"""Router — 视频内容路由层（原 Classifier 升级版）

职责：
  在转录的同时快速分析视频，输出 routing 信号：
    - domain            领域（semiconductor / ai_ml / medicine / ...）
    - content_structure 内容结构（lecture / tutorial / research_talk / meeting）
    - visual_density    视觉密度（low / medium / high）
    - keywords          关键术语列表
    - language          检测语言

与旧 Classifier 的区别：
  - 不依赖转录结果（独立分析首帧 + 首30s音频）
  - 与 Transcribe 并行运行（不阻塞管线）
  - 输出信号归 Router 层，不做参数决策（交给 Policy Engine）
  - 支持缓存（同系列视频秒级命中）
"""

import json
import os
import re
import time
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseAgent
from note_weaver.utils.logger import logger
from note_weaver.utils.config import config


# ── Router 缓存 ───────────────────────────────────────────────

_ROUTER_CACHE_DIR = Path(config.memory_dir) / "router_cache"


def _cache_key_for_video(video_path: str) -> str:
    """文件内容级别缓存键（SHA256 前 16 位）"""
    try:
        with open(video_path, "rb") as f:
            # 只读前 1MB 以加快速度（头部信息足够区分不同视频）
            head = f.read(1024 * 1024)
        return hashlib.sha256(head).hexdigest()[:16]
    except Exception:
        return ""


def _cache_key_for_series(series_pattern: str) -> str:
    """系列级别缓存键（如 "半导体工艺"）"""
    return "series:" + hashlib.md5(series_pattern.encode()).hexdigest()[:16]


def _load_router_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """从磁盘加载 Router 缓存"""
    cache_path = _ROUTER_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 检查是否过期（7 天）
            if time.time() - data.get("cached_at", 0) < 86400 * 7:
                logger.info(f"[Router] 缓存命中: {cache_key}")
                return data.get("result")
        except Exception:
            pass
    return None


def _save_router_cache(cache_key: str, result: Dict[str, Any]):
    """保存 Router 结果到磁盘缓存"""
    _ROUTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_ROUTER_CACHE_DIR / f"{cache_key}.json", "w", encoding="utf-8") as f:
            json.dump({"result": result, "cached_at": time.time()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"[Router] 缓存写入失败（非致命）: {e}")


def _extract_series_pattern(file_name: str) -> Optional[str]:
    """从文件名中提取系列模式

    支持格式：
      - "半导体工艺01" → "半导体工艺"
      - "CMOS工艺（1）" → "CMOS工艺"
      - "Python基础教程_p01" → "Python基础教程"
      - "[莫烦Python] Python 基础教程 p05" → "莫烦Python Python 基础教程"
    """
    # 去掉扩展名
    name = os.path.splitext(file_name)[0]

    # 尝试匹配常见模式
    patterns = [
        (r'^(.+?)[_\s]*(?:p\d+|第\d+[集节章]|\(\d+\)|\d{2})\s*$', 1),         # xxx_p01 / xxx_第1集
        (r'^(.+?)[_\s]*\d+[_\s]*[vV]?\d*\.?\d*$', 1),                           # xxx01 / xxx 01
        (r'^\[.+?\]\s*(.+?)(?:p\d+)', 1),                                        # [莫烦Python] xxx p05
    ]

    for pat, group in patterns:
        m = re.match(pat, name)
        if m:
            return m.group(group).strip().rstrip('_ -')

    # 如果无模式匹配，提取前缀（连续中文 + 英文词）
    m = re.match(r'^([一-鿿\w]+)[_\s\d]', name)
    if m:
        candidate = m.group(1)
        if len(candidate) >= 2:
            return candidate

    return None


# ── 帧差异估算（用于 visual_density） ────────────────────────


def _estimate_visual_density(screenshot_dir: str, file_base: str, max_frames: int = 5) -> str:
    """从截图中估算视觉密度

    通过比较前几张截图的直方图差异来判定画面变化频率。

    Args:
        screenshot_dir: 截图目录
        file_base: 文件名前缀
        max_frames: 最多检查多少张图

    Returns:
        "low" | "medium" | "high"
    """
    if not os.path.isdir(screenshot_dir):
        return "medium"

    images = sorted([
        os.path.join(screenshot_dir, f)
        for f in os.listdir(screenshot_dir)
        if f.startswith(file_base) and f.lower().endswith(('.jpg', '.png'))
    ])

    if len(images) < 2:
        return "medium"

    images = images[:min(max_frames, len(images))]
    diffs = []

    try:
        from PIL import Image
        import numpy as np

        prev_hist = None
        for img_path in images:
            img = Image.open(img_path).convert("L")  # 灰度
            img = img.resize((64, 48))               # 缩略图
            hist = np.array(img.histogram(), dtype=np.float32)
            hist /= (hist.sum() + 1e-8)               # L1 归一化

            if prev_hist is not None:
                # 直方图交叉核（值越接近1越相似）
                similarity = np.minimum(hist, prev_hist).sum()
                diff = 1.0 - similarity
                diffs.append(diff)

            prev_hist = hist
    except ImportError:
        logger.debug("[Router] PIL/numpy 不可用，跳过视觉密度估算")
        return "medium"
    except Exception as e:
        logger.debug(f"[Router] 视觉密度估算失败（非致命）: {e}")
        return "medium"

    # 根据平均差异判定
    if not diffs:
        return "medium"

    from note_weaver.agents.policy import PolicyEngine
    return PolicyEngine.visual_density_from_frames(diffs)


# ── Router Agent ──────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """你是一个视频内容快速分析专家。根据视频文件名和第一段音频文本，快速判断视频的特征。

输出 JSON（只返回 JSON，不要其他文字）：
{
  "domain": "semiconductor | ai_ml | medicine | physics | chemistry | exam_prep | general",
  "content_structure": "lecture | tutorial | research_talk | meeting",
  "visual_density_hint": "low | medium | high",
  "keywords": ["关键术语1", "关键术语2", ...],
  "language": "zh | en",
  "confidence": 0.0~1.0,
  "reasoning": "一句话判断理由"
}

判断依据：
- domain：根据专业领域判断，默认 general
- content_structure：
  * lecture → 教师/PPT/系统的知识讲授
  * tutorial → 操作步骤/代码演示/实验流程
  * research_talk → 学术报告/论文分享/会议演讲
  * meeting → 会议讨论/多人对话
- keywords：提取 3-8 个该视频最可能涉及的关键术语
- visual_density_hint：根据内容类型提示视觉密度（low=纯语音/播客类, medium=PPT讲解, high=频繁切换画面）"""


class RouterAgent(BaseAgent):
    """视频内容路由 Agent — 快速分析视频特征"""

    def __init__(self):
        # Router 使用 fast 模型（轻量快速）
        super().__init__()
        self._prompt_template = None

    def execute(
        self,
        video_path: str = "",
        file_name: str = "",
        audio_first_30s: str = "",
        duration: float = 0.0,
        screenshot_dir: str = "",
        file_base: str = "",
        audio_path: str = "",   # 新增：音频文件路径（独立转写用）
    ) -> Dict[str, Any]:
        """分析视频并返回路由信号

        Args:
            video_path: 视频文件路径
            file_name: 视频文件名（含扩展名）
            audio_first_30s: 前 30 秒音频转录文本（来自 Transcribe，可能尚未就绪）
            duration: 视频时长（秒）
            screenshot_dir: 截图目录（可选，用于视觉密度估算）
            file_base: 文件名前缀（可选，用于视觉密度估算）
            audio_path: 音频文件路径（当 Transcribe 未完成时，独立提取前30s文本）

        Returns:
            {
                "domain": "semiconductor",
                "content_structure": "lecture",
                "visual_density": "medium",          # 实际估算值
                "keywords": ["光刻", "刻蚀"],
                "language": "zh",
                "confidence": 0.85,
                "cache_key": "...",                   # 用于后续缓存
                "series_pattern": "半导体工艺",       # 系列模式（可选）
            }
        """
        result = self._default_result()

        # ── 1. 检查缓存 ──
        # 1a. 文件级缓存
        if video_path:
            cache_key = _cache_key_for_video(video_path)
            cached = _load_router_cache(cache_key)
            if cached:
                cached["cache_key"] = cache_key
                cached["visual_density"] = self._detect_density(screenshot_dir, file_base)
                return cached

        # 1b. 系列级缓存
        series_pattern = _extract_series_pattern(file_name) if file_name else None
        if series_pattern:
            series_key = _cache_key_for_series(series_pattern)
            cached = _load_router_cache(series_key)
            if cached:
                cached["cache_key"] = cache_key if video_path else ""
                cached["series_pattern"] = series_pattern
                cached["visual_density"] = self._detect_density(screenshot_dir, file_base)
                return cached

        # ── 2. 获取前30s音频文本（不依赖 Transcribe，真正并行） ──
        if not audio_first_30s or not audio_first_30s.strip():
            if audio_path and os.path.isfile(audio_path):
                quick_text = self._quick_transcribe(audio_path, max_duration=30)
                if quick_text and quick_text.strip():
                    audio_first_30s = quick_text
                    logger.info(f"[Router] 独立转写前30s完成 ({len(quick_text)}字)")

        # ── 3. 调用 LLM 分类（只依赖文件名 + 音频片段） ──
        if file_name or audio_first_30s:
            llm_result = self._classify_with_llm(file_name, audio_first_30s, duration)
            if llm_result:
                result.update(llm_result)
                logger.info(
                    f"[Router] {file_name} → domain={result['domain']}, "
                    f"structure={result['content_structure']}, "
                    f"confidence={result['confidence']}"
                )

        # ── 4. 估算 visual_density（用实际截图，不依赖 LLM） ──
        result["visual_density"] = self._detect_density(screenshot_dir, file_base)

        # ── 5. 写入缓存 ──
        if video_path and cache_key:
            _save_router_cache(cache_key, {k: v for k, v in result.items()
                                            if k not in ("visual_density", "cache_key", "series_pattern")})
        if series_pattern and series_key:
            _save_router_cache(series_key, {k: v for k, v in result.items()
                                             if k not in ("visual_density", "cache_key", "series_pattern")})

        result["cache_key"] = cache_key if video_path else ""
        result["series_pattern"] = series_pattern

        return result

    def _classify_with_llm(
        self,
        file_name: str,
        audio_sample: str,
        duration: float,
    ) -> Optional[Dict[str, Any]]:
        """调用 LLM 进行快速分类"""
        prompt = f"""视频文件名：{file_name or '(未知)'}
视频时长：{duration:.0f} 秒
前30秒音频文本：
{audio_sample[:500] or '(无音频样本)'}"""

        try:
            raw = self.chat(
                prompt,
                system_instruction=ROUTER_SYSTEM_PROMPT,
                expect_json=True,
            )
            result = json.loads(raw)

            # 验证必要字段
            valid_domains = {"semiconductor", "ai_ml", "medicine", "physics",
                             "chemistry", "exam_prep", "general"}
            valid_structures = {"lecture", "tutorial", "research_talk", "meeting"}

            if result.get("domain") not in valid_domains:
                result["domain"] = "general"
            if result.get("content_structure") not in valid_structures:
                result["content_structure"] = "lecture"

            result["confidence"] = float(result.get("confidence", 0.5))
            result["keywords"] = result.get("keywords", [])[:10]

            return result

        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(f"[Router] LLM 分类失败，使用默认策略: {e}")
            return None

    def _detect_density(self, screenshot_dir: str, file_base: str) -> str:
        """检测实际 visual_density"""
        if screenshot_dir and file_base:
            return _estimate_visual_density(screenshot_dir, file_base)
        return "medium"

    def _quick_transcribe(self, audio_path: str, max_duration: int = 30) -> str:
        """使用 tiny 模型快速转写音频前几秒（与主 Transcriber 并行，不依赖其输出）

        在 Transcribe 尚未完成时，Router 自行提取前30s音频文本用于分类判断。
        使用 faster-whisper tiny 模型（~75MB），转写30s音频约需 1-3 秒。

        Args:
            audio_path: 音频文件路径（必须存在且为 ffmpeg 兼容格式）
            max_duration: 最多转写多少秒

        Returns:
            转写文本（失败时返回空字符串）
        """
        # ── 1. 用 ffmpeg 截取前 max_duration 秒到临时文件 ──
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)
        try:
            ret = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-t", str(max_duration),
                 "-acodec", "copy", tmp_path],
                capture_output=True, timeout=60,
            )
            if ret.returncode != 0 or not os.path.isfile(tmp_path):
                logger.debug(f"[Router] 快速转写: ffmpeg 截取失败")
                return ""

            # ── 2. 用 tiny 模型转写 ──
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel("tiny", device="cpu", compute_type="int8")
                segments, _ = model.transcribe(tmp_path, language="zh")
                text = " ".join(seg.text for seg in segments)
                return text.strip()
            except ImportError:
                logger.debug("[Router] faster-whisper 不可用，跳过快速转写")
                return ""
            except Exception as e:
                logger.debug(f"[Router] 快速转写异常（非致命）: {e}")
                return ""

        except subprocess.TimeoutExpired:
            logger.debug("[Router] 快速转写 ffmpeg 超时")
            return ""
        except Exception as e:
            logger.debug(f"[Router] 快速转写失败（非致命）: {e}")
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @staticmethod
    def _default_result() -> Dict[str, Any]:
        return {
            "domain": "general",
            "content_structure": "lecture",
            "visual_density": "medium",
            "keywords": [],
            "language": "zh",
            "confidence": 0.0,
        }
