"""视觉理解 Agent — 使用 Qwen Vision API 分析截图

继承 BaseAgent（provider="qwen"），共享 client、_encode_image、_clean_json 等能力。

质量控制流水线：
  1. 质量过滤（模糊/过暗/过小/纯色）
  2. 相邻帧相似度去重
  3. top-k 截断
  4. OCR 文字密度优先排序
  5. VLM 分析（可批处理）
"""

import json
import os
import time
from typing import Any, Dict, List

from note_weaver.utils.logger import logger
from note_weaver.utils.config import config
from note_weaver.utils.prompts import VISION_SYSTEM
from note_weaver.agents.base import BaseAgent


class VisionAgent(BaseAgent):
    """Qwen Vision API — 智能截图分析（含质量过滤 + 去重 + 优先排序）

    继承 BaseAgent(provider="qwen")，自动使用 Qwen API 凭证。
    自动模型故障转移：当当前模型的免费额度耗尽时，
    自动切换到下一个备用模型（每个模型有独立 100 万免费额度）。
    """

    # 备用模型列表（按优先级排列，每个有独立免费额度）
    # 参考: https://help.aliyun.com/zh/model-studio/vision-model/
    FALLBACK_MODELS = [
        # 当前配置的模型（由 _active_model 持有，动态确定）
        # — 以下为故障转移备选（每个都有免费额度） —
        "qwen-vl-ocr",                 # 图片文字识别，免费额度
        "qwen3.5-plus",                # 原生视觉语言模型，1M上下文，免费额度
        "qwen3.5-flash",               # 轻量版，免费额度
        "qwen-vl-plus",                # 你当前在用的，免费额度
        "qwen-vl-max",                 # 付费兜底
    ]

    def __init__(self, max_images: int = 10):
        # 继承 BaseAgent(provider="qwen") → 自动使用 Qwen API 凭证
        super().__init__(provider="qwen")
        self.max_images = (
            config.get("vision.max_images_per_batch")   # config.yaml 优先
            if config.get("vision.max_images_per_batch") is not None
            else max_images                              # 否则用构造参数
        )
        self.skip_low_quality = config.get("vision.skip_low_quality", True)
        self.quality_threshold = config.get("vision.quality_threshold", "medium")

        # 当前活跃模型（从配置读取，后续可能因故障转移而变更）
        self._active_model = config.qwen_model_vision
        # 已尝试过的备用模型（避免循环重试）
        self._tried_models: set = set()

    def _is_quota_error(self, e: Exception) -> bool:
        """判断异常是否为配额耗尽错误"""
        err_str = str(e).lower()
        quota_keywords = [
            "quota exhausted", "quota exceeded",
            "insufficient_quota", "429", "too many requests",
            "rate limit", "usage limit", "amount limit",
            "tokens exceeded", "token limit",
        ]
        return any(kw in err_str for kw in quota_keywords)

    def _call_vision_with_failover(self, data_url: str) -> str:
        """调用视觉模型，自动故障转移

        Args:
            data_url: base64 图片 data URL

        Returns:
            API 响应文本

        Raises:
            RuntimeError: 所有模型均失败
        """
        # 构建模型尝试列表：当前模型 + 备用模型中未尝试过的
        models_to_try = [self._active_model]
        for m in self.FALLBACK_MODELS:
            if m != self._active_model and m not in self._tried_models:
                models_to_try.append(m)

        last_error = None
        for model in models_to_try:
            if model in self._tried_models:
                continue

            for attempt in range(1, 4):
                try:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": VISION_SYSTEM},
                            {"role": "user", "content": [
                                {"type": "text", "text": "分析这张教学视频截图。"},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ]},
                        ],
                        temperature=0.3,
                    )
                    content = resp.choices[0].message.content or ""

                    # 成功后更新活跃模型（后续图片继续用这个模型）
                    if model != self._active_model:
                        logger.info(
                            f"[Vision] 故障转移: {self._active_model} → {model}"
                        )
                        self._active_model = model
                    self._tried_models.clear()  # 成功后重置已尝试列表
                    return content

                except Exception as e:
                    last_error = e
                    is_quota = self._is_quota_error(e)

                    if is_quota:
                        self._tried_models.add(model)
                        logger.warning(
                            f"[Vision] {model} 配额耗尽，尝试下一个..."
                        )
                        break  # 换模型，不重试
                    elif attempt < 3:
                        time.sleep(2 * attempt)  # 非配额错误重试3次
                    else:
                        self._tried_models.add(model)
                        logger.warning(
                            f"[Vision] {model} 重试3次仍失败: {e}"
                        )
                        break  # 3次都失败，换模型

        raise RuntimeError(
            f"视觉模型全部不可用: 已尝试 {models_to_try}, 最后错误: {last_error}"
        )

    def execute(
        self,
        screenshot_files: List[str],
    ) -> List[Dict[str, Any]]:
        """质量控制流水线 + VLM 分析"""
        if not screenshot_files:
            logger.info("[Vision] 无截图")
            return []

        original_count = len(screenshot_files)

        # ── 流水线 ──
        # Step 1: 质量过滤
        if self.skip_low_quality:
            screenshot_files = self._quality_filter(
                screenshot_files, min_brightness=10
            )
            logger.info(f"[Vision] 质量过滤: {original_count} → {len(screenshot_files)}")

        # Step 2: 相似度去重
        deduped = self._deduplicate(screenshot_files, sim_threshold=0.95)
        if len(deduped) < len(screenshot_files):
            logger.info(f"[Vision] 去重: {len(screenshot_files)} → {len(deduped)}")
            screenshot_files = deduped

        # Step 3: top-k 截断
        if len(screenshot_files) > self.max_images:
            # 先按文字密度排序，再取 top-k
            screenshot_files = self._prioritize_by_text_density(screenshot_files)
            screenshot_files = screenshot_files[:self.max_images]
            logger.info(f"[Vision] top-k 截断: → {len(screenshot_files)} (最多 {self.max_images})")

        # Step 4: VLM 分析
        logger.info(f"[Vision] {self._active_model} 分析 {len(screenshot_files)} 张...")
        results = []

        for i, img_path in enumerate(screenshot_files):
            img_name = os.path.basename(img_path)
            logger.info(f"[Vision] [{i+1}/{len(screenshot_files)}] {img_name}")

            try:
                data_url = self._encode_image(img_path)

                raw = self._call_vision_with_failover(data_url)
                parsed = json.loads(self._clean_json(raw))
                parsed["image_id"] = img_name
                parsed["image_path"] = img_path
                if "should_include" not in parsed:
                    parsed["should_include"] = True
                results.append(parsed)

                action = "+" if parsed.get("should_include") else "-"
                logger.info(
                    f"  [{action}] {img_name}: {parsed.get('type', '?')} "
                    f"— {parsed.get('content_description', '')[:60]}"
                )

            except Exception as e:
                logger.warning(f"[Vision] {img_name} 失败: {e}")
                results.append({
                    "image_id": img_name, "image_path": img_path,
                    "type": "unknown", "content_description": f"截图 #{i+1}（分析失败）",
                    "key_terms": [], "contains_formula": False, "contains_table": False,
                    "readability": "low", "should_include": True, "suggested_caption": "",
                })

        included = sum(1 for r in results if r.get("should_include"))
        logger.info(
            f"[Vision] 完成: {original_count}→{len(screenshot_files)}→{len(results)} 张, "
            f"采纳 {included} 张"
        )
        return results

    # =================================================================
    # 质量控制方法
    # =================================================================

    def _quality_filter(
        self, images: List[str],
        min_width: int = 100, min_height: int = 100,
        min_brightness: float = 10,
    ) -> List[str]:
        """过滤低质量图片：模糊/过暗/过小/纯色"""
        filtered = []
        from PIL import Image, ImageStat
        for img_path in images:
            try:
                img = Image.open(img_path)
                w, h = img.size
                if w < min_width or h < min_height:
                    continue

                stat = ImageStat.Stat(img)
                mean_brightness = sum(stat.mean) / len(stat.mean)
                if mean_brightness < min_brightness:
                    continue  # 过暗

                mean_std = sum(stat.stddev) / len(stat.stddev)
                if mean_std < 5:
                    continue  # 纯色/过渡帧

                filtered.append(img_path)
            except Exception:
                continue
        return filtered

    def _deduplicate(self, images: List[str], sim_threshold: float = 0.90,
                      min_keep: int = 3) -> List[str]:
        """相邻帧直方图相似度去重

        直方图相似度对白板/幻灯片类内容不敏感（白色背景+深色文字 → 直方图相似度很高）。
        因此强制至少保留 min_keep 帧，防止不同内容的板书被错误合并。

        Args:
            images: 图片路径列表
            sim_threshold: 相似度阈值（超过此值视为重复）
            min_keep: 最少保留帧数（安全网）

        Returns:
            去重后的图片路径列表
        """
        if not images:
            return []

        def _hist_sim(p1: str, p2: str) -> float:
            from PIL import Image
            try:
                h1 = Image.open(p1).histogram()
                h2 = Image.open(p2).histogram()
                return sum(min(a, b) for a, b in zip(h1, h2)) / max(sum(h1), sum(h2))
            except Exception:
                return 0.0

        deduped = [images[0]]
        for img in images[1:]:
            sim = _hist_sim(img, deduped[-1])
            if sim < sim_threshold:
                deduped.append(img)

        # 安全网：至少保留 min_keep 帧（白板/幻灯片直方图相似度高，容易误判）
        if len(deduped) < min_keep < len(images):
            # 从原列表中补充帧（优先补充时间分布均匀的）
            needed = min_keep - len(deduped)
            used = set(id(p) for p in deduped)
            # 按步长均匀抽取补充
            step = max(1, len(images) // (needed + 1))
            extra = [images[i] for i in range(step, len(images), step)
                     if id(images[i]) not in used][:needed]
            deduped.extend(extra)
            deduped.sort(key=lambda p: images.index(p))
            logger.debug(f"[Vision] 去重安全网: 补充 {len(extra)} 帧 → {len(deduped)} 帧")

        return deduped

    def _prioritize_by_text_density(self, images: List[str]) -> List[str]:
        """按文字密度排序（有文字的帧优先处理）"""
        if not images:
            return []

        def _text_density(img_path: str) -> float:
            from PIL import Image, ImageFilter
            import numpy as np
            try:
                img = Image.open(img_path).convert("L")
                edges = img.filter(ImageFilter.FIND_EDGES)
                arr = np.array(edges)
                return float((arr > 30).sum() / arr.size)
            except Exception:
                return 0.0

        scored = [(img, _text_density(img)) for img in images]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [img for img, _ in scored]

