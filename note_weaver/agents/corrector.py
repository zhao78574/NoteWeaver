"""Correction Agent — 领域感知的转录纠错 Agent

职责：
  在 Whisper 转录完成后，Composer 排版之前，
  对原始转录文本进行领域感知的纠错处理。

设计原则：
  - 单一职责：只做纠错，不做笔记生成
  - 领域感知：根据 Router 输出的 domain 注入专业词表
  - 轻量快速：使用 fast 模型，不阻塞管线
  - 保留时间戳：纠错后的 segments 保持原始时间戳不变
"""

import json
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from note_weaver.utils.logger import logger


CORRECTOR_SYSTEM = """你是一个专业的听写纠错专家。你的任务是对语音识别（ASR）的转录文本进行纠错，**只改错误，不改内容风格**。

## 纠错重点

### 1. 专业术语修正
- 识别并修正被误听的专业术语、行业黑话
- 例如："外沿" → "外延"，"客室" → "刻蚀"，"P N姐" → "PN结"
- 如果提供了领域关键词列表，优先检查这些词是否被正确识别

### 2. 数字/单位修正
- 数字和单位可能被误听 (30nm → 300nm, 100kHz → 100赫兹)
- 保持原始单位的规范表达
- 注意中文数字 → 阿拉伯数字（"三十纳米" → "30nm"）

### 3. 英文术语归一化
- 英文缩写字母之间可能被加空格 → 去掉空格："P N 节" → "PN结"，"C V D" → "CVD"
- 连续英文词注意恢复单词边界

### 4. 谐音字/同音字修正
- 中文常见谐音错误（"那个" → 保留但注意上下文人称指代）
- 技术专有名词首字母缩写（"西莫斯" → "CMOS"，"艾完美" → "IME"）

### 5. 保留的内容
- 保持原始的语气词（"嗯、啊"在必要时保留或酌情删除）
- 保持口语化的表达（这是笔记素材，不需要完全书面化）
- 保持英文/数字的原始形式

## 输出格式

返回 JSON：
{
  "corrected_segments": [
    {
      "start": 0.0,
      "end": 12.5,
      "text": "修正后的文本",
      "corrections": ["外沿→外延", "30nm→300nm"]
    }
  ],
  "summary": {
    "total_corrections": 5,
    "domain_terms_corrected": 3
  }
}

注意：
- 保持原 segments 的 start/end 不变
- 只修改 text 字段
- corrections 为空列表表示本次无修改
- 如果原文本没有错误，直接返回原始内容"""


CORRECTOR_USER = """## 领域关键词（参考）
{domain_keywords}

## 转录段落
{segments_text}

请按格式输出 JSON 纠错结果。"""


class CorrectorAgent(BaseAgent):
    """转录纠错 Agent — 领域感知纠错"""

    def __init__(self):
        # Corrector 使用 fast 模型（轻量快速）
        super().__init__()

    def execute(
        self,
        segments: List[Dict[str, Any]],
        domain_keywords: Optional[List[str]] = None,
        domain: str = "",
        raw_text: str = "",
    ) -> Dict[str, Any]:
        """对转录段落进行纠错

        Args:
            segments: 转录段落列表 [{"start": 0.0, "end": 12.5, "text": "..."}, ...]
            domain_keywords: 领域关键词列表（来自 Policy Engine）
            domain: 领域标签（如 "semiconductor"）
            raw_text: 原始全文（可选，用于全文级别纠错）

        Returns:
            {
                "segments": [{"start": ..., "end": ..., "text": "纠错后文本"}, ...],
                "raw_text": "纠错后的全文",
                "timestamped": "纠错后的带时间戳文本",
                "corrections_count": 5,
                "corrections_detail": ["外沿→外延", ...],
            }
        """
        if not segments:
            return {
                "segments": [],
                "raw_text": raw_text,
                "timestamped": "",
                "corrections_count": 0,
                "corrections_detail": [],
            }

        # ── 格式化关键词 ──
        kw_section = ""
        if domain_keywords:
            kw_list = ", ".join(domain_keywords[:30])
            kw_section = f"以下关键词很可能出现在本内容中，请确保它们被正确转写：\n{kw_list}\n"
        if domain and not kw_section:
            kw_section = f"内容领域：{domain}\n"

        # ── 分段纠错（每批最多 20 段，避免 context 过长） ──
        BATCH_SIZE = 20
        corrected_segments = []
        total_corrections = 0
        all_corrections_detail = []

        for batch_start in range(0, len(segments), BATCH_SIZE):
            batch = segments[batch_start:batch_start + BATCH_SIZE]

            # 快速预检：如果领域关键词已在文本中正确出现，跳过 LLM 调用
            if domain_keywords and self._batch_has_all_keywords(batch, domain_keywords):
                logger.debug(
                    f"[Corrector] 跳过批次 {batch_start//BATCH_SIZE + 1}: "
                    f"领域术语已全部正确出现"
                )
                corrected_segments.extend(batch)
                continue

            corrected, corrections_count, corrections_detail = self._correct_batch(
                batch, kw_section
            )
            corrected_segments.extend(corrected)
            total_corrections += corrections_count
            all_corrections_detail.extend(corrections_detail)

        # ── 重建 raw_text 和 timestamped ──
        raw_parts = []
        timestamped_lines = []
        for seg in corrected_segments:
            raw_parts.append(seg["text"])
            m, s = divmod(int(seg["start"]), 60)
            timestamped_lines.append(f"[{m:02d}:{s:02d}] {seg['text']}")

        result = {
            "segments": corrected_segments,
            "raw_text": " ".join(raw_parts),
            "timestamped": "\n".join(timestamped_lines),
            "corrections_count": total_corrections,
            "corrections_detail": all_corrections_detail[:20],  # 最多记录 20 条
        }

        if total_corrections > 0:
            logger.info(
                f"[Corrector] 纠错完成: {total_corrections} 处修正 "
                f"({', '.join(all_corrections_detail[:5])}...)"
            )
        else:
            logger.info("[Corrector] 无修正（转录质量良好）")

        return result

    @staticmethod
    def _batch_has_all_keywords(
        batch: List[Dict[str, Any]],
        domain_keywords: List[str],
    ) -> bool:
        """快速预检：检查一批段落中是否已有足够领域关键词正确出现

        如果 ≥3 个不同的领域关键词在文本中正确出现，说明此批转录质量
        已足够好，无需调用 LLM 纠错，直接跳过以节省 token。

        Args:
            batch: 一批转录段落
            domain_keywords: 领域关键词列表

        Returns:
            True = 关键词覆盖良好，跳过 LLM 调用
        """
        combined = " ".join(seg.get("text", "") for seg in batch).lower()

        found = 0
        for kw in domain_keywords:
            if kw.lower() in combined:
                found += 1
                if found >= 3:   # ≥3 个不同关键词正确出现 → 质量合格
                    return True

        return False

    def _correct_batch(
        self,
        batch: List[Dict[str, Any]],
        kw_section: str,
    ) -> tuple:
        """纠错一批段落

        Returns:
            (corrected_segments, corrections_count, corrections_detail)
        """
        # 构造 batch 文本
        segments_text = json.dumps(
            [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in batch],
            ensure_ascii=False,
            indent=2,
        )

        prompt = CORRECTOR_USER.format(
            domain_keywords=kw_section or "（无特定领域关键词）",
            segments_text=segments_text,
        )

        try:
            raw = self.chat(prompt, system_instruction=CORRECTOR_SYSTEM, expect_json=True)
            result = json.loads(raw)

            corrected = result.get("corrected_segments", [])
            summary = result.get("summary", {})
            corrections_count = summary.get("total_corrections", 0)

            # 提取修正细节
            corrections_detail = []
            for seg in corrected:
                for c in seg.get("corrections", []):
                    corrections_detail.append(c)

            # 如果没有纠正，用原始数据
            if not corrected:
                corrected = batch
                corrections_count = 0

            return corrected, corrections_count, corrections_detail

        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(f"[Corrector] 纠错失败（保持原文）: {e}")
            return batch, 0, []
