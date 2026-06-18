"""质量把关 Agent — 6维评分，不达标自动回退"""

import json
from typing import Any, Dict, List
from .base import BaseAgent
from note_weaver.utils.logger import logger
from note_weaver.utils.prompts import QA_SYSTEM, QA_USER
from note_weaver.utils.config import config


class QAAgent(BaseAgent):
    """笔记质量评估，6维打分 + 修改建议"""

    def __init__(self):
        super().__init__()  # 使用 fast 模型进行审查

    def execute(
        self,
        note_content: str,
        transcript_text: str,
        vision_results: List[Dict[str, Any]],
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        """
        Args:
            note_content: 生成的笔记正文
            transcript_text: 原始转录文本
            vision_results: Vision Agent 的分析结果
            threshold: 本次通过的阈值，None 则用 config.default

        Returns:
            {
                "scores": {"terminology_accuracy": ..., ...},
                "total": 综合分 (0-10),
                "summary": "一句话总评",
                "issues": [...],
                "revision_suggestions": "修改意见",
                "passed": bool,
            }
        """
        logger.info(f"[QA] 开始审核笔记 ({len(note_content)} 字符)")

        # 构建图片描述摘要
        img_desc = self._summarize_vision(vision_results)
        transcript_excerpt = transcript_text[:2000]  # 前2000字足够判断

        prompt = QA_USER.format(
            transcript_length=len(transcript_text),
            transcript_excerpt=transcript_excerpt,
            image_descriptions=img_desc,
            note_content=note_content[:8000],  # 控制输入长度
        )

        try:
            raw = self.chat(prompt, system_instruction=QA_SYSTEM, expect_json=True)
            report = json.loads(raw)
        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(f"[QA] 评分失败，默认通过: {e}")
            return self._default_pass()

        # 计算加权综合分
        weights = config.qa_weights
        scores = report.get("scores", {})
        total = report.get("total", 7.0)

        if not total or total == 0:
            total = sum(
                scores.get(k, 7) * weights.get(k, 0.166)
                for k in weights
            )

        report["total"] = round(total, 1)

        # 判断是否通过（支持递减阈值：首轮7.0，重试逐次降低）
        threshold = config.qa_pass_threshold if threshold is None else threshold
        report["passed"] = total >= threshold

        status = "✅ 通过" if report["passed"] else "❌ 不通过"
        logger.info(
            f"[QA] {status} | total={total:.1f} threshold={threshold:.1f} | "
            f"术语={scores.get('terminology_accuracy', '?')} "
            f"结构={scores.get('structure_clarity', '?')} "
            f"图文={scores.get('image_text_alignment', '?')}"
        )

        if report.get("issues"):
            for issue in report["issues"]:
                logger.info(f"[QA]   问题: {issue}")

        return report

    @staticmethod
    def _summarize_vision(vision_results: List[Dict]) -> str:
        """将 Vision 结果压缩为 QA 可用的摘要"""
        if not vision_results:
            return "（无图片）"

        lines = []
        for r in vision_results:
            if r.get("should_include", True):
                lines.append(
                    f"- {r.get('image_id', '?')}: "
                    f"{r.get('content_description', '?')[:80]}"
                )
        return "\n".join(lines) if lines else "（无可用图片）"

    @staticmethod
    def _default_pass() -> Dict[str, Any]:
        return {
            "scores": {k: 7 for k in config.qa_weights},
            "total": 7.0,
            "summary": "（默认通过 — QA评分异常）",
            "issues": [],
            "revision_suggestions": "",
            "passed": True,
        }
