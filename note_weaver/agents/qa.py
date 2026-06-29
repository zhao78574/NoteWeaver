"""质量把关 Agent — 6维评分 + 结构化缺陷报告

输出升级为 QAReport dataclass，包含：
- dimensions: 各维度分数
- defects: 结构化缺陷列表（含 type/location/severity/suggestion）
- is_passed: 是否通过
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from note_weaver.utils.logger import logger
from note_weaver.utils.prompts import QA_SYSTEM, QA_USER
from note_weaver.utils.config import config


@dataclass
class Defect:
    """结构化缺陷"""
    type: str                                    # missing_content / inaccurate / poor_structure / image_mismatch
    location: str = ""                           # 章节/段落位置
    severity: float = 0.5                        # 0~1
    suggestion: str = ""                         # 修复指令

    @classmethod
    def from_dict(cls, d: dict) -> "Defect":
        return cls(
            type=d.get("type", "inaccurate"),
            location=d.get("location", ""),
            severity=float(d.get("severity", 0.5)),
            suggestion=d.get("suggestion", ""),
        )


@dataclass
class QAReport:
    """结构化质检报告"""
    score: float = 7.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    defects: List[Defect] = field(default_factory=list)
    is_passed: bool = True
    summary: str = ""
    issues: List[str] = field(default_factory=list)
    revision_suggestions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转为 dict（向后兼容旧接口）"""
        return {
            "scores": self.dimensions,
            "total": self.score,
            "summary": self.summary,
            "issues": self.issues,
            "revision_suggestions": self.revision_suggestions,
            "defects": [asdict(d) for d in self.defects],
            "passed": self.is_passed,
        }


class QAAgent(BaseAgent):
    """笔记质量评估，6维打分 + 结构化缺陷报告"""

    def __init__(self):
        super().__init__()  # 使用 fast 模型进行审查

    def execute(
        self,
        note_content: str,
        transcript_text: str,
        vision_results: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            Dict — 兼容旧接口的 dict，包含 scores/total/passed/defects
        """
        logger.info(f"[QA] 开始审核笔记 ({len(note_content)} 字符)")

        # 构建图片描述摘要
        img_desc = self._summarize_vision(vision_results)
        transcript_excerpt = transcript_text[:2000]

        prompt = QA_USER.format(
            transcript_length=len(transcript_text),
            transcript_excerpt=transcript_excerpt,
            image_descriptions=img_desc,
            note_content=note_content[:8000],
        )

        try:
            raw = self.chat(prompt, system_instruction=QA_SYSTEM, expect_json=True)
            report_dict = json.loads(raw)
        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(f"[QA] 评分失败，默认通过: {e}")
            return self._default_pass().to_dict()

        # 解析结构化缺陷
        defects_raw = report_dict.get("defects", [])
        defects = []
        for d in defects_raw:
            try:
                defects.append(Defect.from_dict(d))
            except Exception:
                continue

        # 计算加权综合分
        weights = config.qa_weights
        scores = report_dict.get("scores", {})
        total = report_dict.get("total", 7.0)
        if not total or total == 0:
            total = sum(
                scores.get(k, 7) * weights.get(k, 0.166)
                for k in weights
            )

        report = QAReport(
            score=round(total, 1),
            dimensions=scores,
            defects=defects,
            summary=report_dict.get("summary", ""),
            issues=report_dict.get("issues", []),
            revision_suggestions=report_dict.get("revision_suggestions", ""),
        )

        # 阈值判断
        threshold = config.qa_pass_threshold if threshold is None else threshold
        report.is_passed = total >= threshold

        status = "[OK] 通过" if report.is_passed else "[FAIL] 不通过"
        logger.info(
            f"[QA] {status} | total={report.score:.1f} threshold={threshold:.1f} | "
            f"术语={scores.get('terminology_accuracy', '?')} "
            f"结构={scores.get('structure_clarity', '?')} "
            f"图文={scores.get('image_text_alignment', '?')} "
            f"幻觉={scores.get('hallucination', 'N/A')}"
        )

        if report.defects:
            for d in report.defects:
                logger.info(f"[QA]   [{d.type}] {d.location}: {d.suggestion[:60]}")

        if report.issues:
            for issue in report.issues:
                logger.info(f"[QA]   问题: {issue}")

        return report.to_dict()

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
    def _default_pass() -> QAReport:
        return QAReport(
            score=7.0,
            dimensions={k: 7 for k in config.qa_weights},
            summary="（默认通过 — QA评分异常）",
            is_passed=True,
        )
