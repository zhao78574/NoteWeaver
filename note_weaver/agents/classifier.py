"""视频分类 Agent — 快速识别视频类型和最优处理策略"""

import json
import os
from typing import Any, Dict
from .base import BaseAgent
from note_weaver.utils.logger import logger
from note_weaver.utils.prompts import CLASSIFIER_SYSTEM, CLASSIFIER_USER


class ClassifierAgent(BaseAgent):
    """快速分类视频类型，建议处理策略"""

    def __init__(self):
        super().__init__()  # 使用 deepseek-chat（fast模型）

    def execute(
        self,
        filename: str,
        audio_sample: str,
        duration: float,
    ) -> Dict[str, Any]:
        """
        Args:
            filename: 视频文件名
            audio_sample: 前30秒音频文本片段
            duration: 视频时长（秒）

        Returns:
            {
                "type": "lecture" | "demo" | "meeting" | "other",
                "subtype": ... | null,
                "domain": "semiconductor",
                "difficulty": "beginner" | "intermediate" | "advanced",
                "has_slides": bool,
                "has_whiteboard": bool,
                "suggested_strategy": {
                    "screenshot_interval": int,
                    "note_style": "detailed" | "concise" | "outline" | "step_by_step",
                    "focus_areas": ["..."],
                }
            }
        """
        logger.info(f"[Classifier] 分析视频: {filename} ({duration:.0f}s)")

        prompt = CLASSIFIER_USER.format(
            filename=filename,
            duration=f"{duration:.0f}",
            audio_sample=audio_sample[:500],  # 取前500字符
        )

        try:
            raw = self.chat(
                prompt,
                system_instruction=CLASSIFIER_SYSTEM,
                expect_json=True,
            )
            result = json.loads(raw)

            logger.info(
                f"[Classifier] {filename} → type={result.get('type')}, "
                f"difficulty={result.get('difficulty')}, "
                f"domain={result.get('domain')}"
            )
            return result

        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(f"[Classifier] 返回非JSON或失败，使用默认策略: {e}")
            return self._default_strategy()

    @staticmethod
    def _default_strategy() -> Dict[str, Any]:
        return {
            "type": "lecture",
            "subtype": None,
            "domain": "general",
            "difficulty": "intermediate",
            "has_slides": True,
            "has_whiteboard": False,
            "suggested_strategy": {
                "screenshot_interval": 180,
                "note_style": "detailed",
                "focus_areas": [],
            },
        }
