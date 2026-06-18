"""视觉理解 Agent — 使用 Qwen Vision API 分析截图"""

import json
import os
import time
from typing import Any, Dict, List
from openai import OpenAI
from note_weaver.utils.logger import logger
from note_weaver.utils.config import config
from note_weaver.utils.prompts import VISION_SYSTEM
from note_weaver.agents.base import BaseAgent as _BA  # 复用静态工具方法


class VisionAgent:
    """Qwen Vision API — 逐张分析视频截图，生成语义描述"""

    def __init__(self):
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=config.qwen_api_key,
                base_url=config.qwen_base_url,
            )
        return self._client

    def execute(
        self,
        screenshot_files: List[str],
    ) -> List[Dict[str, Any]]:
        """逐张分析截图，返回语义描述列表"""
        if not screenshot_files:
            logger.info("[Vision] 无截图")
            return []

        logger.info(f"[Vision] Qwen {config.qwen_model_vision} 分析 {len(screenshot_files)} 张...")
        results = []

        for i, img_path in enumerate(screenshot_files):
            img_name = os.path.basename(img_path)
            logger.info(f"[Vision] [{i+1}/{len(screenshot_files)}] {img_name}")

            try:
                data_url = _BA._encode_image(img_path)

                for attempt in range(1, 4):
                    try:
                        resp = self.client.chat.completions.create(
                            model=config.qwen_model_vision,
                            messages=[
                                {"role": "system", "content": VISION_SYSTEM},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "分析这张教学视频截图。"},
                                    {"type": "image_url", "image_url": {"url": data_url}},
                                ]},
                            ],
                            temperature=0.3,
                        )
                        raw = resp.choices[0].message.content or ""
                        parsed = json.loads(_BA._clean_json(raw))
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
                        break

                    except Exception as e:
                        if attempt < 3:
                            time.sleep(2 * attempt)
                        else:
                            raise

            except Exception as e:
                logger.warning(f"[Vision] {img_name} 失败: {e}")
                results.append({
                    "image_id": img_name, "image_path": img_path,
                    "type": "unknown", "content_description": f"截图 #{i+1}（分析失败）",
                    "key_terms": [], "contains_formula": False, "contains_table": False,
                    "readability": "low", "should_include": True, "suggested_caption": "",
                })

        included = sum(1 for r in results if r.get("should_include"))
        logger.info(f"[Vision] 完成: {len(results)} 张, 采纳 {included} 张")
        return results

