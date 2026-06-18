"""Agent 基类 — DeepSeek API (OpenAI 兼容) 统一调用、重试、日志"""

import time
import json
import base64
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from openai import OpenAI
from note_weaver.utils.logger import logger
from note_weaver.utils.config import config


class BaseAgent(ABC):
    """所有 Agent 的抽象基类，封装 DeepSeek API (OpenAI 兼容协议)"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.model_fast
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI 客户端（指向 DeepSeek）"""
        if self._client is None:
            self._client = OpenAI(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
            )
        return self._client

    def chat(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_retries: int = 3,
        expect_json: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """调用 DeepSeek Chat API，自动重试"""
        messages = self._build_messages(prompt, system_instruction)

        for attempt in range(1, max_retries + 1):
            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                # deepseek-reasoner 不支持 temperature 参数
                if self.model_name == "deepseek-reasoner":
                    kwargs.pop("temperature", None)

                response = self.client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content or ""

                if expect_json:
                    text = self._clean_json(text)

                return text

            except Exception as e:
                logger.warning(
                    f"[{self.__class__.__name__}] DeepSeek API err "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(3 * attempt)
                else:
                    raise RuntimeError(
                        f"[{self.__class__.__name__}] API 全部重试失败: {e}"
                    ) from e

    def chat_with_image(
        self,
        prompt: str,
        image_path: str,
        system_instruction: Optional[str] = None,
        max_retries: int = 3,
        expect_json: bool = False,
    ) -> str:
        """调用 DeepSeek Vision API（带图片）"""
        data_url = self._encode_image(image_path)

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                )
                text = response.choices[0].message.content or ""

                if expect_json:
                    text = self._clean_json(text)

                return text

            except Exception as e:
                logger.warning(
                    f"[{self.__class__.__name__}] DeepSeek Vision err "
                    f"(attempt {attempt}/{max_retries}, {os.path.basename(image_path)}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(3 * attempt)
                else:
                    raise RuntimeError(
                        f"[{self.__class__.__name__}] Vision API 全部重试失败: {e}"
                    ) from e

    # ---- 内部辅助 ----

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """构建 messages，deepseek-reasoner 不支持 system role 则合并到 user prompt"""
        if self.model_name == "deepseek-reasoner" and system_instruction:
            merged = f"[系统指令]\n{system_instruction}\n\n---\n\n{prompt}"
            return [{"role": "user", "content": merged}]

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """将本地图片编码为 data:image/...;base64,... URL"""
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime = mime_map.get(ext, "image/jpeg")
        return f"data:{mime};base64,{img_data}"

    @staticmethod
    def _clean_json(text: str) -> str:
        """去除 JSON 外层的 markdown 代码块标记"""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip()

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Agent 的执行入口，子类必须实现"""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} model={self.model_name}>"
