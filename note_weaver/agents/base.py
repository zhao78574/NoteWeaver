"""Agent 基类 — 多 provider 支持 (DeepSeek / Qwen)，统一调用、重试、日志"""

from __future__ import annotations

import time
import base64
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from note_weaver.utils.logger import logger
from note_weaver.utils.config import config

if TYPE_CHECKING:
    from openai import OpenAI


# ── Provider → API 凭证映射 ────────────────────────────────────

_PROVIDER_CREDENTIALS = {
    "deepseek": {
        "api_key_attr": "deepseek_api_key",
        "base_url_attr": "deepseek_base_url",
    },
    "qwen": {
        "api_key_attr": "qwen_api_key",
        "base_url_attr": "qwen_base_url",
    },
}


class BaseAgent(ABC):
    """所有 Agent 的抽象基类，支持多 LLM provider

    provider 参数：
      - "deepseek"（默认）：使用 DeepSeek API，凭证来自 config.deepseek_api_key
      - "qwen"：使用 Qwen / 阿里云百炼 API，凭证来自 config.qwen_api_key

    子类只需传 provider 即可获得对应的 client、chat、stream_chat、
    _encode_image、_clean_json 等通用能力。
    """

    def __init__(self, model_name: Optional[str] = None, provider: str = "deepseek"):
        self.provider = provider
        if model_name is not None:
            self.model_name = model_name
        elif provider == "qwen":
            self.model_name = config.qwen_model_vision
        else:
            self.model_name = config.model_fast
        self._client: Optional[Any] = None

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI 客户端（根据 provider 自动选择凭证）"""
        if self._client is None:
            from openai import OpenAI
            creds = _PROVIDER_CREDENTIALS.get(self.provider, _PROVIDER_CREDENTIALS["deepseek"])
            self._client = OpenAI(
                api_key=getattr(config, creds["api_key_attr"]),
                base_url=getattr(config, creds["base_url_attr"]),
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

    def stream_chat(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        max_retries: int = 3,
        temperature: float = 0.7,
    ) -> str:
        """流式调用 LLM，逐 token 回调 on_token

        Returns:
            完整的响应文本（与 chat() 相同）
        """
        messages = self._build_messages(prompt, system_instruction)

        for attempt in range(1, max_retries + 1):
            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
                if self.model_name == "deepseek-reasoner":
                    kwargs.pop("temperature", None)

                collected: list[str] = []
                response = self.client.chat.completions.create(**kwargs)
                for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        collected.append(delta.content)
                        if on_token:
                            on_token(delta.content)

                full_text = "".join(collected)
                return full_text

            except Exception as e:
                logger.warning(
                    f"[{self.__class__.__name__}] DeepSeek stream API err "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(3 * attempt)
                else:
                    raise RuntimeError(
                        f"[{self.__class__.__name__}] Stream API 全部重试失败: {e}"
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
