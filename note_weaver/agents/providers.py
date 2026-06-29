"""模型 Provider 抽象层 — 注册表 + 工厂函数

允许通过 config.yaml 的 model_registry 配置切换底层模型，
而不修改 Agent 代码。

用法:
    from note_weaver.agents.providers import get_provider

    llm = get_provider("llm", config)
    response = llm.chat([{"role": "user", "content": "Hello"}])

    vision = get_provider("vision", config)
    desc = vision.analyze("image.png", "描述这张图")

    transcriber = get_provider("transcriber", config)
    result = transcriber.transcribe("audio.mp3")
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from note_weaver.utils.logger import logger
from note_weaver.utils.config import config as global_config


# ════════════════════════════════════════════════════════════════
# 抽象基类
# ════════════════════════════════════════════════════════════════


class LLMProvider(ABC):
    """大语言模型 Provider 抽象"""

    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        ...

    @abstractmethod
    def chat_with_image(self, prompt: str, image_path: str,
                        system_instruction: Optional[str] = None) -> str:
        ...


class VisionProvider(ABC):
    """视觉模型 Provider 抽象"""

    @abstractmethod
    def analyze(self, image_path: str, prompt: str) -> str:
        ...


class TranscriberProvider(ABC):
    """语音转录 Provider 抽象"""

    @abstractmethod
    def transcribe(self, audio_path: str, language: str = "zh") -> Dict[str, Any]:
        ...


# ════════════════════════════════════════════════════════════════
# DeepSeek Provider（默认）
# ════════════════════════════════════════════════════════════════


class DeepSeekProvider(LLMProvider):
    """DeepSeek API (OpenAI 兼容协议)"""

    def __init__(self, cfg=None):
        from openai import OpenAI
        cfg = cfg or global_config
        self._client = OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
        )
        self._model = cfg.get("model_registry.llm.model", "deepseek-chat")

    def chat(self, messages: List[Dict], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def chat_with_image(self, prompt: str, image_path: str,
                        system_instruction: Optional[str] = None) -> str:
        import base64, os
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        data_url = f"data:{mime_map.get(ext, 'image/jpeg')};base64,{img_data}"

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
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.3,
        )
        return response.choices[0].message.content or ""


class OpenAIProvider(LLMProvider):
    """OpenAI API Provider"""

    def __init__(self, cfg=None):
        from openai import OpenAI
        cfg = cfg or global_config
        self._client = OpenAI(
            api_key=cfg.openai_api_key,
            base_url="https://api.openai.com/v1",
        )
        self._model = "gpt-4o"

    def chat(self, messages: List[Dict], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs,
        )
        return response.choices[0].message.content or ""

    def chat_with_image(self, prompt: str, image_path: str,
                        system_instruction: Optional[str] = None) -> str:
        import base64, os
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        data_url = f"data:{mime_map.get(ext, 'image/jpeg')};base64,{img_data}"

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
        response = self._client.chat.completions.create(
            model=self._model, messages=messages,
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API Provider (占位，需安装 anthropic 包)"""

    def __init__(self, cfg=None):
        cfg = cfg or global_config
        self._api_key = cfg.anthropic_api_key
        self._model = "claude-sonnet-4-6"
        self._available = False
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
            self._available = True
        except ImportError:
            logger.warning("[Providers] anthropic 未安装，AnthropicProvider 不可用")

    def chat(self, messages: List[Dict], **kwargs) -> str:
        if not self._available:
            raise RuntimeError("anthropic 包未安装")
        import anthropic
        system = None
        real_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                real_messages.append(m)
        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=real_messages,
            max_tokens=4096,
            **kwargs,
        )
        return response.content[0].text if response.content else ""

    def chat_with_image(self, prompt: str, image_path: str,
                        system_instruction: Optional[str] = None) -> str:
        if not self._available:
            raise RuntimeError("anthropic 包未安装")
        import anthropic, base64, os
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        media_type = mime_map.get(ext, "image/jpeg")

        real_messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": img_data,
                }},
                {"type": "text", "text": prompt},
            ],
        }]
        response = self._client.messages.create(
            model=self._model,
            system=system_instruction,
            messages=real_messages,
            max_tokens=4096,
        )
        return response.content[0].text if response.content else ""


# ════════════════════════════════════════════════════════════════
# Vision Providers
# ════════════════════════════════════════════════════════════════


class QwenVLProvider(VisionProvider):
    """Qwen VL (通义千问视觉) Provider — OpenAI 兼容协议"""

    def __init__(self, cfg=None):
        from openai import OpenAI
        cfg = cfg or global_config
        self._client = OpenAI(
            api_key=cfg.qwen_api_key,
            base_url=cfg.qwen_base_url,
        )
        self._model = cfg.get("model_registry.vision.model", "qwen-vl-plus")

    def analyze(self, image_path: str, prompt: str) -> str:
        import base64, os
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        data_url = f"data:{mime_map.get(ext, 'image/jpeg')};base64,{img_data}"

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        )
        return response.choices[0].message.content or ""


class GPT4VProvider(VisionProvider):
    """GPT-4V Provider"""

    def __init__(self, cfg=None):
        from openai import OpenAI
        cfg = cfg or global_config
        self._client = OpenAI(api_key=cfg.openai_api_key, base_url="https://api.openai.com/v1")
        self._model = "gpt-4o"

    def analyze(self, image_path: str, prompt: str) -> str:
        import base64, os
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        data_url = f"data:{mime_map.get(ext, 'image/jpeg')};base64,{img_data}"

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        )
        return response.choices[0].message.content or ""


# ════════════════════════════════════════════════════════════════
# Transcriber Providers
# ════════════════════════════════════════════════════════════════


class FasterWhisperProvider(TranscriberProvider):
    """faster-whisper 本地转录"""

    def __init__(self, cfg=None):
        cfg = cfg or global_config
        wcfg = cfg.whisper_config()
        self._model_size = wcfg["model_size"]
        self._device = wcfg["device"]
        self._compute_type = wcfg["compute_type"]
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size, device=self._device,
                compute_type=self._compute_type,
            )

    def transcribe(self, audio_path: str, language: str = "zh") -> Dict[str, Any]:
        self._load_model()
        segments, info = self._model.transcribe(audio_path, language=language)

        timestamped_lines = []
        raw_parts = []
        seg_list = []

        for seg in segments:
            m, s = divmod(int(seg.start), 60)
            timestamped_lines.append(f"[{m:02d}:{s:02d}] {seg.text}")
            raw_parts.append(seg.text)
            seg_list.append({
                "start": round(seg.start, 1),
                "end": round(seg.end, 1),
                "text": seg.text,
            })

        return {
            "timestamped": "\n".join(timestamped_lines),
            "raw_text": " ".join(raw_parts),
            "segments": seg_list,
            "duration": info.duration,
            "language": info.language,
        }


# ════════════════════════════════════════════════════════════════
# 注册表 + 工厂函数
# ════════════════════════════════════════════════════════════════

# Provider 注册表：类别 → { provider_name → class }
PROVIDER_MAP = {
    "llm": {
        "deepseek": DeepSeekProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    },
    "vision": {
        "qwen-vl": QwenVLProvider,
        "gpt4v": GPT4VProvider,
    },
    "transcriber": {
        "faster-whisper": FasterWhisperProvider,
        # "whisper": WhisperProvider,    # 标准的 whisper（非 faster），暂未实现
    },
}


def get_provider(category: str, cfg=None) -> Any:
    """工厂函数：从 config 中读取 provider 配置，返回对应的 Provider 实例

    Args:
        category: "llm" / "vision" / "transcriber"
        cfg: Config 实例（默认使用全局 config）

    Returns:
        Provider 实例

    Raises:
        ValueError: 未知的 category / provider_name
    """
    cfg = cfg or global_config

    # 从 model_registry 读取 provider 配置
    try:
        registry = cfg.get("model_registry", {})
        cat_cfg = registry.get(category, {})
        provider_name = cat_cfg.get("provider", "")
    except Exception:
        provider_name = ""

    # 无配置时使用默认值
    if not provider_name:
        defaults = {
            "llm": "deepseek",
            "vision": "qwen-vl",
            "transcriber": "faster-whisper",
        }
        provider_name = defaults.get(category, "")
        logger.info(f"[Providers] 使用默认 {category} provider: {provider_name}")

    # 查找注册表
    cat_map = PROVIDER_MAP.get(category, {})
    provider_class = cat_map.get(provider_name)
    if provider_class is None:
        available = ", ".join(cat_map.keys())
        raise ValueError(
            f"未知 {category} provider: '{provider_name}'。"
            f" 可用: {available}"
        )

    logger.info(f"[Providers] 创建 {category} provider: {provider_name}")
    return provider_class(cfg)
