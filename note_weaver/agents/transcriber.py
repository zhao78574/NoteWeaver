"""语音转录 Agent — faster-whisper 封装，从 Auto_Pipeline.py 提取并增强"""

import os
import time
from typing import Any, Dict, List
from faster_whisper import WhisperModel
from .base import BaseAgent
from note_weaver.utils.logger import logger
from note_weaver.utils.config import config


class TranscriberAgent(BaseAgent):
    """faster-whisper 语音识别 Agent"""

    def __init__(self):
        super().__init__()  # 不需要 Gemini 模型，但继承 base 的能力
        self._whisper_model = None
        self._whisper_loaded = False

    def _load_whisper(self):
        """延迟加载 Whisper 模型（全局只加载一次）"""
        if self._whisper_loaded:
            return

        wcfg = config.whisper_config()
        model_size = wcfg["model_size"]
        device = wcfg["device"]
        compute_type = wcfg["compute_type"]

        logger.info(f"加载 Whisper 模型 ({model_size}, {device}, {compute_type})...")
        start = time.time()
        self._whisper_model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        elapsed = time.time() - start
        logger.info(f"Whisper 加载完成 ({elapsed:.1f}s) | device={device} | compute={compute_type}")
        self._whisper_loaded = True

    def execute(
        self,
        audio_path: str,
        language: str = "zh",
    ) -> Dict[str, Any]:
        """
        Args:
            audio_path: 音频文件路径
            language: 语言代码

        Returns:
            {
                "timestamped": "[00:00] 文本...\n[01:23] 文本...",
                "raw_text": "合并后的完整文本",
                "segments": [{"start": 0.0, "end": 12.5, "text": "..."}, ...],
                "duration": 音频总时长(秒),
                "language": "zh",
            }
        """
        self._load_whisper()

        logger.info(f"[Transcriber] 开始转录: {os.path.basename(audio_path)}")

        segments, info = self._whisper_model.transcribe(audio_path, language=language)

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

        result = {
            "timestamped": "\n".join(timestamped_lines),
            "raw_text": " ".join(raw_parts),
            "segments": seg_list,
            "duration": info.duration,
            "language": info.language,
        }

        logger.info(
            f"[Transcriber] 完成: {len(raw_parts)} 字, "
            f"时长 {info.duration:.0f}s, 语言 {info.language}"
        )
        return result
