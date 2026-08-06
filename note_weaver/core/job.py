"""统一任务抽象 — Job = {input, modality, pipeline, output_spec}

所有输入（视频/PDF/网页/文本/重排）统一描述为一个 Job，
Orchestrator.run(job) 根据 modality + pipeline 分发到具体流程。

与 state_machine.Task 的关系：
  - Job = 任务定义（要做什么）
  - Task = 执行追踪（当前做到哪了，状态如何）
"""

from __future__ import annotations

import os
import re
import pathlib
from dataclasses import dataclass, field
from enum import Enum


class Modality(Enum):
    """输入模态"""
    VIDEO = "video"          # .mp4/.mkv/.mov/.avi
    AUDIO = "audio"          # .m4a/.mp3/.wav/.flac/.aac
    PDF = "pdf"              # .pdf
    WEB = "web"              # 普通网页 URL
    TEXT = "text"            # 纯文本/问答
    NOTE = "note"            # 已有 .md 笔记（重排）
    UNKNOWN = "unknown"


class PipelineType(Enum):
    """管线类型"""
    FULL_NOTE = "full_note"                  # 视频→笔记（完整管线）
    PDF_NOTE = "pdf_note"                    # PDF→笔记
    WEB_NOTE = "web_note"                    # 网页→笔记
    REGENERATE = "regenerate"                # 已有笔记重排
    QA_ONLY = "qa_only"                      # 仅问答（不生成笔记）
    BATCH_VIDEO = "batch_video"              # 批量视频
    BATCH_REGENERATE = "batch_regenerate"    # 批量重排
    MERGE_NOTES = "merge_notes"              # 合集合并笔记


@dataclass
class OutputSpec:
    """输出规格说明"""
    format: str = "markdown"          # markdown / html / json
    style: str = "detailed"           # detailed / outline / step_by_step / meeting_minutes
    language: str = "zh"              # zh / en


@dataclass
class Job:
    """统一任务定义

    所有命令的入口统一收敛为 Job，然后 Orchestrator.run(job) 自动分发。
    """
    input: str                          # 文件路径 / URL / 文本
    modality: Modality = Modality.UNKNOWN
    pipeline: PipelineType = PipelineType.QA_ONLY
    output_spec: OutputSpec = field(default_factory=OutputSpec)
    metadata: dict = field(default_factory=dict)

    # 以下字段由 Orchestrator 在执行过程中填充（非 Job 创建时指定）
    file_base: str = ""
    rel_subdir: str = ""

    @classmethod
    def from_input(cls, user_input: str) -> "Job":
        """从用户输入自动推断 Job 参数

        这是 run.py 所有 cmd_* 的收敛入口。
        返回 (Job, 是否识别成功)。
        """
        text = user_input.strip()
        if not text:
            return Job(input=text, modality=Modality.TEXT,
                       pipeline=PipelineType.QA_ONLY)

        # ── 辅助判断 ──
        _VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}
        _AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".aac"}
        _MEDIA_EXTS = _VIDEO_EXTS | _AUDIO_EXTS
        _PDF_EXTS = {".pdf"}
        _NOTE_EXTS = {".md"}

        def _ext(path: str) -> str:
            return os.path.splitext(path)[1].lower()

        def _exists(path: str) -> bool:
            return pathlib.Path(path.strip("\"' ")).exists()

        def _resolve(path: str) -> str:
            return str(pathlib.Path(path.strip("\"' ")).resolve())

        def _is_url(t: str) -> bool:
            low = t.lower()
            return (("youtube.com" in low or "youtu.be" in low
                     or "bilibili.com" in low or "b23.tv" in low)
                    and low.startswith("http"))

        def _is_playlist_url(t: str) -> bool:
            """检测是否为合集/播放列表 URL"""
            low = t.lower()
            # YouTube playlist
            if "youtube.com/playlist" in low or "youtu.be" in low:
                if "list=" in low:
                    return True
            # Bilibili 合集/系列
            if "bilibili.com/list/" in low or "bilibili.com/playlist" in low:
                return True
            if "b23.tv" in low:
                # b23.tv 短链接可能是合集，但无法从URL直接判断
                return False  # 保守起见，b23.tv 统一走单视频
            return False

        def _is_web_url(t: str) -> bool:
            return t.lower().startswith("http") and not _is_url(t)

        def _extract_url(t: str) -> str:
            """从文本中提取 URL，遇到中文/空白/标点自动截断"""
            m = re.search(
                r'https?://[^\s,;)\]}"\''
                r'一-鿿　-〿＀-￯'
                r'，。、；：？！）】」』》【】]+',
                t,
            )
            return m.group(0).rstrip(".,;") if m else ""

        def _extract_path_from_text(t: str):
            """从自然语言中提取路径，返回 (path, is_file) 或 None"""
            # 引号内
            for q in ('"', "“", "'"):
                if q in t:
                    parts = t.split(q)
                    for i, part in enumerate(parts):
                        if i % 2 == 1 and part.strip():
                            p = pathlib.Path(part.strip())
                            if p.exists():
                                return str(p.resolve()), p.is_file()
            # 盘符路径
            for m in re.finditer(r'([A-Za-z]:[\\/][^\s,;)\]}"\']+)', t):
                p = pathlib.Path(m.group(1).strip())
                if p.exists():
                    return str(p.resolve()), p.is_file()
            # 相对路径
            for m in re.finditer(r'([.][.\/\\][^\s,;)\]}"\']+)', t):
                p = pathlib.Path(m.group(1).strip())
                if p.exists():
                    return str(p.resolve()), p.is_file()
            return None

        # ── 先处理特殊命令 ──
        if text.lower() in ("graph", "/graph", "/kg", "知识图谱"):
            return Job(input=text, modality=Modality.TEXT,
                       pipeline=PipelineType.QA_ONLY,
                       metadata={"command": "graph"})

        if text.lower() in ("/stats", "统计", "学习统计", "仪表盘"):
            return Job(input=text, modality=Modality.TEXT,
                       pipeline=PipelineType.QA_ONLY,
                       metadata={"command": "stats"})

        # ── "重排" 命令 ──
        is_regenerate = any(text.startswith(kw) for kw in ("重排", "排版", "regenerate"))
        if is_regenerate:
            search = text
            for kw in ("重排", "排版", "regenerate"):
                search = search.replace(kw, "", 1).strip()
            if search and _exists(search):
                p = _resolve(search)
                ext = _ext(p)
                if ext in _NOTE_EXTS:
                    return Job(input=p, modality=Modality.NOTE,
                               pipeline=PipelineType.REGENERATE)
                if os.path.isdir(p):
                    return Job(input=p, modality=Modality.NOTE,
                               pipeline=PipelineType.BATCH_REGENERATE)
            # 自然语言搜索笔记
            return Job(input=search, modality=Modality.NOTE,
                       pipeline=PipelineType.REGENERATE,
                       metadata={"search_mode": True, "raw_input": text})

        # ── "合集/playlist" 命令 ──
        if any(text.startswith(kw) for kw in ("合集 ", "playlist ")):
            target = text.split(" ", 1)[1].strip().strip("\"'")
            if _is_playlist_url(target):
                return Job(input=target, modality=Modality.VIDEO,
                           pipeline=PipelineType.MERGE_NOTES,
                           metadata={"source": "playlist_url"})
            if _is_url(target):
                return Job(input=target, modality=Modality.VIDEO,
                           pipeline=PipelineType.MERGE_NOTES,
                           metadata={"source": "playlist_url", "force_merge": True})
            # 也可能是目录路径
            if _exists(target):
                p = _resolve(target)
                if os.path.isdir(p):
                    return Job(input=p, modality=Modality.VIDEO,
                               pipeline=PipelineType.MERGE_NOTES,
                               metadata={"source": "local_directory", "force_merge": True})
            err_msg = "请提供有效的合集链接或目录路径"
            return Job(input=text, modality=Modality.TEXT,
                       pipeline=PipelineType.QA_ONLY,
                       metadata={"error": err_msg})

        # ── "from / 下载" 命令 ──
        if text.lower().startswith("from ") or text.lower().startswith("下载 "):
            target = text.split(" ", 1)[1].strip().strip("\"'")
            if _is_playlist_url(target):
                # 合集链接默认逐集单独处理；需合并请用「合集 <url>」
                return Job(input=target, modality=Modality.VIDEO,
                           pipeline=PipelineType.BATCH_VIDEO,
                           metadata={"source": "playlist_url"})
            if _is_url(target):
                return Job(input=target, modality=Modality.VIDEO,
                           pipeline=PipelineType.FULL_NOTE,
                           metadata={"source": "video_url"})
            if _is_web_url(target):
                return Job(input=target, modality=Modality.WEB,
                           pipeline=PipelineType.WEB_NOTE)
            if _exists(target) and _ext(target) in _PDF_EXTS:
                return Job(input=_resolve(target), modality=Modality.PDF,
                           pipeline=PipelineType.PDF_NOTE)
            url = _extract_url(text)
            if url:
                modality = Modality.VIDEO if _is_url(url) else Modality.WEB
                pipeline = PipelineType.FULL_NOTE if modality == Modality.VIDEO else PipelineType.WEB_NOTE
                return Job(input=url, modality=modality, pipeline=pipeline,
                           metadata={"source": "url_from_text"})
            return Job(input=text, modality=Modality.TEXT,
                       pipeline=PipelineType.QA_ONLY,
                       metadata={"error": "请提供有效的链接或 PDF 路径"})

        # ── 显式路径 ──
        if _exists(text):
            p = _resolve(text)
            ext = _ext(p)
            if ext in _VIDEO_EXTS:
                return Job(input=p, modality=Modality.VIDEO,
                           pipeline=PipelineType.FULL_NOTE)
            if ext in _AUDIO_EXTS:
                return Job(input=p, modality=Modality.AUDIO,
                           pipeline=PipelineType.FULL_NOTE)
            if ext in _PDF_EXTS:
                return Job(input=p, modality=Modality.PDF,
                           pipeline=PipelineType.PDF_NOTE)
            if ext in _NOTE_EXTS:
                return Job(input=p, modality=Modality.NOTE,
                           pipeline=PipelineType.REGENERATE)
            if os.path.isdir(p):
                # 检测目录内容类型
                has_media = any(f.suffix.lower() in _MEDIA_EXTS for f in pathlib.Path(p).iterdir())
                has_notes = any(f.suffix.lower() in _NOTE_EXTS for f in pathlib.Path(p).iterdir())
                if has_media:
                    return Job(input=p, modality=Modality.VIDEO,
                               pipeline=PipelineType.BATCH_VIDEO)
                if has_notes:
                    return Job(input=p, modality=Modality.NOTE,
                               pipeline=PipelineType.BATCH_REGENERATE)
            return Job(input=p, modality=Modality.UNKNOWN,
                       pipeline=PipelineType.QA_ONLY)

        # ── 自然语言中提取路径 ──
        extracted = _extract_path_from_text(text)
        if extracted:
            path, is_file = extracted
            ext = _ext(path)
            if is_file:
                if ext in _MEDIA_EXTS:
                    return Job(input=path, modality=Modality.VIDEO if ext in _VIDEO_EXTS else Modality.AUDIO,
                               pipeline=PipelineType.FULL_NOTE)
                if ext in _PDF_EXTS:
                    return Job(input=path, modality=Modality.PDF,
                               pipeline=PipelineType.PDF_NOTE)
                if ext in _NOTE_EXTS:
                    return Job(input=path, modality=Modality.NOTE,
                               pipeline=PipelineType.REGENERATE)
            else:
                has_media = any(f.suffix.lower() in _MEDIA_EXTS for f in pathlib.Path(path).iterdir())
                has_notes = any(f.suffix.lower() in _NOTE_EXTS for f in pathlib.Path(path).iterdir())
                if has_media:
                    return Job(input=path, modality=Modality.VIDEO,
                               pipeline=PipelineType.BATCH_VIDEO)
                if has_notes:
                    return Job(input=path, modality=Modality.NOTE,
                               pipeline=PipelineType.BATCH_REGENERATE)

        # ── URL 识别 ──
        url = _extract_url(text)
        if url:
            if _is_playlist_url(url):
                # 默认情况：合集链接逐集单独处理（不走合并）
                # 需要合并请用「合集 <url>」命令
                return Job(input=url, modality=Modality.VIDEO,
                           pipeline=PipelineType.BATCH_VIDEO,
                           metadata={"source": "playlist_url"})
            if _is_url(url):
                return Job(input=url, modality=Modality.VIDEO,
                           pipeline=PipelineType.FULL_NOTE,
                           metadata={"source": "video_url"})
            if _is_web_url(url):
                return Job(input=url, modality=Modality.WEB,
                           pipeline=PipelineType.WEB_NOTE)

        # ── 兜底：问答 ──
        return Job(input=text, modality=Modality.TEXT,
                   pipeline=PipelineType.QA_ONLY)

    def to_dict(self) -> dict:
        return {
            "input": self.input[:100],
            "modality": self.modality.value,
            "pipeline": self.pipeline.value,
            "output_spec": {
                "format": self.output_spec.format,
                "style": self.output_spec.style,
                "language": self.output_spec.language,
            },
            "metadata": {k: v for k, v in self.metadata.items()
                         if isinstance(v, (str, int, float, bool))},
        }
