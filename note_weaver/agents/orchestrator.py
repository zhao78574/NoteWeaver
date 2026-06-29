"""中央调度 Agent — 任务编排 + 状态管理 + 异常处理

用法:
    orchestrator = Orchestrator()

    # 统一入口（推荐）
    job = Job.from_input("lecture.mp4")
    result = orchestrator.run(job)

    # 向后兼容入口
    task = orchestrator.process_video("lecture.mp4")
"""

import os
import re
import sys
import json
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from note_weaver.core.state_machine import Task, TaskStatus, TaskStateMachine
from note_weaver.core.job import Job, Modality, PipelineType, OutputSpec
from note_weaver.core.cache import PipelineCache
from note_weaver.core.extractor import (
    extract_audio,
    extract_screenshots,
    extract_keyframes,
    clean_screenshot_dir,
    get_video_duration,
)
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger
from note_weaver.utils.style import console

from .classifier import ClassifierAgent
from .transcriber import TranscriberAgent
from .vision import VisionAgent
from .composer import ComposerAgent
from .qa import QAAgent
from .memory_agent import MemoryAgent
from .router import RouterAgent
from .corrector import CorrectorAgent
from .policy import PolicyEngine
from .keyword_manager import KeywordManager                  # 新增


class Orchestrator:
    """NoteWeaver 中央调度器 — 管理整个笔记生成流程"""

    def __init__(self):
        # 基础设施
        self.state_machine = TaskStateMachine()

        # Agent 实例（懒加载）
        self._classifier: Optional[ClassifierAgent] = None
        self._transcriber: Optional[TranscriberAgent] = None
        self._vision: Optional[VisionAgent] = None
        self._composer: Optional[ComposerAgent] = None
        self._qa: Optional[QAAgent] = None
        self._memory: Optional[MemoryAgent] = None
        self._router: Optional[RouterAgent] = None
        self._corrector: Optional[CorrectorAgent] = None
        self._policy: Optional[PolicyEngine] = None
        self._keyword_mgr: Optional[KeywordManager] = None     # 新增

        # 缓存系统
        self.cache = PipelineCache()

        # 确保输出目录存在
        for d in [config.txt_dir,
                   config.note_dir, config.memory_dir, config.log_dir]:
            os.makedirs(d, exist_ok=True)

        logger.info("[Orchestrator] NoteWeaver 初始化完成")

    # =================================================================
    # 统一入口
    # =================================================================

    def run(self, job: Job) -> dict:
        """统一任务入口 — 根据 Job.modality + Job.pipeline 分发到具体流程

        Args:
            job: 任务定义（由 Job.from_input() 创建）

        Returns:
            结果字典，含 output_paths/qa_score/stats 等
        """
        logger.info(f"[Orchestrator] 接收任务: {job.to_dict()}")

        # ── 特殊命令（非管线流程） ──
        cmd = job.metadata.get("command", "")
        if cmd == "graph":
            return self._run_graph()
        if cmd == "stats":
            return self._run_stats()

        # ── 按 PipelineType 分发 ──
        if job.pipeline == PipelineType.FULL_NOTE:
            return self._run_full_note_dag(job)

        elif job.pipeline == PipelineType.PDF_NOTE:
            ctx = {"job": job}
            try:
                results = self._run_dag(PipelineType.PDF_NOTE, ctx)
                return {"ok": True, "type": "pdf",
                        "note_path": results.get("save", {}).get("note_path", "")}
            except Exception as e:
                logger.error(f"[DAG] PDF_NOTE 失败: {e}")
                return {"ok": False, "type": "pdf", "error": str(e)}

        elif job.pipeline == PipelineType.WEB_NOTE:
            ctx = {"job": job}
            try:
                results = self._run_dag(PipelineType.WEB_NOTE, ctx)
                return {"ok": True, "type": "web",
                        "note_path": results.get("save", {}).get("note_path", "")}
            except Exception as e:
                logger.error(f"[DAG] WEB_NOTE 失败: {e}")
                return {"ok": False, "type": "web", "error": str(e)}

        elif job.pipeline == PipelineType.REGENERATE:
            if job.metadata.get("search_mode"):
                return self._run_search_regenerate(job)
            note_path = self._regenerate_note(job)
            return {"ok": True, "type": "regenerate", "note_path": note_path}

        elif job.pipeline == PipelineType.BATCH_VIDEO:
            return self._run_batch_video(job)

        elif job.pipeline == PipelineType.BATCH_REGENERATE:
            return self._run_batch_regenerate(job)

        elif job.pipeline == PipelineType.MERGE_NOTES:
            return self._run_merge_notes(job)

        elif job.pipeline == PipelineType.QA_ONLY:
            return self._run_qa(job)

        else:
            logger.warning(f"[Orchestrator] 未知 Pipeline: {job.pipeline}")
            return {"ok": False, "error": f"未知管线类型: {job.pipeline}"}

    @staticmethod
    def _task_to_result(task, job) -> dict:
        """将 process_video 返回的 Task 对象转为统一结果字典"""
        if task is None or task.status.value == "failed":
            return {
                "ok": False,
                "type": "process_video",
                "error": task.error_message if task else "处理失败",
                "task_id": task.task_id if task else None,
            }
        return {
            "ok": True,
            "type": "process_video",
            "qa_score": task.qa_score,
            "note_path": task.md_path,
            "txt_path": task.txt_path,
            "elapsed": f"{task.elapsed_seconds:.0f}s",
            "task_id": task.task_id,
        }

    # =================================================================
    # 子流程分发
    # =================================================================

    def _run_graph(self) -> dict:
        """生成知识图谱"""
        try:
            self._regenerate_graph_on_disk()
            return {"ok": True, "type": "graph", "message": "知识图谱已生成"}
        except Exception as e:
            return {"ok": False, "type": "graph", "error": str(e)}

    def _run_stats(self) -> dict:
        """获取学习统计"""
        return {"ok": True, "type": "stats", "data": self.get_stats()}

    def _run_qa(self, job: Job) -> dict:
        """问答模式"""
        from note_weaver.skills.chat import chat as chat_notes
        from note_weaver.skills.search import search as search_notes
        config.setup_proxy()
        result = chat_notes(job.input)
        return {"ok": True, "type": "qa", "data": result}

    def _process_pdf(self, job: Job) -> dict:
        """PDF→笔记"""
        from note_weaver.utils.extractors import extract_from_pdf
        from note_weaver.agents.vision import VisionAgent
        from note_weaver.agents.composer import ComposerAgent

        config.setup_proxy()
        p = pathlib.Path(job.input)
        file_base = p.stem.replace(' ', '_')
        note_category = "pdf"
        base_dir = pathlib.Path(config.base_dir)
        img_dir = base_dir / "data" / "Note" / note_category / file_base
        note_dir = base_dir / "data" / "Note" / note_category

        logger.info(f"[PDF] 提取: {p.name}")
        result = extract_from_pdf(job.input, output_dir=str(img_dir))
        logger.info(f"  文本: {len(result['text'])}字 | 图片: {len(result['images'])}张")

        vision_results = []
        if result["images"]:
            logger.info(f"[PDF] 提取图片 {len(result['images'])} 张 → Vision 分析...")
            vision_results = VisionAgent().execute(result["images"])
            included = sum(1 for r in vision_results if r.get("should_include", True))
            logger.info(f"[Vision] {included} 张采纳 / {len(vision_results) - included} 张过滤")

        composer = ComposerAgent()
        note_content = composer.execute(
            file_base=file_base,
            timestamped_text=result["text"][:15000],
            vision_results=vision_results,
            strategy={"note_style": "detailed", "focus_areas": []},
            revision_feedback="这是从PDF提取的内容，整理成结构化学习笔记。",
        )
        note_path = composer.save_note(file_base, note_content, str(note_dir))
        self._export_outputs(note_path)
        self._rebuild_indexes()
        return {"ok": True, "type": "pdf", "note_path": note_path}

    def _process_web(self, job: Job) -> dict:
        """网页→笔记"""
        from note_weaver.utils.extractors import extract_from_url
        from note_weaver.agents.vision import VisionAgent
        from note_weaver.agents.composer import ComposerAgent

        config.setup_proxy()
        import urllib.parse
        parsed = urllib.parse.urlparse(job.input)
        domain = parsed.netloc.replace("www.", "")
        file_base = re.sub(r'[^\w一-鿿]+', '_', domain)[:30]
        base_dir = pathlib.Path(config.base_dir)
        note_category = "web"
        img_dir = base_dir / "data" / "Note" / note_category / file_base
        note_dir = base_dir / "data" / "Note" / note_category

        logger.info(f"[Web] 提取: {job.input}")
        result = extract_from_url(job.input, output_dir=str(img_dir))
        logger.info(f"  标题: {result.get('title','?')[:60]} | 文本: {len(result['text'])}字")

        vision_results = []
        if result.get("images"):
            vision_results = VisionAgent().execute(result["images"])

        composer = ComposerAgent()
        note_content = composer.execute(
            file_base=file_base,
            timestamped_text=result["text"][:15000],
            vision_results=vision_results,
            strategy={"note_style": "detailed", "focus_areas": []},
            revision_feedback=f"来源网页: {job.input}\n整理成结构化学习笔记。",
        )
        note_path = composer.save_note(file_base, note_content, str(note_dir))
        self._export_outputs(note_path)
        self._rebuild_indexes()
        return {"ok": True, "type": "web", "note_path": note_path}

    def _run_search_regenerate(self, job: Job) -> dict:
        """自然语言搜索笔记后重排"""
        from note_weaver.run import _search_and_regenerate
        _search_and_regenerate(job.input, job.metadata.get("raw_input", job.input))
        return {"ok": True, "type": "regenerate", "message": f"搜索重排: {job.input}"}

    def _regenerate_note(self, job: Job) -> str:
        """单个笔记重排"""
        from note_weaver.run import cmd_regenerate_note
        cmd_regenerate_note(job.input)
        return job.input

    def _run_batch_video(self, job: Job) -> dict:
        """批量处理视频目录"""
        from note_weaver.run import cmd_batch
        cmd_batch(job.input)
        return {"ok": True, "type": "batch_video", "directory": job.input}

    def _run_batch_regenerate(self, job: Job) -> dict:
        """批量重排目录"""
        from note_weaver.run import cmd_batch_regenerate
        cmd_batch_regenerate(job.input)
        return {"ok": True, "type": "batch_regenerate", "directory": job.input}

    # =================================================================
    # 合集合并笔记
    # =================================================================

    def _run_merge_from_videos(
        self,
        playlist_title: str,
        videos: list,
        original_url: str = "",
        selected_indices: list = None,
    ) -> dict:
        """从已过滤的视频列表直接执行合并（跳过 playlist 检测 + 用户选择）

        供 run.py 的 cmd_merge_playlist 交互选择后调用。
        如果提供了 original_url + selected_indices，使用 yt-dlp 原生
        playlist_items 下载（更可靠），否则回退到逐个 URL 下载。

        Args:
            playlist_title: 合集标题
            videos: [{"title", "duration", "url", "video_id", "index"}, ...]
            original_url: 原始合集 URL（可选，推荐提供）
            selected_indices: 选中的视频序号列表 [1-based]（可选，与 original_url 配合）

        Returns:
            同 _run_merge_notes
        """
        from note_weaver.agents.downloader import VideoDownloader
        config.setup_proxy()

        dl = VideoDownloader()

        # ── 1) 批量下载（优先用 yt-dlp 原生 playlist_items） ──
        dl_results = []
        if original_url and selected_indices:
            logger.info(f"[Merge] 原生下载合集: {original_url} items={selected_indices}")
            dl_results = dl.download_playlist_range(
                original_url, selected_indices, quality=720,
            )
        else:
            # 回退：逐个 URL 下载
            dl_results = dl.download_playlist_videos(
                {"title": playlist_title, "playlist_count": len(videos), "videos": videos},
                quality=720,
            )

        # ── 2) 逐条处理 ──
        merge_entries = []
        # 安全检查：确保 results 数量匹配
        if len(dl_results) != len(videos):
            logger.warning(f"[Merge] results({len(dl_results)}) 与 videos({len(videos)}) 数量不匹配")
        for vid_info, dl_result in zip(videos, dl_results):
            local_path = dl_result.get("local_path", "")
            error_msg = dl_result.get("error", "")
            if not local_path or not os.path.isfile(local_path):
                reason = error_msg or "无本地文件"
                logger.warning(f"[Merge] 跳过 {vid_info['title']}: {reason}")
                if reason:
                    console.print(f"  [dim]✗ {vid_info.get('title', '?')}: {reason}[/dim]")
                continue

            try:
                job_single = Job(
                    input=local_path,
                    modality=Modality.VIDEO,
                    pipeline=PipelineType.FULL_NOTE,
                )
                result = self._run_full_note_dag(job_single)
                note_content = ""
                task = None
                for t in list(self.state_machine.history):
                    if t.video_path == local_path or t.file_name == os.path.basename(local_path):
                        task = t
                        break
                # 取笔记内容，并标记要删除的单集 .md
                md_to_delete = ""
                if task and task.note_content:
                    note_content = task.note_content
                    md_to_delete = task.md_path or ""
                elif result.get("note_path"):
                    np = result["note_path"]
                    if os.path.isfile(np):
                        with open(np, "r", encoding="utf-8") as f:
                            note_content = f.read()
                        md_to_delete = np
                if md_to_delete and os.path.isfile(md_to_delete):
                    os.remove(md_to_delete)

                # 获取截图目录路径（直接从 task 拿，最可靠）
                ss_dir = task.screenshot_dir if task and task.screenshot_dir else ""
                if not ss_dir and md_to_delete:
                    # 兜底：按 file_base 推测
                    fb = os.path.splitext(os.path.basename(md_to_delete))[0]
                    ss_dir = os.path.join(os.path.dirname(md_to_delete), fb)
                merge_entries.append({
                    "title": vid_info["title"],
                    "duration": vid_info["duration"],
                    "note_content": note_content,
                    "url": vid_info.get("url", ""),
                    "index": vid_info.get("index", 0),
                    "screenshot_dir": ss_dir,  # 截图目录绝对路径，不靠猜
                })
            except Exception as e:
                logger.error(f"[Merge] 处理失败: {vid_info['title']}: {e}")

        # ── 3) 合并笔记 ──
        output_paths = []
        merge_groups = self._split_merge_groups(merge_entries)
        for idx, group in enumerate(merge_groups):
            group_title = playlist_title if len(merge_groups) == 1 else f"{playlist_title}（第{idx+1}部分）"
            merged_content = self.composer.merge_notes(group, group_title)
            safe_title = re.sub(r'[\\/:*?"<>|\s]+', '_', playlist_title).strip('_')[:40]
            range_str = self._get_merge_range_str(group)
            file_base = f"{safe_title}_合集_{range_str}"
            if len(merge_groups) > 1:
                file_base = f"{safe_title}_合集_{range_str}_part{idx+1}"

            note_output_dir = os.path.join(config.note_dir, "bilibili")
            os.makedirs(note_output_dir, exist_ok=True)

            # 统一所有截图到合集图片目录，并更新引用路径
            images_dir = os.path.join(note_output_dir, f"{safe_title}_合集_{range_str}_images")
            merged_content = self._consolidate_merge_images(
                merged_content, group, images_dir,
            )

            md_path = self.composer.save_note(file_base, merged_content, note_output_dir)
            output_paths.append(md_path)
            logger.info(f"[Merge] 合集笔记已保存: {md_path}")

        self._rebuild_indexes()

        return {
            "ok": True,
            "type": "merge_notes",
            "playlist_title": playlist_title,
            "total_videos": len(videos),
            "merged_count": len(merge_entries),
            "output_paths": output_paths,
        }

    def _run_merge_notes(self, job: Job) -> dict:
        """合集/播放列表合并笔记 — 完整流程

        只应在用户明确要求合并时调用（`合集 <url>` 命令）。
        所有视频无条件合并为一篇（或多篇，超长合集自动切分）笔记。

        1. detect_playlist → 获取所有视频列表
        2. download_all → 批量下载
        3. process_each → 逐条转写+笔记
        4. merge_notes → 合并为一篇笔记
        """
        from note_weaver.agents.downloader import VideoDownloader

        url = job.input
        config.setup_proxy()

        # ── 1) 检测合集 ──
        logger.info(f"[Merge] 检测合集: {url}")
        dl = VideoDownloader()
        playlist_info = dl.detect_playlist(url)
        if not playlist_info:
            return {"ok": False, "error": "未能识别为合集/播放列表，或不支持的平台"}
        playlist_title = playlist_info["title"]
        videos = playlist_info["videos"]
        logger.info(
            f"[Merge] 合集「{playlist_title}」: {len(videos)} 个视频"
        )

        # ── 2) 批量下载 ──
        max_videos = job.metadata.get("max_videos")
        videos_to_download = videos[:max_videos] if max_videos else videos

        dl_results = dl.download_playlist_videos(
            {"title": playlist_title, "playlist_count": len(videos_to_download), "videos": videos_to_download},
            quality=720, max_count=max_videos,
        )

        # ── 3) 逐条处理，全部进入合并池 ──
        merge_entries = []
        for vid_info, dl_result in zip(videos_to_download, dl_results):
            local_path = dl_result.get("local_path", "")
            if not local_path or not os.path.isfile(local_path):
                logger.warning(f"[Merge] 跳过 {vid_info['title']}: 无本地文件")
                continue

            try:
                job_single = Job(
                    input=local_path,
                    modality=Modality.VIDEO,
                    pipeline=PipelineType.FULL_NOTE,
                )
                result = self._run_full_note_dag(job_single)
                note_content = ""
                # 从 task 中提取笔记内容
                task = None
                for t in list(self.state_machine.history):
                    if t.video_path == local_path or t.file_name == os.path.basename(local_path):
                        task = t
                        break
                # 取笔记内容，并标记要删除的单集 .md
                md_to_delete = ""
                if task and task.note_content:
                    note_content = task.note_content
                    md_to_delete = task.md_path or ""
                elif result.get("note_path"):
                    np = result["note_path"]
                    if os.path.isfile(np):
                        with open(np, "r", encoding="utf-8") as f:
                            note_content = f.read()
                        md_to_delete = np
                if md_to_delete and os.path.isfile(md_to_delete):
                    os.remove(md_to_delete)

                # 获取截图目录路径（直接从 task 拿，最可靠）
                ss_dir = task.screenshot_dir if task and task.screenshot_dir else ""
                if not ss_dir and md_to_delete:
                    # 兜底：按 file_base 推测
                    fb = os.path.splitext(os.path.basename(md_to_delete))[0]
                    ss_dir = os.path.join(os.path.dirname(md_to_delete), fb)
                merge_entries.append({
                    "title": vid_info["title"],
                    "duration": vid_info["duration"],
                    "note_content": note_content,
                    "url": vid_info.get("url", ""),
                    "index": vid_info.get("index", 0),
                    "screenshot_dir": ss_dir,  # 截图目录绝对路径，不靠猜
                })

            except Exception as e:
                logger.error(f"[Merge] 处理失败: {vid_info['title']}: {e}")

        # ── 4) 合并笔记（超长合集自动切分） ──
        output_paths = []
        merge_groups = self._split_merge_groups(merge_entries)
        for idx, group in enumerate(merge_groups):
            group_title = playlist_title if len(merge_groups) == 1 else f"{playlist_title}（第{idx+1}部分）"
            merged_content = self.composer.merge_notes(group, group_title)
            safe_title = re.sub(r'[\\/:*?"<>|\s]+', '_', playlist_title).strip('_')[:40]
            range_str = self._get_merge_range_str(group)
            file_base = f"{safe_title}_合集_{range_str}"
            if len(merge_groups) > 1:
                file_base = f"{safe_title}_合集_{range_str}_part{idx+1}"

            note_output_dir = os.path.join(config.note_dir, "bilibili")
            os.makedirs(note_output_dir, exist_ok=True)

            # 统一所有截图到合集图片目录，并更新引用路径
            images_dir = os.path.join(note_output_dir, f"{safe_title}_合集_{range_str}_images")
            merged_content = self._consolidate_merge_images(
                merged_content, group, images_dir,
            )

            md_path = self.composer.save_note(file_base, merged_content, note_output_dir)
            output_paths.append(md_path)
            logger.info(f"[Merge] 合集笔记已保存: {md_path}")

        # ── 后处理 ──
        self._rebuild_indexes()

        return {
            "ok": True,
            "type": "merge_notes",
            "playlist_title": playlist_title,
            "total_videos": len(videos),
            "merged_count": len(merge_entries),
            "output_paths": output_paths,
        }

    @staticmethod
    def _get_merge_range_str(entries: list) -> str:
        """从合并条目中提取序号范围，如 "1-10" 或 "1" """
        indices = [e.get("index", 0) for e in entries if e.get("index")]
        if not indices:
            return "合集"
        indices = sorted(set(indices))
        if len(indices) == 1:
            return str(indices[0])
        if indices[-1] - indices[0] + 1 == len(indices):
            return f"{indices[0]}-{indices[-1]}"
        return ",".join(str(i) for i in indices)

    @staticmethod
    def _split_merge_groups(entries: list) -> list:
        """超长合集按总时长上限切分为多组（安全阀）

        当 max_total ≤ 0 时，全部合为一组不切分。
        """
        if not entries:
            return []

        max_total = config.merge_max_total_duration
        if max_total <= 0:
            return [list(entries)]

        groups = []
        current_group = []
        current_total = 0

        for e in entries:
            dur = e.get("duration", 0)
            if current_total + dur > max_total and current_group:
                groups.append(current_group)
                current_group = [e]
                current_total = dur
            else:
                current_group.append(e)
                current_total += dur

        if current_group:
            groups.append(current_group)

        return groups

    @staticmethod
    def _consolidate_merge_images(
        merged_content: str,
        merge_entries: list,
        images_dir: str,
    ) -> str:
        """将所有单集的截图统一复制到一个文件夹，并更新合并笔记中的引用路径

        Args:
            merged_content: 合并后的 Markdown 内容
            merge_entries: 每集信息（含 screenshot_dir 等）
            images_dir: 目标图片文件夹路径

        Returns:
            更新图片路径后的合并笔记内容
        """
        if not merge_entries or not images_dir:
            return merged_content

        os.makedirs(images_dir, exist_ok=True)
        images_dir_name = os.path.basename(images_dir)

        consolidated = False

        for entry in merge_entries:
            ss_dir = entry.get("screenshot_dir", "")
            if not ss_dir or not os.path.isdir(ss_dir):
                continue

            # 获取该目录下所有图片
            images = sorted(
                f for f in os.listdir(ss_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))
            )
            if not images:
                continue

            # 复制到合集目录
            copied = 0
            for img_name in images:
                dst = os.path.join(images_dir, img_name)
                if not os.path.isfile(dst):
                    shutil.copy2(os.path.join(ss_dir, img_name), dst)
                    copied += 1

            if copied:
                logger.info(f"[Merge] 合并截图: {os.path.basename(ss_dir)} → {images_dir_name} ({copied}张)")
                consolidated = True

            # 更新合并笔记中的图片路径：旧目录名/ → 合集目录名/
            old_dir_name = os.path.basename(ss_dir)
            if old_dir_name:
                merged_content = merged_content.replace(
                    f"{old_dir_name}/", f"{images_dir_name}/"
                )

            # 删除旧的零散截图目录
            shutil.rmtree(ss_dir, ignore_errors=True)
            logger.info(f"[Merge] 删除旧截图目录: {os.path.basename(ss_dir)}")

        if not consolidated:
            logger.warning("[Merge] 未找到任何截图目录，请检查 merge_entries 中的 screenshot_dir")
            # 兜底：扫描笔记目录，找遗留的零散截图目录
            logger.info("[Merge] 尝试兜底扫描截图目录...")
            note_dir = config.note_dir
            indices = set(e.get("index", 0) for e in merge_entries if e.get("index"))
            if indices and os.path.isdir(note_dir):
                for root, dirs, unused in os.walk(note_dir):
                    for d in sorted(dirs):
                        for idx in indices:
                            if d.startswith(f"{idx:03d}_") or d.startswith(str(idx)):
                                ss_dir = os.path.join(root, d)
                                if not os.path.isdir(ss_dir):
                                    continue
                                images = [f for f in os.listdir(ss_dir)
                                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
                                if not images:
                                    continue
                                copied = 0
                                for img_name in images:
                                    dst = os.path.join(images_dir, img_name)
                                    if not os.path.isfile(dst):
                                        shutil.copy2(os.path.join(ss_dir, img_name), dst)
                                        copied += 1
                                if copied:
                                    logger.info(f"[Merge] 兜底合并: {d} -> {images_dir_name} ({copied}张)")
                                    consolidated = True
                                merged_content = merged_content.replace(f"{d}/", f"{images_dir_name}/")
                                shutil.rmtree(ss_dir, ignore_errors=True)
                                logger.info(f"[Merge] 兜底删除旧目录: {d}")
                                break
            if not consolidated:
                logger.warning("[Merge] 兜底也未找到截图目录")

        return merged_content

    def _regenerate_graph_on_disk(self):
        """重新生成知识图谱 HTML 文件"""
        script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "weaver_graph.py"
        if script.exists():
            import subprocess
            try:
                subprocess.run(
                    [sys.executable, str(script), "--output",
                     str(pathlib.Path(config.memory_dir) / "knowledge_graph.html")],
                    capture_output=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                logger.warning("[Graph] 图谱生成超时")
            except Exception as e:
                logger.warning(f"[Graph] 图谱生成异常: {e}")

    def _rebuild_indexes(self):
        """后处理：重建索引 + 图谱"""
        self._regenerate_graph_on_disk()
        try:
            from note_weaver.utils.embeddings import EmbeddingIndex
            count = EmbeddingIndex().build(force=True)
            if count:
                logger.info(f"[Embedding] 索引重建完成: {count} 条")
        except Exception as e:
            logger.warning(f"[Embedding] 索引重建失败（非致命）: {e}")

    def _export_outputs(self, note_path: str):
        """笔记完成后生成三种输出形态的元数据

        在笔记目录下生成 _outputs.json，供 CLI/Web UI 展示三态输出入口。
        """
        if not note_path:
            return
        note_dir = pathlib.Path(note_path).parent
        outputs = {
            "markdown": note_path,
            "generated_at": datetime.now().isoformat(),
            "outputs": {
                "markdown": {"path": note_path, "label": "阅读模式"},
                "graph": {"path": str(note_dir / "_knowledge_graph.html"),
                          "label": "图谱模式",
                          "command": "weaver graph"},
                "qa": {"label": "问答模式",
                       "command": "weaver ask \"你的问题\""},
            }
        }
        try:
            with open(note_dir / "_outputs.json", "w", encoding="utf-8") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"导出 _outputs.json 失败: {e}")

    # =================================================================
    # DAG Pipeline — 声明式阶段表 + 拓扑调度
    # =================================================================

    # 管线定义：每个 pipeline 是一个 stage 列表
    #   deps:      前置阶段 ID 列表，用于拓扑排序
    #   fn:        本类的方法名（不含 self）
    #   condition: 可选，lambda(task) → bool，决定是否执行
    #   parallel:  是否与同批就绪的 stage 并行执行
    #   loopback:  是否支持回流（QA 不通过时 re-enqueue 前置 stage）
    PIPELINE_DEFS = {
        PipelineType.FULL_NOTE: {
            "stages": [
                {"id": "prepare",      "deps": [],          "fn": "_stage_prepare"},
                {"id": "extract",      "deps": ["prepare"], "fn": "_stage_extract"},
                # ── 并行阶段 ──
                {"id": "transcribe",   "deps": ["extract"], "fn": "_stage_transcribe",
                 "parallel": True},
                {"id": "vision",       "deps": ["extract"], "fn": "_stage_vision",
                 "parallel": True,
                 "condition": lambda ctx: not ctx.get("is_audio_only", False)},
                {"id": "router",       "deps": ["extract"], "fn": "_stage_router",
                 "parallel": True,
                 "condition": lambda ctx: config.get("router.enabled", True)},
                # ── 纠错（依赖转写 + 路由完成） ──
                {"id": "corrector",    "deps": ["transcribe","router"], "fn": "_stage_corrector",
                 "condition": lambda ctx: config.get("corrector.enabled", True)},
                # ── 后续阶段 ──
                {"id": "chunk",        "deps": ["corrector","transcribe"], "fn": "_stage_chunk",
                 "parallel": True},
                {"id": "compose",      "deps": ["vision","router","corrector","chunk"],
                 "fn": "_stage_compose"},
                {"id": "qa",           "deps": ["compose"],  "fn": "_stage_qa",
                 "loopback": "compose"},
                {"id": "save",         "deps": ["qa"],       "fn": "_stage_save"},
                {"id": "cleanup",      "deps": ["save"],     "fn": "_stage_cleanup"},
            ],
            "max_retries": 2,
            "fallback_classify": True,    # Router 不可用时回退到旧 Classifier
        },
        PipelineType.PDF_NOTE: {
            "stages": [
                {"id": "extract_pdf",  "deps": [],          "fn": "_stage_extract_pdf"},
                {"id": "vision_pdf",   "deps": ["extract_pdf"], "fn": "_stage_vision",
                 "condition": lambda ctx: bool(ctx.get("pdf_images", []))},
                {"id": "compose_pdf",  "deps": ["extract_pdf","vision_pdf"],
                 "fn": "_stage_compose_pdf"},
                {"id": "qa_pdf",       "deps": ["compose_pdf"], "fn": "_stage_qa_pdf",
                 "loopback": "compose_pdf"},
                {"id": "save",         "deps": ["qa_pdf"],    "fn": "_stage_save_pdf"},
            ],
            "max_retries": 2,
        },
        PipelineType.WEB_NOTE: {
            "stages": [
                {"id": "extract_web",  "deps": [],          "fn": "_stage_extract_web"},
                {"id": "compose_web",  "deps": ["extract_web"], "fn": "_stage_compose_web"},
                {"id": "qa_web",       "deps": ["compose_web"], "fn": "_stage_qa_web",
                 "loopback": "compose_web"},
                {"id": "save",         "deps": ["qa_web"],    "fn": "_stage_save_web"},
            ],
            "max_retries": 2,
        },
    }

    def _run_dag(self, pipeline_type: PipelineType, ctx: dict) -> dict:
        """执行 DAG pipeline

        Args:
            pipeline_type: PipelineType 枚举
            ctx: 上下文 dict，各 stage 读取/写入

        Returns:
            结果 dict
        """
        pipeline_def = self.PIPELINE_DEFS.get(pipeline_type)
        if not pipeline_def:
            raise ValueError(f"未知 pipeline: {pipeline_type}")

        stages = pipeline_def["stages"]
        max_retries = pipeline_def.get("max_retries", 0)

        # 构建入度表和依赖图
        deps_in = {s["id"]: set(s.get("deps", [])) for s in stages}
        dep_graph = {s["id"]: s for s in stages}

        def _get_ready(dep_in, completed):
            """返回所有依赖已满足且未完成的 stage"""
            ready = []
            for sid, deps in dep_in.items():
                if sid in completed:
                    continue
                if deps.issubset(completed):
                    # 检查 condition
                    stage_def = dep_graph[sid]
                    cond = stage_def.get("condition")
                    if cond is None or cond(ctx):
                        ready.append(stage_def)
            return ready

        completed = set()
        loopback_count = 0
        ctx["results"] = {}

        pool = ThreadPoolExecutor(max_workers=4)
        try:
            while len(completed) < len(stages):
                ready = _get_ready(deps_in, completed)
                if not ready:
                    # 没有就绪的 stage 但还有未完成的 → 死锁
                    remaining = set(deps_in.keys()) - completed
                    if remaining:
                        raise RuntimeError(f"DAG 死锁: 剩余 {remaining} 无法就绪")
                    break

                # 提交就绪的 stage（含缓存检查）
                futures = {}
                input_path = ctx.get("video_path") or ctx.get("job.input", "")
                for stage_def in ready:
                    sid = stage_def["id"]
                    # 查询缓存（仅缓存有副作用、且昂贵的 stage）
                    # prepare/extract 有 ctx 副作用（设置 task/file_base 等），不缓存
                    use_cache = sid in ("transcribe", "vision", "classify", "chunk")
                    cached_result = None
                    if use_cache and input_path:
                        cache_params = self._cache_params_for_stage(sid, ctx)
                        cached_result = self.cache.get(input_path, sid, cache_params)

                    if cached_result is not None:
                        # 缓存命中，跳过执行
                        ctx["results"][sid] = cached_result
                        completed.add(sid)
                        logger.info(f"[DAG] {sid} 从缓存加载完成")
                        continue

                    fn = getattr(self, stage_def["fn"])
                    fut = pool.submit(fn, ctx)
                    futures[fut] = stage_def

                for future in as_completed(futures):
                    stage_def = futures[future]
                    sid = stage_def["id"]
                    try:
                        result = future.result()
                        ctx["results"][sid] = result
                        completed.add(sid)
                        logger.info(f"[DAG] {sid} 完成")

                        # 写入缓存（仅缓存昂贵 stage）
                        use_cache = sid in ("transcribe", "vision", "classify", "chunk")
                        if use_cache and input_path:
                            cache_params = self._cache_params_for_stage(sid, ctx)
                            self.cache.set(input_path, sid, result, cache_params)

                        # loopback 处理
                        loop_target = stage_def.get("loopback")
                        if loop_target and result is not None:
                            score = result if isinstance(result, (int, float)) else 0
                            if score == 0:
                                loopback_count += 1
                                if loopback_count <= max_retries:
                                    # 重置 compose 到未完成
                                    to_reset = {loop_target}
                                    # 同时也重置 QA
                                    for s in stages:
                                        if s.get("loopback") == loop_target:
                                            to_reset.add(s["id"])
                                    for sid_rm in to_reset:
                                        completed.discard(sid_rm)
                                    logger.info(
                                        f"[DAG] loopback #{loopback_count}: {sid} → "
                                        f"重置 {to_reset}, 重新调度"
                                    )
                    except Exception as e:
                        logger.error(f"[DAG] {sid} 失败: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        ctx["error"] = str(e)
                        # 让外层处理异常
                        raise

        finally:
            pool.shutdown(wait=False)
        return ctx["results"]

    @property
    def classifier(self) -> ClassifierAgent:
        if self._classifier is None:
            self._classifier = ClassifierAgent()
        return self._classifier

    @property
    def transcriber(self) -> TranscriberAgent:
        if self._transcriber is None:
            self._transcriber = TranscriberAgent()
        return self._transcriber

    @property
    def vision(self) -> VisionAgent:
        if self._vision is None:
            self._vision = VisionAgent()
        return self._vision

    @property
    def composer(self) -> ComposerAgent:
        if self._composer is None:
            self._composer = ComposerAgent()
        return self._composer

    @property
    def qa(self) -> QAAgent:
        if self._qa is None:
            self._qa = QAAgent()
        return self._qa

    @property
    def memory(self) -> MemoryAgent:
        if self._memory is None:
            self._memory = MemoryAgent()
        return self._memory

    # ── 新 Agent 懒加载 ────────────────────────────────────

    @property
    def router(self) -> RouterAgent:
        if self._router is None:
            self._router = RouterAgent()
        return self._router

    @property
    def corrector(self) -> CorrectorAgent:
        if self._corrector is None:
            self._corrector = CorrectorAgent()
        return self._corrector

    @property
    def policy(self) -> PolicyEngine:
        if self._policy is None:
            self._policy = PolicyEngine()
        return self._policy

    @property
    def keyword_mgr(self) -> KeywordManager:
        if self._keyword_mgr is None:
            self._keyword_mgr = KeywordManager()
        return self._keyword_mgr

    @staticmethod
    def _cache_params_for_stage(sid: str, ctx: dict) -> Optional[dict]:
        """为每个 stage 生成缓存参数（模型/配置差异 → 不同缓存键）"""
        params = {}
        try:
            from note_weaver.utils.config import config
            if sid == "transcribe":
                params["whisper_model"] = config.get("whisper.model_size", "small")
            elif sid == "vision" or sid == "vision_pdf":
                params["max_images"] = config.get("vision.max_images_per_batch", 10)
            elif sid == "extract":
                params["keyframe_strategy"] = "hybrid"
            elif sid == "router":
                params["enabled"] = config.get("router.enabled", True)
                params["density_check"] = config.get("router.visual_density_check", True)
            elif sid == "corrector":
                params["enabled"] = config.get("corrector.enabled", True)
                params["model"] = config.get("corrector.model", "fast")
            elif sid == "classify":
                params["model"] = config.get("api.deepseek.model_fast", "deepseek-chat")
        except Exception:
            pass
        return params if params else None

    # =================================================================
    # 主处理流程 — DAG Stage 方法
    # =================================================================

    def _stage_prepare(self, ctx: dict) -> dict:
        """Step 0: 文件信息解析 + 去重检查 + 创建 Task"""
        video_path = ctx["video_path"]
        file_name = os.path.basename(video_path)
        file_base = os.path.splitext(file_name)[0].replace(' ', '_')
        for suffix in ('_Auto', '_auto', '_720P', '_1080P', '_final'):
            if file_base.endswith(suffix):
                file_base = file_base[:-len(suffix)]
                file_name = file_base + os.path.splitext(file_name)[1]
                break

        rel_subdir = self._get_rel_subdir(video_path)
        if rel_subdir:
            logger.info(f"[DIR]  输出子目录: {rel_subdir}")
            os.makedirs(os.path.join(config.txt_dir, rel_subdir), exist_ok=True)
            os.makedirs(os.path.join(config.note_dir, rel_subdir), exist_ok=True)

        existing = self.state_machine.find_processed(file_name)
        if existing:
            logger.info(f"[SKIP] 跳过已处理: {file_name}")
            ctx["skipped"] = True
            ctx["task"] = existing
            return {"skipped": True, "task_id": existing.task_id}

        task = self.state_machine.create_task(video_path)
        logger.info(f"=" * 60)
        logger.info(f"[START] 开始处理: {file_name} (task={task.task_id})")
        logger.info(f"=" * 60)

        audio_tmp = os.path.join(config.base_dir, f"temp_audio_{file_base}.mp3")
        is_audio_only = file_name.lower().endswith(
            ('.m4a', '.mp3', '.wav', '.flac', '.aac')
        )
        screenshot_dir = os.path.join(config.note_dir, rel_subdir, file_base)

        ctx.update({
            "video_path": video_path, "file_name": file_name, "file_base": file_base,
            "rel_subdir": rel_subdir, "task": task, "audio_tmp": audio_tmp,
            "is_audio_only": is_audio_only, "screenshot_dir": screenshot_dir,
            "screenshot_files": [], "skipped": False,
        })
        return {"task_id": task.task_id, "file_name": file_name}

    def _stage_extract(self, ctx: dict) -> dict:
        """Step 1: 提取音频 + 截图 (并行)"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.EXTRACTING)

        if ctx["is_audio_only"]:
            logger.info("检测到纯音频文件，跳过截图")
            extract_audio(ctx["video_path"], ctx["audio_tmp"])
            task.audio_path = ctx["audio_tmp"]
            task.screenshot_dir = ctx["screenshot_dir"]
            task.screenshot_files = []
            ctx["screenshot_files"] = []
            return {"audio_only": True}

        clean_screenshot_dir(ctx["screenshot_dir"], ctx["file_base"])

        # 自适应截图间隔
        dur = get_video_duration(ctx["video_path"])
        ctx["duration"] = dur
        if dur <= 300:
            interval = 30
        elif dur <= 900:
            interval = 60
        elif dur <= 1800:
            interval = 90
        elif dur <= 3600:
            interval = 120
        else:
            interval = 180

        # 使用智能帧选择（hybrid 策略：scene change + 均匀采样）
        strategy = "hybrid"
        logger.info(
            f"[Keyframe] 策略={strategy}, 间隔={interval}s "
            f"(视频 {dur:.0f}s)"
        )

        from concurrent.futures import ThreadPoolExecutor as _TPool
        ext_pool = _TPool(max_workers=2)
        audio_future = ext_pool.submit(extract_audio, ctx["video_path"], ctx["audio_tmp"])
        screenshot_files = []
        screen_future = ext_pool.submit(
            extract_keyframes, ctx["video_path"],
            ctx["screenshot_dir"], ctx["file_base"],
            strategy, interval,
        )
        try:
            audio_future.result()
            screenshot_files = screen_future.result()
        except KeyboardInterrupt:
            logger.warning("[Extract] 用户中断提取，跳过结果收集")
            ctx["skipped"] = True
            ext_pool.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception as e:
            logger.error(f"[Extract] 提取失败: {e}")
            ext_pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            ext_pool.shutdown(wait=False)

        task.audio_path = ctx["audio_tmp"]
        task.screenshot_dir = ctx["screenshot_dir"]
        task.screenshot_files = screenshot_files
        ctx["screenshot_files"] = screenshot_files
        ctx["duration"] = dur

        logger.info(f"[OK] 提取完成: 音频 + {len(screenshot_files)} 张截图")
        return {"audio_only": False, "screenshot_count": len(screenshot_files)}

    def _stage_transcribe(self, ctx: dict) -> dict:
        """Step 2a: 语音转录"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.TRANSCRIBING)
        transcript = self.transcriber.execute(task.audio_path)
        task.transcript = transcript
        ctx["transcript"] = transcript
        logger.info(f"[OK] 转录完成: {len(transcript.get('raw_text', ''))}字")
        return {"text_length": len(transcript.get("raw_text", ""))}

    def _stage_vision(self, ctx: dict) -> dict:
        """Step 2b: Vision 截图分析"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.VISION_ANALYZING)
        screenshot_files = ctx.get("screenshot_files", [])
        if screenshot_files:
            vision_results = self.vision.execute(screenshot_files)
        else:
            vision_results = []
        task.vision_results = vision_results
        ctx["vision_results"] = vision_results
        logger.info(f"[OK] 视觉分析完成: {len(vision_results)}张")
        return {"vision_count": len(vision_results)}

    def _stage_router(self, ctx: dict) -> dict:
        """Step 2c: Router 路由分析（与 Transcribe/Vision 并行）

        快速分析视频领域/内容结构/视觉密度，产出 routing 信号。
        不依赖 Transcribe 结果——若 Transcribe 未完成，自动用
        tiny 模型独立转写前30s。
        """
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.ROUTING)

        # 准备 Router 输入
        transcript = ctx.get("transcript") or task.transcript or {}
        segments = transcript.get("segments", [])
        audio_first_30s = ""
        if segments:
            audio_first_30s = " ".join(
                seg["text"] for seg in segments if seg["start"] <= 30
            )
        if not audio_first_30s.strip() and segments:
            audio_first_30s = segments[0]["text"]

        duration = ctx.get("duration", 0)

        # 执行 Router（传入 audio_tmp 用于独立转写，不再依赖 Transcribe）
        router_result = self.router.execute(
            video_path=ctx.get("video_path", ""),
            file_name=ctx.get("file_name", ""),
            audio_first_30s=audio_first_30s,
            duration=duration,
            screenshot_dir=ctx.get("screenshot_dir", ""),
            file_base=ctx.get("file_base", ""),
            audio_path=ctx.get("audio_tmp", ""),    # 传入音频路径，Router 可独立转写
        )

        # Policy Engine: Router 输出 → 管线参数
        policy_result = self.policy.compute(router_result)

        # 合并动态热词（自动扩展的领域高频词）
        try:
            domain = router_result.get("domain", "general")
            merged = self.keyword_mgr.merge_with_static(
                domain, policy_result.get("correction_keywords", []),
            )
            policy_result["correction_keywords"] = merged
            if len(merged) > len(router_result.get("keywords", [])):
                logger.debug(
                    f"[KeywordMgr] 动态合并: {domain} "
                    f"共 {len(merged)} 词（含动态词）"
                )
        except Exception as e:
            logger.debug(f"[KeywordMgr] 动态合并跳过（非致命）: {e}")

        ctx["router_result"] = router_result
        ctx["policy_result"] = policy_result
        ctx["classification"] = {
            "domain": router_result.get("domain", "general"),
            "type": router_result.get("content_structure", "lecture"),
            "visual_density": router_result.get("visual_density", "medium"),
            "keywords": router_result.get("keywords", []),
            "suggested_strategy": {
                "screenshot_interval": policy_result.get("frame_interval", 60),
                "note_style": policy_result.get("note_style", "detailed"),
                "focus_areas": policy_result.get("correction_keywords", []),
            },
        }

        task.classification = ctx["classification"]

        logger.info(
            f"[Router] {ctx['file_name']} → "
            f"domain={router_result.get('domain')}, "
            f"structure={router_result.get('content_structure')}, "
            f"density={router_result.get('visual_density')}"
        )
        return {
            "domain": router_result.get("domain", ""),
            "structure": router_result.get("content_structure", ""),
            "density": router_result.get("visual_density", ""),
        }

    def _stage_corrector(self, ctx: dict) -> dict:
        """Step 3a: 转录纠错（领域感知）

        在 Transcribe 之后、Chunk 之前执行，使用 Router 的领域信号。
        """
        if ctx.get("skipped"):
            return {"skipped": True}
        if not config.corrector_enabled:
            logger.info("[Corrector] 已禁用，跳过纠错")
            return {"skipped": True, "reason": "disabled"}

        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.CORRECTING)

        transcript = ctx.get("transcript") or task.transcript or {}
        segments = transcript.get("segments", [])
        raw_text = transcript.get("raw_text", "")

        if not segments:
            logger.info("[Corrector] 无段落可纠错，跳过")
            return {"skipped": True, "reason": "no_segments"}

        # 从 Router/Policy 获取领域关键词
        policy_result = ctx.get("policy_result", {})
        domain_keywords = policy_result.get("correction_keywords", [])
        router_result = ctx.get("router_result", {})
        domain = router_result.get("domain", "general")

        # 执行纠错
        corrected = self.corrector.execute(
            segments=segments,
            domain_keywords=domain_keywords,
            domain=domain,
            raw_text=raw_text,
        )

        ctx["corrected_transcript"] = corrected

        # 同时更新 task.transcript 的引用（向下兼容）
        if corrected.get("segments"):
            task.transcript = {
                "timestamped": corrected.get("timestamped", transcript.get("timestamped", "")),
                "raw_text": corrected.get("raw_text", raw_text),
                "segments": corrected.get("segments", segments),
                "duration": transcript.get("duration", 0),
                "language": transcript.get("language", "zh"),
            }

        corrections = corrected.get("corrections_count", 0)
        logger.info(f"[Corrector] 完成: {corrections} 处修正")
        return {"corrections_count": corrections}

    def _stage_classify(self, ctx: dict) -> dict:
        """Step 2c: 视频分类"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.CLASSIFYING)

        segments = (ctx.get("transcript") or task.transcript or {}).get("segments", [])
        first_30s_text = " ".join(
            seg["text"] for seg in segments if seg["start"] <= 30
        )
        if not first_30s_text.strip() and segments:
            first_30s_text = segments[0]["text"]

        duration = ctx.get("duration", 0)
        classification = self.classifier.execute(
            filename=ctx["file_name"],
            audio_sample=first_30s_text,
            duration=duration,
        )
        task.classification = classification
        ctx["classification"] = classification
        return {"domain": classification.get("domain", "")}

    def _stage_chunk(self, ctx: dict) -> dict:
        """Step 3: 保存转录文件（优先使用纠错后的文本）"""
        if ctx.get("skipped"):
            return {"skipped": True}

        # 优先使用纠错后的转录
        corrected = ctx.get("corrected_transcript", {})
        if corrected and corrected.get("raw_text"):
            raw_text = corrected["raw_text"]
        else:
            transcript = ctx.get("transcript") or ctx["task"].transcript or {}
            raw_text = transcript.get("raw_text", "")
        self._save_txt(ctx["file_base"], raw_text, ctx.get("rel_subdir", ""))
        txt_path = os.path.join(
            config.txt_dir, ctx.get("rel_subdir", ""), f"{ctx['file_base']}.txt"
        )
        ctx["task"].txt_path = txt_path

        # 保存转录 JSON
        transcript_path = txt_path.replace(".txt", "_transcript.json")
        try:
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Orchestrator] 转录 JSON 保存失败（非致命）: {e}")
        return {"txt_path": txt_path}

    def _stage_compose(self, ctx: dict) -> dict:
        """Step 4a: Composer 排版"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.COMPOSING)

        classification = ctx.get("classification", {}) or task.classification or {}
        domain = classification.get("domain", "")
        difficulty = classification.get("difficulty", "")

        # ── 使用纠错后的转录（如有） ──
        corrected = ctx.get("corrected_transcript", {})
        if corrected and corrected.get("segments"):
            transcript = {
                "timestamped": corrected.get("timestamped", ""),
                "raw_text": corrected.get("raw_text", ""),
                "segments": corrected.get("segments", []),
            }
        else:
            transcript = ctx.get("transcript") or task.transcript or {}

        vision_results = ctx.get("vision_results") or task.vision_results or []
        # V2: 获取转录段落，用于图片时间戳对齐
        segments = transcript.get("segments", [])

        user_context = self.memory.get_context(domain=domain, difficulty=difficulty)

        # ── 策略：优先使用 Policy Engine 的输出 ──
        policy_result = ctx.get("policy_result", {})
        if policy_result:
            strategy = {
                "note_style": policy_result.get("note_style", "detailed"),
                "focus_areas": policy_result.get("correction_keywords", [])[:5],
                "screenshot_interval": policy_result.get("frame_interval", 60),
            }
        else:
            strategy = classification.get("suggested_strategy", {})

        # 获取 QA 反馈（优先使用结构化 defects）
        retry_count = ctx.get("retry_count", 0)
        revision_feedback = ""
        defects = None
        qa_report = ctx.get("qa_report") or ctx.get("results", {}).get("qa")
        if retry_count > 0 and qa_report:
            if isinstance(qa_report, dict):
                defects = qa_report.get("defects")
                revision_feedback = qa_report.get("revision_suggestions", "")
            logger.info(
                f"[DAG] compose loopback #{retry_count}, "
                f"缺陷={len(defects) if defects else '文本反馈'}"
            )

        note_content = self.composer.execute(
            file_base=ctx["file_base"],
            timestamped_text=transcript.get("timestamped", ""),
            vision_results=vision_results,
            strategy=strategy,
            user_context=user_context,
            revision_feedback=revision_feedback,
            defects=defects,
            segments=segments,  # V2: 传入转录段落用于时间戳对齐
        )
        task.note_content = note_content
        ctx["note_content"] = note_content
        return {"length": len(note_content)}

    def _stage_qa(self, ctx: dict) -> float:
        """Step 4b: QA 质检 — 返回分数，0 分触发 loopback"""
        if ctx.get("skipped"):
            return 10.0  # 跳过则直接过
        task = ctx["task"]
        self.state_machine.transition(task, TaskStatus.QA_REVIEWING)

        note_content = ctx.get("note_content") or task.note_content or ""
        transcript = ctx.get("transcript") or task.transcript or {}
        vision_results = ctx.get("vision_results") or task.vision_results or []
        retry_count = ctx.get("retry_count", 0)

        fb = config.qa_fallback_thresholds
        cur_threshold = fb[retry_count] if retry_count < len(fb) else fb[-1]

        qa_report = self.qa.execute(
            note_content=note_content,
            transcript_text=transcript.get("raw_text", ""),
            vision_results=vision_results,
            threshold=cur_threshold,
        )
        task.qa_report = qa_report
        task.qa_score = qa_report.get("total", 0)
        ctx["qa_report"] = qa_report

        passed = qa_report.get("passed", True)
        score = qa_report.get("total", 0)
        if passed:
            logger.info(f"[OK] QA通过 (score={score})")
            return score
        else:
            logger.warning(f"[FAIL] QA不通过 (score={score}, retry={retry_count})")
            ctx["retry_count"] = retry_count + 1
            return 0  # 0 分 = 触发 loopback

    def _stage_save(self, ctx: dict) -> dict:
        """Step 5: 保存笔记 + 更新记忆"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        note_content = ctx.get("note_content") or task.note_content or ""
        classification = ctx.get("classification") or task.classification or {}

        note_output_dir = os.path.join(
            config.note_dir, ctx.get("rel_subdir", "")
        )
        md_path = self.composer.save_note(ctx["file_base"], note_content, note_output_dir)
        task.md_path = md_path
        self._export_outputs(md_path)

        qa_report = ctx.get("qa_report") or task.qa_report or {}
        try:
            self.memory.update_after_note(
                file_base=ctx["file_base"],
                note_content=note_content,
                classification=classification,
                qa_report=qa_report,
            )
        except Exception as mem_err:
            logger.warning(f"[Memory] 更新失败（非致命）: {mem_err}")

        self._rebuild_indexes()
        return {"md_path": md_path}

    def _stage_cleanup(self, ctx: dict) -> dict:
        """Step 6: 清理临时文件 + 标记完成"""
        if ctx.get("skipped"):
            return {"skipped": True}
        task = ctx["task"]
        audio_tmp = ctx.get("audio_tmp", "")
        if os.path.exists(audio_tmp):
            os.remove(audio_tmp)
        self.state_machine.transition(task, TaskStatus.COMPLETED)
        logger.info(
            f"[DONE] 处理完成: {ctx['file_name']} "
            f"(耗时 {task.elapsed_seconds:.0f}s, QA={task.qa_score})"
        )

        # ── 自动热词扩展（从转录文本提取领域高频词） ──
        try:
            transcript = ctx.get("transcript") or task.transcript or {}
            raw_text = transcript.get("raw_text", "")
            router_result = ctx.get("router_result") or {}
            domain = router_result.get("domain", "")
            if raw_text and domain and len(raw_text) > 50:
                self.keyword_mgr.update_from_transcript(raw_text, domain)
        except Exception as e:
            logger.debug(f"[KeywordMgr] 热词更新跳过（非致命）: {e}")
        return {
            "task_id": task.task_id,
            "qa_score": task.qa_score,
            "md_path": task.md_path,
        }

    # ── PDF/Web stages ──

    def _stage_qa_pdf(self, ctx: dict) -> float:
        """PDF: QA 质检 + loopback 支持"""
        note_content = ctx.get("note_content", "")
        vision_results = ctx.get("vision_results", [])
        retry_count = ctx.get("retry_count", 0)

        fb = config.qa_fallback_thresholds
        cur_threshold = fb[retry_count] if retry_count < len(fb) else fb[-1]

        qa_report = self.qa.execute(
            note_content=note_content,
            transcript_text=ctx.get("pdf_text", "")[:15000],
            vision_results=vision_results,
            threshold=cur_threshold,
        )
        ctx["qa_report"] = qa_report

        passed = qa_report.get("passed", True)
        score = qa_report.get("total", 0)
        if passed:
            logger.info(f"[PDF-QA] 通过 (score={score})")
            return score
        else:
            logger.warning(f"[PDF-QA] 不通过 (score={score}, retry={retry_count})")
            ctx["retry_count"] = retry_count + 1
            return 0

    def _stage_qa_web(self, ctx: dict) -> float:
        """Web: QA 质检 + loopback 支持"""
        note_content = ctx.get("note_content", "")
        vision_results = ctx.get("vision_results", [])
        retry_count = ctx.get("retry_count", 0)

        fb = config.qa_fallback_thresholds
        cur_threshold = fb[retry_count] if retry_count < len(fb) else fb[-1]

        qa_report = self.qa.execute(
            note_content=note_content,
            transcript_text=ctx.get("web_text", "")[:15000],
            vision_results=vision_results,
            threshold=cur_threshold,
        )
        ctx["qa_report"] = qa_report

        passed = qa_report.get("passed", True)
        score = qa_report.get("total", 0)
        if passed:
            logger.info(f"[Web-QA] 通过 (score={score})")
            return score
        else:
            logger.warning(f"[Web-QA] 不通过 (score={score}, retry={retry_count})")
            ctx["retry_count"] = retry_count + 1
            return 0

    def _stage_extract_pdf(self, ctx: dict) -> dict:
        from note_weaver.utils.extractors import extract_from_pdf
        config.setup_proxy()
        p = pathlib.Path(ctx["job"].input)
        file_base = p.stem.replace(' ', '_')
        note_dir = pathlib.Path(config.base_dir) / "data" / "Note" / "pdf"
        img_dir = note_dir / file_base
        result = extract_from_pdf(ctx["job"].input, output_dir=str(img_dir))
        ctx.update({
            "file_base": file_base, "note_dir": str(note_dir),
            "pdf_text": result["text"], "pdf_images": result["images"],
        })
        logger.info(f"[PDF] 提取: {p.name} → {len(result['text'])}字, {len(result['images'])}张图")
        return {"text_len": len(result["text"]), "image_count": len(result["images"])}

    def _stage_compose_pdf(self, ctx: dict) -> dict:
        from note_weaver.agents.composer import ComposerAgent
        vision_results = ctx.get("vision_results", [])
        composer = ComposerAgent()
        note_content = composer.execute(
            file_base=ctx["file_base"],
            timestamped_text=ctx["pdf_text"][:15000],
            vision_results=vision_results,
            strategy={"note_style": "detailed", "focus_areas": []},
            revision_feedback="这是从PDF提取的内容，整理成结构化学习笔记。",
        )
        ctx["note_content"] = note_content
        return {"length": len(note_content)}

    def _stage_save_pdf(self, ctx: dict) -> dict:
        from note_weaver.agents.composer import ComposerAgent
        from note_weaver.run import insert_images_by_content, _fix_broken_markdown_images
        composer = ComposerAgent()
        vision_results = ctx.get("vision_results", [])
        note_with_images = insert_images_by_content(
            ctx["note_content"], vision_results, ctx["file_base"]
        )
        note_with_images = _fix_broken_markdown_images(note_with_images)
        note_path = composer.save_note(ctx["file_base"], note_with_images, ctx["note_dir"])
        self._export_outputs(note_path)
        self._rebuild_indexes()
        return {"note_path": note_path}

    def _stage_extract_web(self, ctx: dict) -> dict:
        from note_weaver.utils.extractors import extract_from_url
        config.setup_proxy()
        import urllib.parse
        parsed = urllib.parse.urlparse(ctx["job"].input)
        domain = parsed.netloc.replace("www.", "")
        file_base = re.sub(r'[^\w一-鿿]+', '_', domain)[:30]
        note_dir = pathlib.Path(config.base_dir) / "data" / "Note" / "web"
        img_dir = note_dir / file_base
        result = extract_from_url(ctx["job"].input, output_dir=str(img_dir))
        ctx.update({
            "file_base": file_base, "note_dir": str(note_dir),
            "web_text": result["text"],
        })
        if result.get("images"):
            from note_weaver.agents.vision import VisionAgent
            ctx["vision_results"] = VisionAgent().execute(result["images"])
        else:
            ctx["vision_results"] = []
        logger.info(f"[Web] 提取: {result.get('title','?')[:40]} → {len(result['text'])}字")
        return {"title": result.get("title",""), "text_len": len(result["text"])}

    def _stage_compose_web(self, ctx: dict) -> dict:
        from note_weaver.agents.composer import ComposerAgent
        composer = ComposerAgent()
        note_content = composer.execute(
            file_base=ctx["file_base"],
            timestamped_text=ctx["web_text"][:15000],
            vision_results=ctx.get("vision_results", []),
            strategy={"note_style": "detailed", "focus_areas": []},
            revision_feedback=f"来源网页整理成结构化学习笔记。",
        )
        ctx["note_content"] = note_content
        return {"length": len(note_content)}

    def _stage_save_web(self, ctx: dict) -> dict:
        from note_weaver.agents.composer import ComposerAgent
        from note_weaver.run import insert_images_by_content, _fix_broken_markdown_images
        composer = ComposerAgent()
        note_with_images = insert_images_by_content(
            ctx["note_content"], ctx.get("vision_results", []), ctx["file_base"]
        )
        note_with_images = _fix_broken_markdown_images(note_with_images)
        note_path = composer.save_note(ctx["file_base"], note_with_images, ctx["note_dir"])
        self._export_outputs(note_path)
        self._rebuild_indexes()
        return {"note_path": note_path}

    def _run_full_note_dag(self, job: Job) -> dict:
        """FULL_NOTE 管线的 DAG 入口 — 设置 ctx 并执行 _run_dag"""
        ctx = {"video_path": job.input, "job": job}
        try:
            results = self._run_dag(PipelineType.FULL_NOTE, ctx)
            task = ctx.get("task")
            if ctx.get("skipped") and task:
                return {"ok": True, "skipped": True, "task_id": task.task_id}
            if "save" in results and results["save"].get("md_path"):
                md_path = results["save"]["md_path"]
                qa_score = results.get("cleanup", {}).get("qa_score", 0)
                return {
                    "ok": True, "qa_score": qa_score, "note_path": md_path,
                    "task_id": results["cleanup"].get("task_id", ""),
                }
            if ctx.get("error"):
                return {"ok": False, "error": ctx["error"]}
            return {"ok": True, "results": {k: v for k, v in results.items()
                    if isinstance(v, dict)}}
        except KeyboardInterrupt:
            logger.warning("[DAG] 用户中断处理")
            return {"ok": False, "error": "用户中断"}
        except Exception as e:
            import traceback
            logger.error(f"[DAG] FULL_NOTE 执行失败: {e}")
            logger.error(traceback.format_exc())
            return {"ok": False, "error": str(e)}

    # =================================================================
    # 主处理流程（向后兼容）
    # =================================================================

    def process_video(self, video_path: str) -> Optional[Task]:
        """处理单个视频的完整流程（向后兼容）

        内部使用 DAG pipeline 执行，返回 Task 对象。
        """
        job = Job(input=video_path, modality=Modality.VIDEO,
                  pipeline=PipelineType.FULL_NOTE)
        result = self._run_full_note_dag(job)
        ctx_video_path = video_path  # 用于查找 task
        task = None
        for t in list(self.state_machine.history):
            if t.video_path == video_path or t.file_name == os.path.basename(video_path):
                task = t
                break
        if task is None and self.state_machine.tasks:
            for t in self.state_machine.tasks.values():
                if t.video_path == video_path:
                    task = t
                    break
        return task

    # =================================================================
    # Skills（交互式命令）
    # =================================================================

    def search_notes(self, query: str) -> str:
        """搜索笔记库"""
        concepts = self.memory.search_concepts(query)
        if concepts:
            lines = [f"## 搜索「{query}」— 知识图谱匹配:\n"]
            for c in concepts:
                lines.append(
                    f"- **{c.get('name', '?')}** ({c.get('name_en', '')}): "
                    f"{c.get('definition', '')}"
                )
            return "\n".join(lines)
        return f"未找到与「{query}」相关的概念"

    def get_stats(self) -> str:
        """获取学习统计"""
        stats = self.memory.get_learning_stats()
        active = self.state_machine.get_active_count()
        history = self.state_machine.get_history_summary(5)

        lines = [
            "## NoteWeaver 统计",
            f"- 总笔记数: {stats['total_notes']}",
            f"- 知识图谱概念: {stats['kg_concepts']}",
            f"- 关系数: {stats['kg_relations']}",
            f"- 活跃任务: {active}",
            f"- 薄弱点: {', '.join(stats['weak_points'][:5]) or '无'}",
        ]
        if history:
            lines.append("\n### 最近处理")
            for h in history[:5]:
                lines.append(f"- {h['file_name']} (QA={h['qa_score']})")

        return "\n".join(lines)

    # =================================================================
    # 内部辅助方法
    # =================================================================

    @staticmethod
    def _get_rel_subdir(video_path: str) -> str:
        """计算视频相对于 source_video_dir 的子目录路径

        例如: video_path = "E:/.../Video/1.工艺速通/02_衬底.mp4"
              返回 "1.工艺速通"

        如果视频不在 source_video_dir 下，返回 ""（平铺，向后兼容）
        """
        src_dir = os.path.join(config.base_dir, config.source_video_dir)
        src_dir = os.path.abspath(src_dir)
        try:
            rel = os.path.relpath(os.path.dirname(video_path), src_dir)
            if rel.startswith(".."):
                return ""
            rel = rel.replace(' ', '_')
            return rel if rel != "." else ""
        except ValueError:
            return ""

    def _save_txt(self, file_base: str, raw_text: str, rel_subdir: str = ""):
        """保存原始转录 TXT（按子目录组织）"""
        txt_dir = os.path.join(config.txt_dir, rel_subdir)
        os.makedirs(txt_dir, exist_ok=True)
        formatted = re.sub(r'([。！？；])', r'\1\n', raw_text)
        txt_path = os.path.join(txt_dir, f"{file_base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted.strip())
        logger.info(f"TXT 已保存: {txt_path}")

