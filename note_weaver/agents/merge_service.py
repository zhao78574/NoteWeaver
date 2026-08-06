"""合集合并服务 — 将多集视频处理结果合并为单篇合集笔记

从 Orchestrator 拆分出来，负责 YouTube/Bilibili 播放列表的批量下载、
逐条处理和合并逻辑。
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional, TYPE_CHECKING

from note_weaver.core.job import Job, Modality, PipelineType
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger
from note_weaver.utils.style import console

if TYPE_CHECKING:
    from note_weaver.agents.orchestrator import Orchestrator


class MergeService:
    """合集/播放列表合并服务

    持有 Orchestrator 引用以访问共享的 ComposerAgent 和工具方法。
    """

    def __init__(self, orchestrator: Orchestrator):
        self.orch = orchestrator

    # ── 公共接口 ──

    def merge_from_videos(
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
            同 merge_notes
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
        merge_entries = self._process_videos(videos, dl_results)

        # ── 3) 合并笔记 ──
        return self._merge_and_save(merge_entries, playlist_title)

    def merge_notes(self, job: Job) -> dict:
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

        # ── 3) 逐条处理 ──
        merge_entries = self._process_videos(videos_to_download, dl_results)

        # ── 4) 合并笔记 ──
        return self._merge_and_save(merge_entries, playlist_title)

    # =================================================================
    # 内部方法
    # =================================================================

    def _process_videos(self, videos: list, dl_results: list) -> list:
        """逐条处理下载结果：转写 → 笔记 → 收集合并条目

        Args:
            videos: 视频信息列表 [{"title", "duration", "url", "index"}, ...]
            dl_results: 下载结果列表 [{"local_path", "error"}, ...]

        Returns:
            merge_entries: [{"title", "duration", "note_content", "url", "index", "screenshot_dir"}, ...]
        """
        merge_entries = []

        # 安全检查：确保 results 数量匹配
        if len(dl_results) != len(videos):
            logger.warning(
                f"[Merge] results({len(dl_results)}) 与 videos({len(videos)}) 数量不匹配"
            )

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
                entry = self._process_single_video(vid_info, local_path)
                if entry:
                    merge_entries.append(entry)
            except Exception as e:
                logger.error(f"[Merge] 处理失败: {vid_info['title']}: {e}")

        return merge_entries

    def _process_single_video(self, vid_info: dict, local_path: str) -> Optional[dict]:
        """处理单个视频：运行 DAG 管线 → 提取笔记内容 → 清理单集文件

        Args:
            vid_info: 视频元信息 {"title", "duration", "url", "index"}
            local_path: 本地视频文件路径

        Returns:
            合并条目 dict，失败返回 None
        """
        job_single = Job(
            input=local_path,
            modality=Modality.VIDEO,
            pipeline=PipelineType.FULL_NOTE,
        )
        result = self.orch.dag_runner.run_full_note_dag(job_single)

        # 从状态机历史中查找对应 task
        note_content = ""
        task = None
        for t in list(self.orch.state_machine.history):
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

        # 删除单集 .md（合集笔记会重新生成）
        if md_to_delete and os.path.isfile(md_to_delete):
            os.remove(md_to_delete)

        # 获取截图目录路径（直接从 task 拿，最可靠）
        ss_dir = task.screenshot_dir if task and task.screenshot_dir else ""
        if not ss_dir and md_to_delete:
            fb = os.path.splitext(os.path.basename(md_to_delete))[0]
            ss_dir = os.path.join(os.path.dirname(md_to_delete), fb)

        return {
            "title": vid_info["title"],
            "duration": vid_info["duration"],
            "note_content": note_content,
            "url": vid_info.get("url", ""),
            "index": vid_info.get("index", 0),
            "screenshot_dir": ss_dir,
        }

    def _merge_and_save(self, merge_entries: list, playlist_title: str) -> dict:
        """执行合并并保存（供 merge_from_videos 和 merge_notes 共用）"""
        output_paths = []
        merge_groups = self._split_merge_groups(merge_entries)
        for idx, group in enumerate(merge_groups):
            group_title = playlist_title if len(merge_groups) == 1 else f"{playlist_title}（第{idx+1}部分）"
            merged_content = self.orch.composer.merge_notes(group, group_title)
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

            md_path = self.orch.composer.save_note(file_base, merged_content, note_output_dir)
            output_paths.append(md_path)
            logger.info(f"[Merge] 合集笔记已保存: {md_path}")

        self.orch._rebuild_indexes()

        original_count = sum(len(g) for g in merge_groups)
        return {
            "ok": True,
            "type": "merge_notes",
            "playlist_title": playlist_title,
            "total_videos": original_count,
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

            images = sorted(
                f for f in os.listdir(ss_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))
            )
            if not images:
                continue

            copied = 0
            for img_name in images:
                dst = os.path.join(images_dir, img_name)
                if not os.path.isfile(dst):
                    shutil.copy2(os.path.join(ss_dir, img_name), dst)
                    copied += 1

            if copied:
                logger.info(f"[Merge] 合并截图: {os.path.basename(ss_dir)} → {images_dir_name} ({copied}张)")
                consolidated = True

            old_dir_name = os.path.basename(ss_dir)
            if old_dir_name:
                merged_content = merged_content.replace(
                    f"{old_dir_name}/", f"{images_dir_name}/"
                )

            shutil.rmtree(ss_dir, ignore_errors=True)
            logger.info(f"[Merge] 删除旧截图目录: {os.path.basename(ss_dir)}")

        if not consolidated:
            logger.warning("[Merge] 未找到任何截图目录，请检查 merge_entries 中的 screenshot_dir")
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
