"""视频下载 Agent — yt-dlp 封装，支持 YouTube / Bilibili 链接解析与下载"""

import os
import re
import json
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from note_weaver.utils.logger import logger
from note_weaver.utils.config import config


class VideoDownloader:
    """视频下载器 — 检测 URL → 下载完整视频 → 返回本地路径+元数据

    支持平台:
        - YouTube (youtube.com, youtu.be)
        - Bilibili (bilibili.com, b23.tv)

    用法:
        dl = VideoDownloader()
        result = dl.download("https://youtu.be/xxxxx")
        print(result["local_path"])   # → Video/youtube/xxxxx.mp4
        print(result["metadata"]["title"])
    """

    SUPPORTED_DOMAINS: Dict[str, list] = {
        "youtube": ["youtube.com", "youtu.be"],
        "bilibili": ["bilibili.com", "b23.tv"],
    }

    # 浏览器级请求头，绕过 B站等平台的反爬
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
    }

    def __init__(self, download_dir: Optional[str] = None):
        """
        Args:
            download_dir: 下载根目录，默认 {base_dir}/Video
        """
        base = Path(config.base_dir)  # 项目根目录，由 config.base_dir 指定
        self._download_root = Path(download_dir) if download_dir else (base / "Video")
        self._index_path = self._download_root / ".downloaded_index.json"

        # 按来源分目录
        self._sub_dirs = {
            source: self._download_root / source
            for source in self.SUPPORTED_DOMAINS
        }

    def _base_opts(self, source: str) -> dict:
        """所有 yt-dlp 调用的公共选项"""
        opts = {
            "quiet": True,
            "noplaylist": True,
            "no_warnings": True,
            "http_headers": dict(self._HEADERS),
        }
        if source == "bilibili":
            opts["http_headers"]["Referer"] = "https://www.bilibili.com/"
            opts["http_headers"]["Origin"] = "https://www.bilibili.com"
            opts["geo_bypass"] = True
        return opts

    # ════════════════════════════════════════════════════════
    # 公开接口
    # ════════════════════════════════════════════════════════

    def detect_source(self, url: str) -> Optional[str]:
        """判断链接来源

        Returns:
            'youtube' / 'bilibili' / None
        """
        url_lower = url.lower()
        for source, domains in self.SUPPORTED_DOMAINS.items():
            for domain in domains:
                if domain in url_lower:
                    return source
        return None

    def detect_playlist(self, url: str) -> Optional[dict]:
        """检测 URL 是否为合集/播放列表

        尝试获取播放列表信息，如果返回条目数 > 1 则判定为合集。

        Returns:
            {
                "title": str,           # 合集标题
                "playlist_count": int,  # 视频数量
                "videos": [             # 视频列表
                    {"title": str, "duration": int, "url": str, "index": int, "video_id": str}
                ],
            } 或 None（不是合集）
        """
        source = self.detect_source(url)
        if not source:
            return None

        try:
            # 第一遍：快速获取合集结构（extract_flat 不返回时长）
            ydl_opts = {
                **self._base_opts(source),
                "noplaylist": False,
                "extract_flat": True,
            }
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            playlist_count = info.get("playlist_count", 0)
            if playlist_count <= 1:
                return None

            # 构建视频列表（此时 duration 可能为 0）
            videos = []
            entries = info.get("entries", []) or []
            for entry in entries:
                if entry is None:
                    continue
                video_url = entry.get("url") or entry.get("webpage_url") or ""
                if not video_url and entry.get("id"):
                    if source == "youtube":
                        video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                    elif source == "bilibili":
                        video_url = f"https://www.bilibili.com/video/{entry['id']}"

                # extract_flat 模式下 duration 可能为 None/0
                dur = entry.get("duration") or 0

                videos.append({
                    "title": entry.get("title", f"第{len(videos)+1}集"),
                    "duration": dur,
                    "url": video_url,
                    "index": len(videos) + 1,
                    "video_id": entry.get("id", ""),
                    "source": source,
                })

            # 第二遍：批量补时长（最多补前 30 个，快很多）
            missing = [v for v in videos[:30] if not v.get("duration")]
            if missing:
                logger.info(f"[Downloader] 补全 {len(missing)}/{playlist_count} 个视频时长...")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                    future_map = {}
                    for v in missing:
                        fut = pool.submit(self.get_info, v["url"])
                        future_map[fut] = v
                    for fut in concurrent.futures.as_completed(future_map):
                        v = future_map[fut]
                        try:
                            detail = fut.result(timeout=15)
                            v["duration"] = detail.get("duration", 0) or 0
                            logger.debug(f"[Downloader]   {v['title'][:30]}: {v['duration']}s")
                        except Exception as e:
                            logger.debug(f"[Downloader]   {v['title'][:30]}: 补时长失败 - {e}")

            playlist_title = info.get("title", "合集")
            logger.info(
                f"[Downloader] 检测到合集: 「{playlist_title}」"
                f" ({playlist_count} 集)"
            )
            return {
                "title": playlist_title,
                "playlist_count": playlist_count,
                "videos": videos,
            }

        except Exception as e:
            logger.warning(f"[Downloader] 合集检测失败 (非合集或网络错误): {e}")
            return None

    def get_info(self, url: str) -> dict:
        """提取视频元数据（不下载）

        Returns:
            yt-dlp 的 info dict，包含 title, duration, upload_date, description 等
        """
        source = self.detect_source(url)
        if not source:
            raise ValueError(f"不支持的链接: {url}")

        ydl_opts = {**self._base_opts(source), "noplaylist": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info

    def download_playlist_videos(
        self,
        playlist_info: dict,
        quality: int = 720,
        max_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """批量下载合集内所有视频

        Args:
            playlist_info: detect_playlist() 返回的合集信息
            quality: 分辨率上限（合集短视频降为 720p 节省时间）
            max_count: 最多下载前 N 个（None=全部）

        Returns:
            [{result1}, {result2}, ...]  每个 result 结构与 download() 一致
        """
        videos = playlist_info.get("videos", [])
        if max_count:
            videos = videos[:max_count]

        total = len(videos)
        results = []
        logger.info(f"[Downloader] 批量下载合集: {total} 个视频 (quality≤{quality}p)")

        for i, v in enumerate(videos, 1):
            video_url = v.get("url", "")
            if not video_url:
                logger.warning(f"[Downloader] 跳过第{i}集: 无URL (title={v.get('title','?')})")
                # 仍追加空结果，防止 zip 时索引错位
                results.append({
                    "local_path": "",
                    "metadata": {"title": v.get("title", f"第{i}集"), "duration": 0,
                                 "source": v.get("source", ""), "video_id": v.get("video_id", ""),
                                 "webpage_url": ""},
                    "source": v.get("source", ""), "video_id": v.get("video_id", ""),
                    "already_downloaded": False, "error": "无URL",
                })
                continue

            # 检查缓存
            cached = self._check_cache(video_url)
            if cached and os.path.isfile(cached.get("local_path", "")):
                logger.info(f"[Downloader] [{i}/{total}] 已缓存: {v['title']}")
                results.append(self._make_result(
                    cached["local_path"], cached, already_downloaded=True))
                continue

            logger.info(f"[Downloader] [{i}/{total}] 下载: {v['title']}")
            try:
                result = self.download(video_url, quality=quality)
                results.append(result)
            except Exception as e:
                logger.error(f"[Downloader] [{i}/{total}] 下载失败: {v['title']}: {e}")
                results.append({
                    "local_path": "",
                    "metadata": {
                        "title": v["title"],
                        "duration": v["duration"],
                        "source": v.get("source", ""),
                        "video_id": v.get("video_id", ""),
                        "webpage_url": video_url,
                    },
                    "source": v.get("source", ""),
                    "video_id": v.get("video_id", ""),
                    "already_downloaded": False,
                    "error": str(e),
                })

        return results

    def download_playlist_range(
        self,
        url: str,
        indices: list,
        quality: int = 720,
    ) -> List[Dict[str, Any]]:
        """用 yt-dlp 原生 playlist_items 下载合集指定序号

        Args:
            url: 原始合集/播放列表 URL
            indices: 选中序号列表，1-based，如 [11,12,13,14,15,16,17]
            quality: 分辨率上限

        Returns:
            同 download_playlist_videos — 每个 result 含 local_path
        """
        source = self.detect_source(url)
        if not source:
            logger.error(f"[Downloader] 不支持的链接: {url}")
            return []

        if not indices:
            return []

        # 构造 playlist_items 参数：合并连续区间
        items_parts = []
        sorted_idx = sorted(set(indices))
        run_start = sorted_idx[0]
        run_end = sorted_idx[0]
        for n in sorted_idx[1:]:
            if n == run_end + 1:
                run_end = n
            else:
                items_parts.append(f"{run_start}-{run_end}" if run_start != run_end else str(run_start))
                run_start = run_end = n
        items_parts.append(f"{run_start}-{run_end}" if run_start != run_end else str(run_start))
        playlist_items = ",".join(items_parts)

        logger.info(f"[Downloader] 下载合集 items={playlist_items}: {source}")

        config.setup_proxy()

        # 用 yt-dlp 原生下载指定范围
        ydl_opts = {
            **self._base_opts(source),
            "noplaylist": False,
            "playlist_items": playlist_items,
            "format": f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]",
            "outtmpl": str(self._download_root / source / f"%(playlist_index)03d_%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "writethumbnail": False,
        }

        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

            entries = info.get("entries", []) or []
            results = []
            for entry in entries:
                if entry is None:
                    continue
                # yt-dlp 已下载到指定路径
                idx = entry.get("playlist_index", 0)
                fid = entry.get("id", "")
                ext = entry.get("ext", "mp4")
                local_path = str(self._download_root / source / f"{idx:03d}_{fid}.{ext}")

                if not os.path.isfile(local_path):
                    # 可能扩展名不同，找匹配文件
                    import glob
                    candidates = list(
                        self._download_root.glob(f"{source}/{idx:03d}_{fid}.*")
                    )
                    if candidates:
                        local_path = str(candidates[0])

                results.append({
                    "local_path": local_path if os.path.isfile(local_path) else "",
                    "metadata": {
                        "title": entry.get("title", f"第{idx}集"),
                        "duration": entry.get("duration", 0) or 0,
                        "source": source,
                        "video_id": fid,
                        "webpage_url": entry.get("webpage_url", url),
                    },
                    "source": source,
                    "video_id": fid,
                    "already_downloaded": False,
                })

                if os.path.isfile(local_path):
                    logger.info(f"  ✓ {entry.get('title', '?')[:30]} → {local_path}")
                else:
                    logger.warning(f"  ✗ {entry.get('title', '?')[:30]}: 下载后文件未找到")

            return results

        except Exception as e:
            logger.error(f"[Downloader] 合集下载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def download(self, url: str, quality: int = 1080) -> Dict[str, Any]:
        """下载视频

        Args:
            url: YouTube / Bilibili 链接
            quality: 分辨率上限（默认 1080p）

        Returns:
            {
                "local_path": str,           # 本地视频文件路径
                "metadata": {
                    "title": str,
                    "duration": int,          # 秒
                    "upload_date": str,       # YYYYMMDD
                    "description": str,
                    "source": str,            # youtube / bilibili
                    "video_id": str,
                    "webpage_url": str,
                },
                "source": str,                # youtube / bilibili
                "video_id": str,
                "already_downloaded": bool,   # 是否命中缓存
            }
        """
        source = self.detect_source(url)
        if not source:
            raise ValueError(f"不支持的链接: {url}")

        # ── 检查是否已下载（去重） ──
        cached = self._check_cache(url)
        if cached:
            local_path = cached["local_path"]
            if os.path.isfile(local_path):
                logger.info(f"[Downloader] 命中缓存: {local_path}")
                return self._make_result(local_path, cached, already_downloaded=True)

        # ── 提取元数据（不下载） ──
        logger.info(f"[Downloader] 提取元数据: {url}")
        info = self.get_info(url)
        video_id = info.get("id", "")
        title = info.get("title", "unknown")
        duration = info.get("duration", 0) or 0

        # 检查时长
        if duration > 7200:
            logger.warning(f"[Downloader] 视频过长 ({duration / 60:.0f} 分钟)，可能耗时较久")

        # ── 准备下载目录 ──
        sub_dir = self._sub_dirs[source]
        sub_dir.mkdir(parents=True, exist_ok=True)

        # ── 下载 ──
        ext = self._guess_ext(info) or "mp4"
        file_base = self._sanitize_title(title, video_id)
        filename = f"{file_base}.{ext}"
        local_path = str(sub_dir / filename)

        # 若存在同名文件（不同视频），追加 video_id 去重
        if os.path.isfile(local_path):
            logger.warning(f"[Downloader] 文件名冲突，追加 video_id 去重: {filename}")
            file_base = f"{self._sanitize_title(title, video_id, max_len=40)}_{video_id}"
            filename = f"{file_base}.{ext}"
            local_path = str(sub_dir / filename)

        temp_path = str(sub_dir / f".{video_id}.partial.{ext}")

        ydl_opts = {
            **self._base_opts(source),
            "format": f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]",
            "outtmpl": temp_path,
            "merge_output_format": ext,
        }

        logger.info(f"[Downloader] 开始下载: {title} ({source})")
        start = time.time()

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            # 清理残留文件
            self._clean_partial(sub_dir, video_id)
            logger.error(f"[Downloader] 下载失败: {e}")
            raise RuntimeError(f"下载失败: {e}") from e

        # ── 重命名临时文件 ──
        partial_candidates = list(sub_dir.glob(f".{video_id}.partial.*"))
        if partial_candidates:
            os.rename(str(partial_candidates[0]), local_path)
        elif not os.path.isfile(local_path):
            # yt-dlp 可能直接用 video_id 输出（没有走 partial）
            candidates = list(sub_dir.glob(f"{video_id}.*"))
            if candidates:
                src = str(candidates[0])
                os.rename(src, local_path)
                logger.info(f"[Downloader] 重命名: {candidates[0].name} → {filename}")

        elapsed = time.time() - start
        logger.info(f"[Downloader] 下载完成 ({elapsed:.0f}s): {local_path}")

        if not os.path.isfile(local_path):
            raise RuntimeError(f"下载后未找到视频文件: {local_path}")

        # ── 保存元数据 ──
        metadata = {
            "video_id": video_id,
            "title": title,
            "duration": duration,
            "upload_date": info.get("upload_date", ""),
            "description": info.get("description", "") or "",
            "source": source,
            "webpage_url": info.get("webpage_url", url),
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_metadata(sub_dir, file_base, metadata)

        # ── 更新索引 ──
        self._update_index(url, {
            "source": source,
            "video_id": video_id,
            "local_path": local_path,
            "title": title,
            "duration": duration,
            "downloaded_at": metadata["downloaded_at"],
        })

        return self._make_result(local_path, {
            **metadata,
            "local_path": local_path,
        }, already_downloaded=False)

    def cleanup(self, max_age_days: int = 7):
        """清理超过 max_age_days 的下载文件

        Args:
            max_age_days: 文件保留天数（默认 7）
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        cleaned = 0

        for source in self.SUPPORTED_DOMAINS:
            sub_dir = self._sub_dirs[source]
            if not sub_dir.exists():
                continue

            for f in sub_dir.iterdir():
                if f.suffix in (".mp4", ".mkv", ".webm", ".meta.json"):
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime < cutoff:
                        f.unlink()
                        cleaned += 1
                        logger.info(f"[Downloader] 清理过期文件: {f.name}")

            # 清理空目录
            remaining = [f for f in sub_dir.iterdir() if not f.name.startswith(".")]
            if not remaining:
                shutil.rmtree(sub_dir, ignore_errors=True)

        logger.info(f"[Downloader] 清理完成: 移除 {cleaned} 个文件")
        return cleaned

    # ════════════════════════════════════════════════════════
    # 内部方法
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _sanitize_title(title: str, video_id: str, max_len: int = 55) -> str:
        """将视频标题转为简短明了的文件名

        规则:
            - 移除 Windows 非法文件名字符 (\ / : * ? " < > |)
            - 移除常见 SEO 冗余前缀/后缀
            - 折叠空白字符
            - 截断到 max_len 字符
            - 若结果为空，回退到 video_id

        Args:
            title: 视频原标题
            video_id: 视频唯一 ID（冲突时回退用）
            max_len: 文件名最大长度（默认 55）

        Returns:
            安全的文件名（不含扩展名）
        """
        if not title or title == "unknown":
            return video_id

        safe = title

        # 1) 移除非法文件名字符 → 用下划线替代空格，避免文件名含空格
        safe = re.sub(r'[\\/:*?"<>|]', '_', safe)

        # 2) 移除常见冗余尾缀 — 平台标识、频道名等
        safe = re.sub(
            r'\s*[-–—|/]\s*(YouTube|Bilibili|bilibili|B站|哔哩哔哩)\s*$', '', safe
        )
        safe = re.sub(r'\s*[|]\s*\d+[:：]\d+\s*$', '', safe)

        # 3) 移除行首常见 SEO 冗余（年份标签、推广语等）
        safe = re.sub(
            r'^【\d{4}最新[^】]*】\s*', '', safe
        )
        safe = re.sub(
            r'^(B站|b站|Bilibili|bilibili)(最全|最细|最详细|最强|最完整|宝藏)[^，。！]*[，。！]\s*',
            '', safe
        )
        safe = re.sub(
            r'^(一口气|保姆级|手把手|零基础|从零开始)[^，。！]*[，。！]\s*',
            '', safe
        )

        # 4) 折叠空白 → 下划线
        safe = re.sub(r'\s+', '_', safe).strip('_')

        # 5) 截断 — 尽量在完整词/标点处断开
        if len(safe) > max_len:
            safe = safe[:max_len].rstrip('_')
            safe = re.sub(r'[\s,;.:\-_!?，。；：、！？]+$', '', safe)

        # 6) 若结果为空的防御
        safe = safe.strip('_')
        if not safe:
            return video_id

        return safe

    def _check_cache(self, url: str) -> Optional[dict]:
        """检查 URL 是否已在索引中"""
        if not self._index_path.exists():
            return None
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            return index.get("urls", {}).get(url)
        except (json.JSONDecodeError, IOError):
            return None

    def _update_index(self, url: str, entry: dict):
        """更新下载索引"""
        self._download_root.mkdir(parents=True, exist_ok=True)
        index = {"urls": {}}
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except json.JSONDecodeError:
                index = {"urls": {}}

        index.setdefault("urls", {})[url] = entry

        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def _save_metadata(self, sub_dir: Path, file_base: str, metadata: dict):
        """保存元数据到 {file_base}.meta.json"""
        meta_path = sub_dir / f"{file_base}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _guess_ext(self, info: dict) -> str:
        """从 yt-dlp info 中猜测最佳扩展名"""
        # 优先用视频流的扩展名
        requested_formats = info.get("requested_formats") or []
        for fmt in requested_formats:
            if fmt.get("vcodec") and fmt.get("ext") and fmt["vcodec"] != "none":
                return fmt["ext"]
        # 后备
        return info.get("ext", "mp4")

    def _clean_partial(self, sub_dir: Path, video_id: str):
        """清理未完成的下载文件"""
        for f in sub_dir.glob(f".{video_id}.partial.*"):
            f.unlink(missing_ok=True)

    def _make_result(self, local_path: str, entry: dict,
                     already_downloaded: bool = False) -> dict:
        """统一结果格式"""
        return {
            "local_path": local_path,
            "metadata": {
                "title": entry.get("title", ""),
                "duration": entry.get("duration", 0),
                "upload_date": entry.get("upload_date", ""),
                "description": entry.get("description", ""),
                "source": entry.get("source", ""),
                "video_id": entry.get("video_id", ""),
                "webpage_url": entry.get("webpage_url", ""),
            },
            "source": entry.get("source", ""),
            "video_id": entry.get("video_id", ""),
            "already_downloaded": already_downloaded,
        }
