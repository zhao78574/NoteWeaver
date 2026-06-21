"""视频下载 Agent — yt-dlp 封装，支持 YouTube / Bilibili 链接解析与下载"""

import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import yt_dlp

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
            download_dir: 下载根目录，默认 E:/Claude/NoteWeaver/Video
        """
        base = Path(config.base_dir)  # E:\Claude\NoteWeaver
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

    def get_info(self, url: str) -> dict:
        """提取视频元数据（不下载）

        Returns:
            yt-dlp 的 info dict，包含 title, duration, upload_date, description 等
        """
        source = self.detect_source(url)
        if not source:
            raise ValueError(f"不支持的链接: {url}")

        ydl_opts = self._base_opts(source)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info

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
        filename = f"{video_id}.{ext}"
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
            # yt-dlp 可能直接输出到目标路径
            candidates = list(sub_dir.glob(f"{video_id}.*"))
            if candidates:
                local_path = str(candidates[0])

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
        self._save_metadata(sub_dir, video_id, metadata)

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

    def _save_metadata(self, sub_dir: Path, video_id: str, metadata: dict):
        """保存元数据到 {video_id}.meta.json"""
        meta_path = sub_dir / f"{video_id}.meta.json"
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
