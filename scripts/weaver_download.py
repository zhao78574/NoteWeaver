#!/usr/bin/env python3
"""独立下载入口 — 下载 YouTube/B站视频 → 调用 weaver 处理

用法:
    python scripts/weaver_download.py "https://youtu.be/xxxxx"
    python scripts/weaver_download.py "https://www.bilibili.com/video/BVxxxxx"
    python scripts/weaver_download.py --info "https://youtu.be/xxxxx"    # 仅查看元数据
    python scripts/weaver_download.py --repl                            # 自然语言模式
"""

import sys
import builtins
import subprocess
import argparse
from pathlib import Path

# 确保能找到 note_weaver 包
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from note_weaver.agents.downloader import VideoDownloader
from note_weaver.utils.logger import logger


# ════════════════════════════════════════════════════════
# Windows GBK 终端兼容
# ════════════════════════════════════════════════════════

def _e(text: str) -> str:
    """过滤终端不支持的字符（emoji 等）"""
    try:
        text.encode(sys.stdout.encoding or "utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8"
        )


def print(*args, **kwargs):
    """安全的 print，自动过滤终端不支持的字符"""
    text = " ".join(str(a) for a in args)
    kwargs.pop("flush", None)
    builtins.print(_e(text), **kwargs)


# ════════════════════════════════════════════════════════
# 命令实现
# ════════════════════════════════════════════════════════

def cmd_download(url: str, quality: int = 1080, skip_weaver: bool = False):
    """下载视频 -> 调用 weaver 处理"""
    downloader = VideoDownloader()

    print(f"\n[Download] 检测链接: {url}")
    source = downloader.detect_source(url)
    if not source:
        print(f"[X] 不支持的链接: {url}")
        print(f"   支持的平台: YouTube (youtube.com/youtu.be), Bilibili (bilibili.com/b23.tv)")
        sys.exit(1)

    print(f"   -> 平台: {source.upper()}")

    print(f"[Download] 正在下载视频...")
    try:
        result = downloader.download(url, quality=quality)
    except Exception as e:
        print(f"[X] 下载失败: {e}")
        sys.exit(1)

    meta = result["metadata"]
    duration_min = meta["duration"] / 60

    print(f"\n[OK] 下载完成")
    print(f"  标题: {meta['title']}")
    print(f"  时长: {duration_min:.1f} 分钟")
    print(f"  路径: {result['local_path']}")
    if result["already_downloaded"]:
        print(f"  (使用缓存，已下载过该视频)")

    if skip_weaver:
        print(f"\n[Skip] 跳过 weaver 处理 (--skip-weaver)")
        return

    # 调用 weaver 处理
    print(f"\n[Run] 调用 weaver 处理视频...")
    print(f"   {'=' * 40}")
    try:
        subprocess.run(["weaver", result["local_path"]], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[X] weaver 处理失败 (exit code {e.returncode})")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n[!] 未找到 weaver 命令，请确保 NoteWeaver 已安装")
        print(f"   视频已下载到: {result['local_path']}")
        sys.exit(1)

    print(f"\n[Done] 全部完成！")


def cmd_info(url: str):
    """仅查看视频元数据，不下载"""
    downloader = VideoDownloader()

    source = downloader.detect_source(url)
    if not source:
        print(f"[X] 不支持的链接: {url}")
        sys.exit(1)

    print(f"\n[Info] 提取元数据: {url}")
    try:
        info = downloader.get_info(url)
    except Exception as e:
        print(f"[X] 提取失败: {e}")
        sys.exit(1)

    duration = info.get("duration", 0) or 0
    duration_str = f"{int(duration // 60)} 分 {int(duration % 60)} 秒"

    print(f"\n[Info] 视频信息")
    print(f"   标题:       {info.get('title', 'N/A')}")
    print(f"   平台:       {source.upper()}")
    print(f"   时长:       {duration_str}")
    print(f"   上传日期:   {info.get('upload_date', 'N/A')}")
    print(f"   视频 ID:    {info.get('id', 'N/A')}")
    print(f"   分辨率:     {info.get('height', 'N/A')}p")
    print(f"   格式:       {info.get('ext', 'N/A')}")
    print(f"   视频链接:   {info.get('webpage_url', url)}")

    description = info.get("description", "") or ""
    if description:
        print(f"\n   简介（前 200 字）:")
        print(f"   {description[:200].strip()}")


def cmd_cleanup(max_age_days: int = 7):
    """清理过期下载缓存"""
    downloader = VideoDownloader()
    print(f"\n[Clean] 清理 {max_age_days} 天前的下载缓存...")
    count = downloader.cleanup(max_age_days=max_age_days)
    print(f"   完成，移除了 {count} 个文件")


# ════════════════════════════════════════════════════════
# 自然语言交互模式
# ════════════════════════════════════════════════════════

def cmd_repl():
    """自然语言交互模式"""
    import re

    downloader = VideoDownloader()
    URL_PATTERN = r"https?://[^\s]+"

    print()
    print("=" * 50)
    print("  NoteWeaver 下载助手 (自然语言模式)")
    print("=" * 50)
    print("  直接粘贴链接给我，或者说:")
    print("    [下载这个视频 https://...]")
    print("    [看看这个 https://... 的信息]")
    print("    [清理缓存]")
    print("    [退出] / quit")
    print()

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye 再见！")
            break

        if not text:
            continue

        cmd = text.lower().strip()

        if cmd in ("退出", "quit", "exit", "q"):
            print("Bye 再见！")
            break

        if cmd in ("清理", "清理缓存", "cleanup", "clean"):
            cmd_cleanup(7)
            continue

        # 从文本中提取链接
        urls = re.findall(URL_PATTERN, text)
        if not urls:
            print("   [X] 没找到链接，粘贴 YouTube 或 B站链接给我")
            continue

        for url in urls:
            # 判断意图：包含"看看"/"信息"/"查看" -> info，否则下载
            if any(kw in cmd for kw in ("看看", "信息", "查看", "info")):
                print(f"\n[Info] 查看: {url}")
                try:
                    cmd_info(url)
                except SystemExit:
                    pass
            else:
                # 判断画质
                quality = 1080
                if "720" in cmd or "720p" in cmd:
                    quality = 720
                elif "480" in cmd or "480p" in cmd:
                    quality = 480

                skip_weaver = "仅下载" in cmd or "不处理" in cmd

                print(f"\n[Download]: {url}")
                try:
                    cmd_download(url, quality=quality, skip_weaver=skip_weaver)
                except SystemExit:
                    pass

            print()

    print()


# ════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NoteWeaver 视频下载器 - 下载 YouTube/B站视频并自动处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s "https://youtu.be/dQw4w9WgXcQ"
  %(prog)s "https://www.bilibili.com/video/BV1GJ411x7FD"
  %(prog)s --info "https://youtu.be/xxxxx"
  %(prog)s --quality 720 "https://youtu.be/xxxxx"
  %(prog)s --skip-weaver "https://youtu.be/xxxxx"
  %(prog)s --cleanup
  %(prog)s --repl
        """,
    )
    parser.add_argument("url", nargs="?", help="YouTube / Bilibili 视频链接")
    parser.add_argument("--info", action="store_true", help="仅查看元数据，不下载")
    parser.add_argument("--quality", type=int, default=1080,
                        help="下载分辨率上限（默认 1080）")
    parser.add_argument("--skip-weaver", action="store_true",
                        help="仅下载，不调用 weaver 处理")
    parser.add_argument("--cleanup", action="store_true",
                        help="清理过期下载缓存（默认 7 天）")
    parser.add_argument("--cleanup-days", type=int, default=7,
                        help="清理天数阈值（配合 --cleanup 使用）")
    parser.add_argument("--repl", action="store_true",
                        help="自然语言交互模式")

    args = parser.parse_args()

    if args.repl:
        cmd_repl()
        return

    if args.cleanup:
        cmd_cleanup(args.cleanup_days)
        return

    if args.info:
        if not args.url:
            parser.error("--info 需要提供视频链接")
        cmd_info(args.url)
        return

    if not args.url:
        parser.error("请提供视频链接，或用 --cleanup 清理缓存")
        return

    cmd_download(args.url, quality=args.quality, skip_weaver=args.skip_weaver)


if __name__ == "__main__":
    main()
