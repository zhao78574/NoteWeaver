#!/usr/bin/env python3
"""NoteWeaver Agent — 自主 AI 笔记数字助理

安装: pip install -e /path/to/NoteWeaver
用法:
    weaver                               # Agent 交互模式
    weaver "PIE和TD PIE有什么区别？"      # 单次问答
    weaver ./lecture.mp4                 # 直接处理视频（自动识别路径）
    weaver ./videos/                     # 直接处理整个文件夹
    weaver --batch ./videos/             # 批量处理
"""

import os, sys, argparse, json, pathlib, re
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

__version__ = "1.0.0"

# 作为 pip 包导入时无需手动加 sys.path
if __name__ == "__main__":
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

from note_weaver.utils.config import config
from note_weaver.utils.logger import logger
from note_weaver.agent import NoteWeaverAgent


# ── 支持的文件格式 ──────────────────────────────────────────────
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4a", ".mp3", ".wav", ".flac", ".aac"}


def _is_media_file(text: str) -> bool:
    """判断输入是否为媒体文件路径"""
    p = pathlib.Path(text.strip("\"' "))
    return p.suffix.lower() in _VIDEO_EXTS and p.exists()


def _is_media_dir(text: str) -> bool:
    """判断输入是否为包含媒体文件的目录"""
    p = pathlib.Path(text.strip("\"' "))
    return p.is_dir() and any(
        f.suffix.lower() in _VIDEO_EXTS for f in p.iterdir()
    )


def _extract_path_from_text(text: str):
    """从自然语言中提取文件/文件夹路径

    返回: (path, is_file) 或 None
    """
    # 1. 尝试取引号内的完整路径
    for q in ('"', '"', "'"):
        if q in text:
            parts = text.split(q)
            for i, part in enumerate(parts):
                if i % 2 == 1 and part.strip():
                    p = pathlib.Path(part.strip())
                    if p.exists():
                        return str(p.resolve()), p.is_file()

    # 2. 尝试匹配盘符路径如 E:\xxx 或 E:/xxx
    for m in re.finditer(r'([A-Za-z]:[\\/][^\s,;)\]}"\']+)', text):
        p = pathlib.Path(m.group(1).strip())
        if p.exists():
            return str(p.resolve()), p.is_file()

    # 3. 尝试匹配相对路径 ./ 或 ../
    for m in re.finditer(r'([.][.\/\\][^\s,;)\]}"\']+)', text):
        p = pathlib.Path(m.group(1).strip())
        if p.exists():
            return str(p.resolve()), p.is_file()

    return None


def _resolve_media_path(text: str) -> str:
    """将可能含引号的路径清理后返回绝对路径"""
    p = pathlib.Path(text.strip("\"' "))
    return str(p.resolve())


def _scan_videos(directory: str) -> list:
    """扫描目录中所有支持的媒体文件，返回绝对路径列表"""
    supported = list(_VIDEO_EXTS)
    videos = []
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in [s.lower() for s in supported]:
            videos.append(os.path.join(directory, f))
    return videos


# ── 命令处理 ────────────────────────────────────────────────────

def cmd_agent(user_input: str = ""):
    """Agent 模式：感知→决策→行动→学习→响应"""
    agent = NoteWeaverAgent()
    response = agent.run(user_input)
    print(response)


def cmd_process(video_path: str):
    agent = NoteWeaverAgent()
    print(f"🎬 开始处理: {video_path}")
    response = agent.run(f"处理视频: {video_path}")
    print(response)


def cmd_batch(directory: str):
    if not os.path.isdir(directory):
        logger.error(f"目录不存在: {directory}")
        return
    videos = _scan_videos(directory)
    if not videos:
        print(f"📭 {directory} 中没有视频文件。")
        return
    print(f"📂 从 {directory} 中找到 {len(videos)} 个视频\n")
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {'='*50}")
        cmd_process(vp)


def cmd_interactive():
    """交互式 Agent 对话循环——自然语言 + 视频路径混合输入"""
    agent = NoteWeaverAgent()

    # 先自主检入
    initial = agent.run("")
    print(initial)
    print()

    print("📌 直接输入：问题(问答) | 视频/文件夹路径(处理) | /quit")
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("再见！")
            break

        if user_input.lower() == "/stats":
            user_input = "学习统计"

        if user_input.lower().startswith("/search "):
            user_input = user_input[8:]

        # ── 智能路径识别 ──
        # 情况 A: 纯路径（直接是文件或文件夹路径）
        if _is_media_file(user_input):
            cmd_process(_resolve_media_path(user_input))
        elif _is_media_dir(user_input):
            cmd_batch(_resolve_media_path(user_input))
        else:
            # 情况 B: 自然语言中嵌了路径
            extracted = _extract_path_from_text(user_input)
            if extracted:
                path, is_file = extracted
                if is_file:
                    cmd_process(path)
                else:
                    cmd_batch(path)
            else:
                # 情况 C: 纯自然语言 → Agent 处理
                response = agent.run(user_input)
                print()
                print(response)

        print()


def main():
    parser = argparse.ArgumentParser(
        description="[NoteWeaver Agent] 自主 AI 笔记数字助理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  weaver                                    # 交互模式
  weaver "PIE和TD PIE有什么区别？"          # 单次问答
  weaver lecture.mp4                        # 直接处理单个视频
  weaver E:\\视频\\                          # 直接处理整个文件夹
  weaver --batch ./videos/                  # 批量处理
        """,
    )
    parser.add_argument("message", nargs="?", default="",
                        help="自然语言问题 或 视频/文件夹路径（自动识别）")
    parser.add_argument("--video", help="快捷: 处理单个视频")
    parser.add_argument("--batch", help="快捷: 批量处理目录")
    parser.add_argument("--version", action="store_true", help="显示版本号并退出")

    args = parser.parse_args()

    if args.version:
        print(f"NoteWeaver v{__version__}")
        return

    for d in [config.txt_dir,
               config.note_dir, config.memory_dir, config.log_dir]:
        os.makedirs(d, exist_ok=True)

    if args.video:
        cmd_process(args.video)
    elif args.batch:
        cmd_batch(args.batch)
    elif args.message:
        msg = args.message.strip()
        if _is_media_file(msg):
            cmd_process(_resolve_media_path(msg))
        elif _is_media_dir(msg):
            cmd_batch(_resolve_media_path(msg))
        else:
            cmd_agent(msg)
    else:
        cmd_interactive()


if __name__ == "__main__":
    main()
