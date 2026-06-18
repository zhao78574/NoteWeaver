"""音视频提取器 — 从 Auto_Pipeline.py 提取并增强"""

import os
import subprocess
from note_weaver.utils.logger import logger


def extract_audio(video_path: str, output_path: str, timeout: int = 600) -> str:
    """用 ffmpeg 从视频中提取音频为 mp3

    Args:
        video_path: 视频文件路径
        output_path: 音频输出路径 (.mp3)
        timeout: 超时秒数

    Returns:
        output_path

    Raises:
        subprocess.CalledProcessError: ffmpeg 失败
        subprocess.TimeoutExpired: 超时
    """
    logger.info(f"提取音频: {os.path.basename(video_path)} → {os.path.basename(output_path)}")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", "-y", output_path],
        check=True, capture_output=True, timeout=timeout,
    )
    logger.info(f"音频提取完成: {output_path}")
    return output_path


def extract_screenshots(
    video_path: str,
    output_dir: str,
    file_base: str,
    interval: int = 180,
    timeout: int = 600,
) -> list:
    """用 ffmpeg 定时提取视频关键帧

    Args:
        video_path: 视频文件路径
        output_dir: 截图输出目录
        file_base: 文件名（不含扩展名），用作截图前缀
        interval: 截图间隔秒数
        timeout: 超时秒数

    Returns:
        截图文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    img_pattern = os.path.join(output_dir, f"{file_base}_%03d.jpg")
    logger.info(f"提取截图（间隔 {interval}s）: {os.path.basename(video_path)}")

    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"fps=1/{interval}", img_pattern, "-y"],
        check=True, capture_output=True, timeout=timeout,
    )

    # 收集生成的截图
    screenshots = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(file_base) and f.endswith(".jpg")
    ])
    logger.info(f"截图提取完成: {len(screenshots)} 张, {output_dir}")
    return screenshots


def clean_screenshot_dir(output_dir: str, file_base: str):
    """清理并创建截图文件夹"""
    if os.path.exists(output_dir):
        for fname in os.listdir(output_dir):
            if fname.endswith(".jpg"):
                os.remove(os.path.join(output_dir, fname))
    else:
        os.makedirs(output_dir, exist_ok=True)
    logger.info(f"已清理/创建截图文件夹: {output_dir}")


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0
