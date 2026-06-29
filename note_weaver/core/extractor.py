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


# =================================================================
# 智能帧选择策略
# =================================================================

def extract_keyframes(
    video_path: str,
    output_dir: str,
    file_base: str,
    strategy: str = "hybrid",
    interval: int = 60,
    max_frames: int = 40,
    min_frames: int = 8,
    scene_threshold: float = 0.3,
) -> list:
    """智能关键帧提取

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        file_base: 文件名前缀
        strategy: fixed / scene_change / hybrid
        interval: 固定间隔策略的间隔秒数
        max_frames: 最多帧数
        min_frames: 最少帧数
        scene_threshold: scene change 敏感度（0~1，越低越敏感）

    Returns:
        截图文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)

    if strategy == "fixed":
        return extract_screenshots(video_path, output_dir, file_base, interval)

    elif strategy == "scene_change":
        timestamps = _detect_scenes(video_path, threshold=scene_threshold)
        if len(timestamps) < min_frames:
            logger.info(f"[Keyframe] scene change 只检出 {len(timestamps)} 帧，不足 {min_frames}，改用 hybrid")
            return _extract_at_timestamps(video_path, output_dir, file_base,
                                          _sample_uniform(get_video_duration(video_path),
                                                          min_frames, interval))
        if len(timestamps) > max_frames:
            timestamps = timestamps[::len(timestamps) // max_frames + 1]
        return _extract_at_timestamps(video_path, output_dir, file_base, timestamps)

    elif strategy == "hybrid":
        timestamps = _detect_scenes(video_path, threshold=scene_threshold)
        dur = get_video_duration(video_path)

        if len(timestamps) < min_frames:
            logger.info(f"[Keyframe] scene change {len(timestamps)} 帧不足，均匀采样 {min_frames} 帧")
            return _extract_at_timestamps(video_path, output_dir, file_base,
                                          _sample_uniform(dur, min_frames, interval))

        if len(timestamps) > max_frames:
            step = len(timestamps) // max_frames
            timestamps = timestamps[::step + 1][:max_frames]
            logger.info(f"[Keyframe] scene change {len(timestamps)} 帧（已压缩）")

        return _extract_at_timestamps(video_path, output_dir, file_base, timestamps)

    # fallback
    return extract_screenshots(video_path, output_dir, file_base, interval)


def _detect_scenes(video_path: str, threshold: float = 0.3) -> list:
    """用 ffmpeg scene detection 检测场景切换帧的时间戳

    Windows 兼容：手动 utf-8 解码 stderr，避免 GBK 编码问题。

    Returns:
        场景切换时间戳列表（秒）
    """
    import re
    cmd = [
        "ffmpeg", "-i", video_path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "0", "-f", "null", "-"
    ]
    try:
        # capture_output=True, text=True 在 Windows 上默认用 GBK 解码
        # ffmpeg 的 stderr 是 UTF-8 → UnicodeDecodeError
        # 改为 bytes 模式手动解码
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        timestamps = []
        for line in stderr_text.split("\n"):
            if "pts_time:" in line:
                m = re.search(r"pts_time:([\d.]+)", line)
                if m:
                    t = float(m.group(1))
                    if t > 1.0:  # 跳过第0秒的起始帧
                        timestamps.append(t)
        return timestamps
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"[Keyframe] scene detection 失败: {e}")
        return []


def _extract_at_timestamps(video_path: str, output_dir: str,
                           file_base: str, timestamps: list) -> list:
    """在指定时间戳提取帧

    使用 -ss 逐帧 seek 方式，比 select='eq(t,...)' 更可靠：
    - eq(t, N.NNN) 需要精确命中帧边界，浮点数几乎不可能对齐
    - -ss 会 seek 到最近的关键帧/时间点，保证一定出图
    """
    if not timestamps:
        return []

    # 过滤已存在的文件
    extract_ts = []
    expected_paths = []
    for t in timestamps:
        fname = f"{file_base}_{int(t):04d}s.jpg"
        fpath = os.path.join(output_dir, fname)
        expected_paths.append(fpath)
        if not os.path.exists(fpath):
            extract_ts.append((t, fpath))

    if not extract_ts:
        logger.info(f"[Keyframe] 全部 {len(timestamps)} 帧已存在，跳过提取")
        return sorted(expected_paths)

    # 用 -ss 逐帧提取（每帧一条 ffmpeg 命令）
    # 虽然慢一点，但保证每帧都出来
    success = 0
    for t, fpath in extract_ts:
        cmd = [
            "ffmpeg", "-ss", str(t),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            "-y", fpath,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            if os.path.exists(fpath) and os.path.getsize(fpath) > 1024:
                success += 1
            else:
                logger.debug(f"[Keyframe] ⏭ 跳过无效帧 t={t:.1f}s")
        except Exception as e:
            logger.debug(f"[Keyframe] 帧提取失败 t={t:.1f}s: {e}")

    # 收集结果（全部 expected_paths，包括已存在的）
    result = sorted([p for p in expected_paths if os.path.exists(p)])
    logger.info(f"[Keyframe] 提取 {success} 帧 (共请求 {len(extract_ts)} 帧, 累计 {len(result)})")
    return result


def _sample_uniform(duration: float, n_frames: int, default_interval: int = 60) -> list:
    """在视频时长内均匀采样 n_frames 个时间戳

    跳过第 0 秒（通常是黑屏/空白首帧），从 duration/n_frames 开始采。
    """
    if duration <= 0:
        return []
    step = max(duration / n_frames, default_interval)
    # 从 step/2 开始（跳过第0秒），到 duration - step/2 结束
    start = step * 0.5
    return [start + i * step for i in range(min(n_frames, int(duration / step)))]
