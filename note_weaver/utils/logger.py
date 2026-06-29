"""日志系统 — 复用并增强 Auto_Pipeline.py 的日志能力"""

import os
import logging
import time
from .config import config


def setup_logger(name: str = "note_weaver") -> logging.Logger:
    """创建/获取全局 logger，自动按天分割日志"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # 已初始化

    logger.setLevel(logging.INFO)

    # 日志目录
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"note_weaver_{time.strftime('%Y%m%d')}.log")

    # 只写文件，不写终端（终端输出由 style.py 的 ok/info/step_done 负责）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # 格式
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(fmt)

    logger.addHandler(file_handler)

    return logger


# 全局 logger
logger = setup_logger()
