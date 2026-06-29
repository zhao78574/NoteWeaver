"""PipelineCache — 本地磁盘计算缓存

缓存键设计：
    key = hash(input_path + stage_id + config_version)
    保证输入/模型/配置任一变化 → cache miss

三层缓存：
    raw   → 原始转录结果（最重开销，节省重复转录）
    mid   → Vision 分析结果（节省重复 VLM API 调用，$$$）
    final → 最终笔记（节省重复排版，轻量）

用法:
    from note_weaver.core.cache import PipelineCache

    cache = PipelineCache()
    cached = cache.get(input_path, "transcribe", {"model": "small"})
    if cached:
        return cached
    result = transcribe(input_path)
    cache.set(input_path, "transcribe", result, {"model": "small"})
"""

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from note_weaver.utils.logger import logger


# 缓存层定义
CACHE_LAYERS = {
    "raw": {      # 原始转录
        "stages": ["transcribe"],
        "ttl": 86400 * 30,  # 30 天
    },
    "mid": {      # 中间分析
        "stages": ["vision", "classify", "extract", "router"],
        "ttl": 86400 * 7,   # 7 天
    },
    "final": {    # 最终产物
        "stages": ["compose", "qa", "save"],
        "ttl": 86400 * 365, # 1 年
    },
}


class PipelineCache:
    """三层磁盘缓存系统

    Args:
        cache_dir: 缓存根目录（默认 data/cache）
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 公开接口 ────────────────────────────────────────────────────

    def get(self, input_path: str, stage: str, params: dict = None) -> Optional[Any]:
        """读取缓存

        Args:
            input_path: 输入文件路径
            stage: 阶段 ID（如 "transcribe", "vision"）
            params: 可选的配置参数（影响缓存键）

        Returns:
            缓存数据，miss 返回 None
        """
        cache_path = self._cache_path(input_path, stage, params)
        if not cache_path.exists():
            return None

        # 检查 TTL
        if self._is_expired(input_path, stage):
            logger.debug(f"[Cache] ⏭ {stage}: 已过期，跳过")
            self.invalidate(input_path, stage)
            return None

        try:
            with open(cache_path / "data.pkl", "rb") as f:
                data = pickle.load(f)
            meta = self._read_meta(cache_path)
            size_hint = meta.get("size", 0)
            logger.info(
                f"[Cache] [HIT] {stage} "
                f"({size_hint / 1024:.1f}KB)" if size_hint else
                f"[Cache] [HIT] {stage}"
            )
            return data
        except (pickle.PickleError, EOFError, OSError) as e:
            logger.warning("[Cache] [WARN] 读取失败 (stage={}): {}，清除".format(stage, e))
            self.invalidate(input_path, stage)
            return None

    def set(self, input_path: str, stage: str, data: Any, params: dict = None):
        """写入缓存

        Args:
            input_path: 输入文件路径
            stage: 阶段 ID
            data: 要缓存的数据（需可 pickle）
            params: 可选的配置参数
        """
        cache_path = self._cache_path(input_path, stage, params)
        cache_path.mkdir(parents=True, exist_ok=True)

        try:
            # 写入数据
            with open(cache_path / "data.pkl", "wb") as f:
                pickle.dump(data, f)

            # 写入元数据
            meta = self._build_meta(input_path, stage, data)
            with open(cache_path / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            size_hint = meta.get("size", 0)
            logger.info(
                f"[Cache] [SAVE] {stage} "
                f"({size_hint / 1024:.1f}KB)" if size_hint else
                f"[Cache] [SAVE] {stage}"
            )
        except (pickle.PickleError, OSError) as e:
            logger.warning(f"[Cache] [WARN] 写入失败 (stage={stage}): {e}")

    def invalidate(self, input_path: str, stage: str = None):
        """删除缓存

        Args:
            input_path: 输入文件路径
            stage: 指定阶段（None = 清除该 input 的所有缓存）
        """
        if stage:
            # 删除特定 stage
            cache_path = self._cache_path(input_path, stage)
            if cache_path.exists():
                shutil.rmtree(str(cache_path), ignore_errors=True)
                logger.debug(f"[Cache] [DEL] INVALIDATE {stage}")
        else:
            # 删除该 input 的所有缓存
            pattern = self._input_pattern(input_path)
            for d in self.cache_dir.glob(f"{pattern}*"):
                shutil.rmtree(str(d), ignore_errors=True)
            logger.info(f"[Cache] [DEL] INVALIDATE ALL for {Path(input_path).name}")

    def clear_all(self, layer: str = None):
        """清空全部/指定层缓存

        Args:
            layer: "raw"/"mid"/"final"/None（全部）
        """
        if layer:
            valid_stages = CACHE_LAYERS[layer]["stages"]
            for stage_dir in self.cache_dir.iterdir():
                if stage_dir.is_dir():
                    # 每个 stage_dir 下是各次缓存的子目录
                    for cache_entry in stage_dir.iterdir():
                        if cache_entry.is_dir():
                            meta = self._read_meta(cache_entry)
                            if meta and meta.get("stage") in valid_stages:
                                shutil.rmtree(str(cache_entry), ignore_errors=True)
            logger.info(f"[Cache] [CLEAN] CLEAR layer={layer}")
        else:
            shutil.rmtree(str(self.cache_dir), ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[Cache] [CLEAN] CLEAR ALL")

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计

        Returns:
            {
                "entries": 总缓存条目数,
                "size_bytes": 总磁盘占用,
                "layers": {"raw": {"entries": ..., "size_bytes": ...}, ...},
                "stages": {"transcribe": {"entries": ..., "size_bytes": ...}, ...}
            }
        """
        stats = {
            "entries": 0,
            "size_bytes": 0,
            "layers": {layer: {"entries": 0, "size_bytes": 0} for layer in CACHE_LAYERS},
            "stages": {},
        }
        for stage_dir in self.cache_dir.iterdir():
            if not stage_dir.is_dir():
                continue
            for cache_entry in stage_dir.iterdir():
                if not cache_entry.is_dir():
                    continue
                meta = self._read_meta(cache_entry)
                if not meta:
                    continue
                size = sum(f.stat().st_size for f in cache_entry.rglob("*") if f.is_file())
                stage = meta.get("stage", "unknown")
                layer = meta.get("layer", "unknown")

                stats["entries"] += 1
                stats["size_bytes"] += size
                if stage not in stats["stages"]:
                    stats["stages"][stage] = {"entries": 0, "size_bytes": 0}
                stats["stages"][stage]["entries"] += 1
                stats["stages"][stage]["size_bytes"] += size
                if layer in stats["layers"]:
                    stats["layers"][layer]["entries"] += 1
                    stats["layers"][layer]["size_bytes"] += size

        return stats

    # ── 内部方法 ────────────────────────────────────────────────────

    def _cache_path(self, input_path: str, stage: str, params: dict = None) -> Path:
        """生成缓存路径: cache_dir / <key> / <stage> /"""
        key = self._key(input_path, params)
        return self.cache_dir / key / stage

    def _input_pattern(self, input_path: str) -> str:
        """生成 glob 匹配模式"""
        raw = input_path + "::"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _key(self, input_path: str, params: dict = None) -> str:
        """生成缓存键

        键组成：
        - input_path 的 hash（16 字符）
        - 可选 param hash（用于区分模型/配置变化）
        """
        h = hashlib.sha256()
        h.update(input_path.encode())
        if params:
            h.update(json.dumps(params, sort_keys=True).encode())
        return h.hexdigest()[:16]

    def _layer_for_stage(self, stage: str) -> str:
        """根据 stage 查找所属缓存层"""
        for layer_name, layer_cfg in CACHE_LAYERS.items():
            if stage in layer_cfg["stages"]:
                return layer_name
        return "unknown"

    def _is_expired(self, input_path: str, stage: str) -> bool:
        """检查缓存是否过期"""
        import time
        cache_path = self._cache_path(input_path, stage)
        meta = self._read_meta(cache_path)
        if not meta:
            return True

        created = meta.get("created_at", 0)
        layer = meta.get("layer", "unknown")
        ttl = CACHE_LAYERS.get(layer, {}).get("ttl", 86400)

        return (time.time() - created) > ttl

    def _build_meta(self, input_path: str, stage: str, data: Any) -> dict:
        """构建缓存元数据"""
        import time
        from note_weaver.utils.config import config

        # 估算数据大小（粗略）
        size = 0
        try:
            if isinstance(data, (str, bytes)):
                size = len(data)
            elif isinstance(data, dict):
                size = len(json.dumps(data, default=str))
            elif isinstance(data, (list, tuple)):
                size = sum(
                    len(json.dumps(d, default=str)) if isinstance(d, (dict, list))
                    else len(str(d))
                    for d in data[:100]
                )
        except Exception:
            size = 0

        return {
            "input_path": input_path,
            "input_name": Path(input_path).name,
            "stage": stage,
            "layer": self._layer_for_stage(stage),
            "created_at": time.time(),
            "size": size,
            "config_version": self._config_version(config),
        }

    @staticmethod
    def _read_meta(cache_path: Path) -> Optional[dict]:
        """读取缓存目录中的 meta.json"""
        meta_path = cache_path / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    @staticmethod
    def _config_version(config_obj) -> str:
        """获取配置版本（用于缓存键判等）"""
        try:
            # 提取关键配置字段
            version_parts = [
                config_obj.get("whisper.model_size", "small"),
                config_obj.get("vision.max_images_per_batch", "10"),
                config_obj.get("api.deepseek.model_pro", ""),
            ]
            return "::".join(version_parts)
        except Exception:
            return "unknown"
