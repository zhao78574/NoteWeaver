"""配置管理 — 从 config.yaml + 多层 Key 解析读取，单例模式"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# 用户级全局配置目录（~/.note_weaver/）
_HOME_CONFIG_DIR = Path.home() / ".note_weaver"


class Config:
    """全局配置单例

    API Key 解析策略（Claude Code 风格，按优先级）:
      1️⃣ 环境变量          DEEPSEEK_API_KEY / QWEN_API_KEY
      2️⃣ .env 文件          项目根目录 .env（已 gitignored）
      3️⃣ 全局配置            ~/.note_weaver/config.json
      4️⃣ OS Keychain        系统钥匙串（可选，需 keyring 库）
    """

    _instance: Optional["Config"] = None
    _data: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

        # 解析路径
        base = Path(config_path).parent.parent  # note_weaver/ 的上级
        self._base_dir = base
        paths = self._data.get("paths", {})
        for key in ("txt_dir", "note_dir", "memory_dir", "log_dir"):
            raw = paths.get(key, "")
            paths[key] = str(base / raw) if raw else ""

    # ================================================================
    # 🔑 分层 API Key 解析器（核心改动）
    # ================================================================

    def _resolve_api_key(self, env_var: str, yaml_key_path: list) -> str:
        """按优先级查找 API Key，匹配所有信号源

        Args:
            env_var: 环境变量名，如 "DEEPSEEK_API_KEY"
            yaml_key_path: YAML 中的嵌套 key 路径，如 ["api","deepseek","api_key"]

        Returns:
            API Key 字符串，未找到返回 ""
        """

        # ── 1️⃣ 环境变量（最高优先级） ──
        val = os.environ.get(env_var)
        if val:
            return val

        # ── 2️⃣ .env 文件（项目级，已 gitignored） ──
        dotenv_path = self._base_dir / ".env"
        if dotenv_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path, override=False)
                val = os.environ.get(env_var)
                if val:
                    return val
            except ImportError:
                pass  # python-dotenv 未安装，跳过

        # ── 3️⃣ 全局配置 ~/.note_weaver/config.json ──
        global_cfg = _HOME_CONFIG_DIR / "config.json"
        if global_cfg.exists():
            try:
                with open(global_cfg, encoding="utf-8") as f:
                    gc = json.load(f)
                v = gc
                for k in yaml_key_path:
                    v = v.get(k, {}) if isinstance(v, dict) else {}
                if v and isinstance(v, str):
                    return v
            except Exception:
                pass

        # ── 4️⃣ OS Keychain（可选，需 keyring 库） ──
        try:
            import keyring
            val = keyring.get_password("note_weaver", env_var)
            if val:
                return val
        except ImportError:
            pass  # keyring 未安装，跳过
        except Exception:
            pass

        return ""

    # ================================================================
    # 交互式 Key 补全（CLI 友好，自动保存）
    # ================================================================

    def prompt_api_key(self, env_var: str, display_name: str) -> str:
        """交互式提示用户输入 API Key，支持存入 keychain / 全局配置"""
        print(f"\n⚠️  未检测到 {env_var}")
        val = input(f"请输入你的 {display_name} API Key: ").strip()
        if not val:
            return ""

        # 询问是否保存
        save = input("  保存到系统钥匙扣（keyring）？(y/N): ").strip().lower()
        if save in ("y", "yes"):
            try:
                import keyring
                keyring.set_password("note_weaver", env_var, val)
                print(f"  ✅ 已保存到系统钥匙扣")
                return val
            except ImportError:
                print(f"  ⚠️  keyring 未安装，尝试保存到 ~/.note_weaver/config.json")
                save = "y"

        if save in ("y", "yes"):
            try:
                _HOME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                global_cfg = _HOME_CONFIG_DIR / "config.json"
                gc = {}
                if global_cfg.exists():
                    with open(global_cfg, encoding="utf-8") as f:
                        gc = json.load(f)

                # 按 yaml_key_path 设置嵌套值
                # env_var -> DEEPSEEK_API_KEY -> ["api","deepseek","api_key"]
                key_parts = env_var.lower().replace("_api_key", "").split("_")
                keys = ["api"] + key_parts + ["api_key"]
                d = gc
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = val

                with open(global_cfg, "w", encoding="utf-8") as f:
                    json.dump(gc, f, ensure_ascii=False, indent=2)
                print(f"  ✅ 已保存到 {global_cfg}")
            except Exception as e:
                print(f"  ⚠️  保存失败: {e}")

        return val

    # ================================================================
    # 便捷属性
    # ================================================================

    @property
    def base_dir(self) -> str:
        return str(self._base_dir)

    @property
    def source_video_dir(self) -> str:
        return self._data["paths"]["source_video_dir"]

    @property
    def txt_dir(self) -> str:
        return self._data["paths"]["txt_dir"]

    @property
    def note_dir(self) -> str:
        return self._data["paths"]["note_dir"]

    @property
    def memory_dir(self) -> str:
        return self._data["paths"]["memory_dir"]

    @property
    def log_dir(self) -> str:
        return self._data["paths"]["log_dir"]

    # ---- API (DeepSeek) ----
    @property
    def deepseek_api_key(self) -> str:
        return self._resolve_api_key("DEEPSEEK_API_KEY", ["api", "deepseek", "api_key"])

    @property
    def deepseek_base_url(self) -> str:
        return self._data["api"]["deepseek"]["base_url"]

    @property
    def model_fast(self) -> str:
        return self._data["api"]["deepseek"]["model_fast"]

    @property
    def model_pro(self) -> str:
        return self._data["api"]["deepseek"]["model_pro"]

    @property
    def model_embed(self) -> str:
        return self._data["api"]["deepseek"].get("model_embed", "deepseek-embedding")

    # ---- API (Qwen Vision) ----
    @property
    def qwen_api_key(self) -> str:
        return self._resolve_api_key("QWEN_API_KEY", ["api", "qwen", "api_key"])

    @property
    def qwen_base_url(self) -> str:
        return self._data["api"]["qwen"]["base_url"]

    @property
    def qwen_model_vision(self) -> str:
        return self._data["api"]["qwen"]["model_vision"]

    # ---- Whisper ----
    def whisper_config(self) -> dict:
        w = dict(self._data["whisper"])
        if w["device"] == "auto":
            import torch
            w["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if w["compute_type"] == "auto":
            w["compute_type"] = "float16" if w["device"] == "cuda" else "int8"
        return w

    # ---- QA ----
    @property
    def qa_pass_threshold(self) -> float:
        return self._data["qa"]["pass_threshold"]

    @property
    def qa_fallback_thresholds(self) -> list:
        return self._data["qa"]["fallback_thresholds"]

    @property
    def qa_max_retries(self) -> int:
        return self._data["qa"]["max_retries"]

    @property
    def qa_weights(self) -> dict:
        return self._data["qa"]["dimension_weights"]

    # ---- Screenshot ----
    @property
    def screenshot_default_interval(self) -> int:
        return self._data["screenshot"]["default_interval"]

    # ---- Vision ----
    @property
    def vision_max_images_per_batch(self) -> int:
        return self._data["vision"]["max_images_per_batch"]

    @property
    def vision_skip_low_quality(self) -> bool:
        return self._data["vision"]["skip_low_quality"]

    # ---- Proxy ----
    def setup_proxy(self):
        """设置 HTTP 代理环境变量"""
        p = self._data["proxy"]
        if p.get("enabled", False):
            os.environ["HTTP_PROXY"] = f"http://{p['host']}:{p['port']}"
            os.environ["HTTPS_PROXY"] = f"http://{p['host']}:{p['port']}"

    # ---- 通用访问 ----
    def __getitem__(self, key: str) -> Any:
        keys = key.split(".")
        val = self._data
        for k in keys:
            val = val[k]
        return val

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default

    def __repr__(self):
        return f"<Config base={self._base_dir}>"


# 全局单例
config = Config()
