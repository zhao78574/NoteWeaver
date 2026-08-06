"""NoteWeaver Template Engine — 模板加载、继承合并、Prompt 构建

用法:
    from note_weaver.core.template import TemplateEngine

    # 加载模板（自动继承合并）
    tmpl = TemplateEngine.load("semiconductor")

    # 列出所有可用模板
    TemplateEngine.list_all()

    # 构建 Composer System Prompt
    prompt = TemplateEngine.build_composer_prompt(tmpl)

    # 按视频类型推荐模板
    tmpl = TemplateEngine.recommend("meeting")
"""

import os, yaml
from dataclasses import dataclass, field

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


@dataclass
class NoteTemplate:
    name: str
    label: str = ""
    description: str = ""
    inherits: str = "_base"
    composer: dict = field(default_factory=dict)
    classifier: dict = field(default_factory=dict)
    vision: dict = field(default_factory=dict)
    qa: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


class TemplateEngine:
    """模板引擎 — 负责加载、合并、缓存和 Prompt 构建"""

    _cache: dict[str, NoteTemplate] = {}
    _template_list: list[dict] = []

    @classmethod
    def _scan(cls):
        """扫描 templates/ 目录，建立索引"""
        if cls._template_list:
            return cls._template_list
        cls._template_list = []
        if not os.path.isdir(TEMPLATE_DIR):
            return cls._template_list
        for fname in sorted(os.listdir(TEMPLATE_DIR)):
            if fname.endswith(".yaml") and not fname.startswith("_"):
                path = os.path.join(TEMPLATE_DIR, fname)
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    cls._template_list.append({
                        "name": data["name"],
                        "label": data.get("label", data["name"]),
                        "description": data.get("description", ""),
                        "file": fname,
                    })
        return cls._template_list

    @classmethod
    def list_all(cls) -> list[dict]:
        """返回所有可用模板列表"""
        return cls._scan()

    @classmethod
    def load(cls, name: str = "semiconductor") -> NoteTemplate:
        """加载模板，自动处理继承合并"""
        if name in cls._cache:
            return cls._cache[name]

        path = os.path.join(TEMPLATE_DIR, f"{name}.yaml")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"模板 '{name}' 不存在（{path}）")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 继承合并（深度合并）
        inherit = data.get("inherits")
        if inherit and inherit != "_base":
            base = cls.load(inherit)
            merged = cls._deep_merge(base.__dict__.copy(), data)
        elif inherit == "_base":
            base_path = os.path.join(TEMPLATE_DIR, "_base.yaml")
            if os.path.isfile(base_path):
                with open(base_path, encoding="utf-8") as bf:
                    base_data = yaml.safe_load(bf)
                merged = cls._deep_merge(base_data.copy(), data)
            else:
                merged = data
        else:
            merged = data

        # 确保 name 不变
        merged["name"] = name
        template = NoteTemplate(**merged)
        cls._cache[name] = template
        return template

    @classmethod
    def recommend(cls, video_type: str) -> str:
        """根据视频类型推荐模板名"""
        type_map = {
            "lecture": "academic",
            "seminar": "academic",
            "talk": "academic",
            "meeting": "meeting",
            "discussion": "meeting",
            "demo": "tutorial",
            "code_walkthrough": "tutorial",
            "tutorial": "tutorial",
        }
        return type_map.get(video_type, "semiconductor")

    # ── Prompt 构建 ──────────────────────────────────────

    @classmethod
    def build_composer_prompt(cls, template: NoteTemplate) -> str:
        """从模板构建 Composer 完整 System Prompt"""
        c = template.composer
        role = c.get("role", "笔记整理助手")
        audience = c.get("audience", "读者")
        tone = c.get("tone", "自然清晰")

        sections = c.get("output_sections", [])
        sections_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sections))

        rules = c.get("rules", [])
        rules_text = "\n".join(f"- {r}" for r in rules)

        return f"""你是一位{role}，正在为{audience}整理笔记。

## 笔记风格
{tone}

## 笔记结构
{sections_text}

## 内容要求
{rules_text}

## 输出要求
只输出笔记正文，不要开场白/总结。"""

    @classmethod
    def build_classifier_system_extra(cls, template: NoteTemplate) -> str:
        """从模板构建 Classifier 的额外领域提示"""
        domains = template.classifier.get("domains", [])
        if domains:
            return f"\n可能的领域标签：{', '.join(domains)}"
        return ""

    @classmethod
    def build_vision_focus(cls, template: NoteTemplate) -> str:
        """从模板构建 Vision 的额外关注提示"""
        focus = template.vision.get("focus", [])
        if focus:
            return "\n重点关注画面类型：\n" + "\n".join(f"- {f}" for f in focus)
        return ""

    # ── 内部工具 ──────────────────────────────────────

    @classmethod
    def _deep_merge(cls, base: dict, override: dict) -> dict:
        """深度合并两个字典（override 覆盖 base）"""
        result = base.copy()
        for key, val in override.items():
            if key == "name":
                continue
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = cls._deep_merge(result[key], val)
            else:
                result[key] = val
        return result

    @classmethod
    def clear_cache(cls):
        """清除模板缓存（用于热加载）"""
        cls._cache.clear()
        cls._template_list.clear()
