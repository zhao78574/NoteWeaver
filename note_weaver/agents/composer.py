"""笔记排版 Agent — 融合转录+视觉+上下文，生成高质量笔记"""

import os
import re
from typing import Any, Dict, List, Optional
from .base import BaseAgent
from note_weaver.utils.logger import logger
from note_weaver.utils.prompts import COMPOSER_SYSTEM, build_composer_user_prompt


# ── 笔记后处理工具函数 ─────────────────────────────────────────

def _extract_hash_from_filename(image_id: str) -> str:
    """从 image_id 中解析内容hash，用于跨页去重"""
    m = re.search(r'_([a-f0-9]{8,16})\.[a-z]+$', image_id)
    return m.group(1) if m else ""


def _replace_placeholders(note_text: str, file_base: str, vision_results: list) -> tuple:
    """替换 [图片: path] 占位符为 ![](path) 语法，返回 (替换后的文本, 已替换数)"""
    # 构建图片映射
    img_map = {}
    for r in vision_results:
        if not r.get("should_include", True):
            continue
        img_id = r.get("image_id", "")
        if img_id:
            rel = f"{file_base}/{img_id}"
            img_map[rel] = rel
            img_map[img_id] = rel

    count = 0
    def _replace(m):
        nonlocal count
        key = m.group(1)
        target = img_map.get(key)
        if target:
            count += 1
            return f"\n![]({target})\n"
        return m.group(0)

    note_text = re.sub(r'\[图片:\s*([^\]]+)\]', _replace, note_text)
    return note_text, count


def _fix_broken_markdown_images(note_text: str) -> str:
    """修复 Composer 生成的嵌套/断裂 ![]() markdown

    情况1: ![](![path](path)) → ![](path)
    情况2: ![](path1](path2) → ![](path1)
    """
    note_text = re.sub(
        r'!\[\]\(!\[([^\]]+)\]\(([^)]+)\)\)',
        r'![](\1)',
        note_text
    )
    note_text = re.sub(
        r'!\[\]\(([^)\]]+)\]\([^)]+\)',
        r'![](\1)',
        note_text
    )
    return note_text


class ComposerAgent(BaseAgent):
    """将转录文本 + 图片描述融合成结构化手写感笔记"""

    def __init__(self):
        # Composer 使用 Pro 模型 — 更强的推理和排版能力
        from note_weaver.utils.config import config
        super().__init__(model_name=config.model_pro)

    def execute(
        self,
        file_base: str,
        timestamped_text: str,
        vision_results: List[Dict[str, Any]],
        strategy: Optional[Dict[str, Any]] = None,
        user_context: str = "",
        revision_feedback: str = "",
    ) -> str:
        """
        Args:
            file_base: 文件名（不含扩展名）
            timestamped_text: 带时间戳的转录文本
            vision_results: Vision Agent 的分析结果列表
            strategy: 分类+策略配置
            user_context: Memory Agent 提供的用户背景
            revision_feedback: QA回退时的修改意见（空表示首次生成）

        Returns:
            完整的 Markdown 笔记内容
        """
        strategy = strategy or {}
        note_style = strategy.get("note_style", "detailed")
        focus_areas = ", ".join(strategy.get("focus_areas", []))

        # 构建图片描述文本
        image_descriptions = self._format_vision_results(vision_results, file_base)

        logger.info(
            f"[Composer] 开始排版: {file_base} "
            f"({len(vision_results)} 图, style={note_style}, "
            f"revision={bool(revision_feedback)})"
        )

        # 构建 prompt
        prompt = build_composer_user_prompt(
            file_base=file_base,
            timestamped_text=timestamped_text,
            image_descriptions=image_descriptions,
            user_context=user_context,
            focus_areas=focus_areas,
            note_style=note_style,
        )

        if revision_feedback:
            prompt = (
                f"【修改要求】上次的笔记有以下问题，请针对性修改：\n"
                f"{revision_feedback}\n\n---\n\n"
                f"{prompt}"
            )

        # 调用 Gemini 生成笔记
        note_content = self.chat(
            prompt,
            system_instruction=COMPOSER_SYSTEM,
        )

        # ── 后处理：替换 [图片: path] 占位符 + 修复断裂 markdown ──
        note_content, placeholder_count = _replace_placeholders(
            note_content, file_base, vision_results)
        if placeholder_count:
            logger.info(f"[Composer] 图片占位符替换: {placeholder_count} 张")
        note_content = _fix_broken_markdown_images(note_content)

        logger.info(f"[Composer] 完成: {len(note_content)} 字符")
        return note_content

    def _format_vision_results(
        self, vision_results: List[Dict[str, Any]], file_base: str
    ) -> str:
        """将 Vision 分析结果格式化为 Composer 可用的图片描述文本"""
        if not vision_results:
            return "（无截图可用）\n"

        lines = []
        included = [r for r in vision_results if r.get("should_include", True)]

        if not included:
            return "（所有截图均被过滤，无可用图片）\n"

        lines.append(f"共 {len(included)} 张可用截图：\n")

        for r in included:
            img_id = r.get("image_id", "unknown.jpg")
            rel_path = f"{file_base}/{img_id}"
            desc = r.get("content_description", "（无描述）")
            caption = r.get("suggested_caption", "")
            img_type = r.get("type", "other")

            lines.append(f"- `图片: {rel_path}`")
            lines.append(f"  类型: {img_type}")
            lines.append(f"  内容: {desc}")
            if caption:
                lines.append(f"  建议图注: {caption}")
            if r.get("key_terms"):
                lines.append(f"  关键术语: {', '.join(r['key_terms'])}")
            lines.append("")

        return "\n".join(lines)

    def save_note(
        self,
        file_base: str,
        content: str,
        note_dir: str,
    ) -> str:
        """保存笔记为 Markdown 文件，带精美 header/footer

        Args:
            file_base: 文件名（不含扩展名）
            content: 笔记内容
            note_dir: 笔记输出目录

        Returns:
            保存的 MD 文件路径
        """
        os.makedirs(note_dir, exist_ok=True)

        # 提取首行作为副标题
        lines = [l for l in content.strip().split("\n") if l.strip()]
        first_line = lines[0].replace("#", "").strip() if lines else "核心笔记"
        rest = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        # 组装完整 Markdown
        header = (
            f'# <center><font face="仿宋" color=orange>{file_base}</font></center>\n'
            f'# <font face="微软雅黑"><center>{first_line}</center>\n\n'
        )
        footer = "\n\n</font>\n"
        full = header + rest + footer

        md_path = os.path.join(note_dir, f"{file_base}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full)

        logger.info(f"[Composer] 笔记已保存: {md_path} ({len(full)} 字符)")
        return md_path
