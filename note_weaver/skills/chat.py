"""对话问答 Skill — 基于笔记库的知识问答（带缓存）"""

import os, json
from openai import OpenAI
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger


_SYSTEM = """你是一个学习助理，基于用户的笔记库回答问题。

规则：
1. 优先引用笔记中的具体内容，标注来源笔记文件名
2. 如果笔记中有明确答案，直接引用
3. 如果涉及多个笔记，对比分析
4. 如果笔记中没有答案，诚实告知并建议查阅方向
5. 保持有人味儿的语气，不要AI八股味
6. 回答简洁，问什么答什么"""

# 笔记缓存 {rel_path: {"mtime": float, "content": str}}
_note_cache: dict[str, dict] = {}
_cache_loaded = False


def _load_notes() -> str:
    """加载笔记库（带缓存，只重读有变动的文件）"""
    global _cache_loaded
    note_dir = config.note_dir
    if not os.path.isdir(note_dir):
        return ""

    changed = 0
    context_parts = []

    for root, dirs, files in os.walk(note_dir):
        for f in sorted(files):
            if not f.endswith(".md"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, note_dir)
            try:
                mtime = os.path.getmtime(path)
                # 缓存命中且未变动 → 直接用
                if rel in _note_cache and _note_cache[rel]["mtime"] == mtime:
                    context_parts.append(f"--- {rel} ---\n{_note_cache[rel]['content']}")
                    continue

                # 读文件并缓存
                with open(path, encoding="utf-8") as fp:
                    content = fp.read()
                content = content.replace('<center>', '').replace('</center>', '')
                content = content.replace('<font face="仿宋" color=orange>', '')
                content = content.replace('<font face="微软雅黑">', '')
                content = content.replace('</font>', '')
                content = content[:3000]  # 截断
                _note_cache[rel] = {"mtime": mtime, "content": content}
                context_parts.append(f"--- {rel} ---\n{content}")
                changed += 1
            except Exception:
                continue

    _cache_loaded = True
    if changed:
        logger.info(f"[Chat] 缓存更新: {changed} 篇新/变更笔记")

    if not context_parts:
        return ""

    context = "\n\n".join(context_parts)
    if len(context) > 30000:
        context = context[:30000] + "\n...(truncated)"
    return context


def chat(question: str) -> str:
    """基于笔记库回答用户问题"""
    context = _load_notes()

    if not context:
        return "笔记库为空，请先处理一些视频。"

    # 调用 DeepSeek
    try:
        config.setup_proxy()
        client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        )
        resp = client.chat.completions.create(
            model=config.model_fast,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"## 笔记库\n{context}\n\n## 用户问题\n{question}"},
            ],
            temperature=0.5,
        )
        return resp.choices[0].message.content or "（无响应）"
    except Exception as e:
        return f"问答失败: {e}"
