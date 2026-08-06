"""对话问答 Skill — RAG: BM25 检索 + LLM 自由对话（支持流式输出）

只要设了 DEEPSEEK_API_KEY，用户问什么都能答。
笔记库有内容时自动引用笔记，没有时自由对话。
"""

import re
from typing import Callable, Optional
from openai import OpenAI
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger
from note_weaver.utils.embeddings import HybridRetrieval


_SYSTEM = """# 角色
你是一个 AI 学习助理，名叫 NoteWeaver。

# 能力
1. 如果下面提供了笔记库内容，**必须**引用笔记回答问题
2. 如果笔记库没有相关内容，用自己的知识回答
3. 如果用户问的是纯聊天/日常话题，自然对话即可
4. 回答简洁明了，可画 ASCII 图说明概念
5. 保持有人味儿的语气，不要AI八股文

# 强制规则：来源标注
当引用笔记库内容时，每段引用末尾必须标注来源笔记名，格式为：
[来源：笔记名.md]
如果一条回答引用多篇笔记，分别标注：
[来源：笔记A.md]
[来源：笔记B.md]"""

_source_details_cache: dict = {}

# ── 检索实例缓存（避免每次 new + build） ──
_hybrid_retrieval: Optional[HybridRetrieval] = None
_hybrid_retrieval_built: bool = False


def get_source_details(source_name: str) -> dict:
    """根据来源笔记名查找检索详情（章节名 + 预览 + 行号）

    source_name 可能是 "笔记名.md" 或 "笔记名 — ## 章节名"
    """
    # 精确匹配
    if source_name in _source_details_cache:
        return _source_details_cache[source_name]
    # 模糊匹配：按文件名匹配缓存 key
    for key, val in _source_details_cache.items():
        if key.startswith(source_name) or source_name.startswith(key):
            return val
        # 去掉 .md 再比
        src_clean = source_name.replace(".md", "").strip()
        key_clean = key.split(" — ")[0].strip()
        if src_clean == key_clean:
            return val
    return {}


def _retrieve_context(question: str) -> tuple:
    global _source_details_cache, _hybrid_retrieval, _hybrid_retrieval_built
    _source_details_cache = {}
    context = ""
    has_notes = False
    sources = []
    try:
        # 复用全局检索实例，避免每次 new + build 的开销
        if _hybrid_retrieval is None:
            _hybrid_retrieval = HybridRetrieval(use_semantic=False)
        if not _hybrid_retrieval_built:
            _hybrid_retrieval.build()
            _hybrid_retrieval_built = True
        results = _hybrid_retrieval.search(question, top_k=5)
        if results:
            has_notes = True
            context_parts = []
            seen_sources = set()
            for r in results:
                src = r.get("source", "")        # "笔记名.md"
                title = r.get("title", "")         # "笔记名 — ## 章节名"
                snippet = r.get("snippet", "")
                score = r.get("score", 0)
                context_parts.append(
                    f"--- {title} ({src}) [relevance={score:.2f}] ---\n{snippet}"
                )
                note_name = title or src
                if note_name and note_name not in seen_sources:
                    seen_sources.add(note_name)
                    sources.append(note_name)
                    section = ""
                    if " — " in title:
                        section = title.split(" — ", 1)[-1]
                    # 同时用文件名和标题作为缓存 key
                    clean_src = src.replace(".md", "").strip()
                    details = {
                        "section": section,
                        "snippet": snippet[:200],
                        "source": src,
                        "line_start": r.get("line_start", 0),
                    }
                    _source_details_cache[src] = details           # "笔记名.md"
                    _source_details_cache[clean_src] = details      # "笔记名"
                    _source_details_cache[note_name] = details      # "笔记名 — ## 章节名"
            context = "\n\n".join(context_parts)
            logger.info(
                f"[Chat] RAG: 检索到 {len(results)} 段, "
                f"最高分 {results[0].get('score', 0):.2f}"
            )
    except Exception as e:
        logger.warning(f"[Chat] 检索失败: {e}")
    return context, has_notes, sources


def _build_messages(question: str, context: str, has_notes: bool) -> list:
    user_content = question
    if has_notes and context:
        user_content = (
            f"## 笔记库（检索结果）\n{context}\n\n"
            f"## 用户问题\n{question}"
        )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _post_process(answer: str, has_notes: bool, sources: list) -> str:
    if has_notes and sources:
        has_citation = bool(re.search(r'\[来源：', answer))
        if not has_citation:
            src_lines = "\n".join(f"- {s}" for s in sources)
            answer += f"\n\n---\n参考笔记：\n{src_lines}"
    return answer


def chat(question: str) -> str:
    context, has_notes, sources = _retrieve_context(question)
    messages = _build_messages(question, context, has_notes)
    try:
        config.setup_proxy()
        client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        )
        resp = client.chat.completions.create(
            model=config.model_fast,
            messages=messages,
            temperature=0.7,
        )
        answer = resp.choices[0].message.content or "（无响应）"
    except Exception as e:
        return f"回答失败: {e}"
    return _post_process(answer, has_notes, sources)


def stream_chat(
    question: str,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    context, has_notes, sources = _retrieve_context(question)
    messages = _build_messages(question, context, has_notes)
    try:
        config.setup_proxy()
        client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        )
        resp = client.chat.completions.create(
            model=config.model_fast,
            messages=messages,
            temperature=0.7,
            stream=True,
        )
        collected: list[str] = []
        for chunk in resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                collected.append(delta.content)
                if on_token:
                    on_token(delta.content)
        answer = "".join(collected)
        if not answer:
            answer = "（无响应）"
    except Exception as e:
        return f"回答失败: {e}"
    return _post_process(answer, has_notes, sources)


def invalidate_retrieval_cache():
    """强制重建检索索引（新笔记处理后调用）"""
    global _hybrid_retrieval, _hybrid_retrieval_built
    _hybrid_retrieval = None
    _hybrid_retrieval_built = False
