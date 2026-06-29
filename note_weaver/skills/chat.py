"""对话问答 Skill — RAG: BM25 检索 + LLM 自由对话

只要设了 DEEPSEEK_API_KEY，用户问什么都能答。
笔记库有内容时自动引用笔记，没有时自由对话。
"""

import os
from openai import OpenAI
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger
from note_weaver.utils.embeddings import HybridRetrieval


_SYSTEM = """# 角色
你是一个 AI 学习助理，名叫 NoteWeaver。

# 能力
1. 如果下面提供了笔记库内容，优先引用笔记回答问题，标注来源
2. 如果笔记库没有相关内容，用自己的知识回答
3. 如果用户问的是纯聊天/日常话题，自然对话即可
4. 回答简洁明了，可画 ASCII 图说明概念
5. 保持有人味儿的语气，不要AI八股文"""


def chat(question: str) -> str:
    """RAG 问答：检索 → 生成

    不再加载全部笔记，而是从 BM25 + 可选 Embedding 的混合检索中
    召回最相关的 top-k 段落，拼成上下文后交给 LLM。

    当笔记库为空或检索无结果时，退化为纯 LLM 对话。
    """
    # ── 1. 检索相关段落 ──
    context = ""
    has_notes = False
    try:
        hybrid = HybridRetrieval(use_semantic=False)
        hybrid.build()
        results = hybrid.search(question, top_k=5)
        if results:
            has_notes = True
            context_parts = []
            for r in results:
                src = r.get("source", "")
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                score = r.get("score", 0)
                context_parts.append(
                    f"--- {title} ({src}) [relevance={score:.2f}] ---\n{snippet}"
                )
            context = "\n\n".join(context_parts)
            logger.info(
                f"[Chat] RAG: 检索到 {len(results)} 段, "
                f"最高分 {results[0].get('score', 0):.2f}"
            )
    except Exception as e:
        logger.warning(f"[Chat] 检索失败: {e}")

    # ── 2. 拼上下文 ──
    user_content = question
    if has_notes and context:
        user_content = (
            f"## 笔记库（检索结果）\n{context}\n\n"
            f"## 用户问题\n{question}"
        )

    # ── 3. LLM 生成 ──
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
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content or "（无响应）"
    except Exception as e:
        return f"回答失败: {e}"
