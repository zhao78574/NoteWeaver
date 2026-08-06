"""知识搜索 Skill — BM25 + 可选语义双通道混合检索"""

from note_weaver.utils.logger import logger
from note_weaver.utils.embeddings import HybridRetrieval


def search(query: str, top_k: int = 10) -> str:
    """混合检索笔记库（BM25 + 可选 Embedding → RRF 融合）

    Args:
        query: 搜索词
        top_k: 最大返回条数

    Returns:
        格式化的搜索结果文本
    """
    try:
        hybrid = HybridRetrieval(use_semantic=False)  # 纯BM25（语义模型需从HF下载，国内网络不稳定）
        hybrid.build()
        results = hybrid.search(query, top_k=top_k)
    except Exception as e:
        logger.warning(f"[Search] 检索失败: {e}")
        results = []

    if not results:
        return f"未找到与「{query}」相关的内容"

    lines = [f"## 搜索「{query}」— {len(results)} 条结果\n"]
    for i, r in enumerate(results):
        score_str = f" (score={r['score']})" if r.get("score") else ""
        lines.append(f"### {i+1}. {r['title']}{score_str}")
        lines.append(f"> {r.get('snippet', '')[:200]}")
        if r.get("source"):
            lines.append(f"来源: {r['source']}")
        lines.append("")

    return "\n".join(lines)
