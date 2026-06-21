"""知识搜索 Skill — 向量语义搜索 + 关键词搜索双通道"""

import os
import json
import re
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger
from note_weaver.utils.embeddings import EmbeddingIndex


def search(query: str, top_k: int = 10) -> str:
    """搜索笔记库（向量语义搜索 + 关键词搜索双通道，结果融合排序）

    Args:
        query: 搜索词
        top_k: 最大返回条数

    Returns:
        格式化的搜索结果文本
    """
    results = []

    # ── 通道 1：向量语义搜索 ──
    try:
        idx = EmbeddingIndex()
        idx.build()  # 索引不存在时自动构建
        vec_results = idx.search(query, top_k=top_k)
        for r in vec_results:
            r["channel"] = "向量"
            results.append(r)
    except Exception as e:
        logger.warning(f"[Search] 向量搜索失败: {e}")

    # ── 通道 2：关键词搜索（作为补充/兜底） ──
    kw_results = _keyword_search(query, top_k=top_k)
    for r in kw_results:
        r["channel"] = "关键词"
        results.append(r)

    # ── 融合去重（按 source + title 判重） ──
    seen = set()
    merged = []
    for r in results:
        key = (r.get("source", ""), r.get("title", ""))
        if key not in seen:
            seen.add(key)
            merged.append(r)

    # 按分数排序（向量结果有 score，关键词结果 score=0）
    merged.sort(key=lambda r: r.get("score", 0), reverse=True)

    if not merged:
        return f"未找到与「{query}」相关的内容"

    # ── 格式化输出 ──
    lines = [f"## 搜索「{query}」— {len(merged)} 条结果\n"]
    for i, r in enumerate(merged[:top_k]):
        score_str = f" (score={r['score']})" if r.get("score") else ""
        channel_tag = f" [{r.get('channel', '')}]" if r.get("channel") else ""
        lines.append(f"### {i+1}. {r['title']}{score_str}{channel_tag}")
        lines.append(f"> {r.get('snippet', '')[:200]}")
        if r.get("source"):
            lines.append(f"来源: {r['source']}")
        lines.append("")

    return "\n".join(lines)


def _keyword_search(query: str, top_k: int = 10) -> list:
    """关键词搜索（原 search.py 逻辑）"""
    results = []
    q = query.lower()

    # 1. 知识图谱匹配
    kg_path = os.path.join(config.memory_dir, "knowledge_graph.json")
    if os.path.exists(kg_path):
        with open(kg_path, encoding="utf-8") as f:
            kg = json.load(f)
        for c in kg.get("concepts", []):
            name = (c.get("name", "") + " " + c.get("name_en", "")).lower()
            definition = c.get("definition", "")
            if q in name or q in definition:
                results.append({
                    "source": f"KG:{c.get('category', '')}",
                    "title": f"{c.get('name', '')} ({c.get('name_en', '')})",
                    "snippet": c.get("definition", ""),
                    "score": 0,
                })

    # 2. 笔记全文搜索
    note_dir = config.note_dir
    if os.path.isdir(note_dir):
        for root, dirs, files in os.walk(note_dir):
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path, encoding="utf-8") as fp:
                        content = fp.read()
                except Exception:
                    continue
                if q in content.lower():
                    idx = content.lower().find(q)
                    start = max(0, idx - 40)
                    end = min(len(content), idx + len(query) + 80)
                    snippet = content[start:end].replace("\n", " ").strip()
                    results.append({
                        "source": os.path.relpath(path, note_dir),
                        "title": fname.replace(".md", ""),
                        "snippet": f"...{snippet}...",
                        "score": 0,
                    })

    return results
