"""知识搜索 Skill — 搜索笔记库和知识图谱"""

import os, json, re
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger


def search(query: str, top_k: int = 10) -> str:
    """在笔记库中搜索

    策略: 1) 知识图谱精确匹配 2) 笔记全文关键词匹配 3) 排序去重
    """
    results = []

    # ---- 1. 知识图谱搜索 ----
    kg_path = os.path.join(config.memory_dir, "knowledge_graph.json")
    if os.path.exists(kg_path):
        with open(kg_path, encoding="utf-8") as f:
            kg = json.load(f)
        for c in kg.get("concepts", []):
            name = c.get("name", "") + c.get("name_en", "")
            definition = c.get("definition", "")
            if query.lower() in name.lower() or query in definition:
                results.append({
                    "source": f"[KG] {c.get('category', '')}",
                    "title": f"{c.get('name', '')} ({c.get('name_en', '')})",
                    "snippet": c.get("definition", ""),
                    "notes": c.get("source_notes", []),
                })

    # ---- 2. 笔记全文搜索 ----
    note_dir = config.note_dir
    if os.path.isdir(note_dir):
        for root, dirs, files in os.walk(note_dir):
            for f in files:
                if not f.endswith(".md"):
                    continue
                path = os.path.join(root, f)
                try:
                    with open(path, encoding="utf-8") as fp:
                        content = fp.read()
                except Exception:
                    continue

                if query.lower() in content.lower():
                    # 提取匹配上下文
                    idx = content.lower().find(query.lower())
                    start = max(0, idx - 40)
                    end = min(len(content), idx + len(query) + 80)
                    snippet = content[start:end].replace("\n", " ").strip()
                    results.append({
                        "source": os.path.relpath(path, note_dir),
                        "title": f.replace(".md", ""),
                        "snippet": f"...{snippet}...",
                        "notes": [],
                    })

    # ---- 3. 排序: KG匹配优先 ----
    kg_results = [r for r in results if r["source"].startswith("[KG]")]
    note_results = [r for r in results if not r["source"].startswith("[KG]")]
    ranked = kg_results + note_results

    if not ranked:
        return f"未找到与「{query}」相关的内容"

    # 格式化输出
    lines = [f"## 搜索「{query}」— {len(ranked)} 条结果\n"]
    for i, r in enumerate(ranked[:top_k]):
        lines.append(f"### {i+1}. {r['title']}")
        lines.append(f"> {r['snippet'][:200]}")
        if r["notes"]:
            lines.append(f"来源笔记: {', '.join(r['notes'][:3])}")
        lines.append("")

    return "\n".join(lines)
