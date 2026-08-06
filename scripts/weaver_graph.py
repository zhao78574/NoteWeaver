#!/usr/bin/env python3
"""知识图谱可视化 — 复制 D3.js 星系版模板

用法:
    python scripts/weaver_graph.py                    # 默认打开浏览器
    python scripts/weaver_graph.py --output mymap.html
"""

import sys
import webbrowser
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="NoteWeaver 知识图谱可视化")
    parser.add_argument("--output", "-o", default="",
                        help="输出 HTML 路径（默认自动打开浏览器）")
    parser.add_argument("--category", "-c", default="",
                        help="（已由 HTML 图例筛选替代，此参数仅用于兼容）")
    args = parser.parse_args()

    # 定位 KG 文件（用于提示）
    project_root = Path(__file__).resolve().parent.parent
    kg_path = project_root / "data" / "memory_db" / "knowledge_graph.json"
    if not kg_path.exists():
        print(f"[X] 知识图谱文件不存在: {kg_path}")
        print("   提示: 先处理一些视频生成笔记，知识图谱会自动构建")
        sys.exit(1)

    # 复制 D3.js 星系版模板
    template = project_root / "note_weaver" / "templates" / "knowledge_graph.html"
    if not template.exists():
        print(f"[X] 图谱模板文件不存在: {template}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = project_root / "data" / "memory_db" / "knowledge_graph.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")

    abs_path = str(output_path.resolve())
    print(f"[OK] 知识图谱已生成: {abs_path}")
    print(f"    HTML 支持动态加载 knowledge_graph.json，打开后自动渲染")

    if not args.output:
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
