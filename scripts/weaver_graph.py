#!/usr/bin/env python3
"""知识图谱可视化 — 生成交互式力导向图 HTML

用法:
    python scripts/weaver_graph.py                    # 默认打开浏览器
    python scripts/weaver_graph.py --output mymap.html
    python scripts/weaver_graph.py --category process  # 只看某分类
"""

import sys
import os
import json
import webbrowser
import argparse
from pathlib import Path

# 颜色映射（按分类）
CATEGORY_COLORS = {
    "process":  {"color": "#4CAF50", "shape": "box"},       # 绿色 → 工艺
    "device":   {"color": "#2196F3", "shape": "diamond"},    # 蓝色 → 器件
    "physics":  {"color": "#FF9800", "shape": "ellipse"},    # 橙色 → 物理
    "material": {"color": "#9C27B0", "shape": "hexagon"},    # 紫色 → 材料
    "tool":     {"color": "#F44336", "shape": "triangle"},   # 红色 → 工具
    "other":    {"color": "#607D8B", "shape": "dot"},         # 灰色 → 其他
}

# 难度尺寸映射
DIFFICULTY_SIZE = {
    "beginner": 20,
    "intermediate": 26,
    "advanced": 32,
}

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NoteWeaver 知识图谱</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; height: 100vh; }
#header { position: fixed; top: 0; left: 0; right: 0; z-index: 10; padding: 12px 24px; background: rgba(26,26,46,0.92); backdrop-filter: blur(8px); display: flex; align-items: center; gap: 16px; border-bottom: 1px solid rgba(255,255,255,0.08); }
#header h1 { font-size: 18px; font-weight: 600; }
#header .stats { font-size: 13px; color: #888; }
#legend { margin-left: auto; display: flex; gap: 12px; align-items: center; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 12px; color: #aaa; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
#graph { width: 100vw; height: 100vh; }
#tooltip { position: fixed; display: none; z-index: 20; background: rgba(30,30,60,0.95); border: 1px solid rgba(255,255,255,0.15); border-radius: 10px; padding: 16px 20px; font-size: 13px; line-height: 1.6; box-shadow: 0 8px 32px rgba(0,0,0,0.5); backdrop-filter: blur(12px); pointer-events: none; }
#tooltip .tt-name { font-size: 17px; font-weight: 600; margin-bottom: 2px; }
#tooltip .tt-en { font-size: 12px; color: #888; margin-bottom: 4px; }
#tooltip .tt-def { font-size: 12px; color: #bbb; margin-bottom: 4px; }
#tooltip .tt-meta { font-size: 11px; color: #666; }
#search-box { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); z-index: 15; background: rgba(30,30,60,0.92); border: 1px solid rgba(255,255,255,0.12); border-radius: 24px; padding: 8px 20px; width: 360px; max-width: 90vw; color: #eee; font-size: 14px; outline: none; backdrop-filter: blur(8px); }
#search-box:focus { border-color: #4CAF50; }
.node-label { font-size: 11px; }
</style>
</head>
<body>

<div id="header">
  <h1>🧠 知识图谱</h1>
  <span class="stats" id="stats"></span>
  <div id="legend"></div>
</div>

<div id="graph"></div>
<input id="search-box" type="text" placeholder="🔍 搜索概念…" autocomplete="off">

<div id="tooltip"></div>

<script src="https://cdn.bootcdn.net/ajax/libs/vis-network/9.1.6/dist/vis-network.min.js"></script>
<link rel="stylesheet" href="https://cdn.bootcdn.net/ajax/libs/vis-network/9.1.6/dist/dist/vis-network.min.css">
<script>
const DATA = __DATA_PLACEHOLDER__;

const nodes = new vis.DataSet(DATA.nodes);
const edges = new vis.DataSet(DATA.edges);

document.getElementById('stats').textContent =
  nodes.length + ' 概念 · ' + edges.length + ' 关系';

const colors = DATA.nodeColors || {};
const legend = document.getElementById('legend');
Object.entries(colors).forEach(function(_a) {
  var cat = _a[0], hex = _a[1];
  var item = document.createElement('span');
  item.className = 'legend-item';
  item.innerHTML = '<span class="legend-dot" style="background:' + hex + '"></span>' + cat;
  legend.appendChild(item);
});

var options = {
  physics: {
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {
      gravitationalConstant: -40,
      centralGravity: 0.005,
      springLength: 180,
      springConstant: 0.08,
      damping: 0.4,
    },
    stabilization: { iterations: 200 },
  },
  nodes: {
    font: { color: '#ddd', size: 11, face: 'Microsoft YaHei' },
    borderWidth: 1.5,
    shadow: { enabled: true, size: 4 },
  },
  edges: {
    color: { color: 'rgba(255,255,255,0.12)', highlight: 'rgba(255,255,255,0.3)' },
    width: 1,
    smooth: { type: 'continuous' },
    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
  },
  interaction: {
    hover: true,
    tooltipDelay: 0,
    navigationButtons: true,
    keyboard: true,
  },
};

var container = document.getElementById('graph');
var network = new vis.Network(container, { nodes: nodes, edges: edges }, options);

var tooltip = document.getElementById('tooltip');

network.on('hoverNode', function(params) {
  var node = nodes.get(params.node);
  if (!node || !node._full) return;
  var d = node._full;
  tooltip.innerHTML =
    '<div class="tt-name">' + (d.name || '') + '</div>' +
    (d.name_en ? '<div class="tt-en">' + d.name_en + '</div>' : '') +
    (d.definition ? '<div class="tt-def">' + d.definition + '</div>' : '') +
    '<div class="tt-meta">' +
    (d.category || '') +
    (d.difficulty ? ' &middot; ' + d.difficulty : '') +
    (d.source_notes && d.source_notes.length ? ' &middot; ' + d.source_notes.join(', ') : '') +
    '</div>' +
    (d._relationCount ? '<div class="tt-meta">' + d._relationCount + ' 条关联</div>' : '');
  tooltip.style.display = 'block';
  tooltip.style.width = '';
  tooltip.style.maxWidth = '520px';
  tooltip.style.left = '50%';
  tooltip.style.right = 'auto';
  tooltip.style.top = 'auto';
  tooltip.style.bottom = '80px';
  tooltip.style.transform = 'translateX(-50%)';
});

network.on('blurNode', function() {
  tooltip.style.display = 'none';
});

var searchBox = document.getElementById('search-box');
var lastSearch = '';

searchBox.addEventListener('input', function() {
  var q = this.value.trim().toLowerCase();
  if (q === lastSearch) return;
  lastSearch = q;

  if (!q) {
    nodes.forEach(function(n) { nodes.update({ id: n.id, opacity: 1 }); });
    return;
  }

  var matched = [];
  DATA.rawConcepts.forEach(function(c, i) {
    var searchText = ((c.name || '') + (c.name_en || '') + (c.definition || '')).toLowerCase();
    if (searchText.indexOf(q) !== -1) matched.push(i);
  });

  var matchSet = {};
  matched.forEach(function(id) { matchSet[id] = true; });
  nodes.forEach(function(n) {
    nodes.update({ id: n.id, opacity: matchSet[n.id] ? 1 : 0.12 });
  });
});
</script>
</body>
</html>"""


def build_graph_data(kg_path: str) -> dict:
    """从 knowledge_graph.json 构建 vis-network 数据"""
    with open(kg_path, "r", encoding="utf-8") as f:
        kg = json.load(f)

    concepts = kg.get("concepts", [])
    relations = kg.get("relations", [])

    # 构建索引
    node_map = {}
    vis_nodes = []
    node_colors = {}

    for i, c in enumerate(concepts):
        cat = c.get("category", "other")
        style = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
        diff = c.get("difficulty", "beginner")
        size = DIFFICULTY_SIZE.get(diff, 20)

        node_id = i
        label = c.get("name", "???")
        if len(label) > 8:
            label = label[:8] + "…"

        vis_node = {
            "id": node_id,
            "label": label,
            "title": c.get("name", ""),
            "color": {"background": style["color"], "border": style["color"]},
            "shape": style["shape"],
            "size": size,
            "_full": c,
            "_relationCount": 0,
        }
        vis_nodes.append(vis_node)
        node_map[c.get("name", "")] = node_id
        node_colors[cat] = style["color"]

    # 统计每个节点的关联数
    for r in relations:
        src_name = r.get("source", "")
        tgt_name = r.get("target", "")
        if src_name in node_map:
            vis_nodes[node_map[src_name]]["_full"]["_relationCount"] += 1
        if tgt_name in node_map:
            vis_nodes[node_map[tgt_name]]["_full"]["_relationCount"] += 1

    # 从 related_to 字段构建边
    vis_edges = set()  # 去重
    for i, c in enumerate(concepts):
        for rel_name in c.get("related_to", []):
            if rel_name in node_map:
                j = node_map[rel_name]
                edge_key = (i, j)
                if edge_key not in vis_edges:
                    vis_edges.add(edge_key)

    # 从 relations 字段构建边
    for r in relations:
        src_name = r.get("source", "")
        tgt_name = r.get("target", "")
        if src_name in node_map and tgt_name in node_map:
            edge_key = (node_map[src_name], node_map[tgt_name])
            if edge_key not in vis_edges:
                vis_edges.add(edge_key)

    vis_edges_list = [{"from": f, "to": t} for f, t in sorted(vis_edges)]

    return {
        "nodes": vis_nodes,
        "edges": vis_edges_list,
        "nodeColors": dict(sorted(node_colors.items())),
        "rawConcepts": concepts,
    }


def filter_by_category(data: dict, category: str) -> dict:
    """只保留指定分类的概念"""
    keep_ids = {i for i, c in enumerate(data["rawConcepts"])
                if c.get("category") == category}

    filtered_nodes = [n for n in data["nodes"] if n["id"] in keep_ids]
    node_ids = {n["id"] for n in filtered_nodes}
    filtered_edges = [e for e in data["edges"]
                      if e["from"] in node_ids and e["to"] in node_ids]

    data["nodes"] = filtered_nodes
    data["edges"] = filtered_edges
    return data


def main():
    parser = argparse.ArgumentParser(description="NoteWeaver 知识图谱可视化")
    parser.add_argument("--output", "-o", default="",
                        help="输出 HTML 路径（默认自动打开浏览器）")
    parser.add_argument("--category", "-c", default="",
                        help="只显示指定分类 (process/device/physics/material/tool)")
    args = parser.parse_args()

    # 定位 KG 文件
    project_root = Path(__file__).resolve().parent.parent
    kg_path = project_root / "data" / "memory_db" / "knowledge_graph.json"
    if not kg_path.exists():
        print(f"[X] 知识图谱文件不存在: {kg_path}")
        print("   提示: 先处理一些视频生成笔记，知识图谱会自动构建")
        sys.exit(1)

    # 构建数据
    data = build_graph_data(str(kg_path))
    if args.category:
        data = filter_by_category(data, args.category)
        print(f"[OK] 筛选分类: {args.category} ({len(data['nodes'])} 概念)")

    # 生成 HTML
    import json as _json
    data_json = _json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = project_root / "data" / "memory_db" / "knowledge_graph.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = str(output_path.resolve())
    print(f"[OK] 知识图谱已生成: {abs_path}")
    print(f"    概念: {len(data['nodes'])} | 关系: {len(data['edges'])}")

    if not args.output:
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
