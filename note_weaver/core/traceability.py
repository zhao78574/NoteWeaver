"""可追溯性 — 维护 frame/timestamp → note paragraph 的映射表

提供的功能：
- 从笔记中解析 <!-- frame: ... --> 注释，构建追溯索引
- 记下每个图片来源对应到哪个笔记文件
- 支持按 frame 名 / 时间戳查找笔记段落

用法:
    from note_weaver.core.traceability import TraceabilityIndex

    index = TraceabilityIndex("data/Note")
    index.build("data/Note/半导体物理/Band_Theory.md")
    trace = index.lookup("半导体物理/lecture_p3_0_hash.png")
    # → {"note": "data/Note/半导体物理/Band_Theory.md", "paragraph": "能带弯曲..."}
"""

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from note_weaver.utils.logger import logger
from note_weaver.utils.config import config


class TraceabilityIndex:
    """维护 frame/timestamp → note paragraph 的映射表

    索引以 JSON 格式存储在笔记目录下的 _trace_index.json。
    """

    def __init__(self, note_dir: str = None):
        self.note_dir = Path(note_dir or config.note_dir)
        self._index: Dict[str, Any] = {
            "entries": [],
            "notes": {},
            "updated_at": "",
        }

    # ── 公开接口 ────────────────────────────────────────────────────

    def build(self, note_path: str = None) -> int:
        """扫描笔记文件，构建/更新追溯索引

        Args:
            note_path: 指定笔记路径（None = 扫描全部）

        Returns:
            索引条目数
        """
        from datetime import datetime

        if note_path:
            paths = [Path(note_path)]
        else:
            paths = list(self.note_dir.rglob("*.md"))

        entries = []
        notes_index = {}

        for md_path in paths:
            try:
                content = md_path.read_text(encoding="utf-8")
            except Exception:
                continue

            # 解析 <!-- frame: ... --> 注释
            traces = self._parse_trace_comments(content)
            if not traces:
                continue

            rel_path = str(md_path.relative_to(self.note_dir))
            notes_index[rel_path] = {
                "path": str(md_path),
                "traces": len(traces),
            }

            for trace in traces:
                trace["note"] = rel_path
                entries.append(trace)

        self._index = {
            "entries": entries,
            "notes": notes_index,
            "total_traces": len(entries),
            "total_notes": len(notes_index),
            "updated_at": datetime.now().isoformat(),
        }

        logger.info(
            f"[Traceability] 索引更新: {len(entries)} 条追溯 "
            f"(来自 {len(notes_index)} 篇笔记)"
        )
        return len(entries)

    def lookup(self, frame_name: str) -> Optional[Dict[str, Any]]:
        """按 frame 名查找对应的笔记段落

        Args:
            frame_name: 图片文件名或相对路径

        Returns:
            {"note": "笔记相对路径", "paragraph": "段落文本", ...} 或 None
        """
        if not self._index["entries"]:
            self.build()

        for entry in self._index["entries"]:
            if frame_name in entry.get("source", ""):
                return {
                    "note": entry.get("note", ""),
                    "source": entry.get("source", ""),
                    "context": entry.get("context", ""),
                    "paragraph_index": entry.get("paragraph_index", -1),
                }
        return None

    def get_note_traces(self, note_rel_path: str) -> List[Dict[str, Any]]:
        """获取某篇笔记的所有追溯条目

        Args:
            note_rel_path: 笔记相对路径（如 "半导体物理/Band_Theory.md"）

        Returns:
            追溯条目列表
        """
        return [
            e for e in self._index["entries"]
            if e.get("note") == note_rel_path
        ]

    def save(self):
        """将索引持久化到磁盘"""
        index_path = self.note_dir / "_trace_index.json"
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
            logger.info(f"[Traceability] 索引已保存: {index_path}")
        except Exception as e:
            logger.warning(f"[Traceability] 保存失败: {e}")

    def load(self) -> bool:
        """从磁盘加载索引

        Returns:
            True=加载成功
        """
        index_path = self.note_dir / "_trace_index.json"
        if not index_path.exists():
            return False
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
            logger.info(
                f"[Traceability] 已加载索引: "
                f"{self._index.get('total_traces', 0)} 条追溯"
            )
            return True
        except Exception as e:
            logger.warning(f"[Traceability] 加载失败: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """获取追溯统计"""
        if not self._index["entries"]:
            self.build()
        return {
            "total_traces": len(self._index["entries"]),
            "total_notes": len(self._index["notes"]),
            "notes": dict(self._index["notes"]),
        }

    # ── 内部方法 ────────────────────────────────────────────────────

    @staticmethod
    def _parse_trace_comments(markdown: str) -> List[Dict[str, Any]]:
        """从 Markdown 中解析 <!-- frame: ... --> 注释

        Args:
            markdown: 笔记全文

        Returns:
            [{"source": "file_base/img_id @ timestamp", "context": "...", "paragraph_index": N}, ...]
        """
        traces = []

        # 按段落分割
        paragraphs = re.split(r'\n\n+', markdown)

        for para_idx, para in enumerate(paragraphs):
            # 查找 <!-- frame: ... -->
            matches = re.findall(r'<!--\s*frame:\s*(.*?)\s*-->', para)
            for match in matches:
                # 提取紧随注释后的图片引用
                img_match = re.search(r'!\[\]\(([^)]+)\)', para)
                traces.append({
                    "source": match.strip(),
                    "context": para[:200].strip(),
                    "paragraph_index": para_idx,
                    "image_path": img_match.group(1) if img_match else "",
                })

        return traces
