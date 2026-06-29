"""记忆管理 Agent — 三层记忆：短期会话 + 长期知识 + 情节交互"""

import os
import json
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from .base import BaseAgent
from note_weaver.utils.logger import logger
from note_weaver.utils.config import config


# ════════════════════════════════════════════════════════════════
# 知识图谱 Schema 常量
# ════════════════════════════════════════════════════════════════

NODE_TYPES = ["concept", "document", "chunk", "media_frame"]

EDGE_TYPES = [
    "supports",        # A 支撑/支持 B
    "contradicts",     # A 与 B 相悖
    "derives_from",    # A 衍生自 B
    "similar_to",      # A 相似于 B
    "prerequisite",    # A 是 B 的先修知识
    "part_of",         # A 是 B 的组成部分
    "example_of",      # A 是 B 的一个实例
]


class MemoryAgent(BaseAgent):
    """管理用户画像、知识图谱和交互历史"""

    def execute(self, **kwargs) -> Any:
        """Memory Agent 的通用入口（概念提取等）"""
        return None

    def __init__(self):
        super().__init__()  # 轻量任务用 fast 模型
        self.memory_dir = config.memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)

        # 文件路径
        self._profile_path = os.path.join(self.memory_dir, "user_profile.json")
        self._kg_path = os.path.join(self.memory_dir, "knowledge_graph.json")
        self._episodes_path = os.path.join(self.memory_dir, "episodes.jsonl")

        # 内存缓存
        self.profile: Dict = {}
        self.knowledge_graph: Dict = {"concepts": [], "relations": []}
        self.episodes: List[Dict] = []

        self._loaded = False

    def _ensure_loaded(self):
        """懒加载记忆数据"""
        if self._loaded:
            return
        self.profile = self._load_json(self._profile_path, default={
            "user_id": "default",
            "knowledge_levels": {},
            "note_preferences": {
                "style": "handwritten_style",
                "detail_level": 0.8,
                "use_emojis": True,
                "table_over_paragraph": 0.5,
            },
            "weak_points": [],
            "strong_points": [],
            "term_preferences": {},
            "learning_history": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        })
        self.knowledge_graph = self._load_json(self._kg_path, default={
            "concepts": [],
            "relations": [],
        })
        self._loaded = True
        logger.info(
            f"[Memory] 已加载: 画像({len(self.profile.get('weak_points', []))}弱项), "
            f"知识图谱({len(self.knowledge_graph['concepts'])}概念)"
        )

    # ---- 公共接口 ----

    def get_context(
        self,
        domain: str = "",
        difficulty: str = "",
    ) -> str:
        """获取当前用户的上下文，供 Composer 使用"""
        self._ensure_loaded()
        parts = []

        # 知识水平
        level = self.profile.get("knowledge_levels", {}).get(domain, "")
        if level:
            parts.append(f"学习者在「{domain}」领域的水平为 {level}")

        # 薄弱点
        weak = self.profile.get("weak_points", [])
        if weak:
            parts.append(f"需要特别注意的知识点（学习者历史上理解有困难）: {', '.join(weak[:5])}")

        # 已有知识
        strong = self.profile.get("strong_points", [])
        if strong:
            parts.append(f"学习者已掌握的概念: {', '.join(strong[:8])}")

        # 笔记偏好
        prefs = self.profile.get("note_preferences", {})
        style = prefs.get("style", "")
        if style:
            parts.append(f"偏好的笔记风格: {style}")

        detail = prefs.get("detail_level", 0.8)
        parts.append(f"内容详细度: {detail:.0%}")

        ctx = "\n".join(parts) if parts else ""
        if ctx:
            logger.info(f"[Memory] 提供上下文: {len(ctx)} 字符")

        return ctx

    def update_after_note(
        self,
        file_base: str,
        note_content: str,
        classification: Dict[str, Any],
        qa_report: Dict[str, Any],
    ):
        """笔记生成后更新记忆"""
        self._ensure_loaded()

        # 1. 更新知识图谱（提取新概念）
        concepts = self._extract_concepts(note_content, classification)
        self._merge_concepts(concepts, file_base)

        # 2. 更新用户画像（学习历史）
        domain = classification.get("domain", "")
        if domain:
            history = self.profile.setdefault("learning_history", [])
            history.append({
                "note": file_base,
                "domain": domain,
                "difficulty": classification.get("difficulty", "intermediate"),
                "timestamp": datetime.now().isoformat(),
                "qa_score": qa_report.get("total", 7.0),
            })
            # 只保留最近 100 条
            self.profile["learning_history"] = history[-100:]

        self.profile["last_updated"] = datetime.now().isoformat()

        # 持久化
        self._save_json(self._profile_path, self.profile)
        self._save_json(self._kg_path, self.knowledge_graph)

        logger.info(
            f"[Memory] 记忆已更新: +{len(concepts)}概念, "
            f"学习历史={len(self.profile.get('learning_history', []))}"
        )

    def record_interaction(
        self,
        interaction_type: str,
        user_action: str,
        agent_response: str,
        learning_signal: float = 0.0,
    ):
        """记录一次用户交互（情节记忆）"""
        episode = {
            "timestamp": datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "user_action": user_action,
            "agent_response": agent_response[:500],
            "learning_signal": learning_signal,
        }

        with open(self._episodes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")

        self.episodes.append(episode)

        # 只保留最近 500 条在内存
        if len(self.episodes) > 500:
            self.episodes = self.episodes[-500:]

        logger.debug(f"[Memory] 情节已记录: {interaction_type}")

    def search_concepts(self, query: str, top_k: int = 5) -> List[Dict]:
        """在知识图谱中搜索相关概念（简单关键词匹配）"""
        self._ensure_loaded()
        results = []
        concepts = self.knowledge_graph.get("concepts", [])
        query_lower = query.lower()

        for c in concepts:
            name = c.get("name", "") + c.get("name_en", "")
            if query_lower in name.lower():
                results.append(c)

        return results[:top_k]

    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        self._ensure_loaded()
        history = self.profile.get("learning_history", [])
        domains = {}
        for h in history:
            d = h.get("domain", "unknown")
            domains[d] = domains.get(d, 0) + 1

        return {
            "total_notes": len(history),
            "domains": domains,
            "weak_points": self.profile.get("weak_points", []),
            "strong_points": self.profile.get("strong_points", []),
            "kg_concepts": len(self.knowledge_graph.get("concepts", [])),
            "kg_relations": len(self.knowledge_graph.get("relations", [])),
        }

    # ---- 内部方法 ----

    def _extract_concepts(
        self,
        note_content: str,
        classification: Dict,
    ) -> List[Dict]:
        """从笔记中提取核心概念（含关系类型标注）"""
        from note_weaver.utils.prompts import MEMORY_EXTRACT_CONCEPTS_SYSTEM

        domain = classification.get("domain", "")
        relation_types = ", ".join(EDGE_TYPES)
        prompt = (
            f"领域: {domain}\n\n"
            f"## 关系类型（请对每个概念间关系标注类型）\n"
            f"可用关系类型: {relation_types}\n\n"
            f"- supports（支撑/支持）\n"
            f"- contradicts（矛盾/相悖）\n"
            f"- derives_from（衍生自）\n"
            f"- prerequisite（先修/前置知识）\n"
            f"- part_of（组成/包含）\n"
            f"- example_of（举例/实例）\n"
            f"- similar_to（相似）\n\n"
            f"## 笔记内容\n{note_content[:5000]}"
        )

        try:
            raw = self.chat(
                prompt,
                system_instruction=MEMORY_EXTRACT_CONCEPTS_SYSTEM,
                expect_json=True,
            )
            result = json.loads(raw)
            return result.get("concepts", [])
        except (json.JSONDecodeError, RuntimeError) as e:
            logger.warning(f"[Memory] 概念提取失败: {e}")
            return []

    def _merge_concepts(self, new_concepts: List[Dict], source_note: str):
        """合并新概念到知识图谱（去重 + 类型化关联 + 兼容旧数据）

        支持两种关系格式：
        - 新格式: relations=[{"target": "...", "type": "supports"}, ...]
        - 旧格式: related_to=["概念1", "概念2"] → 自动升级为 "related" 类型
        """
        existing = {c["name"]: c for c in self.knowledge_graph["concepts"]}

        for nc in new_concepts:
            name = nc.get("name", "")
            if not name:
                continue

            if name in existing:
                # 已有概念：追加来源
                sources = existing[name].setdefault("source_notes", [])
                if source_note not in sources:
                    sources.append(source_note)
                # 合并旧关系（type=None 的兼容）
                existing_rels = existing[name].get("relations", [])
                new_rels = nc.get("relations", [])
                existing[name]["relations"] = self._merge_relations(
                    existing_rels, new_rels)
            else:
                nc["source_notes"] = [source_note]
                nc["first_seen"] = datetime.now().isoformat()
                nc.setdefault("relations", [])
                nc.setdefault("related_to", [])
                self.knowledge_graph["concepts"].append(nc)

        # 从 relations 字段构建图边（新格式优先）
        all_new_relations = []  # [(from, to, type), ...]
        for nc in new_concepts:
            name = nc.get("name", "")
            # 新格式 relations
            for rel in nc.get("relations", []):
                target = rel.get("target", "")
                rel_type = rel.get("type", "related")
                if target:
                    all_new_relations.append((name, target, rel_type))
            # 旧格式 related_to（向后兼容）
            for related in nc.get("related_to", []):
                # 避免与 relations 重复
                if not any(r["target"] == related for r in nc.get("relations", [])):
                    all_new_relations.append((name, related, "related"))

        for from_name, to_name, rel_type in all_new_relations:
            # 去重检测
            is_dup = any(
                r["from"] == from_name and r["to"] == to_name
                and (r.get("type", "related") == rel_type or rel_type == "related")
                for r in self.knowledge_graph["relations"]
            )
            if not is_dup:
                self.knowledge_graph["relations"].append({
                    "from": from_name,
                    "to": to_name,
                    "type": rel_type,
                    "source": source_note,
                })

    @staticmethod
    def _merge_relations(existing_relations: List[Dict],
                         new_relations: List[Dict]) -> List[Dict]:
        """合并关系，兼容旧数据（type=None / 无 type 的无类型关系）

        旧格式的 related_to 不会在这里出现（已在 _merge_concepts 中处理）。
        这里处理 relations 列表中的类型化关系。
        """
        seen = set()
        merged = list(existing_relations)  # 保留现有

        for nr in new_relations:
            target = nr.get("target", "")
            rel_type = nr.get("type", "related")
            # 兼容旧数据：无 type 的标记为 related
            if "type" not in nr:
                nr["type"] = "related"
                rel_type = "related"
            key = (target, rel_type)
            if key not in seen:
                seen.add(key)
                # 更新 instead of appending（如果已存在同 target 但不同 type）
                replaced = False
                for i, er in enumerate(merged):
                    if er.get("target") == target:
                        merged[i] = nr
                        replaced = True
                        break
                if not replaced:
                    merged.append(nr)
        return merged

    # ---- JSON 文件操作 ----

    @staticmethod
    def _load_json(path: str, default: Any = None) -> Any:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return default

    @staticmethod
    def _save_json(path: str, data: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
