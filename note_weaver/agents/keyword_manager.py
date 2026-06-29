"""Keyword Manager — 自动热词表扩展 + 衰减机制

职责：
  - 从处理后视频的转录文本中提取高频领域术语，自动扩展热词表
  - 对长期未出现的术语进行衰减，保持词表新鲜度
  - 与 PolicyEngine 集成，为 Corrector 提供动态热词

工作流程：
  1. 每次视频处理完成后，从转录文本提取候选术语
  2. 更新词频和最后出现时间
  3. 定期衰减（每次加载时检查 >30 天未见的 term）
  4. PolicyEngine 合并静态 + 动态词表 → Corrector
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from note_weaver.utils.logger import logger
from note_weaver.utils.config import config


# ── 中文停用词（用于过滤无意义的 2-3 字串） ───────────────────

_STOP_CHARS: Set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一", "个", "上", "也", "很", "到", "说", "要",
    "去", "你", "会", "着", "没", "看", "好", "自", "己", "这",
    "那", "它", "们", "来", "为", "与", "对", "把", "被", "让",
    "从", "向", "往", "以", "但", "而", "所", "如", "因", "或",
    "及", "等", "之", "其", "中", "外", "前", "后", "吗", "呢",
    "吧", "啊", "嗯", "哦", "啦", "嘛", "这个", "那个", "什么",
    "怎么", "因为", "所以", "但是", "可以", "如果", "就是", "没有",
    "一个", "不是", "我们", "他们", "它们", "大家", "可能", "应该",
    "需要", "能够", "时候", "非常", "比较", "然后", "以后", "现在",
    "这样", "那样", "而且", "或者", "还是", "只是", "但是", "虽然",
    "因为", "所以", "已经", "可以", "进行", "通过", "使用", "利用",
    "采用", "具有", "提出", "发现", "得到", "成为", "作为",
}

# ── 领域专有名词后缀（用于识别候选term的边界） ────────────────

_TECH_SUFFIXES: Set[str] = {
    "工艺", "设备", "材料", "结构", "器件", "电路", "系统",
    "方法", "技术", "原理", "步骤", "流程", "参数", "特性",
    "电压", "电流", "浓度", "温度", "压力", "速率", "厚度",
}


class KeywordManager:
    """自动热词表管理 — 扩展 + 衰减 + 合并"""

    def __init__(self, store_path: Optional[str] = None):
        self._store_path = Path(store_path or (
            Path(config.memory_dir) / "domain_keywords.json"
        ))
        self._store: Dict[str, dict] = {}  # {domain: {terms: {term: info}}}
        self._load()

    # ── 持久化 ────────────────────────────────────────────────

    def _load(self):
        """从磁盘加载热词表"""
        if self._store_path.exists():
            try:
                with open(self._store_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._store = raw
                logger.debug(
                    f"[KeywordMgr] 加载热词表: {sum(len(v.get('terms', {})) for v in raw.values())} 词"
                )
            except Exception as e:
                logger.debug(f"[KeywordMgr] 加载失败（重置）: {e}")
                self._store = {}
        else:
            self._store = {}

    def _save(self):
        """保存热词表到磁盘"""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"[KeywordMgr] 保存失败（非致命）: {e}")

    # ── 热词提取 ──────────────────────────────────────────────

    @staticmethod
    def extract_candidates(text: str, top_n: int = 30) -> List[Tuple[str, int]]:
        """从文本中提取高频候选术语

        使用 2-4 字 n-gram 频率统计，过滤停用词。

        Args:
            text: 转录文本
            top_n: 提取前 N 个候选

        Returns:
            [(term, frequency), ...] 按频率降序
        """
        if not text:
            return []

        # 中文部分提取
        chinese_only = re.sub(r'[^一-鿿]', '', text)
        if len(chinese_only) < 10:
            return []

        # 统计 2/3/4-gram 频率
        freq: Dict[str, int] = defaultdict(int)
        for n in (2, 3, 4):
            for i in range(len(chinese_only) - n + 1):
                gram = chinese_only[i:i + n]
                freq[gram] += 1

        # 过滤停用词 + 低频词
        filtered = [
            (gram, count)
            for gram, count in freq.items()
            if count >= 2                          # 至少出现 2 次
            and gram not in _STOP_CHARS             # 不是停用词
            and not all(c in _STOP_CHARS for c in gram)  # 不是全停用字
        ]

        # 按频率降序排列
        filtered.sort(key=lambda x: -x[1])
        return filtered[:top_n]

    def update_from_transcript(self, text: str, domain: str):
        """从转录文本更新领域热词表

        Args:
            text: 转录全文
            domain: 领域标签
        """
        if not text or not domain:
            return

        candidates = self.extract_candidates(text, top_n=30)
        if not candidates:
            return

        # 初始化领域存储
        if domain not in self._store:
            self._store[domain] = {"version": 1, "terms": {}}

        domain_store = self._store[domain]["terms"]
        now = time.time()
        now_date = time.strftime("%Y-%m-%d", time.localtime(now))

        new_count = 0
        for term, count in candidates:
            if term in domain_store:
                # 更新已有词
                domain_store[term]["count"] += count
                domain_store[term]["last_seen"] = now_date
                domain_store[term]["score"] = min(
                    1.0, domain_store[term]["score"] + 0.05
                )
            else:
                # 新增词
                domain_store[term] = {
                    "count": count,
                    "first_seen": now_date,
                    "last_seen": now_date,
                    "score": 0.3,  # 初始分数（需要多次出现才能稳固）
                }
                new_count += 1

        total = len(domain_store)
        self._save()
        logger.info(
            f"[KeywordMgr] 领域={domain}: +{new_count} 新词, "
            f"共 {total} 词活跃"
        )

    # ── 衰减 ──────────────────────────────────────────────────

    def decay(self, half_life_days: float = 30.0):
        """对长期未出现的术语进行衰减

        每次加载/保存时自动调用。衰减逻辑：
          - 上次出现距今 > half_life_days 的 term，score 减半
          - score < 0.05 的 term 从词表中移除

        Args:
            half_life_days: 半衰期（天）
        """
        now = time.time()
        changed = False

        for domain, store in list(self._store.items()):
            terms = store.get("terms", {})
            expired = []

            for term, info in terms.items():
                last_seen_str = info.get("last_seen", "")
                if not last_seen_str:
                    continue

                try:
                    last_seen = time.mktime(time.strptime(last_seen_str, "%Y-%m-%d"))
                except (ValueError, OverflowError):
                    continue

                days_since = (now - last_seen) / 86400.0
                if days_since > half_life_days:
                    # 指数衰减
                    halvings = days_since / half_life_days
                    new_score = info["score"] * (0.5 ** halvings)
                    info["score"] = max(0.0, new_score)
                    changed = True

                    if info["score"] < 0.05:
                        expired.append(term)

            # 移除过期词
            for term in expired:
                del terms[term]
                changed = True

            if expired:
                logger.debug(
                    f"[KeywordMgr] 领域={domain}: 衰减移除 {len(expired)} 词"
                )

        if changed:
            self._save()

    # ── 查询 ──────────────────────────────────────────────────

    def get_dynamic_keywords(
        self, domain: str, top_k: int = 15, min_score: float = 0.1,
    ) -> List[str]:
        """获取指定领域的活跃动态热词

        Args:
            domain: 领域标签
            top_k: 返回前 K 个
            min_score: 最低分数阈值

        Returns:
            关键词列表（按 score 降序）
        """
        self.decay()

        domain_store = self._store.get(domain, {})
        terms = domain_store.get("terms", {})

        # 按 score 排序
        sorted_terms = sorted(
            [(t, info.get("score", 0)) for t, info in terms.items()],
            key=lambda x: -x[1],
        )

        return [t for t, s in sorted_terms if s >= min_score][:top_k]

    def merge_with_static(
        self, domain: str, static_keywords: List[str],
        top_k: int = 30,
    ) -> List[str]:
        """将动态热词与静态策略词表合并

        Args:
            domain: 领域标签
            static_keywords: 静态词表（来自 PolicyEngine）
            top_k: 合并后取前 K 个

        Returns:
            合并后的关键词列表
        """
        dynamic = self.get_dynamic_keywords(domain, top_k=top_k)

        # 合并去重（静态优先）
        seen: Set[str] = set()
        merged = []

        for kw in static_keywords:
            if kw not in seen:
                merged.append(kw)
                seen.add(kw)

        for kw in dynamic:
            if kw not in seen:
                merged.append(kw)
                seen.add(kw)

        return merged[:top_k]

    # ── 管理接口 ─────────────────────────────────────────────

    def add_term(self, domain: str, term: str, score: float = 0.5):
        """手动添加一个热词"""
        if domain not in self._store:
            self._store[domain] = {"version": 1, "terms": {}}

        now_date = time.strftime("%Y-%m-%d")
        self._store[domain]["terms"][term] = {
            "count": 1,
            "first_seen": now_date,
            "last_seen": now_date,
            "score": score,
        }
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        """获取热词表统计"""
        stats = {}
        for domain, store in self._store.items():
            terms = store.get("terms", {})
            active = sum(1 for t in terms.values() if t.get("score", 0) >= 0.1)
            stats[domain] = {
                "total": len(terms),
                "active": active,
            }
        return stats
