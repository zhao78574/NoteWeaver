"""向量嵌入 — 基于字符 n-gram TF-IDF 的语义搜索（纯 numpy，无需外部 API/模型）

用法:
    from note_weaver.utils.embeddings import EmbeddingIndex
    idx = EmbeddingIndex()
    idx.build()                      # 从笔记库重建索引
    results = idx.search("阈值电压")  # 语义搜索
"""

import os
import json
import re
import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional
from note_weaver.utils.config import config
from note_weaver.utils.logger import logger


class EmbeddingIndex:
    """基于字符 n-gram TF-IDF 的本地语义索引

    原理：
    - 对中文文本切分为字符 2-gram（如 "阈值电压" → "阈值", "值电", "电压"）
    - 计算 TF-IDF 向量
    - 余弦相似度搜索
    - 无需 Tokenizer、无需 API、无需 GPU
    """

    def __init__(self, ngram_range: tuple = (2, 3), max_features: int = 50000):
        self._index_path = Path(config.memory_dir) / "tfidf_index.npz"
        self._map_path = Path(config.memory_dir) / "tfidf_map.json"
        self._vocab_path = Path(config.memory_dir) / "tfidf_vocab.json"

        self._ngram_range = ngram_range
        self._max_features = max_features

        # 运行时缓存
        self._tfidf_matrix: Optional["np.ndarray"] = None
        self._items: List[Dict[str, Any]] = []
        self._vocab: Dict[str, int] = {}  # n-gram → idx
        self._idf: List[float] = []
        self._loaded = False

        # 懒惰导入 numpy（只在必要时）
        self._np = None

    def _lazy_np(self):
        if self._np is None:
            import numpy as np
            self._np = np

    # ── 公开接口 ────────────────────────────────────────────────

    def build(self, force: bool = False) -> int:
        """扫描笔记库 + 知识图谱，构建 TF-IDF 索引

        Args:
            force: True=强制重建；False=仅当索引不存在时重建

        Returns:
            索引条目数
        """
        if not force and self._index_path.exists() and self._map_path.exists():
            return self._load()

        items = self._collect_items()
        if not items:
            logger.info("[Embedding] 没有内容需要索引")
            return 0

        texts = [it["text"] for it in items]
        logger.info(f"[Embedding] 构建 TF-IDF 索引: {len(texts)} 条...")

        # 1. 构建词库（n-gram 计数）
        doc_ngrams = []
        all_ngram_counts = Counter()

        for text in texts:
            ngrams = self._extract_ngrams(text)
            doc_ngrams.append(ngrams)
            all_ngram_counts.update(ngrams.keys())

        # 2. 选取 top N 作为词库
        top_ngrams = [ng for ng, _ in all_ngram_counts.most_common(self._max_features)]
        self._vocab = {ng: i for i, ng in enumerate(top_ngrams)}
        self._np = None  # will be set in _lazy_np
        self._lazy_np()

        # 3. 计算 TF-IDF
        n_docs = len(doc_ngrams)
        vocab_size = len(self._vocab)
        idf = self._np.zeros(vocab_size)

        for ngrams in doc_ngrams:
            for ng in ngrams:
                idx = self._vocab.get(ng)
                if idx is not None:
                    idf[idx] += 1

        # IDF = log((N + 1) / (df + 1)) + 1
        idf = self._np.log((n_docs + 1) / (idf + 1)) + 1
        self._idf = idf.tolist()

        # 4. 计算 TF-IDF 矩阵（L2 归一化）
        matrix = self._np.zeros((n_docs, vocab_size), dtype=self._np.float32)

        for doc_idx, ngrams in enumerate(doc_ngrams):
            total = sum(ngrams.values())
            if total == 0:
                continue
            for ng, count in ngrams.items():
                col = self._vocab.get(ng)
                if col is not None:
                    tf = count / total
                    matrix[doc_idx, col] = tf * idf[col]

        # L2 归一化
        norms = self._np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1
        matrix = matrix / norms.reshape(-1, 1)

        self._tfidf_matrix = matrix
        self._items = items

        # 保存
        self._np.savez(str(self._index_path), matrix=matrix)
        with open(self._map_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        with open(self._vocab_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self._vocab, "idf": self._idf}, f, ensure_ascii=False)

        logger.info(f"[Embedding] 索引完成: {len(items)} 条 (词库: {vocab_size})")
        return len(items)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """语义搜索，返回按相似度排序的结果列表

        Args:
            query: 搜索词
            top_k: 返回条数

        Returns:
            [{"source": ..., "title": ..., "snippet": ..., "score": ...}, ...]
        """
        if not self._loaded:
            if not self._load():
                return []

        self._lazy_np()

        # 计算查询向量
        q_ngrams = self._extract_ngrams(query)
        q_vec = self._np.zeros(len(self._vocab), dtype=self._np.float32)
        total = sum(q_ngrams.values())
        if total == 0:
            return []

        for ng, count in q_ngrams.items():
            col = self._vocab.get(ng)
            if col is not None:
                q_vec[col] = (count / total) * self._idf[col]

        # L2 归一化
        q_norm = self._np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # 余弦相似度（点积，因为均已 L2 归一化）
        scores = self._tfidf_matrix @ q_vec

        # 取 top_k
        top_indices = self._np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] < 0.05:
                continue
            item = self._items[idx]
            results.append({
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", "")[:200],
                "score": round(float(scores[idx]), 4),
            })
        return results

    # ── 内部 ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_ngrams(text: str) -> Counter:
        """从中文文本中提取字符 n-gram

        对中文按字切分生成 n-gram（2-gram 和 3-gram），
        对英文/数字按空格切分成词。
        """
        result = Counter()

        # 提取中文字符序列
        chinese_chars = re.findall(r'[一-鿿]+', text)
        for seq in chinese_chars:
            # 2-gram
            for i in range(len(seq) - 1):
                result[seq[i:i+2]] += 1
            # 3-gram
            for i in range(len(seq) - 2):
                result[seq[i:i+3]] += 1

        # 提取英文/数字词
        words = re.findall(r'[a-zA-Z0-9_]+', text)
        for w in words:
            w = w.lower()
            if len(w) >= 2:
                result[w] += 1

        return result

    def _collect_items(self) -> List[Dict[str, Any]]:
        """收集笔记和概念，生成可搜索的文本块"""
        items = []

        # 1. 从知识图谱收集概念
        kg_path = Path(config.memory_dir) / "knowledge_graph.json"
        if kg_path.exists():
            try:
                with open(kg_path, "r", encoding="utf-8") as f:
                    kg = json.load(f)
                for c in kg.get("concepts", []):
                    text = f"{c.get('name', '')} {c.get('name_en', '')} {c.get('definition', '')}"
                    items.append({
                        "source": f"KG:{c.get('category', 'other')}",
                        "title": f"{c.get('name', '')} ({c.get('name_en', '')})",
                        "snippet": c.get("definition", ""),
                        "text": text,
                    })
            except Exception as e:
                logger.warning(f"[Embedding] 读取知识图谱失败: {e}")

        # 2. 从笔记收集
        note_dir = Path(config.note_dir)
        if note_dir.exists():
            for root, dirs, files in os.walk(str(note_dir)):
                for fname in sorted(files):
                    if not fname.endswith(".md"):
                        continue
                    path = Path(root) / fname
                    rel = path.relative_to(note_dir)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                    except Exception:
                        continue

                    content = re.sub(r'<[^>]+>', '', content)
                    content = re.sub(r'!\[\]\([^)]+\)', '', content)
                    content = content.strip()
                    if not content:
                        continue

                    sections = re.split(r'(?=^## )', content, flags=re.MULTILINE)
                    for sec in sections:
                        sec = sec.strip()
                        if len(sec) < 20:
                            continue
                        title_line = sec.split("\n")[0][:80]
                        items.append({
                            "source": str(rel),
                            "title": fname.replace(".md", "") + " — " + title_line,
                            "snippet": sec[:200],
                            "text": sec[:500],
                        })

        return items

    def _load(self) -> bool:
        """从磁盘加载索引到缓存"""
        if not self._index_path.exists() or not self._map_path.exists():
            return False
        try:
            self._lazy_np()
            npz = self._np.load(str(self._index_path))
            self._tfidf_matrix = npz["matrix"]

            with open(self._map_path, "r", encoding="utf-8") as f:
                self._items = json.load(f)

            with open(self._vocab_path, "r", encoding="utf-8") as f:
                vdata = json.load(f)
            self._vocab = vdata["vocab"]
            self._idf = vdata["idf"]

            self._loaded = True
            logger.info(f"[Embedding] 加载索引: {len(self._items)} 条")
            return True
        except Exception as e:
            logger.warning(f"[Embedding] 加载失败: {e}")
            return False
