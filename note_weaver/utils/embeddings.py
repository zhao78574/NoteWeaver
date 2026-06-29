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


# ════════════════════════════════════════════════════════════════
# BM25 索引 — 词袋检索的黄金标准
# ════════════════════════════════════════════════════════════════


class BM25Index:
    """BM25 全文检索索引（基于 rank_bm25）

    替代 EmbeddingIndex 的 n-gram TF-IDF —— BM25 的词频饱和+长度归一化
    对中文和英文检索效果都更好，尤其适合 100+ 篇笔记的场景。

    用法:
        idx = BM25Index()
        idx.build()
        results = idx.search("LOCOS 隔离")
    """

    def __init__(self):
        from pathlib import Path as _Path
        self._index_path = _Path(config.memory_dir) / "bm25_index.pkl"
        self._map_path = _Path(config.memory_dir) / "bm25_map.json"
        self.model = None
        self._items: List[Dict[str, Any]] = []
        self._loaded = False

    # ── 公开接口 ────────────────────────────────────────────────

    def build(self, force: bool = False) -> int:
        """扫描笔记库 + 知识图谱，构建 BM25 索引"""
        if not force and self._index_path.exists() and self._map_path.exists():
            return self._load()

        items = self._collect_items()
        if not items:
            return 0

        texts = [it["text"] for it in items]
        logger.info(f"[BM25] 构建索引: {len(texts)} 条...")

        from rank_bm25 import BM25Okapi
        tokenized = [self._tokenize(t) for t in texts]
        self.model = BM25Okapi(tokenized)
        self._items = items

        import pickle as _p
        import json as _j
        with open(self._index_path, "wb") as f:
            _p.dump(self.model, f)
        with open(self._map_path, "w", encoding="utf-8") as f:
            _j.dump(items, f, ensure_ascii=False, indent=2)

        logger.info(f"[BM25] 索引完成: {len(items)} 条")
        return len(items)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 检索

        Returns:
            [{"source": ..., "title": ..., "snippet": ..., "score": ...}, ...]
        """
        if not self._loaded:
            if not self._load():
                return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores = self.model.get_scores(q_tokens)

        import numpy as _np
        top_indices = _np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] < 0.01:
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
    def _tokenize(text: str) -> List[str]:
        """分词：中文按字拆 + 英文/数字按词"""
        import re as _re
        tokens = []
        tokens.extend(_re.findall(r'[一-鿿]', text))
        for w in _re.findall(r'[a-zA-Z0-9_]+', text):
            w = w.lower()
            if len(w) >= 2:
                tokens.append(w)
        return tokens

    def _collect_items(self) -> List[Dict[str, Any]]:
        return EmbeddingIndex()._collect_items()

    def _load(self) -> bool:
        if not self._index_path.exists() or not self._map_path.exists():
            return False
        try:
            import pickle as _p
            import json as _j
            with open(self._index_path, "rb") as f:
                self.model = _p.load(f)
            with open(self._map_path, "r", encoding="utf-8") as f:
                self._items = _j.load(f)
            self._loaded = True
            logger.info(f"[BM25] 加载索引: {len(self._items)} 条")
            return True
        except Exception as e:
            logger.warning(f"[BM25] 加载失败: {e}")
            return False


# ════════════════════════════════════════════════════════════════
# Hybrid Retrieval — 双通道混合检索（BM25 + 可选语义通道）
# ════════════════════════════════════════════════════════════════


class SemanticIndex:
    """轻量语义向量索引，零外部强制依赖

    基于 sentence-transformers 的可选语义通道。
    当 sentence-transformers 未安装时优雅降级（search 返回空列表）。
    """

    def __init__(self):
        self.model = None
        self._docs: List[Dict[str, Any]] = []
        self._embeddings: Optional["np.ndarray"] = None
        self._np = None

    def _lazy_np(self):
        if self._np is None:
            import numpy as np
            self._np = np

    def _load_model(self) -> bool:
        """尝试加载 sentence-transformers 模型，失败则返回 False"""
        if self.model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            # 用超时线程防止 HF 下载卡死（国内网络不稳定）
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    SentenceTransformer,
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    device='cpu'
                )
                self.model = future.result(timeout=30)
            logger.info("[SemanticIndex] sentence-transformers 模型加载成功")
            return True
        except ImportError:
            logger.warning("[SemanticIndex] sentence-transformers 未安装，语义通道不可用")
            return False
        except concurrent.futures.TimeoutError:
            logger.warning("[SemanticIndex] 模型下载超时（30s），语义通道不可用")
            return False
        except Exception as e:
            logger.warning(f"[SemanticIndex] 模型加载失败: {e}")
            return False

    def build(self, docs: List[Dict[str, str]], force: bool = False) -> bool:
        """构建语义索引

        Args:
            docs: [{"id": str, "text": str}, ...]
            force: 是否强制重建

        Returns:
            True=构建成功, False=不可用（sentence-transformers 未安装）
        """
        if not self._load_model():
            return False

        self._lazy_np()
        self._docs = docs
        texts = [d["text"] for d in docs]

        logger.info(f"[SemanticIndex] 编码 {len(texts)} 条文本...")
        self._embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"[SemanticIndex] 索引完成: {len(docs)} 条")
        return True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """语义搜索

        Args:
            query: 搜索文本
            top_k: 返回条数

        Returns:
            [{"id": str, "score": float}, ...]
        """
        if self._embeddings is None:
            return []
        if not self._load_model():
            return []

        self._lazy_np()
        query_emb = self.model.encode([query])[0]
        scores = self._np.dot(self._embeddings, query_emb)
        top_indices = self._np.argsort(scores)[-top_k:][::-1]

        return [
            {"id": self._docs[i]["id"], "score": float(scores[i])}
            for i in top_indices
        ]

    @property
    def is_available(self) -> bool:
        """语义通道是否可用（模型已加载）"""
        return self.model is not None and self._embeddings is not None


class HybridRetrieval:
    """双通道混合检索：lexical(BM25) + semantic(可选) → RRF 融合排序

    用法:
        hybrid = HybridRetrieval(use_semantic=True)
        hybrid.build()                          # 构建双通道索引
        results = hybrid.search("阈值电压")      # 混合检索
    """

    def __init__(self, use_semantic: bool = False):
        """
        Args:
            use_semantic: 是否启用语义通道（需安装 sentence-transformers ~500MB）
        """
        self.lexical = BM25Index()               # BM25 词袋通道
        self.semantic = SemanticIndex() if use_semantic else None
        self._use_semantic = use_semantic

    def build(self, force: bool = False) -> int:
        """构建双通道索引

        Returns:
            lexical 索引的条目数
        """
        count = self.lexical.build(force=force)

        if self._use_semantic and self.semantic is not None:
            # 从 lexical 收集的 items 构建语义索引
            docs = [
                {"id": str(i), "text": it.get("text", "")}
                for i, it in enumerate(self.lexical._items)
            ]
            available = self.semantic.build(docs)
            if not available:
                logger.info("[HybridRetrieval] 语义通道不可用，降级为纯 lexical")
                self._use_semantic = False

        logger.info(f"[HybridRetrieval] 索引完成: {count} 条 (semantic={self._use_semantic})")
        return count

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """混合检索

        Args:
            query: 搜索词
            top_k: 最终返回条数

        Returns:
            [{"source": ..., "title": ..., "snippet": ..., "score": ...}, ...]
        """
        lexical_results = self.lexical.search(query, top_k=top_k * 2)

        if self._use_semantic and self.semantic and self.semantic.is_available:
            semantic_results = self.semantic.search(query, top_k=top_k * 2)
            fused = self._reciprocal_rank_fusion(lexical_results, semantic_results, top_k)
        else:
            # 纯 lexical
            fused = lexical_results[:top_k]

        return fused

    def _reciprocal_rank_fusion(
        self,
        lexical: List[Dict[str, Any]],
        semantic: List[Dict[str, Any]],
        top_k: int = 5,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """RRF 融合排序

        Args:
            lexical: TF-IDF 搜索结果（带 source/title/snippet/score）
            semantic: 语义搜索结果（带 id/score）
            top_k: 返回条数
            k: RRF 常数（默认 60）

        Returns:
            融合排序后的结果列表
        """
        self._lazy_np()
        scores = {}

        # lexical 通道
        for rank, doc in enumerate(lexical):
            doc_id = doc.get("source", "") + "::" + doc.get("title", "")
            scores[doc_id] = {
                "rrf": 1.0 / (k + rank + 1),
                "source": doc,
                "lexical_score": doc.get("score", 0),
                "semantic_score": 0,
            }

        # semantic 通道
        if not semantic:
            ranked = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
            return [r["source"] for r in ranked[:top_k]]

        for rank, doc in enumerate(semantic):
            doc_id = doc.get("id", str(rank))
            try:
                idx = int(doc_id)
                real_doc = self.lexical._items[idx] if idx < len(self.lexical._items) else None
            except (ValueError, IndexError):
                real_doc = None

            if real_doc:
                real_id = real_doc.get("source", "") + "::" + real_doc.get("title", "")
                if real_id in scores:
                    scores[real_id]["rrf"] += 1.0 / (k + rank + 1)
                    scores[real_id]["semantic_score"] = doc.get("score", 0)
                else:
                    scores[real_id] = {
                        "rrf": 1.0 / (k + rank + 1),
                        "source": real_doc,
                        "lexical_score": 0,
                        "semantic_score": doc.get("score", 0),
                    }

        ranked = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
        return [r["source"] for r in ranked[:top_k]]

    def _lazy_np(self):
        self.lexical._lazy_np()
