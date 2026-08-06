"""向量嵌入 — 基于字符 n-gram TF-IDF 的语义搜索（纯 numpy，无需外部 API/模型）

用法:
    from note_weaver.utils.embeddings import EmbeddingIndex
    idx = EmbeddingIndex()
    idx.build()                      # 从笔记库重建索引
    results = idx.search("阈值电压")  # 语义搜索
"""

from __future__ import annotations

import os
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from note_weaver.utils.config import config

if TYPE_CHECKING:
    import numpy as np
from note_weaver.utils.logger import logger


class EmbeddingIndex:
    """基于字符 n-gram TF-IDF 的本地语义索引"""

    def __init__(self, ngram_range: tuple = (2, 3), max_features: int = 50000):
        self._index_path = Path(config.memory_dir) / "tfidf_index.npz"
        self._map_path = Path(config.memory_dir) / "tfidf_map.json"
        self._vocab_path = Path(config.memory_dir) / "tfidf_vocab.json"

        self._ngram_range = ngram_range
        self._max_features = max_features

        self._tfidf_matrix: Optional["np.ndarray"] = None
        self._items: List[Dict[str, Any]] = []
        self._vocab: Dict[str, int] = {}
        self._idf: List[float] = []
        self._loaded = False

        self._np = None

    def _lazy_np(self):
        if self._np is None:
            import numpy as np
            self._np = np

    # ── 公开接口 ────────────────────────────────────────────────

    def build(self, force: bool = False) -> int:
        if not force and self._index_path.exists() and self._map_path.exists():
            return self._load()

        items = self._collect_items()
        if not items:
            logger.info("[Embedding] 没有内容需要索引")
            return 0

        texts = [it["text"] for it in items]
        logger.info(f"[Embedding] 构建 TF-IDF 索引: {len(texts)} 条...")

        doc_ngrams = []
        all_ngram_counts = Counter()

        for text in texts:
            ngrams = self._extract_ngrams(text)
            doc_ngrams.append(ngrams)
            all_ngram_counts.update(ngrams.keys())

        top_ngrams = [ng for ng, _ in all_ngram_counts.most_common(self._max_features)]
        self._vocab = {ng: i for i, ng in enumerate(top_ngrams)}
        self._np = None
        self._lazy_np()

        n_docs = len(doc_ngrams)
        vocab_size = len(self._vocab)
        idf = self._np.zeros(vocab_size)

        for ngrams in doc_ngrams:
            for ng in ngrams:
                idx = self._vocab.get(ng)
                if idx is not None:
                    idf[idx] += 1

        idf = self._np.log((n_docs + 1) / (idf + 1)) + 1
        self._idf = idf.tolist()

        matrix = self._np.zeros((n_docs, vocab_size), dtype=self._np.float32)
        for doc_idx, ngrams in enumerate(doc_ngrams):
            total = sum(ngrams.values())
            if total == 0:
                continue
            for ng, cnt in ngrams.items():
                feat_idx = self._vocab.get(ng)
                if feat_idx is not None:
                    tf = cnt / total
                    matrix[doc_idx, feat_idx] = tf * idf[feat_idx]

        self._tfidf_matrix = matrix
        self._items = items

        self._save()
        logger.info(f"[Embedding] 索引完成: {n_docs} 条 × {vocab_size} 维")
        return n_docs

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._loaded and not self._load():
            return []

        self._lazy_np()
        q_ngrams = self._extract_ngrams(query)
        q_vec = self._np.zeros(len(self._vocab), dtype=self._np.float32)
        total = sum(q_ngrams.values())
        if total == 0:
            return []

        for ng, cnt in q_ngrams.items():
            idx = self._vocab.get(ng)
            if idx is not None:
                q_vec[idx] = (cnt / total) * self._idf[idx]

        norms = self._np.linalg.norm(self._tfidf_matrix, axis=1)
        q_norm = self._np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        similarities = self._tfidf_matrix.dot(q_vec) / (norms * q_norm + 1e-10)

        top_idx = self._np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_idx:
            item = dict(self._items[idx])
            item["score"] = float(similarities[idx])
            results.append(item)

        return results

    # ── 内部方法 ──────────────────────────────────────────────

    def _extract_ngrams(self, text: str) -> Counter:
        text = text.lower()
        text = re.sub(r'[^a-z0-9一-鿿]', ' ', text)
        ngrams = Counter()
        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                if gram.strip():
                    ngrams[gram] += 1
        return ngrams

    def _collect_items(self) -> List[Dict[str, Any]]:
        items = []

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
                        "line_start": 0,
                    })
            except Exception as e:
                logger.warning(f"[Embedding] 读取知识图谱失败: {e}")

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
                            raw_content = f.read()
                    except Exception:
                        continue

                    content = re.sub(r'<[^>]+>', '', raw_content)
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

                        # 在原始内容中定位该章节标题 → 计算行号
                        first_heading = sec.split("\n")[0].strip()
                        raw_idx = raw_content.find(first_heading)
                        line_start = raw_content[:raw_idx].count('\n') + 1 if raw_idx >= 0 else 0

                        items.append({
                            "source": str(rel),
                            "title": fname.replace(".md", "") + " — " + title_line,
                            "snippet": sec[:200],
                            "text": sec[:500],
                            "line_start": line_start,
                        })

        return items

    def _save(self):
        self._lazy_np()
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._np.savez_compressed(str(self._index_path), matrix=self._tfidf_matrix)
        with open(self._map_path, "w", encoding="utf-8") as f:
            json.dump(self._items, f, ensure_ascii=False, indent=2)
        with open(self._vocab_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self._vocab, "idf": self._idf}, f, ensure_ascii=False)
        logger.info(f"[Embedding] 索引已保存: {len(self._items)} 条")

    def _load(self) -> bool:
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

            # 检测旧索引缺失字段，触发重建
            if self._items and "line_start" not in self._items[0]:
                logger.info("[Embedding] 旧索引缺少 line_start，触发重建")
                return False

            # 🧠 聪明检测：笔记文件有增减 → 自动重建
            if self._items and EmbeddingIndex._notes_changed(self._items):
                return False

            self._loaded = True
            logger.info(f"[Embedding] 加载索引: {len(self._items)} 条")
            return True
        except Exception as e:
            logger.warning(f"[Embedding] 加载失败: {e}")
            return False

    @staticmethod
    def _notes_changed(items: list) -> bool:
        """检测 data/Note/ 下的 .md 文件是否有增减

        对比当前目录中的 .md 文件列表 vs 索引中记录的 .md 来源列表。
        不检测内容修改（代价太高），只检测增删和改名。
        """
        note_dir = Path(config.note_dir)
        if not note_dir.exists():
            return False
        try:
            # 当前磁盘上的 .md 文件集（相对路径）
            on_disk = set()
            for p in note_dir.rglob("*.md"):
                rel = str(p.relative_to(note_dir))
                on_disk.add(rel)

            # 索引中记录的 .md 来源集
            in_index = set()
            for it in items:
                src = it.get("source", "")
                if src.endswith(".md"):
                    in_index.add(src)

            if on_disk != in_index:
                added = on_disk - in_index
                removed = in_index - on_disk
                if added:
                    logger.info(f"[Embedding] 检测到新增笔记: {', '.join(sorted(added))}")
                if removed:
                    logger.info(f"[Embedding] 检测到移除笔记: {', '.join(sorted(removed))}")
                return True
        except Exception as e:
            logger.warning(f"[Embedding] 笔记变化检测异常: {e}")
        return False


# ════════════════════════════════════════════════════════════════
# BM25 索引 — 词袋检索的黄金标准
# ════════════════════════════════════════════════════════════════


class BM25Index:
    """BM25 全文检索索引（基于 rank_bm25）"""

    def __init__(self):
        from note_weaver.utils.config import config as cfg
        self._index_path = Path(cfg.memory_dir) / "bm25_index.pkl"
        self._map_path = Path(cfg.memory_dir) / "bm25_map.json"
        self.model = None
        self._items: List[Dict[str, Any]] = []
        self._loaded = False

    def build(self, force: bool = False) -> int:
        if not force and self._load():
            return len(self._items)

        items = self._collect_items()
        if not items:
            return 0

        texts = [it["text"] for it in items]
        logger.info(f"[BM25] 构建索引: {len(texts)} 条...")

        from rank_bm25 import BM25Okapi
        tokenized = [self._tokenize(t) for t in texts]
        self.model = BM25Okapi(tokenized)
        self._items = items

        self._save()
        logger.info(f"[BM25] 索引完成: {len(items)} 条")
        return len(items)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._loaded and not self._load():
            return []

        tokenized = self._tokenize(query)
        scores = self.model.get_scores(tokenized)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            item = dict(self._items[idx])
            item["score"] = float(scores[idx])
            results.append(item)

        return results

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
            # 检测旧索引缺失字段，触发重建
            if self._items and "line_start" not in self._items[0]:
                logger.info("[BM25] 旧索引缺少 line_start，触发重建")
                return False
            self._loaded = True
            logger.info(f"[BM25] 加载索引: {len(self._items)} 条")
            return True
        except Exception as e:
            logger.warning(f"[BM25] 加载失败: {e}")
            return False

    def _save(self):
        import pickle as _p
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as f:
            _p.dump(self.model, f)
        with open(self._map_path, "w", encoding="utf-8") as f:
            json.dump(self._items, f, ensure_ascii=False, indent=2)
        logger.info(f"[BM25] 索引已保存: {len(self._items)} 条")

    @staticmethod
    def _tokenize(text: str) -> list:
        import re as _re
        text = text.lower()
        tokens = _re.findall(r'[a-z]+|[一-龥]', text)
        return tokens


# ════════════════════════════════════════════════════════════════
# Hybrid Retrieval — 双通道混合检索（BM25 + 可选语义通道）
# ════════════════════════════════════════════════════════════════


class SemanticIndex:
    """轻量语义向量索引，零外部强制依赖"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self._model_name = model_name
        self._model = None
        self._embeddings: Optional["np.ndarray"] = None
        self._items: List[Dict[str, Any]] = []
        self._loaded = False

    def build(self) -> int:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("[Semantic] sentence-transformers 未安装，跳过")
            return 0

        items = EmbeddingIndex()._collect_items()
        if not items:
            return 0

        texts = [it["text"] for it in items]
        logger.info(f"[Semantic] 编码 {len(texts)} 条...")
        self._model = SentenceTransformer(self._model_name)
        self._embeddings = self._model.encode(texts, show_progress_bar=False)
        self._items = items
        self._loaded = True
        logger.info(f"[Semantic] 编码完成: {len(texts)} 条 × {self._embeddings.shape[1]} 维")
        return len(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._loaded:
            return []
        import numpy as np
        q_vec = self._model.encode([query])[0]
        scores = self._embeddings.dot(q_vec) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-10
        )
        top_idx = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_idx:
            item = dict(self._items[idx])
            item["score"] = float(scores[idx])
            results.append(item)
        return results


class HybridRetrieval:
    """BM25 + 可选语义通道的混合检索"""

    def __init__(self, use_semantic: bool = False):
        self._use_semantic = use_semantic
        self._bm25 = BM25Index()
        self._semantic = SemanticIndex() if use_semantic else None
        self._built = False

    def build(self):
        if self._built:
            return
        self._bm25.build()
        if self._semantic:
            self._semantic.build()
        self._built = True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self._bm25.search(query, top_k=top_k * 2)
        if not results and self._semantic:
            results = self._semantic.search(query, top_k=top_k * 2)
        if not results:
            return []

        # 去重（按 title 去重）
        seen = set()
        deduped = []
        for r in results:
            key = r.get("title", "")
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:top_k]
