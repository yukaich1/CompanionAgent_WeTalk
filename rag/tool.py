from __future__ import annotations

import hashlib
import re
from typing import Iterable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from llm import MistralEmbeddingCapacityError, mistral_embed_texts
from rag.models import RAGChunk, RAGQueryPlan, RAGSearchHit, RAGSearchResult
from rag.processing import DocumentProcessor, MarkdownSmartChunker, extract_keywords


def _normalize_vector(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    tokens = re.findall(r"[\u4e00-\u9fff]{1,8}|[A-Za-z][A-Za-z0-9:_/-]*", normalized)
    return [token for token in tokens if token]


class RAGTool:
    def __init__(self, llm=None, document_processor: DocumentProcessor | None = None, chunker: MarkdownSmartChunker | None = None):
        self.llm = llm
        self.document_processor = document_processor or DocumentProcessor()
        self.chunker = chunker or MarkdownSmartChunker()
        self._query_embedding_cache: dict[str, np.ndarray] = {}

    def convert_path_to_markdown(self, file_path: str) -> str:
        return self.document_processor.convert_path_to_markdown(file_path)

    def convert_text_to_markdown(self, text: str, title: str = "") -> str:
        return self.document_processor.convert_text_to_markdown(text, title=title)

    def build_chunks(
        self,
        markdown_text: str,
        document_id: str,
        source_label: str,
        source_type: str = "document",
        priority: float = 1.0,
        metadata: dict | None = None,
    ) -> list[dict]:
        chunks = self.chunker.chunk(
            markdown_text,
            document_id=document_id,
            source_label=source_label,
            source_type=source_type,
            priority=priority,
            metadata=metadata,
        )
        return [RAGChunk(**chunk).model_dump() for chunk in chunks]

    def embed_chunks(self, entries: list[dict], progress_callback=None) -> list[dict]:
        pending: list[tuple[dict, str]] = []
        for entry in entries:
            chunk = RAGChunk(**entry)
            if not chunk.embedding:
                pending.append((entry, chunk.content))
        if not pending:
            return entries
        embeddings = mistral_embed_texts([text for _, text in pending], progress_callback=progress_callback)
        for (entry, _), embedding in zip(pending, embeddings):
            entry["embedding"] = _normalize_vector(embedding).tolist()
        return entries

    def rebuild_index(self, entries: list[dict]) -> tuple[faiss.IndexFlatIP | None, int | None]:
        chunks = [RAGChunk(**entry) for entry in entries if str(entry.get("content", "") or entry.get("text", "") or "").strip()]
        vectors = [np.asarray(chunk.embedding, dtype=np.float32) for chunk in chunks if chunk.embedding]
        if not vectors:
            return None, None
        dim = len(vectors[0])
        index = faiss.IndexFlatIP(dim)
        index.add(np.asarray(vectors, dtype=np.float32))
        return index, dim

    def plan_query(self, query: str, query_type: str = "", enable_multi_query: bool = True, enable_hyde: bool = True) -> RAGQueryPlan:
        direct_query = re.sub(r"\s+", " ", str(query or "")).strip()
        inferred_type = query_type or self._infer_query_type(direct_query)
        if inferred_type in {"story", "persona", "character", "memory", "self_intro", "character_related"}:
            enable_hyde = False
        multi_queries = self._multi_query_expand(direct_query, inferred_type) if enable_multi_query else []
        hyde_document = self._hyde_expand(direct_query, inferred_type) if enable_hyde else ""
        return RAGQueryPlan(
            raw_query=direct_query,
            direct_query=direct_query,
            query_type=inferred_type,
            multi_queries=multi_queries,
            hyde_document=hyde_document,
            metadata={"keywords": extract_keywords(direct_query, limit=8)},
        )

    def search(
        self,
        query: str,
        entries: list[dict],
        index: faiss.IndexFlatIP | None = None,
        top_k: int = 5,
        query_type: str = "",
        enable_multi_query: bool = True,
        enable_hyde: bool = True,
        filters: dict | None = None,
    ) -> RAGSearchResult:
        filters = dict(filters or {})
        chunks = [RAGChunk(**entry) for entry in entries]
        filtered_chunks = [chunk for chunk in chunks if self._match_filters(chunk, filters)]
        plan = self.plan_query(query, query_type=query_type, enable_multi_query=enable_multi_query, enable_hyde=enable_hyde)
        if not filtered_chunks:
            return RAGSearchResult(query_plan=plan, hits=[], metadata={"hit_count": 0, "vector_unavailable": True})

        filtered_index = index if not filters and len(filtered_chunks) == len(chunks) else self.rebuild_index([chunk.model_dump() for chunk in filtered_chunks])[0]
        lexical_index = BM25Okapi([_tokenize(chunk.as_search_text()) for chunk in filtered_chunks])
        retrieval_queries = [plan.direct_query, *plan.multi_queries]
        if plan.hyde_document:
            retrieval_queries.append(plan.hyde_document)

        vector_scores, vector_unavailable = self._vector_search_scores(filtered_index, filtered_chunks, retrieval_queries)
        lexical_scores = self._lexical_scores(lexical_index, filtered_chunks, retrieval_queries)

        hits: list[RAGSearchHit] = []
        min_score = 0.10
        if plan.query_type == "story":
            min_score = 0.18
        elif plan.query_type in {"persona", "character", "self_intro"}:
            min_score = 0.12

        for offset, chunk in enumerate(filtered_chunks):
            vector_score = float(vector_scores[offset])
            lexical_score = float(lexical_scores[offset])
            rerank_score = self._rerank(chunk, plan, vector_score, lexical_score, offset, vector_scores, lexical_scores)
            final_score = rerank_score * float(chunk.priority or 1.0)
            if final_score < min_score:
                continue
            hits.append(
                RAGSearchHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_label=chunk.source_label,
                    content=chunk.content,
                    markdown_path=list(chunk.markdown_path or []),
                    keywords=list(chunk.keywords or []),
                    score=final_score,
                    vector_score=vector_score,
                    lexical_score=lexical_score,
                    rerank_score=rerank_score,
                    priority=float(chunk.priority or 1.0),
                    metadata=dict(chunk.metadata or {}),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return RAGSearchResult(
            query_plan=plan,
            hits=hits[:top_k],
            retrieval_mode="hybrid_cached",
            metadata={"hit_count": len(hits), "vector_unavailable": vector_unavailable},
        )

    def build_context_block(self, title: str, result: RAGSearchResult, char_budget: int = 900) -> str:
        if not result.hits:
            return ""
        lines = [f"=== {title} ==="]
        for index, hit in enumerate(result.hits, start=1):
            path = " > ".join(hit.markdown_path) if hit.markdown_path else hit.source_label
            lines.append(f"[{index}] {path}")
            lines.append(hit.content.strip())
        content = "\n".join(lines).strip()
        if len(content) > char_budget:
            return content[: max(0, char_budget - 3)].rstrip() + "..."
        return content

    def _match_filters(self, chunk: RAGChunk, filters: dict) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            value = getattr(chunk, key, None)
            if value in (None, ""):
                value = chunk.metadata.get(key)
            if isinstance(expected, (set, list, tuple)):
                if value not in expected:
                    return False
            elif value != expected:
                return False
        return True

    def _query_embedding(self, query: str) -> np.ndarray | None:
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()
        if key in self._query_embedding_cache:
            return self._query_embedding_cache[key]
        embedding = mistral_embed_texts([query])[0]
        vector = _normalize_vector(embedding).astype(np.float32)
        self._query_embedding_cache[key] = vector
        return vector

    def _vector_search_scores(self, index: faiss.IndexFlatIP | None, entries: list[RAGChunk], queries: Iterable[str]) -> tuple[np.ndarray, bool]:
        if index is None or index.ntotal == 0:
            return np.zeros(len(entries), dtype=np.float32), True
        scores = np.zeros(len(entries), dtype=np.float32)
        prepared = [query for query in queries if str(query or "").strip()]
        if not prepared:
            return scores, False
        try:
            for query in prepared:
                query_vector = self._query_embedding(query)
                if query_vector is None:
                    continue
                dense_scores, dense_indices = index.search(query_vector[np.newaxis, :], min(len(entries), max(8, len(entries))))
                for similarity, idx in zip(dense_scores[0], dense_indices[0]):
                    if idx < 0 or idx >= len(entries):
                        continue
                    scores[idx] = max(scores[idx], max(0.0, (float(similarity) + 1.0) / 2.0))
            return scores, False
        except MistralEmbeddingCapacityError:
            return np.zeros(len(entries), dtype=np.float32), True
        except Exception:
            return np.zeros(len(entries), dtype=np.float32), True

    def _lexical_scores(self, lexical_index: BM25Okapi, entries: list[RAGChunk], queries: Iterable[str]) -> np.ndarray:
        if not entries:
            return np.zeros(0, dtype=np.float32)
        scores = np.zeros(len(entries), dtype=np.float32)
        for query in queries:
            tokens = _tokenize(query)
            if not tokens:
                continue
            current = np.asarray(lexical_index.get_scores(tokens), dtype=np.float32)
            if current.size:
                peak = float(np.max(current))
                if peak > 0:
                    current = current / peak
                scores = np.maximum(scores, current)
        return scores

    def _rerank(
        self,
        chunk: RAGChunk,
        plan: RAGQueryPlan,
        vector_score: float,
        lexical_score: float,
        index: int,
        vector_scores: np.ndarray,
        lexical_scores: np.ndarray,
    ) -> float:
        text = chunk.as_search_text().lower()
        keyword_bonus = 0.0
        for keyword in plan.metadata.get("keywords", []):
            if keyword and keyword in text:
                keyword_bonus += 0.05
        keyword_bonus = min(keyword_bonus, 0.20)

        path_bonus = 0.0
        if plan.query_type in {"story", "persona", "character", "self_intro"} and chunk.kind == "source_chunk" and vector_score > 0.10:
            path_bonus = 0.08
        elif plan.query_type == "memory" and chunk.kind == "episodic_memory" and vector_score > 0.10:
            path_bonus = 0.08

        vector_rank = int(np.argsort(-vector_scores).tolist().index(index)) + 1 if vector_scores.size else 999
        lexical_rank = int(np.argsort(-lexical_scores).tolist().index(index)) + 1 if lexical_scores.size else 999
        rrf_score = (1.0 / (60 + vector_rank)) + (1.0 / (60 + lexical_rank))
        return max(0.0, 0.55 * vector_score + 0.25 * lexical_score + 6.0 * rrf_score + keyword_bonus + path_bonus)

    def _infer_query_type(self, query: str) -> str:
        text = str(query or "")
        if any(token in text for token in ("故事", "经历", "过去", "那一次", "旅行", "冒险", "讲讲")):
            return "story"
        if any(token in text for token in ("记得", "之前", "上次", "我们聊过", "还记得")):
            return "memory"
        if any(token in text for token in ("你是谁", "自我介绍", "设定", "性格", "口头禅", "喜好", "讨厌", "过去")):
            return "persona"
        return "general"

    def _multi_query_expand(self, query: str, query_type: str) -> list[str]:
        variants: list[str] = []
        keywords = extract_keywords(query, limit=5)
        if query_type == "story":
            variants.extend([f"{query} 经历", f"{query} 故事"])
        elif query_type in {"persona", "character", "self_intro"}:
            variants.extend([f"{query} 角色设定", f"{query} 人物资料"])
        elif query_type == "memory":
            variants.append(f"{query} 对话记忆")
        if len(keywords) >= 2:
            variants.append(" ".join(keywords[:3]))
        deduped: list[str] = []
        for item in variants:
            value = re.sub(r"\s+", " ", str(item or "")).strip()
            if value and value != query and value not in deduped:
                deduped.append(value)
        return deduped[:4]

    def _hyde_expand(self, query: str, query_type: str) -> str:
        if self.llm is None or query_type in {"story", "persona", "character", "self_intro", "memory"}:
            return ""
        prompt = (
            "根据这个问题，写一小段适合检索的中性资料片段。"
            "不要直接回答用户，不要推断未知事实，不要编造人物设定。"
            f"\n问题：{query}\n类型：{query_type}"
        )
        try:
            text = self.llm.generate(prompt, temperature=0.1, max_tokens=120)
        except Exception:
            return ""
        return re.sub(r"\s+", " ", str(text or "")).strip()
