from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from llm import MistralEmbeddingCapacityError, mistral_embed_texts
from rag.models import RAGChunk, RAGQueryPlan, RAGSearchHit, RAGSearchResult
from rag.processing import DocumentProcessor, MarkdownSmartChunker, extract_keywords
from rag.qdrant_store import LocalQdrantStore


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
    def __init__(
        self,
        llm=None,
        document_processor: DocumentProcessor | None = None,
        chunker: MarkdownSmartChunker | None = None,
        vector_store: LocalQdrantStore | None = None,
    ):
        self.llm = llm
        self.document_processor = document_processor or DocumentProcessor()
        self.chunker = chunker or MarkdownSmartChunker()
        self.vector_store = vector_store or LocalQdrantStore()
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

    def embed_chunks(self, entries: list[dict], progress_callback=None, namespace: str = "") -> list[dict]:
        pending: list[tuple[dict, str]] = []
        for entry in list(entries or []):
            chunk = RAGChunk(**entry)
            if not chunk.embedding:
                pending.append((entry, chunk.content))
        if pending:
            embeddings = mistral_embed_texts([text for _, text in pending], progress_callback=progress_callback)
            for (entry, _), embedding in zip(pending, embeddings):
                entry["embedding"] = _normalize_vector(embedding).tolist()
        if namespace:
            self.store_entries(entries, namespace=namespace)
        return entries

    def store_entries(self, entries: list[dict], namespace: str = "default") -> int:
        return self.vector_store.upsert_entries(namespace=namespace, entries=entries)

    def clear_namespace(self, namespace: str) -> None:
        self.vector_store.clear_namespace(namespace)

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
        top_k: int = 5,
        query_type: str = "",
        enable_multi_query: bool = True,
        enable_hyde: bool = True,
        filters: dict | None = None,
        namespace: str = "default",
    ) -> RAGSearchResult:
        filters = dict(filters or {})
        chunks = [RAGChunk(**entry) for entry in list(entries or [])]
        filtered_chunks = [chunk for chunk in chunks if self._match_filters(chunk, filters)]
        plan = self.plan_query(query, query_type=query_type, enable_multi_query=enable_multi_query, enable_hyde=enable_hyde)
        if not filtered_chunks:
            return RAGSearchResult(
                query_plan=plan,
                hits=[],
                retrieval_mode="hybrid_qdrant",
                metadata={"hit_count": 0, "vector_unavailable": True, "vector_backend": "qdrant_local"},
            )

        lexical_index = BM25Okapi([_tokenize(chunk.as_search_text()) for chunk in filtered_chunks])
        retrieval_queries = [plan.direct_query, *plan.multi_queries]
        if plan.hyde_document:
            retrieval_queries.append(plan.hyde_document)

        vector_scores, vector_unavailable = self._vector_search_scores(filtered_chunks, retrieval_queries, namespace=namespace)
        lexical_scores = self._lexical_scores(lexical_index, filtered_chunks, retrieval_queries)

        hits: list[RAGSearchHit] = []
        min_score = 0.06
        if plan.query_type == "story":
            min_score = 0.08
        elif plan.query_type in {"persona", "character", "self_intro"}:
            min_score = 0.07

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
            retrieval_mode="hybrid_qdrant",
            metadata={
                "hit_count": len(hits),
                "vector_unavailable": vector_unavailable,
                "vector_backend": "qdrant_local",
                "namespace": namespace,
            },
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

    def _vector_search_scores(self, entries: list[RAGChunk], queries: Iterable[str], namespace: str) -> tuple[np.ndarray, bool]:
        scores = np.zeros(len(entries), dtype=np.float32)
        prepared = [query for query in queries if str(query or "").strip()]
        if not prepared:
            return scores, False

        vector_entries = [chunk for chunk in entries if chunk.embedding]
        if not vector_entries:
            return scores, True

        self.store_entries([chunk.model_dump() for chunk in vector_entries], namespace=namespace)
        allowed_ids = [chunk.chunk_id for chunk in vector_entries]
        offsets = {chunk.chunk_id: index for index, chunk in enumerate(entries)}
        limit = min(len(vector_entries), max(8, len(entries)))

        try:
            for query in prepared:
                query_vector = self._query_embedding(query)
                if query_vector is None:
                    continue
                current_scores = self.vector_store.search_scores(
                    namespace=namespace,
                    query_vector=query_vector,
                    allowed_ids=allowed_ids,
                    limit=limit,
                )
                for chunk_id, similarity in current_scores.items():
                    index = offsets.get(chunk_id)
                    if index is None:
                        continue
                    scores[index] = max(scores[index], float(similarity))
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
        weights = self._fusion_weights(plan.query_type)
        keyword_bonus = min(keyword_bonus, weights["keyword_cap"])

        vector_rank = int(np.argsort(-vector_scores).tolist().index(index)) + 1 if vector_scores.size else 999
        lexical_rank = int(np.argsort(-lexical_scores).tolist().index(index)) + 1 if lexical_scores.size else 999
        rrf_score = (1.0 / (60 + vector_rank)) + (1.0 / (60 + lexical_rank))
        return max(
            0.0,
            weights["vector"] * vector_score
            + weights["lexical"] * lexical_score
            + weights["rrf"] * rrf_score
            + keyword_bonus,
        )

    def _fusion_weights(self, query_type: str) -> dict[str, float]:
        normalized = str(query_type or "").strip().lower()
        if normalized in {"persona", "character", "self_intro"}:
            return {
                "vector": 0.42,
                "lexical": 0.42,
                "rrf": 5.2,
                "keyword_cap": 0.24,
            }
        if normalized == "story":
            return {
                "vector": 0.66,
                "lexical": 0.18,
                "rrf": 5.6,
                "keyword_cap": 0.16,
            }
        if normalized == "memory":
            return {
                "vector": 0.50,
                "lexical": 0.30,
                "rrf": 5.8,
                "keyword_cap": 0.18,
            }
        return {
            "vector": 0.56,
            "lexical": 0.26,
            "rrf": 6.0,
            "keyword_cap": 0.20,
        }

    def _infer_query_type(self, query: str) -> str:
        text = str(query or "")
        if any(token in text for token in ("故事", "经历", "过去", "那一次", "旅行", "冒险", "讲讲", "发生过")):
            return "story"
        if any(token in text for token in ("记得", "之前", "上次", "我们聊过", "还记得")):
            return "memory"
        if any(token in text for token in ("你是谁", "自我介绍", "设定", "性格", "口头禅", "喜好", "讨厌")):
            return "persona"
        return "general"

    def _multi_query_expand(self, query: str, query_type: str) -> list[str]:
        variants: list[str] = []
        keywords = extract_keywords(query, limit=5)
        if query_type == "story":
            variants.extend([f"{query} 经历", f"{query} 事件", f"{query} 旅途故事"])
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
