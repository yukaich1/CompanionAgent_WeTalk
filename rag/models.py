from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, root_validator


class CompatBaseModel(BaseModel):
    if not hasattr(BaseModel, "model_dump"):
        def model_dump(self, *args, **kwargs):
            return self.dict(*args, **kwargs)


class RAGChunk(CompatBaseModel):
    chunk_id: str
    document_id: str
    source_label: str
    source_type: str = "document"
    content: str
    markdown_path: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    token_count: int = 0
    priority: float = 1.0
    kind: str = "source_chunk"
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    @root_validator(pre=True)
    def _normalize_input(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        payload = dict(values)
        content = str(payload.get("content", "") or payload.get("text", "") or "").strip()
        metadata = dict(payload.get("metadata", {}) or {})
        kind = str(payload.get("kind") or metadata.get("kind") or "source_chunk").strip() or "source_chunk"
        title = str(payload.get("title") or metadata.get("title") or "").strip()
        chunk_id = str(payload.get("chunk_id") or payload.get("id") or "").strip()
        document_id = str(payload.get("document_id") or chunk_id or "").strip()
        payload.update(
            {
                "chunk_id": chunk_id,
                "document_id": document_id or chunk_id,
                "content": content,
                "kind": kind,
                "title": title,
                "metadata": {
                    **metadata,
                    "kind": kind,
                    "title": title,
                },
            }
        )
        return payload

    def as_search_text(self) -> str:
        parts = [
            " ".join(self.markdown_path),
            self.title,
            " ".join(self.keywords),
            self.content,
        ]
        return " ".join(part for part in parts if part).strip()


class RAGQueryPlan(CompatBaseModel):
    raw_query: str
    direct_query: str
    query_type: str = "general"
    multi_queries: list[str] = Field(default_factory=list)
    hyde_document: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGSearchHit(CompatBaseModel):
    chunk_id: str
    document_id: str
    source_label: str
    content: str
    markdown_path: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    score: float = 0.0
    vector_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    priority: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGSearchResult(CompatBaseModel):
    query_plan: RAGQueryPlan
    hits: list[RAGSearchHit] = Field(default_factory=list)
    retrieval_mode: str = "hybrid"
    metadata: dict[str, Any] = Field(default_factory=dict)
