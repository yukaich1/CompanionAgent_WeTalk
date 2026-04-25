from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, cast

import numpy as np
import pydantic.typing as pydantic_typing

from rag.models import RAGChunk


def _patch_pydantic_forwardref_for_py313() -> None:
    original = getattr(pydantic_typing, "evaluate_forwardref", None)
    if original is None or getattr(original, "__name__", "") == "_py313_safe_evaluate_forwardref":
        return

    def _py313_safe_evaluate_forwardref(type_: Any, globalns: Any, localns: Any) -> Any:
        return cast(Any, type_)._evaluate(globalns, localns, recursive_guard=set())

    pydantic_typing.evaluate_forwardref = _py313_safe_evaluate_forwardref


_patch_pydantic_forwardref_for_py313()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, HasIdCondition, HnswConfigDiff, PointStruct, VectorParams


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize_namespace(namespace: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", str(namespace or "default")).strip("_").lower()
    return value[:48] or "default"


class LocalQdrantStore:
    _clients: dict[str, QdrantClient] = {}

    def __init__(self, root_path: str | Path | None = None):
        configured = str(os.getenv("QDRANT_LOCAL_PATH", "") or "").strip()
        base = Path(root_path or configured or (_project_root() / ".qdrant")).resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.root_path = base

    @property
    def client(self) -> QdrantClient:
        key = str(self.root_path)
        client = self._clients.get(key)
        if client is None:
            client = QdrantClient(path=str(self.root_path))
            self._clients[key] = client
        return client

    def collection_name(self, namespace: str) -> str:
        return f"witchtalk_{_sanitize_namespace(namespace)}"

    def clear_namespace(self, namespace: str) -> None:
        collection_name = self.collection_name(namespace)
        if self._collection_exists(collection_name):
            self.client.delete_collection(collection_name)

    def upsert_entries(self, namespace: str, entries: list[dict]) -> int:
        chunks = [RAGChunk(**entry) for entry in list(entries or [])]
        vector_chunks = [chunk for chunk in chunks if chunk.content and chunk.embedding]
        if not vector_chunks:
            return 0

        collection_name = self.collection_name(namespace)
        vector_size = len(vector_chunks[0].embedding or [])
        self._ensure_collection(collection_name, vector_size)

        points = [
            PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=list(chunk.embedding or []),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source_label": chunk.source_label,
                    "source_type": chunk.source_type,
                    "kind": chunk.kind,
                    "title": chunk.title,
                    "priority": float(chunk.priority or 1.0),
                    "keywords": list(chunk.keywords or []),
                    "markdown_path": list(chunk.markdown_path or []),
                    "metadata": dict(chunk.metadata or {}),
                },
            )
            for chunk in vector_chunks
        ]

        batch_size = 64
        for offset in range(0, len(points), batch_size):
            self.client.upsert(collection_name=collection_name, points=points[offset : offset + batch_size], wait=True)
        return len(points)

    def search_scores(self, namespace: str, query_vector: np.ndarray, allowed_ids: list[str], limit: int) -> dict[str, float]:
        filtered_ids = [str(item or "").strip() for item in list(allowed_ids or []) if str(item or "").strip()]
        if not filtered_ids:
            return {}

        collection_name = self.collection_name(namespace)
        if not self._collection_exists(collection_name):
            return {}

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=max(1, min(limit, len(filtered_ids))),
            query_filter=Filter(must=[HasIdCondition(has_id=[self._point_id(chunk_id) for chunk_id in filtered_ids])]),
            with_payload=True,
        )
        return {
            str((point.payload or {}).get("chunk_id") or ""): max(0.0, min(1.0, float(point.score or 0.0)))
            for point in results
            if str((point.payload or {}).get("chunk_id") or "").strip()
        }

    def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if self._collection_exists(collection_name):
            info = self.client.get_collection(collection_name)
            config = getattr(info.config.params, "vectors", None)
            current_size = getattr(config, "size", None)
            if current_size == vector_size:
                return
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        )

    def _collection_exists(self, collection_name: str) -> bool:
        exists = getattr(self.client, "collection_exists", None)
        if callable(exists):
            return bool(exists(collection_name))
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def _point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"witchtalk:{chunk_id}"))
