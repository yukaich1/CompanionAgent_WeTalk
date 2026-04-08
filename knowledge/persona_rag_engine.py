from __future__ import annotations

from pathlib import Path

from knowledge.knowledge_source import PersonaRecallResult
from knowledge.persona_state import PersonaState
from rag.models import RAGChunk
from rag.tool import RAGTool


class PersonaRAGEngine:
    def __init__(self, persona_system=None, persona_state: PersonaState | None = None):
        self.persona_system = persona_system
        self.persona_state = persona_state or PersonaState()
        self.rag = RAGTool(llm=getattr(persona_system, "model", None))

    def set_persona_state(self, persona_state: PersonaState | None) -> None:
        self.persona_state = persona_state or PersonaState()

    def _entries(self, allowed_kinds: set[str] | None = None) -> list[dict]:
        entries: list[dict] = []
        seen_ids: set[str] = set()
        if self.persona_system is None:
            return entries
        for raw in list(getattr(self.persona_system, "entries", []) or []):
            chunk = RAGChunk(**raw)
            if not chunk.content or chunk.chunk_id in seen_ids:
                continue
            if allowed_kinds and chunk.kind not in allowed_kinds:
                continue
            entries.append(chunk.model_dump())
            seen_ids.add(chunk.chunk_id)
        return entries

    def recall(self, query: str, exclude_chunk_ids: list[str] | None = None, preferred_query_type: str = "") -> PersonaRecallResult:
        query = str(query or "").strip()
        if not query:
            return PersonaRecallResult()
        excluded = {str(item or "").strip() for item in list(exclude_chunk_ids or []) if str(item or "").strip()}

        plan = self.rag.plan_query(query, query_type=preferred_query_type or "")
        query_type = plan.query_type or "general"

        if query_type == "story":
            primary_entries = self._entries({"story_chunk"})
            result = self.rag.search(query, primary_entries, top_k=3, query_type="story", enable_hyde=False) if primary_entries else None
            hits = list(result.hits if result is not None else [])
            if excluded:
                filtered_hits = [hit for hit in hits if str(hit.chunk_id or "") not in excluded]
                if filtered_hits:
                    hits = filtered_hits
            if not hits:
                return PersonaRecallResult(
                    integrated_context="",
                    coverage_score=0.0,
                    activated_features=[],
                    evidence_chunks=[],
                    source_breakdown={},
                    metadata={
                        "query_plan": plan.model_dump() if hasattr(plan, "model_dump") else plan.dict(),
                        "hitKinds": [],
                        "vector_unavailable": True,
                        "hits": [],
                        "story_hits": [],
                    },
                )

            top_hit = hits[0]
            story_payload = {
                "title": str(top_hit.metadata.get("title", "") or (top_hit.markdown_path[-1] if top_hit.markdown_path else top_hit.source_label)),
                "content": str(top_hit.content or ""),
                "score": float(top_hit.score or 0.0),
                "source_label": str(top_hit.source_label or ""),
                "chunk_id": str(top_hit.chunk_id or ""),
            }
            displayed_hits = hits[:1]
            metadata = {
                "query_plan": result.query_plan.model_dump() if hasattr(result.query_plan, "model_dump") else result.query_plan.dict(),
                "hitKinds": [str(hit.metadata.get("kind", "")) for hit in displayed_hits],
                "vector_unavailable": bool(result.metadata.get("vector_unavailable", False)),
                "candidate_hit_count": len(hits),
                "hits": [
                    {
                        "chunk_id": str(hit.chunk_id or ""),
                        "title": str(hit.metadata.get("title", "") or (hit.markdown_path[-1] if hit.markdown_path else hit.source_label)),
                        "source_label": str(hit.source_label or ""),
                        "markdown_path": list(hit.markdown_path or []),
                        "keywords": list(hit.keywords or []),
                        "score": round(float(hit.score or 0.0), 4),
                        "vector_score": round(float(hit.vector_score or 0.0), 4),
                        "lexical_score": round(float(hit.lexical_score or 0.0), 4),
                        "content": str(hit.content or "")[:320],
                    }
                    for hit in displayed_hits
                ],
                "story_hits": [story_payload],
            }
            coverage = max(0.0, min(1.0, float(top_hit.score or 0.0)))
            return PersonaRecallResult(
                integrated_context=str(top_hit.content or "").strip(),
                coverage_score=coverage,
                activated_features=[],
                evidence_chunks=[str(top_hit.content or "").strip()],
                source_breakdown={},
                metadata=metadata,
            )

        entries = self._entries({"source_chunk"})
        if not entries:
            return PersonaRecallResult()

        result = self.rag.search(query, entries, top_k=6, query_type=query_type or "persona", enable_hyde=False)
        evidence_chunks = [hit.content for hit in result.hits]
        metadata = {
            "query_plan": result.query_plan.model_dump() if hasattr(result.query_plan, "model_dump") else result.query_plan.dict(),
            "hitKinds": [str(hit.metadata.get("kind", "")) for hit in result.hits],
            "vector_unavailable": bool(result.metadata.get("vector_unavailable", False)),
            "hits": [
                {
                    "chunk_id": str(hit.chunk_id or ""),
                    "title": str(hit.metadata.get("title", "") or (hit.markdown_path[-1] if hit.markdown_path else hit.source_label)),
                    "source_label": str(hit.source_label or ""),
                    "markdown_path": list(hit.markdown_path or []),
                    "keywords": list(hit.keywords or []),
                    "score": round(float(hit.score or 0.0), 4),
                    "vector_score": round(float(hit.vector_score or 0.0), 4),
                    "lexical_score": round(float(hit.lexical_score or 0.0), 4),
                    "content": str(hit.content or "")[:220],
                }
                for hit in result.hits
            ],
            "story_hits": [],
        }
        top_score = result.hits[0].score if result.hits else 0.0
        coverage = max(0.0, min(1.0, top_score))
        return PersonaRecallResult(
            integrated_context="\n\n".join(chunk for chunk in evidence_chunks[:4] if str(chunk or "").strip()).strip(),
            coverage_score=coverage,
            activated_features=[],
            evidence_chunks=evidence_chunks[:6],
            source_breakdown={},
            metadata=metadata,
        )

    def save_snapshot(self, output_path: str | Path) -> None:
        path = Path(output_path)
        result = {
            "keywords": list(self.persona_state.metadata.get("display_keywords", [])),
            "chunk_count": len(self._entries()),
            "persona_name": self.persona_state.immutable_core.identity.name,
        }
        path.write_text(str(result), encoding="utf-8")
