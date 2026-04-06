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

    def _entries(self) -> list[dict]:
        entries: list[dict] = []
        seen_ids: set[str] = set()
        if self.persona_system is None:
            return entries
        for raw in list(getattr(self.persona_system, "entries", []) or []):
            chunk = RAGChunk(**raw)
            if not chunk.content or chunk.chunk_id in seen_ids or chunk.kind != "source_chunk":
                continue
            entries.append(chunk.model_dump())
            seen_ids.add(chunk.chunk_id)
        return entries

    def recall(self, query: str) -> PersonaRecallResult:
        query = str(query or "").strip()
        if not query:
            return PersonaRecallResult()
        entries = self._entries()
        if not entries:
            return PersonaRecallResult()

        plan = self.rag.plan_query(query)
        result = self.rag.search(query, entries, top_k=6, query_type=plan.query_type or "persona", enable_hyde=False)

        evidence_chunks = [hit.content for hit in result.hits]
        story_hits = []
        if plan.query_type == "story":
            story_hits = [
                {
                    "title": str(hit.metadata.get("title", "") or (hit.markdown_path[0] if hit.markdown_path else hit.source_label)),
                    "content": str(hit.content or ""),
                    "score": float(hit.score or 0.0),
                    "source_label": str(hit.source_label or ""),
                }
                for hit in result.hits
            ]

        metadata = {
            "query_plan": result.query_plan.model_dump() if hasattr(result.query_plan, "model_dump") else result.query_plan.dict(),
            "hitKinds": [str(hit.metadata.get("kind", "")) for hit in result.hits],
            "vector_unavailable": bool(result.metadata.get("vector_unavailable", False)),
            "story_hits": story_hits,
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
