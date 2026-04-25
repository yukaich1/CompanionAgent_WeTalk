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

    def _state_parent_entries(self, allowed_kinds: set[str] | None = None) -> list[dict]:
        entries: list[dict] = []
        seen_ids: set[str] = set()
        vault = getattr(self.persona_state, "evidence_vault", None)
        for parent in list(getattr(vault, "parent_chunks", []) or []):
            kind = str(getattr(parent, "kind", "") or "source_chunk")
            if allowed_kinds and kind not in allowed_kinds:
                continue
            chunk_id = str(getattr(parent, "chunk_id", "") or "").strip()
            content = str(getattr(parent, "content", "") or "").strip()
            if not chunk_id or not content or chunk_id in seen_ids:
                continue
            metadata = dict(getattr(parent, "metadata", {}) or {})
            metadata.setdefault("kind", kind)
            metadata.setdefault("title", str(getattr(parent, "title", "") or ""))
            entries.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "kind": kind,
                    "metadata": metadata,
                    "source_label": str(getattr(parent, "source_level", "") or ""),
                    "keywords": list(getattr(parent, "topic_tags", []) or []),
                }
            )
            seen_ids.add(chunk_id)
        return entries

    def _base_template_facts(self, query_type: str) -> list[str]:
        if self.persona_system is None:
            return []
        base_template = getattr(self.persona_system, "base_template", {}) or {}
        facts: list[str] = []
        background = base_template.get("00_BACKGROUND", {})
        profile = background.get("profile", {}) if isinstance(background, dict) else {}
        for key, value in dict(profile or {}).items():
            text = str(value or "").strip()
            if text:
                facts.append(f"{key}: {text}")
        for key in ("A_CORE_PERSONALITY", "C_VALUES_AND_BELIEFS", "E_LIKES", "F_DISLIKES_AND_TABOOS", "B_CATCHPHRASES"):
            section = base_template.get(key, {})
            if not isinstance(section, dict):
                continue
            for field in ("rules", "items", "patterns", "key_experiences"):
                for item in list(section.get(field, []) or []):
                    text = str(item or "").strip()
                    if text:
                        facts.append(text)
        voice_card = str(getattr(self.persona_system, "character_voice_card", "") or "").strip()
        if voice_card and query_type in {"persona", "identity", "self_intro", "general"}:
            facts.append(voice_card)
        return facts

    def _fallback_result(self, query: str, query_type: str, allowed_kinds: set[str]) -> PersonaRecallResult:
        rag_entries = self._entries(allowed_kinds)
        state_entries = self._state_parent_entries(allowed_kinds)
        base_facts = self._base_template_facts(query_type)
        merged_entries = rag_entries + [entry for entry in state_entries if entry["chunk_id"] not in {item["chunk_id"] for item in rag_entries}]
        snippets: list[str] = []
        hit_payloads: list[dict] = []

        if query_type == "story" and not merged_entries:
            return PersonaRecallResult(
                integrated_context="",
                coverage_score=0.0,
                activated_features=[],
                evidence_chunks=[],
                source_breakdown={},
                metadata={
                    "query_plan": {"query": query, "query_type": query_type, "fallback": True},
                    "hitKinds": [],
                    "vector_unavailable": True,
                    "hits": [],
                    "story_hits": [],
                    "fallback_source": "none",
                },
            )

        for entry in merged_entries[:6]:
            content = str(entry.get("content", "") or "").strip()
            if not content:
                continue
            snippets.append(content)
            hit_payloads.append(
                {
                    "chunk_id": str(entry.get("chunk_id", "") or ""),
                    "title": str((entry.get("metadata", {}) or {}).get("title", "") or ""),
                    "source_label": str(entry.get("source_label", "") or "persona_state"),
                    "markdown_path": [],
                    "keywords": list(entry.get("keywords", []) or []),
                    "score": 0.18,
                    "vector_score": 0.0,
                    "lexical_score": 0.0,
                    "content": content[:220],
                }
            )
            if len(snippets) >= 3:
                break

        if query_type != "story" and not hit_payloads and base_facts:
            synthetic_kind = "story_chunk" if query_type == "story" else "source_chunk"
            for index, fact in enumerate(base_facts[:3], start=1):
                hit_payloads.append(
                    {
                        "chunk_id": f"fallback_{synthetic_kind}_{index}",
                        "title": "persona_fallback",
                        "source_label": "persona_base_template",
                        "markdown_path": [],
                        "keywords": [],
                        "score": 0.16,
                        "vector_score": 0.0,
                        "lexical_score": 0.0,
                        "content": str(fact or "")[:220],
                    }
                )
                snippets.append(str(fact or "").strip())

        if query_type == "story" and snippets:
            story_payload = {
                "title": str(hit_payloads[0].get("title", "") or "story_fallback"),
                "content": snippets[0],
                "score": float(hit_payloads[0].get("score", 0.18) or 0.18),
                "source_label": str(hit_payloads[0].get("source_label", "") or "persona_state"),
                "chunk_id": str(hit_payloads[0].get("chunk_id", "") or ""),
            }
            return PersonaRecallResult(
                integrated_context=snippets[0],
                coverage_score=float(hit_payloads[0].get("score", 0.18) or 0.18),
                activated_features=[],
                evidence_chunks=[snippets[0]],
                source_breakdown={},
                metadata={
                    "query_plan": {"query": query, "query_type": query_type, "fallback": True},
                    "hitKinds": [str((entry.get("metadata", {}) or {}).get("kind", "story_chunk")) for entry in merged_entries[:1]],
                    "vector_unavailable": True,
                    "hits": hit_payloads[:1],
                    "story_hits": [story_payload],
                    "fallback_source": "persona_state",
                },
            )

        if base_facts:
            snippets.extend(item for item in base_facts if item not in snippets)
        snippets = [item for item in snippets if str(item or "").strip()][:4]
        return PersonaRecallResult(
            integrated_context="\n\n".join(snippets).strip(),
            coverage_score=float(hit_payloads[0].get("score", 0.18) or 0.18) if snippets and hit_payloads else (0.18 if snippets else 0.0),
            activated_features=[],
            evidence_chunks=snippets,
            source_breakdown={},
            metadata={
                "query_plan": {"query": query, "query_type": query_type, "fallback": True},
                "hitKinds": [str((entry.get("metadata", {}) or {}).get("kind", "source_chunk")) for entry in merged_entries[: len(hit_payloads)]],
                "vector_unavailable": True,
                "hits": hit_payloads,
                "story_hits": [],
                "fallback_source": "persona_state",
            },
        )

    def recall(self, query: str, exclude_chunk_ids: list[str] | None = None, preferred_query_type: str = "") -> PersonaRecallResult:
        query = str(query or "").strip()
        if not query:
            return PersonaRecallResult()
        excluded = {str(item or "").strip() for item in list(exclude_chunk_ids or []) if str(item or "").strip()}

        plan = self.rag.plan_query(query, query_type=preferred_query_type or "")
        query_type = plan.query_type or "general"

        if query_type == "story":
            primary_entries = self._entries({"story_chunk"})
            result = self.rag.search(
                query,
                primary_entries,
                top_k=3,
                query_type="story",
                enable_hyde=False,
                namespace="persona",
            ) if primary_entries else None
            hits = list(result.hits if result is not None else [])
            if excluded:
                filtered_hits = [hit for hit in hits if str(hit.chunk_id or "") not in excluded]
                if filtered_hits:
                    hits = filtered_hits
            if not hits:
                return self._fallback_result(query, query_type, {"story_chunk"})

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
            return self._fallback_result(query, query_type, {"source_chunk"})

        result = self.rag.search(
            query,
            entries,
            top_k=6,
            query_type=query_type or "persona",
            enable_hyde=False,
            namespace="persona",
        )
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
        if not result.hits or float(top_score or 0.0) < 0.22:
            fallback = self._fallback_result(query, query_type, {"source_chunk"})
            if fallback.evidence_chunks:
                return fallback
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
