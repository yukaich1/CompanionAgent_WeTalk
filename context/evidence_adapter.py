from __future__ import annotations

from knowledge.knowledge_source import (
    MemoryRecallResult,
    PersonaRecallResult,
    UnifiedEvidenceItem,
)


class EvidenceAdapter:
    def adapt_persona(self, persona_result: PersonaRecallResult) -> list[UnifiedEvidenceItem]:
        items: list[UnifiedEvidenceItem] = []
        query_plan = dict((persona_result.metadata or {}).get("query_plan", {}) or {})

        if str(persona_result.integrated_context or "").strip():
            items.append(
                UnifiedEvidenceItem(
                    evidence_id="persona_integrated",
                    source_kind="persona_integrated",
                    content=str(persona_result.integrated_context or "").strip(),
                    title=str(query_plan.get("query_type", "") or "persona"),
                    priority=float(persona_result.coverage_score or 0.0),
                    metadata={"coverage_score": persona_result.coverage_score},
                )
            )

        for index, chunk in enumerate(list(persona_result.evidence_chunks or []), start=1):
            content = str(chunk or "").strip()
            if not content:
                continue
            items.append(
                UnifiedEvidenceItem(
                    evidence_id=f"persona_chunk_{index}",
                    source_kind="persona_chunk",
                    content=content,
                    title="persona_evidence",
                    priority=max(0.0, float(persona_result.coverage_score or 0.0) - (index * 0.02)),
                    metadata={"coverage_score": persona_result.coverage_score},
                )
            )
        return items

    def adapt_memory(self, memory_result: MemoryRecallResult) -> list[UnifiedEvidenceItem]:
        items: list[UnifiedEvidenceItem] = []

        for record in list(memory_result.stable_records or []):
            content = str(record.content or "").strip()
            if not content:
                continue
            meta = dict(record.metadata or {})
            items.append(
                UnifiedEvidenceItem(
                    evidence_id=str(record.record_id or ""),
                    source_kind="memory_stable",
                    content=content,
                    title=str(meta.get("domain", "") or "stable_memory"),
                    priority=float(meta.get("confidence", 0.6) or 0.6),
                    scope=str(meta.get("scope", "") or ""),
                    memory_type=str(meta.get("domain", "") or ""),
                    topic_room=str(meta.get("topic_room", "") or ""),
                    metadata=meta,
                )
            )

        for record in list(memory_result.episode_records or []):
            content = str(record.content or "").strip()
            if not content:
                continue
            meta = dict(record.metadata or {})
            items.append(
                UnifiedEvidenceItem(
                    evidence_id=str(record.record_id or ""),
                    source_kind="memory_episode",
                    content=content,
                    title=str(meta.get("summary", "") or meta.get("memory_type", "") or "episode"),
                    priority=float(meta.get("retrieval_score", 0.55) or 0.55),
                    scope=str(meta.get("scope", "") or ""),
                    memory_type=str(meta.get("memory_type", "") or ""),
                    topic_room=str(meta.get("topic_room", "") or ""),
                    metadata=meta,
                )
            )

        relation_state = dict(memory_result.relation_state or {})
        if relation_state:
            items.append(
                UnifiedEvidenceItem(
                    evidence_id="relation_state",
                    source_kind="relation_state",
                    content=str(relation_state),
                    title="relation_state",
                    priority=0.5,
                    metadata=relation_state,
                )
            )

        return items
