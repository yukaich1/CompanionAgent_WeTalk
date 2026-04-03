from __future__ import annotations

from difflib import SequenceMatcher

from knowledge.knowledge_source import DeduplicatedContext, MemoryRecallResult, PersonaRecallResult


class RecallDeduplicator:
    def dedup(self, persona_result: PersonaRecallResult, memory_result: MemoryRecallResult) -> DeduplicatedContext:
        persona_chunks = persona_result.evidence_chunks or [persona_result.integrated_context]
        for record in memory_result.episodic_records:
            for chunk in persona_chunks:
                if chunk and self._similarity(record.content, chunk) > 0.8:
                    record.inject_mode = "reference_only"
                    break
        return DeduplicatedContext(persona=persona_result, memory=memory_result)

    def _similarity(self, left: str, right: str) -> float:
        return SequenceMatcher(None, left or "", right or "").ratio()
