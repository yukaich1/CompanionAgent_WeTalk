from __future__ import annotations

from difflib import SequenceMatcher

from knowledge.knowledge_source import DeduplicatedContext, MemoryRecallResult, PersonaRecallResult


class RecallDeduplicator:
    def dedup(self, persona_result: PersonaRecallResult, memory_result: MemoryRecallResult) -> DeduplicatedContext:
        persona_chunks = [chunk for chunk in (persona_result.evidence_chunks or [persona_result.integrated_context]) if chunk]
        for record in memory_result.episode_records:
            for chunk in persona_chunks:
                if self._should_reference_only(record.content, chunk):
                    record.inject_mode = "reference_only"
                    break
        return DeduplicatedContext(persona=persona_result, memory=memory_result)

    def _should_reference_only(self, memory_text: str, persona_text: str) -> bool:
        left = (memory_text or "").strip()
        right = (persona_text or "").strip()
        if not left or not right:
            return False
        if len(left) < 24 or len(right) < 40:
            return False
        similarity = self._similarity(left, right)
        overlap = self._token_overlap(left, right)
        return similarity > 0.9 and overlap > 0.72

    def _similarity(self, left: str, right: str) -> float:
        return SequenceMatcher(None, left, right).ratio()

    def _token_overlap(self, left: str, right: str) -> float:
        left_tokens = {token for token in left.split() if token}
        right_tokens = {token for token in right.split() if token}
        if not left_tokens or not right_tokens:
            left_tokens = set(left)
            right_tokens = set(right)
        intersection = len(left_tokens & right_tokens)
        denominator = max(1, min(len(left_tokens), len(right_tokens)))
        return intersection / denominator
