from __future__ import annotations

from knowledge.persona_state import GrowthLogEntry, PersonaState


class PersonaEvolutionEngine:
    def check_evolution(
        self,
        state: PersonaState,
        evidence_memory_ids: list[str],
        summary: str,
        original_trait: str = "",
        confidence: float = 0.5,
    ) -> PersonaState:
        if not summary or not evidence_memory_ids:
            return state
        state.slow_change_layer.growth_log.append(
            GrowthLogEntry(
                original_trait=original_trait,
                evolved_trait=summary,
                summary=summary,
                evidence_memory_ids=evidence_memory_ids,
                confidence=confidence,
            )
        )
        return state
