from __future__ import annotations

from knowledge.persona_system import GrowthLogEntry, PersonaState


class PersonaEvolutionEngine:
    def check_evolution(self, state: PersonaState, evidence_memory_ids: list[str], summary: str) -> PersonaState:
        if not summary or not evidence_memory_ids:
            return state
        state.slow_change_layer.growth_log.append(
            GrowthLogEntry(
                original_trait="",
                evolved_trait=summary,
                evidence_memory_ids=evidence_memory_ids,
                confidence=0.5,
            )
        )
        return state
