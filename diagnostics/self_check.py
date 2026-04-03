from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import DEFAULT_CONFIG
from knowledge.persona_system import PersonaState


@dataclass
class SelfCheckResult:
    ok: bool
    warnings: list[str]


class SelfCheck:
    def run(self, persona_state: PersonaState | None = None) -> SelfCheckResult:
        warnings: list[str] = []
        if DEFAULT_CONFIG.response_budget_tokens >= DEFAULT_CONFIG.context_window_tokens:
            raise ValueError("response budget must stay below total context window")
        slot_budget = DEFAULT_CONFIG.slot_budget.model_dump() if hasattr(DEFAULT_CONFIG.slot_budget, "model_dump") else DEFAULT_CONFIG.slot_budget.dict()
        slot_sum = sum(slot_budget.values())
        if slot_sum > DEFAULT_CONFIG.context_window_tokens * 0.8:
            raise ValueError("slot budget exceeds 80% of context window")
        if persona_state is not None:
            if not persona_state.immutable_core.identity.name and not persona_state.immutable_core.core_traits:
                warnings.append("persona_state is empty")
        return SelfCheckResult(ok=True, warnings=warnings)
