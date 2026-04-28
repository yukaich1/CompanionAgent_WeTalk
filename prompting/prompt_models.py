from __future__ import annotations

from dataclasses import dataclass, field

from turn_runtime import ResponsePlan, TurnRuntimeContext


@dataclass(slots=True)
class StablePromptSection:
    character_name: str
    static_persona_prompt: str
    base_rules: list[str]
    dynamic_boundary_marker: str
    cache_key: str


@dataclass(slots=True)
class PromptAssembly:
    system_prompt: str
    dynamic_prompt: str
    cache_key: str
    boundary_marker: str
    compaction_report: dict[str, object] = field(default_factory=dict)

    @property
    def combined_prompt(self) -> str:
        return f"{self.system_prompt}\n\n{self.dynamic_prompt}".strip()


@dataclass(slots=True)
class PromptSections:
    stable: StablePromptSection
    turn_plan: ResponsePlan
    turn_context: TurnRuntimeContext
