from __future__ import annotations

from dataclasses import dataclass

from turn_runtime import ResponsePlan, TurnRuntimeContext


@dataclass(slots=True)
class StablePromptSection:
    character_name: str
    style_prompt: str
    base_rules: list[str]


@dataclass(slots=True)
class PromptSections:
    stable: StablePromptSection
    turn_plan: ResponsePlan
    turn_context: TurnRuntimeContext
