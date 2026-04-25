from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DynamicStateView:
    mood: str
    mood_detail: str
    relation_state: str
    user_emotion_hint: str
    recent_dialogue: str
    memory_snapshot: str
    session_context: str = ""


@dataclass(slots=True)
class EvidenceBundle:
    l0_identity: str = ""
    persona: str = ""
    story: str = ""
    external: str = ""

    def active_sources(self) -> list[str]:
        sources: list[str] = []
        if self.l0_identity.strip():
            sources.append("l0_identity")
        if self.persona.strip():
            sources.append("persona")
        if self.story.strip():
            sources.append("story")
        if self.external.strip():
            sources.append("external")
        return sources


@dataclass(slots=True)
class TurnRuntimeContext:
    user_input: str
    response_mode: str
    persona_focus: str
    character_name: str
    style_prompt: str
    dynamic_state: DynamicStateView
    evidence: EvidenceBundle
    response_contract: str = ""
    persona_focus_contract: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ResponsePlan:
    task_label: str
    evidence_required: bool
    evidence_ready: bool
    evidence_kind: str
    evidence_text: str
    constraints: list[str] = field(default_factory=list)
    max_tokens: int = 900
    temperature: float = 0.55
    continuation_max_tokens: int = 420
    allow_continuation: bool = False
