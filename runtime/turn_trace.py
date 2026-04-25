from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class TurnTrace:
    turn_id: str
    session_id: str
    input_text: str
    normalized_text: str
    created_at: datetime = field(default_factory=datetime.now)
    route_type: str = ""
    response_mode: str = ""
    persona_focus: str = ""
    selected_evidence_sources: list[str] = field(default_factory=list)
    selected_memory_layers: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    tool_policy: dict = field(default_factory=dict)
    recall_query: str = ""
    stage_summary: dict = field(default_factory=dict)
    evidence_gate: dict = field(default_factory=dict)
    emotion_before: dict = field(default_factory=dict)
    emotion_after: dict = field(default_factory=dict)
    relation_before: dict = field(default_factory=dict)
    relation_after: dict = field(default_factory=dict)
    session_context: dict = field(default_factory=dict)
    working_memory: dict = field(default_factory=dict)
    selected_context_view: dict = field(default_factory=dict)
    persistence_boundary: dict = field(default_factory=dict)
    memory_commit: dict = field(default_factory=dict)
    planner: dict = field(default_factory=dict)
    fallback_reason: str = ""
    final_response: str = ""
    latency_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "input_text": self.input_text,
            "normalized_text": self.normalized_text,
            "created_at": self.created_at.isoformat(),
            "route_type": self.route_type,
            "response_mode": self.response_mode,
            "persona_focus": self.persona_focus,
            "selected_evidence_sources": list(self.selected_evidence_sources),
            "selected_memory_layers": list(self.selected_memory_layers),
            "tool_calls": list(self.tool_calls),
            "tool_policy": dict(self.tool_policy),
            "recall_query": self.recall_query,
            "stage_summary": dict(self.stage_summary),
            "evidence_gate": dict(self.evidence_gate),
            "emotion_before": dict(self.emotion_before),
            "emotion_after": dict(self.emotion_after),
            "relation_before": dict(self.relation_before),
            "relation_after": dict(self.relation_after),
            "session_context": dict(self.session_context),
            "working_memory": dict(self.working_memory),
            "selected_context_view": dict(self.selected_context_view),
            "persistence_boundary": dict(self.persistence_boundary),
            "memory_commit": dict(self.memory_commit),
            "planner": dict(self.planner),
            "fallback_reason": self.fallback_reason,
            "final_response": self.final_response,
            "latency_ms": self.latency_ms,
        }
