from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class TurnInput:
    user_text: str
    attached_image: Any | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class TurnObjective:
    route_type: str
    response_mode: str
    persona_focus: str
    requires_tool: bool
    requires_persona: bool
    requires_memory: bool
    requires_story: bool
    requires_external: bool


@dataclass(slots=True)
class TurnArtifacts:
    normalized_text: str = ""
    recent_conversation: str = ""
    intent_stage: Any | None = None
    persona_stage: Any | None = None
    routing_stage: Any | None = None
    memory_stage: Any | None = None
    tool_stage: Any | None = None
    context_stage: Any | None = None
    route_decision: Any | None = None
    intent_result: Any | None = None
    persona_recall: Any | None = None
    memory_result: Any | None = None
    tool_report: Any | None = None
    assembled_context: Any | None = None
    grounding: dict[str, Any] = field(default_factory=dict)
    thought_data: dict[str, Any] = field(default_factory=dict)
    response_text: str = ""
    degraded: bool = False
