from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IntentStageResult:
    normalized_query: str
    recent_conversation: str
    intent_result: Any
    recall_query: str


@dataclass(slots=True)
class PersonaRecallStageResult:
    recall_mode: str
    preferred_query_type: str
    excluded_chunk_ids: list[str] = field(default_factory=list)
    persona_recall: Any = None


@dataclass(slots=True)
class RoutingStageResult:
    route_decision: Any


@dataclass(slots=True)
class MemoryStageResult:
    memory_result: Any = None
    recall_policy: str = ""


@dataclass(slots=True)
class ToolStageResult:
    tool_report: Any = None
    web_persona_context: str = ""
    web_reality_context: str = ""


@dataclass(slots=True)
class ContextStageResult:
    deduped: Any = None
    assembled_context: Any = None
