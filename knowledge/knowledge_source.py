from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeSource(str, Enum):
    USER_CANON = "user_canon"
    WEB_PERSONA = "web_persona"
    WEB_REALITY = "web_reality"
    DIALOGUE_MEMORY = "dialogue_memory"
    MODEL_PRIOR = "model_prior"


SOURCE_PRIORITY: dict[KnowledgeSource, int] = {
    KnowledgeSource.USER_CANON: 1,
    KnowledgeSource.WEB_PERSONA: 2,
    KnowledgeSource.WEB_REALITY: 3,
    KnowledgeSource.DIALOGUE_MEMORY: 4,
    KnowledgeSource.MODEL_PRIOR: 5,
}


class SearchMode(str, Enum):
    NONE = "none"
    PERSONA_SEARCH = "persona_search"
    REALITY_SEARCH = "reality_search"
    BOTH = "both"


class RouteType(str, Enum):
    E1 = "E1"
    E2 = "E2"
    E2B = "E2b"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"


class RouteDecision(BaseModel):
    type: RouteType
    web_search_mode: SearchMode = SearchMode.NONE
    search_hint: list[str] = Field(default_factory=list)
    info_domain: str | None = None


class PersonaRecallResult(BaseModel):
    integrated_context: str = ""
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    activated_features: list[str] = Field(default_factory=list)
    evidence_chunks: list[str] = Field(default_factory=list)
    source_breakdown: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRecordView(BaseModel):
    record_id: str
    content: str
    source: KnowledgeSource = KnowledgeSource.DIALOGUE_MEMORY
    inject_mode: str = "full"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRecallResult(BaseModel):
    episode_records: list[MemoryRecordView] = Field(default_factory=list)
    stable_records: list[MemoryRecordView] = Field(default_factory=list)
    relation_state: dict[str, Any] = Field(default_factory=dict)


class UnifiedEvidenceItem(BaseModel):
    evidence_id: str
    source_kind: str
    content: str
    title: str = ""
    priority: float = 0.0
    scope: str = ""
    memory_type: str = ""
    topic_room: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeduplicatedContext(BaseModel):
    persona: PersonaRecallResult = Field(default_factory=PersonaRecallResult)
    memory: MemoryRecallResult = Field(default_factory=MemoryRecallResult)


class AssembledContext(BaseModel):
    route_type: RouteType
    slots: dict[str, str] = Field(default_factory=dict)
    token_budget: int
    metadata: dict[str, Any] = Field(default_factory=dict)
