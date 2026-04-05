from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EpisodicRecord(BaseModel):
    record_id: str
    event_summary: str
    perspective: Literal["CHARACTER_FIRST", "NARRATOR"] = "CHARACTER_FIRST"
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    emotional_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    character_emotion: str = "neutral"
    relation_impact: dict[str, float] = Field(default_factory=dict)
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    last_recalled_at: datetime | None = None
    recall_count: int = 0
    topic_tags: list[str] = Field(default_factory=list)
    promoted: bool = False
    conflicted: bool = False


class SemanticRecord(BaseModel):
    record_id: str
    source_episode_ids: list[str] = Field(default_factory=list)
    content: str
    domain: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    scope: Literal["USER_SPECIFIC"] = "USER_SPECIFIC"
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated_at: datetime = Field(default_factory=datetime.now)


class RelationState(BaseModel):
    trust: float = Field(default=0.0, ge=0.0, le=1.0)
    affection: float = Field(default=0.0, ge=0.0, le=1.0)
    familiarity: float = Field(default=0.0, ge=0.0, le=1.0)
    stage: str = "stranger"
    last_significant_event: str = ""
    last_updated: datetime = Field(default_factory=datetime.now)


class MemorySystemState(BaseModel):
    episodic_records: list[EpisodicRecord] = Field(default_factory=list)
    semantic_records: list[SemanticRecord] = Field(default_factory=list)
    relation_state: RelationState = Field(default_factory=RelationState)


class MemorySystemStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> MemorySystemState:
        if not self.path.exists():
            return MemorySystemState()
        raw = self.path.read_text(encoding="utf-8")
        if hasattr(MemorySystemState, "model_validate_json"):
            return MemorySystemState.model_validate_json(raw)
        return MemorySystemState.parse_raw(raw)

    def save(self, state: MemorySystemState) -> None:
        if hasattr(state, "model_dump_json"):
            payload = state.model_dump_json(indent=2)
        else:
            payload = state.json(indent=2, ensure_ascii=False)
        self.path.write_text(payload, encoding="utf-8")
