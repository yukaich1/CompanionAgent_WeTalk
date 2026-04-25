from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EpisodicRecord(BaseModel):
    record_id: str
    summary: str
    verbatim_excerpt: str = ""
    user_text: str = ""
    assistant_text: str = ""
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
    memory_type: str = ""
    topic_room: str = ""
    scope: str = ""
    source_session_id: str = ""
    source_turn_index: int = 0
    promoted: bool = False
    conflicted: bool = False

    def recall_text(self) -> str:
        excerpt = str(self.verbatim_excerpt or "").strip()
        if excerpt:
            return excerpt

        user_text = str(self.user_text or "").strip()
        assistant_text = str(self.assistant_text or "").strip()
        if user_text or assistant_text:
            parts = []
            if user_text:
                parts.append(f"User: {user_text}")
            if assistant_text:
                parts.append(f"Assistant: {assistant_text}")
            return "\n".join(parts).strip()

        return str(self.summary or "").strip()

    def display_text(self) -> str:
        return str(self.summary or "").strip() or self.recall_text()


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
    episode_records: list[EpisodicRecord] = Field(default_factory=list)
    stable_records: list[SemanticRecord] = Field(default_factory=list)
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
