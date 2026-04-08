from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from memory.memory_rag_engine import MemoryRAGEngine
from memory.memory_writer import MemoryWriter
from memory.state_models import MemorySystemState
from utils import conversation_to_string, format_timestamp, get_approx_time_ago_str


@dataclass
class MemoryView:
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def format_memory(self):
        timedelta = datetime.now() - self.timestamp
        time_ago_str = get_approx_time_ago_str(timedelta)
        time_format = format_timestamp(self.timestamp)
        return f"<memory timestamp=\"{time_format}\" time_ago=\"{time_ago_str}\">{self.content}</memory>"


class MemorySystem:
    """Coordinates working-memory views and long-term memory recall."""

    def __init__(self, config, state: MemorySystemState | None = None, writer: MemoryWriter | None = None, rag_engine: MemoryRAGEngine | None = None):
        self.config = config
        self.state = state or MemorySystemState()
        self.writer = writer or MemoryWriter()
        self.rag_engine = rag_engine or MemoryRAGEngine()

    def set_state(self, state: MemorySystemState) -> None:
        self.state = state

    def get_beliefs(self) -> list[str]:
        return [record.content for record in self.state.semantic_records if record.content]

    def remember(self, content, emotion=None, is_insight=False):
        summary = str(content or "").strip()
        if not summary:
            return
        importance = 0.8 if is_insight else 0.55
        self.writer.remember(
            self.state,
            event_summary=summary,
            topic_tags=[],
            relation_impact={"familiarity_delta": 0.01},
            importance=importance,
            character_emotion=str(getattr(emotion, "name", "") or "neutral"),
        )

    def recall(self, query: str):
        result = self.rag_engine.recall(query, self.state)
        return [
            MemoryView(content=record.content)
            for record in result.episodic_records
            if str(record.content or "").strip()
        ]

    def tick(self, dt):
        return None

    def consolidate_memories(self):
        return None

    def get_recent_episodic_memories(self):
        recent = sorted(self.state.episodic_records, key=lambda item: item.created_at)[-8:]
        return [MemoryView(content=record.event_summary, timestamp=record.created_at) for record in recent]

    def build_working_memory(self, messages, limit: int = 6):
        filtered = [message for message in list(messages or []) if message.get("role") in {"user", "assistant"}]
        recent = filtered[-limit:]
        views: list[MemoryView] = []
        for message in recent:
            role = "User" if message.get("role") == "user" else "Assistant"
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                        text_parts.append(str(item.get("text", "")))
                content = "\n".join(part for part in text_parts if part).strip()
            else:
                content = str(content or "").strip()
            if not content:
                continue
            views.append(MemoryView(content=f"{role}: {content}"))
        return views

    def recall_memories(self, messages):
        messages = [message for message in messages if message["role"] != "system"]
        query = conversation_to_string(messages[-3:])
        recalled_memories = self.recall(query)
        return self.build_working_memory(messages), recalled_memories
