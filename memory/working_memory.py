from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class ActiveThread:
    thread_id: str
    topic: str
    summary: str
    last_updated_turn: int
    unresolved: bool = True
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class WorkingMemoryState:
    active_topics: list[str] = field(default_factory=list)
    active_threads: list[ActiveThread] = field(default_factory=list)
    recent_emotion_summary: str = ""
    recent_relation_summary: str = ""
    recent_tool_summary: str = ""
    pinned_facts: list[str] = field(default_factory=list)


class WorkingMemoryManager:
    def __init__(self) -> None:
        self.state = WorkingMemoryState()

    def update_from_turn(
        self,
        *,
        turn_index: int,
        topic_hint: str,
        summary: str,
        relation_summary: str = "",
        emotion_summary: str = "",
        tool_summary: str = "",
        pinned_facts: list[str] | None = None,
    ) -> None:
        normalized_topic = str(topic_hint or "").strip()
        normalized_summary = str(summary or "").strip()
        if normalized_topic:
            if normalized_topic in self.state.active_topics:
                self.state.active_topics.remove(normalized_topic)
            self.state.active_topics.insert(0, normalized_topic)
            del self.state.active_topics[6:]

        if normalized_topic and normalized_summary:
            existing = next(
                (thread for thread in self.state.active_threads if thread.topic == normalized_topic),
                None,
            )
            if existing is None:
                existing = ActiveThread(
                    thread_id=f"thread-{turn_index}",
                    topic=normalized_topic,
                    summary=normalized_summary,
                    last_updated_turn=turn_index,
                )
                self.state.active_threads.insert(0, existing)
            else:
                existing.summary = normalized_summary
                existing.last_updated_turn = turn_index
                existing.updated_at = datetime.now()
                existing.unresolved = True
                self.state.active_threads.remove(existing)
                self.state.active_threads.insert(0, existing)
            del self.state.active_threads[4:]

        self.state.recent_relation_summary = str(relation_summary or "").strip()
        self.state.recent_emotion_summary = str(emotion_summary or "").strip()
        self.state.recent_tool_summary = str(tool_summary or "").strip()
        if pinned_facts:
            merged = [str(item or "").strip() for item in pinned_facts if str(item or "").strip()]
            for item in reversed(merged):
                if item in self.state.pinned_facts:
                    self.state.pinned_facts.remove(item)
                self.state.pinned_facts.insert(0, item)
            del self.state.pinned_facts[8:]

    def build_turn_snapshot(self) -> dict[str, object]:
        return {
            "active_topics": list(self.state.active_topics),
            "active_threads": [
                {
                    "thread_id": thread.thread_id,
                    "topic": thread.topic,
                    "summary": thread.summary,
                    "last_updated_turn": thread.last_updated_turn,
                    "unresolved": thread.unresolved,
                }
                for thread in self.state.active_threads
            ],
            "recent_emotion_summary": self.state.recent_emotion_summary,
            "recent_relation_summary": self.state.recent_relation_summary,
            "recent_tool_summary": self.state.recent_tool_summary,
            "pinned_facts": list(self.state.pinned_facts),
        }

    def render_working_memory(self) -> str:
        parts: list[str] = []
        if self.state.active_topics:
            parts.append("Active topics: " + " / ".join(self.state.active_topics))
        if self.state.active_threads:
            thread_lines = [
                f"- {thread.topic}: {thread.summary}"
                for thread in self.state.active_threads
                if str(thread.summary or "").strip()
            ]
            if thread_lines:
                parts.append("Open threads:\n" + "\n".join(thread_lines))
        if self.state.recent_relation_summary:
            parts.append("Relation trend: " + self.state.recent_relation_summary)
        if self.state.recent_emotion_summary:
            parts.append("Emotion trend: " + self.state.recent_emotion_summary)
        if self.state.recent_tool_summary:
            parts.append("Recent tool note: " + self.state.recent_tool_summary)
        if self.state.pinned_facts:
            parts.append("Pinned facts:\n" + "\n".join(f"- {item}" for item in self.state.pinned_facts))
        return "\n\n".join(parts).strip()
