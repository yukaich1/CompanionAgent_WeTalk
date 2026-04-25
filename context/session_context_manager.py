from __future__ import annotations

from dataclasses import dataclass, field

from context.context_compactor import ContextCompactor


@dataclass(slots=True)
class SessionContextState:
    active_topics: list[str] = field(default_factory=list)
    active_thread_summaries: list[str] = field(default_factory=list)
    archived_thread_summaries: list[str] = field(default_factory=list)
    relation_summary: str = ""
    emotion_summary: str = ""
    pinned_facts: list[str] = field(default_factory=list)


class SessionContextManager:
    def __init__(self) -> None:
        self.state = SessionContextState()
        self.compactor = ContextCompactor()

    def update_after_turn(
        self,
        *,
        active_topics: list[str] | None = None,
        active_thread_summaries: list[str] | None = None,
        relation_summary: str = "",
        emotion_summary: str = "",
        pinned_facts: list[str] | None = None,
    ) -> None:
        self.state.active_topics = self.compactor.compact_topics(list(active_topics or []), limit=6)
        active_threads, archived_threads = self.compactor.archive_thread_summaries(
            list(active_thread_summaries or []),
            list(self.state.archived_thread_summaries or []),
            active_limit=4,
            archive_limit=8,
        )
        self.state.active_thread_summaries = active_threads
        self.state.archived_thread_summaries = archived_threads
        self.state.relation_summary = str(relation_summary or "").strip()
        self.state.emotion_summary = str(emotion_summary or "").strip()
        self.state.pinned_facts = self.compactor.compact_pinned_facts(list(pinned_facts or []), limit=8)

    def build_session_context(self) -> dict[str, object]:
        return {
            "active_topics": list(self.state.active_topics),
            "active_thread_summaries": list(self.state.active_thread_summaries),
            "archived_thread_summaries": list(self.state.archived_thread_summaries),
            "relation_summary": self.state.relation_summary,
            "emotion_summary": self.state.emotion_summary,
            "pinned_facts": list(self.state.pinned_facts),
        }

    def restore(self, payload: dict[str, object] | None) -> None:
        payload = payload if isinstance(payload, dict) else {}
        self.state = SessionContextState(
            active_topics=self.compactor.compact_topics(list(payload.get("active_topics", []) or []), limit=6),
            active_thread_summaries=self.compactor.compact_thread_summaries(
                list(payload.get("active_thread_summaries", []) or []),
                limit=4,
            ),
            archived_thread_summaries=self.compactor.compact_thread_summaries(
                list(payload.get("archived_thread_summaries", []) or []),
                limit=8,
            ),
            relation_summary=str(payload.get("relation_summary", "") or "").strip(),
            emotion_summary=str(payload.get("emotion_summary", "") or "").strip(),
            pinned_facts=self.compactor.compact_pinned_facts(list(payload.get("pinned_facts", []) or []), limit=8),
        )

    def render(self) -> str:
        parts: list[str] = []
        if self.state.active_topics:
            parts.append("Session topics: " + " / ".join(self.state.active_topics))
        if self.state.active_thread_summaries:
            parts.append("Session threads:\n" + "\n".join(f"- {item}" for item in self.state.active_thread_summaries))
        if self.state.archived_thread_summaries:
            parts.append(
                "Session archived threads:\n"
                + "\n".join(f"- {item}" for item in self.state.archived_thread_summaries)
            )
        if self.state.relation_summary:
            parts.append("Session relation: " + self.state.relation_summary)
        if self.state.emotion_summary:
            parts.append("Session emotion: " + self.state.emotion_summary)
        if self.state.pinned_facts:
            parts.append("Session pinned facts:\n" + "\n".join(f"- {item}" for item in self.state.pinned_facts))
        return "\n\n".join(parts).strip()
