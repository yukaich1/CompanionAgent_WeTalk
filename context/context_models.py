from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StableContextView:
    identity: str = ""
    style: str = ""
    rules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SessionContextView:
    topics: list[str] = field(default_factory=list)
    threads: list[str] = field(default_factory=list)
    archived_threads: list[str] = field(default_factory=list)
    relation_summary: str = ""
    emotion_summary: str = ""
    pinned_facts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TurnContextView:
    user_input: str = ""
    response_mode: str = ""
    evidence_sources: list[str] = field(default_factory=list)
    evidence_preview: str = ""
    recent_dialogue: str = ""


@dataclass(slots=True)
class SelectedContextView:
    stable: StableContextView
    session: SessionContextView
    turn: TurnContextView

    def as_dict(self) -> dict:
        return {
            "stable": {
                "identity": self.stable.identity,
                "style": self.stable.style,
                "rules": list(self.stable.rules),
            },
            "session": {
                "topics": list(self.session.topics),
                "threads": list(self.session.threads),
                "archived_threads": list(self.session.archived_threads),
                "relation_summary": self.session.relation_summary,
                "emotion_summary": self.session.emotion_summary,
                "pinned_facts": list(self.session.pinned_facts),
            },
            "turn": {
                "user_input": self.turn.user_input,
                "response_mode": self.turn.response_mode,
                "evidence_sources": list(self.turn.evidence_sources),
                "evidence_preview": self.turn.evidence_preview,
                "recent_dialogue": self.turn.recent_dialogue,
            },
        }
