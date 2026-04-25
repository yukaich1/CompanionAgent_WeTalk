from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from memory.episodic_memory_manager import EpisodicMemoryManager
from memory.memory_compactor import MemoryCompactor
from memory.memory_rag_engine import MemoryRAGEngine
from memory.memory_taxonomy import classify_memory_taxonomy
from memory.relation_state_manager import RelationStateManager
from memory.semantic_memory_manager import SemanticMemoryManager
from memory.working_memory import WorkingMemoryManager
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
        self.working_memory = WorkingMemoryManager()
        self.episodic_manager = EpisodicMemoryManager()
        self.semantic_manager = SemanticMemoryManager()
        self.compactor = MemoryCompactor()
        self.relation_state_manager = RelationStateManager()

    def set_state(self, state: MemorySystemState) -> None:
        self.state = state

    def get_beliefs(self) -> list[str]:
        return [record.content for record in self.state.stable_records if record.content]

    def remember(self, content, emotion=None, is_insight=False):
        summary = str(content or "").strip()
        if not summary:
            return
        importance = 0.8 if is_insight else 0.55
        taxonomy = classify_memory_taxonomy(summary, llm=getattr(self.writer, "llm", None))
        self.writer.remember(
            self.state,
            summary=summary,
            verbatim_excerpt=summary,
            topic_tags=[],
            relation_impact={"familiarity_delta": 0.01},
            importance=importance,
            character_emotion=str(getattr(emotion, "name", "") or "neutral"),
            memory_type=str(taxonomy.get("memory_type", "") or ""),
            topic_room=str(taxonomy.get("topic_room", "") or ""),
        )

    def recall(self, query: str, *, policy: str = "episodic_hybrid"):
        result = self.rag_engine.recall(query, self.state, policy=policy)
        self.episodic_manager.reinforce_from_recall(self.state, result.episode_records)
        return [
            MemoryView(content=record.content)
            for record in result.episode_records
            if str(record.content or "").strip()
        ]

    def tick(self, dt):
        self.episodic_manager.decay(self.state, float(dt or 0.0))
        return None

    def consolidate_memories(self):
        self.semantic_manager.merge_semantic_records(self.state)
        self.compactor.compact_episode_window(self.state)
        return None

    def get_recent_episode_memories(self):
        recent = sorted(self.state.episode_records, key=lambda item: item.created_at)[-8:]
        return [MemoryView(content=record.display_text(), timestamp=record.created_at) for record in recent]

    def build_working_memory(self, messages, limit: int = 6):
        structured_snapshot = self.working_memory.render_working_memory()
        if structured_snapshot:
            return [MemoryView(content=structured_snapshot)]
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
        working_snapshot = self.get_working_memory_snapshot()
        query = self._build_recall_query(messages, working_snapshot)
        policy = self._select_recall_policy(query, working_snapshot)
        recalled_memories = self.recall(query, policy=policy)
        return self.build_working_memory(messages), recalled_memories

    def _build_recall_query(self, messages, working_snapshot: dict[str, object]) -> str:
        query_parts = [
            conversation_to_string(messages[-3:]),
            " / ".join(str(item or "").strip() for item in list(working_snapshot.get("active_topics", []) or []) if str(item or "").strip()),
            "\n".join(
                str(thread.get("summary", "") or "").strip()
                for thread in list(working_snapshot.get("active_threads", []) or [])
                if isinstance(thread, dict) and str(thread.get("summary", "") or "").strip()
            ),
        ]
        return "\n".join(part for part in query_parts if str(part or "").strip()).strip()

    def _select_recall_policy(self, query: str, working_snapshot: dict[str, object]) -> str:
        text = str(query or "").strip()
        if any(marker in text for marker in ("还记得", "刚刚", "上次", "之前说过", "我们聊过")):
            return "thread_first"
        active_topics = list(working_snapshot.get("active_topics", []) or [])
        if active_topics:
            return "semantic_first"
        return "episodic_hybrid"

    def update_working_memory(
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
        self.working_memory.update_from_turn(
            turn_index=turn_index,
            topic_hint=topic_hint,
            summary=summary,
            relation_summary=relation_summary,
            emotion_summary=emotion_summary,
            tool_summary=tool_summary,
            pinned_facts=pinned_facts,
        )

    def get_working_memory_snapshot(self) -> dict[str, object]:
        return self.working_memory.build_turn_snapshot()

    def relation_summary(self) -> str:
        return self.relation_state_manager.render_summary(self.state.relation_state)

    def relation_profile(self) -> dict[str, object]:
        return self.relation_state_manager.render_profile(self.state.relation_state)
