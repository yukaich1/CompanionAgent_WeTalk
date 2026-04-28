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

    def build_hot_memory_index(self, *, max_lines: int = 12, max_chars: int = 900) -> str:
        snapshot = self.get_working_memory_snapshot()
        lines: list[str] = []
        topics = [str(item or "").strip() for item in list(snapshot.get("active_topics", []) or []) if str(item or "").strip()]
        if topics:
            lines.append("当前活跃话题: " + " / ".join(topics[:4]))
        threads = list(snapshot.get("active_threads", []) or [])
        for thread in threads[:3]:
            if not isinstance(thread, dict):
                continue
            topic = str(thread.get("topic", "") or "").strip()
            summary = str(thread.get("summary", "") or "").strip()
            if topic or summary:
                lines.append(f"当前线程: {topic or 'general'} - {summary[:120].rstrip()}")
        relation_summary = str(snapshot.get("recent_relation_summary", "") or "").strip()
        if relation_summary:
            lines.append("近期关系摘要: " + relation_summary[:180].rstrip())
        emotion_summary = str(snapshot.get("recent_emotion_summary", "") or "").strip()
        if emotion_summary:
            lines.append("近期情绪摘要: " + emotion_summary[:180].rstrip())
        tool_summary = str(snapshot.get("recent_tool_summary", "") or "").strip()
        if tool_summary:
            lines.append("最近工具结果: " + tool_summary[:160].rstrip())
        pinned = [str(item or "").strip() for item in list(snapshot.get("pinned_facts", []) or []) if str(item or "").strip()]
        for item in pinned[:4]:
            lines.append("重要事实: " + item[:140].rstrip())
        return self._render_memory_lines(lines, max_lines=max_lines, max_chars=max_chars)

    def build_warm_memory_context(self, query: str = "", *, limit: int = 5, max_chars: int = 1600) -> str:
        result = self.rag_engine.recall(query or self.build_hot_memory_index(max_chars=320), self.state, policy="semantic_first")
        blocks: list[str] = []
        for record in list(result.stable_records or [])[: max(1, min(limit, 3))]:
            content = str(getattr(record, "content", "") or "").strip()
            domain = str((getattr(record, "metadata", {}) or {}).get("domain", "") or "").strip()
            if content:
                prefix = f"[长期记忆{': ' + domain if domain else ''}]"
                blocks.append(f"{prefix} {content}")
        remaining = max(0, limit - len(blocks))
        for record in list(result.episode_records or [])[:remaining]:
            content = str(getattr(record, "content", "") or "").strip()
            summary = str((getattr(record, "metadata", {}) or {}).get("summary", "") or "").strip()
            if not content and not summary:
                continue
            body = summary or content
            blocks.append(f"[相关往事] {body[:220].rstrip()}")
        rendered = "\n".join(blocks).strip() or "None"
        return rendered if len(rendered) <= max_chars else rendered[:max_chars].rstrip() + "\n[提示] 温层记忆已压缩。"

    def build_cold_memory_hint(self) -> str:
        episode_count = len(list(self.state.episode_records or []))
        stable_count = len(list(self.state.stable_records or []))
        if episode_count == 0 and stable_count == 0:
            return "None"
        return (
            f"更早的历史互动仍保存在长期状态与会话轨迹中"
            f"（事件记忆 {episode_count} 条，稳定记忆 {stable_count} 条），"
            "当前轮不默认展开，只有在需要时再做深层召回。"
        )

    def build_memory_tiers(self, query: str = "") -> dict[str, str]:
        hot = self.build_hot_memory_index()
        warm = self.build_warm_memory_context(query=query)
        cold = self.build_cold_memory_hint()
        return {
            "hot_memory_index": hot,
            "warm_memory_context": warm,
            "cold_memory_hint": cold,
        }

    def _render_memory_lines(self, lines: list[str], *, max_lines: int, max_chars: int) -> str:
        kept: list[str] = []
        total_chars = 0
        truncated = False
        for line in lines:
            value = str(line or "").strip()
            if not value:
                continue
            projected = total_chars + len(value) + (1 if kept else 0)
            if len(kept) >= max_lines or projected > max_chars:
                truncated = True
                break
            kept.append(value)
            total_chars = projected
        if not kept:
            return "None"
        if truncated:
            kept.append("[提示] 热层记忆索引已截断，当前轮只加载关键指针。")
        return "\n".join(kept).strip()
