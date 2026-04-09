from __future__ import annotations

import uuid
from datetime import datetime

from config import DEFAULT_CONFIG
from diagnostics.conflict_log import ConflictLog
from memory.memory_taxonomy import (
    build_verbatim_excerpt,
    classify_memory_taxonomy,
    normalize_topic_tags,
)
from memory.state_models import EpisodicRecord, MemorySystemState, RelationState, SemanticRecord


class MemoryWriter:
    SEMANTIC_SUMMARY_SCHEMA = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "domain": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["content", "domain", "confidence"],
    }

    def __init__(self, conflict_log: ConflictLog | None = None, llm=None):
        self.conflict_log = conflict_log or ConflictLog()
        self.llm = llm

    def remember(
        self,
        state: MemorySystemState,
        summary: str,
        user_text: str = "",
        assistant_text: str = "",
        verbatim_excerpt: str = "",
        topic_tags: list[str] | None = None,
        relation_impact: dict[str, float] | None = None,
        importance: float = 0.5,
        character_emotion: str = "neutral",
        memory_type: str = "",
        topic_room: str = "",
        scope: str = "",
        source_session_id: str = "",
        source_turn_index: int = 0,
    ) -> MemorySystemState:
        normalized_tags = normalize_topic_tags(topic_tags)
        resolved_summary = str(summary or "").strip()
        resolved_verbatim = str(verbatim_excerpt or "").strip() or build_verbatim_excerpt(
            user_text=user_text,
            assistant_text=assistant_text,
        )
        taxonomy = {}
        if not str(memory_type or "").strip() or not str(topic_room or "").strip():
            taxonomy = classify_memory_taxonomy(
                resolved_summary,
                user_text=user_text,
                assistant_text=assistant_text,
                topic_tags=normalized_tags,
                llm=self.llm,
            )
        resolved_memory_type = str(memory_type or "").strip() or str(taxonomy.get("memory_type", "") or "").strip()
        resolved_topic_room = str(topic_room or "").strip() or str(taxonomy.get("topic_room", "") or "").strip()
        record = EpisodicRecord(
            record_id=str(uuid.uuid4()),
            summary=resolved_summary,
            verbatim_excerpt=resolved_verbatim,
            user_text=str(user_text or "").strip(),
            assistant_text=str(assistant_text or "").strip(),
            emotional_intensity=max(0.0, min(1.0, importance)),
            character_emotion=character_emotion,
            relation_impact=relation_impact or {},
            strength=max(0.3, min(0.8, importance)),
            topic_tags=normalized_tags,
            memory_type=resolved_memory_type,
            topic_room=resolved_topic_room,
            scope=str(scope or "").strip(),
            source_session_id=str(source_session_id or "").strip(),
            source_turn_index=max(0, int(source_turn_index or 0)),
        )
        state.episode_records.append(record)
        self._update_relation_state(state.relation_state, relation_impact or {})
        self._try_promote_semantic(state)
        return state

    def _try_promote_semantic(self, state: MemorySystemState) -> None:
        threshold = DEFAULT_CONFIG.memory.semantic_upgrade_threshold
        unpromoted = [record for record in state.episode_records if not record.promoted]
        if len(unpromoted) < threshold:
            return

        batch = unpromoted[:threshold]
        semantic_payload = self._summarize_semantic_batch(batch)
        semantic = SemanticRecord(
            record_id=str(uuid.uuid4()),
            source_episode_ids=[record.record_id for record in batch],
            content=semantic_payload["content"],
            domain=semantic_payload["domain"],
            confidence=semantic_payload["confidence"],
            last_updated_at=datetime.now(),
        )
        state.stable_records.append(semantic)
        for record in batch:
            record.promoted = True

    def _summarize_semantic_batch(self, batch) -> dict:
        fallback = {
            "content": " | ".join(record.display_text() for record in batch),
            "domain": batch[0].memory_type if batch and batch[0].memory_type else "user_relation",
            "confidence": min(0.5, DEFAULT_CONFIG.memory.semantic_confidence_step_cap * len(batch)),
        }
        if self.llm is None or not batch:
            return fallback

        evidence_lines = []
        for idx, record in enumerate(batch, start=1):
            summary = str(record.display_text() or "").strip()
            if not summary:
                continue
            tags = "、".join(list(record.topic_tags or [])[:5]) or "无"
            evidence_lines.append(
                f"{idx}. 事件: {summary}\n"
                f"   标签: {tags}\n"
                f"   类型: {record.memory_type}/{record.topic_room}\n"
                f"   情绪: {record.character_emotion}"
            )
        if not evidence_lines:
            return fallback

        prompt = f"""
你要把几条已经发生过的对话事件，提炼成一条可长期保留的“稳定记忆摘要”。

目标：
1. 这不是简单拼接事件，而是提炼其中稳定、可持续、值得长期保留的部分。
2. 优先保留用户偏好、关系变化、反复出现的话题、明确承诺、持续状态。
3. 不要保留一次性细枝末节，不要重复原句，不要写成长篇流水账。
4. 输出的 `content` 应该是一条自然、紧凑、可长期检索的中文记忆摘要。
5. `domain` 只能是以下之一：`user_profile`、`user_preference`、`relationship`、`ongoing_topic`、`interaction_pattern`。
6. `confidence` 在 0.0 到 1.0 之间，保守一些。
7. 只输出 JSON。

待提炼事件：
{chr(10).join(evidence_lines)}
""".strip()
        try:
            payload = self.llm.generate(
                prompt,
                return_json=True,
                schema=self.SEMANTIC_SUMMARY_SCHEMA,
                temperature=0.0,
                max_tokens=220,
            )
            content = str((payload or {}).get("content", "") or "").strip()
            domain = str((payload or {}).get("domain", "") or "").strip()
            confidence = float((payload or {}).get("confidence", fallback["confidence"]) or fallback["confidence"])
            if not content or not domain:
                return fallback
            return {
                "content": content,
                "domain": domain,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        except Exception:
            return fallback

    def _update_relation_state(self, relation_state: RelationState, impact: dict[str, float]) -> None:
        cap = DEFAULT_CONFIG.memory.relation_delta_cap
        relation_state.trust = min(1.0, max(0.0, relation_state.trust + max(-cap, min(cap, impact.get("trust_delta", 0.0)))))
        relation_state.affection = min(1.0, max(0.0, relation_state.affection + max(-cap, min(cap, impact.get("affection_delta", 0.0)))))
        relation_state.familiarity = min(1.0, max(0.0, relation_state.familiarity + max(-cap, min(cap, impact.get("familiarity_delta", 0.01)))))
        relation_state.last_updated = datetime.now()
        relation_state.stage = self._stage_from_state(relation_state)

    def _stage_from_state(self, relation_state: RelationState) -> str:
        score = (relation_state.trust + relation_state.affection + relation_state.familiarity) / 3
        if score >= 0.8:
            return "intimate"
        if score >= 0.6:
            return "close"
        if score >= 0.4:
            return "friend"
        if score >= 0.2:
            return "acquaintance"
        return "stranger"
