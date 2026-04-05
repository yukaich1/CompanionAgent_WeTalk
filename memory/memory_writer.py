from __future__ import annotations

import uuid
from datetime import datetime

from config import DEFAULT_CONFIG
from diagnostics.conflict_log import ConflictLog
from memory.state_models import EpisodicRecord, MemorySystemState, RelationState, SemanticRecord


class MemoryWriter:
    def __init__(self, conflict_log: ConflictLog | None = None):
        self.conflict_log = conflict_log or ConflictLog()

    def remember(
        self,
        state: MemorySystemState,
        event_summary: str,
        topic_tags: list[str] | None = None,
        relation_impact: dict[str, float] | None = None,
        importance: float = 0.5,
        character_emotion: str = "neutral",
    ) -> MemorySystemState:
        record = EpisodicRecord(
            record_id=str(uuid.uuid4()),
            event_summary=event_summary,
            emotional_intensity=max(0.0, min(1.0, importance)),
            character_emotion=character_emotion,
            relation_impact=relation_impact or {},
            strength=max(0.3, min(0.8, importance)),
            topic_tags=topic_tags or [],
        )
        state.episodic_records.append(record)
        self._update_relation_state(state.relation_state, relation_impact or {})
        self._try_promote_semantic(state)
        return state

    def _try_promote_semantic(self, state: MemorySystemState) -> None:
        threshold = DEFAULT_CONFIG.memory.semantic_upgrade_threshold
        unpromoted = [record for record in state.episodic_records if not record.promoted]
        if len(unpromoted) < threshold:
            return

        batch = unpromoted[:threshold]
        semantic = SemanticRecord(
            record_id=str(uuid.uuid4()),
            source_episode_ids=[record.record_id for record in batch],
            content=" | ".join(record.event_summary for record in batch),
            domain=batch[0].topic_tags[0] if batch[0].topic_tags else "user_relation",
            confidence=min(0.5, DEFAULT_CONFIG.memory.semantic_confidence_step_cap * len(batch)),
            last_updated_at=datetime.now(),
        )
        state.semantic_records.append(semantic)
        for record in batch:
            record.promoted = True

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
