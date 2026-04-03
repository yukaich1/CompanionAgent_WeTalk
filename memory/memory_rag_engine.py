from __future__ import annotations

from datetime import datetime
from math import exp

from knowledge.knowledge_source import KnowledgeSource, MemoryRecallResult, MemoryRecordView
from memory.memory_system import EpisodicRecord, MemorySystemState


class MemoryRAGEngine:
    def recall(self, query: str, state: MemorySystemState) -> MemoryRecallResult:
        episodic = [record for record in state.episodic_records if record.perspective == "CHARACTER_FIRST"]
        ranked = sorted(episodic, key=lambda record: self._recall_score(query, record), reverse=True)[:5]
        semantic = sorted(state.semantic_records, key=lambda record: record.confidence, reverse=True)[:3]
        return MemoryRecallResult(
            episodic_records=[
                MemoryRecordView(
                    record_id=record.record_id,
                    content=record.event_summary,
                    source=KnowledgeSource.DIALOGUE_MEMORY,
                    metadata={"topic_tags": record.topic_tags, "conflicted": record.conflicted},
                )
                for record in ranked
            ],
            semantic_records=[
                MemoryRecordView(
                    record_id=record.record_id,
                    content=record.content,
                    source=KnowledgeSource.DIALOGUE_MEMORY,
                    metadata={"domain": record.domain, "confidence": record.confidence},
                )
                for record in semantic
            ],
            relation_state=(
                state.relation_state.model_dump()
                if hasattr(state.relation_state, "model_dump")
                else state.relation_state.dict()
            ),
        )

    def _recall_score(self, query: str, record: EpisodicRecord) -> float:
        keyword_hit = 1.0 if any(token and token in record.event_summary for token in query.split()) else 0.0
        age_days = max((datetime.now() - record.created_at).total_seconds() / 86400.0, 0.0)
        time_weight = exp(-0.08 * age_days)
        strength_bonus = 1.2 if record.recall_count > 3 else 1.0
        return 0.4 * strength_bonus + 0.3 * keyword_hit + 0.3 * time_weight
