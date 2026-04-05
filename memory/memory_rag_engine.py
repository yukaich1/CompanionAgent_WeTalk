from __future__ import annotations

import re
from datetime import datetime
from math import exp

from knowledge.knowledge_source import KnowledgeSource, MemoryRecallResult, MemoryRecordView
from memory.state_models import EpisodicRecord, MemorySystemState


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

    def _tokenize(self, query: str) -> list[str]:
        text = re.sub(r"\s+", " ", str(query or "")).strip()
        return [token for token in re.findall(r"[\u4e00-\u9fff]{1,8}|[A-Za-z][A-Za-z0-9:_-]*", text) if token]

    def _keyword_hit(self, query: str, record: EpisodicRecord) -> float:
        tokens = self._tokenize(query)
        if not tokens:
            return 0.0
        summary = str(record.event_summary or "")
        topic_tags = list(record.topic_tags or [])
        text_hits = sum(1 for token in tokens if token in summary)
        tag_hits = sum(2 for token in tokens if token in topic_tags)
        return min(1.0, 0.18 * text_hits + 0.24 * tag_hits)

    def _semantic_hit(self, query: str, record: EpisodicRecord) -> float:
        tokens = self._tokenize(query)
        if not tokens:
            return 0.0
        joined_tags = " ".join(record.topic_tags or [])
        emotional = f"{record.character_emotion} {record.event_summary}"
        score = 0.0
        for token in tokens:
            if token in joined_tags:
                score += 0.35
            elif token in emotional:
                score += 0.2
        return min(1.0, score)

    def _recall_score(self, query: str, record: EpisodicRecord) -> float:
        semantic_sim = self._semantic_hit(query, record)
        keyword_hit = self._keyword_hit(query, record)
        age_days = max((datetime.now() - record.created_at).total_seconds() / 86400.0, 0.0)
        time_weight = exp(-0.08 * age_days)
        if getattr(record, "importance", "").upper() == "HIGH":
            time_weight = 1.0
        strength_weight = min(1.0, float(getattr(record, "strength", 0.5) or 0.5))
        if record.recall_count > 3:
            strength_weight = min(1.0, strength_weight * 1.2)
        return 0.4 * semantic_sim + 0.3 * keyword_hit + 0.2 * time_weight + 0.1 * strength_weight
