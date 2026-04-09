from __future__ import annotations

from datetime import datetime
from math import exp

from knowledge.knowledge_source import KnowledgeSource, MemoryRecallResult, MemoryRecordView
from memory.memory_taxonomy import infer_recall_filters
from memory.state_models import MemorySystemState
from rag.tool import RAGTool


class MemoryRAGEngine:
    def __init__(self, llm=None):
        self.llm = llm
        self.rag = RAGTool(llm=llm)

    def recall(self, query: str, state: MemorySystemState) -> MemoryRecallResult:
        episodic_entries: list[dict] = []
        episodic_index: dict[str, object] = {}
        for record in state.episode_records:
            if record.perspective != "CHARACTER_FIRST":
                continue
            recall_text = record.recall_text()
            if not recall_text:
                continue
            entry = {
                "chunk_id": record.record_id,
                "document_id": record.record_id,
                "source_label": "memory_episode",
                "content": recall_text,
                "markdown_path": [
                    record.memory_type or "dialogue",
                    record.topic_room or "general",
                    record.character_emotion or "neutral",
                ],
                "keywords": list(record.topic_tags or []),
                "priority": self._priority(record),
                "metadata": {
                    "kind": "memory_episode",
                    "record_id": record.record_id,
                    "memory_type": record.memory_type,
                    "topic_room": record.topic_room,
                    "scope": record.scope,
                    "summary": record.summary,
                    "source_session_id": record.source_session_id,
                },
            }
            episodic_entries.append(entry)
            episodic_index[record.record_id] = record

        stable_records = sorted(state.stable_records, key=lambda record: record.confidence, reverse=True)[:3]
        filters = infer_recall_filters(query, llm=self.llm)
        primary_result = self.rag.search(
            query,
            episodic_entries,
            top_k=5,
            query_type="memory",
            filters=filters,
        )
        if primary_result.hits:
            result = primary_result
        else:
            result = self.rag.search(query, episodic_entries, top_k=5, query_type="memory")

        episodic_views: list[MemoryRecordView] = []
        for hit in result.hits:
            record = episodic_index.get(hit.chunk_id)
            if record is None:
                continue
            inject_mode = "reference_only" if hit.score > 0.82 and len(hit.content) > 100 else "full"
            episodic_views.append(
                MemoryRecordView(
                    record_id=record.record_id,
                    content=record.recall_text(),
                    source=KnowledgeSource.DIALOGUE_MEMORY,
                    inject_mode=inject_mode,
                    metadata={
                        "summary": record.summary,
                        "topic_tags": list(record.topic_tags or []),
                        "memory_type": record.memory_type,
                        "topic_room": record.topic_room,
                        "scope": record.scope,
                        "conflicted": bool(record.conflicted),
                        "retrieval_score": hit.score,
                    },
                )
            )

        return MemoryRecallResult(
            episode_records=episodic_views,
            stable_records=[
                MemoryRecordView(
                    record_id=record.record_id,
                    content=record.content,
                    source=KnowledgeSource.DIALOGUE_MEMORY,
                    metadata={"domain": record.domain, "confidence": record.confidence},
                )
                for record in stable_records
            ],
            relation_state=state.relation_state.model_dump() if hasattr(state.relation_state, "model_dump") else state.relation_state.dict(),
        )

    def _priority(self, record) -> float:
        age_days = max((datetime.now() - record.created_at).total_seconds() / 86400.0, 0.0)
        time_weight = exp(-0.08 * age_days)
        strength_weight = min(1.0, float(getattr(record, "strength", 0.5) or 0.5))
        if record.recall_count > 3:
            strength_weight = min(1.0, strength_weight * 1.2)
        verbatim_bonus = 0.05 if str(getattr(record, "verbatim_excerpt", "") or "").strip() else 0.0
        return 0.55 + 0.25 * time_weight + 0.15 * strength_weight + verbatim_bonus
