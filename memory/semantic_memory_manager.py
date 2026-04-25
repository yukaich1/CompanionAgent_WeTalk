from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from memory.state_models import MemorySystemState, SemanticRecord


class SemanticMemoryManager:
    def merge_semantic_records(self, state: MemorySystemState, *, max_records: int = 18) -> None:
        stable_records = list(state.stable_records or [])
        if len(stable_records) <= max_records:
            return

        grouped: dict[str, list[SemanticRecord]] = defaultdict(list)
        for record in stable_records:
            grouped[str(record.domain or "general").strip() or "general"].append(record)

        merged_records: list[SemanticRecord] = []
        for domain, records in grouped.items():
            ordered = sorted(records, key=lambda item: item.last_updated_at, reverse=True)
            if len(ordered) <= 2:
                merged_records.extend(ordered)
                continue

            head = ordered[0]
            merged_content_parts: list[str] = []
            source_ids: list[str] = []
            for record in ordered[:3]:
                content = str(record.content or "").strip()
                if not content or content in merged_content_parts:
                    continue
                merged_content_parts.append(content)
                source_ids.extend(list(record.source_episode_ids or []))

            merged_records.append(
                SemanticRecord(
                    record_id=head.record_id,
                    source_episode_ids=list(dict.fromkeys(source_ids)),
                    content="；".join(merged_content_parts),
                    domain=domain,
                    confidence=max(float(record.confidence or 0.0) for record in ordered[:3]),
                    scope=head.scope,
                    created_at=head.created_at,
                    last_updated_at=datetime.now(),
                )
            )
            merged_records.extend(ordered[3:])

        state.stable_records = sorted(
            merged_records,
            key=lambda item: (float(item.confidence or 0.0), item.last_updated_at),
            reverse=True,
        )[:max_records]
        self.rewrite_relationship_summary(state)

    def rewrite_relationship_summary(self, state: MemorySystemState) -> None:
        relationship_records = [
            record
            for record in list(state.stable_records or [])
            if str(record.domain or "").strip() == "relationship" and str(record.content or "").strip()
        ]
        if len(relationship_records) < 2:
            return

        ordered = sorted(relationship_records, key=lambda item: item.last_updated_at, reverse=True)
        head = ordered[0]
        merged_content = "；".join(
            dict.fromkeys(
                str(record.content or "").strip()
                for record in ordered[:3]
                if str(record.content or "").strip()
            )
        ).strip()
        if not merged_content:
            return

        head.content = merged_content[:320].rstrip("；")
        head.confidence = max(float(record.confidence or 0.0) for record in ordered[:3])
        head.last_updated_at = datetime.now()
        keep_ids = {head.record_id}
        state.stable_records = [
            record
            for record in list(state.stable_records or [])
            if str(record.domain or "").strip() != "relationship" or record.record_id in keep_ids
        ]
