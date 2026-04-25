from __future__ import annotations

from memory.state_models import MemorySystemState


class MemoryCompactor:
    def compact_episode_window(self, state: MemorySystemState, *, max_records: int = 120, keep_recent: int = 72) -> None:
        records = list(state.episode_records or [])
        if len(records) <= max_records:
            return

        ordered = sorted(records, key=lambda item: item.created_at, reverse=True)
        survivors = ordered[:keep_recent]
        remaining = ordered[keep_recent:]
        prioritized = sorted(
            remaining,
            key=lambda item: (
                bool(item.promoted),
                float(item.strength or 0.0),
                int(item.recall_count or 0),
            ),
            reverse=True,
        )
        survivors.extend(prioritized[: max(0, max_records - keep_recent)])
        survivor_ids = {record.record_id for record in survivors}
        state.episode_records = [
            record
            for record in sorted(records, key=lambda item: item.created_at)
            if record.record_id in survivor_ids
        ]
