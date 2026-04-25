from __future__ import annotations

from datetime import datetime

from memory.state_models import MemorySystemState


class EpisodicMemoryManager:
    def reinforce_from_recall(self, state: MemorySystemState, recalled_records: list) -> None:
        if not recalled_records:
            return
        recalled_ids = {
            str(getattr(record, "record_id", "") or "").strip()
            for record in list(recalled_records or [])
            if str(getattr(record, "record_id", "") or "").strip()
        }
        if not recalled_ids:
            return
        now = datetime.now()
        for record in state.episode_records:
            if record.record_id not in recalled_ids:
                continue
            record.recall_count += 1
            record.last_recalled_at = now
            record.strength = min(1.0, float(record.strength or 0.5) + 0.04)

    def decay(self, state: MemorySystemState, dt_seconds: float) -> None:
        if dt_seconds <= 0:
            return
        decay_step = min(0.02, max(0.0, dt_seconds / 86400.0 * 0.01))
        for record in state.episode_records:
            baseline = 0.32 if record.promoted else 0.28
            if float(record.strength or 0.0) > baseline:
                record.strength = max(baseline, float(record.strength or 0.0) - decay_step)
