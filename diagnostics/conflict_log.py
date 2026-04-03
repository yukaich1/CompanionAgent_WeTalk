from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path


class ConflictLog:
    def __init__(self, path: str | Path = "diagnostics_conflict_log.jsonl", max_entries: int = 1000):
        self.path = Path(path)
        self.max_entries = max_entries

    def record(self, event_type: str, payload: dict) -> None:
        entry = {"event_type": event_type, "timestamp": datetime.now().isoformat(), **payload}
        existing = deque(maxlen=self.max_entries)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    existing.append(line)
        existing.append(json.dumps(entry, ensure_ascii=False))
        self.path.write_text("\n".join(existing) + "\n", encoding="utf-8")
