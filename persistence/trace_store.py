from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path


class TraceStore:
    def _json_safe(self, value):
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "as_dict"):
            return self._json_safe(value.as_dict())
        if hasattr(value, "model_dump"):
            return self._json_safe(value.model_dump())
        if hasattr(value, "dict"):
            return self._json_safe(value.dict())
        return str(value)

    def dump(self, path: str, traces: list[dict]) -> None:
        payload = self._json_safe(list(traces or []))
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> list[dict]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return list(payload or [])
