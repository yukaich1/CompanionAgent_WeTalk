from __future__ import annotations

import json
from pathlib import Path


class DerivedStateStore:
    def dump(self, path: str, payload: dict) -> None:
        Path(path).write_text(json.dumps(payload or {}, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> dict:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
