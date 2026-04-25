from __future__ import annotations

import json
from pathlib import Path


class TurnLogger:
    def dump_latest(self, path: str, trace: dict, *, session_context: dict | None = None, health: dict | None = None) -> None:
        payload = {
            "trace": trace if isinstance(trace, dict) else {},
            "session_context": session_context if isinstance(session_context, dict) else {},
            "health": health if isinstance(health, dict) else {},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
