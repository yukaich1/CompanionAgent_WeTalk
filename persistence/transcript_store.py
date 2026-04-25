from __future__ import annotations

import json
from pathlib import Path


class TranscriptStore:
    def dump(self, path: str, messages: list[dict], *, system_prompt: str = "") -> None:
        payload = {
            "system_prompt": str(system_prompt or "").strip(),
            "messages": list(messages or []),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> dict:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
