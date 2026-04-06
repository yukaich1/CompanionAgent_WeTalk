from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path


class AISystemStore:
    def __init__(self, system):
        self.system = system

    def _json_safe(self, value):
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "model_dump"):
            return self._json_safe(value.model_dump())
        if hasattr(value, "dict"):
            return self._json_safe(value.dict())
        return str(value)

    def snapshot(self):
        system = self.system
        system._sync_persona_state()
        return self._json_safe(
            {
                "config": system._config_payload(),
                "buffer": {"system_prompt": system.buffer.system_prompt, "messages": list(system.buffer.messages)},
                "runtime": {"num_messages": system.num_messages, "last_message": system.last_message, "thought_visible": bool(getattr(system.thought_system, "show_thoughts", False))},
                "persona_runtime": {
                    "base_template": getattr(system.persona_system, "base_template", {}),
                    "entries": getattr(system.persona_system, "entries", []),
                    "source_records": getattr(system.persona_system, "source_records", {}),
                    "pending_previews": getattr(system.persona_system, "pending_previews", {}),
                    "display_keywords": getattr(system.persona_system, "display_keywords", []),
                    "style_examples": getattr(system.persona_system, "style_examples", []),
                    "character_voice_card": getattr(system.persona_system, "character_voice_card", ""),
                },
            }
        )

    def restore(self, payload):
        system = self.system
        payload = payload if isinstance(payload, dict) else {}
        buffer_payload = payload.get("buffer", {}) if isinstance(payload.get("buffer", {}), dict) else {}
        system.buffer.set_system_prompt(str(buffer_payload.get("system_prompt", "") or system.config.system_prompt))
        system.buffer.flush()
        for message in buffer_payload.get("messages", []) or []:
            if isinstance(message, dict) and message.get("role"):
                system.buffer.add_message(str(message["role"]), message.get("content", ""))

        runtime_payload = payload.get("runtime", {}) if isinstance(payload.get("runtime", {}), dict) else {}
        system.num_messages = int(runtime_payload.get("num_messages", len(system.buffer.messages)) or 0)
        system.last_message = runtime_payload.get("last_message")
        if isinstance(system.last_message, str):
            try:
                system.last_message = datetime.fromisoformat(system.last_message.strip())
            except ValueError:
                system.last_message = None
        system.thought_system.show_thoughts = bool(runtime_payload.get("thought_visible", getattr(system.thought_system, "show_thoughts", False)))

        persona_payload = payload.get("persona_runtime", {}) if isinstance(payload.get("persona_runtime", {}), dict) else {}
        system.persona_system.base_template = dict(persona_payload.get("base_template", system.persona_system._empty_base_template()))
        system.persona_system.entries = list(persona_payload.get("entries", []) or [])
        system.persona_system.source_records = dict(persona_payload.get("source_records", {}) or {})
        system.persona_system.pending_previews = dict(persona_payload.get("pending_previews", {}) or {})
        system.persona_system.display_keywords = list(persona_payload.get("display_keywords", []) or [])
        system.persona_system.style_examples = list(persona_payload.get("style_examples", []) or [])
        system.persona_system.character_voice_card = str(persona_payload.get("character_voice_card", "") or "")
        system.persona_system._repair_state()
        system._sync_persona_state()

    def save(self, path):
        self.system._save_architecture_state()
        Path(path).write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_payload(self, path):
        path_obj = Path(path)
        if not path_obj.exists():
            return None
        try:
            return json.loads(path_obj.read_text(encoding="utf-8"))
        except Exception:
            return None
