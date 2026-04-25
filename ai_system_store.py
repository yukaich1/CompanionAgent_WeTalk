from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path

from runtime.turn_trace import TurnTrace


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
                "runtime_trace": {
                    "last_turn_trace": getattr(getattr(system, "last_turn_trace", None), "as_dict", lambda: None)(),
                    "turn_traces": [
                        trace.as_dict() if hasattr(trace, "as_dict") else self._json_safe(trace)
                        for trace in list(getattr(system, "turn_traces", []) or [])
                    ],
                    "session_context": getattr(system.session_context_manager, "build_session_context", lambda: {})(),
                    "working_memory": getattr(system.memory_system, "get_working_memory_snapshot", lambda: {})(),
                },
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

        runtime_trace_payload = payload.get("runtime_trace", {}) if isinstance(payload.get("runtime_trace", {}), dict) else {}
        last_turn_trace_payload = runtime_trace_payload.get("last_turn_trace")
        system.last_turn_trace = self._restore_turn_trace(last_turn_trace_payload)
        system.turn_traces.clear()
        for trace_payload in list(runtime_trace_payload.get("turn_traces", []) or []):
            restored = self._restore_turn_trace(trace_payload)
            if restored is not None:
                system.turn_traces.append(restored)
        session_context_payload = runtime_trace_payload.get("session_context", {}) if isinstance(runtime_trace_payload.get("session_context", {}), dict) else {}
        system.session_context_manager.restore(session_context_payload)
        working_memory_payload = runtime_trace_payload.get("working_memory", {}) if isinstance(runtime_trace_payload.get("working_memory", {}), dict) else {}
        for thread in list(working_memory_payload.get("active_threads", []) or []):
            if not isinstance(thread, dict):
                continue
            system.memory_system.update_working_memory(
                turn_index=int(thread.get("last_updated_turn", system.num_messages) or system.num_messages),
                topic_hint=str(thread.get("topic", "") or ""),
                summary=str(thread.get("summary", "") or ""),
                relation_summary=str(working_memory_payload.get("recent_relation_summary", "") or ""),
                emotion_summary=str(working_memory_payload.get("recent_emotion_summary", "") or ""),
                tool_summary=str(working_memory_payload.get("recent_tool_summary", "") or ""),
                pinned_facts=list(working_memory_payload.get("pinned_facts", []) or []),
            )

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
        snapshot = self.snapshot()
        path_obj = Path(path)
        path_obj.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        stem = path_obj.with_suffix("")
        self.system.transcript_store.dump(
            str(stem) + ".transcript.json",
            list(self.system.buffer.messages),
            system_prompt=self.system.buffer.system_prompt,
        )
        self.system.derived_state_store.dump(
            str(stem) + ".derived.json",
            {
                "session_context": getattr(self.system.session_context_manager, "build_session_context", lambda: {})(),
                "working_memory": getattr(self.system.memory_system, "get_working_memory_snapshot", lambda: {})(),
                "runtime": {
                    "session_id": getattr(self.system, "session_id", ""),
                    "num_messages": getattr(self.system, "num_messages", 0),
                    "last_message": self._json_safe(getattr(self.system, "last_message", None)),
                },
            },
        )
        self.system.trace_store.dump(
            str(stem) + ".trace.json",
            [
                trace.as_dict() if hasattr(trace, "as_dict") else self._json_safe(trace)
                for trace in list(getattr(self.system, "turn_traces", []) or [])
            ],
        )

    def load_payload(self, path):
        path_obj = Path(path)
        if not path_obj.exists():
            return None
        try:
            return json.loads(path_obj.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _restore_turn_trace(self, payload):
        if not isinstance(payload, dict):
            return None
        created_at = payload.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif created_at is None:
            created_at = datetime.now()
        return TurnTrace(
            turn_id=str(payload.get("turn_id", "") or ""),
            session_id=str(payload.get("session_id", "") or ""),
            input_text=str(payload.get("input_text", "") or ""),
            normalized_text=str(payload.get("normalized_text", "") or ""),
            created_at=created_at,
            route_type=str(payload.get("route_type", "") or ""),
            response_mode=str(payload.get("response_mode", "") or ""),
            persona_focus=str(payload.get("persona_focus", "") or ""),
            selected_evidence_sources=list(payload.get("selected_evidence_sources", []) or []),
            selected_memory_layers=list(payload.get("selected_memory_layers", []) or []),
            tool_calls=list(payload.get("tool_calls", []) or []),
            tool_policy=dict(payload.get("tool_policy", {}) or {}),
            recall_query=str(payload.get("recall_query", "") or ""),
            stage_summary=dict(payload.get("stage_summary", {}) or {}),
            evidence_gate=dict(payload.get("evidence_gate", {}) or {}),
            emotion_before=dict(payload.get("emotion_before", {}) or {}),
            emotion_after=dict(payload.get("emotion_after", {}) or {}),
            relation_before=dict(payload.get("relation_before", {}) or {}),
            relation_after=dict(payload.get("relation_after", {}) or {}),
            session_context=dict(payload.get("session_context", {}) or {}),
            working_memory=dict(payload.get("working_memory", {}) or {}),
            selected_context_view=dict(payload.get("selected_context_view", {}) or {}),
            persistence_boundary=dict(payload.get("persistence_boundary", {}) or {}),
            memory_commit=dict(payload.get("memory_commit", {}) or {}),
            planner=dict(payload.get("planner", {}) or {}),
            fallback_reason=str(payload.get("fallback_reason", "") or ""),
            final_response=str(payload.get("final_response", "") or ""),
            latency_ms=int(payload.get("latency_ms", 0) or 0),
        )
