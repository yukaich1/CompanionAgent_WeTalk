from __future__ import annotations

import json
from pathlib import Path

from persistence.derived_state_store import DerivedStateStore
from persistence.trace_store import TraceStore
from persistence.transcript_store import TranscriptStore


class SessionReplay:
    def __init__(self) -> None:
        self.transcript_store = TranscriptStore()
        self.derived_state_store = DerivedStateStore()
        self.trace_store = TraceStore()

    def load_session_bundle(self, snapshot_path: str) -> dict:
        snapshot = Path(snapshot_path)
        stem = snapshot.with_suffix("")
        transcript_path = str(stem) + ".transcript.json"
        derived_path = str(stem) + ".derived.json"
        trace_path = str(stem) + ".trace.json"
        return {
            "snapshot_path": str(snapshot),
            "transcript": self.transcript_store.load(transcript_path) if Path(transcript_path).exists() else {},
            "derived_state": self.derived_state_store.load(derived_path) if Path(derived_path).exists() else {},
            "traces": self.trace_store.load(trace_path) if Path(trace_path).exists() else [],
        }

    def summarize_bundle(self, bundle: dict) -> dict:
        bundle = bundle if isinstance(bundle, dict) else {}
        transcript = dict(bundle.get("transcript", {}) or {})
        derived = dict(bundle.get("derived_state", {}) or {})
        traces = list(bundle.get("traces", []) or [])
        runtime = dict(derived.get("runtime", {}) or {})
        return {
            "message_count": len(list(transcript.get("messages", []) or [])),
            "system_prompt_present": bool(str(transcript.get("system_prompt", "") or "").strip()),
            "session_id": str(runtime.get("session_id", "") or ""),
            "num_messages": int(runtime.get("num_messages", 0) or 0),
            "has_session_context": bool(dict(derived.get("session_context", {}) or {})),
            "has_working_memory": bool(dict(derived.get("working_memory", {}) or {})),
            "trace_count": len(traces),
        }

    def load_trace_file(self, path: str) -> list[dict]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return list(payload or [])

    def summarize_traces(self, traces: list[dict]) -> dict:
        traces = list(traces or [])
        modes: dict[str, int] = {}
        fallback_count = 0
        tool_turns = 0
        session_only_tool_turns = 0
        long_term_commit_count = 0
        for trace in traces:
            mode = str(trace.get("response_mode", "") or "unknown")
            modes[mode] = modes.get(mode, 0) + 1
            if str(trace.get("fallback_reason", "") or "").strip():
                fallback_count += 1
            if list(trace.get("tool_calls", []) or []):
                tool_turns += 1
            if str(((trace.get("tool_policy", {}) or {}).get("persist_policy", "") or "")).strip() == "session_only":
                session_only_tool_turns += 1
            if bool(((trace.get("memory_commit", {}) or {}).get("persisted_long_term", False))):
                long_term_commit_count += 1
        return {
            "turn_count": len(traces),
            "fallback_count": fallback_count,
            "tool_turn_count": tool_turns,
            "session_only_tool_turns": session_only_tool_turns,
            "long_term_commit_count": long_term_commit_count,
            "mode_distribution": modes,
        }

    def diff_trace_sets(self, before: list[dict], after: list[dict]) -> dict:
        before = list(before or [])
        after = list(after or [])
        before_summary = self.summarize_traces(before)
        after_summary = self.summarize_traces(after)
        before_modes = dict(before_summary.get("mode_distribution", {}) or {})
        after_modes = dict(after_summary.get("mode_distribution", {}) or {})
        mode_keys = sorted(set(before_modes) | set(after_modes))
        return {
            "before_turn_count": before_summary.get("turn_count", 0),
            "after_turn_count": after_summary.get("turn_count", 0),
            "fallback_delta": int(after_summary.get("fallback_count", 0)) - int(before_summary.get("fallback_count", 0)),
            "tool_turn_delta": int(after_summary.get("tool_turn_count", 0)) - int(before_summary.get("tool_turn_count", 0)),
            "mode_delta": {
                key: int(after_modes.get(key, 0)) - int(before_modes.get(key, 0))
                for key in mode_keys
            },
        }

    def detect_drift(self, traces: list[dict]) -> dict:
        traces = list(traces or [])
        persona_drift = 0
        memory_drift = 0
        grounding_drift = 0
        for trace in traces:
            selected_sources = list(trace.get("selected_evidence_sources", []) or [])
            selected_context = dict(trace.get("selected_context_view", {}) or {})
            session_block = dict(selected_context.get("session", {}) or {})
            turn_block = dict(selected_context.get("turn", {}) or {})
            if trace.get("response_mode") == "self_intro" and "l0_identity" not in selected_sources:
                persona_drift += 1
            if trace.get("response_mode") == "external" and "external" not in list(turn_block.get("evidence_sources", []) or []):
                grounding_drift += 1
            if trace.get("response_mode") in {"casual", "value"} and not session_block.get("threads") and not trace.get("selected_memory_layers"):
                memory_drift += 1
        return {
            "trace_count": len(traces),
            "persona_drift_turns": persona_drift,
            "memory_drift_turns": memory_drift,
            "grounding_drift_turns": grounding_drift,
        }
