from __future__ import annotations

from evaluation.regression_cases import RegressionCase


def evaluate_tool_trace(trace: dict, case: RegressionCase) -> dict:
    tool_policy = dict(trace.get("tool_policy", {}) or {})
    memory_commit = dict(trace.get("memory_commit", {}) or {})
    persist_policy = str(tool_policy.get("persist_policy", "") or "")
    return {
        "case_id": case.case_id,
        "passed_mode": str(trace.get("response_mode", "") or "") == str(case.expected_mode or ""),
        "used_session_only_policy": persist_policy == "session_only",
        "blocked_long_term_persist": (
            not bool(memory_commit.get("persisted_long_term", True))
            if memory_commit
            else persist_policy == "session_only"
        ),
    }
