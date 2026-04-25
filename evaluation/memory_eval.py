from __future__ import annotations

from evaluation.regression_cases import RegressionCase


def evaluate_memory_trace(trace: dict, case: RegressionCase) -> dict:
    sources = list(trace.get("selected_evidence_sources", []) or [])
    if not sources:
        sources = list(
            (((trace.get("selected_context_view", {}) or {}).get("turn", {}) or {}).get("evidence_sources", []) or []
        )
        )
    evidence_kind = str(((trace.get("response_plan", {}) or {}).get("evidence_kind", "") or "")).strip()
    return {
        "case_id": case.case_id,
        "passed_mode": str(trace.get("response_mode", "") or "") == str(case.expected_mode or ""),
        "used_memory_source": evidence_kind == "memory" or any(source in {"memory", "working_memory", "session_context"} for source in sources),
        "memory_layers": list(trace.get("selected_memory_layers", []) or []),
    }
