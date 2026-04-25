from __future__ import annotations

from evaluation.regression_cases import RegressionCase


def evaluate_persona_trace(trace: dict, case: RegressionCase) -> dict:
    return {
        "case_id": case.case_id,
        "passed_mode": str(trace.get("response_mode", "") or "") == str(case.expected_mode or ""),
        "selected_evidence": list(trace.get("selected_evidence_sources", []) or []),
    }
