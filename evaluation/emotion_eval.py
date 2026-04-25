from __future__ import annotations

from evaluation.regression_cases import RegressionCase


def evaluate_emotion_trace(trace: dict, case: RegressionCase) -> dict:
    return {
        "case_id": case.case_id,
        "passed_mode": str(trace.get("response_mode", "") or "") == str(case.expected_mode or ""),
        "emotion_after": dict(trace.get("emotion_after", {}) or {}),
        "relation_after": dict(trace.get("relation_after", {}) or {}),
    }
