from __future__ import annotations

from evaluation.emotion_eval import evaluate_emotion_trace
from evaluation.grounding_eval import evaluate_grounding_trace
from evaluation.memory_eval import evaluate_memory_trace
from evaluation.persona_eval import evaluate_persona_trace
from evaluation.regression_cases import RegressionCase, base_regression_cases
from evaluation.tool_eval import evaluate_tool_trace


class RegressionRunner:
    def __init__(self, cases: list[RegressionCase] | None = None) -> None:
        self.cases = list(cases or base_regression_cases())

    def evaluate_trace(self, trace: dict) -> list[dict]:
        results: list[dict] = []
        for case in self._matching_cases(trace):
            base = {"category": case.category, "expected_mode": case.expected_mode, "actual_mode": str(trace.get("response_mode", "") or "")}
            if case.category == "persona":
                results.append({**base, **evaluate_persona_trace(trace, case)})
            elif case.category == "memory":
                results.append({**base, **evaluate_memory_trace(trace, case)})
            elif case.category == "emotion":
                results.append({**base, **evaluate_emotion_trace(trace, case)})
            elif case.category == "grounding":
                results.append({**base, **evaluate_grounding_trace(trace, case)})
            elif case.category == "tool":
                results.append({**base, **evaluate_tool_trace(trace, case)})
        return results

    def summarize_results(self, traces: list[dict]) -> dict:
        evaluated = [result for trace in list(traces or []) for result in self.evaluate_trace(trace)]
        passed = 0
        total = 0
        for item in evaluated:
            checks = [value for key, value in item.items() if key.startswith("passed_") or key.endswith("_hit") or key.startswith("used_")]
            total += len(checks)
            passed += sum(1 for value in checks if bool(value))
        return {
            "trace_count": len(list(traces or [])),
            "evaluated_case_results": len(evaluated),
            "passed_checks": passed,
            "total_checks": total,
            "pass_rate": (passed / total) if total else 0.0,
            "failing_results": [item for item in evaluated if not all(bool(value) for key, value in item.items() if key.startswith("passed_") or key.endswith("_hit") or key.startswith("used_"))],
            "results": evaluated,
        }

    def _matching_cases(self, trace: dict) -> list[RegressionCase]:
        response_mode = str(trace.get("response_mode", "") or "")
        return [
            case
            for case in self.cases
            if not case.expected_mode or case.expected_mode == response_mode
        ]
