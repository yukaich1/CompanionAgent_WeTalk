from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.regression_cases import base_regression_cases
from evaluation.regression_runner import RegressionRunner
from main import AISystem


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def main() -> int:
    ai = AISystem()
    traces: list[dict] = []
    for case in base_regression_cases():
        ai.session_runtime.handle_user_turn(case.user_input)
        trace = getattr(ai, "last_turn_trace", None)
        traces.append(trace.as_dict() if hasattr(trace, "as_dict") else {})
    report = RegressionRunner().summarize_results(traces)
    print(json.dumps(_json_safe({"traces": traces, "report": report}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
