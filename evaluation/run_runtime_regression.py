from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.regression_cases import base_regression_cases
from evaluation.regression_runner import RegressionRunner
from evaluation.runtime_probe import RuntimeProbe


def main() -> int:
    probe = RuntimeProbe()
    traces = [probe.inspect_turn(case.user_input) for case in base_regression_cases()]
    report = RegressionRunner().summarize_results(traces)
    print(json.dumps({"traces": traces, "report": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
