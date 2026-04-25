from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.regression_runner import RegressionRunner
from persistence.session_replay import SessionReplay


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m evaluation.run_regression_report <trace.json> [baseline-trace.json]")
        return 1
    trace_path = argv[1]
    replay = SessionReplay()
    traces = replay.load_trace_file(trace_path)
    summary = replay.summarize_traces(traces)
    report = RegressionRunner().summarize_results(traces)
    payload = {
        "trace_summary": summary,
        "drift_report": replay.detect_drift(traces),
        "regression_report": report,
    }
    if len(argv) >= 3:
        baseline_traces = replay.load_trace_file(argv[2])
        payload["trace_diff"] = replay.diff_trace_sets(baseline_traces, traces)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
