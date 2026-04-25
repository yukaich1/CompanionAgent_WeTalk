from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persistence.session_replay import SessionReplay


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m evaluation.run_session_diagnostics <snapshot.json>")
        return 1
    replay = SessionReplay()
    bundle = replay.load_session_bundle(argv[1])
    traces = list(bundle.get("traces", []) or [])
    print(
        json.dumps(
            {
                "bundle": bundle,
                "bundle_summary": replay.summarize_bundle(bundle),
                "trace_summary": replay.summarize_traces(traces),
                "drift_report": replay.detect_drift(traces),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
