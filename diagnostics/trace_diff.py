from __future__ import annotations

from persistence.session_replay import SessionReplay


class TraceDiff:
    def __init__(self) -> None:
        self.replay = SessionReplay()

    def compare_files(self, before_path: str, after_path: str) -> dict:
        before = self.replay.load_trace_file(before_path)
        after = self.replay.load_trace_file(after_path)
        return {
            "summary_diff": self.replay.diff_trace_sets(before, after),
            "before_drift": self.replay.detect_drift(before),
            "after_drift": self.replay.detect_drift(after),
        }
