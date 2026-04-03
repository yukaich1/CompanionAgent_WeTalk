from __future__ import annotations

from collections import deque


class HealthMonitor:
    def __init__(self):
        self.coverage_scores = deque(maxlen=50)
        self.tool_latencies = deque(maxlen=100)

    def record_turn_metrics(self, coverage_score: float | None = None, tool_latency: float | None = None) -> None:
        if coverage_score is not None:
            self.coverage_scores.append(coverage_score)
        if tool_latency is not None:
            self.tool_latencies.append(tool_latency)

    def snapshot(self) -> dict:
        return {
            "coverage_scores": list(self.coverage_scores),
            "tool_latencies": list(self.tool_latencies),
        }
