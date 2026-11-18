"""Where the runtime's planner latency L comes from."""
from __future__ import annotations

from records import PlannerResult


class InjectedLatency:
    """A fixed L in simulated seconds, independent of how long the planner really took."""
    name = "injected"

    def __init__(self, latency_s: float) -> None:
        self.value = latency_s

    def latency_s(self, result: PlannerResult) -> float:
        return self.value


class MeasuredLatency:
    """L equal to the planner's measured prefill + decode time (for runs on the target hardware)."""
    name = "measured"

    def latency_s(self, result: PlannerResult) -> float:
        return (result.prefill_ms + result.decode_ms) / 1000.0
