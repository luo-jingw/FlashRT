"""When the runtime starts a planner call. Every policy starts one at the first step of the episode."""
from __future__ import annotations

from records import Observation


class PeriodicTrigger:
    """A call every `period_s` simulated seconds (HPE monitoring). A call due while another is in flight starts as soon
    as that one becomes effective."""

    def __init__(self, period_s: float) -> None:
        self.period_s = period_s
        self._next_due = 0.0

    def reset(self) -> None:
        self._next_due = 0.0

    def due(self, t: float, obs: Observation, t_last_effective: float | None) -> bool:
        if t + 1e-9 >= self._next_due:
            self._next_due = max(self._next_due + self.period_s, t)
            return True
        return False


class BackToBackTrigger:
    """The next call starts `min_gap_s` after the previous result becomes effective (MemER's high level re-queries as
    soon as it finishes). The gap, typically one executor chunk, lets the robot act between calls under the
    synchronous protocol."""

    def __init__(self, min_gap_s: float) -> None:
        self.min_gap_s = min_gap_s
        self._started = False
        self._t_ready: float | None = None

    def reset(self) -> None:
        self._started = False
        self._t_ready = None

    def due(self, t: float, obs: Observation, t_last_effective: float | None) -> bool:
        if not self._started:
            self._started = True
            return True
        if t_last_effective is not None:
            self._t_ready = t_last_effective + self.min_gap_s
        return self._t_ready is not None and t + 1e-9 >= self._t_ready


class BoundaryTrigger:
    """A call when the environment reports a sub-task boundary (the oracle planner with injected delay)."""

    def __init__(self, event: str = "subtask_done") -> None:
        self.event = event
        self._started = False

    def reset(self) -> None:
        self._started = False

    def due(self, t: float, obs: Observation, t_last_effective: float | None) -> bool:
        if not self._started:
            self._started = True
            return True
        return self.event in obs.events
