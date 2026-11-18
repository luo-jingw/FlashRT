"""Interfaces the runtime depends on. Environments, executors, planners, latency sources and trigger policies implement
these; the runtime never reaches into an implementation."""
from __future__ import annotations

from typing import Protocol

from records import ActionChunk, Observation, PlannerResult, PlannerSnapshot


class Env(Protocol):
    dt: float                                  # simulated seconds per control step
    max_steps: int

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: list[float]) -> Observation: ...

    def done(self) -> bool: ...

    def success(self) -> bool: ...

    def subtask_progress(self) -> tuple[int, int]:
        """(completed sub-tasks, total sub-tasks) by the benchmark's own judgement."""
        ...

    def task(self) -> str: ...


class Executor(Protocol):
    exec_steps: int                            # actions executed from each chunk before the next executor call

    def act(self, obs: Observation, instruction: str | None) -> ActionChunk: ...


class Planner(Protocol):
    def reset(self, task: str) -> None: ...

    def decide(self, snapshot: PlannerSnapshot) -> PlannerResult: ...


class LatencySource(Protocol):
    name: str                                  # "injected" or "measured"

    def latency_s(self, result: PlannerResult) -> float: ...


class TriggerPolicy(Protocol):
    def reset(self) -> None: ...

    def due(self, t: float, obs: Observation, t_last_effective: float | None) -> bool:
        """Whether a planner call should start now; called only while no call is in flight."""
        ...
