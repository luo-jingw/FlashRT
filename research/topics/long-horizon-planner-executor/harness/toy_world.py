"""A toy world for checking the runtime: a fixed sequence of sub-tasks, an executor that only makes progress on the
sub-task it is told to do, and an oracle planner that names the current sub-task from ground truth.

Working on a sub-task that is already finished is harmful: each such step breaks the episode with probability
`stale_break_p`, so an async planner whose decision arrives late can cause failure, and a sync planner costs time."""
from __future__ import annotations

import random

from records import ActionChunk, Observation, PlannerResult, PlannerSnapshot


class ToyEnv:
    def __init__(self, n_subtasks: int = 6, steps_per_subtask: int = 40, dt: float = 0.05, max_steps: int = 1200,
                 stale_break_p: float = 0.01) -> None:
        self.n, self.k, self.dt, self.max_steps, self.p = n_subtasks, steps_per_subtask, dt, max_steps, stale_break_p
        self.names = [f"subtask {i}" for i in range(n_subtasks)]

    def reset(self, seed: int) -> Observation:
        self._rng = random.Random(seed)
        self._idx, self._progress, self._broken, self._t = 0, 0, False, 0.0
        return self._obs(())

    def _obs(self, events: tuple[str, ...]) -> Observation:
        return Observation(self._t, {}, [float(self._idx), float(self._progress)], events,
                           {"current": self.names[self._idx] if self._idx < self.n else None})

    def step(self, action: list[float]) -> Observation:
        self._t += self.dt
        target = int(action[0])                # the sub-task index the executor worked on; -1 = no-op
        events: tuple[str, ...] = ()
        if target == self._idx:
            self._progress += 1
            if self._progress >= self.k:
                self._idx, self._progress, events = self._idx + 1, 0, ("subtask_done",)
        elif 0 <= target < self._idx and self._rng.random() < self.p:
            self._broken = True
        return self._obs(events)

    def done(self) -> bool:
        return self._broken or self._idx >= self.n

    def success(self) -> bool:
        return not self._broken and self._idx >= self.n

    def subtask_progress(self) -> tuple[int, int]:
        return self._idx, self.n

    def task(self) -> str:
        return "do " + ", then ".join(self.names)


class ToyExecutor:
    def __init__(self, names: list[str], chunk_len: int = 15, exec_steps: int = 8) -> None:
        self.index = {n: i for i, n in enumerate(names)}
        self.chunk_len, self.exec_steps = chunk_len, exec_steps

    def act(self, obs: Observation, instruction: str | None) -> ActionChunk:
        target = self.index.get(instruction, -1) if instruction is not None else -1
        return ActionChunk([[float(target)]] * self.chunk_len, instruction, 0.0)


class OraclePlanner:
    """Names the sub-task that is current at the snapshot time (from environment ground truth)."""

    def reset(self, task: str) -> None:
        pass

    def decide(self, snapshot: PlannerSnapshot) -> PlannerResult:
        current = snapshot.history[-1].info["current"]
        return PlannerResult(current, "subtask", current or "", True)
