"""One episode of planner + executor on the simulated clock, synchronous or asynchronous."""
from __future__ import annotations

from dataclasses import dataclass

from protocols import Env, Executor, LatencySource, Planner, TriggerPolicy
from records import (ChunkRecord, EpisodeResult, Observation, PlannerCallRecord, PlannerResult, PlannerSnapshot)

PROTOCOLS = ("sync", "async")


@dataclass
class _Pending:
    t_effective: float
    record: PlannerCallRecord


class Runtime:
    """Owns the simulated clock, the in-flight planner call, the current instruction and the chunk queue."""

    def __init__(self, env: Env, executor: Executor, planner: Planner, trigger: TriggerPolicy,
                 latency: LatencySource, protocol: str, max_time_s: float | None = None) -> None:
        """max_time_s: optional limit on simulated time, which counts the time the robot holds still under the
        synchronous protocol; the environment's own step limit counts executed control steps only."""
        if protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
        self.env, self.executor, self.planner = env, executor, planner
        self.trigger, self.latency, self.protocol = trigger, latency, protocol
        self.max_time_s = max_time_s

    def run(self, seed: int) -> EpisodeResult:
        env = self.env
        obs = env.reset(seed)
        self.planner.reset(env.task())
        self.trigger.reset()
        history: list[Observation] = [obs]
        t = 0.0
        steps = 0
        instruction: str | None = None
        unused: list[PlannerCallRecord] = []           # effective decisions not yet read by the executor
        pending: _Pending | None = None
        t_last_effective: float | None = None
        queue: list[list[float]] = []
        calls: list[PlannerCallRecord] = []
        chunks: list[ChunkRecord] = []

        def make_effective(rec: PlannerCallRecord, t_eff: float) -> None:
            nonlocal instruction, t_last_effective
            rec.t_effective = t_eff
            t_last_effective = t_eff
            unused.append(rec)
            if rec.result.instruction is not None:
                instruction = rec.result.instruction   # the executor reads it at the next chunk boundary

        while steps < env.max_steps and not env.done() and (self.max_time_s is None or t < self.max_time_s):
            if pending is None and self.trigger.due(t, obs, t_last_effective):
                t_last_effective = None
                result: PlannerResult = self.planner.decide(PlannerSnapshot(t, env.task(), tuple(history)))
                lat = self.latency.latency_s(result)
                rec = PlannerCallRecord(t, lat, self.latency.name, t + lat, None, result)
                calls.append(rec)
                if self.protocol == "sync":
                    t += lat                           # the robot holds still while the planner runs
                    make_effective(rec, t)
                else:
                    pending = _Pending(t + lat, rec)
            if pending is not None and t + 1e-9 >= pending.t_effective:
                make_effective(pending.record, pending.t_effective)
                pending = None
            if not queue:
                chunk = self.executor.act(obs, instruction)
                queue = list(chunk.actions[: self.executor.exec_steps])
                chunks.append(ChunkRecord(t, instruction, chunk.executor_ms))
                for rec in unused:
                    rec.t_first_use = t
                unused.clear()
            obs = env.step(queue.pop(0))
            history.append(obs)
            t += env.dt
            steps += 1

        done, total = env.subtask_progress()
        return EpisodeResult(seed, self.protocol, env.success(), done, total, steps, t, calls, chunks)
