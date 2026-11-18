"""Observational checks of the runtime on the toy world.

1. With L = 0 the synchronous and asynchronous protocols give identical episodes (same steps, same outcome).
2. SR, sub-task completion and simulated time against injected L, for the boundary-triggered oracle planner (C4's
   design) and the back-to-back trigger (MemER's cadence), under both protocols.
3. Logged L (first use - snapshot) is at least the injected L and at most injected L plus one chunk.

usage: python check_runtime.py
"""
from __future__ import annotations

from latency import InjectedLatency
from runtime import Runtime
from toy_world import OraclePlanner, ToyEnv, ToyExecutor
from triggers import BackToBackTrigger, BoundaryTrigger

SEEDS = range(40)
LATENCIES = [0.0, 0.5, 1.0, 2.0, 4.0]


def make(trigger_name: str, protocol: str, latency_s: float) -> Runtime:
    env = ToyEnv()
    ex = ToyExecutor(env.names)
    chunk_s = ex.exec_steps * env.dt
    trig = BoundaryTrigger() if trigger_name == "boundary" else BackToBackTrigger(min_gap_s=chunk_s)
    return Runtime(env, ex, OraclePlanner(), trig, InjectedLatency(latency_s), protocol)


def main() -> None:
    # 1. L = 0: sync and async agree
    for trig in ("boundary", "back_to_back"):
        a = [make(trig, "sync", 0.0).run(s) for s in SEEDS]
        b = [make(trig, "async", 0.0).run(s) for s in SEEDS]
        same = sum(x.steps == y.steps and x.success == y.success for x, y in zip(a, b))
        print(f"check 1  {trig:<13} L=0 sync vs async identical episodes: {same}/{len(a)}")

    # 2 and 3
    print(f"\n{'trigger':<13} {'protocol':<6} {'L s':>4}  {'SR':>5}  {'subtasks':>8}  {'sim s':>6}  {'calls':>5}"
          f"  {'logged L min / max':>18}")
    chunk_s = ToyExecutor(ToyEnv().names).exec_steps * ToyEnv().dt
    bad = 0
    for trig in ("boundary", "back_to_back"):
        for proto in ("sync", "async"):
            for L in LATENCIES:
                res = [make(trig, proto, L).run(s) for s in SEEDS]
                sr = sum(r.success for r in res) / len(res)
                sub = sum(r.subtasks_done / r.subtasks_total for r in res) / len(res)
                simt = sum(r.sim_time_s for r in res) / len(res)
                calls = sum(len(r.calls) for r in res) / len(res)
                logged = [c.t_first_use - c.t_snapshot for r in res for c in r.calls if c.t_first_use is not None]
                lo, hi = min(logged), max(logged)
                bad += sum(not (L - 1e-6 <= x <= L + chunk_s + 1e-6) for x in logged)
                print(f"{trig:<13} {proto:<6} {L:>4.1f}  {sr:>5.2f}  {sub:>8.2f}  {simt:>6.1f}  {calls:>5.1f}"
                      f"  {lo:>8.2f} / {hi:<8.2f}")
    print(f"\ncheck 3  logged L outside [L, L + one chunk ({chunk_s:.2f} s)]: {bad}")


if __name__ == "__main__":
    main()
