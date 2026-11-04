"""Replay recorded VoLoAgent master calls (record_calls.py) with the ImageWAM executor in one process, at the agent
harness's frequencies: one executor query per 8 env steps at 15 Hz, the calls at their recorded env steps.

  sync      the harness's protocol (VoLoAgent proxy.py:614, :828): at each query point the calls recorded at that step
            run first, then the executor query; one thread. SHARE must be none.
  realtime  the same calls on the wall clock: the executor is released every 8 / 15 s; a planner thread starts each
            call at step / 15 s after the episode start, or when the previous call returns if that is later. SHARE:
            prio   executor stream at the highest priority, planner streams at the default;
            green50 green contexts: planner 50 % of the SMs (88), executor the rest (82);
            coop   cooperative slack scheduling (colocation_qwen27b_5090/coop_gate.py; needs the eager prefill:
                   FLASHRT_QWEN36_TQ_PREFILL_GRAPH=0, FLASHRT_QWEN36_TQ_PREFILL_CHUNK=256).
Episodes (calls files) run back to back in one process; before the first, the executor runs 10 queries and the planner
the first call of the first file, untimed.

Per call: wall and stage times, graphs captured during the call, whether the reply equals the recorded one; per query:
release / start / end and the CUDA-event span. Summary per episode and per cell: model time per episode against the
episode's sim time (sync), executor latency against one control step (66.7 ms) and one chunk (533.3 ms), and the
decision latency of each call (end - its step's time, in ms and env steps; realtime).

usage: python replay_bench.py {sync|realtime} SHARE OUT_JSON CALLS_JSONL [CALLS_JSONL ...]
       python replay_bench.py instruction OUT_JSON CALLS_JSONL [CALLS_JSONL ...]   (executor alone: set_prompt cost)
"""
import collections
import dataclasses
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colocation_5090"))
sys.path.insert(0, os.path.join(HERE, "..", "colocation_qwen27b_5090"))
import volo_calls as vc  # noqa: E402
from executor_imagewam import ImageWamExecutor  # noqa: E402
from planner_replay import CallRecord, PlannerStreams, ReplayPlanner  # noqa: E402

HZ = 15.0
QUERY_S = vc.CHUNK / HZ                  # 0.5333 s between executor queries
STEP_MS = 1e3 / HZ                       # one control step, 66.7 ms
EXEC_INSTRUCTION = "put both the alphabet soup and the tomato sauce in the basket"


def foreign_gpu_pids() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.split()
    return [int(p) for p in out if p.isdigit() and os.path.exists(f"/proc/{p}") and os.stat(f"/proc/{p}").st_uid != os.getuid()]


def pct(x: list[float], q: float) -> float | None:
    return float(np.percentile(x, q)) if len(x) else None


def setup(share: str):
    green, gate, clock = None, None, None
    if share == "green50":
        from green_partition import GreenPartition
        green = GreenPartition(round(torch.cuda.get_device_properties(0).multi_processor_count * 50 / 100))
        exec_stream = green.rest.stream
        streams = PlannerStreams(green.part.stream, green.new_stream(0), green.new_stream(0))
    else:
        exec_stream = torch.cuda.Stream(priority=-1) if share in ("prio", "coop") else torch.cuda.Stream()
        streams = PlannerStreams(*(torch.cuda.Stream(priority=0) for _ in range(3)))
    executor = ImageWamExecutor(exec_stream, EXEC_INSTRUCTION)
    planner = ReplayPlanner(streams)
    if share == "coop":
        from coop_gate import CoopGate, ExecutorClock, install
        clock = ExecutorClock(QUERY_S)
        gate = CoopGate(clock, [streams.main, streams.graph, streams.snap])
        install(planner.fe, gate)
    return executor, planner, green, gate, clock


def run_sync(executor: ImageWamExecutor, planner: ReplayPlanner, rows: list[dict]):
    by_step = collections.defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)
    calls, queries = [], []
    t0 = time.time()
    for k in range(vc.EPISODE_STEPS // vc.CHUNK + 1):
        for r in by_step.get(k * vc.CHUNK, []):
            calls.append(planner.call(r))
        queries.append(executor.query(time.time()))
    return calls, queries, t0, time.time()


def run_realtime(executor: ImageWamExecutor, planner: ReplayPlanner, rows: list[dict], gate, clock):
    calls: list[CallRecord] = []
    t0 = time.time() + 0.5

    def planner_thread() -> None:
        for r in rows:
            due = t0 + r["step"] / HZ
            if time.time() < due:
                time.sleep(due - time.time())
            calls.append(planner.call(r))
            if gate is not None:
                gate.close()

    th = threading.Thread(target=planner_thread, daemon=True)
    th.start()
    queries = []
    n_queries = vc.EPISODE_STEPS // vc.CHUNK + 1
    for k in range(n_queries):
        release = t0 + k * QUERY_S
        if time.time() < release:
            time.sleep(release - time.time())
        if clock is not None:
            clock.tick_start()
        q = executor.query(release)
        queries.append(q)
        if clock is not None:
            clock.tick_end(t0 + (k + 1) * QUERY_S, q.end - q.start)
    if clock is not None:
        clock.stop()  # planner calls still running after the episode's last query run ungated
    th.join()
    return calls, queries, t0, time.time()


def call_dict(c: CallRecord, t0: float) -> dict:
    d = dataclasses.asdict(c)
    d["call_ms"] = c.call_ms
    d["decision_latency_ms"] = (c.t_end - (t0 + c.step / HZ)) * 1e3
    return d


def summarize(calls: list[dict], queries: list[dict], sim_s: float, wall_s: float | None) -> dict:
    lat = [q["latency_ms"] for q in queries]
    by_kind = collections.defaultdict(list)
    for c in calls:
        by_kind[c["kind"]].append(c)
    out = {
        "exec": {"queries": len(lat), "latency_p50_ms": pct(lat, 50), "latency_p99_ms": pct(lat, 99),
                 "latency_max_ms": max(lat) if lat else None,
                 "over_step_pct": 100 * float(np.mean([v > STEP_MS for v in lat])) if lat else None,
                 "over_chunk_pct": 100 * float(np.mean([v > QUERY_S * 1e3 for v in lat])) if lat else None,
                 "gpu_p50_ms": pct([q["gpu_ms"] for q in queries], 50), "sum_s": sum(q["gpu_ms"] for q in queries) / 1e3},
        "calls": {}, "sim_s": sim_s, "wall_s": wall_s,
        "same_reply": sum(c["same_reply"] for c in calls), "n_calls": len(calls),
    }
    for kind, cs in by_kind.items():
        ms = [c["call_ms"] for c in cs]
        warm = [c["call_ms"] for c in cs if c["graphs_captured"] == 0]
        cold = [c["call_ms"] for c in cs if c["graphs_captured"] > 0]
        dl = [c["decision_latency_ms"] for c in cs]
        out["calls"][kind] = {"n": len(cs), "sum_s": sum(ms) / 1e3, "p50_ms": pct(ms, 50), "max_ms": max(ms),
                              "warm_p50_ms": pct(warm, 50), "n_cold": len(cold), "cold_p50_ms": pct(cold, 50),
                              "prefill_p50_ms": pct([c["prefill_ms"] for c in cs], 50),
                              "decode_p50_ms": pct([c["decode_ms"] for c in cs], 50),
                              "vit_p50_ms": pct([c["vit_ms"] for c in cs], 50),
                              "prompt_tokens_p50": pct([c["prompt_tokens"] for c in cs], 50),
                              "output_tokens_p50": pct([c["output_tokens"] for c in cs], 50),
                              "decision_latency_p50_ms": pct(dl, 50), "decision_latency_max_ms": max(dl),
                              "decision_latency_max_steps": max(dl) / STEP_MS}
    out["master_sum_s"] = sum(c["call_ms"] for c in calls) / 1e3
    return out


def main(mode: str, share: str, out: str, files: list[str]) -> None:
    assert (mode == "sync") == (share == "none"), (mode, share)
    foreign_start = foreign_gpu_pids()
    executor, planner, green, gate, clock = setup(share)
    first = [json.loads(line) for line in open(files[0])]
    planner.call(first[0])                          # warm-up, untimed
    if gate is not None:
        gate.enabled = True
    free_GB = torch.cuda.mem_get_info()[0] / 1e9
    episodes = []
    for f in files:
        rows = [json.loads(line) for line in open(f)]
        if mode == "sync":
            calls, queries, t0, t1 = run_sync(executor, planner, rows)
        else:
            calls, queries, t0, t1 = run_realtime(executor, planner, rows, gate, clock)
        cd = [call_dict(c, t0) for c in calls]
        qd = [dict(dataclasses.asdict(q), latency_ms=q.latency_ms) for q in queries]
        s = summarize(cd, qd, vc.EPISODE_STEPS / HZ, t1 - t0)
        episodes.append({"file": f, "calls": cd, "queries": qd, "summary": s})
        e, m = s["exec"], s["calls"].get("monitor", {})
        print(f"{os.path.basename(f)} {mode} {share}: wall {t1 - t0:.1f} s (sim {s['sim_s']:.1f}), master {s['master_sum_s']:.1f} s"
              f" in {s['n_calls']} calls (same reply {s['same_reply']}), exec sum {e['sum_s']:.2f} s, exec lat p50/p99/max "
              f"{e['latency_p50_ms']:.1f}/{e['latency_p99_ms']:.1f}/{e['latency_max_ms']:.1f} ms, >66.7ms {e['over_step_pct']:.1f}%, "
              f"monitor p50 {m.get('p50_ms', 0):.0f} ms (warm {m.get('warm_p50_ms') or 0:.0f}, cold n={m.get('n_cold')}), "
              f"decision lat max {m.get('decision_latency_max_steps', 0):.1f} steps", flush=True)
    all_calls = [c for ep in episodes for c in ep["calls"]]
    all_q = [q for ep in episodes for q in ep["queries"]]
    json.dump({"mode": mode, "share": share, "files": files,
               "prefill_graph": os.environ.get("FLASHRT_QWEN36_TQ_PREFILL_GRAPH", "0"),
               "prefill_chunk": os.environ.get("FLASHRT_QWEN36_TQ_PREFILL_CHUNK"),
               "green_sm": None if green is None else {"planner": green.part.sm_count, "executor": green.rest.sm_count},
               "exec_allocated_GB": executor.allocated_GB, "free_GB": free_GB,
               "foreign_gpu_pids": sorted(set(foreign_start) | set(foreign_gpu_pids())),
               "gate": None if gate is None else {k: dict(v) for k, v in vars(gate.stats).items()},
               "summary": summarize(all_calls, all_q, len(files) * vc.EPISODE_STEPS / HZ, None),
               "episodes": episodes}, open(out, "w"), indent=1)
    os._exit(0)  # skip interpreter teardown: destroying green contexts and graphs at exit aborts the process


def instruction(out: str, files: list[str]) -> None:
    """Executor alone: `set_prompt` cost for the subgoals the recorded decompositions produced (text encoder resident)."""
    from executor_imagewam import instruction_switch_ms
    texts = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == "decompose":
                texts += [t for t in r["subgoals"] if t not in texts]
    res = instruction_switch_ms(texts)
    ms = [c["ms"] for c in res["calls"]]
    print(f"instruction switch: {len(texts)} texts, set_prompt ms first pass {[round(c['ms']) for c in res['calls'][:len(texts)]]}, "
          f"repeat {[round(c['ms']) for c in res['calls'][len(texts):]]}; allocated {res['allocated_with_encoder_GB']:.2f} GB with "
          f"encoder, {res['allocated_slim_GB']:.2f} GB slim", flush=True)
    json.dump(dict(res, files=files, median_ms=float(np.median(ms))), open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "instruction":
        instruction(sys.argv[2], sys.argv[3:])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
