"""Where in the planner's calls do the executor's missed ticks fall? Reads one colocation_bench.py result JSON.

For every tick whose response latency exceeds the period, prints the tick (release and finish, seconds from the
start of the timed window) and the planner call it overlaps most, with the phase at the tick's release: prefill
(t_start to t_start + prefill_ms; the CUDA-event interval also covers the CPU-side input preparation inside
generate), decode (to t_end), or between calls. Then a per-call summary: how many missed ticks each call has and
the latency of the worst one.

usage: python stall_timeline.py CELL_JSON [CELL_JSON ...]
"""
import json
import os
import sys


def phase_at(t: float, calls: list[dict]) -> tuple[int | None, str, float]:
    for i, c in enumerate(calls):
        pre_end = c["t_start"] + c["prefill_ms"] / 1e3
        if c["t_start"] <= t < pre_end:
            return i, "prefill", t - c["t_start"]
        if pre_end <= t < c["t_end"]:
            return i, "decode", t - pre_end
    return None, "between", 0.0


for path in sys.argv[1:]:
    d = json.load(open(path))
    period_s = d["period_ms"] / 1e3
    calls = d["planner_records"]
    print(f"== {os.path.basename(path)}: {d['ticks']} ticks, {len(calls)} planner calls, period {d['period_ms']:.0f} ms")
    for i, c in enumerate(calls):
        print(f"   call {i}: start {c['t_start']:7.2f} s  prefill {c['prefill_ms']:7.0f} ms  decode {c['decode_ms']:6.0f} ms"
              f"  end {c['t_end']:7.2f} s  tokens {c['prompt_tokens']}")
    per_call = {}
    print("   missed ticks: release s, latency ms, call, phase at release, offset into phase s")
    for (rel, start, fin), lat in zip(d["ticks_wall_s"], d["latency_series_ms"]):
        if lat / 1e3 <= period_s:
            continue
        i, ph, off = phase_at(rel, calls)
        print(f"   {rel:8.3f} {lat:8.1f}  {'-' if i is None else i:>3}  {ph:8s} {off:6.2f}")
        n, worst = per_call.get((i, ph), (0, 0.0))
        per_call[(i, ph)] = (n + 1, max(worst, lat))
    print("   per call and phase: misses, worst latency ms")
    for (i, ph), (n, worst) in sorted(per_call.items(), key=lambda kv: (kv[0][0] is None, kv[0][0] or 0, kv[0][1])):
        print(f"   call {'-' if i is None else i} {ph:8s} {n:4d} {worst:8.1f}")
