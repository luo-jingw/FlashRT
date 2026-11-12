"""Align one co-location cell: executor ticks (colocation_bench.py JSON) against the planner process's calls
(planner_process.py JSONL), on the wall clock.

Planner phases per call, from its start time and the synchronized stage times: vit [start, +vit_ms], restore
[+restore_ms], prefill [+prefill_ms], decode [prefill end, end]; the capsule build of the fast arm is one phase
"capsule" (its line has monotonic times only; they are moved to the wall clock by the call lines' constant offset).
Each executor tick (release -> finish) is assigned the phase it overlaps longest, or "none". Only ticks inside the
planner's timed pass (first call due -> last call end) are counted; ticks during the planner's load and warm-up are
left out.

Prints and writes (OUT_JSON):
  executor latency (finish - release) per phase group: ticks, p50 / p99 / max ms, ticks over each threshold;
  planner L_decision and start - due per call kind and wrist image, with the same numbers from an "alone" JSONL
  of the same arm beside them when given (its calls of the same episodes).

usage: python colocation_align.py EXEC_JSON PLANNER_JSONL OUT_JSON [--alone ALONE_JSONL] [--thresholds 200,533]
"""
import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

PHASES = ("vit", "restore", "prefill", "decode", "capsule")


@dataclass
class Interval:
    phase: str
    start: float
    end: float


@dataclass
class TickGroup:
    phase: str
    ticks: int
    p50_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    over: dict[str, int]


@dataclass
class CallGroup:
    kind: str
    wrist: str
    calls: int
    l_decision_med_s: float
    l_decision_max_s: float
    start_delay_med_s: float
    start_delay_max_s: float
    alone_l_decision_med_s: float | None
    alone_l_decision_max_s: float | None


@dataclass
class CellSummary:
    exec_json: str
    planner_jsonl: str
    period_ms: float
    window_s: float
    exec_window_covers_planner: bool
    ticks_in_window: int
    groups: list[TickGroup]
    planner: list[CallGroup]


def read_jsonl(path: str) -> list[dict[str, object]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def planner_intervals(lines: list[dict[str, object]]) -> tuple[list[Interval], float, float]:
    """Phase intervals on the wall clock, and the timed pass's [first due, last end]."""
    calls = [c for c in lines if c["kind"] != "capsule"]
    offset = float(np.median([float(c["start_wall"]) - float(c["start_s"]) for c in calls]))
    out: list[Interval] = []
    for c in calls:
        t = float(c["start_wall"])
        for phase, key in (("vit", "vit_ms"), ("restore", "restore_ms"), ("prefill", "prefill_ms")):
            d = float(c[key]) / 1e3
            if d > 0:
                out.append(Interval(phase, t, t + d))
            t += d
        out.append(Interval("decode", t, float(c["end_wall"])))
    for c in lines:
        if c["kind"] == "capsule":
            out.append(Interval("capsule", float(c["start_s"]) + offset, float(c["end_s"]) + offset))
    first_due = min(float(c["start_wall"]) - (float(c["start_s"]) - float(c["due_s"])) for c in calls)
    last_end = max(float(c["end_wall"]) for c in calls)
    return out, first_due, last_end


def assign(release: float, finish: float, intervals: list[Interval]) -> str:
    best, best_overlap = "none", 0.0
    for iv in intervals:
        ov = min(finish, iv.end) - max(release, iv.start)
        if ov > best_overlap:
            best, best_overlap = iv.phase, ov
    return best


def tick_groups(lat_by_phase: dict[str, list[float]], thresholds: list[float]) -> list[TickGroup]:
    groups = []
    for phase in ("none",) + PHASES + ("all",):
        v = np.array(lat_by_phase.get(phase, []))
        groups.append(TickGroup(phase, len(v),
                                float(np.percentile(v, 50)) if len(v) else None,
                                float(np.percentile(v, 99)) if len(v) else None,
                                float(v.max()) if len(v) else None,
                                {f"{t:g}": int((v > t).sum()) for t in thresholds}))
    return groups


def wrist_of(episode: str) -> str:
    return episode.split("_")[-1]


def call_stats(lines: list[dict[str, object]]) -> dict[tuple[str, str], tuple[list[float], list[float]]]:
    out: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for c in lines:
        if c["kind"] == "capsule":
            continue
        key = (str(c["kind"]), wrist_of(str(c["episode"])))
        ld, sd = out.setdefault(key, ([], []))
        ld.append(float(c["decision_s"]) - float(c["start_s"]))
        sd.append(float(c["start_s"]) - float(c["due_s"]))
    return out


def call_groups(lines: list[dict[str, object]], alone: list[dict[str, object]] | None) -> list[CallGroup]:
    episodes = {c["episode"] for c in lines}
    cur = call_stats(lines)
    ref = call_stats([c for c in alone if c["episode"] in episodes]) if alone is not None else {}
    groups = []
    for (kind, wrist), (ld, sd) in sorted(cur.items()):
        r = ref.get((kind, wrist))
        groups.append(CallGroup(kind, wrist, len(ld), float(np.median(ld)), float(max(ld)), float(np.median(sd)),
                                float(max(sd)), float(np.median(r[0])) if r else None, float(max(r[0])) if r else None))
    return groups


def summarize(exec_json: str, planner_jsonl: str, alone_jsonl: str | None, thresholds: list[float]) -> CellSummary:
    ex = json.load(open(exec_json))
    lines = read_jsonl(planner_jsonl)
    intervals, w0, w1 = planner_intervals(lines)
    t0 = float(ex["t0_wall"])
    lat_by_phase: dict[str, list[float]] = {}
    for release, _start, finish in ex["ticks_wall_s"]:
        r, f = t0 + release, t0 + finish
        if f < w0 or r > w1:
            continue
        lat = (f - r) * 1e3
        lat_by_phase.setdefault(assign(r, f, intervals), []).append(lat)
        lat_by_phase.setdefault("all", []).append(lat)
    exec_end = t0 + ex["ticks_wall_s"][-1][2]
    alone = read_jsonl(alone_jsonl) if alone_jsonl else None
    return CellSummary(exec_json, planner_jsonl, float(ex["period_ms"]), w1 - w0, t0 <= w0 and exec_end >= w1,
                       len(lat_by_phase.get("all", [])), tick_groups(lat_by_phase, thresholds),
                       call_groups(lines, alone))


def fmt(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def print_summary(s: CellSummary) -> None:
    print(f"{s.exec_json}\n  planner timed pass {s.window_s:.1f} s, executor window covers it: "
          f"{s.exec_window_covers_planner}, ticks in it: {s.ticks_in_window}, period {s.period_ms:g} ms")
    keys = list(s.groups[0].over)
    print("  phase    ticks   p50 ms   p99 ms   max ms   " + "   ".join(f">{k} ms" for k in keys))
    for g in s.groups:
        print(f"  {g.phase:8s} {g.ticks:5d} {fmt(g.p50_ms):>8s} {fmt(g.p99_ms):>8s} {fmt(g.max_ms):>8s}   "
              + "   ".join(f"{g.over[k]:>{len(k) + 4}d}" for k in keys))
    print("  kind       wrist     calls  L_dec med / max s  (alone)        start-due med / max s")
    for c in s.planner:
        print(f"  {c.kind:10s} {c.wrist:9s} {c.calls:5d}  {c.l_decision_med_s:6.2f} / {c.l_decision_max_s:5.2f}"
              f"  ({fmt(c.alone_l_decision_med_s, 2)} / {fmt(c.alone_l_decision_max_s, 2)})"
              f"   {c.start_delay_med_s:6.2f} / {c.start_delay_max_s:5.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exec_json")
    ap.add_argument("planner_jsonl")
    ap.add_argument("out_json")
    ap.add_argument("--alone", default=None)
    ap.add_argument("--thresholds", default="200,533")
    a = ap.parse_args()
    s = summarize(a.exec_json, a.planner_jsonl, a.alone, [float(x) for x in a.thresholds.split(",")])
    print_summary(s)
    with open(a.out_json, "w") as f:
        json.dump(asdict(s), f, indent=1)


if __name__ == "__main__":
    main()
