"""Executor latency split by whether a tick overlaps a planner call, from colocation_bench.py result JSONs.

A tick [start, finish] overlaps a planner call [t_start, t_end] when the intervals intersect; an overlapping tick is
further labelled by the planner phase it overlaps longest (prefill: t_start to t_start + prefill_ms; decode: the
rest). Prints, per cell, n / p50 / p99 of response latency for non-overlapping ticks, all overlapping ticks, and
overlapping ticks by phase.

usage: python overlap_stats.py CELL_JSON [CELL_JSON ...]
"""
import json
import os
import sys

import numpy as np


def stats(x: list[float]) -> str:
    if not x:
        return "   0       -       -"
    return f"{len(x):4d} {np.percentile(x, 50):7.1f} {np.percentile(x, 99):7.1f}"


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


print("cell                              | none: n p50 p99      | overlap: n p50 p99   | prefill: n p50 p99   | decode: n p50 p99")
for path in sys.argv[1:]:
    d = json.load(open(path))
    calls = d.get("planner_records") or []
    groups = {"none": [], "overlap": [], "prefill": [], "decode": []}
    for (rel, start, fin), lat in zip(d["ticks_wall_s"], d["latency_series_ms"]):
        pre = dec = 0.0
        for c in calls:
            pe = c["t_start"] + c["prefill_ms"] / 1e3
            pre += overlap(start, fin, c["t_start"], pe)
            dec += overlap(start, fin, pe, c["t_end"])
        if pre == dec == 0.0:
            groups["none"].append(lat)
        else:
            groups["overlap"].append(lat)
            groups["prefill" if pre >= dec else "decode"].append(lat)
    print(f"{os.path.basename(path)[:-5]:33s} | " + " | ".join(stats(groups[k]) for k in groups))
