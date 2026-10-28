"""Collect the R1 cells (r1/*.json) into one markdown table, r1_table.md.

usage: python summarize_r1.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MODE_ORDER = ["isolated", "serial", "process", "prio", "mps10", "mps15", "mps25", "mps50", "mps100"]
HZ_LABEL = {0.0: "continuous", 0.5: "0.5", 1.0: "1", 2.0: "2"}


def fmt(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def row_of(path: str) -> tuple[tuple, str]:
    d = json.load(open(path))
    name = os.path.basename(path)[:-5]
    m = re.match(r"ctx(\d+)_", name)
    ctx = int(m.group(1)) if m else None
    tag = "droptext control" if name.endswith("droptext") else ("MPS executor SM limit, no planner" if "alone" in name else "")
    lat, smi = d["latency_ms"], d["smi_mean"]
    calls = d["planner_calls"]
    rate = calls / d["duration_s"] if calls else None
    key = (ctx if ctx is not None else -1, MODE_ORDER.index(d["mode"]), -d["planner_hz"] if d["planner_hz"] else -99, name)
    cells = [name, "—" if ctx is None else ("~260" if ctx == 0 else "~16.3 k"), d["mode"],
             HZ_LABEL[d["planner_hz"]] if calls else "—", fmt(lat["p50"]), fmt(lat["p99"]), fmt(lat["max"]),
             f"{100 * d['miss_rate_at_period']:.1f}", str(d["ticks"]), str(calls) if calls else "—", fmt(rate, 2),
             fmt(d["planner_call_ms_p50"], 0), fmt(smi["util_mem"], 0), fmt(smi["power_W"], 0),
             fmt(d.get("torch_reserved_GB"), 1), tag]
    return key, "| " + " | ".join(cells) + " |"


rows = sorted(row_of(p) for p in glob.glob(os.path.join(HERE, "r1", "*.json")) if not p.endswith(".planner.json"))
head = ("| cell | planner context (tokens) | mode | planner target Hz | exec p50 ms | exec p99 ms | exec max ms | "
        "miss @50 ms % | exec ticks / 60 s | planner calls | planner calls/s | planner call p50 ms | "
        "DRAM util % (smi) | power W | exec torch reserved GB | note |")
sep = "|" + "---|" * 16
open(os.path.join(HERE, "r1_table.md"), "w").write("\n".join([head, sep] + [r for _, r in rows]) + "\n")
print(f"{len(rows)} cells -> r1_table.md")
