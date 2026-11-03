"""Tables from the cell JSONs of run_grid.sh: one row per cell, and executor latency split by planner phase.

Per cell: executor p50 / p99 / max response latency, miss rate at the period, service time; planner calls, call /
ViT / prefill / decode medians and decode tokens/s; mean SM clock, memory clock, power and DRAM utilization (smi).
Phase split: a tick is labelled by the planner phase it overlaps longest (ViT, prefill, decode; "idle" = no overlap);
per label n, p50, p99 of response latency, and the mean SM clock / power of the smi samples falling in that phase.

usage: python summarize.py CELL_DIR OUT_MD
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def windows(c: dict) -> dict[str, tuple[float, float]]:
    dec0 = c["t_end"] - c["decode_ms"] / 1e3
    pf0 = dec0 - c["prefill_ms"] / 1e3
    vit0 = pf0 - (c["inject_ms"] + c["vit_ms"]) / 1e3
    return {"vit": (vit0, pf0), "prefill": (pf0, dec0), "decode": (dec0, c["t_end"])}


def ov(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def label(t0: float, t1: float, calls: list[dict]) -> str:
    acc = {"vit": 0.0, "prefill": 0.0, "decode": 0.0}
    for c in calls:
        for k, (w0, w1) in windows(c).items():
            acc[k] += ov(t0, t1, w0, w1)
    best = max(acc, key=acc.get)
    return best if acc[best] > 0 else "idle"


def f(x, nd=1) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def main(cell_dir: str, out: str) -> None:
    rows, phase_rows, gpu_rows = [], [], []
    for path in sorted(glob.glob(os.path.join(cell_dir, "*.json"))):
        name = os.path.basename(path)[:-5]
        if name.startswith(("planner_alone", "exec_slim")) or name.endswith(".planner"):
            continue
        d = json.load(open(path))
        if "mode" not in d:
            continue
        pc = d["planner_records"]
        dec_tok_s = (np.median([(c["new_tokens"] - 1) * 1e3 / c["decode_ms"] for c in pc]) if pc else None)
        sm = d["smi_mean"] or {}
        if d.get("tick_host_ms"):
            h = d["tick_host_ms"]
            gpu_rows.append(f"| {name} | {f(d['graph_gpu_ms']['p50'])} | {f(d['graph_gpu_ms']['p99'])} | "
                            f"{f(h['stage_inputs']['p50'])} | {f(h['stage_inputs']['p99'])} | {f(h['graph_launch']['p50'])} | "
                            f"{f(h['wait']['p50'])} | {f(np.median([c['spec_cycles'] for c in pc]) if pc and 'spec_cycles' in pc[0] else None, 0)} | "
                            f"{f(np.median([c['decode_ms'] / max(c['spec_cycles'], 1) for c in pc]) if pc and 'spec_cycles' in pc[0] else None)} |")
        rows.append(f"| {name} | {d['mode']} | {d.get('planner_ctx') if pc else '—'} | {f(d['latency_ms']['p50'])} | "
                    f"{f(d['latency_ms']['p99'])} | {f(d['latency_ms']['max'])} | {d['miss_rate_at_period']*100:.1f} | "
                    f"{f(d['service_ms']['p50'])} | {d['ticks']} | {len(pc)} | {f(d['planner_call_ms_p50'], 0)} | "
                    f"{f(d['planner_vit_ms_p50'])} | {f(d['planner_prefill_ms_p50'])} | {f(d['planner_decode_ms_p50'], 0)} | "
                    f"{f(dec_tok_s)} | {f(sm.get('clk_sm'), 0)} | {f(sm.get('clk_mem'), 0)} | {f(sm.get('power_W'), 0)} | "
                    f"{f(sm.get('util_mem'), 0)} | {d['green_sm']['planner'] if d['green_sm'] else '—'} |")
        if not pc:
            continue
        lab = {"idle": [], "vit": [], "prefill": [], "decode": []}
        gpu = {"idle": [], "vit": [], "prefill": [], "decode": []}
        gser = d.get("tick_gpu_series_ms") or [None] * len(d["latency_series_ms"])
        for (rel, st, fin), lat, g in zip(d["ticks_wall_s"], d["latency_series_ms"], gser):
            k = label(st, fin, pc)
            lab[k].append(lat)
            if g is not None:
                gpu[k].append(g)
        smi = np.array(d["smi_series"]) if d["smi_series"] else np.zeros((0, 6))
        clk = {"idle": [], "vit": [], "prefill": [], "decode": []}
        for r in smi:
            clk[label(r[0], r[0] + 1e-6, pc)].append(r)
        cells = []
        for k in ("idle", "vit", "prefill", "decode"):
            x = lab[k]
            c = np.array(clk[k])
            g = gpu[k]
            cells.append(f"{len(x)} / {f(np.percentile(x, 50) if x else None)} / {f(np.percentile(x, 99) if x else None)}"
                         f" / {f(np.percentile(g, 50) if g else None)}")
            cells.append(f"{f(c[:, 3].mean() if len(c) else None, 0)} / {f(c[:, 5].mean() if len(c) else None, 0)}")
        phase_rows.append(f"| {name} | " + " | ".join(cells) + " |")
    md = ["| cell | mode | planner ctx | exec p50 | exec p99 | exec max | miss % | exec service p50 | ticks | planner "
          "calls | call ms | ViT ms | prefill ms | decode ms | decode tok/s | SM MHz | mem MHz | power W | DRAM util % | "
          "planner SMs |",
          "|" + "---|" * 20, *rows, "",
          "Executor tick split (cells with host timing): graph = event span around the replay (includes a delayed launch); "
          "host stage_inputs / graph launch / wait = wall ms of the three host steps; planner spec-decode cycles per call "
          "and decode ms per cycle (medians):", "",
          "| cell | graph p50 | graph p99 | stage_inputs p50 | stage_inputs p99 | launch p50 | wait p50 | planner cycles | "
          "ms / cycle |", "|" + "---|" * 9, *gpu_rows, "",
          "Executor latency by the planner phase each tick overlaps (n / latency p50 / latency p99 / tick GPU p50, ms), and "
          "mean SM clock MHz / power W "
          "of the smi samples in that phase:", "",
          "| cell | idle ticks | idle clk/W | ViT ticks | ViT clk/W | prefill ticks | prefill clk/W | decode ticks | "
          "decode clk/W |", "|" + "---|" * 9, *phase_rows]
    open(out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
