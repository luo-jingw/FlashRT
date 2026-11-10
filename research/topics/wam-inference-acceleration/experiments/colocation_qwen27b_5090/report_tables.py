"""Main tables of the README from cells/: per planner context, one row per sharing mode.

Executor: p50 / p99 latency, miss rate at 50 ms, achieved ticks/s, p50 slowdown against the executor alone on the same
SM share (isolated, greenalone<P>, mps_exec<E>_alone). Planner: call / prefill medians, decode ms per spec cycle, call
slowdown against the planner alone on the whole device with the same planner settings (planner_alone_sm0*.json:
default, 256-token prefill chunks for coop, MAX_Q_SEQ 1024 for the *_maxq1024 cells; the prefill-graph cells are
divided by the default planner alone). "readback outside GIL" = PLANNER_READBACK_NOGIL=1 with CAPTURE_THREAD_LOCAL=1.
Mean power and SM clock (smi).

usage: python report_tables.py CELL_DIR OUT_MD
"""
import json
import os
import sys

import numpy as np

CELLS = [("serial", "serial", "—"), ("process_maxq1024", "process (time-slicing)", "—"),
         ("mps100_maxq1024", "mps100", "100 % / 100 %"), ("prio", "prio", "—"), ("green25", "green25", "48 / 122"),
         ("green50", "green50", "88 / 82"), ("mps50", "mps50", "50 % / 50 %"), ("mps25", "mps25", "25 % / 75 %"),
         ("coop", "coop", "—"), ("prio_gil05", "prio, switch 0.5 ms", "—"),
         ("green25_gil05", "green25, switch 0.5 ms", "48 / 122"), ("coop_gil05", "coop, switch 0.5 ms", "—"),
         ("prio_nogil", "prio, readback outside GIL", "—"),
         ("green25_nogil", "green25, readback outside GIL", "48 / 122"),
         ("prio_nogil_pfgraph", "prio, readback outside GIL, prefill graphs", "—"),
         ("green25_nogil_pfgraph", "green25, readback outside GIL, prefill graphs", "48 / 122"),
         ("green50_nogil_pfgraph", "green50, readback outside GIL, prefill graphs", "88 / 82"),
         ("coop_nogil", "coop, readback outside GIL", "—")]
EXEC_ALONE = {"green25": "greenalone25", "green50": "greenalone50", "mps50": "mps_exec50_alone",
              "mps25": "mps_exec75_alone"}
PLANNER_ALONE = {"coop": "planner_alone_sm0_chunk256", "maxq1024": "planner_alone_sm0_maxq1024"}


def load(d: str, name: str) -> dict | None:
    p = os.path.join(d, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def main(d: str, out: str) -> None:
    iso = [load(d, n)["latency_ms"]["p50"] for n in ("isolated_start", "isolated_end")]
    lines = []
    for ctx in ("short", "long"):
        lines += [f"### planner context: {ctx}", "",
                  "| mode | planner / executor SMs | exec p50 ms | exec p99 ms | miss % | exec ticks/s | exec slowdown (p50, "
                  "same SMs alone) | planner call ms | planner slowdown | prefill ms | decode ms / cycle | power W | "
                  "SM MHz |", "|" + "---|" * 13]
        for key, label, sms in CELLS:
            c = load(d, f"{ctx}_{key}")
            if c is None:
                continue
            base_key = next((v for k, v in EXEC_ALONE.items() if key.startswith(k)), None)
            base = load(d, base_key)["latency_ms"]["p50"] if base_key else float(np.mean(iso))
            pa_key = next((v for k, v in PLANNER_ALONE.items() if k in key), "planner_alone_sm0")
            pa = load(d, pa_key)["rows"][ctx]["median"]["call_ms"]
            pc = c["planner_records"]
            cyc = np.median([r["decode_ms"] / max(r["spec_cycles"], 1) for r in pc]) if pc and "spec_cycles" in pc[0] else None
            lines.append(f"| {label} | {sms} | {c['latency_ms']['p50']:.1f} | {c['latency_ms']['p99']:.1f} | "
                         f"{c['miss_rate_at_period'] * 100:.1f} | {c['ticks'] / c['duration_s']:.1f} | "
                         f"{c['latency_ms']['p50'] / base:.2f}x | {c['planner_call_ms_p50']:.0f} | "
                         f"{c['planner_call_ms_p50'] / pa:.2f}x | {c['planner_prefill_ms_p50']:.0f} | "
                         f"{'—' if cyc is None else f'{cyc:.1f}'} | {c['smi_mean']['power_W']:.0f} | "
                         f"{c['smi_mean']['clk_sm']:.0f} |")
        lines.append("")
    open(out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
