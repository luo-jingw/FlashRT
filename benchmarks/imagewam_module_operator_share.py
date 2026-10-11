"""Module x operator tables from imagewam_module_operator_profile.py JSON:
GPU ms per infer() (launches) and share of the eager kernel total (%), one
pair of tables per input file, printed as markdown.

usage: python benchmarks/imagewam_module_operator_share.py GRID_JSON [GRID_JSON ...]"""
import collections
import json
import sys

MODULES = (  # (table row, profile module labels merged into it)
    ("VAE encode", ("vae",)),
    ("backbone double (5 blocks)", ("prefill.double",)),
    ("backbone single (20 blocks) + entry", ("prefill.single", "prefill.other")),
    ("ActionDiT double (5 blocks x 10 steps)", ("denoise.double",)),
    ("ActionDiT single (20 blocks x 10 steps) + enc/head/Euler", ("denoise.single", "denoise.step_other",
                                                                  "denoise.loop_other")),
    ("input staging", ("stage_inputs",)),
    ("unattributed", ("outside",)),
)
OPS = ("gemm", "quant/cast", "attention", "conv", "norm", "elementwise", "memcpy/set", "other")


def op_of(cls: str) -> str:
    return "attention" if cls.startswith("attention") or cls == "softmax" else cls


def cells(grid: list) -> dict:
    out = collections.defaultdict(lambda: [0.0, 0.0])
    row_of = {label: row for row, labels in MODULES for label in labels}
    for r in grid:
        cell = out[(row_of.get(r["module"], "unattributed"), op_of(r["op"]))]
        cell[0] += r["ms"]
        cell[1] += r["launches"]
    return out


def table(title: str, t: dict, total: float, as_share: bool) -> None:
    ops = [o for o in OPS if sum(t[(m, o)][0] for m, _ in MODULES) >= 0.005]
    print(f"\n### {title}\n")
    print("| module | " + " | ".join(ops) + " | total |")
    print("|---|" + "---:|" * (len(ops) + 1))
    for m, _ in MODULES:
        row_ms = sum(t[(m, o)][0] for o in ops)
        if row_ms < 0.005:
            continue
        if as_share:
            vals = [f"{100 * t[(m, o)][0] / total:.1f}" if t[(m, o)][0] >= 0.005 else "-" for o in ops]
            print(f"| {m} | " + " | ".join(vals) + f" | {100 * row_ms / total:.1f} |")
        else:
            vals = [f"{t[(m, o)][0]:.2f} ({t[(m, o)][1]:.0f})" if t[(m, o)][0] >= 0.005 else "-" for o in ops]
            print(f"| {m} | " + " | ".join(vals) + f" | {row_ms:.2f} |")
    col = [sum(t[(m, o)][0] for m, _ in MODULES) for o in ops]
    if as_share:
        print("| **total** | " + " | ".join(f"**{100 * c / total:.1f}**" for c in col) + " | **100** |")
    else:
        print("| **total** | " + " | ".join(f"**{c:.2f}**" for c in col) + f" | **{total:.2f}** |")


for path in sys.argv[1:]:
    d = json.load(open(path))
    t = cells(d["grid"])
    total = sum(v[0] for v in t.values())
    tm = d["timing"]
    print(f"\n## {d['precision']} on {d['gpu']}: infer() P50 {tm['infer_p50_ms']:.2f} ms, prefill graph "
          f"{tm['prefill_graph_p50_ms']:.2f} ms, denoise graph {tm['denoise_graph_p50_ms']:.2f} ms, eager kernel "
          f"total {total:.2f} ms, graph-replay kernel total {sum(d['graph_kernel_ms_by_op'].values()):.2f} ms")
    table("GPU ms per infer() (launches)", t, total, as_share=False)
    table("share of eager kernel total (%)", t, total, as_share=True)
