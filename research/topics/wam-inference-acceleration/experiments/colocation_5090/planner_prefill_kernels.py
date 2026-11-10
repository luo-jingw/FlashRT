"""Which GPU work in the planner's long-context prefill holds the SMs long enough to stall a co-located executor?

One Qwen3-VL-2B prefill (model forward over the full prompt, no decode) at PLANNER_CTX_EXTRA extra text tokens,
traced with the torch profiler (CUDA activity). Reports the kernel count, total kernel time, the longest single
kernels, and per-name totals; the raw chrome trace is written next to the JSON.

usage: PLANNER_CTX_EXTRA=16000 python planner_prefill_kernels.py OUT_JSON
"""
import json
import os
import sys
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from planner_qwen3vl import Planner

OUT = sys.argv[1]
planner = Planner()
x = planner.inputs()
with torch.inference_mode():
    for _ in range(2):
        planner.model(**x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        planner.model(**x)
        torch.cuda.synchronize()
trace = OUT[:-5] + "_trace.json"
prof.export_chrome_trace(trace)
kernels = [e for e in json.load(open(trace))["traceEvents"] if e.get("cat") == "kernel"]
by_name = defaultdict(lambda: [0, 0.0, 0.0])
for k in kernels:
    s = by_name[k["name"]]
    s[0] += 1; s[1] += k["dur"]; s[2] = max(s[2], k["dur"])
t0 = min(k["ts"] for k in kernels); t1 = max(k["ts"] + k["dur"] for k in kernels)
longest = sorted(kernels, key=lambda k: -k["dur"])[:15]
res = {"prompt_tokens": int(x["input_ids"].shape[1]), "attn_implementation": planner.model.config._attn_implementation,
       "kernels": len(kernels), "kernel_ms_total": sum(k["dur"] for k in kernels) / 1e3, "span_ms": (t1 - t0) / 1e3,
       "longest": [{"name": k["name"][:120], "ms": k["dur"] / 1e3, "grid": k["args"].get("grid"), "block": k["args"].get("block")} for k in longest],
       "by_name_top": sorted(([n[:120], c, t / 1e3, m / 1e3] for n, (c, t, m) in by_name.items()), key=lambda r: -r[2])[:15]}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "by_name_top"}, indent=1)[:3000])
print("by name (name, count, total ms, max ms):")
for r in res["by_name_top"]:
    print(f"  {r[1]:5d} {r[2]:9.2f} {r[3]:8.2f}  {r[0]}")
