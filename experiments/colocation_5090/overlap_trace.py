"""Why does the executor stall for ~290 ms next to a long-context planner under stream priority?

In-process stream-priority setup (as colocation_bench.py `prio`): ImageWAM fp8 executor on a priority −1 stream,
Qwen3-VL-2B planner (PLANNER_CTX_EXTRA extra text tokens) on a priority 0 stream, one torch profiler (CUDA activity)
tracing both. Two traces:
  alone    N_ALONE executor ticks with no planner;
  overlap  one planner call (generate, prefill + decode) in a thread; executor ticks back-to-back from 50 ms after
           the call starts until it ends.
Executor kernels are the kernels on the streams seen in the alone trace; they are cut into ticks of K kernels, K =
kernels per tick in the alone trace. Per tick: span (first kernel start to last kernel end), busy (sum of executor
kernel durations), wait (span − busy), and the fraction of the span during which a planner kernel is running.

usage: PLANNER_CTX_EXTRA=16000 python overlap_trace.py OUT_JSON
"""
import json
import os
import sys
import threading
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _imagewam_libero_frames import center_crop_resize  # noqa: E402
from planner_qwen3vl import Planner  # noqa: E402

import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

OUT = sys.argv[1]
N_ALONE = 5

exec_stream = torch.cuda.Stream(priority=-1)
with torch.cuda.stream(exec_stream):
    fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision="fp8",
                         ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                         qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                         dataset_stats_path=os.environ["DATASET_STATS_PATH"])
    fe.set_prompt("A video recorded from a robot's point of view executing the following instruction: "
                  "put both the alphabet soup and the tomato sauce in the basket")
fe._qwen3 = None
torch.cuda.empty_cache()
rng = np.random.default_rng(0)
frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
obs = {"view1": torch.from_numpy(center_crop_resize(frame, 224, 224)),
       "view2": torch.from_numpy(center_crop_resize(frame, 224, 224)), "proprio": np.zeros(8, dtype=np.float32)}


def tick() -> None:
    with torch.cuda.stream(exec_stream):
        fe.stage_inputs(obs)
        fe._graph.replay()
    exec_stream.synchronize()


planner = Planner(torch.cuda.Stream(priority=0))
px = planner.inputs()
for _ in range(10):
    tick()
for _ in range(2):
    planner.call(px)
torch.cuda.synchronize()


def kernels_of(prof: profile, name: str) -> list[dict]:
    path = OUT[:-5] + f"_{name}_trace.json"
    prof.export_chrome_trace(path)
    return sorted((e for e in json.load(open(path))["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])


with profile(activities=[ProfilerActivity.CUDA]) as prof_a:
    for _ in range(N_ALONE):
        tick()
alone = kernels_of(prof_a, "alone")
exec_streams = {k["args"]["stream"] for k in alone}
K = len(alone) // N_ALONE

call_done = threading.Event()
call_rec = {}


def planner_call() -> None:
    call_rec.update(planner.call(px).__dict__)
    call_done.set()


with profile(activities=[ProfilerActivity.CUDA]) as prof_o:
    threading.Thread(target=planner_call, daemon=True).start()
    time.sleep(0.05)
    n_ticks = 0
    while not call_done.is_set():
        tick()
        n_ticks += 1
    torch.cuda.synchronize()
over = kernels_of(prof_o, "overlap")
ex = [k for k in over if k["args"]["stream"] in exec_streams]
pl = [k for k in over if k["args"]["stream"] not in exec_streams]
pl_iv = np.array([[k["ts"], k["ts"] + k["dur"]] for k in pl])


def planner_cover(a: float, b: float) -> float:
    if not len(pl_iv):
        return 0.0
    lo, hi = np.maximum(pl_iv[:, 0], a), np.minimum(pl_iv[:, 1], b)
    iv = sorted((x, y) for x, y in zip(lo, hi) if y > x)
    tot, cur = 0.0, None
    for x, y in iv:
        if cur is None or x > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [x, y]
        else:
            cur[1] = max(cur[1], y)
    if cur:
        tot += cur[1] - cur[0]
    return tot / (b - a)


def ticks_of(ks: list[dict]) -> list[dict]:
    out = []
    for i in range(len(ks) // K):
        t = ks[i * K:(i + 1) * K]
        span = (t[-1]["ts"] + t[-1]["dur"] - t[0]["ts"]) / 1e3
        busy = sum(k["dur"] for k in t) / 1e3
        gaps = [(t[j + 1]["ts"] - t[j]["ts"] - t[j]["dur"]) / 1e3 for j in range(K - 1)]
        out.append({"start_ms": t[0]["ts"] / 1e3, "span_ms": span, "busy_ms": busy, "wait_ms": span - busy,
                    "max_gap_ms": max(gaps), "planner_cover": planner_cover(t[0]["ts"], t[-1]["ts"] + t[-1]["dur"])})
    return out


ta, to = ticks_of(alone), ticks_of(ex)
t_first = min(k["ts"] for k in over) / 1e3
for t in to:
    t["start_ms"] -= t_first
for t in ta:
    t["start_ms"] = 0.0
busy_alone = np.median([t["busy_ms"] for t in ta])
res = {"prompt_tokens": int(px["input_ids"].shape[1]), "kernels_per_tick": K, "exec_streams": sorted(exec_streams),
       "planner_kernels": len(pl), "planner_call": call_rec, "ticks_issued": n_ticks, "ticks_traced": len(to),
       "alone": ta, "overlap": to}
json.dump(res, open(OUT, "w"), indent=1)
print(f"K={K} kernels/tick, prompt {res['prompt_tokens']} tokens, planner kernels {len(pl)}, "
      f"call prefill {call_rec['prefill_ms']:.0f} ms decode {call_rec['decode_ms']:.0f} ms")
print("alone   : " + " | ".join(f"span {t['span_ms']:.1f} busy {t['busy_ms']:.1f} wait {t['wait_ms']:.2f}" for t in ta))
print("overlap : start  span   busy   wait  maxgap  planner-cover")
for t in to:
    print(f"        {t['start_ms']:7.0f} {t['span_ms']:6.1f} {t['busy_ms']:6.1f} {t['wait_ms']:6.1f} {t['max_gap_ms']:6.2f}  {t['planner_cover']:.2f}")
