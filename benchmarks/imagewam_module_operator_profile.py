"""ImageWAM module x operator profiling of the served LIBERO configuration,
one precision per process (any GPU the deployment entry supports).

Configuration: ImageWAMWorkload.libero() (2 views 224x224, action horizon 64,
action_dim 7, proprio 8, text 512, 10 denoise steps), real LIBERO checkpoint,
real FLUX.2 AE, Qwen3-4B text encoder, built through load_imagewam (the
deployment entry, default profile). Observation: LIBERO-format uint8 frames
(content does not change kernel time) + proprio.

Measurements:
  timing     infer() CUDA-event P50 (warmup 20 / iters 100); stage times:
             VAE encode (the frontend's own staging call), prefill graph,
             denoise-loop graph, full-graph replay (round robin, 100 samples);
  attribution  run_eager() + VAE staging under torch.profiler, with every
             module function wrapped in record_function and a device sync at
             range entry/exit, so each kernel's GPU interval lies inside
             exactly one innermost range. Kernels are grouped by
             (module, operator class). Kernel durations are those of eager
             launches of the same kernels the graph replays.
  graph check  full infer() under torch.profiler (graph replay): kernel time
             by operator class, to confirm eager kernel time matches replay.
Environment: CKPT_PATH, FLUX2_AE_MODEL_PATH, FLUX2_SRC, QWEN3_MODEL_SPEC,
DATASET_STATS_PATH. Tables: benchmarks/imagewam_module_operator_share.py.

usage: python benchmarks/imagewam_module_operator_profile.py PRECISION OUT_JSON [--names]"""
import collections
import functools
import json
import re
import sys
import time

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk
import flash_rt.frontends.torch.imagewam_thor as F
import flash_rt.models.imagewam.pipeline_thor as P
from flash_rt.models.imagewam.workload import ImageWAMWorkload

import os

PRECISION, OUT = sys.argv[1], sys.argv[2]
DUMP_NAMES = "--names" in sys.argv
PROMPT = "pick up the red block and place it in the tray"

wl = ImageWAMWorkload.libero()
t0 = time.time()
fe = F.load_imagewam(os.environ["CKPT_PATH"], wl, precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
fe.set_prompt(PROMPT)
torch.cuda.synchronize()
opts = fe._resolved.options
print(f"built {PRECISION} in {time.time() - t0:.0f}s; text_trim={opts.text_trim} vae_graph_input={opts.vae_graph_input} "
      f"use_fa4={opts.use_fa4} active x0={fe._active_dims['x0']} a0={fe._active_dims['a0']} "
      f"num_action={fe._active_dims['num_action']}", flush=True)

rng = np.random.default_rng(0)
obs = {"view1": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
       "view2": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
       "proprio": rng.standard_normal(8).astype(np.float32)}
views = (torch.as_tensor(obs["view1"]), torch.as_tensor(obs["view2"]))
for _ in range(3):
    out = fe.infer(obs)
torch.cuda.synchronize()
act = np.asarray(out["actions"])
print("actions", act.shape, "finite", bool(np.isfinite(act).all()), flush=True)

# ---------------- timing ----------------
ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def timed(fn):
    ev0.record(); fn(); ev1.record(); ev1.synchronize()
    return ev0.elapsed_time(ev1)


s = torch.cuda.Stream()


def graph_of(fn):
    with torch.cuda.stream(s):
        fn(s.cuda_stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn(s.cuda_stream)
    return g


def prefill_fn(st):
    P.imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe._active_dims, stream=st,
                       attn=fe._attn, mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
                       rope_table=fe._rope_table.data_ptr())


def denoise_fn(st):
    P.imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe._active_dims, stream=st,
                            attn=fe._attn, action_mods=fe._action_mods, head_mods=fe._head_mods,
                            action_rope_table=fe._action_rope_table.data_ptr(), deltas=fe._deltas)


g_pre, g_den = graph_of(prefill_fn), graph_of(denoise_fn)
infer_t, pre_t, den_t, vae_t = [], [], [], []
for _ in range(20):
    fe.infer(obs); g_pre.replay(); g_den.replay()
torch.cuda.synchronize()
for _ in range(100):
    infer_t.append(timed(lambda: fe.infer(obs)))
    pre_t.append(timed(g_pre.replay))
    den_t.append(timed(g_den.replay))
    vae_t.append(timed(lambda: fe.stage_images(*views)))
med = lambda x: float(np.median(x))
timing = {"infer_p50_ms": med(infer_t), "infer_p10_ms": float(np.percentile(infer_t, 10)),
          "infer_p90_ms": float(np.percentile(infer_t, 90)), "prefill_graph_p50_ms": med(pre_t),
          "denoise_graph_p50_ms": med(den_t), "vae_stage_p50_ms": med(vae_t)}
print("timing", json.dumps({k: round(v, 3) for k, v in timing.items()}), flush=True)

# ---------------- attribution (eager, ranged) ----------------
def ranged(label, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        torch.cuda.synchronize()
        with torch.profiler.record_function(label):
            r = fn(*a, **k)
            torch.cuda.synchronize()
        return r
    return w


RANGES = {"prefill.double": ("_double_stream_layer",), "prefill.single": ("_single_stream_layer",
          "_single_stream_layer_kv_only"), "denoise.double": ("_action_double_layer",),
          "denoise.single": ("_action_single_layer",), "denoise.step_other": ("imagewam_denoise_step",)}
saved = []
for label, names in RANGES.items():
    for n in names:
        saved.append((P, n, getattr(P, n)))
        setattr(P, n, ranged(label, getattr(P, n)))
for mod in (P, F):
    saved.append((mod, "imagewam_prefill", getattr(mod, "imagewam_prefill")))
    setattr(mod, "imagewam_prefill", ranged("prefill.other", saved[-1][2]))
    saved.append((mod, "imagewam_denoise_loop", getattr(mod, "imagewam_denoise_loop")))
    setattr(mod, "imagewam_denoise_loop", ranged("denoise.loop_other", saved[-1][2]))
attn_run = fe._attn.run
fe._attn.run = ranged("ATTN", attn_run)
vae_run = fe._vae_stage.run if fe._vae_stage is not None else None
if vae_run is not None:
    fe._vae_stage.run = ranged("vae", vae_run)
LABELS = set(RANGES) | {"prefill.other", "denoise.loop_other", "ATTN", "vae", "stage_inputs"}

ITERS = 3
for _ in range(2):
    fe.stage_images(*views); fe.run_eager()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(ITERS):
        torch.cuda.synchronize()
        with torch.profiler.record_function("stage_inputs"):
            fe.stage_images(*views)
            torch.cuda.synchronize()
        fe.run_eager()
        torch.cuda.synchronize()
for mod, n, fn in saved:
    setattr(mod, n, fn)
fe._attn.run = attn_run
if vae_run is not None:
    fe._vae_stage.run = vae_run

evs = prof.events()
ranges = [(e.time_range.start, e.time_range.end, e.name) for e in evs
          if e.name in LABELS and e.device_type == torch.autograd.DeviceType.CPU]
# record_function ranges also appear on the GPU timeline as user annotations
# named after the range: exclude them, they are not kernels.
kernels = [e for e in evs if e.device_type == torch.autograd.DeviceType.CUDA and e.name not in LABELS]


def innermost(t):
    best = None
    for st, en, name in ranges:
        if st <= t <= en and (best is None or en - st < best[1] - best[0]):
            best = (st, en, name)
    return best[2] if best else None


ranges.sort()
def attribute(t):
    """innermost module range and whether the kernel sits inside an ATTN range"""
    mod, in_attn, span = None, False, None
    for st, en, name in ranges:
        if st > t:
            break
        if t <= en:
            if name == "ATTN":
                in_attn = True
            elif span is None or en - st < span:
                mod, span = name, en - st
    return mod, in_attn


OPS = [  # (op class, regex on kernel name), first match wins
    ("conv", r"fprop|implicit_gemm|conv|Conv|cudnn|nhwc"),
    # quant/cast before attention: Thor NVFP4 kernels live in flash_rt::fp4::
    # and `flash` would otherwise steal kernel_quantize_* (4 ms on nvfp4).
    ("quant/cast", r"quantiz|kernel_quantize|cast_|_cast|to_fp8|fp8_|nvfp4"),
    ("attention", r"fmha|flash_fwd|FlashAttention|fa4cute|cuteFlash"),
    ("gemm", r"nvjet|GemmUniversal|gemm|Gemm|cutlass|splitKreduce|xmma|cublas|matmul"),
    ("norm", r"norm|Norm|rms|gn_|ada"),
    ("softmax", r"softmax|Softmax"),
    ("memcpy/set", r"Memcpy|Memset|memcpy|memset"),
    ("elementwise", r"elementwise|silu|glu|gelu|residual|add|mul|gate|rope|rotary|copy|scale|reduce|cat|fill|index|unrolled|vectorized|euler|preprocess"),
]


def opclass(name):
    for cls, rx in OPS:
        if re.search(rx, name):
            return cls
    return "other"


grid = collections.defaultdict(lambda: [0.0, 0])
names = collections.defaultdict(lambda: [0.0, 0, set()])
unattributed = 0.0
for k in kernels:
    mod, in_attn = attribute(k.time_range.start)
    mod = mod or "outside"
    if mod == "outside":
        unattributed += k.time_range.elapsed_us()
    cls = "attention:" + opclass(k.name) if in_attn else opclass(k.name)
    cell = grid[(mod, cls)]
    cell[0] += k.time_range.elapsed_us() / ITERS / 1000.0
    cell[1] += 1 / ITERS
    nm = names[k.name]
    nm[0] += k.time_range.elapsed_us() / ITERS / 1000.0
    nm[1] += 1 / ITERS
    nm[2].add(f"{mod}|{cls}")
total_eager = sum(v[0] for v in grid.values())
print(f"eager kernel total {total_eager:.2f} ms/infer; unattributed {unattributed / ITERS / 1000:.3f} ms", flush=True)
for (mod, cls), (ms, n) in sorted(grid.items(), key=lambda kv: -kv[1][0]):
    print(f"  {mod:20s} {cls:22s} {ms:9.3f} ms  {n:7.0f} launches  {1000 * ms / max(n, 1):8.2f} us/launch")
if DUMP_NAMES:
    for nm, (ms, n, where) in sorted(names.items(), key=lambda kv: -kv[1][0]):
        print(f"    {ms:8.3f} ms {n:6.0f}x  {nm[:120]}  <- {sorted(where)}")

# ---------------- graph check ----------------
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as gprof:
    for _ in range(ITERS):
        fe.infer(obs)
    torch.cuda.synchronize()
graph_cls = collections.defaultdict(float)
for e in gprof.key_averages():
    if e.self_device_time_total > 0:
        graph_cls[opclass(e.key)] += e.self_device_time_total / ITERS / 1000.0
print("graph-replay kernel ms by op class:", {k: round(v, 3) for k, v in sorted(graph_cls.items())},
      "total", round(sum(graph_cls.values()), 3), flush=True)

json.dump({"precision": PRECISION, "timing": timing, "eager_kernel_total_ms": total_eager,
           "unattributed_ms": unattributed / ITERS / 1000,
           "grid": [{"module": m, "op": c, "ms": v[0], "launches": v[1]} for (m, c), v in grid.items()],
           "kernels": [{"name": nm, "ms": v[0], "launches": v[1], "cells": sorted(v[2])} for nm, v in names.items()],
           "graph_kernel_ms_by_op": dict(graph_cls),
           "config": {"workload": "ImageWAMWorkload.libero()", "active_dims": {k: fe._active_dims[k] for k in
                      ("x0", "a0", "num_action", "total")}, "text_trim": opts.text_trim,
                      "vae_graph_input": str(opts.vae_graph_input), "use_fa4": opts.use_fa4},
           "gpu": torch.cuda.get_device_name()}, open(OUT, "w"), indent=1)
print("wrote", OUT)
