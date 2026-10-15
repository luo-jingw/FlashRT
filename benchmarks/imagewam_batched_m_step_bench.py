#!/usr/bin/env python
"""Cost of one ImageWAM ActionDiT denoise step at M = 64k action rows (RTX 5090).

Question (opportunities.md OPT-038, candidate 1, experiment 1): how does the
time of one ActionDiT step grow when k denoise timesteps are evaluated in one
batched forward (M = 64k GEMM rows)? Picard / ParaDiGMS parallel-in-time
sampling pays sweeps x t(M = 64k); it wins when t(640) / t(64) is small.

Cost proxy: the served frontend built with `action_horizon = 64k` (the
checkpoint structure's `max_action_horizon` raised to 64k, every other field
the checkpoint's own). Differences from a real batched step:
  - attention: here every action row sees all 64k action rows plus the
    prefix; a real batched step sees the prefix plus its own 64 rows. The
    attention time is reported separately, together with k x t_attn(k = 1)
    (k separate 64-row attentions), which brackets it;
  - AdaLN modulation: one (shift, scale, gate) for all rows here, one per
    64-row group in a real batched step (elementwise either way).

Per k: one denoise step (`imagewam_denoise_step`, step 0) captured as a CUDA
graph, CUDA-event P50 over ITERS replays (WARMUP first); one attention call
(`attn.run("mot", 0, ...)`) captured alone, same timing; an eager step under
torch.profiler (a throwaway session first), kernels classified as in
`imagewam_module_operator_profile.py` with every kernel inside `attn.run`
counted as attention. Inputs: LIBERO-10 task 0 (official prompt template),
episode 11 frame 0, noise seed 0; prefill runs once per k before timing.
The 10-step loop graph at k = 1 is timed as a cross-check (10 x step).

usage: python benchmarks/imagewam_batched_m_step_bench.py PRECISION K_LIST OUT_JSON
  PRECISION  fp16 | fp8 | nvfp4
  K_LIST     e.g. 1,2,3,4,5,6,7,8,9,10
Required env: CKPT_PATH, DATASET_STATS_PATH, FLUX2_AE_MODEL_PATH, FLUX2_SRC,
QWEN3_MODEL_SPEC, DATA_ROOT, FLASHRT_ROOT.
"""
import collections
import dataclasses
import functools
import gc
import json
import os
import re
import subprocess
import sys

import av
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
from _imagewam_libero_frames import _episode_parquet, _episode_video, _read_jsonl, center_crop_resize  # noqa: E402

from flash_rt import flash_rt_kernels as fvk  # noqa: E402
import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
import flash_rt.models.imagewam.pipeline_thor as P  # noqa: E402
from flash_rt.models.imagewam.structure import ImageWAMStructure  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

PRECISION = sys.argv[1]
KS = [int(k) for k in sys.argv[2].split(",")]
OUT = sys.argv[3]
WARMUP, ITERS, PROF_ITERS = 20, 200, 5
SUITE, TASK, EPISODE, FRAME = "libero_10", 0, 11, 0
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROOT = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")

OPS = [  # same table as benchmarks/imagewam_module_operator_profile.py
    ("conv", r"fprop|implicit_gemm|conv|Conv|cudnn|nhwc"),
    ("quant/cast", r"quantiz|kernel_quantize|cast_|_cast|to_fp8|fp8_|nvfp4"),
    ("attention", r"fmha|flash_fwd|FlashAttention|fa4cute|cuteFlash"),
    ("gemm", r"nvjet|GemmUniversal|gemm|Gemm|cutlass|splitKreduce|xmma|cublas|matmul"),
    ("norm", r"norm|Norm|rms|gn_|ada"),
    ("softmax", r"softmax|Softmax"),
    ("memcpy/set", r"Memcpy|Memset|memcpy|memset"),
    ("elementwise", r"elementwise|silu|glu|gelu|residual|add|mul|gate|rope|rotary|copy|scale|reduce|cat|fill|index|unrolled|vectorized|euler|preprocess"),
]


def opclass(name: str) -> str:
    for cls, rx in OPS:
        if re.search(rx, name):
            return cls
    return "other"


def observation() -> tuple[dict, str]:
    tasks = {t["task_index"]: t["task"] for t in _read_jsonl(f"{ROOT}/meta/tasks.jsonl")}
    df = pd.read_parquet(_episode_parquet(ROOT, EPISODE))
    if int(df["task_index"].iloc[0]) != TASK:
        raise RuntimeError(f"episode {EPISODE} is task {int(df['task_index'].iloc[0])}, expected {TASK}")
    frames = []
    for key in ("observation.images.image", "observation.images.wrist_image"):
        with av.open(_episode_video(ROOT, key, EPISODE)) as c:
            for i, f in enumerate(c.decode(video=0)):
                if i == FRAME:
                    frames.append(f.to_ndarray(format="rgb24"))
                    break
    obs = {"view1": torch.from_numpy(center_crop_resize(frames[0], 224, 224)),
           "view2": torch.from_numpy(center_crop_resize(frames[1], 224, 224)),
           "proprio": np.array(df["observation.state"].iloc[FRAME], dtype=np.float32)}
    return obs, DEFAULT_PROMPT.format(task=tasks[TASK])


def timed_graph(fn, stream: torch.cuda.Stream) -> tuple[float, float, float]:
    with torch.cuda.stream(stream):
        fn(stream.cuda_stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        fn(stream.cuda_stream)
    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(ITERS):
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    del g
    return float(np.median(ts)), float(np.percentile(ts, 10)), float(np.percentile(ts, 90))


def run_k(k: int, obs: dict, prompt: str, base_structure: ImageWAMStructure) -> dict:
    na = 64 * k
    structure = dataclasses.replace(base_structure, max_action_horizon=max(na, base_structure.max_action_horizon))
    workload = dataclasses.replace(ImageWAMWorkload.libero(), action_horizon=na)
    fe = F.load_imagewam(os.environ["CKPT_PATH"], workload, structure=structure, precision=PRECISION,
                         ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                         qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                         dataset_stats_path=os.environ["DATASET_STATS_PATH"])
    fe.set_prompt(prompt)
    d = fe._active_dims
    if d["num_action"] != na:
        raise RuntimeError(f"num_action {d['num_action']} != {na}")
    g = torch.Generator(device="cuda").manual_seed(0)
    noise = 0.01 * torch.randn(na, d["action_dim"], generator=g, device="cuda")
    out = fe.infer(obs, action_noise=noise)["actions"]
    finite = bool(np.isfinite(out).all())
    fe.stage_inputs(obs, noise=noise)
    fe.run_eager()  # prefill K/V for the timed step
    torch.cuda.synchronize()
    mod_double, mod_single = fe._action_mods[0]
    rope = fe._action_rope_table.data_ptr()
    delta = None if fe._deltas is None else fe._deltas[0]

    def step_fn(st: int) -> None:
        P.imagewam_denoise_step(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, d, 0, stream=st, attn=fe._attn,
                                mod_double=mod_double, mod_single=mod_single, head_mod=fe._head_mods[0],
                                action_rope_table=rope, delta=delta)

    def attn_fn(st: int) -> None:
        fe._attn.run("mot", 0, q_seq=na, kv_seq=d["total"], stream=st, x0=d["x0"], a0=d["a0"])

    def loop_fn(st: int) -> None:
        P.imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, d, stream=st, attn=fe._attn,
                                action_mods=fe._action_mods, head_mods=fe._head_mods,
                                action_rope_table=rope, deltas=fe._deltas)

    s = torch.cuda.Stream()
    step_p50, step_p10, step_p90 = timed_graph(step_fn, s)
    attn_p50, _, _ = timed_graph(attn_fn, s)
    loop_p50 = timed_graph(loop_fn, s)[0] if k == 1 else None

    # eager attribution
    def ranged(label, fn):
        @functools.wraps(fn)
        def w(*a, **kw):
            torch.cuda.synchronize()
            with torch.profiler.record_function(label):
                r = fn(*a, **kw)
                torch.cuda.synchronize()
            return r
        return w
    attn_run = fe._attn.run
    fe._attn.run = ranged("ATTN", attn_run)
    st = torch.cuda.current_stream().cuda_stream
    for _ in range(2):
        step_fn(st)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
        step_fn(st); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(PROF_ITERS):
            torch.cuda.synchronize()
            with torch.profiler.record_function("STEP"):
                step_fn(st)
                torch.cuda.synchronize()
    fe._attn.run = attn_run
    evs = prof.events()
    labels = {"ATTN", "STEP"}
    attn_ranges = sorted((e.time_range.start, e.time_range.end) for e in evs
                         if e.name == "ATTN" and e.device_type == torch.autograd.DeviceType.CPU)
    step_ranges = sorted(e.time_range.start for e in evs
                         if e.name == "STEP" and e.device_type == torch.autograd.DeviceType.CPU) + [float("inf")]
    kernels = [e for e in evs if e.device_type == torch.autograd.DeviceType.CUDA and e.name not in labels]
    split = collections.defaultdict(lambda: [0.0, 0.0])
    for kn in kernels:
        t0 = kn.time_range.start
        in_attn = any(a <= t0 <= b for a, b in attn_ranges)
        cls = "attention" if in_attn else opclass(kn.name)
        split[cls][0] += kn.time_range.elapsed_us() / PROF_ITERS / 1000.0
        split[cls][1] += 1 / PROF_ITERS
    per_iter = [sum(1 for kn in kernels if step_ranges[i] <= kn.time_range.start < step_ranges[i + 1])
                for i in range(PROF_ITERS)]
    rec = {"k": k, "M": na, "x0": d["x0"], "a0": d["a0"], "total": d["total"], "actions_finite": finite,
           "step_graph_ms": {"p50": round(step_p50, 4), "p10": round(step_p10, 4), "p90": round(step_p90, 4)},
           "attn_graph_ms_layer0": round(attn_p50, 4),
           "loop10_graph_ms": None if loop_p50 is None else round(loop_p50, 4),
           "eager_split_ms": {c: round(v[0], 4) for c, v in sorted(split.items(), key=lambda kv: -kv[1][0])},
           "eager_launches": {c: round(v[1]) for c, v in split.items()},
           "eager_kernel_total_ms": round(sum(v[0] for v in split.values()), 4),
           "kernels_per_iter": per_iter}
    del fe
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main() -> None:
    obs, prompt = observation()
    base = ImageWAMStructure.from_checkpoint(os.environ["CKPT_PATH"])
    commit = subprocess.run(["git", "-C", os.environ["FLASHRT_ROOT"], "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    rows = []
    for k in KS:
        r = run_k(k, obs, prompt, base)
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump({"precision": PRECISION, "gpu": torch.cuda.get_device_name(), "commit": commit, "prompt": prompt,
               "episode": EPISODE, "frame": FRAME, "warmup": WARMUP, "iters": ITERS, "prof_iters": PROF_ITERS,
               "rows": rows}, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
