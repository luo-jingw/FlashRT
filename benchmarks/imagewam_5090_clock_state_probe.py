#!/usr/bin/env python
"""Does ImageWAM latency on RTX 5090 move between clock / memory states over time?

One process, one precision, LIBERO frame and official prompt (as
imagewam_batched_m_step_bench.py). Before timing, waits until the GPU is
back at its idle P-state (P8, memory clock <= 810 MHz, up to WAIT_S).

  cold run  RUN_S seconds of back-to-back windows; each window: ITERS_W
            infer() calls (CUDA-event ms each), then 5 prefill-graph and 5
            denoise-loop-graph replays (the infer() graph's two halves,
            captured separately as in imagewam_module_operator_profile.py);
  idle gaps after the cold run, for each gap in GAPS_S: sleep, then time
            the first FIRST_N infer() calls one by one;
  bandwidth 512 MiB device-to-device copy, 10 repeats, before the wait for
            idle (cold GPU) and after the cold run (warm GPU), not adjacent
            to the timed windows.
Clocks: `nvidia-smi --query-gpu=... --loop-ms=100` runs alongside and
writes CSV; windows carry wall-clock timestamps to join against it.

usage: python benchmarks/imagewam_5090_clock_state_probe.py PRECISION OUT_PREFIX
"""
import json
import os
import subprocess
import sys
import time

import av
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
from _imagewam_libero_frames import _episode_parquet, _episode_video, _read_jsonl, center_crop_resize  # noqa: E402

from flash_rt import flash_rt_kernels as fvk  # noqa: E402
import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
import flash_rt.models.imagewam.pipeline_thor as P  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

PRECISION, OUT = sys.argv[1], sys.argv[2]
RUN_S, ITERS_W, WAIT_S = 240.0, 20, 180.0
GAPS_S, FIRST_N = [1, 5, 30, 120], 10
SUITE, TASK, EPISODE, FRAME = "libero_10", 0, 11, 0
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROOT = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")
SMI_FIELDS = "timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active"


def smi_now() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.mem,clocks.sm", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(", ")
    return {"pstate": out[0], "mem": int(out[1]), "sm": int(out[2])}


def bandwidth() -> list[float]:
    n = 512 << 20
    a = torch.empty(n, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    b.copy_(a); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(10):
        e0.record(); b.copy_(a); e1.record(); e1.synchronize()
        out.append(round(2 * n / (e0.elapsed_time(e1) * 1e-3) / 1e9, 1))  # read + write, GB/s
    del a, b
    torch.cuda.empty_cache()
    return out


def observation() -> tuple[dict, str]:
    tasks = {t["task_index"]: t["task"] for t in _read_jsonl(f"{ROOT}/meta/tasks.jsonl")}
    df = pd.read_parquet(_episode_parquet(ROOT, EPISODE))
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


smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,nounits", "--loop-ms=100"],
                       stdout=open(f"{OUT}_smi.csv", "w"), stderr=subprocess.DEVNULL)
try:
    t_start = time.time()
    obs, prompt = observation()
    fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                         ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                         qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                         dataset_stats_path=os.environ["DATASET_STATS_PATH"])
    fe.set_prompt(prompt)
    d = fe._active_dims
    s = torch.cuda.Stream()

    def graph_of(fn):
        with torch.cuda.stream(s):
            fn(s.cuda_stream)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn(s.cuda_stream)
        return g

    g_pre = graph_of(lambda st: P.imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, d, stream=st, attn=fe._attn,
                                                   mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
                                                   rope_table=fe._rope_table.data_ptr()))
    g_den = graph_of(lambda st: P.imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, d, stream=st, attn=fe._attn,
                                                        action_mods=fe._action_mods, head_mods=fe._head_mods,
                                                        action_rope_table=fe._action_rope_table.data_ptr(), deltas=fe._deltas))
    for _ in range(3):
        fe.infer(obs)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(fn) -> float:
        e0.record(); fn(); e1.record(); e1.synchronize()
        return e0.elapsed_time(e1)

    bw_cold = bandwidth()
    t_wait0 = time.time()
    while time.time() - t_wait0 < WAIT_S:
        st = smi_now()
        if st["pstate"] == "P8" and st["mem"] <= 810:
            break
        time.sleep(1.0)
    idle_state = smi_now()
    idle_waited = round(time.time() - t_wait0, 1)

    windows = []
    t0 = time.time()
    while time.time() - t0 < RUN_S:
        tw = time.time()
        inf = [timed(lambda: fe.infer(obs)) for _ in range(ITERS_W)]
        pre = [timed(g_pre.replay) for _ in range(5)]
        den = [timed(g_den.replay) for _ in range(5)]
        windows.append({"t": round(tw - t0, 3), "wall": tw, "infer": [round(x, 4) for x in inf],
                        "prefill": [round(x, 4) for x in pre], "denoise": [round(x, 4) for x in den]})
    bw_warm = bandwidth()

    gaps = []
    for gap in GAPS_S:
        time.sleep(gap)
        state = smi_now()
        tg = time.time()
        first = [round(timed(lambda: fe.infer(obs)), 4) for _ in range(FIRST_N)]
        gaps.append({"gap_s": gap, "state_before": state, "wall": tg, "first_infer_ms": first})
finally:
    smi.terminate()

json.dump({"precision": PRECISION, "gpu": torch.cuda.get_device_name(), "t_start": t_start,
           "idle_state_at_start": idle_state, "idle_waited_s": idle_waited,
           "bandwidth_cold_GBps": bw_cold, "bandwidth_warm_GBps": bw_warm,
           "windows": windows, "gaps": gaps}, open(f"{OUT}.json", "w"))
p50 = lambda xs: float(np.median(xs))
first, last = windows[:5], windows[-5:]
print(f"{PRECISION}: idle at start {idle_state} after {idle_waited}s; "
      f"infer P50 first 5 windows {p50(sum((w['infer'] for w in first), [])):.3f} ms, "
      f"last 5 windows {p50(sum((w['infer'] for w in last), [])):.3f} ms; "
      f"bandwidth cold {p50(bw_cold):.0f} / warm {p50(bw_warm):.0f} GB/s")
for g in gaps:
    print(f"  after {g['gap_s']}s idle ({g['state_before']}): first infer {g['first_infer_ms'][:4]} ... median {p50(g['first_infer_ms']):.3f}")
