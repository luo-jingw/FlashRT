"""Served latency of ImageWAM with the denoise loop cut to 2 evaluations (grid points {0, 9}).

One process, one precision, LIBERO-10 task 0 frame + official prompt (as
imagewam_5090_clock_state_probe.py). The frontend is loaded and set_prompt()
captures the served graph (VAE stage + prefill + 10-step loop) = G10. Then
the denoise schedule is replaced by the {0, 9} subset of the same grid:
  step A = served step 0's AdaLN/head modulation, Euler size t9 - t0 (= sum of deltas[0:9])
  step B = served step 9's modulation, Euler size 0 - t9 (= deltas[9])
and the frontend's own _capture_graph() records G2 (same capture path,
num_denoise_steps = 2). infer() is timed with CUDA events (stage_inputs +
replay + action readback), G10 / G2 alternated in ABBA blocks of BLOCK
calls. Also timed alone: the prefill graph and the 10- and 2-step denoise
loops (captured separately, as in the clock-state probe).
Checks: G2 vs run_eager() with the same 2-step schedule (same noise), and
G2 vs G10 action MAE (xyz+rpy, de-normalized), fixed noise.

usage: python grid2_latency.py PRECISION OUT_JSON
"""
import json
import os
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
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

PRECISION, OUT = sys.argv[1], sys.argv[2]
WARMUP, BLOCK, ROUNDS = 30, 50, 4
SUITE, TASK, EPISODE, FRAME = "libero_10", 0, 11, 0
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROOT = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")
GRID = (0, 9)


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


obs, prompt = observation()
fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
fe.set_prompt(prompt)
d = fe._active_dims
na, ad, nstep = d["num_action"], d["action_dim"], d["num_denoise_steps"]
assert nstep == 10 and fe._deltas is not None
noise = 0.01 * torch.randn(na, ad, generator=torch.Generator(device="cuda").manual_seed(0), device="cuda")

# ---- 10-step served graph and its reference actions
g10 = fe._graph
sched10 = (fe._action_mods, fe._head_mods, fe._deltas)  # kept alive: G10 reads these tensors
a10 = fe.infer(obs, action_noise=noise)["actions"]

# ---- 2-point grid schedule, captured by the frontend's own capture path
deltas10 = [float(x) for x in fe._deltas]
sched2 = ([fe._action_mods[j] for j in GRID], [fe._head_mods[j] for j in GRID],
          [sum(deltas10[0:9]), deltas10[9]])
fe._action_mods, fe._head_mods, fe._deltas = sched2
d["num_denoise_steps"] = 2
fe._capture_graph()
g2 = fe._graph
a2 = fe.infer(obs, action_noise=noise)["actions"]
fe.stage_inputs(obs, noise=noise)
fe.run_eager()
torch.cuda.synchronize()
a2_eager = np.asarray(fe.read_actions()).copy()
mae = lambda x, y: float(np.abs(np.asarray(x)[..., :6] - np.asarray(y)[..., :6]).mean())

# ---- timing: infer() with G10 / G2 alternated
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def timed_infer() -> float:
    e0.record()
    fe.infer(obs)
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1)


def use(graph, sched, n):
    fe._graph = graph
    fe._action_mods, fe._head_mods, fe._deltas = sched
    d["num_denoise_steps"] = n


t = {"G10": [], "G2": []}
for name, g, s, n in (("G10", g10, sched10, 10), ("G2", g2, sched2, 2)):
    use(g, s, n)
    for _ in range(WARMUP):
        fe.infer(obs)
for r in range(ROUNDS):
    order = (("G10", g10, sched10, 10), ("G2", g2, sched2, 2))
    for name, g, s, n in (order if r % 2 == 0 else order[::-1]):
        use(g, s, n)
        t[name] += [timed_infer() for _ in range(BLOCK)]

# ---- stage graphs (prefill alone, 10- and 2-step loops alone)
st = torch.cuda.Stream()


def graph_of(fn):
    with torch.cuda.stream(st):
        fn(st.cuda_stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        fn(st.cuda_stream)
    return g


def loop(sched, n):
    dd = dict(d, num_denoise_steps=n)
    return lambda s: P.imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, dd, stream=s, attn=fe._attn,
                                             action_mods=sched[0], head_mods=sched[1],
                                             action_rope_table=fe._action_rope_table.data_ptr(), deltas=sched[2])


use(g10, sched10, 10)
fe.stage_inputs(obs, noise=noise)
g_pre = graph_of(lambda s: P.imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, d, stream=s, attn=fe._attn,
                                              mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
                                              rope_table=fe._rope_table.data_ptr()))
g_d10, g_d2 = graph_of(loop(sched10, 10)), graph_of(loop(sched2, 2))


def replay_ms(g, n=200) -> float:
    for _ in range(20):
        g.replay()
    xs = []
    for _ in range(n):
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        xs.append(e0.elapsed_time(e1))
    return float(np.median(xs))


stage = {"prefill_graph_ms": replay_ms(g_pre), "denoise10_graph_ms": replay_ms(g_d10), "denoise2_graph_ms": replay_ms(g_d2)}
p = lambda xs, q: float(np.percentile(xs, q))
res = {"precision": PRECISION, "gpu": torch.cuda.get_device_name(), "x0": d["x0"], "a0": d["a0"],
       "grid": list(GRID), "deltas2": sched2[2], "deltas10": deltas10,
       "infer_ms": {k: {"p10": p(v, 10), "p50": p(v, 50), "p90": p(v, 90), "n": len(v)} for k, v in t.items()},
       "stage_ms": stage,
       "check": {"g2_vs_eager2_max_abs": float(np.abs(np.asarray(a2) - a2_eager).max()),
                 "g2_vs_g10_mae": mae(a2, a10), "g10_vs_eager2_mae": mae(a10, a2_eager)}}
json.dump(res, open(OUT, "w"), indent=1)
i10, i2 = res["infer_ms"]["G10"]["p50"], res["infer_ms"]["G2"]["p50"]
print(f"{PRECISION}: infer P50 10-step {i10:.2f} ms, 2-step {i2:.2f} ms ({i10 / i2:.2f}x, -{i10 - i2:.2f} ms); "
      f"prefill {stage['prefill_graph_ms']:.2f}, denoise10 {stage['denoise10_graph_ms']:.2f}, denoise2 {stage['denoise2_graph_ms']:.2f} ms; "
      f"check {res['check']}")
