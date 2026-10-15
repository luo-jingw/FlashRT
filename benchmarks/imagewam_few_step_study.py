#!/usr/bin/env python
"""Fewer ActionDiT denoise steps on the served 10-step grid (ImageWAM, LIBERO frames).

The velocity field is close to v_t(x) = (x - x0_hat) / t with x0_hat nearly
independent of x (benchmarks/imagewam_velocity_jacobian_study.py), for which
an Euler step from t to t' is exact when x0_hat does not change. This script
measures how much x0_hat changes along the trajectory and what step subsets
of the served grid give.

Per sample (served noise 0.01 * N(0, 1)):
  x0hat_j  actions of x_j - t_j * v_j(x_j) on the serial trajectory, j = 0..9
           (t_j = 1 - sum_{i<j} |d_i|), MAE vs the serial result and vs the
           dataset actions (first 10 steps);
  subsets  Euler over a subset S of the 10 grid points: from grid point s to
           the next point of S (or t = 0), velocity evaluated with step s's own
           conditioning, step size t_next - t_s.
MAE over xyz + rpy of the denormalized 64-step chunk; gripper sign agreement.

usage: python benchmarks/imagewam_few_step_study.py PRECISION EPS_PER_TASK STRIDE OUT_JSON
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
from flash_rt.models.imagewam.pipeline_thor import _wrap_fp16  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

PRECISION, EPT, STRIDE, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
SUITE, SEED = "libero_10", 2
SUBSETS = {"10": list(range(10)), "5": [0, 2, 4, 6, 8], "5b": [0, 3, 5, 7, 8], "4": [0, 3, 6, 8], "3": [0, 4, 7],
           "3b": [0, 6, 8], "2": [0, 6], "2b": [0, 8], "1": [0],
           "3c": [0, 5, 9], "3d": [0, 7, 9], "2c": [0, 9], "2d": [5, 9], "1b": [9], "1c": [8]}
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROOT = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")
DEV = "cuda"


def decode(path: str) -> list:
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


tasks = {t["task_index"]: t["task"] for t in _read_jsonl(f"{ROOT}/meta/tasks.jsonl")}
by_task: dict[int, list[int]] = {}
for ep in _read_jsonl(f"{ROOT}/meta/episodes.jsonl"):
    e = int(ep["episode_index"])
    ti = int(pd.read_parquet(_episode_parquet(ROOT, e), columns=["task_index"])["task_index"].iloc[0])
    by_task.setdefault(ti, []).append(e)
rng = np.random.default_rng(SEED)
picks = {ti: sorted(rng.choice(eps, EPT, replace=False).tolist()) for ti, eps in sorted(by_task.items())}

fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
NA, AD = fe.dims["num_action"], fe.dims["action_dim"]
NSTEP = fe.dims["num_denoise_steps"]
vel_buf = _wrap_fp16(fe._bufs["velocity"], NA, AD)
deltas = [float(x) for x in fe._deltas]
tgrid = [1.0 - sum(abs(d) for d in deltas[:j]) for j in range(NSTEP)] + [0.0]
sign = -1.0 if deltas[0] < 0 else 1.0


def velocity(x: torch.Tensor, j: int) -> torch.Tensor:
    fe._action_latent.copy_(x)
    mod_double, mod_single = fe._action_mods[j]
    P.imagewam_denoise_step(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe._active_dims, j,
                            stream=torch.cuda.current_stream().cuda_stream, attn=fe._attn,
                            mod_double=mod_double, mod_single=mod_single, head_mod=fe._head_mods[j],
                            action_rope_table=fe._action_rope_table.data_ptr(), delta=deltas[j])
    return vel_buf.float().clone()


def actions_of(x: torch.Tensor) -> np.ndarray:
    fe._action_latent.copy_(x)
    return np.asarray(fe.read_actions()).copy()


def run_subset(x0: torch.Tensor, S: list[int]) -> torch.Tensor:
    x = x0.clone()
    for i, s in enumerate(S):
        t_next = tgrid[S[i + 1]] if i + 1 < len(S) else 0.0
        x = x + sign * (tgrid[s] - t_next) * velocity(x, s)
    return x


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a[..., :6] - b[..., :6]).mean())


def grip(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.sign(a[..., 6] - 0.5) == np.sign(b[..., 6] - 0.5)).mean())


recs = []
for ti, eps in picks.items():
    fe.set_prompt(DEFAULT_PROMPT.format(task=tasks[ti]))
    for e in eps:
        df = pd.read_parquet(_episode_parquet(ROOT, e))
        v1 = decode(_episode_video(ROOT, "observation.images.image", e))
        v2 = decode(_episode_video(ROOT, "observation.images.wrist_image", e))
        gt = np.stack(df["action"].to_numpy()).astype(np.float32)
        for fr in range(0, len(df), STRIDE):
            obs = {"view1": torch.from_numpy(center_crop_resize(v1[fr], 224, 224)),
                   "view2": torch.from_numpy(center_crop_resize(v2[fr], 224, 224)),
                   "proprio": np.array(df["observation.state"].iloc[fr], dtype=np.float32)}
            g = torch.Generator(device=DEV).manual_seed(len(recs))
            x0 = 0.01 * torch.randn(NA, AD, generator=g, device=DEV)
            fe.stage_inputs(obs, noise=x0)
            fe.run_eager()
            served = np.asarray(fe.read_actions()).copy()
            xs = [x0.clone()]
            hat = []
            for j in range(NSTEP):
                v = velocity(xs[-1], j)
                hat.append(actions_of(xs[-1] - tgrid[j] * v))
                xs.append(xs[-1] + deltas[j] * v)
            ref = actions_of(xs[-1])
            g10 = gt[fr:fr + 10]
            rec = {"task": ti, "episode": e, "frame": fr, "served_vs_serial": mae(served, ref),
                   "serial_vs_gt": float(np.abs(ref[:len(g10), :6] - g10[:, :6]).mean()),
                   "x0hat_vs_serial": [mae(h, ref) for h in hat],
                   "x0hat_vs_gt": [float(np.abs(h[:len(g10), :6] - g10[:, :6]).mean()) for h in hat], "subsets": {}}
            for name, S in SUBSETS.items():
                a = actions_of(run_subset(x0, S))
                rec["subsets"][name] = [mae(a, ref), grip(a, ref), float(np.abs(a[:len(g10), :6] - g10[:, :6]).mean())]
            recs.append(rec)
        print(f"task {ti} ep {e}: {len(recs)} samples", flush=True)

m = lambda xs: round(float(np.mean(xs)), 6)
summary = {"precision": PRECISION, "n": len(recs), "tgrid": [round(t, 4) for t in tgrid], "subsets": SUBSETS,
           "served_vs_serial": m([r["served_vs_serial"] for r in recs]),
           "serial_vs_gt": m([r["serial_vs_gt"] for r in recs]),
           "x0hat_vs_serial": [m([r["x0hat_vs_serial"][j] for r in recs]) for j in range(NSTEP)],
           "x0hat_vs_gt": [m([r["x0hat_vs_gt"][j] for r in recs]) for j in range(NSTEP)],
           "subset": {n: {"mae_vs_serial": m([r["subsets"][n][0] for r in recs]),
                          "mae_vs_serial_p90": round(float(np.percentile([r["subsets"][n][0] for r in recs], 90)), 6),
                          "grip_agree": m([r["subsets"][n][1] for r in recs]),
                          "mae_vs_gt": m([r["subsets"][n][2] for r in recs])} for n in SUBSETS}}
json.dump({"summary": summary, "samples": recs}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1))
