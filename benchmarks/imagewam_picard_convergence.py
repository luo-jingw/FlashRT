#!/usr/bin/env python
"""Picard (parallel-in-time) sampling of the ImageWAM ActionDiT on LIBERO frames.

The served sampler is 10 serial Euler steps x_{j+1} = x_j + d_j v_j(x_j).
Picard iteration evaluates all steps of a window in parallel from the
current trajectory guess and updates it by the cumulative sum
x_{j+1} = x_s + sum_{i=s..j} d_i v_i(x_i). Velocities here are evaluated one
step at a time with the served kernels (`imagewam_denoise_step`, step j's
own AdaLN modulation and delta; the velocity is read from
`bufs["velocity"]`), so the iteration count is the algorithm's, not a
kernel's.

Per sample (one dataset frame, one initial noise):
  serial    the 10-step serial result (reference);
  full      window = all 10 steps, initial guess x_j = x_0 for all j; after
            each sweep s = 1..10 the final latent x_10 is denormalized and
            compared with serial (exact at s = 10);
  sliding   ParaDiGMS sliding window, window W in WINDOWS, tolerance tau in
            TAUS: a step is converged when the RMS change of its latent in
            the last sweep is < tau; the window start moves to the first
            unconverged step and new window slots start from the last
            computed latent; reported: sweeps until all 10 steps are
            converged and the deviation of the result from serial.
Reference scales per sample: |serial(noise seed a) - serial(noise seed b)|
(sampling spread) and |serial - dataset action| (first 10 steps).
Deviations: MAE over xyz + rpy (dims 0..5) of the denormalized 64-step chunk,
and gripper sign agreement.

usage: python benchmarks/imagewam_picard_convergence.py PRECISION EPS_PER_TASK STRIDE OUT_JSON
"""
import json
import os
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
from flash_rt.models.imagewam.pipeline_thor import _wrap_fp16  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

PRECISION, EPT, STRIDE, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
SUITE, SEED = "libero_10", 0
WINDOWS = [2, 4, 6, 8, 10]
TAUS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROOT = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")


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
deltas = [float(x) for x in fe._deltas] if fe._deltas is not None else [float(fe.dims["dt"])] * NSTEP


def velocity(x: torch.Tensor, j: int) -> torch.Tensor:
    """v_j(x): one served denoise step at step j from latent x (f32, (NA, AD))."""
    fe._action_latent.copy_(x)
    mod_double, mod_single = fe._action_mods[j]
    st = torch.cuda.current_stream().cuda_stream
    P.imagewam_denoise_step(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe._active_dims, j, stream=st,
                            attn=fe._attn, mod_double=mod_double, mod_single=mod_single,
                            head_mod=fe._head_mods[j], action_rope_table=fe._action_rope_table.data_ptr(),
                            delta=deltas[j])
    return vel_buf.float().clone()


def actions_of(x: torch.Tensor) -> np.ndarray:
    fe._action_latent.copy_(x)
    return np.asarray(fe.read_actions()).copy()


def serial(x0: torch.Tensor) -> torch.Tensor:
    x = x0.clone()
    for j in range(NSTEP):
        x = x + deltas[j] * velocity(x, j)
    return x


def full_window(x0: torch.Tensor) -> list[torch.Tensor]:
    xs = [x0.clone() for _ in range(NSTEP + 1)]
    finals = []
    for _ in range(NSTEP):
        vs = [velocity(xs[j], j) for j in range(NSTEP)]
        new = [x0.clone()]
        for j in range(NSTEP):
            new.append(new[-1] + deltas[j] * vs[j])
        xs = new
        finals.append(xs[NSTEP].clone())
    return finals


def sliding(x0: torch.Tensor, W: int, tau: float) -> tuple[torch.Tensor, int, list[int]]:
    xs = [x0.clone() for _ in range(NSTEP + 1)]
    start, sweeps, widths = 0, 0, []
    while start < NSTEP:
        end = min(start + W, NSTEP)
        vs = [velocity(xs[j], j) for j in range(start, end)]
        sweeps += 1
        widths.append(end - start)
        new = [xs[start]]
        for i, j in enumerate(range(start, end)):
            new.append(new[-1] + deltas[j] * vs[i])
        err = [float((new[i + 1] - xs[start + i + 1]).pow(2).mean().sqrt()) for i in range(end - start)]
        for i in range(end - start):
            xs[start + i + 1] = new[i + 1]
        # advance past the leading run of converged steps (the first step of a
        # window is always exact: its input x_start is final)
        adv = 1
        while adv < end - start and err[adv] < tau:
            adv += 1
        start += adv
        for j in range(end + 1, min(start + W, NSTEP) + 1):
            xs[j] = xs[end].clone()
    return xs[NSTEP], sweeps, widths


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a[..., :6] - b[..., :6]).mean())


def grip_agree(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.sign(a[..., 6] - 0.5) == np.sign(b[..., 6] - 0.5)).mean())


samples = []
t0 = time.time()
for ti, eps in picks.items():
    fe.set_prompt(DEFAULT_PROMPT.format(task=tasks[ti]))
    for e in eps:
        df = pd.read_parquet(_episode_parquet(ROOT, e))
        v1 = decode(_episode_video(ROOT, "observation.images.image", e))
        v2 = decode(_episode_video(ROOT, "observation.images.wrist_image", e))
        gt = np.stack(df["action"].to_numpy()).astype(np.float32)
        for fr in range(0, len(df), STRIDE):
            i = len(samples)
            obs = {"view1": torch.from_numpy(center_crop_resize(v1[fr], 224, 224)),
                   "view2": torch.from_numpy(center_crop_resize(v2[fr], 224, 224)),
                   "proprio": np.array(df["observation.state"].iloc[fr], dtype=np.float32)}
            g = torch.Generator(device="cuda").manual_seed(2 * i)
            xa = 0.01 * torch.randn(NA, AD, generator=g, device="cuda")
            g = torch.Generator(device="cuda").manual_seed(2 * i + 1)
            xb = 0.01 * torch.randn(NA, AD, generator=g, device="cuda")
            fe.stage_inputs(obs, noise=xa)
            fe.run_eager()  # prefill for this frame (and one served pass)
            served = np.asarray(fe.read_actions()).copy()
            ref = actions_of(serial(xa))
            ref_b = actions_of(serial(xb))
            g10 = gt[fr:fr + 10]
            rec = {"task": ti, "episode": e, "frame": fr,
                   "eager_vs_serial_mae": mae(served, ref),
                   "seed_spread_mae": mae(ref, ref_b), "seed_spread_grip": grip_agree(ref, ref_b),
                   "serial_vs_gt_mae": float(np.abs(ref[:len(g10), :6] - g10[:, :6]).mean()),
                   "full": [], "sliding": {}}
            for xf in full_window(xa):
                a = actions_of(xf)
                rec["full"].append([mae(a, ref), grip_agree(a, ref)])
            for W in WINDOWS:
                for tau in TAUS:
                    xs, sw, widths = sliding(xa, W, tau)
                    a = actions_of(xs)
                    rec["sliding"][f"{W}/{tau}"] = [sw, mae(a, ref), grip_agree(a, ref), widths]
            samples.append(rec)
        print(f"task {ti} ep {e}: {len(samples)} samples, {time.time() - t0:.0f}s", flush=True)


def agg(vals: list) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": round(float(a.mean()), 5), "p50": round(float(np.median(a)), 5),
            "p90": round(float(np.percentile(a, 90)), 5), "max": round(float(a.max()), 5)}


summary = {"precision": PRECISION, "suite": SUITE, "eps_per_task": EPT, "stride": STRIDE, "n_samples": len(samples),
           "deltas": deltas, "windows": WINDOWS, "taus": TAUS,
           "eager_vs_serial_mae": agg([s["eager_vs_serial_mae"] for s in samples]),
           "seed_spread_mae": agg([s["seed_spread_mae"] for s in samples]),
           "seed_spread_grip": agg([s["seed_spread_grip"] for s in samples]),
           "serial_vs_gt_mae": agg([s["serial_vs_gt_mae"] for s in samples]),
           "full": [{"sweeps": s + 1, "mae": agg([r["full"][s][0] for r in samples]),
                     "grip": agg([r["full"][s][1] for r in samples])} for s in range(NSTEP)],
           "sliding": {key: {"sweeps": agg([r["sliding"][key][0] for r in samples]),
                             "mae": agg([r["sliding"][key][1] for r in samples]),
                             "grip": agg([r["sliding"][key][2] for r in samples]),
                             "widths_example": samples[0]["sliding"][key][3]}
                       for key in samples[0]["sliding"]}}
json.dump({"summary": summary, "samples": samples}, open(OUT, "w"))
print(json.dumps({k: summary[k] for k in ("n_samples", "eager_vs_serial_mae", "seed_spread_mae", "serial_vs_gt_mae")}))
for r in summary["full"]:
    print("full sweeps", r["sweeps"], "mae", r["mae"], "grip", r["grip"]["mean"])
for k, v in summary["sliding"].items():
    print("sliding", k, "sweeps", v["sweeps"], "mae", v["mae"]["mean"], v["mae"]["p90"], "grip", v["grip"]["mean"])
