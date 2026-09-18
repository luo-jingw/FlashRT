"""A served precision's fidelity to `fp16` on real held-out LIBERO frames.

For each evaluation frame (the end-to-end harness's rule: libero_spatial,
first episode of each task, frames FRAMES), an `fp16` reference frontend
and a candidate frontend (PRECISION, optional CALIBRATION file) run the
captured graph on the same real observation, prompt and initial noise.
Reported per frame and summarized (min / median / mean):

  backbone_hidden   cosine of the prefill output residual (a0 x hidden)
  action_hidden     cosine of the ActionDiT residual after the last
                    denoise step's blocks, before the head (num_action x
                    action_hidden_dim)
  action_latent     cosine of the final flow-matching state, normalized
                    action space (num_action x 7)
  actions           cosine of the denormalized action chunk
  mae_*_vs_gt       mean |action - ground truth| over the chunk, raw units

then `infer()` latency (P10/P50/P90 over ITERS calls, wall clock around
the synchronizing call; indicative only on a shared GPU).

The frontends run one after the other (reference first, outputs kept on
the CPU), so peak memory is one frontend plus Qwen3.

Env: CKPT_PATH, FLUX2_AE_MODEL_PATH, FLUX2_SRC, QWEN3_MODEL_SPEC, DATA_ROOT;
PRECISION (candidate, default fp8_static), CALIBRATION (path, optional),
N_TASKS (10), FRAMES ("0,60"), NOISE_SCALE (1.0 = official N(0,1) sampler;
0.01 = the served infer() convention, issues.md ISSUE-002), ITERS (30).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.libero_dims import LIBERO_HORIZON, LIBERO_REAL_DIMS
from flash_rt.models.imagewam.libero_frames import LiberoFrame, evaluation_frames, load_frame

DEV = "cuda"
BF16 = torch.bfloat16
CKPT = os.environ["CKPT_PATH"]
PRECISION = os.environ.get("PRECISION", "fp8_static")
CALIBRATION = os.environ.get("CALIBRATION") or None
N_TASKS = int(os.environ.get("N_TASKS", "10"))
FRAMES = tuple(int(x) for x in os.environ.get("FRAMES", "0,60").split(","))
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", "1.0"))
ITERS = int(os.environ.get("ITERS", "30"))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def noise_for(i: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(i)
    n = torch.randn((1, LIBERO_HORIZON, 7), generator=g, dtype=torch.float32)
    return n.to(DEV, BF16).float()[0] * NOISE_SCALE


def build(precision: str, calibration: str | None) -> ImageWAMTorchFrontendThor:
    return ImageWAMTorchFrontendThor(
        precision=precision, dims_override=dict(LIBERO_REAL_DIMS), ckpt_path=CKPT,
        ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
        qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
        dataset_stats_path=os.path.join(os.path.dirname(CKPT), "dataset_stats.json"),
        calibration_path=calibration)


def obs_of(fr: LiberoFrame) -> dict:
    return {"view1": torch.from_numpy(fr.view1), "view2": torch.from_numpy(fr.view2), "proprio": fr.state}


@torch.no_grad()
def run(fe: ImageWAMTorchFrontendThor, frames: list[LiberoFrame]) -> list[dict]:
    """Graph replay per frame; returns the four compared tensors on CPU."""
    d = fe.dims
    out = []
    for i, fr in enumerate(frames):
        fe.set_prompt(fr.task)
        fe.stage_inputs(obs_of(fr), noise=noise_for(i))
        fe._graph.replay()
        torch.cuda.synchronize()
        bufs = fe._bufs
        n_act, ahd = d["num_action"], d["action_hidden_dim"]
        interface = {"data": (int(bufs["action_hidden"]), False), "shape": (n_act, ahd),
                     "typestr": "<f2", "version": 3}
        action_hidden = torch.as_tensor(type("_V", (), {"__cuda_array_interface__": interface})(),
                                        device=DEV).float().cpu()
        latent = fe._action_latent.detach().clone()
        out.append(dict(backbone_hidden=fe._backbone_hidden.float().cpu(),
                        action_hidden=action_hidden, action_latent=latent.cpu(),
                        actions=fe._action_norm.backward(latent).cpu()))
    return out


def latency(fe: ImageWAMTorchFrontendThor, fr: LiberoFrame) -> tuple[float, float, float]:
    obs = obs_of(fr)
    for _ in range(5):
        fe.infer(obs)
    ts = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fe.infer(obs)
        ts.append((time.perf_counter() - t0) * 1e3)
    return tuple(float(v) for v in np.percentile(ts, [10, 50, 90]))


def main() -> None:
    refs = evaluation_frames(os.environ["DATA_ROOT"], n_tasks=N_TASKS, frames=FRAMES)
    frames = [load_frame(os.environ["DATA_ROOT"], r, LIBERO_HORIZON) for r in refs]
    print(f"{len(frames)} held-out frames (libero_spatial, frames {FRAMES}); candidate "
          f"precision={PRECISION} calibration={CALIBRATION}; noise scale {NOISE_SCALE}", flush=True)

    fe = build("fp16", None)
    ref_out = run(fe, frames)
    ref_lat = latency(fe, frames[0])
    del fe
    torch.cuda.empty_cache()

    fe = build(PRECISION, CALIBRATION)
    cand_out = run(fe, frames)
    cand_lat = latency(fe, frames[0])

    rows = []
    for fr, r, c in zip(frames, ref_out, cand_out):
        gt = torch.from_numpy(fr.gt)
        n = min(len(gt), LIBERO_HORIZON)
        row = dict(ep=fr.ref.episode, frame=fr.ref.frame,
                   backbone_hidden=cos(c["backbone_hidden"], r["backbone_hidden"]),
                   action_hidden=cos(c["action_hidden"], r["action_hidden"]),
                   action_latent=cos(c["action_latent"], r["action_latent"]),
                   actions=cos(c["actions"], r["actions"]),
                   mae_fp16_vs_gt=(r["actions"][:n] - gt[:n]).abs().mean().item(),
                   mae_cand_vs_gt=(c["actions"][:n] - gt[:n]).abs().mean().item(),
                   finite=bool(torch.isfinite(c["actions"]).all()))
        rows.append(row)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    print(f"\n=== {PRECISION} vs fp16, calibration={CALIBRATION}, noise scale {NOISE_SCALE} ===")
    for col in ("backbone_hidden", "action_hidden", "action_latent", "actions",
                "mae_fp16_vs_gt", "mae_cand_vs_gt"):
        print(f"{col:16s} min={df[col].min():.5f} median={df[col].median():.5f} mean={df[col].mean():.5f}")
    print(f"MAE ratio (candidate / fp16): {df['mae_cand_vs_gt'].mean() / df['mae_fp16_vs_gt'].mean():.3f}")
    print(f"all finite: {bool(df['finite'].all())}")
    print(f"infer() ms  fp16: P10={ref_lat[0]:.1f} P50={ref_lat[1]:.1f} P90={ref_lat[2]:.1f} | "
          f"{PRECISION}: P10={cand_lat[0]:.1f} P50={cand_lat[1]:.1f} P90={cand_lat[2]:.1f}")
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
