"""Thor check for the `e0m3_hadamard` precision tier vs `nvfp4` and `fp16`.

Runs the served `ImageWAMTorchFrontendThor.infer()` (real checkpoint,
real VAE with the default preprocessing, real Qwen3 context, real
proprio, 10-step shift schedule, merged single-stream `linear1`/`linear2`)
on real LIBERO frames, first as the `fp16` reference and then once per
compared precision, with the same N(0,1) initial action noise for every
run (`infer(action_noise=...)`), and reports per compared precision:

  bh_cos    backbone_hidden (prefill output) cosine vs fp16
  al_cos    action_latent (normalized actions) cosine vs fp16
  act_cos   denormalized 64-step actions cosine vs fp16
  mae_gt    mean |actions - ground truth| over the chunk (open loop)

then `infer()` latency for every compared precision, measured in one
process with the calls interleaved (P10/P50/P90 of wall time around
`infer()`, which includes the VAE encode, plus CUDA-event time of the
graph replay alone).

Env (all optional): PRECISIONS ("nvfp4,e0m3_hadamard"; the compared
precisions, `fp16` allowed as a sanity row), SUITE (libero_spatial),
N_TASKS (10), FRAMES ("0,20,40,60,80"), LAT_ITERS (50). Attention
follows the frontend's own rule (`use_fa4=None`: cuBLAS unless
`FLASHRT_THOR_FA4=1`); the resolved choice is printed.

Expected from the H100 simulation (`opportunities.md` OPT-024, merged
`linear2`, 20 frames): `e0m3_hadamard` actions cosine vs fp16 about
0.9997 and `backbone_hidden` about 0.9996, both above `nvfp4` (about
0.9993 and 0.998), and open-loop MAE vs ground truth about equal to
fp16's.
Required env: CKPT_PATH (dataset_stats.json beside it),
FLUX2_AE_MODEL_PATH, FLUX2_SRC, QWEN3_MODEL_SPEC, DATA_ROOT, and the
ImageWAM `src/` plus FLUX2_SRC/src on PYTHONPATH.

Frontends are built one at a time; the reference frontend is freed
after its frames are collected, the compared ones are kept for the
latency section.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ["FLUX2_SRC"] + "/src")
sys.path.insert(0, os.environ["FLUX2_SRC"])

from _imagewam_libero_frames import LiberoFrame, load_libero_frames  # noqa: E402

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor  # noqa: E402
from flash_rt.models.imagewam.text_encoder import encode_prompts, load_real_text_encoder  # noqa: E402

DEV = "cuda"
BF16 = torch.bfloat16
CKPT = os.environ["CKPT_PATH"]
STATS = os.path.join(os.path.dirname(CKPT), "dataset_stats.json")
PRECISIONS = os.environ.get("PRECISIONS", "nvfp4,e0m3_hadamard").split(",")
SUITE = os.environ.get("SUITE", "libero_spatial")
N_TASKS = int(os.environ.get("N_TASKS", "10"))
FRAMES = [int(x) for x in os.environ.get("FRAMES", "0,20,40,60,80").split(",")]
LAT_ITERS = int(os.environ.get("LAT_ITERS", "50"))
HORIZON, STEPS, SHIFT = 64, 10, 5.0

REAL_DIMS = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=HORIZON, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=1.0 / STEPS, num_denoise_steps=STEPS,
    ref_h=14, ref_w=28, proprio_dim=8, shift=SHIFT, num_train_timesteps=1000,
)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def noise_for(frame_idx: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(frame_idx)
    return torch.randn((1, HORIZON, 7), generator=g, dtype=torch.float32).to(DEV, BF16).float()[0]


def build(precision: str) -> ImageWAMTorchFrontendThor:
    t = time.time()
    fe = ImageWAMTorchFrontendThor(
        precision=precision, dims_override=dict(REAL_DIMS), ckpt_path=CKPT,
        ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
        dataset_stats_path=STATS)
    print(f"[{precision}] constructed in {time.time() - t:.1f}s; use_fa4={fe.use_fa4} "
          f"merge_linear2={fe.dims.get('merge_linear2')}", flush=True)
    return fe


def observation(f: LiberoFrame) -> dict:
    return {"view1": torch.from_numpy(f.view1.copy()), "view2": torch.from_numpy(f.view2.copy()),
            "proprio": f.state.copy()}


def run_frame(fe: ImageWAMTorchFrontendThor, ctx: dict, f: LiberoFrame, idx: int) -> dict:
    """Served `infer()` with the initial action noise fixed per frame, so
    every precision sees identical inputs. `set_prompt` caches on
    `(prompt_text, context given)`, so the cache is cleared to load each
    task's context."""
    c, msk = ctx[f.task]
    fe._current_prompt = None
    fe.set_prompt(context=c, context_mask=msk)
    act = torch.from_numpy(fe.infer(observation(f), action_noise=noise_for(idx))["actions"]).float()
    return dict(bh=fe._backbone_hidden.detach().float().cpu(),
                al=fe._action_latent.detach().float().cpu(), act=act)


def latency(fes: dict, f: LiberoFrame) -> dict:
    obs = observation(f)
    wall = {p: [] for p in fes}
    replay = {p: [] for p in fes}
    for fe in fes.values():
        for _ in range(5):
            fe.infer(obs)
    for _ in range(LAT_ITERS):
        for p, fe in fes.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fe.infer(obs)
            wall[p].append((time.perf_counter() - t0) * 1e3)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fe._graph.replay()
            e1.record()
            torch.cuda.synchronize()
            replay[p].append(e0.elapsed_time(e1))
    out = {}
    for p in fes:
        w, r = np.array(wall[p]), np.array(replay[p])
        out[p] = dict(infer_p10=float(np.percentile(w, 10)), infer_p50=float(np.percentile(w, 50)),
                      infer_p90=float(np.percentile(w, 90)), replay_p10=float(np.percentile(r, 10)),
                      replay_p50=float(np.percentile(r, 50)), replay_p90=float(np.percentile(r, 90)))
    return out


@torch.no_grad()
def main() -> None:
    frames = load_libero_frames(os.environ["DATA_ROOT"], SUITE, N_TASKS, FRAMES, HORIZON)
    print(f"device: {torch.cuda.get_device_name(0)}; frames: {len(frames)} ({SUITE}, {FRAMES})", flush=True)
    q_model, q_tok = load_real_text_encoder(os.environ["QWEN3_MODEL_SPEC"])
    ctx = {}
    for task in sorted({f.task for f in frames}):
        c, msk = encode_prompts(q_model, q_tok, [task])
        ctx[task] = (c[0].clone(), msk[0].clone())
    del q_model, q_tok
    gc.collect()
    torch.cuda.empty_cache()

    fe = build("fp16")
    ref = [run_frame(fe, ctx, f, i) for i, f in enumerate(frames)]
    del fe
    gc.collect()
    torch.cuda.empty_cache()
    outputs, keep = {"fp16 (reference)": ref}, {}
    for prec in PRECISIONS:
        fe = build(prec)
        outputs[prec] = [run_frame(fe, ctx, f, i) for i, f in enumerate(frames)]
        keep[prec] = fe
        print(f"[{prec}] frames done; allocated {torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB", flush=True)

    rows = {}
    print("\nper precision (median / min over frames)")
    print(f"{'precision':18s}{'bh_cos':>18s}{'al_cos':>18s}{'act_cos':>18s}{'mae_gt':>9s}{'finite':>8s}")
    for prec in outputs:
        r = []
        for i, f in enumerate(frames):
            o = outputs[prec][i]
            gt = torch.from_numpy(f.gt)
            n_gt = min(len(gt), HORIZON)
            r.append(dict(ep=f.episode, frame=f.frame, bh_cos=cos(o["bh"], ref[i]["bh"]),
                          al_cos=cos(o["al"], ref[i]["al"]), act_cos=cos(o["act"], ref[i]["act"]),
                          mae_gt=float((o["act"][:n_gt] - gt[:n_gt]).abs().mean()),
                          finite=bool(torch.isfinite(o["act"]).all())))
        rows[prec] = r
        a = {k: np.array([x[k] for x in r]) for k in ("bh_cos", "al_cos", "act_cos", "mae_gt")}
        print(f"{prec:18s}" + "".join(f"{np.median(a[k]):9.5f}/{a[k].min():8.5f}" for k in ("bh_cos", "al_cos", "act_cos"))
              + f"{a['mae_gt'].mean():9.5f}{str(all(x['finite'] for x in r)):>8s}")
    lat = latency(keep, frames[0])
    if lat:
        print(f"\ninfer() latency, {LAT_ITERS} interleaved iterations (ms)")
        for p, s in lat.items():
            print(f"{p:16s} infer P10/P50/P90 = {s['infer_p10']:.1f}/{s['infer_p50']:.1f}/{s['infer_p90']:.1f}   "
                  f"graph replay P10/P50/P90 = {s['replay_p10']:.1f}/{s['replay_p50']:.1f}/{s['replay_p90']:.1f}")
    print("\nJSON " + json.dumps(dict(frames=[(f.episode, f.frame) for f in frames],
                                      use_fa4={p: fe.use_fa4 for p, fe in keep.items()},
                                      per_frame=rows, latency=lat)))


if __name__ == "__main__":
    main()
