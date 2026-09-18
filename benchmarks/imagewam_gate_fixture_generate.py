"""Generate a versioned ImageWAM LIBERO regression-gate fixture.

Produces ``fixture.npz`` + ``manifest.json`` (format:
``flash_rt/datasets/imagewam_gate_fixture.py``) and copies the manifest
into ``tests/fixtures/imagewam_gate/<name>.manifest.json`` for git. The
fixture data itself stays out of git.

Data loading, preprocessing and the official reference are reused from
``imagewam_e2e_official_compare.py`` (same env contract). Two phases, so
only one model is resident at a time:

1. Official ImageWAM (bf16): per task the official Qwen3 context and
   mask; per observation and seed the initial action noise, drawn
   exactly as ``infer_action_flux2(seed=...)`` draws it (CPU generator,
   float32, rounded through bfloat16), and the official normalized
   action chunk.
2. FlashRT ``fp16`` (real checkpoint, real VAE, dataset stats; no Qwen3):
   the served ``infer(obs, action_noise=noise)`` with the stored context,
   renormalized, as the FlashRT fp16 reference.

Required env: as ``imagewam_e2e_official_compare.py`` (FLUX2_SRC,
CKPT_PATH, FLUX2_MODEL_PATH, FLUX2_AE_MODEL_PATH, QWEN3_MODEL_SPEC,
DATA_ROOT, ImageWAM ``src/`` on PYTHONPATH). Fixture v1:

    N_TASKS=10 FRAMES=0,60 SEEDS=0,1 SUITE=libero_spatial \\
    python benchmarks/imagewam_gate_fixture_generate.py \\
        --name imagewam_libero_gate_v1 \\
        --output-dir /home/user1/workspace/jingwu/artifacts/deploy-gates/imagewam_libero_gate_v1
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from imagewam_e2e_official_compare import (
    CKPT,
    FRAMES,
    HORIZON,
    N_TASKS,
    REAL_DIMS,
    SEEDS,
    SHIFT,
    STATS,
    STEPS,
    SUITE,
    build_official,
    center_crop_resize,
    load_samples,
)

from flash_rt.core.parity import parity_metrics
from flash_rt.datasets.imagewam_gate_fixture import GateFixtureStore, ImageWAMGateFixture
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.dataset_stats import load_real_normalizers

DEV = "cuda"
BF16 = torch.bfloat16
VIEW_SIZE = 224
REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "tests" / "fixtures" / "imagewam_gate"


def official_noise(seed: int, action_dim: int) -> torch.Tensor:
    """The initial latent ``infer_action_flux2(seed=seed)`` draws, as float32 on the device."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, HORIZON, action_dim), generator=generator, dtype=torch.float32)
    return noise.to(DEV, BF16).float()[0]


def official_image(view1: np.ndarray, view2: np.ndarray) -> torch.Tensor:
    """Two 224x224 views side by side, [-1, 1], ``[1, 3, 224, 448]``."""
    image = np.concatenate([view1, view2], axis=1)
    return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) * (2.0 / 255.0) - 1.0


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    return {"commit": head, "tracked_changes": bool(dirty)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return parity_metrics(torch.from_numpy(a), torch.from_numpy(b))["cosine"]


def masked_mae(pred_real: np.ndarray, gt: np.ndarray, gt_len: int) -> float:
    return float(np.abs(pred_real[:gt_len] - gt[:gt_len]).mean())


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="fixture name, e.g. imagewam_libero_gate_v1")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", type=Path, default=None,
                        help="copy of the manifest for git (default tests/fixtures/imagewam_gate/<name>.manifest.json)")
    args = parser.parse_args()
    manifest_out = args.manifest_out or MANIFEST_DIR / f"{args.name}.manifest.json"
    if (args.output_dir / "fixture.npz").exists():
        raise FileExistsError(f"{args.output_dir}/fixture.npz exists; fixtures are immutable, pick a new name")

    generator_git = git_state()
    samples = load_samples()
    print(f"samples: {len(samples)} ({SUITE}, tasks {N_TASKS}, frames {FRAMES}, seeds {SEEDS})", flush=True)
    prompts: list[str] = []
    for s in samples:
        if s["task"] not in prompts:
            prompts.append(s["task"])
    n, n_seeds = len(samples), len(SEEDS)
    view1 = np.stack([center_crop_resize(s["v1"], VIEW_SIZE, VIEW_SIZE) for s in samples])
    view2 = np.stack([center_crop_resize(s["v2"], VIEW_SIZE, VIEW_SIZE) for s in samples])
    state = np.stack([s["state"] for s in samples]).astype(np.float32)
    action_dim = int(samples[0]["gt"].shape[1])
    gt_actions = np.full((n, HORIZON, action_dim), np.nan, dtype=np.float32)
    gt_len = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(samples):
        rows = min(len(s["gt"]), HORIZON)
        gt_actions[i, :rows] = s["gt"][:rows]
        gt_len[i] = rows

    state_norm, action_norm = load_real_normalizers(STATS, device=DEV)
    noise = np.stack([np.stack([official_noise(seed, action_dim).cpu().numpy() for seed in SEEDS])
                      for _ in samples]).astype(np.float32)

    # Phase 1: official model.
    t0 = time.time()
    official = build_official()
    print(f"official loaded in {time.time() - t0:.1f}s", flush=True)
    contexts, masks = [], []
    for prompt in prompts:
        ctx, mask = official._prepare_flux2_infer_text(prompt, None, None)
        contexts.append(ctx[0].detach().cpu())
        masks.append(mask[0].detach().cpu().bool())
    official_actions = np.zeros((n, n_seeds, HORIZON, action_dim), dtype=np.float32)
    for i, s in enumerate(samples):
        t = prompts.index(s["task"])
        ctx = contexts[t].unsqueeze(0).to(DEV)
        mask = masks[t].unsqueeze(0).to(DEV)
        proprio = state_norm.forward(torch.as_tensor(state[i], device=DEV).reshape(1, -1))
        image = official_image(view1[i], view2[i])
        for j, seed in enumerate(SEEDS):
            out = official.infer_action_flux2(
                prompt=None, input_image=image, action_horizon=HORIZON, proprio=proprio, context=ctx,
                context_mask=mask, num_inference_steps=STEPS, sigma_shift=SHIFT, seed=seed)["action"]
            official_actions[i, j] = out.float().numpy()
        print(f"official {i + 1}/{n} ep={s['ep']} frame={s['frame']}", flush=True)
    official_peak_gib = torch.cuda.max_memory_allocated() / 2**30
    del official
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Phase 2: FlashRT fp16 served path.
    t0 = time.time()
    fe = ImageWAMTorchFrontendThor(
        precision="fp16", dims_override=dict(REAL_DIMS), ckpt_path=CKPT,
        ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
        dataset_stats_path=STATS)
    print(f"flashrt fp16 constructed in {time.time() - t0:.1f}s", flush=True)
    fp16_reference = np.zeros_like(official_actions)
    current_task = None
    for i, s in enumerate(samples):
        t = prompts.index(s["task"])
        if t != current_task:
            # set_prompt(context=...) caches on (None, True) and would skip a
            # new context; reset the cache key (issues.md ISSUE-060).
            fe._current_prompt = None
            fe.set_prompt(context=contexts[t], context_mask=masks[t])
            current_task = t
        obs = {"view1": torch.from_numpy(view1[i]), "view2": torch.from_numpy(view2[i]), "proprio": state[i]}
        for j in range(n_seeds):
            real = fe.infer(obs, action_noise=torch.from_numpy(noise[i, j]).to(DEV))["actions"]
            fp16_reference[i, j] = action_norm.forward(torch.from_numpy(real)).cpu().numpy()
    flashrt_peak_gib = torch.cuda.max_memory_allocated() / 2**30

    # Summary against the end-to-end baseline (per seed, normalized space).
    def denorm(x: np.ndarray) -> np.ndarray:
        return action_norm.backward(torch.from_numpy(x)).cpu().numpy()

    reference_summary: dict[str, object] = {}
    for j, seed in enumerate(SEEDS):
        cos = [cosine(fp16_reference[i, j], official_actions[i, j]) for i in range(n)]
        mae_fr = [masked_mae(denorm(fp16_reference[i, j]), gt_actions[i], gt_len[i]) for i in range(n)]
        mae_off = [masked_mae(denorm(official_actions[i, j]), gt_actions[i], gt_len[i]) for i in range(n)]
        reference_summary[f"seed_{seed}"] = {
            "fp16_vs_official_median": float(np.median(cos)), "fp16_vs_official_min": float(np.min(cos)),
            "fp16_mae_vs_gt_mean": float(np.mean(mae_fr)), "official_mae_vs_gt_mean": float(np.mean(mae_off))}
    spread = [cosine(official_actions[i, 0], official_actions[i, -1]) for i in range(n)]
    reference_summary["official_seed_spread_median"] = float(np.median(spread))
    reference_summary["official_seed_spread_min"] = float(np.min(spread))
    print(json.dumps(reference_summary, indent=1), flush=True)

    fixture = ImageWAMGateFixture(
        view1=view1, view2=view2, state=state,
        task_index=np.array([prompts.index(s["task"]) for s in samples], dtype=np.int64),
        episode=np.array([s["ep"] for s in samples], dtype=np.int64),
        frame=np.array([s["frame"] for s in samples], dtype=np.int64),
        gt_actions=gt_actions, gt_len=gt_len, prompts=np.array(prompts),
        context_bf16_bits=torch.stack(contexts).to(BF16).view(torch.int16).numpy().view(np.uint16),
        context_mask=torch.stack(masks).numpy(),
        seeds=np.array(SEEDS, dtype=np.int64), noise=noise,
        official_actions=official_actions, fp16_reference_actions=fp16_reference)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/imagewam_gate_fixture_generate.py",
        "git": generator_git,
        "source": {"dataset": "yuanty/LIBERO-fastwam (LeRobot v2.1)", "suite": SUITE, "n_tasks": N_TASKS,
                   "frames": FRAMES, "seeds": SEEDS, "views": "observation.images.image, "
                   "observation.images.wrist_image; official center-crop resize to 224x224 each"},
        "sampler": {"horizon": HORIZON, "num_inference_steps": STEPS, "sigma_shift": SHIFT,
                    "noise": "torch.randn((1,H,A), CPU generator seeded per seed, float32) -> bfloat16 -> float32"},
        "checkpoint": {"path": CKPT, "bytes": os.path.getsize(CKPT), "sha256": file_sha256(CKPT)},
        "dataset_stats": {"path": STATS, "sha256": file_sha256(STATS)},
        "official": {"dtype": "bfloat16", "qwen3_model_spec": os.environ.get("QWEN3_MODEL_SPEC", "")},
        "fp16_reference": {"precision": "fp16", "frontend": "ImageWAMTorchFrontendThor.infer(action_noise=...)",
                           "dims": {k: v for k, v in REAL_DIMS.items()}},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "peak_gib": {"official": round(official_peak_gib, 2), "flashrt_fp16": round(flashrt_peak_gib, 2)},
        "reference_summary": reference_summary,
    }
    manifest = GateFixtureStore(args.output_dir).save(fixture, args.name, metadata)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.write(manifest_out)
    print(f"fixture: {args.output_dir}/fixture.npz ({manifest.files['fixture.npz'].bytes / 2**20:.1f} MiB)")
    print(f"manifest: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
