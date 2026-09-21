#!/usr/bin/env python
"""Steady-state latency of the OFFICIAL ImageWAM torch implementation.

The baseline row of the final result tables (`benchmarks/imagewam_result_table.py`).
It times the same call boundary as FlashRT's `infer()`: camera frames and
proprio in, the action chunk out, with the text context already encoded
(passed as `context=` / `context_mask=`, so the Qwen3 text encoder is outside
the timed region, as `set_prompt` puts it outside FlashRT's). The image
encode, the proprio projection, the backbone prefill and the denoise loop are
inside it. bf16 eager, `torch.no_grad()`, no `torch.compile` (inductor does not
compile on Thor and `cudagraphs` is slower, THOR_STATUS_SUMMARY.md).

The inputs are random by design (a random image in the official's [-1, 1]
pixel range, a random normalized proprio vector, a fixed instruction), like
`benchmarks/imagewam_thor_path_bench.py`: latency does not depend on their
values. The official model masks the padded text keys but still runs them, so
the instruction's own length does not change its cost.

    CKPT_PATH=... FLUX2_SRC=... FLUX2_MODEL_PATH=... FLUX2_AE_MODEL_PATH=... \\
    QWEN3_MODEL_SPEC=... python benchmarks/imagewam_official_torch_bench.py \\
        --workload libero

The workload flags are `benchmarks/_imagewam_workload_cli.py`'s: the step count,
the shift, the horizon and the image size come from the named workload, so the
same script produces the baseline row of every table.

Prints one line, `official P10=.. P50=.. P90=.. ms (n=..)`, in the format of
the path bench's rows, after the workload and the Jetson clock state.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _imagewam_workload_cli import add_workload_args, workload_from_args  # noqa: E402
from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state  # noqa: E402

DEV = "cuda"
DEFAULT_PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
REQUIRED_ENV = ("CKPT_PATH", "FLUX2_SRC", "FLUX2_MODEL_PATH", "FLUX2_AE_MODEL_PATH", "QWEN3_MODEL_SPEC")


def percentiles(ms: list[float]) -> tuple[float, float, float]:
    q = np.percentile(np.asarray(ms), (10, 50, 90))
    return float(q[0]), float(q[1]), float(q[2])


def build_official(ckpt: str):
    """The official model, as `benchmarks/imagewam_e2e_official_compare.py:build_official` builds it."""
    from omegaconf import OmegaConf
    from imagewam.runtime import create_imagewam_flux2_klein

    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt), "config.yaml")).model
    m = create_imagewam_flux2_klein(
        flux2_model_path=os.environ["FLUX2_MODEL_PATH"], ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"],
        flux2_src_path=os.environ["FLUX2_SRC"], variant=cfg.variant,
        qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"], load_text_encoder=True,
        action_dit_config=cfg.action_dit_config, action_dit_pretrained_path=None,
        proprio_dim=cfg.proprio_dim, video_scheduler=cfg.video_scheduler,
        action_scheduler=cfg.action_scheduler, loss=cfg.loss,
        mot_checkpoint_mixed_attn=cfg.mot_checkpoint_mixed_attn,
        mot_gqa_implementation=cfg.mot_gqa_implementation,
        mot_force_flash_attention=cfg.mot_force_flash_attention,
        pack_proprio_after_text=cfg.pack_proprio_after_text, device=DEV)
    m.load_checkpoint(ckpt)
    return m.eval()


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_workload_args(ap)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="the instruction, encoded once outside the timing")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="seed of the random image and proprio, and of the sampler")
    args = ap.parse_args()
    if args.iters <= 0 or args.warmup < 0:
        ap.error("--iters must be > 0 and --warmup >= 0")
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        ap.error(f"missing environment: {', '.join(missing)}")
    sys.path.insert(0, os.environ["FLUX2_SRC"] + "/src")
    sys.path.insert(0, os.environ["FLUX2_SRC"])

    workload = workload_from_args(args)
    print(f"workload: {workload}", flush=True)
    report_jetson_clock_state()

    t = time.time()
    off = build_official(os.environ["CKPT_PATH"])
    print(f"official loaded in {time.time() - t:.1f}s", flush=True)

    # The text context: encoded once, outside the timed region.
    ctx, mask = off._prepare_flux2_infer_text(args.prompt, None, None)
    torch.cuda.synchronize()

    # The views concatenated along the width in the official's [-1, 1] pixel range: (1, 3, H, num_views * W).
    rng = np.random.default_rng(args.seed)
    img = rng.integers(0, 256, (workload.image_h, workload.num_views * workload.image_w, 3), dtype=np.uint8)
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) * (2.0 / 255.0) - 1.0
    proprio = torch.from_numpy(rng.uniform(-0.5, 0.5, (1, workload.proprio_dim)).astype(np.float32)).to(DEV)

    def call() -> torch.Tensor:
        return off.infer_action_flux2(
            prompt=None, input_image=x, action_horizon=workload.action_horizon, proprio=proprio,
            context=ctx, context_mask=mask, num_inference_steps=workload.num_steps,
            sigma_shift=workload.shift, seed=args.seed)["action"]

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    ms = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = call()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3)
    if not torch.isfinite(out.float()).all():
        print("official output is not finite", file=sys.stderr)
        return 1
    p10, p50, p90 = percentiles(ms)
    print(f"official P10={p10:.2f} P50={p50:.2f} P90={p90:.2f} ms (n={len(ms)}) "
          f"steps={workload.num_steps} horizon={workload.action_horizon}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
