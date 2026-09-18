#!/usr/bin/env python
"""ImageWAM VAE stage: profile and in-process A/B of every variant
(roadmap items 2 and 5, `plan.md`).

Sections (`--section`, default `all`):

  profile     torch-profiler op table of the real `AutoEncoder.encode` at
              the real 224x448 input, plus kernels per encode.
  preprocess  served torch `_prep_view` + cat vs the fused
              `imagewam_vae_preprocess_bf16` kernel, for raw 512x512 and
              pre-resized 224x224 views, CPU and GPU uint8 inputs.

Timing: variants are called round-robin in one process. Each sample is
wall-clock around one call with `torch.cuda.synchronize()` on both
sides (so CPU-side work such as host dtype conversion counts), and
P10/P50/P90 over `--iters` samples are printed per variant. On a shared
GPU these numbers are indicative only.

Required env: FLUX2_SRC, AE_MODEL_PATH (or FLUX2_AE_MODEL_PATH).
Optional env: DATA_ROOT (LIBERO-fastwam; real frames, else synthetic).
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Callable

import numpy as np
import torch
from PIL import Image

from flash_rt.models.imagewam.vae_encoder import _prep_view, encode_to_tokens, load_real_ae
from flash_rt.models.imagewam.vae_preprocess import VaePreprocessor

DEV = "cuda"
BF16 = torch.bfloat16


def _flux2_src() -> str:
    src = os.environ["FLUX2_SRC"]
    return os.path.join(src, "src") if os.path.isdir(os.path.join(src, "src", "flux2")) else src


def _ae_path() -> str:
    return os.environ.get("AE_MODEL_PATH") or os.environ["FLUX2_AE_MODEL_PATH"]


def load_views() -> tuple[np.ndarray, np.ndarray, str]:
    """Two real 512x512 LIBERO views (agent + wrist, episode 0 frame 0),
    or synthetic smooth frames when DATA_ROOT is not set."""
    root = os.environ.get("DATA_ROOT")
    if root:
        import av
        base = os.path.join(root, "libero_spatial_no_noops_lerobot", "videos", "chunk-000")
        frames = []
        for key in ("observation.images.image", "observation.images.wrist_image"):
            with av.open(os.path.join(base, key, "episode_000000.mp4")) as c:
                for f in c.decode(video=0):
                    frames.append(f.to_ndarray(format="rgb24"))
                    break
        return frames[0], frames[1], "real LIBERO"
    y, x = np.meshgrid(np.linspace(0, 1, 512), np.linspace(0, 1, 512), indexing="ij")
    base = (np.stack([x, y, 1 - x], axis=-1) * 255).astype(np.uint8)
    return base, base[::-1].copy(), "synthetic"


def center_crop_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Official eval `_center_crop_resize` (PIL bilinear + center crop)."""
    pil = Image.fromarray(img)
    sw, sh = pil.size
    scale = max(w / sw, h / sh)
    r = pil.resize((round(sw * scale), round(sh * scale)), resample=Image.BILINEAR)
    rw, rh = r.size
    left, top = (rw - w) // 2, (rh - h) // 2
    return np.array(r.crop((left, top, left + w, top + h)))


def ab_time(variants: dict[str, Callable[[], object]], iters: int, warmup: int = 5) -> None:
    for fn in variants.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {k: [] for k in variants}
    for _ in range(iters):
        for name, fn in variants.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - t0) * 1e3)
    for name, ts in samples.items():
        p10, p50, p90 = np.percentile(ts, [10, 50, 90])
        print(f"  {name:48s} P10={p10:8.3f}  P50={p50:8.3f}  P90={p90:8.3f} ms  (n={len(ts)})")


def gpu_kernels(fn: Callable[[], object], n: int = 20) -> tuple[float, float, float]:
    """(kernels per call, summed GPU kernel time per call in ms, host
    enqueue time per call in ms). Kernel time from the torch profiler,
    so it excludes the idle gaps a shared GPU adds between kernels;
    enqueue time is wall-clock of `n` calls issued back to back."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    enqueue_ms = (time.perf_counter() - t0) * 1e3 / n
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
    return len(kernels) / n, sum(e.device_time for e in kernels) / n / 1e3, enqueue_ms


def report_gpu(variants: dict[str, Callable[[], object]]) -> None:
    for name, fn in variants.items():
        k, gpu_ms, enq_ms = gpu_kernels(fn)
        print(f"  {name:48s} kernels={k:6.1f}  gpu_kernel_time={gpu_ms:8.3f} ms  host_enqueue={enq_ms:8.3f} ms")


def section_profile(ae: torch.nn.Module, v1: np.ndarray, v2: np.ndarray) -> None:
    from torch.profiler import ProfilerActivity, profile
    x = torch.cat([_prep_view(torch.from_numpy(v).to(DEV), (224, 224), DEV, BF16) for v in (v1, v2)], dim=-1)
    print(f"\n=== profile: AutoEncoder.encode, input {tuple(x.shape)} {x.dtype} ===")
    with torch.no_grad():
        for _ in range(5):
            ae.encode(x)
        torch.cuda.synchronize()
        n = 10
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                ae.encode(x)
            torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
    total_us = sum(e.device_time for e in kernels) / n
    print(f"CUDA kernels per encode: {len(kernels) / n:.0f}; GPU time per encode: {total_us / 1e3:.3f} ms")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=90))


def section_preprocess(ae: torch.nn.Module, v1: np.ndarray, v2: np.ndarray, iters: int) -> None:
    pre = VaePreprocessor(resize="area")
    inputs = {
        "raw 512x512 CPU": (torch.from_numpy(v1), torch.from_numpy(v2)),
        "raw 512x512 GPU": (torch.from_numpy(v1).to(DEV), torch.from_numpy(v2).to(DEV)),
        "pre-resized 224x224 CPU": (torch.from_numpy(center_crop_resize(v1, 224, 224)),
                                    torch.from_numpy(center_crop_resize(v2, 224, 224))),
    }
    for label, (a, b) in inputs.items():
        def torch_prep() -> torch.Tensor:
            return torch.cat([_prep_view(a, (224, 224), DEV, BF16), _prep_view(b, (224, 224), DEV, BF16)], dim=-1)

        def kernel_prep() -> torch.Tensor:
            out = torch.empty(1, 3, 224, 448, dtype=BF16, device=DEV)
            pre.run([a.to(DEV).contiguous(), b.to(DEV).contiguous()], out, torch.cuda.current_stream().cuda_stream)
            return out

        same = torch.equal(torch_prep(), kernel_prep())
        print(f"\n=== preprocess, {label}: kernel bit-identical to _prep_view: {same} ===")
        variants = {"torch _prep_view x2 + cat (served before)": torch_prep,
                    "fused kernel (vae_preprocess)": kernel_prep}
        ab_time(variants, iters)
        report_gpu(variants)
        with torch.no_grad():
            t_old = encode_to_tokens(ae, a, b)
            t_new = encode_to_tokens(ae, a, b, preprocessor=pre)
        print(f"  encode_to_tokens tokens bit-identical (legacy vs kernel preprocess): {torch.equal(t_old, t_new)}")
        with torch.no_grad():
            ab_time({"encode_to_tokens, torch preprocess": lambda: encode_to_tokens(ae, a, b),
                     "encode_to_tokens, kernel preprocess": lambda: encode_to_tokens(ae, a, b, preprocessor=pre)},
                    max(iters // 4, 10))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=["all", "profile", "preprocess"])
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}")
    ae = load_real_ae(_ae_path(), _flux2_src())
    v1, v2, src = load_views()
    print(f"views: {src} {v1.shape} + {v2.shape}")
    if args.section in ("all", "profile"):
        section_profile(ae, v1, v2)
    if args.section in ("all", "preprocess"):
        section_preprocess(ae, v1, v2, args.iters)


if __name__ == "__main__":
    main()
