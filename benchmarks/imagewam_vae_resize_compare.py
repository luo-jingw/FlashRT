#!/usr/bin/env python
"""VAE-token effect of the image preprocessing choices (issues.md ISSUE-030).

For the same LIBERO frames as `imagewam_e2e_official_compare.py`
(`SUITE`, `N_TASKS`, `FRAMES`), encodes the two camera views through the
real FLUX.2 VAE after four preprocessing chains and prints token cosine
statistics (min / median / max over frames) for each pair:

  area      served default: `VaePreprocessor(resize="area")`
            (torch area resize, float32 normalization table)
  pil       `VaePreprocessor(resize="pil_bilinear")`: the official eval's
            PIL BILINEAR center-crop resize, served float32 normalization
  official  official LIBERO eval (`eval_libero_single._obs_to_model_input`):
            PIL resize, then `x * (2/255) - 1` computed in BF16
  training  training transform (`config.yaml` processor): uint8 / 255 in
            float32, `torchvision.transforms.Resize([224, 224])`
            (bilinear, antialias), `Normalize(0.5, 0.5)`, then BF16

`--proxy-size S` first PIL-downscales each 512x512 dataset frame to SxS
(BILINEAR), a stand-in for a simulator rendering at SxS (the official
eval renders at 256x256, `libero_utils.LIBERO_ENV_RESOLUTION`).

Required env: FLUX2_SRC, AE_MODEL_PATH (or FLUX2_AE_MODEL_PATH), DATA_ROOT.
"""
from __future__ import annotations

import argparse
import json
import os

import av
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from flash_rt.models.imagewam.vae_encoder import encode_to_tokens, load_real_ae
from flash_rt.models.imagewam.vae_preprocess import VaePreprocessor

DEV = "cuda"
BF16 = torch.bfloat16
SUITE = os.environ.get("SUITE", "libero_spatial")
N_TASKS = int(os.environ.get("N_TASKS", "10"))
FRAMES = [int(x) for x in os.environ.get("FRAMES", "0,60").split(",")]


def read_frame(path: str, idx: int) -> np.ndarray:
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == idx:
                return f.to_ndarray(format="rgb24")
    raise IndexError(idx)


def load_views() -> list[tuple[np.ndarray, np.ndarray]]:
    """Same task/episode/frame selection as imagewam_e2e_official_compare.load_samples."""
    root = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")
    eps = [json.loads(line) for line in open(f"{root}/meta/episodes.jsonl")]
    seen, out = set(), []
    for ep in eps:
        e = ep["episode_index"]
        df = pd.read_parquet(f"{root}/data/chunk-000/episode_{e:06d}.parquet")
        ti = int(df["task_index"].iloc[0])
        if ti in seen:
            continue
        seen.add(ti)
        for fr in FRAMES:
            if fr >= len(df):
                continue
            out.append(tuple(read_frame(f"{root}/videos/chunk-000/observation.images.{k}/episode_{e:06d}.mp4", fr)
                             for k in ("image", "wrist_image")))
        if len(seen) >= N_TASKS:
            break
    return out


def pil_resize(img: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((size, size), resample=Image.BILINEAR))


def official_image(views: tuple[np.ndarray, ...]) -> torch.Tensor:
    """PIL BILINEAR resize to 224x224 (the eval's center crop is a no-op
    for square frames), then the eval's normalization in the model dtype:
    bf16(bf16(v * 2/255) - 1)."""
    u8 = torch.empty(1, 3, 224, 448, dtype=torch.float32, device=DEV)
    for i, v in enumerate(views):
        r = np.array(Image.fromarray(v).resize((224, 224), resample=Image.BILINEAR))
        u8[..., i * 224:(i + 1) * 224] = torch.from_numpy(r).permute(2, 0, 1).float().to(DEV)
    x = u8.to(BF16)
    return x * (2.0 / 255.0) - 1.0


def training_image(views: tuple[np.ndarray, ...]) -> torch.Tensor:
    parts = []
    for v in views:
        t = torch.from_numpy(v).permute(2, 0, 1).to(DEV).to(torch.float32) / 255.0
        t = TF.resize(t, [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        parts.append(TF.normalize(t, mean=[0.5] * 3, std=[0.5] * 3))
    return torch.cat(parts, dim=-1).unsqueeze(0).to(BF16)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-size", type=int, default=0, help="PIL-downscale frames to SxS first (0: keep 512)")
    args = ap.parse_args()
    src = os.environ["FLUX2_SRC"]
    src = os.path.join(src, "src") if os.path.isdir(os.path.join(src, "src", "flux2")) else src
    ae = load_real_ae(os.environ.get("AE_MODEL_PATH") or os.environ["FLUX2_AE_MODEL_PATH"], src)
    area, pil = VaePreprocessor(resize="area"), VaePreprocessor(resize="pil_bilinear")
    samples = load_views()
    if args.proxy_size:
        samples = [tuple(pil_resize(v, args.proxy_size) for v in s) for s in samples]
    h, w = samples[0][0].shape[:2]
    print(f"{len(samples)} frame pairs ({SUITE}, {N_TASKS} tasks, frames {FRAMES}), camera input {h}x{w}")

    pairs = [("area", "pil"), ("pil", "official"), ("area", "official"),
             ("pil", "training"), ("official", "training"), ("area", "training")]
    rows: dict[str, list[float]] = {f"{a} vs {b}": [] for a, b in pairs}
    for views in samples:
        tv = [torch.from_numpy(np.ascontiguousarray(v)) for v in views]
        tok = {
            "area": encode_to_tokens(ae, list(tv), preprocessor=area)[0],
            "pil": encode_to_tokens(ae, list(tv), preprocessor=pil)[0],
            "official": ae.encode(official_image(views)).permute(0, 2, 3, 1).reshape(-1, 128),
            "training": ae.encode(training_image(views)).permute(0, 2, 3, 1).reshape(-1, 128),
        }
        for a, b in pairs:
            rows[f"{a} vs {b}"].append(cos(tok[a], tok[b]))
    print("token cosine over frames:")
    for k, v in rows.items():
        print(f"  {k:24s} min={np.min(v):.5f} median={np.median(v):.5f} max={np.max(v):.5f}")


if __name__ == "__main__":
    main()
