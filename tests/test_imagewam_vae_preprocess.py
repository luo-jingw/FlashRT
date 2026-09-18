"""`imagewam_vae_preprocess_bf16` kernel vs the served torch preprocessing
and vs the official PIL resize (roadmap item 2, `plan.md`).

Observational: every case prints cosine, max-abs, rel_l2 and the count
of BF16 elements that differ from the reference. The asserts only
require what the kernel is designed to guarantee: zero differing
elements (bit-exact).

References:
- `resize="area"`: `vae_encoder._prep_view` per view + `torch.cat`
  (the served path).
- `resize="pil_bilinear"`: the official `_center_crop_resize` (PIL
  `BILINEAR` + center crop, uint8) followed by the served normalization
  expression.

Real LIBERO frames (512x512) are used when `DATA_ROOT` points at the
LIBERO-fastwam dataset; otherwise only synthetic inputs run.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from PIL import Image

from flash_rt.models.imagewam.vae_encoder import _prep_view
from flash_rt.models.imagewam.vae_preprocess import VaePreprocessor

DEV = "cuda"
BF16 = torch.bfloat16


def _real_frames() -> list[np.ndarray]:
    root = os.environ.get("DATA_ROOT")
    if not root:
        return []
    base = os.path.join(root, "libero_spatial_no_noops_lerobot", "videos", "chunk-000")
    paths = [os.path.join(base, "observation.images.image", "episode_000000.mp4"),
             os.path.join(base, "observation.images.wrist_image", "episode_000000.mp4")]
    if not all(os.path.isfile(p) for p in paths):
        return []
    import av
    frames = []
    for p in paths:
        with av.open(p) as c:
            for f in c.decode(video=0):
                frames.append(f.to_ndarray(format="rgb24"))
                break
    return frames


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Official ImageWAM eval `eval_libero_single._center_crop_resize`."""
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def _stats(name: str, out: torch.Tensor, ref: torch.Tensor) -> int:
    a, b = out.float().flatten(), ref.float().flatten()
    cos = (a @ b / (a.norm() * b.norm() + 1e-12)).item()
    maxabs = (a - b).abs().max().item()
    rel_l2 = ((a - b).norm() / (b.norm() + 1e-12)).item()
    ndiff = int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())
    print(f"{name}: shape={tuple(out.shape)} cosine={cos:.8f} max_abs={maxabs:.3e} "
          f"rel_l2={rel_l2:.3e} differing_bf16={ndiff}/{out.numel()}")
    return ndiff


def _cases() -> list[tuple[str, list[np.ndarray]]]:
    rng = np.random.default_rng(0)
    cases = []
    real = _real_frames()
    if real:
        cases.append(("real LIBERO 512x512 x2", real))
    cases += [
        ("random 512x512 x2", [rng.integers(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(2)]),
        ("random 256x256 x2", [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(2)]),
        ("random 224x224 x2 (no resize)", [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(2)]),
        ("random 480x640 x1 (non-square)", [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)]),
        ("random 100x150 x1 (upsample)", [rng.integers(0, 256, (100, 150, 3), dtype=np.uint8)]),
    ]
    return cases


def test_area_matches_served_prep_view_bitexact():
    pre = VaePreprocessor(resize="area")
    for name, views in _cases():
        gpu_views = [torch.from_numpy(v).to(DEV) for v in views]
        ref = torch.cat([_prep_view(v, (224, 224), DEV, BF16) for v in gpu_views], dim=-1)
        out = torch.empty(1, 3, 224, 224 * len(views), dtype=BF16, device=DEV)
        pre.run(gpu_views, out, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert _stats(f"area    {name}", out, ref) == 0


def test_pil_bilinear_matches_official_center_crop_resize_bitexact():
    pre = VaePreprocessor(resize="pil_bilinear")
    for name, views in _cases():
        gpu_views = [torch.from_numpy(v).to(DEV) for v in views]
        resized = [torch.from_numpy(_center_crop_resize(v, 224, 224).copy()).to(DEV) for v in views]
        ref = torch.cat([_prep_view(r, (224, 224), DEV, BF16) for r in resized], dim=-1)
        out = torch.empty(1, 3, 224, 224 * len(views), dtype=BF16, device=DEV)
        pre.run(gpu_views, out, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        assert _stats(f"pil     {name}", out, ref) == 0


def test_area_vs_pil_difference_on_same_input():
    """Not a correctness check: the size of the served-vs-official
    preprocessing difference itself (issues.md ISSUE-030)."""
    area, pil = VaePreprocessor(resize="area"), VaePreprocessor(resize="pil_bilinear")
    for name, views in _cases()[:2]:
        gpu_views = [torch.from_numpy(v).to(DEV) for v in views]
        a = torch.empty(1, 3, 224, 224 * len(views), dtype=BF16, device=DEV)
        p = torch.empty_like(a)
        area.run(gpu_views, a, 0)
        pil.run(gpu_views, p, 0)
        torch.cuda.synchronize()
        _stats(f"area-vs-pil {name}", a, p)


def test_invalid_arguments_raise():
    pre = VaePreprocessor(resize="area")
    v = torch.zeros(64, 64, 3, dtype=torch.uint8, device=DEV)
    with pytest.raises(ValueError):
        pre.run([v], torch.empty(1, 3, 224, 448, dtype=BF16, device=DEV), 0)
    with pytest.raises(ValueError):
        pre.run([v.float()], torch.empty(1, 3, 224, 224, dtype=BF16, device=DEV), 0)
    with pytest.raises(ValueError):
        VaePreprocessor(resize="bicubic")
