"""`ImageWAMVaeStage` (roadmap item 5, `plan.md`): the fixed-address VAE
stage, eager and captured into a CUDA graph, against the served
`encode_to_tokens` path.

Needs the real FLUX.2 AE (`AE_MODEL_PATH`) and a `flux2` clone
(`FLUX2_SRC`); skips otherwise. Uses real LIBERO 512x512 frames when
`DATA_ROOT` is set, else synthetic frames. Prints cosine / max-abs and
the token statistics next to the real Thor values (mean -0.02, std
0.97, absmax 4.91).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

DEV = "cuda"
BF16 = torch.bfloat16

_FLUX2_SRC = os.environ.get("FLUX2_SRC", "")
_FLUX2_SRC = os.path.join(_FLUX2_SRC, "src") if os.path.isdir(os.path.join(_FLUX2_SRC, "src", "flux2")) else _FLUX2_SRC
_AE_PATH = os.environ.get("AE_MODEL_PATH", "")
_AVAILABLE = os.path.isdir(_FLUX2_SRC) and os.path.isfile(_AE_PATH)
pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="real flux2 clone / AE checkpoint not present")


def _frames(n: int) -> list[np.ndarray]:
    root = os.environ.get("DATA_ROOT")
    base = os.path.join(root or "", "libero_spatial_no_noops_lerobot", "videos", "chunk-000")
    out: list[np.ndarray] = []
    if root and os.path.isdir(base):
        import av
        for key in ("observation.images.image", "observation.images.wrist_image"):
            with av.open(os.path.join(base, key, "episode_000000.mp4")) as c:
                for i, f in enumerate(c.decode(video=0)):
                    if i in (0, 40):
                        out.append(f.to_ndarray(format="rgb24"))
                    if i >= 40:
                        break
        return out[:n]
    rng = np.random.default_rng(0)
    y, x = np.meshgrid(np.linspace(0, 1, 512), np.linspace(0, 1, 512), indexing="ij")
    for _ in range(n):
        base_img = np.stack([x, y, 1 - x], axis=-1) * 255 + rng.normal(0, 8, (512, 512, 3))
        out.append(np.clip(base_img, 0, 255).astype(np.uint8))
    return out


def _cmp(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    x, y = a.float().flatten(), b.float().flatten()
    cos = (x @ y / (x.norm() * y.norm() + 1e-12)).item()
    f = a.float()
    print(f"{name}: cosine={cos:.8f} max_abs={(x - y).abs().max().item():.3e} "
          f"mean={f.mean().item():.4f} std={f.std().item():.4f} absmax={f.abs().max().item():.4f}")


@pytest.fixture(scope="module")
def ae():
    from flash_rt.models.imagewam.vae_encoder import load_real_ae
    return load_real_ae(_AE_PATH, _FLUX2_SRC)


def _graph(fn) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def test_stage_eager_and_graph_match_encode_to_tokens(ae):
    from flash_rt.models.imagewam.vae_encoder import encode_to_tokens
    from flash_rt.models.imagewam.vae_preprocess import VaePreprocessor
    from flash_rt.models.imagewam.vae_stage import ImageWAMVaeStage, VaeStageSpec

    # _frames order: agent f0, agent f40, wrist f0, wrist f40 (real) or 4 synthetic frames.
    frames = [torch.from_numpy(f) for f in _frames(4)]
    first, second = [frames[0], frames[2]], [frames[1], frames[3]]
    pre = VaePreprocessor(resize="area")
    spec = VaeStageSpec(num_views=2, in_h=512, in_w=512)
    img_raw = torch.zeros(spec.img_len, 128, dtype=BF16, device=DEV)
    stage = ImageWAMVaeStage(ae, pre, spec, img_raw)

    with torch.no_grad():
        ref1 = encode_to_tokens(ae, *first)[0]
        ref2 = encode_to_tokens(ae, *second)[0]
    stage.stage(first)
    stage.run()
    torch.cuda.synchronize()
    _cmp("stage eager vs encode_to_tokens", img_raw, ref1)
    assert torch.equal(img_raw, ref1)

    graph = _graph(stage.run)
    for views, ref in ((second, ref2), (first, ref1)):
        img_raw.zero_()
        stage.stage(views)
        graph.replay()
        torch.cuda.synchronize()
        _cmp("stage graph replay vs encode_to_tokens", img_raw, ref)
        assert torch.equal(img_raw, ref)


def test_stage_rejects_wrong_view_shape(ae):
    from flash_rt.models.imagewam.vae_preprocess import VaePreprocessor
    from flash_rt.models.imagewam.vae_stage import ImageWAMVaeStage, VaeStageSpec

    spec = VaeStageSpec(num_views=2, in_h=512, in_w=512)
    stage = ImageWAMVaeStage(ae, VaePreprocessor(), spec, torch.zeros(spec.img_len, 128, dtype=BF16, device=DEV))
    with pytest.raises(ValueError):
        stage.stage([torch.zeros(224, 224, 3, dtype=torch.uint8)] * 2)
    with pytest.raises(ValueError):
        stage.stage([torch.zeros(512, 512, 3, dtype=torch.uint8)])
