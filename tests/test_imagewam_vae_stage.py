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


# Small random-weight dims whose image span matches the real 2x224x224
# VAE output (a0 - x0 = 392 tokens, 14x28 grid).
_SMALL_DIMS = dict(
    hidden=256, HD=128, NH=2, mlp_hidden=384, joint_attention_dim=64,
    x0=3, a0=395, num_layers_double=2, num_layers_single=3,
    action_hidden_dim=128, action_attn_width=256, action_mlp_hidden=192,
    action_dim=7, num_action=4, total=399,
    action_num_layers_double=2, action_num_layers_single=3,
    dt=0.5, num_denoise_steps=2, ref_h=14, ref_w=28,
)


def _eager_pipeline(fe) -> None:
    """The captured graph's prefill + denoise, run eagerly on the same
    frontend (same buffers, weights and GEMM algorithm cache)."""
    import flash_rt.flash_rt_kernels as fvk
    from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_loop, imagewam_prefill
    stream = torch.cuda.current_stream().cuda_stream
    imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=stream, attn=fe._attn,
                     mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
                     rope_table=fe._rope_table.data_ptr())
    imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=stream, attn=fe._attn,
                          action_mods=fe._action_mods, head_mods=fe._head_mods,
                          action_rope_table=fe._action_rope_table.data_ptr(), deltas=fe._deltas)
    torch.cuda.synchronize()


def test_native_encoder_tokens_near_exact(ae):
    """`NativeFlux2Encoder` (NHWC + FlashRT GroupNorm) vs the torch
    AutoEncoder: same function, different accumulation order."""
    from flash_rt.models.imagewam.vae_encoder import encode_to_tokens
    from flash_rt.models.imagewam.vae_native_encoder import NativeFlux2Encoder
    from flash_rt.models.imagewam.vae_preprocess import VaePreprocessor
    from flash_rt.models.imagewam.vae_stage import ImageWAMVaeStage, VaeStageSpec

    native = NativeFlux2Encoder(ae)
    frames = [torch.from_numpy(f) for f in _frames(4)]
    spec = VaeStageSpec(num_views=2, in_h=512, in_w=512)
    img_raw = torch.zeros(spec.img_len, 128, dtype=BF16, device=DEV)
    stage = ImageWAMVaeStage(native, VaePreprocessor(), spec, img_raw)
    graph = _graph(stage.run)
    for views in ([frames[0], frames[2]], [frames[1], frames[3]]):
        with torch.no_grad():
            ref = encode_to_tokens(ae, *views)[0]
        stage.stage(views)
        stage.run()
        torch.cuda.synchronize()
        eager = img_raw.clone()
        _cmp("native stage eager vs torch encode_to_tokens", eager, ref)
        x, y = eager.float().flatten(), ref.float().flatten()
        assert (x @ y / (x.norm() * y.norm())).item() > 0.9999
        img_raw.zero_()
        stage.stage(views)
        graph.replay()
        torch.cuda.synchronize()
        _cmp("native stage graph vs native stage eager", img_raw, eager)
        assert torch.equal(img_raw, eager)


@pytest.mark.parametrize("vae_encoder", ["torch", "native"])
def test_frontend_vae_in_graph_matches_eager_reference(vae_encoder):
    """VAE inside the main graph (`vae_graph_input`): one replay writes
    img_raw exactly as `encode_to_tokens` with the same encoder does
    eagerly, and the actions equal an eager run of the same frontend's
    prefill + denoise fed with those tokens and the same noise."""
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
    from flash_rt.models.imagewam.vae_encoder import encode_to_tokens

    torch.manual_seed(0)
    fe = ImageWAMTorchFrontendThor(dims_override=dict(_SMALL_DIMS), precision="fp16",
                                   ae_model_path=_AE_PATH, flux2_src=_FLUX2_SRC,
                                   vae_encoder=vae_encoder, vae_graph_input=(2, 512, 512))
    fe.set_prompt()
    frames = [torch.from_numpy(f) for f in _frames(4)]
    for v1, v2 in ((frames[0], frames[2]), (frames[1], frames[3])):
        torch.manual_seed(1)
        actions = torch.from_numpy(fe.infer({"view1": v1, "view2": v2})["actions"])
        graph_tokens = fe._img_raw.clone()
        with torch.no_grad():
            ref_tokens = encode_to_tokens(fe._ae, v1, v2, preprocessor=fe._vae_pre, encoder=fe._vae_encoder)[0]
        _cmp(f"frontend graph[{vae_encoder}] img_raw vs encode_to_tokens (same encoder)", graph_tokens, ref_tokens)
        assert torch.equal(graph_tokens, ref_tokens)

        fe._img_raw.copy_(ref_tokens)
        torch.manual_seed(1)
        fe._action_latent.normal_()
        fe._action_latent.mul_(0.01)
        _eager_pipeline(fe)
        ref_actions = fe._action_latent.detach().cpu()
        _cmp(f"frontend graph[{vae_encoder}] actions vs eager reference", actions, ref_actions)
        assert torch.isfinite(actions).all()
        assert torch.equal(actions, ref_actions)


def test_frontend_eager_native_matches_encode_to_tokens():
    """`vae_encoder="native"` outside the graph: img_raw equals
    `encode_to_tokens` with the native encoder."""
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
    from flash_rt.models.imagewam.vae_encoder import encode_to_tokens
    fe = ImageWAMTorchFrontendThor(dims_override=dict(_SMALL_DIMS), precision="fp16", ae_model_path=_AE_PATH,
                                   flux2_src=_FLUX2_SRC, vae_encoder="native")
    fe.set_prompt()
    frames = [torch.from_numpy(f) for f in _frames(4)]
    fe.infer({"view1": frames[0], "view2": frames[2]})
    with torch.no_grad():
        ref = encode_to_tokens(fe._ae, frames[0], frames[2], preprocessor=fe._vae_pre, encoder=fe._vae_encoder)[0]
    _cmp("frontend eager[native] img_raw vs encode_to_tokens (native)", fe._img_raw, ref)
    assert torch.equal(fe._img_raw, ref)


def test_frontend_vae_flag_validation():
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
    with pytest.raises(ValueError):
        ImageWAMTorchFrontendThor(dims_override=dict(_SMALL_DIMS), precision="fp16", vae_graph_input=(2, 512, 512))
    with pytest.raises(ValueError):
        ImageWAMTorchFrontendThor(dims_override=dict(_SMALL_DIMS), precision="fp16", vae_encoder="bogus")
    with pytest.raises(ValueError):
        ImageWAMTorchFrontendThor(dims_override=dict(_SMALL_DIMS), precision="fp16", vae_resize="bogus")
    fe = ImageWAMTorchFrontendThor(dims_override=dict(_SMALL_DIMS), precision="fp16", ae_model_path=_AE_PATH,
                                   flux2_src=_FLUX2_SRC, vae_graph_input=(2, 224, 224))
    fe.set_prompt()
    with pytest.raises(ValueError):
        fe.infer({})
    with pytest.raises(ValueError):
        fe.infer({"view1": torch.zeros(512, 512, 3, dtype=torch.uint8),
                  "view2": torch.zeros(512, 512, 3, dtype=torch.uint8)})
