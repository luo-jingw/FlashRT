"""ImageWAM `io="python"` model runtime with the real VAE, both placements.

Random-weight backbone at small widths but the real image token count
(`img_len` 392 = a 14 x 28 grid from two 224 x 224 views), the real
FLUX.2 AE through the served preprocessing kernel. A ctypes consumer
stages two uint8 frames through the ABI and, with every buffer the tick
writes NaN-poisoned first (the in-graph uint8 views zeroed), must
reproduce `infer()` bit for bit; with the `images` verb made a no-op the
same tick must fail:

- VAE outside the graph: `images` STAGED (encode now, write `img_raw`);
- VAE inside the graph (`vae_graph_input`): `images` STAGED into the
  graph's uint8 view buffer, and the same buffer written raw through the
  `image_views` SWAP window.

Both placements drive those 224x224 frames, the size the workload delivers a
view at and the size it is encoded at, so the two differ only in where the
encode happens: the declared `images` shape, the in-graph `views_u8` and the
`image_views` window, and the 392 tokens in `img_raw` are the same in both.

Needs the real AE (`AE_MODEL_PATH` / `FLUX2_AE_MODEL_PATH`), a `flux2`
clone (`FLUX2_SRC`) and the exec/ + runtime/ builds; skips otherwise.
"""
import os

import numpy as np
import pytest
import torch

pytest.importorskip("flash_rt.runtime.exec", exc_type=ImportError)
pytest.importorskip("flash_rt.runtime.export", exc_type=ImportError)

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor  # noqa: E402
from flash_rt.models.imagewam.runtime_export import FrtImageView  # noqa: E402
from _helpers.imagewam_abi_checks import bits, poison_tick_state, python_verb_noop  # noqa: E402
from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path, make_image_views  # noqa: E402

_FLUX2_SRC = os.environ.get("FLUX2_SRC", "")
_FLUX2_SRC = os.path.join(_FLUX2_SRC, "src") if os.path.isdir(os.path.join(_FLUX2_SRC, "src", "flux2")) else _FLUX2_SRC
_AE_PATH = os.environ.get("AE_MODEL_PATH") or os.environ.get("FLUX2_AE_MODEL_PATH", "")
pytestmark = pytest.mark.skipif(not (os.path.isdir(_FLUX2_SRC) and os.path.isfile(_AE_PATH)),
                                reason="real flux2 clone / AE checkpoint not present")

IMG_DIMS = {"x0": 3, "a0": 3 + 392, "total": 3 + 392 + 4, "ref_h": 14, "ref_w": 28}
# Both placements take LIBERO's own per-view size: a view is encoded at the
# size it is delivered at (`VaeStageSpec.encode_hw`, the frontend's
# `_input_view_shape()`), so the in-graph stage built from `(2, 224, 224)`
# has `encode_hw` `(224, 224)`, `views_u8` `(2, 224, 224, 3)` and the 392
# tokens `IMG_DIMS` carries. Frames of another size encode at that size
# instead: 256x256 views are a 16 x 32 grid, i.e. 512 tokens.
FRAME_HW = (224, 224)
SEED = 11


def _frames(hw: tuple[int, int]) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    return [np.ascontiguousarray(rng.integers(0, 256, (*hw, 3), dtype=np.uint8)) for _ in range(2)]


def _build(vae_graph_input):
    """FA4 off explicitly: this test fixes where the VAE runs, not the
    attention path, and `FLASHRT_THOR_FA4` defaults to the machine's answer,
    so stating the choice keeps the two placements' rows the same on every
    machine."""
    fe = ImageWAMTorchFrontendThor(precision="fp16", use_fa4=False, dims_override=dict(IMG_DIMS),
                                   ae_model_path=_AE_PATH, flux2_src=_FLUX2_SRC,
                                   vae_graph_input=vae_graph_input)
    fe.set_prompt("pick up the red cup")
    frames = _frames(FRAME_HW)
    obs = {"view1": torch.from_numpy(frames[0]), "view2": torch.from_numpy(frames[1])}
    noise = torch.randn(4, 7, device="cuda") * 0.01
    ref = fe.infer(obs, action_noise=noise)["actions"]
    return fe, frames, noise, ref, bits(fe._img_raw)


def _tick(fe, frames, noise, *, images_noop=False):
    """One poisoned ABI tick through `images` STAGED (and, with the VAE in
    the graph, a second one through the `image_views` SWAP window)."""
    if images_noop:
        with python_verb_noop("images"):
            mr = fe.export_model_runtime()
    else:
        mr = fe.export_model_runtime()
    consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
    try:
        names = [p.name for p in consumer.ports]
        poison_tick_state(fe)
        consumer.set_input("images", make_image_views(frames, FrtImageView))
        consumer.write_swap("noise", noise.cpu().numpy())
        consumer.step()
        staged = consumer.get_output("actions", np.float32, (4, 7))
        staged_tokens = bits(fe._img_raw)
        swap = None
        if "image_views" in names:
            poison_tick_state(fe)
            consumer.write_swap("image_views", np.stack(frames))
            consumer.write_swap("noise", noise.cpu().numpy())
            consumer.step()
            swap = consumer.get_output("actions", np.float32, (4, 7))
        return names, staged, staged_tokens, swap, mr.identity
    finally:
        consumer.close()
        mr.release()


def _run(vae_graph_input):
    fe, frames, noise, ref, ref_tokens = _build(vae_graph_input)
    names, staged, staged_tokens, swap, identity = _tick(fe, frames, noise)
    _, noop, noop_tokens, _, _ = _tick(fe, frames, noise, images_noop=True)
    print(f"mutant images no-op: tokens array_equal={np.array_equal(noop_tokens, ref_tokens)} "
          f"actions array_equal={np.array_equal(noop, ref)}")
    assert not np.array_equal(noop, ref) and not np.array_equal(noop_tokens, ref_tokens)
    return names, ref, ref_tokens, staged, staged_tokens, swap, identity


def test_vae_outside_graph_images_port():
    names, ref, ref_tokens, staged, staged_tokens, swap, identity = _run(None)
    print(f"ports={names}; images STAGED: tokens array_equal={np.array_equal(staged_tokens, ref_tokens)} "
          f"actions array_equal={np.array_equal(staged, ref)} max_abs={np.abs(staged - ref).max():.3g}")
    assert names == ["images", "image_tokens", "noise", "actions", "actions_raw"]
    assert "vae_in_graph=False" in identity and "vae_resize=area" in identity
    assert np.array_equal(staged_tokens, ref_tokens) and np.array_equal(staged, ref)


def test_vae_inside_graph_images_and_image_views_ports():
    names, ref, ref_tokens, staged, staged_tokens, swap, identity = _run((2, *FRAME_HW))
    print(f"ports={names}; images STAGED: tokens array_equal={np.array_equal(staged_tokens, ref_tokens)} "
          f"actions array_equal={np.array_equal(staged, ref)}; image_views SWAP: actions "
          f"array_equal={np.array_equal(swap, ref)} max_abs={np.abs(swap - ref).max():.3g}")
    assert names == ["images", "image_views", "noise", "actions", "actions_raw"]
    assert "vae_in_graph=True" in identity
    assert np.array_equal(staged_tokens, ref_tokens) and np.array_equal(staged, ref)
    assert np.array_equal(swap, ref)
