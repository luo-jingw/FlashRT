"""ImageWAM `io="python"` model runtime with the real VAE, both placements.

Random-weight backbone at small widths but the real image token count
(`img_len` 392 = a 14 x 28 grid from two 224 x 224 views), the real
FLUX.2 AE through the served preprocessing kernel. A ctypes consumer
stages two uint8 frames through the ABI and must reproduce `infer()` bit
for bit:

- VAE outside the graph: `images` STAGED (encode now, write `img_raw`);
- VAE inside the graph (`vae_graph_input`): `images` STAGED into the
  graph's uint8 view buffer, and the same buffer written raw through the
  `image_views` SWAP window.

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
from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path, make_image_views  # noqa: E402

_FLUX2_SRC = os.environ.get("FLUX2_SRC", "")
_FLUX2_SRC = os.path.join(_FLUX2_SRC, "src") if os.path.isdir(os.path.join(_FLUX2_SRC, "src", "flux2")) else _FLUX2_SRC
_AE_PATH = os.environ.get("AE_MODEL_PATH") or os.environ.get("FLUX2_AE_MODEL_PATH", "")
pytestmark = pytest.mark.skipif(not (os.path.isdir(_FLUX2_SRC) and os.path.isfile(_AE_PATH)),
                                reason="real flux2 clone / AE checkpoint not present")

IMG_DIMS = {"x0": 3, "a0": 3 + 392, "total": 3 + 392 + 4, "ref_h": 14, "ref_w": 28}
RAW_HW = (256, 256)
SEED = 11


def _frames(hw: tuple[int, int]) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    return [np.ascontiguousarray(rng.integers(0, 256, (*hw, 3), dtype=np.uint8)) for _ in range(2)]


def _run(vae_graph_input):
    hw = RAW_HW if vae_graph_input else (224, 224)
    fe = ImageWAMTorchFrontendThor(precision="fp16", dims_override=dict(IMG_DIMS), ae_model_path=_AE_PATH,
                                   flux2_src=_FLUX2_SRC, vae_graph_input=vae_graph_input)
    fe.set_prompt("pick up the red cup")
    frames = _frames(hw)
    obs = {"view1": torch.from_numpy(frames[0]), "view2": torch.from_numpy(frames[1])}
    noise = torch.randn(4, 7, device="cuda") * 0.01
    ref = fe.infer(obs, action_noise=noise)["actions"]
    ref_tokens = fe._img_raw.detach().clone()
    mr = fe.export_model_runtime()
    consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
    try:
        names = [p.name for p in consumer.ports]
        views = make_image_views(frames, FrtImageView)
        consumer.set_input("images", views)
        consumer.write_swap("noise", noise.cpu().numpy())
        consumer.step()
        staged = consumer.get_output("actions", np.float32, (4, 7))
        staged_tokens = fe._img_raw.detach().clone()
        swap = None
        if "image_views" in names:
            fe._vae_stage.views_u8.zero_()
            torch.cuda.synchronize()
            consumer.write_swap("image_views", np.stack(frames))
            consumer.write_swap("noise", noise.cpu().numpy())
            consumer.step()
            swap = consumer.get_output("actions", np.float32, (4, 7))
        return names, ref, ref_tokens, staged, staged_tokens, swap, mr.identity
    finally:
        consumer.close()
        mr.release()


def test_vae_outside_graph_images_port():
    names, ref, ref_tokens, staged, staged_tokens, swap, identity = _run(None)
    print(f"ports={names}; images STAGED: tokens array_equal={torch.equal(staged_tokens, ref_tokens)} "
          f"actions array_equal={np.array_equal(staged, ref)} max_abs={np.abs(staged - ref).max():.3g}")
    assert names == ["images", "image_tokens", "noise", "actions", "actions_raw"]
    assert "vae_in_graph=False" in identity and "vae_resize=area" in identity
    assert torch.equal(staged_tokens, ref_tokens) and np.array_equal(staged, ref)


def test_vae_inside_graph_images_and_image_views_ports():
    names, ref, ref_tokens, staged, staged_tokens, swap, identity = _run((2, *RAW_HW))
    print(f"ports={names}; images STAGED: tokens array_equal={torch.equal(staged_tokens, ref_tokens)} "
          f"actions array_equal={np.array_equal(staged, ref)}; image_views SWAP: actions "
          f"array_equal={np.array_equal(swap, ref)} max_abs={np.abs(swap - ref).max():.3g}")
    assert names == ["images", "image_views", "noise", "actions", "actions_raw"]
    assert "vae_in_graph=True" in identity
    assert torch.equal(staged_tokens, ref_tokens) and np.array_equal(staged, ref)
    assert np.array_equal(swap, ref)
