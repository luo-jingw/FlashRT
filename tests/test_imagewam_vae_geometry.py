"""The VAE encode geometry follows the workload's per-view size.

The workload's layout divides `image_h`/`image_w` by the patch stride to get
the token grid (`ref_h x ref_w`, `img_len`) the graph's `img_raw` and the
image RoPE are built on, and the VAE encodes exactly one view per camera --
so a view is encoded at the workload's own per-view size. Encoding the 3x256
target workload at a fixed 224x224 produced 14x42 = 588 tokens for the 3x48
= 768 `img_raw` the frontend had already sized, which is what this file
pins, in both placements:

* `VaeStageSpec.encode_hw` / `latent_hw` / `img_len` (the in-graph stage's
  own token geometry): the views' own `in_h x in_w` by default, so
  `(3, 256, 256)` gives 16x48 = 768 rows, while `(2, 224, 224)` keeps
  LIBERO's 14x28 = 392. An explicit `out_hw` is still honoured for a caller
  that stages another delivered size and resizes;
* `ImageWAMVaeStage(..., img_raw)` accepting exactly that token count, and
  rejecting a preprocessor whose encode size disagrees with the spec's;
* the per-view size the frontend's outside-graph path encodes at, read off
  the `out_hw` it passes to `encode_to_tokens` and the preprocessing kernel
  it builds, for `TARGET_WORKLOAD` (256x256) and `ImageWAMWorkload.libero()`
  (224x224) -- the numbers must match each workload's own layout.

CPU only: no CUDA, no compiled extension, no frontend construction (the
constructor allocates CUDA tensors and loads multi-GB checkpoints), so the
frontend is built with `unittest.mock` the way
`tests/test_imagewam_workload_views.py` does it, the preprocessing kernel
and `encode_to_tokens` are replaced by recording stand-ins. The real
numbers (that LIBERO's two 224x224 views still produce today's exact
tokens, and that the 3x256 stage fills a real `img_raw`) need the real AE
on a GPU: `tests/test_imagewam_vae_stage.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest import mock

import pytest
import torch

from benchmarks._imagewam_workload_cli import TARGET_WORKLOAD
from flash_rt.frontends.torch import imagewam_thor
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam import vae_encoder
from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.vae_stage import ImageWAMVaeStage, VaeStageSpec
from flash_rt.models.imagewam.workload import ImageWAMWorkload

LIBERO_WORKLOAD = ImageWAMWorkload.libero()
STRUCTURE = ImageWAMStructure.libero()
LIBERO = "libero"
TARGET = "target"


def _layout(workload: ImageWAMWorkload):
    return workload.layout(STRUCTURE)


# -- the stage spec's own token geometry -------------------------------------


@pytest.mark.parametrize("workload", [LIBERO_WORKLOAD, TARGET_WORKLOAD], ids=[LIBERO, TARGET])
def test_stage_spec_geometry_follows_the_view_size(workload):
    """No `out_hw`: the views' own size IS the encode size, so the spec's
    token grid is the workload's own layout."""
    lay = _layout(workload)
    spec = VaeStageSpec(num_views=workload.num_views, in_h=workload.image_h, in_w=workload.image_w)
    assert spec.encode_hw == (workload.image_h, workload.image_w)
    assert spec.latent_hw == (lay.ref_h, lay.ref_w)
    assert spec.img_len == lay.img_len


def test_target_workload_stage_spec_is_768_tokens():
    """The 3x256 target workload: 16x48, the grid `img_raw` (768, 128) and
    the image RoPE are sized for -- not the 224-derived 14x42 = 588."""
    spec = VaeStageSpec(num_views=3, in_h=256, in_w=256)
    assert spec.latent_hw == (16, 48)
    assert spec.img_len == 3 * 256 // 16 * (256 // 16) == 768


def test_libero_stage_spec_is_unchanged():
    """Two 224x224 views: 14x28 = 392, exactly the geometry of every
    LIBERO-served build before the encode size followed the workload."""
    spec = VaeStageSpec(num_views=2, in_h=224, in_w=224)
    assert spec.encode_hw == (224, 224)
    assert spec.latent_hw == (14, 28)
    assert spec.img_len == 392


@pytest.mark.parametrize("out_hw,expected_img_len", [((224, 224), 392), ((256, 256), 512)])
def test_stage_spec_honours_an_explicit_encode_size(out_hw, expected_img_len):
    """A caller that stages another delivered size and resizes still pins
    the encode size itself (the LIBERO 512x512 frames encoded at 224x224 of
    `benchmarks/imagewam_vae_stage_bench.py`)."""
    spec = VaeStageSpec(num_views=2, in_h=512, in_w=512, out_hw=out_hw)
    assert spec.encode_hw == out_hw
    assert spec.img_len == expected_img_len


# -- what the stage accepts --------------------------------------------------


class _StubPreprocessor:
    """`VaePreprocessor`'s construction interface without a CUDA device
    (the real one allocates its table on one). `run()` is not part of these
    tests -- only the graphs and the resize mode are reachable on CPU."""

    def __init__(self, out_hw: tuple[int, int]) -> None:
        self.out_hw = out_hw
        self.prepared: list[tuple[int, int]] = []

    def prepare(self, in_h: int, in_w: int) -> None:
        self.prepared.append((in_h, in_w))

    def run(self, views, out, stream: int) -> None:
        raise AssertionError("these tests do not run the preprocessing kernel")


@pytest.mark.parametrize("workload", [LIBERO_WORKLOAD, TARGET_WORKLOAD], ids=[LIBERO, TARGET])
def test_stage_accepts_the_img_raw_the_workload_sizes(workload):
    """`ImageWAMVaeStage(..., img_raw)` takes exactly the workload's own
    `img_len` x 128 buffer, and its fixed buffers are the workload's own
    view size and token grid."""
    lay = _layout(workload)
    spec = VaeStageSpec(num_views=workload.num_views, in_h=workload.image_h, in_w=workload.image_w)
    img_raw = torch.zeros(lay.img_len, 128, dtype=torch.bfloat16)
    pre = _StubPreprocessor(spec.encode_hw)
    stage = ImageWAMVaeStage(object(), pre, spec, img_raw)
    assert tuple(stage.views_u8.shape) == (workload.num_views, workload.image_h, workload.image_w, 3)
    assert tuple(stage.image.shape) == (1, 3, workload.image_h, workload.num_views * workload.image_w)
    assert tuple(stage._tokens_nhwc.shape) == (1, lay.ref_h, lay.ref_w, 128)
    assert pre.prepared == [(workload.image_h, workload.image_w)]


def test_stage_rejects_the_224_token_count_for_three_256_views():
    """The observed failure: a `(588, 128)` stage against the workload's
    `(768, 128)` `img_raw` is a rejected mismatch, not a silent 588-token
    encode."""
    spec = VaeStageSpec(num_views=3, in_h=256, in_w=256)
    with pytest.raises(ValueError, match=r"img_raw must be contiguous BF16 \(768, 128\)"):
        ImageWAMVaeStage(object(), _StubPreprocessor(spec.encode_hw), spec,
                         torch.zeros(588, 128, dtype=torch.bfloat16))


def test_stage_rejects_a_preprocessor_for_another_encode_size():
    """The preprocessing kernel and the stage must encode the same size:
    otherwise the tokens the encoder produces would not be the grid the
    spec reserves."""
    spec = VaeStageSpec(num_views=3, in_h=256, in_w=256)
    with pytest.raises(ValueError, match=r"preprocessor.out_hw=\(224, 224\) != spec.encode_hw=\(256, 256\)"):
        ImageWAMVaeStage(object(), _StubPreprocessor((224, 224)), spec,
                         torch.zeros(spec.img_len, 128, dtype=torch.bfloat16))


# -- the frontend's own per-view size ----------------------------------------


class _RecordingPreprocessor:
    """`VaePreprocessor` stand-in: records the resize mode and the per-view
    size the frontend builds it for."""

    instances: list["_RecordingPreprocessor"] = []

    def __init__(self, *, resize: str, out_hw: tuple[int, int], device: str = "cuda") -> None:
        self.resize = resize
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        _RecordingPreprocessor.instances.append(self)


@dataclass(frozen=True)
class _StagedCall:
    """What `stage_images` handed the VAE, as this file's own interface."""
    out_hw: tuple[int, int]
    preprocessor: _RecordingPreprocessor
    encoder: object
    num_views: int
    frontend: ImageWAMTorchFrontendThor


def _mocked_frontend(**attributes) -> ImageWAMTorchFrontendThor:
    """A frontend whose `__init__` only sets `attributes` (the real one
    allocates CUDA tensors)."""
    def init(self, **kwargs) -> None:
        del kwargs
        for name, value in attributes.items():
            setattr(self, name, value)

    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", init):
        return ImageWAMTorchFrontendThor()


def _staged_geometry(workload: ImageWAMWorkload) -> _StagedCall:
    """What the frontend's outside-graph `stage_images` hands the VAE for a
    frontend serving `workload`: the per-view `out_hw`, the preprocessing
    kernel it built, the encoder, the camera count and the frontend."""
    lay = _layout(workload)
    _RecordingPreprocessor.instances = []
    fe = _mocked_frontend(_workload=workload, _vae_stage=None, _ae=object(),
                          _vae_encoder=object(), _vae_pre=None, _vae_resize="area",
                          _img_raw=torch.zeros(lay.img_len, 128, dtype=torch.bfloat16))
    views = [torch.zeros(workload.image_h, workload.image_w, 3, dtype=torch.uint8)
             for _ in range(workload.num_views)]
    with mock.patch.object(imagewam_thor, "VaePreprocessor", _RecordingPreprocessor), \
            mock.patch.object(vae_encoder, "encode_to_tokens",
                              return_value=torch.zeros(1, lay.img_len, 128, dtype=torch.bfloat16)) as encode:
        fe.stage_images(*views)
    assert len(_RecordingPreprocessor.instances) == 1
    return _StagedCall(out_hw=tuple(encode.call_args.kwargs["out_hw"]),
                       preprocessor=_RecordingPreprocessor.instances[0],
                       encoder=encode.call_args.kwargs["encoder"],
                       num_views=len(encode.call_args.args[1]), frontend=fe)


@pytest.mark.parametrize("workload", [LIBERO_WORKLOAD, TARGET_WORKLOAD], ids=[LIBERO, TARGET])
def test_stage_images_encodes_at_the_workloads_per_view_size(workload):
    """The VAE outside the graph gets the workload's own per-view size
    (through `_input_view_shape()`, the frontend's one view-shape
    resolution) both as `encode_to_tokens(out_hw=...)` and as the
    preprocessing kernel's encode size, and it encodes every camera of the
    workload."""
    staged = _staged_geometry(workload)
    assert staged.out_hw == (workload.image_h, workload.image_w)
    assert staged.preprocessor.out_hw == (workload.image_h, workload.image_w)
    assert staged.preprocessor.resize == "area"
    assert staged.encoder is staged.frontend._vae_encoder
    assert staged.num_views == workload.num_views
    # The call returned the workload's own `img_len` rows and `stage_images`
    # copied them into `_img_raw` without a shape error: the workload's
    # layout is what the frontend's own buffer is sized for.
    assert staged.frontend._img_raw.shape[0] == _layout(workload).img_len


def test_hand_passed_dims_still_encode_at_liberos_own_size():
    """A caller that passes dims by hand names no workload:
    `_input_view_shape()`'s own `(2, 224, 224)` is the per-view size, so
    that path is unchanged (and a 512x512 LIBERO frame is resized to it)."""
    lay = _layout(LIBERO_WORKLOAD)
    _RecordingPreprocessor.instances = []
    fe = _mocked_frontend(_workload=None, _vae_stage=None, _ae=object(),
                          _vae_encoder=object(), _vae_pre=None, _vae_resize="area",
                          _img_raw=torch.zeros(lay.img_len, 128, dtype=torch.bfloat16))
    tokens = torch.zeros(1, lay.img_len, 128, dtype=torch.bfloat16)
    with mock.patch.object(imagewam_thor, "VaePreprocessor", _RecordingPreprocessor), \
            mock.patch.object(vae_encoder, "encode_to_tokens", return_value=tokens) as encode:
        fe.stage_images(torch.zeros(512, 512, 3, dtype=torch.uint8),
                        torch.zeros(512, 512, 3, dtype=torch.uint8))
    assert tuple(encode.call_args.kwargs["out_hw"]) == (224, 224)
    assert _RecordingPreprocessor.instances[0].out_hw == (224, 224)
    assert fe._img_raw.shape == (lay.img_len, 128)


def test_encode_to_tokens_default_stays_liberos_per_view_size():
    """The frontend passes `out_hw` explicitly; the direct callers that
    encode LIBERO frames keep the default, which is that release's own
    per-view size."""
    import inspect

    parameter = inspect.signature(vae_encoder.encode_to_tokens).parameters["out_hw"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == (LIBERO_WORKLOAD.image_h, LIBERO_WORKLOAD.image_w) == (224, 224)
