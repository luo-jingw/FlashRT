"""The view count and the text length come from the workload, not from a
LIBERO constant (issues.md ISSUE-084, ISSUE-081, ISSUE-083).

CPU only: no CUDA, no compiled extension, no frontend construction (the
constructor allocates CUDA tensors and loads multi-GB checkpoints). What is
pinned here, one mechanism at a time:

* `observation_views(observation, num_views)` -- the observation path's own
  view list: `view1` ... `view<num_views>`, a missing key raising a
  `ValueError` that names it, and the two-view call returning exactly what
  the former `ImageWAMTorchFrontendThor._observation_views` returned;
* `ImageWAMTorchFrontendThor.num_views` / `_input_view_shape()` -- the
  authority order (the resolved workload, then the in-graph VAE stage, then
  LIBERO's two 224x224 views for a caller that passed dims by hand), with
  `__init__` replaced by `unittest.mock` the way
  `tests/test_imagewam_public_entry.py` does it;
* `stage_images` rejecting a view count other than `num_views`, before it
  reads the VAE;
* `ImageWAMTorchFrontendThor._text_max_length()` -- the padded text length
  the frontend's own dims imply (`x0 - 1` with proprio, `x0` without), for
  LIBERO (512) and for the three-view 256x256 target workload of
  `benchmarks/_imagewam_workload_cli.TARGET_WORKLOAD` (128);
* `vae_encoder.encode_to_tokens`'s own view validation, which runs before it
  touches the autoencoder at all (so no VAE and no CUDA are needed here).

The numerics are NOT pinned here: that two views still produce today's exact
tokens, and that the ABI declares the right frame shape, need the real AE on
a GPU (`tests/test_imagewam_vae_encoder.py`,
`tests/test_imagewam_model_runtime_vae.py`).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from benchmarks._imagewam_workload_cli import TARGET_WORKLOAD
from flash_rt.frontends.torch.imagewam_thor import (
    ImageWAMTorchFrontendThor,
    observation_views,
)
from flash_rt.models.imagewam.config_resolver import resolve_config
from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload

LIBERO_WORKLOAD = ImageWAMWorkload.libero()
STRUCTURE = ImageWAMStructure.libero()


def _observation(num_views: int, shape: tuple[int, int, int] = (4, 5, 3)) -> dict:
    return {f"view{i + 1}": np.zeros(shape, dtype=np.uint8) for i in range(num_views)}


def _mocked_frontend(**attributes) -> ImageWAMTorchFrontendThor:
    """A frontend whose `__init__` only sets `attributes`.

    The real constructor allocates CUDA tensors; this is the same
    `mock.patch.object(..., "__init__", ...)` the config-mapping tests use.
    """
    def init(self, **kwargs) -> None:
        del kwargs
        for name, value in attributes.items():
            setattr(self, name, value)

    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", init):
        return ImageWAMTorchFrontendThor()


def _resolved_dims(workload: ImageWAMWorkload) -> dict:
    return resolve_config(workload, STRUCTURE).dims


# -- the observation's views (ISSUE-084) -------------------------------------


@pytest.mark.parametrize("num_views", [1, 2, 3])
def test_observation_views_returns_the_workloads_views(num_views):
    observation = _observation(num_views)
    views = observation_views(observation, num_views)
    assert views == [observation[f"view{i + 1}"] for i in range(num_views)]
    assert all(views[i] is observation[f"view{i + 1}"] for i in range(num_views))


def test_observation_views_reports_the_missing_key_and_the_expected_count():
    observation = _observation(2)
    with pytest.raises(ValueError) as excinfo:
        observation_views(observation, 3)
    assert "view3" in str(excinfo.value)
    assert "num_views=3" in str(excinfo.value)
    # the frames that ARE present are not silently used as a shorter run
    assert isinstance(excinfo.value, ValueError)


def test_observation_views_two_views_match_the_former_helper():
    """The former `_observation_views` was
    `[observation["view1"]] + ([observation["view2"]] if "view2" in
    observation else [])`; the two-view call is the same list, element for
    element."""
    observation = _observation(2)
    former = [observation["view1"]] + ([observation["view2"]] if "view2" in observation else [])
    assert observation_views(observation, 2) == former
    assert all(a is b for a, b in zip(observation_views(observation, 2), former))
    only_view1 = {"view1": observation["view1"]}
    assert observation_views(only_view1, 1) == [only_view1["view1"]]


def test_observation_views_takes_torch_frames_too():
    observation = {f"view{i + 1}": torch.zeros(4, 5, 3, dtype=torch.uint8) for i in range(3)}
    views = observation_views(observation, 3)
    assert views[2] is observation["view3"] and isinstance(views[0], torch.Tensor)


# -- the frontend's own view count (ISSUE-084) -------------------------------


def test_from_config_reports_the_workloads_view_count():
    """A frontend built for a workload reports that workload's cameras, even
    though its dims carry no view count of their own."""
    resolved = resolve_config(TARGET_WORKLOAD, STRUCTURE)
    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", lambda self, **kw: None):
        fe = ImageWAMTorchFrontendThor.from_config(resolved, workload=TARGET_WORKLOAD)
    assert fe.num_views == TARGET_WORKLOAD.num_views == 3
    assert fe._input_view_shape() == TARGET_WORKLOAD.vae_graph_input() == (3, 256, 256)


def test_hand_passed_dims_report_liberos_two_views():
    """A caller that passes dims by hand names no workload: two 224x224
    views, the geometry that caller got before the workload object existed."""
    fe = _mocked_frontend(_workload=None, _vae_stage=None)
    assert fe.num_views == 2
    assert fe._input_view_shape() == (2, 224, 224)


def test_in_graph_stage_reports_its_own_spec():
    stage = SimpleNamespace(spec=SimpleNamespace(num_views=3, in_h=256, in_w=256))
    fe = _mocked_frontend(_workload=None, _vae_stage=stage)
    assert fe.num_views == 3
    assert fe._input_view_shape() == (3, 256, 256)


def test_the_workload_outranks_the_stage_spec():
    """The workload is the authority; the stage (built from that same
    `vae_graph_input()`) is the fallback for a frontend that has no
    workload."""
    stage = SimpleNamespace(spec=SimpleNamespace(num_views=2, in_h=224, in_w=224))
    fe = _mocked_frontend(_workload=TARGET_WORKLOAD, _vae_stage=stage)
    assert fe.num_views == 3 and fe._input_view_shape() == (3, 256, 256)


def test_stage_images_rejects_a_view_count_other_than_num_views():
    """Before it reads the VAE: a three-view workload with two views is a
    mismatch, not a two-view run."""
    fe = _mocked_frontend(_workload=TARGET_WORKLOAD, _vae_stage=None, _ae=object())
    with pytest.raises(ValueError) as excinfo:
        fe.stage_images(object(), object())
    assert "3 views" in str(excinfo.value) and "got 2" in str(excinfo.value)

    two_view = _mocked_frontend(_workload=None, _vae_stage=None, _ae=object())
    with pytest.raises(ValueError) as excinfo:
        two_view.stage_images(object())
    assert "2 views" in str(excinfo.value) and "got 1" in str(excinfo.value)

    no_ae = _mocked_frontend(_workload=None, _vae_stage=None, _ae=None)
    with pytest.raises(RuntimeError, match="stage_images requires ae_model_path"):
        no_ae.stage_images(object(), object())


# -- the text length the workload implies (ISSUE-083) ------------------------


def _text_max_length(dims: dict, *, proprio: bool) -> int:
    """`_text_max_length()` reading only the two attributes it owns
    (`dims` and the constructor's resolved `_proprio_dim`)."""
    stub = SimpleNamespace(dims=dims, _proprio_dim=dims.get("proprio_dim") if proprio else None)
    return ImageWAMTorchFrontendThor._text_max_length(stub)


@pytest.mark.parametrize("workload", [LIBERO_WORKLOAD, TARGET_WORKLOAD],
                         ids=["libero", "target-3x256"])
def test_text_length_follows_the_workload(workload):
    dims = _resolved_dims(workload)
    assert dims["proprio_dim"] == workload.proprio_dim == 8
    # the proprio row takes one context row: the encoder pads to x0 - 1
    assert _text_max_length(dims, proprio=True) == dims["x0"] - 1 == workload.text_max_len
    # without a proprio row the context IS the encoder's own length
    assert _text_max_length(dims, proprio=False) == dims["x0"] == workload.text_max_len + 1


def test_libero_still_encodes_at_flux2s_own_length():
    """LIBERO: `x0=513` and `proprio_dim=8`, so the live Qwen3 branch asks
    for 512 rows -- exactly `_set_context_with_optional_proprio`'s rule
    (`x0 == text_len + 1`) and FLUX.2's own `MAX_LENGTH`."""
    dims = _resolved_dims(LIBERO_WORKLOAD)
    assert dims["x0"] == 513
    text_len = _text_max_length(dims, proprio=True)
    assert text_len == 512
    assert dims["x0"] == text_len + 1


def test_target_workload_encodes_at_its_own_128():
    """The 3x256 workload of `benchmarks/_imagewam_workload_cli`: three views
    of 256x256 give a 16x48 image-token grid (768 rows), `x0=129`, `a0=897`;
    the encoder pads to 128, not 512."""
    dims = _resolved_dims(TARGET_WORKLOAD)
    assert (dims["x0"], dims["a0"]) == (129, 897)
    assert (dims["ref_h"], dims["ref_w"]) == (16, 48)
    assert dims["ref_h"] * dims["ref_w"] == dims["a0"] - dims["x0"] == 768
    assert _text_max_length(dims, proprio=True) == 128 == TARGET_WORKLOAD.text_max_len


def test_set_prompt_passes_the_length_its_dims_imply():
    """The live-Qwen3 branch reads `_text_max_length()`, and the encoder
    module's own `_MAX_LENGTH` stays the default for a caller that passes
    none."""
    import inspect

    from flash_rt.models.imagewam import text_encoder

    src = inspect.getsource(ImageWAMTorchFrontendThor.set_prompt)
    assert "max_length=self._text_max_length()" in src
    parameter = inspect.signature(text_encoder.encode_prompts).parameters["max_length"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == text_encoder._MAX_LENGTH == 512


# -- encode_to_tokens' view validation (ISSUE-084) ---------------------------


class _NoAeAccess:
    """Stub autoencoder: `parameters()` is the first thing
    `encode_to_tokens` calls on `ae`, so reaching it means the view
    validation did not run first."""

    def parameters(self):
        raise AssertionError("encode_to_tokens read the autoencoder before validating `views`")


def test_encode_to_tokens_rejects_no_views():
    from flash_rt.models.imagewam.vae_encoder import encode_to_tokens

    with pytest.raises(ValueError, match="at least one view"):
        encode_to_tokens(_NoAeAccess(), [])


@pytest.mark.parametrize("view", [
    torch.zeros(224, 224, dtype=torch.uint8),        # missing the channel axis
    torch.zeros(224, 224, 4, dtype=torch.uint8),     # four channels
    torch.zeros(224, 224, 3, dtype=torch.float32),   # not uint8
    torch.zeros(3, dtype=torch.uint8),               # not an image at all
])
def test_encode_to_tokens_rejects_a_wrong_view(view):
    from flash_rt.models.imagewam.vae_encoder import encode_to_tokens

    with pytest.raises(ValueError, match="view 0 must be"):
        encode_to_tokens(_NoAeAccess(), [view])


def test_encode_to_tokens_names_the_offending_view():
    from flash_rt.models.imagewam.vae_encoder import encode_to_tokens

    views = [torch.zeros(224, 224, 3, dtype=torch.uint8), torch.zeros(224, 224)]
    with pytest.raises(ValueError, match=r"view 1 must be"):
        encode_to_tokens(_NoAeAccess(), views)


def test_encode_to_tokens_rejects_a_bare_tensor_instead_of_a_sequence():
    """The former two-view call style (`encode_to_tokens(ae, v1, v2)`) is
    gone; a single tensor passed where the sequence belongs says so."""
    from flash_rt.models.imagewam.vae_encoder import encode_to_tokens

    with pytest.raises(ValueError, match="must be a sequence"):
        encode_to_tokens(_NoAeAccess(), torch.zeros(224, 224, 3, dtype=torch.uint8))
