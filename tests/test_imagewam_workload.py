"""`ImageWAMWorkload` layout derivation and validation (plan.md W1, W8).

CPU only. The LIBERO numbers are pinned twice: as a literal table in this
file (so a change to the derivation cannot silently move them) and
against `libero_dims.LIBERO_REAL_DIMS`, which is itself the `dims`
mapping `resolve_config(libero(), structure)` produces.

`layout()` reaches the structure through `patch_stride` and
`max_action_horizon` only, so the generic derivation and the invalid-input
cases use a local stand-in; the LIBERO cases use the real
`ImageWAMStructure.libero()`.
"""
from __future__ import annotations

import dataclasses
from collections import namedtuple

import pytest

from flash_rt.models.imagewam.config_resolver import resolve_config
from flash_rt.models.imagewam.libero_dims import (
    LIBERO_HORIZON, LIBERO_REAL_DIMS, LIBERO_SHIFT, LIBERO_STEPS,
)
from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload, SequenceLayout

Struct = namedtuple("Struct", "patch_stride max_action_horizon")
LIBERO_STRUCT = Struct(patch_stride=16, max_action_horizon=64)
LAYOUT_KEYS = ("x0", "a0", "total", "ref_h", "ref_w", "dt")

# The literal dims of the real ImageWAM-FLUX.2-4B-LIBERO release: the keys
# `LIBERO_REAL_DIMS` held before it was derived from the workload and the
# structure, plus `action_dim` and the camera geometry (`num_views`,
# `image_h`, `image_w`), which the resolver's mapping adds.
LIBERO_LITERAL_DIMS: dict = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    action_dim=7, num_action=64, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=0.1, num_denoise_steps=10,
    ref_h=14, ref_w=28, proprio_dim=8, shift=5.0, num_train_timesteps=1000,
    num_views=2, image_h=224, image_w=224,
)
# `img_len` is not a `dims` key; it is the image token count, `a0 - x0`.
LIBERO_LITERAL_IMG_LEN = 392


def _with(**kw) -> ImageWAMWorkload:
    return dataclasses.replace(ImageWAMWorkload.libero(), **kw)


def test_libero_real_dims_is_the_literal_table():
    assert LIBERO_REAL_DIMS == LIBERO_LITERAL_DIMS
    assert (LIBERO_HORIZON, LIBERO_STEPS, LIBERO_SHIFT) == (64, 10, 5.0)


def test_libero_layout_equals_libero_real_dims():
    w = ImageWAMWorkload.libero()
    lay = w.layout(ImageWAMStructure.libero())
    assert lay.img_len == LIBERO_LITERAL_IMG_LEN
    for key in LAYOUT_KEYS:
        assert getattr(lay, key) == LIBERO_REAL_DIMS[key] == LIBERO_LITERAL_DIMS[key], key
    assert w.action_horizon == LIBERO_REAL_DIMS["num_action"] == LIBERO_LITERAL_DIMS["num_action"]
    assert lay.img_len == LIBERO_REAL_DIMS["a0"] - LIBERO_REAL_DIMS["x0"]
    assert (lay.x0, lay.img_len, lay.a0, lay.total, lay.ref_h, lay.ref_w) == (513, 392, 905, 969, 14, 28)


def test_resolve_config_dims_equal_libero_real_dims():
    """plan.md: `LIBERO_REAL_DIMS` is the resolver's dims mapping for
    `libero()` and the real structure, so its 27 literals are one
    definition and not a copy."""
    r = resolve_config(ImageWAMWorkload.libero(), ImageWAMStructure.libero())
    assert r.dims == LIBERO_REAL_DIMS
    assert set(r.dims) == set(LIBERO_REAL_DIMS) == set(LIBERO_LITERAL_DIMS)


def test_libero_scalars_equal_libero_real_dims():
    w = ImageWAMWorkload.libero()
    assert w.action_horizon == LIBERO_REAL_DIMS["num_action"] == LIBERO_HORIZON
    assert w.action_dim == LIBERO_REAL_DIMS["action_dim"] == 7
    assert w.proprio_dim == LIBERO_REAL_DIMS["proprio_dim"]
    assert w.num_steps == LIBERO_REAL_DIMS["num_denoise_steps"] == LIBERO_STEPS
    assert w.shift == LIBERO_REAL_DIMS["shift"] == LIBERO_SHIFT
    assert w.num_train_timesteps == LIBERO_REAL_DIMS["num_train_timesteps"]
    # The camera geometry the layout was derived from is part of the dims
    # (the calibration file's identity reads it, calibration_file.py).
    for key in ("num_views", "image_h", "image_w"):
        assert getattr(w, key) == LIBERO_REAL_DIMS[key] == LIBERO_LITERAL_DIMS[key], key
    assert (LIBERO_REAL_DIMS["num_views"], LIBERO_REAL_DIMS["image_h"], LIBERO_REAL_DIMS["image_w"]) \
        == (2, 224, 224)


@pytest.mark.parametrize("views,h,w,ref_h,ref_w", [
    (1, 224, 224, 14, 14),
    (2, 224, 224, 14, 28),
    (3, 224, 224, 14, 42),
    (2, 256, 320, 16, 40),
    (1, 448, 224, 28, 14),
])
def test_grid_derivation(views, h, w, ref_h, ref_w):
    lay = _with(num_views=views, image_h=h, image_w=w).layout(LIBERO_STRUCT)
    assert (lay.ref_h, lay.ref_w) == (ref_h, ref_w)
    assert lay.img_len == ref_h * ref_w
    assert lay.a0 == lay.x0 + lay.img_len
    assert lay.total == lay.a0 + 64


def test_single_view_layout():
    lay = _with(num_views=1).layout(LIBERO_STRUCT)
    assert (lay.img_len, lay.a0, lay.total) == (196, 709, 773)


def test_text_horizon_steps_derivation():
    lay = _with(text_max_len=128, action_horizon=16, num_steps=4).layout(LIBERO_STRUCT)
    assert lay.x0 == 129
    assert lay.a0 == 129 + 392
    assert lay.total == lay.a0 + 16
    assert lay.dt == 0.25


def test_layout_is_a_frozen_sequence_layout():
    lay = ImageWAMWorkload.libero().layout(LIBERO_STRUCT)
    assert isinstance(lay, SequenceLayout)
    with pytest.raises(dataclasses.FrozenInstanceError):
        lay.x0 = 1


def test_workload_is_frozen():
    w = ImageWAMWorkload.libero()
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.num_views = 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.text_max_len = 1
    assert hash(w) == hash(ImageWAMWorkload.libero())


@pytest.mark.parametrize("field", [
    "num_views", "image_h", "image_w", "text_max_len", "action_horizon", "action_dim",
    "proprio_dim", "num_steps", "num_train_timesteps"])
@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_dimension_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _with(**{field: value})


@pytest.mark.parametrize("value", [0.0, -5.0])
def test_non_positive_shift_rejected(value):
    with pytest.raises(ValueError, match="shift"):
        _with(shift=value)


def test_num_views_below_one_rejected():
    with pytest.raises(ValueError, match="num_views"):
        _with(num_views=0)


@pytest.mark.parametrize("field,value", [("image_h", 225), ("image_w", 230), ("image_h", 8)])
def test_image_size_not_multiple_of_stride_rejected(field, value):
    w = _with(**{field: value})  # divisibility depends on the structure, checked in layout()
    with pytest.raises(ValueError, match=field):
        w.layout(LIBERO_STRUCT)


def test_stride_is_read_from_structure():
    w = _with(image_h=225, image_w=225)
    lay = w.layout(Struct(patch_stride=15, max_action_horizon=64))
    assert (lay.ref_h, lay.ref_w) == (15, 30)
    with pytest.raises(ValueError, match="image_h"):
        w.layout(Struct(patch_stride=16, max_action_horizon=64))


def test_horizon_above_structure_limit_rejected():
    with pytest.raises(ValueError, match="action_horizon"):
        _with(action_horizon=65).layout(LIBERO_STRUCT)
    assert _with(action_horizon=65).layout(Struct(16, 65)).total == 905 + 65


def test_non_integer_field_rejected():
    with pytest.raises(ValueError, match="num_views"):
        _with(num_views=2.5)
    with pytest.raises(ValueError, match="num_views"):
        _with(num_views=True)


def test_bad_structure_stride_rejected():
    with pytest.raises(ValueError, match="patch_stride"):
        ImageWAMWorkload.libero().layout(Struct(patch_stride=0, max_action_horizon=64))


def test_libero_vae_graph_input():
    # benchmarks/imagewam_e2e_official_compare.py: (2,) + (224, 224) with pre-sized views
    assert ImageWAMWorkload.libero().vae_graph_input() == (2, 224, 224)
    assert _with(num_views=3, image_h=256, image_w=320).vae_graph_input() == (3, 256, 320)
