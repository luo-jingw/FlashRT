"""`ImageWAMWorkload` layout derivation and validation (plan.md W1).

CPU only. `layout()` reads the structure through `patch_stride` and
`max_action_horizon`, so a local stand-in replaces `ImageWAMStructure`.
"""
from __future__ import annotations

import dataclasses
from collections import namedtuple

import pytest

from flash_rt.models.imagewam.libero_dims import LIBERO_REAL_DIMS
from flash_rt.models.imagewam.workload import ImageWAMWorkload, SequenceLayout

Struct = namedtuple("Struct", "patch_stride max_action_horizon")
LIBERO_STRUCT = Struct(patch_stride=16, max_action_horizon=64)
LAYOUT_KEYS = ("x0", "a0", "total", "ref_h", "ref_w", "dt")


def _with(**kw) -> ImageWAMWorkload:
    return dataclasses.replace(ImageWAMWorkload.libero(), **kw)


def test_libero_layout_equals_libero_real_dims():
    lay = ImageWAMWorkload.libero().layout(LIBERO_STRUCT)
    for key in LAYOUT_KEYS:
        assert getattr(lay, key) == LIBERO_REAL_DIMS[key], key
    assert lay.img_len == LIBERO_REAL_DIMS["a0"] - LIBERO_REAL_DIMS["x0"] == 392
    assert (lay.x0, lay.img_len, lay.a0, lay.total, lay.ref_h, lay.ref_w) == (513, 392, 905, 969, 14, 28)


def test_libero_scalars_equal_libero_real_dims():
    w = ImageWAMWorkload.libero()
    assert w.action_horizon == LIBERO_REAL_DIMS["num_action"]
    assert w.proprio_dim == LIBERO_REAL_DIMS["proprio_dim"]
    assert w.num_steps == LIBERO_REAL_DIMS["num_denoise_steps"]
    assert w.shift == LIBERO_REAL_DIMS["shift"]
    assert w.num_train_timesteps == LIBERO_REAL_DIMS["num_train_timesteps"]


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
