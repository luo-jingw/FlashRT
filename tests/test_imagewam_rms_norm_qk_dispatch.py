"""`_rms_norm_qk`'s `dims["vec_rms_norm"]` dispatch
(`flash_rt/models/imagewam/pipeline_thor.py`, OPT-032 candidate 5).

CPU only: no CUDA, no build needed -- `fvk` is a plain mock, so this
checks only the CALL-ORDER / dispatch contract (which function gets
called, with which positional arguments) and the default-off behaviour.
The warp-per-row kernel's own numerics (`rms_norm_fp16_vec`,
`csrc/kernels/vec_fp16_backbone.cu`) and its real speed/cosine at
ImageWAM's production shape need Thor/RTX hardware -- not re-checked
here.
"""
from __future__ import annotations

from unittest import mock

from flash_rt.models.imagewam.pipeline_thor import _rms_norm_qk

ARGS = (111, 222, 333, 64 * 24, 128, 1e-6, 0)  # x, w, out, rows, dim, eps, stream


def test_default_off_calls_the_plain_kernel():
    fvk = mock.Mock()
    _rms_norm_qk(fvk, {}, *ARGS)
    fvk.rms_norm_fp16.assert_called_once_with(*ARGS)
    fvk.rms_norm_fp16_vec.assert_not_called()


def test_vec_rms_norm_false_calls_the_plain_kernel():
    fvk = mock.Mock()
    _rms_norm_qk(fvk, {"vec_rms_norm": False}, *ARGS)
    fvk.rms_norm_fp16.assert_called_once_with(*ARGS)
    fvk.rms_norm_fp16_vec.assert_not_called()


def test_vec_rms_norm_true_calls_the_warp_per_row_kernel():
    fvk = mock.Mock()
    _rms_norm_qk(fvk, {"vec_rms_norm": True}, *ARGS)
    fvk.rms_norm_fp16_vec.assert_called_once_with(*ARGS)
    fvk.rms_norm_fp16.assert_not_called()
