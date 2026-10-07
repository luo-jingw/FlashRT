"""`_rms_norm_qk`'s `dims["vec_rms_norm"]` dispatch
(`flash_rt/models/imagewam/pipeline_thor.py`, OPT-032 candidate 5).

CPU only: no CUDA, no build needed -- `fvk` is a plain mock, so this
checks only the CALL-ORDER / dispatch contract (which function gets
called, with which positional arguments), the default-off behaviour,
and the return-code check added after a real gap was found on RTX 5090
(`rms_norm_fp16_vec` silently skips its computation and returns -1 on
an unmet precondition -- dim%8!=0 or an unaligned pointer -- instead of
raising on its own; nothing before this test caught that it was never
checked). The warp-per-row kernel's own numerics
(`csrc/kernels/vec_fp16_backbone.cu`) and its real speed/cosine at
ImageWAM's production shape need Thor/RTX hardware -- not re-checked
here (both have been confirmed there, see this function's own
docstring).
"""
from __future__ import annotations

from unittest import mock

import pytest

from flash_rt.models.imagewam.pipeline_thor import _rms_norm_qk

ARGS = (111, 222, 333, 64 * 24, 128, 1e-6, 0)  # x, w, out, rows, dim, eps, stream


def _fvk_with_vec_rc(rc: int = 0):
    fvk = mock.Mock()
    fvk.rms_norm_fp16_vec.return_value = rc
    return fvk


def test_default_off_calls_the_plain_kernel():
    fvk = _fvk_with_vec_rc()
    _rms_norm_qk(fvk, {}, *ARGS)
    fvk.rms_norm_fp16.assert_called_once_with(*ARGS)
    fvk.rms_norm_fp16_vec.assert_not_called()


def test_vec_rms_norm_false_calls_the_plain_kernel():
    fvk = _fvk_with_vec_rc()
    _rms_norm_qk(fvk, {"vec_rms_norm": False}, *ARGS)
    fvk.rms_norm_fp16.assert_called_once_with(*ARGS)
    fvk.rms_norm_fp16_vec.assert_not_called()


def test_vec_rms_norm_true_calls_the_warp_per_row_kernel():
    fvk = _fvk_with_vec_rc(0)
    _rms_norm_qk(fvk, {"vec_rms_norm": True}, *ARGS)
    fvk.rms_norm_fp16_vec.assert_called_once_with(*ARGS)
    fvk.rms_norm_fp16.assert_not_called()


def test_vec_rms_norm_nonzero_return_code_raises():
    """The real gap found on RTX 5090: `rms_norm_fp16_vec` returning -1 (an
    unmet precondition) used to be silently ignored -- the computation
    never ran and the caller got on with stale/garbage `out` data with no
    error at all."""
    fvk = _fvk_with_vec_rc(-1)
    with pytest.raises(RuntimeError, match="rms_norm_fp16_vec returned -1"):
        _rms_norm_qk(fvk, {"vec_rms_norm": True}, *ARGS)
