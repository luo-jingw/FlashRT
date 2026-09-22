"""`_fused_gate_res`'s OPT-032 candidate 3 dispatch (`dims["fuse_res_norm_fp4"]`,
`flash_rt/models/imagewam/pipeline_thor.py`) and `_awq_target`'s `lin`
attachment.

CPU only: no CUDA, no `flash_rt.flash_rt_fp4` build needed. A fake
`Nvfp4Linear` is constructed via `object.__new__` (skipping its real
`__init__`, which imports the Blackwell-only `flash_rt.flash_rt_fp4`
module this machine cannot build) so `isinstance(lin, Nvfp4Linear)`
still holds -- this checks the CALL-ORDER / dispatch contract only.
The fused kernel's own numerics are checked separately: bit-exact on
Ada (`tests/test_fused_norm_fp4_kernel.py`) and confirmed on Thor
against the currently-wired unfused pair (THOR_CHECKLIST.md X6).
Whether this dispatch reaches the real kernel with the right pointers
end-to-end, and the resulting layer output, needs a Thor rebuild with
NVFP4 (`ENABLE_SM100_CUTLASS`) -- not re-checked here.
"""
from __future__ import annotations

from unittest import mock

import pytest
import torch

from flash_rt.models.imagewam.pipeline_thor import AdaLNTarget, _awq_target, _fused_gate_res
from flash_rt.models.imagewam.quant_linear import Nvfp4Linear

DIM = 8


def _fp32(dim=DIM):
    """`(1, 1, dim)` FP32, matching the real modulation chunk shape
    (`adaln.modulation`'s own chunk output) -- the fp4_direct branch
    indexes `[0, 0]` into it, unlike `_mod_vec_ptr`'s plain-path
    contract which only needs `numel() == dim` and a contiguous last
    dim, so this shape satisfies both call paths."""
    return torch.zeros(1, 1, dim, dtype=torch.float32)


def _fake_nvfp4_linear(*, awq_inv_s=None):
    """A real `Nvfp4Linear` instance (so `isinstance` holds) built
    without running its Blackwell-only `__init__`."""
    lin = object.__new__(Nvfp4Linear)
    lin._awq_inv_s = awq_inv_s
    lin.scratch = mock.Mock(
        max_M=1024,
        packed=mock.Mock(data_ptr=lambda: 0x1000),
        sfa=mock.Mock(data_ptr=lambda: 0x2000),
    )
    lin._ensure_scratch = mock.Mock()  # already sized in this test
    lin._fvk_fp4 = mock.Mock(
        gate_res_ada_layer_norm_fp4_sfa_bf16res=mock.Mock(),
        gate_res_ada_layer_norm_fp4_sfa_fp16res=mock.Mock(),
    )
    return lin


def _fake_fvk():
    return mock.Mock(
        gate_res_ada_layer_norm_bf16res=mock.Mock(),
        gate_res_ada_layer_norm_fp16=mock.Mock(),
    )


def test_fp4_direct_dispatches_to_the_fused_fp4_kernel_not_the_plain_one():
    fvk = _fake_fvk()
    lin = _fake_nvfp4_linear()
    target = AdaLNTarget(_fp32(), _fp32(), out_ptr=0xDEAD, lin=lin)

    _fused_gate_res(fvk, 0x10, _fp32(), 0x20, 4, DIM, target, 0,
                     bf16_residual=True, eps=1e-6, fp4_direct=True)

    lin._fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_bf16res.assert_called_once()
    lin._fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_fp16res.assert_not_called()
    fvk.gate_res_ada_layer_norm_bf16res.assert_not_called()
    lin._ensure_scratch.assert_called_once_with(4)

    args = lin._fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_bf16res.call_args.args
    residual, gemm_out, gate, scale, shift, inv_s, packed, sfa, seq_len, dim, eps, stream = args
    assert (residual, gemm_out) == (0x20, 0x10)
    assert (packed, sfa) == (0x1000, 0x2000)
    assert (seq_len, dim, eps, stream) == (4, DIM, 1e-6, 0)
    assert inv_s == 0  # no AWQ fold in this test


def test_fp16_residual_variant_picked_when_bf16_residual_is_false():
    fvk = _fake_fvk()
    lin = _fake_nvfp4_linear()
    target = AdaLNTarget(_fp32(), _fp32(), out_ptr=0xDEAD, lin=lin)

    _fused_gate_res(fvk, 0x10, _fp32(), 0x20, 4, DIM, target, 0,
                     bf16_residual=False, eps=1e-6, fp4_direct=True)

    lin._fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_fp16res.assert_called_once()
    lin._fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_bf16res.assert_not_called()


def test_awq_inv_s_forwarded_when_present():
    """`inv_s` is `const __half*` in the kernel (same as gate/scale/
    shift), so `_fused_gate_res` must round the FP32 `awq_inv_s` to a
    fresh FP16 tensor before passing its pointer -- checking pointer
    identity against the original FP32 tensor would be wrong (and
    would have hidden the real bug this branch had, THOR_STATUS_SUMMARY.md
    `0922f`: passing the FP32 pointer directly to a `const __half*`
    parameter). Non-zero and distinct from the no-AWQ (0) case is what
    a CPU-only test can check."""
    fvk = _fake_fvk()
    lin = _fake_nvfp4_linear(awq_inv_s=_fp32())
    target = AdaLNTarget(_fp32(), _fp32(), out_ptr=0xDEAD, lin=lin)

    _fused_gate_res(fvk, 0x10, _fp32(), 0x20, 4, DIM, target, 0,
                     bf16_residual=True, eps=1e-6, fp4_direct=True)

    args = lin._fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_bf16res.call_args.args
    assert args[5] != 0
    assert args[5] != lin.awq_inv_s.data_ptr()  # a fresh FP16 tensor, not the original FP32 one


def test_gate_scale_shift_are_rounded_to_fp16_not_passed_as_fp32():
    """Regression test for the exact bug found on Thor (`0922f`): this
    kernel's `gate`/`scale`/`shift` are `const __half*`, unlike
    `gate_res_ada_layer_norm_bf16res`'s `const float*` -- passing the
    FP32 chunk's own pointer reads garbage at half the intended stride.

    Reads the bytes back via `ctypes` INSIDE the mock kernel's own
    `side_effect` -- i.e. synchronously, while `_fused_gate_res`'s local
    FP16 tensors are still alive and referenced on its stack frame.
    Reading them back from the saved `call_args` after the function has
    already returned is unsound for a CPU tensor with no consumer (the
    real kernel call, which keeps a CUDA-stream-ordered claim on the
    memory until it actually runs): its refcount hits zero the moment
    `_fused_gate_res` returns, and CPython/the CPU allocator are free to
    reuse that memory for anything before this test ever reads it --
    caught by a first attempt at this test that read stale/reused
    bytes, not a real production bug."""
    import ctypes
    import struct

    def _read_fp16(ptr: int) -> float:
        buf = (ctypes.c_uint16 * 1).from_address(ptr)
        return struct.unpack("<e", struct.pack("<H", buf[0]))[0]

    seen = {}

    def _capture(residual, gemm_out, gate_ptr, scale_ptr, shift_ptr, *rest):
        seen["gate"] = _read_fp16(gate_ptr)
        seen["scale"] = _read_fp16(scale_ptr)
        seen["shift"] = _read_fp16(shift_ptr)

    fvk = _fake_fvk()
    lin = _fake_nvfp4_linear()
    lin._fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_bf16res.side_effect = _capture
    gate = _fp32()
    gate[0, 0, 0] = 3.5
    scale = _fp32()
    scale[0, 0, 0] = -1.25
    shift = _fp32()
    shift[0, 0, 0] = 0.75
    target = AdaLNTarget(shift, scale, out_ptr=0xDEAD, lin=lin)

    _fused_gate_res(fvk, 0x10, gate, 0x20, 4, DIM, target, 0,
                     bf16_residual=True, eps=1e-6, fp4_direct=True)

    assert seen["gate"] == pytest.approx(3.5)
    assert seen["scale"] == pytest.approx(-1.25)
    assert seen["shift"] == pytest.approx(0.75)


def test_fp4_direct_default_off_keeps_the_plain_path_even_for_an_nvfp4_target():
    """Backward compatibility: existing callers that don't pass
    `fp4_direct` (every call site before this round) must be unaffected
    even if `target.lin` happens to be an `Nvfp4Linear` -- the flag,
    not the object's type, gates the new behavior."""
    fvk = _fake_fvk()
    lin = _fake_nvfp4_linear()
    target = AdaLNTarget(_fp32(), _fp32(), out_ptr=0xDEAD, lin=lin)

    _fused_gate_res(fvk, 0x10, _fp32(), 0x20, 4, DIM, target, 0, bf16_residual=True, eps=1e-6)

    fvk.gate_res_ada_layer_norm_bf16res.assert_called_once()
    lin._fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_bf16res.assert_not_called()


def test_non_nvfp4_target_keeps_the_plain_path_even_with_fp4_direct_true():
    fvk = _fake_fvk()
    target = AdaLNTarget(_fp32(), _fp32(), out_ptr=0xDEAD, lin=object())

    _fused_gate_res(fvk, 0x10, _fp32(), 0x20, 4, DIM, target, 0,
                     bf16_residual=True, eps=1e-6, fp4_direct=True)

    fvk.gate_res_ada_layer_norm_bf16res.assert_called_once()


def test_target_none_takes_the_plain_no_target_path():
    fvk = _fake_fvk()
    _fused_gate_res(fvk, 0x10, _fp32(), 0x20, 4, DIM, None, 0,
                     bf16_residual=True, eps=1e-6, fp4_direct=True)
    fvk.gate_res_ada_layer_norm_bf16res.assert_called_once_with(0x10, mock.ANY, 0x20, 0, 0, 0, 4, DIM, 1e-6, 0)


def test_awq_target_always_attaches_lin_even_without_awq_scale():
    plain_lin = object()
    target = AdaLNTarget(_fp32(), _fp32(), out_ptr=0xDEAD)
    out = _awq_target(plain_lin, target)
    assert out.lin is plain_lin
    assert out.shift is target.shift and out.scale is target.scale and out.out_ptr == target.out_ptr


def test_awq_target_none_passthrough():
    assert _awq_target(object(), None) is None
