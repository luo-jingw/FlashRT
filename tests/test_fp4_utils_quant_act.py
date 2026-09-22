"""`quant_act_nvfp4`'s vectorized-first, scalar-fallback call sequence
(`flash_rt/executors/fp4_utils.py`).

CPU only: `flash_rt.flash_rt_fp4` is stubbed, so no compiled kernel or GPU is
needed. The vectorized kernel (`quantize_fp4_dynamic_sfa_fp16_vec`,
`csrc/quantize/quantize_fp4_sfa_vec.cu`) is documented bit-exact with the
scalar one and refuses (rc != 0) only on misaligned pointers or K not a
multiple of 16; its numerical correctness needs a real GPU and is not
re-checked here (Thor; this machine has no compiled kernel to run it
against).
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
import torch


@pytest.fixture
def fp4_utils(monkeypatch):
    """`flash_rt.executors.fp4_utils` imported fresh under a stub
    `flash_rt.flash_rt_fp4` that records every call, restored afterwards."""
    calls: list[tuple[str, tuple]] = []
    stub = types.ModuleType("flash_rt.flash_rt_fp4")

    def make(name, rc):
        def fn(*args):
            calls.append((name, args))
            return rc[0]
        return fn

    rc_vec, rc_scalar = [0], [0]
    stub.quantize_fp4_dynamic_sfa_fp16_vec = make("vec", rc_vec)
    stub.quantize_fp4_dynamic_sfa_fp16 = make("scalar", rc_scalar)
    stub.sfa_size_bytes = lambda m, k, is_sfb: 64

    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fp4", stub)
    monkeypatch.delitem(sys.modules, "flash_rt.executors.fp4_utils", raising=False)  # force a fresh import
    import importlib

    mod = importlib.import_module("flash_rt.executors.fp4_utils")
    yield types.SimpleNamespace(mod=mod, calls=calls, rc_vec=rc_vec, rc_scalar=rc_scalar)
    monkeypatch.delitem(sys.modules, "flash_rt.executors.fp4_utils", raising=False)


class _FakeScratch:
    def __init__(self, K):
        self.K = K
        self.max_M = 1024
        self.packed = mock.Mock(data_ptr=lambda: 111)
        self.sfa = mock.Mock(data_ptr=lambda: 222)


def _x(K=32):
    x = mock.Mock(spec=torch.Tensor)
    x.dtype = torch.float16
    x.device = mock.Mock(type="cuda")
    x.data_ptr = lambda: 100
    return x


def test_vec_tried_first_and_scalar_not_called_when_it_succeeds(fp4_utils):
    fp4_utils.mod.quant_act_nvfp4(_x(), _FakeScratch(32), M=64)
    assert [name for name, _ in fp4_utils.calls] == ["vec"]


def test_falls_back_to_scalar_when_vec_refuses(fp4_utils):
    fp4_utils.rc_vec[0] = -1
    fp4_utils.mod.quant_act_nvfp4(_x(), _FakeScratch(32), M=64)
    assert [name for name, _ in fp4_utils.calls] == ["vec", "scalar"]


def test_raises_when_both_fail(fp4_utils):
    fp4_utils.rc_vec[0] = -1
    fp4_utils.rc_scalar[0] = -1
    with pytest.raises(RuntimeError, match="failed rc=-1"):
        fp4_utils.mod.quant_act_nvfp4(_x(), _FakeScratch(32), M=64)
    assert [name for name, _ in fp4_utils.calls] == ["vec", "scalar"]


def test_both_calls_use_the_same_arguments(fp4_utils):
    fp4_utils.rc_vec[0] = -1
    x, scratch = _x(K=48), _FakeScratch(48)
    fp4_utils.mod.quant_act_nvfp4(x, scratch, M=17, stream=3)
    (_, vec_args), (_, scalar_args) = fp4_utils.calls
    assert vec_args == scalar_args == (100, 111, 222, 17, 48, False, 3)
