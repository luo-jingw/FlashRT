"""GPU-free dispatch tests for the Pi0.5 RTX NVFP4 encoder (use_fp4_encoder).

The kernel module is a recorder: every call is logged with its arguments,
so the tests check which kernels run, in which order, on which pointers,
with which tile variant -- not numerics (that is the RTX 5090 run's job).
"""
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest


class _Buf:
    def __init__(self, ptr, nbytes=1):
        self.ptr = SimpleNamespace(value=ptr)
        self.nbytes = nbytes


class _RecordingFvk:
    """Records every kernel call as (name, args)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _kernel(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return _kernel


def _symbols():
    try:
        from flash_rt.models.pi05 import pipeline_rtx
        from flash_rt.models.pi05.nvfp4_sm120 import Nvfp4ActBufferSm120, Nvfp4WeightSm120
    except (ImportError, OSError) as exc:
        pytest.skip(f"Pi05 RTX imports unavailable: {exc}")
    return SimpleNamespace(pipeline_rtx=pipeline_rtx, Act=Nvfp4ActBufferSm120,
                           Weight=Nvfp4WeightSm120)


_PROJ_N_K = {
    "encoder_attn_qkv_w": lambda s: ((s.ENC_NH + 2 * s.ENC_NKV) * s.ENC_HD, s.ENC_D),
    "encoder_attn_o_w": lambda s: (s.ENC_D, s.ENC_D),
    "encoder_ffn_gate_up_w": lambda s: (2 * s.ENC_H, s.ENC_D),
    "encoder_ffn_down_w": lambda s: (s.ENC_D, s.ENC_H),
}


def _make_pipe(sym, seq=781):
    P = sym.pipeline_rtx
    pipe = P.Pi05Pipeline.__new__(P.Pi05Pipeline)
    pipe.fvk = _RecordingFvk()
    pipe.gemm = Mock()  # cuBLASLt runner: must not be used by the NVFP4 encoder
    pipe.use_fp4_encoder = True
    pipe.use_fp8 = True
    pipe.fp8_calibrated = True
    pipe.use_int8_encoder = False
    pipe.encoder_seq_len = seq
    pipe.bufs = {name: _Buf(100 + idx) for idx, name in enumerate(
        ["encoder_x", "encoder_x_norm", "encoder_QKV", "encoder_rope_weights",
         "encoder_gate_merged", "encoder_hidden"])}
    pipe._rms_ones_enc = _Buf(900)
    pipe._attn_ptrs = {"enc_Q": 950}
    pipe._enc_kv_layer_ptrs = MethodType(lambda self, i, offset_tokens=0: (960 + i, 980 + i), pipe)
    pipe.attn = SimpleNamespace(run=Mock(return_value=777))
    pipe.enc_act_fp4 = sym.Act(packed=_Buf(1001), sf=_Buf(1002), rows=seq, k=P.ENC_D)
    pipe.enc_act_fp4_large = sym.Act(packed=_Buf(1011), sf=_Buf(1012), rows=seq, k=P.ENC_H)
    nvfp4 = {}
    for i in range(P.ENC_L):
        for base, nk in _PROJ_N_K.items():
            n, k = nk(P)
            nvfp4[f"{base}_{i}"] = sym.Weight(
                packed_ptr=hash((base, i, "p")) % 10**6, sf_ptr=hash((base, i, "s")) % 10**6,
                alpha=0.5, n=n, k=k)
    pipe.weights = {"nvfp4": nvfp4}
    return pipe


def _gemm_calls(fvk):
    return [(name, args) for name, args, _ in fvk.calls if name.startswith("fp4_w4a16_gemm_sm120")]


def test_middle_layer_runs_four_nvfp4_gemms_with_measured_tiles():
    sym = _symbols()
    P = sym.pipeline_rtx
    pipe = _make_pipe(sym)
    i, seq = 3, pipe.encoder_seq_len

    P.Pi05Pipeline._encoder_layer(pipe, i, seq, fuse_b1=True, stream=5)

    gemms = _gemm_calls(pipe.fvk)
    assert [name for name, _ in gemms] == [
        "fp4_w4a16_gemm_sm120_bf16out",           # qkv: plain
        "fp4_w4a16_gemm_sm120_bf16out",           # o: plain
        "fp4_w4a16_gemm_sm120_bf16out",           # gate_up: plain
        "fp4_w4a16_gemm_sm120_bf16out_pingpong",  # down: pingpong
    ]
    W = pipe.weights["nvfp4"]
    B = pipe.bufs
    expected = [
        (f"encoder_attn_qkv_w_{i}", pipe.enc_act_fp4, B["encoder_QKV"].ptr.value),
        (f"encoder_attn_o_w_{i}", pipe.enc_act_fp4, B["encoder_x_norm"].ptr.value),
        (f"encoder_ffn_gate_up_w_{i}", pipe.enc_act_fp4, B["encoder_gate_merged"].ptr.value),
        (f"encoder_ffn_down_w_{i}", pipe.enc_act_fp4_large, B["encoder_x_norm"].ptr.value),
    ]
    for (_, args), (wname, act, out_ptr) in zip(gemms, expected):
        w = W[wname]
        assert args == (act.packed.ptr.value, w.packed_ptr, out_ptr, seq, w.n, w.k,
                        act.sf.ptr.value, w.sf_ptr, w.alpha, 5)


@pytest.mark.parametrize("layer", [0, 5])
def test_each_gemm_input_is_quantized_from_the_right_buffer(layer):
    sym = _symbols()
    P = sym.pipeline_rtx
    pipe = _make_pipe(sym)
    seq = pipe.encoder_seq_len
    B = pipe.bufs
    act, act_large = pipe.enc_act_fp4, pipe.enc_act_fp4_large

    P.Pi05Pipeline._encoder_layer(pipe, layer, seq, fuse_b1=False, stream=0)

    norm_quant = [(name, args) for name, args, _ in pipe.fvk.calls if "rms_norm_to_nvfp4" in name]
    x, x_norm, ones = B["encoder_x"].ptr.value, B["encoder_x_norm"].ptr.value, pipe._rms_ones_enc.ptr.value
    fused_residual = ("residual_add_rms_norm_to_nvfp4_swizzled_bf16",
                      (x, x_norm, x, ones, act.packed.ptr.value, act.sf.ptr.value, seq, P.ENC_D, 1e-6, 0))
    if layer == 0:  # no previous residual to fold
        b1 = ("rms_norm_to_nvfp4_swizzled_bf16",
              (x, ones, act.packed.ptr.value, act.sf.ptr.value, seq, P.ENC_D, 1e-6, 0))
    else:  # previous layer's post-FFN residual folded in, in place
        b1 = fused_residual
    assert norm_quant == [b1, fused_residual]  # B1 -> qkv, B4 -> gate_up

    quants = [args for name, args, _ in pipe.fvk.calls if name == "quantize_bf16_to_nvfp4_swizzled"]
    assert quants == [
        (777, act.packed.ptr.value, act.sf.ptr.value, seq, P.ENC_D, 0),  # attention out -> o
        (B["encoder_hidden"].ptr.value, act_large.packed.ptr.value, act_large.sf.ptr.value,
         seq, P.ENC_H, 0),                                               # GeGLU -> down
    ]
    names = [name for name, _, _ in pipe.fvk.calls]
    assert "residual_add" not in names and "rms_norm" not in names
    assert not any("fp8" in name for name in names)
    assert pipe.gemm.method_calls == []


def test_last_layer_runs_qkv_only():
    sym = _symbols()
    P = sym.pipeline_rtx
    pipe = _make_pipe(sym)

    P.Pi05Pipeline._encoder_layer(pipe, P.ENC_L - 1, pipe.encoder_seq_len, fuse_b1=True, stream=0)

    gemms = _gemm_calls(pipe.fvk)
    assert len(gemms) == 1
    assert gemms[0][1][1] == pipe.weights["nvfp4"][f"encoder_attn_qkv_w_{P.ENC_L - 1}"].packed_ptr
    pipe.attn.run.assert_not_called()


def test_scratch_requires_nvfp4_weights():
    sym = _symbols()
    P = sym.pipeline_rtx
    pipe = P.Pi05Pipeline.__new__(P.Pi05Pipeline)
    pipe.use_fp4_encoder = True
    pipe.weights = {}
    pipe.encoder_seq_len = 781
    with pytest.raises(ValueError, match='weights\\["nvfp4"\\]'):
        P.Pi05Pipeline._allocate_fp4_encoder_scratch(pipe)


def test_gemm_rejects_mismatched_k():
    sym = _symbols()
    from flash_rt.models.pi05.nvfp4_sm120 import gemm_bf16out
    act = sym.Act(packed=_Buf(1), sf=_Buf(2), rows=16, k=2048)
    w = sym.Weight(packed_ptr=3, sf_ptr=4, alpha=1.0, n=2048, k=16384)
    with pytest.raises(ValueError, match="K=2048"):
        gemm_bf16out(_RecordingFvk(), "plain", act, w, 5, 16, 0)


def _frontend_for_validation(**overrides):
    try:
        from flash_rt.frontends.torch import pi05_rtx
    except (ImportError, OSError) as exc:
        pytest.skip(f"Pi05 RTX frontend imports unavailable: {exc}")
    fe = pi05_rtx.Pi05TorchFrontendRtx.__new__(pi05_rtx.Pi05TorchFrontendRtx)
    fe.use_fp8 = True
    fe._force_bf16 = False
    fe._force_int8_decoder = False
    fe._int8_encoder_only = False
    fe._use_int8_vision = False
    for k, v in overrides.items():
        setattr(fe, k, v)
    return pi05_rtx, fe


@pytest.mark.parametrize("overrides, match", [
    ({"use_fp8": False}, "use_fp8=True"),
    ({"_force_bf16": True}, "BF16 fallback"),
    ({"_force_int8_decoder": True}, "FORCE_INT8"),
    ({"_int8_encoder_only": True}, "FORCE_INT8"),
    ({"_use_int8_vision": True}, "FORCE_INT8"),
])
def test_frontend_rejects_unsupported_configs(monkeypatch, overrides, match):
    pi05_rtx, fe = _frontend_for_validation(**overrides)
    monkeypatch.setattr(pi05_rtx.torch.cuda, "get_device_capability", lambda *a: (12, 0))
    with pytest.raises(ValueError, match=match):
        fe._validate_fp4_encoder_config()


def test_frontend_rejects_non_sm120(monkeypatch):
    pi05_rtx, fe = _frontend_for_validation()
    monkeypatch.setattr(pi05_rtx.torch.cuda, "get_device_capability", lambda *a: (8, 9))
    with pytest.raises(ValueError, match="SM120"):
        fe._validate_fp4_encoder_config()


def test_rl_and_batched_modes_reject_fp4_encoder():
    pi05_rtx, fe = _frontend_for_validation()
    fe.use_fp4_encoder = True
    with pytest.raises(ValueError, match="use_fp4_encoder"):
        fe.set_rl_mode(cfg_enable=True)
    with pytest.raises(ValueError, match="use_fp4_encoder"):
        fe.set_batched_mode(enable=True)
