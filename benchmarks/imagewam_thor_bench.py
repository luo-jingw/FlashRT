#!/usr/bin/env python
"""ImageWAM Thor structural-dry-run speed benchmark (plan.md, all phases).

**Rewritten 2026-09-14 alongside `pipeline_thor.py`'s real-math rewrite**
(opportunities.md OPT-002 / PROJECT.md "Confirmed end goal") -- measures
the REAL per-layer math (per-head K/V, QK-Norm, real 4-axis RoPE, real
AdaLN modulation, real SiLU-gated-GLU MLP, real no-mask attention) at
ImageWAM's REAL per-head geometry (NH=24, HD=128, backbone
hidden=3072/mlp=9216, ActionDiT hidden=1024/attn_width=3072/mlp=4096 --
from `_imagewam_thor_spec.py`'s confirmed FLUX.2-klein-4B/ActionDiT
config values), then derives whole-pipeline totals by simple
arithmetic (layer count x per-layer time). This is NOT accuracy
validation -- random weights, explicitly out of scope per PROJECT.md
-- and NOT a Thor number: this runs on the project's own Ada (sm_89)
8GB GPU.

Deliberately does NOT allocate the full 25-layer weight set at once
(~5.5GB of random FP16 weights at real dims, now larger still with the
real per-head K/V and doubled MLP-gate widths) -- this machine has
~6.7GB free on an 8GB shared laptop GPU. Instead, each layer TYPE
(backbone double-stream, backbone single-stream, ActionDiT
double-stream, ActionDiT single-stream) is benchmarked in isolation
with a 1-layer attention spec, called repeatedly at layer_idx=0 --
valid because every layer of the same type has identical shapes and
therefore identical steady-state cost (confirmed by GemmRunner's own
per-(op,M,N,K) cache: repeated identical-shape calls hit the same
cached cuBLASLt algorithm).

Sequence lengths are a representative choice, not a confirmed real
ImageWAM deployment value (that number was never established in this
project's research) -- kept at total=960 (a0=896 + num_action=64,
64 = max_action_horizon from _imagewam_thor_spec.py) to stay under
softmax_mot_joint_fp16's confirmed 1024-column ceiling (see
csrc/kernels/softmax.cu: SM_MAX_COLS=1024 -- columns beyond that are
silently never read by the single-warp-per-row reduction loop, which
would make a >1024 timing number meaningless: the kernel would be
doing LESS work than a correct implementation needs, not more).
"""
from __future__ import annotations

import os
import statistics

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.pipeline_real import compute_action_modulation, compute_shared_modulation
from flash_rt.models.imagewam.pipeline_thor import (
    _action_double_layer,
    _action_single_layer,
    _double_stream_layer,
    _single_stream_layer,
)
from flash_rt.models.imagewam.quant_linear import Fp16Linear, Fp8Linear, Nvfp4Linear, StaticFp8Linear
from flash_rt.models.imagewam.rope import build_action_rope_table, build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16
F32 = torch.float32

# Real confirmed dims (_imagewam_thor_spec.py).
HIDDEN, HD, NH, MLP_HIDDEN, JOINT_ATTN_DIM = 3072, 128, 24, 9216, 7680
ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH, ACTION_MLP_HIDDEN = 1024, 3072, 4096
NUM_DOUBLE, NUM_SINGLE = 5, 20
MAX_ACTION_HORIZON = 64

# Representative (not confirmed-real) sequence lengths, see module docstring.
X0, A0 = 128, 896
NUM_ACTION = MAX_ACTION_HORIZON
TOTAL = A0 + NUM_ACTION  # 960, under the 1024 softmax ceiling

WARMUP, ITERS = 15, 50

# OPT-005: FA4 for the "backbone" site only ("mot"/ActionDiT has no
# FA4-equivalent mask support -- unaffected either way). Opt-in via env
# var, not a hardcoded default here, since this same script also runs
# on this dev machine's own Ada GPU (no FA4 runtime at all). Verified
# correct AND fast on real Thor hardware for the real per-head
# convention this script already uses (cosine=1.000000, 3.75x
# standalone at real a0=896 dims -- opportunities.md OPT-005's own
# 2026-09-14 entry): `IMAGEWAM_USE_FA4=1 python3 imagewam_thor_bench.py`
# on Thor to fold that win into the backbone_double/backbone_single
# numbers below.
USE_FA4 = os.environ.get("IMAGEWAM_USE_FA4", "0") == "1"

# OPT-004 step 5: FP8/NVFP4 quantized GEMM (plan.md). Opt-in via env
# var, same reasoning as IMAGEWAM_USE_FA4 -- this dev machine can
# WIRE both paths but cannot numerically verify either one locally
# (FP8: pre-existing Ada cuBLASLt gap, `cublasLtMatmulAlgoGetHeuristic`
# status 15 at every shape; NVFP4: flash_rt.flash_rt_fp4 is a
# Blackwell/Thor-only compiled extension, not built on Ada) -- see
# quant_linear.py's own module docstring. Needs real Thor hardware to
# produce meaningful timing/correctness numbers for anything but
# "fp16": `IMAGEWAM_PRECISION=fp8 python3 imagewam_thor_bench.py`.
#
# OPT-004 step 6 (plan.md): "fp8_static"/"fp8_static_cutlass" add a
# one-time `.calibrate()` call before this file's own `_autotune`/
# `_time_ms` (see `_calibrate_static_fp8` below) -- static, calibrate-
# once activation scale instead of Fp8Linear's per-call dynamic one;
# "_cutlass" additionally swaps the GEMM itself to
# `cutlass_fp8_sq`/`_wide`/`_t1` (Thor-only, same `ENABLE_SM100_CUTLASS`
# gate NVFP4 already uses). Both untestable on this Ada machine for the
# same reasons as "fp8" above.
PRECISION = os.environ.get("IMAGEWAM_PRECISION", "fp16")

_keepalive = []


def _rand(*shape, dtype=FP16, scale=0.02):
    t = (torch.randn(*shape, dtype=torch.float32, device=DEV) * scale).to(dtype)
    _keepalive.append(t)
    return t


def _lin(n, k, scale=0.02):
    """Weight in GEMM (K,N) convention -- see test_imagewam_prefill.py's
    own docstring for the dangling-pointer trap this helper avoids
    (`.contiguous()` on a non-contiguous `.t()` allocates a NEW tensor
    that must itself be kept alive, not just its pre-transpose input)."""
    t = (torch.randn(n, k, dtype=torch.float32, device=DEV) * scale).to(FP16).t().contiguous()
    _keepalive.append(t)
    return t


def _zeros(*shape, dtype=FP16):
    t = torch.zeros(*shape, dtype=dtype, device=DEV)
    _keepalive.append(t)
    return t


def _norm_scale(HD):
    t = (torch.randn(HD, dtype=torch.float32, device=DEV).abs() + 0.5).to(FP16)
    _keepalive.append(t)
    return t


def _time_ms(fn, warmup=WARMUP, iters=ITERS) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    return p50, p90, statistics.mean(times)


def _make_linear(gemm, n, k):
    """Build the weight-projection linear op selected by IMAGEWAM_PRECISION
    -- OPT-004 step 5, mirrors imagewam_thor.py's own `_rnd_linear`."""
    w = _lin(n, k)
    if PRECISION == "fp16":
        return Fp16Linear(gemm, w.data_ptr(), n, k)
    if PRECISION == "fp8":
        return Fp8Linear(w.data_ptr(), n, k)
    if PRECISION == "nvfp4":
        return Nvfp4Linear(w.data_ptr(), n, k)
    if PRECISION == "fp8_static":
        return StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False)
    if PRECISION == "fp8_static_cutlass":
        return StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=True)
    raise ValueError(f"unknown IMAGEWAM_PRECISION={PRECISION!r}")


def _calibrate_static_fp8(weights, m):
    """Freeze every `StaticFp8Linear` weight's activation scale ONCE,
    before `_time_ms`'s own warmup loop -- OPT-004 step 6 (plan.md),
    mirrors `imagewam_thor.py`'s own `_calibrate_fp8` (same
    representative-M simplification: one M per layer-type family, not
    per exact call site inside that layer -- these bench functions are
    already split by family, so `m` is just that function's own
    dominant sequence length). No-op for every other precision."""
    if PRECISION not in ("fp8_static", "fp8_static_cutlass"):
        return
    scratch_by_k = {}
    for lin in weights.values():
        if not isinstance(lin, StaticFp8Linear):
            continue
        x = scratch_by_k.get(lin.k)
        if x is None:
            x = torch.randn(m, lin.k, dtype=FP16, device=DEV) * 0.1
            scratch_by_k[lin.k] = x
        lin.calibrate(x.data_ptr(), m, 0)
    torch.cuda.synchronize()


def _autotune(gemm, shapes):
    """Autotune `gemm.fp16_nn` once per distinct (M,N,K) shape -- OPT-004
    step 4, now the frontend's own default (see imagewam_thor.py's
    `_autotune_gemm`). Uses disposable zero-filled scratch, not this
    benchmark's own real weight/activation buffers -- autotune only
    times candidate algorithms, it needs no meaningful values."""
    for m, n, k in shapes:
        x = torch.zeros(m, k, dtype=FP16, device=DEV)
        w = torch.zeros(k, n, dtype=FP16, device=DEV)
        out = torch.zeros(m, n, dtype=FP16, device=DEV)
        gemm.autotune_fp16_nn(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, 16)
    torch.cuda.synchronize()


def _make_1layer_backend(*, kind: str):
    """A 1-layer AttentionSpec/backend for isolated per-layer-type timing."""
    max_seq = A0 if kind == "backbone" else TOTAL
    spec = make_imagewam_attention_spec(max_prefix_seq=A0, max_total_seq=TOTAL,
                                         num_layers=1, num_heads=NH, head_dim=HD)
    ctx = fvk.FvkContext()
    # Real per-head K/V: (1, max_seq, HIDDEN), not the old broadcast
    # (1, max_seq, HD) shape (opportunities.md OPT-002).
    K_cache = _zeros(1, max_seq, HIDDEN)
    V_cache = _zeros(1, max_seq, HIDDEN)
    Q_O = _zeros(max_seq, HIDDEN)
    logits = _zeros(max_seq * NH, max_seq + (max_seq % 2))
    backend = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
        },
        mot_slots={
            "Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
            "layer_stride": K_cache[0].numel() * 2,
        },
        use_perhead_kv=True, use_real_mot_mask=True,
        # Only "backbone" has an FA4 dispatch branch at all; harmless
        # to pass for "mot" too (ImageWAMAttnBackend simply never
        # reads it there), but gated on `kind` for clarity.
        use_fa4=(USE_FA4 and kind == "backbone"),
    )
    return ctx, backend


def bench_backbone_double():
    ctx, attn = _make_1layer_backend(kind="backbone")
    gemm = fvk.GemmRunner()
    img_len = A0 - X0
    weights = {("backbone", "double", 0, "txt_in.weight"): _make_linear(gemm, HIDDEN, JOINT_ATTN_DIM)}
    for prefix in ("txt", "img"):
        weights[("backbone", "double", 0, f"{prefix}_qkv.weight")] = _make_linear(gemm, 3 * HIDDEN, HIDDEN)
        weights[("backbone", "double", 0, f"{prefix}_proj.weight")] = _make_linear(gemm, HIDDEN, HIDDEN)
        weights[("backbone", "double", 0, f"{prefix}_mlp0.weight")] = _make_linear(gemm, MLP_HIDDEN * 2, HIDDEN)
        weights[("backbone", "double", 0, f"{prefix}_mlp2.weight")] = _make_linear(gemm, HIDDEN, MLP_HIDDEN)
        weights[("backbone", "double", 0, f"{prefix}_query_norm")] = _norm_scale(HD).data_ptr()
        weights[("backbone", "double", 0, f"{prefix}_key_norm")] = _norm_scale(HD).data_ptr()
    dims = dict(hidden=HIDDEN, HD=HD, NH=NH, mlp_hidden=MLP_HIDDEN,
                joint_attention_dim=JOINT_ATTN_DIM, x0=X0, a0=A0)
    bufs = {
        "context": _rand(X0, JOINT_ATTN_DIM).data_ptr(),
        "backbone_hidden": _rand(A0, HIDDEN, scale=0.1).data_ptr(),
        "modded_scratch": _zeros(A0, HIDDEN).data_ptr(),
        "txt_qkv_merged": _zeros(X0, 3 * HIDDEN).data_ptr(),
        "img_qkv_merged": _zeros(img_len, 3 * HIDDEN).data_ptr(),
        "txt_mlp_merged": _zeros(X0, MLP_HIDDEN * 2).data_ptr(),
        "txt_mlp_gated": _zeros(X0, MLP_HIDDEN).data_ptr(),
        "img_mlp_merged": _zeros(img_len, MLP_HIDDEN * 2).data_ptr(),
        "img_mlp_gated": _zeros(img_len, MLP_HIDDEN).data_ptr(),
        "proj_scratch": _zeros(A0, HIDDEN).data_ptr(),
    }
    mod_w = {
        "time_in_w1": torch.randn(HIDDEN, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(HIDDEN, HIDDEN, dtype=F32, device=DEV) * 0.02,
        "mod_double_txt": torch.randn(6 * HIDDEN, HIDDEN, dtype=F32, device=DEV) * 0.02,
        "mod_double_img": torch.randn(6 * HIDDEN, HIDDEN, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * HIDDEN, HIDDEN, dtype=F32, device=DEV) * 0.02,
    }
    mod_txt, mod_img, _ = compute_shared_modulation(torch.zeros(1, device=DEV), mod_w, HIDDEN)
    table = build_backbone_rope_table(X0, img_len, 1, device=DEV)
    _calibrate_static_fp8(weights, A0)
    _autotune(gemm, {
        (X0, HIDDEN, JOINT_ATTN_DIM), (X0, 3 * HIDDEN, HIDDEN), (X0, HIDDEN, HIDDEN),
        (X0, MLP_HIDDEN * 2, HIDDEN), (X0, HIDDEN, MLP_HIDDEN),
        (img_len, 3 * HIDDEN, HIDDEN), (img_len, HIDDEN, HIDDEN), (img_len, MLP_HIDDEN * 2, HIDDEN),
        (img_len, HIDDEN, MLP_HIDDEN),
    })

    def run():
        _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, 0, 0, attn,
                              mod_txt, mod_img, table.data_ptr())

    return _time_ms(run)


def bench_backbone_single():
    ctx, attn = _make_1layer_backend(kind="backbone")
    gemm = fvk.GemmRunner()
    weights = {
        ("backbone", "single", 0, "qkv.weight"): _make_linear(gemm, 3 * HIDDEN, HIDDEN),
        ("backbone", "single", 0, "mlp_in.weight"): _make_linear(gemm, MLP_HIDDEN * 2, HIDDEN),
        ("backbone", "single", 0, "attn_out_proj.weight"): _make_linear(gemm, HIDDEN, HIDDEN),
        ("backbone", "single", 0, "mlp_down.weight"): _make_linear(gemm, HIDDEN, MLP_HIDDEN),
        ("backbone", "single", 0, "query_norm"): _norm_scale(HD).data_ptr(),
        ("backbone", "single", 0, "key_norm"): _norm_scale(HD).data_ptr(),
    }
    dims = dict(hidden=HIDDEN, HD=HD, NH=NH, mlp_hidden=MLP_HIDDEN, a0=A0)
    bufs = {
        "backbone_hidden": _rand(A0, HIDDEN, scale=0.1).data_ptr(),
        "modded_scratch": _zeros(A0, HIDDEN).data_ptr(),
        "single_qkv_merged": _zeros(A0, 3 * HIDDEN).data_ptr(),
        "single_mlp_merged": _zeros(A0, MLP_HIDDEN * 2).data_ptr(),
        "single_mlp_gated": _zeros(A0, MLP_HIDDEN).data_ptr(),
        "proj_scratch": _zeros(A0, HIDDEN).data_ptr(),
        "proj_scratch2": _zeros(A0, HIDDEN).data_ptr(),
    }
    mod_w = {
        "time_in_w1": torch.randn(HIDDEN, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(HIDDEN, HIDDEN, dtype=F32, device=DEV) * 0.02,
        "mod_double_txt": torch.randn(6 * HIDDEN, HIDDEN, dtype=F32, device=DEV) * 0.02,
        "mod_double_img": torch.randn(6 * HIDDEN, HIDDEN, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * HIDDEN, HIDDEN, dtype=F32, device=DEV) * 0.02,
    }
    _, _, mod_single = compute_shared_modulation(torch.zeros(1, device=DEV), mod_w, HIDDEN)
    table = build_backbone_rope_table(X0, A0 - X0, 1, device=DEV)
    _calibrate_static_fp8(weights, A0)
    _autotune(gemm, {
        (A0, 3 * HIDDEN, HIDDEN), (A0, HIDDEN, HIDDEN), (A0, MLP_HIDDEN * 2, HIDDEN), (A0, HIDDEN, MLP_HIDDEN),
    })

    def run():
        _single_stream_layer(ctx, fvk, gemm, bufs, weights, dims, 0, 0, 0, attn,
                              mod_single, table.data_ptr())

    return _time_ms(run)


def bench_action_double():
    ctx, attn = _make_1layer_backend(kind="mot")
    gemm = fvk.GemmRunner()
    weights = {
        ("action_dit", "double", 0, "qkv.weight"): _make_linear(gemm, 3 * ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
        ("action_dit", "double", 0, "proj.weight"): _make_linear(gemm, ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
        ("action_dit", "double", 0, "mlp0.weight"): _make_linear(gemm, ACTION_MLP_HIDDEN * 2, ACTION_HIDDEN_DIM),
        ("action_dit", "double", 0, "mlp2.weight"): _make_linear(gemm, ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
        ("action_dit", "double", 0, "query_norm"): _norm_scale(HD).data_ptr(),
        ("action_dit", "double", 0, "key_norm"): _norm_scale(HD).data_ptr(),
    }
    dims = dict(action_hidden_dim=ACTION_HIDDEN_DIM, action_attn_width=ACTION_ATTN_WIDTH,
                HD=HD, NH=NH, action_mlp_hidden=ACTION_MLP_HIDDEN, x0=X0, a0=A0,
                num_action=NUM_ACTION, total=TOTAL)
    bufs = {
        "action_hidden": _rand(NUM_ACTION, ACTION_HIDDEN_DIM, scale=0.1).data_ptr(),
        "action_modded": _zeros(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_qkv_merged": _zeros(NUM_ACTION, 3 * ACTION_ATTN_WIDTH).data_ptr(),
        "action_proj_scratch": _zeros(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_mlp_merged": _zeros(NUM_ACTION, ACTION_MLP_HIDDEN * 2).data_ptr(),
        "action_mlp_gated": _zeros(NUM_ACTION, ACTION_MLP_HIDDEN).data_ptr(),
    }
    mod_w = {
        "time_in_w1": torch.randn(ACTION_HIDDEN_DIM, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(ACTION_HIDDEN_DIM, ACTION_HIDDEN_DIM, dtype=F32, device=DEV) * 0.02,
        "mod_double": torch.randn(6 * ACTION_HIDDEN_DIM, ACTION_HIDDEN_DIM, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * ACTION_HIDDEN_DIM, ACTION_HIDDEN_DIM, dtype=F32, device=DEV) * 0.02,
    }
    mod_double, _ = compute_action_modulation(torch.ones(1, device=DEV), mod_w, ACTION_HIDDEN_DIM)
    action_table = build_action_rope_table(NUM_ACTION, device=DEV)
    _calibrate_static_fp8(weights, NUM_ACTION)
    _autotune(gemm, {
        (NUM_ACTION, 3 * ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
        (NUM_ACTION, ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
        (NUM_ACTION, ACTION_MLP_HIDDEN * 2, ACTION_HIDDEN_DIM),
        (NUM_ACTION, ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
    })

    def run():
        _action_double_layer(ctx, fvk, gemm, bufs, weights, dims, 0, 0, 0, attn,
                              mod_double, action_table.data_ptr())

    return _time_ms(run)


def bench_action_single():
    ctx, attn = _make_1layer_backend(kind="mot")
    gemm = fvk.GemmRunner()
    weights = {
        ("action_dit", "single", 0, "qkv.weight"): _make_linear(gemm, 3 * ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
        ("action_dit", "single", 0, "mlp_in.weight"): _make_linear(gemm, ACTION_MLP_HIDDEN * 2, ACTION_HIDDEN_DIM),
        ("action_dit", "single", 0, "attn_out_proj.weight"): _make_linear(gemm, ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
        ("action_dit", "single", 0, "mlp_down.weight"): _make_linear(gemm, ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
        ("action_dit", "single", 0, "query_norm"): _norm_scale(HD).data_ptr(),
        ("action_dit", "single", 0, "key_norm"): _norm_scale(HD).data_ptr(),
    }
    dims = dict(action_hidden_dim=ACTION_HIDDEN_DIM, action_attn_width=ACTION_ATTN_WIDTH,
                HD=HD, NH=NH, action_mlp_hidden=ACTION_MLP_HIDDEN, x0=X0, a0=A0,
                num_action=NUM_ACTION, total=TOTAL)
    bufs = {
        "action_hidden": _rand(NUM_ACTION, ACTION_HIDDEN_DIM, scale=0.1).data_ptr(),
        "action_modded": _zeros(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_qkv_merged": _zeros(NUM_ACTION, 3 * ACTION_ATTN_WIDTH).data_ptr(),
        "action_proj_scratch": _zeros(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_proj_scratch2": _zeros(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_mlp_merged": _zeros(NUM_ACTION, ACTION_MLP_HIDDEN * 2).data_ptr(),
        "action_mlp_gated": _zeros(NUM_ACTION, ACTION_MLP_HIDDEN).data_ptr(),
    }
    mod_w = {
        "time_in_w1": torch.randn(ACTION_HIDDEN_DIM, 256, dtype=F32, device=DEV) * 0.02,
        "time_in_w2": torch.randn(ACTION_HIDDEN_DIM, ACTION_HIDDEN_DIM, dtype=F32, device=DEV) * 0.02,
        "mod_double": torch.randn(6 * ACTION_HIDDEN_DIM, ACTION_HIDDEN_DIM, dtype=F32, device=DEV) * 0.02,
        "mod_single": torch.randn(3 * ACTION_HIDDEN_DIM, ACTION_HIDDEN_DIM, dtype=F32, device=DEV) * 0.02,
    }
    _, mod_single = compute_action_modulation(torch.ones(1, device=DEV), mod_w, ACTION_HIDDEN_DIM)
    action_table = build_action_rope_table(NUM_ACTION, device=DEV)
    _calibrate_static_fp8(weights, NUM_ACTION)
    _autotune(gemm, {
        (NUM_ACTION, 3 * ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
        (NUM_ACTION, ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
        (NUM_ACTION, ACTION_MLP_HIDDEN * 2, ACTION_HIDDEN_DIM),
        (NUM_ACTION, ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
    })

    def run():
        _action_single_layer(ctx, fvk, gemm, bufs, weights, dims, 0, 0, 0, attn,
                              mod_single, action_table.data_ptr())

    return _time_ms(run)


def bench_mot_joint_kernel_only():
    ctx = fvk.FvkContext()
    q = _rand(NUM_ACTION * NH, HD)
    k = _rand(TOTAL, HIDDEN)
    v = _rand(TOTAL, HIDDEN)
    total_pad = TOTAL + (TOTAL % 2)
    logits = _zeros(NUM_ACTION * NH, total_pad)
    out = _zeros(NUM_ACTION * NH, HD)
    scale = 1.0 / (HD ** 0.5)

    def run():
        fvk.attention_qkv_fp16_perhead(ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                        logits.data_ptr(), out.data_ptr(),
                                        NUM_ACTION, TOTAL, NH, HD, scale, 0)

    return _time_ms(run)


def bench_standard_attn_kernel_only():
    ctx = fvk.FvkContext()
    q = _rand(A0 * NH, HD)
    k = _rand(A0, HIDDEN)
    v = _rand(A0, HIDDEN)
    logits = _zeros(A0 * NH, A0)
    out = _zeros(A0 * NH, HD)
    scale = 1.0 / (HD ** 0.5)

    def run():
        fvk.attention_qkv_fp16_perhead(ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                        logits.data_ptr(), out.data_ptr(),
                                        A0, A0, NH, HD, scale, 0)

    return _time_ms(run)


def main():
    print(f"Dims: hidden={HIDDEN} HD={HD} NH={NH} mlp_hidden={MLP_HIDDEN} "
          f"| action_hidden_dim={ACTION_HIDDEN_DIM} action_attn_width={ACTION_ATTN_WIDTH} "
          f"action_mlp_hidden={ACTION_MLP_HIDDEN}")
    print(f"Seq: x0={X0} a0={A0} num_action={NUM_ACTION} total={TOTAL} "
          f"(representative, not a confirmed real ImageWAM deployment length)")
    print(f"Real math (opportunities.md OPT-002): per-head K/V, QK-Norm, RoPE, "
          f"AdaLN, SiLU-GLU MLP, no-mask attention -- not the old approximation.")
    print(f"Warmup={WARMUP} iters={ITERS}, CUDA-event timing, P50/P90/mean in ms\n")

    results = {}
    for name, fn in [
        ("backbone_double_layer", bench_backbone_double),
        ("backbone_single_layer", bench_backbone_single),
        ("action_double_layer", bench_action_double),
        ("action_single_layer", bench_action_single),
        ("mot_joint_kernel_only", bench_mot_joint_kernel_only),
        ("standard_attn_kernel_only", bench_standard_attn_kernel_only),
    ]:
        p50, p90, mean = fn()
        results[name] = p50
        print(f"{name:28s} P50={p50:8.3f}  P90={p90:8.3f}  mean={mean:8.3f}")
        _keepalive.clear()
        torch.cuda.empty_cache()

    backbone_total = NUM_DOUBLE * results["backbone_double_layer"] + \
        NUM_SINGLE * results["backbone_single_layer"]
    action_step_total = NUM_DOUBLE * results["action_double_layer"] + \
        NUM_SINGLE * results["action_single_layer"]

    print(f"\nDerived (layer_count x per-layer P50, NOT a single measured run):")
    print(f"  backbone prefill total ({NUM_DOUBLE} double + {NUM_SINGLE} single): "
          f"{backbone_total:.2f} ms")
    print(f"  one ActionDiT denoise step ({NUM_DOUBLE} double + {NUM_SINGLE} single): "
          f"{action_step_total:.2f} ms")
    for n_steps in (1, 4, 10):
        print(f"  prefill + {n_steps}-step denoise loop: "
              f"{backbone_total + n_steps * action_step_total:.2f} ms")


if __name__ == "__main__":
    main()
