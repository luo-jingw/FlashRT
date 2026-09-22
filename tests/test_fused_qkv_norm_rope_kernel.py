"""Bit-exact + informal-timing validation for the standalone fused
QKV-split + RMSNorm(Q,K) + RoPE(Q,K) + V-copy kernel
(csrc/kernels/fused_qkv_norm_rope/qkv_split_norm_rope_fp16.cu).

This kernel is NOT wired into the production pipeline or the main
CMake build (see that directory's own header comment) -- it is
JIT-compiled here via `torch.utils.cpp_extension.load` and checked
`torch.equal` (bit-exact, fp16) against the exact 7-call production
sequence it fuses, reproduced verbatim from
`flash_rt/models/imagewam/pipeline_thor.py`'s `_single_stream_layer`
(same sequence, per source-range, in `_double_stream_layer`'s
txt/img halves and `_action_double_layer`/`_action_single_layer`):

    _copy_slice(Q_O, qkv, rows, hidden, src_row_stride=...)      # col q_col_offset
    _copy_slice(K_cache, qkv, rows, hidden, src_row_stride=...)  # col k_col_offset
    _copy_slice(V_cache, qkv, rows, hidden, src_row_stride=...)  # col v_col_offset
    rms_norm_fp16(Q_O, q_norm, Q_O, rows*NH, HD, eps, stream)
    rms_norm_fp16(K_cache, k_norm, K_cache, rows*NH, HD, eps, stream)
    rope_apply_fp16_perhead(Q_O, rope_table, rows, NH, HD, stream)
    rope_apply_fp16_perhead(K_cache, rope_table, rows, NH, HD, stream)

using the project's own production `rms_norm_fp16`/`rope_apply_fp16_perhead`
(via `flash_rt.flash_rt_kernels`) as the reference implementation --
these are the numerical ground truth, not reimplemented here.

Real ImageWAM shapes (flash_rt/frontends/torch/_imagewam_thor_spec.py):
  backbone_hidden = 3072 (NH=24, HD=128), rows in {25, 417, 905}
  action_attn_width = 3072 (NH=24, HD=128, == backbone, required for
    mot_joint), num_action = max_action_horizon = 64
  backbone_mlp_hidden = int(3072*3.0) = 9216 (used for the merged
  `linear1` src_row_stride variant, `merge_qkv_mlp=True` call sites)
"""
import os
import time

import pytest
import torch
from torch.utils.cpp_extension import load

import flash_rt.flash_rt_kernels as fvk

DEV = "cuda"
EPS = 1e-6

_KERNEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "csrc", "kernels", "fused_qkv_norm_rope")

# Same precision-affecting flags CMakeLists.txt uses for the
# flash_rt_kernels target that builds csrc/kernels/norm.cu and
# csrc/kernels/rope.cu (the reference kernels this must match
# bit-exact) -- see CMakeLists.txt's `target_compile_options(flash_rt_kernels ...)`.
# NOT the --use_fast_math flags some OTHER targets in that file use.
_CUDA_FLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    "--ftz=true",
    "--prec-div=false",
    "--prec-sqrt=false",
]

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="fused_qkv_norm_rope_test_ext",
            sources=[
                os.path.join(_KERNEL_DIR, "binding.cpp"),
                os.path.join(_KERNEL_DIR, "qkv_split_norm_rope_fp16.cu"),
            ],
            extra_cuda_cflags=_CUDA_FLAGS,
            verbose=False,
        )
    return _ext


def _make_rope_table(rows: int, HD: int, device: str) -> torch.Tensor:
    """Interleaved cos/sin-per-pair table, same layout
    `rope_apply_fp16_perhead_kernel` reads (rope_table[seq_pos*HD+2d]=cos,
    [...+2d+1]=sin) -- see csrc/kernels/rope.cu."""
    half = HD // 2
    pos = torch.arange(rows, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1)))
    ang = torch.einsum("s,d->sd", pos, freqs)
    table = torch.zeros(rows, HD, dtype=torch.float32, device=device)
    table[:, 0:2 * half:2] = torch.cos(ang)
    table[:, 1:2 * half:2] = torch.sin(ang)
    return table.half().contiguous()


def _make_inputs(rows: int, NH: int, HD: int, hidden: int, src_row_stride: int, seed: int = 0):
    torch.manual_seed(seed)
    qkv = (torch.randn(rows, src_row_stride, dtype=torch.float16, device=DEV) * 0.5)
    q_norm = (1.0 + 0.1 * torch.randn(HD, dtype=torch.float16, device=DEV))
    k_norm = (1.0 + 0.1 * torch.randn(HD, dtype=torch.float16, device=DEV))
    rope_table = _make_rope_table(rows, HD, DEV)
    return qkv, q_norm, k_norm, rope_table


def _reference(qkv, q_norm, k_norm, rope_table, rows, NH, HD, hidden,
                q_off, k_off, v_off, stream=0):
    """Reproduces the exact production 7-call sequence, using the
    project's own rms_norm_fp16/rope_apply_fp16_perhead as ground truth."""
    Q = qkv[:, q_off:q_off + hidden].clone().contiguous()
    K = qkv[:, k_off:k_off + hidden].clone().contiguous()
    V = qkv[:, v_off:v_off + hidden].clone().contiguous()
    fvk.rms_norm_fp16(Q.data_ptr(), q_norm.data_ptr(), Q.data_ptr(), rows * NH, HD, EPS, stream)
    fvk.rms_norm_fp16(K.data_ptr(), k_norm.data_ptr(), K.data_ptr(), rows * NH, HD, EPS, stream)
    fvk.rope_apply_fp16_perhead(Q.data_ptr(), rope_table.data_ptr(), rows, NH, HD, stream)
    fvk.rope_apply_fp16_perhead(K.data_ptr(), rope_table.data_ptr(), rows, NH, HD, stream)
    torch.cuda.synchronize()
    return Q, K, V


def _fused(qkv, q_norm, k_norm, rope_table, rows, NH, HD, hidden,
           src_row_stride, q_off, k_off, v_off, stream=0):
    ext = _get_ext()
    Q = torch.empty(rows, hidden, dtype=torch.float16, device=DEV)
    K = torch.empty(rows, hidden, dtype=torch.float16, device=DEV)
    V = torch.empty(rows, hidden, dtype=torch.float16, device=DEV)
    ext.qkv_split_norm_rope_fp16(
        qkv.data_ptr(), q_norm.data_ptr(), k_norm.data_ptr(), rope_table.data_ptr(),
        Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        rows, NH, HD, hidden, src_row_stride,
        q_off, k_off, v_off, hidden, EPS, stream)
    torch.cuda.synchronize()
    return Q, K, V


# ──────────────────────────────────────────────────────────────────
# Bit-exact shape sweep: real ImageWAM backbone + ActionDiT shapes,
# plain qkv GEMM output (src_row_stride = 3*hidden, col offsets
# 0/hidden/2*hidden) -- the `_double_stream_layer` / non-merged
# `_single_stream_layer` / `_action_*_layer` call shape.
# ──────────────────────────────────────────────────────────────────
BACKBONE_HIDDEN, BACKBONE_NH, BACKBONE_HD = 3072, 24, 128
ACTION_ATTN_WIDTH, ACTION_NH, ACTION_HD = 3072, 24, 128
NUM_ACTION = 64
BACKBONE_MLP_HIDDEN = int(BACKBONE_HIDDEN * 3.0)  # 9216, backbone_mlp_ratio

PLAIN_SHAPES = [
    ("backbone_rows25", BACKBONE_HIDDEN, BACKBONE_NH, BACKBONE_HD, 25),
    ("backbone_rows417", BACKBONE_HIDDEN, BACKBONE_NH, BACKBONE_HD, 417),
    ("backbone_rows905", BACKBONE_HIDDEN, BACKBONE_NH, BACKBONE_HD, 905),
    ("action_rows64", ACTION_ATTN_WIDTH, ACTION_NH, ACTION_HD, NUM_ACTION),
]


@pytest.mark.parametrize("name,hidden,NH,HD,rows", PLAIN_SHAPES, ids=[s[0] for s in PLAIN_SHAPES])
def test_bit_exact_plain_qkv(name, hidden, NH, HD, rows):
    src_row_stride = 3 * hidden
    q_off, k_off, v_off = 0, hidden, 2 * hidden
    qkv, q_norm, k_norm, rope_table = _make_inputs(rows, NH, HD, hidden, src_row_stride, seed=1)

    Q_ref, K_ref, V_ref = _reference(qkv, q_norm, k_norm, rope_table, rows, NH, HD, hidden,
                                      q_off, k_off, v_off)
    Q_f, K_f, V_f = _fused(qkv, q_norm, k_norm, rope_table, rows, NH, HD, hidden,
                            src_row_stride, q_off, k_off, v_off)

    assert torch.equal(Q_ref, Q_f), f"{name}: Q mismatch"
    assert torch.equal(K_ref, K_f), f"{name}: K mismatch"
    assert torch.equal(V_ref, V_f), f"{name}: V mismatch"


def test_bit_exact_merged_linear1_qkv():
    """The `merge_qkv_mlp=True` call shape (`_single_stream_layer` /
    `_action_single_layer`'s `linear1` branch): src_row_stride =
    3*hidden + 2*mlp_hidden (Q/K/V columns followed by the MLP
    gate/up columns), Q/K/V still at column offsets 0/hidden/2*hidden."""
    hidden, NH, HD, rows = BACKBONE_HIDDEN, BACKBONE_NH, BACKBONE_HD, 25
    mlp_hidden = BACKBONE_MLP_HIDDEN
    src_row_stride = 3 * hidden + 2 * mlp_hidden
    q_off, k_off, v_off = 0, hidden, 2 * hidden
    qkv, q_norm, k_norm, rope_table = _make_inputs(rows, NH, HD, hidden, src_row_stride, seed=2)

    Q_ref, K_ref, V_ref = _reference(qkv, q_norm, k_norm, rope_table, rows, NH, HD, hidden,
                                      q_off, k_off, v_off)
    Q_f, K_f, V_f = _fused(qkv, q_norm, k_norm, rope_table, rows, NH, HD, hidden,
                            src_row_stride, q_off, k_off, v_off)

    assert torch.equal(Q_ref, Q_f), "merged-linear1: Q mismatch"
    assert torch.equal(K_ref, K_f), "merged-linear1: K mismatch"
    assert torch.equal(V_ref, V_f), "merged-linear1: V mismatch"


def test_even_head_dim_is_a_documented_precondition():
    """The kernel reads/writes Q/K/V via __half2 (4-byte) vectorized
    loads at per-head byte offset head*HD*sizeof(half); that offset is
    only guaranteed 4-byte aligned when HD is even. Real ImageWAM
    shapes (backbone and ActionDiT) always use HD=128 (divisible by
    8), so this is a documented precondition, not a limitation this
    project needs to design around (see this project's convention of
    validating kernels at real, fixed production shapes rather than
    arbitrary N -- confirmed by construction here, not by trying an
    odd HD, which segfaults/misaligns rather than merely producing a
    numerically different result)."""
    assert BACKBONE_HD % 2 == 0
    assert ACTION_HD % 2 == 0


def test_double_stream_split_call_matches_joint_rope():
    """`_double_stream_layer` runs RMSNorm separately per stream (txt
    then img, two different source qkv buffers) but RoPE ONCE jointly
    over the combined [txt|img] row range of Q_O/K_cache. RoPE is
    purely row-independent/elementwise, so calling the fused kernel
    TWICE (once per stream, each with its own correctly row-offset
    rope_table pointer) must reproduce the SAME per-row result as the
    reference's separate-norm-then-joint-rope sequence -- this test
    checks that equivalence directly, combined-buffer bit-exact.
    """
    hidden, NH, HD = BACKBONE_HIDDEN, BACKBONE_NH, BACKBONE_HD
    x0, img_len = 17, 40
    a0 = x0 + img_len
    src_row_stride = 3 * hidden
    q_off, k_off, v_off = 0, hidden, 2 * hidden

    torch.manual_seed(4)
    txt_qkv = torch.randn(x0, src_row_stride, dtype=torch.float16, device=DEV) * 0.5
    img_qkv = torch.randn(img_len, src_row_stride, dtype=torch.float16, device=DEV) * 0.5
    q_norm = 1.0 + 0.1 * torch.randn(HD, dtype=torch.float16, device=DEV)
    k_norm = 1.0 + 0.1 * torch.randn(HD, dtype=torch.float16, device=DEV)
    rope_table = _make_rope_table(a0, HD, DEV)  # ONE table over the combined range

    # ---- reference: separate-source RMSNorm per stream into a
    # COMBINED (a0, hidden) buffer, THEN one joint RoPE call over the
    # whole a0 range (exactly mirrors _double_stream_layer's own
    # sequence). ----
    Q_ref = torch.empty(a0, hidden, dtype=torch.float16, device=DEV)
    K_ref = torch.empty(a0, hidden, dtype=torch.float16, device=DEV)
    Q_ref[:x0] = txt_qkv[:, q_off:q_off + hidden]
    K_ref[:x0] = txt_qkv[:, k_off:k_off + hidden]
    Q_ref[x0:] = img_qkv[:, q_off:q_off + hidden]
    K_ref[x0:] = img_qkv[:, k_off:k_off + hidden]
    fvk.rms_norm_fp16(Q_ref[:x0].data_ptr(), q_norm.data_ptr(), Q_ref[:x0].data_ptr(), x0 * NH, HD, EPS, 0)
    fvk.rms_norm_fp16(K_ref[:x0].data_ptr(), k_norm.data_ptr(), K_ref[:x0].data_ptr(), x0 * NH, HD, EPS, 0)
    img_Q_ptr = Q_ref.data_ptr() + x0 * hidden * 2
    img_K_ptr = K_ref.data_ptr() + x0 * hidden * 2
    fvk.rms_norm_fp16(img_Q_ptr, q_norm.data_ptr(), img_Q_ptr, img_len * NH, HD, EPS, 0)
    fvk.rms_norm_fp16(img_K_ptr, k_norm.data_ptr(), img_K_ptr, img_len * NH, HD, EPS, 0)
    fvk.rope_apply_fp16_perhead(Q_ref.data_ptr(), rope_table.data_ptr(), a0, NH, HD, 0)
    fvk.rope_apply_fp16_perhead(K_ref.data_ptr(), rope_table.data_ptr(), a0, NH, HD, 0)
    torch.cuda.synchronize()

    # ---- fused: TWO calls, one per stream, each with its own
    # row-offset rope_table pointer -- writing into the SAME combined
    # destination buffer at the matching row offset. ----
    ext = _get_ext()
    Q_f = torch.empty(a0, hidden, dtype=torch.float16, device=DEV)
    K_f = torch.empty(a0, hidden, dtype=torch.float16, device=DEV)
    V_f = torch.empty(a0, hidden, dtype=torch.float16, device=DEV)
    rope_bytes_per_row = HD * 2  # fp16
    ext.qkv_split_norm_rope_fp16(
        txt_qkv.data_ptr(), q_norm.data_ptr(), k_norm.data_ptr(), rope_table.data_ptr(),
        Q_f.data_ptr(), K_f.data_ptr(), V_f.data_ptr(),
        x0, NH, HD, hidden, src_row_stride, q_off, k_off, v_off, hidden, EPS, 0)
    ext.qkv_split_norm_rope_fp16(
        img_qkv.data_ptr(), q_norm.data_ptr(), k_norm.data_ptr(),
        rope_table.data_ptr() + x0 * rope_bytes_per_row,
        Q_f.data_ptr() + x0 * hidden * 2, K_f.data_ptr() + x0 * hidden * 2, V_f.data_ptr() + x0 * hidden * 2,
        img_len, NH, HD, hidden, src_row_stride, q_off, k_off, v_off, hidden, EPS, 0)
    torch.cuda.synchronize()

    assert torch.equal(Q_ref, Q_f), "double-stream split-call: Q mismatch"
    assert torch.equal(K_ref, K_f), "double-stream split-call: K mismatch"


# ──────────────────────────────────────────────────────────────────
# Informal timing (Ada, informative only, does not predict Thor).
# ──────────────────────────────────────────────────────────────────
def test_informal_timing_ada_only():
    hidden, NH, HD, rows = BACKBONE_HIDDEN, BACKBONE_NH, BACKBONE_HD, 905
    src_row_stride = 3 * hidden
    q_off, k_off, v_off = 0, hidden, 2 * hidden
    qkv, q_norm, k_norm, rope_table = _make_inputs(rows, NH, HD, hidden, src_row_stride, seed=5)
    ext = _get_ext()

    Q_r = torch.empty(rows, hidden, dtype=torch.float16, device=DEV)
    K_r = torch.empty(rows, hidden, dtype=torch.float16, device=DEV)
    Q_f = torch.empty(rows, hidden, dtype=torch.float16, device=DEV)
    K_f = torch.empty(rows, hidden, dtype=torch.float16, device=DEV)
    V_f = torch.empty(rows, hidden, dtype=torch.float16, device=DEV)

    def run_reference():
        Q_r.copy_(qkv[:, q_off:q_off + hidden])
        K_r.copy_(qkv[:, k_off:k_off + hidden])
        fvk.rms_norm_fp16(Q_r.data_ptr(), q_norm.data_ptr(), Q_r.data_ptr(), rows * NH, HD, EPS, 0)
        fvk.rms_norm_fp16(K_r.data_ptr(), k_norm.data_ptr(), K_r.data_ptr(), rows * NH, HD, EPS, 0)
        fvk.rope_apply_fp16_perhead(Q_r.data_ptr(), rope_table.data_ptr(), rows, NH, HD, 0)
        fvk.rope_apply_fp16_perhead(K_r.data_ptr(), rope_table.data_ptr(), rows, NH, HD, 0)

    def run_fused():
        ext.qkv_split_norm_rope_fp16(
            qkv.data_ptr(), q_norm.data_ptr(), k_norm.data_ptr(), rope_table.data_ptr(),
            Q_f.data_ptr(), K_f.data_ptr(), V_f.data_ptr(),
            rows, NH, HD, hidden, src_row_stride, q_off, k_off, v_off, hidden, EPS, 0)

    n_warmup, n_iter = 10, 100
    for _ in range(n_warmup):
        run_reference()
        run_fused()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        run_reference()
    end.record()
    torch.cuda.synchronize()
    ref_ms = start.elapsed_time(end) / n_iter

    start.record()
    for _ in range(n_iter):
        run_fused()
    end.record()
    torch.cuda.synchronize()
    fused_ms = start.elapsed_time(end) / n_iter

    ratio = ref_ms / fused_ms if fused_ms > 0 else float("inf")
    print(f"\n[Ada RTX 4060 Laptop, informative only, does NOT predict Thor]")
    print(f"  rows={rows} hidden={hidden} NH={NH} HD={HD}")
    print(f"  reference (7 launches): {ref_ms:.4f} ms/iter")
    print(f"  fused (1 launch):       {fused_ms:.4f} ms/iter")
    print(f"  ratio (reference/fused): {ratio:.2f}x")
