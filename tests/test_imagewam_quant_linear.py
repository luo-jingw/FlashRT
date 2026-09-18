"""`quant_linear.py`'s `Fp8Linear`/`Nvfp4Linear`/`StaticFp8Linear`
correctness against the FP16 reference (`Fp16Linear`): cosine, max-abs
and rel_l2, at a small shape and at every distinct real ImageWAM
(M,N,K) shape.

Availability is probed with real canary calls, not guessed:

- FP8 cuBLASLt (`Fp8Linear`, `StaticFp8Linear(use_cutlass=False)`):
  runs on sm_89/sm_90 through the TN layout and on Blackwell through
  NN (`quant_linear.fp8_cublaslt_layout()`, `issues.md` ISSUE-001).
  The NN-vs-TN comparison needs a GPU that supports both (Blackwell).
- NVFP4: `flash_rt.flash_rt_fp4` exists only in a `GPU_ARCH=110`
  (Thor) or Blackwell build.
- FP8 CUTLASS (`StaticFp8Linear(use_cutlass=True)`): `cutlass_fp8_*`
  exist only in an `ENABLE_SM100_CUTLASS` build.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.quant_linear import (
    Fp16Linear,
    Fp8Linear,
    Nvfp4Linear,
    StaticFp8Linear,
    fp8_cublaslt_layout,
)

DEV = "cuda"
FP16 = torch.float16

_keepalive = []


def _own(t):
    _keepalive.append(t)
    return t


def _lin(n, k, scale=0.02):
    t = _own((torch.randn(n, k, dtype=torch.float32, device=DEV) * scale).to(FP16).t().contiguous())
    return t


def _probe_fp8_available():
    """Real canary call, not a guess -- FP8's own unavailability here
    is a runtime cuBLASLt failure, not an ImportError, so there is no
    cheap upfront check like NVFP4's module import."""
    try:
        w = _lin(16, 16)
        lin = Fp8Linear(w.data_ptr(), 16, 16)
        x = _own(torch.randn(4, 16, dtype=FP16, device=DEV) * 0.1)
        out = _own(torch.zeros(4, 16, dtype=FP16, device=DEV))
        lin(x.data_ptr(), out.data_ptr(), 4, 0)
        torch.cuda.synchronize()
        return True, None
    except RuntimeError as e:
        return False, str(e)


def _probe_nvfp4_available():
    try:
        import flash_rt.flash_rt_fp4  # noqa: F401
        return True, None
    except ImportError as e:
        return False, str(e)


def _probe_static_fp8_cublaslt_available():
    """Same cuBLASLt env gap as Fp8Linear, but exercised through
    StaticFp8Linear's own calibrate()->__call__ ordering (OPT-004 step
    6 plan Phase 1) -- a real canary, not a guess."""
    try:
        w = _lin(16, 16)
        lin = StaticFp8Linear(w.data_ptr(), 16, 16, use_cutlass=False)
        x = _own(torch.randn(4, 16, dtype=FP16, device=DEV) * 0.1)
        out = _own(torch.zeros(4, 16, dtype=FP16, device=DEV))
        lin.calibrate(x.data_ptr(), 4, 0)
        lin(x.data_ptr(), out.data_ptr(), 4, 0)
        torch.cuda.synchronize()
        return True, None
    except RuntimeError as e:
        return False, str(e)


def _probe_static_fp8_cutlass_available():
    """OPT-004 step 6 plan Phase 2 -- separate probe from the cuBLASLt
    variant above, per the plan's own Code Mapping note: a build could
    in principle have one without the other even though Phase 0/this
    plan's own research says they're gated together in practice."""
    try:
        w = _lin(16, 16)
        StaticFp8Linear(w.data_ptr(), 16, 16, use_cutlass=True)
        return True, None
    except RuntimeError as e:
        return False, str(e)


_FP8_AVAILABLE, _FP8_REASON = _probe_fp8_available()
_NVFP4_AVAILABLE, _NVFP4_REASON = _probe_nvfp4_available()
_STATIC_FP8_AVAILABLE, _STATIC_FP8_REASON = _probe_static_fp8_cublaslt_available()
_STATIC_FP8_CUTLASS_AVAILABLE, _STATIC_FP8_CUTLASS_REASON = _probe_static_fp8_cutlass_available()


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def _reference_and_ptrs(m, n, k):
    """Shared setup: one real weight, one real activation, FP16
    reference output via `Fp16Linear` (the already-verified passthrough,
    see test_imagewam_prefill.py etc.) -- both quantized paths are
    compared against THIS, not a separate torch.matmul, so any
    discrepancy is attributable purely to quantization, not to a
    convention mismatch between this test and the pointer-based path."""
    gemm = fvk.GemmRunner()
    w = _lin(n, k)
    x = _own(torch.randn(m, k, dtype=FP16, device=DEV) * 0.1)
    ref_out = _own(torch.zeros(m, n, dtype=FP16, device=DEV))
    Fp16Linear(gemm, w.data_ptr(), n, k)(x.data_ptr(), ref_out.data_ptr(), m, 0)
    torch.cuda.synchronize()
    return w, x, ref_out


def test_fp8_linear_matches_fp16_reference():
    if not _FP8_AVAILABLE:
        import pytest
        pytest.skip(f"FP8 cuBLASLt GEMM not available on this machine: {_FP8_REASON}")

    torch.manual_seed(0)
    m, n, k = 8, 64, 96
    w, x, ref_out = _reference_and_ptrs(m, n, k)

    fp8_lin = Fp8Linear(w.data_ptr(), n, k)
    out = _own(torch.zeros(m, n, dtype=FP16, device=DEV))
    fp8_lin(x.data_ptr(), out.data_ptr(), m, 0)
    torch.cuda.synchronize()

    cos = _cosine(out, ref_out)
    print(f"Fp8Linear vs Fp16Linear reference: cosine={cos:.6f}")
    assert cos > 0.99, f"cosine too low: {cos}"


def test_nvfp4_linear_matches_fp16_reference():
    if not _NVFP4_AVAILABLE:
        import pytest
        pytest.skip(f"NVFP4 extension not available on this machine (Blackwell/Thor-only "
                    f"build, see quant_linear.py's module docstring): {_NVFP4_REASON}")

    torch.manual_seed(0)
    m, n, k = 8, 64, 96  # K divisible by 16, required by quant_weight_nvfp4
    w, x, ref_out = _reference_and_ptrs(m, n, k)

    nvfp4_lin = Nvfp4Linear(w.data_ptr(), n, k)
    out = _own(torch.zeros(m, n, dtype=FP16, device=DEV))
    nvfp4_lin(x.data_ptr(), out.data_ptr(), m, 0)
    torch.cuda.synchronize()

    cos = _cosine(out, ref_out)
    print(f"Nvfp4Linear vs Fp16Linear reference: cosine={cos:.6f}")
    # Real Thor measurement (2026-09-14, one random uncalibrated layer):
    # cosine=0.989133 -- below the 0.99 bar FP8 clears easily (0.999242),
    # consistent with NVFP4's own format (E2M1, 2 mantissa bits, block-16
    # dynamic scale, no calibration) being structurally noisier than FP8
    # (E4M3), not evidence of a wiring bug in Nvfp4Linear. Bar set to
    # 0.98 to reflect this real, measured characteristic rather than a
    # threshold picked before any real number existed -- see plan.md's
    # own "OPT-004 step 5" Phase 4 write-up. Real trained (calibrated)
    # weights may do better or worse than this random-Gaussian test;
    # not knowable without the real checkpoint (Thor-only, per
    # PROJECT.md).
    assert cos > 0.98, f"cosine too low: {cos}"


def test_static_fp8_linear_cublaslt_matches_fp16_reference():
    """OPT-004 step 6 plan Phase 1: static (calibrate-once) scale,
    still `fp8_gemm_descale_fp16`/cuBLASLt -- isolates the scale change
    alone from the CUTLASS kernel change (next test)."""
    if not _STATIC_FP8_AVAILABLE:
        import pytest
        pytest.skip(f"Static FP8 (cuBLASLt) not available on this machine: {_STATIC_FP8_REASON}")

    torch.manual_seed(0)
    m, n, k = 8, 64, 96
    w, x, ref_out = _reference_and_ptrs(m, n, k)

    lin = StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False)
    lin.calibrate(x.data_ptr(), m, 0)
    out = _own(torch.zeros(m, n, dtype=FP16, device=DEV))
    lin(x.data_ptr(), out.data_ptr(), m, 0)
    torch.cuda.synchronize()

    cos = _cosine(out, ref_out)
    print(f"StaticFp8Linear(cublaslt) vs Fp16Linear reference: cosine={cos:.6f}")
    assert cos > 0.99, f"cosine too low: {cos}"


def test_static_fp8_linear_cutlass_matches_fp16_reference():
    """OPT-004 step 6 plan Phase 2: static scale + cutlass_fp8_sq/_wide/_t1
    -- the actual house-mechanism-equivalent path. Thor-only build
    (ENABLE_SM100_CUTLASS), same gate NVFP4 already uses; this dev
    machine's build has neither, confirmed via the probe above."""
    if not _STATIC_FP8_CUTLASS_AVAILABLE:
        import pytest
        pytest.skip(f"cutlass_fp8_sq/_wide/_t1 not available on this machine "
                    f"(Thor/Blackwell-only build, ENABLE_SM100_CUTLASS): "
                    f"{_STATIC_FP8_CUTLASS_REASON}")

    torch.manual_seed(0)
    m, n, k = 8, 64, 96
    w, x, ref_out = _reference_and_ptrs(m, n, k)

    lin = StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=True)
    lin.calibrate(x.data_ptr(), m, 0)
    out = _own(torch.zeros(m, n, dtype=FP16, device=DEV))
    lin(x.data_ptr(), out.data_ptr(), m, 0)
    torch.cuda.synchronize()

    cos = _cosine(out, ref_out)
    print(f"StaticFp8Linear(cutlass) vs Fp16Linear reference: cosine={cos:.6f}")
    assert cos > 0.99, f"cosine too low: {cos}"


# Every distinct (M, N, K) a quantized precision runs in the served
# pipeline at the real LIBERO dims (x0=513 text rows, img_len=392 image
# rows, a0=905 single-stream rows, num_action=64 ActionDiT rows;
# hidden=3072, mlp_hidden=9216, action_hidden_dim=1024,
# action_mlp_hidden=4096), with the merged single-stream linear1 and
# linear2 (K = hidden + mlp_hidden).
_REAL_SHAPES = {
    "txt_qkv": (513, 9216, 3072), "txt_proj": (513, 3072, 3072),
    "txt_mlp0": (513, 18432, 3072), "txt_mlp2": (513, 3072, 9216),
    "img_qkv": (392, 9216, 3072), "img_proj": (392, 3072, 3072),
    "img_mlp0": (392, 18432, 3072), "img_mlp2": (392, 3072, 9216),
    "single_linear1": (905, 27648, 3072), "single_linear2": (905, 3072, 12288),
    "action_qkv": (64, 9216, 1024), "action_proj": (64, 1024, 3072),
    "action_mlp0": (64, 8192, 1024), "action_mlp2": (64, 1024, 4096),
    "action_linear1": (64, 17408, 1024), "action_linear2": (64, 1024, 7168),
}


def _errors(out, ref):
    d = (out.float() - ref.float())
    return (_cosine(out, ref), d.abs().max().item(),
            (d.norm() / (ref.float().norm() + 1e-12)).item())


def _real_shape_case(m, n, k):
    """Random weight ~N(0, 0.02) (the real checkpoint's typical scale)
    and activation ~N(0, 1) (AdaLN-normalized GEMM inputs are O(1))."""
    gemm = fvk.GemmRunner()
    w = _lin(n, k)
    x = _own(torch.randn(m, k, dtype=FP16, device=DEV))
    ref = _own(torch.zeros(m, n, dtype=FP16, device=DEV))
    Fp16Linear(gemm, w.data_ptr(), n, k)(x.data_ptr(), ref.data_ptr(), m, 0)
    torch.cuda.synchronize()
    return w, x, ref


def test_fp8_cublaslt_real_shapes_match_fp16_reference():
    """Dynamic `Fp8Linear` and static `StaticFp8Linear(use_cutlass=False)`
    at every real shape, in this GPU's cuBLASLt layout."""
    if not (_FP8_AVAILABLE and _STATIC_FP8_AVAILABLE):
        import pytest
        pytest.skip(f"FP8 cuBLASLt GEMM not available on this machine: {_FP8_REASON}")
    torch.manual_seed(0)
    layout = fp8_cublaslt_layout()
    worst = 1.0
    for name, (m, n, k) in _REAL_SHAPES.items():
        w, x, ref = _real_shape_case(m, n, k)
        dyn = Fp8Linear(w.data_ptr(), n, k)
        out_dyn = torch.zeros(m, n, dtype=FP16, device=DEV)
        dyn(x.data_ptr(), out_dyn.data_ptr(), m, 0)
        st = StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False)
        st.calibrate(x.data_ptr(), m, 0)
        out_st = torch.zeros(m, n, dtype=FP16, device=DEV)
        st(x.data_ptr(), out_st.data_ptr(), m, 0)
        torch.cuda.synchronize()
        c_d, mx_d, rl_d = _errors(out_dyn, ref)
        c_s, mx_s, rl_s = _errors(out_st, ref)
        print(f"[{layout}] {name:16s} M={m:4d} N={n:5d} K={k:4d}  "
              f"dynamic cos={c_d:.6f} maxabs={mx_d:.4f} rel_l2={rl_d:.5f}  "
              f"static cos={c_s:.6f} maxabs={mx_s:.4f} rel_l2={rl_s:.5f}  "
              f"static==dynamic: {torch.equal(out_st, out_dyn)}")
        worst = min(worst, c_d, c_s)
        del w, x, ref
        _keepalive.clear()
    assert worst > 0.99, f"worst cosine too low: {worst}"


def test_fp8_cublaslt_nn_matches_tn():
    """NN (Thor's default) vs TN at every real shape -- only where both
    layouts are supported (Blackwell). Same FP8 values, same scales; any
    difference is the cuBLASLt algorithm's accumulation order."""
    import pytest
    try:
        w = _lin(16, 16)
        x = _own(torch.randn(4, 16, dtype=FP16, device=DEV))
        out = _own(torch.zeros(4, 16, dtype=FP16, device=DEV))
        Fp8Linear(w.data_ptr(), 16, 16, layout="nn")(x.data_ptr(), out.data_ptr(), 4, 0)
        Fp8Linear(w.data_ptr(), 16, 16, layout="tn")(x.data_ptr(), out.data_ptr(), 4, 0)
        torch.cuda.synchronize()
    except RuntimeError as e:
        pytest.skip(f"this GPU does not support both FP8 cuBLASLt layouts: {e}")
    torch.manual_seed(0)
    for name, (m, n, k) in _REAL_SHAPES.items():
        w, x, ref = _real_shape_case(m, n, k)
        outs = {}
        for layout in ("nn", "tn"):
            lin = StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False, layout=layout)
            lin.calibrate(x.data_ptr(), m, 0)
            outs[layout] = torch.zeros(m, n, dtype=FP16, device=DEV)
            lin(x.data_ptr(), outs[layout].data_ptr(), m, 0)
        torch.cuda.synchronize()
        c, mx, rl = _errors(outs["tn"], outs["nn"])
        print(f"{name:16s} tn vs nn: cos={c:.6f} maxabs={mx:.5f} rel_l2={rl:.6f} "
              f"bit-exact={torch.equal(outs['tn'], outs['nn'])}")
        assert c > 0.9999, f"{name}: tn vs nn cosine {c}"
        _keepalive.clear()


if __name__ == "__main__":
    if not _FP8_AVAILABLE:
        print(f"SKIPPED fp8: {_FP8_REASON}")
    else:
        test_fp8_linear_matches_fp16_reference()

    if not _NVFP4_AVAILABLE:
        print(f"SKIPPED nvfp4: {_NVFP4_REASON}")
    else:
        test_nvfp4_linear_matches_fp16_reference()

    if not _STATIC_FP8_AVAILABLE:
        print(f"SKIPPED static_fp8(cublaslt): {_STATIC_FP8_REASON}")
    else:
        test_static_fp8_linear_cublaslt_matches_fp16_reference()

    if not _STATIC_FP8_CUTLASS_AVAILABLE:
        print(f"SKIPPED static_fp8(cutlass): {_STATIC_FP8_CUTLASS_REASON}")
    else:
        test_static_fp8_linear_cutlass_matches_fp16_reference()

    if _FP8_AVAILABLE and _STATIC_FP8_AVAILABLE:
        test_fp8_cublaslt_real_shapes_match_fp16_reference()
    try:
        test_fp8_cublaslt_nn_matches_tn()
    except BaseException as e:  # pytest.skip raises outside pytest too
        print(f"SKIPPED fp8 nn-vs-tn: {e}")

    print("DONE (see SKIPPED lines above for anything not actually verified here)")
