"""`quant_linear.py`'s `Fp8Linear`/`Nvfp4Linear` correctness (plan.md's
"OPT-004 step 5" Phase 1/2: cosine>0.99 against the FP16 reference,
small dims, no speed measurement).

**Both precisions are UNTESTABLE on this project's own dev machine
(Ada sm_89), for two independent reasons** -- see `quant_linear.py`'s
own module docstring for the full account:

- FP8: this venv's cuBLASLt (12.8.04, CUDA 12.8, Ada compute
  capability (8,9)) returns `cublasLtMatmulAlgoGetHeuristic failed
  with cuBLAS status 15` for `fp8_gemm_descale_fp16` at EVERY shape --
  a pre-existing, already-documented environment gap (`plan.md`'s own
  "Ada FP8 Environment Gap" section), not a wiring bug, not fixable
  from this project's code, not a hardware limitation (the user's real
  Thor run already produced real FP8 numbers with this exact kernel).
- NVFP4: `flash_rt.flash_rt_fp4` (the compiled extension `Nvfp4Linear`
  needs) only exists in a build configured with `-DGPU_ARCH=110`
  (Thor) or a Blackwell target -- not built on this Ada machine at all.

Following `test_imagewam_fa4_backbone.py`'s own established pattern
for a kernel this dev machine cannot run: build the test, detect
unavailability with a real probe (not a guess), skip cleanly rather
than asserting a bar that can never be cleared here. Needs real Thor
hardware to actually verify the cosine>0.99 bar.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.quant_linear import Fp16Linear, Fp8Linear, Nvfp4Linear

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


_FP8_AVAILABLE, _FP8_REASON = _probe_fp8_available()
_NVFP4_AVAILABLE, _NVFP4_REASON = _probe_nvfp4_available()


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
        pytest.skip(f"FP8 GEMM not available on this machine (known Ada cuBLASLt "
                    f"environment gap, see quant_linear.py's module docstring): {_FP8_REASON}")

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
    assert cos > 0.99, f"cosine too low: {cos}"


if __name__ == "__main__":
    if not _FP8_AVAILABLE:
        print(f"SKIPPED fp8: {_FP8_REASON}")
    else:
        test_fp8_linear_matches_fp16_reference()

    if not _NVFP4_AVAILABLE:
        print(f"SKIPPED nvfp4: {_NVFP4_REASON}")
    else:
        test_nvfp4_linear_matches_fp16_reference()

    print("DONE (see SKIPPED lines above for anything not actually verified here)")
