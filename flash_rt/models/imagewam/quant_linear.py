"""ImageWAM quantized weight-projection GEMM wrappers (OPT-004 step 5,
`plan.md`'s "Plan: OPT-004 step 5 — FP8/NVFP4 quantized GEMM for
ImageWAM"). `pipeline_thor.py`'s real-math layer functions dispatch
every weight-projection GEMM through one of these, uniformly, via
`weights[key](x_ptr, out_ptr, m, stream)` -- the caller never branches
on precision; all of that lives here.

Promotes the ALREADY-PROVEN `_Fp8Linear`/`_Fp4Linear` wrapper pattern
from `benchmarks/imagewam_thor_fp8_bench.py`/`imagewam_thor_fp4_bench.py`
(disconnected, old-approximate-math scripts) into real, reusable,
pointer-based code that operates on a REAL weight (quantized once at
construction), not a random one, and that plugs directly into
`pipeline_thor.py`'s own pointer-passing convention end to end
(`x_ptr`/`out_ptr` ints, not `torch.Tensor` args). The underlying
quantization/GEMM KERNELS themselves are not new (see `plan.md`'s
Phase 0: `fp8_gemm_descale_fp16` dispatches through `cublasLtMatmul`,
architecture-portable by design; `fp4_gemm` is a genuine CUTLASS
kernel built with `-arch=sm_110a` specifically for Thor's own
tensor-core generation) -- this module is glue code, not new CUDA.

**FP8 is ALSO untestable for real numeric correctness on this dev
machine, for a completely different reason than NVFP4** (confirmed
2026-09-14 while wiring this module, and already independently
documented in `plan.md`'s own "Ada FP8 Environment Gap — Confirmed,
Not This Project's Bug" section from an earlier session): this venv's
cuBLASLt (12.8.04, CUDA 12.8, Ada compute capability (8,9)) returns
`cublasLtMatmulAlgoGetHeuristic failed with cuBLAS status 15`
(`CUBLAS_STATUS_NOT_SUPPORTED`) for `fp8_gemm_descale_fp16` at EVERY
shape tried, including trivial ones -- not a shape problem, not a bug
in this module or in `pipeline_thor.py`'s own wiring, and not a
hardware limitation (Ada Lovelace has real FP8 tensor cores; the
user's own real Thor run already produced real FP8 numbers with this
exact kernel). `Fp8Linear` itself is straightforward and mirrors
`_Fp8Linear`'s own already-Thor-proven call pattern exactly -- this
note exists so a `RuntimeError` from `Fp8Linear.__call__` on THIS
machine is correctly read as "known environment gap, needs Thor," not
as a wiring bug to chase.

**NVFP4 is UNTESTED on this project's own dev machine** (Ada sm_89):
`flash_rt.flash_rt_fp4` (the compiled NVFP4 extension) only exists in
a build configured with `-DGPU_ARCH=110` (Thor) or a Blackwell target
-- confirmed by reading `benchmarks/imagewam_thor_fp4_bench.py`'s own
docstring ("this script has never executed successfully anywhere")
and by directly attempting the import here (`ModuleNotFoundError`).
`Nvfp4Linear` therefore imports `flash_rt.flash_rt_fp4`/
`flash_rt.executors.fp4_utils` LAZILY, inside `__init__`, so this
module stays importable everywhere; only actually constructing an
`Nvfp4Linear` requires the real extension. FP8 (`Fp8Linear`) has no
such restriction -- `fp8_gemm_descale_fp16` lives in the main
`flash_rt_kernels` extension and is exercised by every existing test
this session already ran on Ada.
"""
from __future__ import annotations

import torch

import flash_rt.flash_rt_kernels as fvk

DEV = "cuda"
FP16 = torch.float16
F8 = torch.float8_e4m3fn


def _wrap_fp16(ptr: int, m: int, n: int) -> torch.Tensor:
    """Zero-copy CUDA tensor view over a raw fp16 pointer, contiguous
    `(m, n)` -- same technique as `pipeline_thor.py`'s own `_wrap_fp16`
    (not shared via import, matching that module's own reasoning: no
    cross-file coupling for a handful of lines every pointer-based
    module in this project already re-derives)."""
    interface = {
        "data": (int(ptr), False),
        "shape": (int(m), int(n)),
        "typestr": "<f2",
        "version": 3,
    }
    owner = type("_QuantLinearFp16View", (), {"__cuda_array_interface__": interface})()
    return torch.as_tensor(owner, device=DEV)


class Fp16Linear:
    """Plain FP16 passthrough -- `out[M,N] = x[M,K] @ W[K,N]`, no
    quantization. Exists purely so `pipeline_thor.py`'s call sites are
    UNIFORM regardless of precision (`weights[key](...)` always works,
    whether that value is this class or `Fp8Linear`/`Nvfp4Linear`).
    """

    def __init__(self, gemm, weight_ptr: int, n: int, k: int):
        self.gemm = gemm
        self.weight_ptr = int(weight_ptr)
        self.n, self.k = int(n), int(k)

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        self.gemm.fp16_nn(x_ptr, self.weight_ptr, out_ptr, m, self.n, self.k, stream)


class Fp8Linear:
    """`out[M,N]` (fp16) `= x[M,K]` (fp16, quantized to fp8 on the fly)
    `@ W[K,N]` (fp8, quantized ONCE at construction from a REAL fp16
    weight). Both scales are real, dynamically measured values
    (`quantize_fp8_device_fp16`'s own GPU-side absmax -> compute_scale
    -> quantize, no calibration, no host sync, no fixed placeholder --
    matches `imagewam_thor_fp8_bench.py`'s own already-established
    convention for why this is the right primitive to use here).
    """

    def __init__(self, weight_fp16_ptr: int, n: int, k: int):
        self.n, self.k = int(n), int(k)
        w = _wrap_fp16(weight_fp16_ptr, self.k, self.n)  # (K,N), matches gemm.fp16_nn's own convention
        self.w_f8 = torch.empty(self.k, self.n, dtype=F8, device=DEV)
        self.w_scale = torch.zeros(1, dtype=torch.float32, device=DEV)
        fvk.quantize_fp8_device_fp16(w.data_ptr(), self.w_f8.data_ptr(), self.w_scale.data_ptr(),
                                      self.k * self.n, 0)
        self.act_scale = torch.zeros(1, dtype=torch.float32, device=DEV)
        self.act_f8 = None
        self._max_m = 0

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        if self.act_f8 is None or m > self._max_m:
            self.act_f8 = torch.empty(m, self.k, dtype=F8, device=DEV)
            self._max_m = m
        fvk.quantize_fp8_device_fp16(x_ptr, self.act_f8.data_ptr(), self.act_scale.data_ptr(),
                                      m * self.k, stream)
        fvk.fp8_gemm_descale_fp16(self.act_f8.data_ptr(), self.w_f8.data_ptr(), out_ptr,
                                   m, self.n, self.k, self.act_scale.data_ptr(), self.w_scale.data_ptr(),
                                   stream)


class Nvfp4Linear:
    """`out[M,N]` (fp16) `= x[M,K]` (fp16, quantized on the fly) `@
    W[N,K]^T` (nvfp4, quantized ONCE at construction). See module
    docstring for the Blackwell/Thor-only availability constraint --
    raises a clear `RuntimeError` at construction (not at import time)
    on any build without `flash_rt.flash_rt_fp4`.

    NVFP4's own weight convention is `[N,K]` (out-major), NOT this
    project's usual `(K,N)` GEMM storage convention every OTHER weight
    in `pipeline_thor.py`'s `weights` dict uses -- confirmed by reading
    `flash_rt/executors/fp4_utils.py`'s own `quant_weight_nvfp4`
    docstring directly. The one-time transpose+copy at construction
    (`.t().contiguous()`) handles this; every OTHER call site in this
    module and in `pipeline_thor.py` itself stays in the usual `(K,N)`
    convention, only this class's own constructor differs internally.
    """

    def __init__(self, weight_fp16_ptr: int, n: int, k: int):
        try:
            import flash_rt.flash_rt_fp4  # noqa: F401 -- import-time availability check
            from flash_rt.executors.fp4_utils import FP4ActScratch, fp4_gemm, quant_act_nvfp4, quant_weight_nvfp4
        except ImportError as e:
            raise RuntimeError(
                "Nvfp4Linear requires a Blackwell/Thor NVFP4 build "
                "(flash_rt.flash_rt_fp4) -- configure cmake with "
                "-DGPU_ARCH=110 and rebuild (see "
                "benchmarks/imagewam_thor_fp4_bench.py's own docstring)."
            ) from e
        if k % 16 != 0:
            raise ValueError(f"NVFP4 requires K divisible by 16, got K={k}")
        self._fp4_act_scratch_cls = FP4ActScratch
        self._fp4_gemm = fp4_gemm
        self._quant_act = quant_act_nvfp4
        self.n, self.k = int(n), int(k)

        w_kn = _wrap_fp16(weight_fp16_ptr, self.k, self.n)  # (K,N), this project's own convention
        w_nk = w_kn.t().contiguous()  # (N,K), NVFP4's own convention -- one-time real copy
        self.w_quant = quant_weight_nvfp4(w_nk)
        self.scratch = None

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        if self.scratch is None:
            self.scratch = self._fp4_act_scratch_cls(m, self.k, device=DEV)
        x = _wrap_fp16(x_ptr, m, self.k)
        out = _wrap_fp16(out_ptr, m, self.n)
        self._quant_act(x, self.scratch, m, stream)
        self._fp4_gemm(self.scratch, self.w_quant, out, m, self.n, self.k, stream=stream)
