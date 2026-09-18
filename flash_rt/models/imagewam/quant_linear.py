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

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.blockscaled_ref import BLOCK, prepare_e0m3_hadamard_weight

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16
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


class Bf16OutLinear:
    """`out[M,N]` (BF16) `= x[M,K]` (BF16) `@ W[K,N]` (BF16) --
    ImageWAM real-Qwen3-conditioning fix (opportunities.md OPT-001 "FP16
    residual overflow"). Used ONLY for `txt_in.weight`/`img_in.weight`
    (the two projections that WRITE the persistent backbone residual
    buffer directly, before any layer even runs) -- every other
    weight-projection GEMM in `pipeline_thor.py` still reads/writes
    plain FP16 via `Fp16Linear` (or a quantized variant), since real
    Thor tracing showed only the residual accumulator itself ever
    reaches magnitudes FP16 cannot hold (~120000 with real Qwen3-4B
    text conditioning at x0=512); everything downstream of each layer's
    own AdaLayerNorm (which re-normalizes back to O(1-10) regardless of
    the residual's own scale) stays comfortably FP16-safe.

    Applied regardless of `self._precision` (unlike `Fp16Linear`/
    `Fp8Linear`/etc., which are selected BY `self._precision`) --
    quantizing this one small entry-point GEMM is orthogonal to, and
    would not fix, the residual-overflow problem this class exists for.
    Uses `GemmRunner.bf16_nn`, which (like `fp16_nn`) requires `A`/`B`/`D`
    all be the same dtype -- so the real weight is cast to BF16 once at
    construction (see `imagewam_thor.py`'s `_wrap_bf16out_linear`), and
    the caller (`pipeline_thor.py`) must pass a BF16 `x_ptr` (the real
    Qwen3/VAE-encoded `context`/`img_raw` buffers) and a BF16 `out_ptr`
    (the `backbone_hidden` residual buffer).
    """

    def __init__(self, gemm, weight_ptr: int, n: int, k: int):
        self.gemm = gemm
        self.weight_ptr = int(weight_ptr)
        self.n, self.k = int(n), int(k)

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        self.gemm.bf16_nn(x_ptr, self.weight_ptr, out_ptr, m, self.n, self.k, stream)


def _pick_fp16_cutlass_variant(n: int, k: int) -> str:
    """Shape-based tile-config pick for the `cutlass_fp16_*` family
    (opportunities.md OPT-013) -- mirrors `_pick_fp8_cutlass_variant`
    above EXACTLY (same "wide-N -> a wide-tile variant" heuristic,
    same caveat: ImageWAM's own shape mix has M always small relative
    to N/K, this has NOT been calibrated against a real Thor profiling
    sweep across the full `plain/sq/t1/wide/k64/2sm21` family -- `sq`/
    `wide` are the two variants `_pick_fp8_cutlass_variant` already
    uses for this same shape mix, kept as the default here too rather
    than guessing among the other four with no evidence). Pass
    `variant=` explicitly to `CutlassFp16Linear` to override for a
    real Thor A/B sweep against `k64`/`2sm21`/etc -- do not treat this
    default as settled without one."""
    if n >= 4 * k:
        return "wide"
    return "sq"


class CutlassFp16Linear:
    """`out[M,N]` (fp16) `= x[M,K]` (fp16) `@ W[K,N]` (fp16) -- SAME
    math as `Fp16Linear` (no quantization, `alpha=1, beta=0`), but
    dispatched through a hand-tuned CUTLASS kernel
    (`cutlass_fp16_plain/sq/t1/wide/k64/2sm21`, `csrc/gemm/cutlass_sm100_fp16.cu`)
    instead of cuBLASLt's own generic autotuned algorithm selection --
    opportunities.md OPT-013, found while surveying what FlashRT's
    OTHER Thor models (GROOT/Pi0.5's `shared_primitives.py`) already do
    that ImageWAM didn't: a real, Thor-tuned FP16 CUTLASS GEMM family
    already exists in this codebase (confirmed via a code comment
    dated 2026-05-18 recording real Thor timing), just never wired
    into ImageWAM. Zero new math vs `Fp16Linear` -- this is a pure
    GEMM-backend swap, so a real Thor A/B against the existing
    `precision="fp16"` default is the only way to know if it actually
    wins here (ImageWAM's own (M,N,K) shapes and Thor occupancy may
    differ from whatever shape this kernel family was originally tuned
    against elsewhere in this codebase).

    **Thor/Blackwell-only, same gate as `StaticFp8Linear(use_cutlass=True)`/
    `Nvfp4Linear`** -- these SM100/SM110 CUTLASS kernels are absent
    from a plain Ada (`GPU_ARCH=89`) build; `hasattr(fvk, "cutlass_fp16_k64")`
    probes availability, raising a clear `RuntimeError` (not silently
    falling back) on any other build, matching this module's own
    established pattern for every other CUTLASS-gated class.

    **CUTLASS's own B-matrix convention differs from this project's
    usual `(K,N)` GEMM-storage layout**, same as `Nvfp4Linear`/
    `StaticFp8Linear(use_cutlass=True)`: CUTLASS FP16 reads B as
    `[N,K]` row-major (`cutlass_sm100_fp16.cu`'s own header comment:
    "same as PyTorch nn.Linear weights") -- the one-time
    `.t().contiguous()` at construction handles this, identical to
    those two classes' own pattern.
    """

    def __init__(self, weight_fp16_ptr: int, n: int, k: int, *, variant: str | None = None):
        self.n, self.k = int(n), int(k)
        if not hasattr(fvk, "cutlass_fp16_k64"):
            raise RuntimeError(
                "CutlassFp16Linear requires a build with ENABLE_SM100_CUTLASS "
                "(Thor/Blackwell, GPU_ARCH=110) -- cutlass_fp16_* kernels are "
                "absent from this flash_rt_kernels build. Same gate FP8-static+"
                "CUTLASS/NVFP4 already use; see quant_linear.py's own module "
                "docstring.")
        self._variant = variant if variant is not None else _pick_fp16_cutlass_variant(self.n, self.k)
        self._cutlass_fn = getattr(fvk, f"cutlass_fp16_{self._variant}")
        w_kn = _wrap_fp16(weight_fp16_ptr, self.k, self.n)  # (K,N), this project's usual convention
        w_nk = w_kn.t().contiguous()  # (N,K), CUTLASS's own out-major convention -- one-time real copy
        self._w = w_nk
        self.weight_ptr = w_nk.data_ptr()

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        rc = self._cutlass_fn(x_ptr, self.weight_ptr, out_ptr, m, self.n, self.k,
                               1.0, 0.0, stream)
        if rc != 0:
            raise RuntimeError(
                f"cutlass_fp16_{self._variant} failed rc={rc} (M={m}, N={self.n}, K={self.k})")


class CutlassFp16SwiGluMlp:
    """ImageWAM's real MLP up-projection, `out[M,mlp_hidden] =
    SiLU(x[M,K] @ W_gate[K,mlp_hidden]) * (x[M,K] @ W_up[K,mlp_hidden])`
    -- the real SwiGLU gate (opportunities.md OPT-013), fused into TWO
    CUTLASS kernels instead of today's default path (ONE wide cuBLASLt
    GEMM producing a merged `(seq, 2*mlp_hidden)` `[gate;up]` buffer,
    THEN a separate `silu_glu_merged_fp16` elementwise kernel reading
    it back to compute `silu(gate)*up`):
      1. `cutlass_fp16_k64_silu`: `gate_buf = SiLU(x @ W_gate)` (SiLU
         fused into this GEMM's own epilogue -- ImageWAM's real
         activation, confirmed against `csrc/kernels/activation.cu`'s
         own `silu_glu_merged_kernel` formula, NOT the `GeluTanhApprox`
         a different model's own GEMM+GELU epilogue variant uses).
      2. `cutlass_fp16_k64_mul_aux`: `out = (x @ W_up) * gate_buf`
         (the up-projection GEMM and the gate multiply fused into ONE
         kernel's own epilogue, via CUTLASS's `LinCombDeEltAct` aux-tensor
         load).
    Net: zero separate elementwise kernel, same total GEMM FLOPs as
    today's one wide GEMM (split into two `mlp_hidden`-wide ones
    instead of one `2*mlp_hidden`-wide one) -- whether this nets out
    faster than the current cuBLASLt-wide-GEMM-plus-elementwise-kernel
    path is a real Thor timing question, not assumed.

    **Splits the real checkpoint's own merged `mlp0.weight`
    `(K, 2*mlp_hidden)` (columns `[0:mlp_hidden)`=gate,
    `[mlp_hidden:2*mlp_hidden)`=up -- confirmed against
    `silu_glu_merged_kernel`'s own indexing) into two separate
    `(mlp_hidden, K)` CUTLASS-layout weights ONCE at construction** --
    a plain column-slice + `.t().contiguous()`, no new checkpoint
    format needed, no accuracy change (same real trained weight values,
    just split into two tensors instead of read as one).

    Allocates its own `gate_buf` scratch (`(max_m, mlp_hidden)` fp16)
    lazily on first call, sized from that call's own `m` and reused
    (same convention as `Nvfp4Linear`'s own lazy scratch) -- safe
    across CUDA Graph replay since `m` is fixed per capture (this
    project's own established invariant: shapes never change between
    capture and replay).

    Same Thor/Blackwell-only gate as `CutlassFp16Linear`.
    """

    def __init__(self, merged_weight_fp16_ptr: int, mlp_hidden: int, k: int):
        self.mlp_hidden, self.k = int(mlp_hidden), int(k)
        if not hasattr(fvk, "cutlass_fp16_k64_silu"):
            raise RuntimeError(
                "CutlassFp16SwiGluMlp requires a build with ENABLE_SM100_CUTLASS "
                "(Thor/Blackwell, GPU_ARCH=110) -- cutlass_fp16_k64_silu/"
                "_k64_mul_aux are absent from this flash_rt_kernels build.")
        w_merged_kn = _wrap_fp16(merged_weight_fp16_ptr, self.k, 2 * self.mlp_hidden)
        w_gate_kn = w_merged_kn[:, :self.mlp_hidden].contiguous()
        w_up_kn = w_merged_kn[:, self.mlp_hidden:].contiguous()
        self._w_gate = w_gate_kn.t().contiguous()  # (mlp_hidden, K), CUTLASS's own out-major convention
        self._w_up = w_up_kn.t().contiguous()
        self.w_gate_ptr = self._w_gate.data_ptr()
        self.w_up_ptr = self._w_up.data_ptr()
        self._gate_buf = None
        self._max_m = 0

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        if self._gate_buf is None or m > self._max_m:
            self._gate_buf = torch.zeros(m, self.mlp_hidden, dtype=FP16, device=DEV)
            self._max_m = m
        gate_ptr = self._gate_buf.data_ptr()
        rc = fvk.cutlass_fp16_k64_silu(x_ptr, self.w_gate_ptr, gate_ptr,
                                        m, self.mlp_hidden, self.k, 1.0, 0.0, stream)
        if rc != 0:
            raise RuntimeError(f"cutlass_fp16_k64_silu failed rc={rc} (M={m}, N={self.mlp_hidden}, K={self.k})")
        rc = fvk.cutlass_fp16_k64_mul_aux(x_ptr, self.w_up_ptr, gate_ptr, out_ptr,
                                           m, self.mlp_hidden, self.k, stream)
        if rc != 0:
            raise RuntimeError(f"cutlass_fp16_k64_mul_aux failed rc={rc} (M={m}, N={self.mlp_hidden}, K={self.k})")


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


def _pick_fp8_cutlass_variant(n: int, k: int) -> str:
    """Shape-based tile-config pick for `cutlass_fp8_sq`/`_wide`/`_t1`
    (OPT-004 step 6 plan). Mirrors `fp4_utils.py`'s own `pick_variant`
    in SPIRIT (wide-N -> a wide-tile variant), but its thresholds are
    NOT calibrated against a real profiling sweep the way
    `fp4_utils.py`'s own thresholds were (`docs/v2/fp4_kernel_impl_progress.md`
    §4.3) -- this is a provisional heuristic for ImageWAM's own shape
    mix (M always small relative to N/K here, unlike Pi0.5's decoder
    M=10 case that motivated `_t1` there), not yet a measured choice.
    Phase 4's real Thor per-layer P50s should confirm or retune this
    before treating it as settled -- do not hardcode past that point
    without validating at the real production shape."""
    if n >= 4 * k:
        return "wide"
    return "sq"


# Every `cutlass_fp8_*` FP16-output GEMM tile a `StaticFp8Linear(use_cutlass=True)`
# can switch to (`gemm_variant_tuner.VariantTunableGemm`). The first four are
# the 256-row / clustered tiles `csrc/gemm/cutlass_sm100.cu` has always had;
# the `t128x*` tiles are the 1-SM, cluster 1x1x1 small-M tiles
# (`gemm_types_sm100.h`, `sm100_small_m`), `t128x64x256` being Pi0.5's v10
# decoder tile shape.
FP8_CUTLASS_VARIANTS = ("sq", "wide", "t1", "plain",
                        "t128x64x256", "t128x64x128", "t128x128x128", "t128x256x128")


class StaticFp8Linear:
    """`out[M,N]` (fp16) `= x[M,K]` (fp16, quantized to fp8 with a
    FROZEN static scale) `@ W[K,N]` (fp8, quantized ONCE at
    construction) -- OPT-004 step 6 plan, the house calibration
    pattern every OTHER FlashRT Thor model uses (`docs/calibration.md`),
    NOT the per-call dynamic scale `Fp8Linear` above uses. `Fp8Linear`
    is NOT removed/replaced by this class -- it stays the "no
    calibration step needed" fallback (its own docstring already
    explains that tradeoff); this class exists for the specific case
    (`imagewam_thor.py`'s CUDA-Graph-captured production frontend)
    where a fixed set of kernel launches gets replayed unchanged many
    times, making a ONE-TIME calibration cost worth paying to remove a
    PER-REPLAY amax reduction.

    Two independently switchable changes, per the plan's own Problem
    section (isolated so a real Thor measurement can tell which one, if
    either, explains ImageWAM's own dynamic-FP8 regression):
    - Static activation scale (`calibrate()` freezes it once via the
      SAME `quantize_fp8_device_fp16` kernel `Fp8Linear` calls every
      forward; `__call__` then uses the cheaper `quantize_fp8_static_fp16`,
      no amax reduction, every time) -- always on for this class.
    - GEMM kernel choice (`use_cutlass=False`: keep `fp8_gemm_descale_fp16`
      /cuBLASLt, isolating the scale change alone -- Phase 1.
      `use_cutlass=True`: `cutlass_fp8_sq`/`_wide`/`_t1`, gated behind
      the SAME `ENABLE_SM100_CUTLASS`/Thor-only build flag NVFP4 already
      uses successfully on the user's Thor build -- Phase 2. Probed via
      `hasattr(fvk, "cutlass_fp8_sq")`, not a separate extension import
      like NVFP4's `flash_rt.flash_rt_fp4` -- these symbols live in the
      same `flash_rt_kernels` module, just conditionally compiled in.)

    **CUTLASS weight layout differs from this project's usual (K,N)
    GEMM-storage convention**: `cutlass_fp8_sq`/`_wide`/`_t1` read B as
    `[N,K]` row-major (confirmed by reading `csrc/gemm/cutlass_sm100.cu`'s
    own `cutlass_run_impl`: `stride_B` is packed for `{N,K,1}`) -- the
    SAME out-major convention `Nvfp4Linear` already handles via a
    one-time `.t().contiguous()`. `use_cutlass=False` (cuBLASLt) keeps
    the usual (K,N) layout, matching `Fp8Linear`'s own convention --
    the two GEMM backends genuinely need different weight storage, not
    just a different function call.

    **Ordering contract, enforced, not just documented**: `calibrate()`
    MUST run before the first `__call__` (a captured CUDA Graph replays
    identical kernel launches forever -- calibrating after capture would
    freeze a scale the graph could never actually use, or worse, replay
    stale/uninitialized scale memory). `__call__` raises `RuntimeError`
    if `calibrate()` has not run yet; `calibrate()` after even one
    `__call__` also raises, since that ordering bug is just as real
    (silently producing a captured graph whose replay never matches
    what was verified before capture) and just as cheap to catch here.

    **Tile variant (`use_cutlass=True` only)**: implements
    `gemm_variant_tuner.VariantTunableGemm` with `family="fp8_cutlass"`.
    `default_variant` is `_pick_fp8_cutlass_variant`'s pick and is what
    `__call__` runs until `set_variant` changes it; a CUDA graph captures
    the variant current at capture time. The tuning methods stage their
    own activation copy and scale and never touch `act_scale` or the
    calibrate/call ordering state. With `use_cutlass=False` (cuBLASLt)
    there is one GEMM path: `family="fp8_cublaslt"`, `variant="cublaslt"`,
    and the tuning methods raise.
    """

    def __init__(self, weight_fp16_ptr: int, n: int, k: int, *, use_cutlass: bool = False):
        self.n, self.k = int(n), int(k)
        self.use_cutlass = use_cutlass
        self.family = "fp8_cutlass" if use_cutlass else "fp8_cublaslt"
        self.default_variant = "cublaslt"
        self.variant = "cublaslt"
        self._calibrated = False
        self._called = False

        w_kn = _wrap_fp16(weight_fp16_ptr, self.k, self.n)  # (K,N), this project's usual convention
        if use_cutlass:
            if not hasattr(fvk, "cutlass_fp8_sq"):
                raise RuntimeError(
                    "StaticFp8Linear(use_cutlass=True) requires a build with "
                    "ENABLE_SM100_CUTLASS (Thor/Blackwell, GPU_ARCH=110) -- "
                    "cutlass_fp8_sq/_wide/_t1 are absent from this "
                    "flash_rt_kernels build. Same gate NVFP4 already uses; "
                    "see quant_linear.py's own module docstring.")
            self.default_variant = _pick_fp8_cutlass_variant(self.n, self.k)
            self.variant = self.default_variant
            w_nk = w_kn.t().contiguous()  # (N,K), CUTLASS's own out-major convention
            self.w_f8 = torch.empty(self.n, self.k, dtype=F8, device=DEV)
            self._quantize_weight(w_nk, self.w_f8)
        else:
            self.w_f8 = torch.empty(self.k, self.n, dtype=F8, device=DEV)
            self._quantize_weight(w_kn, self.w_f8)

        self.act_scale = torch.zeros(1, dtype=torch.float32, device=DEV)
        self.act_f8 = None
        self._max_m = 0
        self._tune_act_f8 = None
        self._tune_scale = None
        self._tune_alpha = 1.0

    def _quantize_weight(self, w: torch.Tensor, out_f8: torch.Tensor) -> None:
        w_scale = torch.zeros(1, dtype=torch.float32, device=DEV)
        fvk.quantize_fp8_device_fp16(w.data_ptr(), out_f8.data_ptr(), w_scale.data_ptr(),
                                      w.numel(), 0)
        self.w_scale = w_scale

    def calibrate(self, x_ptr: int, m: int, stream: int = 0) -> None:
        """Measure and FREEZE the activation scale from one
        representative forward pass -- see class docstring for the
        ordering contract this enforces. `x_ptr`/`m` are the SAME
        random dry-run activations this project already uses everywhere
        (NOT real calibration data; see the plan's own "deliberately
        not in scope" note for why real house calibration waits for a
        real checkpoint, OPT-001)."""
        if self._called:
            raise RuntimeError("StaticFp8Linear.calibrate() called after __call__ -- "
                                "would silently invalidate an already-captured graph")
        scratch = torch.empty(m, self.k, dtype=F8, device=DEV)
        fvk.quantize_fp8_device_fp16(x_ptr, scratch.data_ptr(), self.act_scale.data_ptr(),
                                      m * self.k, stream)
        torch.cuda.synchronize()
        # Host-side alpha, computed HERE (before any graph capture), never
        # inside __call__ -- a captured CUDA Graph cannot re-issue a host
        # sync (`.item()`) on every replay, and docs/calibration.md's own
        # "must be f32, not f64" rule (a historical Pi0.5 regression,
        # 0.9992 -> 0.9878, from implicit f64 multiplication) applies here
        # identically.
        self._alpha_host = float(np.float32(self.act_scale.item()) * np.float32(self.w_scale.item()))
        self._calibrated = True

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        if not self._calibrated:
            raise RuntimeError("StaticFp8Linear.__call__ before calibrate()")
        self._called = True
        if self.act_f8 is None or m > self._max_m:
            self.act_f8 = torch.empty(m, self.k, dtype=F8, device=DEV)
            self._max_m = m
        fvk.quantize_fp8_static_fp16(x_ptr, self.act_f8.data_ptr(), self.act_scale.data_ptr(),
                                      m * self.k, stream)
        if self.use_cutlass:
            cutlass_fn = getattr(fvk, f"cutlass_fp8_{self.variant}")
            rc = cutlass_fn(self.act_f8.data_ptr(), self.w_f8.data_ptr(), out_ptr,
                             m, self.n, self.k, self._alpha_host, 0.0, stream)
            if rc != 0:
                raise RuntimeError(f"cutlass_fp8_{self.variant} failed rc={rc}")
        else:
            fvk.fp8_gemm_descale_fp16(self.act_f8.data_ptr(), self.w_f8.data_ptr(), out_ptr,
                                       m, self.n, self.k, self.act_scale.data_ptr(), self.w_scale.data_ptr(),
                                       stream)

    def _require_cutlass(self) -> None:
        if not self.use_cutlass:
            raise RuntimeError("tile variants exist only for StaticFp8Linear(use_cutlass=True)")

    def candidate_variants(self) -> tuple[str, ...]:
        self._require_cutlass()
        return FP8_CUTLASS_VARIANTS

    def set_variant(self, variant: str) -> None:
        self._require_cutlass()
        if variant not in FP8_CUTLASS_VARIANTS:
            raise ValueError(f"unknown FP8 CUTLASS variant {variant!r}; known: {FP8_CUTLASS_VARIANTS}")
        self.variant = variant

    def prepare_tuning_input(self, x_ptr: int, m: int, stream: int = 0) -> None:
        """Quantize `x` into this op's own tuning buffer with its own
        dynamic scale (not `act_scale`, which `calibrate()` owns)."""
        self._require_cutlass()
        self._tune_act_f8 = torch.empty(m, self.k, dtype=F8, device=DEV)
        self._tune_scale = torch.zeros(1, dtype=torch.float32, device=DEV)
        fvk.quantize_fp8_device_fp16(x_ptr, self._tune_act_f8.data_ptr(), self._tune_scale.data_ptr(),
                                      m * self.k, stream)
        torch.cuda.synchronize()
        self._tune_alpha = float(np.float32(self._tune_scale.item()) * np.float32(self.w_scale.item()))

    def launch_variant(self, variant: str, out_ptr: int, m: int, stream: int = 0) -> int:
        self._require_cutlass()
        if self._tune_act_f8 is None:
            raise RuntimeError("launch_variant() before prepare_tuning_input()")
        cutlass_fn = getattr(fvk, f"cutlass_fp8_{variant}")
        return int(cutlass_fn(self._tune_act_f8.data_ptr(), self.w_f8.data_ptr(), out_ptr,
                              m, self.n, self.k, self._tune_alpha, 0.0, stream))


# `cutlass_fp4_gemm_variant` indices (csrc/gemm/fp4/cutlass_fp4_gemm_variants.cu)
# an `Nvfp4Linear` can switch to: every cluster-1x1x1 tile. v4 128x128x128,
# v5 128x64x128, v6 128x256x128, v7 128x128x256, v8 128x256x256, v10
# 128x64x256 (Pi0.5's decoder tile). The clustered tiles (v0-v3, v9) are left
# out: Pi0.5 measured cluster tiles winning in isolation and losing in the
# pipeline on Thor (docs/pi05_thor_decoder_fp4_e2e.md). The default pick
# (`fp4_utils.pick_variant`) is always a candidate too.
NVFP4_VARIANTS = ("v4", "v5", "v6", "v7", "v8", "v10")


def _nvfp4_variant_index(variant: str) -> int:
    if not variant.startswith("v") or not variant[1:].isdigit():
        raise ValueError(f"NVFP4 variant must look like 'v<index>', got {variant!r}")
    return int(variant[1:])


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

    **Tile variant**: implements `gemm_variant_tuner.VariantTunableGemm`
    with `family="nvfp4"`. `default_variant` is `fp4_utils.pick_variant`'s
    pick (what `fp4_gemm` has always selected here) and is what
    `__call__` runs until `set_variant` changes it; a CUDA graph captures
    the variant current at capture time.
    """

    family = "nvfp4"

    def __init__(self, weight_fp16_ptr: int, n: int, k: int):
        try:
            import flash_rt.flash_rt_fp4 as fvk_fp4
            from flash_rt.executors.fp4_utils import (
                FP4ActScratch,
                fp4_gemm,
                pick_variant,
                quant_act_nvfp4,
                quant_weight_nvfp4,
            )
        except ImportError as e:
            raise RuntimeError(
                "Nvfp4Linear requires a Blackwell/Thor NVFP4 build "
                "(flash_rt.flash_rt_fp4) -- configure cmake with "
                "-DGPU_ARCH=110 and rebuild (see "
                "benchmarks/imagewam_thor_fp4_bench.py's own docstring)."
            ) from e
        if k % 16 != 0:
            raise ValueError(f"NVFP4 requires K divisible by 16, got K={k}")
        self._fvk_fp4 = fvk_fp4
        self._fp4_act_scratch_cls = FP4ActScratch
        self._fp4_gemm = fp4_gemm
        self._quant_act = quant_act_nvfp4
        self.n, self.k = int(n), int(k)
        self.default_variant = f"v{pick_variant(self.n, self.k)}"
        self.variant = self.default_variant

        w_kn = _wrap_fp16(weight_fp16_ptr, self.k, self.n)  # (K,N), this project's own convention
        w_nk = w_kn.t().contiguous()  # (N,K), NVFP4's own convention -- one-time real copy
        self.w_quant = quant_weight_nvfp4(w_nk)
        self.scratch = None

    def _ensure_scratch(self, m: int) -> None:
        if self.scratch is None or m > self.scratch.max_M:
            self.scratch = self._fp4_act_scratch_cls(m, self.k, device=DEV)

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        self._ensure_scratch(m)
        x = _wrap_fp16(x_ptr, m, self.k)
        out = _wrap_fp16(out_ptr, m, self.n)
        self._quant_act(x, self.scratch, m, stream)
        self._fp4_gemm(self.scratch, self.w_quant, out, m, self.n, self.k,
                       variant_idx=_nvfp4_variant_index(self.variant), stream=stream)

    def candidate_variants(self) -> tuple[str, ...]:
        return NVFP4_VARIANTS

    def set_variant(self, variant: str) -> None:
        idx = _nvfp4_variant_index(variant)
        count = int(self._fvk_fp4.cutlass_fp4_gemm_num_variants())
        if not 0 <= idx < count:
            raise ValueError(f"NVFP4 variant {variant!r} out of range [0, {count})")
        self.variant = variant

    def prepare_tuning_input(self, x_ptr: int, m: int, stream: int = 0) -> None:
        """Quantize `x` into this op's own activation scratch (the same
        buffer `__call__` uses; the next `__call__` overwrites it)."""
        self._ensure_scratch(m)
        self._quant_act(_wrap_fp16(x_ptr, m, self.k), self.scratch, m, stream)

    def launch_variant(self, variant: str, out_ptr: int, m: int, stream: int = 0) -> int:
        if self.scratch is None:
            raise RuntimeError("launch_variant() before prepare_tuning_input()")
        return int(self._fvk_fp4.cutlass_fp4_gemm_variant(
            _nvfp4_variant_index(variant),
            self.scratch.packed.data_ptr(), self.scratch.sfa.data_ptr(),
            self.w_quant['packed'].data_ptr(), self.w_quant['sfb'].data_ptr(),
            out_ptr, m, self.n, self.k, 1.0, 0.0, stream))


class Nvfp4SwiGluMlp:
    """ImageWAM's own NVFP4 MLP gate/up fusion (opportunities.md op-fusion
    audit finding 2, mirrors `CutlassFp16SwiGluMlp` above but for NVFP4).

    Today's `nvfp4` MLP-gate path (via the generic `_wrap_linear`/
    `Nvfp4Linear` dispatch every OTHER precision also uses) is: one WIDE
    NVFP4 GEMM against the real checkpoint's own merged `(2*mlp_hidden, K)`
    weight -> a full `(m, 2*mlp_hidden)` fp16 buffer written -> the plain
    `silu_glu_merged_fp16` kernel reads it -> a `(m, mlp_hidden)` fp16
    gated buffer written. This class instead: splits the SAME merged
    weight into two `(mlp_hidden, K)` NVFP4-quantized halves at
    construction (one-time, mirrors `CutlassFp16SwiGluMlp`'s own column-
    split) -> two separate NVFP4 GEMMs, each producing an `(m, mlp_hidden)`
    FP4-PACKED (4-bit, ~1/4 the bytes of fp16) intermediate via
    `fp4out_gemm`/`FP4Buffer` (the "split-GU FFN path" building blocks
    this codebase already had for a different model, never wired to
    ImageWAM) -> the new TRUE-SiLU combiner kernel
    (`silu_glu_two_fp4_to_fp16`, added alongside this class -- the
    existing `geglu_two_fp4_to_fp4` computes GELU-tanh, confirmed by
    reading its actual device-code formula, not SiLU, despite its
    "silu_mul" internal naming) reads both FP4-packed intermediates and
    writes the `(m, mlp_hidden)` fp16 gated buffer directly, no FP4
    requantization needed since the down-projection GEMM immediately
    after (unchanged, still the generic `Nvfp4Linear` dispatch) already
    re-quantizes its own fp16 input internally.

    Net effect: the intermediate gate/up representation is FP4-packed
    instead of a full-width fp16 merged buffer -- real DRAM-traffic
    reduction on the intermediate, not just a launch-count change (see
    this codebase's own `silu_mul_two_fp4_to_fp4.cu` module docstring:
    "reads HALF the activation DRAM... vs fp16 today" for the same
    mechanism in a different model). Activation is quantized to FP4 ONCE
    per call and reused for both GEMMs (matches the existing single-wide-
    GEMM path's own single activation-quant cost, not doubled).
    """

    def __init__(self, merged_weight_fp16_ptr: int, mlp_hidden: int, k: int):
        try:
            import flash_rt.flash_rt_fp4  # noqa: F401 -- import-time availability check
            from flash_rt.executors.fp4_utils import (
                FP4ActScratch,
                FP4Buffer,
                fp4out_gemm,
                quant_act_nvfp4,
                quant_weight_nvfp4,
                silu_glu_two_fp4_to_fp16,
            )
        except ImportError as e:
            raise RuntimeError(
                "Nvfp4SwiGluMlp requires a Blackwell/Thor NVFP4 build "
                "(flash_rt.flash_rt_fp4) -- configure cmake with "
                "-DGPU_ARCH=110 and rebuild (see "
                "benchmarks/imagewam_thor_fp4_bench.py's own docstring)."
            ) from e
        if k % 16 != 0:
            raise ValueError(f"NVFP4 requires K divisible by 16, got K={k}")
        self.mlp_hidden, self.k = int(mlp_hidden), int(k)
        self._fp4_act_scratch_cls = FP4ActScratch
        self._fp4_buffer_cls = FP4Buffer
        self._fp4out_gemm = fp4out_gemm
        self._quant_act = quant_act_nvfp4
        self._combine = silu_glu_two_fp4_to_fp16

        w_kn = _wrap_fp16(merged_weight_fp16_ptr, self.k, 2 * self.mlp_hidden)  # (K, 2*mlp_hidden)
        w_gate_nk = w_kn[:, :self.mlp_hidden].t().contiguous()  # (mlp_hidden, K), NVFP4's own convention
        w_up_nk = w_kn[:, self.mlp_hidden:].t().contiguous()    # (mlp_hidden, K)
        self.w_gate_quant = quant_weight_nvfp4(w_gate_nk)
        self.w_up_quant = quant_weight_nvfp4(w_up_nk)

        self.scratch = None
        self._gate_buf = None
        self._up_buf = None
        self._max_m = 0

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        if self.scratch is None or m > self._max_m:
            self.scratch = self._fp4_act_scratch_cls(m, self.k, device=DEV)
            self._gate_buf = self._fp4_buffer_cls(m, self.mlp_hidden, device=DEV)
            self._up_buf = self._fp4_buffer_cls(m, self.mlp_hidden, device=DEV)
            self._max_m = m
        x = _wrap_fp16(x_ptr, m, self.k)
        out = _wrap_fp16(out_ptr, m, self.mlp_hidden)
        self._quant_act(x, self.scratch, m, stream)
        self._fp4out_gemm(self.scratch, self.w_gate_quant,
                           self._gate_buf.packed.data_ptr(), self._gate_buf.sfa.data_ptr(),
                           m, self.mlp_hidden, self.k, stream=stream)
        self._fp4out_gemm(self.scratch, self.w_up_quant,
                           self._up_buf.packed.data_ptr(), self._up_buf.sfa.data_ptr(),
                           m, self.mlp_hidden, self.k, stream=stream)
        self._combine(self._gate_buf.packed.data_ptr(), self._gate_buf.sfa.data_ptr(),
                      self._up_buf.packed.data_ptr(), self._up_buf.sfa.data_ptr(),
                      out.data_ptr(), m, self.mlp_hidden, stream)


class E0m3HadamardLinear:
    """`out[M,N]` (fp16) `= x[M,K]` (fp16) `@ W[K,N]` with both operands
    in E0M3 (uniform INT4, codes -7..7, one UE4M3 scale per 16 K values)
    after the same orthonormal 16-point Hadamard rotation of every
    16-wide K block (`opportunities.md` OPT-024). The rotation is its own
    inverse and block-diagonal along K, so the product is unchanged in
    exact arithmetic and any `K % 16 == 0` works.

    Construction (once): transpose the `(K,N)` fp16 weight to `(N,K)`,
    prepare it with `blockscaled_ref.prepare_e0m3_hadamard_weight`
    (butterfly rotation in fp32, per-tensor power of two `2^e` chosen so
    the largest block scale `amax/7` stays at or below UE4M3's 448, which
    moves the block scales out of UE4M3's subnormal range where ImageWAM's
    weights otherwise put more than 90% of them, fp16), and quantize with
    `quantize_e0m3_dynamic_sfa_fp16` into packed codes + SFB. The GEMM
    `alpha = 2^-e` undoes the pre-scale exactly.

    Call: `quantize_e0m3_dynamic_sfa_fp16_vec(use_rht=1)` rotates and
    quantizes the activation into this object's scratch (allocated on
    the first call, grown if `m` grows), then
    `cutlass_fp4_gemm_e0m3w_variant(a_format=0)` runs the SM110
    runtime-descriptor block-scaled GEMM with the tile the NVFP4 path
    picks for the same `(N, K)` (`fp4_utils.pick_variant`).

    Thor/Blackwell only: raises `RuntimeError` at construction without
    `flash_rt.flash_rt_fp4`.
    """

    def __init__(self, weight_fp16_ptr: int, n: int, k: int) -> None:
        try:
            import flash_rt.flash_rt_fp4 as fvk_fp4
            from flash_rt.executors.fp4_utils import pick_variant
        except ImportError as e:
            raise RuntimeError(
                "E0m3HadamardLinear requires a Blackwell/Thor build "
                "(flash_rt.flash_rt_fp4) -- configure cmake with -DGPU_ARCH=110 and rebuild."
            ) from e
        if k % BLOCK != 0 or n % BLOCK != 0:
            raise ValueError(f"E0M3 requires N and K divisible by {BLOCK}, got N={n} K={k}")
        self._fp4 = fvk_fp4
        self.n, self.k = int(n), int(k)
        # Rotation (butterfly, fp32), per-tensor 2^e pre-scale, fp16.
        w_in, self.alpha = prepare_e0m3_hadamard_weight(
            _wrap_fp16(weight_fp16_ptr, self.k, self.n).t().contiguous())
        self.w_packed = torch.empty(self.n, self.k // 2, dtype=torch.uint8, device=DEV)
        # Zero-init: SF layout padding is never written and must stay inert.
        self.w_sfb = torch.zeros(fvk_fp4.sfa_size_bytes(self.n, self.k, True), dtype=torch.uint8, device=DEV)
        rc = fvk_fp4.quantize_e0m3_dynamic_sfa_fp16(
            w_in.data_ptr(), self.w_packed.data_ptr(), self.w_sfb.data_ptr(), self.n, self.k, True, 0)
        torch.cuda.synchronize()
        if rc != 0:
            raise RuntimeError(f"quantize_e0m3_dynamic_sfa_fp16 (weight) failed rc={rc}")
        self.variant = int(pick_variant(self.n, self.k))
        self.a_packed = None
        self.a_sfa = None
        self._max_m = 0

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        if m > self._max_m:
            self.a_packed = torch.empty(m, self.k // 2, dtype=torch.uint8, device=DEV)
            self.a_sfa = torch.zeros(self._fp4.sfa_size_bytes(m, self.k, False), dtype=torch.uint8, device=DEV)
            self._max_m = m
        rc = self._fp4.quantize_e0m3_dynamic_sfa_fp16_vec(
            x_ptr, self.a_packed.data_ptr(), self.a_sfa.data_ptr(), m, self.k, False, 1, stream)
        if rc != 0:
            raise RuntimeError(f"quantize_e0m3_dynamic_sfa_fp16_vec failed rc={rc} (x_ptr must be 16-byte aligned)")
        rc = self._fp4.cutlass_fp4_gemm_e0m3w_variant(
            self.variant, self.a_packed.data_ptr(), self.a_sfa.data_ptr(),
            self.w_packed.data_ptr(), self.w_sfb.data_ptr(), out_ptr,
            m, self.n, self.k, self.alpha, 0.0, stream, 0)
        if rc != 0:
            raise RuntimeError(f"cutlass_fp4_gemm_e0m3w_variant({self.variant}) failed rc={rc:#x}")
