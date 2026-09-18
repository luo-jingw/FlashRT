"""ImageWAM precision tiers and their properties, as data.

Before this module the properties of a precision were spread over
`flash_rt/frontends/torch/imagewam_thor.py` as four string tuples
(`_PRECISIONS`, `_NVFP4_PRECISIONS`, `_STATIC_FP8_PRECISIONS`,
`_VARIANT_TUNED_PRECISIONS`), the `if self._precision == ...` chain of
`_wrap_linear` (alignment fallbacks), and three single-precision
comparisons (`merge_qkv_mlp`, the fused SwiGLU MLP, the `fp16_nn` GEMM
autotune). `PROPERTIES` below is that knowledge in one table; every entry
is pinned against the frontend source by
`tests/test_imagewam_precision_table.py`.

This module is a leaf: standard library only. It does not import torch,
the compiled kernels or the frontend, so the resolver, the tests and
tooling can reason about a precision without a GPU stack. (Importing it
as `flash_rt.models.imagewam.precision` still runs the package's
`__init__`, which imports the pipeline; load the file by path where that
matters.)

`Precision` is a `str` enum whose values are today's precision strings,
so `Precision("nvfp4") == "nvfp4"`, `"nvfp4" in tuple_of_strings`, and
`str(Precision.NVFP4)` / f-string formatting (used in runtime-identity
strings) all keep producing the plain string.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class PrecisionProperties:
    """One row of the precision property table.

    alignment: the multiple that BOTH `n` and `k` of a `(k, n)` linear must
        be for this precision's native GEMM (`Precision.alignment_fallback`).
        `1` means no constraint (the plain fp16 path).
    needs_calibration: a one-time activation-scale calibration runs before
        graph capture (`_calibrate_fp8`; `_STATIC_FP8_PRECISIONS`). With a
        calibration file the scales are real; without one the frontend
        today falls back to a logged N(0, 0.1) PLACEHOLDER and continues,
        it does not refuse.
    supports_native_runtime: every linear this precision builds is one the
        native pipeline can describe (`pipeline_resources.linear_resource`
        accepts `Fp16Linear`, `Bf16OutLinear` and `Nvfp4Linear` only) and
        it uses the merged single-stream path the native pipeline records.
    supports_awq: the NVFP4 family, the only one that takes `awq_inv_s`
        (`_NVFP4_PRECISIONS`).
    supports_tile_autotune: switchable CUTLASS tile, the only ones
        `gemm_variant_autotune=True` applies to (`_VARIANT_TUNED_PRECISIONS`).
    merge_qkv_mlp: the single-stream `linear1` (qkv + mlp gate/up) runs as
        one GEMM; also the precondition for `merge_linear2`.
    fused_swiglu_mlp: the MLP gate/up is `CutlassFp16SwiGluMlp`, a
        standalone weight (which is why this precision cannot merge).
    fp16_nn_backbone_gemm: the backbone weight GEMMs run on the runner's
        `fp16_nn` path, so a new text length autotunes those shapes too.
    """

    alignment: int
    needs_calibration: bool
    supports_native_runtime: bool
    supports_awq: bool
    supports_tile_autotune: bool
    merge_qkv_mlp: bool
    fused_swiglu_mlp: bool
    fp16_nn_backbone_gemm: bool


class Precision(str, Enum):
    """The precision tiers of the ImageWAM Thor frontend.

    Member order equals the frontend's `_PRECISIONS` tuple. `Precision(x)`
    raises `ValueError` for anything else.
    """

    FP16 = "fp16"
    FP16_CUTLASS = "fp16_cutlass"
    FP8 = "fp8"
    NVFP4 = "nvfp4"
    FP8_STATIC = "fp8_static"
    FP8_STATIC_CUTLASS = "fp8_static_cutlass"
    E0M3_HADAMARD = "e0m3_hadamard"
    # NVFP4 numerics emulated with fp16 GEMMs (SimNvfp4Linear); accuracy
    # work on GPUs without Blackwell FP4, not a fast path.
    NVFP4_SIM = "nvfp4_sim"

    def __str__(self) -> str:
        return self.value

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)

    @property
    def properties(self) -> PrecisionProperties:
        return PROPERTIES[self]

    @property
    def alignment(self) -> int:
        return PROPERTIES[self].alignment

    @property
    def needs_calibration(self) -> bool:
        return PROPERTIES[self].needs_calibration

    @property
    def supports_native_runtime(self) -> bool:
        return PROPERTIES[self].supports_native_runtime

    @property
    def supports_awq(self) -> bool:
        return PROPERTIES[self].supports_awq

    @property
    def supports_tile_autotune(self) -> bool:
        return PROPERTIES[self].supports_tile_autotune

    @property
    def merge_qkv_mlp(self) -> bool:
        return PROPERTIES[self].merge_qkv_mlp

    @property
    def merge_linear2_allowed(self) -> bool:
        """`merge_linear2=True` needs `merge_qkv_mlp` (the merged path
        writes the SiLU-GLU output straight into the `linear2` input)."""
        return PROPERTIES[self].merge_qkv_mlp

    @property
    def fused_swiglu_mlp(self) -> bool:
        return PROPERTIES[self].fused_swiglu_mlp

    @property
    def fp16_nn_backbone_gemm(self) -> bool:
        return PROPERTIES[self].fp16_nn_backbone_gemm

    def accepts_calibration_path(self, *, nvfp4_awq: bool = False) -> bool:
        """Whether a calibration file is consumed: by the static-FP8 tiers
        (activation scales) and, with `nvfp4_awq`, by the AWQ tiers
        (per-channel statistics). Mirrors the frontend's `calibration_path`
        check."""
        return self.needs_calibration or (nvfp4_awq and self.supports_awq)

    def alignment_fallback(self, n: int, k: int) -> bool:
        """True when a `(k, n)` linear must use the plain fp16 linear
        (`Fp16Linear`, cuBLASLt) instead of this precision's native linear.

        Reproduces `_wrap_linear`: the native GEMM needs `n` and `k` both
        divisible by `alignment` (8 for `fp16_cutlass` and the FP8 tiers,
        16 for the block-scaled tiers `nvfp4`, `e0m3_hadamard`,
        `nvfp4_sim`); otherwise it falls back. `fp16` has no native
        alternative, so it never falls back. In the real model only
        `action_encoder` (K=7) and `head.linear` (N=7) fall back; AWQ
        eligibility (`_plan_awq`, `% 16` on both dims) is the same rule for
        the AWQ tiers.
        """
        a = PROPERTIES[self].alignment
        return n % a != 0 or k % a != 0


_MERGED_QUANT = dict(merge_qkv_mlp=True, fused_swiglu_mlp=False, fp16_nn_backbone_gemm=False)

PROPERTIES: dict[Precision, PrecisionProperties] = {
    Precision.FP16: PrecisionProperties(
        alignment=1, needs_calibration=False, supports_native_runtime=True,
        supports_awq=False, supports_tile_autotune=False,
        merge_qkv_mlp=True, fused_swiglu_mlp=False, fp16_nn_backbone_gemm=True),
    Precision.FP16_CUTLASS: PrecisionProperties(
        alignment=8, needs_calibration=False, supports_native_runtime=False,
        supports_awq=False, supports_tile_autotune=False,
        merge_qkv_mlp=False, fused_swiglu_mlp=True, fp16_nn_backbone_gemm=False),
    Precision.FP8: PrecisionProperties(
        alignment=8, needs_calibration=False, supports_native_runtime=False,
        supports_awq=False, supports_tile_autotune=False, **_MERGED_QUANT),
    Precision.NVFP4: PrecisionProperties(
        alignment=16, needs_calibration=False, supports_native_runtime=True,
        supports_awq=True, supports_tile_autotune=True, **_MERGED_QUANT),
    Precision.FP8_STATIC: PrecisionProperties(
        alignment=8, needs_calibration=True, supports_native_runtime=False,
        supports_awq=False, supports_tile_autotune=False, **_MERGED_QUANT),
    Precision.FP8_STATIC_CUTLASS: PrecisionProperties(
        alignment=8, needs_calibration=True, supports_native_runtime=False,
        supports_awq=False, supports_tile_autotune=True, **_MERGED_QUANT),
    Precision.E0M3_HADAMARD: PrecisionProperties(
        alignment=16, needs_calibration=False, supports_native_runtime=False,
        supports_awq=False, supports_tile_autotune=False, **_MERGED_QUANT),
    Precision.NVFP4_SIM: PrecisionProperties(
        alignment=16, needs_calibration=False, supports_native_runtime=False,
        supports_awq=True, supports_tile_autotune=False, **_MERGED_QUANT),
}

assert set(PROPERTIES) == set(Precision), "PROPERTIES must have one row per Precision member"
