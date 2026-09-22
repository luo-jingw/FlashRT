"""Standalone bit-exact test for the fused_norm_fp4 kernel
(csrc/kernels/fused_norm_fp4/), NOT wired into the production pipeline.

Compares the fused kernel's NVFP4-packed bytes + CUTLASS SFA scale bytes
against the real project's own UNFUSED two-step path, run at real ImageWAM
shapes on the real local GPU:

    fvk.gate_res_bf16res / gate_res_fp16          (csrc/kernels/decoder_fused.cu)
    fvk.ada_layer_norm_bf16in_fp16out / _fp16      (csrc/kernels/norm.cu)
    quantize_fp4_dynamic_sfa_fp16                  (csrc/quantize/quantize_fp4_sfa.cu,
                                                     compiled here unmodified --
                                                     flash_rt.flash_rt_fp4 is a
                                                     Thor/Blackwell-only build
                                                     and is not present on this
                                                     Ada machine)

against the new fused kernel:

    gate_res_ada_layer_norm_fp4_sfa_bf16res / _fp16res
                                                    (csrc/kernels/fused_norm_fp4/)

The quantize step is plain bit/integer manipulation (e2m1 packing + CUTLASS
SFA tile-interleave addressing) with no tensor-core MMA instruction, so it
compiles and runs correctly on this Ada (sm_89) GPU even though the NVFP4
GEMM that later consumes this output needs Blackwell/Thor -- only the
QUANTIZE OUTPUT BYTES are exercised here, never a GEMM.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/test_fused_norm_fp4_kernel.py
"""
from __future__ import annotations

import os
import sys

import torch
from torch.utils.cpp_extension import load

REPO = "/home/ljw/projects/pi0.5/FlashRT"
WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTLASS_INC = f"{REPO}/third_party/cutlass/include"
CUTLASS_UTIL_INC = f"{REPO}/third_party/cutlass/tools/util/include"
BUILD_DIR = "/tmp/claude-1000/-home-ljw-projects-pi0-5/4d8ac22d-337e-4d3f-9b41-5f44259ae605/scratchpad/fused_norm_fp4_build"

EPS = 1e-6  # matches pipeline_thor.py's hardcoded eps for every ada_layer_norm call


def _build_extension():
    os.makedirs(BUILD_DIR, exist_ok=True)
    return load(
        name="fused_norm_fp4_ext",
        sources=[
            f"{WORKTREE}/csrc/kernels/fused_norm_fp4/fused_norm_fp4.cu",
            f"{WORKTREE}/csrc/kernels/fused_norm_fp4/fused_norm_fp4_bindings.cpp",
            f"{WORKTREE}/csrc/quantize/quantize_fp4_sfa.cu",
            f"{WORKTREE}/csrc/quantize/reshape_scales_sfa.cu",
        ],
        extra_include_paths=[f"{WORKTREE}/csrc", CUTLASS_INC, CUTLASS_UTIL_INC],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--ftz=true",
            "--prec-div=false",
            "--prec-sqrt=false",
            "--expt-relaxed-constexpr",
            "-std=c++17",
            "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            "-gencode=arch=compute_89,code=sm_89",
        ],
        build_directory=BUILD_DIR,
        verbose=True,
    )


def _e2m1_to_float(nibble: torch.Tensor) -> torch.Tensor:
    """Dequantize a uint8 e2m1 nibble (0..15) to its represented float value
    -- used only for human-readable ULP/diff diagnostics on mismatch, not for
    the pass/fail comparison itself (that is torch.equal on raw bytes)."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=nibble.device)
    mag = lut[(nibble & 0x7).long()]
    sign = torch.where((nibble & 0x8) != 0, -1.0, 1.0)
    return mag * sign


def _diagnose_packed_mismatch(packed_ref: torch.Tensor, packed_fused: torch.Tensor,
                               sfa_ref: torch.Tensor, sfa_fused: torch.Tensor) -> str:
    diff_mask = packed_ref != packed_fused
    n_diff = int(diff_mask.sum().item())
    lo_ref = (packed_ref & 0xF).to(torch.uint8)
    hi_ref = ((packed_ref >> 4) & 0xF).to(torch.uint8)
    lo_fused = (packed_fused & 0xF).to(torch.uint8)
    hi_fused = ((packed_fused >> 4) & 0xF).to(torch.uint8)
    lo_diff = (_e2m1_to_float(lo_ref) - _e2m1_to_float(lo_fused)).abs()
    hi_diff = (_e2m1_to_float(hi_ref) - _e2m1_to_float(hi_fused)).abs()
    max_level_diff = max(float(lo_diff.max()), float(hi_diff.max()))
    sfa_diff = int((sfa_ref != sfa_fused).sum().item())
    return (f"packed byte mismatches={n_diff}/{packed_ref.numel()} "
            f"max e2m1-level |Δ|={max_level_diff:.3f} (grid step=0.5-2.0) "
            f"sfa byte mismatches={sfa_diff}/{sfa_ref.numel()}")


def run_case(ext, fvk, rows: int, dim: int, res_dtype: torch.dtype) -> bool:
    device = "cuda"
    torch.manual_seed(1234 + rows * 131 + dim)

    # Real value ranges: BF16 residual (backbone) legitimately reaches wide
    # magnitude (opportunities.md OPT-001); FP16 residual (ActionDiT) must
    # stay well inside fp16's ~65504 ceiling.
    res_scale = 200.0 if res_dtype == torch.bfloat16 else 8.0
    residual0 = (torch.randn(rows, dim, device=device, dtype=torch.float32) * res_scale).to(res_dtype)
    gemm_out = (torch.randn(rows, dim, device=device, dtype=torch.float32) * 1.5).to(torch.float16)
    gate = (torch.randn(dim, device=device, dtype=torch.float32) * 0.8).to(torch.float16)
    scale = (torch.randn(dim, device=device, dtype=torch.float32) * 0.5).to(torch.float16)
    shift = (torch.randn(dim, device=device, dtype=torch.float32) * 0.5).to(torch.float16)
    gate_full = gate.unsqueeze(0).expand(rows, dim).contiguous()

    residual_ref = residual0.clone().contiguous()
    residual_fused = residual0.clone().contiguous()
    modded = torch.empty(rows, dim, device=device, dtype=torch.float16)

    if res_dtype == torch.bfloat16:
        fvk.gate_res_bf16res(gemm_out.data_ptr(), gate_full.data_ptr(),
                              residual_ref.data_ptr(), rows * dim, 0)
        fvk.ada_layer_norm_bf16in_fp16out(residual_ref.data_ptr(), scale.data_ptr(),
                                           shift.data_ptr(), modded.data_ptr(),
                                           rows, dim, EPS, 0)
        fused_fn = ext.gate_res_ada_layer_norm_fp4_sfa_bf16res
    else:
        fvk.gate_res_fp16(gemm_out.data_ptr(), gate_full.data_ptr(),
                           residual_ref.data_ptr(), rows * dim, 0)
        fvk.ada_layer_norm_fp16(residual_ref.data_ptr(), scale.data_ptr(),
                                 shift.data_ptr(), modded.data_ptr(),
                                 rows, dim, EPS, 0)
        fused_fn = ext.gate_res_ada_layer_norm_fp4_sfa_fp16res

    sfa_bytes = ext.sfa_size_bytes(rows, dim, False)
    packed_ref = torch.empty(rows, dim // 2, device=device, dtype=torch.uint8)
    sfa_ref = torch.zeros(sfa_bytes, device=device, dtype=torch.uint8)
    rc = ext.quantize_fp4_dynamic_sfa_fp16(modded.data_ptr(), packed_ref.data_ptr(),
                                            sfa_ref.data_ptr(), rows, dim, False, 0)
    assert rc == 0, f"reference quantize_fp4_dynamic_sfa_fp16 rc={rc}"

    packed_fused = torch.empty(rows, dim // 2, device=device, dtype=torch.uint8)
    sfa_fused = torch.zeros(sfa_bytes, device=device, dtype=torch.uint8)
    rc = fused_fn(residual_fused.data_ptr(), gemm_out.data_ptr(), gate.data_ptr(),
                   scale.data_ptr(), shift.data_ptr(), 0,
                   packed_fused.data_ptr(), sfa_fused.data_ptr(),
                   rows, dim, EPS, 0)
    assert rc == 0, f"fused kernel rc={rc}"

    torch.cuda.synchronize()

    residual_match = torch.equal(residual_ref, residual_fused)
    packed_match = torch.equal(packed_ref, packed_fused)
    sfa_match = torch.equal(sfa_ref, sfa_fused)
    ok = residual_match and packed_match and sfa_match

    tag = "bf16res" if res_dtype == torch.bfloat16 else "fp16res"
    status = "PASS bit-exact" if ok else "FAIL"
    print(f"[{tag}] rows={rows:4d} dim={dim:4d}: residual_match={residual_match} "
          f"packed_match={packed_match} sfa_match={sfa_match}  -> {status}")
    if not ok:
        if not residual_match:
            d = (residual_ref.float() - residual_fused.float()).abs()
            print(f"    residual mismatch: max|Δ|={float(d.max())}, n_diff={int((d>0).sum())}")
        if not packed_match or not sfa_match:
            print("    " + _diagnose_packed_mismatch(packed_ref, packed_fused, sfa_ref, sfa_fused))
    return ok


def run_awq_smoke(ext, fvk) -> bool:
    """Diagnostic-only check for the AWQ input-scale-fold path (optional
    `inv_s` argument) -- NOT part of the required bit-exact suite (the
    task's own producer/consumer contract has no `inv_s`; this exists only
    to characterize the AWQ question the task asked about).

    The fused kernel multiplies by `inv_s` on the full-precision (fp32)
    modulated value and rounds through fp16 ONCE afterwards -- the same
    single-rounding convention csrc/fused_fp4/layer_norm_fp4_sfa.cu's own
    `inv_s` parameter uses. There is no way to build a bit-exact "unfused"
    reference for this from the existing two real kernels: `modded` from
    `ada_layer_norm_bf16in_fp16out` is ALREADY rounded to fp16, so folding
    `inv_s` in afterward and quantizing necessarily double-rounds (fp16
    round of the LN output, then a second fp16 round of modded*inv_s) --
    a real, expected discrepancy from the fused kernel's single-rounding
    path, not a bug in either. This function measures exactly that gap
    (expected to NOT be bit-exact) rather than asserting a false pass/fail;
    see the module docstring / final report for the conclusion this
    supports: the AWQ fold can reuse the SAME kernel (one extra nullable
    `inv_s` arg, already implemented), but a bit-exact regression test for
    it would need a THIRD reference kernel (fold inv_s before the only
    fp16 rounding), which is out of this task's scope."""
    device = "cuda"
    rows, dim = 64, 3072
    torch.manual_seed(999)
    residual0 = (torch.randn(rows, dim, device=device, dtype=torch.float32) * 100.0).to(torch.bfloat16)
    gemm_out = (torch.randn(rows, dim, device=device, dtype=torch.float32) * 1.5).to(torch.float16)
    gate = (torch.randn(dim, device=device, dtype=torch.float32) * 0.8).to(torch.float16)
    scale = (torch.randn(dim, device=device, dtype=torch.float32) * 0.5).to(torch.float16)
    shift = (torch.randn(dim, device=device, dtype=torch.float32) * 0.5).to(torch.float16)
    inv_s = (torch.rand(dim, device=device, dtype=torch.float32) * 0.9 + 0.1).to(torch.float16)
    gate_full = gate.unsqueeze(0).expand(rows, dim).contiguous()

    residual_ref = residual0.clone().contiguous()
    residual_fused = residual0.clone().contiguous()
    modded = torch.empty(rows, dim, device=device, dtype=torch.float16)

    fvk.gate_res_bf16res(gemm_out.data_ptr(), gate_full.data_ptr(),
                          residual_ref.data_ptr(), rows * dim, 0)
    fvk.ada_layer_norm_bf16in_fp16out(residual_ref.data_ptr(), scale.data_ptr(),
                                       shift.data_ptr(), modded.data_ptr(),
                                       rows, dim, EPS, 0)
    torch.cuda.synchronize()
    modded_awq = (modded.float() * inv_s.float().unsqueeze(0)).to(torch.float16).contiguous()

    sfa_bytes = ext.sfa_size_bytes(rows, dim, False)
    packed_ref = torch.empty(rows, dim // 2, device=device, dtype=torch.uint8)
    sfa_ref = torch.zeros(sfa_bytes, device=device, dtype=torch.uint8)
    rc = ext.quantize_fp4_dynamic_sfa_fp16(modded_awq.data_ptr(), packed_ref.data_ptr(),
                                            sfa_ref.data_ptr(), rows, dim, False, 0)
    assert rc == 0

    packed_fused = torch.empty(rows, dim // 2, device=device, dtype=torch.uint8)
    sfa_fused = torch.zeros(sfa_bytes, device=device, dtype=torch.uint8)
    rc = ext.gate_res_ada_layer_norm_fp4_sfa_bf16res(
        residual_fused.data_ptr(), gemm_out.data_ptr(), gate.data_ptr(),
        scale.data_ptr(), shift.data_ptr(), inv_s.data_ptr(),
        packed_fused.data_ptr(), sfa_fused.data_ptr(), rows, dim, EPS, 0)
    assert rc == 0
    torch.cuda.synchronize()

    packed_match = torch.equal(packed_ref, packed_fused)
    sfa_match = torch.equal(sfa_ref, sfa_fused)
    ok = packed_match and sfa_match
    print(f"[awq-fold diagnostic, NOT required] rows={rows} dim={dim}: "
          f"packed_match={packed_match} sfa_match={sfa_match} "
          f"-> {'bit-exact (unexpected)' if ok else 'differs (expected: double-rounding, see docstring)'}")
    print("    " + _diagnose_packed_mismatch(packed_ref, packed_fused, sfa_ref, sfa_fused))
    return ok


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}")

    ext = _build_extension()
    import flash_rt.flash_rt_kernels as fvk

    shapes = [(r, d) for r in (25, 417, 905, 64) for d in (3072, 1024)]
    dtypes = [torch.bfloat16, torch.float16]

    results = []
    for res_dtype in dtypes:
        for rows, dim in shapes:
            results.append(run_case(ext, fvk, rows, dim, res_dtype))

    n_pass = sum(results)
    n_total = len(results)
    print(f"\nTOTAL (required suite): {n_pass}/{n_total} bit-exact")

    print()
    run_awq_smoke(ext, fvk)  # diagnostic only, not part of the pass/fail gate

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
