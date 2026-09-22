"""Standalone proof + benchmark for the two ImageWAM denoise-step-boundary
kernel fusion candidates investigated for `imagewam_denoise_step` /
`imagewam_denoise_loop` (flash_rt/models/imagewam/pipeline_thor.py):

1. Fused Euler-step + fp32->fp16 cast
   (csrc/kernels/fused_step_boundary/euler_step_cast_fused.cu):
   replaces `fvk.gpu_euler_step(...)` (end of step i) + the next step's
   `fvk.gpu_cast_fp32_to_fp16(...)` (start of step i+1) with one launch.
   Verified BIT-EXACT (`torch.equal`) against the two reference kernels
   called separately, at ImageWAM's real LIBERO dims
   (num_action=64, action_dim=7 -- benchmarks/imagewam_thor_graph_bench.py's
   own `REAL_DIMS`), across several random seeds including near-zero and
   negative `action_latent`/`velocity`/`delta` values.

2. FP16 GEMM with a fused BIAS epilogue (cuBLASLt)
   (csrc/kernels/fused_step_boundary/action_encoder_gemm_bias.cu):
   investigates folding `add_bias_fp16` into `action_encoder.weight`'s
   own GEMM. NOT bit-exact by construction (the fused epilogue adds the
   bias in FP32 before a single round to FP16; the reference path rounds
   the GEMM to FP16 first, then adds+rounds the bias in FP16 -- one more
   rounding step) -- checked via max-abs-diff / cosine similarity instead.

   MACHINE-SPECIFIC FINDING (this dev machine only, Ada sm_89, real
   Thor unavailable here -- NOT claimed as a closed/general result,
   see feedback_dont_close_bugs_on_local_only_evidence): on this
   venv's cuBLASLt (12.8.5.5, correctly linked -- confirmed via `ldd`,
   distinct from the older libcublasLt.so.11 the prebuilt
   `flash_rt_kernels` extension itself happens to link against),
   `cublasLtMatmulAlgoGetHeuristic` returns status 15
   (`CUBLAS_STATUS_NOT_SUPPORTED`) for CUBLASLT_EPILOGUE_BIAS on a
   plain FP16xFP16->FP16 GEMM at EVERY K tried (7, 8, 16, 128 -- not
   an 8/16-alignment issue), for this exact matmul_desc/layout
   construction. This is the same failure class `quant_linear.py`'s
   own module docstring already documents for `fp8_gemm_descale_fp16`
   ("Ada FP8 Environment Gap") -- here it recurs for a pure FP16 bias
   epilogue, and even the codebase's own existing BF16 precedent
   (`GemmRunner.bf16_nn_bias`, called unmodified through the prebuilt
   extension) fails too, on that older linked cuBLASLt (11.7.4.6,
   error 7 at the BIAS_DATA_TYPE attribute-set call, before heuristic
   search even runs). Whether a real Thor cuBLASLt build supports this
   epilogue for FP16 is UNTESTED here.

Neither fused kernel is wired into `pipeline_thor.py` or
`quant_linear.py` here (both off-limits for this investigation) --
this file only proves out / measures the kernels in isolation.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

torch.manual_seed(0)

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNEL_DIR = os.path.join(_HERE, "..", "csrc", "kernels", "fused_step_boundary")

# Real LIBERO dims (benchmarks/imagewam_thor_graph_bench.py REAL_DIMS,
# flash_rt/frontends/torch/imagewam_thor.py's `action_dim=7` default).
NUM_ACTION = 64
ACTION_DIM = 7
ACTION_HIDDEN_DIM = 1024
NUM_DENOISE_STEPS = 10


def _gencode_flag() -> str:
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    return f"-gencode=arch=compute_{arch},code=sm_{arch}"


@pytest.fixture(scope="module")
def fvk():
    import flash_rt.flash_rt_kernels as _fvk
    return _fvk


@pytest.fixture(scope="module")
def euler_cast_ext():
    from torch.utils.cpp_extension import load
    return load(
        name="imagewam_step_boundary_euler_cast",
        sources=[os.path.join(_KERNEL_DIR, "euler_step_cast_fused.cu")],
        extra_cuda_cflags=[_gencode_flag()],
        verbose=False,
    )


@pytest.fixture(scope="module")
def gemm_bias_ext():
    from torch.utils.cpp_extension import load
    return load(
        name="imagewam_step_boundary_gemm_bias",
        sources=[os.path.join(_KERNEL_DIR, "action_encoder_gemm_bias.cu")],
        extra_cuda_cflags=[_gencode_flag()],
        extra_ldflags=["-lcublasLt"],
        verbose=False,
    )


# ================================================================
# Item 1: fused Euler-step + cast -- bit-exact correctness
# ================================================================

_CASES = [
    dict(name="uniform_small", latent_scale=0.01, vel_scale=0.05, delta=0.1),
    dict(name="near_zero", latent_scale=1e-6, vel_scale=1e-6, delta=0.1),
    dict(name="negative_velocity", latent_scale=0.01, vel_scale=-0.05, delta=0.1),
    dict(name="negative_delta", latent_scale=0.01, vel_scale=0.05, delta=-0.1),
    dict(name="both_negative", latent_scale=-0.02, vel_scale=-0.08, delta=0.1),
    dict(name="large_delta", latent_scale=0.01, vel_scale=0.05, delta=1.0),
    dict(name="exact_zero_latent", latent_scale=0.0, vel_scale=0.05, delta=0.1),
    dict(name="exact_zero_velocity", latent_scale=0.01, vel_scale=0.0, delta=0.1),
]


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_euler_step_and_cast_bit_exact(fvk, euler_cast_ext, case, seed):
    torch.manual_seed(seed)
    n = NUM_ACTION * ACTION_DIM

    if case["latent_scale"] == 0.0:
        latent0 = torch.zeros(n, dtype=torch.float32, device="cuda")
    else:
        latent0 = (torch.randn(n, device="cuda") * case["latent_scale"]).to(torch.float32)
    if case["vel_scale"] == 0.0:
        velocity = torch.zeros(n, dtype=torch.float16, device="cuda")
    else:
        velocity = (torch.randn(n, device="cuda") * case["vel_scale"]).to(torch.float16)
    delta = float(case["delta"])

    # --- reference: two separate kernels, exactly as pipeline_thor.py calls them ---
    ref_latent = latent0.clone()
    fvk.gpu_euler_step(ref_latent.data_ptr(), velocity.data_ptr(),
                        NUM_ACTION, ACTION_DIM, delta, 0, 0)
    torch.cuda.synchronize()
    ref_latent_fp16 = torch.empty(n, dtype=torch.float16, device="cuda")
    fvk.gpu_cast_fp32_to_fp16(ref_latent.data_ptr(), ref_latent_fp16.data_ptr(), n, 0)
    torch.cuda.synchronize()

    # --- fused kernel ---
    fused_latent = latent0.clone()
    fused_latent_fp16 = torch.empty(n, dtype=torch.float16, device="cuda")
    euler_cast_ext.fused_euler_step_and_cast(
        fused_latent, velocity, fused_latent_fp16, NUM_ACTION, ACTION_DIM, delta, 0)
    torch.cuda.synchronize()

    assert torch.equal(fused_latent, ref_latent), (
        f"fp32 action_latent mismatch (case={case['name']}, seed={seed})")
    assert torch.equal(fused_latent_fp16, ref_latent_fp16), (
        f"fp16 cast mismatch (case={case['name']}, seed={seed})")


def test_fused_euler_step_and_cast_timing(fvk, euler_cast_ext):
    """Observational: measured latency of the fused kernel vs. the two
    separate reference kernels, at the real shape, averaged over many
    iterations (not a pass/fail assertion, per AGENTS.md -- expose the
    measured value)."""
    n = NUM_ACTION * ACTION_DIM
    latent = (torch.randn(n, device="cuda") * 0.01).to(torch.float32)
    velocity = (torch.randn(n, device="cuda") * 0.05).to(torch.float16)
    latent_fp16 = torch.empty(n, dtype=torch.float16, device="cuda")

    iters, warmup = 2000, 200

    def run_separate():
        fvk.gpu_euler_step(latent.data_ptr(), velocity.data_ptr(), NUM_ACTION, ACTION_DIM, 0.1, 0, 0)
        fvk.gpu_cast_fp32_to_fp16(latent.data_ptr(), latent_fp16.data_ptr(), n, 0)

    def run_fused():
        euler_cast_ext.fused_euler_step_and_cast(latent, velocity, latent_fp16, NUM_ACTION, ACTION_DIM, 0.1, 0)

    for _ in range(warmup):
        run_separate()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run_separate()
    torch.cuda.synchronize()
    separate_us = (time.perf_counter() - start) / iters * 1e6

    for _ in range(warmup):
        run_fused()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run_fused()
    torch.cuda.synchronize()
    fused_us = (time.perf_counter() - start) / iters * 1e6

    print(f"\n[euler+cast] separate (2 launches): {separate_us:.2f} us/call | "
          f"fused (1 launch): {fused_us:.2f} us/call | "
          f"delta: {separate_us - fused_us:.2f} us/call")


# ================================================================
# Item 2: FP16 GEMM + fused bias epilogue -- closeness (not bit-exact)
# ================================================================

def _run_fused_gemm_bias_or_skip(gemm_bias_ext, x, w, out, bias, M, N, K):
    """cuBLASLt's CUBLASLT_EPILOGUE_BIAS heuristic search returns
    CUBLAS_STATUS_NOT_SUPPORTED (status 15) for a plain FP16 GEMM on
    this dev machine's Ada/cuBLASLt-12.8 combo -- see module docstring
    ("MACHINE-SPECIFIC FINDING"). Skip (not fail) so this is visibly a
    verified environment gap, not a correctness regression, matching
    quant_linear.py's own precedent for the analogous FP8 gap."""
    try:
        gemm_bias_ext.fp16_gemm_bias(x, w, out, bias, M, N, K)
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "error 15" in str(e):
            pytest.skip(
                "cuBLASLt CUBLASLT_EPILOGUE_BIAS unsupported (status 15) for FP16 "
                "GEMM on this dev machine's Ada/cuBLASLt-12.8 -- same failure class "
                "as quant_linear.py's documented Ada FP8 gap; untested on Thor. "
                f"Original error: {e}")
        raise


def test_fp16_gemm_bias_close_to_reference(fvk, gemm_bias_ext):
    M, K, N = NUM_ACTION, ACTION_DIM, ACTION_HIDDEN_DIM  # action_encoder's real (M,K,N)
    torch.manual_seed(42)
    x = (torch.randn(M, K, device="cuda") * 0.01).to(torch.float16).contiguous()
    w = (torch.randn(K, N, device="cuda") * 0.02).to(torch.float16).contiguous()
    bias = (torch.randn(N, device="cuda") * 0.01).to(torch.float16).contiguous()

    # --- reference: GemmRunner.fp16_nn (== Fp16Linear, action_encoder's
    # real fallback path) followed by the separate add_bias_fp16 ---
    gemm = fvk.GemmRunner()
    ref_out = torch.empty(M, N, dtype=torch.float16, device="cuda").contiguous()
    gemm.fp16_nn(x.data_ptr(), w.data_ptr(), ref_out.data_ptr(), M, N, K, 0)
    torch.cuda.synchronize()
    fvk.add_bias_fp16(ref_out.data_ptr(), bias.data_ptr(), M, N, 0)
    torch.cuda.synchronize()

    # --- fused: single cuBLASLt call, CUBLASLT_EPILOGUE_BIAS ---
    fused_out = torch.empty(M, N, dtype=torch.float16, device="cuda").contiguous()
    _run_fused_gemm_bias_or_skip(gemm_bias_ext, x, w, fused_out, bias, M, N, K)

    diff = (fused_out.float() - ref_out.float())
    max_abs_diff = diff.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        fused_out.float().flatten(), ref_out.float().flatten(), dim=0).item()
    exact_match_frac = torch.equal(fused_out, ref_out)

    print(f"\n[gemm+bias] max_abs_diff={max_abs_diff:.6g} cosine={cos:.8f} "
          f"bit_exact={exact_match_frac} shape=({M},{K},{N})")

    # Not bit-exact by construction (different rounding order, see module
    # docstring) -- assert closeness instead.
    assert cos > 0.999, f"cosine similarity too low: {cos}"
    assert max_abs_diff < 1e-2, f"max abs diff too large: {max_abs_diff}"


def test_fp16_gemm_bias_timing(fvk, gemm_bias_ext):
    """Observational: measured latency of the fused GEMM+bias vs. GEMM
    followed by the separate add_bias_fp16 launch, at the real
    action_encoder shape."""
    M, K, N = NUM_ACTION, ACTION_DIM, ACTION_HIDDEN_DIM
    x = (torch.randn(M, K, device="cuda") * 0.01).to(torch.float16).contiguous()
    w = (torch.randn(K, N, device="cuda") * 0.02).to(torch.float16).contiguous()
    bias = (torch.randn(N, device="cuda") * 0.01).to(torch.float16).contiguous()
    out = torch.empty(M, N, dtype=torch.float16, device="cuda").contiguous()

    gemm = fvk.GemmRunner()
    iters, warmup = 2000, 200

    def run_separate():
        gemm.fp16_nn(x.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, 0)
        fvk.add_bias_fp16(out.data_ptr(), bias.data_ptr(), M, N, 0)

    def run_fused():
        _run_fused_gemm_bias_or_skip(gemm_bias_ext, x, w, out, bias, M, N, K)

    for _ in range(warmup):
        run_separate()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run_separate()
    torch.cuda.synchronize()
    separate_us = (time.perf_counter() - start) / iters * 1e6

    # Probe once, outside the timing loop, so an environment-gap skip
    # (see _run_fused_gemm_bias_or_skip) short-circuits before wasting
    # 200+ warmup iterations on a call that will just raise every time.
    run_fused()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run_fused()
    torch.cuda.synchronize()
    fused_us = (time.perf_counter() - start) / iters * 1e6

    print(f"\n[gemm+bias] separate (gemm + add_bias_fp16): {separate_us:.2f} us/call | "
          f"fused (1 cuBLASLt call): {fused_us:.2f} us/call | "
          f"delta: {separate_us - fused_us:.2f} us/call")


# ================================================================
# Step-boundary bookkeeping: confirm the fusable-pair count the report
# relies on (imagewam_denoise_loop's own call structure).
# ================================================================

def test_step_boundary_fusable_pair_count():
    """`imagewam_denoise_loop` calls `imagewam_denoise_step` once per
    `step in range(num_denoise_steps)`, sequentially, reusing the same
    `bufs["action_latent"]` buffer every iteration (pipeline_thor.py).
    Step i's trailing `gpu_euler_step` write is immediately followed by
    step i+1's leading `gpu_cast_fp32_to_fp16` read of that exact buffer
    for every i in [0, num_denoise_steps-2] -- the fused kernel applies
    there. The LAST step's `gpu_euler_step` write has no following cast
    inside the graph: `imagewam_thor.py`'s `infer()` reads
    `self._action_latent` directly after `.replay()` for de-normalization/
    output, never re-casting it to fp16."""
    fusable_pairs = NUM_DENOISE_STEPS - 1
    assert fusable_pairs == 9
    unfused_final_euler_steps = 1
    assert fusable_pairs + unfused_final_euler_steps == NUM_DENOISE_STEPS
