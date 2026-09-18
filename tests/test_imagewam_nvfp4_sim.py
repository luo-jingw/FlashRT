"""`precision="nvfp4_sim"`: its quantizer (`blockscaled_ref`'s E2M1
path, which `SimNvfp4Linear` runs) against the real NVFP4 quantizer, bit
for bit, and `SimNvfp4Linear` in the served pipeline.

Reference kernel: `quantize_fp4_dynamic_sfa_fp16`
(`csrc/quantize/quantize_fp4_sfa.cu`), the quantizer `Nvfp4Linear` runs
for its weights (SFB layout) and activations (SFA layout). It comes from
`flash_rt.flash_rt_fp4` on a Blackwell/Thor build. Elsewhere this test
compiles the unmodified source for the local GPU with the production
flags of the `fp4_kernels_obj` target (`-O3 --use_fast_math
--expt-relaxed-constexpr`, `CMakeLists.txt`), together with the C shim
`tools/blockscaled_quantizers_shim.cu` and the sources it links, as
`tools/check_blockscaled_quantizers_sm90.py` does; the quantizer is
CUDA-core code with no Blackwell instruction. Packed codes and the
tile-interleaved scale bytes are compared with
`blockscaled_ref.pack_codes` / `pack_scales`.

Also checks `SimNvfp4Linear` against the Thor-measured `Nvfp4Linear`
cosine on `test_imagewam_quant_linear.py`'s own small case.
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile

import pytest
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.blockscaled_ref import (
    dequantize_blocks,
    pack_codes,
    pack_scales,
    quantize_blocks,
    sf_size_bytes,
)
from flash_rt.models.imagewam.quant_linear import Fp16Linear, SimNvfp4Linear

DEV = "cuda"
FP16 = torch.float16
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The shim and the sources it links (tools/check_blockscaled_quantizers_sm90.py).
_SHIM_SOURCES = ("csrc/quantize/quantize_e0m3_sfa.cu", "csrc/quantize/quantize_fp4_sfa.cu",
                 "csrc/fused_fp4/pi05_e0m3_act.cu", "tools/blockscaled_quantizers_shim.cu")
# fp4_kernels_obj's CUDA compile options (CMakeLists.txt) and its define.
_PRODUCTION_FLAGS = ("-O3", "--use_fast_math", "--expt-relaxed-constexpr",
                     "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1")


def _build_shim() -> str:
    """nvcc-build the shim for the local GPU; cached by source contents."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    cutlass = os.path.join(_REPO, "third_party", "cutlass")
    h = hashlib.sha256(" ".join(_PRODUCTION_FLAGS + (arch,)).encode())
    for src in _SHIM_SOURCES + ("csrc/quantize/quantize_fp4_sfa.cuh",):
        with open(os.path.join(_REPO, src), "rb") as f:
            h.update(f.read())
    lib = os.path.join(tempfile.gettempdir(), f"imagewam_nvfp4_sfa_ref_{h.hexdigest()[:16]}.so")
    if not os.path.exists(lib):
        cmd = [os.environ.get("NVCC", "nvcc"), "-std=c++17", *_PRODUCTION_FLAGS,
               f"-gencode=arch=compute_{arch},code=sm_{arch}", "-Xcompiler", "-fPIC", "-shared",
               f"-I{_REPO}/csrc", f"-I{cutlass}/include", f"-I{cutlass}/tools/util/include",
               *[os.path.join(_REPO, s) for s in _SHIM_SOURCES], "-o", lib + ".tmp"]
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(lib + ".tmp", lib)
    return lib


def _real_quantizer():
    """Returns `fn(src_ptr, packed_ptr, sf_ptr, N, D, is_sfb) -> rc` and its origin."""
    try:
        import flash_rt.flash_rt_fp4 as fp4
        return ((lambda s, p, f, n, d, sfb: fp4.quantize_fp4_dynamic_sfa_fp16(s, p, f, n, d, sfb, 0)),
                "flash_rt_fp4")
    except ImportError:
        pass
    shim = ctypes.CDLL(_build_shim())
    shim.shim_nvfp4.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 3
    major, minor = torch.cuda.get_device_capability()
    return ((lambda s, p, f, n, d, sfb: shim.shim_nvfp4(s, p, f, n, d, int(sfb))),
            f"quantize_fp4_sfa.cu built for sm_{major}{minor} with -O3 --use_fast_math")


def _adversarial_input(rows: int, k: int) -> torch.Tensor:
    """Blocks spanning every regime of the quantizer: normal-range E4M3
    scales, subnormal scales, scales that underflow to 0, saturation at
    448 (amax/6 > 448), all-zero blocks, and elements placed exactly on
    E2M1 rounding thresholds."""
    g = torch.Generator(device="cpu").manual_seed(0)
    nb = k // 16
    mag = torch.pow(10.0, torch.empty(rows, nb, 1).uniform_(-6, 4.7, generator=g))
    x = torch.randn(rows, nb, 16, generator=g) * mag
    x[0, :4] = 0.0
    # threshold ties: block amax 6 -> scale e4m3(1.0) = 1, elements on the midpoints
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
                         -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0])
    x[1, :4] = ties
    return x.reshape(rows, k).clamp(-65000, 65000).to(FP16)


def test_sim_quantizer_bit_exact_vs_real_kernel():
    try:
        quant, origin = _real_quantizer()
    except Exception as e:  # no nvcc / no CUTLASS checkout
        pytest.skip(f"no real NVFP4 quantizer available: {e}")
    torch.manual_seed(0)
    cases = [("adversarial", _adversarial_input(256, 3072))]
    cases += [("randn", torch.randn(905, 3072, device=DEV).to(FP16)),
              ("weight-like 0.02", (torch.randn(1024, 3072, device=DEV) * 0.02).to(FP16))]
    for name, x in cases:
        x = x.to(DEV).contiguous()
        n, d = x.shape
        q = quantize_blocks(x, "e2m1")
        for layout, is_sfb in (("SFA (activation)", False), ("SFB (weight)", True)):
            packed = torch.zeros(n, d // 2, dtype=torch.uint8, device=DEV)
            sf = torch.zeros(sf_size_bytes(n, d), dtype=torch.uint8, device=DEV)
            assert quant(x.data_ptr(), packed.data_ptr(), sf.data_ptr(), n, d, is_sfb) == 0
            torch.cuda.synchronize()
            code_mismatch = (pack_codes(q.codes) != packed).sum().item()
            scale_mismatch = (pack_scales(q.scale_bytes) != sf).sum().item()
            print(f"[{origin}] {name} {layout}: {n}x{d}, packed-byte mismatches={code_mismatch}, "
                  f"scale-byte mismatches={scale_mismatch}")
            assert code_mismatch == 0 and scale_mismatch == 0
        sub = (q.scales < 2 ** -6).float().mean().item()
        print(f"  subnormal-or-zero block scales: {sub:.1%}")
        # dequantized values (E2M1 value x UE4M3 scale) are exact in fp16
        val32 = dequantize_blocks(q)
        assert torch.equal(val32.half().float(), val32)


def test_sim_linear_reproduces_thor_nvfp4_cosine():
    """Same inputs as test_imagewam_quant_linear.py's NVFP4 case; the real
    Nvfp4Linear gave cosine 0.989133 against Fp16Linear on Thor."""
    torch.manual_seed(0)
    m, n, k = 8, 64, 96
    gemm = fvk.GemmRunner()
    w = (torch.randn(n, k, dtype=torch.float32, device=DEV) * 0.02).to(FP16).t().contiguous()
    x = torch.randn(m, k, dtype=FP16, device=DEV) * 0.1
    ref = torch.zeros(m, n, dtype=FP16, device=DEV)
    Fp16Linear(gemm, w.data_ptr(), n, k)(x.data_ptr(), ref.data_ptr(), m, 0)
    out = torch.zeros(m, n, dtype=FP16, device=DEV)
    SimNvfp4Linear(gemm, w.data_ptr(), n, k)(x.data_ptr(), out.data_ptr(), m, 0)
    torch.cuda.synchronize()
    a, b = out.float().flatten(), ref.float().flatten()
    cos = (a @ b / (a.norm() * b.norm())).item()
    print(f"SimNvfp4Linear vs Fp16Linear: cosine={cos:.6f} (Thor Nvfp4Linear: 0.989133)")
    assert abs(cos - 0.989133) < 5e-6


def test_nvfp4_sim_precision_graph_replay_equals_eager():
    """`precision="nvfp4_sim"` at toy dims: the captured graph (which
    records the simulator's torch ops) reproduces the eager run exactly."""
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
    torch.manual_seed(3)
    fe = ImageWAMTorchFrontendThor(precision="nvfp4_sim")
    fe.set_prompt()
    n_sim = sum(isinstance(v, SimNvfp4Linear) for v in fe.weights.values())
    noise = torch.randn(fe.dims["num_action"], fe.dims["action_dim"], device=DEV)
    fe.stage_inputs({}, noise=noise)
    img_raw = fe._img_raw.clone()
    fe._graph.replay()
    torch.cuda.synchronize()
    replay = fe._action_latent.clone()
    fe.stage_inputs({}, noise=noise)
    fe._img_raw.copy_(img_raw)
    fe.run_eager()
    torch.cuda.synchronize()
    print(f"{n_sim} SimNvfp4Linear weights; replay == eager: {torch.equal(replay, fe._action_latent)}; "
          f"finite: {bool(torch.isfinite(replay).all())}")
    assert n_sim > 0 and torch.isfinite(replay).all()
    assert torch.equal(replay, fe._action_latent)
