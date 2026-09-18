"""`nvfp4_sim.py` against the real NVFP4 quantizer, bit for bit.

Reference kernel: `quantize_fp4_dynamic_fp16` (`csrc/quantize/
quantize_fp4_dynamic.cu`), the linear-scale-layout twin of the
`quantize_fp4_sfa.cu` quantizer `Nvfp4Linear` runs (same device helpers
and arithmetic; only the scale byte's address differs). It comes from
`flash_rt.flash_rt_fp4` on a Blackwell/Thor build; elsewhere this test
JIT-compiles that same `.cu` file for the local GPU (pure CUDA-core code,
no Blackwell instruction), so the simulator is checked against the real
device code on any GPU with nvcc available.

Also checks `SimNvfp4Linear` against the Thor-measured `Nvfp4Linear`
cosine on `test_imagewam_quant_linear.py`'s own small case.
"""
import os
import tempfile

import pytest
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.nvfp4_sim import SimNvfp4Linear, dequantize_nvfp4, quantize_nvfp4
from flash_rt.models.imagewam.quant_linear import Fp16Linear

DEV = "cuda"
FP16 = torch.float16
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BINDING = r"""
#include <torch/extension.h>
#include "quantize_fp4_dynamic.cuh"
int quantize(int64_t src, int64_t packed, int64_t scales, int n, int d) {
  return flash_rt::fp4::quantize_fp4_dynamic_fp16(
      reinterpret_cast<const void*>(src), reinterpret_cast<void*>(packed),
      reinterpret_cast<void*>(scales), n, d, 0);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("quantize_fp4_dynamic_fp16", &quantize); }
"""


def _real_quantizer():
    """Returns `fn(src_ptr, packed_ptr, scales_ptr, N, D) -> rc` and its origin."""
    try:
        import flash_rt.flash_rt_fp4 as fp4
        return (lambda s, p, c, n, d: fp4.quantize_fp4_dynamic_fp16(s, p, c, n, d, 0)), "flash_rt_fp4"
    except ImportError:
        pass
    from torch.utils.cpp_extension import load
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    build_dir = os.path.join(tempfile.gettempdir(), f"imagewam_nvfp4_quant_ref_sm{major}{minor}")
    os.makedirs(build_dir, exist_ok=True)
    binding = os.path.join(build_dir, "binding.cpp")
    with open(binding, "w") as f:
        f.write(_BINDING)
    mod = load(name=f"imagewam_nvfp4_quant_ref_sm{major}{minor}",
               sources=[binding, os.path.join(_REPO, "csrc/quantize/quantize_fp4_dynamic.cu")],
               extra_include_paths=[os.path.join(_REPO, "csrc/quantize")],
               build_directory=build_dir, verbose=False)
    return mod.quantize_fp4_dynamic_fp16, f"JIT-compiled quantize_fp4_dynamic.cu (sm_{major}{minor})"


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
    except Exception as e:  # no nvcc / no ninja
        pytest.skip(f"no real NVFP4 quantizer available: {e}")
    torch.manual_seed(0)
    cases = [("adversarial", _adversarial_input(256, 3072))]
    cases += [("randn", torch.randn(905, 3072, device=DEV).to(FP16)),
              ("weight-like 0.02", (torch.randn(1024, 3072, device=DEV) * 0.02).to(FP16))]
    for name, x in cases:
        x = x.to(DEV).contiguous()
        n, d = x.shape
        packed = torch.zeros(n, d // 2, dtype=torch.uint8, device=DEV)
        scales = torch.zeros(n, d // 16, dtype=torch.uint8, device=DEV)
        assert quant(x.data_ptr(), packed.data_ptr(), scales.data_ptr(), n, d) == 0
        torch.cuda.synchronize()
        codes, sc = quantize_nvfp4(x)
        sim_packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        sim_scales = sc.view(torch.uint8)
        code_mismatch = (sim_packed != packed).sum().item()
        scale_mismatch = (sim_scales != scales).sum().item()
        sub = (sc.float() < 2 ** -6).float().mean().item()
        print(f"[{origin}] {name}: {n}x{d}, packed-byte mismatches={code_mismatch}, "
              f"scale mismatches={scale_mismatch}, subnormal-or-zero block scales={sub:.1%}")
        assert code_mismatch == 0 and scale_mismatch == 0
        # dequantized values (E2M1 value x E4M3 scale) are exact in fp16
        mag = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=DEV)[(codes & 7).long()]
        val32 = (mag.reshape(n, d // 16, 16) * sc.float().unsqueeze(-1)).reshape(n, d)
        assert torch.equal(val32.half().float(), val32)
        assert torch.equal(dequantize_nvfp4(codes, sc).abs().float(), val32)


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
