// ============================================================================
//  Standalone pybind11 module for the fused_norm_fp4 kernel + the REAL
//  project quantize_fp4_sfa.cu / reshape_scales_sfa.cu kernels (compiled
//  unmodified as additional JIT sources -- see
//  tests/test_fused_norm_fp4_kernel.py), so the "unfused two-step" reference
//  path in that test uses the project's actual quantize kernel, not a
//  reimplementation.
//
//  All pointer args are passed as uintptr_t (Tensor.data_ptr()), matching
//  the established fvk / flash_rt_fp4 convention.
// ============================================================================
#include <pybind11/pybind11.h>
#include <cstdint>

#include "kernels/fused_norm_fp4/fused_norm_fp4.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "quantize/reshape_scales_sfa.cuh"

namespace py = pybind11;

PYBIND11_MODULE(fused_norm_fp4_ext, m) {
  m.def("gate_res_ada_layer_norm_fp4_sfa_bf16res",
        [](uintptr_t residual, uintptr_t gemm_out, uintptr_t gate,
           uintptr_t scale, uintptr_t shift, uintptr_t inv_s,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, float eps, uintptr_t stream) -> int {
          return flash_rt::fused_norm_fp4::gate_res_ada_layer_norm_fp4_sfa_bf16res(
              reinterpret_cast<void*>(residual),
              reinterpret_cast<const void*>(gemm_out),
              reinterpret_cast<const void*>(gate),
              reinterpret_cast<const void*>(scale),
              reinterpret_cast<const void*>(shift),
              inv_s ? reinterpret_cast<const void*>(inv_s) : nullptr,
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              seq_len, dim, eps,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("gemm_out"), py::arg("gate"),
        py::arg("scale"), py::arg("shift"), py::arg("inv_s") = 0,
        py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("eps"), py::arg("stream") = 0,
        "Fused gate*residual(BF16) + AdaLayerNorm(no affine) + NVFP4 quantize + SFA.");

  m.def("gate_res_ada_layer_norm_fp4_sfa_fp16res",
        [](uintptr_t residual, uintptr_t gemm_out, uintptr_t gate,
           uintptr_t scale, uintptr_t shift, uintptr_t inv_s,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, float eps, uintptr_t stream) -> int {
          return flash_rt::fused_norm_fp4::gate_res_ada_layer_norm_fp4_sfa_fp16res(
              reinterpret_cast<void*>(residual),
              reinterpret_cast<const void*>(gemm_out),
              reinterpret_cast<const void*>(gate),
              reinterpret_cast<const void*>(scale),
              reinterpret_cast<const void*>(shift),
              inv_s ? reinterpret_cast<const void*>(inv_s) : nullptr,
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              seq_len, dim, eps,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("gemm_out"), py::arg("gate"),
        py::arg("scale"), py::arg("shift"), py::arg("inv_s") = 0,
        py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("eps"), py::arg("stream") = 0,
        "Fused gate*residual(FP16) + AdaLayerNorm(no affine) + NVFP4 quantize + SFA.");

  // ── Reference kernels, compiled from the REAL, unmodified project
  // sources (csrc/quantize/quantize_fp4_sfa.cu, reshape_scales_sfa.cu). ──
  m.def("quantize_fp4_dynamic_sfa_fp16",
        [](uintptr_t src, uintptr_t packed, uintptr_t sfa,
           int N, int D, bool is_sfb, uintptr_t stream) -> int {
          return flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(
              reinterpret_cast<const void*>(src),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              N, D, is_sfb,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("sfa"),
        py::arg("N"), py::arg("D"), py::arg("is_sfb"), py::arg("stream") = 0,
        "Reference: fp16 -> NVFP4 packed + CUTLASS tile-interleaved SFA/SFB "
        "(real csrc/quantize/quantize_fp4_sfa.cu, unmodified).");

  m.def("sfa_size_bytes", &flash_rt::fp4::sfa_size_bytes,
        py::arg("rows"), py::arg("D"), py::arg("is_sfb"),
        "Reference: byte size of the CUTLASS SFA/SFB buffer "
        "(real csrc/quantize/reshape_scales_sfa.cu, unmodified).");
}
