// Standalone pybind11 binding for the fused QKV split + RMSNorm(Q,K)
// + RoPE(Q,K) + V-copy kernel — JIT-compiled by
// tests/test_fused_qkv_norm_rope_kernel.py via
// torch.utils.cpp_extension.load. Not part of the production
// flash_rt_kernels extension (csrc/bindings.cpp is untouched).
//
// Same calling convention as the existing flash_rt.flash_rt_kernels
// ops this kernel fuses (uintptr_t raw device pointers, plain ints
// for shapes, uintptr_t for the CUDA stream) — see csrc/bindings.cpp's
// own `rms_norm_fp16`/`rope_apply_fp16_perhead` bindings.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "qkv_split_norm_rope_fp16.cuh"

namespace {

cudaStream_t to_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }

void qkv_split_norm_rope_fp16_py(
    uintptr_t qkv, uintptr_t q_norm_weight, uintptr_t k_norm_weight, uintptr_t rope_table,
    uintptr_t Q_out, uintptr_t K_out, uintptr_t V_out,
    int rows, int NH, int HD, int hidden,
    int src_row_stride,
    int q_col_offset, int k_col_offset, int v_col_offset,
    int dst_row_stride,
    float eps,
    uintptr_t stream) {
    qkv_split_norm_rope_fp16(
        reinterpret_cast<const __half*>(qkv),
        reinterpret_cast<const __half*>(q_norm_weight),
        reinterpret_cast<const __half*>(k_norm_weight),
        reinterpret_cast<const __half*>(rope_table),
        reinterpret_cast<__half*>(Q_out),
        reinterpret_cast<__half*>(K_out),
        reinterpret_cast<__half*>(V_out),
        rows, NH, HD, hidden,
        src_row_stride,
        q_col_offset, k_col_offset, v_col_offset,
        dst_row_stride,
        eps,
        to_stream(stream));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qkv_split_norm_rope_fp16", &qkv_split_norm_rope_fp16_py,
          "Fused QKV column-slice split + RMSNorm(Q,K) + RoPE(Q,K) + V-copy (fp16)",
          py::arg("qkv"), py::arg("q_norm_weight"), py::arg("k_norm_weight"), py::arg("rope_table"),
          py::arg("Q_out"), py::arg("K_out"), py::arg("V_out"),
          py::arg("rows"), py::arg("NH"), py::arg("HD"), py::arg("hidden"),
          py::arg("src_row_stride"),
          py::arg("q_col_offset"), py::arg("k_col_offset"), py::arg("v_col_offset"),
          py::arg("dst_row_stride"),
          py::arg("eps") = 1e-6f,
          py::arg("stream") = 0);
}
