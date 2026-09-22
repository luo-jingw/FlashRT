// ================================================================
// FlashRT — standalone proof: FP16 GEMM with a fused BIAS epilogue
// (cuBLASLt), investigating whether ImageWAM's `action_encoder`
// GEMM + `add_bias_fp16` pair (pipeline_thor.py's
// `imagewam_denoise_step`, item 2 of the step-boundary kernel
// investigation) can drop the separate bias-add launch.
//
// Precedent found in csrc/gemm/gemm_runner.cu: `GemmRunner::bf16_nn_bias`
// (BF16 A/B/D + BF16 bias, CUBLASLT_EPILOGUE_BIAS, CUBLAS_COMPUTE_32F)
// and `GemmRunner::fp8_nn_bias`/`fp8_nn_bias_bf16` (FP8 A/B). Neither
// matches `action_encoder`'s actual dtype family: `_wrap_linear` in
// flash_rt/frontends/torch/imagewam_thor.py always falls back to plain
// `Fp16Linear` (GemmRunner::fp16_nn, FP16 A/B/D, no bias epilogue) for
// this GEMM at every precision tier, because K=7/N=7 fails every
// backend's 8/16-alignment check. `GemmRunner::fp16_nn` itself has no
// bias-epilogue sibling. This file adapts `bf16_nn_bias`'s exact
// cuBLASLt call structure (same epilogue enum, same compute type) to
// CUDA_R_16F in place of CUDA_R_16BF -- the minimal change needed,
// not a new GEMM design -- to measure whether the fused epilogue is
// worth carrying into production. NOT wired into quant_linear.py /
// pipeline_thor.py (both off-limits here); this is a standalone
// proof + benchmark only.
//
// NOTE ON EXACTNESS: unlike the Euler+cast fusion (euler_step_cast_fused.cu,
// which is bit-exact by construction -- same math, same rounding, just
// one fewer memory round trip), this fusion changes ROUNDING ORDER:
//   reference : D_fp16 = fp16(fp32_accum(A@B));  out = fp16(fp16(D_fp16) + fp16(bias))   [double-rounded]
//   fused     : out = fp16(fp32_accum(A@B) + fp32(bias))                                  [single-rounded, in cuBLASLt's epilogue]
// so the fused result is expected to be close (and typically MORE
// accurate, one fewer rounding step) but not always bit-identical to
// the reference. Verified below via max-abs-diff / cosine, not
// `torch.equal`.
// ================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdexcept>
#include <string>

#define CUBLAS_CHECK(x)                                                          \
    do {                                                                         \
        cublasStatus_t st = (x);                                                 \
        if (st != CUBLAS_STATUS_SUCCESS) {                                       \
            throw std::runtime_error("cuBLASLt error " + std::to_string((int)st) \
                                      + " at " __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                         \
    } while (0)

// D(M,N) = A(M,K) @ B(K,N) + bias(N), all FP16, row-major, matching
// GemmRunner::fp16_nn's own (M,K)/(K,N)/(M,N) row-major convention
// (see _wrap_linear / `_rnd_linear`'s (k,n) weight-storage convention
// in imagewam_thor.py) -- adapted line-for-line from `bf16_nn_bias`.
static void fp16_nn_bias(cublasLtHandle_t handle, void* workspace, size_t workspace_size,
                          const void* A, const void* B, void* D, const void* bias,
                          int M, int N, int K, cudaStream_t stream) {
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t A_desc, B_desc, D_desc;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    cudaDataType_t bias_type = CUDA_R_16F;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_16F, M, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_16F, K, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&D_desc, CUDA_R_16F, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc,
        A_desc, B_desc, D_desc, D_desc, preference, 1, &heuristic, &returned_results));

    if (returned_results == 0) {
        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
        throw std::runtime_error("cuBLASLt fp16_nn_bias: no algorithm found");
    }

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle, matmul_desc,
        &alpha, A, A_desc, B, B_desc, &beta, D, D_desc, D, D_desc,
        &heuristic.algo, workspace, workspace_size, stream));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(D_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// ---- minimal persistent handle + workspace, kept small (shared GPU) ----
namespace {
cublasLtHandle_t g_handle = nullptr;
void* g_workspace = nullptr;
size_t g_workspace_size = 4 * 1024 * 1024;  // 4 MB -- plenty for this tiny GEMM

void ensure_handle() {
    if (g_handle == nullptr) {
        CUBLAS_CHECK(cublasLtCreate(&g_handle));
        cudaMalloc(&g_workspace, g_workspace_size);
    }
}
}  // namespace

// ---- Torch-tensor entry point for the standalone JIT test ----
void fp16_gemm_bias_torch(torch::Tensor A, torch::Tensor B, torch::Tensor D, torch::Tensor bias,
                           int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && D.is_cuda() && bias.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16 && B.scalar_type() == torch::kFloat16 &&
                D.scalar_type() == torch::kFloat16 && bias.scalar_type() == torch::kFloat16,
                "all tensors must be fp16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && D.is_contiguous() && bias.is_contiguous(),
                "all tensors must be contiguous");
    ensure_handle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fp16_nn_bias(g_handle, g_workspace, g_workspace_size,
                 A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), D.data_ptr<at::Half>(), bias.data_ptr<at::Half>(),
                 static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp16_gemm_bias", &fp16_gemm_bias_torch,
          "FP16 GEMM D=A@B+bias via cuBLASLt CUBLASLT_EPILOGUE_BIAS (standalone proof)");
}
