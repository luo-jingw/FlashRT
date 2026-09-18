#include "proprio_projection.h"

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <string>

namespace flashrt {
namespace models {
namespace imagewam {

namespace {

constexpr size_t kWorkspaceBytes = 4u << 20;

bool lt_ok(cublasStatus_t status, const char* what, std::string* error) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    if (error) *error = std::string("proprio projection: ") + what + " failed (cuBLAS status " +
                        std::to_string(static_cast<int>(status)) + ")";
    return false;
}

}  // namespace

ProprioProjection::~ProprioProjection() { release(); }

void ProprioProjection::release() {
    if (d_) cublasLtMatrixLayoutDestroy(d_);
    if (b_) cublasLtMatrixLayoutDestroy(b_);
    if (a_) cublasLtMatrixLayoutDestroy(a_);
    if (matmul_) cublasLtMatmulDescDestroy(matmul_);
    if (workspace_) cudaFree(workspace_);
    if (handle_) cublasLtDestroy(handle_);
    d_ = b_ = a_ = nullptr;
    matmul_ = nullptr;
    workspace_ = nullptr;
    handle_ = nullptr;
}

bool ProprioProjection::init(const void* weight_t, const void* bias, int k, int n,
                             std::string* error) {
    release();
    if (!weight_t || !bias || k <= 0 || n <= 0) {
        if (error) *error = "proprio projection: null weight/bias or empty shape";
        return false;
    }
    weight_t_ = weight_t;
    k_ = k;
    n_ = n;
    if (!lt_ok(cublasLtCreate(&handle_), "cublasLtCreate", error)) return false;
    if (cudaMalloc(&workspace_, kWorkspaceBytes) != cudaSuccess) {
        workspace_ = nullptr;
        if (error) *error = "proprio projection: workspace cudaMalloc failed";
        return false;
    }
    workspace_bytes_ = kWorkspaceBytes;

    const cublasOperation_t op_n = CUBLAS_OP_N;
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    const cudaDataType_t bias_type = CUDA_R_16BF;
    if (!lt_ok(cublasLtMatmulDescCreate(&matmul_, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc", error) ||
        !lt_ok(cublasLtMatmulDescSetAttribute(matmul_, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)),
               "transa", error) ||
        !lt_ok(cublasLtMatmulDescSetAttribute(matmul_, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)),
               "transb", error) ||
        !lt_ok(cublasLtMatmulDescSetAttribute(matmul_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                              sizeof(epilogue)),
               "epilogue", error) ||
        !lt_ok(cublasLtMatmulDescSetAttribute(matmul_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                              sizeof(bias)),
               "bias pointer", error) ||
        !lt_ok(cublasLtMatmulDescSetAttribute(matmul_, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                              sizeof(bias_type)),
               "bias dtype", error)) {
        return false;
    }
    // Column-major view: D (n x 1) = A (n x k) * B (k x 1) + bias (n).
    // A is weight_t's row-major (k, n) memory read as column-major (n, k),
    // ld = n; the BIAS epilogue adds one value per row of D.
    if (!lt_ok(cublasLtMatrixLayoutCreate(&a_, CUDA_R_16BF, n, k, n), "A layout", error) ||
        !lt_ok(cublasLtMatrixLayoutCreate(&b_, CUDA_R_16BF, k, 1, k), "B layout", error) ||
        !lt_ok(cublasLtMatrixLayoutCreate(&d_, CUDA_R_16BF, n, 1, n), "D layout", error)) {
        return false;
    }

    cublasLtMatmulPreference_t preference = nullptr;
    if (!lt_ok(cublasLtMatmulPreferenceCreate(&preference), "preference", error)) return false;
    bool ok = lt_ok(cublasLtMatmulPreferenceSetAttribute(
                        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_,
                        sizeof(workspace_bytes_)),
                    "preference workspace", error);
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    if (ok) {
        ok = lt_ok(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_, a_, b_, d_, d_, preference, 1,
                                                  &heuristic, &returned),
                   "heuristic", error);
    }
    cublasLtMatmulPreferenceDestroy(preference);
    if (!ok) return false;
    if (returned == 0) {
        if (error) *error = "proprio projection: cuBLASLt returned no algorithm";
        return false;
    }
    algo_ = heuristic.algo;
    return true;
}

bool ProprioProjection::run(const void* x, void* out, cudaStream_t stream,
                            std::string* error) const {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return lt_ok(cublasLtMatmul(handle_, matmul_, &alpha, weight_t_, a_, x, b_, &beta, out, d_,
                                out, d_, &algo_, workspace_, workspace_bytes_, stream),
                 "cublasLtMatmul", error);
}

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt
