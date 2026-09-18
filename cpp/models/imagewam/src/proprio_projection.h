// The proprio_encoder linear of ImageWAM, run by a native verb:
// out(1, N) bf16 = x(1, K) bf16 @ weight_t(K, N) bf16 + bias(N) bf16,
// one cuBLASLt matmul with a bias epilogue (fp32 accumulate, one rounding).
// Descriptors, algorithm and workspace are created once; `run` allocates
// nothing and is safe inside stream capture.
#ifndef FLASHRT_CPP_MODELS_IMAGEWAM_PROPRIO_PROJECTION_H
#define FLASHRT_CPP_MODELS_IMAGEWAM_PROPRIO_PROJECTION_H

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <string>

namespace flashrt {
namespace models {
namespace imagewam {

class ProprioProjection {
public:
    ProprioProjection() = default;
    ~ProprioProjection();
    ProprioProjection(const ProprioProjection&) = delete;
    ProprioProjection& operator=(const ProprioProjection&) = delete;

    // `weight_t` (k, n) and `bias` (n) are borrowed device pointers.
    // Returns false with `error` set on failure.
    bool init(const void* weight_t, const void* bias, int k, int n, std::string* error);

    // x: device bf16 (1, k); out: device bf16 (1, n).
    bool run(const void* x, void* out, cudaStream_t stream, std::string* error) const;

private:
    void release();

    cublasLtHandle_t handle_ = nullptr;
    cublasLtMatmulDesc_t matmul_ = nullptr;
    cublasLtMatrixLayout_t a_ = nullptr;
    cublasLtMatrixLayout_t b_ = nullptr;
    cublasLtMatrixLayout_t d_ = nullptr;
    cublasLtMatmulAlgo_t algo_{};
    void* workspace_ = nullptr;
    size_t workspace_bytes_ = 0;
    const void* weight_t_ = nullptr;
    int k_ = 0;
    int n_ = 0;
};

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_IMAGEWAM_PROPRIO_PROJECTION_H
