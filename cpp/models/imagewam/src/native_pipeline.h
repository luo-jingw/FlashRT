// NativePipeline: ImageWAM's backbone prefill and ActionDiT denoise loop,
// recorded in C++ against the existing csrc kernels in the exact order of
// flash_rt/models/imagewam/pipeline_thor.py (merged single-stream linear1,
// per-head K/V, real mot mask, cuBLAS-decomposed attention). Every device
// pointer comes from a borrowed frt_imagewam_pipeline_config; the pipeline
// owns only its GemmRunner (with the algorithms handed off by the setup
// producer) and its cuBLAS handle. Methods enqueue on `stream` and are
// safe inside stream capture once a warm-up run has happened.
#ifndef FLASHRT_CPP_MODELS_IMAGEWAM_NATIVE_PIPELINE_H
#define FLASHRT_CPP_MODELS_IMAGEWAM_NATIVE_PIPELINE_H

#include "flashrt/cpp/models/imagewam/c_api.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

class GemmRunner;

namespace flashrt {
namespace models {
namespace imagewam {

class NativePipeline {
public:
    ~NativePipeline();
    NativePipeline(const NativePipeline&) = delete;
    NativePipeline& operator=(const NativePipeline&) = delete;

    // Validates and copies the config tables; null with `error` on failure.
    static std::unique_ptr<NativePipeline> create(const frt_imagewam_pipeline_config& config,
                                                  std::string* error);

    const std::vector<frt_imagewam_gemm_shape>& gemm_shapes() const { return gemm_shapes_; }
    void set_gemm_algo(const frt_imagewam_gemm_shape& shape, const void* algo);

    void double_layer(int index, cudaStream_t stream);
    void single_layer(int index, cudaStream_t stream);
    void prefill(cudaStream_t stream);
    void denoise_step(int step, cudaStream_t stream);
    void denoise(cudaStream_t stream);

    const frt_imagewam_pipeline_config& config() const { return c_; }

private:
    NativePipeline() = default;
    void collect_gemm_shapes();
    void add_gemm_shape(const frt_imagewam_linear& linear, int m);
    void linear(const frt_imagewam_linear& l, const void* x, void* out, int m,
                cudaStream_t stream);
    void mlp_gate_up(const frt_imagewam_linear& l, const void* x, void* merged, void* gated,
                     int m, int mlp_hidden, cudaStream_t stream);
    void copy_qkv(const void* merged, int merged_width, int rows, int width, void* q, void* k,
                  void* v, cudaStream_t stream);
    void attention(void* q_out, int layer, int q_rows, int kv_rows, cudaStream_t stream);
    void action_double_layer(int index, const frt_imagewam_action_step& mods,
                             cudaStream_t stream);
    void action_single_layer(int index, const frt_imagewam_action_step& mods,
                             cudaStream_t stream);
    void* k_layer(int layer) const;
    void* v_layer(int layer) const;

    frt_imagewam_pipeline_config c_{};
    std::vector<frt_imagewam_double_layer> double_layers_;
    std::vector<frt_imagewam_single_layer> single_layers_;
    std::vector<frt_imagewam_action_double_layer> action_double_layers_;
    std::vector<frt_imagewam_single_layer> action_single_layers_;
    std::vector<frt_imagewam_action_step> steps_;
    std::vector<frt_imagewam_gemm_shape> gemm_shapes_;
    std::unique_ptr<GemmRunner> gemm_;
    cublasHandle_t cublas_ = nullptr;
};

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_IMAGEWAM_NATIVE_PIPELINE_H
