// NativeRuntime: the hot-path half of ImageWAM's `io="native"` model
// runtime. Implements set_input(proprio), get_output(actions) and step on
// its own CUDA stream against borrowed device windows; see
// include/flashrt/cpp/models/imagewam/c_api.h for the ownership contract.
// Not thread-safe: calls on one instance must not overlap (threading
// contract in c_api.h); only the declaration count is atomic.
//
// One graph per text length: the handle holds a graph variant table keyed
// by the context length x0 (the same key space the exec contract's ShapeKey
// uses for this model), the lengths the io config declared. `context_rows_`
// is the ACTIVE key: `step` replays that key's exec and `set_proprio_row` /
// `set_pipeline` are validated against it.
#ifndef FLASHRT_CPP_MODELS_IMAGEWAM_NATIVE_RUNTIME_H
#define FLASHRT_CPP_MODELS_IMAGEWAM_NATIVE_RUNTIME_H

#include "flashrt/cpp/models/imagewam/c_api.h"
#include "flashrt/model_runtime.h"

#include "io_transforms.h"
#include "native_pipeline.h"
#include "native_schema.h"
#include "proprio_projection.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace imagewam {

class NativeRuntime {
public:
    // Returns null with `error` set when the config is invalid or a CUDA
    // resource cannot be created.
    static std::unique_ptr<NativeRuntime> create(const frt_imagewam_io_config& config,
                                                 std::string* error);
    ~NativeRuntime();
    NativeRuntime(const NativeRuntime&) = delete;
    NativeRuntime& operator=(const NativeRuntime&) = delete;

    const char* last_error() const { return last_error_.c_str(); }
    cudaStream_t stream() const { return stream_; }

    // The graph variant table. `use_graph` adopts the producer's exec under
    // a declared length; `capture` adopts the one this handle recorded under
    // the active length. `graph_exec` / `graph_nodes` describe the active
    // length's variant, which is what `step` replays.
    int use_graph(uint64_t key, cudaGraphExec_t graph_exec);
    int has_variant(uint64_t key) const;
    cudaGraphExec_t variant_exec(uint64_t key) const;
    int set_text_length(uint64_t key);
    uint64_t text_length() const { return context_rows_; }
    cudaGraphExec_t graph_exec() const { return variant_exec(context_rows_); }

    int set_proprio_row(int32_t row);
    int schema_records(char* out, uint64_t capacity, uint64_t* written) const;
    int bind_declaration(const frt_model_runtime_v1* declaration);

    // Native pipeline (setup): install, hand off GEMM algorithms, run eager
    // segments for parity checks, capture the graph `step` replays.
    int set_pipeline(const frt_imagewam_pipeline_config& config);
    int gemm_shapes(frt_imagewam_gemm_shape* out, uint64_t capacity, uint64_t* count) const;
    int set_gemm_algo(const frt_imagewam_gemm_shape& shape, const void* algo, uint64_t bytes);
    int run(uint32_t segment, int32_t index);
    int capture();
    uint64_t graph_nodes() const {
        const int at = variant_index(context_rows_);
        return at < 0 ? 0 : variants_[at].nodes;
    }

    // Model runtimes built over this handle's verbs (owner callbacks of
    // frt_model_runtime_override_verbs). While one is live, the graph it
    // adopted and the pipeline behind it cannot be replaced: set_pipeline,
    // capture and use_graph fail.
    void declaration_retained() { declarations_.fetch_add(1, std::memory_order_acq_rel); }
    void declaration_released() { declarations_.fetch_sub(1, std::memory_order_acq_rel); }

    int set_input(uint32_t port, const void* data, uint64_t bytes, int stream);
    int get_output(uint32_t port, void* out, uint64_t capacity, uint64_t* written, int stream);
    int step();

private:
    NativeRuntime() = default;

    // One graph exec of the variant table: the text length (x0) it was
    // captured for, the exec, whether this handle recorded it, and the node
    // count `capture` measured (0 for an adopted exec).
    struct GraphVariant {
        uint64_t key;
        cudaGraphExec_t exec;
        bool owned;
        uint64_t nodes;
    };

    int fail(int status, const std::string& message);
    int refuse_while_exported(const char* what);
    void drop_owned_graphs();
    int variant_index(uint64_t key) const;      // -1 when the table has no such key
    bool is_declared(uint64_t key) const;
    std::string declared_keys() const;          // "[6, 14]", for messages
    std::string adopted_keys() const;           // "[6, 14]" / "none", for messages
    int check_stream(int stream);
    int stage_proprio(const void* data, uint64_t bytes);
    int read_actions(void* out, uint64_t capacity, uint64_t* written);

    NativeDims dims_{};
    uint32_t context_rows_ = 0;                // active text length (x0) = the active key
    uint32_t context_width_ = 0;
    std::vector<uint32_t> text_lengths_;        // declared lengths; {context_rows} when none declared
    void* img_raw_ = nullptr;
    void* context_ = nullptr;
    void* action_latent_ = nullptr;
    AffineField state_norm_;
    AffineField action_norm_;
    ProprioProjection projection_;
    NativeSchema schema_;

    cudaStream_t stream_ = nullptr;
    std::vector<GraphVariant> variants_;        // one entry per length the handle holds a graph for
    uint32_t pipeline_x0_ = 0;                  // x0 of the installed pipeline (0 = none)
    std::unique_ptr<NativePipeline> pipeline_;
    int32_t proprio_row_ = -1;
    void* proprio_device_ = nullptr;           // bf16 (1, proprio_dim) staging scratch
    std::vector<float> proprio_normalized_;
    std::vector<uint16_t> proprio_bf16_;

    std::atomic<int> declarations_{0};
    bool bound_ = false;
    int export_stream_id_ = -1;
    std::vector<int> port_role_;               // declaration port index -> NativePort (int)
    std::string last_error_;
};

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_IMAGEWAM_NATIVE_RUNTIME_H
