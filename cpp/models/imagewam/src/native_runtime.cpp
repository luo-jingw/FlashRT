#include "native_runtime.h"

#include "flashrt/cpp/models/imagewam/c_api.h"
#include "flashrt/model_runtime.h"
#include "flashrt/runtime.h"

#include "gemm_runner.h"
#include "io_transforms.h"
#include "native_pipeline.h"
#include "native_schema.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <exception>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace imagewam {

namespace {

constexpr int kOk = 0;
constexpr int kInvalid = -1;
constexpr int kNotFound = -2;
constexpr int kUnsupported = -3;
constexpr int kShape = -4;
constexpr int kStorage = -5;
constexpr int kBackend = -6;

const char* kBufferNames[] = {"img_raw", "context", "action_latent"};

constexpr int GemmRunnerAlgoBytes() { return GemmRunner::kAlgoBytes; }

bool copy_field(const float* scale, const float* offset, uint32_t n, const char* what,
                AffineField* field, std::string* error) {
    if (!scale && !offset) return true;
    if (!scale || !offset) {
        *error = std::string(what) + " normalization needs both scale and offset";
        return false;
    }
    field->scale.assign(scale, scale + n);
    field->offset.assign(offset, offset + n);
    return true;
}

std::string cuda_message(const char* what, cudaError_t rc) {
    return std::string(what) + ": " + cudaGetErrorString(rc);
}

}  // namespace

std::unique_ptr<NativeRuntime> NativeRuntime::create(const frt_imagewam_io_config& c,
                                                     std::string* error) {
    if (c.struct_size < sizeof(frt_imagewam_io_config)) {
        *error = "frt_imagewam_io_config.struct_size is smaller than this library's config";
        return nullptr;
    }
    if (!c.img_len || !c.token_dim || !c.num_action || !c.action_dim || !c.context_rows ||
        !c.context_width) {
        *error = "frt_imagewam_io_config has a zero dimension";
        return nullptr;
    }
    if (!c.img_raw || !c.context || !c.action_latent) {
        *error = "frt_imagewam_io_config has a null device window";
        return nullptr;
    }
    if (c.proprio_dim && (!c.proprio_weight_t || !c.proprio_bias)) {
        *error = "proprio_dim > 0 needs proprio_weight_t and proprio_bias";
        return nullptr;
    }
    std::unique_ptr<NativeRuntime> rt(new NativeRuntime());
    rt->dims_ = {c.img_len, c.token_dim, c.num_action, c.action_dim, c.proprio_dim};
    rt->context_rows_ = c.context_rows;
    rt->context_width_ = c.context_width;
    rt->img_raw_ = c.img_raw;
    rt->context_ = c.context;
    rt->action_latent_ = c.action_latent;
    if (!copy_field(c.state_scale, c.state_offset, c.proprio_dim, "state", &rt->state_norm_, error) ||
        !copy_field(c.action_scale, c.action_offset, c.action_dim, "action", &rt->action_norm_,
                    error)) {
        return nullptr;
    }
    cudaError_t rc = cudaStreamCreateWithFlags(&rt->stream_, cudaStreamNonBlocking);
    if (rc != cudaSuccess) {
        rt->stream_ = nullptr;
        *error = cuda_message("cudaStreamCreateWithFlags", rc);
        return nullptr;
    }
    if (c.proprio_dim) {
        rc = cudaMalloc(&rt->proprio_device_, size_t(c.proprio_dim) * 2);
        if (rc != cudaSuccess) {
            rt->proprio_device_ = nullptr;
            *error = cuda_message("proprio scratch cudaMalloc", rc);
            return nullptr;
        }
        if (!rt->projection_.init(c.proprio_weight_t, c.proprio_bias, int(c.proprio_dim),
                                  int(c.context_width), error)) {
            return nullptr;
        }
        rt->proprio_normalized_.resize(c.proprio_dim);
        rt->proprio_bf16_.resize(c.proprio_dim);
    }
    rt->schema_ = build_native_schema(rt->dims_);
    return rt;
}

NativeRuntime::~NativeRuntime() {
    if (stream_) cudaStreamSynchronize(stream_);
    if (owned_graph_) cudaGraphExecDestroy(owned_graph_);
    pipeline_.reset();
    if (proprio_device_) cudaFree(proprio_device_);
    if (stream_) cudaStreamDestroy(stream_);
}

int NativeRuntime::fail(int status, const std::string& message) {
    last_error_ = message;
    return status;
}

int NativeRuntime::use_graph(cudaGraphExec_t graph_exec) {
    if (!graph_exec) return fail(kInvalid, "use_graph: null graph exec");
    if (owned_graph_) {
        cudaStreamSynchronize(stream_);
        cudaGraphExecDestroy(owned_graph_);
        owned_graph_ = nullptr;
        graph_nodes_ = 0;
    }
    graph_ = graph_exec;
    return kOk;
}

int NativeRuntime::set_pipeline(const frt_imagewam_pipeline_config& config) {
    if (config.action_latent != action_latent_ || config.img_raw != img_raw_ ||
        config.context != context_) {
        return fail(kInvalid, "set_pipeline: IO windows differ from the runtime's");
    }
    if (config.num_action != int32_t(dims_.num_action) ||
        config.action_dim != int32_t(dims_.action_dim) ||
        config.x0 != int32_t(context_rows_) ||
        config.joint_attention_dim != int32_t(context_width_) ||
        config.a0 - config.x0 != int32_t(dims_.img_len) ||
        config.head_dim != int32_t(dims_.token_dim)) {
        return fail(kShape, "set_pipeline: dimensions differ from the runtime's IO config");
    }
    std::string error;
    std::unique_ptr<NativePipeline> pipeline = NativePipeline::create(config, &error);
    if (!pipeline) return fail(kInvalid, "set_pipeline: " + error);
    pipeline_ = std::move(pipeline);
    return kOk;
}

int NativeRuntime::gemm_shapes(frt_imagewam_gemm_shape* out, uint64_t capacity,
                               uint64_t* count) const {
    if (!pipeline_) return kInvalid;
    const auto& shapes = pipeline_->gemm_shapes();
    if (count) *count = shapes.size();
    if (capacity < shapes.size()) return kStorage;
    for (size_t i = 0; i < shapes.size(); ++i) out[i] = shapes[i];
    return kOk;
}

int NativeRuntime::set_gemm_algo(const frt_imagewam_gemm_shape& shape, const void* algo,
                                 uint64_t bytes) {
    if (!pipeline_) return fail(kInvalid, "set_gemm_algo: set_pipeline first");
    if (!algo || bytes != uint64_t(GemmRunnerAlgoBytes())) {
        return fail(kShape, "set_gemm_algo: algorithm must be " +
                                std::to_string(GemmRunnerAlgoBytes()) + " bytes");
    }
    try {
        pipeline_->set_gemm_algo(shape, algo);
    } catch (const std::exception& e) {
        return fail(kBackend, std::string("set_gemm_algo: ") + e.what());
    }
    return kOk;
}

int NativeRuntime::run(uint32_t segment, int32_t index) {
    if (!pipeline_) return fail(kInvalid, "run: set_pipeline first");
    try {
        switch (segment) {
            case FRT_IMAGEWAM_SEGMENT_DOUBLE_LAYER: pipeline_->double_layer(index, stream_); break;
            case FRT_IMAGEWAM_SEGMENT_SINGLE_LAYER: pipeline_->single_layer(index, stream_); break;
            case FRT_IMAGEWAM_SEGMENT_PREFILL: pipeline_->prefill(stream_); break;
            case FRT_IMAGEWAM_SEGMENT_DENOISE_STEP: pipeline_->denoise_step(index, stream_); break;
            case FRT_IMAGEWAM_SEGMENT_DENOISE: pipeline_->denoise(stream_); break;
            case FRT_IMAGEWAM_SEGMENT_FULL:
                pipeline_->prefill(stream_);
                pipeline_->denoise(stream_);
                break;
            default: return fail(kInvalid, "run: unknown segment");
        }
    } catch (const std::out_of_range&) {
        return fail(kNotFound, "run: layer or step index out of range");
    } catch (const std::exception& e) {
        return fail(kBackend, std::string("run: ") + e.what());
    }
    const cudaError_t rc = cudaStreamSynchronize(stream_);
    if (rc != cudaSuccess) return fail(kBackend, cuda_message("run", rc));
    return kOk;
}

int NativeRuntime::capture() {
    if (!pipeline_) return fail(kInvalid, "capture: set_pipeline first");
    // Warm-up: lazy cuBLAS/cuBLASLt initialisation must not happen under capture.
    int rc = run(FRT_IMAGEWAM_SEGMENT_FULL, 0);
    if (rc != kOk) return rc;
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) return fail(kBackend, cuda_message("cudaStreamBeginCapture", err));
    std::string record_error;
    try {
        pipeline_->prefill(stream_);
        pipeline_->denoise(stream_);
    } catch (const std::exception& e) {
        record_error = e.what();
    }
    err = cudaStreamEndCapture(stream_, &graph);
    if (!record_error.empty() || err != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        return fail(kBackend, record_error.empty() ? cuda_message("cudaStreamEndCapture", err)
                                                   : "capture: " + record_error);
    }
    size_t nodes = 0;
    cudaGraphGetNodes(graph, nullptr, &nodes);
    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) return fail(kBackend, cuda_message("cudaGraphInstantiate", err));
    if (owned_graph_) cudaGraphExecDestroy(owned_graph_);
    owned_graph_ = exec;
    graph_ = exec;
    graph_nodes_ = nodes;
    return kOk;
}

int NativeRuntime::set_proprio_row(int32_t row) {
    if (!dims_.proprio_dim) return fail(kUnsupported, "set_proprio_row: no proprio port");
    if (row < 0 || uint32_t(row) >= context_rows_) {
        return fail(kInvalid, "set_proprio_row: row " + std::to_string(row) +
                                  " outside the context rows");
    }
    proprio_row_ = row;
    return kOk;
}

int NativeRuntime::schema_records(char* out, uint64_t capacity, uint64_t* written) const {
    const std::string records = render_schema_records(schema_);
    if (written) *written = records.size();
    if (capacity < records.size()) return kStorage;
    if (out) std::memcpy(out, records.data(), records.size());
    return kOk;
}

int NativeRuntime::bind_declaration(const frt_model_runtime_v1* m) {
    if (!m || m->abi_version != FRT_MODEL_RUNTIME_ABI_VERSION ||
        m->struct_size < FRT_MODEL_RUNTIME_V1_BASE_SIZE || !m->exp) {
        return fail(kInvalid, "bind_declaration: not a v1 model runtime");
    }
    const frt_runtime_export_v1* exp = m->exp;
    if (exp->n_streams < 1 || exp->streams[0].native_handle != static_cast<void*>(stream_)) {
        return fail(kInvalid, "bind_declaration: stream 0 is not this runtime's native stream");
    }
    if (m->n_stages != 1 || exp->n_graphs < 1) {
        return fail(kInvalid, "bind_declaration: expected one graph stage");
    }
    if (exp->n_buffers != 3) return fail(kInvalid, "bind_declaration: expected 3 buffers");
    for (uint64_t b = 0; b < 3; ++b) {
        if (std::strcmp(exp->buffers[b].name, kBufferNames[b]) != 0) {
            return fail(kInvalid, std::string("bind_declaration: buffer ") + std::to_string(b) +
                                      " is not " + kBufferNames[b]);
        }
    }
    if (m->n_ports != schema_.ports.size()) {
        return fail(kShape, "bind_declaration: port count differs from the native schema");
    }
    std::vector<int> roles(m->n_ports);
    for (uint64_t i = 0; i < m->n_ports; ++i) {
        const frt_runtime_port_desc& d = m->ports[i];
        const NativePortSpec& s = schema_.ports[i];
        bool same = s.name == d.name && s.modality == d.modality && s.dtype == d.dtype &&
                    s.layout == d.layout && s.direction == d.direction && s.update == d.update &&
                    s.required == d.required && s.shape.size() == d.rank && s.bytes == d.bytes &&
                    d.offset == 0;
        for (uint32_t k = 0; same && k < d.rank; ++k) same = s.shape[k] == d.shape[k];
        if (same && s.buffer == NativeBuffer::kNone) same = d.buffer == nullptr;
        if (same && s.buffer != NativeBuffer::kNone) {
            const frt_runtime_buffer_desc& buf = exp->buffers[static_cast<int64_t>(s.buffer)];
            same = d.buffer == buf.handle && buf.bytes == s.bytes;
        }
        if (!same) {
            return fail(kShape, "bind_declaration: port " + std::to_string(i) + " (" + d.name +
                                    ") differs from the native schema");
        }
        roles[i] = static_cast<int>(s.role);
    }
    port_role_ = roles;
    export_stream_id_ = exp->streams[0].stream_id;
    bound_ = true;
    return kOk;
}

int NativeRuntime::check_stream(int stream) {
    if (stream == -1 || stream == export_stream_id_) return kOk;
    return fail(kInvalid, "stream " + std::to_string(stream) + " is not an exported stream");
}

int NativeRuntime::set_input(uint32_t port, const void* data, uint64_t bytes, int stream) {
    if (!bound_) return fail(kInvalid, "set_input: bind_declaration has not succeeded");
    if (port >= port_role_.size()) return fail(kNotFound, "set_input: unknown port index");
    if (check_stream(stream) != kOk) return kInvalid;
    if (static_cast<NativePort>(port_role_[port]) != NativePort::kProprio) {
        return fail(kUnsupported, "set_input: SWAP port, write its buffer window");
    }
    return stage_proprio(data, bytes);
}

int NativeRuntime::stage_proprio(const void* data, uint64_t bytes) {
    const uint32_t n = dims_.proprio_dim;
    if (!data || bytes != uint64_t(n) * 4) {
        return fail(kShape, "set_input(proprio): payload must be " + std::to_string(n * 4) +
                                " bytes of f32");
    }
    if (proprio_row_ < 0) return fail(kInvalid, "set_input(proprio): set_proprio_row first");
    normalize_state(static_cast<const float*>(data), state_norm_, proprio_normalized_.data(), n);
    for (uint32_t i = 0; i < n; ++i) proprio_bf16_[i] = float_to_bf16_rne(proprio_normalized_[i]);
    const cudaError_t rc = cudaMemcpyAsync(proprio_device_, proprio_bf16_.data(), size_t(n) * 2,
                                           cudaMemcpyHostToDevice, stream_);
    if (rc != cudaSuccess) return fail(kBackend, cuda_message("proprio H2D", rc));
    void* row = static_cast<char*>(context_) +
                size_t(proprio_row_) * size_t(context_width_) * 2;
    std::string error;
    if (!projection_.run(proprio_device_, row, stream_, &error)) return fail(kBackend, error);
    return kOk;
}

int NativeRuntime::get_output(uint32_t port, void* out, uint64_t capacity, uint64_t* written,
                              int stream) {
    if (!bound_) return fail(kInvalid, "get_output: bind_declaration has not succeeded");
    if (port >= port_role_.size()) return fail(kNotFound, "get_output: unknown port index");
    if (check_stream(stream) != kOk) return kInvalid;
    if (static_cast<NativePort>(port_role_[port]) != NativePort::kActions) {
        return fail(kUnsupported, "get_output: SWAP port, read its buffer window");
    }
    return read_actions(out, capacity, written);
}

int NativeRuntime::read_actions(void* out, uint64_t capacity, uint64_t* written) {
    const uint64_t bytes = uint64_t(dims_.num_action) * dims_.action_dim * 4;
    if (written) *written = bytes;
    if (capacity < bytes) return fail(kStorage, "get_output(actions): buffer too small");
    if (!out) return fail(kInvalid, "get_output(actions): null output");
    cudaError_t rc = cudaMemcpyAsync(out, action_latent_, bytes, cudaMemcpyDeviceToHost, stream_);
    if (rc == cudaSuccess) rc = cudaStreamSynchronize(stream_);
    if (rc != cudaSuccess) return fail(kBackend, cuda_message("actions D2H", rc));
    denormalize_actions_inplace(static_cast<float*>(out), action_norm_, dims_.num_action,
                                dims_.action_dim);
    return kOk;
}

int NativeRuntime::step() {
    if (!graph_) return fail(kInvalid, "step: no graph (use_graph or capture first)");
    const cudaError_t rc = cudaGraphLaunch(graph_, stream_);
    if (rc != cudaSuccess) return fail(kBackend, cuda_message("cudaGraphLaunch", rc));
    return kOk;
}

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt
