// C ABI of libflashrt_imagewam_native.so: handle lifetime and the static
// verbs table. Every entry point catches exceptions; nothing unwinds into C.
#include "flashrt/cpp/models/imagewam/c_api.h"

#include "flashrt/model_runtime.h"

#include "native_runtime.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <exception>
#include <memory>
#include <string>

struct frt_imagewam_native {
    std::unique_ptr<flashrt::models::imagewam::NativeRuntime> runtime;
    std::atomic<int> refs{1};
};

namespace {

using flashrt::models::imagewam::NativeRuntime;

constexpr int kInvalid = -1;
constexpr int kUnsupported = -3;
constexpr int kBackend = -6;

thread_local std::string g_create_error;

NativeRuntime* runtime_of(void* self) {
    return self ? static_cast<frt_imagewam_native*>(self)->runtime.get() : nullptr;
}

int verb_set_input(void* self, uint32_t port, const void* data, uint64_t bytes, int stream) {
    NativeRuntime* rt = runtime_of(self);
    if (!rt) return kInvalid;
    try {
        return rt->set_input(port, data, bytes, stream);
    } catch (...) {
        return kBackend;
    }
}

int verb_get_output(void* self, uint32_t port, void* out, uint64_t capacity, uint64_t* written,
                    int stream) {
    NativeRuntime* rt = runtime_of(self);
    if (!rt) return kInvalid;
    try {
        return rt->get_output(port, out, capacity, written, stream);
    } catch (...) {
        return kBackend;
    }
}

int verb_prepare(void*, uint32_t, frt_shape_key) { return kUnsupported; }

int verb_step(void* self) {
    NativeRuntime* rt = runtime_of(self);
    if (!rt) return kInvalid;
    try {
        return rt->step();
    } catch (...) {
        return kBackend;
    }
}

const char* verb_last_error(void* self) {
    NativeRuntime* rt = runtime_of(self);
    return rt ? rt->last_error() : "null ImageWAM native handle";
}

const frt_model_runtime_verbs kVerbs = {
    sizeof(frt_model_runtime_verbs), 0u, &verb_set_input, &verb_get_output,
    &verb_prepare, &verb_step, &verb_last_error,
};

}  // namespace

extern "C" {

void frt_imagewam_native_abi_sizes(uint64_t out[4]) {
    if (!out) return;
    out[0] = sizeof(frt_imagewam_io_config);
    out[1] = sizeof(frt_imagewam_pipeline_config);
    out[2] = sizeof(frt_imagewam_linear);
    out[3] = sizeof(frt_imagewam_action_step);
}

int frt_imagewam_native_create(const frt_imagewam_io_config* config, frt_imagewam_native** out) {
    if (!out) return kInvalid;
    *out = nullptr;
    if (!config) {
        g_create_error = "null config";
        return kInvalid;
    }
    try {
        std::string error;
        std::unique_ptr<NativeRuntime> rt = NativeRuntime::create(*config, &error);
        if (!rt) {
            g_create_error = error;
            return kInvalid;
        }
        auto* h = new frt_imagewam_native();
        h->runtime = std::move(rt);
        *out = h;
        return 0;
    } catch (const std::exception& e) {
        g_create_error = e.what();
        return kBackend;
    } catch (...) {
        g_create_error = "frt_imagewam_native_create failed";
        return kBackend;
    }
}

void frt_imagewam_native_retain(void* h) {
    if (h) static_cast<frt_imagewam_native*>(h)->refs.fetch_add(1, std::memory_order_relaxed);
}

void frt_imagewam_native_release(void* h) {
    if (!h) return;
    auto* handle = static_cast<frt_imagewam_native*>(h);
    if (handle->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete handle;
}

void frt_imagewam_native_declaration_retain(void* h) {
    if (!h) return;
    frt_imagewam_native_retain(h);
    static_cast<frt_imagewam_native*>(h)->runtime->declaration_retained();
}

void frt_imagewam_native_declaration_release(void* h) {
    if (!h) return;
    static_cast<frt_imagewam_native*>(h)->runtime->declaration_released();
    frt_imagewam_native_release(h);
}

const char* frt_imagewam_native_last_error(const frt_imagewam_native* h) {
    if (!h) return g_create_error.c_str();
    return h->runtime->last_error();
}

void* frt_imagewam_native_stream(frt_imagewam_native* h) {
    return h ? static_cast<void*>(h->runtime->stream()) : nullptr;
}

int frt_imagewam_native_use_graph(frt_imagewam_native* h, uint64_t key, void* graph_exec) {
    if (!h) return kInvalid;
    return h->runtime->use_graph(key, static_cast<cudaGraphExec_t>(graph_exec));
}

int frt_imagewam_native_has_variant(frt_imagewam_native* h, uint64_t key) {
    return h ? h->runtime->has_variant(key) : 0;
}

void* frt_imagewam_native_variant_exec(frt_imagewam_native* h, uint64_t key) {
    return h ? static_cast<void*>(h->runtime->variant_exec(key)) : nullptr;
}

int frt_imagewam_native_set_text_length(frt_imagewam_native* h, uint64_t key) {
    if (!h) return kInvalid;
    return h->runtime->set_text_length(key);
}

uint64_t frt_imagewam_native_text_length(frt_imagewam_native* h) {
    return h ? h->runtime->text_length() : 0;
}

void* frt_imagewam_native_graph_exec(frt_imagewam_native* h) {
    return h ? static_cast<void*>(h->runtime->graph_exec()) : nullptr;
}

int frt_imagewam_native_set_proprio_row(frt_imagewam_native* h, int32_t row) {
    if (!h) return kInvalid;
    return h->runtime->set_proprio_row(row);
}

int frt_imagewam_native_schema_records(const frt_imagewam_native* h, char* out, uint64_t capacity,
                                       uint64_t* written) {
    if (!h) return kInvalid;
    try {
        return h->runtime->schema_records(out, capacity, written);
    } catch (...) {
        return kBackend;
    }
}

int frt_imagewam_native_bind_declaration(frt_imagewam_native* h,
                                         const frt_model_runtime_v1* declaration) {
    if (!h) return kInvalid;
    try {
        return h->runtime->bind_declaration(declaration);
    } catch (...) {
        return kBackend;
    }
}

const frt_model_runtime_verbs* frt_imagewam_native_verbs(void) { return &kVerbs; }

int frt_imagewam_native_set_pipeline(frt_imagewam_native* h,
                                     const frt_imagewam_pipeline_config* config) {
    if (!h || !config) return kInvalid;
    try {
        return h->runtime->set_pipeline(*config);
    } catch (...) {
        return kBackend;
    }
}

int frt_imagewam_native_gemm_shapes(const frt_imagewam_native* h, frt_imagewam_gemm_shape* out,
                                    uint64_t capacity, uint64_t* count) {
    if (!h) return kInvalid;
    return h->runtime->gemm_shapes(out, capacity, count);
}

int frt_imagewam_native_set_gemm_algo(frt_imagewam_native* h, const frt_imagewam_gemm_shape* shape,
                                      const void* algo, uint64_t bytes) {
    if (!h || !shape) return kInvalid;
    return h->runtime->set_gemm_algo(*shape, algo, bytes);
}

int frt_imagewam_native_run(frt_imagewam_native* h, uint32_t segment, int32_t index) {
    if (!h) return kInvalid;
    return h->runtime->run(segment, index);
}

int frt_imagewam_native_capture(frt_imagewam_native* h) {
    if (!h) return kInvalid;
    return h->runtime->capture();
}

int frt_imagewam_native_graph_nodes(const frt_imagewam_native* h, uint64_t* count) {
    if (!h || !count) return kInvalid;
    *count = h->runtime->graph_nodes();
    return 0;
}

}  // extern "C"
