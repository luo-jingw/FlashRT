/* ImageWAM native runtime — C ABI (libflashrt_imagewam_native.so).
 *
 * The native half of ImageWAM's `io="native"` model-runtime face. A setup
 * producer (the Python frontend today) owns every device allocation and
 * builds the frt_model_runtime_v1 declaration; this library supplies the
 * hot-path verbs (set_input / get_output / step) as plain C functions, so a
 * tick runs without Python or the GIL.
 *
 * Ownership: every device pointer passed in a config is BORROWED. The
 * library owns only its CUDA stream, the graph exec it captures, its
 * cuBLAS/cuBLASLt handles and workspace, and small staging scratch. The
 * producer must keep borrowed memory alive while any reference to the
 * handle exists (the model runtime anchors both).
 *
 * Status codes follow the model-runtime convention: 0 ok, -1 invalid,
 * -2 not found, -3 unsupported, -4 shape mismatch, -5 insufficient
 * storage, -6 backend. The message of the last failure is available from
 * frt_imagewam_native_last_error.
 *
 * Interface record: docs/imagewam_native_cpp.md.
 */
#ifndef FLASHRT_CPP_MODELS_IMAGEWAM_C_API_H
#define FLASHRT_CPP_MODELS_IMAGEWAM_C_API_H

#include <stddef.h>
#include <stdint.h>

#include "flashrt/model_runtime.h"

#if defined(__GNUC__) || defined(__clang__)
#define FLASHRT_IMAGEWAM_C_API __attribute__((visibility("default")))
#else
#define FLASHRT_IMAGEWAM_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct frt_imagewam_native frt_imagewam_native;

/* Per-tick IO surface of one captured ImageWAM deployment. */
typedef struct frt_imagewam_io_config {
    uint32_t struct_size;          /* = sizeof(frt_imagewam_io_config)        */
    uint32_t img_len;              /* image_tokens rows (392 for LIBERO)      */
    uint32_t token_dim;            /* image_tokens columns (HD = 128)         */
    uint32_t num_action;           /* action chunk rows (64)                  */
    uint32_t action_dim;           /* action chunk columns (7)                */
    uint32_t proprio_dim;          /* raw robot state width; 0 = no proprio   */
    uint32_t context_rows;         /* x0                                      */
    uint32_t context_width;        /* joint_attention_dim                     */
    /* Borrowed device windows. */
    void* img_raw;                 /* bf16 (img_len, token_dim)               */
    void* context;                 /* bf16 (context_rows, context_width)      */
    void* action_latent;           /* f32  (num_action, action_dim)           */
    /* Borrowed proprio projection (proprio_dim > 0): bf16 row-major
     * weight^T (proprio_dim, context_width) and bias (context_width). */
    const void* proprio_weight_t;
    const void* proprio_bias;
    /* Copied min/max normalization constants; null pointers mean identity.
     * state: proprio_dim entries each; action: action_dim entries each. */
    const float* state_scale;
    const float* state_offset;
    const float* action_scale;
    const float* action_offset;
} frt_imagewam_io_config;

/* Create a handle with one reference. Validates the config, copies the
 * normalization constants, creates the native stream and the proprio
 * projection plan. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_create(
    const frt_imagewam_io_config* config, frt_imagewam_native** out);

/* Reference counting; `h` is an frt_imagewam_native*. Thread-safe. The
 * handle is destroyed when the count reaches zero. Their addresses are the
 * owner callbacks for frt_model_runtime_override_verbs. */
FLASHRT_IMAGEWAM_C_API void frt_imagewam_native_retain(void* h);
FLASHRT_IMAGEWAM_C_API void frt_imagewam_native_release(void* h);

FLASHRT_IMAGEWAM_C_API const char* frt_imagewam_native_last_error(
    const frt_imagewam_native* h);

/* The cudaStream_t every verb and the graph run on. */
FLASHRT_IMAGEWAM_C_API void* frt_imagewam_native_stream(frt_imagewam_native* h);

/* Replay `graph_exec` (a cudaGraphExec_t captured by the setup producer
 * over the borrowed windows) on the native stream in `step`. The exec is
 * borrowed. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_use_graph(
    frt_imagewam_native* h, void* graph_exec);

/* The graph exec `step` replays (null before use_graph/capture). */
FLASHRT_IMAGEWAM_C_API void* frt_imagewam_native_graph_exec(frt_imagewam_native* h);

/* The context row the proprio token is written to. Setup only: the
 * producer calls it after every prompt change. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_set_proprio_row(
    frt_imagewam_native* h, int32_t row);

/* Canonical port / region / stage records of the `io="native"` schema this
 * handle implements, in the runtime builder's identity format (one record
 * per line), for the declaration's buffer order img_raw, context,
 * action_latent. `written` receives the byte length; -5 when `capacity`
 * is too small. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_schema_records(
    const frt_imagewam_native* h, char* out, uint64_t capacity,
    uint64_t* written);

/* Check a producer declaration against the native schema (port names,
 * modality, dtype, direction, update, shape, window size) and remember the
 * port indices. Required before the verbs are used. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_bind_declaration(
    frt_imagewam_native* h, const frt_model_runtime_v1* declaration);

/* The native verbs; `self` is the handle. Every entry is non-null. */
FLASHRT_IMAGEWAM_C_API const frt_model_runtime_verbs* frt_imagewam_native_verbs(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLASHRT_CPP_MODELS_IMAGEWAM_C_API_H */
