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
 * Threading: calls on one handle must not overlap. The verbs, the setup
 * calls and last_error share the handle's stream, staging scratch and
 * error string, so the host serializes them (one tick at a time per
 * handle). Any host thread may make them; there is no thread affinity.
 * The last_error string stays valid until the next call on the handle.
 * retain / release and declaration_retain / declaration_release are
 * thread-safe. Separate handles are independent, except that run and
 * capture synchronize the device (cudaDeviceSynchronize), so they must
 * not run while another thread captures a CUDA graph in global mode.
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
    uint32_t context_rows;         /* x0 of the length active at creation     */
    uint32_t context_width;        /* joint_attention_dim                     */
    /* The text lengths (x0) this deployment serves: the graph variant keys
     * the handle accepts, in the key space the exec contract uses for this
     * model (`ShapeKey = x0`, the Python side's `GraphVariants`). 0 = one
     * length, `context_rows`; otherwise `text_lengths` points at
     * `num_text_lengths` values, distinct, `context_rows` among them, and
     * copied by frt_imagewam_native_create. `context_rows` is the active
     * length at creation (the variant default key): it bounds
     * set_proprio_row and set_pipeline until set_text_length selects
     * another declared length. */
    uint32_t num_text_lengths;
    const uint32_t* text_lengths;
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

/* sizeof of the config structs as compiled into the library, for binding
 * layout checks: out[0] io_config, out[1] pipeline_config, out[2] linear,
 * out[3] action_step. */
FLASHRT_IMAGEWAM_C_API void frt_imagewam_native_abi_sizes(uint64_t out[4]);

/* Create a handle with one reference. Validates the config, copies the
 * normalization constants, the declared text lengths, creates the native
 * stream and the proprio projection plan. A declared length table must
 * contain `context_rows` (the active length at creation). */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_create(
    const frt_imagewam_io_config* config, frt_imagewam_native** out);

/* Reference counting; `h` is an frt_imagewam_native*. Thread-safe. The
 * handle is destroyed when the count reaches zero. */
FLASHRT_IMAGEWAM_C_API void frt_imagewam_native_retain(void* h);
FLASHRT_IMAGEWAM_C_API void frt_imagewam_native_release(void* h);

/* The owner callbacks for frt_model_runtime_override_verbs: a reference,
 * plus a count of live model runtimes over this handle. While the count is
 * non-zero, use_graph, set_pipeline and capture fail with -1 (the model
 * runtime adopted the current graph exec). */
FLASHRT_IMAGEWAM_C_API void frt_imagewam_native_declaration_retain(void* h);
FLASHRT_IMAGEWAM_C_API void frt_imagewam_native_declaration_release(void* h);

/* Message of the handle's last failure; with a null handle, of the
 * calling thread's last failed frt_imagewam_native_create. */
FLASHRT_IMAGEWAM_C_API const char* frt_imagewam_native_last_error(
    const frt_imagewam_native* h);

/* The cudaStream_t every verb and the graph run on. */
FLASHRT_IMAGEWAM_C_API void* frt_imagewam_native_stream(frt_imagewam_native* h);

/* Replay `graph_exec` (a cudaGraphExec_t captured by the setup producer
 * over the borrowed windows) on the native stream in `step` for the text
 * length `key` (the context length x0 the exec was captured for, one of
 * the config's declared lengths). The exec is borrowed; a graph this handle
 * captured for the same key is destroyed. The handle's active length is
 * unchanged: select the key with set_text_length. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_use_graph(
    frt_imagewam_native* h, uint64_t key, void* graph_exec);

/* 1 when the handle holds a graph for the text length `key`, 0 when not. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_has_variant(
    frt_imagewam_native* h, uint64_t key);

/* The graph exec the handle holds for the text length `key` (null when it
 * holds none). Adopted execs are the producer's, captured ones this
 * handle's. */
FLASHRT_IMAGEWAM_C_API void* frt_imagewam_native_variant_exec(
    frt_imagewam_native* h, uint64_t key);

/* Select the text length (x0) the next ticks serve: the graph `step`
 * replays, the bound of set_proprio_row and the text length every pipeline
 * call (set_pipeline, gemm_shapes, set_gemm_algo, run, capture) resolves its
 * pipeline against. Setup only, and legal while a model runtime is live: the
 * setup producer calls it after every prompt change, next to set_proprio_row
 * (C++ cannot see the Python prompt). -2 for a length the handle has no
 * variant for. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_set_text_length(
    frt_imagewam_native* h, uint64_t key);

/* The text length (x0) the next ticks serve. */
FLASHRT_IMAGEWAM_C_API uint64_t frt_imagewam_native_text_length(
    frt_imagewam_native* h);

/* The graph exec `step` replays now, i.e. the one of the active text
 * length (null when the handle holds none). */
FLASHRT_IMAGEWAM_C_API void* frt_imagewam_native_graph_exec(frt_imagewam_native* h);

/* The context row the proprio token is written to; it must be a row of the
 * active text length's context block. Setup only: the producer calls it
 * after every prompt change, after set_text_length. */
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

/* ------------------------------------------------------------------ */
/* Native pipeline: records the backbone prefill and the ActionDiT     */
/* denoise loop against the csrc kernels (the order of                 */
/* flash_rt/models/imagewam/pipeline_thor.py) from a borrowed resource */
/* table, and captures them as the graph `step` replays.               */
/* ------------------------------------------------------------------ */

enum frt_imagewam_linear_kind {
    FRT_IMAGEWAM_LINEAR_FP16  = 0,   /* GemmRunner::fp16_nn, weight (k, n) fp16   */
    FRT_IMAGEWAM_LINEAR_BF16  = 1,   /* GemmRunner::bf16_nn, weight (k, n) bf16   */
    FRT_IMAGEWAM_LINEAR_NVFP4 = 2    /* dynamic NVFP4 activation + CUTLASS FP4    */
};

typedef struct frt_imagewam_linear {
    uint32_t kind;                   /* frt_imagewam_linear_kind                  */
    int32_t n;                       /* output features                            */
    int32_t k;                       /* input features                             */
    int32_t fp4_variant;             /* NVFP4: CUTLASS variant index               */
    const void* weight;              /* FP16/BF16 (k, n); NVFP4 packed (n, k/2)    */
    const void* weight_scales;       /* NVFP4: SFB                                  */
    void* act_packed;                /* NVFP4: activation scratch (m, k/2)          */
    void* act_scales;                /* NVFP4: activation SFA                       */
} frt_imagewam_linear;

/* One AdaLN site. fp16 shift/scale (dim) feed a standalone AdaLN and the
 * gate materialized to (rows, dim) feeds gate_res_* (the unfused path);
 * the fp32 (dim) vectors are the modulation output the fused gated
 * residual + next AdaLN kernel reads (fuse_res_norm). The gate-less head
 * leaves both gates null. */
typedef struct frt_imagewam_adaln {
    const void* shift;
    const void* scale;
    const void* gate;
    const float* shift_f32;
    const float* scale_f32;
    const float* gate_f32;
} frt_imagewam_adaln;

typedef struct frt_imagewam_double_layer {
    frt_imagewam_linear txt_qkv, img_qkv, txt_proj, img_proj;
    frt_imagewam_linear txt_mlp0, img_mlp0, txt_mlp2, img_mlp2;
    const void* txt_query_norm;
    const void* txt_key_norm;
    const void* img_query_norm;
    const void* img_key_norm;
} frt_imagewam_double_layer;

/* Backbone or ActionDiT single-stream block (merged linear1). With
 * merge_linear2 only `linear2` (K = attn width + mlp hidden) is used,
 * otherwise only attn_out_proj and mlp_down. */
typedef struct frt_imagewam_single_layer {
    frt_imagewam_linear linear1, attn_out_proj, mlp_down, linear2;
    const void* query_norm;
    const void* key_norm;
} frt_imagewam_single_layer;

typedef struct frt_imagewam_action_double_layer {
    frt_imagewam_linear qkv, proj, mlp0, mlp2;
    const void* query_norm;
    const void* key_norm;
} frt_imagewam_action_double_layer;

typedef struct frt_imagewam_action_step {
    frt_imagewam_adaln double1, double2, single, head;
    float delta;                     /* Euler step size of this step            */
    uint32_t reserved;
} frt_imagewam_action_step;

typedef struct frt_imagewam_pipeline_config {
    uint32_t struct_size;            /* = sizeof(frt_imagewam_pipeline_config)  */
    int32_t hidden, head_dim, num_heads, mlp_hidden, joint_attention_dim;
    int32_t x0, a0, total, num_action, action_dim;
    int32_t action_hidden_dim, action_attn_width, action_mlp_hidden;
    int32_t num_double, num_single, action_num_double, action_num_single, num_steps;
    int32_t merge_linear2;           /* single-stream linear2 as one GEMM         */
    int32_t fuse_res_norm;           /* gated residual fused with the next AdaLN  */
    float eps;
    /* Borrowed buffers (pipeline_thor.py `bufs`); bf16: context, backbone_hidden,
     * img_raw; f32: action_latent; every other one fp16. */
    void *context, *backbone_hidden, *img_raw, *modded_scratch;
    void *txt_qkv_merged, *img_qkv_merged, *single_linear1_merged, *action_linear1_merged;
    void *action_qkv_merged, *action_latent_fp16, *velocity, *head_modded;
    void *txt_mlp_merged, *txt_mlp_gated, *img_mlp_merged, *img_mlp_gated, *single_mlp_gated;
    void *proj_scratch, *proj_scratch2, *action_latent, *action_hidden, *action_modded;
    void *action_proj_scratch, *action_proj_scratch2, *action_mlp_merged, *action_mlp_gated;
    void *single_linear2_in, *action_linear2_in;   /* merged linear2 inputs      */
    /* Attention (borrowed): shared Q/O, per-layer K/V at base + layer * stride. */
    void *q_o, *k_cache, *v_cache, *logits;
    uint64_t kv_layer_stride_bytes;
    float attn_scale;
    uint32_t reserved;
    const void* rope_table;          /* backbone (a0, HD) fp16                   */
    const void* action_rope_table;   /* action (num_action, HD) fp16             */
    /* Shared weights. */
    frt_imagewam_linear txt_in, img_in, action_encoder, head_linear;
    const void* action_encoder_bias;
    /* Backbone modulation, shared by every layer of its stream type. */
    frt_imagewam_adaln txt_mod1, txt_mod2, img_mod1, img_mod2, single_mod;
    /* Arrays: num_double, num_single, action_num_double, action_num_single,
     * num_steps entries. Copied by set_pipeline. */
    const frt_imagewam_double_layer* double_layers;
    const frt_imagewam_single_layer* single_layers;
    const frt_imagewam_action_double_layer* action_double_layers;
    const frt_imagewam_single_layer* action_single_layers;
    const frt_imagewam_action_step* steps;
} frt_imagewam_pipeline_config;

/* Install the pipeline (copies the tables, creates the pipeline's own
 * GemmRunner and cuBLAS handle) for the text length `config.x0`, which must be
 * one of the config's declared lengths, and select that key as the active text
 * length. One pipeline (and one captured graph) per length: installing a key
 * again replaces that key's pipeline and destroys the graph captured from it
 * (after synchronizing the native stream), while the other keys keep both.
 * `step` fails for a key whose graph was dropped until its next `capture`, and
 * the resources the replaced table pointed to may be freed once this returns.
 * Setup only. The pipeline calls that follow (gemm_shapes, set_gemm_algo, run,
 * capture) all resolve against the active text length. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_set_pipeline(
    frt_imagewam_native* h, const frt_imagewam_pipeline_config* config);

/* One GEMM shape the pipeline launches through GemmRunner. */
typedef struct frt_imagewam_gemm_shape {
    int32_t kind;                    /* 0 = bf16_nn, 1 = fp16_nn (GemmRunner hand-off kinds) */
    int32_t m, n, k;
} frt_imagewam_gemm_shape;

/* The distinct GEMM shapes of the pipeline installed for the ACTIVE text
 * length; `count` receives the number; -5 when `capacity` is too small, -1
 * when no pipeline is installed for it. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_gemm_shapes(
    const frt_imagewam_native* h, frt_imagewam_gemm_shape* out, uint64_t capacity,
    uint64_t* count);

/* Install the cuBLASLt algorithm (GemmRunner::kAlgoBytes bytes) the setup
 * producer selected for one shape, so both pipelines run the same kernel:
 * into the pipeline installed for the ACTIVE text length. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_set_gemm_algo(
    frt_imagewam_native* h, const frt_imagewam_gemm_shape* shape,
    const void* algo, uint64_t bytes);

/* Eager segments on the native stream, of the pipeline installed for the
 * ACTIVE text length. They write the frontend's buffers,
 * so `run` (and `capture`, which runs FULL once as its warm-up) first waits
 * for all prior work on the device (cudaDeviceSynchronize), then
 * synchronizes the native stream before return. */
enum frt_imagewam_segment {
    FRT_IMAGEWAM_SEGMENT_DOUBLE_LAYER = 0,   /* backbone double-stream block `index` */
    FRT_IMAGEWAM_SEGMENT_SINGLE_LAYER = 1,   /* backbone single-stream block `index` */
    FRT_IMAGEWAM_SEGMENT_PREFILL      = 2,
    FRT_IMAGEWAM_SEGMENT_DENOISE_STEP = 3,   /* denoise step `index`                 */
    FRT_IMAGEWAM_SEGMENT_DENOISE      = 4,   /* all denoise steps                    */
    FRT_IMAGEWAM_SEGMENT_FULL         = 5    /* prefill + denoise                    */
};
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_run(
    frt_imagewam_native* h, uint32_t segment, int32_t index);

/* Warm up once eagerly, then capture prefill + denoise on the native stream
 * into a graph the handle owns (replacing the one it holds for that length, if
 * it captured one) for the ACTIVE text length, whose pipeline must be
 * installed; `step` replays it from then on. -1 when no pipeline is installed
 * for the active text length. */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_capture(frt_imagewam_native* h);

/* Number of kernel nodes in the graph `step` replays, when this handle
 * captured it (0 before capture, and for an exec the producer adopted). */
FLASHRT_IMAGEWAM_C_API int frt_imagewam_native_graph_nodes(
    const frt_imagewam_native* h, uint64_t* count);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLASHRT_CPP_MODELS_IMAGEWAM_C_API_H */
