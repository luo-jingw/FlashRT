#include "native_pipeline.h"

#include "flashrt/cpp/models/imagewam/c_api.h"

#include "activation.cuh"
#include "attention_cublas.cuh"
#include "csrc_operations.h"
#include "decoder_fused.cuh"
#include "elementwise.cuh"
#include "fp4_linear.h"
#include "gemm_runner.h"
#include "norm.cuh"
#include "rope.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace imagewam {

namespace {

constexpr int kGemmBf16 = 0;
constexpr int kGemmFp16 = 1;

const __half* h(const void* p) { return static_cast<const __half*>(p); }
__half* hm(void* p) { return static_cast<__half*>(p); }
__nv_bfloat16* bm(void* p) { return static_cast<__nv_bfloat16*>(p); }

// Byte offset of `rows` rows of a 2-byte-element matrix with `width` columns.
void* row_offset(void* base, long long rows, long long width) {
    return static_cast<char*>(base) + rows * width * 2;
}

bool valid_linear(const frt_imagewam_linear& l) {
    if (l.n <= 0 || l.k <= 0 || !l.weight) return false;
    if (l.kind == FRT_IMAGEWAM_LINEAR_FP16 || l.kind == FRT_IMAGEWAM_LINEAR_BF16) return true;
    return l.kind == FRT_IMAGEWAM_LINEAR_NVFP4 && nvfp4_available() && l.weight_scales &&
           l.act_packed && l.act_scales && l.fp4_variant >= 0;
}

bool valid_adaln(const frt_imagewam_adaln& a, bool gate) {
    return a.shift && a.scale && (!gate || a.gate);
}

void check_launch(const char* what) {
    const cudaError_t rc = cudaGetLastError();
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc));
    }
}

}  // namespace

NativePipeline::~NativePipeline() {
    if (cublas_) cublasDestroy(cublas_);
}

std::unique_ptr<NativePipeline> NativePipeline::create(const frt_imagewam_pipeline_config& c,
                                                       std::string* error) {
    if (c.struct_size < sizeof(frt_imagewam_pipeline_config)) {
        *error = "frt_imagewam_pipeline_config.struct_size is smaller than this library's config";
        return nullptr;
    }
    if (c.hidden <= 0 || c.head_dim <= 0 || c.num_heads * c.head_dim != c.hidden ||
        c.x0 <= 0 || c.a0 <= c.x0 || c.total != c.a0 + c.num_action || c.num_action <= 0 ||
        c.action_dim <= 0 || c.action_attn_width != c.hidden || c.num_double < 0 ||
        c.num_single < 0 || c.action_num_double < 0 || c.action_num_single < 0 ||
        c.num_steps <= 0) {
        *error = "frt_imagewam_pipeline_config has inconsistent dimensions";
        return nullptr;
    }
    const void* buffers[] = {
        c.context, c.backbone_hidden, c.img_raw, c.modded_scratch, c.txt_qkv_merged,
        c.img_qkv_merged, c.single_linear1_merged, c.action_linear1_merged, c.action_qkv_merged,
        c.action_latent_fp16, c.velocity, c.head_modded, c.txt_mlp_merged, c.txt_mlp_gated,
        c.img_mlp_merged, c.img_mlp_gated, c.single_mlp_gated, c.proj_scratch, c.proj_scratch2,
        c.action_latent, c.action_hidden, c.action_modded, c.action_proj_scratch,
        c.action_proj_scratch2, c.action_mlp_merged, c.action_mlp_gated, c.q_o, c.k_cache,
        c.v_cache, c.logits, c.rope_table, c.action_rope_table, c.action_encoder_bias};
    for (const void* p : buffers) {
        if (!p) {
            *error = "frt_imagewam_pipeline_config has a null buffer";
            return nullptr;
        }
    }
    if (!valid_linear(c.txt_in) || !valid_linear(c.img_in) || !valid_linear(c.action_encoder) ||
        !valid_linear(c.head_linear) || !valid_adaln(c.txt_mod1, true) ||
        !valid_adaln(c.txt_mod2, true) || !valid_adaln(c.img_mod1, true) ||
        !valid_adaln(c.img_mod2, true) || !valid_adaln(c.single_mod, true)) {
        *error = "frt_imagewam_pipeline_config has an invalid shared weight or modulation";
        return nullptr;
    }
    if ((c.num_double && !c.double_layers) || (c.num_single && !c.single_layers) ||
        (c.action_num_double && !c.action_double_layers) ||
        (c.action_num_single && !c.action_single_layers) || !c.steps) {
        *error = "frt_imagewam_pipeline_config has a null layer table";
        return nullptr;
    }
    std::unique_ptr<NativePipeline> p(new NativePipeline());
    p->c_ = c;
    p->double_layers_.assign(c.double_layers, c.double_layers + c.num_double);
    p->single_layers_.assign(c.single_layers, c.single_layers + c.num_single);
    p->action_double_layers_.assign(c.action_double_layers,
                                    c.action_double_layers + c.action_num_double);
    p->action_single_layers_.assign(c.action_single_layers,
                                    c.action_single_layers + c.action_num_single);
    p->steps_.assign(c.steps, c.steps + c.num_steps);
    p->c_.double_layers = nullptr;
    p->c_.single_layers = nullptr;
    p->c_.action_double_layers = nullptr;
    p->c_.action_single_layers = nullptr;
    p->c_.steps = nullptr;
    for (const auto& l : p->double_layers_) {
        for (const frt_imagewam_linear* w : {&l.txt_qkv, &l.img_qkv, &l.txt_proj, &l.img_proj,
                                             &l.txt_mlp0, &l.img_mlp0, &l.txt_mlp2, &l.img_mlp2}) {
            if (!valid_linear(*w)) {
                *error = "invalid double-stream linear";
                return nullptr;
            }
        }
        if (!l.txt_query_norm || !l.txt_key_norm || !l.img_query_norm || !l.img_key_norm) {
            *error = "null double-stream QK norm";
            return nullptr;
        }
    }
    for (const auto* table : {&p->single_layers_, &p->action_single_layers_}) {
        for (const auto& l : *table) {
            if (!valid_linear(l.linear1) || !valid_linear(l.attn_out_proj) ||
                !valid_linear(l.mlp_down) || !l.query_norm || !l.key_norm) {
                *error = "invalid single-stream layer";
                return nullptr;
            }
        }
    }
    for (const auto& l : p->action_double_layers_) {
        if (!valid_linear(l.qkv) || !valid_linear(l.proj) || !valid_linear(l.mlp0) ||
            !valid_linear(l.mlp2) || !l.query_norm || !l.key_norm) {
            *error = "invalid action double-stream layer";
            return nullptr;
        }
    }
    for (const auto& s : p->steps_) {
        if (!valid_adaln(s.double1, true) || !valid_adaln(s.double2, true) ||
            !valid_adaln(s.single, true) || !valid_adaln(s.head, false)) {
            *error = "invalid action step modulation";
            return nullptr;
        }
    }
    try {
        p->gemm_.reset(new GemmRunner());
    } catch (const std::exception& e) {
        *error = std::string("GemmRunner: ") + e.what();
        return nullptr;
    }
    if (cublasCreate(&p->cublas_) != CUBLAS_STATUS_SUCCESS) {
        p->cublas_ = nullptr;
        *error = "cublasCreate failed";
        return nullptr;
    }
    p->collect_gemm_shapes();
    return p;
}

void NativePipeline::add_gemm_shape(const frt_imagewam_linear& l, int m) {
    int kind;
    if (l.kind == FRT_IMAGEWAM_LINEAR_FP16) {
        kind = kGemmFp16;
    } else if (l.kind == FRT_IMAGEWAM_LINEAR_BF16) {
        kind = kGemmBf16;
    } else {
        return;
    }
    for (const auto& s : gemm_shapes_) {
        if (s.kind == kind && s.m == m && s.n == l.n && s.k == l.k) return;
    }
    gemm_shapes_.push_back({kind, m, l.n, l.k});
}

void NativePipeline::collect_gemm_shapes() {
    const int img_len = c_.a0 - c_.x0;
    add_gemm_shape(c_.txt_in, c_.x0);
    add_gemm_shape(c_.img_in, img_len);
    for (const auto& l : double_layers_) {
        for (const frt_imagewam_linear* w : {&l.txt_qkv, &l.txt_proj, &l.txt_mlp0, &l.txt_mlp2})
            add_gemm_shape(*w, c_.x0);
        for (const frt_imagewam_linear* w : {&l.img_qkv, &l.img_proj, &l.img_mlp0, &l.img_mlp2})
            add_gemm_shape(*w, img_len);
    }
    for (const auto& l : single_layers_) {
        for (const frt_imagewam_linear* w : {&l.linear1, &l.attn_out_proj, &l.mlp_down})
            add_gemm_shape(*w, c_.a0);
    }
    add_gemm_shape(c_.action_encoder, c_.num_action);
    add_gemm_shape(c_.head_linear, c_.num_action);
    for (const auto& l : action_double_layers_) {
        for (const frt_imagewam_linear* w : {&l.qkv, &l.proj, &l.mlp0, &l.mlp2})
            add_gemm_shape(*w, c_.num_action);
    }
    for (const auto& l : action_single_layers_) {
        for (const frt_imagewam_linear* w : {&l.linear1, &l.attn_out_proj, &l.mlp_down})
            add_gemm_shape(*w, c_.num_action);
    }
}

void NativePipeline::set_gemm_algo(const frt_imagewam_gemm_shape& s, const void* algo) {
    gemm_->set_cached_algo(s.kind, s.m, s.n, s.k, algo);
}

void* NativePipeline::k_layer(int layer) const {
    return static_cast<char*>(c_.k_cache) + static_cast<size_t>(layer) * c_.kv_layer_stride_bytes;
}

void* NativePipeline::v_layer(int layer) const {
    return static_cast<char*>(c_.v_cache) + static_cast<size_t>(layer) * c_.kv_layer_stride_bytes;
}

void NativePipeline::linear(const frt_imagewam_linear& l, const void* x, void* out, int m,
                            cudaStream_t stream) {
    void* xv = const_cast<void*>(x);
    void* wv = const_cast<void*>(l.weight);
    switch (l.kind) {
        case FRT_IMAGEWAM_LINEAR_FP16:
            gemm_->fp16_nn(xv, wv, out, m, l.n, l.k, stream);
            return;
        case FRT_IMAGEWAM_LINEAR_BF16:
            gemm_->bf16_nn(xv, wv, out, m, l.n, l.k, stream);
            return;
        case FRT_IMAGEWAM_LINEAR_NVFP4:
            nvfp4_linear(l, x, out, m, stream);
            return;
        default:
            throw std::invalid_argument("unknown frt_imagewam_linear kind");
    }
}

// pipeline_thor._mlp_gate_up for a plain linear: one merged GEMM, then SiLU-GLU.
void NativePipeline::mlp_gate_up(const frt_imagewam_linear& l, const void* x, void* merged,
                                 void* gated, int m, int mlp_hidden, cudaStream_t stream) {
    linear(l, x, merged, m, stream);
    silu_glu_merged_fp16(h(merged), hm(gated), m, mlp_hidden, stream, 0);
}

// pipeline_thor._copy_slice x3: Q/K/V column thirds of a fused projection.
void NativePipeline::copy_qkv(const void* merged, int merged_width, int rows, int width, void* q,
                              void* k, void* v, cudaStream_t stream) {
    gpu_strided_copy_fp16(h(merged), hm(q), rows, width, merged_width, 0, stream);
    gpu_strided_copy_fp16(h(merged), hm(k), rows, width, merged_width, width, stream);
    gpu_strided_copy_fp16(h(merged), hm(v), rows, width, merged_width, 2 * width, stream);
}

// ImageWAMAttnBackend.run with use_perhead_kv=True, use_real_mot_mask=True:
// plain per-head attention for both sites, output written over the queries.
void NativePipeline::attention(void* q_out, int layer, int q_rows, int kv_rows,
                               cudaStream_t stream) {
    attention_qkv_fp16_perhead(cublas_, h(q_out), h(k_layer(layer)), h(v_layer(layer)),
                               hm(c_.logits), hm(q_out), q_rows, kv_rows, c_.num_heads,
                               c_.head_dim, c_.attn_scale, stream);
}

void NativePipeline::double_layer(int i, cudaStream_t s) {
    const frt_imagewam_double_layer& w = double_layers_.at(i);
    const int hidden = c_.hidden, x0 = c_.x0, a0 = c_.a0, img_len = a0 - x0;
    const int nh = c_.num_heads, hd = c_.head_dim;
    const float eps = c_.eps;
    void* combined = c_.backbone_hidden;
    void* modded = c_.modded_scratch;
    void* q = c_.q_o;
    void* k = k_layer(i);
    void* v = v_layer(i);

    // text stream: rows [0, x0) of the persistent bf16 residual
    ada_layer_norm_bf16in_fp16out(bm(combined), h(c_.txt_mod1.scale), h(c_.txt_mod1.shift),
                                  hm(modded), x0, hidden, eps, s);
    linear(w.txt_qkv, modded, c_.txt_qkv_merged, x0, s);
    copy_qkv(c_.txt_qkv_merged, 3 * hidden, x0, hidden, q, k, v, s);
    rms_norm_fp16(h(q), h(w.txt_query_norm), hm(q), x0 * nh, hd, eps, s);
    rms_norm_fp16(h(k), h(w.txt_key_norm), hm(k), x0 * nh, hd, eps, s);

    // image stream: rows [x0, a0)
    void* img_x = row_offset(combined, x0, hidden);
    void* img_modded = row_offset(modded, x0, hidden);
    void* img_q = row_offset(q, x0, hidden);
    void* img_k = row_offset(k, x0, hidden);
    void* img_v = row_offset(v, x0, hidden);
    ada_layer_norm_bf16in_fp16out(bm(img_x), h(c_.img_mod1.scale), h(c_.img_mod1.shift),
                                  hm(img_modded), img_len, hidden, eps, s);
    linear(w.img_qkv, img_modded, c_.img_qkv_merged, img_len, s);
    copy_qkv(c_.img_qkv_merged, 3 * hidden, img_len, hidden, img_q, img_k, img_v, s);
    rms_norm_fp16(h(img_q), h(w.img_query_norm), hm(img_q), img_len * nh, hd, eps, s);
    rms_norm_fp16(h(img_k), h(w.img_key_norm), hm(img_k), img_len * nh, hd, eps, s);

    rope_apply_fp16_perhead(hm(q), h(c_.rope_table), a0, nh, hd, s);
    rope_apply_fp16_perhead(hm(k), h(c_.rope_table), a0, nh, hd, s);
    attention(q, i, a0, a0, s);

    void* proj = c_.proj_scratch;
    linear(w.txt_proj, q, proj, x0, s);
    gate_res_bf16res(h(proj), h(c_.txt_mod1.gate), bm(combined), x0 * hidden, s);
    void* img_proj = row_offset(proj, x0, hidden);
    linear(w.img_proj, img_q, img_proj, img_len, s);
    gate_res_bf16res(h(img_proj), h(c_.img_mod1.gate), bm(img_x), img_len * hidden, s);

    ada_layer_norm_bf16in_fp16out(bm(combined), h(c_.txt_mod2.scale), h(c_.txt_mod2.shift),
                                  hm(modded), x0, hidden, eps, s);
    mlp_gate_up(w.txt_mlp0, modded, c_.txt_mlp_merged, c_.txt_mlp_gated, x0, c_.mlp_hidden, s);
    linear(w.txt_mlp2, c_.txt_mlp_gated, proj, x0, s);
    gate_res_bf16res(h(proj), h(c_.txt_mod2.gate), bm(combined), x0 * hidden, s);

    ada_layer_norm_bf16in_fp16out(bm(img_x), h(c_.img_mod2.scale), h(c_.img_mod2.shift),
                                  hm(img_modded), img_len, hidden, eps, s);
    mlp_gate_up(w.img_mlp0, img_modded, c_.img_mlp_merged, c_.img_mlp_gated, img_len,
                c_.mlp_hidden, s);
    linear(w.img_mlp2, c_.img_mlp_gated, img_proj, img_len, s);
    gate_res_bf16res(h(img_proj), h(c_.img_mod2.gate), bm(img_x), img_len * hidden, s);
    check_launch("double-stream layer");
}

void NativePipeline::single_layer(int i, cudaStream_t s) {
    const frt_imagewam_single_layer& w = single_layers_.at(i);
    const int site = c_.num_double + i;
    const int hidden = c_.hidden, a0 = c_.a0, nh = c_.num_heads, hd = c_.head_dim;
    const int linear1_width = 3 * hidden + 2 * c_.mlp_hidden;
    const float eps = c_.eps;
    void* combined = c_.backbone_hidden;
    void* modded = c_.modded_scratch;
    void* q = c_.q_o;
    void* k = k_layer(site);
    void* v = v_layer(site);

    ada_layer_norm_bf16in_fp16out(bm(combined), h(c_.single_mod.scale), h(c_.single_mod.shift),
                                  hm(modded), a0, hidden, eps, s);
    linear(w.linear1, modded, c_.single_linear1_merged, a0, s);
    copy_qkv(c_.single_linear1_merged, linear1_width, a0, hidden, q, k, v, s);
    silu_glu_merged_fp16(h(c_.single_linear1_merged) + 3 * hidden, hm(c_.single_mlp_gated), a0,
                         c_.mlp_hidden, s, linear1_width);
    rms_norm_fp16(h(q), h(w.query_norm), hm(q), a0 * nh, hd, eps, s);
    rms_norm_fp16(h(k), h(w.key_norm), hm(k), a0 * nh, hd, eps, s);
    rope_apply_fp16_perhead(hm(q), h(c_.rope_table), a0, nh, hd, s);
    rope_apply_fp16_perhead(hm(k), h(c_.rope_table), a0, nh, hd, s);
    attention(q, site, a0, a0, s);

    linear(w.attn_out_proj, q, c_.proj_scratch, a0, s);
    linear(w.mlp_down, c_.single_mlp_gated, c_.proj_scratch2, a0, s);
    residual_add_fp16(hm(c_.proj_scratch), h(c_.proj_scratch2), a0 * hidden, s);
    gate_res_bf16res(h(c_.proj_scratch), h(c_.single_mod.gate), bm(combined), a0 * hidden, s);
    check_launch("single-stream layer");
}

void NativePipeline::prefill(cudaStream_t s) {
    const int img_len = c_.a0 - c_.x0;
    linear(c_.txt_in, c_.context, c_.backbone_hidden, c_.x0, s);
    linear(c_.img_in, c_.img_raw, row_offset(c_.backbone_hidden, c_.x0, c_.hidden), img_len, s);
    for (int i = 0; i < c_.num_double; ++i) double_layer(i, s);
    for (int i = 0; i < c_.num_single; ++i) single_layer(i, s);
}

void NativePipeline::action_double_layer(int i, const frt_imagewam_action_step& mods,
                                         cudaStream_t s) {
    const frt_imagewam_action_double_layer& w = action_double_layers_.at(i);
    const int ahd = c_.action_hidden_dim, aaw = c_.action_attn_width, na = c_.num_action;
    const int nh = c_.num_heads, hd = c_.head_dim;
    const float eps = c_.eps;
    void* x = c_.action_hidden;
    void* modded = c_.action_modded;
    void* aq = row_offset(c_.q_o, c_.a0, aaw);
    void* ak = row_offset(k_layer(i), c_.a0, aaw);
    void* av = row_offset(v_layer(i), c_.a0, aaw);

    ada_layer_norm_fp16(h(x), h(mods.double1.scale), h(mods.double1.shift), hm(modded), na, ahd,
                        eps, s);
    linear(w.qkv, modded, c_.action_qkv_merged, na, s);
    copy_qkv(c_.action_qkv_merged, 3 * aaw, na, aaw, aq, ak, av, s);
    rms_norm_fp16(h(aq), h(w.query_norm), hm(aq), na * nh, hd, eps, s);
    rms_norm_fp16(h(ak), h(w.key_norm), hm(ak), na * nh, hd, eps, s);
    rope_apply_fp16_perhead(hm(aq), h(c_.action_rope_table), na, nh, hd, s);
    rope_apply_fp16_perhead(hm(ak), h(c_.action_rope_table), na, nh, hd, s);
    attention(aq, i, na, c_.total, s);

    void* proj = c_.action_proj_scratch;
    linear(w.proj, aq, proj, na, s);
    gate_res_fp16(h(proj), h(mods.double1.gate), hm(x), na * ahd, s);
    ada_layer_norm_fp16(h(x), h(mods.double2.scale), h(mods.double2.shift), hm(modded), na, ahd,
                        eps, s);
    mlp_gate_up(w.mlp0, modded, c_.action_mlp_merged, c_.action_mlp_gated, na,
                c_.action_mlp_hidden, s);
    linear(w.mlp2, c_.action_mlp_gated, proj, na, s);
    gate_res_fp16(h(proj), h(mods.double2.gate), hm(x), na * ahd, s);
    check_launch("action double-stream layer");
}

void NativePipeline::action_single_layer(int i, const frt_imagewam_action_step& mods,
                                         cudaStream_t s) {
    const frt_imagewam_single_layer& w = action_single_layers_.at(i);
    const int site = c_.action_num_double + i;
    const int ahd = c_.action_hidden_dim, aaw = c_.action_attn_width, na = c_.num_action;
    const int nh = c_.num_heads, hd = c_.head_dim;
    const int linear1_width = 3 * aaw + 2 * c_.action_mlp_hidden;
    const float eps = c_.eps;
    void* x = c_.action_hidden;
    void* modded = c_.action_modded;
    void* aq = row_offset(c_.q_o, c_.a0, aaw);
    void* ak = row_offset(k_layer(site), c_.a0, aaw);
    void* av = row_offset(v_layer(site), c_.a0, aaw);

    ada_layer_norm_fp16(h(x), h(mods.single.scale), h(mods.single.shift), hm(modded), na, ahd, eps,
                        s);
    linear(w.linear1, modded, c_.action_linear1_merged, na, s);
    copy_qkv(c_.action_linear1_merged, linear1_width, na, aaw, aq, ak, av, s);
    silu_glu_merged_fp16(h(c_.action_linear1_merged) + 3 * aaw, hm(c_.action_mlp_gated), na,
                         c_.action_mlp_hidden, s, linear1_width);
    rms_norm_fp16(h(aq), h(w.query_norm), hm(aq), na * nh, hd, eps, s);
    rms_norm_fp16(h(ak), h(w.key_norm), hm(ak), na * nh, hd, eps, s);
    rope_apply_fp16_perhead(hm(aq), h(c_.action_rope_table), na, nh, hd, s);
    rope_apply_fp16_perhead(hm(ak), h(c_.action_rope_table), na, nh, hd, s);
    attention(aq, site, na, c_.total, s);

    linear(w.attn_out_proj, aq, c_.action_proj_scratch, na, s);
    linear(w.mlp_down, c_.action_mlp_gated, c_.action_proj_scratch2, na, s);
    residual_add_fp16(hm(c_.action_proj_scratch), h(c_.action_proj_scratch2), na * ahd, s);
    gate_res_fp16(h(c_.action_proj_scratch), h(mods.single.gate), hm(x), na * ahd, s);
    check_launch("action single-stream layer");
}

void NativePipeline::denoise_step(int step, cudaStream_t s) {
    const frt_imagewam_action_step& mods = steps_.at(step);
    const int na = c_.num_action, ad = c_.action_dim, ahd = c_.action_hidden_dim;

    gpu_cast_fp32_to_fp16(static_cast<const float*>(c_.action_latent), hm(c_.action_latent_fp16),
                          na * ad, s);
    linear(c_.action_encoder, c_.action_latent_fp16, c_.action_hidden, na, s);
    add_bias_fp16(hm(c_.action_hidden), h(c_.action_encoder_bias), na, ahd, s);
    for (int i = 0; i < c_.action_num_double; ++i) action_double_layer(i, mods, s);
    for (int i = 0; i < c_.action_num_single; ++i) action_single_layer(i, mods, s);

    ada_layer_norm_fp16(h(c_.action_hidden), h(mods.head.scale), h(mods.head.shift),
                        hm(c_.head_modded), na, ahd, c_.eps, s);
    linear(c_.head_linear, c_.head_modded, c_.velocity, na, s);
    gpu_euler_step(static_cast<float*>(c_.action_latent), h(c_.velocity), na, ad, mods.delta, 0,
                   s);
    check_launch("denoise step");
}

void NativePipeline::denoise(cudaStream_t s) {
    for (int step = 0; step < c_.num_steps; ++step) denoise_step(step, s);
}

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt
