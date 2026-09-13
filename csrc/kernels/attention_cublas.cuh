// ================================================================
// FlashRT — cuBLAS decomposed attention (GQA-compatible)
// QK^T + softmax + PV, matching pi05 engine exactly.
// Stateless: receives cuBLAS handle from caller (FvkContext).
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Full attention: Q @ K^T → softmax → @ V → out
// Supports GQA: K/V have 1 head, Q has NH heads.
void attention_qkv_fp16(
    cublasHandle_t handle,   // caller's cuBLAS handle (from FvkContext)
    const __half* Q,         // (S, NH*HD) = (S*NH, HD) contiguous
    const __half* K,         // (S_kv, HD) single KV head
    const __half* V,         // (S_kv, HD)
    __half* logits,          // (S*NH, S_kv) scratch buffer
    __half* out,             // (S*NH, HD) = (S, NH, HD) output
    int S, int S_kv, int NH, int HD,
    float attn_scale,        // 1/sqrt(HD)
    cudaStream_t stream = 0);

// Fixed-shape attention with a device-side valid K/V length.
// QK and PV keep the graph-captured S_kv_max shape; logits rows
// [seqused_k[0], S_kv_max) are masked before softmax.
void attention_qkv_fp16_seqused(
    cublasHandle_t handle,
    const __half* Q,         // (S*NH, HD)
    const __half* K,         // (S_kv_max, HD)
    const __half* V,         // (S_kv_max, HD)
    __half* logits,          // scratch: (S*NH, S_kv_max)
    __half* out,             // (S*NH, HD)
    int S, int S_kv_max, int NH, int HD,
    const int* seqused_k,    // device int32[1], valid K/V rows
    float attn_scale,
    cudaStream_t stream = 0);

// Same as attention_qkv_fp16 but supports ODD S_kv.
// Internally pads logits leading dimension to even for __half2 alignment.
// logits buffer must have room for S*NH * (S_kv+1) elements when S_kv is odd.
void attention_qkv_fp16_padded(
    cublasHandle_t handle,
    const __half* Q,         // (S*NH, HD)
    const __half* K,         // (S_kv, HD)
    const __half* V,         // (S_kv, HD)
    __half* logits,          // scratch: (S*NH, S_kv_padded) where padded = S_kv rounded up to even
    __half* out,             // (S*NH, HD)
    int S, int S_kv, int NH, int HD,
    float attn_scale,
    cudaStream_t stream = 0);

// Single-call attention with state token masking for Pi0.
// State token (first 1 query) can only attend to the first `state_nk` keys.
// Remaining keys are masked with -inf before softmax.
// This replaces the split attention (2 calls) with 1 call + 1 mask kernel.
// Handles odd S_kv via padded lda (same as attention_qkv_fp16_padded).
void attention_qkv_fp16_state_masked(
    cublasHandle_t handle,
    const __half* Q,         // (S*NH, HD)
    const __half* K,         // (S_kv, HD)
    const __half* V,         // (S_kv, HD)
    __half* logits,          // scratch: (S*NH, S_kv_padded)
    __half* out,             // (S*NH, HD)
    int S, int S_kv, int NH, int HD,
    int state_nk,            // number of keys visible to state token (typically enc_seq+1)
    float attn_scale,
    cudaStream_t stream = 0);

// Single-call self-attention with ImageWAM's MoT joint block mask.
// Q/K/V already represent ONE combined sequence
// [prefix (text+ref) | target-image | action], laid out by the
// caller (this function does not concatenate anything) -- self-
// attention, so S == S_kv == total (the combined sequence length).
// Row group visibility (see softmax_mot_joint_fp16 for the exact
// rule): prefix sees [0,x0); target-image sees [0,a0); action sees
// [0,x0) U [a0,total) -- action never attends to target-image.
// Handles odd total via padded lda (same convention as
// attention_qkv_fp16_state_masked).
void attention_qkv_fp16_mot_joint(
    cublasHandle_t handle,
    const __half* Q,         // (total*NH, HD)
    const __half* K,         // (total, HD)
    const __half* V,         // (total, HD)
    __half* logits,          // scratch: (total*NH, total_padded)
    __half* out,             // (total*NH, HD)
    int total, int NH, int HD,
    int x0, int a0,          // block boundaries, see softmax_mot_joint_fp16
    float attn_scale,
    cudaStream_t stream = 0);

// Same joint attention, ACTION QUERIES ONLY (OPT-003, opportunities.md).
// Q covers just the action rows (num_action*NH), not the whole
// combined sequence -- during the real ImageWAM denoise loop the
// prefix/image rows never have a live query (their own Q was already
// consumed once during prefill), so attention_qkv_fp16_mot_joint above
// computes total*NH query rows to obtain only num_action*NH useful
// ones. K/V still cover the WHOLE [0, total) combined sequence (the
// action rows' visibility spans [0,x0) U [a0,total), which needs the
// full K/V resident either way) -- only Q shrinks. S != S_kv here:
// S = num_action, S_kv = total.
void attention_qkv_fp16_mot_joint_action(
    cublasHandle_t handle,
    const __half* Q,         // (num_action*NH, HD) -- action rows only
    const __half* K,         // (total, HD) -- full combined K
    const __half* V,         // (total, HD) -- full combined V
    __half* logits,          // scratch: (num_action*NH, total_padded)
    __half* out,             // (num_action*NH, HD)
    int num_action, int total, int NH, int HD,
    int x0, int a0,          // block boundaries, see softmax_mot_joint_action_fp16
    float attn_scale,
    cudaStream_t stream = 0);
