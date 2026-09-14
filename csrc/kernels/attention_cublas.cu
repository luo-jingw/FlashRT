// ================================================================
// FlashRT — cuBLAS attention (GQA-compatible)
// Direct port of pi05 engine attention pattern:
//   1. QK^T via cublasGemmEx (CUBLAS_OP_T, CUBLAS_OP_N)
//   2. softmax_fp16 (separate kernel)
//   3. PV via cublasGemmEx (CUBLAS_OP_N, CUBLAS_OP_N)
//
// Stateless: no static handles. Caller provides cublasHandle_t.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "softmax.cuh"

// Fill every `stride`-th element with -inf (fp16 = 0xFBFF = -65504)
// Used to mask pad columns in logits for odd-N attention.
__global__ void fill_neginf_strided_kernel(__half* data, int stride, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i * stride] = __float2half(-65504.0f);
}

void fill_neginf_strided_fp16(__half* data, int stride, int count, cudaStream_t stream) {
    fill_neginf_strided_kernel<<<(count + 255) / 256, 256, 0, stream>>>(data, stride, count);
}

// Complete attention: QK^T → softmax → PV
void attention_qkv_fp16(
    cublasHandle_t handle,
    const __half* Q,         // (S, NH*HD)
    const __half* K,         // (S_kv, HD) — single KV head
    const __half* V,         // (S_kv, HD)
    __half* logits,          // (S*NH, S_kv) scratch
    __half* out,             // (S*NH, HD) output = (S, NH, HD)
    int S, int S_kv, int NH, int HD,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    // Step 1: QK^T → logits
    // logits(S_kv, S*NH) = K^T(S_kv,HD)^T * Q(HD, S*NH) in col-major
    float zero = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv, S * NH, HD,
        &attn_scale,
        K, CUDA_R_16F, HD,
        Q, CUDA_R_16F, HD,
        &zero,
        logits, CUDA_R_16F, S_kv,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Step 2: softmax (in-place, per-row)
    softmax_fp16(logits, S * NH, S_kv, stream);

    // Step 3: PV → out
    // out(HD, S*NH) = V(HD, S_kv) * logits(S_kv, S*NH) in col-major
    float one = 1.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S * NH, S_kv,
        &one,
        V, CUDA_R_16F, HD,
        logits, CUDA_R_16F, S_kv,
        &zero,
        out, CUDA_R_16F, HD,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// Mask rows [seqused_k[0], S_kv_max) in every logits column.
// logits is col-major [S_kv_max, S*NH].
__global__ void mask_seqused_logits_kernel(
    __half* logits, int S_kv_max, int S_NH, const int* __restrict__ seqused_k)
{
    int valid = seqused_k[0];
    if (valid < 0) valid = 0;
    if (valid > S_kv_max) valid = S_kv_max;
    int mask_rows = S_kv_max - valid;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S_NH * mask_rows;
    if (idx >= total) return;
    int col = idx / mask_rows;
    int row = valid + (idx % mask_rows);
    logits[col * S_kv_max + row] = __float2half(-65504.0f);
}

void mask_seqused_logits_fp16(__half* logits, int S_kv_max, int S_NH,
                              const int* seqused_k, cudaStream_t stream) {
    // Launch even when valid == S_kv_max; the kernel exits with total=0.
    int total = S_NH * S_kv_max;
    if (total > 0) {
        mask_seqused_logits_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            logits, S_kv_max, S_NH, seqused_k);
    }
}

void attention_qkv_fp16_seqused(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int S, int S_kv_max, int NH, int HD,
    const int* seqused_k,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    float zero = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv_max, S * NH, HD,
        &attn_scale,
        K, CUDA_R_16F, HD,
        Q, CUDA_R_16F, HD,
        &zero,
        logits, CUDA_R_16F, S_kv_max,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    mask_seqused_logits_fp16(logits, S_kv_max, S * NH, seqused_k, stream);
    softmax_fp16(logits, S * NH, S_kv_max, stream);

    float one = 1.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S * NH, S_kv_max,
        &one,
        V, CUDA_R_16F, HD,
        logits, CUDA_R_16F, S_kv_max,
        &zero,
        out, CUDA_R_16F, HD,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}


// Combined mask kernel: pad column + state mask in one launch.
// logits is col-major [S_kv_pad, S_NH].
// 1. If has_pad: set row S_kv to -inf in ALL columns (pad column)
// 2. Set rows [state_nk, S_kv_pad) to -inf in first NH columns (state mask)
__global__ void mask_pad_and_state_kernel(
    __half* logits, int S_kv_pad, int S_kv,
    int S_NH, int NH, int state_nk, bool has_pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Part 1: pad column — S_NH elements
    int pad_count = has_pad ? S_NH : 0;
    if (idx < pad_count) {
        int col = idx;
        logits[col * S_kv_pad + S_kv] = __float2half(-65504.0f);
        return;
    }

    // Part 2: state mask — NH * (S_kv_pad - state_nk) elements
    int sidx = idx - pad_count;
    int num_mask_rows = S_kv_pad - state_nk;
    int state_count = NH * num_mask_rows;
    if (sidx < state_count) {
        int col = sidx / num_mask_rows;
        int row = state_nk + sidx % num_mask_rows;
        logits[col * S_kv_pad + row] = __float2half(-65504.0f);
    }
}

void mask_pad_and_state_fp16(__half* logits, int S_kv_pad, int S_kv,
                              int S_NH, int NH, int state_nk, bool has_pad,
                              cudaStream_t stream) {
    int pad_count = has_pad ? S_NH : 0;
    int state_count = NH * (S_kv_pad - state_nk);
    int total = pad_count + state_count;
    if (total > 0) {
        mask_pad_and_state_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            logits, S_kv_pad, S_kv, S_NH, NH, state_nk, has_pad);
    }
}

// Mask kernel: set logits[col][row] = -inf for state token's hidden columns.
// logits is col-major [S_kv_pad, S*NH] with ldc=S_kv_pad.
// State token occupies the first NH columns (cols 0..NH-1).
// We mask rows [state_nk, S_kv_pad) in those columns to -inf.
__global__ void mask_state_logits_kernel(
    __half* logits, int S_kv_pad, int NH,
    int state_nk, int S_kv)
{
    // Each thread handles one element to mask
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_mask_rows = S_kv_pad - state_nk;  // rows to mask per column
    int total = NH * num_mask_rows;
    if (idx >= total) return;
    int col = idx / num_mask_rows;       // which of the NH state columns
    int row = state_nk + idx % num_mask_rows;  // which row within that column
    logits[col * S_kv_pad + row] = __float2half(-65504.0f);
}

// Padded version: supports odd S_kv by using padded leading dimension.
// logits buffer must have room for S*NH * S_kv_pad elements.
void attention_qkv_fp16_padded(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int S, int S_kv, int NH, int HD,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    int S_kv_pad = S_kv + (S_kv & 1);  // round up to even

    // Step 1: QK^T → logits with padded ldc
    // logits col-major: (S_kv_pad, S*NH) with ldc=S_kv_pad
    float zero = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv, S * NH, HD,       // M=S_kv (actual), N=S*NH, K=HD
        &attn_scale,
        K, CUDA_R_16F, HD,      // lda=HD
        Q, CUDA_R_16F, HD,      // ldb=HD
        &zero,
        logits, CUDA_R_16F, S_kv_pad,  // ldc=S_kv_pad (padded!)
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Fill pad column with -inf so softmax gives it zero weight
    if (S_kv & 1) {
        int rows = S * NH;
        // logits col-major: [S_kv_pad, rows] with ldc=S_kv_pad
        // Pad element is at row index S_kv (last row) in each column
        // In memory: logits[S_kv + col * S_kv_pad] for each col
        // = contiguous block at offset S_kv, stride S_kv_pad, length rows
        // Use cudaMemset2DAsync with byte pattern 0xFB for fp16 -inf (0xFBFF)
        // Actually 0xFBFF byte-by-byte is {0xFF, 0xFB} in little-endian
        // cudaMemset2DAsync fills with a single byte value, can't do 2-byte pattern
        // Use a simple kernel instead:
        extern void fill_neginf_strided_fp16(__half* data, int stride, int count, cudaStream_t stream);
        fill_neginf_strided_fp16(logits + S_kv, S_kv_pad, rows, stream);
    }

    // Step 2: softmax with padded cols
    softmax_fp16(logits, S * NH, S_kv_pad, stream);

    // Step 3: PV → out with padded logits
    // out(HD, S*NH) = V(HD, S_kv) * logits(S_kv_pad, S*NH)
    // But V has S_kv rows and logits has S_kv_pad rows.
    // The pad row in logits has weight ~0 (from -inf), and we don't need V[S_kv_pad-1].
    // Use K=S_kv_pad but V needs padding too... or use K=S_kv and ignore pad weight.
    // Actually: logits pad column has ~0 weight, so PV result is dominated by valid columns.
    // But cuBLAS K dimension must match: V has S_kv rows, logits has S_kv_pad rows.
    // We must use K=S_kv for the PV GEMM, with logits ldb=S_kv_pad.
    float one = 1.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S * NH, S_kv,       // M=HD, N=S*NH, K=S_kv (actual)
        &one,
        V, CUDA_R_16F, HD,      // lda=HD
        logits, CUDA_R_16F, S_kv_pad,  // ldb=S_kv_pad (padded stride!)
        &zero,
        out, CUDA_R_16F, HD,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}


// State-masked attention: single call with mask for Pi0 state token.
// Combines QK^T + mask + softmax + PV in one function.
// State token (first 1 query, NH rows in logits) can only see first state_nk keys.
void attention_qkv_fp16_state_masked(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int S, int S_kv, int NH, int HD,
    int state_nk,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    int S_kv_pad = S_kv + (S_kv & 1);

    // Step 1: QK^T for ALL queries (S*NH) against ALL keys (S_kv)
    float zero = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv, S * NH, HD,
        &attn_scale,
        K, CUDA_R_16F, HD,
        Q, CUDA_R_16F, HD,
        &zero,
        logits, CUDA_R_16F, S_kv_pad,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Step 2: Fused softmax with state masking + pad handling (single kernel, no extra launch)
    // State rows [0, NH): mask cols [state_nk, S_kv_pad)
    // Action rows [NH, S*NH): mask cols [S_kv, S_kv_pad) (pad column only, if odd)
    softmax_state_masked_fp16(logits, S * NH, S_kv_pad,
                               NH, state_nk, S_kv, stream);

    // Step 4: PV
    float one = 1.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S * NH, S_kv,
        &one,
        V, CUDA_R_16F, HD,
        logits, CUDA_R_16F, S_kv_pad,
        &zero,
        out, CUDA_R_16F, HD,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}


// ImageWAM MoT joint-attention: self-attention over one combined
// [prefix | target-image | action] sequence with the block mask
// described in attention_cublas.cuh / softmax_mot_joint_fp16.
// Structurally identical to attention_qkv_fp16_state_masked above
// (QK^T -> masked softmax -> PV, one extra kernel launch for the
// mask+softmax fusion, same odd-length padding convention) --
// only the masking rule differs.
void attention_qkv_fp16_mot_joint(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int total, int NH, int HD,
    int x0, int a0,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    int total_pad = total + (total & 1);

    // Step 1: QK^T for ALL queries (total*NH) against ALL keys (total)
    float zero = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        total, total * NH, HD,
        &attn_scale,
        K, CUDA_R_16F, HD,
        Q, CUDA_R_16F, HD,
        &zero,
        logits, CUDA_R_16F, total_pad,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Step 2: fused block-masked softmax (also masks the pad column
    // via the `total` parameter, same as state_masked's pad_start).
    softmax_mot_joint_fp16(logits, total * NH, total_pad,
                            NH, x0, a0, total, stream);

    // Step 3: PV
    float one = 1.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, total * NH, total,
        &one,
        V, CUDA_R_16F, HD,
        logits, CUDA_R_16F, total_pad,
        &zero,
        out, CUDA_R_16F, HD,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}


// ImageWAM MoT joint-attention, ACTION QUERIES ONLY (OPT-003 fix).
// Structurally identical to attention_qkv_fp16_mot_joint above --
// same QK^T -> masked softmax -> PV composition -- except Q covers
// only num_action rows instead of total, and correspondingly this is
// NOT self-attention (S=num_action != S_kv=total): K/V still cover the
// full combined sequence.
void attention_qkv_fp16_mot_joint_action(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int num_action, int total, int NH, int HD,
    int x0, int a0,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    int total_pad = total + (total & 1);

    // Step 1: QK^T for action queries (num_action*NH) against ALL keys (total)
    float zero = 0.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        total, num_action * NH, HD,
        &attn_scale,
        K, CUDA_R_16F, HD,
        Q, CUDA_R_16F, HD,
        &zero,
        logits, CUDA_R_16F, total_pad,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Step 2: fused mask+softmax, single collapsed rule for every row
    // (see softmax_mot_joint_action_fp16).
    softmax_mot_joint_action_fp16(logits, num_action * NH, total_pad,
                                   x0, a0, total, stream);

    // Step 3: PV
    float one = 1.0f;
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, num_action * NH, total,
        &one,
        V, CUDA_R_16F, HD,
        logits, CUDA_R_16F, total_pad,
        &zero,
        out, CUDA_R_16F, HD,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}


// ================================================================
// Real per-head K/V attention (OPT-002) -- see attention_cublas.cuh
// for the full layout convention. Both kernels below batch over the
// NH head dimension via cublasGemmStridedBatchedEx: for a fixed head
// h, a token's data sits at base offset h*HD within that token's own
// NH*HD-wide row, so the per-head "matrix" for QK^T/PV has leading
// dimension NH*HD (the stride between consecutive tokens) and batch
// stride HD (the offset from head h to head h+1 within one token).
// The `logits` buffer keeps its existing (row=tok*NH+head, col=key)
// layout, so head h's own logits sub-matrix has leading dimension
// NH*total_pad (stride between consecutive query tokens' rows for the
// SAME head) and batch stride total_pad (offset from head h to h+1).
// ================================================================

void attention_qkv_fp16_perhead(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int S, int S_kv, int NH, int HD,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    int S_kv_pad = S_kv + (S_kv & 1);
    long long strideQK = HD;              // head h -> h+1 within one token
    long long strideLogits = S_kv_pad;    // head h -> h+1 within one query row-block

    // Step 1: QK^T per head, batched over NH.
    // logits_h(S_kv, S) = K_h^T(HD,S_kv)^T * Q_h(HD,S), col-major.
    float zero = 0.0f;
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        S_kv, S, HD,
        &attn_scale,
        K, CUDA_R_16F, NH * HD, strideQK,
        Q, CUDA_R_16F, NH * HD, strideQK,
        &zero,
        logits, CUDA_R_16F, NH * S_kv_pad, strideLogits,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (S_kv & 1) {
        // Pad column at row S_kv in EVERY one of the S*NH query rows --
        // same fill kernel already used by attention_qkv_fp16_padded,
        // just over the full S*NH row count (head-interleaving doesn't
        // change which rows need the pad fill).
        fill_neginf_strided_fp16(logits + S_kv, S_kv_pad, S * NH, stream);
    }

    // Step 2: softmax (in-place, per-row) -- unchanged, row layout is
    // identical to the broadcast-K/V kernels.
    softmax_fp16(logits, S * NH, S_kv_pad, stream);

    // Step 3: PV per head, batched over NH.
    // out_h(HD, S) = V_h(HD,S_kv) * logits_h(S_kv,S), col-major.
    float one = 1.0f;
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, S, S_kv,
        &one,
        V, CUDA_R_16F, NH * HD, strideQK,
        logits, CUDA_R_16F, NH * S_kv_pad, strideLogits,
        &zero,
        out, CUDA_R_16F, NH * HD, strideQK,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}


void attention_qkv_fp16_mot_joint_action_perhead(
    cublasHandle_t handle,
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* logits,
    __half* out,
    int num_action, int total, int NH, int HD,
    int x0, int a0,
    float attn_scale,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);

    int total_pad = total + (total & 1);
    long long strideQK = HD;
    long long strideLogits = total_pad;

    // Step 1: QK^T for action queries, batched over NH heads.
    float zero = 0.0f;
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        total, num_action, HD,
        &attn_scale,
        K, CUDA_R_16F, NH * HD, strideQK,
        Q, CUDA_R_16F, NH * HD, strideQK,
        &zero,
        logits, CUDA_R_16F, NH * total_pad, strideLogits,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Step 2: same fused mask+softmax as the broadcast-K/V version --
    // row layout (tok*NH+head) is unchanged, mask is a function of
    // token position only.
    softmax_mot_joint_action_fp16(logits, num_action * NH, total_pad,
                                   x0, a0, total, stream);

    // Step 3: PV, batched over NH heads.
    float one = 1.0f;
    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        HD, num_action, total,
        &one,
        V, CUDA_R_16F, NH * HD, strideQK,
        logits, CUDA_R_16F, NH * total_pad, strideLogits,
        &zero,
        out, CUDA_R_16F, NH * HD, strideQK,
        NH,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
