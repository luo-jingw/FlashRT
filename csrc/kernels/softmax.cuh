// ================================================================
// FlashRT — Softmax kernel (FP16)
// Port of pi05 softmax_bf16_kernel. In-place row-wise softmax.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void softmax_fp16(__half* data, int rows, int cols, cudaStream_t stream = 0);

// Causal softmax: per-head row-wise softmax that masks upper-triangular
// positions. Logits buffer is laid out as (NH * S_q, S_kv_pad) — for
// global row index ``r``, the per-head Q index is ``q = r % S_q`` and we
// mask cols ``j > q`` (strict upper-triangular). Also masks pad columns
// at ``j >= pad_start`` (= S_kv) regardless of row.
// Used by the causal LLM attention path in the GROOT N1.7 pipeline.
void softmax_causal_fp16(__half* data, int rows, int cols,
                          int S_q, int pad_start,
                          cudaStream_t stream = 0);

// Softmax with state token masking: first `mask_rows` rows have cols [mask_start, cols) set to -inf.
// Eliminates separate mask kernel launch. Used by Pi0 state-masked attention.
// mask_rows: first N rows get mask_start applied
// mask_start: state token's key limit (enc_seq+1)
// pad_start: actual S_kv before padding (for pad column masking on all rows)
void softmax_state_masked_fp16(__half* data, int rows, int cols,
                                int mask_rows, int mask_start, int pad_start,
                                cudaStream_t stream = 0);

// Softmax with ImageWAM's MoT joint-attention block mask: three row
// groups over [0, total) — prefix [0, x0), target-image [x0, a0),
// action [a0, total) — with visibility:
//   prefix row:       visible cols = [0, x0)
//   target-image row: visible cols = [0, a0)              (contiguous)
//   action row:       visible cols = [0, x0) U [a0, total) (NOT contiguous —
//                      action attends to prefix and itself but never to
//                      target-image)
// `total` is the real (unpadded) key count; `cols` is the padded
// column count the logits buffer was allocated/written with (same
// even-padding convention as softmax_state_masked_fp16 — columns in
// [total, cols) are masked on every row regardless of group).
// Rows are grouped by `NH` (heads) exactly like softmax_state_masked_fp16's
// own `mask_rows` convention: row index r belongs to query token
// q = r / NH, and q's group is determined by comparing q against x0/a0.
//
// NOTE: like every other kernel in this file, this is a single-warp-
// per-row reduction (SM_MAX_COLS = 1024 columns max, see softmax.cu).
// ImageWAM's real total sequence length (prefix + target-image patches
// + action tokens) has not yet been confirmed to fit under that limit
// for a real deployment image resolution — verify before relying on
// this kernel at production scale; a full image-patch count exceeding
// 1024 needs a block-level (not warp-level) reduction instead.
void softmax_mot_joint_fp16(__half* data, int rows, int cols,
                             int NH, int x0, int a0, int total,
                             cudaStream_t stream = 0);
