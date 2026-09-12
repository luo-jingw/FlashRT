"""ImageWAM (FLUX.2-4B variant) Thor compute path — backbone prefill.

plan.md Phase 3. Structural dry-run scope only (see PROJECT.md /
plan.md): random-initialized weights, BF16/FP16 only, no FP8, no
calibration, no real checkpoint. Goal is a populated, correctly-shaped
KV cache with no NaN/Inf, not accuracy against a trained model.

# Simplifications versus the real FLUX.2-klein-4B / ImageWAM backbone
=======================================================================

Documented explicitly here (not silently assumed) because each one
would need to be revisited before this pipeline could produce
accurate output against a real checkpoint. None of them affect
correctness of *this* phase's stated goal (finite output, correct
pointer-interface, correct shapes).

1. **Unweighted (unit-scale) RMS norm only, no AdaLN modulation.**
   `_imagewam_thor_spec.py` (Phase 1) declares no norm/modulation
   weight tensors at all, so there is no per-layer learnable norm
   scale to load. This pipeline still calls `fvk.rms_norm_fp16` before
   every attention and MLP sub-block, passing a single shared,
   all-ones "weight" buffer (`bufs["norm_ones"]`) instead of a real
   learned scale -- true normalization (bounds activation magnitude),
   just with an always-1.0 elementwise gain. This was not optional:
   an earlier version of this file had NO normalization at all, and
   the Phase 3 wiring test caught real NaN/Inf after only a few
   layers -- unbounded residual growth through repeated random-weight
   GEMMs in FP16 overflows quickly with no norm anywhere, regardless
   of random-vs-real weights. Real FLUX double/single-stream blocks
   additionally apply per-block AdaLN shift/scale/gate from a
   timestep+text conditioning vector; that part is still not modeled.

2. **Single-shared K/V per position, not real per-head MHA.** The
   `attention_qkv_fp16` / `attention_qkv_fp16_mot_joint` kernels this
   pipeline calls through `ImageWAMAttnBackend` both take K/V as a
   single `(seq, HD)` buffer broadcast across all `NH` query heads
   (confirmed by reading `csrc/kernels/attention_cublas.cu` directly —
   this is the same shape `attention_qkv_fp16_state_masked` already
   uses for Pi0.5's own GQA `num_kv_heads=1` sites). Real FLUX/DiT
   attention uses full per-head K/V (`num_kv_heads == num_q_heads`).
   This pipeline's own K/V projection weights are therefore declared
   at `HD` width (128), not `hidden` width (3072) like the real
   checkpoint's fused QKV tensor — a genuine architectural
   simplification, not just a random-vs-real-weight difference.
   Tracked as a follow-up in `opportunities.md` (a real per-head-K/V
   kernel is a separate, substantial CUDA task).

3. **No persistent text-stream residual across double-stream layers.**
   `_imagewam_thor_spec.py` declares one `txt_in` weight PER
   double-stream layer (matching the real checkpoint's tensor names).
   This pipeline re-derives the text stream fresh from raw `context`
   at every double-stream layer via that layer's own `txt_in`,
   discarding the previous layer's text-side attention+MLP update
   rather than carrying it forward as a residual. The image stream
   does not have this limitation — it keeps one true residual across
   all 5 double-stream layers.

# Weight pointer keys this file expects (post-split, not the raw
# checkpoint-shaped WEIGHT_SPEC declared in Phase 1)
=======================================================================

Phase 1's `_imagewam_thor_spec.py` declares checkpoint-shaped tensors
(a fused `img_attn.qkv.weight`, etc.) for real-checkpoint-loading
compatibility. This pipeline instead consumes already-split,
already-transposed-to-(K,N) pointers (splitting a fused QKV tensor
into separate Q/K/V matrices, and transposing PyTorch's `(out, in)`
convention to `(in, out)` for `GemmRunner.fp16_nn`, are both one-time
operations that belong in the frontend's weight-loading step — see
`CosmosEdgeThor.__init__`'s own `.t().contiguous()` for the established
precedent). Phase 5 (frontend) is responsible for producing these keys
from Phase 1's declared shapes (real load) or by direct random-fill
(this stage). Keys, all `weights[("backbone", stream, layer, slot)]`:

  stream="double", layer in [0, 5):
    txt_in (joint_attention_dim, hidden), txt_q/txt_proj (hidden, hidden),
    txt_k/txt_v (hidden, HD), txt_mlp0 (hidden, mlp_hidden),
    txt_mlp2 (mlp_hidden, hidden), and the img_* analogs of all but txt_in.

  stream="single", layer in [0, 20):
    q/attn_out_proj (hidden, hidden), k/v (hidden, HD),
    mlp_in (hidden, mlp_hidden), mlp_down (mlp_hidden, hidden).

`bufs` keys (pipeline-owned scratch, pre-allocated once by the
frontend, fp16 throughout):
    context           (max_txt_seq, joint_attention_dim)  -- input
    backbone_hidden   (a0, hidden)   -- the persistent residual;
                                         rows [0,x0) = text, [x0,a0) = image
    normed_scratch    (a0, hidden)   -- pre-norm landing pad, reused by
                                         every attention/MLP sub-block
    norm_ones         (hidden,)      -- all-1.0, the unweighted RMS norm
                                         "weight" every call shares
    txt_mlp_hidden    (x0, mlp_hidden)
    img_mlp_hidden    (a0 - x0, mlp_hidden)
    single_mlp_hidden (a0, mlp_hidden)
    proj_scratch      (a0, hidden)   -- GEMM output landing pad before
                                         the residual_add_fp16 accumulate
"""
from __future__ import annotations


def _ptr_offset(base_ptr: int, row_offset: int, row_width: int) -> int:
    """Byte offset into a row-major fp16 buffer -- 2 bytes/element."""
    return int(base_ptr) + int(row_offset) * int(row_width) * 2


def _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, stream, attn):
    """One FLUX.2 double-stream block: separate img/txt QKV+MLP, joint attn.

    Reads/writes `bufs["backbone_hidden"]` rows [0,x0) (text) and
    [x0,a0) (image) in place.
    """
    hidden = dims["hidden"]
    HD = dims["HD"]
    mlp_hidden = dims["mlp_hidden"]
    joint_attention_dim = dims["joint_attention_dim"]
    x0 = dims["x0"]
    a0 = dims["a0"]
    img_len = a0 - x0

    combined = bufs["backbone_hidden"]  # (a0, hidden)
    normed = bufs["normed_scratch"]     # (a0, hidden) scratch
    ones = bufs["norm_ones"]
    eps = 1e-6
    key = lambda slot: weights[("backbone", "double", layer_idx, slot)]

    ptrs = attn.get_slot_ptrs("backbone", layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]

    # --- text stream: freshly re-derived from raw context (see module
    # docstring simplification #3), overwrites rows [0,x0) of combined ---
    gemm.fp16_nn(bufs["context"], key("txt_in"), combined, x0, hidden, joint_attention_dim, stream)
    txt_x = combined  # rows [0, x0)
    txt_normed = normed

    fvk.rms_norm_fp16(txt_x, ones, txt_normed, x0, hidden, eps, stream)
    gemm.fp16_nn(txt_normed, key("txt_q"), Q_O, x0, hidden, hidden, stream)
    gemm.fp16_nn(txt_normed, key("txt_k"), K_cache, x0, HD, hidden, stream)
    gemm.fp16_nn(txt_normed, key("txt_v"), V_cache, x0, HD, hidden, stream)

    # --- image stream: persistent residual, rows [x0,a0) of combined ---
    img_x_ptr = _ptr_offset(combined, x0, hidden)
    img_normed_ptr = _ptr_offset(normed, x0, hidden)
    img_Q_ptr = _ptr_offset(Q_O, x0, hidden)
    img_K_ptr = _ptr_offset(K_cache, x0, HD)
    img_V_ptr = _ptr_offset(V_cache, x0, HD)
    fvk.rms_norm_fp16(img_x_ptr, ones, img_normed_ptr, img_len, hidden, eps, stream)
    gemm.fp16_nn(img_normed_ptr, key("img_q"), img_Q_ptr, img_len, hidden, hidden, stream)
    gemm.fp16_nn(img_normed_ptr, key("img_k"), img_K_ptr, img_len, HD, hidden, stream)
    gemm.fp16_nn(img_normed_ptr, key("img_v"), img_V_ptr, img_len, HD, hidden, stream)

    # --- joint self-attention over the whole [text | image] sequence ---
    attn.run("backbone", layer_idx, q_seq=a0, stream=stream)
    # Output lands back at Q_O, split by the same row ranges.

    # --- separate output projections, residual-accumulated ---
    proj = bufs["proj_scratch"]
    gemm.fp16_nn(Q_O, key("txt_proj"), proj, x0, hidden, hidden, stream)
    fvk.residual_add_fp16(txt_x, proj, x0 * hidden, stream)

    img_proj_ptr = _ptr_offset(proj, x0, hidden)
    gemm.fp16_nn(img_Q_ptr, key("img_proj"), img_proj_ptr, img_len, hidden, hidden, stream)
    fvk.residual_add_fp16(img_x_ptr, img_proj_ptr, img_len * hidden, stream)

    # --- separate MLPs, residual-accumulated ---
    fvk.rms_norm_fp16(txt_x, ones, txt_normed, x0, hidden, eps, stream)
    txt_mlp = bufs["txt_mlp_hidden"]
    gemm.fp16_nn(txt_normed, key("txt_mlp0"), txt_mlp, x0, mlp_hidden, hidden, stream)
    fvk.gelu_inplace_fp16(txt_mlp, x0 * mlp_hidden, stream)
    gemm.fp16_nn(txt_mlp, key("txt_mlp2"), proj, x0, hidden, mlp_hidden, stream)
    fvk.residual_add_fp16(txt_x, proj, x0 * hidden, stream)

    fvk.rms_norm_fp16(img_x_ptr, ones, img_normed_ptr, img_len, hidden, eps, stream)
    img_mlp = bufs["img_mlp_hidden"]
    gemm.fp16_nn(img_normed_ptr, key("img_mlp0"), img_mlp, img_len, mlp_hidden, hidden, stream)
    fvk.gelu_inplace_fp16(img_mlp, img_len * mlp_hidden, stream)
    gemm.fp16_nn(img_mlp, key("img_mlp2"), img_proj_ptr, img_len, hidden, mlp_hidden, stream)
    fvk.residual_add_fp16(img_x_ptr, img_proj_ptr, img_len * hidden, stream)


def _single_stream_layer(ctx, fvk, gemm, bufs, weights, dims, weight_layer_idx,
                          site_layer_idx, stream, attn):
    """One FLUX.2 single-stream block: merged img+txt, fused QKV+MLP-in.

    Operates on the whole `bufs["backbone_hidden"]` (a0, hidden) buffer.
    ``weight_layer_idx`` (0..19) indexes this stream's own declared
    weights; ``site_layer_idx`` (continues after the double-stream
    layers) indexes the "backbone" attention site's shared 25-layer KV
    cache -- these are deliberately different counters, see
    ``imagewam_prefill``.
    """
    hidden = dims["hidden"]
    HD = dims["HD"]
    mlp_hidden = dims["mlp_hidden"]
    a0 = dims["a0"]

    combined = bufs["backbone_hidden"]
    normed = bufs["normed_scratch"]
    ones = bufs["norm_ones"]
    eps = 1e-6
    key = lambda slot: weights[("backbone", "single", weight_layer_idx, slot)]

    ptrs = attn.get_slot_ptrs("backbone", site_layer_idx)
    Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]

    # One norm feeds Q/K/V and mlp_in alike -- matches real FLUX single-
    # stream blocks, which apply one fused `linear1` GEMM to one
    # normalized input; this pipeline only split that GEMM into four
    # separate ones (see module docstring's weight-key convention).
    fvk.rms_norm_fp16(combined, ones, normed, a0, hidden, eps, stream)
    gemm.fp16_nn(normed, key("q"), Q_O, a0, hidden, hidden, stream)
    gemm.fp16_nn(normed, key("k"), K_cache, a0, HD, hidden, stream)
    gemm.fp16_nn(normed, key("v"), V_cache, a0, HD, hidden, stream)

    mlp = bufs["single_mlp_hidden"]
    gemm.fp16_nn(normed, key("mlp_in"), mlp, a0, mlp_hidden, hidden, stream)
    fvk.gelu_inplace_fp16(mlp, a0 * mlp_hidden, stream)

    attn.run("backbone", site_layer_idx, q_seq=a0, stream=stream)

    proj = bufs["proj_scratch"]
    gemm.fp16_nn(Q_O, key("attn_out_proj"), proj, a0, hidden, hidden, stream)
    fvk.residual_add_fp16(combined, proj, a0 * hidden, stream)

    gemm.fp16_nn(mlp, key("mlp_down"), proj, a0, hidden, mlp_hidden, stream)
    fvk.residual_add_fp16(combined, proj, a0 * hidden, stream)


def imagewam_encode_once(ctx, fvk, gemm, bufs, weights, dims, stream=0):
    """Target-image encode step -- explicit no-op in this scope.

    ImageWAM's real encode step is a VAE forward
    (`_encode_flux2_image_tokens`); no VAE weights are declared (out of
    scope -- see `_imagewam_thor_spec.py` and `opportunities.md`
    OPT-001). The frontend fills `bufs["backbone_hidden"]`'s image rows
    `[x0, a0)` directly with random data once, standing in for
    already-encoded image patch tokens, so there is nothing left for
    this function to compute. Kept as its own pipeline-stage function
    (matching the Interface in plan.md) so a real VAE integration has
    an unambiguous place to go later, rather than folding "encode" into
    "prefill" silently.
    """
    return


def imagewam_prefill(ctx, fvk, gemm, bufs, weights, dims, stream=0, *, attn=None):
    """One backbone forward (5 double-stream + 20 single-stream layers)
    over the [prefix | target-image] sequence, populating the per-layer
    KV cache the later denoise loop (Phase 4) reads through the "mot"
    attention site.

    Required `dims` keys: hidden, HD, NH, mlp_hidden, joint_attention_dim,
    x0, a0, num_layers_double, num_layers_single.

    `gemm` must be a `fvk.GemmRunner` constructed once by the caller
    (frontend, Phase 5) outside any CUDA-graph-capturable region --
    `GemmRunner()` does `cudaMalloc` for a 256MB workspace at
    construction (see `docs/adding_new_model.md`'s own pointer-interface
    contract example, which threads `gemm` in the same way). An earlier
    draft of this function constructed a fresh `GemmRunner()` inside
    each per-layer helper (25 times per call) -- wrong on two counts:
    it allocates inside what must become a graph-capturable region, and
    it produced a real `cublasLtMatmul` internal error at runtime on
    this machine's 8GB GPU (found by running the Phase 3 wiring test,
    not by inspection).
    """
    if attn is None:
        raise ValueError("imagewam_prefill requires an ImageWAMAttnBackend via attn=")
    num_double = dims["num_layers_double"]
    for layer_idx in range(num_double):
        _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, layer_idx, stream, attn)
    # The "backbone" attention site's per-layer KV cache is ONE
    # contiguous 25-layer range (num_layers_double + num_layers_single);
    # single-stream layers continue that same indexing rather than
    # restarting at 0, which would otherwise alias double-stream layer
    # 0..4's own K/V cache slots.
    for i in range(dims["num_layers_single"]):
        _single_stream_layer(ctx, fvk, gemm, bufs, weights, dims, i, num_double + i, stream, attn)
