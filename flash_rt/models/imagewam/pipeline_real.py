"""ImageWAM real backbone prefill: loops the verified real
DoubleStreamBlock/SingleStreamBlock forwards across all 25 real
backbone layers (opportunities.md OPT-002).

Real per-layer structure, confirmed from `Flux2.forward`
(`black-forest-labs/flux2`'s `src/flux2/model.py`, pinned commit
`50fe5162777813d869182b139e83b10743caef15`, read directly):

    vec = time_in(timestep_embedding(timestep))
    mod_txt = double_stream_modulation_txt(vec)   # SHARED across all 5 double layers
    mod_img = double_stream_modulation_img(vec)   # SHARED across all 5 double layers
    mod_single = single_stream_modulation(vec)    # SHARED across all 20 single layers
    for block in double_blocks: img, txt = block.forward_kv_extract(img, txt, ..., mod_img, mod_txt)
    combined = cat([txt, img])
    for block in single_blocks: combined = block.forward_kv_extract(combined, ..., mod_single)

The modulation values are computed ONCE from a single per-forward
timestep and REUSED across every layer of a given stream type -- only
the layers' own GEMM weights (qkv/proj/mlp/query_norm/key_norm) differ
per layer. This module is a NEW, additive real-math pipeline path --
it does NOT modify or replace `flash_rt/models/imagewam/pipeline_thor.py`
(the existing, already-tested, broadcast-K/V approximate pipeline every
current benchmark script and the registered `_PIPELINE_MAP` frontend
use). Kept fully separate to carry zero risk to what already works,
matching this project's own standing convention for every other
real-math addition this round (opt-in flags, new modules, never in-
place replacement) -- see opportunities.md for what remains before
this could become the default.

**Correctness verification only, not a steady-state performance
path**: every buffer inside each block call is freshly allocated
(`torch.zeros(...)`), with no pre-allocated scratch reuse across layers
or across repeated calls -- consistent with this module's scope (does
the real math compute the right thing at all, run once), not with the
`imagewam_thor_*_bench.py` scripts' own steady-state-loop convention.
On this dev machine's tight 8GB budget, running the full 25-layer real
backbone once from a near-empty GPU works fine, but running it
repeatedly in a tight loop without buffer reuse risks real memory
pressure (observed directly: one run failed with a CUDA OOM when the
GPU already had ~5.3GB in use from unrelated earlier processes in the
same session, and succeeded cleanly once that was cleared). Turning
this into a real perf-comparable path would need the same
allocate-once-during-warmup discipline every benchmark script already
uses.
"""
from __future__ import annotations

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.adaln import head_modulation, mlp_embedder, modulation, timestep_embedding_real
from flash_rt.models.imagewam.real_action_expert import (
    real_action_double_block_forward_fp16,
    real_action_single_block_forward_fp16,
)
from flash_rt.models.imagewam.real_double_stream_block import real_double_stream_block_forward_fp16
from flash_rt.models.imagewam.real_single_stream_block import real_single_stream_block_forward_fp16

DEV = "cuda"
FP16 = torch.float16
F32 = torch.float32


def compute_shared_modulation(timestep: torch.Tensor, weights: dict, hidden: int):
    """weights keys: `time_in_w1`, `time_in_w2` (MLPEmbedder), `mod_double_txt`,
    `mod_double_img` (each (6*hidden,hidden), double=True), `mod_single`
    ((3*hidden,hidden), double=False). Returns (mod_txt, mod_img, mod_single)."""
    emb = timestep_embedding_real(timestep)
    vec = mlp_embedder(emb, weights["time_in_w1"], weights["time_in_w2"])
    mod_txt = modulation(vec, weights["mod_double_txt"], double=True)
    mod_img = modulation(vec, weights["mod_double_img"], double=True)
    mod_single, _ = modulation(vec, weights["mod_single"], double=False)
    return mod_txt, mod_img, mod_single


def imagewam_prefill_real(
    gemm, ctx,
    txt: torch.Tensor, img: torch.Tensor,
    double_layer_weights: list[dict], single_layer_weights: list[dict],
    mod_double_txt: tuple, mod_double_img: tuple, mod_single: tuple,
    rope_table: torch.Tensor,
    NH: int, HD: int, hidden: int, mlp_hidden: int,
    attn_scale: float,
    *, collect_kv_cache: bool = False,
):
    """Real 25-layer backbone prefill (5 double + 20 single). Returns
    the final combined [txt | img] sequence (total, hidden).

    `double_layer_weights[i]` / `single_layer_weights[i]`: per-layer
    weight dicts, see `real_double_stream_block_forward_fp16` /
    `real_single_stream_block_forward_fp16` for the required keys.
    Modulation is SHARED across all layers of a given stream type
    (real architecture property, not a simplification) -- computed
    once via `compute_shared_modulation` and passed in here.

    `collect_kv_cache`: when True, also returns a 25-entry list of this
    forward's own per-layer, post-QKNorm+RoPE `(K, V)` (each
    `(x0+img_len, NH, HD)`) -- the FROZEN cache
    `imagewam_full_forward_real`'s action-expert denoise loop reads via
    `real_action_double_block_forward_fp16`/`real_action_single_block_forward_fp16`'s
    own `cached_k`/`cached_v` parameters, in the same double-then-single
    layer order used everywhere else in this module. Returns
    `(combined, kv_cache)` instead of `combined` when set; default
    False keeps every existing caller unaffected.
    """
    x0 = txt.shape[0]
    kv_cache = [] if collect_kv_cache else None
    for w in double_layer_weights:
        if collect_kv_cache:
            txt, img, K, V = real_double_stream_block_forward_fp16(
                gemm, ctx, txt, img, w, mod_double_txt, mod_double_img,
                rope_table, NH, HD, hidden, mlp_hidden, attn_scale, return_kv=True)
            kv_cache.append((K, V))
        else:
            txt, img = real_double_stream_block_forward_fp16(
                gemm, ctx, txt, img, w, mod_double_txt, mod_double_img,
                rope_table, NH, HD, hidden, mlp_hidden, attn_scale)

    combined = torch.cat([txt, img], dim=0).contiguous()
    for w in single_layer_weights:
        if collect_kv_cache:
            combined, K, V = real_single_stream_block_forward_fp16(
                gemm, ctx, combined, w, mod_single,
                rope_table, NH, HD, hidden, mlp_hidden, attn_scale, return_kv=True)
            kv_cache.append((K, V))
        else:
            combined = real_single_stream_block_forward_fp16(
                gemm, ctx, combined, w, mod_single,
                rope_table, NH, HD, hidden, mlp_hidden, attn_scale)

    if collect_kv_cache:
        return combined, kv_cache
    return combined


def compute_action_head_modulation(timestep: torch.Tensor, weights: dict, hidden: int):
    """OPT-001: `Flux2ActionHead`'s own AdaLN modulation (shift, scale
    only, no gate -- see `head_modulation`'s own docstring), computed
    from the SAME `vec` `compute_action_modulation` derives, but kept
    as its own small standalone function rather than a 3rd return value
    there -- `compute_action_modulation`'s existing 2-tuple return is a
    shared dependency of `test_imagewam_pipeline_full_real.py` (which
    has no use for head's own modulation, that test's own scope is
    `imagewam_full_forward_real`'s already-documented action_hidden_dim-
    width placeholder, unrelated to this fix); recomputing `vec` here
    is trivial (see adaln.py's own docstring: "a handful of KB...
    negligible") and keeps that signature untouched.
    `weights` keys: `time_in_w1`, `time_in_w2` (SAME ActionDiT `time_in`
    weights `compute_action_modulation` already needs), `head_adaln`
    (`(2*hidden,hidden)`, `Flux2ActionHead.adaLN_modulation`'s own
    Linear weight). Returns `(shift, scale)`."""
    emb = timestep_embedding_real(timestep)
    vec = mlp_embedder(emb, weights["time_in_w1"], weights["time_in_w2"])
    return head_modulation(vec, weights["head_adaln"])


def compute_action_modulation(timestep: torch.Tensor, weights: dict, hidden: int):
    """Same shape as `compute_shared_modulation`, for ActionDiT's own
    weights. `weights` keys: `time_in_w1`, `time_in_w2` (ActionDiT's
    own `time_in` MLPEmbedder -- a SEPARATE weight set from the
    backbone's, confirmed from `mot.py`: each expert owns its own
    `time_in`), `mod_double` (`(6*hidden,hidden)`, double=True --
    ActionDiT's double block is img-only, so there is only ONE
    double-type modulation here, unlike the backbone's separate
    txt/img pair), `mod_single` (`(3*hidden,hidden)`, double=False).
    Returns `(mod_double, mod_single)`.
    """
    emb = timestep_embedding_real(timestep)
    vec = mlp_embedder(emb, weights["time_in_w1"], weights["time_in_w2"])
    mod_double = modulation(vec, weights["mod_double"], double=True)
    mod_single, _ = modulation(vec, weights["mod_single"], double=False)
    return mod_double, mod_single


def imagewam_full_forward_real(
    gemm, ctx,
    txt: torch.Tensor, img: torch.Tensor, action_latent: torch.Tensor,
    backbone_double_weights: list[dict], backbone_single_weights: list[dict],
    backbone_shared_weights: dict,
    action_double_weights: list[dict], action_single_weights: list[dict],
    action_shared_weights: dict,
    backbone_rope_table: torch.Tensor, action_rope_table: torch.Tensor,
    NH: int, HD: int, hidden: int, mlp_hidden: int,
    action_hidden: int, action_mlp_hidden: int,
    attn_scale: float,
    num_denoise_steps: int,
    *, backbone_timestep: float = 0.0,
) -> torch.Tensor:
    """The FULL real ImageWAM forward: one backbone prefill (producing
    the frozen per-layer K/V cache) followed by the whole ActionDiT
    flow-matching denoise loop, reading that cache through the real
    (unmasked -- opportunities.md's mask correction) joint attention.
    Extends `imagewam_prefill_real`'s own backbone-only scope to the
    complete model, still a **correctness-verification path, not a
    steady-state performance path** (same caveat as every function in
    this module -- fresh buffer allocation every call, no CUDA Graph,
    no `pipeline_thor.py` wiring; see opportunities.md OPT-002 for what
    a real steady-state port still needs).

    Real per-forward structure, following `imagewam_prefill_real`'s own
    backbone loop plus `mot.py`'s `forward_flux2_action_with_video_cache`
    called once per denoise step (`infer_action_flux2`'s own loop):

        combined, kv_cache = imagewam_prefill_real(..., collect_kv_cache=True)
        for step in range(num_denoise_steps):
            action_timestep = 1.0 - step / num_denoise_steps   # flow-matching schedule
            mod_double, mod_single = compute_action_modulation(action_timestep, ...)
            action = action_latent (cast to fp16)
            for i, w in enumerate(action_double_weights):
                action = real_action_double_block_forward_fp16(..., cached_k=kv_cache[i][0], cached_v=kv_cache[i][1], ...)
            for i, w in enumerate(action_single_weights):
                action = real_action_single_block_forward_fp16(..., cached_k=kv_cache[5+i][0], cached_v=kv_cache[5+i][1], ...)
            action_latent = action_latent + dt * action   # Euler step, dt = 1/num_denoise_steps

    **Two structural approximations, both already established elsewhere
    in this codebase (not new to this function), documented explicitly**:
    (1) the backbone's own conditioning timestep is fixed at
    ``backbone_timestep=0.0`` (matching the exact value used by
    `benchmarks/imagewam_real_checkpoint_validation.py`'s real-checkpoint
    run, where `video_timestep = torch.zeros(1)` -- confirmed correct
    there, cosine=0.999927 against the real reference; a real backbone
    forward conditioned on a DIFFERENT timestep is not validated here);
    (2) ActionDiT's own raw `action_latent` is fed directly into its
    transformer blocks with no `action_encoder` (a real
    `Linear(action_dim, hidden_dim)` WITH bias that projects raw,
    small-width action values up to `action_hidden` first) --
    `pipeline_thor.py`'s own dry-run pipeline makes this identical
    simplification (`action_latent` allocated directly at
    `action_hidden_dim` width, see its own module docstring); a real
    action-dim projection head is real-checkpoint-dependent work, same
    status as OPT-001.

    `backbone_shared_weights`/`action_shared_weights`: see
    `compute_shared_modulation`/`compute_action_modulation` for the
    required keys.

    Returns the final `action_latent` (float32, `(num_action,
    action_hidden)`, the flow-matching output).
    """
    mod_txt, mod_img, mod_single_bb = compute_shared_modulation(
        torch.full((1,), backbone_timestep, dtype=F32, device=DEV), backbone_shared_weights, hidden)
    combined, kv_cache = imagewam_prefill_real(
        gemm, ctx, txt, img, backbone_double_weights, backbone_single_weights,
        mod_txt, mod_img, mod_single_bb, backbone_rope_table,
        NH, HD, hidden, mlp_hidden, attn_scale, collect_kv_cache=True)
    del combined  # only the per-layer kv_cache is read by the denoise loop below

    num_action = action_latent.shape[0]
    num_double = len(action_double_weights)
    dt = 1.0 / num_denoise_steps
    for step in range(num_denoise_steps):
        action_timestep = 1.0 - step * dt
        mod_double, mod_single = compute_action_modulation(
            torch.full((1,), action_timestep, dtype=F32, device=DEV), action_shared_weights, action_hidden)

        action = action_latent.to(FP16)
        for i, w in enumerate(action_double_weights):
            K, V = kv_cache[i]
            action = real_action_double_block_forward_fp16(
                gemm, action, w, mod_double, action_rope_table, K, V,
                NH, HD, action_hidden, action_mlp_hidden, attn_scale)
        for i, w in enumerate(action_single_weights):
            K, V = kv_cache[num_double + i]
            action = real_action_single_block_forward_fp16(
                gemm, action, w, mod_single, action_rope_table, K, V,
                NH, HD, action_hidden, action_mlp_hidden, attn_scale)

        action_latent = action_latent + dt * action.float()

    return action_latent
