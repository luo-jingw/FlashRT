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

from flash_rt.models.imagewam.adaln import mlp_embedder, modulation, timestep_embedding_real
from flash_rt.models.imagewam.real_double_stream_block import real_double_stream_block_forward_fp16
from flash_rt.models.imagewam.real_single_stream_block import real_single_stream_block_forward_fp16

DEV = "cuda"
FP16 = torch.float16


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
) -> torch.Tensor:
    """Real 25-layer backbone prefill (5 double + 20 single). Returns
    the final combined [txt | img] sequence (total, hidden).

    `double_layer_weights[i]` / `single_layer_weights[i]`: per-layer
    weight dicts, see `real_double_stream_block_forward_fp16` /
    `real_single_stream_block_forward_fp16` for the required keys.
    Modulation is SHARED across all layers of a given stream type
    (real architecture property, not a simplification) -- computed
    once via `compute_shared_modulation` and passed in here.
    """
    x0 = txt.shape[0]
    for w in double_layer_weights:
        txt, img = real_double_stream_block_forward_fp16(
            gemm, ctx, txt, img, w, mod_double_txt, mod_double_img,
            rope_table, NH, HD, hidden, mlp_hidden, attn_scale)

    combined = torch.cat([txt, img], dim=0).contiguous()
    for w in single_layer_weights:
        combined = real_single_stream_block_forward_fp16(
            gemm, ctx, combined, w, mod_single,
            rope_table, NH, HD, hidden, mlp_hidden, x0, attn_scale)

    return combined
