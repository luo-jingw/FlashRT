"""Real ImageWAM checkpoint loading (OPT-001, `plan.md`'s own "OPT-001"
plan, Phase 2).

**Reads the checkpoint's raw `state_dict` by KEY NAME, not by
constructing the live `ImageWAM`/`flux2` PyTorch model and walking its
attributes** -- a deliberate departure from
`benchmarks/imagewam_real_checkpoint_validation.py`'s own
`extract_*_weights` functions (which DO need the live model, hence
`imagewam`/`flux2` importable). This module needs NEITHER: `model.pt`'s
own `torch.load(..., map_location='cpu', mmap=True)['mot']` IS already
a flat `key -> tensor` mapping (an `OrderedDict`, confirmed by reading
it directly on this dev machine, 2026-09-15 -- see `PROJECT.md`'s
corrected "Real checkpoint testing" note), so no live module
construction is needed at all to pull the tensors themselves.

**Real key names, confirmed directly against the actual release
checkpoint on this machine** (`/home/ljw/projects/pi0.5/models/imagewam_flux2_4b_libero/model.pt`,
NOT re-derived from `imagewam_real_checkpoint_validation.py`'s own
Python-attribute-path documentation, which describes the LIVE
attribute chain, e.g. `model.video_expert.transformer.double_blocks[i]`,
not the state_dict key string):

    mixtures.video.transformer.txt_in.weight              (hidden, joint_attention_dim)
    mixtures.video.transformer.img_in.weight               (hidden, HD)
    mixtures.video.transformer.time_in.{in_layer,out_layer}.weight
    mixtures.video.transformer.double_stream_modulation_{img,txt}.lin.weight
    mixtures.video.transformer.single_stream_modulation.lin.weight
    mixtures.video.transformer.double_blocks.{i}.{txt,img}_attn.{qkv,proj}.weight
    mixtures.video.transformer.double_blocks.{i}.{txt,img}_attn.norm.{query,key}_norm.scale
    mixtures.video.transformer.double_blocks.{i}.{txt,img}_mlp.{0,2}.weight
    mixtures.video.transformer.single_blocks.{i}.{linear1,linear2}.weight
    mixtures.video.transformer.single_blocks.{i}.norm.{query,key}_norm.scale
    mixtures.action.action_encoder.{weight,bias}           (action_hidden_dim, action_dim) + bias
    mixtures.action.time_in.{in_layer,out_layer}.weight
    mixtures.action.double_stream_modulation_img.lin.weight   -- IMG-ONLY, no _txt variant
    mixtures.action.single_stream_modulation.lin.weight
    mixtures.action.double_blocks.{i}.img_attn.{qkv,proj}.weight / norm.*
    mixtures.action.double_blocks.{i}.img_mlp.{0,2}.weight
    mixtures.action.single_blocks.{i}.{linear1,linear2}.weight / norm.*
    mixtures.action.head.adaLN_modulation.1.weight          (2*action_hidden_dim, action_hidden_dim)
    mixtures.action.head.linear.weight                      (action_dim, action_hidden_dim)

`mixtures.video.double_blocks.*`/`mixtures.video.single_blocks.*`
(WITHOUT the `.transformer.` segment) ALSO exist in the checkpoint,
confirmed to be the exact same tensor storage as their `.transformer.`
counterparts (`.data_ptr()` equal, `torch.equal` true) -- a harmless
duplicate registration artifact of how this checkpoint was saved, not
two different weight sets. This loader always reads the
`.transformer.` path, matching `imagewam_real_checkpoint_validation.py`'s
own already-proven attribute chain.

**`txt_in`/`img_in` are SHARED across every double layer** (real
FLUX.2's own architecture: ONE `transformer.txt_in`/`img_in`, not one
per layer) -- unlike this project's own random-weight dry run, which
gives every layer an independent random copy (harmless there, wrong
here). This loader reads each ONCE and reuses the SAME tensor object
for every layer's `("backbone","double",L,"txt_in.weight")`/
`"img_in.weight"` key.

**Still Thor-only for an actual forward pass**: `flux2` (the model-
definition code) is not cloned on this dev machine, and the real model
needs ~18-23GB, more than this machine's 8GB VRAM. This module only
reads raw tensors from the checkpoint file -- it does not construct
`ImageWAM`, does not need `flux2`, and needs no GPU at all to run
(everything stays on CPU as `torch.float16` until the caller decides
what to do with it, e.g. move to CUDA / wrap in `Fp16Linear` etc.).
"""
from __future__ import annotations

import torch

FP16 = torch.float16


def _w(sd: dict, key: str) -> torch.Tensor:
    """Real `nn.Linear` weight, `(out,in)` bf16 -> FlashRT `(in,out)`
    fp16 (this project's own GEMM-storage convention, matching every
    `_lin()`/`_rnd_linear` helper elsewhere)."""
    return sd[key].detach().t().contiguous().to(FP16)


def _v(sd: dict, key: str) -> torch.Tensor:
    """1D vector (norm scale, bias), bf16 -> fp16, no transpose."""
    return sd[key].detach().contiguous().to(FP16)


def _extract_double_block(sd: dict, prefix: str, *, sides: tuple[str, ...], prefixed: bool = True) -> dict:
    """One double-stream block's own weights. `sides` is `("txt","img")`
    for the backbone, `("img",)` for ActionDiT (img-only, see
    `real_action_expert.py`'s own docstring).

    `prefixed` controls the OUTPUT slot name, independent of the real
    checkpoint's own path segment (which always says `{side}_attn`/
    `{side}_mlp`, even for ActionDiT's single "img" side): the backbone
    keys every slot `f"{side}_{slot}"` (`pipeline_thor.py`'s own
    `_double_stream_layer` dual-stream convention), but ActionDiT's own
    `_action_double_layer` uses PLAIN, unprefixed slot names (`"qkv.weight"`,
    not `"img_qkv.weight"` -- confirmed against `imagewam_thor.py`'s own
    `_alloc_random_weights`, which never prefixes action_dit slots)."""
    out = {}
    for side in sides:
        attn = f"{prefix}.{side}_attn"
        mlp = f"{prefix}.{side}_mlp"
        p = f"{side}_" if prefixed else ""
        out[f"{p}qkv.weight"] = _w(sd, f"{attn}.qkv.weight")
        out[f"{p}proj.weight"] = _w(sd, f"{attn}.proj.weight")
        out[f"{p}mlp0.weight"] = _w(sd, f"{mlp}.0.weight")
        out[f"{p}mlp2.weight"] = _w(sd, f"{mlp}.2.weight")
        out[f"{p}query_norm"] = _v(sd, f"{attn}.norm.query_norm.scale")
        out[f"{p}key_norm"] = _v(sd, f"{attn}.norm.key_norm.scale")
    return out


def _extract_single_block(sd: dict, prefix: str, *, attn_dim: int,
                           merge_qkv_mlp: bool = False, merge_linear2: bool = False,
                           kv_only: bool = False) -> dict:
    """One single-stream block: splits the real fused `linear1`/`linear2`
    into the separate GEMMs `pipeline_thor.py` expects -- mathematically
    identical column-range split, see `real_single_stream_block.py`'s
    own docstring for why. `linear1`: `(3*attn_dim + 2*mlp_hidden, hidden)`
    real `(out,in)`; `linear2`: `(hidden, attn_dim + mlp_hidden)`.

    `kv_only=True` (OPT-032 candidate 8, `docs/imagewam_last_block_kv_only.md`):
    ALSO returns `"linear1_kv.weight"`, a one-time column slice of the
    same raw `linear1` tensor down to its K,V output columns
    `[attn_dim, 3*attn_dim)` (Q is `[0,attn_dim)`, MLP gate/up is
    `[3*attn_dim, end)` -- the caller's job to decide which layer this
    applies to (only the LAST backbone single-stream layer's own
    self-attention/MLP work is dead; every earlier layer's full
    `linear1.weight` is still needed and still returned regardless of
    this flag). Bit-exact against the full block's own K/V output at
    real production shape, confirmed in
    `tests/test_imagewam_last_block_kv_only.py`.

    `merge_qkv_mlp=True` (opportunities.md op-fusion audit finding 1):
    returns the real, UNSPLIT `linear1.weight` (covers qkv+mlp-gate/up
    in one tensor) under `"linear1.weight"` instead of splitting it
    into `qkv.weight`/`mlp_in.weight` -- `pipeline_thor.py`'s own
    merged-GEMM path reads Q/K/V and the mlp gate/up columns directly
    out of that one GEMM's output via strided views (no separate
    `mlp_in` GEMM).

    `merge_linear2=True` (roadmap item 4): returns the real, UNSPLIT
    `linear2.weight` under `"linear2.weight"`, `(attn_dim + mlp_hidden,
    hidden)` in the (K,N) GEMM convention, instead of
    `attn_out_proj.weight`/`mlp_down.weight` -- `pipeline_thor.py`'s
    merged path runs it as ONE GEMM over `[attn_out | mlp_act]`, like
    the official block."""
    l1 = sd[f"{prefix}.linear1.weight"].detach()  # (out, in) bf16
    l2 = sd[f"{prefix}.linear2.weight"].detach()
    out = {
        "query_norm": _v(sd, f"{prefix}.norm.query_norm.scale"),
        "key_norm": _v(sd, f"{prefix}.norm.key_norm.scale"),
    }
    if merge_linear2:
        out["linear2.weight"] = l2.t().contiguous().to(FP16)
    else:
        out["attn_out_proj.weight"] = l2[:, :attn_dim].t().contiguous().to(FP16)
        out["mlp_down.weight"] = l2[:, attn_dim:].t().contiguous().to(FP16)
    if merge_qkv_mlp:
        out["linear1.weight"] = l1.t().contiguous().to(FP16)
    else:
        qkv_w = l1[: 3 * attn_dim, :]
        mlp_in_w = l1[3 * attn_dim:, :]
        out["qkv.weight"] = qkv_w.t().contiguous().to(FP16)
        out["mlp_in.weight"] = mlp_in_w.t().contiguous().to(FP16)
    if kv_only:
        kv_w = l1[attn_dim:3 * attn_dim, :]
        out["linear1_kv.weight"] = kv_w.t().contiguous().to(FP16)
    return out


def load_real_imagewam_state_dict(ckpt_path: str) -> dict:
    """Loads `model.pt`'s own `mot` sub-dict (the flat real state_dict)
    -- the ONE call in this module that touches disk/the checkpoint
    format itself, kept separate from the key-parsing logic below so a
    caller can load once and call `build_real_weights`/
    `build_real_modulation_weights` against the same in-memory dict
    without re-reading the file.

    `map_location='cpu', mmap=True`: keeps the ~9GB file OFF the GPU
    entirely (this project's target Thor deployment still needs the
    weights on GPU eventually, but that's the CALLER's decision, e.g.
    per-tensor as each gets wrapped in `Fp16Linear`/`StaticFp8Linear`/
    etc. -- not this function's job) and avoids materializing the
    whole file into RAM at once (verified working on this dev machine,
    9GB file, 20GB free RAM, 2026-09-15).
    """
    payload = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    if "mot" not in payload:
        raise ValueError(
            f"expected a top-level 'mot' key in {ckpt_path} (matches the real "
            f"yuyangalin/ImageWAM-FLUX.2-4B-LIBERO release layout) -- got keys "
            f"{list(payload.keys())}, this may be a different checkpoint format")
    return payload["mot"]


def build_real_weights(sd: dict, *, num_double: int, num_single: int,
                        action_num_double: int, action_num_single: int,
                        action_attn_width: int, merge_qkv_mlp: bool = False,
                        merge_linear2: bool = False, last_layer_kv_only: bool = False) -> dict:
    """Returns a flat dict keyed EXACTLY like `imagewam_thor.py`'s own
    `self._weights` (same 4-tuples `pipeline_thor.py` already expects)
    -- values are raw fp16 `torch.Tensor` (CPU; NOT yet wrapped in
    `Fp16Linear`/etc, and NOT yet moved to CUDA -- same division of
    labor as `_rnd_linear`: this function only produces the real
    tensor, the frontend decides precision/device).

    `last_layer_kv_only=True` (OPT-032 candidate 8): the LAST backbone
    single-stream layer's block ALSO gets `"linear1_kv.weight"`
    (`_extract_single_block`'s own `kv_only`) -- ActionDiT's blocks are
    unaffected, this candidate is backbone-only
    (`docs/imagewam_last_block_kv_only.md`'s own scope).
    """
    weights = {}

    # Shared, once, reused for every double layer (see module docstring's
    # "second finding").
    txt_in_w = _w(sd, "mixtures.video.transformer.txt_in.weight")
    img_in_w = _w(sd, "mixtures.video.transformer.img_in.weight")
    # txt_in_w is (K,N) = (joint_attention_dim, hidden) after _w()'s own
    # transpose -- derive the real backbone hidden width from the
    # checkpoint's own tensor shape rather than requiring a separate
    # caller-supplied param that could silently drift out of sync.
    hidden = txt_in_w.shape[1]
    for L in range(num_double):
        weights[("backbone", "double", L, "txt_in.weight")] = txt_in_w
        weights[("backbone", "double", L, "img_in.weight")] = img_in_w
        block = _extract_double_block(
            sd, f"mixtures.video.transformer.double_blocks.{L}", sides=("txt", "img"))
        for slot, tensor in block.items():
            weights[("backbone", "double", L, slot)] = tensor

    for L in range(num_single):
        block = _extract_single_block(
            sd, f"mixtures.video.transformer.single_blocks.{L}", attn_dim=hidden,
            merge_qkv_mlp=merge_qkv_mlp, merge_linear2=merge_linear2,
            kv_only=last_layer_kv_only and L == num_single - 1)
        for slot, tensor in block.items():
            weights[("backbone", "single", L, slot)] = tensor

    # ActionDiT (img-only, see real_action_expert.py's own docstring).
    weights[("action_dit", "shared", 0, "action_encoder.weight")] = _w(sd, "mixtures.action.action_encoder.weight")
    weights[("action_dit", "shared", 0, "action_encoder.bias")] = _v(sd, "mixtures.action.action_encoder.bias")
    weights[("action_dit", "shared", 0, "head.linear.weight")] = _w(sd, "mixtures.action.head.linear.weight")
    for L in range(action_num_double):
        block = _extract_double_block(sd, f"mixtures.action.double_blocks.{L}", sides=("img",), prefixed=False)
        for slot, tensor in block.items():
            weights[("action_dit", "double", L, slot)] = tensor
    for L in range(action_num_single):
        block = _extract_single_block(
            sd, f"mixtures.action.single_blocks.{L}", attn_dim=action_attn_width,
            merge_qkv_mlp=merge_qkv_mlp, merge_linear2=merge_linear2)
        for slot, tensor in block.items():
            weights[("action_dit", "single", L, slot)] = tensor

    return weights


def build_real_modulation_weights(sd: dict) -> dict:
    """Returns `{"backbone": mod_w, "action": mod_w, "head_adaln": tensor}`
    matching `compute_shared_modulation`/`compute_action_modulation`/
    `compute_action_head_modulation`'s own required `weights` dict keys
    (`pipeline_real.py`) -- these are FLOAT32 (matching every other
    call site's own `mod_w` dtype, e.g. `imagewam_thor.py`'s random
    path, NOT fp16 -- the vec/modulation computation is deliberately
    plain PyTorch at fp32, see `adaln.py`'s own docstring for why).

    **NOT transposed**, unlike `_w()` above: `adaln.py`'s `mlp_embedder`/
    `modulation`/`head_modulation` are plain `F.linear(x, weight)` calls
    (`x @ weight.t()`), which need the weight in its REAL, native
    `(out,in)` layout directly -- `imagewam_thor.py`'s own random
    `mod_w` construction (`torch.randn(hidden, 256, ...)` for
    `time_in_w1`) already confirms this convention; only the FlashRT
    GEMM-backed weights (`_w()` above, feeding `gemm.fp16_nn`/
    `Fp16Linear`) need the `(K,N)` transpose."""
    def w32(key):
        return sd[key].detach().contiguous().to(torch.float32)

    backbone = {
        "time_in_w1": w32("mixtures.video.transformer.time_in.in_layer.weight"),
        "time_in_w2": w32("mixtures.video.transformer.time_in.out_layer.weight"),
        "mod_double_txt": w32("mixtures.video.transformer.double_stream_modulation_txt.lin.weight"),
        "mod_double_img": w32("mixtures.video.transformer.double_stream_modulation_img.lin.weight"),
        "mod_single": w32("mixtures.video.transformer.single_stream_modulation.lin.weight"),
    }
    action = {
        "time_in_w1": w32("mixtures.action.time_in.in_layer.weight"),
        "time_in_w2": w32("mixtures.action.time_in.out_layer.weight"),
        "mod_double": w32("mixtures.action.double_stream_modulation_img.lin.weight"),
        "mod_single": w32("mixtures.action.single_stream_modulation.lin.weight"),
        "head_adaln": w32("mixtures.action.head.adaLN_modulation.1.weight"),
    }
    return {"backbone": backbone, "action": action}


def load_real_proprio_weights(ckpt_path: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Real `proprio_encoder` (a plain biased `nn.Linear(proprio_dim,
    joint_attention_dim)`, e.g. `(7680,8)` weight + `(7680,)` bias for
    the real LIBERO release, `proprio_dim=8` per that release's own
    `config.yaml`) -- closed-loop real-robot-state conditioning, found
    2026-09-15 while scoping real closed-loop testing (opportunities.md).

    **Lives at the TOP LEVEL of the checkpoint payload, a SIBLING of
    `mot` (NOT inside it)** -- confirmed by reading the real checkpoint
    directly: `torch.load(ckpt_path, mmap=True)` has top-level keys
    `{'mot', 'step', 'torch_dtype', 'proprio_encoder'}`.
    `load_real_imagewam_state_dict` above only returns `payload["mot"]`
    (by design -- every other loader in this module takes that flat
    dict directly, and changing its return shape would break every
    existing caller), so this is a small, separate loader that does
    its own `torch.load` rather than widening that function's contract.
    `mmap=True` keeps this cheap even though it re-opens the same file
    (the 9GB backbone/action tensors are never touched by this call).

    Returns `None` if this checkpoint has no `proprio_encoder` key
    (i.e. `proprio_dim=None` for that release -- not every ImageWAM
    checkpoint necessarily uses proprio conditioning), else
    `(weight, bias)` as real `(out,in)`-layout CPU tensors (matching
    `adaln.py`'s own plain-`F.linear` convention, NOT the transposed
    `(K,N)` GEMM-storage convention `_w()` above uses -- this weight is
    applied OUTSIDE the CUDA graph via plain PyTorch, same as
    `text_encoder.py`/`vae_encoder.py`, never through `Fp16Linear`).
    """
    payload = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    pe = payload.get("proprio_encoder")
    if pe is None:
        return None
    return pe["weight"].detach().contiguous(), pe["bias"].detach().contiguous()
