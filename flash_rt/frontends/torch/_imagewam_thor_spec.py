"""ImageWAM (FLUX.2-4B variant) Thor weight specification.

**Rewritten 2026-09-14 to declare REAL-shaped tensors** (per-head K/V,
QK-Norm scales, AdaLN modulation weights, real SiLU-GLU MLP widths),
following opportunities.md OPT-002's now fully-verified real math
(including against the real trained checkpoint on Thor, cosine=0.9999+
-- see `benchmarks/imagewam_real_checkpoint_validation.py`). Random
initialization remains the only fill method here (no real checkpoint
loading in this file) -- see `flash_rt/frontends/torch/imagewam_thor.py`
for where real-checkpoint loading would eventually plug in (still not
implemented; that requires the real `imagewam`/`flux2` packages, only
available on Thor).

Dimensions are architecture configs only (no weight download):
- FLUX.2-klein-4B: `black-forest-labs/FLUX.2-klein-4B`,
  `transformer/config.json` (num_attention_heads=24,
  attention_head_dim=128, num_layers=5, num_single_layers=20,
  in_channels=128, joint_attention_dim=7680).
- ActionDiT (FLUX.2 variant): `configs/model/imagewam_flux2_klein_4b_base.yaml`
  in the ImageWAM repository (hidden_dim=1024, num_heads=24,
  attn_head_dim=128, num_layers_double=5, num_layers_single=20,
  max_action_horizon=64). `num_heads`/`attn_head_dim` match the
  backbone exactly — required for the `mot_joint` attention kernel
  (see `flash_rt/hardware/thor/attn_backend.py`) to operate on both
  streams' Q/K/V in the same per-head geometry.

**Per-layer shapes below now match what
`flash_rt/models/imagewam/pipeline_thor.py`'s real-math layer helpers
consume directly.** Q/K/V are declared as ONE fused `{prefix}_qkv.weight`
tensor (`(width, 3*width)`, GEMM (K,N) convention) — a 2026-09-14
follow-up (OPT-004 step 2, opportunities.md) that both cuts the
projection from 3 GEMM launches to 1 AND matches a real checkpoint's
own fused `qkv` tensor directly (a real checkpoint's `img_attn.qkv`/
`txt_attn.qkv`/`linear1`'s own QKV slice loads into this shape with NO
splitting needed, unlike an earlier version of this file that
deliberately kept Q/K/V separate for pointer-code simplicity — that
turned out to cost both a real GEMM launch and checkpoint-native-ness
for no benefit). In double-stream blocks `proj` (the OUTPUT projection)
and `mlp0`/`mlp2` remain separate GEMMs, since `pipeline_thor.py`'s own
`_double_stream_layer`/etc. need Q and K to land in two DIFFERENT
persistent buffers (`Q_O`/`K_cache`) afterward regardless (one GEMM
into a wide scratch buffer, then 3 slice-copies).

Single-stream blocks: the `*_SINGLE_LAYER_SHAPES` tables below list the
split layout (`qkv` + `mlp_in`, `attn_out_proj` + `mlp_down`), which only
`precision="fp16_cutlass"` loads. Every other precision loads the real
checkpoint's two fused tensors unsplit (`dims["merge_qkv_mlp"]`,
`dims["merge_linear2"]`), in the same (K,N) GEMM convention:
`linear1.weight` `(width_in, 3*attn_width + 2*mlp_hidden)` and
`linear2.weight` `(attn_width + mlp_hidden, width_out)` -- backbone
`(3072, 27648)` / `(12288, 3072)`, ActionDiT `(1024, 20480)` /
`(7168, 1024)`. `tests/test_imagewam_thor_precision_routing.py` holds
the per-precision slot contract.

Real checkpoint loading (when it exists) will go through ImageWAM's
own Python model classes plus `benchmarks/imagewam_real_checkpoint_validation.py`'s
own `extract_*_weights` functions (already verified against the real
checkpoint, cosine=0.9999+), which read from real `nn.Module` objects
directly — not from this file's declared shapes.

K/V are now real per-head width (`backbone_hidden`/`action_attn_width`,
matching `NH*HD`), not the old broadcast-K/V `HD` width — see
opportunities.md OPT-002. MLP first-projection width is `mlp_hidden*2`
(real SiLU-gated GLU, not plain GELU on `mlp_hidden`) — see
`flash_rt/models/imagewam/real_mlp.py`. `*_query_norm`/`*_key_norm`
((HD,) each) are new: real QK-Norm scales, applied via the existing
`rms_norm_fp16` kernel per (token,head) row (opportunities.md OPT-002 —
confirmed this kernel already computes the exact real QK-Norm formula,
no new kernel needed).

`SHARED_MOD_SHAPES`/`ACTION_SHARED_MOD_SHAPES` are NEW: real AdaLN
modulation is driven by a per-FORWARD (not per-layer) timestep
embedding, shared across every layer of a given stream type (a real
architecture property, confirmed from `Flux2.forward` — see
`flash_rt/models/imagewam/pipeline_real.py`'s `compute_shared_modulation`/
`compute_action_modulation`). These are allocated ONCE per model
instance (backbone) or ONCE PER DENOISE STEP (ActionDiT, since its own
conditioning timestep changes every step), never per-layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ImageWAMThorSpec:
    backbone_num_heads: int = 24
    backbone_head_dim: int = 128
    backbone_num_layers_double: int = 5
    backbone_num_layers_single: int = 20
    backbone_in_channels: int = 128
    backbone_mlp_ratio: float = 3.0
    joint_attention_dim: int = 7680  # context input width, not a weight

    action_hidden_dim: int = 1024
    action_num_heads: int = 24        # == backbone_num_heads, required for mot_joint
    action_head_dim: int = 128        # == backbone_head_dim, required for mot_joint
    action_num_layers_double: int = 5  # == backbone_num_layers_double
    action_num_layers_single: int = 20  # == backbone_num_layers_single
    action_mlp_ratio: float = 4.0
    max_action_horizon: int = 64

    @property
    def backbone_hidden(self) -> int:
        return self.backbone_num_heads * self.backbone_head_dim  # 3072

    @property
    def backbone_mlp_hidden(self) -> int:
        return int(self.backbone_hidden * self.backbone_mlp_ratio)  # 9216

    @property
    def action_attn_width(self) -> int:
        return self.action_num_heads * self.action_head_dim  # 3072

    @property
    def action_mlp_hidden(self) -> int:
        return int(self.action_hidden_dim * self.action_mlp_ratio)  # 4096


SPEC = ImageWAMThorSpec()


# ──────────────────────────────────────────────────────────────────
# Backbone (FLUX.2-klein-4B), double-stream blocks: separate img/txt
# Q/K/V+proj+MLP, jointly attended within the block. K/V at real
# per-head width (backbone_hidden), not broadcast HD.
# ──────────────────────────────────────────────────────────────────

BACKBONE_DOUBLE_LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    "img_qkv.weight": (SPEC.backbone_hidden, 3 * SPEC.backbone_hidden),
    "img_proj.weight": (SPEC.backbone_hidden, SPEC.backbone_hidden),
    "img_mlp0.weight": (SPEC.backbone_mlp_hidden * 2, SPEC.backbone_hidden),
    "img_mlp2.weight": (SPEC.backbone_hidden, SPEC.backbone_mlp_hidden),
    "img_query_norm": (SPEC.backbone_head_dim,),
    "img_key_norm": (SPEC.backbone_head_dim,),
    # txt stream: context arrives at joint_attention_dim, projected
    # into the shared attention width once per layer before qkv.
    "txt_in.weight": (SPEC.backbone_hidden, SPEC.joint_attention_dim),
    "txt_qkv.weight": (SPEC.backbone_hidden, 3 * SPEC.backbone_hidden),
    "txt_proj.weight": (SPEC.backbone_hidden, SPEC.backbone_hidden),
    "txt_mlp0.weight": (SPEC.backbone_mlp_hidden * 2, SPEC.backbone_hidden),
    "txt_mlp2.weight": (SPEC.backbone_hidden, SPEC.backbone_mlp_hidden),
    "txt_query_norm": (SPEC.backbone_head_dim,),
    "txt_key_norm": (SPEC.backbone_head_dim,),
}

# Single-stream blocks: img+txt merged into one stream. Split layout of
# the real fused linear1(QKV+MLP-up)/linear2(attn-out+MLP-down), loaded
# only under fp16_cutlass; every other precision loads linear1.weight /
# linear2.weight unsplit -- see module docstring.
BACKBONE_SINGLE_LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    "qkv.weight": (SPEC.backbone_hidden, 3 * SPEC.backbone_hidden),
    "attn_out_proj.weight": (SPEC.backbone_hidden, SPEC.backbone_hidden),
    "mlp_in.weight": (SPEC.backbone_mlp_hidden * 2, SPEC.backbone_hidden),
    "mlp_down.weight": (SPEC.backbone_hidden, SPEC.backbone_mlp_hidden),
    "query_norm": (SPEC.backbone_head_dim,),
    "key_norm": (SPEC.backbone_head_dim,),
}

# Shared (not per-layer) AdaLN modulation weights, allocated ONCE.
SHARED_MOD_SHAPES: dict[str, tuple[int, ...]] = {
    "time_in_w1": (SPEC.backbone_hidden, 256),
    "time_in_w2": (SPEC.backbone_hidden, SPEC.backbone_hidden),
    "mod_double_txt": (6 * SPEC.backbone_hidden, SPEC.backbone_hidden),
    "mod_double_img": (6 * SPEC.backbone_hidden, SPEC.backbone_hidden),
    "mod_single": (3 * SPEC.backbone_hidden, SPEC.backbone_hidden),
}


# ──────────────────────────────────────────────────────────────────
# ActionDiT (FLUX.2 variant): single action-token stream, IMG-ONLY
# double block (no txt branch -- see real_action_expert.py). K/V at
# real per-head width (action_attn_width = NH*HD), not broadcast HD;
# residual-stream width (action_hidden_dim) differs from attention
# width (action_attn_width) here, unlike the backbone.
# ──────────────────────────────────────────────────────────────────

ACTION_DIT_DOUBLE_LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    "qkv.weight": (SPEC.action_hidden_dim, 3 * SPEC.action_attn_width),
    "proj.weight": (SPEC.action_hidden_dim, SPEC.action_attn_width),
    "mlp0.weight": (SPEC.action_mlp_hidden * 2, SPEC.action_hidden_dim),
    "mlp2.weight": (SPEC.action_hidden_dim, SPEC.action_mlp_hidden),
    "query_norm": (SPEC.action_head_dim,),
    "key_norm": (SPEC.action_head_dim,),
}

ACTION_DIT_SINGLE_LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    "qkv.weight": (SPEC.action_hidden_dim, 3 * SPEC.action_attn_width),
    "attn_out_proj.weight": (SPEC.action_hidden_dim, SPEC.action_attn_width),
    "mlp_in.weight": (SPEC.action_mlp_hidden * 2, SPEC.action_hidden_dim),
    "mlp_down.weight": (SPEC.action_hidden_dim, SPEC.action_mlp_hidden),
    "query_norm": (SPEC.action_head_dim,),
    "key_norm": (SPEC.action_head_dim,),
}

# ActionDiT's own time_in/modulation weights -- a SEPARATE weight set
# from the backbone's own (confirmed from mot.py: each expert owns its
# own time_in). Only ONE double-type modulation (img-only block, no
# txt), unlike the backbone's separate txt/img pair. Allocated ONCE
# PER DENOISE STEP (the action-expert's conditioning timestep changes
# every step; the backbone's own SHARED_MOD_SHAPES stays fixed for the
# whole forward -- see module docstring).
ACTION_SHARED_MOD_SHAPES: dict[str, tuple[int, ...]] = {
    "time_in_w1": (SPEC.action_hidden_dim, 256),
    "time_in_w2": (SPEC.action_hidden_dim, SPEC.action_hidden_dim),
    "mod_double": (6 * SPEC.action_hidden_dim, SPEC.action_hidden_dim),
    "mod_single": (3 * SPEC.action_hidden_dim, SPEC.action_hidden_dim),
}


# ──────────────────────────────────────────────────────────────────
# Inputs (not weights): random-filled the same way every weight is
# for this plan, but never written to a "weights" dict -- these are
# per-call/per-prompt buffers.
# ──────────────────────────────────────────────────────────────────

INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "context": (512, SPEC.joint_attention_dim),  # qwen_context_len, joint_attention_dim
    "context_mask": (512,),
}


def backbone_layer_key(layer: int, is_single: bool, suffix: str) -> str:
    stream = "single" if is_single else "double"
    return f"backbone.{stream}.{layer}.{suffix}"


def action_dit_layer_key(layer: int, is_single: bool, suffix: str) -> str:
    stream = "single" if is_single else "double"
    return f"action_dit.{stream}.{layer}.{suffix}"


def iter_expected_shapes() -> Iterable[tuple[str, tuple[int, ...]]]:
    """Every PER-LAYER weight tensor this plan's frontend must allocate.

    Does not include `INPUT_SHAPES` (per-call inputs) or the shared
    modulation weights (`SHARED_MOD_SHAPES`/`ACTION_SHARED_MOD_SHAPES`,
    allocated once, not per layer -- see module docstring).
    """
    for layer in range(SPEC.backbone_num_layers_double):
        for suffix, shape in BACKBONE_DOUBLE_LAYER_SHAPES.items():
            yield backbone_layer_key(layer, False, suffix), shape
    for layer in range(SPEC.backbone_num_layers_single):
        for suffix, shape in BACKBONE_SINGLE_LAYER_SHAPES.items():
            yield backbone_layer_key(layer, True, suffix), shape
    for layer in range(SPEC.action_num_layers_double):
        for suffix, shape in ACTION_DIT_DOUBLE_LAYER_SHAPES.items():
            yield action_dit_layer_key(layer, False, suffix), shape
    for layer in range(SPEC.action_num_layers_single):
        for suffix, shape in ACTION_DIT_SINGLE_LAYER_SHAPES.items():
            yield action_dit_layer_key(layer, True, suffix), shape
