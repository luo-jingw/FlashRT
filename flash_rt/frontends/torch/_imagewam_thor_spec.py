"""ImageWAM (FLUX.2-4B variant) Thor weight specification.

Structural dry-run scope only: every tensor declared here is intended
for random initialization (see `flash_rt/frontends/torch/imagewam_thor.py`),
not for loading a real checkpoint. No Qwen3-4B weights are declared:
every ImageWAM model config (including the FLUX.2-4B one this targets)
sets `load_text_encoder: false` — ImageWAM never loads or runs the
text encoder itself, and always receives a precomputed `context`/
`context_mask` pair. `context` is declared as an input shape below,
not a weight.

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

Tensor name fragments (`qkv`, `proj`, `linear1`, `linear2`,
`img_mlp.{0,2}`, `txt_mlp.{0,2}`) are taken from that same YAML's
`flux2_lora_config.target_suffixes` list, confirmed real names in the
ImageWAM/FLUX.2 checkpoint. Full per-tensor shapes for the backbone's
double-stream and single-stream blocks follow FLUX's published
double-stream (separate img/txt QKV+MLP, jointly attended) /
single-stream (img+txt merged, QKV+MLP-up fused into `linear1`,
attn-out+MLP-down fused into `linear2`) block design. ActionDiT's own
per-block shapes (single action-token stream, no separate txt stream)
are a structural approximation of the same double/single-stream split
at `action_hidden_dim` width — not yet verified against
`imagewam/src/imagewam/models/backbones/action_dit_flux2.py`'s actual
block implementation; sufficient for a random-init structural dry run,
not for loading a real ActionDiT checkpoint.
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
# QKV+proj+MLP, jointly attended within the block.
# ──────────────────────────────────────────────────────────────────

BACKBONE_DOUBLE_LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    # img stream operates directly at backbone_hidden.
    "img_attn.qkv.weight": (3 * SPEC.backbone_hidden, SPEC.backbone_hidden),
    "img_attn.proj.weight": (SPEC.backbone_hidden, SPEC.backbone_hidden),
    "img_mlp.0.weight": (SPEC.backbone_mlp_hidden, SPEC.backbone_hidden),
    "img_mlp.2.weight": (SPEC.backbone_hidden, SPEC.backbone_mlp_hidden),
    # txt stream: context arrives at joint_attention_dim, projected
    # into the shared attention width once per layer before qkv.
    "txt_in.weight": (SPEC.backbone_hidden, SPEC.joint_attention_dim),
    "txt_attn.qkv.weight": (3 * SPEC.backbone_hidden, SPEC.backbone_hidden),
    "txt_attn.proj.weight": (SPEC.backbone_hidden, SPEC.backbone_hidden),
    "txt_mlp.0.weight": (SPEC.backbone_mlp_hidden, SPEC.backbone_hidden),
    "txt_mlp.2.weight": (SPEC.backbone_hidden, SPEC.backbone_mlp_hidden),
}

# Single-stream blocks: img+txt merged into one stream; QKV+MLP-up
# fused into linear1, attn-out+MLP-down fused into linear2.
BACKBONE_SINGLE_LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    "linear1.weight": (
        3 * SPEC.backbone_hidden + SPEC.backbone_mlp_hidden,
        SPEC.backbone_hidden,
    ),
    "linear2.weight": (
        SPEC.backbone_hidden,
        SPEC.backbone_hidden + SPEC.backbone_mlp_hidden,
    ),
}


# ──────────────────────────────────────────────────────────────────
# ActionDiT (FLUX.2 variant): single action-token stream. Structural
# approximation of the backbone's own double/single-stream split at
# action_hidden_dim width -- see module docstring.
# ──────────────────────────────────────────────────────────────────

ACTION_DIT_DOUBLE_LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    "action_attn.qkv.weight": (3 * SPEC.action_attn_width, SPEC.action_hidden_dim),
    "action_attn.proj.weight": (SPEC.action_hidden_dim, SPEC.action_attn_width),
    "action_mlp.0.weight": (SPEC.action_mlp_hidden, SPEC.action_hidden_dim),
    "action_mlp.2.weight": (SPEC.action_hidden_dim, SPEC.action_mlp_hidden),
}

ACTION_DIT_SINGLE_LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    "linear1.weight": (
        3 * SPEC.action_attn_width + SPEC.action_mlp_hidden,
        SPEC.action_hidden_dim,
    ),
    "linear2.weight": (
        SPEC.action_hidden_dim,
        SPEC.action_attn_width + SPEC.action_mlp_hidden,
    ),
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
    """Every weight tensor this plan's frontend must allocate.

    Does not include INPUT_SHAPES (context/context_mask) -- those are
    per-call inputs, not weights; the frontend allocates them
    separately (see imagewam_thor.py).
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
