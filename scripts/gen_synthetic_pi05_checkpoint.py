#!/usr/bin/env python3
"""Generate a synthetic, random-weight Pi0.5 checkpoint directory.

``Pi05TorchFrontendThor`` (``flash_rt/frontends/torch/pi05_thor.py``) and
``Pi05TorchFrontendRtx`` (``flash_rt/frontends/torch/pi05_rtx.py``) both
require a real ``checkpoint_dir`` at construction and have no random-weights
or shape-override construction mode. This script produces a directory that
satisfies their actual, on-disk loading contract (``model.safetensors`` plus
``norm_stats.json``) with random-but-finite weight values, so a synthetic
shape can be benchmarked without a real trained checkpoint.

Checkpoint content is the raw HuggingFace-style key/shape set both frontends
read (confirmed by reading ``Pi05TorchFrontendThor._load_weights``,
``Pi05TorchFrontendRtx.convert_pi05_safetensors``, and the declarative
``_pi05_thor_spec.py`` / ``_thor_spec_common.py`` weight specs end to end):
every dimension below is a **fixed Pi0.5 architecture constant** imported
from ``flash_rt.models.pi05.pipeline_rtx`` (the module that defines them,
literally commented "Fixed Pi0.5 model dimensions"), not a checkpoint- or
config-driven value. There is no ``config.json`` anywhere in this loading
path for either frontend: neither class reads any JSON file except
``norm_stats.json`` (action/state unnormalization statistics), and the
"chunk_size" constructor argument threaded through
``flash_rt/models/pi05/pipeline_thor.py``'s ``bind_runtime_export`` (around
line 820-836) is unrelated metadata for the runtime-export container, not
the action-chunk length the decoder actually runs at.

Confirmed field names (see the module docstring of each usage site):

- ``action_dim``: the hardcoded module constant ``ACTION_DIM = 32`` in
  ``flash_rt/models/pi05/pipeline_rtx.py``. ``Pi05TorchFrontendThor`` never
  imports this constant but hardcodes the same value directly in
  ``_load_weights`` (e.g. ``self._ae_action_f32 = torch.empty(Sa, 32, ...)``,
  ``self._g_noise = torch.zeros(Sa, 32, ...)``) and via the on-disk shape of
  ``action_in_proj`` / ``action_out_proj``. Neither frontend exposes a
  constructor kwarg or config path to change it; this generator therefore
  only supports ``--action-dim 32`` and raises otherwise.
- ``chunk_size`` / action horizon: ``Pi05TorchFrontendRtx.__init__`` takes it
  as the real constructor kwarg ``chunk_size`` (default ``CHUNK_SIZE = 10``,
  attribute ``self.chunk_size``) -- fully independent of checkpoint content,
  since no weight tensor's shape depends on it (only runtime buffer/style
  shapes do). ``Pi05TorchFrontendThor`` hardcodes it as the literal
  ``Sa = 10`` inside ``_load_weights`` (attribute ``self.Sa``); its
  ``__init__`` signature has no ``chunk_size`` or ``num_steps`` parameter at
  all, and ``flash_rt.api.load_model()`` only forwards ``num_steps`` when
  ``"num_steps" in inspect.signature(pipe_cls).parameters``, which is False
  for this class -- so there is no path, checkpoint-side or kwarg-side, to
  make ``Pi05TorchFrontendThor`` run at a chunk size other than 10 today.
- ``num_flow_steps`` (the denoising step count): ``Pi05TorchFrontendRtx``'s
  real constructor kwarg is ``num_steps`` (default
  ``NUM_STEPS_DEFAULT = 10``, attribute ``self._num_steps``), independent of
  checkpoint content for the same reason. ``Pi05TorchFrontendThor`` hardcodes
  it as the literal ``steps = 10`` inside ``_load_weights`` (attribute
  ``self.steps``); same non-overridable situation as chunk_size.
- ``num_views``: a real constructor kwarg on both classes, and confirmed NOT
  stored in checkpoint content -- every view is processed through the same
  shared vision-tower weights, so no safetensors tensor shape depends on it.

Net effect: this generator's ``--chunk-size`` and ``--num-flow-steps`` flags
exist to match the requested CLI surface and are recorded in an informational
sidecar (``synthetic_checkpoint_metadata.json``, NOT read by either loader),
but they cannot change the generated ``model.safetensors`` content (nothing
in it depends on them), and a value other than 10 is only actually honored by
constructing ``Pi05TorchFrontendRtx`` directly with matching ``chunk_size=`` /
``num_steps=`` kwargs -- ``Pi05TorchFrontendThor`` silently ignores both and
always runs at chunk_size=10 / num_flow_steps=10.

OpenPI's own PyTorch eager reference (the official baseline used throughout
this repo's benchmark docs, e.g. ``docs/pi05_thor_decoder_fp4_e2e.md``) reads
the SAME ``model.safetensors`` format: that document states plainly that
"Both FlashRT and the official reference load the same converted weights"
from one ``model.safetensors`` file. This generator's output is therefore
usable as a synthetic drop-in for the official reference too, subject to the
same fixed-architecture-constant caveat above (the official reference does
not expose a different action_dim/chunk_size/num_flow_steps either -- it is
the same trained Pi0.5 checkpoint format).

Usage::

    python scripts/gen_synthetic_pi05_checkpoint.py \\
        --out /tmp/pi05_synthetic_checkpoint \\
        --action-dim 32 --chunk-size 30 --num-flow-steps 10 \\
        --num-views 3 --seed 0

The vocabulary size (257152) is the real PaliGemma SentencePiece vocabulary
(``sentencepiece.SentencePieceProcessor(...).GetPieceSize()`` on the real
``paligemma_tokenizer.model``, matching the same hardcoded fallback already
used in ``flash_rt/frontends/torch/pi0fast.py``). It must match exactly:
both frontends embed real tokenizer output through this table via
``torch.nn.functional.embedding``, so a smaller synthetic vocabulary would
index out of range and raise at ``set_prompt()`` time for ordinary prompts.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib

import torch
from safetensors.torch import save_file

from flash_rt.models.pi05.pipeline_rtx import (
    ACTION_DIM,
    DEC_D,
    DEC_H,
    DEC_HD,
    DEC_L,
    DEC_NH,
    DEC_NKV,
    ENC_D,
    ENC_H,
    ENC_HD,
    ENC_L,
    ENC_NH,
    ENC_NKV,
    NUM_STEPS_DEFAULT,
    VIS_D,
    VIS_H,
    VIS_L,
)
from flash_rt.frontends.torch.pi05_rtx import CHUNK_SIZE, IMG_HW

logger = logging.getLogger(__name__)

# PaliGemma SentencePiece vocabulary size. Fixed by the real tokenizer this
# checkpoint's embedding table must be indexable by (see module docstring).
_PALIGEMMA_VOCAB_SIZE = 257152

# SigLIP patch size (so400m-style 224x224 image, 14x14 patches -> 16x16 = 256
# patch tokens per view). Not exported as a named constant anywhere in the
# pipeline modules; fixed by ``vp.embeddings.patch_embedding.weight``'s shape
# in the real checkpoint and by ``IMG_HW // _PATCH_SIZE == 16``.
_PATCH_SIZE = 14
_PATCH_CHANNELS = 3
_NUM_PATCHES = (IMG_HW // _PATCH_SIZE) ** 2

_WEIGHT_STD = 0.02  # small-magnitude init; keeps every GEMM/quantize finite.

_VISION_ROOT = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
_PROJECTOR_ROOT = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
_ENCODER_ROOT = "paligemma_with_expert.paligemma.model.language_model.layers"
_EMBEDDING_KEY = "paligemma_with_expert.paligemma.lm_head.weight"
_DECODER_ROOT = "paligemma_with_expert.gemma_expert.model.layers"
_DECODER_FINAL_NORM_ROOT = "paligemma_with_expert.gemma_expert.model.norm.dense"


def _randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Small-magnitude random weight, guaranteed finite at FP16."""
    return (torch.randn(shape, generator=generator, dtype=torch.float32)
            * _WEIGHT_STD).to(torch.float16)


def _zeros(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float16)


def _build_vision_tower(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """SigLIP vision tower: embeddings + 27 encoder layers + final norm."""
    out: dict[str, torch.Tensor] = {
        f"{_VISION_ROOT}.embeddings.patch_embedding.weight":
            _randn((VIS_D, _PATCH_CHANNELS, _PATCH_SIZE, _PATCH_SIZE), generator),
        f"{_VISION_ROOT}.embeddings.patch_embedding.bias": _zeros((VIS_D,)),
        f"{_VISION_ROOT}.embeddings.position_embedding.weight":
            _randn((_NUM_PATCHES, VIS_D), generator),
        f"{_VISION_ROOT}.post_layernorm.weight": _randn((VIS_D,), generator),
        f"{_VISION_ROOT}.post_layernorm.bias": _zeros((VIS_D,)),
    }
    for i in range(VIS_L):
        lp = f"{_VISION_ROOT}.encoder.layers.{i}"
        out[f"{lp}.layer_norm1.weight"] = _randn((VIS_D,), generator)
        out[f"{lp}.layer_norm1.bias"] = _zeros((VIS_D,))
        out[f"{lp}.layer_norm2.weight"] = _randn((VIS_D,), generator)
        out[f"{lp}.layer_norm2.bias"] = _zeros((VIS_D,))
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            out[f"{lp}.self_attn.{proj}.weight"] = _randn((VIS_D, VIS_D), generator)
            out[f"{lp}.self_attn.{proj}.bias"] = _zeros((VIS_D,))
        out[f"{lp}.mlp.fc1.weight"] = _randn((VIS_H, VIS_D), generator)
        out[f"{lp}.mlp.fc1.bias"] = _zeros((VIS_H,))
        out[f"{lp}.mlp.fc2.weight"] = _randn((VIS_D, VIS_H), generator)
        out[f"{lp}.mlp.fc2.bias"] = _zeros((VIS_D,))
    return out


def _build_projector(generator: torch.Generator) -> dict[str, torch.Tensor]:
    return {
        f"{_PROJECTOR_ROOT}.weight": _randn((ENC_D, VIS_D), generator),
        f"{_PROJECTOR_ROOT}.bias": _zeros((ENC_D,)),
    }


def _build_encoder(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Gemma-2B encoder: 18 GQA layers (8 query heads, 1 KV head), no biases."""
    q_dim = ENC_NH * ENC_HD
    kv_dim = ENC_NKV * ENC_HD
    out: dict[str, torch.Tensor] = {}
    for i in range(ENC_L):
        ep = f"{_ENCODER_ROOT}.{i}"
        out[f"{ep}.input_layernorm.weight"] = _randn((ENC_D,), generator)
        out[f"{ep}.self_attn.q_proj.weight"] = _randn((q_dim, ENC_D), generator)
        out[f"{ep}.self_attn.k_proj.weight"] = _randn((kv_dim, ENC_D), generator)
        out[f"{ep}.self_attn.v_proj.weight"] = _randn((kv_dim, ENC_D), generator)
        out[f"{ep}.self_attn.o_proj.weight"] = _randn((ENC_D, q_dim), generator)
        out[f"{ep}.post_attention_layernorm.weight"] = _randn((ENC_D,), generator)
        out[f"{ep}.mlp.gate_proj.weight"] = _randn((ENC_H, ENC_D), generator)
        out[f"{ep}.mlp.up_proj.weight"] = _randn((ENC_H, ENC_D), generator)
        out[f"{ep}.mlp.down_proj.weight"] = _randn((ENC_D, ENC_H), generator)
    return out


def _build_embedding(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Shared PaliGemma token embedding / lm_head table."""
    return {_EMBEDDING_KEY: _randn((_PALIGEMMA_VOCAB_SIZE, ENC_D), generator)}


def _build_decoder(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Gemma-300M action-expert decoder: 18 layers with AdaRMSNorm modulation
    Dense layers (3x hidden dim: shift/scale/gate), plus the final norm mod."""
    q_dim = DEC_NH * DEC_HD
    kv_dim = DEC_NKV * DEC_HD
    mod_dim = 3 * DEC_D
    out: dict[str, torch.Tensor] = {}
    for i in range(DEC_L):
        dp = f"{_DECODER_ROOT}.{i}"
        out[f"{dp}.input_layernorm.dense.weight"] = _randn((mod_dim, DEC_D), generator)
        out[f"{dp}.input_layernorm.dense.bias"] = _zeros((mod_dim,))
        out[f"{dp}.self_attn.q_proj.weight"] = _randn((q_dim, DEC_D), generator)
        out[f"{dp}.self_attn.k_proj.weight"] = _randn((kv_dim, DEC_D), generator)
        out[f"{dp}.self_attn.v_proj.weight"] = _randn((kv_dim, DEC_D), generator)
        out[f"{dp}.self_attn.o_proj.weight"] = _randn((DEC_D, q_dim), generator)
        out[f"{dp}.post_attention_layernorm.dense.weight"] = _randn((mod_dim, DEC_D), generator)
        out[f"{dp}.post_attention_layernorm.dense.bias"] = _zeros((mod_dim,))
        out[f"{dp}.mlp.gate_proj.weight"] = _randn((DEC_H, DEC_D), generator)
        out[f"{dp}.mlp.up_proj.weight"] = _randn((DEC_H, DEC_D), generator)
        out[f"{dp}.mlp.down_proj.weight"] = _randn((DEC_D, DEC_H), generator)
    out[f"{_DECODER_FINAL_NORM_ROOT}.weight"] = _randn((mod_dim, DEC_D), generator)
    out[f"{_DECODER_FINAL_NORM_ROOT}.bias"] = _zeros((mod_dim,))
    return out


def _build_time_mlp_and_action_heads(
        generator: torch.Generator, action_dim: int) -> dict[str, torch.Tensor]:
    """Sinusoidal time-embedding MLP (DEC_D -> DEC_D) + action in/out projections."""
    return {
        "time_mlp_in.weight": _randn((DEC_D, DEC_D), generator),
        "time_mlp_in.bias": _zeros((DEC_D,)),
        "time_mlp_out.weight": _randn((DEC_D, DEC_D), generator),
        "time_mlp_out.bias": _zeros((DEC_D,)),
        "action_in_proj.weight": _randn((DEC_D, action_dim), generator),
        "action_in_proj.bias": _zeros((DEC_D,)),
        "action_out_proj.weight": _randn((action_dim, DEC_D), generator),
        "action_out_proj.bias": _zeros((action_dim,)),
    }


def build_state_dict(*, action_dim: int, seed: int) -> dict[str, torch.Tensor]:
    """Assemble the complete raw HF-style Pi0.5 checkpoint state dict.

    Every key/shape here is read directly by both
    ``Pi05TorchFrontendThor._load_weights`` and
    ``Pi05TorchFrontendRtx.convert_pi05_safetensors`` (confirmed by reading
    both end to end plus the declarative ``_pi05_thor_spec.py`` /
    ``_thor_spec_common.py`` weight specs); this is not a guess.
    """
    if action_dim != ACTION_DIM:
        raise ValueError(
            f"action_dim={action_dim} is not supported: both frontends "
            f"hardcode ACTION_DIM={ACTION_DIM} in buffer shapes and weight "
            "projections with no config or constructor override (see this "
            "module's docstring). Pass --action-dim "
            f"{ACTION_DIM}.")
    generator = torch.Generator().manual_seed(seed)
    state_dict: dict[str, torch.Tensor] = {}
    state_dict.update(_build_vision_tower(generator))
    state_dict.update(_build_projector(generator))
    state_dict.update(_build_encoder(generator))
    state_dict.update(_build_embedding(generator))
    state_dict.update(_build_decoder(generator))
    state_dict.update(_build_time_mlp_and_action_heads(generator, action_dim))
    return state_dict


def build_norm_stats(action_dim: int, state_dim: int = 8) -> dict:
    """Minimal openpi-schema norm_stats: only ``q01``/``q99`` are read by
    ``flash_rt.core.utils.actions.unnormalize_actions``; ``state`` is
    included for completeness (used only in state-in-prompt mode)."""
    return {
        "actions": {
            "q01": [-1.0] * action_dim,
            "q99": [1.0] * action_dim,
        },
        "state": {
            "q01": [-1.0] * state_dim,
            "q99": [1.0] * state_dim,
        },
    }


def _warn_if_unhonorable_by_thor(chunk_size: int, num_flow_steps: int) -> None:
    if chunk_size != CHUNK_SIZE or num_flow_steps != NUM_STEPS_DEFAULT:
        logger.warning(
            "chunk_size=%d / num_flow_steps=%d requested, but "
            "Pi05TorchFrontendThor hardcodes chunk_size=%d and "
            "num_flow_steps=%d internally (pi05_thor.py _load_weights: "
            "`Sa, ... = %d, ...` / `steps = %d`) with no constructor kwarg "
            "or config path to override either value. Constructing "
            "Pi05TorchFrontendThor from this checkpoint will silently run "
            "at chunk_size=%d / num_flow_steps=%d regardless of this "
            "request. Only Pi05TorchFrontendRtx(checkpoint_dir, "
            "chunk_size=%d, num_steps=%d, ...) actually honors these "
            "values -- pass them to its constructor; this generator does "
            "not (and cannot) bake them into model.safetensors, because no "
            "weight tensor's shape depends on them.",
            chunk_size, num_flow_steps, CHUNK_SIZE, NUM_STEPS_DEFAULT,
            CHUNK_SIZE, NUM_STEPS_DEFAULT, CHUNK_SIZE, NUM_STEPS_DEFAULT,
            chunk_size, num_flow_steps)


def generate_checkpoint(
        out_dir: pathlib.Path, *, action_dim: int, chunk_size: int,
        num_flow_steps: int, num_views: int, seed: int) -> None:
    """Write ``model.safetensors`` + ``norm_stats.json`` +
    ``synthetic_checkpoint_metadata.json`` under ``out_dir``."""
    if num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if num_flow_steps < 1:
        raise ValueError(f"num_flow_steps must be >= 1, got {num_flow_steps}")
    _warn_if_unhonorable_by_thor(chunk_size, num_flow_steps)

    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = build_state_dict(action_dim=action_dim, seed=seed)
    save_file(state_dict, str(out_dir / "model.safetensors"))
    logger.info("Wrote %d tensors to %s", len(state_dict),
                out_dir / "model.safetensors")

    norm_stats = build_norm_stats(action_dim)
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    logger.info("Wrote %s", out_dir / "norm_stats.json")

    # Informational only -- NOT read by either frontend's loader (there is
    # no config.json in this checkpoint format; see module docstring).
    metadata = {
        "note": (
            "Informational only. Neither Pi05TorchFrontendThor nor "
            "Pi05TorchFrontendRtx reads this file; it exists only to "
            "record how this synthetic checkpoint was generated."),
        "generator": "scripts/gen_synthetic_pi05_checkpoint.py",
        "seed": seed,
        "requested": {
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "num_flow_steps": num_flow_steps,
            "num_views": num_views,
        },
        "fixed_architecture_constants": {
            "action_dim": ACTION_DIM,
            "vis_layers": VIS_L, "vis_d": VIS_D, "vis_h": VIS_H,
            "enc_layers": ENC_L, "enc_d": ENC_D, "enc_h": ENC_H,
            "dec_layers": DEC_L, "dec_d": DEC_D, "dec_h": DEC_H,
            "vocab_size": _PALIGEMMA_VOCAB_SIZE,
        },
        "honored_by": {
            "Pi05TorchFrontendThor": {
                "num_views": "constructor kwarg",
                "chunk_size": f"hardcoded to {CHUNK_SIZE}, not overridable",
                "num_flow_steps":
                    f"hardcoded to {NUM_STEPS_DEFAULT}, not overridable",
            },
            "Pi05TorchFrontendRtx": {
                "num_views": "constructor kwarg",
                "chunk_size": "constructor kwarg `chunk_size`",
                "num_flow_steps": "constructor kwarg `num_steps`",
            },
        },
    }
    with open(out_dir / "synthetic_checkpoint_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote %s", out_dir / "synthetic_checkpoint_metadata.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic random-weight Pi0.5 checkpoint directory "
            "(model.safetensors + norm_stats.json) for shape/speed testing "
            "without a real trained checkpoint."))
    parser.add_argument("--out", required=True, type=pathlib.Path,
                        help="Output checkpoint directory (created if missing).")
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM,
                        help=(f"Action dimension. Only {ACTION_DIM} is "
                              "supported (fixed architecture constant; see "
                              "module docstring)."))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=(f"Action horizon / chunk size (default "
                              f"{CHUNK_SIZE}). Only actually honored by "
                              "Pi05TorchFrontendRtx's constructor kwarg of "
                              "the same name -- Pi05TorchFrontendThor "
                              "hardcodes it and ignores this value. Not "
                              "encoded into model.safetensors."))
    parser.add_argument("--num-flow-steps", type=int, default=NUM_STEPS_DEFAULT,
                        help=(f"Denoising step count (default "
                              f"{NUM_STEPS_DEFAULT}). Only actually honored "
                              "by Pi05TorchFrontendRtx's `num_steps` "
                              "constructor kwarg -- Pi05TorchFrontendThor "
                              "hardcodes it and ignores this value. Not "
                              "encoded into model.safetensors."))
    parser.add_argument("--num-views", type=int, default=2,
                        help=("Number of camera views. Recorded only in the "
                              "informational metadata sidecar -- pass the "
                              "same value to the frontend constructor; no "
                              "checkpoint tensor shape depends on it."))
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible weight values.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    generate_checkpoint(
        args.out, action_dim=args.action_dim, chunk_size=args.chunk_size,
        num_flow_steps=args.num_flow_steps, num_views=args.num_views,
        seed=args.seed)


if __name__ == "__main__":
    main()
