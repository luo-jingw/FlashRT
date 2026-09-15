#!/usr/bin/env python
"""ImageWAM real-checkpoint validation: compares FlashRT's verified
real-math modules (flash_rt/models/imagewam/real_*.py, opportunities.md
OPT-002) against ImageWAM's own real PyTorch model, loaded with REAL
trained weights from the released checkpoint.

**Run on real Thor hardware by the user, 2026-09-14, against the real
`yuyangalin/ImageWAM-FLUX.2-4B-LIBERO` release checkpoint**:
Backbone cosine=0.999927, ActionDiT cosine=0.999963 (see plan.md "Real
Thor Hardware Result -- Real-Checkpoint Validation"). A few env details
below were corrected against that real run (this project's own dev
machine still has no checkpoint/`imagewam`/`flux2` locally, so this
script itself is unchanged from what was actually run).

## Prerequisites (run this on the machine with the checkpoint)

  - `imagewam` package importable via `PYTHONPATH=<ImageWAM repo>/src`
    (NOT `pip install -e .` -- that pulls in the package's own pinned
    deps and downgraded Thor's `torch==2.9.1+cu130` to `2.7.1` in
    practice; point `PYTHONPATH` at `src/` instead).
  - `black-forest-labs/flux2` cloned at the pinned commit and
    importable: `FLUX2_SRC` env var pointing at it (see ImageWAM's own
    `docs/dependencies.md` / `README_zh.md` "模型准备" section).
  - Downloaded checkpoint files, env vars:
      `FLUX2_MODEL_PATH`   -- base FLUX.2 klein-base-4B safetensors
      `FLUX2_AE_MODEL_PATH` -- FLUX.2 autoencoder safetensors
      `CKPT_PATH`          -- ImageWAM release checkpoint file. **The
                              real release (yuyangalin/ImageWAM-FLUX.2-4B-LIBERO)
                              names this `model.pt`, not `checkpoint.pt`**
                              (the docstring originally guessed wrong
                              here -- confirmed on the real Thor run).
  - FlashRT itself importable (this repo).
  - `ACTION_DIM` env var (raw per-timestep action dimension). For the
    real LIBERO release this is **7** (LIBERO's own 7-DoF action
    space), confirmed against the release's sibling `config.yaml` (NOT
    `train_config.yaml` as originally guessed) -- still check this
    against your own release's config rather than trusting this
    default blindly for a different task/robot.

## What this validates, and what it deliberately skips

Only the CORE TRANSFORMER MATH (backbone `DoubleStreamBlock`/
`SingleStreamBlock` + ActionDiT `SlimFlux2DoubleBlock`/`SlimFlux2SingleBlock`)
against REAL TRAINED WEIGHTS. Does **not** need the real VAE or Qwen3
text encoder: `Flux2VideoExpert.pre_dit`/`ActionDiTFlux2.pre_dit` both
accept ALREADY-TOKENIZED inputs (`ref_image_hidden_states`, `context`,
`action_tokens`) directly -- feeding them random tensors of the REAL
shape exercises the exact same weight computation a real image/prompt
would, without needing to actually run the VAE/Qwen3 (this project's
whole established convention: random inputs are fine for validating
COMPUTATION correctness, only the WEIGHTS need to be real here).

Calls the model's own REAL orchestration functions for the reference
(`model.mot.prefill_flux2_video_cache` /
`model.mot.forward_flux2_action_with_video_cache`) rather than
lower-level block methods directly -- this is deliberate:
`DoubleStreamBlock.forward_kv_extract`/`causal_attn_fn` (a different,
LOWER-level function on the block itself) use a DIFFERENT mask rule
than what ImageWAM's own `MoT` orchestration actually uses at
inference time (see opportunities.md's "Major correction" for the full
story) -- calling the block method directly would silently validate
against the WRONG reference.

## LoRA-merge branch -- checked, not an issue for the LIBERO release

`ImageWAM.load_checkpoint`'s real source has a LoRA-merge branch for
`stack == "flux2"` (`merge_lora_state_dict_to_plain` /
`remap_plain_linear_keys_to_lora_base`) that runs unconditionally for
this stack -- meaning the real checkpoint's weights COULD in principle
be stored in a different key layout than the plain module structure
this script reads weights from directly
(`model.video_expert.transformer.double_blocks[i].img_attn.qkv.weight`
etc.). **Checked on the real Thor run**: `model.load_checkpoint(...)`
reported `missing_keys=0 unexpected_keys=0` for the LIBERO release --
the plain module attributes this script reads from directly are
correct as-is; no LoRA-layout mismatch for this specific checkpoint.
If you see missing/unexpected key warnings on a DIFFERENT release,
that is still the signal something needs investigating before trusting
the comparison below.
"""
from __future__ import annotations

import os
import sys

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.adaln import modulation, timestep_embedding_real, mlp_embedder
from flash_rt.models.imagewam.pipeline_real import imagewam_prefill_real
from flash_rt.models.imagewam.real_action_expert import (
    real_action_double_block_forward_fp16,
    real_action_single_block_forward_fp16,
)
from flash_rt.models.imagewam.rope import build_action_rope_table, build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16

# Real confirmed dims (this project's own _imagewam_thor_spec.py /
# benchmarks, cross-checked against Klein4BParams in the real
# flux2/model.py at the pinned commit).
HIDDEN, HD, NH, MLP_HIDDEN, JOINT_ATTN_DIM = 3072, 128, 24, 9216, 7680
ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH, ACTION_MLP_HIDDEN = 1024, 3072, 4096
NUM_DOUBLE, NUM_SINGLE = 5, 20
MAX_ACTION_HORIZON = 64
# CONFIRMED real (2026-09-15): x0=512 (Qwen3's own real max_length,
# confirmed via flux2.text_encoder.MAX_LENGTH and
# _imagewam_thor_spec.py's own declared context shape) -- superseding
# this script's own earlier x0=128 placeholder. a0=904 (real Thor +
# real FLUX.2-dev VAE against real libero_spatial_no_noops_lerobot
# frames, 224x448 input -> 8x VAE downsample + 2x2 patch merge ->
# 14x28 grid) -- superseding this script's own earlier 24x32=768 (a
# 384x512 input guess). A0-X0==392==REF_H*REF_W.
X0, A0 = 512, 904  # text tokens, text+ref tokens
NUM_ACTION = MAX_ACTION_HORIZON
REF_H, REF_W = 14, 28


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def load_real_model(action_dim: int):
    """Real loading sequence, ported from ImageWAM.from_flux2_klein_pretrained
    + ImageWAM.load_checkpoint (imagewam.py, read directly)."""
    flux2_src = os.environ.get("FLUX2_SRC")
    if flux2_src and flux2_src not in sys.path:
        sys.path.insert(0, flux2_src)
    from imagewam.models.backbones.imagewam import ImageWAM

    model = ImageWAM.from_flux2_klein_pretrained(
        flux2_model_path=os.environ["FLUX2_MODEL_PATH"],
        ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"],
        action_dit_config=dict(
            action_dim=action_dim,
            hidden_dim=ACTION_HIDDEN_DIM,
            num_heads=NH,
            attn_head_dim=HD,
            num_layers_double=NUM_DOUBLE,
            num_layers_single=NUM_SINGLE,
            mlp_ratio=4.0,
            max_action_horizon=MAX_ACTION_HORIZON,
        ),
        action_dit_pretrained_path=None,  # overwritten by load_checkpoint below
        flux2_src_path=flux2_src,
        variant="klein-base-4b",
        device=DEV,
        torch_dtype=torch.bfloat16,
    )
    model.load_checkpoint(os.environ["CKPT_PATH"])
    model.eval()
    return model


def _w(t: torch.Tensor) -> torch.Tensor:
    """Real nn.Linear weight (out,in), bf16 -> FlashRT GEMM (in,out), fp16."""
    return t.detach().t().contiguous().to(FP16)


def _v(t: torch.Tensor) -> torch.Tensor:
    """1D vector (e.g. a norm scale), bf16 -> fp16."""
    return t.detach().contiguous().to(FP16)


def extract_backbone_double_weights(block) -> dict:
    """block: one of model.video_expert.transformer.double_blocks[i]."""
    w = {}
    for side, attn, mlp in (("txt", block.txt_attn, block.txt_mlp), ("img", block.img_attn, block.img_mlp)):
        w[f"{side}_qkv"] = _w(attn.qkv.weight)
        w[f"{side}_proj"] = _w(attn.proj.weight)
        w[f"{side}_mlp_in"] = _w(mlp[0].weight)
        w[f"{side}_mlp_out"] = _w(mlp[2].weight)
        w[f"{side}_query_norm"] = _v(attn.norm.query_norm.scale)
        w[f"{side}_key_norm"] = _v(attn.norm.key_norm.scale)
    return w


def extract_backbone_single_weights(block) -> dict:
    """block: one of model.video_expert.transformer.single_blocks[i].
    Splits the real fused linear1/linear2 into the separate GEMMs
    real_single_stream_block.py expects (see that module's own
    docstring for why this split is mathematically identical). Uses the
    module-level NH/HD constants directly (this script's fixed real
    dims, not a parameter) -- intentional, since this script only ever
    runs against the one real checkpoint whose architecture those
    constants already describe; not written to be dimension-generic."""
    attn_dim = NH * HD
    l1 = block.linear1.weight.detach()  # (3*attn_dim + 2*mlp_hidden, hidden)
    l2 = block.linear2.weight.detach()  # (hidden, attn_dim + mlp_hidden)
    qkv_w = l1[: 3 * attn_dim, :]
    mlp_in_w = l1[3 * attn_dim:, :]
    attn_out_w = l2[:, :attn_dim]
    mlp_out_w = l2[:, attn_dim:]
    return {
        "qkv": _w(qkv_w),
        "mlp_in": _w(mlp_in_w),
        "attn_out": _w(attn_out_w),
        "mlp_out": _w(mlp_out_w),
        "query_norm": _v(block.norm.query_norm.scale),
        "key_norm": _v(block.norm.key_norm.scale),
    }


def extract_action_double_weights(block) -> dict:
    """block: model.action_expert.double_blocks[i] (SlimFlux2DoubleBlock,
    IMG-ONLY -- see real_action_expert.py's own docstring)."""
    return {
        "qkv": _w(block.img_attn.qkv.weight),
        "proj": _w(block.img_attn.proj.weight),
        "mlp_in": _w(block.img_mlp[0].weight),
        "mlp_out": _w(block.img_mlp[2].weight),
        "query_norm": _v(block.img_attn.norm.query_norm.scale),
        "key_norm": _v(block.img_attn.norm.key_norm.scale),
    }


def extract_action_single_weights(block) -> dict:
    """block: model.action_expert.single_blocks[i] (SlimFlux2SingleBlock).
    Uses the module-level NH/HD constants directly, same as
    extract_backbone_single_weights above (not dimension-generic)."""
    attn_dim = NH * HD
    l1 = block.linear1.weight.detach()
    l2 = block.linear2.weight.detach()
    qkv_w = l1[: 3 * attn_dim, :]
    mlp_in_w = l1[3 * attn_dim:, :]
    attn_out_w = l2[:, :attn_dim]
    mlp_out_w = l2[:, attn_dim:]
    return {
        "qkv": _w(qkv_w),
        "mlp_in": _w(mlp_in_w),
        "attn_out": _w(attn_out_w),
        "mlp_out": _w(mlp_out_w),
        "query_norm": _v(block.norm.query_norm.scale),
        "key_norm": _v(block.norm.key_norm.scale),
    }


def real_modulation(model, timestep: torch.Tensor):
    """Real vec/modulation computation, matching
    Flux2VideoExpert.pre_dit's own real call sequence directly."""
    from flux2.model import timestep_embedding

    transformer = model.video_expert.transformer
    vec = transformer.time_in(timestep_embedding(timestep, 256))
    mod_img = transformer.double_stream_modulation_img(vec)
    mod_txt = transformer.double_stream_modulation_txt(vec)
    mod_single, _ = transformer.single_stream_modulation(vec)
    return vec, mod_img, mod_txt, mod_single


def real_action_modulation(model, timestep: torch.Tensor):
    from flux2.model import timestep_embedding

    action_expert = model.action_expert
    vec = action_expert.time_in(timestep_embedding(timestep, 256))
    mod_img = action_expert.double_stream_modulation_img(vec)
    mod_single, _ = action_expert.single_stream_modulation(vec)
    return vec, mod_img, mod_single


def _mod_to_fp16(mod):
    """Real mod tuples hold bf16 tensors; cast to fp16 for FlashRT."""
    def cast3(m):
        return tuple(t.to(FP16) for t in m)
    if isinstance(mod, tuple) and len(mod) == 2 and isinstance(mod[0], tuple):
        return cast3(mod[0]), cast3(mod[1])
    return cast3(mod)


def main():
    action_dim = int(os.environ.get("ACTION_DIM", "7"))
    print(f"Loading real ImageWAM model (action_dim={action_dim}, "
          f"CHECK this matches the release's own train_config.yaml)...")
    model = load_real_model(action_dim)
    print("Loaded.")

    torch.manual_seed(0)
    # Random inputs of the REAL shape -- see module docstring for why
    # this is sufficient (validates computation, not semantic content).
    context = torch.randn(1, X0, JOINT_ATTN_DIM, dtype=torch.bfloat16, device=DEV)
    ref_tokens = torch.randn(1, A0 - X0, HD, dtype=torch.bfloat16, device=DEV)
    ref_img_ids = model.video_expert.build_img_ids(
        batch_size=1, token_height=REF_H, token_width=REF_W,
        time_value=10.0, device=DEV, dtype=torch.bfloat16)
    empty_target = ref_tokens.new_zeros(1, 0, HD)
    empty_target_ids = ref_img_ids.new_zeros(1, 0, 4)
    video_timestep = torch.zeros(1, dtype=torch.bfloat16, device=DEV)

    with torch.no_grad():
        video_pre = model.video_expert.pre_dit(
            x=empty_target, timestep=video_timestep, context=context,
            context_mask=None, ref_image_hidden_states=ref_tokens,
            target_img_ids=empty_target_ids, ref_img_ids=ref_img_ids,
        )
        prefix_mask = model._build_mot_attention_mask_flux2(
            batch_size=1, txt_len=X0, target_len=0, cond_len=A0 - X0,
            action_len=0, device=DEV, text_attention_mask=video_pre["text_mask"],
        )
        video_kv_cache = model.mot.prefill_flux2_video_cache(
            video_tokens=video_pre["tokens"], video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"], attention_mask=prefix_mask,
        )
    ref_backbone_out = torch.cat(
        [video_kv_cache["final_video"]["txt"], video_kv_cache["final_video"]["img"]], dim=1)[0]
    print(f"Real backbone forward done. Final shape: {tuple(ref_backbone_out.shape)}")

    # -- FlashRT side: extract real weights, run imagewam_prefill_real --
    double_weights = [extract_backbone_double_weights(b) for b in model.video_expert.transformer.double_blocks]
    single_weights = [extract_backbone_single_weights(b) for b in model.video_expert.transformer.single_blocks]

    txt_flat = context[0].to(FP16).contiguous()
    txt_hidden = torch.zeros(X0, HIDDEN, dtype=FP16, device=DEV)
    txt_in_w = _w(model.video_expert.transformer.txt_in.weight)
    gemm = fvk.GemmRunner()
    gemm.fp16_nn(txt_flat.data_ptr(), txt_in_w.data_ptr(), txt_hidden.data_ptr(), X0, HIDDEN, JOINT_ATTN_DIM, 0)
    img_flat = ref_tokens[0].to(FP16).contiguous()
    img_hidden = torch.zeros(A0 - X0, HIDDEN, dtype=FP16, device=DEV)
    img_in_w = _w(model.video_expert.transformer.img_in.weight)
    gemm.fp16_nn(img_flat.data_ptr(), img_in_w.data_ptr(), img_hidden.data_ptr(), A0 - X0, HIDDEN, HD, 0)
    torch.cuda.synchronize()

    _, mod_img, mod_txt, mod_single = real_modulation(model, video_timestep)
    mod_img_fp16 = _mod_to_fp16(mod_img)
    mod_txt_fp16 = _mod_to_fp16(mod_txt)
    mod_single_fp16 = _mod_to_fp16(mod_single)

    table = build_backbone_rope_table(X0, REF_H, REF_W, device=DEV)
    ctx = fvk.FvkContext()
    attn_scale = 1.0 / (HD ** 0.5)

    flashrt_out = imagewam_prefill_real(
        gemm, ctx, txt_hidden, img_hidden, double_weights, single_weights,
        mod_txt_fp16, mod_img_fp16, mod_single_fp16, table, NH, HD, HIDDEN, MLP_HIDDEN, attn_scale)

    cos_backbone = _cosine(ref_backbone_out, flashrt_out)
    print(f"Backbone real-checkpoint validation: cosine={cos_backbone:.6f} "
          f"(expect close to but maybe not exactly 1.0 -- real weights are "
          f"bf16, FlashRT is fp16, a real precision difference)")

    # -- ActionDiT side --
    action_timestep = torch.zeros(1, dtype=torch.bfloat16, device=DEV)
    action_tokens_raw = torch.randn(1, NUM_ACTION, action_dim, dtype=torch.bfloat16, device=DEV)
    with torch.no_grad():
        action_pre = model.action_expert.pre_dit(
            action_tokens=action_tokens_raw, timestep=action_timestep, context=None, context_mask=None)
        full_mask = model._build_mot_attention_mask_flux2(
            batch_size=1, txt_len=X0, target_len=0, cond_len=A0 - X0,
            action_len=NUM_ACTION, device=DEV, text_attention_mask=video_pre["text_mask"])
        ref_action_out = model.mot.forward_flux2_action_with_video_cache(
            action_tokens=action_pre["tokens"], action_ids=action_pre["ids"],
            action_t_mod=action_pre["t_mod"], video_kv_cache=video_kv_cache,
            attention_mask=full_mask, video_seq_len=A0,
        )[0]
    print(f"Real ActionDiT forward done. Final shape: {tuple(ref_action_out.shape)}")

    action_double_weights = [extract_action_double_weights(b) for b in model.action_expert.double_blocks]
    action_single_weights = [extract_action_single_weights(b) for b in model.action_expert.single_blocks]
    _, action_mod_img, action_mod_single = real_action_modulation(model, action_timestep)
    action_mod_img_fp16 = _mod_to_fp16(action_mod_img)
    action_mod_single_fp16 = _mod_to_fp16(action_mod_single)

    action_hidden = torch.zeros(NUM_ACTION, ACTION_HIDDEN_DIM, dtype=FP16, device=DEV)
    action_encoder_w = _w(model.action_expert.action_encoder.weight)
    action_encoder_b = _v(model.action_expert.action_encoder.bias)
    action_in = action_tokens_raw[0].to(FP16).contiguous()
    gemm.fp16_nn(action_in.data_ptr(), action_encoder_w.data_ptr(), action_hidden.data_ptr(),
                 NUM_ACTION, ACTION_HIDDEN_DIM, action_dim, 0)
    action_hidden += action_encoder_b  # real action_encoder HAS bias, unlike everything else

    action_rope_table = build_action_rope_table(NUM_ACTION, device=DEV)
    # this layer's own real per-layer backbone K/V cache, real per-head, fp16
    cached_double_kv = [(c["k"][0].reshape(A0, NH, HD).to(FP16).contiguous(),
                         c["v"][0].reshape(A0, NH, HD).to(FP16).contiguous())
                        for c in video_kv_cache["double"]]
    cached_single_kv = [(c["k"][0].reshape(A0, NH, HD).to(FP16).contiguous(),
                         c["v"][0].reshape(A0, NH, HD).to(FP16).contiguous())
                        for c in video_kv_cache["single"]]

    for i, w in enumerate(action_double_weights):
        cached_k, cached_v = cached_double_kv[i]
        action_hidden = real_action_double_block_forward_fp16(
            gemm, action_hidden, w, action_mod_img_fp16, action_rope_table,
            cached_k, cached_v, NH, HD, ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN, attn_scale)
    for i, w in enumerate(action_single_weights):
        cached_k, cached_v = cached_single_kv[i]
        action_hidden = real_action_single_block_forward_fp16(
            gemm, action_hidden, w, action_mod_single_fp16, action_rope_table,
            cached_k, cached_v, NH, HD, ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN, attn_scale)

    cos_action = _cosine(ref_action_out, action_hidden)
    print(f"ActionDiT real-checkpoint validation: cosine={cos_action:.6f}")

    print("\n=== SUMMARY ===")
    print(f"Backbone cosine: {cos_backbone:.6f}")
    print(f"ActionDiT cosine: {cos_action:.6f}")
    print("If either is well below ~0.99, check ACTION_DIM first (most "
          "likely wrong-guess parameter), then the LoRA-merge note in "
          "this script's own module docstring.")


if __name__ == "__main__":
    main()
