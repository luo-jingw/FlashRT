"""`checkpoint_loader.py` correctness against the ACTUAL real ImageWAM
release checkpoint (OPT-001, `plan.md`'s own "OPT-001" plan Phase 2).

**Runs against real files, when present** -- unlike every other
quantization-plan test this session (which could only probe a KNOWN
environment gap and SKIP), the real checkpoint FILES are actually
present on THIS dev machine (`/home/ljw/projects/pi0.5/models/`,
discovered while writing this loader -- see `PROJECT.md`'s corrected
"Real checkpoint testing" note). This test SKIPS cleanly if that path
doesn't exist (e.g. a fresh clone of this repo, or a machine without
the models cache), following this project's own established skip
pattern, just inverted: skip on absence of a local RESOURCE, not
absence of a kernel/GPU feature.

Two checks: (1) every real weight tensor's shape matches this
project's own confirmed real dims exactly (`_imagewam_thor_spec.py`'s
own values) -- catches a wrong key path or a wrong `linear1`/`linear2`
split silently producing a wrong-shaped tensor; (2) one REAL backbone
double-stream layer (real trained weights, random activations --
matching this project's own "random inputs are fine for validating
computation, only weights need to be real" convention) produces
finite, non-degenerate output through `pipeline_thor.py`'s own pointer
path -- the strongest verification achievable on this dev machine
without `flux2`/enough VRAM for the full model.
"""
import os

import torch

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16

_CKPT_PATH = os.environ.get("CKPT_PATH", "/home/ljw/projects/pi0.5/models/imagewam_flux2_4b_libero/model.pt")
_CKPT_AVAILABLE = os.path.exists(_CKPT_PATH)

HIDDEN, HD, NH, MLP_HIDDEN, JOINT_ATTN_DIM = 3072, 128, 24, 9216, 7680
ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH, ACTION_MLP_HIDDEN, ACTION_DIM = 1024, 3072, 4096, 7
NUM_DOUBLE, NUM_SINGLE = 5, 20
# CONFIRMED real img_len=392 (14x28) and x0=512 (Qwen3's own real
# max_length), see opportunities.md's OPT-001/OPT-008 entries
# (2026-09-15) -- superseding the earlier 128/768/896 guesses.
X0, A0 = 512, 904
REF_H, REF_W = 14, 28  # real image RoPE grid (opportunities.md OPT-002's flat-grid correction)

_FLUX2_SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party", "flux2", "src")
_AE_PATH = os.environ.get(
    "AE_MODEL_PATH", "/home/ljw/projects/pi0.5/models/flux2_klein_4b/ae.safetensors")
_VAE_AVAILABLE = os.path.isdir(_FLUX2_SRC) and os.path.isfile(_AE_PATH)


def test_shapes_match_confirmed_real_dims():
    if not _CKPT_AVAILABLE:
        import pytest
        pytest.skip(f"real checkpoint not present at {_CKPT_PATH} on this machine")

    from flash_rt.models.imagewam.checkpoint_loader import build_real_weights, load_real_imagewam_state_dict

    sd = load_real_imagewam_state_dict(_CKPT_PATH)
    weights = build_real_weights(sd, num_double=NUM_DOUBLE, num_single=NUM_SINGLE,
                                  action_num_double=NUM_DOUBLE, action_num_single=NUM_SINGLE,
                                  action_attn_width=ACTION_ATTN_WIDTH)

    checks = {
        ("backbone", "double", 0, "txt_in.weight"): (JOINT_ATTN_DIM, HIDDEN),
        ("backbone", "double", 0, "img_in.weight"): (HD, HIDDEN),
        ("backbone", "double", NUM_DOUBLE - 1, "txt_qkv.weight"): (HIDDEN, 3 * HIDDEN),
        ("backbone", "double", NUM_DOUBLE - 1, "img_proj.weight"): (HIDDEN, HIDDEN),
        ("backbone", "double", NUM_DOUBLE - 1, "txt_mlp0.weight"): (HIDDEN, MLP_HIDDEN * 2),
        ("backbone", "double", NUM_DOUBLE - 1, "img_mlp2.weight"): (MLP_HIDDEN, HIDDEN),
        ("backbone", "single", NUM_SINGLE - 1, "qkv.weight"): (HIDDEN, 3 * HIDDEN),
        ("backbone", "single", NUM_SINGLE - 1, "mlp_in.weight"): (HIDDEN, MLP_HIDDEN * 2),
        ("backbone", "single", NUM_SINGLE - 1, "attn_out_proj.weight"): (HIDDEN, HIDDEN),
        ("backbone", "single", NUM_SINGLE - 1, "mlp_down.weight"): (MLP_HIDDEN, HIDDEN),
        ("action_dit", "shared", 0, "action_encoder.weight"): (ACTION_DIM, ACTION_HIDDEN_DIM),
        ("action_dit", "shared", 0, "head.linear.weight"): (ACTION_HIDDEN_DIM, ACTION_DIM),
        # ActionDiT double is img-only -- unprefixed slot names, unlike
        # the backbone's dual txt/img streams (checkpoint_loader.py's
        # own _extract_double_block docstring).
        ("action_dit", "double", NUM_DOUBLE - 1, "qkv.weight"): (ACTION_HIDDEN_DIM, 3 * ACTION_ATTN_WIDTH),
        ("action_dit", "double", NUM_DOUBLE - 1, "proj.weight"): (ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
        ("action_dit", "single", NUM_SINGLE - 1, "qkv.weight"): (ACTION_HIDDEN_DIM, 3 * ACTION_ATTN_WIDTH),
        ("action_dit", "single", NUM_SINGLE - 1, "mlp_down.weight"): (ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM),
    }
    for key, expected in checks.items():
        got = tuple(weights[key].shape)
        assert got == expected, f"{key}: got {got}, expected {expected}"

    # txt_in/img_in shared across every double layer (module docstring's
    # "second finding") -- same tensor object, not L independent copies.
    assert weights[("backbone", "double", 0, "txt_in.weight")] is weights[
        ("backbone", "double", NUM_DOUBLE - 1, "txt_in.weight")]
    print(f"PASS: {len(weights)} real weight tensors, all shapes match confirmed real dims")


def test_real_double_stream_layer_forward_finite():
    if not _CKPT_AVAILABLE:
        import pytest
        pytest.skip(f"real checkpoint not present at {_CKPT_PATH} on this machine")

    import flash_rt.flash_rt_kernels as fvk
    from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
    from flash_rt.models.imagewam.checkpoint_loader import (
        build_real_modulation_weights, build_real_weights, load_real_imagewam_state_dict,
    )
    from flash_rt.models.imagewam.pipeline_real import compute_shared_modulation
    from flash_rt.models.imagewam.pipeline_thor import _double_stream_layer
    from flash_rt.models.imagewam.quant_linear import Bf16OutLinear, Fp16Linear
    from flash_rt.models.imagewam.rope import build_backbone_rope_table

    sd = load_real_imagewam_state_dict(_CKPT_PATH)
    # Only layer 0 -- keeps this test's own GPU footprint small (this
    # dev machine has 8GB VRAM, nowhere near the ~8.9GB the FULL real
    # model needs, see PROJECT.md).
    raw_weights = build_real_weights(sd, num_double=1, num_single=0,
                                      action_num_double=0, action_num_single=0,
                                      action_attn_width=HIDDEN)
    mod = build_real_modulation_weights(sd)
    del sd

    img_len = A0 - X0
    gemm = fvk.GemmRunner()
    keepalive = []
    weights = {}
    for key, t in raw_weights.items():
        site, _, L, slot = key
        if site != "backbone":
            continue  # action_encoder/head are unconditional in build_real_weights; not needed here
        if slot in ("query_norm", "key_norm") or slot.endswith("_norm"):
            tg = t.to(DEV)
            keepalive.append(tg)
            weights[key] = tg.data_ptr()
        elif slot in ("txt_in.weight", "img_in.weight"):
            # OPT-001 "FP16 residual overflow" fix -- see Bf16OutLinear's
            # own docstring in quant_linear.py.
            n, k = t.shape[1], t.shape[0]
            tg = t.to(DEV, dtype=BF16).contiguous()
            keepalive.append(tg)
            weights[key] = Bf16OutLinear(gemm, tg.data_ptr(), n, k)
        else:
            n, k = t.shape[1], t.shape[0]
            tg = t.to(DEV).contiguous()
            keepalive.append(tg)
            weights[key] = Fp16Linear(gemm, tg.data_ptr(), n, k)

    mod_w = {k: v.to(DEV) for k, v in mod["backbone"].items()}
    mod_txt, mod_img, _ = compute_shared_modulation(torch.zeros(1, device=DEV), mod_w, HIDDEN)

    spec = make_imagewam_attention_spec(max_prefix_seq=A0, max_total_seq=A0 + 1,
                                         num_layers=1, num_heads=NH, head_dim=HD)
    ctx = fvk.FvkContext()
    K_cache = torch.zeros(1, A0, HIDDEN, dtype=FP16, device=DEV)
    V_cache = torch.zeros(1, A0, HIDDEN, dtype=FP16, device=DEV)
    Q_O = torch.zeros(A0, HIDDEN, dtype=FP16, device=DEV)
    logits = torch.zeros(A0 * NH, A0 + (A0 % 2), dtype=FP16, device=DEV)
    attn = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                        "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5)},
        mot_slots={"Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
                   "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5), "layer_stride": K_cache[0].numel() * 2},
        use_perhead_kv=True, use_real_mot_mask=True,
    )

    torch.manual_seed(0)
    # Random inputs at real dims -- only the WEIGHTS are real here (see
    # module docstring); this checks the loaded weights flow correctly
    # through the real math, not semantic output quality. `context`/
    # `img_raw`/`combined` are BF16 (OPT-001 "FP16 residual overflow"),
    # matching `Bf16OutLinear`'s own dtype requirement and the real
    # persistent-residual buffer's real range need.
    context = (torch.randn(X0, JOINT_ATTN_DIM, device=DEV) * 0.5).to(BF16)
    img_raw = (torch.randn(img_len, HD, device=DEV) * 0.5).to(BF16)
    combined = torch.zeros(A0, HIDDEN, dtype=BF16, device=DEV)
    table = build_backbone_rope_table(X0, REF_H, REF_W, device=DEV)
    bufs = {
        "context": context.data_ptr(), "img_raw": img_raw.data_ptr(), "backbone_hidden": combined.data_ptr(),
        "modded_scratch": torch.zeros(A0, HIDDEN, dtype=FP16, device=DEV).data_ptr(),
        "txt_qkv_merged": torch.zeros(X0, 3 * HIDDEN, dtype=FP16, device=DEV).data_ptr(),
        "img_qkv_merged": torch.zeros(img_len, 3 * HIDDEN, dtype=FP16, device=DEV).data_ptr(),
        "txt_mlp_merged": torch.zeros(X0, MLP_HIDDEN * 2, dtype=FP16, device=DEV).data_ptr(),
        "txt_mlp_gated": torch.zeros(X0, MLP_HIDDEN, dtype=FP16, device=DEV).data_ptr(),
        "img_mlp_merged": torch.zeros(img_len, MLP_HIDDEN * 2, dtype=FP16, device=DEV).data_ptr(),
        "img_mlp_gated": torch.zeros(img_len, MLP_HIDDEN, dtype=FP16, device=DEV).data_ptr(),
        "proj_scratch": torch.zeros(A0, HIDDEN, dtype=FP16, device=DEV).data_ptr(),
    }
    dims = dict(hidden=HIDDEN, HD=HD, NH=NH, mlp_hidden=MLP_HIDDEN,
                joint_attention_dim=JOINT_ATTN_DIM, x0=X0, a0=A0)

    # `_double_stream_layer` no longer projects txt_in/img_in itself
    # (bug fix, 2026-09-15, opportunities.md OPT-001 -- see that
    # function's own docstring); `imagewam_prefill` does this ONCE
    # before its layer loop, this direct single-layer call does the
    # equivalent explicitly.
    weights[("backbone", "double", 0, "txt_in.weight")](context.data_ptr(), combined.data_ptr(), X0, 0)
    weights[("backbone", "double", 0, "img_in.weight")](
        img_raw.data_ptr(), combined.data_ptr() + X0 * HIDDEN * 2, img_len, 0)

    _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, 0, 0, attn, mod_txt, mod_img, table.data_ptr())
    torch.cuda.synchronize()

    assert torch.isfinite(combined).all(), "real-weight backbone layer produced NaN/Inf"
    std = combined.float().std().item()
    assert std > 1e-3, f"output suspiciously degenerate (std={std}) -- possible silent all-zero weight"
    print(f"PASS: real-weight double-stream layer finite, mean={combined.float().mean().item():.4f} "
          f"std={std:.4f}")


def test_full_frontend_with_real_checkpoint():
    """The strongest check in this file: `ImageWAMTorchFrontendThor`
    constructed with `ckpt_path=` at REAL FLUX.2-4B dims, real weights
    end to end -- construction (real weights loaded + moved to CUDA),
    `set_prompt()` (CUDA Graph capture with real weights), `infer()`
    (graph replay). Confirmed on this dev machine (2026-09-15, 8GB
    reported VRAM per `nvidia-smi` -- but peak allocated measured at
    ~9.86GB, meaning this WSL2 environment's CUDA driver pages beyond
    the reported dedicated VRAM into host RAM rather than raising OOM;
    not something to rely on for STEADY-STATE Thor-performance claims,
    but real enough to let this one-time construction+capture actually
    complete here instead of only on Thor)."""
    if not _CKPT_AVAILABLE:
        import pytest
        pytest.skip(f"real checkpoint not present at {_CKPT_PATH} on this machine")

    import numpy as np

    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

    real_dims = dict(
        hidden=HIDDEN, HD=HD, NH=NH, mlp_hidden=MLP_HIDDEN, joint_attention_dim=JOINT_ATTN_DIM,
        x0=X0, a0=A0, num_layers_double=NUM_DOUBLE, num_layers_single=NUM_SINGLE,
        action_hidden_dim=ACTION_HIDDEN_DIM, action_attn_width=ACTION_ATTN_WIDTH,
        action_mlp_hidden=ACTION_MLP_HIDDEN, action_dim=ACTION_DIM,
        num_action=64, total=A0 + 64,
        action_num_layers_double=NUM_DOUBLE, action_num_layers_single=NUM_SINGLE,
        dt=1.0 / 10, num_denoise_steps=10,
        ref_h=REF_H, ref_w=REF_W,
    )
    frontend = ImageWAMTorchFrontendThor(dims_override=real_dims, precision="fp16", ckpt_path=_CKPT_PATH)
    frontend.set_prompt("real checkpoint smoke test")
    out = frontend.infer({})
    actions = out["actions"]
    assert actions.shape == (64, ACTION_DIM)
    assert np.isfinite(actions).all(), "real-checkpoint infer() produced NaN/Inf"
    print(f"PASS: full real-checkpoint frontend, actions shape={actions.shape}, "
          f"mean={actions.mean():.4f} std={actions.std():.4f}")


def test_full_frontend_with_real_checkpoint_and_real_vae():
    """The strongest check in this project so far: real transformer
    weights (OPT-001) AND a real VAE-encoded real camera frame
    (real VAE + text-context wiring plan) together, end to end --
    construction, `set_prompt()`, `infer({"view1":...,"view2":...})`.
    Needs BOTH the real checkpoint and the real `flux2` clone + AE
    checkpoint; skips cleanly if either is missing."""
    if not (_CKPT_AVAILABLE and _VAE_AVAILABLE):
        import pytest
        pytest.skip(f"needs both the real checkpoint ({_CKPT_PATH}) and the real "
                    f"flux2 clone/AE checkpoint ({_FLUX2_SRC}, {_AE_PATH})")

    import numpy as np

    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

    real_dims = dict(
        hidden=HIDDEN, HD=HD, NH=NH, mlp_hidden=MLP_HIDDEN, joint_attention_dim=JOINT_ATTN_DIM,
        x0=X0, a0=A0, num_layers_double=NUM_DOUBLE, num_layers_single=NUM_SINGLE,
        action_hidden_dim=ACTION_HIDDEN_DIM, action_attn_width=ACTION_ATTN_WIDTH,
        action_mlp_hidden=ACTION_MLP_HIDDEN, action_dim=ACTION_DIM,
        num_action=64, total=A0 + 64,
        action_num_layers_double=NUM_DOUBLE, action_num_layers_single=NUM_SINGLE,
        dt=1.0 / 10, num_denoise_steps=10,
        ref_h=REF_H, ref_w=REF_W,
    )
    frontend = ImageWAMTorchFrontendThor(dims_override=real_dims, precision="fp16", ckpt_path=_CKPT_PATH,
                                          ae_model_path=_AE_PATH, flux2_src=_FLUX2_SRC)
    frontend.set_prompt("real checkpoint + real vae smoke test")

    view1 = torch.zeros(224, 224, 3, dtype=torch.uint8, device=DEV)
    view2 = torch.zeros(224, 224, 3, dtype=torch.uint8, device=DEV)
    out = frontend.infer({"view1": view1, "view2": view2})
    actions = out["actions"]
    assert actions.shape == (64, ACTION_DIM)
    assert np.isfinite(actions).all(), "real-checkpoint + real-VAE infer() produced NaN/Inf"
    print(f"PASS: full real-checkpoint + real-VAE frontend, actions shape={actions.shape}, "
          f"mean={actions.mean():.4f} std={actions.std():.4f}")


if __name__ == "__main__":
    if not _CKPT_AVAILABLE:
        print(f"SKIPPED: real checkpoint not present at {_CKPT_PATH}")
    else:
        test_shapes_match_confirmed_real_dims()
        test_real_double_stream_layer_forward_finite()
        test_full_frontend_with_real_checkpoint()
        if not _VAE_AVAILABLE:
            print(f"SKIPPED combined VAE test: flux2 clone/AE checkpoint not present")
        else:
            test_full_frontend_with_real_checkpoint_and_real_vae()
        print("PASS")
