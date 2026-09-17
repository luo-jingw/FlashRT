"""Real closed-loop robot-state conditioning (proprio in, action out) --
found 2026-09-15 while scoping real closed-loop testing
(opportunities.md). Ported facts, confirmed by directly reading the
real `imagewam` package source and the real LIBERO release checkpoint
on this dev machine, NOT assumed:

- The real checkpoint's `config.yaml` has `proprio_dim: 8`,
  `pack_proprio_after_text: true`.
- The real checkpoint's top-level payload (a SIBLING of `mot`, not
  inside it) has a real trained `proprio_encoder` -- `weight`
  `(7680, 8)`, `bias` `(7680,)` -- confirmed via
  `torch.load(ckpt_path, mmap=True)['proprio_encoder']`.
- `imagewam.py`'s own real `_append_proprio_to_context`
  (`pack_proprio_after_text=True` branch) inserts the proprio token at
  row `context_mask.sum()` (right after the last real text token),
  shifting padding one row later -- NOT a fixed position, and NOT a
  simple append at the end.
- `imagewam.py`'s own real `_append_proprio_to_context_if_enabled`
  raises `ValueError` if `proprio_encoder` exists but `proprio is None`
  -- proprio is NOT optional for a checkpoint trained with it.
- The real `config.yaml` for this release: `use_stepwise_action_norm:
  false`, `norm_default_mode: min/max`, `norm_exception_mode: null` --
  so BOTH `state` (proprio, in) and `action` (out) use plain
  `global_min`/`global_max` linear normalization (never `stepwise_*`,
  `q01/q99`, or `z-score`, for THIS release specifically).
"""
import os

import torch

DEV = "cuda"
BF16 = torch.bfloat16

_CKPT_PATH = "/home/ljw/projects/pi0.5/models/imagewam_flux2_4b_libero/model.pt"
_STATS_PATH = "/home/ljw/projects/pi0.5/models/imagewam_flux2_4b_libero/dataset_stats.json"
_CKPT_AVAILABLE = os.path.exists(_CKPT_PATH)
_STATS_AVAILABLE = os.path.exists(_STATS_PATH)


def test_min_max_normalizer_round_trips_and_matches_real_stats():
    if not _STATS_AVAILABLE:
        import pytest
        pytest.skip(f"real dataset_stats.json not present at {_STATS_PATH} on this machine")

    from flash_rt.models.imagewam.dataset_stats import load_dataset_stats, load_real_normalizers

    state_norm, action_norm = load_real_normalizers(_STATS_PATH)
    stats = load_dataset_stats(_STATS_PATH)

    # A real action value (the dataset's own mean) must round-trip through
    # forward (normalize) -> backward (denormalize) to itself.
    mean_action = torch.tensor([stats["action"]["default"]["global_mean"]], device=DEV)
    normed = action_norm.forward(mean_action)
    back = action_norm.backward(normed)
    diff = (back - mean_action).abs().max().item()
    assert diff < 1e-4, f"round-trip diff too large: {diff}"

    # forward() must land within [-1,1] for any value inside [global_min, global_max]
    # (a real proprio state, not just the mean) -- confirms the scale/offset sign.
    gmin = torch.tensor([stats["state"]["default"]["global_min"]], device=DEV)
    gmax = torch.tensor([stats["state"]["default"]["global_max"]], device=DEV)
    assert torch.allclose(state_norm.forward(gmin), torch.full_like(gmin, -1.0), atol=1e-4)
    assert torch.allclose(state_norm.forward(gmax), torch.full_like(gmax, 1.0), atol=1e-4)
    print("PASS: MinMaxNormalizer round-trips and matches real dataset_stats.json bounds")


def test_proprio_scatter_matches_real_insertion_rule():
    """`_set_context_with_optional_proprio` must replicate `imagewam.py`'s
    own real `_append_proprio_to_context` scatter EXACTLY: real tokens
    keep their rank, the proprio slot lands at `valid_counts`, padding
    shifts one row later. Toy dims/random weights -- this checks the
    WIRING (row placement), not real-checkpoint accuracy."""
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

    JA = 16
    # Wiring test (proprio row placement), not a precision test -- pin
    # fp16 explicitly since the class default is now "nvfp4" (Stage 3
    # decision, opportunities.md), which requires a Blackwell/Thor
    # build this dev machine doesn't have.
    f = ImageWAMTorchFrontendThor(
        dims_override=dict(x0=9, a0=14, total=18, joint_attention_dim=JA, proprio_dim=3),
        precision="fp16")

    text_ctx = torch.arange(8 * JA, dtype=torch.float32, device=DEV).reshape(8, JA)
    text_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool, device=DEV)
    f.set_prompt(context=text_ctx.to(BF16), context_mask=text_mask)

    assert f._proprio_row == 3, f"expected proprio_row=3 (valid_counts), got {f._proprio_row}"
    ctx = f._context.float()
    assert torch.allclose(ctx[:3], text_ctx[:3]), "real tokens must keep their original rank"
    assert (ctx[3] == 0).all(), "proprio row must be zero before the first infer() call"
    assert torch.allclose(ctx[4:9], text_ctx[3:8]), "padding must shift exactly one row later"

    # infer() must overwrite ONLY the proprio row, leaving text/padding untouched.
    out = f.infer({"proprio": [0.1, 0.2, 0.3]})
    import numpy as np
    assert np.isfinite(out["actions"]).all()
    ctx2 = f._context.float()
    assert not (ctx2[3] == 0).all(), "proprio row must be non-zero after infer()"
    assert torch.allclose(ctx2[:3], ctx[:3]) and torch.allclose(ctx2[4:9], ctx[4:9]), \
        "infer() must not touch the text/padding rows, only the proprio row"
    print("PASS: proprio scatter matches the real insertion rule; infer() only touches its own row")


def test_infer_raises_on_missing_proprio():
    """Matches `_append_proprio_to_context_if_enabled`'s own real
    `raise ValueError(...)` when `proprio_encoder` exists but no
    `proprio` is given -- proprio is NOT optional once enabled."""
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

    # Wiring test (missing-proprio contract), not a precision test --
    # pin fp16 explicitly, same reason as above.
    f = ImageWAMTorchFrontendThor(dims_override=dict(proprio_dim=8), precision="fp16")
    f.set_prompt()
    try:
        f.infer({})
        raise AssertionError("expected ValueError for missing observation['proprio']")
    except ValueError as e:
        assert "proprio" in str(e)
    print("PASS: infer() raises on missing proprio, matching the real model's own contract")


def test_real_proprio_encoder_loads_with_real_checkpoint():
    if not _CKPT_AVAILABLE:
        import pytest
        pytest.skip(f"real checkpoint not present at {_CKPT_PATH} on this machine")

    from flash_rt.models.imagewam.checkpoint_loader import load_real_proprio_weights

    w, b = load_real_proprio_weights(_CKPT_PATH)
    assert tuple(w.shape) == (7680, 8), f"unexpected proprio_encoder.weight shape {tuple(w.shape)}"
    assert tuple(b.shape) == (7680,), f"unexpected proprio_encoder.bias shape {tuple(b.shape)}"
    print(f"PASS: real proprio_encoder loaded, weight={tuple(w.shape)} bias={tuple(b.shape)}")


if __name__ == "__main__":
    test_proprio_scatter_matches_real_insertion_rule()
    test_infer_raises_on_missing_proprio()
    if not _STATS_AVAILABLE:
        print(f"SKIPPED normalizer test: {_STATS_PATH} not present")
    else:
        test_min_max_normalizer_round_trips_and_matches_real_stats()
    if not _CKPT_AVAILABLE:
        print(f"SKIPPED real checkpoint test: {_CKPT_PATH} not present")
    else:
        test_real_proprio_encoder_loads_with_real_checkpoint()
    print("PASS")
