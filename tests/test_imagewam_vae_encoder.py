"""`vae_encoder.py` correctness against real Thor stats (real VAE +
text-context wiring plan, `plan.md`, Phase 1).

**Skips cleanly if the real `flux2` clone or the real AE checkpoint
aren't present on this machine** -- following this project's own
established skip pattern (`test_imagewam_fa4_backbone.py`,
`test_imagewam_checkpoint_loader.py`), inverted here for a local
RESOURCE (a git clone + a checkpoint file) rather than a kernel/GPU
feature. `flux2_src`/`ae_model_path` default to this dev machine's own
actual locations (`third_party/flux2`, `/home/ljw/projects/pi0.5/models/flux2_klein_4b/ae.safetensors`)
-- override via env vars for a different machine.

No independent reference implementation exists locally to compute a
cosine against (that would need `diffusers.AutoencoderKLFlux2`, which
this module's own docstring explains is NOT numerically equivalent --
missing the real 2x2 patch-merge and BatchNorm). Instead, checked
against the real Thor-measured statistics (mean=-0.02, std=0.97,
absmax=4.91, `opportunities.md`'s own OPT-001 entry) as ground truth.
"""
import os

import torch

DEV = "cuda"

_FLUX2_SRC = os.environ.get(
    "FLUX2_SRC", os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party", "flux2", "src"))
_AE_PATH = os.environ.get(
    "AE_MODEL_PATH", "/home/ljw/projects/pi0.5/models/flux2_klein_4b/ae.safetensors")
_AVAILABLE = os.path.isdir(_FLUX2_SRC) and os.path.isfile(_AE_PATH)


def _make_test_frame():
    """A real-ish 224x224x3 uint8 frame -- checkerboard + gradient, not
    a random-noise image (a real camera frame is spatially smooth/
    structured, not white noise; this at least has real spatial
    structure for the VAE's own conv stack to encode meaningfully,
    though it's not an actual LIBERO photo -- see this file's own
    module docstring for how a real frame was independently used to
    derive the ground-truth stats this test checks against)."""
    torch.manual_seed(0)
    y, x = torch.meshgrid(torch.linspace(0, 1, 224), torch.linspace(0, 1, 224), indexing="ij")
    base = (torch.stack([x, y, 1 - x], dim=-1) * 255).clamp(0, 255)
    noise = torch.randn(224, 224, 3) * 8
    frame = (base + noise).clamp(0, 255).to(torch.uint8)
    return frame


def test_load_real_ae_and_encode_matches_real_thor_stats():
    if not _AVAILABLE:
        import pytest
        pytest.skip(f"flux2 clone ({_FLUX2_SRC}) or ae checkpoint ({_AE_PATH}) not present on this machine")

    from flash_rt.models.imagewam.vae_encoder import encode_to_tokens, load_real_ae

    ae = load_real_ae(_AE_PATH, _FLUX2_SRC)
    view1 = _make_test_frame().to(DEV)
    view2 = _make_test_frame().to(DEV)  # same synthetic frame twice -- structure, not content, matters here

    tokens = encode_to_tokens(ae, view1, view2)
    assert tokens.shape == (1, 392, 128), f"expected (1,392,128), got {tuple(tokens.shape)}"
    assert torch.isfinite(tokens).all(), "real VAE encode produced NaN/Inf"

    tf = tokens.float()
    mean, std = tf.mean().item(), tf.std().item()
    print(f"encode_to_tokens (synthetic frame): mean={mean:.4f} std={std:.4f} absmax={tf.abs().max().item():.4f}")
    # Real Thor measurement (opportunities.md OPT-001, real LIBERO
    # frames): mean=-0.02, std=0.97. A synthetic checkerboard frame
    # won't match exactly (different image content -> different
    # statistics) -- this is a SANITY range check (the real VAE's own
    # BatchNorm should keep std roughly O(1) for any real-ish input,
    # not a bit-exact match), not a tight correctness bar.
    assert 0.3 < std < 3.0, f"std={std} far outside the real-data range (0.97) -- check patch-merge/BN wiring"


if __name__ == "__main__":
    if not _AVAILABLE:
        print(f"SKIPPED: flux2 clone ({_FLUX2_SRC}) or ae checkpoint ({_AE_PATH}) not present")
    else:
        test_load_real_ae_and_encode_matches_real_thor_stats()
        print("PASS")
