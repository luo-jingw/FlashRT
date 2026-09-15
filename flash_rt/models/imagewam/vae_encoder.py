"""Real ImageWAM/FLUX.2 VAE encoding (real VAE + text-context wiring
plan, `plan.md`, Phase 1).

**Uses the REAL `flux2.autoencoder.AutoEncoder` class, NOT
`diffusers.AutoencoderKLFlux2`** (the class `model_index.json` names).
Confirmed by reading `imagewam.py`'s own real VAE construction directly
(`AutoEncoder(AutoEncoderParams())` + a strict `safetensors` load) that
this is what real inference actually uses. Confirmed the two classes
are NOT interchangeable by running both on this dev machine against a
real LIBERO-fastwam frame: `diffusers.AutoencoderKLFlux2.encode().latent_dist.mode()`
gives mean=-0.031/std=1.72/absmax=8.31 -- WRONG, that class defines an
identical `self.bn` `BatchNorm2d` submodule but never calls it inside
`encode()`, and never applies the real 2x2 patch-merge either. The
REAL `flux2.autoencoder.AutoEncoder.encode()` (`moments = encoder(x)`
-> take the mean half -> `rearrange` 2x2 patch-merge -> `self.bn` in
eval mode using its own TRAINED `running_mean`/`running_var`) gives
mean=-0.012/std=0.973/absmax=4.72 -- matching the real Thor
measurement (mean=-0.02, std=0.97, absmax=4.91) closely. Use the real
class; the diffusers one is not a safe substitute despite matching
architecture/weights format.

**`flux2` source availability, corrected**: this project's own docs
(`PROJECT.md`, earlier `plan.md` sections) assumed `black-forest-labs/flux2`
was not reachable from this dev machine. Found false while building
this module: `git clone https://github.com/black-forest-labs/flux2.git`
succeeds directly and pins to the exact commit
(`50fe5162777813d869182b139e83b10743caef15`) this project's own docs
have referenced by hash for weeks without a local checkout. `flux2_src`
below should point at such a clone (`FLUX2_SRC` env var, matching
`imagewam_real_checkpoint_validation.py`'s own established convention)
-- lazy-imported here, not a hard dependency of this module at import
time, matching `Nvfp4Linear`'s own guarded-import pattern for a
not-always-present package.
"""
from __future__ import annotations

import sys

import torch

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16


def load_real_ae(ae_model_path: str, flux2_src: str, *, device: str = DEV,
                  dtype: torch.dtype = torch.bfloat16):
    """Loads the real FLUX.2 AutoEncoder from `ae_model_path`
    (`ae.safetensors`), matching `imagewam.py`'s own
    `AutoEncoder(AutoEncoderParams())` + `load_sft` + strict
    `load_state_dict` exactly -- confirmed on this dev machine against
    the real Klein-4B `ae.safetensors`: `missing_keys=0 unexpected_keys=0`.

    `flux2_src` must point at a real `black-forest-labs/flux2` clone
    (`git clone https://github.com/black-forest-labs/flux2.git`,
    confirmed reachable from this dev machine 2026-09-15) -- lazy
    `sys.path.insert` here, not a module-level import, so importing
    THIS module never requires `flux2` to be present.
    """
    if flux2_src not in sys.path:
        sys.path.insert(0, flux2_src)
    try:
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams
    except ImportError as e:
        raise RuntimeError(
            f"load_real_ae requires a real flux2 clone at flux2_src={flux2_src!r} "
            f"(git clone https://github.com/black-forest-labs/flux2.git) -- "
            f"import failed: {e}") from e
    from safetensors.torch import load_file

    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())
    state = load_file(ae_model_path)
    missing, unexpected = ae.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"load_real_ae: {ae_model_path} does not match AutoEncoderParams()'s "
            f"own default architecture -- missing={missing[:5]} unexpected={unexpected[:5]} "
            f"(expected 0/0, confirmed against the real Klein-4B checkpoint)")
    return ae.to(device=device, dtype=dtype).eval()


def _prep_view(view: torch.Tensor, out_hw: tuple[int, int], device: str, dtype: torch.dtype) -> torch.Tensor:
    """One real camera view, `(H,W,3)` uint8, -> `(1,3,out_h,out_w)`
    normalized `x*2/255-1` (real ImageWAM/FLUX.2 preprocessing,
    confirmed against `libero_spatial_no_noops_lerobot`'s own real
    eval preprocessing: resize to 224x224 per view)."""
    import torch.nn.functional as F

    if view.dtype != torch.uint8 or view.ndim != 3 or view.shape[-1] != 3:
        raise ValueError(f"view must be (H,W,3) uint8, got shape={tuple(view.shape)} dtype={view.dtype}")
    x = view.to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    if tuple(x.shape[-2:]) != out_hw:
        x = F.interpolate(x, size=out_hw, mode="area")
    x = x * 2.0 / 255.0 - 1.0
    return x.to(dtype=dtype)


@torch.no_grad()
def encode_to_tokens(ae, view1: torch.Tensor, view2: torch.Tensor | None = None,
                      *, out_hw: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Real image -> real `(1, img_len, HD)` **BF16** CUDA tokens, ready
    to copy into `img_raw` directly. BF16 here isn't about THIS
    tensor's own range -- real VAE tokens are small (absmax~4.7-4.9,
    confirmed against real Thor measurement) -- it's so `img_raw`
    matches `img_in.weight`'s own `Bf16OutLinear` dtype requirement
    (`GemmRunner.bf16_nn` needs `A`/`B`/`D` all the same dtype; see
    opportunities.md OPT-001 "FP16 residual overflow", which is driven
    entirely by real Qwen3 TEXT conditioning, not the image tokens).

    `view1`/`view2`: `(H,W,3)` uint8 tensors, one or two real camera
    views. With two views, concatenated horizontally AFTER each is
    independently resized to `out_hw` (matching real
    `libero_spatial_no_noops_lerobot` eval preprocessing exactly: two
    224x224 views -> one 224x448 input -> real VAE -> 14x28 packed
    grid -> 392 tokens). With one view, encoded alone (whatever
    `img_len` that resolution produces -- caller's own responsibility
    to match `imagewam_thor.py`'s own configured `img_raw` shape).
    """
    device = next(ae.parameters()).device
    dtype = next(ae.parameters()).dtype
    x1 = _prep_view(view1, out_hw, str(device), dtype)
    if view2 is not None:
        x2 = _prep_view(view2, out_hw, str(device), dtype)
        x = torch.cat([x1, x2], dim=-1)  # (1,3,H,2W) -- horizontal concat, real convention
    else:
        x = x1

    z = ae.encode(x)  # (1, HD, latent_h, latent_w) -- real 2x2 patch-merge + BatchNorm baked in
    tokens = z.permute(0, 2, 3, 1).reshape(z.shape[0], -1, z.shape[1])  # (1, img_len, HD)
    return tokens.to(dtype=BF16)
