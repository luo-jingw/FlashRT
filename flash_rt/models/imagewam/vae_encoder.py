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
from collections.abc import Sequence

import torch

from flash_rt.models.imagewam.vae_preprocess import VaePreprocessor
from flash_rt.models.imagewam.vae_stage import VaeEncoder

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
    normalized `x*2/255-1` (real ImageWAM/FLUX.2 preprocessing: resize to
    the per-view encode size `out_hw`, which is the served workload's
    `image_h x image_w`, or 224x224 for the LIBERO release)."""
    import torch.nn.functional as F

    if view.dtype != torch.uint8 or view.ndim != 3 or view.shape[-1] != 3:
        raise ValueError(f"view must be (H,W,3) uint8, got shape={tuple(view.shape)} dtype={view.dtype}")
    x = view.to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    if tuple(x.shape[-2:]) != out_hw:
        x = F.interpolate(x, size=out_hw, mode="area")
    x = x * 2.0 / 255.0 - 1.0
    return x.to(dtype=dtype)


@torch.no_grad()
def encode_to_tokens(ae, views: Sequence[torch.Tensor],
                      *, out_hw: tuple[int, int] = (224, 224),
                      preprocessor: VaePreprocessor | None = None,
                      encoder: VaeEncoder | None = None) -> torch.Tensor:
    """Real image -> real `(1, img_len, HD)` **BF16** CUDA tokens, ready
    to copy into `img_raw` directly. BF16 here isn't about THIS
    tensor's own range -- real VAE tokens are small (absmax~4.7-4.9,
    confirmed against real Thor measurement) -- it's so `img_raw`
    matches `img_in.weight`'s own `Bf16OutLinear` dtype requirement
    (`GemmRunner.bf16_nn` needs `A`/`B`/`D` all the same dtype; see
    opportunities.md OPT-001 "FP16 residual overflow", which is driven
    entirely by real Qwen3 TEXT conditioning, not the image tokens).

    `views`: one or more `(H,W,3)` uint8 real camera views, in the served
    order (view1 first). Each is resized to `out_hw` on its own -- the
    per-view size the real ImageWAM/FLUX.2 preprocessing uses, not the
    concatenated one -- then the resized views are concatenated
    horizontally into one `(1,3,out_h,out_w*len(views))` image and encoded
    by ONE `encode` call. The view count is the workload's
    (`ImageWAMWorkload.num_views`): two views reproduce real
    `libero_spatial_no_noops_lerobot` eval preprocessing exactly (two
    224x224 views -> one 224x448 input -> real VAE -> 14x28 packed grid ->
    392 tokens), and the resulting `img_len` is whatever that resolution
    produces -- caller's own responsibility to match
    `imagewam_thor.py`'s own configured `img_raw` shape.

    `out_hw`: the per-view size every view is encoded at. The default
    `(224, 224)` is LIBERO's own per-view size (`ImageWAMWorkload.libero()`
    serves two 224x224 cameras). A workload-driven caller passes the
    workload's `image_h`/`image_w` -- the size `ImageWAMWorkload.layout`
    divides by the patch stride, so the tokens are that workload's own
    `img_len` -- which is what `ImageWAMTorchFrontendThor.stage_images`
    does, from `_input_view_shape()`. A view already of size `out_hw` is
    not resized.

    `views` is validated before `ae` is touched: at least one view, every
    view `(H,W,3)` uint8 (`ValueError` naming the offending view).

    `preprocessor`: when given, the views are preprocessed by the fused
    `imagewam_vae_preprocess_bf16` kernel (`vae_preprocess.py`) instead
    of `_prep_view` + `torch.cat`: only uint8 bytes cross to the GPU and
    one launch per view writes the concatenated BF16 image. With
    `resize="area"` the result is bit-identical to the `_prep_view`
    path (tests/test_imagewam_vae_preprocess.py); `_prep_view` remains
    the reference implementation.

    `encoder`: the object whose `encode` runs (default `ae` itself), e.g.
    `vae_native_encoder.NativeFlux2Encoder(ae)`; `ae` still sets the
    device and dtype.
    """
    if not isinstance(views, Sequence):
        raise ValueError(f"views must be a sequence of (H,W,3) uint8 camera views, "
                         f"got {type(views).__name__}")
    if len(views) == 0:
        raise ValueError("encode_to_tokens needs at least one view, got an empty sequence")
    for i, view in enumerate(views):
        if view.dtype != torch.uint8 or view.ndim != 3 or view.shape[-1] != 3:
            raise ValueError(f"view {i} must be (H,W,3) uint8, got shape={tuple(view.shape)} "
                             f"dtype={view.dtype}")
    device = next(ae.parameters()).device
    dtype = next(ae.parameters()).dtype
    if preprocessor is not None:
        if tuple(preprocessor.out_hw) != tuple(out_hw):
            raise ValueError(f"preprocessor.out_hw={preprocessor.out_hw} != out_hw={out_hw}")
        if dtype != torch.bfloat16:
            raise ValueError(f"the preprocessing kernel writes BF16; the AE is {dtype}")
        gpu_views = [v.to(device=device).contiguous() for v in views]
        x = torch.empty(1, 3, out_hw[0], out_hw[1] * len(gpu_views), dtype=dtype, device=device)
        preprocessor.run(gpu_views, x, torch.cuda.current_stream(device).cuda_stream)
    else:
        # (1,3,out_h,out_w*N) -- horizontal concat, real convention.
        x = torch.cat([_prep_view(v, out_hw, str(device), dtype) for v in views], dim=-1)

    z = (ae if encoder is None else encoder).encode(x)  # (1, HD, latent_h, latent_w), 2x2 patch-merge + BatchNorm
    tokens = z.permute(0, 2, 3, 1).reshape(z.shape[0], -1, z.shape[1])  # (1, img_len, HD)
    return tokens.to(dtype=BF16)
