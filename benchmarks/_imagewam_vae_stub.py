"""Representative FLUX/SD-family latent-diffusion VAE encoder for the
ImageWAM full-pipeline speed benchmarks.

**Not the real FLUX.2 AE.** That source (`flux2.autoencoder.AutoEncoder`)
lives in `black-forest-labs/flux2` on GitHub, which ImageWAM's own
`docs/dependencies.md` documents as "user clones upstream, not
vendored" -- it is not present anywhere on this machine, and this
project has no network access or real checkpoint to obtain the real
architecture's exact channel counts or a real `ae.safetensors` from.
Consistent with every other component in this project's scope (random
weights, no real checkpoint, see PROJECT.md), this is a standard,
public-knowledge latent-diffusion VAE encoder shape (the same
conv/resnet/attention structure used across the SD/FLUX family of
autoencoders: `ch=128`, 2 ResnetBlocks per stage, one mid-block self-
attention, GroupNorm(32)+SiLU throughout). This gives a real,
non-trivial conv workload of the right computational order for a speed
measurement, not a claim of bit-exact architecture match.

**Token width/downsample factor, corrected from an earlier guess after
reading the real upstream source directly**: `imagewam.py`'s own
`_encode_flux2_image_tokens` asserts image H,W must be multiples of 16,
and `Flux2VideoExpert.pre_dit`'s own docstring states the packed image
tokens it consumes must be `[B,N,128]` -- confirmed real, not assumed
(`flux2_video_expert.py`, both lines quoted verbatim in this project's
git history). Also confirmed real: `Flux2VideoExpert.pack_latents` is
`rearrange(latents, "b c h w -> b (h w) c")` -- a PURE reshape, no 2x2
patch-merge step at all. Together these mean the real FLUX.2 VAE
downsamples 16x spatially AND emits 128 channels directly (not the
classic SD/FLUX.1 pattern of 8x downsample + 16 latent channels + a
separate 2x2-merge to 64-dim tokens, which an earlier version of this
module wrongly assumed by analogy). This encoder now downsamples 16x
in one conv stack (`ch_mult` has 5 stages, 4 downsamples between them)
and emits 128 channels directly via `conv_out` -- `pack_latents` below
is correspondingly a pure reshape now too, matching the real one
exactly instead of doing an extra (wrong) patch-merge. `VAE_IMG_H=384,
VAE_IMG_W=512` are chosen to produce exactly 768 tokens -- matching
`A0-X0` already used by every `imagewam_thor_*_bench.py` script for the
image-token span, so this module's output drops directly into the
existing `combined[x0:a0]` buffer via a single `img_in` projection,
same convention as the existing `txt_in` projection for text tokens.

Runs on plain PyTorch/cuDNN ops throughout (Conv2d, GroupNorm, SiLU,
`scaled_dot_product_attention`) -- no new FlashRT kernel needed, same
choice `flash_rt/models/cosmos3_edge/vae_native.py` already made for
its own (different, causal-3D) VAE: leave convolution on the official
cuDNN path, since that is already fast and correct, and only worth
replacing with a custom kernel if profiling shows it dominates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VAE_IMG_H, VAE_IMG_W = 384, 512
VAE_LATENT_CH = 128  # real FLUX.2 packed-token width, see module docstring
VAE_PATCH_TOKEN_DIM = VAE_LATENT_CH  # no patch-merge -- pack_latents is a pure reshape
VAE_NUM_TOKENS = (VAE_IMG_H // 16) * (VAE_IMG_W // 16)  # 24*32 = 768


class _ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _AttnBlock(nn.Module):
    """Single-head full self-attention over spatial positions (mid-block)."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).permute(1, 0, 3, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [b, h*w, c]
        out = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))
        out = out.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class Flux2VaeEncoderStub(nn.Module):
    def __init__(self, base_ch: int = 128, ch_mult=(1, 2, 4, 4, 4), num_res_blocks: int = 2,
                 latent_ch: int = VAE_LATENT_CH):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]
        self.conv_in = nn.Conv2d(3, chs[0], 3, padding=1)

        self.stages = nn.ModuleList()
        in_ch = chs[0]
        for i, out_ch in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(_ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if i < len(chs) - 1 else None
            self.stages.append(nn.ModuleDict({"blocks": blocks, "down": downsample} if downsample is not None
                                              else {"blocks": blocks}))

        mid_ch = chs[-1]
        self.mid_block1 = _ResnetBlock(mid_ch, mid_ch)
        self.mid_attn = _AttnBlock(mid_ch)
        self.mid_block2 = _ResnetBlock(mid_ch, mid_ch)

        self.norm_out = nn.GroupNorm(32, mid_ch)
        self.conv_out = nn.Conv2d(mid_ch, 2 * latent_ch, 3, padding=1)
        self.latent_ch = latent_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for stage in self.stages:
            for block in stage["blocks"]:
                h = block(h)
            if "down" in stage:
                h = stage["down"](h)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        mean, _logvar = h.chunk(2, dim=1)
        return mean  # deterministic encode -- fine for a speed-only benchmark


def build_vae_encoder(device: str = "cuda", dtype: torch.dtype = torch.float16,
                       compile: bool = False) -> Flux2VaeEncoderStub:
    """`compile=True` (opt-in, NOT the default) wraps the module in
    `torch.compile(mode="default")`.

    Real measured effect on this dev machine, in isolation (Ada sm_89,
    profiled with `torch.profiler` first -- convolution itself dominates
    at ~66% of GPU time, with the rest split across GroupNorm/SiLU/
    layout-conversion kernels cuDNN inserts converting between NCHW and
    its preferred NHWC layout for some conv algorithms; `channels_last` +
    `cudnn.benchmark` alone, tried first, made things WORSE, 55ms
    baseline -> 63ms, likely because the attention block's own reshape/
    permute ops silently force a layout conversion back): `torch.compile`
    fuses the elementwise GroupNorm/SiLU/residual-add chains via
    Inductor, taking the standalone `vae_encode` time from ~55-61ms down
    to ~48ms with `mode="default"` (~44ms with `mode="max-autotune"`,
    a bit faster still) -- confirmed by re-running the full INT4/FP16
    benchmark scripts end to end, not just this module alone.

    **Reverted to `compile=False` as the default despite the real gain
    above, because of demonstrated unreliability on this 8GB machine**:
    `mode="max-autotune"` first made the FP8 script's GPU sit at 100%
    util / ~7.9-7.92GB of 8GB total (near-OOM) for 45+ seconds with zero
    forward progress logged, requiring a hard kill. Switching to
    `mode="default"` fixed INT4/FP16 (both ran cleanly, confirming
    `mode="default"` itself is not inherently broken), but the SAME FP8
    script then stalled again under `mode="default"` too -- 65+ seconds
    stuck inside `FullImageWAMFP8.__init__` itself (before "Built." even
    printed, i.e. before any VAE forward call happens), GPU pinned at
    100% util, memory near the same ~7.9GB ceiling, again killed by
    hand. Not root-caused (no lingering compile-worker process found;
    INT4/FP16 use the identical `build_vae_encoder` call and did not
    reproduce it) -- plausibly some interaction between `torch.compile`'s
    lazy backend/cache initialization and this script's own FP8-specific
    weight-quantization loop in `__init__`, not investigated further.
    Given this project's standing memory-safety discipline on an
    already-tight 8GB machine, an unexplained, unpredictable path to a
    near-OOM multi-minute stall is not an acceptable default trade for a
    ~7-13ms (~13-20%) gain. Kept as an explicit opt-in for further
    investigation (e.g. on Thor, which has much more memory headroom and
    may not reproduce this at all) rather than silently dropped.

    The one-time JIT compile happens on first call, i.e. inside each
    bench script's own WARMUP loop when `compile=True` is passed
    explicitly, not inside the measured steady-state window -- same
    convention already used for cuDNN algorithm caching and CUDA context
    init elsewhere in this project.
    """
    vae = Flux2VaeEncoderStub().to(device=device, dtype=dtype).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    if compile:
        torch.backends.cudnn.benchmark = True
        vae = torch.compile(vae, mode="default")
    return vae


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, H*W, C], a PURE reshape -- matches the real
    `Flux2VideoExpert.pack_latents` exactly (`rearrange(latents,
    "b c h w -> b (h w) c")`, confirmed by reading flux2_video_expert.py
    directly: no patch-merge step at all, since this encoder already
    downsamples 16x and emits the real 128-channel token width directly
    -- see this module's own docstring for the correction history)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1).contiguous()  # [B, tokens, C]
