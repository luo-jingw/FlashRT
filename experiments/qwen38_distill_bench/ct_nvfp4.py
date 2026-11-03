"""compressed-tensors ``nvfp4-pack-quantized`` weight encoding and decoding.

Layout read by ``flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py``:
  weight_packed        u8       (out, in/2)   two E2M1 codes per byte, element 2j in the low nibble
  weight_scale         f8_e4m3  (out, in/16)  one scale per 16-element block
  weight_global_scale  f32      (1,)          global; w = e2m1 * weight_scale / weight_global_scale

E2M1 code: bit 3 is the sign, bits 0-2 index the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
"""
from dataclasses import dataclass

import torch

E2M1_MAX = 6.0
E4M3_MAX = 448.0
BLOCK = 16
_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@dataclass
class CtNvfp4Weight:
    packed: torch.Tensor        # u8 (out, in/2)
    scale: torch.Tensor         # float8_e4m3fn (out, in/16)
    global_scale: torch.Tensor  # f32 (1,)


def global_scale_for(amax: float) -> float:
    """Global scale that maps the tensor's largest block scale onto the E4M3 maximum."""
    return E4M3_MAX * E2M1_MAX / amax if amax > 0.0 else 1.0


def encode(w: torch.Tensor, global_scale: float) -> CtNvfp4Weight:
    """Quantize a 2-D (out, in) weight on its device; ``in`` must be a multiple of 16."""
    out_f, in_f = w.shape
    if in_f % BLOCK != 0:
        raise ValueError(f"in_features {in_f} is not a multiple of {BLOCK}")
    blocks = w.float().view(out_f, in_f // BLOCK, BLOCK)
    block_amax = blocks.abs().amax(dim=-1)
    scale = (block_amax / E2M1_MAX * global_scale).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    step = scale.float() / global_scale
    scaled = torch.where(step.unsqueeze(-1) > 0, blocks / step.clamp(min=1e-30).unsqueeze(-1), torch.zeros_like(blocks))
    mids = torch.tensor(_MIDPOINTS, device=w.device, dtype=torch.float32)
    mag = torch.bucketize(scaled.abs().contiguous(), mids, right=False).to(torch.uint8)
    codes = (mag | ((scaled < 0) & (mag > 0)).to(torch.uint8) << 3).view(out_f, in_f)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return CtNvfp4Weight(packed=packed, scale=scale.contiguous(),
                         global_scale=torch.tensor([global_scale], dtype=torch.float32, device=w.device))


def decode(q: CtNvfp4Weight) -> torch.Tensor:
    """Dequantize to float32 (out, in)."""
    out_f, half = q.packed.shape
    codes = torch.stack([q.packed & 0x0F, q.packed >> 4], dim=-1).view(out_f, half * 2)
    levels = torch.tensor(_LEVELS, device=q.packed.device, dtype=torch.float32)
    mag = levels[(codes & 0x07).long()]
    val = torch.where((codes & 0x08) != 0, -mag, mag)
    step = q.scale.float() / q.global_scale.float()
    return (val.view(out_f, -1, BLOCK) * step.unsqueeze(-1)).view(out_f, half * 2)


def dequant_fp8_block128(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """FP8 e4m3 weight with a (ceil(out/128), ceil(in/128)) scale grid -> float32 (out, in)."""
    out_f, in_f = w.shape
    s = scale_inv.float().repeat_interleave(128, dim=0)[:out_f].repeat_interleave(128, dim=1)[:, :in_f]
    return w.float() * s
