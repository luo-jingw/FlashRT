"""Qwen3.5-architecture text RoPE rows from interleaved-MRoPE position ids.

Produces rows in the layout of ``Qwen36TorchFrontendRtx._rope_cos_table`` / ``_rope_sin_table``: (S, rotary_dim)
BF16, rotate_half convention, the rotary_dim/2 frequencies duplicated (``cat([f, f])``). Frequency ``j`` takes the
temporal position, except ``j % 3 == 1`` for ``j < 3 * mrope_section[1]`` (height) and ``j % 3 == 2`` for
``j < 3 * mrope_section[2]`` (width), as in HF ``Qwen3_5TextRotaryEmbedding``. The arithmetic matches the frontend's
``_build_rope_table`` (float32 ``position * inv_freq``), so rows whose three axes are equal reproduce the 1-D table.
"""
import torch


def text_rope_rows(position_ids: torch.Tensor, *, rotary_dim: int, rope_theta: float,
                   mrope_section: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(3, S) long position ids (t, h, w) -> (cos, sin), each (S, rotary_dim) BF16 on the ids' device."""
    if position_ids.dim() != 2 or position_ids.shape[0] != 3:
        raise ValueError(f"position_ids must be (3, S), got {tuple(position_ids.shape)}")
    if sum(mrope_section) != rotary_dim // 2:
        raise ValueError(f"mrope_section {mrope_section} does not sum to rotary_dim / 2 = {rotary_dim // 2}")
    dev = position_ids.device
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, device=dev, dtype=torch.float32) / rotary_dim))
    freqs = position_ids.to(torch.float32)[:, :, None] * inv_freq[None, None, :]
    mixed = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        idx = slice(offset, mrope_section[axis] * 3, 3)
        mixed[:, idx] = freqs[axis][:, idx]
    emb = torch.cat([mixed, mixed], dim=-1)
    return emb.cos().to(torch.bfloat16).contiguous(), emb.sin().to(torch.bfloat16).contiguous()
