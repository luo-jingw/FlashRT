"""Text context trimmed to the prompt's valid tokens (issues.md ISSUE-020).

Official ImageWAM (`imagewam.py` `_build_mot_attention_mask_flux2`)
masks the padded text keys for every query, at the backbone prefill and
at the action site: `mask[:, :, t0:r0] &= text_valid`. With proprio
packing (`_append_proprio_to_context`, `pack_proprio_after_text=True`)
the valid tokens are packed by rank, the proprio token sits right after
them at row `n_valid`, and the mask becomes the prefix `[0, n_valid]`.
Text RoPE positions are `0..L-1` over the whole context
(`Flux2VideoExpert.build_txt_ids`), so rows `[0, n_valid]` get positions
`0..n_valid` whatever the padded length `L` is, and image positions do
not depend on `L`. The padded rows are inert under that mask: no query
reads their keys, and nothing else reads their outputs. A sequence that
holds only rows `[0, n_valid]` therefore computes the official masked
math, with no mask.

`pack_trimmed_context` builds those rows; `trimmed_sequence_dims` gives
the sequence dims of a text length inside the frontend's max dims.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedTextContext:
    """The rows of a trimmed context.

    `rows`: `(n_valid + 1, D)` with a proprio slot (zero, filled per
    `infer()`), `(n_valid, D)` without; same dtype and device as the
    input context. `proprio_row`: `n_valid` with a proprio slot, else
    `None`.
    """
    rows: torch.Tensor
    n_valid: int
    proprio_row: int | None


def pack_trimmed_context(text_ctx: torch.Tensor, text_mask: torch.Tensor, *,
                         proprio_slot: bool) -> PackedTextContext:
    """Valid text rows in rank order, then the proprio slot.

    `text_ctx`: `(L, D)`; `text_mask`: `(L,)` bool, `True` for a valid
    token. With `proprio_slot`, the order is official
    `_append_proprio_to_context`'s for any mask. Without it, official
    keeps each token at its padded position, so the mask must be a
    prefix for the trimmed positions to match.
    """
    if text_ctx.ndim != 2 or text_mask.ndim != 1 or text_mask.shape[0] != text_ctx.shape[0]:
        raise ValueError(f"text context (L, D) and mask (L,) expected, got {tuple(text_ctx.shape)} "
                         f"and {tuple(text_mask.shape)}")
    mask = text_mask.to(device=text_ctx.device, dtype=torch.bool)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        raise ValueError("text context has no valid token (context_mask is all False)")
    if not proprio_slot and not bool(mask[:n_valid].all().item()):
        raise ValueError("without a proprio slot the context_mask must be a prefix (valid tokens first) "
                         "for the trimmed text positions to match the padded ones")
    valid = text_ctx[mask]
    if not proprio_slot:
        return PackedTextContext(rows=valid, n_valid=n_valid, proprio_row=None)
    rows = torch.zeros(n_valid + 1, text_ctx.shape[1], dtype=text_ctx.dtype, device=text_ctx.device)
    rows[:n_valid].copy_(valid)
    return PackedTextContext(rows=rows, n_valid=n_valid, proprio_row=n_valid)


def trimmed_sequence_dims(max_dims: dict, x0: int) -> dict:
    """Sequence dims for a context of `x0` rows inside `max_dims`.

    The image and action blocks keep their lengths, so `a0` and `total`
    shrink by the same `x0_max - x0` rows as `x0`. Returns
    `max_dims` itself (not a copy) when `x0` is the max, so the
    untrimmed path keeps its dims object; otherwise a copy with the three
    keys replaced.
    """
    x0_max = int(max_dims["x0"])
    if not 1 <= x0 <= x0_max:
        raise ValueError(f"x0={x0} outside [1, {x0_max}] (the buffers are sized for x0={x0_max})")
    if x0 == x0_max:
        return max_dims
    drop = x0_max - x0
    return dict(max_dims, x0=x0, a0=int(max_dims["a0"]) - drop, total=int(max_dims["total"]) - drop)
