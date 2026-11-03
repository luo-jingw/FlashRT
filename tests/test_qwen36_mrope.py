"""text_rope_rows: shape and layout, equality with the 1-D table on equal axes, axis interleaving, input checks."""
import types

import pytest
import torch

from flash_rt.frontends.torch._qwen36_mrope import text_rope_rows

ROT, THETA, SEC = 64, 10_000_000.0, (11, 11, 10)


def _table_1d(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx
    ns = types.SimpleNamespace(_quant_format="nvfp4", device="cpu", max_seq=n,
                               _cfg={"head_dim": 256, "partial_rotary_factor": 0.25, "rope_theta": THETA})
    Qwen36TorchFrontendRtx._build_rope_table(ns)
    return ns._rope_cos_table, ns._rope_sin_table


def test_equal_axes_reproduce_the_1d_table():
    pos = torch.arange(300).view(1, -1).expand(3, -1)
    cos, sin = text_rope_rows(pos, rotary_dim=ROT, rope_theta=THETA, mrope_section=SEC)
    c1, s1 = _table_1d(300)
    assert cos.dtype == torch.bfloat16 and cos.shape == (300, ROT)
    assert torch.equal(cos, c1) and torch.equal(sin, s1)


def test_halves_are_duplicated_and_axes_interleave():
    pos = torch.stack([torch.full((4,), 5), torch.full((4,), 9), torch.full((4,), 13)])
    cos, _ = text_rope_rows(pos, rotary_dim=ROT, rope_theta=THETA, mrope_section=SEC)
    assert torch.equal(cos[:, :ROT // 2], cos[:, ROT // 2:])
    ref = {a: text_rope_rows(torch.full((3, 4), p), rotary_dim=ROT, rope_theta=THETA, mrope_section=SEC)[0]
           for a, p in (("t", 5), ("h", 9), ("w", 13))}
    for j in range(ROT // 2):
        axis = "h" if j % 3 == 1 and j < 3 * SEC[1] else "w" if j % 3 == 2 and j < 3 * SEC[2] else "t"
        assert torch.equal(cos[:, j], ref[axis][:, j]), j


@pytest.mark.parametrize("pos, section", [(torch.zeros(2, 5, dtype=torch.long), SEC),
                                          (torch.zeros(3, 5, dtype=torch.long), (11, 11, 11))])
def test_rejects_bad_inputs(pos, section):
    with pytest.raises(ValueError):
        text_rope_rows(pos, rotary_dim=ROT, rope_theta=THETA, mrope_section=section)
