"""Dry-run test for `benchmarks/imagewam_real_checkpoint_validation.py`'s
weight-extraction functions, using FAKE modules that mirror the exact
real attribute paths (`block.img_attn.qkv.weight`,
`block.norm.query_norm.scale`, etc.) but with random weights instead of
a real checkpoint.

This does NOT validate against real weights (impossible without the
actual checkpoint file, which this project's dev machine doesn't have)
-- it validates that the extraction code itself (`extract_backbone_double_weights`,
`extract_backbone_single_weights`, `extract_action_double_weights`,
`extract_action_single_weights`) produces correctly-shaped weight dicts
that `real_double_stream_block_forward_fp16`/`real_single_stream_block_forward_fp16`/
`real_action_double_block_forward_fp16`/`real_action_single_block_forward_fp16`
accept and run without shape/attribute errors -- catching a wiring bug
in the extraction script BEFORE the user runs it against the real
checkpoint on Thor, where a shape mismatch would otherwise surface as a
confusing runtime error deep inside FlashRT.
"""
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, "benchmarks")
from imagewam_real_checkpoint_validation import (  # noqa: E402
    extract_action_double_weights,
    extract_action_single_weights,
    extract_backbone_double_weights,
    extract_backbone_single_weights,
)

import flash_rt.flash_rt_kernels as fvk  # noqa: E402
from flash_rt.models.imagewam.real_action_expert import (  # noqa: E402
    real_action_double_block_forward_fp16,
    real_action_single_block_forward_fp16,
)
from flash_rt.models.imagewam.real_double_stream_block import real_double_stream_block_forward_fp16  # noqa: E402
from flash_rt.models.imagewam.real_single_stream_block import real_single_stream_block_forward_fp16  # noqa: E402
from flash_rt.models.imagewam.rope import build_action_rope_table, build_backbone_rope_table  # noqa: E402

DEV = "cuda"
BF16 = torch.bfloat16
FP16 = torch.float16


def _fake_linear(out_f, in_f):
    lin = nn.Linear(in_f, out_f, bias=False).to(device=DEV, dtype=BF16)
    return lin


def _fake_qknorm(dim):
    ns = types.SimpleNamespace()
    ns.query_norm = types.SimpleNamespace(scale=torch.randn(dim, dtype=BF16, device=DEV))
    ns.key_norm = types.SimpleNamespace(scale=torch.randn(dim, dtype=BF16, device=DEV))
    return ns


def _make_fake_backbone_double_block(hidden, mlp_hidden, HD, NH):
    attn_dim = NH * HD
    assert attn_dim == hidden  # real backbone property
    block = types.SimpleNamespace()
    for side in ("txt", "img"):
        attn = types.SimpleNamespace(
            qkv=_fake_linear(3 * attn_dim, hidden),
            proj=_fake_linear(hidden, attn_dim),
            norm=_fake_qknorm(HD),
        )
        mlp = nn.Sequential(
            _fake_linear(mlp_hidden * 2, hidden), nn.SiLU(), _fake_linear(hidden, mlp_hidden)
        )
        setattr(block, f"{side}_attn", attn)
        setattr(block, f"{side}_mlp", mlp)
    return block


def _make_fake_backbone_single_block(hidden, mlp_hidden, HD, NH):
    attn_dim = NH * HD
    assert attn_dim == hidden
    block = types.SimpleNamespace()
    block.linear1 = _fake_linear(3 * attn_dim + mlp_hidden * 2, hidden)
    block.linear2 = _fake_linear(hidden, attn_dim + mlp_hidden)
    block.norm = _fake_qknorm(HD)
    return block


def _make_fake_action_double_block(hidden, attn_dim, mlp_hidden, HD):
    block = types.SimpleNamespace()
    block.img_attn = types.SimpleNamespace(
        qkv=_fake_linear(3 * attn_dim, hidden),
        proj=_fake_linear(hidden, attn_dim),
        norm=_fake_qknorm(HD),
    )
    block.img_mlp = nn.Sequential(
        _fake_linear(mlp_hidden * 2, hidden), nn.SiLU(), _fake_linear(hidden, mlp_hidden)
    )
    return block


def _make_fake_action_single_block(hidden, attn_dim, mlp_hidden, HD):
    block = types.SimpleNamespace()
    block.linear1 = _fake_linear(3 * attn_dim + mlp_hidden * 2, hidden)
    block.linear2 = _fake_linear(hidden, attn_dim + mlp_hidden)
    block.norm = _fake_qknorm(HD)
    return block


def test_backbone_double_extraction_and_forward_run_without_error():
    hidden, mlp_hidden, HD, NH = 512, 768, 128, 4
    x0, img_len = 3, 5
    total = x0 + img_len
    scale = 1.0 / (HD ** 0.5)

    block = _make_fake_backbone_double_block(hidden, mlp_hidden, HD, NH)
    w = extract_backbone_double_weights(block)
    expected_keys = {"txt_qkv", "txt_proj", "txt_mlp_in", "txt_mlp_out", "txt_query_norm", "txt_key_norm",
                      "img_qkv", "img_proj", "img_mlp_in", "img_mlp_out", "img_query_norm", "img_key_norm"}
    assert set(w.keys()) == expected_keys
    assert w["txt_qkv"].shape == (hidden, 3 * hidden)  # (K,N) GEMM convention
    assert w["txt_query_norm"].shape == (HD,)
    assert w["txt_qkv"].dtype == FP16

    txt = torch.randn(x0, hidden, dtype=FP16, device=DEV)
    img = torch.randn(img_len, hidden, dtype=FP16, device=DEV)

    def one_mod():
        return (torch.zeros(1, 1, hidden, dtype=FP16, device=DEV),
                torch.zeros(1, 1, hidden, dtype=FP16, device=DEV),
                torch.zeros(1, 1, hidden, dtype=FP16, device=DEV))
    mod_txt = (one_mod(), one_mod())
    mod_img = (one_mod(), one_mod())
    table = build_backbone_rope_table(x0, img_len, 1, device=DEV)

    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()
    txt_out, img_out = real_double_stream_block_forward_fp16(
        gemm, ctx, txt, img, w, mod_txt, mod_img, table, NH, HD, hidden, mlp_hidden, scale)
    assert torch.isfinite(txt_out).all()
    assert torch.isfinite(img_out).all()
    print("PASS: backbone double block extraction + forward run without error")


def test_backbone_single_extraction_and_forward_run_without_error():
    # extract_backbone_single_weights uses the real project's fixed
    # NH=24, HD=128 (attn_dim=3072) directly, not a parameter -- see
    # its own docstring. Must match here for the row/column slicing to
    # land on the right boundaries; only hidden/mlp_hidden are free to
    # shrink for a fast test.
    from imagewam_real_checkpoint_validation import HD, NH
    hidden, mlp_hidden = NH * HD, 96  # hidden == attn_dim, a real backbone property
    total = 8
    scale = 1.0 / (HD ** 0.5)

    block = _make_fake_backbone_single_block(hidden, mlp_hidden, HD, NH)
    w = extract_backbone_single_weights(block)
    expected_keys = {"qkv", "mlp_in", "attn_out", "mlp_out", "query_norm", "key_norm"}
    assert set(w.keys()) == expected_keys
    assert w["qkv"].shape == (hidden, 3 * NH * HD)
    assert w["attn_out"].shape == (NH * HD, hidden)

    x = torch.randn(total, hidden, dtype=FP16, device=DEV)
    mod = (torch.zeros(1, 1, hidden, dtype=FP16, device=DEV),) * 3
    table = build_backbone_rope_table(3, 5, 1, device=DEV)

    gemm = fvk.GemmRunner()
    ctx = fvk.FvkContext()
    out = real_single_stream_block_forward_fp16(gemm, ctx, x, w, mod, table, NH, HD, hidden, mlp_hidden, scale)
    assert torch.isfinite(out).all()
    print("PASS: backbone single block extraction + forward run without error")


def test_action_double_extraction_and_forward_run_without_error():
    # extract_action_double_weights takes NH/HD directly from its
    # caller's own args (it's a plain function param there, unlike the
    # *_single variants) -- no fixed-dim constraint here.
    hidden, attn_dim, mlp_hidden, HD, NH = 256, 512, 384, 128, 4
    num_action, backbone_total = 5, 8
    scale = 1.0 / (HD ** 0.5)

    block = _make_fake_action_double_block(hidden, attn_dim, mlp_hidden, HD)
    w = extract_action_double_weights(block)
    expected_keys = {"qkv", "proj", "mlp_in", "mlp_out", "query_norm", "key_norm"}
    assert set(w.keys()) == expected_keys
    assert w["qkv"].shape == (hidden, 3 * attn_dim)
    assert w["proj"].shape == (attn_dim, hidden)

    action = torch.randn(num_action, hidden, dtype=FP16, device=DEV)
    cached_k = torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV)
    cached_v = torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV)
    mod = ((torch.zeros(1, 1, hidden, dtype=FP16, device=DEV),) * 3,
           (torch.zeros(1, 1, hidden, dtype=FP16, device=DEV),) * 3)
    table = build_action_rope_table(num_action, device=DEV)

    gemm = fvk.GemmRunner()
    out = real_action_double_block_forward_fp16(
        gemm, action, w, mod, table, cached_k, cached_v, NH, HD, hidden, mlp_hidden, scale)
    assert torch.isfinite(out).all()
    print("PASS: action double block extraction + forward run without error")


def test_action_single_extraction_and_forward_run_without_error():
    # extract_action_single_weights uses the real project's fixed
    # NH=24, HD=128 directly for its row/column slicing, same
    # constraint as extract_backbone_single_weights -- attn_dim must
    # match here; hidden/mlp_hidden are free to shrink.
    from imagewam_real_checkpoint_validation import HD, NH
    attn_dim = NH * HD
    hidden, mlp_hidden = 256, 384
    num_action, backbone_total = 5, 8
    scale = 1.0 / (HD ** 0.5)

    block = _make_fake_action_single_block(hidden, attn_dim, mlp_hidden, HD)
    w = extract_action_single_weights(block)
    assert w["qkv"].shape == (hidden, 3 * attn_dim)
    assert w["attn_out"].shape == (attn_dim, hidden)

    action = torch.randn(num_action, hidden, dtype=FP16, device=DEV)
    cached_k = torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV)
    cached_v = torch.randn(backbone_total, NH, HD, dtype=FP16, device=DEV)
    mod = (torch.zeros(1, 1, hidden, dtype=FP16, device=DEV),) * 3
    table = build_action_rope_table(num_action, device=DEV)

    gemm = fvk.GemmRunner()
    out = real_action_single_block_forward_fp16(
        gemm, action, w, mod, table, cached_k, cached_v, NH, HD, hidden, mlp_hidden, scale)
    assert torch.isfinite(out).all()
    print("PASS: action single block extraction + forward run without error")


if __name__ == "__main__":
    test_backbone_double_extraction_and_forward_run_without_error()
    test_backbone_single_extraction_and_forward_run_without_error()
    test_action_double_extraction_and_forward_run_without_error()
    test_action_single_extraction_and_forward_run_without_error()
    print("PASS")
