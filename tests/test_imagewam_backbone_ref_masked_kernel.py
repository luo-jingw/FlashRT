"""`attention_qkv_fp16_backbone_ref_masked_perhead` kernel correctness.

**Status update, found while investigating ActionDiT's real structure**:
this kernel's mask rule ("txt sees all, ref sees only itself") was
based on `flux2/model.py`'s own `causal_attn_fn`, which turned out to
be the WRONG source function for ImageWAM's real inference path.
ImageWAM's own `MoT._mixed_attention` (with a mask from `imagewam.py`'s
`_build_mot_attention_mask_flux2`) is what `infer_action_flux2` actually
uses, and that function's real call sites always pass `target_len=0`,
which reduces to NO masking between text and ref at all -- see
`opportunities.md` OPT-002 for the full correction. This kernel is
therefore **not used by `real_backbone_attn.py`/`real_double_stream_block.py`/
`real_single_stream_block.py` any more** (they use plain
`attention_qkv_fp16_perhead` instead). The kernel itself is still a
real, correct implementation of the 2-group rule it was built for (a
mask ImageWAM's own `flux2/model.py`-internal `causal_attn_fn` DOES use,
just not on the path this project's real deployment target exercises)
-- kept, with this test, as a validated building block for a
hypothetical future need (e.g. if `target_len>0` were ever relevant),
not deleted just because it's currently unused.
"""
import torch

import flash_rt.flash_rt_kernels as fvk

DEV = "cuda"
FP16 = torch.float16


def _ref_backbone_ref_masked(Q, K, V, scale, x0, total):
    """Independent PyTorch reference: txt rows [0,x0) see all columns
    [0,total); ref rows [x0,total) see only columns [x0,total)."""
    q = Q.permute(1, 0, 2).float()  # (NH, total, HD)
    k = K.permute(1, 0, 2).float()
    v = V.permute(1, 0, 2).float()
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # (NH, total, total)

    mask = torch.ones(total, total, dtype=torch.bool, device=Q.device)
    # ref rows (>= x0) may only see columns >= x0
    mask[x0:total, 0:x0] = False
    logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, v)
    return out.permute(1, 0, 2).contiguous()


def _run(total, NH, HD, x0, seed):
    torch.manual_seed(seed)
    scale = 1.0 / (HD ** 0.5)
    Q = torch.randn(total, NH, HD, dtype=FP16, device=DEV)
    K = torch.randn(total, NH, HD, dtype=FP16, device=DEV)
    V = torch.randn(total, NH, HD, dtype=FP16, device=DEV)

    total_pad = total + (total % 2)
    logits = torch.zeros(total * NH, total_pad, dtype=FP16, device=DEV)
    out = torch.zeros(total, NH, HD, dtype=FP16, device=DEV)
    ctx = fvk.FvkContext()
    fvk.attention_qkv_fp16_backbone_ref_masked_perhead(
        ctx, Q.data_ptr(), K.data_ptr(), V.data_ptr(),
        logits.data_ptr(), out.data_ptr(),
        total, NH, HD, x0, scale, 0)
    torch.cuda.synchronize()
    return Q, K, V, out, scale


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def test_backbone_ref_masked_small_shape():
    x0, total = 3, 8
    Q, K, V, out, scale = _run(total, NH=4, HD=16, x0=x0, seed=0)
    ref = _ref_backbone_ref_masked(Q, K, V, scale, x0, total)
    cos = _cosine(ref, out)
    print(f"backbone_ref_masked (small): cosine={cos:.6f}")
    assert cos > 0.999


def test_backbone_ref_masked_real_dims():
    """Real ImageWAM dims: x0=512 text tokens (Qwen3's own real
    max_length, confirmed via flux2.text_encoder.MAX_LENGTH and
    _imagewam_thor_spec.py's own declared context shape), 904 total
    (512 text + 392 real ref-image tokens, opportunities.md OPT-001)."""
    X0, A0 = 512, 904
    Q, K, V, out, scale = _run(A0, NH=24, HD=128, x0=X0, seed=1)
    ref = _ref_backbone_ref_masked(Q, K, V, scale, X0, A0)
    cos = _cosine(ref, out)
    print(f"backbone_ref_masked (real dims): cosine={cos:.6f}")
    assert cos > 0.999


def test_txt_rows_see_everything_ref_rows_see_only_self():
    """Direct behavioral check, independent of the reference-attention
    math above: a txt row's output must change if ref-side K/V change
    (it attends to ref), but a ref row's output must NOT change if
    txt-side K/V change (it never attends to txt) -- exactly the real
    asymmetric visibility rule, checked by perturbation rather than by
    recomputing the whole softmax."""
    x0, total, NH, HD = 3, 8, 4, 16
    scale = 1.0 / (HD ** 0.5)
    torch.manual_seed(2)
    Q = torch.randn(total, NH, HD, dtype=FP16, device=DEV)
    K = torch.randn(total, NH, HD, dtype=FP16, device=DEV)
    V = torch.randn(total, NH, HD, dtype=FP16, device=DEV)
    ctx = fvk.FvkContext()
    total_pad = total + (total % 2)

    def run(K_, V_):
        logits = torch.zeros(total * NH, total_pad, dtype=FP16, device=DEV)
        out = torch.zeros(total, NH, HD, dtype=FP16, device=DEV)
        fvk.attention_qkv_fp16_backbone_ref_masked_perhead(
            ctx, Q.data_ptr(), K_.data_ptr(), V_.data_ptr(),
            logits.data_ptr(), out.data_ptr(),
            total, NH, HD, x0, scale, 0)
        torch.cuda.synchronize()
        return out.clone()

    out_base = run(K, V)

    K_perturbed_txt = K.clone()
    V_perturbed_txt = V.clone()
    K_perturbed_txt[0:x0] += 5.0
    V_perturbed_txt[0:x0] += 5.0
    out_txt_perturbed = run(K_perturbed_txt, V_perturbed_txt)

    K_perturbed_ref = K.clone()
    V_perturbed_ref = V.clone()
    K_perturbed_ref[x0:total] += 5.0
    V_perturbed_ref[x0:total] += 5.0
    out_ref_perturbed = run(K_perturbed_ref, V_perturbed_ref)

    # ref rows [x0,total) must be UNCHANGED when only txt-side K/V moved.
    ref_rows_unaffected = torch.allclose(out_base[x0:total], out_txt_perturbed[x0:total], atol=1e-3)
    # txt rows [0,x0) must CHANGE when ref-side K/V moved (it attends to ref).
    txt_rows_affected = not torch.allclose(out_base[0:x0], out_ref_perturbed[0:x0], atol=1e-3)
    # ref rows must also change when their OWN K/V moved (self-attention).
    ref_rows_affected_by_own_kv = not torch.allclose(out_base[x0:total], out_ref_perturbed[x0:total], atol=1e-3)

    print(f"ref rows unaffected by txt perturbation: {ref_rows_unaffected}")
    print(f"txt rows affected by ref perturbation: {txt_rows_affected}")
    print(f"ref rows affected by own-KV perturbation: {ref_rows_affected_by_own_kv}")
    assert ref_rows_unaffected
    assert txt_rows_affected
    assert ref_rows_affected_by_own_kv


if __name__ == "__main__":
    test_backbone_ref_masked_small_shape()
    test_backbone_ref_masked_real_dims()
    test_txt_rows_see_everything_ref_rows_see_only_self()
    print("PASS")
