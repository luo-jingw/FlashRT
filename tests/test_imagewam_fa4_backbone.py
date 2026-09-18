"""ImageWAM backbone self-attention via FA4 vs. the existing cuBLAS-
composed kernel (OPT-005, opportunities.md).

UNTESTED ON REAL HARDWARE by this project -- FA4 needs an active Thor
(sm_110) FA4 runtime (nvidia-cutlass-dsl + quack-kernels, the
"thor-fa4" pip extra), which this project's own dev machine does not
have (confirmed: `ImageWAMAttnBackend(..., use_fa4=True)` raises
cleanly here with "ModuleNotFoundError: No module named 'cutlass'").
Written by mirroring `ThorFlashAttnBackend`'s own already-verified
Pi0.5 "encoder"-site FA4 call pattern as closely as possible -- see
`ImageWAMAttnBackend.__init__`'s own `use_fa4` docstring for the exact
correspondence -- but this specific ImageWAM integration has never run.

This is NOT a re-verification of attention math from scratch: it
compares FA4's output against `attention_qkv_fp16`/`_perhead` (this
project's own already-PyTorch-reference-verified kernels, see
test_imagewam_mot_joint_kernel.py's sibling tests) on identical random
Q/K/V. If FA4 and the cuBLAS path agree closely, both are validated by
extension; if they disagree, at least one has a real bug worth finding
before using either in production.

**Two cases, both need Thor to actually run**: `test_fa4_matches_cublas_backbone_attention`
covers the OLD broadcast-K/V convention (`use_perhead_kv=False`,
`pack_gqa=True`); `test_fa4_matches_cublas_backbone_attention_perhead`
covers the NEW real per-head K/V convention that's now this class's
own default (`use_perhead_kv=True`, `pack_gqa=False`) -- added
2026-09-14 after auditing `ImageWAMAttnBackend.run()`'s FA4 branch and
finding it silently assumed the OLD broadcast shape even when real
per-head K/V was in use (fixed in the same commit; see that method's
own docstring for the full account). Both cases are written and ready,
but this dev machine can only confirm they SKIP cleanly, not that
either passes.

`test_fa4_matches_cublas_both_sites_real_shapes` (OPT-019) also covers
the "mot" site at its real shape. The frontend runs "mot" with the real
unmasked rule (`use_real_mot_mask=True`), which FA4 expresses as plain
non-causal attention; the legacy three-region mask has no FA4 path.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec

try:
    from flash_rt.hardware.thor import fa4_backend
    _FA4_AVAILABLE = fa4_backend.fa4_fwd() is not None
except ImportError:
    _FA4_AVAILABLE = False


def _run_backend(*, use_fa4: bool, q_seq: int, kv_seq: int, NH: int, HD: int,
                  hidden: int, num_layers: int, use_perhead_kv: bool = False):
    spec = make_imagewam_attention_spec(max_prefix_seq=q_seq, max_total_seq=q_seq + 1,
                                         num_layers=num_layers, num_heads=NH, head_dim=HD)
    ctx = fvk.FvkContext()
    device = "cuda"
    kv_width = NH * HD if use_perhead_kv else HD
    Q_O = torch.zeros(q_seq * NH, HD, dtype=torch.float16, device=device)
    K_all = torch.randn(num_layers, kv_seq, kv_width, dtype=torch.float16, device=device)
    V_all = torch.randn(num_layers, kv_seq, kv_width, dtype=torch.float16, device=device)
    logits = torch.zeros(q_seq * NH, max(kv_seq, HD), dtype=torch.float16, device=device)
    fa4_out = torch.zeros(q_seq * NH, HD, dtype=torch.float16, device=device)
    layer_stride = K_all[0].numel() * 2

    backend = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": Q_O.data_ptr(), "K": K_all.data_ptr(), "V": V_all.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
            "fa4_out": fa4_out.data_ptr(), "fa4_out_numel": fa4_out.numel(),
        },
        mot_slots={
            "Q_O": Q_O.data_ptr(), "K": K_all.data_ptr(), "V": V_all.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
            "layer_stride": layer_stride,
        },
        use_fa4=use_fa4, use_perhead_kv=use_perhead_kv,
    )

    Q_seed = torch.randn(q_seq * NH, HD, dtype=torch.float16, device=device)
    Q_O.copy_(Q_seed)
    backend.run("backbone", 0, q_seq=q_seq, kv_seq=kv_seq, stream=0)
    torch.cuda.synchronize()
    return Q_O.clone(), K_all, V_all


def test_fa4_matches_cublas_backbone_attention():
    if not _FA4_AVAILABLE:
        import pytest
        pytest.skip(f"FA4 runtime not available on this machine: "
                    f"{fa4_backend.status() if 'fa4_backend' in dir() else 'import failed'}")

    torch.manual_seed(0)
    NH, HD, hidden = 24, 128, 3072
    q_seq = kv_seq = 8  # even, required by the cuBLAS path's own softmax alignment
    num_layers = 1

    # Same seed, same K/V for both paths -- only the Q buffer differs
    # per call (each backend construction re-randomizes it inside
    # _run_backend, so pin the seed before each to get identical inputs).
    torch.manual_seed(1)
    out_cublas, K_ref, V_ref = _run_backend(
        use_fa4=False, q_seq=q_seq, kv_seq=kv_seq, NH=NH, HD=HD, hidden=hidden, num_layers=num_layers)

    torch.manual_seed(1)
    out_fa4, K_fa4, V_fa4 = _run_backend(
        use_fa4=True, q_seq=q_seq, kv_seq=kv_seq, NH=NH, HD=HD, hidden=hidden, num_layers=num_layers)

    assert torch.equal(K_ref, K_fa4) and torch.equal(V_ref, V_fa4), (
        "K/V inputs differ between the two runs -- comparison is not apples-to-apples")

    a = out_cublas.float().flatten()
    b = out_fa4.float().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    rel_l2 = (b - a).norm() / (a.norm() + 1e-12)
    print(f"FA4 vs cuBLAS backbone attention: cosine={cos.item():.6f} rel_l2={rel_l2.item():.6f}")
    assert cos.item() > 0.999, f"cosine too low: {cos.item()}"


def test_fa4_matches_cublas_backbone_attention_perhead():
    """Same comparison, but `use_perhead_kv=True` -- the real per-head
    K/V convention that's this class's own default since OPT-002.
    Exercises the `pack_gqa=False` branch this same commit fixed (see
    module docstring); the OLD-convention test above only exercises
    `pack_gqa=True` and would not have caught the bug this fixes."""
    if not _FA4_AVAILABLE:
        import pytest
        pytest.skip(f"FA4 runtime not available on this machine: "
                    f"{fa4_backend.status() if 'fa4_backend' in dir() else 'import failed'}")

    torch.manual_seed(0)
    NH, HD, hidden = 24, 128, 3072
    q_seq = kv_seq = 8
    num_layers = 1

    torch.manual_seed(1)
    out_cublas, K_ref, V_ref = _run_backend(
        use_fa4=False, use_perhead_kv=True, q_seq=q_seq, kv_seq=kv_seq,
        NH=NH, HD=HD, hidden=hidden, num_layers=num_layers)

    torch.manual_seed(1)
    out_fa4, K_fa4, V_fa4 = _run_backend(
        use_fa4=True, use_perhead_kv=True, q_seq=q_seq, kv_seq=kv_seq,
        NH=NH, HD=HD, hidden=hidden, num_layers=num_layers)

    assert torch.equal(K_ref, K_fa4) and torch.equal(V_ref, V_fa4), (
        "K/V inputs differ between the two runs -- comparison is not apples-to-apples")

    a = out_cublas.float().flatten()
    b = out_fa4.float().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    rel_l2 = (b - a).norm() / (a.norm() + 1e-12)
    print(f"FA4 vs cuBLAS backbone attention (perhead K/V): cosine={cos.item():.6f} rel_l2={rel_l2.item():.6f}")
    assert cos.item() > 0.999, f"cosine too low: {cos.item()}"


def _real_shape_backend(*, use_fa4: bool, use_fa4_mot: bool, total: int, a0: int, NH: int, HD: int):
    """Real per-head K/V + unmasked real "mot" rule, as the frontend builds it."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    hidden, num_layers = NH * HD, 2
    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=total,
                                         num_layers=num_layers, num_heads=NH, head_dim=HD)
    q_o = torch.randn(total, hidden, generator=gen, device="cuda").to(torch.float16)
    k = torch.randn(num_layers, total, hidden, generator=gen, device="cuda").to(torch.float16)
    v = torch.randn(num_layers, total, hidden, generator=gen, device="cuda").to(torch.float16)
    logits = torch.zeros(total * NH, total + total % 2, dtype=torch.float16, device="cuda")
    fa4_out = torch.zeros(total, hidden, dtype=torch.float16, device="cuda")
    slots = {"Q_O": q_o.data_ptr(), "K": k.data_ptr(), "V": v.data_ptr(),
             "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
             "fa4_out": fa4_out.data_ptr(), "fa4_out_numel": fa4_out.numel()}
    backend = ImageWAMAttnBackend(
        spec, fvk.FvkContext(), backbone_slots=dict(slots),
        mot_slots=dict(slots, layer_stride=k[0].numel() * 2),
        use_fa4=use_fa4, use_perhead_kv=True, use_real_mot_mask=True, use_fa4_mot=use_fa4_mot)
    return backend, (q_o, k, v, logits, fa4_out)


def _cos_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    return cos.item(), (b - a).abs().max().item(), ((b - a).norm() / (a.norm() + 1e-12)).item()


def test_fa4_matches_cublas_both_sites_real_shapes():
    """OPT-019: FA4 against the cuBLAS chain at the REAL shapes of both
    sites -- "backbone" q=kv=a0=905 (odd) and "mot" q=64 action rows at
    row offset a0 over kv=total=969 -- with real per-head K/V and the
    unmasked real "mot" rule the frontend uses."""
    if not _FA4_AVAILABLE:
        import pytest
        pytest.skip(f"FA4 runtime not available on this machine: {fa4_backend.status()}")
    NH, HD, a0, total, num_action = 24, 128, 905, 969, 64
    ref, ref_bufs = _real_shape_backend(use_fa4=False, use_fa4_mot=False, total=total, a0=a0, NH=NH, HD=HD)
    fa4, fa4_bufs = _real_shape_backend(use_fa4=True, use_fa4_mot=True, total=total, a0=a0, NH=NH, HD=HD)
    ref.run("backbone", 1, q_seq=a0, kv_seq=a0, stream=0)
    fa4.run("backbone", 1, q_seq=a0, kv_seq=a0, stream=0)
    torch.cuda.synchronize()
    cos, mx, rel = _cos_rel(ref_bufs[0][:a0], fa4_bufs[0][:a0])
    print(f"FA4 vs cuBLAS backbone q=kv={a0}: cosine={cos:.6f} max_abs={mx:.3e} rel_l2={rel:.6f}")
    assert cos > 0.999, f"backbone cosine too low: {cos}"
    before = fa4_bufs[0][:a0].clone()
    ref.run("mot", 1, q_seq=num_action, kv_seq=total, stream=0, x0=513, a0=a0)
    fa4.run("mot", 1, q_seq=num_action, kv_seq=total, stream=0, x0=513, a0=a0)
    torch.cuda.synchronize()
    rows = slice(a0, a0 + num_action)
    cos, mx, rel = _cos_rel(ref_bufs[0][rows], fa4_bufs[0][rows])
    print(f"FA4 vs cuBLAS mot q={num_action} kv={total}: cosine={cos:.6f} max_abs={mx:.3e} rel_l2={rel:.6f}")
    assert cos > 0.999, f"mot cosine too low: {cos}"
    assert torch.equal(fa4_bufs[0][:a0], before), "mot FA4 wrote outside the action rows"


if __name__ == "__main__":
    if not _FA4_AVAILABLE:
        print(f"SKIPPED: FA4 not available on this machine "
              f"({fa4_backend.status() if 'fa4_backend' in dir() else 'import failed'})")
    else:
        test_fa4_matches_cublas_backbone_attention()
        test_fa4_matches_cublas_backbone_attention_perhead()
        test_fa4_matches_cublas_both_sites_real_shapes()
        print("PASS")
