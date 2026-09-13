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
compares FA4's output against `attention_qkv_fp16` (this project's own
already-PyTorch-reference-verified kernel, see
test_imagewam_mot_joint_kernel.py's sibling tests) on identical random
Q/K/V. If FA4 and the cuBLAS path agree closely, both are validated by
extension; if they disagree, at least one has a real bug worth finding
before using either in production.

Only covers the "backbone" site (plain self-attention, no mask) --
FA4's plain causal/non-causal API has no evaluated equivalent for the
"mot" site's three-region mask, not attempted here.
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
                  hidden: int, num_layers: int):
    spec = make_imagewam_attention_spec(max_prefix_seq=q_seq, max_total_seq=q_seq + 1)
    ctx = fvk.FvkContext()
    device = "cuda"
    Q_O = torch.zeros(q_seq * NH, HD, dtype=torch.float16, device=device)
    K_all = torch.randn(num_layers, kv_seq, HD, dtype=torch.float16, device=device)
    V_all = torch.randn(num_layers, kv_seq, HD, dtype=torch.float16, device=device)
    logits = torch.zeros(q_seq * NH, max(kv_seq, HD), dtype=torch.float16, device=device)
    layer_stride = K_all[0].numel() * 2

    backend = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": Q_O.data_ptr(), "K": K_all.data_ptr(), "V": V_all.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
        },
        mot_slots={
            "Q_O": Q_O.data_ptr(), "K": K_all.data_ptr(), "V": V_all.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
            "layer_stride": layer_stride,
        },
        use_fa4=use_fa4,
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


if __name__ == "__main__":
    if not _FA4_AVAILABLE:
        print(f"SKIPPED: FA4 not available on this machine "
              f"({fa4_backend.status() if 'fa4_backend' in dir() else 'import failed'})")
    else:
        test_fa4_matches_cublas_backbone_attention()
        print("PASS")
