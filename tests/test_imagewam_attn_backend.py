"""ImageWAMAttnBackend wiring smoke test (plan.md Phase 2 correction).

Not a re-verification of attention math (tests/test_imagewam_mot_joint_kernel.py
already does that against a PyTorch reference). This only exercises the
dispatch plumbing added after discovering that ThorFlashAttnBackend can
never be constructed for ImageWAM's sites (see attn_backend.py's
ImageWAMAttnBackend docstring): pointer arithmetic, per-layer K/V
indexing, and both kernel branches ("standard" for "backbone",
"mot_joint" for "mot") reachable end-to-end through the same
AttentionBackend protocol every other FlashRT model pipeline uses.
"""
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import (
    ImageWAMAttnBackend,
    make_imagewam_attention_spec,
)


def test_backbone_and_mot_sites_run_without_nan():
    NH, HD = 24, 128
    x0, a0, total = 4, 8, 12  # 4 prefix + 4 target-image + 4 action tokens
    scale = 1.0 / (HD ** 0.5)
    device = "cuda"

    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=total)
    ctx = fvk.FvkContext()

    # One shared per-layer K/V buffer pair, sized for the full [prefix |
    # target-image | action] sequence -- "backbone" writes rows [0, a0)
    # once, "mot" reads the whole thing every denoise step (see the
    # class docstring's cache-ownership note).
    num_layers = spec.site("mot").num_layers
    K_all = torch.randn(num_layers, total, HD, dtype=torch.float16, device=device)
    V_all = torch.randn(num_layers, total, HD, dtype=torch.float16, device=device)
    layer_stride = K_all[0].numel() * 2  # bytes, fp16

    backbone_Q_O = torch.zeros(a0 * NH, HD, dtype=torch.float16, device=device)
    backbone_logits = torch.zeros(a0 * NH, a0, dtype=torch.float16, device=device)
    mot_Q_O = torch.zeros(total * NH, HD, dtype=torch.float16, device=device)
    total_pad = total + (total % 2)
    mot_logits = torch.zeros(total * NH, total_pad, dtype=torch.float16, device=device)

    backend = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": backbone_Q_O.data_ptr(), "K": K_all.data_ptr(),
            "V": V_all.data_ptr(), "logits": backbone_logits.data_ptr(),
            "scale": scale,
        },
        mot_slots={
            "Q_O": mot_Q_O.data_ptr(), "K": K_all.data_ptr(),
            "V": V_all.data_ptr(), "logits": mot_logits.data_ptr(),
            "scale": scale, "layer_stride": layer_stride,
        },
    )

    # Backbone: self-attention over [prefix|image] (rows [0,a0) of K/V).
    torch.randn(a0 * NH, HD, dtype=torch.float16, device=device, out=backbone_Q_O)
    out_ptr = backend.run("backbone", 0, q_seq=a0, stream=0)
    torch.cuda.synchronize()
    assert out_ptr == backbone_Q_O.data_ptr()
    assert torch.isfinite(backbone_Q_O).all()

    # Mot: joint attention, action queries ONLY (OPT-003 fix) -- q_seq is
    # num_action, kv_seq is the full combined sequence; output lands at
    # row-offset a0 within mot_Q_O, not at row 0.
    num_action = total - a0
    torch.randn(total * NH, HD, dtype=torch.float16, device=device, out=mot_Q_O)
    out_ptr = backend.run("mot", 0, q_seq=num_action, kv_seq=total, stream=0, x0=x0, a0=a0)
    torch.cuda.synchronize()
    assert out_ptr == mot_Q_O.data_ptr() + a0 * NH * HD * 2  # row width is NH*HD, not HD
    assert torch.isfinite(mot_Q_O).all()

    print("PASS: both sites dispatch and produce finite output")


def test_backbone_and_mot_sites_run_perhead_without_nan():
    """Same wiring smoke test, but with use_perhead_kv=True (OPT-002).

    K/V buffers are now (num_layers, total, NH, HD) -- real per-head,
    NH times wider per token than the broadcast test above -- since
    that's the format both new *_perhead kernels require (see
    ImageWAMAttnBackend's own use_perhead_kv docstring). Additionally
    cross-checks against the broadcast path: when the per-head K/V
    buffer happens to store the SAME vector repeated across every head
    (a broadcast K/V is a special case of per-head K/V), the backend's
    two dispatch paths must agree -- this is the backend-level version
    of test_imagewam_perhead_attention_kernel.py's own strongest check,
    confirming the wiring (pointer arithmetic, slot plumbing) preserves
    the kernel-level guarantee already proven there, not just that
    something runs without crashing.
    """
    NH, HD = 24, 128
    x0, a0, total = 4, 8, 12
    scale = 1.0 / (HD ** 0.5)
    device = "cuda"

    spec = make_imagewam_attention_spec(max_prefix_seq=a0, max_total_seq=total)
    ctx = fvk.FvkContext()
    num_layers = spec.site("mot").num_layers

    torch.manual_seed(0)
    # Real per-head K/V -- but every head stores an IDENTICAL copy, so
    # this is directly comparable to the broadcast path below.
    K_shared = torch.randn(num_layers, total, HD, dtype=torch.float16, device=device)
    V_shared = torch.randn(num_layers, total, HD, dtype=torch.float16, device=device)
    K_perhead = K_shared.unsqueeze(2).expand(-1, -1, NH, -1).contiguous()
    V_perhead = V_shared.unsqueeze(2).expand(-1, -1, NH, -1).contiguous()
    layer_stride = K_perhead[0].numel() * 2  # bytes, fp16

    backbone_Q_O = torch.zeros(a0 * NH, HD, dtype=torch.float16, device=device)
    backbone_logits = torch.zeros(a0 * NH, a0, dtype=torch.float16, device=device)
    mot_Q_O = torch.zeros(total * NH, HD, dtype=torch.float16, device=device)
    total_pad = total + (total % 2)
    mot_logits = torch.zeros(total * NH, total_pad, dtype=torch.float16, device=device)

    backend_perhead = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": backbone_Q_O.data_ptr(), "K": K_perhead.data_ptr(),
            "V": V_perhead.data_ptr(), "logits": backbone_logits.data_ptr(),
            "scale": scale,
        },
        mot_slots={
            "Q_O": mot_Q_O.data_ptr(), "K": K_perhead.data_ptr(),
            "V": V_perhead.data_ptr(), "logits": mot_logits.data_ptr(),
            "scale": scale, "layer_stride": layer_stride,
        },
        use_perhead_kv=True,
    )

    Q_seed = torch.randn(a0 * NH, HD, dtype=torch.float16, device=device)
    backbone_Q_O.copy_(Q_seed)
    out_ptr = backend_perhead.run("backbone", 0, q_seq=a0, stream=0)
    torch.cuda.synchronize()
    assert out_ptr == backbone_Q_O.data_ptr()
    assert torch.isfinite(backbone_Q_O).all()
    perhead_backbone_out = backbone_Q_O.clone()

    num_action = total - a0
    Q_seed_mot = torch.randn(total * NH, HD, dtype=torch.float16, device=device)
    mot_Q_O.copy_(Q_seed_mot)
    out_ptr = backend_perhead.run("mot", 0, q_seq=num_action, kv_seq=total, stream=0, x0=x0, a0=a0)
    torch.cuda.synchronize()
    assert out_ptr == mot_Q_O.data_ptr() + a0 * NH * HD * 2
    assert torch.isfinite(mot_Q_O).all()
    perhead_mot_out = mot_Q_O.clone()

    # Now the broadcast path, with the SAME Q seeds and the same
    # underlying K/V (as the (total,HD) shared vectors), and confirm
    # the two backends agree.
    backbone_Q_O.copy_(Q_seed)
    mot_Q_O.copy_(Q_seed_mot)
    backend_broadcast = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": backbone_Q_O.data_ptr(), "K": K_shared.data_ptr(),
            "V": V_shared.data_ptr(), "logits": backbone_logits.data_ptr(),
            "scale": scale,
        },
        mot_slots={
            "Q_O": mot_Q_O.data_ptr(), "K": K_shared.data_ptr(),
            "V": V_shared.data_ptr(), "logits": mot_logits.data_ptr(),
            "scale": scale, "layer_stride": K_shared[0].numel() * 2,
        },
    )
    backend_broadcast.run("backbone", 0, q_seq=a0, stream=0)
    torch.cuda.synchronize()
    backend_broadcast.run("mot", 0, q_seq=num_action, kv_seq=total, stream=0, x0=x0, a0=a0)
    torch.cuda.synchronize()

    def _cosine(a, b):
        a_ = a.float().flatten()
        b_ = b.float().flatten()
        return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()

    cos_backbone = _cosine(perhead_backbone_out, backbone_Q_O)
    cos_mot = _cosine(perhead_mot_out, mot_Q_O)
    print(f"backend-level perhead vs broadcast: backbone cosine={cos_backbone:.6f}, "
          f"mot cosine={cos_mot:.6f}")
    assert cos_backbone > 0.999
    assert cos_mot > 0.999

    print("PASS: both sites dispatch (use_perhead_kv=True) and match the broadcast path")


if __name__ == "__main__":
    test_backbone_and_mot_sites_run_without_nan()
    test_backbone_and_mot_sites_run_perhead_without_nan()
