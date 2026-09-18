#!/usr/bin/env python
"""OPT-005: FA4 vs. the existing cuBLAS-composed kernel for ImageWAM's
"backbone" self-attention (no mask, single-shared-KV-head), at real
dims. Thor-only -- see tests/test_imagewam_fa4_backbone.py's own
docstring for why this project cannot run or verify this locally.

Run tests/test_imagewam_fa4_backbone.py FIRST and confirm it passes
before trusting any speed number here -- a fast-but-wrong kernel is
not a win.
"""
from __future__ import annotations

import statistics

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec

try:
    from flash_rt.hardware.thor import fa4_backend
    _FA4_AVAILABLE = fa4_backend.fa4_fwd() is not None
except ImportError:
    _FA4_AVAILABLE = False

WARMUP, ITERS = 20, 100
_KEEP: list = []
NH, HD, A0 = 24, 128, 896  # real ImageWAM backbone dims, see plan.md


def _build(use_fa4: bool):
    spec = make_imagewam_attention_spec(max_prefix_seq=A0, max_total_seq=A0 + 64)
    spec.sites["backbone"].num_layers = 1
    ctx = fvk.FvkContext()
    device = "cuda"
    Q_O = torch.randn(A0 * NH, HD, dtype=torch.float16, device=device)
    K = torch.randn(1, A0, HD, dtype=torch.float16, device=device)
    V = torch.randn(1, A0, HD, dtype=torch.float16, device=device)
    logits = torch.zeros(A0 * NH, A0, dtype=torch.float16, device=device)
    fa4_out = torch.zeros(A0 * NH, HD, dtype=torch.float16, device=device)
    # The backend holds raw pointers only; keep every buffer it reads alive.
    _KEEP.extend((K, V, logits, fa4_out))
    backend = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": Q_O.data_ptr(), "K": K.data_ptr(), "V": V.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
            "fa4_out": fa4_out.data_ptr(), "fa4_out_numel": fa4_out.numel(),
        },
        mot_slots={
            "Q_O": Q_O.data_ptr(), "K": K.data_ptr(), "V": V.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
            "layer_stride": K[0].numel() * 2,
        },
        use_fa4=use_fa4,
    )
    return backend, Q_O


def _time_ms(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2], statistics.mean(times)


def main():
    print(f"Real ImageWAM backbone self-attention shape: NH={NH} HD={HD} a0={A0}\n")

    backend_cublas, _ = _build(use_fa4=False)
    p50_c, mean_c = _time_ms(lambda: backend_cublas.run("backbone", 0, q_seq=A0, kv_seq=A0, stream=0))
    print(f"cuBLAS-composed (attention_qkv_fp16): P50={p50_c:.4f} ms  mean={mean_c:.4f} ms")

    if not _FA4_AVAILABLE:
        print(f"\nFA4 not available on this machine "
              f"({fa4_backend.status() if 'fa4_backend' in dir() else 'import failed'}) "
              f"-- skipping the FA4 side of this comparison.")
        return

    backend_fa4, _ = _build(use_fa4=True)
    p50_f, mean_f = _time_ms(lambda: backend_fa4.run("backbone", 0, q_seq=A0, kv_seq=A0, stream=0))
    print(f"FA4:                                   P50={p50_f:.4f} ms  mean={mean_f:.4f} ms")
    print(f"\nSpeedup: {p50_c / p50_f:.2f}x" if p50_f > 0 else "")


if __name__ == "__main__":
    main()
