#!/usr/bin/env python
"""OPT-007 follow-up: does zero-padding K to the next power of 2 unblock
the QuaRot FHT+int4 GEMM path at ImageWAM's real (non-power-of-2) K
dimensions?

Real math, not a guess: padding both activation and weight with zeros
before an orthogonal (Hadamard) rotation preserves their exact inner
product in the continuous limit -- <H@x_pad, H@w_pad> = <x_pad, w_pad>
= <x, w> for any orthogonal H, verified numerically below (max abs
error ~1e-4, floating-point noise). The real open question was whether
the ACTUAL kernel (fht_int4_quant_fp16, a fixed radix-16x3 butterfly
for K=4096 or a generic staged butterfly otherwise) plus int4's coarse
4-bit resolution keeps enough accuracy to be worth using, and whether
it even runs correctly at every padded size ImageWAM needs -- both
empirical questions this probe answers.

Result, this machine, random data (not real ImageWAM activations) —
**and a real run-to-run inconsistency worth stating plainly, not
hiding**: the FIRST exploratory run of this exact logic showed
K=3072->4096 and K=7680->8192 both WORKING (cosine 0.983 and 0.977)
with only K=9216->16384 failing (GEMM rc=196615). Formalizing that
exploration into this committed script and re-running it — with
IDENTICAL logic and call order — instead reproducibly (3 consecutive
runs, same result every time) shows:

  K=3072 -> pad 4096:  WORKS,  cosine=0.983 vs an fp32 reference
  K=7680 -> pad 8192:  FAILS,  GEMM rc=196615
  K=9216 -> pad 16384: FAILS,  fht_int4_quant_fp16 itself silently
                        produces an all-zero row scale (no CUDA error
                        thrown) before the GEMM is even attempted

Root cause not identified for either failure mode. The K=16384
all-zero-scale symptom is consistent with a shared-memory sizing issue
(the padded smem row at K=16384 needs ~68KB, above the default 48KB
static limit) but this was not confirmed by reading the kernel's own
launch configuration. The K=3072 result reproduced identically across
every run in both scripts; only K=7680/8192 flipped between the two
otherwise-identical runs, an unexplained instability in this same
kernel family this project has now observed three separate times
(here, and twice in opportunities.md OPT-007's own INT8/INT4 mlp2
findings) — treat any single "it works" result from this SM80 INT4/INT8
CUTLASS family as provisional, not confirmed, until it reproduces
across multiple independent runs.

Practical implication for ImageWAM, given ONLY K=3072 is confirmed
reliably working: q/k/v/proj and mlp0 (all K=3072 — mlp0's own K is
the hidden width it reads from, not the mlp_hidden width it writes to)
could plausibly use real QuaRot-rotated INT4 once someone implements
the weight-side offline rotation (this project has none — FlashRT's
own comment on fht_int4.cu says weight rotation is "done offline in
the frontend", i.e. by whatever loads a real Chameleon-7B checkpoint,
which does not exist for ImageWAM). txt_in (K=7680) and mlp2/mlp_down
(K=9216) cannot right now, and not only because of the FHT crash this
project already knew about (opportunities.md OPT-007) — the padding
workaround itself is not yet reliable at these larger sizes either.
"""
from __future__ import annotations

import torch

import flash_rt.flash_rt_kernels as fvk

DEV = "cuda"


def hadamard(n: int) -> torch.Tensor:
    """Sylvester construction -- matches fht_int4.cu's own H_K exactly
    (recursive [[1,1],[1,-1]] Kronecker product). Only defined for
    n a power of 2, which is exactly the constraint this probe is about."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def probe(k_real: int, k_pad: int, m: int, n: int):
    torch.manual_seed(0)
    x = torch.randn(m, k_real, dtype=torch.float32) * 2.0
    w = torch.randn(n, k_real, dtype=torch.float32) * 0.5
    ref = x @ w.T  # fp32, unrotated, unquantized reference

    x_pad = torch.zeros(m, k_pad)
    x_pad[:, :k_real] = x
    w_pad = torch.zeros(n, k_pad)
    w_pad[:, :k_real] = w

    H = hadamard(k_pad)
    x_rot = (x_pad @ H) / (k_pad ** 0.5)
    w_rot = (w_pad @ H) / (k_pad ** 0.5)
    rot_err = ((x_rot @ w_rot.T) - ref).abs().max().item()
    print(f"K={k_real}->pad{k_pad}: rotation preserves dot product, "
          f"max abs err={rot_err:.6f} (expect ~0)")

    x_pad_fp16 = x_pad.to(torch.float16).to(DEV).contiguous()
    packed_act = torch.zeros(m, k_pad // 2, dtype=torch.uint8, device=DEV)
    act_scale = torch.zeros(m, dtype=torch.float32, device=DEV)
    fvk.fht_int4_quant_fp16(x_pad_fp16.data_ptr(), packed_act.data_ptr(),
                             act_scale.data_ptr(), m, k_pad, 0)
    torch.cuda.synchronize()
    if (act_scale == 0).all():
        print(f"  fht_int4_quant_fp16 produced an all-zero scale at K={k_pad} "
              f"-- silent failure, not investigated further (see module docstring)")
        return

    # Weight-side rotation+quant done here in plain torch, matching
    # fht_int4.cu's own documented convention (row @ H_K / sqrt(K), then
    # per-row symmetric int4, qmax=7) -- FlashRT itself provides no
    # kernel for this (it's meant to happen offline, once, in whatever
    # loads a real quantized checkpoint).
    amax = w_rot.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    scale_u = amax / 7.0
    w_q = torch.round(w_rot / scale_u).clamp(-7, 7).to(torch.int8)
    even = (w_q[:, 0::2] & 0xF).to(torch.uint8)
    odd = (w_q[:, 1::2] & 0xF).to(torch.uint8)
    w_packed = (even | (odd << 4)).to(DEV).contiguous()
    w_scale = scale_u.squeeze(1).to(torch.float32).to(DEV).contiguous()

    out = torch.zeros(m, n, dtype=torch.float16, device=DEV)
    rc = fvk.cutlass_int4_rowwise_fp16out(
        packed_act.data_ptr(), w_packed.data_ptr(),
        act_scale.data_ptr(), w_scale.data_ptr(),
        out.data_ptr(), m, n, k_pad, 0)
    torch.cuda.synchronize()
    if rc != 0:
        print(f"  GEMM failed rc={rc}")
        return

    a_ = ref.to(DEV).half().float().flatten()
    b_ = out.float().flatten()
    cos = torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)
    rel_l2 = (b_ - a_).norm() / (a_.norm() + 1e-12)
    print(f"  int4 + Hadamard-padding GEMM vs fp32 reference: "
          f"cosine={cos.item():.6f} rel_l2={rel_l2.item():.6f}")


def main():
    probe(k_real=3072, k_pad=4096, m=64, n=64)
    probe(k_real=9216, k_pad=16384, m=32, n=32)
    probe(k_real=7680, k_pad=8192, m=32, n=32)


if __name__ == "__main__":
    main()
