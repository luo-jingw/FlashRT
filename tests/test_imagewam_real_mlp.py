"""ImageWAM/FLUX.2 real MLP correctness (opportunities.md OPT-002
follow-up).

Verifies `flash_rt.models.imagewam.real_mlp.real_mlp_fp16` against an
independent transcription of the real `DoubleStreamBlock.img_mlp`/
`txt_mlp` + `SiLUActivation` formula (`black-forest-labs/flux2`'s
`src/flux2/model.py` at the pinned commit
`50fe5162777813d869182b139e83b10743caef15`, fetched and read directly).
"""
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.real_mlp import real_mlp_fp16

DEV = "cuda"
FP16 = torch.float16


def _ref_mlp(x, w_in, w_out):
    """w_in: (mlp_hidden*2, hidden) real nn.Linear weight (out,in).
    w_out: (hidden, mlp_hidden)."""
    h = F.linear(x.float(), w_in.float())
    g, u = h.chunk(2, dim=-1)
    gated = F.silu(g) * u
    return F.linear(gated.to(x.dtype).float(), w_out.float())


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def _run(seq, hidden, mlp_hidden, seed):
    torch.manual_seed(seed)
    x = torch.randn(seq, hidden, dtype=FP16, device=DEV)
    w_in = torch.randn(mlp_hidden * 2, hidden, dtype=FP16, device=DEV) * 0.02
    w_out = torch.randn(hidden, mlp_hidden, dtype=FP16, device=DEV) * 0.02

    ref = _ref_mlp(x, w_in, w_out)

    gemm = fvk.GemmRunner()
    # gemm.fp16_nn expects weight stored (K,N) = (in,out) -- transpose
    # from real nn.Linear's own (out,in) convention.
    w_in_kn = w_in.t().contiguous()
    w_out_kn = w_out.t().contiguous()
    merged = torch.zeros(seq, mlp_hidden * 2, dtype=FP16, device=DEV)
    gated = torch.zeros(seq, mlp_hidden, dtype=FP16, device=DEV)
    out = torch.zeros(seq, hidden, dtype=FP16, device=DEV)

    real_mlp_fp16(gemm, x.data_ptr(), w_in_kn.data_ptr(), w_out_kn.data_ptr(),
                  merged.data_ptr(), gated.data_ptr(), out.data_ptr(),
                  seq, hidden, mlp_hidden, 0)
    torch.cuda.synchronize()
    return ref, out


def test_real_mlp_small_shape():
    ref, out = _run(seq=8, hidden=32, mlp_hidden=48, seed=0)
    cos = _cosine(ref, out)
    print(f"real_mlp_fp16 (small): cosine={cos:.6f}")
    assert cos > 0.999


def test_real_mlp_real_dims():
    """Real ImageWAM dims: hidden=3072, mlp_hidden=9216 (mlp_ratio=3.0)."""
    ref, out = _run(seq=896, hidden=3072, mlp_hidden=9216, seed=1)
    cos = _cosine(ref, out)
    print(f"real_mlp_fp16 (real dims): cosine={cos:.6f}")
    assert cos > 0.999


if __name__ == "__main__":
    test_real_mlp_small_shape()
    test_real_mlp_real_dims()
    print("PASS")
