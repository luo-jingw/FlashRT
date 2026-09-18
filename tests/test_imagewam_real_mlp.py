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


def _silu_glu_ref(gate_up: torch.Tensor) -> torch.Tensor:
    """Torch transcription of `silu_glu_merged_kernel`: fp32
    `g/(1+exp(-g))*u`, rounded once to fp16."""
    g, u = gate_up.float().chunk(2, dim=-1)
    return (g / (1.0 + torch.exp(-g)) * u).to(FP16)


def _silu_glu_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """The kernel on a packed `(seq, 2*mlp_hidden)` copy with a packed
    output -- the layout every pre-stride caller uses."""
    seq, width = gate_up.shape
    src = gate_up.contiguous()
    out = torch.zeros(seq, width // 2, dtype=FP16, device=DEV)
    fvk.silu_glu_merged_fp16(src.data_ptr(), out.data_ptr(), seq, width // 2, 0)
    return out


def _silu_glu_case(seq, mlp_hidden, in_width, in_col, out_width, out_col, seed):
    """Runs `silu_glu_merged_fp16` reading gate/up from columns
    `[in_col, in_col + 2*mlp_hidden)` of a `(seq, in_width)` buffer and
    writing `[out_col, out_col + mlp_hidden)` of a `(seq, out_width)`
    buffer. Returns (max-abs vs the torch reference, bit-exact vs the
    packed-layout kernel, untouched output columns all zero)."""
    torch.manual_seed(seed)
    src = torch.randn(seq, in_width, dtype=FP16, device=DEV)
    dst = torch.zeros(seq, out_width, dtype=FP16, device=DEV)
    fvk.silu_glu_merged_fp16(src.data_ptr() + in_col * 2, dst.data_ptr() + out_col * 2, seq, mlp_hidden, 0,
                             in_width, out_width)
    gate_up = src[:, in_col:in_col + 2 * mlp_hidden]
    packed = _silu_glu_packed(gate_up)
    torch.cuda.synchronize()
    got = dst[:, out_col:out_col + mlp_hidden]
    rest = torch.cat([dst[:, :out_col], dst[:, out_col + mlp_hidden:]], dim=1)
    max_abs = (got.float() - _silu_glu_ref(gate_up).float()).abs().max().item()
    return max_abs, torch.equal(got, packed), bool((rest == 0).all())


def test_silu_glu_merged_strides():
    """`silu_glu_merged_fp16` input/output row strides at the real
    single-stream shapes: the `linear1`-merged strided input, and the
    `linear2`-merged strided output (roadmap item 4) writing the MLP
    columns of a `(seq, attn_width + mlp_hidden)` buffer. Every strided
    case must equal the packed-layout call bit for bit (the stride only
    moves addresses) and leave the other output columns untouched."""
    hidden, mlp_hidden, a0 = 3072, 9216, 905
    l1_width, l2_width = 3 * hidden + 2 * mlp_hidden, hidden + mlp_hidden
    aaw, amh, num_action = 3072, 4096, 64
    cases = {
        "linear1 in / packed out": (a0, mlp_hidden, l1_width, 3 * hidden, mlp_hidden, 0),
        "linear1 in / linear2 out (backbone)": (a0, mlp_hidden, l1_width, 3 * hidden, l2_width, hidden),
        "linear1 in / linear2 out (action)": (num_action, amh, 3 * aaw + 2 * amh, 3 * aaw, aaw + amh, aaw),
    }
    for i, (name, case) in enumerate(cases.items()):
        max_abs, exact, rest_zero = _silu_glu_case(*case, seed=i)
        print(f"silu_glu_merged_fp16 [{name}]: bit_exact_vs_packed={exact} "
              f"max_abs_vs_torch={max_abs:.3e} other_columns_zero={rest_zero}")
        assert exact and rest_zero
        assert max_abs < 1e-2


if __name__ == "__main__":
    test_real_mlp_small_shape()
    test_real_mlp_real_dims()
    test_silu_glu_merged_strides()
    print("PASS")
