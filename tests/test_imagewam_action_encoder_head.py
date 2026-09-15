"""ImageWAM real `action_encoder`/`head` correctness (OPT-001, plan.md
"OPT-001" Phase 1's own second fix, alongside `img_in`).

Ported directly from `imagewam/models/backbones/action_dit_flux2.py`
(read at `/home/ljw/projects/pi0.5/tmp/ImageWAM/src/imagewam/models/backbones/action_dit_flux2.py`,
this project's own local read-only reference clone -- see `PROJECT.md`'s
"Onboarding" note): `ActionDiTFlux2.pre_dit`'s `tokens = self.action_encoder(action_tokens)`
(a real `nn.Linear(action_dim, hidden_dim)` WITH bias -- the only
biased weight anywhere in this project) and `Flux2ActionHead.forward`'s
`linear((1+scale)*norm_final(x)+shift)` (AdaLN, no gate, since it's a
FINAL output layer, not a residual block).

Verifies `pipeline_thor.py`'s own pointer-path calls
(`key("action_encoder.weight")` + `add_bias_fp16`; `ada_layer_norm_fp16`
+ `key("head.linear.weight")`) against a plain-PyTorch `F.linear`
reference built directly from the same real formulas -- not the full
`imagewam_denoise_step` (that's covered end-to-end, wiring-only, by
`test_imagewam_denoise.py`; this file isolates just the two new GEMM+
AdaLN primitives at real-dims scale, matching this project's own
narrow-kernel-test convention, e.g. `test_imagewam_qknorm_reuse.py`).
"""
import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.quant_linear import Fp16Linear

DEV = "cuda"
FP16 = torch.float16
F32 = torch.float32

_keepalive = []


def _own(t):
    _keepalive.append(t)
    return t


def _lin(n, k, scale=0.02):
    return _own((torch.randn(n, k, dtype=torch.float32, device=DEV) * scale).to(FP16).t().contiguous())


def _cosine(a, b):
    a_, b_ = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_, b_) / (a_.norm() * b_.norm() + 1e-12)).item()


def _check_action_encoder(num_action, action_dim, action_hidden_dim):
    torch.manual_seed(0)
    gemm = fvk.GemmRunner()
    w = _lin(action_hidden_dim, action_dim)  # (K=action_dim, N=action_hidden_dim), FlashRT convention
    bias = _own((torch.randn(action_hidden_dim, dtype=torch.float32, device=DEV) * 0.02).to(FP16))
    x = _own(torch.randn(num_action, action_dim, dtype=FP16, device=DEV) * 0.1)

    out = _own(torch.zeros(num_action, action_hidden_dim, dtype=FP16, device=DEV))
    Fp16Linear(gemm, w.data_ptr(), action_hidden_dim, action_dim)(x.data_ptr(), out.data_ptr(), num_action, 0)
    fvk.add_bias_fp16(out.data_ptr(), bias.data_ptr(), num_action, action_hidden_dim, 0)
    torch.cuda.synchronize()

    # Real nn.Linear(action_dim, action_hidden_dim): weight is (out,in)
    # = (action_hidden_dim, action_dim) -- w here is FlashRT's (K,N) =
    # (action_dim, action_hidden_dim), so w.t() recovers the real
    # (out,in) layout F.linear expects.
    ref = F.linear(x.float(), w.t().float(), bias.float())

    cos = _cosine(out, ref)
    print(f"action_encoder (num_action={num_action}): cosine={cos:.6f}")
    assert cos > 0.999, f"cosine too low: {cos}"


def _check_head(num_action, action_dim, action_hidden_dim):
    torch.manual_seed(1)
    gemm = fvk.GemmRunner()
    linear_w = _lin(action_dim, action_hidden_dim)  # (K=action_hidden_dim, N=action_dim)
    shift = _own((torch.randn(action_hidden_dim, dtype=torch.float32, device=DEV) * 0.1).to(FP16))
    scale = _own((torch.randn(action_hidden_dim, dtype=torch.float32, device=DEV) * 0.1).to(FP16))
    x = _own(torch.randn(num_action, action_hidden_dim, dtype=FP16, device=DEV) * 0.3)

    modded = _own(torch.zeros(num_action, action_hidden_dim, dtype=FP16, device=DEV))
    fvk.ada_layer_norm_fp16(x.data_ptr(), scale.data_ptr(), shift.data_ptr(),
                             modded.data_ptr(), num_action, action_hidden_dim, 1e-6, 0)
    out = _own(torch.zeros(num_action, action_dim, dtype=FP16, device=DEV))
    Fp16Linear(gemm, linear_w.data_ptr(), action_dim, action_hidden_dim)(
        modded.data_ptr(), out.data_ptr(), num_action, 0)
    torch.cuda.synchronize()

    # Real Flux2ActionHead.forward: linear((1+scale)*norm_final(x)+shift).
    # norm_final is elementwise_affine=False LayerNorm -- matches
    # ada_layer_norm_fp16's own internal normalization exactly.
    x_normed = F.layer_norm(x.float(), (action_hidden_dim,), eps=1e-6)
    ref_modded = (1 + scale.float()) * x_normed + shift.float()
    ref = F.linear(ref_modded, linear_w.t().float())

    cos = _cosine(out, ref)
    print(f"head (num_action={num_action}): cosine={cos:.6f}")
    assert cos > 0.999, f"cosine too low: {cos}"


def test_action_encoder_small():
    _check_action_encoder(num_action=4, action_dim=7, action_hidden_dim=96)


def test_action_encoder_real_dims():
    _check_action_encoder(num_action=64, action_dim=7, action_hidden_dim=1024)


def test_head_small():
    _check_head(num_action=4, action_dim=7, action_hidden_dim=96)


def test_head_real_dims():
    _check_head(num_action=64, action_dim=7, action_hidden_dim=1024)


if __name__ == "__main__":
    test_action_encoder_small()
    test_action_encoder_real_dims()
    test_head_small()
    test_head_real_dims()
    print("PASS")
