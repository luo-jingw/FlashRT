"""FLUX.2 AutoEncoder encode in NHWC with the FlashRT GroupNorm(+SiLU)
kernel (roadmap item 5 phase 4, `plan.md`).

`NativeFlux2Encoder(ae).encode(x)` computes the same function as
`flux2.autoencoder.AutoEncoder.encode(x)`, in the same op order:

    conv_in -> 4 levels x (2 ResnetBlocks [+ Downsample]) -> mid ResnetBlock
    -> mid AttnBlock -> mid ResnetBlock -> GroupNorm+swish -> conv_out
    -> quant_conv -> mean half -> 2x2 patch merge -> BatchNorm (eval)

with three changes of execution, not of math:

- every activation stays in `torch.channels_last` (NHWC) memory, and the
  convolution weights are channels_last copies, so cuDNN runs its NHWC
  kernels without the NCHW<->NHWC transposes it inserts around every
  convolution of the NCHW module;
- each `GroupNorm` (+ `swish`) is one `imagewam_groupnorm_nhwc_bf16`
  call (`csrc/kernels/imagewam_vae_groupnorm.cu`) instead of torch's
  GroupNorm, sigmoid and multiply kernels;
- inside each ResnetBlock the convolutions run without bias; conv1's
  bias is folded into norm2's reads, and conv2's bias, the
  nin_shortcut bias and the residual add are one
  `imagewam_bias_residual_nhwc_bf16` pass
  (`csrc/kernels/imagewam_vae_residual.cu`), each with torch's BF16
  rounding after every add;
- the attention block's q/k/v 1x1 convolutions are one GEMM over the
  NHWC rows (weights concatenated), and `proj_out` is a GEMM too.

Accumulation order differs from the NCHW module (cuDNN NHWC algorithms,
cuBLAS GEMMs, the GroupNorm reduction order), so tokens are near-exact,
not bit-identical; tests/test_imagewam_vae_stage.py records the gap.
The encoder weights are copied (about 70 MB BF16); `ae` is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

import flash_rt.flash_rt_kernels as fvk

_CL = torch.channels_last


@dataclass(frozen=True)
class _Conv:
    weight: torch.Tensor          # channels_last BF16
    bias: torch.Tensor
    stride: int
    padding: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def without_bias(self, x: torch.Tensor) -> torch.Tensor:
        """The convolution before torch's separate bias add; the caller
        applies `self.bias` in a fused kernel."""
        return F.conv2d(x, self.weight, None, stride=self.stride, padding=self.padding)


@dataclass(frozen=True)
class _GroupNorm:
    gamma: torch.Tensor
    beta: torch.Tensor
    groups: int
    eps: float

    def __call__(self, x: torch.Tensor, silu: bool, bias: torch.Tensor | None = None) -> torch.Tensor:
        """`x`: `(1, C, H, W)` BF16 in channels_last memory; normalizes
        `bf16(x + bias)` when `bias` is given."""
        if x.dtype != torch.bfloat16 or not x.is_contiguous(memory_format=_CL):
            raise ValueError(f"native GroupNorm expects channels_last BF16, got {x.dtype} strides={x.stride()}")
        n, c, h, w = x.shape
        y = torch.empty_like(x, memory_format=_CL)
        nbytes = fvk.imagewam_groupnorm_nhwc_workspace_bytes(n, h * w, c, self.groups)
        ws = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
        rc = fvk.imagewam_groupnorm_nhwc_bf16(
            x.data_ptr(), 0 if bias is None else bias.data_ptr(), self.gamma.data_ptr(), self.beta.data_ptr(),
            y.data_ptr(), ws.data_ptr(), nbytes,
            n, h * w, c, self.groups, self.eps, int(silu), torch.cuda.current_stream(x.device).cuda_stream)
        if rc != 0:
            raise RuntimeError(f"imagewam_groupnorm_nhwc_bf16 failed rc={rc} for {tuple(x.shape)}")
        return y


@dataclass(frozen=True)
class _ResnetBlock:
    norm1: _GroupNorm
    conv1: _Conv
    norm2: _GroupNorm
    conv2: _Conv
    shortcut: _Conv | None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1.without_bias(self.norm1(x, silu=True))
        h = self.conv2.without_bias(self.norm2(h, silu=True, bias=self.conv1.bias))
        res, res_bias = x, None
        if self.shortcut is not None:
            res, res_bias = self.shortcut.without_bias(x), self.shortcut.bias
        n, c, hh, ww = h.shape
        if not (h.is_contiguous(memory_format=_CL) and res.is_contiguous(memory_format=_CL)):
            raise ValueError("native ResnetBlock expects channels_last convolution outputs")
        rc = fvk.imagewam_bias_residual_nhwc_bf16(
            h.data_ptr(), self.conv2.bias.data_ptr(), res.data_ptr(),
            0 if res_bias is None else res_bias.data_ptr(), h.data_ptr(), n * hh * ww, c,
            torch.cuda.current_stream(h.device).cuda_stream)
        if rc != 0:
            raise RuntimeError(f"imagewam_bias_residual_nhwc_bf16 failed rc={rc} for {tuple(h.shape)}")
        return h


@dataclass(frozen=True)
class _Downsample:
    conv: _Conv                   # stride 2, padding 0, after a (0,1,0,1) zero pad

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


@dataclass(frozen=True)
class _AttnBlock:
    norm: _GroupNorm
    qkv_weight: torch.Tensor      # (3C, C): q, k, v 1x1 conv weights stacked
    qkv_bias: torch.Tensor        # (3C,)
    proj_weight: torch.Tensor     # (C, C)
    proj_bias: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        rows = self.norm(x, silu=False).permute(0, 2, 3, 1).reshape(n * h * w, c)
        qkv = F.linear(rows, self.qkv_weight, self.qkv_bias).view(n, 1, h * w, 3 * c)
        q, k, v = qkv.split(c, dim=-1)
        o = F.scaled_dot_product_attention(q, k, v)
        out = F.linear(o.reshape(n * h * w, c), self.proj_weight, self.proj_bias)
        return x + out.view(n, h, w, c).permute(0, 3, 1, 2)


def _conv(m: torch.nn.Conv2d) -> _Conv:
    return _Conv(weight=m.weight.detach().contiguous(memory_format=_CL).clone(memory_format=_CL),
                 bias=m.bias.detach().clone(), stride=int(m.stride[0]), padding=int(m.padding[0]))


def _norm(m: torch.nn.GroupNorm) -> _GroupNorm:
    return _GroupNorm(gamma=m.weight.detach().clone(), beta=m.bias.detach().clone(),
                      groups=int(m.num_groups), eps=float(m.eps))


def _resblock(m: torch.nn.Module) -> _ResnetBlock:
    shortcut = _conv(m.nin_shortcut) if m.in_channels != m.out_channels else None
    return _ResnetBlock(norm1=_norm(m.norm1), conv1=_conv(m.conv1), norm2=_norm(m.norm2),
                        conv2=_conv(m.conv2), shortcut=shortcut)


def _attn(m: torch.nn.Module) -> _AttnBlock:
    c = int(m.in_channels)
    qkv_w = torch.cat([m.q.weight, m.k.weight, m.v.weight], dim=0).detach().reshape(3 * c, c).contiguous()
    qkv_b = torch.cat([m.q.bias, m.k.bias, m.v.bias], dim=0).detach().contiguous()
    return _AttnBlock(norm=_norm(m.norm), qkv_weight=qkv_w, qkv_bias=qkv_b,
                      proj_weight=m.proj_out.weight.detach().reshape(c, c).contiguous(),
                      proj_bias=m.proj_out.bias.detach().clone())


class NativeFlux2Encoder:
    """NHWC FLUX.2 AutoEncoder encoder with fused GroupNorm(+SiLU);
    satisfies `vae_stage.VaeEncoder`."""

    def __init__(self, ae: torch.nn.Module) -> None:
        enc = ae.encoder
        if next(enc.parameters()).dtype != torch.bfloat16:
            raise ValueError("NativeFlux2Encoder expects the AE in BF16")
        if any(len(level.attn) > 0 for level in enc.down):
            raise ValueError("NativeFlux2Encoder supports the FLUX.2 encoder layout (no down-level attention)")
        self._conv_in = _conv(enc.conv_in)
        self._levels: list[tuple[list[_ResnetBlock], _Downsample | None]] = []
        for i_level, level in enumerate(enc.down):
            blocks = [_resblock(b) for b in level.block]
            down = _Downsample(_conv(level.downsample.conv)) if i_level != enc.num_resolutions - 1 else None
            self._levels.append((blocks, down))
        self._mid_block_1 = _resblock(enc.mid.block_1)
        self._mid_attn = _attn(enc.mid.attn_1)
        self._mid_block_2 = _resblock(enc.mid.block_2)
        self._norm_out = _norm(enc.norm_out)
        self._conv_out = _conv(enc.conv_out)
        self._quant_conv = _conv(enc.quant_conv)
        self._ps = (int(ae.ps[0]), int(ae.ps[1]))
        self._bn_mean = ae.bn.running_mean.detach().clone()
        self._bn_var = ae.bn.running_var.detach().clone()
        self._bn_eps = float(ae.bn.eps)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """`(1, 3, H, W)` BF16 -> `(1, 128, H/16, W/16)` BF16."""
        h = self._conv_in(x)
        for blocks, down in self._levels:
            for block in blocks:
                h = block(h)
            if down is not None:
                h = down(h)
        h = self._mid_block_1(h)
        h = self._mid_attn(h)
        h = self._mid_block_2(h)
        h = self._conv_out(self._norm_out(h, silu=True))
        moments = self._quant_conv(h)
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = rearrange(mean, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=self._ps[0], pj=self._ps[1])
        return F.batch_norm(z, self._bn_mean, self._bn_var, None, None, False, 0.0, self._bn_eps)
