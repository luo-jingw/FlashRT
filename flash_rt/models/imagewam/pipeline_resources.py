"""Interface between a captured ImageWAM frontend and the native pipeline.

`ImageWAMPipelineResources` describes everything the C++ pipeline
(cpp/models/imagewam/src/native_pipeline.cpp) needs to record the same
prefill and denoise loop as `pipeline_thor.py`: dimensions, borrowed
buffer and attention pointers, every weight as a `LinearResource`, the
AdaLN modulation precomputed into the fp16 form the kernels consume, and
the per-step Euler sizes. The frontend builds it
(`ImageWAMTorchFrontendThor.pipeline_resources`) and keeps owning every
buffer; the tensors materialized here (fp16 modulation) are owned by the
returned object and must outlive the native handle.

One table describes one context length: the sequence dims (`x0`, `a0`,
`total`) and the backbone RoPE table are the frontend's active length's,
while the buffers are the ones it allocated for the longest declared
length and every length's table points at them. A deployment that serves
several text lengths therefore hands over one table per length
(`ImageWAMTextLengthPipelineSource`), each installed and captured as its
own native pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from flash_rt.models.imagewam.quant_linear import (
    Bf16OutLinear,
    Fp16Linear,
    Nvfp4Linear,
    _nvfp4_variant_index,
)

LINEAR_FP16 = 0
LINEAR_BF16 = 1
LINEAR_NVFP4 = 2


@dataclass(frozen=True)
class LinearResource:
    """One weight projection: `out (m, n) = x (m, k) @ W`."""

    kind: int                # LINEAR_FP16 / LINEAR_BF16 / LINEAR_NVFP4
    n: int
    k: int
    weight: int              # FP16/BF16: (k, n) device pointer; NVFP4: packed (n, k/2)
    weight_scales: int = 0   # NVFP4: SFB
    act_packed: int = 0      # NVFP4: activation scratch (m, k/2)
    act_scales: int = 0      # NVFP4: activation SFA
    fp4_variant: int = -1    # NVFP4: CUTLASS variant index


def linear_resource(op: object) -> LinearResource:
    """The `LinearResource` of one frontend weight op (quant_linear.py)."""
    if isinstance(op, Fp16Linear):
        return LinearResource(LINEAR_FP16, op.n, op.k, op.weight_ptr)
    if isinstance(op, Bf16OutLinear):
        return LinearResource(LINEAR_BF16, op.n, op.k, op.weight_ptr)
    if isinstance(op, Nvfp4Linear):
        if op.scratch is None:
            raise RuntimeError("Nvfp4Linear activation scratch is allocated on first call; capture first")
        # The CUTLASS tile the op launches now (its (N, K) default, or the
        # one gemm_variant_autotune measured), as the captured graph does.
        return LinearResource(LINEAR_NVFP4, op.n, op.k, op.w_quant["packed"].data_ptr(),
                              op.w_quant["sfb"].data_ptr(), op.scratch.packed.data_ptr(),
                              op.scratch.sfa.data_ptr(), _nvfp4_variant_index(op.variant))
    raise ValueError(f"the native pipeline does not support {type(op).__name__} weights")


@dataclass(frozen=True)
class AdaLNResource:
    """One AdaLN site. `shift`/`scale`: fp16 `(dim,)` and `gate`: fp16
    materialized to `(rows, dim)`, exactly as `fp16_adaln_operands` builds them
    (unfused path, and the standalone AdaLN that starts each chain).
    `*_f32`: the `(1, 1, dim)` FP32 modulation chunks the fused gated
    residual + next AdaLN kernel reads (`dims["fuse_res_norm"]`). Gates are
    `None` for the gate-less head."""

    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor | None
    shift_f32: torch.Tensor
    scale_f32: torch.Tensor
    gate_f32: torch.Tensor | None


@dataclass(frozen=True)
class DoubleLayerResource:
    txt_qkv: LinearResource
    img_qkv: LinearResource
    txt_proj: LinearResource
    img_proj: LinearResource
    txt_mlp0: LinearResource
    img_mlp0: LinearResource
    txt_mlp2: LinearResource
    img_mlp2: LinearResource
    txt_query_norm: int
    txt_key_norm: int
    img_query_norm: int
    img_key_norm: int


@dataclass(frozen=True)
class SingleLayerResource:
    """With `dims["merge_linear2"]` only `linear2` is set, otherwise only
    `attn_out_proj` and `mlp_down`."""

    linear1: LinearResource
    attn_out_proj: LinearResource | None
    mlp_down: LinearResource | None
    linear2: LinearResource | None
    query_norm: int
    key_norm: int


@dataclass(frozen=True)
class ActionDoubleLayerResource:
    qkv: LinearResource
    proj: LinearResource
    mlp0: LinearResource
    mlp2: LinearResource
    query_norm: int
    key_norm: int


@dataclass(frozen=True)
class ActionStepResource:
    double1: AdaLNResource
    double2: AdaLNResource
    single: AdaLNResource
    head: AdaLNResource
    delta: float


@dataclass(frozen=True)
class PipelineDims:
    hidden: int
    head_dim: int
    num_heads: int
    mlp_hidden: int
    joint_attention_dim: int
    x0: int
    a0: int
    total: int
    num_action: int
    action_dim: int
    action_hidden_dim: int
    action_attn_width: int
    action_mlp_hidden: int
    num_double: int
    num_single: int
    action_num_double: int
    action_num_single: int
    num_steps: int
    merge_linear2: bool
    fuse_res_norm: bool
    eps: float


@dataclass(frozen=True)
class PipelineBuffers:
    """pipeline_thor.py `bufs` device pointers (bf16: context,
    backbone_hidden, img_raw; f32: action_latent; the rest fp16)."""

    context: int
    backbone_hidden: int
    img_raw: int
    modded_scratch: int
    txt_qkv_merged: int
    img_qkv_merged: int
    single_linear1_merged: int
    action_linear1_merged: int
    action_qkv_merged: int
    action_latent_fp16: int
    velocity: int
    head_modded: int
    txt_mlp_merged: int
    txt_mlp_gated: int
    img_mlp_merged: int
    img_mlp_gated: int
    single_mlp_gated: int
    proj_scratch: int
    proj_scratch2: int
    action_latent: int
    action_hidden: int
    action_modded: int
    action_proj_scratch: int
    action_proj_scratch2: int
    action_mlp_merged: int
    action_mlp_gated: int
    single_linear2_in: int
    action_linear2_in: int


@dataclass(frozen=True)
class AttentionResource:
    q_o: int
    k_cache: int
    v_cache: int
    logits: int
    kv_layer_stride_bytes: int
    scale: float
    rope_table: int
    action_rope_table: int


@dataclass(frozen=True)
class ImageWAMPipelineResources:
    dims: PipelineDims
    buffers: PipelineBuffers
    attention: AttentionResource
    txt_in: LinearResource
    img_in: LinearResource
    action_encoder: LinearResource
    head_linear: LinearResource
    action_encoder_bias: int
    txt_mod1: AdaLNResource
    txt_mod2: AdaLNResource
    img_mod1: AdaLNResource
    img_mod2: AdaLNResource
    single_mod: AdaLNResource
    double_layers: tuple[DoubleLayerResource, ...]
    single_layers: tuple[SingleLayerResource, ...]
    action_double_layers: tuple[ActionDoubleLayerResource, ...]
    action_single_layers: tuple[SingleLayerResource, ...]
    steps: tuple[ActionStepResource, ...]


class ImageWAMPipelineSource(Protocol):
    """Frontend operations the native pipeline setup uses."""

    def pipeline_resources(self) -> ImageWAMPipelineResources:
        ...

    def gemm_algo(self, kind: int, m: int, n: int, k: int) -> bytes | None:
        ...


class ImageWAMTextLengthPipelineSource(ImageWAMPipelineSource, Protocol):
    """A `ImageWAMPipelineSource` that carries one resource table per text
    length: the context lengths `x0` it has captured, the call that makes one
    of them the active length (its table then describes that one), and the
    length it is active on. `ImageWAMNativeRuntime.capture_pipeline_text_lengths`
    walks them to install and capture the native pipeline once per length."""

    @property
    def captured_text_lengths(self) -> tuple[int, ...]:
        """The context lengths `x0` this source has captured, ascending; a
        read, as on the frontend
        (`ImageWAMTorchFrontendThor.captured_text_lengths`)."""
        ...

    def _activate_text_length(self, x0: int) -> None:
        ...

    @property
    def active_dims(self) -> dict:
        ...
