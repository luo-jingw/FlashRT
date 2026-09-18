"""Builds the handoff structs of the ImageWAM native library from a
captured frontend's `ImageWAMRuntimeSurface`.

Every device pointer in a struct is borrowed from the frontend; tensors
this module materializes for the handoff (the transposed proprio weight)
and the host arrays behind float pointers are returned in the keepalive
list, which the caller holds for as long as the native handle lives.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
import torch

from flash_rt.models.imagewam.native_library import (
    PIPELINE_BUFFER_FIELDS,
    PIPELINE_DIM_FIELDS,
    ImageWAMActionDoubleLayer,
    ImageWAMActionStep,
    ImageWAMAdaLN,
    ImageWAMDoubleLayer,
    ImageWAMIoConfig,
    ImageWAMLinear,
    ImageWAMPipelineConfig,
    ImageWAMSingleLayer,
)
from flash_rt.models.imagewam.pipeline_resources import (
    AdaLNResource,
    ImageWAMPipelineResources,
    LinearResource,
)
from flash_rt.models.imagewam.runtime_surface import ImageWAMRuntimeSurface


@dataclass(frozen=True)
class NativeHandoff:
    """A filled config struct plus everything its pointers reference."""

    config: ctypes.Structure
    keepalive: tuple


def _host_f32(t: torch.Tensor | None, keep: list) -> ctypes.POINTER(ctypes.c_float) | None:
    if t is None:
        return None
    arr = np.ascontiguousarray(t.detach().float().cpu().numpy(), dtype=np.float32)
    keep.append(arr)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def build_io_config(surface: ImageWAMRuntimeSurface) -> NativeHandoff:
    keep: list = []
    c = ImageWAMIoConfig()
    c.struct_size = ctypes.sizeof(ImageWAMIoConfig)
    c.img_len = surface.img_len
    c.token_dim = surface.token_dim
    c.num_action = surface.num_action
    c.action_dim = surface.action_dim
    c.proprio_dim = surface.proprio_dim or 0
    c.context_rows = surface.context_rows
    c.context_width = surface.context_width
    c.img_raw = surface.img_raw.data_ptr()
    c.context = surface.context.data_ptr()
    c.action_latent = surface.action_latent.data_ptr()
    if surface.proprio_dim:
        weight_t = surface.proprio_weight.detach().to(torch.bfloat16).t().contiguous()
        bias = surface.proprio_bias.detach().to(torch.bfloat16).contiguous()
        keep.extend((weight_t, bias))
        c.proprio_weight_t = weight_t.data_ptr()
        c.proprio_bias = bias.data_ptr()
        c.state_scale = _host_f32(surface.state_scale, keep)
        c.state_offset = _host_f32(surface.state_offset, keep)
    c.action_scale = _host_f32(surface.action_scale, keep)
    c.action_offset = _host_f32(surface.action_offset, keep)
    return NativeHandoff(config=c, keepalive=tuple(keep))


def _linear(r: LinearResource) -> ImageWAMLinear:
    return ImageWAMLinear(kind=r.kind, n=r.n, k=r.k, fp4_variant=r.fp4_variant, weight=r.weight,
                          weight_scales=r.weight_scales or None, act_packed=r.act_packed or None,
                          act_scales=r.act_scales or None)


def _adaln(r: AdaLNResource) -> ImageWAMAdaLN:
    return ImageWAMAdaLN(shift=r.shift.data_ptr(), scale=r.scale.data_ptr(),
                         gate=None if r.gate is None else r.gate.data_ptr())


def build_pipeline_config(resources: ImageWAMPipelineResources) -> NativeHandoff:
    """`frt_imagewam_pipeline_config` over the frontend's resources; the
    layer and step arrays and `resources` itself are in the keepalive."""
    c = ImageWAMPipelineConfig()
    c.struct_size = ctypes.sizeof(ImageWAMPipelineConfig)
    for name in PIPELINE_DIM_FIELDS:
        setattr(c, name, int(getattr(resources.dims, name)))
    c.eps = resources.dims.eps
    for name in PIPELINE_BUFFER_FIELDS:
        setattr(c, name, int(getattr(resources.buffers, name)))
    a = resources.attention
    c.q_o, c.k_cache, c.v_cache, c.logits = a.q_o, a.k_cache, a.v_cache, a.logits
    c.kv_layer_stride_bytes = a.kv_layer_stride_bytes
    c.attn_scale = a.scale
    c.rope_table, c.action_rope_table = a.rope_table, a.action_rope_table
    c.txt_in = _linear(resources.txt_in)
    c.img_in = _linear(resources.img_in)
    c.action_encoder = _linear(resources.action_encoder)
    c.head_linear = _linear(resources.head_linear)
    c.action_encoder_bias = resources.action_encoder_bias
    for name in ("txt_mod1", "txt_mod2", "img_mod1", "img_mod2", "single_mod"):
        setattr(c, name, _adaln(getattr(resources, name)))

    doubles = (ImageWAMDoubleLayer * max(1, len(resources.double_layers)))()
    for i, L in enumerate(resources.double_layers):
        for name, _ in ImageWAMDoubleLayer._fields_:
            value = getattr(L, name)
            setattr(doubles[i], name, _linear(value) if isinstance(value, LinearResource) else value)
    singles = (ImageWAMSingleLayer * max(1, len(resources.single_layers)))()
    for i, L in enumerate(resources.single_layers):
        singles[i] = ImageWAMSingleLayer(_linear(L.linear1), _linear(L.attn_out_proj),
                                         _linear(L.mlp_down), L.query_norm, L.key_norm)
    action_doubles = (ImageWAMActionDoubleLayer * max(1, len(resources.action_double_layers)))()
    for i, L in enumerate(resources.action_double_layers):
        action_doubles[i] = ImageWAMActionDoubleLayer(_linear(L.qkv), _linear(L.proj), _linear(L.mlp0),
                                                      _linear(L.mlp2), L.query_norm, L.key_norm)
    action_singles = (ImageWAMSingleLayer * max(1, len(resources.action_single_layers)))()
    for i, L in enumerate(resources.action_single_layers):
        action_singles[i] = ImageWAMSingleLayer(_linear(L.linear1), _linear(L.attn_out_proj),
                                                _linear(L.mlp_down), L.query_norm, L.key_norm)
    steps = (ImageWAMActionStep * len(resources.steps))()
    for i, st in enumerate(resources.steps):
        steps[i] = ImageWAMActionStep(_adaln(st.double1), _adaln(st.double2), _adaln(st.single),
                                      _adaln(st.head), st.delta, 0)
    c.double_layers = doubles
    c.single_layers = singles
    c.action_double_layers = action_doubles
    c.action_single_layers = action_singles
    c.steps = steps
    return NativeHandoff(config=c, keepalive=(resources, doubles, singles, action_doubles,
                                               action_singles, steps))
