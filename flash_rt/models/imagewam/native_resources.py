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

from flash_rt.models.imagewam.native_library import ImageWAMIoConfig
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
