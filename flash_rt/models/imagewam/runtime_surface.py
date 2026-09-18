"""Interface between a captured ImageWAM frontend and its runtime export.

`ImageWAMRuntimeSurface` names the captured graph, its stream and the
three device windows the graph reads and writes. `ImageWAMRuntimeSource`
is the set of frontend operations the model-runtime verbs call; the
frontend owns every buffer and every staging operation, and the export
module (`runtime_export.py`) only wraps pointers and dispatches verbs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

# The workload fields a captured graph depends on, as runtime-identity keys
# (`workload.<field>`). `num_views`, `image_h` and `image_w` name the image
# geometry, `text_max_len` with the proprio row gives `x0`, `action_horizon`
# the action block, `action_dim` the head width, `proprio_dim` whether a
# proprio row exists, and `num_steps` with `shift` the denoise schedule.
# `ImageWAMWorkload` owns the values; this tuple only names them.
WORKLOAD_IDENTITY_FIELDS = ("num_views", "image_h", "image_w", "text_max_len", "action_horizon",
                            "action_dim", "proprio_dim", "num_steps", "shift")


def workload_identity(workload) -> tuple[tuple[str, str], ...]:
    """`(("workload.<field>", <value>), ...)` for every
    `WORKLOAD_IDENTITY_FIELDS` entry of `workload` (`ImageWAMWorkload`)."""
    return tuple((f"workload.{name}", str(getattr(workload, name)))
                 for name in WORKLOAD_IDENTITY_FIELDS)


@dataclass(frozen=True)
class ImageWAMRuntimeSurface:
    """What one captured ImageWAM frontend exposes to the runtime export.

    - `graph_exec`: the instantiated exec of the captured prefill + denoise
      graph (`torch.cuda.CUDAGraph.raw_cuda_graph_exec()`), owned by the
      frontend's `CUDAGraph`.
    - `stream`: the capture stream. ABI replay and every staging verb run
      on it, so staged writes are ordered before the replay.
    - `img_raw`: `(img_len, token_dim)` bf16 VAE tokens, read by the graph.
    - `context`: `(x0, joint_attention_dim)` bf16 prompt context including
      the proprio row, read by the graph.
    - `action_latent`: `(num_action, action_dim)` f32; the initial noise on
      entry, the normalized action chunk after replay (in place).
    - `setup_identity`: ordered `(key, value)` pairs describing the setup
      (precision, dimensions, flags) that the captured graph depends on.
      `dims.<key>` carries the resolved dims the frontend built its buffers
      from; a frontend constructed from a resolved configuration also
      carries `workload.<field>` (`WORKLOAD_IDENTITY_FIELDS`), so a runtime
      and a calibration file are checked against the workload that was
      served and not only against the dims it produced.
    - `proprio_row`: the context row the proprio token goes to for the
      current prompt (`None` without proprio); changes with the prompt.
    - `proprio_weight` / `proprio_bias`: the real `proprio_encoder`,
      `(joint_attention_dim, proprio_dim)` / `(joint_attention_dim,)` bf16.
    - `state_scale` / `state_offset` / `action_scale` / `action_offset`:
      f32 min/max normalization constants from `dataset_stats.json`
      (`None` when not loaded).
    - `view_shape`: `(views, H, W)` of the uint8 frames `stage_images`
      takes: `(2, 224, 224)` with the VAE outside the graph, the
      frontend's `vae_graph_input` with the VAE inside it.
    - `views_u8`: the graph's `(views, H, W, 3)` uint8 view buffer when
      the VAE runs inside the graph (the graph then writes `img_raw`
      itself), else `None`.
    - `owner`: the object that owns `graph_exec` and every buffer and
      weight the graph reads or writes (the frontend). Holding the surface
      keeps them alive.
    """

    graph_exec: int
    stream: torch.cuda.Stream
    img_raw: torch.Tensor
    context: torch.Tensor
    action_latent: torch.Tensor
    img_len: int
    token_dim: int
    num_action: int
    action_dim: int
    proprio_dim: int | None
    has_vae: bool
    has_text_encoder: bool
    action_denormalized: bool
    setup_identity: tuple[tuple[str, str], ...]
    context_rows: int
    context_width: int
    proprio_row: int | None
    proprio_weight: torch.Tensor | None
    proprio_bias: torch.Tensor | None
    state_scale: torch.Tensor | None
    state_offset: torch.Tensor | None
    action_scale: torch.Tensor | None
    action_offset: torch.Tensor | None
    view_shape: tuple[int, int, int]
    views_u8: torch.Tensor | None
    owner: object = field(repr=False, compare=False)


class ImageWAMRuntimeSource(Protocol):
    """Frontend operations the ImageWAM model-runtime verbs dispatch to."""

    def runtime_surface(self) -> ImageWAMRuntimeSurface:
        ...

    def stage_images(self, *views: torch.Tensor) -> None:
        ...

    def stage_proprio(self, proprio: np.ndarray) -> None:
        ...

    def set_prompt(self, prompt_text: str | None = None, *,
                   context: torch.Tensor | None = None,
                   context_mask: torch.Tensor | None = None) -> None:
        ...

    def read_actions(self) -> np.ndarray:
        ...
