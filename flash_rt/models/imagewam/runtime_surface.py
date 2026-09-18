"""Interface between a captured ImageWAM frontend and its runtime export.

`ImageWAMRuntimeSurface` names the captured graph, its stream and the
three device windows the graph reads and writes. `ImageWAMRuntimeSource`
is the set of frontend operations the model-runtime verbs call; the
frontend owns every buffer and every staging operation, and the export
module (`runtime_export.py`) only wraps pointers and dispatches verbs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


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
    - `proprio_row`: the context row the proprio token goes to for the
      current prompt (`None` without proprio); changes with the prompt.
    - `proprio_weight` / `proprio_bias`: the real `proprio_encoder`,
      `(joint_attention_dim, proprio_dim)` / `(joint_attention_dim,)` bf16.
    - `state_scale` / `state_offset` / `action_scale` / `action_offset`:
      f32 min/max normalization constants from `dataset_stats.json`
      (`None` when not loaded).
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


class ImageWAMRuntimeSource(Protocol):
    """Frontend operations the ImageWAM model-runtime verbs dispatch to."""

    def runtime_surface(self) -> ImageWAMRuntimeSurface:
        ...

    def stage_images(self, view1: torch.Tensor, view2: torch.Tensor | None) -> None:
        ...

    def stage_proprio(self, proprio: np.ndarray) -> None:
        ...

    def set_prompt(self, prompt_text: str | None = None, *,
                   context: torch.Tensor | None = None,
                   context_mask: torch.Tensor | None = None) -> None:
        ...

    def read_actions(self) -> np.ndarray:
        ...
