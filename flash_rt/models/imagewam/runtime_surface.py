"""Interface between a captured ImageWAM frontend and its runtime export.

`ImageWAMRuntimeSurface` names the captured graphs, the stream they replay
on and the three device windows they read and write.
`ImageWAMRuntimeSource` is the set of frontend operations the model-runtime
verbs call; the frontend owns every buffer and every staging operation, and
the export module (`runtime_export.py`) only wraps pointers and dispatches
verbs.

A frontend captures one graph per text-context length (`text_trim`), so the
surface carries a variant table rather than one exec: `GraphVariants` is
that table (the key is the context length `x0` the graph was captured for),
and `graph_variant_plan` turns it into the export's declaration
(`runtime/export.py` `GraphSpec`) — one adopted exec per key, the active
length as the default key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

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
class TextLengthGraph:
    """One captured text-context length and the graph variant serving it.

    - `key`: the context length `x0` the graph was captured for — the rows
      its attention reads (`n_valid + 1` with a proprio row, `n_valid`
      without). It is the `ShapeKey` this model varies on, so it keys the
      export's variant table.
    - `graph_exec`: the instantiated exec of that capture
      (`torch.cuda.CUDAGraph.raw_cuda_graph_exec()`), owned by the
      frontend's `CUDAGraph`.
    """

    key: int
    graph_exec: int


@dataclass(frozen=True)
class GraphVariants:
    """Every graph variant one captured frontend serves, keyed by text
    length.

    - `active_key`: the context length of the graph the frontend replays
      now (`active_dims["x0"]`). `step` selects this key.
    - `entries`: one `TextLengthGraph` per captured length, `key`
      ascending, the active one included. It is the set
      `precapture_text_lengths` fills and `set_prompt` extends.
    - `per_prompt_length`: True when the frontend keeps one graph per
      prompt length (`text_trim=True`), so a later `set_prompt` may change
      the active key; False when one graph serves every prompt at
      `dims["x0"]`.
    """

    active_key: int
    entries: tuple[TextLengthGraph, ...]
    per_prompt_length: bool


@dataclass(frozen=True)
class GraphVariantPlan:
    """What an export declares for a surface's graph table: the key a plain
    host fires (`runtime/export.py` `GraphSpec.default_key`), every adopted
    key (`GraphSpec.keys`, ascending) and the exec `Graph`'s LRU cap
    (`ctx.graph(name, max_variants)`)."""

    default_key: int
    keys: tuple[int, ...]
    max_variants: int


def graph_variant_plan(variants: GraphVariants) -> GraphVariantPlan:
    """`variants` as the export's declaration: one adopted exec per captured
    length, `default_key` the active one (`graph_exec` and the default key
    then describe the same graph).

    `max_variants` is the table's own size. The export adopts exactly the
    lengths `variants` names and never captures (`step` refuses a length
    with no variant instead), so the cap can be exact and no LRU eviction
    can drop a declared length.

    Raises `ValueError` when the table is empty, when a key repeats, or
    when `active_key` is not in it: the active graph would then be missing
    from the declaration.
    """
    keys = tuple(entry.key for entry in variants.entries)
    if not keys:
        raise ValueError("a runtime surface must expose at least one graph variant")
    if len(set(keys)) != len(keys):
        raise ValueError(f"graph variant keys repeat: {keys}")
    if variants.active_key not in keys:
        raise ValueError(f"the active key {variants.active_key} is not in the graph variant table "
                         f"{keys}")
    return GraphVariantPlan(default_key=variants.active_key, keys=tuple(sorted(keys)),
                            max_variants=len(keys))


def uncaptured_text_length_message(key: int, keys: tuple[int, ...]) -> str:
    """The `step` failure for the length a prompt set when the runtime
    adopted no variant for it: names the length and the lengths it did
    adopt."""
    adopted = ", ".join(str(k) for k in keys) if keys else "none"
    return (f"step: no graph variant for text length x0={key}; this runtime was exported with the "
            f"lengths [{adopted}]. The length was never captured: declare it in "
            f"precapture_text_lengths at startup, before exporting, or set a prompt of an exported "
            f"length")


@dataclass(frozen=True)
class ImageWAMRuntimeSurface:
    """What one captured ImageWAM frontend exposes to the runtime export.

    Every field describes the *active* graph: with `text_trim=True` the
    frontend runs one graph per prompt length, so `img_len`, `context_rows`,
    `setup_identity` and `graph_exec` are those of the length the last
    `set_prompt` set, and `graph_variants` is the table of every captured
    length (its `active_key` names the one the rest of the surface
    describes).

    - `graph_exec`: the instantiated exec of the captured prefill + denoise
      graph for the active length
      (`torch.cuda.CUDAGraph.raw_cuda_graph_exec()`), owned by the
      frontend's `CUDAGraph`. It is the entry `graph_variants` holds under
      `active_key`.
    - `graph_variants`: every captured length and its exec, keyed by the
      context length `x0` the graph was captured for. `text_trim=False`:
      one entry, `dims["x0"]`, `per_prompt_length=False`.
    - `stream`: the capture stream. ABI replay and every staging verb run
      on it, so staged writes are ordered before the replay.
    - `img_raw`: `(img_len, token_dim)` bf16 VAE tokens, read by the graph.
      `img_len` is `a0 - x0`, which trimming leaves unchanged (the image
      and action blocks keep their lengths).
    - `context`: `(x0_max, joint_attention_dim)` bf16 prompt context
      including the proprio row; the active graph reads its first
      `context_rows` rows.
    - `action_latent`: `(num_action, action_dim)` f32; the initial noise on
      entry, the normalized action chunk after replay (in place).
    - `setup_identity`: ordered `(key, value)` pairs describing the setup
      (precision, dimensions, flags) that the captured graph depends on.
      `dims.<key>` carries the resolved dims of the active graph (`x0`,
      `a0` and `total` of the active length with `text_trim=True`, the
      max dims otherwise); `text_trim` records whether one graph per prompt
      length is in play. A frontend constructed from a resolved
      configuration also carries `workload.<field>`
      (`WORKLOAD_IDENTITY_FIELDS`), so a runtime and a calibration file are
      checked against the workload that was served and not only against the
      dims it produced.
    - `proprio_row`: the context row the proprio token goes to for the
      current prompt (`None` without proprio); changes with the prompt.
    - `proprio_weight` / `proprio_bias`: the real `proprio_encoder`,
      `(joint_attention_dim, proprio_dim)` / `(joint_attention_dim,)` bf16.
    - `state_scale` / `state_offset` / `action_scale` / `action_offset`:
      f32 min/max normalization constants from `dataset_stats.json`
      (`None` when not loaded).
    - `view_shape`: `(views, H, W)` of the uint8 frames `stage_images`
      takes, as the frontend resolves it
      (`ImageWAMTorchFrontendThor._input_view_shape`): the resolved
      workload's `vae_graph_input()` when the frontend has a workload,
      else the in-graph VAE stage's own spec, else `(2, 224, 224)` for a
      caller that passed dims by hand.
    - `views_u8`: the graph's `(views, H, W, 3)` uint8 view buffer when
      the VAE runs inside the graph (the graph then writes `img_raw`
      itself), else `None`.
    - `owner`: the object that owns `graph_exec` and every buffer and
      weight the graph reads or writes (the frontend). Holding the surface
      keeps them alive.
    """

    graph_exec: int
    graph_variants: GraphVariants
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

    @property
    def active_dims(self) -> dict[str, Any]:
        """The dims the currently active graph runs. `step` reads `x0`: it
        is the variant key of the length the prompt actually set."""
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
