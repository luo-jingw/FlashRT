"""Checks shared by the ImageWAM model-runtime gates and tests.

- `poison_tick_state`: NaN-fills every buffer an ABI tick must write or
  refresh, so a tick cannot pass on values the reference `infer()` left
  behind in the shared buffers.
- `python_verb_noop` / `python_step_noop`: mutants of the `io="python"`
  verbs (the verb reports success and does nothing). They patch
  `ImageWAMPythonVerbs`, so the runtime must be exported inside the
  context.
- `MutatedPipelineSource`: an `ImageWAMPipelineSource` whose resource
  table drops or misroutes work, for mutants of the native pipeline.

A gate row is only meaningful if its mutant makes it fail; the gates and
tests run both.
"""
from __future__ import annotations

import contextlib
import dataclasses
from typing import Iterator

import torch

from flash_rt.models.imagewam import runtime_export
from flash_rt.models.imagewam.pipeline_resources import ImageWAMPipelineResources, ImageWAMPipelineSource

PIPELINE_MUTATIONS = ("no_backbone", "skip_last_single", "skip_last_step", "swap_single_weight")


def poison_tick_state(fe, *, whole_context: bool = False) -> None:
    """NaN-fill `img_raw`, the proprio context row (the whole context with
    `whole_context`), the K/V caches, `Q_O`, the backbone residual and the
    action latent; zero the in-graph VAE's uint8 view buffer."""
    torch.cuda.synchronize()
    nan = float("nan")
    fe._img_raw.fill_(nan)
    if whole_context:
        fe._context.fill_(nan)
    elif fe._proprio_row is not None:
        fe._context[fe._proprio_row].fill_(nan)
    for t in (fe._K_cache, fe._V_cache, fe._Q_O, fe._backbone_hidden, fe._action_latent):
        t.fill_(nan)
    if fe._vae_stage is not None:
        fe._vae_stage.views_u8.zero_()
    torch.cuda.synchronize()


def bits(t: torch.Tensor):
    """Host copy of a tensor's raw bits (NaN-safe exact comparison)."""
    t = t.detach().contiguous()
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    return t.view(view).cpu().numpy()


@contextlib.contextmanager
def python_verb_noop(port: str) -> Iterator[None]:
    """`set_input(port)` of the `io="python"` face returns success and stages nothing."""
    original = runtime_export.ImageWAMPythonVerbs.set_input

    def noop(self, index: int, payload: bytes, stream: int) -> int:
        if 0 <= index < len(self._layout.names) and self._layout.names[index] == port:
            return runtime_export.STATUS_OK
        return original(self, index, payload, stream)

    runtime_export.ImageWAMPythonVerbs.set_input = noop
    try:
        yield
    finally:
        runtime_export.ImageWAMPythonVerbs.set_input = original


@contextlib.contextmanager
def python_step_noop() -> Iterator[None]:
    """`step` of the `io="python"` face returns success and replays nothing."""
    original = runtime_export.ImageWAMPythonVerbs.step
    runtime_export.ImageWAMPythonVerbs.step = lambda self: runtime_export.STATUS_OK
    try:
        yield
    finally:
        runtime_export.ImageWAMPythonVerbs.step = original


class MutatedPipelineSource:
    """`source`'s pipeline resources with one mutation applied:

    - `no_backbone`: prefill runs only `txt_in` / `img_in`, no backbone block;
    - `skip_last_single`: prefill omits the last backbone single-stream block;
    - `skip_last_step`: the denoise loop runs one step fewer;
    - `swap_single_weight`: the second-to-last backbone single-stream block
      uses the last block's `linear1` weight.
    """

    def __init__(self, source: ImageWAMPipelineSource, mutation: str):
        if mutation not in PIPELINE_MUTATIONS:
            raise ValueError(f"unknown mutation {mutation!r}")
        self._source = source
        self.mutation = mutation

    def pipeline_resources(self) -> ImageWAMPipelineResources:
        r = self._source.pipeline_resources()
        d = r.dims
        if self.mutation == "no_backbone":
            return dataclasses.replace(r, dims=dataclasses.replace(d, num_double=0, num_single=0),
                                       double_layers=(), single_layers=())
        if self.mutation == "skip_last_single":
            return dataclasses.replace(r, dims=dataclasses.replace(d, num_single=d.num_single - 1),
                                       single_layers=r.single_layers[:-1])
        if self.mutation == "skip_last_step":
            return dataclasses.replace(r, dims=dataclasses.replace(d, num_steps=d.num_steps - 1),
                                       steps=r.steps[:-1])
        layers = list(r.single_layers)
        layers[-2] = dataclasses.replace(layers[-2], linear1=layers[-1].linear1)
        return dataclasses.replace(r, single_layers=tuple(layers))

    def gemm_algo(self, kind: int, m: int, n: int, k: int) -> bytes | None:
        return self._source.gemm_algo(kind, m, n, k)
