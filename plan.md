# Plan

Plan Status: proposed

# Problem

## Current

FlashRT has no ImageWAM support. ImageWAM (`yuyangalin/ImageWAM`, cloned
read-only at `/home/ljw/projects/pi0.5/tmp/ImageWAM`) is a world-action
model whose real inference path (`infer_action_flux2` in
`src/imagewam/models/backbones/imagewam.py`) has this shape:

1. A diffusion-transformer image-editing backbone (FLUX.2 4B, the only
   variant with a released checkpoint) runs once per observation, at a
   fixed timestep, to build a per-layer KV cache
   (`prefill_flux2_video_cache`).
2. A flow-matching action expert (ActionDiT) runs an iterative denoise
   loop that reuses that cache through a joint attention mechanism
   (`mot.py::_mixed_attention`) which concatenates the backbone's and
   the action expert's Q/K/V into one shared softmax before splitting
   the output back into two streams.
3. The text instruction is encoded by an external Qwen3-4B encoder.
   `infer_action_flux2` already accepts a precomputed `context`/
   `context_mask` pair and skips the encoder call entirely when they
   are supplied (`_prepare_flux2_infer_text`).

FlashRT's own new-model workflow (`docs/adding_new_model.md`,
`flash_rt/frontends/torch/_template/`) assumes one attention stream per
site: `SiteSpec.kernel` is one of `mha`/`gqa`/`mqa`/`causal`/
`bidirectional`, dispatched in `flash_rt/hardware/thor/attn_backend.py`.
No existing value or dispatch branch covers a joint softmax across two
attention streams. `WEIGHT_SPEC` (`_template/weights_spec.py`,
`flash_rt/models/cosmos3_edge/weights.py`) is state-dict-only: every
row's `torch_key` is resolved against a loaded checkpoint, with no path
to allocate and randomly fill a weight instead.

## Problem

ImageWAM's joint (MoT) attention pattern and its no-real-checkpoint
requirement both fall outside what FlashRT's existing model-integration
abstractions (`AttentionSpec`, `WEIGHT_SPEC`) support.

## Goal

A working, non-NaN, BF16 ImageWAM (FLUX.2-4B shape) forward pass on
Jetson AGX Thor (sm_110), running through FlashRT's pointer-interface /
CUDA-Graph dispatch machinery, with randomly initialized weights and
measured latency.

Out of scope for this plan: RTX/Orin frontends, FP8 quantization,
calibration, real checkpoint loading, and accuracy validation against
a reference. These require real weights and a calibration dataset,
neither of which this plan uses.

# Structure

## Modules

- **ImageWAM weight declaration** — owns every weight tensor's shape,
  dtype, and initialization source (random, for this plan) for the
  FLUX.2-4B backbone, the Qwen3-4B text encoder, and ActionDiT.
- **ImageWAM attention declaration** — owns the per-site attention
  shapes and the new joint-attention kernel dispatch.
- **ImageWAM pipeline forward** — owns the pointer-interface forward
  functions: image/text encode-once, backbone prefill, action-expert
  denoise loop.
- **ImageWAM frontend** — owns buffer allocation, weight upload (random
  path), CUDA Graph capture, and the public `set_prompt()`/`infer()`
  entry points.

## Responsibilities

- Weight declaration is read by the frontend at load time and by
  nothing else.
- Attention declaration is read by the pipeline forward (through
  `attn.run(site=...)`) and by nothing else.
- Pipeline forward reads only pointers and dimensions; it owns no
  persistent state of its own — all state lives in buffers the
  frontend allocated.
- The frontend owns every buffer, the CUDA Graph object, and the
  cached text context; it is the only module that allocates memory or
  touches the CUDA Graph API directly.

## State Ownership

| State | Owner |
|---|---|
| Weight buffers (backbone, text encoder, ActionDiT) | ImageWAM frontend |
| Cached Qwen3-4B `context`/`context_mask` | ImageWAM frontend |
| Per-layer KV cache (backbone prefill output) | ImageWAM frontend (buffer), written by pipeline forward |
| CUDA Graph object | ImageWAM frontend |
| Attention site shapes | ImageWAM attention declaration (read-only after construction) |

# Interface

## Interfaces

```python
# flash_rt/frontends/torch/_imagewam_thor_spec.py
@dataclass
class WeightSpec:
    name: str
    torch_key: Optional[str]   # None -> allocate + randomly fill, skip state-dict lookup
    shape: tuple[int, ...]
    dtype: str
    quant: Optional[QuantSpec] # unused while precision work is out of scope

WEIGHT_SPEC: list[WeightSpec]
```

```python
# flash_rt/hardware/thor/attn_backend.py
@register_attention_spec("imagewam")
def make_imagewam_attention_spec(...) -> AttentionSpec:
    ...
    SiteSpec(name="mot", ..., kernel="mot_joint", extra={...})
```

`kernel="mot_joint"` is a new `SiteSpec.kernel` value. Its dispatch
branch in `attn_backend.py` concatenates the two experts' Q/K/V,
computes one shared softmax, and splits the output back into the two
streams — the direct implementation of `mot.py::_mixed_attention`'s
math in FlashRT's own attention backend.

```python
# flash_rt/models/imagewam/pipeline_thor.py
def imagewam_encode_once(gemm, fvk_module, bufs: dict[str, int],
                          weights: dict[str, int], dims: dict[str, int],
                          *, stream: int = 0) -> None: ...

def imagewam_prefill(gemm, fvk_module, bufs: dict[str, int],
                      weights: dict[str, int], dims: dict[str, int],
                      *, attn=None, stream: int = 0) -> None: ...

def imagewam_denoise_step(gemm, fvk_module, bufs: dict[str, int],
                           weights: dict[str, int], dims: dict[str, int],
                           *, attn=None, stream: int = 0) -> None: ...
```

`bufs` and `weights` are `dict[str, int]` (raw `.data_ptr()` values);
`dims` is `dict[str, int]`. No tensor object crosses a forward
function boundary, matching every existing FlashRT pipeline.

```python
# flash_rt/frontends/torch/imagewam_thor.py
class ImageWAMTorchFrontendThor(FrontendBase):
    def set_prompt(self, prompt: str) -> None: ...
    def infer(self, observation: dict) -> dict:  # {"actions": np.ndarray}
        ...
```

## Inputs

- `set_prompt`: a plain instruction string.
- `infer`: one observation (an image tensor plus optional proprioceptive
  state, matching `infer_action_flux2`'s own `input_image`/`proprio`
  arguments).

## Outputs

- `infer` returns an actions array, shape `[action_horizon, action_dim]`.

## State Changes

- `set_prompt` overwrites the cached `context`/`context_mask`; every
  subsequent `infer` call until the next `set_prompt` uses the cached
  value.
- `infer` overwrites the per-layer KV cache buffer every call (no
  cross-call KV reuse — each call is a new observation).

# Flow

1. `set_prompt(prompt)` (first call, or whenever the instruction
   changes): runs the Qwen3-4B encoder once, writes `context`/
   `context_mask` into cached buffers on the frontend.
2. `infer(observation)`:
   - `imagewam_encode_once`: encodes the current observation image into
     patch tokens; reads the cached text context, does not recompute it.
   - `imagewam_prefill`: one backbone forward at fixed timestep,
     writing the per-layer KV cache buffer.
   - A loop of `imagewam_denoise_step` calls (the flow-matching
     schedule length), each reading the KV cache through the
     `mot_joint` attention site and advancing the action latent one
     step. The loop body is captured once as a single CUDA Graph
     (`flash_rt/models/cosmos3_edge/static_engine.py`'s `StaticEngine`
     pattern: capture the whole loop once at frontend construction,
     `replay()` on every `infer` call; the per-step timestep is a
     device-side scalar written before each replay, not a Python
     argument).
   - The final action latent is read back and returned.

# Code Mapping

## Modules

| Module | File |
|---|---|
| ImageWAM weight declaration | `flash_rt/frontends/torch/_imagewam_thor_spec.py` (new) |
| ImageWAM attention declaration | `flash_rt/hardware/thor/attn_backend.py` (append) |
| ImageWAM pipeline forward | `flash_rt/models/imagewam/pipeline_thor.py` (new) |
| ImageWAM frontend | `flash_rt/frontends/torch/imagewam_thor.py` (new) |

## Interfaces

| Interface | File |
|---|---|
| `WeightSpec`, `WEIGHT_SPEC` | `flash_rt/frontends/torch/_imagewam_thor_spec.py` |
| `make_imagewam_attention_spec`, `kernel="mot_joint"` dispatch | `flash_rt/hardware/thor/attn_backend.py` |
| `imagewam_encode_once`, `imagewam_prefill`, `imagewam_denoise_step` | `flash_rt/models/imagewam/pipeline_thor.py` |
| `ImageWAMTorchFrontendThor` | `flash_rt/frontends/torch/imagewam_thor.py` |

## State

| State | File |
|---|---|
| Weight buffers, KV cache buffer, cached text context, CUDA Graph object | `flash_rt/frontends/torch/imagewam_thor.py` (`ImageWAMTorchFrontendThor`) |

# Implementation

## Phase 1 — Weight declaration with a random-init path

Phase Status: pending

### Goal

Declare every weight tensor FLUX.2-4B-shaped ImageWAM needs (backbone,
Qwen3-4B text encoder, ActionDiT), with a random-fill path in place of
a state-dict lookup.

### Files

`flash_rt/frontends/torch/_imagewam_thor_spec.py` (new).

### Structures

`WeightSpec.torch_key: Optional[str]`; `None` triggers allocation and
random fill in the loader instead of a state-dict lookup. Real
tensor shapes come from
`configs/model/imagewam_flux2_klein_4b_base.yaml` and the upstream
FLUX.2/Qwen3-4B configs — reading those configs is part of this
phase's own work, not a precondition for approving it.

### Affected Modules

ImageWAM weight declaration only.

### Observation

The loader allocates every declared buffer and fills it with random
values without a state-dict present; total allocated byte count is
logged and checked against the sum of declared shapes/dtypes.

## Phase 2 — Joint attention kernel

Phase Status: pending

### Goal

A `mot_joint` attention kernel implementing
`mot.py::_mixed_attention`'s concatenated-QKV joint softmax.

### Files

`flash_rt/hardware/thor/attn_backend.py` (append `kernel="mot_joint"`
dispatch branch and its implementation).

### Structures

New `SiteSpec.kernel` value `"mot_joint"`.

### Affected Modules

ImageWAM attention declaration only.

### Observation

A standalone test compares the new kernel's output against a
plain-PyTorch concatenated-softmax reference, on random inputs at
ImageWAM's real attention shape. This checks the kernel against its
own mathematical definition; it is not a calibration or accuracy
check against a trained model, and stays in scope even though
calibration itself does not.

## Phase 3 — Encode-once and backbone prefill

Phase Status: pending

### Goal

`imagewam_encode_once` and `imagewam_prefill`, producing a populated
KV cache buffer from random weights without NaN/Inf.

### Files

`flash_rt/models/imagewam/pipeline_thor.py` (new).

### Structures

None new; consumes Phase 1's buffers and Phase 2's attention site.

### Affected Modules

ImageWAM pipeline forward only.

### Observation

Given random weights, the two functions run without NaN/Inf and
without violating the pointer-interface contract (no `.cpu()`,
`.numpy()`, or `torch.empty()` inside either function); output tensor
shapes match the declared dims.

## Phase 4 — Denoise loop and CUDA Graph capture

Phase Status: pending

### Goal

`imagewam_denoise_step`, called in a loop whose body is captured once
as a single CUDA Graph, replayed on every `infer` call.

### Files

`flash_rt/models/imagewam/pipeline_thor.py` (same file as Phase 3).

### Structures

Reuses `flash_rt/models/cosmos3_edge/static_engine.py`'s capture-once/
replay-many pattern; whether `StaticEngine` itself is reusable as-is or
needs an ImageWAM-specific subclass is resolved during this phase, not
assumed here. The flow-matching step schedule is ImageWAM's own
(`src/imagewam/models/backbones/schedulers/scheduler_continuous.py`),
not cosmos3_edge's UniPC scheduler — only the capture/replay/device-
scalar pattern is reused, not the scheduler class.

### Affected Modules

ImageWAM pipeline forward only.

### Observation

The captured graph replays repeatedly without recapture; P50 latency
over repeated replays is measured and recorded.

## Phase 5 — Frontend and text-context caching

Phase Status: pending

### Goal

`ImageWAMTorchFrontendThor`, wiring Phases 1-4 into `set_prompt()` and
`infer()`.

### Files

`flash_rt/frontends/torch/imagewam_thor.py` (new).

### Structures

None new; owns the state listed in Structure's State Ownership table.

### Affected Modules

ImageWAM frontend only.

### Observation

`set_prompt()` followed by repeated `infer()` calls (fixed prompt,
varying random observations) succeeds on Thor with random weights,
returns non-NaN actions, and reports P50 `infer()` latency.
