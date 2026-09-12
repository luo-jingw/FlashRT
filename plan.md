# Plan

Plan Status: approved

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
  FLUX.2-4B backbone and ActionDiT. Does not include the Qwen3-4B text
  encoder: every ImageWAM config sets `load_text_encoder: false`, so
  text conditioning is always an input (`context`/`context_mask`),
  never a weight this integration owns.
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
# Modeled on the real _cosmos3_edge_thor_spec.py, not the generic
# template: shape declaration is separate from data loading, no
# loader callables live in this file at all.
@dataclass(frozen=True)
class ImageWAMThorSpec:
    backbone_num_heads: int = 24
    backbone_head_dim: int = 128
    backbone_num_layers_double: int = 5
    backbone_num_layers_single: int = 20
    backbone_in_channels: int = 128
    joint_attention_dim: int = 7680      # context input width, not a weight
    action_hidden_dim: int = 1024
    action_num_heads: int = 24           # == backbone_num_heads, required for mot_joint
    action_head_dim: int = 128           # == backbone_head_dim, required for mot_joint
    action_num_layers_double: int = 5    # == backbone_num_layers_double
    action_num_layers_single: int = 20   # == backbone_num_layers_single
    max_action_horizon: int = 64

SPEC = ImageWAMThorSpec()
BACKBONE_LAYER_SHAPES: dict[str, tuple[int, ...]]
ACTION_DIT_LAYER_SHAPES: dict[str, tuple[int, ...]]

def iter_expected_shapes() -> Iterable[tuple[str, tuple[int, ...]]]: ...
```

```python
# flash_rt/hardware/thor/attn_backend.py
# Real shape (flash_rt/hardware/backend.py): AttentionSpec.add_site is
# a plain method, not a decorator-registered function -- there is no
# @register_attention_spec mechanism in this codebase (an earlier draft
# of this plan assumed one; it does not exist). make_imagewam_attention_spec
# is a plain function placed next to the existing make_pi05_attention_spec,
# called directly by the frontend, following that file's own convention.
def make_imagewam_attention_spec(...) -> AttentionSpec:
    spec = AttentionSpec()
    spec.add_site(
        "mot",
        num_layers=...,
        num_q_heads=24,       # shared with the FLUX.2 backbone, see Phase 1
        num_kv_heads=24,      # ImageWAM's MoT attention is not GQA
        head_dim=128,
        max_q_seq=...,        # total concatenated sequence length
        extra={"kernel": "mot_joint", "block_boundaries": ("t0", "r0", "x0", "a0")},
    )
    return spec
```

`kernel="mot_joint"` is a new `SiteSpec.kernel` value, confirmed
necessary: `flash_rt/hardware/thor/attn_backend.py::run()` dispatches
only `"standard"` (`fvk.attention_qkv_fp16`, plain/GQA self-attention,
no mask argument at all) and `"state_masked"`
(`fvk.attention_qkv_fp16_state_masked`, a single leading-visibility
boundary specific to Pi0's decoder). Neither accepts an arbitrary mask
or matches ImageWAM's block structure. (`"mha"`, listed in
`docs/adding_new_model.md`'s kernel table, does not exist anywhere in
this Thor backend or in `flash_rt/hardware/backend.py` — that table
entry does not apply here.)

ImageWAM's real mask (`mot.py::_mixed_attention`, plain masked
`F.scaled_dot_product_attention`; mask built by
`imagewam.py::_build_mot_attention_mask_flux2`) is a fixed three-block
pattern over `[prefix (text+ref) | target-image | action]`. Only TWO
boundaries actually affect visibility, `x0` and `a0` — `t0`/`r0` (the
text/ref internal split) never matter for the mask, since prefix is
always visible or invisible as one unit: prefix sees `[0,x0)`;
target-image sees `[0,a0)`; action sees `[0,x0) U [a0,total)` — NOT
target-image. The boundaries are scalars fixed per call, not a full
`[total, total]` boolean tensor — `kernel="mot_joint"`'s dispatch
takes them as small device-side scalars, the same pattern
`"state_masked"` already uses for its own single boundary
(`state_nk`), rather than materializing and uploading a full mask
every call. Implemented and verified in Phase 2 (see below).

```python
# flash_rt/models/imagewam/pipeline_thor.py
# Real signature (flash_rt/frontends/torch/_template/pipeline.py):
# (ctx, fvk, bufs, weights, dims, stream=0, *, attn=None) -- an earlier
# draft of this plan omitted `ctx` (the FvkContext / cuBLAS-handle
# object, positional) and used non-matching param names.
def imagewam_encode_once(ctx, fvk, bufs: dict, weights: dict, dims: dict,
                          stream: int = 0) -> None: ...

def imagewam_prefill(ctx, fvk, bufs: dict, weights: dict, dims: dict,
                      stream: int = 0, *, attn=None) -> None: ...

def imagewam_denoise_step(ctx, fvk, bufs: dict, weights: dict, dims: dict,
                           stream: int = 0, *, attn=None) -> None: ...
```

`bufs` values are raw `.data_ptr()` integers (allocated once by the
frontend via `CudaBuffer`, per the template). `weights` is keyed by
the same `(site, layer_idx, slot_name)` tuples as `WEIGHT_SPEC`, each
value a pointer. `dims` is `dict[str, int]`. No tensor object crosses
a forward function boundary, matching every existing FlashRT
pipeline. No `quantize_fp8_static`/`gemm_fp8_fp16`/alpha-scale calls
appear in these three functions — this plan's forward is BF16-only
(plain `gemm_bf16_nn`-style GEMMs), matching the template's
`*_forward_calibrate` compute shape but without its
`_measure_scale_gpu` calls, since no calibration happens either.

```python
# flash_rt/frontends/torch/imagewam_thor.py
# Real shape (flash_rt/frontends/torch/_template/frontend.py): a plain
# class, no base class -- an earlier draft of this plan assumed a
# FrontendBase that does not appear in the template.
class ImageWAMTorchFrontendThor:
    def __init__(self, checkpoint_dir=None, **kwargs) -> None: ...
        # checkpoint_dir is accepted for interface parity with every
        # other frontend but unused while this plan random-fills every
        # shape from _imagewam_thor_spec.iter_expected_shapes() instead
        # of loading a real checkpoint.
    def set_prompt(self, prompt_text: str) -> None: ...
        # No calibration-cache lookup, no _recalibrate_with_real_data,
        # and no real Qwen3-4B forward: ImageWAM's own config always
        # sets load_text_encoder=false, so `context`/`context_mask` are
        # inputs this integration never computes -- set_prompt random-
        # fills the context input buffer to the expected shape, then
        # captures the CUDA Graph directly.
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
   changes): random-fills the `context`/`context_mask` input buffers
   to their expected shape (no real Qwen3-4B forward — see Interface).
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
| `ImageWAMThorSpec`, `BACKBONE_LAYER_SHAPES`, `ACTION_DIT_LAYER_SHAPES`, `iter_expected_shapes` | `flash_rt/frontends/torch/_imagewam_thor_spec.py` |
| `make_imagewam_attention_spec`, `kernel="mot_joint"` dispatch | `flash_rt/hardware/thor/attn_backend.py` |
| `imagewam_encode_once`, `imagewam_prefill`, `imagewam_denoise_step` | `flash_rt/models/imagewam/pipeline_thor.py` |
| `ImageWAMTorchFrontendThor` | `flash_rt/frontends/torch/imagewam_thor.py` |

## State

| State | File |
|---|---|
| Weight buffers, KV cache buffer, cached text context, CUDA Graph object | `flash_rt/frontends/torch/imagewam_thor.py` (`ImageWAMTorchFrontendThor`) |

# Implementation

## Phase 1 — Weight declaration with a random-init path

Phase Status: completed

### Goal

Declare every weight tensor FLUX.2-4B-shaped ImageWAM needs (backbone,
ActionDiT — not the Qwen3-4B text encoder, see Structure), with a
random-fill path in place of a state-dict lookup.

### Files

`flash_rt/frontends/torch/_imagewam_thor_spec.py` (new).

### Structures

Modeled on `flash_rt/frontends/torch/_cosmos3_edge_thor_spec.py` (a
real, production spec for another diffusion-transformer model, not the
generic `_template/weights_spec.py`) rather than the dataclass-list or
loader-dict shapes earlier drafts of this plan assumed. That file
separates shape declaration from data loading cleanly: a frozen
dataclass of real dimensions, `GLOBAL_SHAPES`/`LAYER_SHAPES` dicts
mapping tensor name to shape (no loader callables, no checkpoint
reading at all in the spec file itself), and an `iter_expected_shapes()`
generator. Checkpoint loading and shape validation
(`load_transformer_weight_map`/`validate_transformer_shapes`) live in
the same file but are separate functions, not required for this plan.

This plan's `_imagewam_thor_spec.py` follows the same split:
`ImageWAMThorSpec` (frozen dataclass), `BACKBONE_LAYER_SHAPES`/
`ACTION_DIT_LAYER_SHAPES` dicts, `iter_expected_shapes()`. The
random-init path needs no new mechanism on the spec side at all — the
frontend (Phase 5) fills every shape from `iter_expected_shapes()`
with random data instead of calling a checkpoint loader function; the
spec file is identical either way.

Real dimensions (architecture configs only — no weight download,
consistent with the random-init goal):

FLUX.2-klein-4B backbone (`transformer/config.json`, upstream
`black-forest-labs/FLUX.2-klein-4B`, not vendored in this repository):
`num_attention_heads=24`, `attention_head_dim=128`, `num_layers=5`
(double-stream), `num_single_layers=20`, `in_channels=128`,
`patch_size=1`, `mlp_ratio=3.0`, `joint_attention_dim=7680`.
`joint_attention_dim` is 3x Qwen3-4B's own `hidden_size` (2560) —
likely a multi-layer concatenation of Qwen3 hidden states used as text
conditioning; confirming the exact construction is part of this
phase's work, not assumed here.

Qwen3-4B is not part of this weight declaration. Every ImageWAM model
config, including the FLUX.2-4B one, sets `load_text_encoder: false`:
ImageWAM itself never loads or runs the text encoder as part of the
model — `infer_action_flux2` always receives a precomputed `context`/
`context_mask`. This integration follows the same contract: `context`
is an INPUT buffer (`[qwen_context_len=512, joint_attention_dim=7680]`,
per `configs/model/imagewam_flux2_klein_4b_base.yaml`), not a weight,
and is random-filled the same way any other input is for this plan —
no Qwen3-4B weight (hidden_size=2560, 36 layers, etc.) is declared or
loaded anywhere in this integration.

ActionDiT, FLUX.2 variant (same YAML): `hidden_dim=1024`,
`num_heads=24`, `attn_head_dim=128`, `num_layers_double=5`,
`num_layers_single=20`, `mlp_ratio=4.0`, `max_action_horizon=64`.
`num_heads`/`attn_head_dim` match the FLUX.2 backbone exactly — this
is what makes the `mot_joint` attention (Phase 2) valid: the two
experts keep separate residual widths (1024 vs. 24x128=3072) but
share the same per-head attention geometry. `action_dim` is resolved
from the training data config at runtime (`${data.train.processor.action_output_dim}`);
this plan picks a placeholder LIBERO-shaped value if the real one
cannot be resolved from `configs/` alone.

### Affected Modules

ImageWAM weight declaration only.

### Observation

`iter_expected_shapes()` enumerates every declared tensor; a
standalone check random-fills each one and confirms the total
allocated byte count matches the sum of declared shapes/dtypes. This
runs on plain numpy, no GPU or compiled `flash_rt_kernels` extension
required — the actual GPU upload happens in Phase 5.

Ran: 145 tensors declared, no duplicate keys, every random-filled
array's shape matches its declaration. Total ~3.47B parameters
(backbone ~2.95B across 5 double-stream + 20 single-stream layers,
ActionDiT ~525M across the same layer split at `action_hidden_dim`
width) — the right order of magnitude for the real "FLUX.2-4B"
checkpoint's stated size, a sanity check that the declared shapes are
architecturally plausible, not a claim of exact match with the real
checkpoint (see this file's own docstring on tensor-name/shape
confidence).

## Phase 2 — Joint attention kernel

Phase Status: completed

### Goal

A `mot_joint` attention kernel implementing
`mot.py::_mixed_attention`'s concatenated-QKV joint softmax over the
`[prefix | target-image | action]` mask described in Interface.

Refined during implementation: the mask only depends on TWO real
boundaries, `x0` and `a0` — `t0`/`r0` (the text/ref internal split)
never affect attention visibility, since prefix (text+ref together)
is always visible or invisible as one unit. `state_masked` (Pi0's own
existing masked kernel) turned out to be the closer real precedent
than assumed: reading `csrc/kernels/attention_cublas.cu` directly
showed Thor's masked-attention kernels are cuBLAS QK^T -> a small
fused mask+softmax kernel -> cuBLAS PV, not a hand-fused
flash-attention-style kernel from scratch. `mot_joint` follows the
exact same three-step shape; only the masking rule inside the fused
softmax kernel differs (three row-groups instead of one threshold,
with the action group's visibility being two disjoint ranges instead
of one contiguous range).

### Files

- `csrc/kernels/softmax.cu`/`.cuh` — new `softmax_mot_joint_fp16`,
  modeled directly on the existing `softmax_state_masked_fp16`.
- `csrc/kernels/attention_cublas.cu`/`.cuh` — new
  `attention_qkv_fp16_mot_joint` (QK^T -> masked softmax -> PV),
  modeled directly on `attention_qkv_fp16_state_masked`.
- `csrc/bindings.cpp` — pybind entry `attention_qkv_fp16_mot_joint`.
- `flash_rt/hardware/thor/attn_backend.py` — `run()` gains `x0`/`a0`
  keyword arguments and a `kernel == "mot_joint"` dispatch branch;
  new `make_imagewam_attention_spec()` — two sites, `"backbone"`
  (`kernel="standard"`) and `"mot"` (`kernel="mot_joint"`), see the
  Structures correction below for why one site was not enough; both
  25 layers (`backbone_num_layers_double + backbone_num_layers_single`),
  `num_q_heads=24`/`head_dim=128` shared with both experts.

### Structures

New `SiteSpec.kernel` value `"mot_joint"`; `run()`'s new `x0`/`a0`
keyword arguments follow the exact same optional-scalar pattern
`state_nk` already uses — no new argument shape for this backend.
The two experts' Q/K/V are combined into ONE buffer by the pipeline
forward (Phase 3/4) before calling `run()`; this site's own slot
allocation just needs to be sized for the combined sequence length
(`max_q_seq` already supports this — no new slot-allocation mechanism
needed, resolving a complexity this plan's Interface section had
flagged as an open question).

**Correction (found while starting Phase 3): TWO attention sites are
needed, not one.** Re-tracing `infer_action_flux2` closely:
`self.video_expert.pre_dit(...)` runs ONCE, before the denoise loop,
and IS the backbone's own 25-layer forward — self-attention over just
its own `[prefix | target-image]` tokens (action tokens do not exist
yet at that point, so `mot_joint`'s three-region mask does not apply
there at all). Only `mot.prefill_flux2_video_cache`'s OUTPUT (that
forward's own K/V) feeds the LATER `mot.forward_action_with_video_cache`
calls inside the denoise loop, where `mot_joint` is actually used.
`make_imagewam_attention_spec()` now declares two sites: `"backbone"`
(`kernel="standard"`, `imagewam_prefill`'s own self-attention) and
`"mot"` (`kernel="mot_joint"`, `imagewam_denoise_step`'s joint
attention against the backbone's cached K/V). Both share
`num_q_heads=24`/`head_dim=128`; `num_layers=25` on both since every
backbone layer has a corresponding action-expert layer at the same
index.

### Affected Modules

ImageWAM attention declaration only.

### Observation

Ran a standalone test (`test_mot_joint.py`) comparing the new kernel
against a plain-PyTorch `F.scaled_dot_product_attention` reference
with an explicit boolean mask matching the same three-region rule, at
ImageWAM's real per-head geometry (`NH=24`, `HD=128`), on a small
(`total=24`: 8 prefix + 8 target-image + 8 action tokens) but complete
example covering all three row-groups including the non-contiguous
action-row case. Built `flash_rt_kernels` locally (Ada sm_89, this
machine's own GPU — correctness only, not a Thor timing claim) via
`cmake -B build -S . -DGPU_ARCH=89 -DFA2_ARCH_NATIVE_ONLY=ON` (see
`PROJECT.md` for the exact build fix needed: the system pybind11 was
too old for this venv's Python 3.11). Result: `cosine=1.000000,
rel_l2=0.000441` — the small residual is fp16-rounding-scale, not a
correctness gap. This checks the kernel against its own mathematical
definition; it is not a calibration or accuracy check against a
trained model, and stays in scope even though calibration itself does
not.

Not yet verified: the `>=1024`-column ceiling on this single-warp-per-
row softmax kernel style, against ImageWAM's real total sequence
length at a real deployment image resolution (see
`softmax_mot_joint_fp16`'s own docstring) — recorded as an open item,
not blocking this phase's own completion.

**Correction (found while starting Phase 3): `run()`'s `mot_joint`
branch was added to the wrong class and was unreachable dead code.**
`docs/adding_new_model.md` says to "extend the dispatch branches in
`ThorFlashAttnBackend.run`" for a new kernel value, which is what the
first version of this phase did. But `ThorFlashAttnBackend`'s own
constructor unconditionally rejects any site set other than Pi0.5's
fixed `{"siglip", "encoder", "decoder"}` (its module docstring says so
explicitly: "Currently supports Pi0.5's three sites... Pi0/GROOT out
of scope for Stage 1") — so a `ThorFlashAttnBackend` instance can never
actually be constructed for ImageWAM's `{"backbone", "mot"}` sites, and
the added branch could never run. Fixed by removing that branch from
`ThorFlashAttnBackend.run()` (reverting it to `'standard'`/
`'state_masked'` only, Pi0.5 untouched) and adding a new, separate
`ImageWAMAttnBackend(AttentionBackendBase)` class in the same file,
implementing the same `get_slot_ptrs`/`run()` protocol for just
`"backbone"`/`"mot"`. This is deliberately a standalone class rather
than a generalization of `ThorFlashAttnBackend`'s own site validation
— touching Pi0.5's already-shipped, feature-heavy class (FA4 wiring,
fixed-shape state-prompt masking, siglip-specific checks, none of
which ImageWAM needs) to serve a second, unrelated model would be a
large, unnecessary risk for a personal fork not going upstream.

Also discovered while investigating this: FlashRT already has an
unrelated, RTX-only, G1-stage model called **Motus**
(`flash_rt/models/motus/`, `flash_rt/hardware/rtx/attn_backend_motus.py`)
that also does joint video+action+text attention and happens to name
its own *site* `"mot_joint"` — a naming coincidence with this plan's
*kernel value* `"mot_joint"`, not a functional overlap (different
files, different hardware, different mask: Motus's own `"mot_joint"`
site is a full, unmasked joint MHA per its own docstring, unlike
ImageWAM's three-region masked attention). No code changed as a result
of this check; noted here only so a future reader searching for
`"mot_joint"` does not conflate the two.

Verified the fix with a new wiring smoke test,
`tests/test_imagewam_attn_backend.py`: constructs `ImageWAMAttnBackend`
for both sites with real per-layer K/V pointer arithmetic, runs
`"backbone"` (`kernel="standard"`) and `"mot"` (`kernel="mot_joint"`,
`x0=4, a0=8, total=12`) end-to-end, asserts finite output. This is a
plumbing check, not a re-verification of attention math (already done
above against the PyTorch reference).

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
