# Plan

Plan Status: completed

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

Phase Status: completed

### Goal

`imagewam_encode_once` and `imagewam_prefill`, producing a populated
KV cache buffer from random weights without NaN/Inf.

### Files

- `flash_rt/models/imagewam/pipeline_thor.py` (new) — `imagewam_encode_once`
  (explicit no-op, see its own docstring: the real encode step is a VAE
  forward, out of scope, no VAE weights declared), `imagewam_prefill`
  (5 double-stream + 20 single-stream layers), and private
  `_double_stream_layer`/`_single_stream_layer` helpers.
- `flash_rt/models/imagewam/__init__.py` (new).
- `flash_rt/hardware/thor/attn_backend.py` — added an even-`kv_seq`
  guard to `ImageWAMAttnBackend.run()`'s `"standard"` branch (see
  Structures correction below).
- `tests/test_imagewam_prefill.py` (new).

### Structures

Consumes Phase 1's declared shapes and Phase 2's `ImageWAMAttnBackend`,
but **not** Phase 1's checkpoint-shaped `WEIGHT_SPEC` tensors directly.
`pipeline_thor.py`'s own docstring defines a second, pipeline-facing
weight-key convention (already-split Q/K/V, already transposed to
`(K,N)` for `GemmRunner.fp16_nn`) that Phase 5's frontend must produce
— splitting a fused checkpoint tensor and transposing it are one-time
operations that belong at weight-load time, not inside a
graph-capturable forward (same precedent as `CosmosEdgeThor.__init__`'s
own `.t().contiguous()`).

Three real corrections, found by running the phase's own test, not by
inspection:

1. **`GemmRunner` must be constructed once, outside the forward.** An
   earlier draft created a fresh `fvk.GemmRunner()` inside each
   per-layer helper (25 times per `imagewam_prefill` call). Besides
   being wasteful, `GemmRunner()` does `cudaMalloc` for a 256MB
   workspace at construction — doing that repeatedly inside what must
   become a CUDA-graph-capturable region is invalid, and it produced a
   real `cublasLtMatmul` internal error at runtime on this machine's
   8GB GPU. Fixed by adding `gemm` as an explicit parameter to
   `imagewam_encode_once`/`imagewam_prefill` (matching
   `docs/adding_new_model.md`'s own pointer-interface contract example,
   which threads `gemm: fvk.GemmRunner` the same way) — constructed
   once by the caller.

2. **`attention_qkv_fp16` (the `"standard"` kernel) requires an even
   `kv_seq`.** Its softmax (`softmax_fp16_kernel`,
   `csrc/kernels/softmax.cu`) reinterprets each logits row as
   `__half2` with no internal even-padding, unlike `mot_joint`/
   `state_masked` (both compute their own `*_pad = n + (n & 1)`). An
   odd `kv_seq` makes odd-indexed rows start at a 2-byte-, not 4-byte-,
   aligned address — a real CUDA "misaligned address" crash, not a
   FlashRT bug specific to this project (Pi0.5's own `"standard"`
   sites apparently never hit it because their `enc_seq_max` happens to
   always be even in practice). Fixed by adding an explicit check in
   `ImageWAMAttnBackend.run()`'s `"standard"` branch rather than
   silently relying on the caller to know this; `a0` (the "backbone"
   site's `kv_seq`) must be even.

3. **Zero normalization anywhere caused real NaN/Inf**, not a
   pointer-interface bug. Random-weight GEMM chains without any
   normalization overflow FP16 within a handful of unnormalized
   residual layers, independent of random-vs-real weights. Fixed by
   calling `fvk.rms_norm_fp16` before every attention and MLP
   sub-block, passing a shared all-ones "weight" buffer
   (`bufs["norm_ones"]`) instead of a learned scale — real
   normalization with an always-1.0 elementwise gain, since Phase 1
   declared no learnable norm weights. This is a *better* description
   of this phase's own scope than the original plan's "no normalization
   at all," not a new limitation: `pipeline_thor.py`'s docstring
   simplification list was rewritten accordingly (unweighted RMS norm
   only, no AdaLN modulation — AdaLN is still not modeled).

Also documented (not fixed, tracked in `opportunities.md`): the
`attention_qkv_fp16`/`attention_qkv_fp16_mot_joint` kernels this
pipeline calls take K/V as a single `(seq, HD)` buffer broadcast across
all `NH` query heads (confirmed in Phase 2), not real per-head MHA —
this pipeline's own K/V projection weights are declared at `HD` width,
not `hidden` width, a genuine architectural simplification versus real
FLUX/DiT attention, separate from the random-vs-real-weight difference.

### Affected Modules

ImageWAM pipeline forward only.

### Observation

Ran `tests/test_imagewam_prefill.py` (small dims: `hidden=96, HD=16,
NH=6, mlp_hidden=192, joint_attention_dim=64, x0=4, a0=8`, 2
double-stream + 3 single-stream layers — not real FLUX.2-4B size, this
only checks the wiring) with fully random FP16 weights and inputs.
Result: `imagewam_prefill` runs to completion, `backbone_hidden` and
the per-layer K/V cache are all finite, and the K/V cache is
confirmed actually written (nonzero). Also fixed a real bug in the
test itself while getting here: several buffers were allocated as
bare `torch.zeros(...).data_ptr()` expressions with no surviving
Python reference — PyTorch's caching allocator is free to reuse that
memory for the next allocation the instant the tensor's refcount hits
zero, silently corrupting an already-stored pointer. Fixed by keeping
every buffer/weight tensor alive in a `_keepalive` list for the test's
duration.

## Phase 4 — Denoise loop and CUDA Graph capture

Phase Status: completed

### Goal

`imagewam_denoise_step`, called in a loop whose body is captured once
as a single CUDA Graph, replayed on every `infer` call.

### Files

- `flash_rt/models/imagewam/pipeline_thor.py` — `_action_double_layer`,
  `_action_single_layer` (ActionDiT's own 5+20 layer split, structurally
  identical to the backbone's "img"/"single" halves, single-stream
  only), `imagewam_denoise_step` (one Euler step), `imagewam_denoise_loop`
  (the whole N-step loop — what Phase 5's frontend captures as one
  CUDA Graph, not this phase itself: no `torch.cuda.graph(...)` call
  lives in `pipeline_thor.py`, matching every other file split in this
  codebase between compute (`models/`) and IO/graph-capture (`frontends/`)).
- `tests/test_imagewam_denoise.py` (new).

### Structures

**Correction: `static_engine.py`'s `EdgeStaticBufferEngine` is not the
capture/replay pattern** — an earlier draft of this plan named it as
such without having read `models/cosmos3_edge/pipeline_thor.py` in
full. `EdgeStaticBufferEngine` is a one-time bring-up/reference helper
used *inside* `CosmosEdgeThor.__init__`; the real capture-once/
replay-many pattern is `CosmosEdgeThor.capture()`/`.denoise()`
(`self.graph = torch.cuda.CUDAGraph(); with torch.cuda.graph(self.graph):
self.run_loop()`, then `denoise()` does one `self.graph.replay()`
covering the whole N-step loop). This plan reuses that real pattern:
`imagewam_denoise_loop` is this project's `run_loop()` analog (called
once per capture, called via `.replay()` thereafter by Phase 5's
frontend) — but as a free pointer-interface function here rather than
a method, since Phase 5 (unlike cosmos3_edge) keeps compute and
IO/graph-capture in separate files per this codebase's own file-split
rule (`docs/adding_new_model.md` §0).

Also corrected: **the per-step timestep is a Python int, not a
device-side scalar** — this plan's own Interface/Flow sections had
assumed a device-scalar-before-each-replay pattern without verifying
it against real code. Reading `CosmosEdgeThor.run_loop()`/`capture()`
in full shows the opposite: the entire N-step loop, `step` included, is
captured as ONE graph with `step` unrolled at *capture* time into N
constant-indexed kernel sequences (`self.t_emb[step:step+1]`), not read
from a device scalar written before each replay. This plan's own
`dt` follows the same shape: a fixed uniform `1.0 / num_denoise_steps`
Python float in `dims`, not a buffer.

**ActionDiT's own attention-facing width differs from its residual
width** — found by running this phase's own test with deliberately
different `action_hidden_dim`/`hidden` values (32 vs 96) after an
initial version silently used `action_hidden_dim` for both and passed
anyway with a same-valued placeholder. `_imagewam_thor_spec.py`
(Phase 1) already declares `action_attn_width` as a *distinct*,
computed property precisely equal to `backbone_hidden` for
`mot_joint` validity, but the first draft of this phase's pipeline code
used `action_hidden_dim` for the Q/proj GEMM widths (and hence for the
row width of the shared `Q_O`/K/V-cache offset arithmetic) instead —
wrong for any case where `action_hidden_dim != action_attn_width` (true
for the real model: 1024 vs 3072). Fixed by adding `dims["action_attn_width"]`
and threading it through `_action_double_layer`/`_action_single_layer`'s
own Q/proj GEMMs and pointer offsets, keeping `action_hidden_dim` scoped
to only the residual/MLP width (`q`/`k`/`v` input dim, `mlp0`/`mlp_in`
input dim, `mlp2`/`mlp_down` output dim) and `action_attn_width` scoped
to only the attention-facing GEMMs (`q`/`proj` output/input width, the
`Q_O` row-offset arithmetic).

**Per-step attention correctness relies on row-independence, not
correct backbone-row Q.** Only the action rows have a live query during
a denoise step (the backbone/image rows' own Q was already consumed
during prefill and is never read again) — `_action_*_layer` writes
fresh Q/K/V into rows `[a0, total)` of the shared `Q_O`/K/V-cache every
step but leaves rows `[0, a0)` of `Q_O` exactly as prefill last wrote
them (stale/meaningless). `attn.run("mot", ...)` still computes
attention over the WHOLE `total` sequence — wasted compute for the
backbone/image rows (tracked as part of OPT-002), but not a correctness
bug: attention is row-independent (each row's output depends only on
its own Q and the shared K/V), so a stale Q on a row whose output is
never read cannot corrupt the action rows' own, correctly-computed
output.

**ActionDiT has no declared output-projection head.** Phase 1 declares
only q/k/v/proj/mlp0/mlp2 and linear1/linear2, ending at
`action_hidden_dim` width, not a real (typically much smaller) action
dimension. This phase treats ActionDiT's own final hidden state as the
velocity directly (`bufs["action_hidden"]`, same shape as
`bufs["action_latent"]`) — a documented placeholder, not a bug,
grouped with OPT-001/OPT-002 as real-checkpoint-dependent follow-up
work.

The flow-matching Euler update itself uses two already-existing,
previously-unused-by-this-plan kernels rather than a new one:
`fvk.gpu_cast_fp32_to_fp16` (seeds `action_hidden` from the F32
`action_latent` at the start of each step) and `fvk.gpu_euler_step`
(`actions[i] += dt * velocity[i]`, F32 actions / FP16 velocity — the
same real convention Pi0's own diffusion decoder uses, confirmed by
its call-site placement in `csrc/bindings.cpp` next to Pi0-specific
kernels). `action_latent` is F32 for this reason, not FP16 like every
other buffer in this pipeline.

### Affected Modules

ImageWAM pipeline forward only.

### Observation

Ran `tests/test_imagewam_denoise.py`: extends Phase 3's test setup
(same small dims) with ActionDiT weights (`action_hidden_dim=32`,
`action_attn_width=96=hidden`, `action_mlp_hidden=64`, 3 action
tokens, `total=11`), runs a real `imagewam_prefill` first, then
`imagewam_denoise_loop` over 2 steps. Result: `action_latent` stays
finite and is confirmed actually advanced (not equal to its
pre-loop value); the backbone's own K/V cache rows `[0, a0)` remain
finite and are not the ones the denoise loop rewrites, checked
explicitly. This is a wiring check (no NaN/Inf, correct shapes, the
loop actually mutates state) — not a Thor performance measurement
(CUDA Graph capture itself belongs to Phase 5's frontend, not
`pipeline_thor.py`) and not accuracy against a trained model.
over repeated replays is measured and recorded.

## Phase 5 — Frontend and text-context caching

Phase Status: completed

### Goal

`ImageWAMTorchFrontendThor`, wiring Phases 1-4 into `set_prompt()` and
`infer()`.

### Files

- `flash_rt/frontends/torch/imagewam_thor.py` (new).
- `tests/test_imagewam_frontend.py` (new).

### Structures

Owns the state listed in Structure's State Ownership table (weight
buffers, KV cache buffer, cached text context, CUDA Graph object), as
already assigned. One structural deviation from the generic template,
made deliberately rather than found as a bug: buffer allocation uses
plain `torch.cuda.Tensor` + `.data_ptr()` throughout, not
`_template/frontend.py`'s `CudaBuffer` ctypes wrapper — matching the
real, working `CosmosEdgeThor` precedent and everything this plan's
own Phases 2-4 have already used, rather than introducing a second,
inconsistent buffer-ownership convention into the same project.

The whole `imagewam_prefill` + `imagewam_denoise_loop` sequence is
captured as ONE CUDA Graph in `_capture_graph()` (two warm-up calls on
a side stream, then one more inside `torch.cuda.graph(...)`), following
`CosmosEdgeThor.capture()`'s real pattern (Phase 4's own corrected
understanding of it) rather than the generic template's separate
encoder/decoder capture. `set_prompt()` triggers capture on the first
call only (`context_mask`, per its own module-docstring note, is
accepted but never consumed by the pipeline; `context` is random-filled
in place — a real, in-place `.normal_()` write to a buffer whose
address the graph already captured, not a reallocation); `infer()`
random-fills the observation-derived rows of `backbone_hidden` and a
fresh noise seed into `action_latent`, then replays.

### Affected Modules

ImageWAM frontend only.

### Observation

Ran `tests/test_imagewam_frontend.py`: `set_prompt()` once, a second
`set_prompt()` call with the same prompt string confirmed to skip
recapture (same graph object), then 5x `infer()` with varying random
observations. Result: every call returns a finite `(num_action,
action_hidden_dim)` actions array, no two consecutive calls return
identical output (confirming graph replay actually reads the
freshly-written buffers rather than stale ones from capture time), P50
latency 0.77ms — this machine's own small structural-dry-run dims
(`_DEFAULT_DIMS`), not a Thor number and not comparable to a real
FLUX.2-4B-scale latency.

## Plan Completion

All 5 phases completed. `imagewam_prefill` + `imagewam_denoise_loop`
run end-to-end through `ImageWAMTorchFrontendThor.set_prompt()`/
`infer()`, captured as one CUDA Graph, on random weights at a small
structural-dry-run scale, verified on this project's own Ada (sm_89)
GPU. What this does NOT establish, tracked in `opportunities.md`:
accuracy against a trained checkpoint (OPT-001), real per-head K/V
attention (OPT-002), Thor-specific (sm_110) performance or even
correctness at Thor scale (this machine has no Thor hardware), and
real-resolution sequence lengths against the `softmax_mot_joint_fp16`
`>=1024`-column ceiling flagged in Phase 2. This plan's own stated
goal — extend FlashRT's Thor pipeline machinery to cover ImageWAM's
structural shape with random weights, deferring precision — is met.

## Ada (sm_89) Steady-State Speed (this machine, not Thor)

`benchmarks/imagewam_thor_bench.py`: per-layer-type steady-state
latency (CUDA-event timed, 15 warmup + 50 measured iterations, P50) at
ImageWAM's REAL confirmed per-head geometry (`hidden=3072, HD=128,
NH=24, mlp_hidden=9216` backbone; `action_hidden_dim=1024,
action_attn_width=3072, action_mlp_hidden=4096` ActionDiT — from
`_imagewam_thor_spec.py`), at a representative (not confirmed-real)
sequence length `x0=128, a0=896, num_action=64 (=max_action_horizon),
total=960` — kept under `softmax_mot_joint_fp16`'s confirmed
`SM_MAX_COLS=1024` ceiling (`csrc/kernels/softmax.cu`) so the number
reflects real work, not silently-truncated work.

Deliberately benchmarks one layer of each type in isolation (1-layer
`AttentionSpec`, called repeatedly at `layer_idx=0`) rather than
allocating the full 25-layer weight set (~5.5GB of random FP16 weights
at real dims) — this machine has ~6.7GB free on an 8GB shared laptop
GPU. Whole-pipeline totals below are `layer_count × per-layer P50`,
not one single measured run — valid because every layer of a given
type has identical shapes and therefore identical steady-state cost.

| Component | P50 |
|---|---|
| backbone double-stream layer | 6.26 ms |
| backbone single-stream layer | 5.79 ms |
| ActionDiT double-stream layer | 1.07 ms |
| ActionDiT single-stream layer | 1.07 ms |
| `mot_joint` kernel alone (no GEMMs) | 0.87 ms |
| `standard` attn kernel alone (no GEMMs) | 0.74 ms |
| **backbone prefill total** (5 double + 20 single) | **147.1 ms** |
| **one ActionDiT denoise step** (5 double + 20 single) | **26.7 ms** |
| prefill + 1-step denoise | 173.8 ms |
| prefill + 4-step denoise | 253.9 ms |
| prefill + 10-step denoise | 414.1 ms |

Not a Thor number (this machine's own Ada sm_89 GPU) and not an
accuracy claim (random weights). The per-layer numbers are dominated
by the large GEMMs (backbone `mlp0`/`mlp2` alone are `896×9216×3072`
each way), not by the attention kernels themselves — the standalone
kernel timings above are roughly an order of magnitude below a full
layer's own time, consistent with that. `attn.run` inside a real layer
also carries the `get_slot_ptrs`/dispatch Python overhead this
micro-benchmark's own direct kernel calls skip, so the "kernel only"
row is a lower bound, not the attention share of a real layer.

## FP4 (NVFP4) Full-Scale Benchmark — Written for Thor, Not Run Here

`benchmarks/imagewam_thor_fp4_bench.py`: the user asked to actually run
FP4 (not derive it from a BF16 estimate) and see full-scale (all 25
backbone + 25 ActionDiT layers) steady-state latency. Two real,
hardware-level facts made that impossible on this dev machine, found
before writing anything (not attempted-then-failed):

1. **FP4 tensor cores are Blackwell-only.** `CMakeLists.txt` gates
   `ENABLE_NVFP4` (SM120, RTX 50-series) and
   `ENABLE_CUTLASS_SM100_NVFP4_W4A16` (SM100/Thor SM110) both strictly
   on `GPU_ARCH`, printing `"DISABLED (requires Blackwell
   sm_120a/sm_121a, current: sm_${GPU_ARCH})"` for anything else. This
   machine is Ada (sm_89) — no rebuild unlocks this, the hardware has
   no FP4 tensor core unit.
2. **Full 25-layer BF16/FP16 was also judged too risky**: ~6.95GB of
   weights alone (see the estimate above) against ~6.7GB actually free
   on this 8GB shared laptop GPU.

The user chose (via `AskUserQuestion`): write the FP4 benchmark code
for them to run on Thor, rather than attempt BF16 here or stop.

**Written from real, existing FlashRT code, not guessed**:
`flash_rt.executors.fp4_utils` (documented as the "Pi0.5 FP4 frontend,
Phase 4.3" NVFP4 wrapper — `quant_weight_nvfp4`, `FP4ActScratch`,
`quant_act_nvfp4`, `fp4_gemm`) and the real Thor SM100 NVFP4 W4A16
GEMM build block (`csrc/gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm100.cu`,
gated on `GPU_ARCH STREQUAL "110"` independent of the SM120 flag,
already production-used by `qwen36_thor.py`). `fp4_gemm`'s own output
is FP16 already, so FP4 GEMMs write directly into the same
`Q_O`/`K_cache`/`V_cache` buffers this plan's real (FP16) attention
kernels already expect — no BF16↔FP16 cast layer anywhere. Attention
itself is completely unchanged (still the FP16 kernels from Phase 2-4,
OPT-002's broadcast-K/V simplification still applies, the 1024-column
softmax ceiling still applies and the same `total=960` sequence choice
avoids it).

**Verified the ORCHESTRATION, not the FP4 kernels themselves**: wrote a
throwaway dry run (not committed) substituting plain `fp16_nn` for the
real FP4 GEMM inside `_Fp4Linear.__call__`, keeping every loop, weight
lookup, and pointer-offset call identical. It ran clean on this Ada
machine and reproduced the earlier per-layer-extrapolated FP16 numbers
almost exactly (backbone 145.8ms vs. the 147.1ms extrapolation above;
one denoise step 27.8ms vs. 26.7ms) — real cross-validation that the
25+25-layer loop structure, weight dict keys, and attention dispatch
are correct. What this does NOT validate: the actual `fp4_gemm`/
`quant_weight_nvfp4`/`quant_act_nvfp4` calls' argument shapes, dtypes,
or the `alpha=1.0`/`variant_idx=-1` placeholders against the real
compiled Blackwell kernels — genuinely untestable without Blackwell
hardware. The script's own docstring says this plainly: "UNTESTED ON
REAL HARDWARE... has never executed successfully anywhere," and fails
with a clear, actionable message (not a raw traceback — confirmed on
this machine) if `flash_rt.flash_rt_fp4` isn't built in.

Also found and fixed while checking this: importing
`flash_rt.executors.fp4_utils` itself raises `ModuleNotFoundError`
without a Blackwell build (it does `import flash_rt.flash_rt_fp4` at
module level unconditionally) — confirmed directly on this machine.
The script's import guard covers both imports in one `try/except`, not
just the more obvious `flash_rt.flash_rt_fp4` one.

## Real Thor (SM110) Results — First Real-Hardware Run

Run by the user on actual Jetson AGX Thor hardware, random weights (no
checkpoint download), `benchmarks/imagewam_thor_fp4_bench.py` for FP4
and the equivalent FP16/BF16/FP8 variants of the same graph-free
per-layer-type structure. Same dims as the Ada estimates above
(`hidden=3072`, 5 double + 20 single, `x0=128, a0=896, num_action=64,
total=960`). `has_cutlass_sm100=True`, `has_nvfp4=True` — the FP4 path
this plan wrote untested actually ran successfully on its first real
try.

| Precision | backbone prefill (25L) | one denoise step (25L) | prefill + 10-step |
|---|---|---|---|
| FP16 | 81.5 ms | 35.3 ms | 433.9 ms |
| BF16 | 85.5 ms | 36.8 ms | 452.9 ms |
| FP8 | 61.0 ms | 34.7 ms | 407.5 ms |
| FP4 (NVFP4) | 55.8 ms | 35.6 ms | 411.1 ms |

Notable, and diagnosable from this plan's own known design, not a
mystery:

- **Quantization only helps prefill, barely touches the denoise step.**
  FP4 prefill is 1.46x FP16; FP4 denoise step is statistically flat
  (35.6ms vs. FP16's 35.3ms). This is `attn.run("mot", ...)`'s own
  documented inefficiency (Phase 4's Structures section, OPT-002):
  every denoise step computes `mot_joint` attention over ALL `total=960`
  rows even though only the `num_action=64` action rows' output is ever
  read — `total*NH=23,040` query rows computed for `num_action*NH=1,536`
  useful ones, roughly **15x overcompute**. Quantizing the surrounding
  GEMMs cannot speed up a step that is attention-bound by construction;
  this real measurement is the first hard evidence of how much that
  costs, not just a theoretical concern.
- **BF16 is slower than FP16, consistently** (85.5 vs 81.5ms prefill;
  452.9 vs 433.9ms full). This pipeline's attention kernels
  (`attention_qkv_fp16`/`attention_qkv_fp16_mot_joint`) are FP16-only
  (`__half`) — a BF16 GEMM path would need a cast at the attention
  boundary that a pure-FP16 path skips. (This benchmark's own FP4/FP8
  variants avoid this because their GEMM output is FP16 already, same
  as `fp4_gemm`'s documented contract used in the FP4 script.)
- **None of FlashRT's real kernel-fusion or autotuning machinery is in
  this pipeline yet** — see "What This Confirms" below.

### What This Confirms: No FlashRT "Core Optimizations" Are In This Path Yet

Direct answer to "是不是还没迁移flashRT的核心优化": correct, essentially
none are. This was in scope from the start as deferred, not an
oversight found now — Phase 1-5's own stated goal was wiring
correctness on random weights, explicitly deferring precision AND
performance work (`opportunities.md` OPT-001/OPT-002 already existed
before this real-hardware run; this run is the first hard evidence of
their actual cost, not a new discovery of missing scope). Concretely,
compared to a real FlashRT model like `cosmos3_edge`:

1. **No CUDA Graph capture in this benchmark** (the user's own note:
   "graph-free 路径"). Every kernel call pays Python/pybind11 dispatch
   overhead and a host-device round trip; a captured graph's `.replay()`
   has near-zero CPU overhead and lets the GPU execute back-to-back.
   Phase 5's `ImageWAMTorchFrontendThor` DOES capture a graph, but this
   speed benchmark (both the Ada script and its FP4 Thor counterpart)
   deliberately runs graph-free per-layer-type, to isolate per-layer
   cost — the real, graph-captured, whole-pipeline number is not yet
   measured on Thor.
2. **Zero kernel fusion.** Every double/single-stream block here is
   norm → GEMM → GEMM → GEMM (separate calls) → attention → GEMM →
   residual_add → norm → GEMM → gelu → GEMM → residual_add — one kernel
   launch per math op. Real FlashRT models fuse aggressively:
   `residual_add_rms_norm_fp8` (residual + norm + quantize in one
   launch), fused QKV projections (one wide GEMM instead of three),
   `bias_gate_mul_residual_bf16`. None of that exists in
   `pipeline_thor.py` yet — Phase 3/4's own goal was a correct 1:1
   translation of the math, not a fused one.
3. **`GemmRunner` never autotunes for these shapes.** `get_or_create_cached`
   only ever requests cuBLASLt's top-1 heuristic result
   (`cublasLtMatmulAlgoGetHeuristic(..., 1, &heuristic, ...)` in
   `csrc/gemm/gemm_runner.cu`) — it never calls the same file's own
   `autotune_cached` (multi-algorithm benchmark, used elsewhere in this
   codebase) for ImageWAM's specific shapes.
4. **Real per-head K/V attention (OPT-002) and real FP8/FP4 calibration
   (OPT-001) are still not implemented** — this run reused the same
   broadcast-K/V simplification and untuned `alpha=1.0`/auto-picked
   GEMM variant this plan already flagged as placeholders.

None of this is a regression from the plan's own stated scope — it is
exactly what "extend FlashRT's Thor pipeline machinery to ImageWAM's
structural shape... deferring precision" (and, implicitly, performance)
committed to delivering. The performance work is real, scoped,
concrete follow-up, not a vague "make it faster" — see OPT-003/OPT-004
below.

## Full-Pipeline FP16 — Real Local Confirmation, and a VRAM Estimate Correction

`benchmarks/imagewam_thor_fp16_bench.py` (new): the exact same 25+25
layer structure as `imagewam_thor_fp4_bench.py`, but plain `fp16_nn`
for every projection — fully run on this dev machine (Ada sm_89), not
just isolated-per-layer-extrapolated:

| | backbone prefill (25L) | one denoise step (25L) | prefill + 10-step |
|---|---|---|---|
| Ada, this machine | 152.1 ms | 29.5 ms | 447.8 ms |
| Thor, user-reported | 81.5 ms | 35.3 ms | 433.9 ms |

Close enough on the "full" number to be a useful cross-check (Thor
faster on the GEMM-heavy prefill as expected of newer hardware; the two
are closer on the attention-bound denoise step, consistent with
OPT-003's diagnosis that denoise cost is dominated by `mot_joint`'s own
overcompute rather than raw GEMM throughput, which would otherwise
scale more with hardware generation).

**Correction to the earlier VRAM estimate**: this successful local run
(real weight footprint ~5.56GB — recomputed from what actually gets
allocated: backbone 2.406B params + ActionDiT 0.374B params = 2.78B,
this pipeline's own reduced-K/V-width convention) is SMALLER than the
~6.95GB figure quoted earlier for "the declared subset" — that earlier
number used the CHECKPOINT-shaped (full K/V width) parameter count
(3.473B, from `_imagewam_thor_spec.py`'s own declared shapes), not what
this pipeline's own pointer-interface functions actually allocate.
Both numbers are real and correctly computed for what they each
describe — they just describe two different things (real-checkpoint-
compatible declaration vs. this project's own reduced-K/V pipeline
convention) that were not clearly distinguished when first quoted.

`benchmarks/imagewam_thor_fp8_bench.py` (new): identical structure,
`_Fp8Linear` in place of `_Fp16Linear`. NOT verified end-to-end here —
see "Ada FP8 Environment Gap" below — but its orchestration was
verified with the same fp16-substitution dry-run technique used for
FP4, reproducing near-identical numbers to the FP16 script (150.0ms /
28.8ms / 435.3ms) as expected (same loop structure, same weight
layout, only the linear op differs).

## Ada FP8 Environment Gap — Confirmed, Not This Project's Bug

Attempting `benchmarks/imagewam_thor_fp8_bench.py` for real on this
machine fails at the very first GEMM call (`txt_in`, after all 65+
weight matrices quantized successfully): `cublasLtMatmulAlgoGetHeuristic
failed with cuBLAS status 15` (`CUBLAS_STATUS_NOT_SUPPORTED`). Isolated
with `benchmarks/imagewam_gemm_precision_compare.py`: this venv's
cuBLASLt (12.8.04, CUDA 12.8, compute capability (8,9)) returns this
same failure for FP8 (E4M3) matmul at EVERY shape tried, including a
trivial 64x64x64, via two independent code paths
(`GemmRunner.fp8_nn_dev_fp16` and the standalone `fp8_gemm_descale_fp16`).
Ada Lovelace has real FP8 tensor core hardware in general — this is a
confirmed environment/library-version gap in this specific venv, not a
FlashRT bug or a hardware limitation: the user's own Thor run already
produced real FP8 numbers with the exact same `quantize_fp8_static_fp16`
+ `fp8_gemm_descale_fp16` API pair this project's FP8 script uses.

## GEMM-Only Precision Comparison, Including INT4 (Ada, Real Numbers)

`benchmarks/imagewam_gemm_precision_compare.py`: isolated GEMM-only
timing (no attention, no norm, no pipeline loop) at ImageWAM's real
backbone projection shapes, comparing plain `fp16_nn` against the
SEPARATE SM80-family CUTLASS INT8/INT4 rowwise GEMM path
(`csrc/gemm/cutlass_sm80_int4_rowwise.cu`, built for Jetson Orin SM87's
QuaRot path, `cutlass::arch::Sm80` — confirmed to build and run on Ada
sm_89 after reconfiguring with `-DENABLE_SM80_INT8_CUTLASS=ON
-DFLASHRT_ENABLE_CHAMELEON=ON` — see OPT-007). Warmup=20, iters=100,
CUDA-event P50, this machine:

| shape | fp16 | fp8 | int8 (SM80) | int4 (SM80) |
|---|---|---|---|---|
| q/proj [M=896,N=3072,K=3072] | 0.810ms | FAIL (env) | 0.213ms | 0.088ms |
| k/v [M=896,N=128,K=3072] | 0.045ms | FAIL (env) | 0.025ms | 0.015ms |
| mlp0 [M=896,N=9216,K=3072] | 1.937ms | FAIL (env) | 0.534ms | 0.288ms |
| mlp2 [M=896,N=3072,K=9216] | 1.872ms | FAIL (env) | FAIL (shape) | FAIL (shape) |

INT4/INT8 (where they work) are genuinely fast — ~9x and ~4x over
FP16 respectively at the `q/proj` shape — but the `mlp2` failure
(`K=9216`, non-zero CUTLASS return code, not a crash) is real and
unexplained (not investigated further this pass; recorded as OPT-007's
own open item). This is a GEMM-only measurement — no correctness claim
(random already-packed int4/int8 bytes, no QuaRot Hadamard rotation
applied, which this specific kernel's own file header says is required
for real activations to survive int4's dynamic range) and no full-
pipeline integration.

## Full-Pipeline INT4 (Ada, GEMM-only) — Real Steady-State Number

Follow-up to the above, once the user asked specifically for the
*whole inference pipeline's* INT4 steady state, not just isolated
GEMMs. `benchmarks/imagewam_thor_int4_bench.py` (new): same 25+25-layer
structure as the FP16/FP8/FP4 scripts, real weight packing, ran clean
end to end on this machine (the `mlp2`/`K=9216` failure recorded above
did NOT reproduce here — see OPT-007's correction: it turned out to be
a flaky artifact specific to `imagewam_gemm_precision_compare.py`'s own
mixed-precision-in-one-process sequence, not a hard `K` limit; separate
targeted repro attempts ruled out shape order, the preceding fp8/int8
failures, and the timing loop pattern as the cause — root cause
unresolved but the real full-pipeline run is unaffected by it):

| | backbone prefill (25L) | one denoise step (25L) | prefill + 10-step |
|---|---|---|---|
| INT4 (GEMM-only, this machine) | 42.4 ms | 24.2 ms | 283.3 ms |
| FP16 (this machine) | 152.1 ms | 29.5 ms | 447.8 ms |

**The critical caveat, found while trying to make this number honest
rather than just fast**: this is GEMM-only, with NO per-call activation
quantization — weights and activations are both random already-packed
int4 bytes, reused across every replay. Tried to include the real
quantization step (`fht_int4_quant_fp16`, the only real activation
quantizer for this specific QuaRot-family INT4 scheme) and it CRASHES
with an illegal memory access at ImageWAM's real hidden dims (3072,
9216, 7680 — confirmed one shape per fresh process to avoid the crash
corrupting further tests): it works cleanly at 128/1024/4096 (all
powers of 2) and crashes at 3072 (not a power of 2). This FHT kernel
needs a power-of-2 transform size; none of ImageWAM's real hidden
dimensions are powers of 2. So this number is a genuine "how fast could
the GEMMs be" signal, not a number a real INT4 deployment could
actually achieve without first resolving this — recorded in full in
`opportunities.md` OPT-007, which this finding meaningfully updates
(both the mlp2-flakiness correction and the FHT crash).
