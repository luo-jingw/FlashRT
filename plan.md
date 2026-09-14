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

## OPT-003 Fixed: mot_joint Attention Restricted to Action Queries

Following the priority order agreed after the "为什么慢" discussion:
OPT-003 (the biggest, best-understood win) implemented and verified
first, ahead of OPT-004/OPT-002/OPT-001.

New kernel pair (`attention_qkv_fp16_mot_joint_action` +
`softmax_mot_joint_action_fp16`, `csrc/kernels/attention_cublas.cu`/`.cuh`
and `softmax.cu`/`.cuh`): identical cuBLAS-composed structure to the
original `mot_joint` kernel, but Q covers only the `num_action` action
rows instead of the whole `total` sequence — K/V still cover `total`
(action rows attend into the frozen prefix K/V). Collapses the
softmax's three-row-group mask into one uniform rule per row, since
every remaining row is an action row. `ImageWAMAttnBackend.run()`'s
`"mot_joint"` branch now takes `q_seq=num_action` + explicit
`kv_seq=total` and computes the Q/output pointer offset internally
from `a0`; `pipeline_thor.py` and all four benchmark scripts updated
to the new call contract (two-line change each — the pointer
arithmetic the pipeline already did before/after the call did not need
to change, since the new kernel writes to the same offset).

Verified two ways (`tests/test_imagewam_mot_joint_action_kernel.py`):
against a PyTorch reference (cosine=1.000000), and bit-for-bit
equivalence with the ORIGINAL `mot_joint` kernel's own output for the
same action rows (cosine=1.000000) — the contract that actually
matters: zero behavior change for the rows anything reads, pure speed.
All existing tests still pass.

**Real measured speedup, this machine (Ada)**:

| | one denoise step (25L) | prefill + 10-step |
|---|---|---|
| FP16 before → after | 29.5 ms → **5.68 ms (5.2x)** | 447.8 ms → **203.2 ms (2.2x)** |
| INT4 (GEMM-only) before → after | 24.2 ms → **4.37 ms (5.5x)** | 283.3 ms → **91.1 ms (3.1x)** |

Full details, including the exact mechanism and remaining Thor
re-measurement gap, recorded in `opportunities.md` OPT-003 (now marked
RESOLVED on Ada; Thor confirmation still pending — the user's own
Thor-side FP16/BF16/FP8/FP4 numbers from before this fix are the
baseline to re-run against).

## OPT-004 Step 1: Graph Capture Measured — Compute-Bound, Not Launch-Bound (Ada)

`benchmarks/imagewam_thor_graph_bench.py` (new): built the real
`ImageWAMTorchFrontendThor` at real dims (post-OPT-003 fix), captured
its CUDA Graph, measured steady-state `infer()`.

| | prefill + 10-step denoise |
|---|---|
| Graph-free (post-OPT-003) | 203.2 ms |
| CUDA-graph-captured | **198.5 ms** (~2% faster) |

A real, somewhat unexpected finding: at these shapes, individual GEMMs
are large enough (hundreds of µs to a few ms each) that per-launch
dispatch overhead is a small fraction of the total — this pipeline is
solidly compute-bound, not launch-bound, on Ada. CUDA Graph capture is
real and already built (Phase 5) but is not where further gains are
hiding here. Redirects priority toward OPT-004's remaining steps (fuse
QKV into one wide GEMM, fuse residual+norm, `GemmRunner.autotune_cached`)
— real compute reduction, not launch-overhead elimination. NOT yet
re-confirmed on Thor, where the launch-vs-compute balance could differ
(faster GEMMs there could make it relatively MORE launch-bound, not
less) — recorded as an open item in `opportunities.md` OPT-004.

## OPT-004 Step 4: GemmRunner Autotuning — Real but Modest

`benchmarks/imagewam_thor_fp16_autotuned_bench.py` (new): each linear
op autotunes once (real cuBLASLt algorithm benchmarking, mutating the
same cache `fp16_nn` reads from — cannot regress correctness). Result:
backbone prefill 143.5ms → 138.8ms (~3%), full 203.2ms → 195.1ms (~4%).
Small because cuBLASLt's default heuristic already picks close to the
best available algorithm for these shapes on this hardware — most
shapes had only 1 candidate algorithm to begin with. Real, safe,
worth keeping, but not the big remaining lever. Full detail in
`opportunities.md` OPT-004.

The user went to sleep after asking for local-only follow-up work
(OPT-005/FA4 needs Thor, deferred). Per their own direction ("本机可以
用int4尝试替换"), the rest of this session's autonomous work explored
whether OPT-007's INT4 path could be extended (see the Hadamard-padding
probe recorded in `opportunities.md` OPT-004/OPT-007) rather than
continuing into OPT-005's Thor-only, locally-unverifiable territory.

Remaining, deferred until Thor access returns: OPT-005 (FA2/FA4 for
backbone), OPT-004 steps 2-3 (fuse QKV, fuse residual+norm — real
engineering work best not done blind without being able to verify
correctness against a running comparison), and re-confirming OPT-003's
graph-vs-graph-free and autotune findings on real Thor hardware.

## Real Thor (SM110) Results — Post-OPT-003 Run

The user ran the full checklist above on real Thor hardware. All 6
correctness tests PASS (`test_imagewam_mot_joint_kernel.py`,
`test_imagewam_mot_joint_action_kernel.py` — both sub-tests,
`test_imagewam_attn_backend.py`, `test_imagewam_prefill.py`,
`test_imagewam_denoise.py`, `test_imagewam_frontend.py`).

**Speed, Ada vs Thor (prefill / one denoise step / full 10-step), all
post-OPT-003**:

| | Ada | Thor | Thor vs Ada (full) |
|---|---|---|---|
| FP16, graph-free | 143.5 / 5.68 / 203.2 | 81.6 / 5.89 / 140.4 | 1.45x faster |
| FP16 + autotune | — / — / 195.1 (+4%) | 67.9 / 5.80 / 126.1 (+10%) | bigger relative gain on Thor |
| FP16, graph-captured | — / — / 198.5 (+2%) | — / — / 129.9 (+7.5%) | bigger relative gain on Thor, but autotune (126.1) beats it |
| FP8 | not runnable (env gap) | **60.0 / 4.65 / 106.6** | first real run — best full number |
| FP4 (NVFP4) | never run (no Blackwell) | 55.8 / 5.52 / 111.1 | first real run |
| INT4 (SM80 CUTLASS) | 45.2 / 4.31 / 88.0 (GEMM-only optimistic) | 721 / 49.4 / **1214** | **~8.6x SLOWER than FP16 on Thor** |

**OPT-003 confirmed on Thor, and is the dominant fix end to end.**
Before the fix, Thor's own denoise step was ~35.3ms (matching the
"Real Thor Results" baseline table above); after, 5.89ms — **6.0x**
(slightly better than Ada's 5.2x). Full 10-step: ~434ms → 140.4ms.
Query rows shrank 15x (960→64) but the measured speedup is only 6x
because the remaining denoise cost is GEMM-bound, not
attention-bound, once the attention overcompute is gone — matches the
mechanism exactly, no surprise. **Real consequence for where to look
next**: prefill is now 58% of the full path (was the minority share
before the fix) — the optimization center of gravity has moved from
denoise attention to backbone GEMM.

**Graph-vs-graph-free direction holds on Thor, but the gap is bigger
than Ada's (~2%) — 7.5% (140.4→129.9ms, saving 10.5ms)**. Faster GEMMs
on Thor do make launch overhead a somewhat bigger relative factor,
exactly the direction predicted when this was flagged as unconfirmed.
Still not the biggest lever, though: **autotune alone (126.1ms) already
beats graph-capture-with-default-heuristic (129.9ms)** — real evidence
that better GEMM algorithm selection matters more here than reducing
launch count, at least at this shape mix.

**FP8/FP4 now show a real, meaningful full-pipeline win, because
OPT-003 removed the fixed ~35ms/step floor that used to cap them
regardless of GEMM precision.** FP8 106.6ms (1.32x vs FP16's 140.4ms),
FP4 111.1ms (1.26x). FP8 edges out FP4 again, consistent with the
pre-OPT-003 baseline's own finding. Both only requantize the big
GEMMs; attention stays FP16 throughout (OPT-002 still open).

**INT4 (SM80 CUTLASS, Orin-targeted) is a real, clear negative result
on Thor — confirms the user's own standing caution not to use it
there.** The build (`-DENABLE_SM80_INT8_CUTLASS=ON
-DFLASHRT_ENABLE_CHAMELEON=ON`) compiles, links, and every GEMM shape
returns success (rc=0) — but "succeeds" only means no error thrown,
not numerically correct (untested) — and it is dramatically slower
than FP16 on Thor's own tensor cores (e.g. `q/proj`: 3.30ms INT4 vs
0.127ms FP16 — ~26x slower; full pipeline 1214ms vs 140ms — ~8.6x
slower). This SM80-templated kernel almost certainly falls back to a
compatibility path that does not use Thor's native (Blackwell)
tensor-core instructions at all. **Conclusion: Thor's own native NVFP4
(SM100) path is the right low-precision target there — the SM80
INT4/Orin path is a dead end on Thor, not a slower-but-usable
alternative.** The Thor build was reset back to the default slim
config (Chameleon/INT4 off) afterward specifically to avoid this slow
path being used by accident later.

**Hadamard-padding probe re-run on Thor**: `K=3072→4096` cosine=0.983
(matches Ada exactly). `K=7680→8192` cosine=0.977 this time (matches
Ada's very first exploratory run, not Ada's later — and reproducible —
failures at this size: the instability itself reproduces across
hardware, not just its specific symptom). `K=9216→16384` fails with a
different symptom again ("CUDA invalid argument" on Thor vs an
all-zero scale with no thrown error on Ada). Net: this kernel family's
real instability at large K is now confirmed on two different GPUs,
not an Ada-specific quirk — treat OPT-007's own INT4 path as unreliable
at K>4096 regardless of hardware, on top of it now also being
confirmed a poor fit for Thor's tensor cores specifically.

### What This Changes About Priority

1. **FP8 is currently the best real, verified full-pipeline number
   (106.6ms)** — and unlike before OPT-003, this is now a genuinely
   meaningful win over FP16, not one capped by a fixed attention floor.
   Getting FP8 (or FP4) real calibration and accuracy validation
   working (OPT-001) is now higher-value than it was pre-OPT-003, since
   there is a real speed prize waiting behind it.
2. **Autotune beating graph-capture on Thor** suggests the next
   concrete, low-risk step is combining both (graph-capture an
   already-autotuned pipeline) — not yet tried — before investing in
   OPT-004's remaining fusion steps (2-3), whose value is still
   unconfirmed and was already judged likely-modest under a
   compute-bound regime.
3. **OPT-007 (SM80 INT4) should be considered closed/shelved for Thor**
   specifically — real, measured, dramatically negative — while
   remaining a legitimate (if still K-limited) option on true Ampere/
   Orin-class hardware, which is what it was built for.

## OPT-005: FA4 for Backbone Self-Attention — Implemented, Untested Here

Per the original priority order (OPT-003 → OPT-004 step 1 → OPT-002/
OPT-005 → OPT-001 → OPT-004 steps 2-3/OPT-006), and now that the user
has an active Thor testing loop, OPT-005 was implemented: opt-in
`use_fa4=True` on `ImageWAMAttnBackend`, dispatching the "backbone"
site's plain self-attention through FA4 instead of the cuBLAS-composed
`attention_qkv_fp16`, modeled directly on `ThorFlashAttnBackend`'s own
already-verified Pi0.5 "encoder"-site FA4 call pattern.

**Correction to the original OPT-005 hypothesis, found while reading
the real Pi0.5 pattern closely (not re-derived from the earlier SigLIP-
based survey)**: Pi0.5's own GQA/single-KV-head sites use FA4 with
`pack_gqa=True` and a SINGLE shared K/V head — the SAME broadcast-K/V
convention OPT-002 is about, not real per-head MHA. FA4 does not fix
OPT-002 "for free" as first hoped; it is a faster kernel for the exact
same math this pipeline already computes. Still worth having — real
speed, zero risk to existing behavior (opt-in, default off, full
regression suite passes unchanged) — just doesn't double as an OPT-002
fix.

Default `use_fa4=False` — every existing test and call site is
unaffected. Confirmed `use_fa4=True` raises a clean `RuntimeError` on
this non-Thor machine, matching the FP4 script's own error-handling
discipline. `tests/test_imagewam_fa4_backbone.py` and
`benchmarks/imagewam_fa4_vs_cublas_bench.py` are ready for the user's
Thor agent — full detail, including what to check first if the
correctness test fails, in `opportunities.md` OPT-005.

## Real Thor Hardware Result — FA4, VAE Encode, and Graph/Autotune Re-Confirmed

User ran the full Thor checklist against commit `a0702b7` (`has_nvfp4()
== True`). All 6 ImageWAM correctness tests still PASS (no OPT-003
regression); `resolve_pipeline_class('imagewam','torch','thor')`
resolves correctly (the new `_PIPELINE_MAP` entry works on Thor too).

**OPT-005 (FA4), first real-hardware run**: `test_imagewam_fa4_backbone.py`
PASS, `cosine=1.000000, rel_l2=0.000412`. Speed
(`imagewam_fa4_vs_cublas_bench.py`, real `NH=24,HD=128,a0=896` shape):
cuBLAS 0.821ms vs FA4 0.203ms — **4.05x** on the isolated backbone
self-attention call. Not yet wired into the full 25-layer prefill
benchmarks (still opt-in, off everywhere) — see opportunities.md
OPT-005 for the pipeline-integration follow-up this implies.

Environment note the user recorded: this Thor venv's FlashRT reuses
the sibling openpi project's jax 0.5.3 via a `.pth` file;
`nvidia-cutlass-dsl` 4.5.1's `cutlass.jax` submodule needs
`jnp.float8_e8m0fnu`, which that jax version lacks, crashing `import
cutlass` outright even though FA4 only needs `cutlass.cute`. Fixed with
a venv-local try/except around that jax import inside the venv's own
`cutlass/__init__.py` — not a change to any vendored or repository
source. `fa4_backend.status() == "active"` afterward.

**OPT-008 (VAE encode), first real-hardware run** — the standout
finding this round:

| | vae_encode | prefill (25L+VAE) | one denoise step | full (10-step) | full, no VAE (earlier) |
|---|---|---|---|---|---|
| FP16 | 43.9 ms | 124.5 ms | 5.87 ms | 183.3 ms | 140.4 ms |
| FP8 (dynamic scale) | 43.7 ms | 107.8 ms | 6.17 ms | 169.5 ms | 106.6 ms (old fixed-scale, no VAE) |
| FP4 (NVFP4) | 43.8 ms | 98.7 ms | 5.51 ms | 153.7 ms | 111.1 ms |

VAE cost (~44ms) is essentially fixed regardless of GEMM precision, as
expected (same VAE module in every script). Backing it back out
reproduces the earlier no-VAE numbers closely (FP16 ≈80.6 vs 81.6ms,
FP4 ≈54.9 vs 55.8ms) — confirms no regression, just a real addition.
FP8's own backbone cost grew a bit (prefill +~4ms, denoise
4.65->6.17ms) specifically from switching to genuine dynamic
`quantize_fp8_device_fp16` scale measurement (see plan.md's earlier FP8
scale-strategy entry) — expected, not a bug.

**VAE is ~24-28% of full pipeline latency** — bigger than the entire
post-OPT-003 denoise loop, and bigger than OPT-004's graph+autotune win
combined (~10ms). Re-checked graph/autotune with this in view: those
two scripts still don't include the VAE (random image tokens, as
designed) — graph-captured (129.3ms) + VAE outside the graph (43.9ms)
≈ 173ms vs. graph-free+VAE (183ms): graph capture now saves only ~10ms
against a VAE cost 4x that size sitting right next to it. This
sharpens OPT-004's "compute-bound, not launch-bound" conclusion and
gives a concrete next target: **VAE optimization is now the single
highest-value remaining performance item**, promoted ahead of both
OPT-005's pipeline-integration follow-up and further backbone GEMM
work — see opportunities.md OPT-008's updated Promotion Condition.

## Real Thor Hardware Result — OPT-002 Round Full Verification + 5-Precision Comparison (commit e329d3a)

User rebuilt on Thor with `-DGPU_ARCH=110` (no new build flags needed --
all of this round's kernels are plain cuBLAS/CUDA, same as everything
else). All 14 tests pass with no platform-specific compile or numeric
issues:

**New correctness tests (this round's real-math work), all PASS,
cosine matching Ada exactly**: `test_imagewam_perhead_attention_kernel.py`
(5 sub-checks, cosine=1.000000), `test_imagewam_attn_backend.py`
(`use_perhead_kv=True`, cosine=1.000000 both sites), `test_imagewam_rope_kernel.py`
(embed diff=0, apply cosine=1.000000), `test_imagewam_qknorm_reuse.py`
(cosine=1.000000 incl. in-place), `test_imagewam_backbone_ref_masked_kernel.py`
(cosine=1.000000, all 3 behavioral perturbation checks correct),
`test_imagewam_real_backbone_attention.py` (small=1.000000, real
dims=0.999999), `test_imagewam_adaln.py` (timestep diff=0, rest
cosine=1.000000), `test_imagewam_real_mlp.py` (cosine=1.000000 both
shapes), `test_imagewam_real_double_stream_block.py` (the full combined
block, cosine=1.000000 both txt/img streams, real dims 3072/9216).

**Regression suite**: `mot_joint`, `mot_joint_action`, `denoise`,
`frontend`, `prefill` all still PASS -- the old broadcast path is
untouched by this round's additive work.

**Full 5-precision comparison, all with the corrected (128-channel,
16x-downsample) VAE, graph-free, random weights, still the OLD
approximate pipeline (broadcast K/V + GELU MLP -- the new real-math
work from this round is NOT wired into the pipeline yet)**:

| precision | vae | prefill(+VAE) | denoise x1 | **full (prefill+10-step)** | vs FP16 |
|---|---|---|---|---|---|
| FP16 | 38.6 | 119.8 | 5.85 | **178.3 ms** | — |
| FP8 (dynamic scale) | 39.0 | 104.6 | 6.19 | **166.5 ms** | 1.07x |
| FP4 (NVFP4) | 39.3 | 94.4 | 5.53 | **149.7 ms** | **1.19x** |
| INT8 (SM80) | 38.8 | 114.9 | 6.21 | **177.1 ms** | ~1.00x |
| INT4 (SM80) | 38.9 | 761.5 | 49.4 | **1255 ms** | **0.14x (7.0x slower)** |

DiT-only (VAE subtracted: prefill-VAE + 10x denoise): FP16 140ms, FP8
128ms, FP4 110ms, INT8 138ms, INT4 1216ms.

**New finding: INT8 (SM80) now runs the FULL pipeline cleanly on
Thor**, including the exact K=9216 (mlp_down) shape that reliably fails
on this dev machine's Ada GPU -- confirms that failure is Ada-specific
(a real hardware/driver quirk of this SM80-templated kernel on that
specific architecture), not a general property of the kernel family.
However, Thor's INT8 latency is essentially identical to FP16 (no real
tensor-core benefit at this shape on Thor either) -- INT8 remains not
worth pursuing on Thor for a different reason than on Ada (no crash,
but no speedup). INT4 remains the confirmed dead end on Thor (OPT-007,
unchanged, ~7x slower).

**Takeaways**: NVFP4 is Thor's only precision tier with a clear,
meaningful full-pipeline win (1.19x). FP8's dynamic-scale cost mostly
cancels its own GEMM speedup (1.07x, barely worth it as measured here
with zero calibration). VAE (~39ms) is still ~22% of the FP16 full
path, larger than the entire 10-step denoise loop -- OPT-008's VAE-
optimization priority stands.

`benchmarks/imagewam_thor_int8_bench.py` updated to include the real
VAE encode step, matching every other precision sibling script (it
previously had none) -- brought back from the user's own Thor-side
edit into this repo for consistency.

## Real Thor Hardware Result — Real-Checkpoint Validation (commit dc5e0ca, run 2026-09-14)

User ran `benchmarks/imagewam_real_checkpoint_validation.py` on Thor
against the real, downloaded `yuyangalin/ImageWAM-FLUX.2-4B-LIBERO`
release checkpoint (not a random/fake one). Environment corrections
vs. this round's own docstring guesses, confirmed against the actual
release layout: checkpoint file is `model.pt` (not `checkpoint.pt`),
sibling config is `config.yaml` (not `train_config.yaml`), `action_dim`
= 7 (LIBERO 7-DoF, explicitly checked against the release config, not
the script's default), `imagewam` installed via `PYTHONPATH=ImageWAM/src`
(a real `pip install -e .` would have downgraded Thor's
`torch==2.9.1+cu130` to `2.7.1` via the package's own dependency pins
-- avoided).

`model.load_checkpoint` reported `missing_keys=0 unexpected_keys=0` --
the LoRA-merge branch flagged earlier as an open uncertainty was a
non-issue for this specific checkpoint (only an unrelated
`proprio_encoder` warning, expected since the validation script never
passes `proprio_dim`).

Result, comparing FlashRT's real-math modules (this round's
`real_double_stream_block.py`/`real_single_stream_block.py`/
`pipeline_real.py`/`real_action_expert.py`) against the ACTUAL official
reference path with real trained bf16 weights (`model.video_expert.pre_dit`
+ `_build_mot_attention_mask_flux2` + `mot.prefill_flux2_video_cache`
for the 25-layer backbone; `model.action_expert.pre_dit` +
`mot.forward_flux2_action_with_video_cache` for ActionDiT):

| component | shape | cosine |
|---|---|---|
| Backbone (25-layer full prefill) | (896, 3072) | **0.999927** |
| ActionDiT (double+single, video-cache joint attn) | (64, 1024) | **0.999963** |

Both ~0.9999, the expected small headroom below 1.0 from bf16 (real
weights) vs. fp16 (FlashRT compute) precision, not from a math error.
**This confirms every real-math correction from this OPT-002 round --
per-head K/V, real 4-axis RoPE, QK-Norm, AdaLN modulation, real
LayerNorm, the real SiLU-gated-GLU MLP, and the corrected no-mask
attention rule -- end-to-end against real trained weights, not just
independent PyTorch references.** OPT-002's real-math coverage is now
fully validated, not just theoretically verified.

**What remains open**: everything above is still only in the new,
additive `real_*.py`/`pipeline_real.py` modules; `pipeline_thor.py`
itself (the actual served pipeline, and every `imagewam_thor_*_bench.py`
script) is unchanged and still uses the old approximate math by
default. Wiring the validated real math into the actual serving path
is the next real piece of work, not yet started.

**Superseded 2026-09-14**: this was fully done afterward — see
`opportunities.md` OPT-002's own final entries and the "# Plan: OPT-004
step 5" section below for what followed. `pipeline_thor.py` is now the
real-math path by default; the mechanism-integration plan below is the
current active one.

---

# Plan: OPT-004 step 5 — FP8/NVFP4 quantized GEMM for ImageWAM

Plan Status: approved

## Problem

### Current

`pipeline_thor.py`'s real-math layer functions (OPT-002, confirmed
Thor-verified this session, further sped up by OPT-004 steps 1-4 —
autotune, QKV fusion, fused AdaLN/gated-residual kernels, FA4, all
four combined giving a real -23% backbone-prefill win on Thor) run
EVERY GEMM in plain FP16 (`gemm.fp16_nn`). FlashRT already has proven,
working FP8 (`fvk.quantize_fp8_device_fp16` + `fvk.fp8_gemm_descale_fp16`)
and NVFP4 (`quant_act_nvfp4` + `fp4_gemm`) quantization primitives,
but they are only exercised inside `benchmarks/imagewam_thor_fp8_bench.py`/
`imagewam_thor_fp4_bench.py` — standalone scripts that (confirmed by
direct inspection) still model the OLD approximate math (broadcast K/V
at `HD` width via `_zeros_fp16(num_layers, TOTAL, HD)`, the OLD
`make_imagewam_attention_spec` override-after-construction calling
convention) and do not import or touch `pipeline_thor.py` at all.

Previously measured wins (OLD approximate math, real FLUX.2 dims, real
Thor hardware, from this file's own now-superseded tables above): FP8
dynamic-scale ~1.07x full-pipeline, NVFP4 ~1.19x full-pipeline. INT8/
INT4 are closed dead ends (OPT-006/007, unrelated to this plan).

A related project (pi0.5_ggml, cross-session memory `pi05_fp8_quant_scheme_candidates`
ISSUE-018) found that an FP8 kernel written for/tuned on Ada does NOT
automatically reach Thor's own native FP8 tensor cores when merely
recompiled for sm_110 — real Thor hardware there showed F16 beating
FP8 by MORE (1.58x) than Ada did (1.22x), the opposite of what a
genuine Thor-native FP8 kernel should show. That is a DIFFERENT kernel
family (ggml-cuda's own CUDA kernels, not FlashRT's `fp8_gemm_descale_fp16`/
`fp4_gemm`) — whether FlashRT's own kernels have this same gap is
UNKNOWN and must be checked before trusting any number this plan
produces (see Phase 0).

### Problem

No path exists to run ImageWAM's CURRENT real math (post-OPT-002:
real per-head K/V, fused QKV at 3x width, real MLP-gate at
`mlp_hidden*2` width, fused AdaLN kernels — all now `pipeline_thor.py`'s
own defaults) at a quantized precision at all. The existing FP8/FP4
evidence is measured against different (old, narrower) shapes and a
disconnected code path, so it cannot be trusted to predict this
plan's own result.

### Measurable goal

Quantized (FP8 and/or NVFP4) GEMM wired into `pipeline_thor.py`'s
actual real-math layer functions as an opt-in construction parameter
(matching the `use_perhead_kv`/`use_real_mot_mask`/`use_fa4`
precedent), verified for correctness (cosine against the FP16
real-math reference, at a LOSSY-precision tolerance — e.g. >0.99, NOT
the >0.999 bar used for every FP16-only mechanism change so far, since
quantization is genuinely lossy) and for real per-layer/full-prefill
speed on Thor hardware, at the CURRENT real-math shapes.

## Structure

- `flash_rt/models/imagewam/pipeline_thor.py` — OWNS per-layer GEMM
  dispatch. Every weight-projection `gemm.fp16_nn(...)` call site
  becomes a uniform call through a uniform "linear op" object,
  independent of which precision that object was built with.
- NEW `flash_rt/models/imagewam/quant_linear.py` — OWNS the linear-op
  classes (`Fp16Linear` passthrough, `Fp8Linear`, `Nvfp4Linear`),
  promoting the ALREADY-PROVEN `_Fp8Linear`/`_Fp4Linear` pattern from
  the disconnected benchmark scripts into a real, importable, reusable
  module. The underlying KERNELS are not new (Phase 0 decides whether
  they need to be); this module is new glue code, not new CUDA.
- `flash_rt/frontends/torch/imagewam_thor.py` — OWNS frontend
  construction; gains a `precision: str = "fp16"` parameter, threaded
  into weight allocation (each real weight tensor gets wrapped in the
  selected linear-op class ONCE at construction, matching the
  autotune/FA4 opt-in precedent already in this file).
- `flash_rt/frontends/torch/_imagewam_thor_spec.py` — UNCHANGED.
  Quantization is a storage-format/dispatch choice, not a shape
  change.
- `benchmarks/imagewam_thor_bench.py` — OWNS per-layer speed
  measurement; gains the same `precision=`/env-toggle pattern just
  established for `IMAGEWAM_USE_FA4` (OPT-005), so this plan reports
  through the SAME already-trusted harness instead of a fourth
  parallel disconnected script.
- The 5 existing `imagewam_thor_{fp16,fp8,fp4,int8,int4}_bench.py`
  scripts stay OUT OF SCOPE (already flagged elsewhere as stale;
  updating them to real-math shapes is separate, not-yet-started work).

State ownership: quantized weight copies are owned by the SAME
`self._weights` dict `imagewam_thor.py` already owns — each dict value
becomes a linear-op OBJECT (callable) instead of a raw pointer int,
not a second parallel structure. Open decision for Phase 1: whether
random-weight dry-run mode should skip allocating the now-redundant
FP16 copy once a weight is quantized (saves memory, adds a branch) or
just keep both (simpler, matches this stage's own "not yet
memory-optimized" scope) — default to keeping both unless it blocks
something.

## Interface

```python
# flash_rt/models/imagewam/quant_linear.py
class Fp16Linear:
    def __init__(self, gemm, weight_ptr: int, n: int, k: int): ...
    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None: ...

class Fp8Linear:
    def __init__(self, weight_fp16_ptr: int, n: int, k: int): ...  # quantizes once
    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None: ...
    # internally: quantize_fp8_device_fp16(activation) -> fp8_gemm_descale_fp16

class Nvfp4Linear:
    def __init__(self, weight_fp16_ptr: int, n: int, k: int): ...
    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None: ...
    # internally: quant_act_nvfp4(activation) -> fp4_gemm
```

`pipeline_thor.py` call-site change (mechanical, applied at every
weight-projection site in all 4 layer functions):
```python
# before:
gemm.fp16_nn(modded, key("txt_qkv.weight"), txt_qkv_merged, x0, 3 * hidden, hidden, stream)
# after:
key("txt_qkv.weight")(modded, txt_qkv_merged, x0, stream)
```
`key(slot)` now returns the linear-op OBJECT (built once at
construction), not a raw pointer — `weights` dict values change type
uniformly. Non-weight-projection calls (QKV-slice copies, QK-Norm,
RoPE, attention, SiLU-GLU) are UNCHANGED — precision only touches the
weight-projection GEMMs, matching what the existing benchmark scripts
already scope FP8/FP4 to.

## Flow

1. `imagewam_thor.py.__init__`: allocate real FP16 weight tensors (as
   today) → for each, construct the linear-op object selected by
   `precision` (`Fp16Linear` wraps the plain pointer + `gemm`;
   `Fp8Linear`/`Nvfp4Linear` quantize the weight once) → store the
   OBJECT (not a raw int) as the `self._weights` dict value.
2. Per layer: every call site is now `key(slot)(x_ptr, out_ptr, m, stream)`
   uniformly; no precision branching inside `pipeline_thor.py` itself.
3. Graph capture: unaffected in shape/mechanism — quantization becomes
   "what happens inside one call," still fully capturable (same
   graph-safety reasoning already established this session for
   `_fuse_mod_group`/autotune).

## Code Mapping

| module | file | task |
|---|---|---|
| linear-op classes | `flash_rt/models/imagewam/quant_linear.py` (new) | `Fp16Linear`/`Fp8Linear`/`Nvfp4Linear` |
| GEMM dispatch | `flash_rt/models/imagewam/pipeline_thor.py` | every weight-projection call site → `key(slot)(...)` |
| frontend construction | `flash_rt/frontends/torch/imagewam_thor.py` | `precision=` param, weight-wrapping |
| per-layer bench | `benchmarks/imagewam_thor_bench.py` | matching `precision=`/env toggle |
| correctness tests | `tests/test_imagewam_quant_linear.py` (new) | Fp8/Nvfp4 vs Fp16 cosine, small dims |
| wiring re-check | `tests/test_imagewam_thor_real_wiring.py` | extend to cover a quantized pass at lossy tolerance |
| record | `opportunities.md` OPT-004 | final "step 5" entry once measured |

## Implementation Phases

### Phase 0 — research, BLOCKING, no code

Phase Status: completed

Goal: resolve whether `fp8_gemm_descale_fp16`/`fp4_gemm` are already
Thor-native (sm_110 in their own build target list, or a genuinely
per-arch-tuned dispatch) or Ada-bound, mirroring pi0.5_ggml's own
ISSUE-018 finding for a DIFFERENT kernel family.

**Result: NOT the pi0.5_ggml gap. Both kernels are Thor-aware by
construction, for two different reasons:**

- **`fp4_gemm`** (`csrc/gemm/fp4/cutlass_fp4_gemm.cu`): a genuine
  CUTLASS 4.x kernel (CUTLASS's own example 72a, Blackwell NVFP4x
  NVFP4→bf16, adapted for fp16 output), explicitly gated by
  `ENABLE_SM100_CUTLASS` and built with `-arch=sm_110a`/`sm_100a` (the
  `'a'` suffix required for `TCGEN05_MXF4_MMA`, a Blackwell-generation
  tensor-core MMA instruction — confirmed by reading the kernel
  source's own top-of-file comment directly). `CMakeLists.txt:43-44`
  confirms `GPU_ARCH=110` (Thor) sets `ENABLE_SM100_CUTLASS=ON`
  automatically. This is compiled FOR Thor's own tensor-core
  generation, not a recompiled Ada kernel — the ~1.19x real Thor
  number already on record is trustworthy as a genuine Thor-native
  result.
- **`fp8_gemm_descale_fp16`** (`csrc/kernels/decoder_fused.cu:310`):
  NOT a custom hand-written kernel at all — it's a `cublasLtMatmul`
  call (NVIDIA's own vendor library), which dispatches to whatever
  tensor-core path is appropriate for the ACTUAL GPU it runs on via
  its own internal heuristic (`cublasLtMatmulAlgoGetHeuristic`) — this
  is architecturally the opposite situation from pi0.5_ggml's own
  custom, hand-tuned-for-Ada CUDA kernel. The only residual, much
  lower-severity question is whether cuBLASLt's own tactic selection
  is AS WELL-TUNED for Thor's FP8 tensor cores as for Ada's (a vendor-
  library maturity question, not a "wrong architecture entirely"
  question) — not something this project's own code can fix either
  way, and not blocking.

Modified files: none. Affected modules: none (pure investigation).
Conclusion: proceed to Phase 1 — no reason to expect either kernel
undershoots Thor's real potential the way pi0.5_ggml's did.

### Phase 1 — `quant_linear.py` + FP8 wiring, correctness only, small dims

Phase Status: pending

Goal: `Fp16Linear`/`Fp8Linear` implemented and cosine-verified against
the FP16 real-math reference, small test dims, no speed measurement.
Modified files: new `flash_rt/models/imagewam/quant_linear.py`;
`pipeline_thor.py` (call-site interface change); new
`tests/test_imagewam_quant_linear.py`.
Affected modules: `pipeline_thor.py`'s dispatch layer only — no
weight-shape, attention, or AdaLN changes.
Observation method: cosine check (>0.99) for one layer of each of the
4 real-math layer types (double/single backbone, double/single
action), FP8 vs. the already-trusted FP16 reference.

### Phase 2 — NVFP4 wiring, same bar

Phase Status: pending

Goal: `Nvfp4Linear`, same verification pattern as Phase 1.
Modified files: `quant_linear.py`, tests.
Affected modules: same as Phase 1.
Observation method: same cosine bar, NVFP4 vs. FP16 reference.

### Phase 3 — frontend + per-layer benchmark integration

Phase Status: pending

Goal: `imagewam_thor.py`'s `precision=` param; `imagewam_thor_bench.py`'s
matching toggle; real per-layer timing at real FLUX.2 dims (Ada first
-- correctness doesn't need Thor, timing here is only a sanity check
given OPT-004's own repeated Ada-underestimates-Thor pattern).
Modified files: `imagewam_thor.py`, `imagewam_thor_bench.py`.
Affected modules: frontend construction, bench harness.
Observation method: per-layer P50 table (FP16 vs FP8 vs NVFP4), same
style as every other OPT-004 step's own table.

### Phase 4 — real Thor measurement + close-out

Phase Status: pending

Goal: hand to the user for a real Thor correctness + speed run,
following the exact checklist pattern already established for OPT-004
steps 1-4.
Modified files: `opportunities.md` only (recording the result).
Affected modules: none (measurement only).
Observation method: Thor cosine + P50 table, compared against the
104.7ms real-math prefill baseline OPT-004/OPT-005 already established.
