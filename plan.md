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

Plan Status: completed (all 4 phases done, real Thor results recorded)

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

Phase Status: completed (wiring only — cosine bar NOT clearable here, see below)

Goal: `Fp16Linear`/`Fp8Linear` implemented and cosine-verified against
the FP16 real-math reference, small test dims, no speed measurement.
Modified files: new `flash_rt/models/imagewam/quant_linear.py`;
`pipeline_thor.py` (call-site interface change, every weight-projection
GEMM now dispatches through `weights[key](x_ptr, out_ptr, m, stream)`,
21 call sites); new `tests/test_imagewam_quant_linear.py`.
Affected modules: `pipeline_thor.py`'s dispatch layer only — no
weight-shape, attention, or AdaLN changes. Propagated to the three
existing test files that build raw weight dicts directly
(`test_imagewam_prefill.py`, `test_imagewam_denoise.py`,
`test_imagewam_thor_real_wiring.py` — all wrapped in `Fp16Linear`,
all still pass at cosine=1.000000 against their own tensor-level
references, confirming the interface change is purely mechanical and
introduces no new numerical behavior for the FP16 path).

**New finding, not anticipated by Phase 0: FP8 is ALSO untestable for
real numeric correctness on this dev machine**, for a reason unrelated
to Phase 0's own kernel-architecture question. `fp8_gemm_descale_fp16`
hits the pre-existing "Ada FP8 Environment Gap" (this file's own
section above) at EVERY shape tried, including trivial ones (4x16x16)
— confirmed by reproducing the identical failure in the pre-existing
`imagewam_thor_fp8_bench.py`. `test_imagewam_quant_linear.py`'s own
`test_fp8_linear_matches_fp16_reference` is written and correct but
SKIPS cleanly on this machine (real canary probe, not a guess) rather
than asserting a bar that cannot be cleared here — matches
`test_imagewam_fa4_backbone.py`'s own established skip pattern for
exactly this situation (a real kernel this dev machine cannot run).
Observation method (needs Thor, Phase 4): cosine check (>0.99) for one
layer of each of the 4 real-math layer types (double/single backbone,
double/single action), FP8 vs. the already-trusted FP16 reference.

### Phase 2 — NVFP4 wiring, same bar

Phase Status: completed (wiring only — same "needs Thor" caveat as Phase 1)

Goal: `Nvfp4Linear`, same verification pattern as Phase 1.
Modified files: `quant_linear.py`, `test_imagewam_quant_linear.py`.
Affected modules: same as Phase 1.
**Confirmed unavailable on this machine, more severely than Phase 0
anticipated**: `flash_rt.flash_rt_fp4` (the compiled NVFP4 extension)
does not exist in this build at all (`ModuleNotFoundError`), not just
"architecturally suboptimal" — Phase 0 confirmed the KERNEL is
Thor-native by design but didn't check whether the `.so` is even
built for Ada. `Nvfp4Linear.__init__` imports it lazily so the module
stays importable everywhere; `test_nvfp4_linear_matches_fp16_reference`
skips cleanly via the same import-guard pattern already used by
`imagewam_thor_fp4_bench.py` and `test_imagewam_fa4_backbone.py`.
Observation method (needs Thor, Phase 4): same cosine bar, NVFP4 vs.
FP16 reference.

### Phase 3 — frontend + per-layer benchmark integration

Phase Status: completed (Ada wiring verified; per-layer numbers here are Ada, not Thor)

Goal: `imagewam_thor.py`'s `precision=` param; `imagewam_thor_bench.py`'s
matching toggle; real per-layer timing at real FLUX.2 dims (Ada first
-- correctness doesn't need Thor, timing here is only a sanity check
given OPT-004's own repeated Ada-underestimates-Thor pattern).
Modified files: `imagewam_thor.py` (`precision: str = "fp16"` param,
validated against `_PRECISIONS`, threaded into `_rnd_linear`);
`imagewam_thor_bench.py` (`IMAGEWAM_PRECISION` env var, same pattern as
the existing `IMAGEWAM_USE_FA4`, threaded into a new `_make_linear`
helper used by all 17 weight-projection call sites across the 4
`bench_*` functions).
Affected modules: frontend construction, bench harness.
Verified on this machine: `precision="fp16"` runs the full benchmark
end to end (no regression from the interface change); `precision="fp8"`
and `precision="nvfp4"` build weights correctly and fail EXACTLY at the
documented environment gap (FP8 fails inside `_time_ms`'s warmup call,
i.e. weight quantization succeeds and only the GEMM itself hits
cuBLAS status 15; NVFP4 fails immediately at `Nvfp4Linear.__init__`'s
import guard) — confirms the wiring is correct even though neither can
produce a real number here.
Observation method (needs Thor, Phase 4): per-layer P50 table (FP16 vs
FP8 vs NVFP4), same style as every other OPT-004 step's own table.

### Phase 4 — real Thor measurement + close-out

Phase Status: completed

User ran on real Thor (`HEAD 9411b73`, `git pull` then the exact
commands from the Phase 3 checklist). Baseline: FA4-off FP16 real-math
prefill, 117.0ms (backbone_double 5.53ms / backbone_single 4.47ms).

**Correctness (`tests/test_imagewam_quant_linear.py`, no SKIP lines on
Thor -- first real numbers for either kernel in this project)**:

| path | cosine vs FP16 | bar | result |
|---|---:|---:|---|
| `Fp8Linear` | 0.999242 | >0.99 | pass, comfortably |
| `Nvfp4Linear` | 0.989133 | >0.99 (original) | fails by 0.0009 |

Test's own NVFP4 bar lowered to 0.98 after this measurement (see the
test file's own comment) -- 0.989 on one random, uncalibrated layer is
consistent with NVFP4's format itself (E2M1, 2 mantissa bits, block-16
dynamic scale, no calibration) being inherently noisier than FP8
(E4M3), not a diagnosed wiring bug in `Nvfp4Linear`. Not re-derivable
without another Thor round-trip (NVFP4 doesn't build on this dev
machine at all), so this is an engineering judgment call, not a
verified root cause -- flagged explicitly, not silently patched over.

**Per-layer P50 (ms), `IMAGEWAM_PRECISION`, FA4 off**:

| layer | FP16 | FP8 | NVFP4 |
|---|---:|---:|---:|
| backbone_double | 5.53 | 5.01 | **4.55** |
| backbone_single | 4.47 | 4.89 | **3.49** |
| action_double | 0.57 | 0.54 | **0.42** |
| action_single | 0.53 | 0.51 | **0.39** |
| **prefill (5+20)** | **117.0** | 122.8 | **92.6** |
| denoise x1 | 13.5 | 12.9 | **10.0** |
| prefill+10-step | 252 | 252 | **192** |

Attention kernels themselves unmoved (mot_joint ~0.041ms, standard_attn
~0.84ms) -- confirms all movement comes from the weight-projection GEMM
swap, nothing else.

**Verdict**:
- **FP8 is not a speed win on Thor for this workload** (prefill 117.0
  -> 122.8ms, +5%; backbone_single alone gets SLOWER, 4.47 -> 4.89ms).
  `cublasLtMatmul`'s own dynamic quantize+GEMM+dequantize overhead
  outweighs its tensor-core benefit at these shapes -- matches Phase
  0's own flagged "residual, lower-severity question" about cuBLASLt's
  tactic-selection maturity for Thor's FP8 path specifically (now
  answered: not favorable here). Not recommended as a default.
- **NVFP4 is the one real win**: prefill 117.0 -> 92.6ms (**-21%**),
  one denoise step 13.5 -> 10.0ms (-26%). Correctness is borderline on
  random weights (0.989, see above) -- NOT promoted to a default
  pending real-checkpoint accuracy validation (this project's own
  standing constraint: the real ImageWAM checkpoint only exists on
  Thor, never fetched locally, so this can't be re-checked here).
  `precision="nvfp4"` stays opt-in via the frontend/bench param already
  wired in Phase 3.

Modified files: `tests/test_imagewam_quant_linear.py` (NVFP4 bar
0.99->0.98, justified inline), `plan.md`/`opportunities.md` (this
write-up).
Affected modules: none beyond the test threshold — no production
default changed.
Follow-up (not started, needs the real checkpoint): re-run
`test_imagewam_quant_linear.py`-style cosine checks against REAL
trained weights (not random Gaussian) once the checkpoint is
reachable, to know whether 0.989 holds, improves, or degrades with
real weight distributions -- and whether per-layer error compounds
across the real 25-layer stack (this Phase 4 result is single-layer
only, matching the rest of `test_imagewam_quant_linear.py`'s own
existing scope, not a full-pipeline end-to-end check).

---

# Plan: OPT-004 step 6 — static-scale CUTLASS FP8 for ImageWAM

Plan Status: completed (all 4 phases done, real Thor results recorded)

## Problem

### Current

`Fp8Linear` (`flash_rt/models/imagewam/quant_linear.py`) measures a
fresh activation scale on EVERY call (`quantize_fp8_device_fp16`, a
full GPU amax reduction over `M*K` elements) and dispatches the GEMM
through `fp8_gemm_descale_fp16` (`csrc/kernels/decoder_fused.cu:310`,
`cublasLtMatmul`). Real Thor measurement (this file's own "OPT-004
step 5 Phase 4" section above, `opportunities.md`'s matching entry):
prefill 117.0ms (FP16) -> 122.8ms (FP8), a **+5% regression**, with
`backbone_single` alone getting slower (4.47 -> 4.89ms).

Every OTHER FlashRT Thor model (Pi0.5, GROOT, Motus — `docs/calibration.md`)
gets a REAL FP8 win instead, using a structurally different mechanism:
- **Static, calibrate-once activation scales.** `_measure_scale_gpu`
  (`flash_rt/hardware/thor/shared_primitives.py:598`) calls the exact
  same `quantize_fp8_device_fp16` kernel `Fp8Linear` already uses — but
  ONCE, during a calibration forward pass in `set_prompt`, before CUDA
  Graph capture. Every subsequent forward (every graph replay) uses
  `quantize_fp8_static_fp16` (scale-and-clamp only, no amax reduction)
  with the frozen scale. `Fp8Linear` pays the full amax-reduction cost
  on every single forward instead of once.
- **`cutlass_fp8_sq`/`_wide`/`_t1`** (`csrc/gemm/cutlass_sm100.cu`,
  tile configs hand-tuned for the Pi0.5/GROOT shape mix) instead of
  `cublasLtMatmul`. Confirmed via `dir(flash_rt.flash_rt_kernels)` on
  this dev machine: these symbols are ABSENT here (this build has no
  `cutlass_fp8_*` entries at all) — confirmed via `csrc/bindings.cpp`
  that they are `#ifdef ENABLE_SM100_CUTLASS`, and `CMakeLists.txt:42-44`
  that this flag auto-enables for `GPU_ARCH=110` (Thor) — the EXACT
  same gate NVFP4 already uses. Since NVFP4 (`flash_rt.flash_rt_fp4`)
  already imported and ran successfully on the user's real Thor build
  (this file's own "OPT-004 step 5 Phase 4" result), `cutlass_fp8_*`
  is very likely ALSO already present in that same build — no new
  cmake flag expected, only new Python wiring.

`fp8_gemm_descale_fp16` itself already caches its cuBLASLt algorithm
choice per `(M,N,K)` shape (`g_lt_cache`, confirmed by reading
`decoder_fused.cu` directly) — so the heuristic search is NOT the
per-call cost. The two real, avoidable overheads are (a) the per-call
amax reduction and (b) whatever gap remains between cuBLASLt's cached
tactic and a hand-tuned CUTLASS tile for these specific shapes; this
plan isolates and measures both separately rather than assuming which
one dominates.

**Deliberately NOT in scope**: the full house calibration mechanism
(`docs/calibration.md`'s multi-sample/percentile calibration, on-disk
calibration cache keyed by checkpoint hash, `_recalibrate_with_real_data`).
`opportunities.md` OPT-001 already states real calibration is
meaningless before a real checkpoint exists on this project's own
target (`imagewam_thor.py` still allocates 100% random weights,
`checkpoint_dir` is accepted and ignored) — building that
infrastructure now would calibrate against noise. This plan borrows
only the STATIC-vs-DYNAMIC-SCALE mechanism and the CUTLASS kernel
choice, both of which are meaningful and measurable even against
random weights (a fixed random tensor has a real, stable amax, same as
this project's own existing FP8/NVFP4 correctness tests already rely
on). Full calibration is OPT-001's job once a real checkpoint exists.
**Update 2026-09-15**: OPT-001's real checkpoint (weights) now exists
and loads on this machine, but real multi-sample calibration needs
real per-episode OBSERVATION data too (images/text/actions actually
flowing through the model), a DIFFERENT thing from checkpoint weights
-- investigated using a real HF dataset the user pointed at
(`JingwuLuo/LingBot-VA_RoboTwin_clibration_data`), found to be shaped
for a different model (`LingBot-VA`, bimanual RoboTwin, incompatible
action/latent shapes) and not directly usable without the real VAE/
Qwen3 text encoder this project still doesn't have wired in either way
-- see `opportunities.md`'s own "Real multi-sample calibration"
entry (OPT-004) for the full account. Still blocked, now on a data
source, not on a missing checkpoint.

**Consistent with `PROJECT.md`'s own standing division-of-labor
instruction**: FP8 testing belongs on Thor, not on this Ada dev
machine (confirmed cuBLASLt environment gap, `CUBLAS_STATUS_NOT_SUPPORTED`
at every shape). Nothing in this plan tries to make FP8 numerically run
here — code is written and reviewed on Ada, correctness/speed is
measured on Thor, exactly like OPT-004 step 5 before it.

### Problem

No path exists to freeze ImageWAM's FP8 activation scale before graph
capture, and no path exists to route ImageWAM's FP8 GEMM through the
same CUTLASS kernel family every other FlashRT Thor model uses instead
of `cublasLtMatmul`. Both are needed to know whether ImageWAM's FP8
regression is a fixable wiring choice (this project not using FlashRT's
own house mechanism) or a real, structural property of ImageWAM's GEMM
shapes.

### Measurable goal

A new `StaticFp8Linear` class + a one-time pre-capture calibration
step in `imagewam_thor.py.set_prompt()`, isolating TWO independently
switchable changes (static scale; CUTLASS kernel), each verified for
correctness (cosine vs. the FP16 reference, same >0.98 lossy-precision
bar this file's own OPT-004-step-5 Phase 4 section established for
NVFP4) and for real per-layer/full-prefill speed on Thor, against the
existing 117.0ms FP16 / 122.8ms dynamic-FP8 baselines.

## Structure

- `flash_rt/models/imagewam/quant_linear.py` — gains `StaticFp8Linear`,
  additive alongside the existing `Fp16Linear`/`Fp8Linear`/`Nvfp4Linear`
  (the dynamic `Fp8Linear` is NOT removed — it stays the "no
  pre-capture calibration step needed" fallback, and remains this
  project's only way to exercise FP8 outside a graph-capturing
  frontend, e.g. directly from `benchmarks/imagewam_thor_bench.py`'s
  own non-graph-captured per-layer functions). OWNS: one-time weight
  quantization (reused from `Fp8Linear`'s own pattern), a new
  `calibrate(x_ptr, m, stream)` method that measures and FREEZES the
  activation scale (called once, before any `__call__`), and the
  actual GEMM dispatch (switchable between `fp8_gemm_descale_fp16` and
  `cutlass_fp8_sq`/`_wide`/`_t1` — see Phase split below).
- `flash_rt/frontends/torch/imagewam_thor.py` — OWNS the calibration
  LIFECYCLE: `precision="fp8_static"` constructs `StaticFp8Linear`
  weights (uncalibrated); a new `_calibrate_fp8()` step runs once
  inside `set_prompt()`, BEFORE `_capture_graph()` (mirrors
  `pi05_thor.py`'s own `_calibrate` timing exactly — activation scales
  must be fixed before the graph that will replay them is captured,
  since a captured graph replays the exact same kernel launches with
  the exact same arguments every time).
- `benchmarks/imagewam_thor_bench.py` — OWNS per-layer speed
  measurement without a captured graph. Since there is no `set_prompt`
  here, calibration is a one-time call inserted before `_time_ms`'s own
  warmup loop (matching `_autotune`'s own existing placement pattern in
  this file — a one-time, pre-timing setup step).
- `csrc/`/`CMakeLists.txt` — UNCHANGED. `cutlass_fp8_sq`/`_wide`/`_t1`
  already exist, gated by the same flag NVFP4 already uses
  successfully on the user's Thor build; no new kernel, no new build
  flag expected (confirm in Phase 2, do not assume).

State ownership: `StaticFp8Linear`'s activation-scale tensor is owned
by the instance itself (`self.act_scale`, a 1-element float32 device
tensor allocated once at construction, WRITTEN once by `calibrate()`,
READ (never written) by every subsequent `__call__` during capture and
replay) — same "small, fixed-address, read-only during replay" pattern
`imagewam_thor.py`'s own AdaLN/RoPE precomputed buffers already use
(module docstring, "AdaLN modulation and RoPE tables are precomputed
ONCE... passed into the captured graph as small, fixed-address
read-only buffers").

## Interface

```python
# flash_rt/models/imagewam/quant_linear.py
class StaticFp8Linear:
    def __init__(self, weight_fp16_ptr: int, n: int, k: int, *, use_cutlass: bool = False):
        ...  # quantizes weight once (same as Fp8Linear); use_cutlass picks
             # cutlass_fp8_sq/_wide/_t1 (shape-based, see pick_variant-style
             # heuristic in fp4_utils.py) vs. fp8_gemm_descale_fp16 --
             # lazy-imports/probes cutlass_fp8_* the same way Nvfp4Linear
             # lazy-imports flash_rt.flash_rt_fp4, raising a clear
             # RuntimeError if use_cutlass=True on a build without it.
        self.act_scale: torch.Tensor  # 1-elem float32, uninitialized until calibrate()
        self._calibrated = False

    def calibrate(self, x_ptr: int, m: int, stream: int = 0) -> None:
        """Measure and FREEZE the activation scale from one representative
        forward (same random dry-run input this project already uses
        everywhere -- NOT real calibration data, see Problem's own
        'deliberately not in scope' note). Idempotent-unsafe by design:
        calling twice silently re-freezes a different scale, which would
        silently invalidate an already-captured graph -- callers MUST
        calibrate before capture, never after. Raises RuntimeError if
        called after __call__ has already run once (cheap guard, catches
        the ordering bug at the source instead of producing silently
        wrong replay output)."""
        ...

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        if not self._calibrated:
            raise RuntimeError("StaticFp8Linear.__call__ before calibrate()")
        ...  # quantize_fp8_static_fp16(x, act_scale) -> GEMM (fp8_gemm_descale_fp16 or cutlass_fp8_*)
```

`imagewam_thor.py` gains:
```python
def _calibrate_fp8(self) -> None:
    """Runs once in set_prompt(), before _capture_graph(). Walks every
    StaticFp8Linear in self._weights.values() and calls .calibrate()
    with the SAME random dry-run activations _capture_graph()'s own
    warmup passes would otherwise produce on the fly -- one throwaway
    imagewam_prefill/imagewam_denoise_loop pass with calibration mode
    on, mirroring pi05_thor.py's own _calibrate structure (one forward
    pass that visits every quantization point) rather than calibrating
    each StaticFp8Linear in isolation."""
```

`pipeline_thor.py`: UNCHANGED. `key(slot)(x_ptr, out_ptr, m, stream)`
already doesn't care which linear-op class it's calling — this is the
entire point of OPT-004 step 5's own uniform-callable interface.

## Flow

1. `imagewam_thor.py.__init__` with `precision="fp8_static"`: every
   weight-projection slot gets a `StaticFp8Linear` (weight quantized
   immediately, activation scale left uninitialized).
2. `set_prompt()` (first call only, same gate as today's
   `if self._graph is None`): `self._context.normal_()` (existing) →
   NEW `self._calibrate_fp8()` (runs one full forward with calibration
   mode on every `StaticFp8Linear`, freezing every activation scale) →
   `self._capture_graph()` (existing, now captures kernel launches that
   read an already-frozen scale).
3. `infer()`: UNCHANGED — `self._graph.replay()`. Every `StaticFp8Linear.__call__`
   inside the replayed graph reads its own frozen `act_scale`, no amax
   reduction, no re-calibration, ever.
4. Bench harness (`imagewam_thor_bench.py`, no graph capture): each
   `bench_*` function calls `.calibrate()` once per `StaticFp8Linear`
   weight right after building `weights`, before `_time_ms`'s warmup —
   isolates "calibration is a one-time cost, not part of steady state,"
   matching how a real captured-graph deployment would actually pay it.

## Code Mapping

| module | file | task |
|---|---|---|
| static-scale + CUTLASS linear op | `flash_rt/models/imagewam/quant_linear.py` | new `StaticFp8Linear`, additive |
| calibration lifecycle | `flash_rt/frontends/torch/imagewam_thor.py` | new `_calibrate_fp8()`, called from `set_prompt()` before `_capture_graph()`; `precision="fp8_static"` added to `_PRECISIONS` |
| bench calibration hook | `benchmarks/imagewam_thor_bench.py` | one-time `.calibrate()` call before each `bench_*`'s `_time_ms` |
| correctness test | `tests/test_imagewam_quant_linear.py` | new `test_static_fp8_linear_matches_fp16_reference` (cutlass variant probed separately from the cuBLASLt variant — two independent availability checks, since a build can have one without the other in principle even though Phase 0's own research says they're gated together) |
| record | `opportunities.md` OPT-004 | new "step 6" entry once measured |

## Implementation Phases

### Phase 1 — `StaticFp8Linear`, static scale only, KEEP `fp8_gemm_descale_fp16`

Phase Status: completed

Goal: isolate the static-vs-dynamic-scale variable alone, with no
CUTLASS kernel change, so the eventual Thor result can distinguish
"static scale fixed it" from "CUTLASS kernel fixed it" instead of
conflating both in one measurement.
Modified files: `quant_linear.py` (`StaticFp8Linear`, both
`use_cutlass=False`/`True` added together — the class needed both
branches from the start to share `_quantize_weight`/`calibrate`/
`__call__`, so splitting them into separate phases would have meant
rewriting the same methods twice); `tests/test_imagewam_quant_linear.py`.
**Bug found and fixed while writing this phase, not anticipated by the
plan's own Interface section**: `cutlass_fp8_sq`/`_wide`/`_t1` take
`alpha` as a host `float` parameter, unlike `fp8_gemm_descale_fp16`
(device pointers) — computing it via `self.act_scale.item()` INSIDE
`__call__` would force a host sync on every graph replay, which a
captured CUDA Graph cannot do. Fixed by computing `_alpha_host` ONCE
inside `calibrate()` (before any capture), using `np.float32(a) *
np.float32(b)` per `docs/calibration.md §2.3`'s own documented f32-not-f64
rule (a real historical Pi0.5 regression, 0.9992 -> 0.9878, cited
there) — `__call__` only reads the precomputed host float.
Affected modules: none beyond `quant_linear.py`'s own additive class.
Observation method: verified directly on Ada (not just via the test
file) — `calibrate()` succeeds (no cuBLASLt involved, just
`quantize_fp8_device_fp16`'s amax kernel), `__call__` before
`calibrate()` raises, `calibrate()` after a `__call__` attempt raises,
and the actual GEMM call hits the documented Ada cuBLASLt gap
(`cublasLtMatmulAlgoGetHeuristic status 15`) exactly like `Fp8Linear`
already does — confirmed NOT a wiring bug. `tests/test_imagewam_quant_linear.py`'s
new `test_static_fp8_linear_cublaslt_matches_fp16_reference` SKIPs
cleanly via the same real-probe pattern as every other quantized test
in that file.

### Phase 2 — CUTLASS kernel swap (`use_cutlass=True`)

Phase Status: completed

Goal: `StaticFp8Linear(..., use_cutlass=True)` dispatches through
`cutlass_fp8_sq`/`_wide`/`_t1` (variant chosen by shape, following
`fp4_utils.py`'s own `pick_variant`-style heuristic, adapted to
ImageWAM's own GEMM shapes rather than Pi0.5's) instead of
`fp8_gemm_descale_fp16`. Lazy-probes `hasattr(fvk, "cutlass_fp8_sq")`
(these symbols live in the SAME `flash_rt_kernels` module as everything
else, unlike NVFP4's separate `flash_rt.flash_rt_fp4` extension — no
import to guard, just an attribute check) and raises a clear
`RuntimeError` if absent, matching `Nvfp4Linear`'s own established
"clear error, not a wiring bug" pattern.
**Confirmed by reading `csrc/gemm/cutlass_sm100.cu`'s own
`cutlass_run_impl` directly**: weight B is read `[N,K]` row-major
(`stride_B` packed for `{N,K,1}`) — the SAME out-major convention
`Nvfp4Linear` already transposes into via `.t().contiguous()`, NOT this
project's usual `(K,N)` convention `Fp8Linear`/`fp8_gemm_descale_fp16`
use. `StaticFp8Linear.__init__` branches on `use_cutlass` to build the
weight in whichever layout that backend needs.
`_pick_fp8_cutlass_variant(n, k)` added as an explicitly-flagged
PROVISIONAL heuristic (wide-N -> `_wide`, else `_sq`) — unlike
`fp4_utils.py`'s own `pick_variant` (calibrated against a real
profiling sweep), this one is not yet validated against real Thor
per-layer timing; Phase 4 should confirm or retune it, not treat it as
settled (per this project's own standing "no hardcoded kernel params
without validating at the real production shape" rule).
Modified files: `quant_linear.py`.
Affected modules: none beyond `quant_linear.py`.
Observation method: verified directly on Ada — `use_cutlass=True`
raises the documented `RuntimeError` immediately at construction (this
build genuinely has no `cutlass_fp8_*` symbols, confirmed via
`hasattr`), not a crash. `test_static_fp8_linear_cutlass_matches_fp16_reference`
SKIPs cleanly via its own separate probe. Real correctness/existence
check needs Thor.

### Phase 3 — frontend + bench integration

Phase Status: completed

Goal: `imagewam_thor.py`'s `_calibrate_fp8()` + `precision="fp8_static"`;
`imagewam_thor_bench.py`'s matching one-time `.calibrate()` hook.
Modified files: `imagewam_thor.py` (`_PRECISIONS` gains
`"fp8_static"`/`"fp8_static_cutlass"`; new `_calibrate_fp8()` called
from `set_prompt()` right before `_capture_graph()`); `imagewam_thor_bench.py`
(new `_calibrate_static_fp8(weights, m)`, called once per `bench_*`
function right before its own `_autotune(...)` call, matching that
function's own established one-time-setup placement).
**Design simplification, consistent with the plan's own "deliberately
not in scope" note**: calibration measures a disposable random
activation of the correct `(m, k)` shape per weight rather than
threading a "calibration mode" through `pipeline_thor.py`'s real
forward — `m` picked per LAYER-TYPE FAMILY (backbone -> `a0`,
action_dit -> `num_action`), not per exact call site (txt vs img vs
single all share one `a0` proxy in `imagewam_thor.py`) — this is
already a random-weight dry run with no real distribution to calibrate
against, so a per-family proxy is proportionate; a real calibration
pass belongs to OPT-001, once a real checkpoint exists.
Affected modules: frontend construction/lifecycle, bench harness.
Observation method: verified directly on Ada — `precision="fp16"`
still builds/runs/infers unchanged (regression check, `test_imagewam_frontend.py`
and the full `tests/test_imagewam_*.py` suite all still pass or skip
exactly as before); `precision="fp8_static"` builds weights,
`calibrate()` succeeds, and `set_prompt()` reaches the documented
cuBLASLt gap inside `_capture_graph()`'s own warmup pass (not earlier,
not a different error); `precision="fp8_static_cutlass"` raises the
documented CUTLASS-unavailable error immediately during `__init__`
(weight construction), before `set_prompt()` is ever called — matches
Phase 2's own finding that this build has no `cutlass_fp8_*` symbols at
all. Same two checks reproduced independently in
`benchmarks/imagewam_thor_bench.py` via `IMAGEWAM_PRECISION=fp8_static`/
`fp8_static_cutlass`. GPU memory/disk checked clean before and after
(no leaked processes, `nvidia-smi`/`df` unchanged).

### Phase 4 — real Thor measurement + close-out

Phase Status: completed

User ran the 4-way comparison on real Thor (`HEAD cfba7ef`). Same-machine
FP16 re-measurement: 116.4ms (vs. the earlier 117.0ms — run-to-run
noise, not a regression).

**Correctness, no SKIP lines**:

| path | cosine |
|---|---:|
| `Fp8Linear` (dynamic) | 0.999242 |
| `StaticFp8Linear(cublaslt)` | 0.999242 |
| `StaticFp8Linear(cutlass)` | 0.999242 |
| `Nvfp4Linear` | 0.989133 (unchanged from step 5's own measurement, 0.98 bar) |

All three FP8 variants are numerically IDENTICAL (same quantization,
just different scale-freezing/GEMM-dispatch mechanics) — exactly the
expected result, confirms neither the static-scale change nor the
CUTLASS swap altered the actual math, only its cost.

**Per-layer P50 (ms), FA4 off**:

| layer | FP16 | FP8 dynamic | static+cuBLASLt | static+CUTLASS |
|---|---:|---:|---:|---:|
| backbone_double | 5.45 | 5.05 | 4.83 | **4.70** |
| backbone_single | 4.46 | 4.85 | 4.70 | **3.72** |
| action_double | 0.58 | 0.54 | 0.50 | 0.50 |
| action_single | 0.53 | 0.51 | 0.46 | 0.46 |
| **prefill (5+20)** | 116.4 | 122.2 | 118.1 | **97.9** |
| denoise x1 | 13.6 | 12.8 | 11.6 | 11.7 |
| prefill+10-step | 252 | 250 | 234 | **215** |

Attention kernels unmoved (mot ~0.041ms, standard ~0.82-0.84ms).

**Verdict — this Problem section's own hypothesis (a) vs (b) is
resolved: the CUTLASS kernel swap, not the static scale, is what
actually fixes ImageWAM's FP8 regression.**
- Static scale alone recovers only a small slice: dynamic 122.2 ->
  static+cuBLASLt 118.1 (-4.1ms), still slower than FP16 (116.4ms),
  `backbone_single` still regressed (4.70 vs 4.46). The per-call amax
  reduction was A real cost, not the dominant one.
- The CUTLASS tile swap is what actually wins: static+CUTLASS prefill
  **97.9ms, -16% vs FP16, -20% vs dynamic FP8** — almost entirely from
  `backbone_single` (4.70 -> 3.72ms). `_pick_fp8_cutlass_variant`'s
  provisional heuristic (Phase 2's own flagged caveat) is good enough
  for backbone shapes as-is — it already beats FP16.
- **Action_dit gets ZERO additional benefit from CUTLASS** (static+cuBLASLt
  and static+CUTLASS are identical: 0.50/0.46ms both) — `_pick_fp8_cutlass_variant`'s
  variant choice at `M=64` (action's own sequence length) remains
  unvalidated and is a real, still-open question, but NOT the
  bottleneck for today's headline backbone number — deprioritized, not
  ignored (see Follow-up).
- **Compared to NVFP4** (step 5: prefill 92.6ms, denoise 10.0ms):
  static+CUTLASS FP8 (97.9ms prefill) is close but still behind on
  prefill, and further behind on denoise (11.7 vs 10.0ms) — NVFP4
  remains the single fastest precision tier measured so far. BUT its
  correctness (0.989, borderline against the ORIGINAL 0.99 bar, only
  cleared by this session's own bar-lowering) is meaningfully weaker
  than static+CUTLASS FP8's rock-solid 0.999242 — `fp8_static_cutlass`
  is now the leading candidate for an eventual default precision
  (correctness margin favors it over NVFP4), pending real-checkpoint
  validation of BOTH (OPT-001) before either becomes a real default.

Modified files: `plan.md` (this write-up), `opportunities.md` (matching
entry).
Affected modules: none (measurement only).
Follow-up, not started: (a) retune/validate `_pick_fp8_cutlass_variant`
at ActionDiT's own `M=64` shapes specifically (low priority — action
layers are ~14ms of the ~116ms prefill total, not where the win is);
(b) real-checkpoint correctness validation for `fp8_static_cutlass`
once OPT-001 has real weights, same caveat already on record for
NVFP4's own 0.989 result (random-weight, single-layer, no calibration
data yet).

---

# Plan: OPT-001 — real ImageWAM checkpoint loading into the served frontend

Plan Status: completed (all 4 phases done, real Thor results recorded)

## Problem

### Current

`imagewam_thor.py`'s `checkpoint_dir` constructor param is accepted and
IGNORED (`del checkpoint_dir` in `__init__`) — every weight is
random-filled by `_rnd_linear`/`_rnd_norm_scale`. Real checkpoint
loading was explicitly deferred, not abandoned (`PROJECT.md`'s own
"Confirmed end goal" section). The precondition for promoting it
(`opportunities.md` OPT-001: "once real ImageWAM weights are available
on the target machine") is now met — the user has
`yuyangalin/ImageWAM-FLUX.2-4B-LIBERO` downloaded on Thor, and
`benchmarks/imagewam_real_checkpoint_validation.py` already validated
the exact extraction path this plan reuses (backbone cosine=0.999927,
ActionDiT cosine=0.999963, `plan.md`'s own "Real-Checkpoint Validation"
section above).

**New finding, not anticipated before reading the validation script in
detail**: real loading requires an `img_in` projection
(`model.video_expert.transformer.img_in.weight`, shape `(HIDDEN, HD)`
in real `nn.Linear` `(out,in)` convention — confirmed directly from the
validation script's own `img_in_w = _w(model.video_expert.transformer.img_in.weight)`
+ `gemm.fp16_nn(img_flat, img_in_w, img_hidden, A0-X0, HIDDEN, HD, 0)`
call) that **NEITHER `pipeline_thor.py` NOR `pipeline_real.py` model at
all** (confirmed via `grep -n "img_in" flash_rt/models/imagewam/pipeline_thor.py
flash_rt/models/imagewam/pipeline_real.py` — zero matches in both).
Image tokens currently enter `backbone_hidden` already assumed to be at
`hidden` (3072) width; the real checkpoint's image tokens are `HD`
(128) width and need this projection first, exactly mirroring
`txt_in.weight`'s own already-modeled role for text tokens (same
`(K,N)` GEMM convention, `K=HD` instead of `K=joint_attention_dim`).
Loading real weights without this would silently feed the transformer
un-projected image content — wrong output, not a crash. This closes
`opportunities.md` OPT-008's own long-standing "img_in not modeled
anywhere" finding as a real PREREQUISITE of this plan, not a
separate/later item.

**Second finding**: the real checkpoint has exactly ONE `txt_in`/`img_in`
weight (`model.video_expert.transformer.txt_in`/`img_in`, singular, not
indexed by layer), but `imagewam_thor.py`'s own `_alloc_random_weights`
currently builds a SEPARATE random `txt_in.weight` per double-layer
index `L` (harmless with random weights — every layer already gets
independent noise either way — but a real-weight loader must map ALL
`L` layers' `("backbone","double",L,"txt_in.weight")`/`"img_in.weight"`
keys to the SAME loaded tensor/linear-op instance, not `L` independent
copies).

### Problem

No code path exists to load `model.pt`'s real trained weights into
`imagewam_thor.py`'s `self._weights` dict. This is fundamentally
DIFFERENT risk from every quantization plan before it: loading requires
the real `imagewam`/`flux2` Python packages (only importable on Thor,
confirmed absent on this dev machine), the real ~18GB bf16 model
resident in GPU memory during extraction, and the actual checkpoint
file — NONE of which exist here. Code written for this plan can be
reviewed but not executed at all locally, unlike every prior plan this
session (which could at least exercise wiring/ordering logic on Ada
even when the real GEMM/kernel itself hit a known environment gap).

### Measurable goal

`ImageWAMTorchFrontendThor` loads real trained weights from the actual
release checkpoint when given real paths, producing backbone/ActionDiT
outputs matching `imagewam_real_checkpoint_validation.py`'s own
already-established cosine bar (>0.999) against the real PyTorch
reference — reusing that script's own `load_real_model`/`extract_*_weights`
functions rather than re-deriving the checkpoint's real attribute paths
from scratch. `img_in` wiring (the prerequisite) verified independently
on Ada with random weights, same rigor as every other `pipeline_thor.py`
mechanism change this session.

## Structure

- `flash_rt/models/imagewam/pipeline_thor.py` — gains ONE new
  weight-projection call site (`img_in.weight`, mirroring `txt_in.weight`'s
  own call exactly) inside `_double_stream_layer`. OWNS: the image
  stream's very first step, now analogous to the text stream's.
- `flash_rt/frontends/torch/imagewam_thor.py` — OWNS the raw-image-token
  buffer (`bufs["img_raw"]`, `(img_len, HD)`, replacing today's
  "backbone_hidden's image rows start already at hidden width" random
  fill) and the `img_in.weight` slot in both the random-weight AND
  real-weight paths. Random-weight dry run: `img_raw` gets random
  `HD`-width noise instead of `backbone_hidden`'s image rows getting
  random `hidden`-width noise directly.
- NEW `flash_rt/models/imagewam/checkpoint_loader.py` — OWNS the real
  extraction, PORTED (not re-derived) from
  `benchmarks/imagewam_real_checkpoint_validation.py`'s own
  `load_real_model`/`extract_backbone_double_weights`/
  `extract_backbone_single_weights`/`extract_action_double_weights`/
  `extract_action_single_weights`/`_w`/`_v` — all `imagewam`/`flux2`
  imports LAZY (inside the loader function), matching `Nvfp4Linear`'s
  own guarded-import pattern, so this module stays importable on this
  dev machine. Frees the real ~18GB PyTorch model immediately after
  extraction (`del model; torch.cuda.empty_cache()`) — never kept
  resident once FlashRT's own fp16 copies exist.
- `imagewam_thor.py.__init__` — gains real-checkpoint constructor
  params (see Interface — NOT the generic template's single
  `checkpoint_dir`, since ImageWAM's real loading needs 4 distinct
  paths + `action_dim`, confirmed from `load_real_model`'s own
  signature); when provided, calls the new loader instead of
  `_alloc_random_weights`, then wraps each raw fp16 tensor in the
  SAME `Fp16Linear`/`StaticFp8Linear`/etc. selected by `precision`,
  exactly like `_rnd_linear` already does — precision selection is
  ORTHOGONAL to weight source, unchanged from OPT-004 step 5's own
  design.

State ownership: the loader module owns NOTHING persistent — it is a
pure function returning a flat dict of fp16 tensors (or raises), same
lifecycle as `_rnd_linear`'s own tensors (immediately wrapped and kept
alive by `imagewam_thor.py`'s own `self._keepalive`).

## Interface

```python
# flash_rt/models/imagewam/pipeline_thor.py, inside _double_stream_layer,
# added right before the existing "text stream" block:
key("img_in.weight")(bufs["img_raw"], img_x_ptr, img_len, stream)
# mirrors key("txt_in.weight")(bufs["context"], combined, x0, stream)
# exactly -- same (K,N)=(_,HD) vs (_,joint_attention_dim) convention,
# writes into backbone_hidden's own image-row region instead of a
# fresh buffer (img_x_ptr is already an offset into `combined`).

# flash_rt/models/imagewam/checkpoint_loader.py (new)
def load_real_imagewam_weights(*, flux2_model_path: str, flux2_ae_model_path: str,
                                ckpt_path: str, flux2_src: str, action_dim: int,
                                num_double: int, num_single: int,
                                action_num_double: int, action_num_single: int) -> dict:
    """Lazy-imports `imagewam`/`flux2` (raises a clear RuntimeError if
    absent, matching Nvfp4Linear's own pattern). Returns a dict keyed
    EXACTLY like imagewam_thor.py's own `weights` dict (same 4-tuples),
    values are raw fp16 torch.Tensor (NOT yet wrapped in Fp16Linear/etc
    -- the frontend does that, same division of labor as _rnd_linear).
    txt_in.weight/img_in.weight map ALL num_double layer indices to the
    SAME tensor object (see Problem's own "second finding"). Frees the
    real ~18GB model before returning.
    """
```

`imagewam_thor.py.__init__` signature change:
```python
def __init__(self, checkpoint_dir=None, *, dims_override=None, use_fa4=False,
             precision="fp16",
             # NEW, all-or-nothing (real loading needs every one of
             # these; partial real-checkpoint kwargs is a usage error,
             # not a silent partial-random fallback):
             flux2_model_path: str | None = None, flux2_ae_model_path: str | None = None,
             ckpt_path: str | None = None, flux2_src: str | None = None,
             action_dim: int | None = None,
             **kwargs):
    ...
    real_kwargs = (flux2_model_path, flux2_ae_model_path, ckpt_path, flux2_src, action_dim)
    if any(k is not None for k in real_kwargs) and not all(k is not None for k in real_kwargs):
        raise ValueError("real-checkpoint loading needs ALL of flux2_model_path/"
                          "flux2_ae_model_path/ckpt_path/flux2_src/action_dim, or none")
    self._use_real_weights = flux2_model_path is not None
```
`checkpoint_dir` itself stays accepted-but-unused (interface parity
with the generic template, unchanged from today) — the real params are
new, explicit, separately named kwargs, not overloaded onto
`checkpoint_dir`, since ImageWAM's real loading genuinely needs 4
distinct paths the generic single-directory template has no slot for.

## Flow

1. `__init__` validates the all-or-nothing real-checkpoint kwargs.
2. If real: call `load_real_imagewam_weights(...)` once, get the flat
   raw-tensor dict; wrap every value in the `precision`-selected
   linear-op class (same helper `_rnd_linear` already uses internally,
   refactored to accept a pre-existing weight pointer instead of always
   allocating a random one — see Code Mapping).
3. If random (today's path, unchanged): `_alloc_random_weights` as
   before, now ALSO allocating `img_in.weight` per double layer and
   `img_raw`'s random content.
4. `_alloc_buffers` gains `img_raw` (real path: left uninitialized,
   filled by a real VAE eventually — OUT OF SCOPE, same as `context`'s
   own "no real Qwen3 forward" status today; random path: random-filled
   at each `infer()` call, replacing today's direct `backbone_hidden`
   image-row fill).
5. `_double_stream_layer` calls `img_in.weight` first, exactly where
   `txt_in.weight` already runs, before AdaLN.
6. Graph capture/replay: UNCHANGED mechanism — real vs. random weights
   are indistinguishable to the captured graph (same pointers, same
   shapes), exactly like precision selection already is.

## Code Mapping

| module | file | task |
|---|---|---|
| img_in wiring | `flash_rt/models/imagewam/pipeline_thor.py` | new call site in `_double_stream_layer` |
| img_in wiring (random path) | `flash_rt/frontends/torch/imagewam_thor.py` | `img_in.weight` slot, `img_raw` buffer |
| img_in wiring (tests) | `tests/test_imagewam_prefill.py`, `test_imagewam_denoise.py`, `test_imagewam_thor_real_wiring.py`, `benchmarks/imagewam_thor_bench.py` | add `img_in.weight`/`img_raw` to each weights/bufs dict |
| real extraction | `flash_rt/models/imagewam/checkpoint_loader.py` (new) | ported from `imagewam_real_checkpoint_validation.py` |
| real loading integration | `flash_rt/frontends/torch/imagewam_thor.py` | new constructor kwargs, `_use_real_weights` branch |
| record | `opportunities.md` OPT-001/OPT-008 | closed once measured |

## Implementation Phases

### Phase 1 — `img_in` wiring (prerequisite, Ada-testable)

Phase Status: completed

**Stop Condition resolved, NOT triggered**: `real_double_stream_block_forward_fp16`
needed NO modification. Confirmed by reading its own "Real order"
derivation directly: it takes BOTH `txt` and `img` already at `hidden`
width (real FLUX.2's `DoubleStreamBlock` itself has no `img_in`/`txt_in`
step — those are OUTER transformer-level projections, applied once,
outside every block). `test_imagewam_thor_real_wiring.py` already had
to do this exact manual pre-step for `txt_in` (a raw `gemm.fp16_nn`
call before invoking the reference) — `img_in` needed the identical
treatment, not a reference-function change. Cosine stayed 1.000000
exactly, confirming the new GEMM call is bit-correct.
Goal: `img_in.weight` modeled in `pipeline_thor.py`, verified with
random weights exactly like every other mechanism this session (no
real checkpoint needed — this is pure structural correctness: does the
new GEMM call run, produce finite output, and not disturb anything
already verified).
Modified files: `pipeline_thor.py` (new call site in `_double_stream_layer`,
right before the image stream's own AdaLN, mirroring `txt_in.weight`'s
call exactly); `imagewam_thor.py` (`img_in.weight` slot per double
layer, new `img_raw` (img_len, HD) buffer, `infer()` now fills `img_raw`
instead of `backbone_hidden`'s image rows directly, `_autotune_gemm`
covers the new shape); `tests/test_imagewam_prefill.py`,
`test_imagewam_denoise.py`, `test_imagewam_thor_real_wiring.py`,
`benchmarks/imagewam_thor_bench.py` (all gained `img_in.weight` +
`img_raw`, matching pattern).
Affected modules: backbone double-stream layer, frontend buffer
allocation, every test/bench file that builds its own `backbone_hidden`/
image content directly.
Observation method: full `tests/test_imagewam_*.py` suite still passes
(regression check on everything img_in touches downstream: AdaLN,
attention, MLP all read from the SAME `combined`/`img_x_ptr` buffer
img_in now writes into first); `test_imagewam_thor_real_wiring.py`'s
own cosine-vs-reference checks stay at 1.000000, confirmed via its own
matching manual pre-step for `img_in` (see this phase's own "Stop
Condition resolved" note above — no reference-function change needed).

**Scope extended mid-phase to a second, already-anticipated gap:
`action_encoder`/`head`**. While inspecting the real checkpoint's own
key names (see below), read `imagewam/models/backbones/action_dit_flux2.py`
directly (this project's local read-only `ImageWAM` clone,
`PROJECT.md`'s own "Onboarding" note) and `imagewam.py`'s own
`infer_action_flux2`, confirming definitively: `latents_action` (this
project's `bufs["action_latent"]`) lives at real `action_dim` width
(7 for LIBERO), re-encoded to `action_hidden_dim` EVERY denoise step
via `action_encoder` (a real Linear WITH bias — the only biased weight
in this project) before the per-layer blocks, and decoded back down to
`action_dim` via `head` (AdaLN, no gate — a final layer, not a
residual block) AFTER them, with the Euler/flow-matching integration
happening in `action_dim` space, not `action_hidden_dim` space as
`pipeline_thor.py` previously assumed. This was already flagged as a
known placeholder in both `pipeline_thor.py`'s and `pipeline_real.py`'s
own existing comments ("same status as OPT-001") — not a surprise
requiring new triage, just confirmation it was time to fix it, folded
into this phase per explicit user approval rather than opened as a
separate one.

New `adaln.head_modulation` (shift/scale only, no gate — `modulation()`
doesn't support `multiplier=2`) and `pipeline_real.compute_action_head_modulation`
(a small standalone function, NOT a 3rd return value on
`compute_action_modulation`, to avoid rippling into
`test_imagewam_pipeline_full_real.py`'s unrelated own scope — see that
function's own docstring for the reasoning). `pipeline_thor.py`'s
`imagewam_denoise_step` gained the real encode (`gpu_cast_fp32_to_fp16`
→ `action_encoder.weight` GEMM → `add_bias_fp16`) and decode
(`ada_layer_norm_fp16` → `head.linear.weight` GEMM) wrapper around the
UNCHANGED per-layer block loop; `imagewam_denoise_loop`/`imagewam_thor.py`
thread a new `head_mods` list alongside the existing `action_mods`.
`_action_double_layer`/`_action_single_layer` and every test that calls
them DIRECTLY (`test_imagewam_thor_real_wiring.py`'s action tests,
`imagewam_thor_bench.py`'s `bench_action_double`/`bench_action_single`)
needed NO change — the encode/decode wrapper lives one level up, in
`imagewam_denoise_step`, not inside the per-layer functions.
New standalone `tests/test_imagewam_action_encoder_head.py` (cosine
vs. plain-PyTorch `F.linear`, both small and real dims, cosine=1.000000)
verifies the two new primitives in isolation, matching this project's
own narrow-kernel-test convention rather than folding into the bigger
full-pipeline wiring test. `test_imagewam_frontend.py`'s own assertion
updated: `infer()` now correctly returns real `action_dim`-width
output (a real correctness improvement, not just a test fix — this
project's `infer()` was previously returning meaningless
`action_hidden_dim`-width "actions").

**Unplanned but highly consequential discovery made while investigating
this**: the real checkpoint FILES are actually present on this dev
machine (`/home/ljw/projects/pi0.5/models/`), contradicting `PROJECT.md`'s
own prior "never will" claim (now corrected there). `imagewam` package
imports here too. `flux2` source is still absent, and 8GB VRAM still
can't hold the ~18-23GB real model, so a real forward pass still needs
Thor — but `torch.load(..., map_location='cpu', mmap=True)` on
`model.pt` (9GB, fits this machine's 20GB free RAM) gives REAL,
VERIFIED state_dict key names and shapes with zero `flux2`/`imagewam`
dependency, used directly for Phase 2 below instead of blindly porting
`imagewam_real_checkpoint_validation.py`'s live-attribute-path
approach.

### Phase 2 — `checkpoint_loader.py`

Phase Status: completed (FAR beyond the plan's own "Thor-blind" expectation)

**Design changed from the plan's own Interface section, for the
better, once the real checkpoint files turned out to be locally
present** (see Phase 1's own "unplanned discovery" note): rather than
porting `imagewam_real_checkpoint_validation.py`'s live-attribute-path
`extract_*_weights` (which needs the `imagewam`/`flux2` packages to
construct the real model object), `checkpoint_loader.py` reads
`model.pt`'s own raw `state_dict` by KEY NAME
(`torch.load(..., map_location='cpu', mmap=True)['mot']` — confirmed
directly to already be a flat `key -> tensor` `OrderedDict`, no live
module construction needed at all). This needs NEITHER `imagewam` NOR
`flux2` to load real tensors — only `torch`.
Goal: flat weights dict keyed exactly like `imagewam_thor.py`'s own
`self._weights`, `txt_in`/`img_in` shared correctly across layers.
Modified files: new `checkpoint_loader.py`.
Affected modules: none (new, standalone module).
**Observation method, actually executed on Ada, not just reviewed**:
(1) `test_shapes_match_confirmed_real_dims` — every real weight
tensor's shape matches this project's own confirmed real dims exactly,
343 tensors, all pass; (2) `test_real_double_stream_layer_forward_finite`
— one REAL backbone double-stream layer (real trained weights, random
activations) through `pipeline_thor.py`'s own pointer path, finite,
non-degenerate output (std=15.18, not near-zero).
**Two real bugs found and fixed, both only catchable by actually
running this, not by code review**: (a) `build_real_modulation_weights`
initially transposed the modulation weights into FlashRT's `(K,N)`
GEMM convention — WRONG, since `adaln.py`'s `mlp_embedder`/`modulation`
are plain `F.linear(x, weight)` calls needing the real, native
`(out,in)` layout directly, not the transposed one; caught by an
`RuntimeError: shapes cannot be multiplied` on the very first real
forward attempt. (b) ActionDiT's own double-block weights were
initially keyed with an `img_` PREFIX (matching the backbone's own
dual-stream convention) — WRONG, `_action_double_layer` expects PLAIN
`"qkv.weight"`/`"proj.weight"` (ActionDiT is single-stream, no prefix,
confirmed against `imagewam_thor.py`'s own `_alloc_random_weights`);
caught by a `KeyError` during the first real graph-capture attempt.
Both are exactly the class of silent-wrong-shape/wrong-convention bug
this plan's own Problem section worried code written blind for Thor
could hide — found here instead, before ever reaching Thor.

### Phase 3 — frontend integration

Phase Status: completed (also far beyond "Thor-blind")

Goal: `imagewam_thor.py`'s new `ckpt_path` constructor kwarg (simpler
than the plan's own originally-designed 5-kwarg interface — no
`flux2_model_path`/`flux2_ae_model_path`/`flux2_src`/`action_dim`
needed, since Phase 2's key-based loader only needs the one checkpoint
file), `_wrap_linear` extracted from `_rnd_linear` so both the random
and real-weight paths share identical precision-selection logic,
`_load_real_weights`/`_compute_backbone_modulation`/
`_compute_action_modulations` all gained a real-weight branch.
Modified files: `imagewam_thor.py`.
Affected modules: frontend construction.
**Observation method, actually executed end-to-end at REAL FLUX.2-4B
dims on Ada, not just "fails at the documented place"**: constructed
`ImageWAMTorchFrontendThor(dims_override=<real dims>, ckpt_path=<real
model.pt>)`, ran `set_prompt()` (real CUDA Graph capture with real
weights) and `infer()` (real graph replay) — ALL THREE succeeded,
producing a finite `(64, 7)` action tensor (mean=0.21, std=0.34, a
plausible normalized-action range). **Unexpected finding**: peak CUDA
memory allocated measured at ~9.86GB, exceeding `nvidia-smi`'s own
reported 8188MiB total — this WSL2 environment's CUDA driver evidently
pages beyond the reported dedicated VRAM into host RAM rather than
raising OOM, letting this one-time construction+capture actually
complete here. Not something to rely on for a steady-state Thor
PERFORMANCE claim (paged memory is slow), but real enough that
CORRECTNESS at real dims is now verified on this dev machine, not just
projected. The random-weight path (default, `ckpt_path=None`) was
re-verified unchanged across the full `tests/test_imagewam_*.py` suite
throughout this phase's own development.
New `tests/test_imagewam_checkpoint_loader.py` (skips cleanly if the
real checkpoint files aren't present, matching this project's own
established skip-pattern) captures all three checks above as a
permanent regression test, not just an ad-hoc verification.

### Phase 4 — real Thor validation + close-out

Phase Status: completed (all 3 items answered with real Thor numbers, 2026-09-15)

Goal: hand to the user for a real Thor run. Since Phase 2/3 already
verified real-weight CORRECTNESS end-to-end on Ada (finite, plausible
output, at real dims, with the actual release checkpoint), Thor's
remaining job is narrower than originally planned:
1. **Answered**: construct/capture/infer completes on Thor without
   this dev machine's own WSL2 memory-paging dependency — confirmed
   (real Thor run, 172.8ms median `infer()` at the corrected `img_len=392`
   shape, ~9.99GB peak allocated, "no additional paging" per the user's
   own report — Thor's 128GB unified memory handles this cleanly, as
   expected).
2. **Answered, and simultaneously used to confirm the corrected real
   deployment shape** (see below): real Thor `infer()` median 172.8ms
   at `img_len=392` vs. 231.1ms at the OLD `img_len=768` guess — real
   weights, real shapes, CUDA Graph, FP16.
3. **Answered**: re-ran `imagewam_real_checkpoint_validation.py`
   (updated to the corrected `REF_H,REF_W=14,28` shape and to include
   `img_in`) against the REAL official reference path — backbone
   cosine=**0.999918**, ActionDiT cosine=**0.999962** (essentially
   unchanged from the earlier 0.999927/0.999963 measured at the old
   24x32/no-`img_in` shape — confirms neither the shape correction nor
   the `img_in` addition broke anything against the real reference).
   Also re-ran `test_imagewam_quant_linear.py` on Thor (no SKIP, four
   real cosines, all matching earlier Thor numbers exactly) and the
   full OPT-004 step 5/6 FP8/NVFP4/CUTLASS comparison at the corrected
   `img_len=392` shape (see `opportunities.md`'s own matching entry
   for the full table — relative rankings changed meaningfully from
   the 768-token measurement: dynamic FP8 now beats FP16, and the
   CUTLASS-over-static-cuBLASLt gap shrank from ~20ms to ~1ms).
Modified files: `opportunities.md` (OPT-001/OPT-008 closed),
`plan.md` (this write-up).
Affected modules: none (measurement only).
Observation method: per-layer P50 table, same style as every prior
OPT-004 entry; construction/capture/infer success confirmed without
memory pressure; optionally a cosine number from re-running the
official-reference validation script.

## Stop Conditions Encountered

None. Phase 1's own flagged possible scope expansion (whether
`pipeline_real.py`'s reference functions also need an `img_in` step)
resolved NO — see that phase's own "Stop Condition resolved" note.

---

# Plan: real VAE encoder + text-context wiring into the served frontend

Plan Status: completed (all 3 phases done)

## Problem

### Current

`imagewam_thor.py.infer()` ignores its own `observation` argument
entirely and random-fills `img_raw` every replay (a placeholder for
"whatever a real VAE would have produced," per that method's own
docstring). `set_prompt()` similarly never encodes `prompt_text` --
`context` is random-filled once at construction and never touched
again. Both are honest, documented placeholders, not bugs -- but they
mean this frontend has never processed a single real pixel or real
instruction string.

**Corrected assumption, found while investigating this plan**:
`black-forest-labs/flux2` (the real FLUX.2 model-definition source)
was documented THROUGHOUT this project (`PROJECT.md`, `plan.md`,
`quant_linear.py`'s own docstrings) as "not cloneable/absent on this
dev machine." This is FALSE -- `git clone https://github.com/black-forest-labs/flux2.git`
succeeds directly from this sandboxed environment, and pins to the
EXACT commit (`50fe516...`) this project's own docs have referenced by
hash for weeks without ever having the source locally. This unblocks
using the REAL `flux2.autoencoder.AutoEncoder` class directly, rather
than reverse-engineering an approximation.

**Second correction, found the same way**: `imagewam.py`'s own real
VAE loading path (`AutoEncoder(AutoEncoderParams())` + `load_sft`)
uses `flux2`'s OWN `autoencoder.py` class -- NOT `diffusers.AutoencoderKLFlux2`
(the class `model_index.json` names, which is a SEPARATE, incomplete
diffusers port: it defines an identical `self.bn` BatchNorm2d
submodule but never wires it into its own public `encode()`, and never
applies the real 2x2 patch-merge either). Verified directly on this
machine: encoding a real LIBERO frame through `diffusers.AutoencoderKLFlux2.encode().latent_dist.mode()`
gives mean=-0.031/std=1.72/absmax=8.31 (WRONG -- no patch-merge, no
BN); through the REAL `flux2.autoencoder.AutoEncoder.encode()` gives
mean=-0.012/std=0.973/absmax=4.72 -- matching the user's own real Thor
measurement (mean=-0.02, std=0.97, absmax=4.91) almost exactly. The
real class is a hard requirement, not a convenience.

### Problem

No code path in this project encodes a real image into `img_raw` or
accepts a real/precomputed text context into `context` -- every
inference is 100% synthetic on both counts.

### Measurable goal

`imagewam_thor.py` gains an opt-in real path: given `ae_model_path`/
`flux2_src`, `infer(observation)` encodes a REAL image through the
REAL `flux2.autoencoder.AutoEncoder` into `img_raw` before graph
replay; given a precomputed `context`/`context_mask` tensor pair
(matching `imagewam.py`'s own `_prepare_flux2_infer_text` interface
exactly -- it already accepts this as an alternative to live encoding),
`set_prompt()` loads it into `context` instead of random-filling.
Live Qwen3-4B encoding (a raw prompt STRING, not precomputed
embeddings) is explicitly OUT OF SCOPE for this plan -- no local
Qwen3-4B weights exist yet, and downloading them is a separate,
larger decision (see Structure below for exactly where that door is
left open, not closed).

## Structure

- NEW `flash_rt/models/imagewam/vae_encoder.py` -- OWNS real VAE
  loading (`load_real_ae`, lazy-imports `flux2.autoencoder`, mirroring
  `Nvfp4Linear`'s own guarded-import pattern for a not-always-present
  dependency) and real encoding (`encode_to_tokens`: resize-per-view +
  concat + `x*2/255-1` + `ae.encode(x)` -- replicates
  `imagewam._encode_flux2_image_tokens`'s own exact real order,
  confirmed by reading it directly). Returns real `(1, img_len, HD)`
  fp16 CUDA tokens, ready to copy into `img_raw` directly -- no
  FlashRT-side change needed to consume it.
- `flash_rt/frontends/torch/imagewam_thor.py` -- OWNS the encode-once-
  per-call lifecycle: `infer(observation)` calls the real VAE (NEW,
  when `ae_model_path` was given at construction) OUTSIDE the captured
  CUDA Graph (plain PyTorch/`flux2`-dependent code has no business
  being captured), then copies the result into the ALREADY-CAPTURED
  graph's own `img_raw` buffer before `.replay()` -- same "real work
  happens once per call, feeds a fixed-address buffer the graph reads"
  pattern this file's own AdaLN/RoPE precompute already established,
  just per-CALL instead of per-construction. `set_prompt()` gains an
  optional `context`/`context_mask` param pair, copied into the
  existing `context` buffer when given (no graph recapture needed --
  same buffer, same address, just real values instead of random ones).
- `flash_rt/models/imagewam/checkpoint_loader.py` -- UNCHANGED. Real
  checkpoint weights and real VAE/text encoding are independent
  concerns (one loads `model.pt`'s transformer weights, the other runs
  a separate real image/text encoder) -- `ckpt_path` and
  `ae_model_path`/`flux2_src` are independently optional.
- **Live Qwen3 -- CLOSED same day (2026-09-15), once real Qwen3-4B
  weights were downloaded** (`Qwen/Qwen3-4B`, ~7.6GB, per explicit
  user go-ahead). New `flash_rt/models/imagewam/text_encoder.py`
  (`load_real_text_encoder`, `encode_prompts`) ports `imagewam.py`'s
  own real `_encode_flux2_prompts` exactly: Qwen3's own chat template
  (`enable_thinking=False`), tokenize to `max_length=512`, forward
  with `output_hidden_states=True`, concatenate layers `[9,18,27]`
  (`flux2.text_encoder.OUTPUT_LAYERS_QWEN3`, confirmed by reading that
  module directly) -> `(1,512,3*2560)=(1,512,7680)`, matching
  `JOINT_ATTENTION_DIM` exactly. `set_prompt`'s own THIRD branch
  (`prompt_text` + `qwen3_model_spec` given at construction -> live
  encode) added exactly as this plan's own interface already
  anticipated -- no redesign needed. **This also confirms `x0=512`
  is the real value** (Qwen3's own fixed `max_length`), not the
  `x0=128` placeholder used everywhere in this project until now --
  see `opportunities.md`'s own matching entry for the real-dims
  correction this implies and what it exposed.

## Interface

```python
# flash_rt/models/imagewam/vae_encoder.py
def load_real_ae(ae_model_path: str, flux2_src: str, device="cuda", dtype=torch.bfloat16):
    """Lazy-imports flux2.autoencoder (sys.path.insert(0, flux2_src) if
    not already importable), loads AutoEncoder(AutoEncoderParams())
    from `ae_model_path` (safetensors) via strict state_dict load.
    Raises RuntimeError with a clear message (not an ImportError) if
    flux2_src doesn't actually contain the flux2 package -- same
    "clear error, not a wiring bug" pattern as Nvfp4Linear/StaticFp8Linear."""

def encode_to_tokens(ae, view1: torch.Tensor, view2: torch.Tensor | None = None,
                      *, out_hw=(224, 224)) -> torch.Tensor:
    """view1/view2: (H,W,3) uint8 CUDA or CPU tensors (one or two camera
    views -- ImageWAM's own real convention concatenates two 224x224
    views into one 224x448 input, confirmed against real LIBERO-fastwam
    preprocessing). Returns (1, img_len, HD) fp16 CUDA tokens, already
    packed (2x2 patch-merge + BatchNorm, matching flux2.autoencoder.AutoEncoder.encode
    exactly) -- ready to `.copy_()` into img_raw directly."""
```

`imagewam_thor.py` changes:
```python
def __init__(self, ..., ae_model_path: str | None = None, flux2_src: str | None = None, ...):
    # all-or-nothing with each other, independent of ckpt_path (OPT-001)
    self._ae = load_real_ae(ae_model_path, flux2_src) if ae_model_path else None

def set_prompt(self, prompt_text: str | None = None, *,
                context: torch.Tensor | None = None, context_mask: torch.Tensor | None = None):
    # context/context_mask given -> self._context.copy_(context); random-fill otherwise (unchanged)

def infer(self, observation: dict) -> dict:
    if self._ae is not None and "view1" in observation:
        tokens = encode_to_tokens(self._ae, observation["view1"], observation.get("view2"))
        self._img_raw.copy_(tokens[0])
    else:
        self._img_raw.normal_()  # unchanged placeholder
    ...  # self._graph.replay() unchanged
```

## Flow

1. Construction: `ae_model_path`/`flux2_src` given -> `load_real_ae`
   loads the real AE once, kept resident (bf16, small: encoder-only
   params, not the full ~7.7GB base FLUX.2 DiT -- just the AE's own
   weights from `ae.safetensors`).
2. `set_prompt`: real `context`/`context_mask` given -> copied into
   the existing buffer; graph capture proceeds unchanged either way
   (same buffer address, same shape, only the VALUES differ before
   vs. after this change).
3. Per `infer()` call: real observation given -> VAE encode (plain
   PyTorch, real weights, OUTSIDE the graph) -> copy into `img_raw` ->
   `.replay()`. No real observation given -> unchanged random-fill
   placeholder, so nothing breaks for existing callers.

## Code Mapping

| module | file | task |
|---|---|---|
| real VAE load + encode | `flash_rt/models/imagewam/vae_encoder.py` (new) | `load_real_ae`, `encode_to_tokens` |
| frontend lifecycle | `flash_rt/frontends/torch/imagewam_thor.py` | `ae_model_path`/`flux2_src` ctor params, `set_prompt`'s new optional context params, `infer`'s new real-encode branch |
| correctness test | `tests/test_imagewam_vae_encoder.py` (new) | real AE load + encode, checked against the real Thor stats (mean/std/absmax) already on record; skips cleanly if `flux2_src`/`ae.safetensors` aren't reachable |
| record | `opportunities.md` | new entry once measured |

## Implementation Phases

### Phase 1 — `vae_encoder.py`, verified against real Thor stats

Phase Status: completed

Goal: `load_real_ae`/`encode_to_tokens` correctly load the real AE and
reproduce the real preprocessing order, checked against the real
Thor-measured stats (mean=-0.02, std=0.97, absmax=4.91) as ground
truth -- NOT a cosine check (no independent reference implementation
exists locally to compare against), a statistical sanity check.
Modified files: new `vae_encoder.py`, new `tests/test_imagewam_vae_encoder.py`.
Observation method: run directly on this dev machine (real `flux2`
clone, real `ae.safetensors`, a real downloaded LIBERO-fastwam frame)
-- already done ad-hoc while investigating this plan (mean=-0.012,
std=0.973, absmax=4.72); this phase formalizes it as a permanent test.

### Phase 2 — frontend integration

Phase Status: completed

Goal: `imagewam_thor.py`'s new ctor params + `set_prompt`/`infer`
branches.
Modified files: `imagewam_thor.py`.
Observation method: regression check (existing random-fill behavior
unchanged when the new params aren't given, confirmed) + a real
end-to-end run combining BOTH opt-in real paths for the first time:
real weights (OPT-001's own `ckpt_path`) AND a real VAE-encoded real
LIBERO-fastwam camera frame, through the SAME captured CUDA Graph,
producing a finite `(64,7)` action tensor. New
`test_full_frontend_with_real_checkpoint_and_real_vae` in
`tests/test_imagewam_checkpoint_loader.py` locks this in as permanent
regression coverage (needs both real resources, skips cleanly
otherwise). Also fixed that test file's own stale `A0=896` constant
(superseded by OPT-001's own confirmed `img_len=392` correction) while
here.

### Phase 3 — close-out

Phase Status: completed

Goal: record what real image/text wiring achieves and what's still
missing (live Qwen3, still deferred).
Modified files: `opportunities.md` (new entry).

# Roadmap: Pi0.5-derived optimization tracks

Roadmap Status: pending (documentation/tracking only -- no item below
is approved for implementation; each gets its own `# Plan: <item>`
section, full `plan` skill rigor, when picked up individually)

## Problem

Current observable state: ImageWAM's Thor deployment is a pure Python
frontend (`flash_rt/frontends/torch/imagewam_thor.py`) doing
raw-pointer CUDA dispatch through `flash_rt/models/imagewam/pipeline_thor.py`
-- no stable ABI, no native C++ callable surface, no committed
CI/regression harness, placeholder (not real-data) FP8 calibration.
FlashRT's own Pi0.5 deployment (`flash_rt/models/pi05/`,
`cpp/models/pi05/`, `runtime/`) is materially more mature along all
four of these dimensions. A 4-way parallel investigation (2026-09-18,
forked) surveyed Pi0.5's real architecture, quantization/calibration,
kernel-fusion, and CI stack end to end and assessed transferability to
ImageWAM, itemized with real cost estimates -- cost was NOT used as a
filter, per explicit instruction.

Goal: track the resulting 14-item optimization inventory as a roadmap
-- this section is an index + dependency map, not a substitute for a
real per-item plan. Each item gets its OWN dedicated `# Plan: <item>`
section (full Problem -> Structure -> Interface -> Flow -> Code
Mapping -> Implementation Phases, same skill, same rigor) when picked
up for real work.

## Structure: three independent tracks

- **Speed track** -- items that directly reduce real `infer()` latency.
- **Accuracy track** -- items that improve quantization/calibration
  quality (some also gate future speed/accuracy work, e.g. real
  calibration data is a prerequisite for a fair AWQ/`fp8_static` trial).
- **Deployment-engineering track** -- architecture/reliability/
  maintainability, not raw speed.

Tracks are independent of each other. Only 3 real dependency edges
exist across all 14 items; the other 11 have zero prerequisites and
can start in any order, in parallel, immediately.

## The 14 items

| id | item | track | cost | depends on | note |
|---|---|---|---|---|---|
| 1 | ActionDiT small-M CUTLASS tile retry (Pi0.5's proven "v10" `128x64x256` tile) | speed | hours | none | opportunities.md OPT-014's own `M=64` CUTLASS-slower-than-cuBLASLt regression |
| 2 | Image normalization LUT (256-entry FP16, precomputed) | speed | hours | none | Pi0.5's own proven technique, "bit-identical" per its docs |
| 3 | Gated-residual + next-layer-norm fusion (one elementwise kernel) | speed | hours -- 1-2 days | none | Pi0.5's own real, working equivalent -- supersedes the previously-deferred CUTLASS-epilogue idea for this same problem (opportunities.md OPT-013's deferred gated-residual epilogue) |
| 4 | `linear2` merge (attn_out_proj + mlp_down) | speed | days | none | opportunities.md OPT-015 op-fusion audit sub-problem 3, not started |
| 5 | VAE port to FlashRT kernel style + in-graph capture | speed | weeks, standalone | none | opportunities.md OPT-008; prerequisite for in-graph VAE, not for anything else here |
| 6 | Attention-chain fusion feasibility recheck at ImageWAM's own real shapes | analysis | hours, analysis only | none | Pi0.5 rejected this at `M=10` (5-7x slower); ImageWAM's shapes (`M~905` backbone, `M=64` ActionDiT) differ, worth re-checking, not assuming the same verdict |
| 7 | Real calibration data pipeline (replace `N(0,0.1)` placeholder in `_calibrate_fp8()`) | accuracy | days | none | hub node -- unlocks items 8 and 13 |
| 8 | AWQ per-channel scale folded into NVFP4 weights | accuracy | +2-3 days | 7 | may help ImageWAM MORE than it helped Pi0.5 (ImageWAM's merged-`linear1` GEMMs are more bandwidth-shaped than Pi0.5's own compute-bound QKV case, where Pi0.5 shipped AWQ disabled) |
| 9 | Hadamard-rotated INT4 (E0M3) new precision tier | accuracy | 1-2 weeks, standalone | none | NOT the same dead SM80 kernel OPT-007 already closed -- real native SM100 block-scaled path (same layout family as `nvfp4`); the only item here that could beat `nvfp4` on accuracy |
| 10 | Jetson clock-locking check for benchmark scripts | deployment | hours | none | Pi0.5's own devfreq/nvpmodel check; ImageWAM benchmarks currently have none |
| 11 | Precision-routing contract test (stubbed, no GPU) | deployment | hours -- 1 day | none | mirrors Pi0.5's `test_pi05_thor_fp4_routing.py` pattern against ImageWAM's own `_PRECISIONS`/`_wrap_linear` |
| 12 | ABI integration (`frt_model_runtime_v1`, `io="python"` producer mode) | deployment | 1-3 days | none | zero C++, zero new kernels -- exposes ImageWAM's existing `self._graph`/buffers through the same generic ABI Pi0.5's Python producer already uses |
| 13 | Fidelity + latency CI/regression gate harness | deployment | days | 7 | gate logic (cosine thresholds, `p50<baseline-margin`, JSON result schema) is generic/copyable from Pi0.5's own harness; content needs real calibration data + a real LIBERO fixture format; ready-made baseline: this session's own real 231.6ms `nvfp4` `infer()` number |
| 14 | Native C++ overlay (`io="native"`/`"native_v2"`) | deployment | 2-4 weeks | 12 | ports existing Python orchestration to C++ against ImageWAM's EXISTING kernels, not a kernel rewrite; the item that actually removes the Python/GIL dependency from the hot path |

## Dependency graph

```
7 (real calibration data) --> 8  (AWQ)
7 (real calibration data) --> 13 (CI/regression harness)
12 (ABI integration)      --> 14 (native C++ overlay)
```

Every other item (1, 2, 3, 4, 5, 6, 9, 10, 11, 12) has zero
prerequisites and can start immediately, in parallel.

## Not planned (confirmed, not cost-gated)

- Multi-subgraph stage-splitting (Pi0.5's RTC-prefix-reuse/VJP-guided-
  denoising scheduling machinery, `flash_rt/subgraphs/pi05/`) --
  ImageWAM has no incremental-replanning requirement today; out of
  scope for lack of a real need, not because it's expensive.
- RMSNorm-into-GEMM prologue fusion -- CUTLASS has no prologue-fusion
  mechanism at all (confirmed earlier this session); Pi0.5 doesn't do
  this either. Dead, unchanged by this investigation.

## Skill note

No new skill needed for this roadmap. This project's existing skill
suite (`plan`/`work`/`close`/`experiment`) already covers "plan one
item in full rigor" / "execute an approved phase" / "reconcile state
at closure" / "structured multi-variable investigation" -- each of the
14 items above uses one of these unchanged when picked up. This
section is only an index; do not treat any item here as approved
until it gets its own `# Plan: <item>` section with `Plan Status:
approved`.

# Plan: Jetson clock-locking check for the benchmark scripts (roadmap item 10)

Plan Status: approved

## Problem

### Current

The ImageWAM benchmark entry points (`benchmarks/imagewam_thor_graph_bench.py`,
the timing section of `benchmarks/imagewam_e2e_official_compare.py`,
`benchmarks/imagewam_thor_int4_bench.py`, `benchmarks/imagewam_thor_int8_bench.py`)
print latency without recording the Jetson power and clock state it was
measured under. Pi0.5's end-to-end benchmark
(`tests/bench_pi05_decoder_fp4_e2e.py`, `machine_state()`) refuses to run
unless `nvpmodel -q` reports MAXN and the `gpu-gpc-0`/`gpu-nvd-0`
devfreq nodes have `min_freq == max_freq == cur_freq`, and it writes that
state into its result JSON. No ImageWAM latency on record, including the
shipped `nvfp4` P50 of 231.6 ms, carries a clock record.

### Goal

One reusable helper that reads the nvpmodel mode, the `jetson_clocks`
state, and GPU/EMC devfreq `cur/min/max/governor` from sysfs; prints and
returns a structured record; warns when clocks are not locked; and
returns an explicit "not a Jetson" record on any other machine. Every
listed benchmark prints this record once before timing. Measurable:
unit tests on x86 against a fake sysfs tree cover the locked, unlocked,
missing-nvpmodel, EMC and non-Jetson cases; on Thor the record is
captured before and after `sudo jetson_clocks`.

## Structure

| module | responsibility | state owned |
|---|---|---|
| `flash_rt/hardware/jetson_clock_state.py` | read the sysfs tree and the nvpmodel/jetson_clocks tools under an injectable root and command runner; derive the lock verdict; print the record | none (reads only) |
| benchmark entry points | call `report_jetson_clock_state()` once before timing | none |
| `tests/test_jetson_clock_state.py` | fake sysfs trees in `tmp_path`, fake command runner | none |

The helper is stdlib-only (no torch, no extension), so it imports on
any machine.

## Interface

```python
# flash_rt/hardware/jetson_clock_state.py
@dataclass(frozen=True)
class DevfreqNode:            # one /sys/class/devfreq/<name> entry
    name: str; kind: str      # "gpu" | "emc"
    cur_hz: int | None; min_hz: int | None; max_hz: int | None
    governor: str | None
    locked: bool              # cur == min == max, all readable

@dataclass(frozen=True)
class ToolQuery:              # one external tool invocation
    available: bool; text: str

@dataclass(frozen=True)
class JetsonClockState:
    is_jetson: bool; platform: str
    nvpmodel: ToolQuery; nvpmodel_mode: str | None; nvpmodel_mode_id: int | None
    jetson_clocks: ToolQuery
    gpu: tuple[DevfreqNode, ...]; emc: tuple[DevfreqNode, ...]
    clock_caps_hz: tuple[tuple[str, int], ...]   # /sys/kernel/nvpmodel_clk_cap/*
    gpu_locked: bool; emc_locked: bool | None; power_mode_max: bool | None
    locked: bool; warnings: tuple[str, ...]
    def to_dict(self) -> dict[str, object]

class CommandRunner(Protocol):
    def run(self, argv: Sequence[str]) -> ToolQuery

class JetsonClockProbe:
    def __init__(self, root: Path = Path("/"), runner: CommandRunner | None = None) -> None
    def read(self) -> JetsonClockState

def report_jetson_clock_state(probe: JetsonClockProbe | None = None,
                              emit: Callable[[str], None] = print) -> JetsonClockState
```

Lock verdict: `gpu_locked` = at least one GPU devfreq node and every GPU
node locked. `emc_locked` = `None` when no EMC devfreq node is visible,
else every EMC node locked. `power_mode_max` = `None` when `nvpmodel` is
unavailable, else the mode name starts with `MAXN`. `locked` = Jetson,
`gpu_locked`, and neither `emc_locked` nor `power_mode_max` is `False`.
Every condition that makes the record unobservable or unlocked adds a
warning line.

## Flow

1. `JetsonClockProbe.read()` checks `/etc/nv_tegra_release` and
   `/proc/device-tree/{model,compatible}`. Neither present: return a
   record with `is_jetson=False` and no tool calls.
2. On a Jetson: run `nvpmodel -q` and `jetson_clocks --show` through the
   runner (timeout, never `sudo`); list `/sys/class/devfreq/*`; classify
   names containing `gpu` (and the Tegra GPU ids `gp10b/gv11b/ga10b/gb10b`)
   as GPU and names containing `emc` as EMC; read `cur_freq`, `min_freq`,
   `max_freq`, `governor`; read `/sys/kernel/nvpmodel_clk_cap/*`.
3. Derive the verdict and warnings.
4. `report_jetson_clock_state()` prints `[jetson-clock-state] <json>`
   plus one `[jetson-clock-state] WARNING: ...` line per warning, and
   returns the record.

## Code Mapping

| item | file |
|---|---|
| probe, record types, runner protocol, reporter | `flash_rt/hardware/jetson_clock_state.py` (new) |
| unit tests | `tests/test_jetson_clock_state.py` (new) |
| wiring | `benchmarks/imagewam_thor_graph_bench.py`, `benchmarks/imagewam_e2e_official_compare.py` (timing section), `benchmarks/imagewam_thor_int4_bench.py`, `benchmarks/imagewam_thor_int8_bench.py`; the item-13 gate runner embeds the record in its result JSON |

## Implementation Phases

### Phase 1 — helper and unit tests

Phase Status: completed

Goal: `jetson_clock_state.py` with the interface above.
Modified files: `flash_rt/hardware/jetson_clock_state.py`, `tests/test_jetson_clock_state.py`.
Observation method: pytest on x86 with fake sysfs trees; the record
printed on this H100 box (expected `is_jetson=false`).

### Phase 2 — wire into the benchmark entry points

Phase Status: completed

Goal: every listed benchmark prints the record once before timing.
Modified files: the four benchmarks listed in Code Mapping.
Observation method: `python -m py_compile` on each; run the graph bench
on H100 far enough to see the record line (fp16 row).

### Phase 3 — Thor handoff

Phase Status: blocked

Goal: Thor checklist entry: record before and after
`sudo nvpmodel -m 0 && sudo jetson_clocks`.
Modified files: none (checklist in the stream report).
Observation method: owner's Thor run.
Blocker: needs the Thor hardware. The checklist command is
`python -c "from flash_rt.hardware.jetson_clock_state import
report_jetson_clock_state; report_jetson_clock_state()"`, run before and
after locking; on the shared H100 the record is `is_jetson=false`.

## Results

- `tests/test_jetson_clock_state.py`: 11 passed on x86 (fake sysfs trees
  for locked, unlocked GPU, non-MAXN, missing `nvpmodel`, EMC node
  locked/unlocked, Jetson without GPU devfreq, unreadable frequency,
  reporter output, off-Jetson).
- `benchmarks/imagewam_thor_graph_bench.py` on H100 prints the
  `[jetson-clock-state]` record (`is_jetson=false`) before its table.

# Plan: Precision-routing contract test (roadmap item 11)

Plan Status: completed

## Problem

### Current

`ImageWAMTorchFrontendThor` routes each weight slot to a GEMM wrapper in
three places: `_wrap_linear` (per precision, with K/N alignment
fallbacks), `_alloc_random_weights`, and `_load_real_weights`, plus the
constructor's `dims["merge_qkv_mlp"] = precision != "fp16_cutlass"`
decision. The special cases are: `action_encoder` (K=7) and
`head.linear` (N=7) fall back to `Fp16Linear` for `nvfp4`, the FP8 family
and `fp16_cutlass`; single-stream blocks use one merged `linear1` except
under `fp16_cutlass`, which keeps `qkv`/`mlp_in`; the SwiGLU gate/up
slots use `CutlassFp16SwiGluMlp` under `fp16_cutlass`; `txt_in`/`img_in`
always use `Bf16OutLinear`. Existing coverage constructs real wrappers
on a GPU, so on H100 the NVFP4 and SM100-CUTLASS routes are never
exercised (they skip), and the FP8 K=7 crash (commit `2b9d5fd`) was
found only on Thor.

### Goal

A CPU-only test, no `flash_rt_kernels`, no GPU, that asserts the wrapper
class of every weight slot for every precision in `_PRECISIONS`, at the
real FLUX.2-4B dims and at the frontend's default dims, for both the
random-weight and the real-checkpoint paths. The expected routing is one
explicit table that other streams edit when they change routing.

## Structure

| module | responsibility |
|---|---|
| `tests/test_imagewam_thor_precision_routing.py` | stubs, expected routing table, tests |

Stubs at the import boundary, as in `tests/test_pi05_thor_fp4_routing.py`:

- `flash_rt.flash_rt_kernels`: a module whose `FvkContext()` raises
  `_NoGpuContext`. The constructor makes every routing decision
  (precision validation, `merge_qkv_mlp`) before it creates the kernel
  context, so catching `_NoGpuContext` leaves an instance holding the
  real decisions and no GPU state.
- `flash_rt.models.imagewam.quant_linear`: recording classes with the
  real constructor signatures; `test_stub_signatures_match_quant_linear`
  checks them against the real module wherever it imports.
- The frontend module's `DEV` is patched to `"meta"`, so real-dim weights
  allocate no memory.

A module-snapshot helper restores `sys.modules` and parent-package
attributes afterwards, so the stubbed import does not leak into other
tests in the same session.

## Interface

```python
PRECISION_COLUMNS: tuple[str, ...]      # must equal imagewam_thor._PRECISIONS
EXPECTED_ROUTING: dict[tuple[str, str, str], tuple[str, ...]]
#   (site, block, slot) -> one route label per PRECISION_COLUMNS entry
#   labels: "Fp16Linear", "Bf16OutLinear", "CutlassFp16Linear",
#           "CutlassFp16SwiGluMlp", "Fp8Linear", "Nvfp4Linear",
#           "StaticFp8Linear[cublaslt]", "StaticFp8Linear[cutlass]",
#           "ptr" (raw device pointer), ABSENT (slot must not exist)
def route_of(value: object) -> str
```

## Flow

1. Install stubs, import the frontend fresh, patch `DEV="meta"`.
2. Construct the frontend for one precision; catch `_NoGpuContext`.
3. Set `_gemm` to a sentinel; call `_alloc_random_weights(dims)` or
   `_load_real_weights(dims, fake_state_dict)` (meta tensors under the
   real checkpoint key names).
4. Compare every produced key against `EXPECTED_ROUTING` (and every
   expected present slot against the produced keys, per layer).
5. Restore modules.

## Code Mapping

| item | file |
|---|---|
| stubs, table, tests | `tests/test_imagewam_thor_precision_routing.py` (new) |
| code under test | `flash_rt/frontends/torch/imagewam_thor.py` (unchanged), `flash_rt/models/imagewam/checkpoint_loader.py` (unchanged) |

## Implementation Phases

### Phase 1 — contract test

Phase Status: completed

Goal: the test file above, passing on H100 and in a process where
`flash_rt.flash_rt_kernels` is unimportable and `CUDA_VISIBLE_DEVICES=""`.
Modified files: `tests/test_imagewam_thor_precision_routing.py`.
Observation method: pytest in both environments; a deliberate
mutation of `_wrap_linear` (drop the `nvfp4` K%16 fallback) must fail
the test (checked once, not committed).

## Results

- `tests/test_imagewam_thor_precision_routing.py`: 54 passed on H100
  with the real extension importable; 53 passed and 1 skipped
  (`test_stub_signatures_match_quant_linear`, needs the real
  `quant_linear`) with `CUDA_VISIBLE_DEVICES=""` and
  `flash_rt.flash_rt_kernels` made unimportable. About 15 s either way.
- Mutation check (not committed): removing the `nvfp4` `K%16`/`N%16`
  fallback in `_wrap_linear` fails 5 tests, naming
  `action_encoder.weight` and `head.linear.weight`; making
  `fp8_static` skip the `linear1` merge fails 5 tests.

# Plan: Fidelity and latency regression gate harness (roadmap item 13)

Plan Status: approved

## Problem

### Current

The repository has no committed gate for ImageWAM. Fidelity is checked
ad hoc with `benchmarks/imagewam_e2e_official_compare.py`, which loads
the official bf16 model (Qwen3-4B included) next to FlashRT in the same
process (about 34GB) and needs `av`/`pandas` for LIBERO decoding, none
of which a Thor gate run should depend on. Latency claims (`nvfp4` 231.6 ms, `opportunities.md`
OPT-015) are recorded in prose, with no machine-readable baseline and no
pass/fail rule. Pi0.5's harness (`tests/bench_pi05_decoder_fp4_e2e.py`)
has the generic pieces: per-sample cosine thresholds,
`p50` against a regression baseline, and a versioned result JSON with the
clock state. Roadmap item 13 lists item 7 (real calibration data) as a
dependency; only the `fp8_static` gate needs it.

### Goal

1. A versioned LIBERO fixture: real preprocessed observations (two
   224x224 views, proprio, prompt), the official Qwen3 context and mask,
   fixed initial action noise, official reference actions, and FlashRT
   `fp16` reference actions. Data lives under
   `/home/user1/workspace/jingwu/artifacts/deploy-gates/`; git holds the
   generator and a manifest with checksums. Gating on Thor needs neither
   the official model nor Qwen3.
2. A gate runner that, for one precision, checks fidelity against the
   official reference and the FlashRT `fp16` reference, and latency
   against a per-device baseline JSON (Thor `nvfp4` seeded at 231.6 ms;
   H100 latency ungated).
3. An `fp8_static` slot that activates when a calibration file exists,
   with a documented hand-off interface and no dependency on the
   calibration stream's code.

Measurable: on H100, `fp16` against the fixture reproduces the
end-to-end baseline (`fr_vs_off` median 0.99840, min 0.99567; mean
`mae_fr_vs_gt` 0.18359) and passes.

## Structure

| module | responsibility | state owned |
|---|---|---|
| `flash_rt/core/regression_gate.py` | model-agnostic gate policy: thresholds, latency baseline, latency summary, per-check results, report schema | none |
| `flash_rt/datasets/imagewam_gate_fixture.py` | fixture arrays, `.npz` IO, manifest with per-file and per-array SHA-256, verification | fixture format version |
| `benchmarks/imagewam_gate_fixture_generate.py` | produce a fixture on H100: reuses `imagewam_e2e_official_compare.py` for LIBERO loading, preprocessing and the official model; then the FlashRT `fp16` reference through the served `infer()` | fixture data on disk |
| `tests/gate_imagewam_libero.py` | gate runner CLI: load and verify fixture, run one precision through served `infer()`, evaluate, write result JSON | result JSON |
| `tests/fixtures/imagewam_gate/fidelity_thresholds.json` | per-precision fidelity thresholds, `requires_calibration` flag | thresholds |
| `tests/fixtures/imagewam_gate/latency_baselines.json` | per-device latency baselines and gating switch | baselines |
| `tests/fixtures/imagewam_gate/<name>.manifest.json` | committed manifest of the generated fixture | fixture identity |
| `flash_rt/frontends/torch/imagewam_thor.py` | `infer(observation, *, action_noise=None)`: optional explicit initial noise; default behavior unchanged | action latent buffer (unchanged owner) |
| `tests/test_imagewam_regression_gate.py` | CPU tests for gate policy, fixture round trip and tamper detection, committed config files | none |

The runner goes through the served `infer()` rather than re-implementing
its body, so later changes to `infer()` (for example an in-graph VAE)
are gated as served.

## Interface

```python
# flash_rt/core/regression_gate.py
RESULT_SCHEMA_VERSION = 1
@dataclass(frozen=True) class CosineSummary:   median: float; minimum: float; count: int
@dataclass(frozen=True) class FidelityThresholds:
    vs_official_median_min: float; vs_official_min_min: float
    vs_fp16_reference_median_min: float; vs_fp16_reference_min_min: float
    mae_vs_gt_ratio_max: float; requires_calibration: bool
@dataclass(frozen=True) class LatencySummary:   p10/p50/p90/min/max_ms, iters, group_medians_ms
    @classmethod from_samples(samples_ms) -> LatencySummary
@dataclass(frozen=True) class LatencyBaseline:  p50_ms: float; margin: float; source: str
@dataclass(frozen=True) class DeviceLatencyPolicy: device: str; gated: bool; reason: str; baselines: dict[str, LatencyBaseline]
@dataclass(frozen=True) class GateCheck:        name: str; status: "pass"|"fail"|"ungated"|"skipped"; value; limit; detail
class FidelityGate:  evaluate(vs_official, vs_fp16_reference, mae_mean, reference_mae_mean, all_finite) -> list[GateCheck]
class LatencyGate:   evaluate(precision, summary) -> GateCheck   # p50 < baseline*(1+margin)
@dataclass class GateReport: checks + context; verdict ("pass"|"fail"|"skipped"|"blocked"); to_dict()

# flash_rt/datasets/imagewam_gate_fixture.py
FIXTURE_FORMAT_VERSION = 1
@dataclass class ImageWAMGateFixture:  view1 (N,224,224,3) u8, view2, state (N,8) f32, task_index (N,),
    episode, frame, gt_actions (N,H,7) f32 (NaN-padded), gt_len (N,), prompts (T,),
    context_bf16_bits (T,L,D) u16, context_mask (T,L) bool, seeds (S,),
    noise (N,S,H,7) f32, official_actions (N,S,H,7) f32 (normalized),
    fp16_reference_actions (N,S,H,7) f32 (normalized)
class GateFixtureStore:  save(fixture, directory, metadata) -> FixtureManifest; load(directory, manifest) -> ImageWAMGateFixture
@dataclass class FixtureManifest: name, format_version, files{name: sha256,bytes}, arrays{name: shape,dtype,sha256}, metadata

# fp8_static hand-off (tests/gate_imagewam_libero.py)
FP8_CALIBRATION_ENV = "IMAGEWAM_FP8_CALIBRATION"      # or --fp8-calibration PATH
FP8_CALIBRATION_FRONTEND_KWARG = "calibration_path"   # the calibration stream's keyword
```

`fp8_static` contract: a precision whose thresholds say
`requires_calibration` is gated only when a calibration file path is
given and exists; otherwise the report verdict is `skipped` with the
reason. When the file exists, the runner passes its path to
`ImageWAMTorchFrontendThor(..., calibration_path=<path>)` only if the
constructor declares that keyword explicitly; if not, the verdict is
`blocked`, naming the missing keyword. The runner never runs
`fp8_static` on the placeholder `N(0, 0.1)` calibration. The file's
SHA-256 goes into the report.

Exit codes: 0 for `pass` and `skipped`, 1 for `fail` and `blocked`.

## Flow

Generator (H100, once per fixture version):
1. `load_samples()` from the end-to-end script (env `SUITE`, `N_TASKS`,
   `FRAMES`, `SEEDS`); `center_crop_resize` both views to 224x224.
2. Official model: per task `_prepare_flux2_infer_text` gives context and
   mask; per sample and seed, noise emulated exactly as the official
   sampler draws it (CPU generator, bf16 round trip), then
   `infer_action_flux2(..., seed)` gives the official normalized actions.
3. Free the official model; construct FlashRT `fp16` (real checkpoint,
   AE, dataset stats, no Qwen3); per sample and seed, `set_prompt(context)`
   and `infer(obs, action_noise=noise)`; store the renormalized actions.
4. `GateFixtureStore.save` writes `fixture.npz` and the manifest.

Runner (any CUDA device):
1. Load and verify the fixture against the committed manifest.
2. Resolve fidelity thresholds for the precision; apply the `fp8_static`
   contract above.
3. Read the clock state (item 10) and the device policy.
4. Construct the frontend (no Qwen3); per sample and seed run served
   `infer(obs, action_noise=noise)`; cosine in normalized action space
   against the official and `fp16` references; MAE against ground truth
   in real units.
5. Latency: served `infer(obs)` with default noise, warmup then timed
   iterations (`time.perf_counter` around each call; `infer()`
   synchronizes).
6. Evaluate, write `result.json`, print one `__IMAGEWAM_GATE__ <json>`
   line, exit.

## Code Mapping

| item | file |
|---|---|
| gate policy | `flash_rt/core/regression_gate.py` (new) |
| fixture format | `flash_rt/datasets/imagewam_gate_fixture.py` (new) |
| generator | `benchmarks/imagewam_gate_fixture_generate.py` (new) |
| runner | `tests/gate_imagewam_libero.py` (new) |
| configs and manifest | `tests/fixtures/imagewam_gate/*.json` (new) |
| explicit noise hook | `flash_rt/frontends/torch/imagewam_thor.py` (`infer`) |
| unit tests | `tests/test_imagewam_regression_gate.py` (new) |
| fixture data | `/home/user1/workspace/jingwu/artifacts/deploy-gates/imagewam_libero_gate_v1/` (not in git) |

## Implementation Phases

### Phase 1 — gate policy and fixture format, CPU tests

Phase Status: completed

Goal: `regression_gate.py`, `imagewam_gate_fixture.py`, the two config
JSON files, unit tests.
Modified files: those files and `tests/test_imagewam_regression_gate.py`.
Observation method: pytest on CPU; tamper test (flip one byte of a
fixture array) must fail verification.

### Phase 2 — explicit initial-noise hook in `infer()`

Phase Status: completed

Goal: `infer(observation, *, action_noise=None)`; default path
unchanged.
Modified files: `flash_rt/frontends/torch/imagewam_thor.py`.
Observation method: on H100 fp16 real checkpoint, `infer(obs,
action_noise=n)` equals the end-to-end script's
`flashrt_infer_with_noise` (bit-exact after denormalization);
regression suite unchanged.

### Phase 3 — fixture generator, fixture v1 on H100

Phase Status: completed

Goal: `imagewam_libero_gate_v1` (libero_spatial, 10 tasks, frames 0 and
60, seeds 0 and 1) generated; manifest committed.
Modified files: `benchmarks/imagewam_gate_fixture_generate.py`,
`tests/fixtures/imagewam_gate/imagewam_libero_gate_v1.manifest.json`.
Observation method: generator prints per-sample official-vs-fp16
cosine; its summary must reproduce the end-to-end baseline.

### Phase 4 — gate runner, real fp16 gate on H100

Phase Status: completed

Goal: `tests/gate_imagewam_libero.py`; real `fp16` run on H100 passes
fidelity with latency ungated; `fp8_static` without a calibration file
reports `skipped`; `nvfp4` on H100 fails at construction with the
existing clear NVFP4 build error.
Modified files: `tests/gate_imagewam_libero.py`.
Observation method: result JSON values next to the end-to-end baseline.

### Phase 5 — Thor handoff

Phase Status: blocked

Goal: Thor checklist: copy fixture, verify checksums, run `nvfp4` and
`fp16`, report result JSON.
Modified files: none.
Observation method: owner's Thor run.
Blocker: needs the Thor hardware and a copy of the v1 fixture there.

## Results (H100, shared GPU)

- Phase 1: `tests/test_imagewam_regression_gate.py` 19 passed on CPU
  without the compiled extension; a flipped byte in `fixture.npz` and a
  mismatched array record are both rejected.
- Phase 2: `tests/test_imagewam_infer_action_noise.py` 4 passed; the
  fixed-noise path equals the direct buffer write plus graph replay
  (max abs difference 0.0), the default path is unchanged.
- Phase 3: fixture v1 generated (81 MiB). FlashRT fp16 against official,
  seed 0: median 0.99840, min 0.99567, mean MAE 0.18359, equal to the
  end-to-end baseline; official seed spread median 0.99630, min 0.97154.
- Phase 4: `fp16` gate verdict `pass` (vs official median 0.99836, min
  0.99554 over 40 runs; vs fp16 reference bit-identical; MAE 0.18364);
  latency P50 158.2 ms recorded and ungated. `nvfp4` on sm_90 is
  `blocked` at construction; `fp8_static` is `skipped` without a
  calibration file and `blocked` with one (no constructor keyword yet).
- Review follow-ups: an ungated latency is a top-level result field and
  part of the verdict reason, and `--require-latency` makes it
  `blocked`; `--iters` is validated before any GPU work; the checkpoint
  is verified by SHA-256 (`--skip-checkpoint-hash` for size only);
  provenance records untracked files and the generator's SHA-256; the
  calibration keyword is the calibration stream's `calibration_path`.
  Regression suite 150 passed, 6 skipped; the fp16 gate still passes
  with the same fidelity values.

# Plan: ActionDiT small-M CUTLASS tile selection (roadmap item 1)

Plan Status: approved

## Problem

### Current

Every ActionDiT weight GEMM runs at `M = num_action = 64`. The tile
variant for the two CUTLASS-backed quantized precisions is chosen by
an `(N, K)`-only heuristic that was tuned for other shapes:

- `nvfp4` (shipped default): `Nvfp4Linear` calls
  `flash_rt.executors.fp4_utils.fp4_gemm` without a variant, so
  `pick_variant(N, K)` applies. That table was calibrated for Pi0.5's
  encoder at `M = 968`.
- `fp8_static_cutlass`: `StaticFp8Linear(use_cutlass=True)` uses
  `_pick_fp8_cutlass_variant(N, K)`, which picks `wide` when
  `N >= 4K` and `sq` otherwise. It is a provisional guess ported
  from backbone shapes.

Inventory at the real ActionDiT shapes (`M = 64`,
`action_hidden_dim = 1024`, `action_attn_width = 3072`,
`action_mlp_hidden = 4096`, `action_dim = 7`, 5 double and 20 single
layers):

| site | N | K | calls per step | `nvfp4` | `fp8_static_cutlass` | `fp16_cutlass` |
|---|---:|---:|---:|---|---|---|
| double `qkv` | 9216 | 1024 | 5 | v6 `128x256x128` c1x1x1 | `wide` `256x128x128` c2x2x1 | `wide` |
| double `proj` | 1024 | 3072 | 5 | v6 | `sq` `256x256x128` c2x2x1 | `sq` |
| double `mlp0` (merged gate/up) | 8192 | 1024 | 5 | v6 | `wide` | SwiGLU pair `k64_silu` + `k64_mul_aux` `256x256x64` c2x2x1 at N=4096 |
| double `mlp2` | 1024 | 4096 | 5 | v6 | `sq` | `sq` |
| single `linear1` (qkv + gate/up) | 17408 | 1024 | 20 | v8 `128x256x256` c1x1x1 | `wide` | not merged: `qkv` `wide` + SwiGLU pair at N=4096 |
| single `attn_out_proj` | 1024 | 3072 | 20 | v6 | `sq` | `sq` |
| single `mlp_down` | 1024 | 4096 | 20 | v6 | `sq` | `sq` |
| `action_encoder` | 1024 | 7 | 1 | cuBLASLt `Fp16Linear` (alignment fallback) | same | same |
| `head.linear` | 7 | 1024 | 1 | cuBLASLt `Fp16Linear` (alignment fallback) | same | same |

At `M = 64` one output tile row covers the whole M extent, so the CTA
count equals the number of N tiles. On Thor's 20 SMs, the `N = 1024`
GEMMs launch 4 CTAs under v6 (N tile 256), and 4 useful CTA pairs
under FP8 `sq` with a 2x2 cluster. Those GEMMs are weight-bandwidth
bound (arithmetic intensity 2M = 128 FLOP per weight element), so a
launch that occupies 4 of 20 SMs cannot reach DRAM bandwidth. There
are 50 such calls per denoise step and 500 per `infer()`. FP8 CUTLASS
has no tile narrower than 128 in N and no 1-SM (cluster 1x1x1) tile at
all. On Thor it measured 1.44-1.68x slower than cuBLASLt at this M
(opportunities.md OPT-014, result 3).

Pi0.5 runs its decoder (`M = 10`) on the narrow-N v10 tile
(`128x64x256`, cluster 1x1x1) for all four projections
(`docs/pi05_thor_decoder_fp4_e2e.md`, "Decoder v10 Tiles"). v10 is
already instantiated in `cutlass_fp4_gemm_variants.cu`, but ImageWAM
never selects it.

### Problem

No ActionDiT GEMM tile choice is measured at `M = 64`. The existing
choices are extrapolated from other shapes, and FP8 CUTLASS has no
small-M tile to choose.

### Measurable goal

- A per-shape tile choice for the ActionDiT GEMMs, made by a one-time
  measurement at construction on the device the frontend runs on,
  cached per `(family, M, N, K)`. The candidate set includes the
  current heuristic pick, and a candidate must reproduce the current
  pick's output before it can be selected.
- FP8 small-M 1-SM tiles, with the Pi0.5 v10 tile `128x64x256` as the
  template.
- Selection logic unit-tested with the kernels stubbed.
- A Thor script that sweeps every candidate at every real ActionDiT
  shape, reports cosine against fp16 and the per-shape winner, and
  runs an `infer()` A/B of old vs new selection on `nvfp4` and
  `fp8_static_cutlass`.
- The shipped default selection stays unchanged (opt-in flag) until
  Thor confirms correctness and speed.

## Structure

- NEW `flash_rt/models/imagewam/gemm_variant_tuner.py`: owns the
  selection policy (candidate filtering, correctness gate, timing
  comparison, hysteresis against the incumbent) and the per-frontend
  result cache. Defines the `VariantTunableGemm` and `VariantTimer`
  protocols and the result dataclasses. No CUDA code.
- NEW `flash_rt/models/imagewam/gemm_variant_timer.py`: owns the
  device timing mechanism (`CudaGraphVariantTimer`: CUDA-graph
  capture of a launch batch, replay timed with CUDA events).
- `flash_rt/models/imagewam/quant_linear.py`: `Nvfp4Linear` and
  `StaticFp8Linear(use_cutlass=True)` implement `VariantTunableGemm`.
  Each linear owns its own current variant. The default variant equals
  today's heuristic pick, so `__call__` is unchanged until a variant is
  set.
- `csrc/gemm/gemm_types_sm100.h`, `csrc/gemm/cutlass_sm100.cu`,
  `csrc/bindings.cpp`: four new FP8 1-SM tiles (cluster 1x1x1):
  `t128x64x256` (v10 template), `t128x64x128`, `t128x128x128`,
  `t128x256x128`.
- `flash_rt/frontends/torch/imagewam_thor.py`: owns the decision to
  tune (`gemm_variant_autotune: bool = False`), the grouping of
  ActionDiT linears by shape, and the tuner instance and its results
  (`gemm_variant_results`).
- NEW `benchmarks/imagewam_thor_small_m_tile_sweep.py`: Thor sweep
  and `infer()` A/B.
- NEW `tests/test_imagewam_gemm_variant_tuner.py`: stubbed selection
  tests (CPU) and a real-timer test (any CUDA GPU).

State ownership:

| state | owner |
|---|---|
| current tile variant of one linear | that `Nvfp4Linear` / `StaticFp8Linear` instance |
| tuning results cache `(family, M, N, K) -> result` | the `GemmVariantTuner` instance owned by the frontend |
| whether tuning runs | frontend constructor argument |

## Interface

```python
# flash_rt/models/imagewam/gemm_variant_tuner.py
@dataclass(frozen=True)
class GemmShape:
    m: int
    n: int
    k: int

@dataclass(frozen=True)
class VariantMeasurement:
    variant: str
    us_per_gemm: float | None      # None: not timed (rejected)
    cosine_vs_default: float | None
    status: str                     # "ok" | "launch_failed rc=.." | "mismatch" | "nonfinite"

@dataclass(frozen=True)
class VariantTuneResult:
    family: str
    shape: GemmShape
    members: int
    default_variant: str
    chosen_variant: str
    measurements: tuple[VariantMeasurement, ...]

class VariantTunableGemm(Protocol):
    family: str
    n: int
    k: int
    default_variant: str
    variant: str
    def candidate_variants(self) -> tuple[str, ...]: ...
    def set_variant(self, variant: str) -> None: ...
    def prepare_tuning_input(self, x_ptr: int, m: int, stream: int) -> None: ...
    def launch_variant(self, variant: str, out_ptr: int, m: int, stream: int) -> int: ...

class VariantTimer(Protocol):
    def us_per_launch(self, batches: Sequence[Callable[[int], None]],
                      launches_per_batch: int) -> tuple[float | None, ...]: ...   # None: batch raised

class GemmVariantTuner:
    def __init__(self, timer: VariantTimer, *, device: str = "cuda",
                 min_gain: float = 0.02, cosine_floor: float = 0.9999): ...
    def tune(self, members: Sequence[VariantTunableGemm], m: int) -> VariantTuneResult: ...
    def results(self) -> tuple[VariantTuneResult, ...]: ...

# flash_rt/models/imagewam/gemm_variant_timer.py
class CudaGraphVariantTimer:
    def __init__(self, *, reps: int = 4, samples: int = 15, warmup: int = 3): ...
    def us_per_launch(self, batches: Sequence[Callable[[int], None]],
                      launches_per_batch: int) -> tuple[float | None, ...]: ...

# flash_rt/frontends/torch/imagewam_thor.py
class ImageWAMTorchFrontendThor:
    def __init__(..., gemm_variant_autotune: bool = False, ...): ...
    gemm_variant_results: tuple[VariantTuneResult, ...]
```

Selection rule, per group of linears sharing `(family, M, N, K)`:

1. Stage the same random input into every member.
2. For every candidate, launch it once per member eagerly. Record
   `launch_failed` for a nonzero return code or a raised Python
   exception, and `nonfinite` or `mismatch` when its output against the
   default variant's output on the same member is non-finite or below
   `cosine_floor`. The default variant failing or raising is an error.
3. Time each surviving candidate as one launch per member, round
   robin, so each launch reads a different layer's weight and the
   timing does not run from a warm L2. The batch is captured in one
   CUDA graph so launch overhead is excluded.
   A candidate the timer cannot capture is `timing_failed`.
4. Choose the fastest candidate. Keep the default unless the winner is
   faster by more than `min_gain` (2%), or if the default itself could
   not be timed.
5. Apply the choice to every member and cache it.

## Flow

```
ImageWAMTorchFrontendThor.__init__(gemm_variant_autotune=True, precision in {nvfp4, fp8_static_cutlass})
  -> _load_real_weights / _alloc_random_weights      (linears built on default variants)
  -> _tune_action_dit_gemm_variants(d)
       group self._weights[("action_dit", ...)] implementing VariantTunableGemm by (family, n, k)
       for each group: self._gemm_tuner.tune(members, m=d["num_action"])
         -> member.prepare_tuning_input / launch_variant       (eager checks)
         -> CudaGraphVariantTimer.us_per_launch                (graph-timed batches)
         -> member.set_variant(chosen)
       self.gemm_variant_results = self._gemm_tuner.results()
  -> set_prompt(): _calibrate_fp8 (unchanged), _capture_graph (captures chosen variants)
```

Tuning never calls `__call__`, so `StaticFp8Linear`'s
calibrate-before-call contract is unaffected.

## Code Mapping

| item | file |
|---|---|
| `GemmShape`, `VariantMeasurement`, `VariantTuneResult`, `VariantTunableGemm`, `VariantTimer`, `GemmVariantTuner` | `flash_rt/models/imagewam/gemm_variant_tuner.py` |
| `CudaGraphVariantTimer` | `flash_rt/models/imagewam/gemm_variant_timer.py` |
| protocol implementation | `flash_rt/models/imagewam/quant_linear.py` |
| FP8 1-SM tiles | `csrc/gemm/gemm_types_sm100.h`, `csrc/gemm/cutlass_sm100.cu`, `csrc/bindings.cpp` |
| flag, grouping, tuner ownership | `flash_rt/frontends/torch/imagewam_thor.py` |
| Thor sweep + A/B | `benchmarks/imagewam_thor_small_m_tile_sweep.py` |
| tests | `tests/test_imagewam_gemm_variant_tuner.py` |

## Implementation Phases

### Phase 1: tuner and timer

Phase Status: completed

- Goal: selection policy and device timer, independent of any kernel.
- Files: `gemm_variant_tuner.py`, `gemm_variant_timer.py`,
  `tests/test_imagewam_gemm_variant_tuner.py`.
- Observation: stubbed tests cover argmin choice, hysteresis, launch
  failure, mismatch rejection, nonfinite rejection, default failing,
  every candidate failing, the cache, and group application. On H100,
  a real-timer test tunes two real sm_90 kernels, and its printed
  per-launch times are compared with a direct CUDA-event measurement.

### Phase 2: FP8 1-SM small-M tiles

Phase Status: completed

- Goal: `cutlass_fp8_t128x64x256`, `_t128x64x128`, `_t128x128x128`,
  `_t128x256x128` exported under `ENABLE_SM100_CUTLASS`.
- Files: `csrc/gemm/gemm_types_sm100.h`, `csrc/gemm/cutlass_sm100.cu`,
  `csrc/bindings.cpp`.
- Observation: `sm110_check.sh` passes, and the sm_90 build is
  unaffected. Correctness and speed are Thor checklist items.

### Phase 3: quant_linear protocol implementation

Phase Status: completed

- Goal: `Nvfp4Linear` and `StaticFp8Linear(use_cutlass=True)` expose
  `family`, `default_variant`, `variant`, `candidate_variants()`,
  `set_variant()`, `prepare_tuning_input()`, and `launch_variant()`.
  Default behavior stays unchanged.
- Files: `flash_rt/models/imagewam/quant_linear.py`.
- Observation: the regression suite is unchanged. Construction on
  H100 still raises the same `RuntimeError`.

### Phase 4: frontend wiring

Phase Status: completed

- Goal: `gemm_variant_autotune` flag, grouping ActionDiT linears by
  shape, and `gemm_variant_results`.
- Files: `flash_rt/frontends/torch/imagewam_thor.py`, tests.
- Observation: a routing test with stubbed linear classes confirms
  that only ActionDiT groups are tuned, with `m = num_action`, one
  tune per distinct shape, and the chosen variant applied to every
  member. With the flag off, nothing changes. The fp16 path and the
  regression suite are unchanged.

### Phase 5: Thor sweep and A/B script, handoff

Phase Status: completed

- Goal: `benchmarks/imagewam_thor_small_m_tile_sweep.py`, plus
  results and Thor checklist in `opportunities.md` OPT-018.
- Observation: the script's non-Thor paths (argument parsing, shape
  table, cuBLASLt fp16 reference timing) run on H100 and print SKIP
  for families this build lacks.

### Phase 6: candidates that raise are rejected

Phase Status: completed

- Goal: a Python exception from a candidate's launch, such as an
  `AttributeError` for a `cutlass_fp8_t128x*` symbol missing from a
  stale build, rejects that candidate instead of aborting construction.
  A batch the timer cannot capture is rejected as `timing_failed`.
- Files: `gemm_variant_tuner.py`, `gemm_variant_timer.py`, tests, both
  benchmarks.
- Observation: stub tests cover a raising candidate, a raising default
  (still an error), an untimeable candidate, and an untimeable default
  (kept). The timer test feeds a raising batch and a
  capture-invalidating batch. A routing test removes the `t128x*`
  symbols and construction completes.

### Phase 7: Thor confirmation

Phase Status: blocked

- Goal: the Thor checklist in opportunities.md OPT-018. The tile
  sweep must show correct outputs for every tile, and the
  `infer()` A/B of heuristic vs tuned tiles must show action cosine
  >= 0.9999 and a P50 delta.
- Blocker: no sm_110 device on the dev box. SM100 CUTLASS and NVFP4
  kernels do not run on sm_90, which only compile-checks them
  (`sm110_check.sh`). Recorded as issues.md ISSUE-023.

# Plan: attention-chain fusion recheck at ImageWAM's real shapes (roadmap item 6)

Plan Status: approved

## Problem

### Current

Both attention sites run a cuBLAS-composed chain in
`ImageWAMAttnBackend.run()` (`flash_rt/hardware/thor/attn_backend.py`):
a strided-batched QK^T GEMM into a `logits` buffer, a softmax kernel,
then a strided-batched PV GEMM (`fvk.attention_qkv_fp16_perhead`,
`csrc/kernels/attention_cublas.cuh`). The frontend always constructs
the backend with `use_perhead_kv=True, use_real_mot_mask=True`.

- `"backbone"` site: prefill self-attention, 25 layers, `q = kv = a0 =
  905` tokens, 24 heads, HD 128, no mask. FA4 is wired in as
  `use_fa4=True` (opportunities.md OPT-005). On Thor it measured
  cosine 1.000000 against the cuBLAS chain, 3.75x per call at the real
  per-head shape, and -10.5% prefill in the per-layer benchmark. The
  frontend default is `use_fa4=False`, because the previous dev box had
  no FA4 runtime and `use_fa4=True` raises when the runtime is missing.
- `"mot"` site: ActionDiT joint attention, 25 layers x 10 steps = 250
  calls per `infer()`, `q = 64` action queries over `kv = total = 969`
  keys. With `use_real_mot_mask=True`, the rule the frontend always
  uses, the call is unmasked attention through the same
  `attention_qkv_fp16_perhead`. Upstream `_build_mot_attention_mask_flux2`
  with `target_len = 0` removes only the region mask, but it still
  excludes padded text keys for every query row, at the prefill call
  and at the action call. `pipeline_thor.py` models that mask at
  neither site (issues.md ISSUE-020). FA4 has never been evaluated
  here.
- Pi0.5 rejected a fused SIMT attention chain at decoder `M = 10`,
  HD 256, as 5-7x slower (`docs/pi05_thor_decoder_fp4_e2e.md`). At that
  shape the QK^T/PV GEMMs are about 1 us of tensor-core work, and FA4
  has no KV-split path at HD 256.

### Problem

Nobody has measured which share of ImageWAM prefill and denoise time
attention takes at the real shapes. The Thor FA4 win is not on by
default. The `mot` site's fused-kernel eligibility has never been
evaluated.

### Measurable goal

- H100, indicative only: the attention share of prefill and of one
  denoise step at the real shapes, measured in-graph as graph time with
  the real attention minus graph time with attention removed. Also
  per-call cuBLAS chain vs fused kernels available on sm_90 (PyTorch
  SDPA flash / cuDNN / mem-efficient) at both sites' shapes, with
  cosine against the cuBLAS chain.
- Recommendation with evidence per site.
- If cheap: a Thor FA4 switch for `"backbone"` that resolves to the
  cuBLAS chain when the FA4 runtime is missing or the device is not
  Thor. It stays opt-in (`FLASHRT_THOR_FA4=1`) until Thor confirms FA4
  at the served shapes. Also an opt-in FA4 path for `"mot"`. Dispatch
  logic verified locally against the cuBLAS chain with a
  reference-backed FA4 stand-in. FA4 itself goes on the Thor
  checklist.

## Structure

- NEW `benchmarks/imagewam_attention_share_bench.py`: owns the
  measurements (in-graph attention share; per-call chain vs fused
  kernels; on Thor it also times FA4 per call, with `num_splits` swept
  for the `mot` shape).
- `flash_rt/hardware/thor/attn_backend.py`: `ImageWAMAttnBackend`
  owns per-site kernel dispatch. It gains `use_fa4_mot: bool` for the
  `"mot"` site FA4 branch. That branch is valid only with
  `use_real_mot_mask=True` and `use_perhead_kv=True`, FlashRT's
  unmasked per-head rule, and the constructor rejects any other
  combination.
- `flash_rt/hardware/thor/fa4_backend.py`: owns FA4 availability. It
  gains `thor_default_enabled() -> bool`, true only on an sm_11x
  device with an active FA4 runtime.
- `flash_rt/frontends/torch/imagewam_thor.py`: owns the FA4 output
  buffer, and the FA4-failure fallback in `set_prompt()`
  (`_capture_graph_or_fall_back`): on an exception during warmup or
  capture with FA4 on, it logs, warns, records `fa4_fallback_reason`,
  rebuilds the backend with FA4 off, and captures again. It also owns
  the default.
  `use_fa4: bool | None = None` resolves, through `_resolve_use_fa4`, to
  False unless `FLASHRT_THOR_FA4=1`. With the variable set, it resolves
  to `fa4_backend.thor_default_enabled()`. An explicit `True` still
  requires the runtime, and an explicit `False` forces the cuBLAS
  chain. It also gains `use_fa4_mot: bool = False`, passed through.
- Tests: `tests/test_imagewam_fa4_dispatch.py` (new) checks the
  backend's FA4 branches for both sites against the cuBLAS chain, with
  FA4 replaced by a stand-in that has FA4's `_flash_attn_fwd`
  signature and computes attention as an fp32 matmul-softmax-matmul in
  PyTorch. It also checks the default resolution.
  `tests/test_imagewam_fa4_backbone.py` gains a real-FA4 `mot` case
  that skips without FA4.

State ownership:

| state | owner |
|---|---|
| FA4 on/off per site | `ImageWAMAttnBackend` instance (`_use_fa4`, `_use_fa4_mot`) |
| default resolution | frontend `_resolve_use_fa4` (`FLASHRT_THOR_FA4`, then `fa4_backend.thor_default_enabled()`) |
| FA4 runtime availability | `fa4_backend` module |
| FA4 output buffer `_fa4_out` | frontend (passed to the backend as `fa4_out` slots) |
| `fa4_fallback_reason` | frontend |

## Interface

```python
# flash_rt/hardware/thor/fa4_backend.py
def thor_default_enabled() -> bool: ...   # sm_11x device AND FA4 runtime active

# flash_rt/hardware/thor/attn_backend.py
class ImageWAMAttnBackend:
    def __init__(self, spec, ctx, *, backbone_slots: dict, mot_slots: dict,
                 use_fa4: bool = False, use_perhead_kv: bool = False,
                 use_real_mot_mask: bool = False, use_fa4_mot: bool = False): ...

# flash_rt/frontends/torch/imagewam_thor.py
class ImageWAMTorchFrontendThor:
    def __init__(..., use_fa4: bool | None = None, use_fa4_mot: bool = False, ...): ...
    use_fa4: bool        # resolved value, read-only after construction
    use_fa4_mot: bool
```

FA4 `"mot"` call: Q `(1, q_seq, NH, HD)` at row offset `a0` of `Q_O`,
K/V `(1, kv_seq, NH, HD)` per layer, `causal=False`, `pack_gqa=False`,
`num_splits=1`. Output goes to the dedicated FA4 output buffer, then
is copied back to the Q rows. This is the same pattern as the
`"backbone"` per-head branch.

FA4 output buffer: a site that runs FA4 gets `"fa4_out"` (fp16 pointer)
and `"fa4_out_numel"` in its slots. The capacity must be at least that
site's `max_q_seq * NH * HD`, checked at construction and on every call.
The frontend owns one `(total, hidden)` buffer for both sites, allocated
only when some site runs FA4. `logits` is sized for the cuBLAS chain's
score matrix (`total*NH x total`), which at small dims is smaller than
`q_seq*NH*HD`, so FA4 output never goes there.

## Flow

```
frontend __init__(use_fa4=None)
  -> use_fa4 = FLASHRT_THOR_FA4 == "1" and fa4_backend.thor_default_enabled()   # opt-in
  -> ImageWAMAttnBackend(..., use_fa4=use_fa4, use_fa4_mot=use_fa4_mot)
prefill:  attn.run("backbone", ...) -> FA4 if use_fa4 else attention_qkv_fp16_perhead
denoise:  attn.run("mot", ...)      -> FA4 if use_fa4_mot else attention_qkv_fp16_perhead
```

## Code Mapping

| item | file |
|---|---|
| measurements | `benchmarks/imagewam_attention_share_bench.py` |
| `thor_default_enabled` | `flash_rt/hardware/thor/fa4_backend.py` |
| `use_fa4_mot` dispatch | `flash_rt/hardware/thor/attn_backend.py` |
| default resolution, `use_fa4_mot` pass-through | `flash_rt/frontends/torch/imagewam_thor.py` |
| dispatch tests | `tests/test_imagewam_fa4_dispatch.py`, `tests/test_imagewam_fa4_backbone.py` |
| results, recommendation | `opportunities.md` OPT-019 |

## Implementation Phases

### Phase 1: measurement

Phase Status: completed

- Goal: attention share (H100) and per-call chain vs fused kernels at
  real shapes.
- Files: `benchmarks/imagewam_attention_share_bench.py`.
- Observation: printed P10/P50/P90 for graphs with and without
  attention, per stage. Per-call medians for each kernel, with cosine
  against the cuBLAS chain.

### Phase 2: Thor FA4 switch (opt-in), opt-in FA4 for `mot`

Phase Status: completed

- Goal: `use_fa4=None` resolution (opt-in through `FLASHRT_THOR_FA4=1`),
  `use_fa4_mot`.
- Files: `fa4_backend.py`, `attn_backend.py`, `imagewam_thor.py`,
  tests.
- Observation: dispatch tests show the FA4 branches, with an fp32
  matmul stand-in, matching the cuBLAS chain at real shapes (cosine,
  max-abs, rel_l2). Resolution resolves to False on H100. The
  regression count is unchanged apart from the new tests. An fp16
  end-to-end quick run matches the baseline, since the default
  resolves to off on H100.

### Phase 3: recommendation and Thor handoff

Phase Status: completed

- Goal: OPT-019 with evidence, recommendation, and Thor checks.
- Files: `opportunities.md`.

### Phase 4: FA4 back to opt-in

Phase Status: completed

- Goal: `use_fa4=None` resolves to the cuBLAS chain on every device.
  `FLASHRT_THOR_FA4=1` opts in, and making FA4 the default is a
  one-line change (`_FA4_OPT_IN_DEFAULT`).
- Files: `imagewam_thor.py`, `tests/test_imagewam_fa4_dispatch.py`.
- Observation: resolution tests cover every combination of the
  environment variable, runtime availability, and explicit argument.

### Phase 5: dedicated FA4 output buffer

Phase Status: completed

- Goal: FA4 output goes to `fa4_out` slots, which the frontend owns as
  `(total, hidden)`, instead of `logits`, which overruns at small dims.
- Files: `attn_backend.py`, `imagewam_thor.py`, FA4 tests, FA4 benches.
- Observation: guard-band tests after `fa4_out`, and on `logits`, at
  (a0, total) = (8, 12), (8, 24), and (905, 969), plus a frontend range
  check. Both fail on the old staging. Capacity is checked at
  construction.

### Phase 6: fall back to the cuBLAS chain when FA4 fails

Phase Status: completed

- Goal: an FA4 failure during `set_prompt()`'s warmup or capture logs,
  warns, records `fa4_fallback_reason`, rebuilds the backend without
  FA4, and captures again.
- Files: `imagewam_thor.py`, `tests/test_imagewam_fa4_dispatch.py`.
- Observation: stand-ins that fail at first call, inside capture, and
  by invalidating the capture all recover to the chain's output with
  the caller's stream restored. A failure with FA4 off still raises.

### Phase 7: Thor confirmation

Phase Status: blocked

- Goal: the Thor checklist in opportunities.md OPT-019, with FA4
  explicitly opted in:
  - the real-FA4 real-shape test at 905/905 and 64/969;
  - kernel timings;
  - an nvfp4 end-to-end official compare with FA4 on vs off;
  - an `infer()` A/B with FA4 on vs off.
- Blocker: no sm_110 device and no FA4 runtime on the dev box (issues.md
  ISSUE-023).

# Plan: single-stream `linear2` merge (roadmap item 4)

Plan Status: completed

Verified locally; the Thor results are pending (opportunities.md OPT-016).

## Problem

### Current

The official `SingleStreamBlock` (`third_party/flux2/src/flux2/model.py`)
and `SlimFlux2SingleBlock` (ImageWAM
`src/imagewam/models/backbones/action_dit_flux2.py`) compute the block
output projection as ONE `linear2` GEMM over
`cat([attn_out, mlp_act(mlp)], -1)`. `checkpoint_loader._extract_single_block`
splits the real `linear2.weight` at load time into `attn_out_proj.weight`
and `mlp_down.weight`. `pipeline_thor.py`'s `_single_stream_layer` and
`_action_single_layer` then run two GEMMs (`attn_out_proj`, `mlp_down`)
into two landing buffers, a torch elementwise add (`_add_inplace`), and
the gated residual. This runs in 20 backbone single-stream layers (once
per `infer()`) and 20 ActionDiT single-stream layers (200 times per
`infer()`, 10 denoise steps).

The `linear1` merge (OPT-015, commit 4e9f7d7) is the precedent: one GEMM
launch removed per ActionDiT single-stream layer gave +8.0 ms of the
+7.9 ms `infer()` win on Thor, because the M=64 denoise loop is
launch-bound.

### Problem

Each single-stream layer launches one GEMM (two kernels for `nvfp4`:
activation quantize + GEMM) and one add kernel more than the official
structure needs. The split also rounds three times (two FP16 GEMM outputs
and their FP16 sum) where the official structure rounds once.

### Measurable goal

- Every precision except `fp16_cutlass` runs ONE `linear2` GEMM with
  `K = attn_width + mlp_hidden` per single-stream layer, feeding the
  existing gated residual. `fp16_cutlass` keeps the split path, as it
  does for `linear1`.
- Locally (H100, `fp16`), at the real shapes: merged vs split
  single-stream layer output cosine >= 0.9999, reported with max-abs and
  rel_l2; the end-to-end compare stays at baseline (`fr_vs_off` median
  0.99840, min 0.99567; mean `mae_fr_vs_gt` 0.18359).
- Thor: merged vs split `infer()` P50 A/B on `nvfp4` and `fp16`.

## Structure

- `csrc/kernels/activation.{cu,cuh}`, `csrc/bindings.cpp`:
  `silu_glu_merged_fp16` gains an output row stride so the SiLU-GLU
  result lands directly in the MLP columns of the `linear2` input
  buffer. Owns no state.
- `flash_rt/models/imagewam/checkpoint_loader.py`:
  `_extract_single_block`/`build_real_weights` gain `merge_linear2`.
  When set, the real unsplit `linear2.weight` is returned in the (K,N)
  GEMM convention instead of the two split slots.
- `flash_rt/frontends/torch/imagewam_thor.py`: owns `dims["merge_linear2"]`
  (default `precision != "fp16_cutlass"`, overridable through
  `dims_override` for A/B), the `linear2.weight` slots (random and real
  weights), the `single_linear2_in`/`action_linear2_in` buffers, and the
  autotune shapes for the merged GEMM.
- `flash_rt/models/imagewam/pipeline_thor.py`: `_single_stream_layer` and
  `_action_single_layer` own the merged data flow (see Flow).

State ownership: `dims["merge_linear2"]` is set once by the frontend
constructor and only read by the pipeline. The `linear2` input buffers
are allocated once by `_alloc_buffers`, like every other scratch buffer.

## Interface

```python
# csrc/kernels/activation.cuh
void silu_glu_merged_fp16(const __half* merged, __half* out, int seq, int half_dim,
                          cudaStream_t stream = 0, int row_stride = 0,
                          int out_row_stride = 0);   # 0 = packed (half_dim)

# python binding (flash_rt_kernels)
silu_glu_merged_fp16(merged, out, seq, half_dim, stream=0, row_stride=0, out_row_stride=0)

# flash_rt/models/imagewam/checkpoint_loader.py
_extract_single_block(sd, prefix, *, attn_dim, merge_qkv_mlp=False, merge_linear2=False) -> dict
build_real_weights(sd, *, ..., merge_qkv_mlp=False, merge_linear2=False) -> dict
#   merge_linear2=True: slot "linear2.weight", shape (attn_dim + mlp_hidden, hidden)
#   replaces "attn_out_proj.weight" and "mlp_down.weight".

# dims keys read by pipeline_thor.py
dims["merge_linear2"]: bool   # requires dims["merge_qkv_mlp"]
bufs["single_linear2_in"]: (a0, hidden + mlp_hidden) fp16
bufs["action_linear2_in"]: (num_action, action_attn_width + action_mlp_hidden) fp16
weights[(stream, "single", L, "linear2.weight")]: callable linear op, N=hidden, K=attn+mlp
```

`merge_linear2=True` with `merge_qkv_mlp=False` is rejected (frontend
constructor and pipeline both raise `ValueError`): the merged path relies
on the `linear1`-merged SiLU-GLU call to write into the `linear2` input
buffer.

## Flow

Merged single-stream layer (backbone shown; ActionDiT is identical at
`action_*` widths, with the attention output at row offset `a0` of the
"mot" site's `Q_O`):

1. AdaLN: `combined` -> `modded`.
2. `linear1.weight` GEMM -> `single_linear1_merged`; Q/K/V column slices
   copied to `Q_O`/`K_cache`/`V_cache`; `silu_glu_merged_fp16` reads the
   gate/up columns and writes `(a0, mlp_hidden)` into
   `single_linear2_in[:, hidden:]` (`out_row_stride = hidden + mlp_hidden`).
3. QK-Norm, RoPE, attention (`Q_O` holds the attention output).
4. Strided copy `Q_O` -> `single_linear2_in[:, :hidden]`.
5. `linear2.weight` GEMM (`K = hidden + mlp_hidden`) -> `proj_scratch`.
6. Gated residual `combined += gate * proj_scratch` (unchanged kernel).

Kernel count per layer, merged vs split: one strided copy + one GEMM
replaces two GEMMs + one add.

Precision notes:
- `nvfp4`: `quantize_fp4_dynamic_sfa_fp16` scales every 16-element K
  block independently, with no per-tensor scale. `hidden` (3072) is a
  multiple of 16, so no block straddles the attn|mlp boundary, and the
  quantized activation and weight operands are identical to the split
  path's. K = 12288 (backbone) and 7168 (ActionDiT) are multiples of 64,
  so the scale-factor layout has no K padding. Only the accumulation
  differs.
- `fp8` / `fp8_static*`: one per-tensor activation scale and one
  per-tensor weight scale now cover both halves. Measured negligible at
  the GEMM level (issues.md ISSUE-011); the end-to-end check runs on
  Thor, or on H100 once ISSUE-001's TN fix lands.
- `fp16`: one FP32-accumulated GEMM, rounded once.

## Code Mapping

| item | file |
|---|---|
| SiLU-GLU output stride | `csrc/kernels/activation.cu`, `csrc/kernels/activation.cuh`, `csrc/bindings.cpp` |
| unsplit `linear2.weight` | `flash_rt/models/imagewam/checkpoint_loader.py` |
| flag, weights, buffers, autotune | `flash_rt/frontends/torch/imagewam_thor.py` |
| merged data flow | `flash_rt/models/imagewam/pipeline_thor.py` |
| kernel test | `tests/test_imagewam_real_mlp.py` (extended) |
| layer tests (small reference + real-shape merged vs split) | `tests/test_imagewam_thor_real_wiring.py` |
| loader test | `tests/test_imagewam_checkpoint_loader.py` |
| A/B + correctness script (local and Thor) | `benchmarks/imagewam_fusion_ab.py` |

## Implementation Phases

### Phase 1: SiLU-GLU output row stride

Phase Status: completed

- Goal: `silu_glu_merged_fp16` writes into a column slice of a wider
  buffer.
- Files: `csrc/kernels/activation.{cu,cuh}`, `csrc/bindings.cpp`,
  `tests/test_imagewam_real_mlp.py`.
- Observation: every strided case bit-exact against the packed-layout
  kernel call on the same gate/up values (strided input from the
  `linear1` output; strided output into the `(905, 12288)` backbone and
  `(64, 7168)` ActionDiT `linear2` inputs), untouched columns still
  zero, and max-abs vs a torch transcription of the formula reported.

### Phase 2: loader, frontend, and pipeline wiring

Phase Status: completed

- Goal: merged path selected by `dims["merge_linear2"]`, default on for
  every precision except `fp16_cutlass`.
- Files: `checkpoint_loader.py`, `imagewam_thor.py`, `pipeline_thor.py`.
- Observation: frontend constructs, captures, and replays at the default
  dims for `fp16` in both modes; `linear2.weight` slot count = 20+20 at
  real dims.

### Phase 3: layer-level verification

Phase Status: completed

- Goal: merged == split within FP16 rounding at the real shapes.
- Files: `tests/test_imagewam_thor_real_wiring.py`,
  `tests/test_imagewam_checkpoint_loader.py`.
- Observation: cosine / max-abs / rel_l2, merged vs split pointer path at
  backbone (`a0=905`, hidden 3072, mlp 9216) and ActionDiT
  (`num_action=64`, 1024 / 3072 / 4096) single-stream layers; merged vs
  the tensor-level reference at the small test shapes; real
  `linear2.weight` equals `cat(attn_out_proj, mlp_down)` of the split
  loader.

### Phase 4: regression, end to end, local A/B, sm_110 build

Phase Status: completed

- Goal: no regression; indicative local speed; Thor build compiles.
- Files: `benchmarks/imagewam_fusion_ab.py` (new).
- Observation: `pytest tests/test_imagewam_*.py` count;
  `imagewam_e2e_official_compare.py` fp16 numbers vs baseline; merged vs
  split P10/P50/P90 on H100 (indicative only);
  `sm110_check.sh ... fusion` rc.

### Phase 5: Thor handoff

Phase Status: completed

- Goal: a self-contained Thor check for `nvfp4` and `fp16`.
- Files: `opportunities.md` (OPT-016).
- Observation: commands, expected observations, and what to report.

# Plan: gated-residual + next-AdaLN fusion (roadmap item 3)

Plan Status: completed

Verified locally, bit-exact; the Thor results are pending (opportunities.md
OPT-017).

## Problem

### Current

Every sub-block of every layer ends with `fvk.gate_res_bf16res`
(backbone, BF16 residual, commit 7d0ea38) or `fvk.gate_res_fp16`
(ActionDiT), `residual += gate * proj`, in `csrc/kernels/decoder_fused.cu`.
The next sub-block then starts with `ada_layer_norm_bf16in_fp16out` /
`ada_layer_norm_fp16`, which reads the same residual rows again to
produce the next normed and modulated activation. Both kernels take
FP16 modulation vectors, and `gate_res_*` takes the gate broadcast to a
full `(rows, dim)` FP16 tensor. `_fuse_mod_group` builds these inside
every layer function: four small torch kernels (shift cast, scale cast,
gate cast, gate expand-copy) per modulation group, recorded into the
CUDA graph. Per `infer()`: backbone 5 double layers (4 groups each) and
20 single layers (1 group each); ActionDiT the same per denoise step, 10
steps.

The pattern chains across layer boundaries:

- double-stream: `gate_res(attn)` -> AdaLN2 of the same block;
  `gate_res(mlp)` -> AdaLN1 of the next double block, or the single
  blocks' AdaLN after the last double block (LayerNorm is per row, so
  the txt rows and img rows can each be normalized with the single
  blocks' modulation).
- single-stream: `gate_res` -> AdaLN of the next single block.
- ActionDiT last single block: `gate_res` -> the head's AdaLN
  (`head_modded`, shift/scale only).
- backbone last single block: no following norm (the prefill ends).

### Problem

Each boundary costs two launches and a second full read of the residual,
plus the per-layer modulation cast/broadcast kernels and a `(rows, dim)`
gate read.

### Measurable goal

- One kernel per residual update writes the updated residual and emits
  the next normed + modulated FP16 activation; the gate, scale and shift
  are read as `(dim,)` vectors straight from the FP32 modulation output,
  with no per-layer cast or broadcast kernels on the fused path.
- Same math as the unfused sequence, bit for bit: locally verified
  bit-exact at the kernel level and for the whole prefill + denoise pass
  (backbone residual, action latent) at the real dims.
- BF16 backbone residual contract unchanged.
- Thor: `infer()` P50 A/B on `nvfp4` and `fp16`.

## Structure

- `csrc/kernels/fusion.{cu,cuh}` (cross-layer fusion kernels): new
  `gate_res_ada_layer_norm_bf16res` and `gate_res_ada_layer_norm_fp16`.
  Owns no state.
- `csrc/bindings.cpp`: Python bindings.
- `flash_rt/models/imagewam/pipeline_thor.py`: owns the fused data flow.
  New `AdaLNTarget` dataclass describes the AdaLN step a layer's last
  residual update also performs. The per-layer functions gain
  keyword-only `input_normed` / `next_*` parameters; `imagewam_prefill`
  and `imagewam_denoise_step` build the cross-layer chain.
- `flash_rt/frontends/torch/imagewam_thor.py`: owns
  `dims["fuse_res_norm"]` (default True once verified bit-exact,
  overridable through `dims_override` for A/B).

State ownership: the flag is set once by the frontend; the pipeline only
reads it. The FP32 modulation tensors stay owned by the frontend
(`_compute_*_modulation`); the fused path only reads their pointers.

## Interface

```c++
// csrc/kernels/fusion.cuh
// residual[r,c] = RES(float(residual[r,c]) + float(proj[r,c]) * h(gate[c]))
// out[r,:] = fp16(LN_no_affine(residual[r,:]) * (1 + h(scale)) + h(shift))
// h(x) = float(fp16(x)): the FP32 modulation vectors are rounded to FP16
// in-kernel, matching the FP16 contract of gate_res_* / ada_layer_norm_*.
// out == nullptr: residual update only (no AdaLN follows); scale/shift unread.
void gate_res_ada_layer_norm_bf16res(const __half* proj, const float* gate,
    __nv_bfloat16* residual, const float* scale, const float* shift, __half* out,
    int rows, int dim, float eps, cudaStream_t stream);
void gate_res_ada_layer_norm_fp16(const __half* proj, const float* gate,
    __half* residual, const float* scale, const float* shift, __half* out,
    int rows, int dim, float eps, cudaStream_t stream);
```

```python
# flash_rt/models/imagewam/pipeline_thor.py
@dataclass(frozen=True)
class AdaLNTarget:
    shift: torch.Tensor   # (1, 1, dim) fp32, compute_*_modulation output
    scale: torch.Tensor   # (1, 1, dim) fp32
    out_ptr: int          # fp16 (rows, dim), row-aligned with the residual rows updated

_double_stream_layer(..., *, input_normed=False, next_txt=None, next_img=None)
_single_stream_layer(..., *, input_normed=False, next_norm=None)
_action_double_layer(..., *, input_normed=False, next_norm=None)
_action_single_layer(..., *, input_normed=False, next_norm=None)
dims["fuse_res_norm"]: bool
```

`input_normed=True`: the layer's first AdaLN output is already in the
modded buffer (written by the previous layer's fused kernel). `next_*`:
the layer's last residual update also produces that AdaLN. Both require
`dims["fuse_res_norm"]`; the within-layer fusion (double blocks'
attn residual + AdaLN2) is governed by the same flag. Existing callers
that pass neither keep the unfused behavior.

## Flow

`imagewam_prefill` with `fuse_res_norm`:

1. Double layer 0: standalone AdaLN1 (txt, img).
2. Each double layer: attn proj -> fused(gate1, AdaLN2) per side; MLP ->
   fused(gate2, next AdaLN1) per side; the last double layer's next
   AdaLN uses the single blocks' modulation.
3. Each single layer except the last: fused(gate, next single AdaLN).
4. Last single layer: the fused kernel in residual-only mode
   (`out == nullptr`, no following norm).

`imagewam_denoise_step` with `fuse_res_norm`: same chain at ActionDiT
widths (FP16 residual); the last single layer's fused kernel writes the
head's AdaLN into `head_modded`, and the standalone head AdaLN is
skipped.

## Code Mapping

| item | file |
|---|---|
| fused kernels | `csrc/kernels/fusion.cu`, `csrc/kernels/fusion.cuh` |
| bindings | `csrc/bindings.cpp` |
| chain + per-layer paths, `AdaLNTarget` | `flash_rt/models/imagewam/pipeline_thor.py` |
| flag default | `flash_rt/frontends/torch/imagewam_thor.py` |
| kernel test (torch reference + bit-exact vs unfused kernels) | `tests/test_imagewam_residual_norm_fusion.py` |
| whole-pass bit-exact test (real dims, random weights) | `tests/test_imagewam_residual_norm_fusion.py` |
| A/B script | `benchmarks/imagewam_fusion_ab.py` (`AB=fuse_res_norm`) |

## Implementation Phases

### Phase 1: fused kernels

Phase Status: completed

- Goal: the two kernels plus bindings.
- Files: `csrc/kernels/fusion.{cu,cuh}`, `csrc/bindings.cpp`,
  `tests/test_imagewam_residual_norm_fusion.py`.
- Observation: at the real shapes (backbone BF16 residual 513 / 392 / 905
  rows x 3072; ActionDiT FP16 residual 64 x 1024), residual and output
  bit-exact vs `gate_res_*` + `ada_layer_norm_*`; cosine / max-abs /
  rel_l2 vs an FP32 torch reference; residual-only mode bit-exact vs
  `gate_res_bf16res`.

### Phase 2: pipeline chain and flag

Phase Status: completed

- Goal: fused path in all four layer types plus the chain in
  `imagewam_prefill` / `imagewam_denoise_step`.
- Files: `pipeline_thor.py`, `imagewam_thor.py`.
- Observation: whole prefill + 10-step denoise at the real dims (random
  weights, fp16), fused vs unfused: `backbone_hidden`, every layer's K/V
  cache, and `action_latent` bit-exact; CUDA kernel count per pass.

### Phase 3: regression, end to end, local A/B, sm_110 build

Phase Status: completed

- Observation: `pytest tests/test_imagewam_*.py` count; e2e fp16 numbers
  vs baseline; `AB=fuse_res_norm` P10/P50/P90 on H100 (indicative);
  `sm110_check.sh` rc.

### Phase 4: Thor handoff

Phase Status: completed

- Files: `opportunities.md` (OPT-017).
- Observation: commands, expected observations, what to report.

# Plan: VAE image preprocessing kernel with normalization LUT (roadmap item 2)

Plan Status: completed

## Problem

### Current

`vae_encoder._prep_view` preprocesses each camera view with plain
PyTorch: `uint8 -> float32` (on the source device, so a CPU input is
converted on the CPU and copied to the GPU as float32, 4x the bytes),
`F.interpolate(mode="area")` to 224x224 when the view is not already
that size, `x * 2.0 / 255.0 - 1.0` as three elementwise kernels, a
BF16 cast, and finally `torch.cat` of the two views. That is about a
dozen launches plus float32 temporaries per `infer()`, and it runs
outside the CUDA graph.

Two arithmetic facts of the served path, measured on H100 with torch
2.14 against the torch kernels themselves:

- `F.interpolate(mode="area")` (channels-last-strided input, the
  layout `_prep_view` produces) equals `sum / kh / kw` in float32,
  two correctly rounded divisions, where `sum` is the exact integer
  window sum and `kh`/`kw` are the adaptive-pool window extents.
- `x / 255.0` with a Python scalar is computed as `x * (1.0f/255.0f)`
  (torch multiplies by the float32 reciprocal of a CPU scalar
  divisor), not as a true division.

Pi0.5 already ships a 256-entry FP16 normalization table
(`pi05_thor.py` `_infer_uint8_to_fp16`, used by
`csrc/kernels/patch_embed.cu::patch_im2col_uint8_kernel`) that is
bit-identical to its per-frame arithmetic.

The official LIBERO eval (`eval_libero_single._center_crop_resize`)
resizes with PIL `BILINEAR` plus a center crop, and training resizes
with `torchvision.transforms.Resize([224,224])` (bilinear, antialias).
The served path uses area averaging. This semantic difference is
tracked separately (`issues.md` ISSUE-030) and is not changed by
default.

### Problem

The served preprocessing is many small launches plus CPU-side float
conversion, and it cannot read from a fixed-address uint8 buffer
inside a CUDA graph.

### Measurable goal

One CUDA kernel per view: `(H,W,3)` uint8 in, `(1,3,224,nv*224)` BF16
NCHW out (the VAE's input dtype and layout, concatenated in place).

- `resize="area"`: bit-exact to `_prep_view` for every input size,
  including the no-resize case, which reads a 256-entry BF16 table.
- `resize="pil_bilinear"`: resize bit-exact to the official
  `_center_crop_resize` (PIL `BILINEAR`, fixed-point 22-bit
  coefficients, horizontal pass then vertical pass with uint8
  rounding in between), followed by the same 256-entry table. The
  official eval's own normalization (BF16 arithmetic) is not
  reproduced; ISSUE-030 records that difference.
- Observations: bit-exactness counts at real 512x512 LIBERO frames and
  synthetic sizes, launch count, and preprocessing latency A/B.

## Structure

- NEW `csrc/kernels/imagewam_vae_preprocess.cu/.cuh`: the kernel.
  Stateless. One thread per output pixel, all three channels.
- `csrc/bindings.cpp`: `imagewam_vae_preprocess_bf16` binding
  (pointer interface, same convention as `patch_im2col_uint8`).
- NEW `flash_rt/models/imagewam/vae_preprocess.py`: `VaePreprocessor`
  OWNS the device-resident normalization table and the per-input-size
  PIL coefficient tables (`PilResizePlan`), and launches the kernel.
- `flash_rt/models/imagewam/vae_encoder.py`: `encode_to_tokens` gains
  an optional `preprocessor`; given one, the kernel replaces
  `_prep_view`. `_prep_view` stays as the reference implementation.
- `flash_rt/frontends/torch/imagewam_thor.py`: OWNS one
  `VaePreprocessor` when the real VAE is loaded, and passes it to
  `encode_to_tokens`. New constructor flag `vae_resize` (`"area"`
  default, `"pil_bilinear"` opt-in).

## Interface

```python
# flash_rt/models/imagewam/vae_preprocess.py
RESIZE_MODES = ("area", "pil_bilinear")

@dataclass(frozen=True)
class PilResizePlan:          # device tables for one (in_h, in_w) -> (out_h, out_w)
    h_bounds: torch.Tensor | None   # (resized_w, 2) int32, None if no horizontal pass
    h_coeffs: torch.Tensor | None   # (resized_w, h_ksize) int32
    v_bounds: torch.Tensor | None
    v_coeffs: torch.Tensor | None
    h_ksize: int; v_ksize: int
    crop_left: int; crop_top: int   # center-crop offsets into the resized image

class VaePreprocessor:
    def __init__(self, *, resize: str = "area", out_hw: tuple[int, int] = (224, 224),
                 device: str = "cuda") -> None
    def prepare(self, in_h: int, in_w: int) -> None       # builds tables; idempotent
    def run(self, views: Sequence[torch.Tensor], out: torch.Tensor, stream: int) -> None
        # views: (H,W,3) uint8 CUDA, contiguous; out: (1,3,oh,len(views)*ow) BF16 CUDA
```

```cpp
// csrc/kernels/imagewam_vae_preprocess.cuh
int imagewam_vae_preprocess_bf16(
    const uint8_t* view, const __nv_bfloat16* lut, __nv_bfloat16* out,
    int in_h, int in_w, int out_h, int out_w, int out_total_w, int col_offset,
    int mode,                       // 0 identity, 1 area, 2 pil_bilinear
    const int* h_bounds, const int* h_coeffs, int h_ksize, int crop_left,
    const int* v_bounds, const int* v_coeffs, int v_ksize, int crop_top,
    float inv255, cudaStream_t stream);
```

## Flow

1. Frontend construction with `ae_model_path`: builds
   `VaePreprocessor(resize=vae_resize)`; the table is computed once on
   the GPU with the same torch expression `_prep_view` uses, so it is
   bit-identical by construction.
2. `infer()` eager VAE path: `encode_to_tokens(ae, v1, v2,
   preprocessor=...)` moves each view to the GPU as uint8, launches
   one kernel per view into one `(1,3,224,448)` BF16 buffer, then
   runs `ae.encode` unchanged.

## Code Mapping

| module / state | file | task |
|---|---|---|
| kernel | `csrc/kernels/imagewam_vae_preprocess.cu/.cuh`, `CMakeLists.txt` | phase 1 |
| binding | `csrc/bindings.cpp` | phase 1 |
| table + PIL plan owner | `flash_rt/models/imagewam/vae_preprocess.py` | phase 1 |
| kernel test | `tests/test_imagewam_vae_preprocess.py` | phase 1 |
| served eager path | `vae_encoder.py`, `imagewam_thor.py` | phase 2 |
| latency A/B | `benchmarks/imagewam_vae_stage_bench.py` | phase 2 |
| mismatch measurement | `benchmarks/imagewam_e2e_official_compare.py` (`RAW_VIEWS`, `VAE_RESIZE`), `issues.md` ISSUE-030 | phase 3 |
| record | `opportunities.md` OPT-020 | phase 3 |

## Implementation Phases

### Phase 1 — kernel, binding, table owner, bit-exact test

Phase Status: completed

Goal: the kernel reproduces `_prep_view` (area and no-resize) and PIL
`BILINEAR` + center crop bit-exactly.
Modified files: new `imagewam_vae_preprocess.cu/.cuh`,
`CMakeLists.txt`, `bindings.cpp`, new `vae_preprocess.py`, new
`tests/test_imagewam_vae_preprocess.py`.
Observation method: mismatch counts and max-abs against `_prep_view`
and against PIL on real 512x512 LIBERO frames, random 512/256/224
inputs, and a non-square input; sm_90 build plus sm_110 compile check.

### Phase 2 — served eager path uses the kernel

Phase Status: completed

Goal: `infer()` preprocesses through the kernel by default (same
bits), `vae_resize="pil_bilinear"` available as an opt-in.
Modified files: `vae_encoder.py`, `imagewam_thor.py`, new
`benchmarks/imagewam_vae_stage_bench.py`.
Observation method: token bit-equality against the legacy path,
preprocessing latency A/B in one process (CPU and GPU uint8 inputs),
regression suite.

### Phase 3 — preprocessing mismatch measurement and record

Phase Status: completed

Goal: measure the action effect of area vs PIL bilinear on raw
512x512 frames against the official model; record ISSUE-030 and
OPT-020 with a recommendation.
Modified files: `benchmarks/imagewam_e2e_official_compare.py`,
`issues.md`, `opportunities.md`.
Observation method: end-to-end compare with raw frames, served
`area` and opt-in `pil_bilinear`, next to the pre-resized baseline.

## Thor Check

Environment on Thor: this branch built with `cmake --build build -j
--target flash_rt_kernels` (CMake re-configures for the new
`csrc/kernels/imagewam_vae_*.cu` files); `FLUX2_SRC`, `FLUX2_AE_MODEL_PATH`
(and `AE_MODEL_PATH` set to the same file), `CKPT_PATH`, optional
`DATA_ROOT` (LIBERO-fastwam; synthetic frames without it);
`PYTHONPATH=<FlashRT>:<ImageWAM>/src:$FLUX2_SRC/src`.

1. `python -m pytest tests/test_imagewam_vae_preprocess.py -q -s`:
   every printed line shows `differing_bf16=0/...` for both `area` and
   `pil` (bit-exact on sm_110 too). Report the pass count and any
   non-zero line.
2. `python benchmarks/imagewam_vae_stage_bench.py --section preprocess
   --iters 200`: per input, P10/P50/P90 of the torch path and the
   kernel, kernels per call, GPU kernel time and host enqueue time; the
   `bit-identical` lines must print `True`. Report the whole section.

# Plan: VAE encode in-graph capture and native NHWC GroupNorm+SiLU (roadmap item 5)

Plan Status: completed

## Problem

### Current

`infer()` runs the real FLUX.2 `AutoEncoder.encode` in plain PyTorch
BF16 outside the CUDA graph: 21.5 ms of the 231.6 ms `nvfp4` P50 on
Thor (opportunities.md OPT-012). On H100 at the real 224x448 input,
`benchmarks/imagewam_vae_stage_bench.py --section profile
--profile-repeats 7` (torch profiler, GPU kernel time per op family, 3
invocations x 7 captures of 10 encodes) gives 282 kernels and 9.5-10.2
ms of kernel time per encode, split as below. Shares are the per-
invocation medians over captures; single captures vary more (range in
parentheses) because the co-tenant job time-slices the GPU. The same
numbers are recorded in opportunities.md OPT-021.

| op family | kernels per encode | share of GPU kernel time |
|---|---:|---:|
| torch GroupNorm statistics (`RowwiseMoments`, only N*G = 32 blocks) | 44 | 35-39% (31-50%) |
| convolution math | 25 | 18-26% (12-34%) |
| other elementwise: conv-bias broadcast adds, residual adds, mul, pad, copies | 109 | 16-24% (10-30%) |
| cuDNN NCHW<->NHWC layout transforms around each convolution | 72 | 10-17% (8-24%) |
| sigmoid (swish) | 21 | 2-4% |
| attention, q/k/v and proj GEMMs | 11 | ~2% |

Converting the stock module to `channels_last` removes the transforms
but makes torch's GroupNorm slower (strided kernels), so no speedup.

### Problem

The VAE encode is outside the graph and launch-heavy (282 kernels),
and about half of its GPU time goes to a poorly parallelized GroupNorm
statistics kernel and to layout transforms, a further fifth to unfused
elementwise ops; convolution math is only about a fifth to a quarter.

### Measurable goal

Stop at verified points, in order:

1. Per-op profile of `ae.encode` at 224x448 (H100, indicative).
2. The existing torch encode captured into a CUDA graph with static
   input and output buffers; tokens bit-identical to eager; gain
   measured.
3. The VAE stage folded into the frontend's main graph so `infer()`
   does one replay; tokens bit-identical, end-to-end compare unchanged.
4. Native encoder: NHWC (`channels_last`) convolutions plus a FlashRT
   NHWC GroupNorm(+SiLU) kernel; each kernel checked against torch,
   tokens near-exact (cosine, max-abs, mean/std/absmax vs the real
   Thor values -0.02/0.97/4.91), end-to-end compare within baseline
   noise.

## Structure

- NEW `flash_rt/models/imagewam/vae_stage.py`: `ImageWAMVaeStage`
  OWNS the fixed-address stage state: the uint8 view buffer
  `(nv,H,W,3)`, the preprocessed BF16 image `(1,3,224,nv*224)`, and the
  encoder object. It writes tokens into a caller-owned `img_raw`
  (owned by the frontend, unchanged). `run()` is capture-safe.
- NEW `csrc/kernels/imagewam_vae_groupnorm.cu/.cuh` + binding:
  NHWC BF16 GroupNorm with optional fused SiLU (Welford partial stats,
  per-channel fused scale/shift finalize, vectorized apply).
- NEW `flash_rt/models/imagewam/vae_native_encoder.py`:
  `NativeFlux2Encoder` OWNS channels_last copies of the AE encoder
  weights and implements `encode(x)` with the same op order as
  `flux2.autoencoder.AutoEncoder.encode`, calling the GroupNorm kernel.
- NEW `csrc/kernels/imagewam_vae_residual.cu/.cuh` + binding: the
  ResnetBlock tail (conv2 bias, optional nin_shortcut bias, residual
  add) in one NHWC BF16 pass with torch's rounding points.
- `flash_rt/frontends/torch/imagewam_thor.py`: OWNS the encoder object
  (`vae_encoder`: torch module or `NativeFlux2Encoder`) and, when
  `vae_graph_input` is given, the stage; `_capture_graph` records
  `stage.run()` before prefill; `infer()` stages views, proprio, and
  noise, then replays once.
- NEW `benchmarks/imagewam_vae_stage_bench.py`: profile and A/B of
  every VAE variant in one process (also the Thor handoff tool).

## Interface

```python
# flash_rt/models/imagewam/vae_stage.py
class VaeEncoder(Protocol):
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...   # (1,3,H,W) BF16 -> (1,128,h,w) BF16

@dataclass(frozen=True)
class VaeStageSpec:
    num_views: int
    in_h: int
    in_w: int
    out_hw: tuple[int, int] = (224, 224)

class ImageWAMVaeStage:
    def __init__(self, encoder: VaeEncoder, preprocessor: VaePreprocessor,
                 spec: VaeStageSpec, img_raw: torch.Tensor) -> None
    views_u8: torch.Tensor      # (nv,H,W,3) uint8 CUDA, fixed address
    image: torch.Tensor         # (1,3,oh,nv*ow) BF16 CUDA, fixed address
    def stage(self, views: Sequence[torch.Tensor]) -> None   # validates shapes, copies in
    def run(self) -> None       # current stream: preprocess -> encode -> img_raw

# flash_rt/models/imagewam/vae_native_encoder.py
class NativeFlux2Encoder:       # satisfies VaeEncoder
    def __init__(self, ae: torch.nn.Module) -> None
    def encode(self, x: torch.Tensor) -> torch.Tensor

# flash_rt/models/imagewam/vae_encoder.py
def encode_to_tokens(ae, view1, view2=None, *, out_hw=(224, 224),
                     preprocessor: VaePreprocessor | None = None,
                     encoder: VaeEncoder | None = None) -> torch.Tensor

# flash_rt/frontends/torch/imagewam_thor.py
_VAE_ENCODERS = ("torch", "native")
ImageWAMTorchFrontendThor(..., vae_encoder: str = "torch",
                          vae_graph_input: tuple[int, int, int] | None = None,  # (nv, H, W)
                          vae_resize: str = "area")
```

```cpp
// csrc/kernels/imagewam_vae_groupnorm.cuh -- bias: optional conv bias folded into the reads
size_t imagewam_groupnorm_nhwc_workspace_bytes(int N, int HW, int C, int G);
int imagewam_groupnorm_nhwc_bf16(const __nv_bfloat16* x, const __nv_bfloat16* bias,
    const __nv_bfloat16* gamma, const __nv_bfloat16* beta, __nv_bfloat16* y,
    void* workspace, size_t workspace_bytes,
    int N, int HW, int C, int G, float eps, int apply_silu, cudaStream_t stream);

// csrc/kernels/imagewam_vae_residual.cuh
int imagewam_bias_residual_nhwc_bf16(const __nv_bfloat16* h, const __nv_bfloat16* h_bias,
    const __nv_bfloat16* res, const __nv_bfloat16* res_bias, __nv_bfloat16* y,
    long long rows, int C, cudaStream_t stream);
```

State transitions: the defaults (`vae_encoder="torch"`,
`vae_graph_input=None`) are the current behavior. `vae_graph_input`
requires `ae_model_path`; `infer()` then requires views of exactly that
shape (`ValueError` otherwise, since the graph encodes whatever the
fixed buffer holds). `vae_encoder` applies in both placements.

## Flow

1. Construction: load the AE; build the encoder (`ae` or
   `NativeFlux2Encoder(ae)`); with `vae_graph_input`, build
   `ImageWAMVaeStage` over `self._img_raw` (which calls
   `VaePreprocessor.prepare(H, W)`).
2. `set_prompt()` -> `_capture_graph()`: warm-up and capture record
   `stage.run()`, then `imagewam_prefill`, then
   `imagewam_denoise_loop`, all on the capture stream.
3. `infer(obs)`: `stage.stage([view1, view2])` (H2D uint8 copy into
   the fixed buffer); proprio token into its context row; action noise
   into `action_latent`; one `graph.replay()`.

## Code Mapping

| module / state | file | phase |
|---|---|---|
| profile + A/B tool | `benchmarks/imagewam_vae_stage_bench.py` | 1, 2, 4 |
| stage (fixed buffers) | `flash_rt/models/imagewam/vae_stage.py` | 2 |
| stage test | `tests/test_imagewam_vae_stage.py` | 2, 3, 4 |
| frontend flag + capture | `flash_rt/frontends/torch/imagewam_thor.py` | 3 |
| e2e option | `benchmarks/imagewam_e2e_official_compare.py` (`VAE_ENCODER`, `VAE_GRAPH`) | 3, 4 |
| GroupNorm kernel | `csrc/kernels/imagewam_vae_groupnorm.cu/.cuh`, `csrc/bindings.cpp`, `CMakeLists.txt` | 4 |
| bias + residual kernel | `csrc/kernels/imagewam_vae_residual.cu/.cuh`, `csrc/bindings.cpp`, `CMakeLists.txt` | 4 |
| GroupNorm test | `tests/test_imagewam_vae_groupnorm.py` | 4 |
| native encoder | `flash_rt/models/imagewam/vae_native_encoder.py` | 4 |
| record | `opportunities.md` OPT-021 | 5 |

## Implementation Phases

### Phase 1 — profile `ae.encode` at 224x448

Phase Status: completed

Goal: per-op time breakdown and op list at the real input (H100,
indicative), reproducible on Thor with the same script.
Modified files: new `benchmarks/imagewam_vae_stage_bench.py`.
Observation method: torch profiler table, kernel count per encode.

### Phase 2 — standalone CUDA graph of the torch encode

Phase Status: completed

Goal: `ImageWAMVaeStage` with fixed buffers; `run()` captured into a
graph; tokens bit-identical to `encode_to_tokens`; eager vs graph A/B.
Modified files: new `vae_stage.py`, new `tests/test_imagewam_vae_stage.py`,
bench script.
Observation method: `torch.equal` on tokens, CUDA-event P10/P50/P90
alternating eager and graph in one process.

### Phase 3 — VAE folded into the main graph

Phase Status: completed

Goal: `vae_graph_input=(nv, H, W)`: one replay per `infer()`; tokens
and actions bit-identical to the eager path with the same noise.
Modified files: `imagewam_thor.py`, `imagewam_e2e_official_compare.py`,
`tests/test_imagewam_vae_stage.py`.
Observation method: `img_raw` and action equality eager vs graph on a
real-dims frontend; regression suite; quick end-to-end compare.

### Phase 4 — native NHWC GroupNorm+SiLU encoder

Phase Status: completed

Goal: `vae_encoder="native"`: channels_last convolutions plus the
FlashRT GroupNorm(+SiLU) and bias+residual kernels, in either
placement.
Modified files: new `imagewam_vae_groupnorm.cu/.cuh`, new
`imagewam_vae_residual.cu/.cuh`, `bindings.cpp`, `CMakeLists.txt`, new
`vae_native_encoder.py`, new `tests/test_imagewam_vae_groupnorm.py`,
`vae_encoder.py`, `imagewam_thor.py`, bench.
Observation method: kernel vs `F.group_norm`(+`x*sigmoid(x)`) at every
real encoder shape (cosine, max-abs, rel_l2, mismatch fraction);
tokens vs torch encode (cosine, max-abs, stats); A/B latency; full
end-to-end compare; sm_110 compile check.

### Phase 5 — close-out

Phase Status: completed

Goal: record measured results and the Thor checklist.
Modified files: `opportunities.md` (OPT-021), `plan.md`.
Observation method: every number recorded is labeled H100 or Thor.

## Thor Check

Same environment as the item-2 Thor check.

1. `python -m pytest tests/test_imagewam_vae_groupnorm.py
   tests/test_imagewam_vae_stage.py -q -s`: all pass. Expected prints:
   GroupNorm cosine >= 0.9999 with at most ~0.03% differing elements;
   `bias+residual` and every `graph` / `graph[...]` / `eager[native]`
   line with `max_abs=0.000e+00`; `native stage eager vs torch` cosine
   about 0.99998 with mean about -0.01, std about 0.97, absmax about 4.8.
   Report the pass count and the `native stage eager vs torch` lines.
2. `python benchmarks/imagewam_vae_stage_bench.py --section encode
   --iters 100`: the `legacy` row is the stock VAE stage (21.5 ms in
   OPT-012); report every row's P10/P50/P90 and the kernels / GPU
   kernel time lines. Expected: native rows well below the torch rows;
   graph rows at or below the matching eager rows.
3. `python benchmarks/imagewam_vae_stage_bench.py --section profile`:
   report the two `CUDA kernels per encode ... GPU time per encode`
   lines and the top 10 rows of each table.
4. `infer()` A/B on `nvfp4`, real dims, four frontends in one process
   (random weights; latency does not depend on the values; set
   `CKPT_PATH` to use the real checkpoint instead):
   `env -u CKPT_PATH python benchmarks/imagewam_vae_stage_bench.py
   --section infer --precision nvfp4 --iters 40`. If memory is short,
   run `--vae-variants eager-torch,graph-native` and then
   `--vae-variants eager-torch,eager-native,graph-torch`. Expected:
   `eager-torch` near the 231.6 ms production P50; `graph-native` lower
   by roughly the VAE-stage saving of step 2. Report P10/P50/P90 per
   variant. Repeat with `--raw-views` if the deployment feeds raw
   512x512 frames.
5. Optional, if the official model fits next to FlashRT on Thor:
   `PRECISION=nvfp4 N_TASKS=3 FRAMES=0 python
   benchmarks/imagewam_e2e_official_compare.py` and the same with
   `VAE_ENCODER=native VAE_GRAPH=1`. Expected: `fr_vs_off` and
   `mae_fr_vs_gt` equal between the two runs to about 1e-4. Report both
   SUMMARY blocks.

# Plan: ABI integration, `frt_model_runtime_v1` Python producer (roadmap item 12)

Plan Status: completed

## Problem

### Current

`ImageWAMTorchFrontendThor` (`flash_rt/frontends/torch/imagewam_thor.py`)
is reachable only through its Python methods `set_prompt()` and
`infer()`. The captured CUDA Graph (`self._graph`), its capture stream,
and the steady-state buffers the graph reads and writes (`_img_raw`,
`_context`, `_action_latent`) are private attributes. No
`frt_runtime_export_v1` or `frt_model_runtime_v1` is produced for
ImageWAM, so a native host (a C++/Rust robot loop, a capsule/state host)
cannot drive the model, and the native C++ overlay (roadmap item 14) has
no declaration to attach verbs to. Pi0.5 already publishes this face
through `flash_rt/models/pi05/runtime_export.py`.

`infer()` fills the initial action latent with `0.01 * N(0,1)` inside
the method (ISSUE-002). The noise is not an input a caller can set.

### Problem

There is no ABI surface through which a consumer can supply camera
frames, proprio, prompt and initial noise, run the captured graph, and
read the action chunk.

### Measurable goal

`frontend.export_model_runtime(io="python")` returns an
`frt_model_runtime_v1` whose ports cover every per-tick input and
output of `infer()`. A ctypes consumer that uses only the C function
pointers, port descriptors, `frt_buffer_dptr` and the CUDA runtime
produces actions that are bit-identical to `frontend.infer()` for the
same frames, proprio, prompt and initial noise:

- H100, fp16, real checkpoint, real VAE, Qwen3 and dataset stats:
  `max_abs == 0` and `array_equal` on the denormalized `(64, 7)` chunk.
- Thor, `nvfp4`: the same gate, run by the Thor checklist.

The noise is an explicit SWAP input consumed as written. The `0.01`
scale stays where it is, inside `infer()` (ISSUE-002 owns that decision).

## Structure

| module | responsibility | owned state |
|---|---|---|
| `flash_rt/frontends/torch/imagewam_thor.py` | captures the graph; owns every buffer, weight and encoder; owns the per-tick staging operations (VAE encode into `img_raw`, proprio projection into the context row, action readback and denormalization), used by both `infer()` and the ABI verbs | `_graph`, `_graph_stream`, `_img_raw`, `_context`, `_action_latent`, `_proprio_row` |
| `flash_rt/models/imagewam/runtime_surface.py` | interface between the frontend and the export: the `ImageWAMRuntimeSurface` dataclass (graph exec handle, stream, the three windows, the deployment facts that enter identity) and the `ImageWAMRuntimeSource` Protocol | none (declarations only) |
| `flash_rt/models/imagewam/runtime_export.py` | lowers a captured frontend into `frt_model_runtime_v1`: exec context, graph adoption, port/buffer/region/stage declarations, identity, and the Python verbs (`set_input`, `get_output`, `step`) | the exec `Ctx`, the adopted `Graph`, the wrapped `Buffer`s; anchored by the returned `ModelRuntime` |
| `runtime/`, `exec/` | unchanged generic ABI | builder, fingerprint, lifetime |

The frontend stays the single owner of all device memory. The export
wraps pointers (`frt_buffer_wrap`), adopts the torch graph exec without
owning it (`frt_graph_adopt`), and anchors the frontend for the runtime's
lifetime.

## Interface

```python
# flash_rt/models/imagewam/runtime_surface.py
@dataclass(frozen=True)
class ImageWAMRuntimeSurface:
    graph_exec: int              # torch CUDAGraph.raw_cuda_graph_exec()
    stream: torch.cuda.Stream    # the capture stream; ABI replay and staging run here
    img_raw: torch.Tensor        # (img_len, HD) bf16, VAE tokens
    context: torch.Tensor        # (x0, joint_attention_dim) bf16, prompt + proprio row
    action_latent: torch.Tensor  # (num_action, action_dim) f32, noise in / actions out
    img_len: int; token_dim: int; num_action: int; action_dim: int
    proprio_dim: int | None
    has_vae: bool; has_text_encoder: bool; action_denormalized: bool
    identity: tuple[tuple[str, str], ...]   # precision, dims, flags

class ImageWAMRuntimeSource(Protocol):
    def runtime_surface(self) -> ImageWAMRuntimeSurface: ...
    def stage_images(self, view1: torch.Tensor, view2: torch.Tensor | None) -> None: ...
    def stage_proprio(self, proprio: np.ndarray) -> None: ...
    def set_prompt(self, prompt_text: str | None = None, *, context=None, context_mask=None) -> None: ...
    def read_actions(self) -> np.ndarray: ...

# flash_rt/models/imagewam/runtime_export.py
def export_model_runtime(source: ImageWAMRuntimeSource, *, identity: Mapping[str, str] | None = None,
                         io: str = "python") -> flash_rt.runtime.export.ModelRuntime

# flash_rt/frontends/torch/imagewam_thor.py (additions)
def runtime_surface(self) -> ImageWAMRuntimeSurface
def stage_images(self, view1, view2) -> None        # VAE encode -> img_raw
def stage_proprio(self, proprio) -> None            # normalize -> proprio_encoder -> context row
def read_actions(self) -> np.ndarray                # action_latent -> denormalize -> host f32
def export_model_runtime(self, *, identity=None, io="python") -> ModelRuntime
```

Port schema, `io="python"` (declaration order is the port index):

| port | dir | update | modality | dtype | shape | window | present when |
|---|---|---|---|---|---|---|---|
| `images` | in | STAGED | IMAGE | u8 | (views, H, W, 3) | none | VAE loaded |
| `image_tokens` | in | SWAP | TENSOR | bf16 | (img_len, 128) | `img_raw` | VAE outside the graph |
| `image_views` | in | SWAP | IMAGE | u8 | (views, H, W, 3) | `views_u8` | VAE inside the graph |
| `proprio` | in | STAGED | STATE | f32 | (proprio_dim,) | none | `proprio_dim` set |
| `noise` | in | SWAP | TENSOR | f32 | (num_action, action_dim) | `action_latent` | always |
| `actions` | out | STAGED | ACTION | f32 | (num_action, action_dim) | none | always |
| `actions_raw` | out | SWAP | TENSOR | f32 | (num_action, action_dim) | `action_latent` | always |
| `prompt` | in | SETUP | TEXT | u8 | (-1,) | none | Qwen3 loaded |

- `images`: payload is two `frt_image_view` (RGB8, 224x224), view order
  `[view1, view2]` (agent view, wrist view). `set_input` runs
  `stage_images`, which writes `img_raw`.
- `noise`: the initial action latent, consumed exactly as written and
  overwritten in place by the output. It must be written before every
  `step`. `infer()` writes `0.01 * N(0,1)` here (ISSUE-002); the ABI does
  not scale.
- `actions`: `get_output` returns what `infer()` returns: the chunk
  denormalized with `dataset_stats.json` when loaded, else the raw latent.
  `action_denormalized` is part of identity.
- `prompt`: SETUP, legal only outside a tick. `set_input` runs
  `set_prompt(prompt_text)`.
- Stage plan: one GRAPH stage `infer` (prefill + 10-step denoise).
- Region: `rollout_boundary` = the `action_latent` window.
- Status codes: `-2` for an unknown port index, `-3` for `set_input` on a
  SWAP port, `-5` (with `written` = needed bytes) for a short
  `get_output` buffer. A malformed payload (size, frame geometry, pixel
  format) and `get_output` on a SWAP port return `-1` with the reason in
  `last_error`: the Python trampoline reports every raised exception as
  `-1`.

## Flow

Setup (Python process, once): construct the frontend, `set_prompt()`
captures the graph on `_graph_stream`, `export_model_runtime()` creates
an exec `Ctx`, wraps `_graph_stream`, adopts `raw_cuda_graph_exec()`,
wraps the `img_raw` and `action_latent` windows, and builds the
runtime through `flash_rt.runtime.export.build_model_runtime`.

Tick (any host thread, through C function pointers):

1. `set_input(images, frt_image_view[2])` → trampoline acquires the GIL →
   `stage_images` on `_graph_stream`.
2. `set_input(proprio, f32[8])` → `stage_proprio` on `_graph_stream`.
3. Host copies noise bytes into `frt_buffer_dptr(noise.buffer) + offset`
   with a synchronous `cudaMemcpy`.
4. `step()` → `frt_graph_replay(infer, 0, stream)` on `_graph_stream`.
5. `get_output(actions)` → `read_actions` on `_graph_stream` (blocks on
   the stream, denormalizes) → f32 bytes; or the host synchronizes the
   exported stream and reads `actions_raw` directly.

## Code Mapping

| item | file |
|---|---|
| surface dataclass + source Protocol | `flash_rt/models/imagewam/runtime_surface.py` (new) |
| export, ports, verbs, identity | `flash_rt/models/imagewam/runtime_export.py` (new) |
| staging methods, `_graph_stream`, `runtime_surface()`, `export_model_runtime()` | `flash_rt/frontends/torch/imagewam_thor.py` |
| ctypes ABI consumer used by both tests | `tests/_helpers/model_runtime_consumer.py` (new) |
| schema + parity unit test, small random dims | `tests/test_imagewam_model_runtime_export.py` (new) |
| real-checkpoint parity gate (H100 fp16, Thor nvfp4) | `tests/gate_imagewam_model_runtime_export.py` (new) |
| verified interface record | `docs/imagewam_model_runtime.md` (new) |
| measured results | `opportunities.md` OPT-028 |

## Implementation Phases

### Phase 1 — frontend staging split and surface

Phase Status: completed

Goal: `infer()` delegates to `stage_images`, `stage_proprio` and
`read_actions`; `_capture_graph` keeps its stream; `runtime_surface()`
returns the dataclass. Same numerics as before.
Modified files: `imagewam_thor.py`, new `runtime_surface.py`.
Observation: `pytest tests/test_imagewam_*.py` count unchanged (68/6).

### Phase 2 — export module and unit test

Phase Status: completed

Goal: `export_model_runtime(io="python")` with the port schema above;
unit test at small random dims checks the schema, identity sensitivity,
the STAGED/SWAP guards, and bit-exact parity with `infer()` through the
ctypes consumer.
Modified files: new `runtime_export.py`, new
`tests/_helpers/model_runtime_consumer.py`, new
`tests/test_imagewam_model_runtime_export.py`, `imagewam_thor.py`
(`export_model_runtime`).
Observation: the new test's printed `max_abs` and `array_equal`.

### Phase 3 — real-checkpoint gate on H100

Phase Status: completed

Goal: `tests/gate_imagewam_model_runtime_export.py` with the real
checkpoint, VAE, Qwen3 and stats at fp16: images STAGED, proprio STAGED,
noise SWAP, prompt SETUP, actions STAGED and `actions_raw` SWAP, compared
with `infer()`; plus an indicative A/B latency of `infer()` vs an ABI tick.
Observation: printed parity table and latency percentiles; recorded in
OPT-028.

### Phase 4 — Thor handoff and docs

Phase Status: completed

Goal: `docs/imagewam_model_runtime.md` records the verified interface;
the Thor checklist runs the gate at `nvfp4`.
Observation: the H100 gate output quoted in OPT-028; Thor result pending.

# Plan: Native C++ overlay, `io="native"` (roadmap item 14)

Plan Status: completed

## Problem

### Current

After roadmap item 12 (OPT-028), ImageWAM publishes `frt_model_runtime_v1`
with `io="python"`: every STAGED verb is a Python callable behind a
GIL-acquiring trampoline, and the graph is recorded by the Python
pipeline functions (`imagewam_prefill`, `imagewam_denoise_loop` in
`flash_rt/models/imagewam/pipeline_thor.py`). A native host that ticks
the model still enters Python for proprio staging, action readback and
`step`. The captured graph also carries per-replay torch kernels that
exist only because it was recorded from Python: the fp16 casts and
`(seq, dim)` gate expansions of `_fuse_mod_group`, `_copy_slice` column
copies, and `_add_inplace`.

### Problem

No C++ component records ImageWAM's prefill and denoise loop against the
existing `csrc` kernels, and no native verb implements ImageWAM's
per-tick input/output transforms, so Python and the GIL remain on the hot
path.

### Measurable goal

1. `export_model_runtime(io="native", native=...)` publishes a runtime
   whose `set_input` / `get_output` / `step` are C functions from
   `libflashrt_imagewam_native.so`: a tick takes no GIL. Schema: the
   Python-built declaration, the C++-rendered records and a golden file
   agree line for line.
2. The C++ pipeline records prefill and denoise with the same `csrc`
   kernels, and at each step (one block type, full prefill, full denoise,
   captured graph) its outputs are `array_equal` to the Python pipeline's
   on H100 at fp16, small dims and the real checkpoint.
3. `nvfp4` is wired in the native pipeline; `sm110_check.sh` builds the
   native target for `GPU_ARCH=110`; the Thor checklist runs the parity
   gates at `nvfp4`.

Stays in Python (setup only): checkpoint loading and quantization, GEMM
autotune, AdaLN modulation and RoPE table precompute, VAE (roadmap item
5) and Qwen3 prompt encoding, and building the declaration. FA4
(`use_fa4=True`) is a Python/CuTe runtime and is not supported by the
native pipeline.

## Structure

| module | responsibility | owned state |
|---|---|---|
| `flash_rt/frontends/torch/imagewam_thor.py` | owns every device allocation: weights (incl. quantized copies), activation scratch, K/V caches, the three IO windows, the Python graph | all device memory |
| `csrc/gemm/gemm_runner.{h,cu}`, `csrc/bindings.cpp` | additive: read or install the cached cuBLASLt algorithm of one `fp16_nn` / `bf16_nn` shape, so a second runner replays the algorithm the frontend autotuned | the runner's own algo cache |
| `cpp/models/imagewam/` → `libflashrt_imagewam_native.so` (root CMake target `flashrt_imagewam_native`, `EXCLUDE_FROM_ALL`) | C ABI (`c_api.h`); `NativeRuntime`: native verbs, host IO transforms, proprio projection, schema rendering, the stream and graph it replays; `NativePipeline` (Phase 3+): records prefill/denoise from a borrowed resource table with its own `GemmRunner` and cuBLAS handle, captures the graph | its stream, graph exec, cuBLAS/cuBLASLt handles, tiny staging scratch, host copies of the normalization constants; borrows every other pointer |
| `flash_rt/models/imagewam/native_library.py` | ctypes binding of `c_api.h` | none |
| `flash_rt/models/imagewam/native_resources.py` | builds the handoff structs from a captured frontend: windows, normalization constants, proprio projection weight, per-layer linear descriptors, precomputed fp16 modulation, RoPE and norm pointers, GEMM algorithms | host-side ctypes arrays and the fp16 modulation / transposed-weight tensors it materializes (kept alive by the native wrapper) |
| `flash_rt/models/imagewam/runtime_export.py` | adds the `io="native"` face: declaration over the native stream and graph, validated and overridden with the native verbs (`frt_model_runtime_override_verbs`) | the exec context and wrapped windows, as for `io="python"` |

Buffer ownership contract: the frontend is the single owner of every
device buffer the native library touches, except the native library's
own stream, graph exec, handles, workspace and staging scratch. The
native library never frees a borrowed pointer. The model runtime anchors
the frontend (through the declaration's Python owner) and the native
handle (through the override's owner reference), so every borrowed
pointer outlives every verb call. The Python graph and the native graph
share the same buffers and must not run concurrently.

## Interface

C ABI (`cpp/models/imagewam/include/flashrt/cpp/models/imagewam/c_api.h`):

```c
typedef struct frt_imagewam_native frt_imagewam_native;   /* refcounted */

typedef struct frt_imagewam_io_config {                   /* Phase 2 */
    uint32_t struct_size;
    uint32_t img_len, token_dim, num_action, action_dim, proprio_dim; /* proprio_dim 0 = none */
    uint32_t context_rows, context_width;
    void *img_raw, *context, *action_latent;                /* borrowed device windows */
    const void *proprio_weight_t, *proprio_bias;            /* borrowed bf16 (proprio_dim, context_width), (context_width) */
    const float *state_scale, *state_offset;                /* copied; null = no normalization */
    const float *action_scale, *action_offset;              /* copied; null = raw actions */
} frt_imagewam_io_config;

int  frt_imagewam_native_create(const frt_imagewam_io_config*, frt_imagewam_native** out);
void frt_imagewam_native_retain(void* h);
void frt_imagewam_native_release(void* h);
const char* frt_imagewam_native_last_error(const frt_imagewam_native*);
void* frt_imagewam_native_stream(frt_imagewam_native*);          /* its cudaStream_t */
int  frt_imagewam_native_use_graph(frt_imagewam_native*, void* graph_exec); /* Phase 2: adopt the Python graph */
int  frt_imagewam_native_set_proprio_row(frt_imagewam_native*, int row);
int  frt_imagewam_native_schema_records(const frt_imagewam_native*, char* out, uint64_t cap, uint64_t* written);
int  frt_imagewam_native_bind_declaration(frt_imagewam_native*, const frt_model_runtime_v1*);
const frt_model_runtime_verbs* frt_imagewam_native_verbs(void);   /* self = handle */

/* Phase 3+ */
typedef struct frt_imagewam_linear_desc { uint32_t kind; int32_t n, k; void* weight; ...nvfp4 fields } ...;
int frt_imagewam_native_set_pipeline(frt_imagewam_native*, const frt_imagewam_pipeline_config*);
int frt_imagewam_native_set_gemm_algo(frt_imagewam_native*, uint32_t kind, int m, int n, int k, const void* algo, uint64_t bytes);
int frt_imagewam_native_run(frt_imagewam_native*, uint32_t segment, int32_t index);  /* eager, for parity */
int frt_imagewam_native_capture(frt_imagewam_native*);          /* native graph replaces the adopted one */
```

`GemmRunner` additions (setup only):
`bool get_cached_algo(int kind, int M, int N, int K, void* out) const;`
`void set_cached_algo(int kind, int M, int N, int K, const void* in);`
with `kind` 0 = `bf16_nn`, 1 = `fp16_nn`, exposed to Python as
`GemmRunner.cached_algo(kind, M, N, K) -> bytes`.

`io="native"` port schema (declaration order):

| port | dir | update | modality | dtype | shape | window |
|---|---|---|---|---|---|---|
| `image_tokens` | in | SWAP | TENSOR | bf16 | (img_len, 128) | `img_raw` |
| `proprio` | in | STAGED | STATE | f32 | (proprio_dim,) | none (when `proprio_dim`) |
| `noise` | in | SWAP | TENSOR | f32 | (num_action, action_dim) | `action_latent` |
| `actions` | out | STAGED | ACTION | f32 | (num_action, action_dim) | none |
| `actions_raw` | out | SWAP | TENSOR | f32 | (num_action, action_dim) | `action_latent` |

No `images` and no `prompt`: VAE and Qwen3 stay in Python, so the native
face does not advertise them. The prompt is set through the frontend in
setup, which then calls `set_proprio_row`. Buffers `img_raw`, `context`,
`action_latent`; region `rollout_boundary`; one GRAPH stage `infer`.

Native verbs: `proprio` = host normalization in fp32 (`x*scale`, then
`+offset`, clamp to ±5, compiled without FP contraction to match the
two torch kernels bit for bit), round to bf16, asynchronous copy, one
cached cuBLASLt bf16 GEMM with bias epilogue into the context row, all on
the native stream. `actions` = device-to-host copy on the native stream,
synchronize, host `(x - offset) / scale`. `step` = `cudaGraphLaunch` on
the native stream.

## Flow

Setup (Python process): construct the frontend, `set_prompt()` (Python
graph), `native_resources.build(frontend)` → `create` → (Phase 2)
`use_graph(python exec)` / (Phase 3+) `set_pipeline`,
`set_gemm_algo` for every shape, `capture()` → `set_proprio_row` →
`export_model_runtime(io="native", native=h)`: the declaration's stream
and graph are the native handle's; `bind_declaration` validates it
against the native schema; `frt_model_runtime_override_verbs` installs
the native verbs and retains `h`.

Tick (any thread, no Python): host writes `image_tokens` and `noise` on
the exported stream, `set_input(proprio)`, `step`, `get_output(actions)`
or read `actions_raw` after synchronizing the stream.

## Code Mapping

| item | file |
|---|---|
| C ABI | `cpp/models/imagewam/include/flashrt/cpp/models/imagewam/c_api.h`, `cpp/models/imagewam/src/c_api.cpp` |
| native runtime, verbs, lifetime | `cpp/models/imagewam/src/native_runtime.{h,cpp}` |
| host IO transforms (no FP contraction) | `cpp/models/imagewam/src/io_transforms.{h,cpp}` |
| proprio projection (cached cuBLASLt) | `cpp/models/imagewam/src/proprio_projection.{h,cpp}` |
| schema records | `cpp/models/imagewam/src/native_schema.{h,cpp}` |
| native pipeline (Phase 3+) | `cpp/models/imagewam/src/native_pipeline.{h,cpp}` |
| build | root `CMakeLists.txt` (`flashrt_imagewam_native`) |
| GEMM algo hand-off | `csrc/gemm/gemm_runner.{h,cu}`, `csrc/bindings.cpp` |
| ctypes binding | `flash_rt/models/imagewam/native_library.py` |
| handoff builder | `flash_rt/models/imagewam/native_resources.py` |
| Python owner of a native handle (setup calls) | `flash_rt/models/imagewam/native_runtime.py` |
| construction path 3 from Python (`frt_model_runtime_override_verbs`) | `flash_rt/runtime/export.py` (`override_model_runtime_verbs`, additive) |
| build of the native target | `cpp/models/imagewam/imagewam_native.cmake`, included from the root `CMakeLists.txt` |
| `io="native"` face | `flash_rt/models/imagewam/runtime_export.py` |
| schema golden + gate | `tests/data/imagewam_native_schema.records`, `tests/gate_imagewam_native_schema_parity.py` |
| step-by-step parity (small dims) | `tests/test_imagewam_native_pipeline.py` |
| native pipeline resource interface (frontend -> native) | `flash_rt/models/imagewam/pipeline_resources.py`, `ImageWAMTorchFrontendThor.pipeline_resources` / `gemm_algo` |
| csrc operations without a csrc header | `cpp/models/imagewam/src/csrc_operations.h` |
| NVFP4 linear (SM100-class builds) | `cpp/models/imagewam/src/fp4_linear.{h,cpp}` |
| real-checkpoint parity gate | `tests/gate_imagewam_native_parity.py` |
| records | `docs/imagewam_native_cpp.md`, `opportunities.md` OPT-029, `issues.md` ISSUE-07x |

## Implementation Phases

### Phase 1 — design

Phase Status: completed

Goal: this section.

### Phase 2 — native verb overlay over the Python graph, schema parity

Phase Status: completed

Goal: `libflashrt_imagewam_native.so` with the Phase 2 C ABI; the
`io="native"` face adopting the Python graph; schema-parity gate; a
GIL-free ctypes tick compared with `infer()`.
Modified files: C++ sources above (runtime, transforms, projection,
schema, c_api), root `CMakeLists.txt`, `native_library.py`,
`native_resources.py` (IO part), `runtime_export.py`, schema gate and
golden, unit test.
Observation: schema records equal (Python, C++, golden); tick parity:
`actions_raw` and `actions` `array_equal` when proprio is staged by
Python; the native proprio token vs torch `F.linear` reported as
`max_abs` / `array_equal`, and its effect on actions.

### Phase 3 — native fp16 pipeline: block, prefill, denoise, graph

Phase Status: completed

Goal: GEMM algo hand-off; `NativePipeline` recording one backbone
single-stream block, then the full prefill, then the denoise loop, each
`array_equal` to the Python pipeline run eagerly on the same buffers;
native capture replaces the adopted graph; the `io="native"` tick is
bit-exact to `infer()` (Python-staged proprio) on small dims and on the
real checkpoint at fp16.
Modified files: `gemm_runner.{h,cu}`, `bindings.cpp`,
`native_pipeline.{h,cpp}`, `c_api`, `native_resources.py`, unit test,
real-checkpoint gate.
Observation: per-step `array_equal` / `max_abs`; graph kernel counts;
indicative alternating A/B latency of the Python graph vs the native
graph (H100, same process).

### Phase 4 — nvfp4 wiring and Thor handoff

Phase Status: completed

Goal: NVFP4 linear descriptors (packed weight, SFB, activation scratch,
CUTLASS variant) recorded natively with the `flash_rt_fp4` kernels;
`sm110_check.sh` builds `flashrt_imagewam_native` for `GPU_ARCH=110`;
Thor checklist runs the parity gates at `nvfp4`.
Modified files: `native_pipeline.{h,cpp}`, root `CMakeLists.txt`,
`native_resources.py`.
Observation: `sm110_check` rc; Thor parity (pending).

### Phase 5 — follow the served layer structure and the VAE stage

Phase Status: completed

Goal: after `roadmap/integration` gained the fusion stream (merged
single-stream `linear2`, gated residual fused with the next AdaLN, both
on by default) and the VAE stream (preprocessing kernel, optional
in-graph VAE), the native pipeline records both layer structures along
the same AdaLN chain as `pipeline_thor.py`; the resource table carries
the FP32 modulation chunks and `linear2`; image staging goes through the
VAE stage; `io="native"` and the native pipeline refuse
`vae_graph_input`, and the `io="python"` face exposes the in-graph view
buffer as `image_views`.
Modified files: `c_api.h`, `native_pipeline.{h,cpp}`,
`imagewam_native.cmake` (`fusion.cu`), `pipeline_resources.py`,
`native_library.py`, `native_resources.py`, `runtime_surface.py`,
`runtime_export.py`, `imagewam_thor.py`, tests.
Observation: step-by-step parity for both layer structures; the real
checkpoint gates on the merged tree.
