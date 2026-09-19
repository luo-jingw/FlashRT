# ImageWAM configuration: workload, structure, precision, profiles

How the ImageWAM Thor frontend decides what to serve and how to serve it.
One resolution step turns a workload, a structure, a named profile and a few
overrides into the dims and the switches the frontend is built from. Nothing
is re-derived afterwards.

| Question | Owner | File |
|---|---|---|
| What is served (cameras, image size, text length, horizon, proprio dim, denoise loop, shift) | `ImageWAMWorkload` | `flash_rt/models/imagewam/workload.py` |
| What the checkpoint is (backbone widths, action-expert dims, horizon limit, patch stride) | `ImageWAMStructure` | `flash_rt/models/imagewam/structure.py` |
| What a precision implies (calibration, AWQ, tile autotune, alignment fallback, native runtime, merged GEMMs) | `Precision` / `PROPERTIES` | `flash_rt/models/imagewam/precision.py` |
| Which combination is legal, and what a named profile sets | `resolve_config`, `PROFILES`, rules R1-R11 / V1 | `flash_rt/models/imagewam/config_resolver.py` |
| Building the frontend from a resolved configuration | `from_config`, `load_imagewam` | `flash_rt/frontends/torch/imagewam_thor.py` |
| The served LIBERO dims as one mapping | `LIBERO_REAL_DIMS` | `flash_rt/models/imagewam/libero_dims.py` |

## Entry point

```python
from flash_rt.frontends.torch.imagewam_thor import load_imagewam
from flash_rt.models.imagewam.workload import ImageWAMWorkload

fe = load_imagewam(ckpt_path, ImageWAMWorkload.libero(), profile="default",
                   calibration_path=None, ae_model_path=..., flux2_src=...)
fe.set_prompt(context=ctx, context_mask=mask)
actions = fe.infer({"view1": v1, "view2": v2, "proprio": state})
```

`load_imagewam(ckpt_path, workload, *, structure=None, profile="default",
precision=None, calibration_path=None, ae_model_path=None, flux2_src=None,
qwen3_model_spec=None, dataset_stats_path=None, consumer="infer",
allow_placeholder_calibration=False, vae_resize="area",
precapture_text_lengths=None, **expert)`.

- `structure=None` reads `ImageWAMStructure.from_checkpoint(ckpt_path)`:
  backbone widths come from the checkpoint's tensor shapes, the action-expert
  dims and `max_action_horizon` from the `config.yaml` beside it, and a
  disagreement between the two raises. `structure` is required when
  `ckpt_path` is `None` (random-weight run).
- `consumer` is what the configuration is for: `"infer"` (the frontend's own
  `infer()`), `"abi"` (`runtime_surface()` / `export_model_runtime(io="python")`),
  `"native"` (`pipeline_resources()` / the native runtime). Which
  combinations they allow is rules R5 and R6: `text_trim` is served through
  the ABI (one graph per trimmed length) and still refused for the native
  pipeline, which describes one fixed graph.
- `precapture_text_lengths` (x0 values, valid tokens + 1 with proprio, as
  `captured_text_lengths()` reports them): the frontend captures a graph for
  each of those lengths once at construction, so the first `set_prompt` of
  one of them only switches graphs. It needs `text_trim=True`. The cache is
  bounded by `text_trim_cache_size` (expert option, default 32): the least
  recently used non-active length is dropped when the bound is reached, and a
  dropped length captures again the next time it is used.
- `**expert` carries the expert tier only (`config_resolver.EXPERT_KEYS`:
  `use_fa4`, `use_fa4_mot`, `text_trim`, `text_trim_cache_size`,
  `vae_encoder`, `vae_graph`, `nvfp4_awq`, `awq_alpha`, `awq_scope`,
  `gemm_variant_autotune`, `gemm_runner`, `merge_qkv_mlp`, `merge_linear2`).
- Every legality decision is `resolve_config`'s, and it runs before the
  frontend constructor: an illegal combination raises `ConfigError` whose
  message starts with the rule id, so nothing is allocated for a
  configuration that cannot run.

`ImageWAMTorchFrontendThor.__init__` keeps its own signature and behaviour
(`dims_override=`, the same switches), so callers that pass dims by hand are
unaffected. `from_config(resolved, *, workload=None,
precapture_text_lengths=None, **kwargs)` is the
resolved path; `frontend_kwargs_from_config(resolved, **kwargs)` is the one
place that maps a `ResolvedConfig` onto constructor arguments. A frontend
built through `from_config` records what it was built from: `workload` and
the `resolved_config` property (both `None` on the constructor path).

## Workload, and what is derived from it

`ImageWAMWorkload` holds only served quantities: `num_views`, `image_h`,
`image_w` (per view, pixels), `text_max_len`, `action_horizon`, `action_dim`,
`proprio_dim`, `num_steps`, `shift`, `num_train_timesteps`.
`ImageWAMWorkload.libero()` is `2 x 224x224`, 512 text tokens, horizon 64,
7 action dims, 8 proprio dims, 10 steps, shift 5.0.

`workload.layout(structure)` derives, and raises on inconsistent input:

| Derived | LIBERO value | Rule |
|---|---|---|
| `ref_h`, `ref_w` | 14, 28 | `image_h / patch_stride`, `num_views * image_w / patch_stride`; the views are concatenated along the width |
| `x0` | 513 | `text_max_len + 1` (the proprio row) |
| `img_len` | 392 | `ref_h * ref_w` |
| `a0` | 905 | `x0 + img_len` |
| `total` | 969 | `a0 + action_horizon` |
| `dt` | 0.1 | `1 / num_steps` |
| `vae_graph_input` | `(2, 224, 224)` | `(num_views, image_h, image_w)`, used when the VAE runs inside the graph |

Rejected: an image side that is not a multiple of the patch stride (16), an
`action_horizon` above `structure.max_action_horizon`, and any non-positive
field. `resolve_config` reports these as rule R7.

## Structure

`ImageWAMStructure` has one place per source:

- `from_checkpoint(ckpt_path)`: backbone `hidden`, `HD`, `NH`, `mlp_hidden`,
  `joint_attention_dim` and the layer counts are read from the `mot`
  state_dict's tensor shapes and key counts (`config.yaml` does not carry
  them); `action_hidden_dim`, `action_attn_width`, `action_mlp_hidden`, the
  action-expert layer counts and `max_action_horizon` come from
  `model.action_dit_config` of the `config.yaml` beside `model.pt`,
  cross-checked against the tensors where a tensor also carries the value.
  `model.pt` is opened with `mmap=True`, so the weights are not materialised.
- `libero()`: the constant table of the real `ImageWAM-FLUX.2-4B-LIBERO`
  release (3072/128/24/9216/7680, 5 double + 20 single for both the backbone
  and the action expert, 1024/3072/4096, `max_action_horizon=64`).
- `toy()`: the small random-weight dims of `imagewam_thor._DEFAULT_DIMS`.
- `patch_stride = 16`: the FLUX.2 autoencoder constant (8x conv downsampling
  and a 2x2 patch merge), not a checkpoint value.

`LIBERO_REAL_DIMS` is `dict(resolve_config(ImageWAMWorkload.libero(),
ImageWAMStructure.libero()).dims)`: benchmarks, gates and tests read the
served dims from that one mapping instead of re-typing them.

## Precision

`Precision` is a `str` enum whose values are the precision strings the
frontend and every log use (`fp16`, `fp16_cutlass`, `fp8`, `nvfp4`,
`fp8_static`, `fp8_static_cutlass`, `e0m3_hadamard`, `nvfp4_sim`), so
`Precision("nvfp4") == "nvfp4"` and f-strings keep producing the plain
string. `PROPERTIES` holds one row per member:

| Property | Meaning |
|---|---|
| `alignment` | both `n` and `k` of a `(k, n)` linear must be a multiple of it for the tier's native GEMM; 1 = no constraint |
| `needs_calibration` | static-FP8 tiers: a calibration file carries the activation scales |
| `supports_awq` | the NVFP4 family: takes folded per-channel input scales |
| `supports_tile_autotune` | `gemm_variant_autotune` applies (switchable CUTLASS tile) |
| `supports_native_runtime` | every linear it builds is one `pipeline_resources.linear_resource` describes, and it uses the merged linear1 |
| `merge_qkv_mlp`, `fused_swiglu_mlp` | the single-stream `linear1` (qkv + mlp gate/up) merge, and the tier that keeps its own fused SwiGLU MLP instead |
| `fp16_nn_backbone_gemm` | the backbone GEMMs run on the `fp16_nn` path, so a new text length autotunes those shapes |

`Precision.alignment_fallback(n, k)` is the fallback the frontend applies:
`True` means the plain `Fp16Linear` (cuBLASLt) instead of the tier's native
linear. In the real model only `action_encoder` (K=7) and `head.linear`
(N=7) fall back.

`precision.py` is a leaf: standard library only, no torch, no compiled
extension, so a resolver or a test can reason about a precision without a
GPU stack.

## Profiles and rules

`PROFILES` maps a name to a set of tier-2 options:

| Profile | Contents |
|---|---|
| `default` | the frontend's own defaults: `nvfp4`, no text trim, FA4 backbone as the frontend resolves it (`FLASHRT_THOR_FA4`, off unless set), no FA4 mot, torch VAE encoder outside the graph, no AWQ |
| `fast` | `text_trim` + FA4 backbone + FA4 mot + native VAE encoder inside the graph. PROVISIONAL: the contents and whether any switch becomes the default are an owner decision (measured 106.1 ms against 203.3 ms for `default`, `nvfp4`, `libero_spatial`); `text_trim` is refused with the `abi`/`native` consumers (rule R5) |

`resolve_config` raises `ConfigError("<rule id>: <combination>")`:

| Rule | Combination refused |
|---|---|
| R1 | static-FP8 precision without a calibration file (the constructor path keeps its logged N(0, 0.1) placeholder; `allow_placeholder_calibration=True` accepts it here) |
| R2 | `gemm_variant_autotune` with a precision that has no switchable tile |
| R3 | a real VAE encoder, or the VAE inside the graph, without `ae_model_path` / `flux2_src` |
| R4 | `nvfp4_awq` with a non-AWQ precision, or without a calibration file |
| R5 | `text_trim` with the `abi` or `native` consumer (one graph per prompt length against a single-graph surface; ISSUE-080 condition 5) |
| R6 | native consumer with something the native pipeline does not carry: AWQ, a precision it cannot describe, FA4, or the VAE stage |
| R7 | workload layout inconsistent with the structure |
| R8 | a calibration file whose identity differs from the resolved dims or `text_trim` |
| R9 | merge flags inconsistent with the precision |
| R10 | a calibration file nothing in the configuration would use |
| R11 | a structure invariant the frontend asserts (`HD == 128`, `action_attn_width == hidden`) |
| V1 | a value outside its domain (unknown profile, precision, `vae_encoder`, consumer, expert key, or a non-bool switch) |

## The resolved configuration in logs and identities

`format_effective_config(options, *, use_fa4=None, use_fa4_mot=None,
fa4_fallback_reason=None)` produces the one line every log and matrix row
carries:

```
effective_config precision=nvfp4 text_trim=False vae_encoder=torch vae_graph=False use_fa4=auto use_fa4_mot=False fa4_fallback_reason=None calibration=None awq=False
```

The key order is `EFFECTIVE_CONFIG_FIELDS`. `use_fa4` is resolved at
construction (it depends on the machine) and prints `auto` when it is not yet
known; the compare script and the matrix print the frontend's own resolved
configuration (`frontend.resolved_config.options`) with the runtime-resolved
`use_fa4`, `use_fa4_mot` and `fa4_fallback_reason`, which is what
`scripts/imagewam_thor_matrix.sh:parse_log` reads.

The runtime identity (`ImageWAMRuntimeSurface.setup_identity`, carried into
the `frt_model_runtime_v1` export) lists the pipeline, the precision, the
attention choices, the calibration digest, the VAE setup, then
`workload.<field>` pairs (`runtime_surface.workload_identity`, fields in
`WORKLOAD_IDENTITY_FIELDS`) and finally `dims.<key>` for every resolved dims
entry. The workload entries are additive: they name the workload the graph
was built for, beside the dims it produced. A frontend built by the
constructor with hand-passed dims has no workload to name and reports only the
`dims.<key>` entries.

The calibration file's identity (`calibration_file.IDENTITY_DIM_KEYS`) covers
every dims entry that changes GEMM shapes or activation distributions,
including the ones a workload changes (`x0`, `a0`, `num_action`, `action_dim`,
`num_denoise_steps`, `shift`, `proprio_dim`, `ref_h`, `ref_w`), plus
`text_trim` and the checkpoint hash; a file recorded for another workload is
refused with the differing field named.
