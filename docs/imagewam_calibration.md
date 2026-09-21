# ImageWAM activation calibration

ImageWAM's static-scale FP8 precisions (`fp8_static`,
`fp8_static_cutlass`) quantize every GEMM input with one frozen
per-tensor scale. This page describes how those scales are derived from
real data, the file that carries them, and how the Thor frontend uses
it. The same file carries the per-channel statistics the NVFP4 AWQ path
uses ([`imagewam_nvfp4_awq.md`](imagewam_nvfp4_awq.md)).

## Build

```bash
python benchmarks/imagewam_build_calibration.py \
    --out <dir>/imagewam_libero_calib_n64_v1.safetensors --n 64
```

Environment: `CKPT_PATH` (with `dataset_stats.json` beside it),
`FLUX2_AE_MODEL_PATH`, `FLUX2_SRC`, `QWEN3_MODEL_SPEC`, `DATA_ROOT`
(LIBERO-fastwam, LeRobot v2.1).

`--text-trim` records with a trimming frontend instead. The file's identity
carries the switch it was recorded with, and a frontend refuses a file whose
identity differs from its own, so an untrimmed build serves only untrimmed
configurations and a `--text-trim` build only trimming ones. The served
`default` profile trims, so it loads a `--text-trim` build
(`plan.md`, "Decisions pending": `text_trim` is the served default).

1. Frames (`benchmarks/_imagewam_libero_frames.py`): `n` split
   evenly over `libero_object`, `libero_goal`, `libero_10`; each share
   stratified by episode x frame position with the house sampler
   (`flash_rt.core.calibration.stratified_sample_indices`). Every episode
   of the evaluation set (the end-to-end harness's `libero_spatial`
   frames) is excluded, and the builder refuses any overlap. The two
   camera views get the official eval preprocessing (PIL bilinear
   resize + center crop to 224x224).
2. Forward: an `fp16` frontend with the real checkpoint, VAE, Qwen3
   prompt encoding, proprio and the 10-step shift schedule. Per frame:
   `set_prompt(task)`, `stage_inputs(obs, noise)` with official-sampler
   noise `N(0,1)` (`torch.Generator("cpu").manual_seed(i)`, bf16-rounded),
   then `run_eager()` (prefill + denoise without a CUDA Graph,
   bit-exact to graph replay).
3. Recording (`activation_recorder.py`): every fp16 GEMM input (142
   sites at the real dims with the merged single-stream `linear1` and
   `linear2`; `txt_in`/`img_in` are BF16 and never quantized) gives absmax, the 99 / 99.9 / 99.99th percentiles of |x|,
   and per-input-channel absmax. Calls within one sample are reduced by
   max, so ActionDiT sites cover all 10 denoise steps
   (`docs/calibration.md` §4.2).
4. Reduction (`calibration_file.build_calibration`): across samples with
   `accumulate_amax(percentile=99.9)` (linear interpolation along the
   sample axis). The static FP8 scale is `amax / 448` in float32, floored
   at 1e-12, the same arithmetic as `compute_scale_kernel`.

The site set follows the frontend's GEMM structure (`merge_qkv_mlp`,
`merge_linear2` are part of the file's identity), so a file must be built
with the same structure it is served with.

## File

One safetensors file. Tensors `<site>.channel_amax` (K floats) and
`<site>.sample_absmax` (N floats); JSON metadata `imagewam_calibration`
with the format name and version, checkpoint identity, dims, percentile,
frame list, noise description, and per site `amax`, `rows` and the |x|
percentiles. Site names are the frontend's weight keys joined with dots,
e.g. `backbone.single.3.linear1.weight`.

Identity: checkpoint hash (`_checkpoint_hash`: SHA-256 of the first 64KB
plus the file size) and size, every dims entry that changes GEMM
shapes or activations (`calibration_file.IDENTITY_DIM_KEYS`), and
`text_trim`. The dims include the workload's camera geometry
(`num_views`, `image_h`, `image_w`): two workloads can share `ref_h`,
`ref_w`, `x0` and `a0` (2 views of 224x224 and 4 views of 224x112 both
give the 14 x 28 grid) while the VAE is fed different images, so the
layout alone does not identify the workload. The frontend raises
`ValueError` on any mismatch, naming the differing keys.

A frontend gets the camera geometry from its workload
(`load_imagewam` / `ImageWAMTorchFrontendThor.from_config`). One built by
hand with `dims_override` has none unless the override carries the three
keys (`libero_dims.LIBERO_REAL_DIMS` does), and it cannot validate a file:
the error shows each missing key against the file's value.

Format version 3 is the only one read. Version 1 (no `text_trim`) and
version 2 (no camera geometry) files predate the workload identity;
`load_calibration` refuses them, naming the version, and they are
re-recorded with `benchmarks/imagewam_build_calibration.py` (the command
above; add `--text-trim` for the trimming build), not migrated.

## Frontend

`ImageWAMTorchFrontendThor(precision="fp8_static" | "fp8_static_cutlass",
ckpt_path=..., calibration_path=...)` loads and validates the file before
loading the checkpoint. `set_prompt()` -> `_calibrate_fp8()` sets each
`StaticFp8Linear` scale with `set_activation_scale(site.fp8_act_scale())`
before graph capture; a missing site raises. Without `calibration_path`
the scales come from `N(0, 0.1)` noise and a warning is logged.

`benchmarks/imagewam_e2e_official_compare.py` takes the file through
`CALIBRATION=`; `benchmarks/imagewam_precision_fidelity.py` compares any
precision to `fp16` on held-out frames.

## Recorded statistics (real checkpoint, N = 64)

Per-site amax, median (min-max) over the layers of each group:

| site group | amax | p99.99 / amax |
|---|---|---:|
| backbone double `txt_qkv` | 11.6 (7.7-46.2) | 0.39 |
| backbone double `txt_mlp2` | 2679 (2191-7067) | 0.018 |
| backbone double `img_qkv` | 14.8 (6.8-24.4) | 0.48 |
| backbone double `img_mlp2` | 41.6 (29.2-102) | 0.17 |
| backbone single `linear1` | 36.4 (34.8-38.5) | 0.12 |
| backbone single `linear2` | 33.2 (14.8-90.2) | 0.12 |
| ActionDiT single `linear1` | 18.3 (13.2-22.9) | 0.92 |
| ActionDiT single `linear2` | 13.1 (4.8-135) | 0.57 |

The placeholder's `N(0, 0.1)` noise has an absmax near 0.5 at these
sizes, so its scales clip real activations by 1-4 orders of magnitude.
A second 64-frame set with a different suite mix gave per-site amax
within a few percent (`txt_mlp2` max 7051 vs 7067, `linear1` median
36.47 vs 36.42).

## Measured accuracy (H100, real checkpoint)

`fp8_static` against `fp16`, 20 held-out `libero_spatial` frames
(`imagewam_precision_fidelity.py`, official-sampler noise; the real-file
row on the served structure with merged `linear2` and the fused
residual+AdaLN, the placeholder row before those merges):

| calibration | backbone_hidden | action_hidden | action_latent | actions | MAE / fp16 MAE |
|---|---|---|---|---|---:|
| placeholder `N(0, 0.1)` | 0.456 (0.425-) | 0.687 | 0.872 (0.671-) | 0.901 (0.735-) | 1.697 |
| real file, N = 64 | 0.99994 (0.99984-) | 0.99995 | 0.99997 (0.99996-) | 0.99997 (0.99994-) | 1.000 |

Cells are median (min-) cosine. Against official ImageWAM
(`imagewam_e2e_official_compare.py`, `N_TASKS=10 FRAMES=0,60 SEEDS=0,1`):

| path | fr_vs_off median | min | mean MAE vs GT |
|---|---:|---:|---:|
| `fp16` | 0.99840 | 0.99566 | 0.18359 |
| `fp8_static`, placeholder (before the merges) | 0.87576 | 0.66887 | 0.30207 |
| `fp8_static`, real file | 0.99837 | 0.99571 | 0.18370 |

The result does not depend on the calibration set's size or suite mix
(measured before the merges): N = 8 (3/3/2 frames per suite) and a second N = 64 set (31/27/6) give
the same `backbone_hidden` (0.99993-0.99994 median) and `actions`
(0.99997 median) cosines and MAE ratio (1.000-1.001).
`tests/gate_imagewam_libero.py --precision fp8_static --fp8-calibration
<file> --no-text-trim --override vae_graph=false --override vae_encoder=torch --manifest
tests/fixtures/imagewam_gate/imagewam_libero_gate_v1.manifest.json` passes
on H100 (vs official median 0.99830, min 0.99553; vs its
`fp16` reference median 0.999969; MAE 0.18373 against 0.18364).

Every recorded number on this page belongs to the untrimmed configuration.
The gate row is fixture v1, whose `fp16` reference is untrimmed, and it
loads the untrimmed N = 64 build of the Build section (a frontend refuses a
calibration file recorded with the other `text_trim`). The switches are
stated in the invocation because the gate's own defaults are the served
configuration now: fixture v2, trimming on and the native VAE encoder inside
the graph, so a bare run gates the trimmed pipeline (the two `--override`s put
the torch VAE back outside the graph, as it was in the recorded run). That gate has run at `nvfp4` only, and no `fp8_static`
number against fixture v2 is recorded. The two accuracy tables are the same
untrimmed configuration: the e2e rows were recorded before `text_trim` was
served, and `imagewam_precision_fidelity.py` runs untrimmed unless
`TEXT_TRIM=1`. All of them are H100 rows, where `use_fa4=None` resolves to
the cuBLAS chain (FA4 needs a compute-capability-11.x device), so the
backbone site's FA4 default does not enter them.
