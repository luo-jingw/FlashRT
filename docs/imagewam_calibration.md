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

1. Frames (`flash_rt/models/imagewam/libero_frames.py`): `n` split
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
3. Recording (`activation_recorder.py`): every fp16 GEMM input (182
   sites at the real dims; `txt_in`/`img_in` are BF16 and never
   quantized) gives absmax, the 99 / 99.9 / 99.99th percentiles of |x|,
   and per-input-channel absmax. Calls within one sample are reduced by
   max, so ActionDiT sites cover all 10 denoise steps
   (`docs/calibration.md` §4.2).
4. Reduction (`calibration_file.build_calibration`): across samples with
   `accumulate_amax(percentile=99.9)` (linear interpolation along the
   sample axis). The static FP8 scale is `amax / 448` in float32, floored
   at 1e-12, the same arithmetic as `compute_scale_kernel`.

On the H100 dev machine the N = 64 build records at 2.6-3.0 s per sample.

## File

One safetensors file. Tensors `<site>.channel_amax` (K floats) and
`<site>.sample_absmax` (N floats); JSON metadata `imagewam_calibration`
with the format name and version, checkpoint identity, dims, percentile,
frame list, noise description, and per site `amax`, `rows` and the |x|
percentiles. Site names are the frontend's weight keys joined with dots,
e.g. `backbone.single.3.linear1.weight`.

Identity: checkpoint hash (`_checkpoint_hash`: SHA-256 of the first 64KB
plus the file size) and size, and every dims entry that changes GEMM
shapes or activations (`calibration_file.IDENTITY_DIM_KEYS`). The
frontend raises `ValueError` on any mismatch.

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
| backbone single `mlp_down` | 33.2 (14.8-89.7) | 0.13 |
| ActionDiT single `linear1` | 18.3 (13.2-22.9) | 0.92 |
| ActionDiT single `mlp_down` | 13.1 (4.8-135) | 0.56 |

The placeholder's `N(0, 0.1)` noise has an absmax near 0.5 at these
sizes, so its scales clip real activations by 1-4 orders of magnitude.
A second 64-frame set with a different suite mix gave per-site amax
within a few percent (`txt_mlp2` max 7051 vs 7067, `linear1` median
36.47 vs 36.42).

## Measured accuracy (H100, real checkpoint)

`fp8_static` against `fp16`, 20 held-out `libero_spatial` frames
(`imagewam_precision_fidelity.py`, official-sampler noise):

| calibration | backbone_hidden | action_hidden | action_latent | actions | MAE / fp16 MAE |
|---|---|---|---|---|---:|
| placeholder `N(0, 0.1)` | 0.456 (0.425-) | 0.687 | 0.872 (0.671-) | 0.901 (0.735-) | 1.697 |
| real file, N = 64 | 0.99994 (0.99984-) | 0.99995 | 0.99997 (0.99995-) | 0.99997 (0.99989-) | 1.000 |

Cells are median (min-) cosine. Against official ImageWAM
(`imagewam_e2e_official_compare.py`, `N_TASKS=10 FRAMES=0,60 SEEDS=0,1`):

| path | fr_vs_off median | min | mean MAE vs GT |
|---|---:|---:|---:|
| `fp16` | 0.99840 | 0.99567 | 0.18359 |
| `fp8_static`, placeholder | 0.87576 | 0.66887 | 0.30207 |
| `fp8_static`, real file | 0.99844 | 0.99559 | 0.18372 |
