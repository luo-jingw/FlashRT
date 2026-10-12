# ImageWAM on Jetson AGX Thor: module x operator profile

GPU time per `infer()` of the served ImageWAM LIBERO configuration on
Thor, split by model module and operator class, for `fp16`, `fp8` and
`nvfp4`, with the bottleneck of each part where the measurement supports
one, and a same-cell comparison to the RTX 5090 profile
(`docs/imagewam_rtx5090_profile.md`).

## Configuration

- Workload: `ImageWAMWorkload.libero()` — 2 views 224x224, action horizon
  64, `action_dim` 7, proprio 8, text padded to 512, 10 denoise steps.
- Weights: real LIBERO checkpoint (`checkpoints/imagewam_release/libero/flux2_klein_4b/model.pt`),
  real FLUX.2 AE (`checkpoints/flux2/FLUX.2-dev/ae.safetensors`), Qwen3-4B
  (`checkpoints/Qwen3-4B`); built with `load_imagewam` (default profile).
  Served options as resolved on this machine: `text_trim=True` (prompt
  "pick up the red block and place it in the tray" → `x0` = 24, `a0` =
  416, `num_action` = 64, `total` = 480), VAE inside the CUDA graph
  (`vae_graph_input=(2,224,224)`), FA4 at both attention sites (`use_fa4`
  in the profile is `None` / auto; construction enables FA4 —
  `flashrt_fa4` / `FlashAttentionForwardSm100` is in the kernel dump).
  `vec_rms_norm` on (frontend `setdefault`; the QK RMSNorm kernel is
  `rms_norm_fp16_vec`, not the 256-thread `rms_norm_kernel`).
  `fuse_qkv_norm_rope` on (frontend `setdefault`; the Q/K/V split,
  per-head Q/K RMSNorm and RoPE run as one `qkv_split_norm_rope_fp16`
  kernel per layer site).
- GEMM M: backbone text stream 24, image stream 392, single-stream 416;
  ActionDiT 64.
- Precisions: `fp16`, `fp8` (dynamic-scale `Fp8Linear`), `nvfp4`. ImageWAM
  has no `bf16` precision. Each precision ran in its own process.
- Machine: Jetson AGX Thor Developer Kit, SM 11.0, torch 2.9.1+cu130,
  CUDA 13.0, FlashRT `a21507e`. Power mode MAXN (`nvpmodel -q`). GPU
  clocks pinned (read-only): GPC 1.575 GHz, NVD 1.692 GHz; the EMC
  frequency is not readable without root (nvpmodel EMC cap 4266 MHz). No
  `nvpmodel -m` / `jetson_clocks` / sysfs write. GPU exclusive during the
  timed runs. Environment: `FlashRT/env.sh` + in-tree `.venv`.

## Method

Experiment grid (one variable): precision ∈ {fp16, fp8, nvfp4}. Everything
else fixed as above.

- Wall time: `infer()` CUDA-event P50 (warmup 20, 100 iterations);
  prefill and denoise loop captured as separate CUDA graphs with the
  arguments `run_eager()` uses. Script:
  `benchmarks/imagewam_module_operator_profile.py`.
- Attribution: `run_eager()` under `torch.profiler`, every module function
  of `pipeline_thor.py` (`_double_stream_layer`, `_single_stream_layer[_kv_only]`,
  `_action_double_layer`, `_action_single_layer`, `imagewam_denoise_step`,
  `imagewam_prefill`, `imagewam_denoise_loop`), the attention backend's
  `run` and the VAE stage's `run` wrapped in a profiler range with a device
  sync at entry and exit. Operator class by kernel name; kernels inside
  `attn.run` are "attention". The fused `qkv_split_norm_rope_fp16` kernel
  counts as "norm", so the Q/K/V split copies and RoPE it absorbs are in
  "norm", not "elementwise".
- OPS regex (Thor-only fixes vs the 5090 dump, same script): `preprocess`
  → elementwise (`imagewam_vae_preprocess_kernel` was "other");
  `quant/cast` matched before attention, and `flash` was narrowed to
  `flash_fwd|FlashAttention|fa4cute|cuteFlash`, so
  `flash_rt::fp4::kernel_quantize_fp4_sfa_vec` is quant/cast and
  not attention. `absmax_kernel` stays "other", as on 5090. Unattributed
  kernel time: 0.000 ms on all three cells.
- CUPTI warmup: on Thor the first `torch.profiler` session of a process
  initializes CUPTI and drops about half of the kernel records (symptoms:
  non-integer per-infer launch counts, counts below the structural floor,
  eager total far below `infer()` P50). The script runs a throwaway
  profiler session before the attributed one and prints the kernel count
  of each attributed iteration; unequal counts invalidate the grid.
- Cross-check: kernels per iteration were equal in every cell. Eager
  kernel total, graph-replay kernel total and `infer()` P50 (ms):

  | precision | kernels per iteration | eager | `infer()` P50 | graph-replay | eager − P50 | graph − P50 |
  |---|---|---:|---:|---:|---:|---:|
  | fp16 | [2834, 2834, 2834] (equal) | 131.28 | 139.17 | 141.84 | −5.7% | +1.9% |
  | fp8 | [5304, 5304, 5304] (equal) | 104.97 | 111.86 | 113.04 | −6.2% | +1.1% |
  | nvfp4 | [3259, 3259, 3259] (equal) | 80.53 | 80.22 | 81.75 | +0.4% | +1.9% |

  Launch counts are integers, and 1670 below the `fuse_qkv_norm_rope`-off
  counts (4504 / 6974 / 4929), the 1670 launches the fused kernel removes.
  `qkv_split_norm_rope_fp16_kernel` is the only Q/K norm and RoPE kernel
  in all three dumps (280 launches, 3.64 ms per `infer()`);
  `rms_norm_fp16_vec` and `rope_apply_fp16_perhead` do not appear.
- No GEMM-peak or DRAM-ceiling measurement on this Thor in this run.
  Achieved TFLOPS and weight GB/s below reuse the 5090 doc's layer-shape
  inventory. No compute-bound vs bandwidth-bound call is made from those
  rates.
- Scripts: `benchmarks/imagewam_module_operator_profile.py`,
  `benchmarks/imagewam_module_operator_share.py`. Raw JSON is local
  (`logs/1011_thor_profile_fused_default/`, not in the repository).

## Wall time (ms)

| precision | `infer()` P50 | prefill graph | denoise-loop graph | `infer()` P50, `fuse_qkv_norm_rope` off |
|---|---:|---:|---:|---:|
| fp16 | 139.17 | 46.63 | 83.46 | 149.38 |
| fp8 | 111.86 | 38.31 | 64.57 | 118.81 |
| nvfp4 | 80.22 | 28.23 | 43.10 | 87.34 |

Remainder (`infer()` − prefill − denoise) is 9.08 / 8.98 / 8.89 ms. The
last column is the same configuration at `74b5efe`, where the flag was
off (separate processes). The flag saves 7.1–10.2 ms per `infer()`. The
within-process A/B, 7.3–8.3 ms over three processes, is in
opportunities.md OPT-032 candidate 1. `vae_stage_p50` from the script is
near zero because the VAE runs inside the full infer graph, not in
`stage_images` alone.

## GPU ms per `infer()` by module and operator (launches)

Attention = FA4 (`FlashAttentionForwardSm100`) + the memcpy that
`attn.run` issues around it. "norm" includes the fused
`qkv_split_norm_rope_fp16` kernel. ActionDiT rows are summed over 10
denoise steps. Totals are the eager kernel total (input staging, 0.01 ms,
not shown).

### fp16

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.15 (5) | - | 0.33 (1) | 3.93 (25) | 2.36 (68) | 1.94 (29) | 8.70 |
| backbone double (5 blocks) | 11.44 (45) | - | 0.23 (10) | - | 0.80 (32) | 0.68 (14) | 13.16 |
| backbone single (20 blocks) + entry | 26.48 (43) | - | 0.95 (40) | - | 2.25 (40) | 3.41 (40) | 33.09 |
| ActionDiT double (5 blocks × 10 steps) | 11.22 (250) | - | 1.68 (100) | - | 0.97 (160) | 0.46 (70) | 14.33 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 50.17 (630) | 0.01 (10) | 6.49 (400) | - | 2.67 (400) | 2.65 (420) | 61.99 |
| **total** | **99.45** | **0.01** | **9.69** | **3.93** | **9.05** | **9.14** | **131.28** |

### fp8

| module | GEMM | quant/cast | absmax | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.14 (5) | - | - | 0.33 (1) | 3.95 (25) | 2.35 (68) | 1.91 (29) | 8.69 |
| backbone double (5 blocks) | 9.40 (45) | 0.43 (40) | 0.33 (40) | 0.23 (10) | - | 0.80 (32) | 0.68 (54) | 11.90 |
| backbone single (20 blocks) + entry | 16.64 (43) | 1.34 (40) | 0.68 (40) | 0.99 (40) | - | 2.32 (40) | 3.38 (80) | 25.37 |
| ActionDiT double (5 blocks × 10 steps) | 6.21 (200) | 0.53 (200) | 0.96 (200) | 1.68 (100) | - | 0.97 (160) | 0.67 (270) | 11.10 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 31.91 (430) | 1.52 (410) | 2.19 (400) | 6.41 (400) | - | 2.64 (400) | 3.06 (820) | 47.89 |
| **total** | **64.30** | **3.82** | **4.17** | **9.64** | **3.95** | **9.09** | **9.71** | **104.97** |

"absmax" is `absmax_kernel`. The memsets around it (0.28 ms) are not in
the cells but are in the row totals.

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.15 (5) | - | 0.33 (1) | 3.93 (25) | 2.36 (68) | 1.96 (29) | 8.73 |
| backbone double (5 blocks) | 6.11 (40) | 0.51 (40) | 0.23 (10) | - | 0.79 (32) | 0.59 (14) | 8.24 |
| backbone single (20 blocks) + entry | 12.20 (43) | 1.36 (40) | 0.97 (40) | - | 2.26 (40) | 3.31 (40) | 20.10 |
| ActionDiT double (5 blocks × 10 steps) | 6.01 (200) | 0.82 (200) | 1.72 (100) | - | 0.92 (160) | 0.46 (70) | 9.93 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 19.44 (430) | 2.25 (410) | 6.54 (400) | - | 2.61 (400) | 2.68 (420) | 33.52 |
| **total** | **43.91** | **4.95** | **9.80** | **3.93** | **8.93** | **8.99** | **80.53** |

"quant/cast" in nvfp4 is `kernel_quantize_fp4_sfa_vec` plus the ActionDiT
`cast_fp32_fp16` on the Euler path.

## Share of eager kernel total by module and operator (%)

Same runs. Each cell is the cell's eager GPU time divided by that
precision's eager kernel total (fp16 131.28, fp8 104.97, nvfp4 80.53 ms).

### fp16

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | 0.3 | 3.0 | 1.8 | 1.5 | 6.6 |
| backbone double (5 blocks) | 8.7 | - | 0.2 | - | 0.6 | 0.5 | 10.0 |
| backbone single (20 blocks) + entry | 20.2 | - | 0.7 | - | 1.7 | 2.6 | 25.2 |
| ActionDiT double (5 blocks × 10 steps) | 8.5 | - | 1.3 | - | 0.7 | 0.4 | 10.9 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 38.2 | 0.0 | 4.9 | - | 2.0 | 2.0 | 47.2 |
| **total** | **75.8** | **0.0** | **7.4** | **3.0** | **6.9** | **7.0** | **100** |

### fp8

| module | GEMM | quant/cast | absmax | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | - | 0.3 | 3.8 | 2.2 | 1.8 | 8.3 |
| backbone double (5 blocks) | 9.0 | 0.4 | 0.3 | 0.2 | - | 0.8 | 0.6 | 11.3 |
| backbone single (20 blocks) + entry | 15.9 | 1.3 | 0.6 | 0.9 | - | 2.2 | 3.2 | 24.2 |
| ActionDiT double (5 blocks × 10 steps) | 5.9 | 0.5 | 0.9 | 1.6 | - | 0.9 | 0.6 | 10.6 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 30.4 | 1.4 | 2.1 | 6.1 | - | 2.5 | 2.9 | 45.6 |
| **total** | **61.3** | **3.6** | **4.0** | **9.2** | **3.8** | **8.7** | **9.3** | **100** |

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.2 | - | 0.4 | 4.9 | 2.9 | 2.4 | 10.8 |
| backbone double (5 blocks) | 7.6 | 0.6 | 0.3 | - | 1.0 | 0.7 | 10.2 |
| backbone single (20 blocks) + entry | 15.1 | 1.7 | 1.2 | - | 2.8 | 4.1 | 25.0 |
| ActionDiT double (5 blocks × 10 steps) | 7.5 | 1.0 | 2.1 | - | 1.1 | 0.6 | 12.3 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 24.1 | 2.8 | 8.1 | - | 3.2 | 3.3 | 41.6 |
| **total** | **54.5** | **6.1** | **12.2** | **4.9** | **11.1** | **11.2** | **100** |

## GEMM achieved rates

FLOPs and weight bytes from the same layer shapes used in
`docs/imagewam_rtx5090_profile.md` (all 20 single-stream layers counted
as full layers). Weight bytes per parameter: fp16 2, fp8 1, nvfp4 0.5625.
No Thor peak is available, so there is no "% of GEMM peak" / "% of DRAM"
column.

| module | precision | GEMM ms | TFLOPS | weight GB/s |
|---|---|---:|---:|---:|
| backbone double | fp16 | 11.44 | 45 | 214 |
| backbone double | fp8 | 9.40 | 54 | 131 |
| backbone double | nvfp4 | 6.11 | 84 | 113 |
| backbone single | fp16 | 26.48 | 77 | 185 |
| backbone single | fp8 | 16.64 | 123 | 147 |
| backbone single | nvfp4 | 12.20 | 167 | 113 |
| ActionDiT double | fp16 | 11.22 | 14 | 224 |
| ActionDiT double | fp8 | 6.21 | 26 | 203 |
| ActionDiT double | nvfp4 | 6.01 | 27 | 118 |
| ActionDiT single | fp16 | 50.17 | 13 | 201 |
| ActionDiT single | fp8 | 31.91 | 20 | 158 |
| ActionDiT single | nvfp4 | 19.44 | 33 | 146 |

## Bottleneck per part

Only claims that the tables themselves support. No Thor roofline.

| part | share of eager fp16 / fp8 / nvfp4 | what the tables show |
|---|---|---|
| ActionDiT single-stream GEMM | 38% / 30% / 24% | Largest GEMM cell in every precision; M = 64, 10 steps |
| backbone single-stream GEMM | 20% / 16% / 15% | Second GEMM cell; M = 416 |
| attention (FA4) | 7% / 9% / 12% | 9.6–9.8 ms in every precision; FA4, not the 5090 cuBLAS chain |
| elementwise | 7% / 9% / 11% | 9.0–9.7 ms: `silu_glu_merged` 5.7 ms over 280 launches (backbone single 3.4 ms over 40), the single-stream attention-output copies 1.3 ms over 220, VAE `bias_residual` 0.9 ms; fp8 adds `compute_scale` |
| norm | 7% / 9% / 11% | 8.9–9.1 ms: `qkv_split_norm_rope_fp16` 3.64 ms over 280 launches, VAE `gn_apply` 1.76 ms, `gate_res_ada_layer_norm` 1.5 ms over 300 |
| VAE encode | 7% / 8% / 11% | 8.69–8.73 ms, 25 conv; precision-independent |
| nvfp4 quant/cast | – / – / 6.1% | 4.95 ms, mostly `kernel_quantize_fp4_sfa_vec` |
| fp8 absmax | – / 4.0% / – | `absmax_kernel`, 680 launches |

Fastest served tier on this GPU for this configuration: `nvfp4`
(`infer()` P50 80.22 ms).

## Same-cell comparison to RTX 5090

5090 numbers from `docs/imagewam_rtx5090_profile.md` (eager ms, and % of
that machine's eager total), measured at the same commit and served
defaults: `vec_rms_norm` on, `fuse_qkv_norm_rope` on. Thor numbers are from
the tables above. Wall time is `infer()` P50. GEMM ms are the module
cells, so "backbone single" includes the entry projection and "ActionDiT
single" the ActionDiT enc/head GEMMs on both machines. Configuration
differs where the default profile does: Thor FA4 on, 5090 cuBLAS
attention. Both have VAE in graph, `text_trim` and `x0` = 24.

### Wall time (ms)

| precision | Thor `infer()` | 5090 `infer()` | Thor / 5090 |
|---|---:|---:|---:|
| fp16 | 139.17 | 32.88 | 4.2× |
| fp8 | 111.86 | 23.85 | 4.7× |
| nvfp4 | 80.22 | 28.23 | 2.8× |

5090's fastest tier is fp8; Thor's is nvfp4.

### Eager GEMM ms (same module)

| module | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| backbone double | 11.44 | 3.50 | 9.40 | 1.40 | 6.11 | 1.10 |
| backbone single | 26.48 | 10.43 | 16.64 | 3.42 | 12.20 | 2.35 |
| ActionDiT double | 11.22 | 2.19 | 6.21 | 1.42 | 6.01 | 2.38 |
| ActionDiT single | 50.17 | 7.90 | 31.91 | 4.59 | 19.44 | 7.52 |

### Eager norm, elementwise, attention and VAE ms

| part | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| norm total | 9.05 | 2.11 | 9.09 | 2.08 | 8.93 | 2.08 |
| elementwise total | 9.14 | 1.17 | 9.71 | 1.70 | 8.99 | 1.14 |
| attention total | 9.69 | 3.38 | 9.64 | 3.41 | 9.80 | 3.39 |
| VAE encode total | 8.70 | 3.10 | 8.69 | 3.09 | 8.73 | 3.09 |

The same fused kernel takes 3.64 ms on Thor and 0.73–0.77 ms on 5090
(280 launches each). Elementwise differs most: `silu_glu_merged` alone
is 5.7 ms on Thor against 0.58 ms on 5090. Attention is FA4 vs cuBLAS.
VAE is precision-independent on both machines.

### Share of eager total, GEMM and norm (%)

| cell | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM total | 75.8 | 72.6 | 61.3 | 48.3 | 54.5 | 47.2 |
| norm total | 6.9 | 6.4 | 8.7 | 9.2 | 11.1 | 7.3 |
| ActionDiT single GEMM | 38.2 | 23.8 | 30.4 | 20.4 | 24.1 | 26.5 |
| backbone single GEMM | 20.2 | 31.5 | 15.9 | 15.2 | 15.1 | 8.3 |

## Conclusions

1. Served wall time on Thor, LIBERO config, default profile
   (`vec_rms_norm` and `fuse_qkv_norm_rope` on): nvfp4 80.22 ms, fp8
   111.86 ms, fp16 139.17 ms. nvfp4 is the fastest of the three (5090's
   fastest is fp8). The 98.0 ms figure elsewhere is the 3-view /
   horizon-30 target shape, not this LIBERO row.
2. Kernels per iteration are equal in every cell (2834 / 5304 / 3259),
   and eager totals track `infer()` P50 and graph-replay kernel time to a
   few percent.
3. GEMM is 55–76% of eager time; the ActionDiT single-stream GEMM (M = 64,
   10 steps) is the largest cell in every precision.
4. The precision-independent part is about 37 ms: VAE 8.7, attention
   9.7, norm 9.0, elementwise 9.0–9.7 ms. On 5090 the same part is about
   10 ms.
   - With the fused Q/K/V kernel, the largest elementwise item is
     `silu_glu_merged` (5.7 ms).
   - The largest norm item is the fused kernel itself (3.64 ms).
5. FA4 is on by default on Thor and was left on. Attention is 9.7 ms
   eager vs 3.4 ms cuBLAS on 5090 — a different operator, not a slower
   instance of the same one.
6. NVFP4 activation quantization on Thor is `kernel_quantize_fp4_sfa_vec`
   (4.95 ms), not the 5090 `cast_fp16_to_bf16` + `quantize_bf16_to_nvfp4`
   + `cast_bf16_to_fp16` sequence (5.96 ms).
7. Defaults that match 5090: `text_trim`, VAE in graph, `x0` = 24,
   `vec_rms_norm` on, `fuse_qkv_norm_rope` on. Default that does not: FA4.
