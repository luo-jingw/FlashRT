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
  `fuse_qkv_norm_rope` off (its default at the measured commit; it is
  default on from the commit after `a2ac9c2`, measured in
  opportunities.md OPT-032 candidate 1).
- GEMM M: backbone text stream 24, image stream 392, single-stream 416;
  ActionDiT 64.
- Precisions: `fp16`, `fp8` (dynamic-scale `Fp8Linear`), `nvfp4`. ImageWAM
  has no `bf16` precision. Each precision ran in its own process.
- Machine: Jetson AGX Thor Developer Kit, SM 11.0, torch 2.9.1+cu130,
  CUDA 13.0, FlashRT `acc0b10` plus the `vec_rms_norm` default-on change
  measured here. Power mode MAXN (`nvpmodel -q`). GPU clocks pinned
  (read-only): GPC 1.575 GHz, NVD 1.692 GHz, `emc_locked` unknown. No
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
  `attn.run` are "attention".
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
  | fp16 | [4504, 4504, 4504] (equal) | 139.49 | 149.38 | 153.03 | −6.6% | +2.4% |
  | fp8 | [6974, 6974, 6974] (equal) | 112.39 | 118.81 | 120.62 | −5.4% | +1.5% |
  | nvfp4 | [4929, 4929, 4929] (equal) | 88.47 | 87.34 | 90.00 | +1.3% | +3.0% |

  Launch counts are integers. Structural floors hold. The same LIBERO
  shape with `vec_rms_norm` off on this machine was nvfp4 106.65 ms
  (norm 28.5 ms); with the flag on, nvfp4 is 87.34 ms (norm 8.2 ms),
  matching the earlier real-checkpoint confirmation (86.4 ms) within
  1 ms. The 98.0 ms number is a different shape (3-view, horizon 30,
  `action_dim` 32, random weights, flag on).
- No GEMM-peak or DRAM-ceiling measurement on this Thor in this run.
  Achieved TFLOPS and weight GB/s below reuse the 5090 doc's layer-shape
  inventory. No compute-bound vs bandwidth-bound call is made from those
  rates.
- Scripts: `benchmarks/imagewam_module_operator_profile.py`,
  `benchmarks/imagewam_module_operator_share.py`. Raw JSON is local
  (`logs/1011_thor_profile_vecrms/`, not in the repository).

## Wall time (ms)

| precision | `infer()` P50 | prefill graph | denoise-loop graph |
|---|---:|---:|---:|
| fp16 | 149.38 | 49.32 | 90.78 |
| fp8 | 118.81 | 40.66 | 69.05 |
| nvfp4 | 87.34 | 31.17 | 47.25 |

Remainder (`infer()` − prefill − denoise) is 9.28 / 9.10 / 8.92 ms.
`vae_stage_p50` from the script is ~0.08–0.09 ms because the VAE runs
inside the full infer graph, not in `stage_images` alone.

## GPU ms per `infer()` by module and operator (launches)

Attention = FA4 (`FlashAttentionForwardSm100`) + the memcpy that
`attn.run` issues around it. ActionDiT rows are summed over 10 denoise
steps. Totals are the eager kernel total.

### fp16

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.15 (5) | - | 0.33 (1) | 3.97 (25) | 2.36 (68) | 1.93 (29) | 8.73 |
| backbone double (5 blocks) | 11.25 (45) | - | 0.21 (10) | - | 0.76 (42) | 1.25 (54) | 13.48 |
| backbone single (20 blocks) + entry | 26.68 (43) | - | 0.83 (40) | - | 1.66 (60) | 6.58 (140) | 35.75 |
| ActionDiT double (5 blocks × 10 steps) | 11.48 (250) | - | 1.57 (100) | - | 0.93 (210) | 1.60 (320) | 15.58 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 50.16 (630) | 0.01 (10) | 6.13 (400) | - | 2.51 (600) | 7.13 (1420) | 65.94 |
| **total** | **99.72** | **0.01** | **9.07** | **3.97** | **8.21** | **18.49** | **139.49** |

### fp8

| module | GEMM | quant/cast | absmax | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.14 (5) | - | - | 0.33 (1) | 3.97 (25) | 2.37 (68) | 1.95 (29) | 8.76 |
| backbone double (5 blocks) | 9.32 (45) | 0.43 (40) | 0.33 (40) | 0.21 (10) | - | 0.76 (42) | 1.25 (94) | 12.33 |
| backbone single (20 blocks) + entry | 16.51 (43) | 1.34 (40) | 0.68 (40) | 0.85 (40) | - | 1.67 (60) | 6.56 (180) | 27.63 |
| ActionDiT double (5 blocks × 10 steps) | 6.19 (200) | 0.53 (200) | 0.96 (200) | 1.53 (100) | - | 0.92 (210) | 1.82 (520) | 12.02 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 31.72 (430) | 1.55 (410) | 2.20 (400) | 5.98 (400) | - | 2.54 (600) | 7.52 (1820) | 51.64 |
| **total** | **63.89** | **3.85** | **4.17** | **8.90** | **3.97** | **8.26** | **19.09** | **112.39** |

"absmax" is `absmax_kernel`. Memset around it is 0.26 ms and is omitted
from the cells above.

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.15 (5) | - | 0.33 (1) | 3.98 (25) | 2.36 (68) | 1.96 (29) | 8.76 |
| backbone double (5 blocks) | 6.08 (40) | 0.51 (40) | 0.21 (10) | - | 0.73 (42) | 1.15 (54) | 8.68 |
| backbone single (20 blocks) + entry | 12.27 (43) | 1.36 (40) | 0.83 (40) | - | 1.67 (60) | 6.53 (140) | 22.65 |
| ActionDiT double (5 blocks × 10 steps) | 6.20 (200) | 0.83 (200) | 1.54 (100) | - | 0.91 (210) | 1.58 (320) | 11.06 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 19.29 (430) | 2.28 (410) | 6.09 (400) | - | 2.57 (600) | 7.07 (1420) | 37.30 |
| **total** | **43.98** | **4.98** | **9.00** | **3.98** | **8.23** | **18.29** | **88.47** |

"quant/cast" in nvfp4 is `kernel_quantize_fp4_sfa_vec` plus the ActionDiT
`cast_fp32_fp16` on the Euler path.

## Share of eager kernel total by module and operator (%)

Same runs. Each cell is the cell's eager GPU time divided by that
precision's eager kernel total (fp16 139.49, fp8 112.39, nvfp4 88.47 ms).

### fp16

| module | GEMM | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | 0.2 | 2.8 | 1.7 | 1.4 | 6.3 |
| backbone double (5 blocks) | 8.1 | 0.2 | - | 0.5 | 0.9 | 9.7 |
| backbone single (20 blocks) + entry | 19.1 | 0.6 | - | 1.2 | 4.7 | 25.6 |
| ActionDiT double (5 blocks × 10 steps) | 8.2 | 1.1 | - | 0.7 | 1.1 | 11.2 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 36.0 | 4.4 | - | 1.8 | 5.1 | 47.3 |
| **total** | **71.5** | **6.5** | **2.8** | **5.9** | **13.3** | **100** |

### fp8

| module | GEMM | quant/cast | absmax | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | - | 0.3 | 3.5 | 2.1 | 1.7 | 7.8 |
| backbone double (5 blocks) | 8.3 | 0.4 | 0.3 | 0.2 | - | 0.7 | 1.1 | 11.0 |
| backbone single (20 blocks) + entry | 14.7 | 1.2 | 0.6 | 0.8 | - | 1.5 | 5.8 | 24.6 |
| ActionDiT double (5 blocks × 10 steps) | 5.5 | 0.5 | 0.9 | 1.4 | - | 0.8 | 1.6 | 10.7 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 28.2 | 1.4 | 2.0 | 5.3 | - | 2.3 | 6.7 | 45.9 |
| **total** | **56.8** | **3.4** | **3.7** | **7.9** | **3.5** | **7.4** | **17.0** | **100** |

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.2 | - | 0.4 | 4.5 | 2.7 | 2.2 | 9.9 |
| backbone double (5 blocks) | 6.9 | 0.6 | 0.2 | - | 0.8 | 1.3 | 9.8 |
| backbone single (20 blocks) + entry | 13.9 | 1.5 | 0.9 | - | 1.9 | 7.4 | 25.6 |
| ActionDiT double (5 blocks × 10 steps) | 7.0 | 0.9 | 1.7 | - | 1.0 | 1.8 | 12.5 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 21.8 | 2.6 | 6.9 | - | 2.9 | 8.0 | 42.2 |
| **total** | **49.7** | **5.6** | **10.2** | **4.5** | **9.3** | **20.7** | **100** |

## GEMM achieved rates

FLOPs and weight bytes from the same layer shapes used in
`docs/imagewam_rtx5090_profile.md` (all 20 single-stream layers counted
as full layers). Weight bytes per parameter: fp16 2, fp8 1, nvfp4 0.5625.
No Thor peak is available, so there is no "% of GEMM peak" / "% of DRAM"
column.

| module | precision | GEMM ms | TFLOPS | weight GB/s |
|---|---|---:|---:|---:|
| backbone double | fp16 | 11.25 | 45 | 218 |
| backbone double | fp8 | 9.32 | 55 | 132 |
| backbone double | nvfp4 | 6.08 | 84 | 113 |
| backbone single | fp16 | 26.68 | 77 | 184 |
| backbone single | fp8 | 16.51 | 124 | 149 |
| backbone single | nvfp4 | 12.27 | 167 | 113 |
| ActionDiT double | fp16 | 11.48 | 14 | 219 |
| ActionDiT double | fp8 | 6.19 | 26 | 203 |
| ActionDiT double | nvfp4 | 6.20 | 26 | 114 |
| ActionDiT single | fp16 | 50.16 | 13 | 201 |
| ActionDiT single | fp8 | 31.72 | 20 | 159 |
| ActionDiT single | nvfp4 | 19.29 | 33 | 147 |

## Bottleneck per part

Only claims that the tables themselves support. No Thor roofline.

| part | share of eager fp16 / fp8 / nvfp4 | what the tables show |
|---|---|---|
| ActionDiT single-stream GEMM | 36% / 28% / 22% | Largest GEMM cell in every precision; M = 64, 10 steps |
| backbone single-stream GEMM | 19% / 15% / 14% | Second GEMM cell; M = 416 |
| elementwise | 13% / 17% / 21% | Launch-heavy: 1,400+ `at::native` copies plus SiLU-GLU / RoPE |
| attention (FA4) | 7% / 8% / 10% | 9.0 ms in every precision; FA4, not the 5090 cuBLAS chain |
| VAE encode | 6% / 8% / 10% | 8.73–8.76 ms, 25 conv; precision-independent |
| norm | 6% / 7% / 9% | 8.2 ms; VAE group-norm ~2.4 ms plus `rms_norm_fp16_vec` |
| nvfp4 quant/cast | – / – / 5.6% | 4.98 ms, mostly `kernel_quantize_fp4_sfa_vec` |
| fp8 absmax | – / 3.7% / – | `absmax_kernel`, 680 launches |

Fastest served tier on this GPU for this configuration: `nvfp4`
(`infer()` P50 87.34 ms).

## Same-cell comparison to RTX 5090

5090 numbers from `docs/imagewam_rtx5090_profile.md` (eager ms and % of
that machine's eager total). Both machines run `vec_rms_norm` on. Thor
numbers from the tables above. Wall time is
`infer()` P50. Configuration also differs where the default profile
does: Thor FA4 on, 5090 cuBLAS attention; both have VAE in graph,
`text_trim`, `x0` = 24.

### Wall time (ms)

| precision | Thor `infer()` | 5090 `infer()` | Thor / 5090 |
|---|---:|---:|---:|
| fp16 | 149.38 | 34.80 | 4.3× |
| fp8 | 118.81 | 25.98 | 4.6× |
| nvfp4 | 87.34 | 30.35 | 2.9× |

5090's fastest tier is fp8; Thor's is nvfp4.

### Eager GEMM ms (same module)

| module | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| backbone double | 11.25 | 3.49 | 9.32 | 1.40 | 6.08 | 1.10 |
| backbone single | 26.68 | 10.35 | 16.51 | 3.38 | 12.27 | 2.30 |
| ActionDiT double | 11.48 | 2.18 | 6.19 | 1.42 | 6.20 | 2.38 |
| ActionDiT single | 50.16 | 7.67 | 31.72 | 4.55 | 19.29 | 7.44 |

### Eager norm, attention and VAE ms

| part | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| norm total | 8.21 | 2.24 | 8.26 | 2.24 | 8.23 | 2.24 |
| attention total | 9.07 | 3.38 | 8.90 | 3.40 | 9.00 | 3.40 |
| VAE encode total | 8.73 | 3.09 | 8.76 | 3.09 | 8.76 | 3.09 |

Thor QK RMSNorm is now `rms_norm_fp16_vec`; the remaining ~8.2 ms of
"norm" is mostly VAE group-norm plus the cheap warp-per-row QK path
(with the flag off the same cell was 28.5 ms); on 5090 the same flag
leaves 2.24 ms. Attention is FA4 vs cuBLAS. VAE is precision-independent on both machines.

### Share of eager total, GEMM and norm (%)

| cell | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM total | 71.5 | 67.8 | 56.8 | 44.1 | 49.7 | 43.7 |
| norm total | 5.9 | 6.4 | 7.4 | 9.1 | 9.3 | 7.3 |
| ActionDiT single GEMM | 36.0 | 21.8 | 28.2 | 18.4 | 21.8 | 24.4 |
| backbone single GEMM | 19.1 | 29.4 | 14.7 | 13.7 | 13.9 | 7.5 |

## Conclusions

1. Served wall time on Thor, LIBERO config, default profile with
   `vec_rms_norm` on: nvfp4 87.34 ms, fp8 118.81 ms, fp16 149.38 ms.
   nvfp4 is the fastest of the three (5090's fastest was fp8). 87.34 ms
   matches the earlier real-checkpoint `vec_rms_norm=True` number
   (86.4 ms) within 1 ms. The 98.0 ms figure is the 3-view / horizon-30
   target shape, not this LIBERO row.
2. Kernels per iteration are equal in every cell (4504 / 6974 / 4929),
   and eager totals track `infer()` P50 and graph-replay kernel time to a
   few percent.
3. `vec_rms_norm` is the served default (`setdefault` on the frontend).
   Norm is 8.2 ms / 6–9% of eager time, vs 28.5 ms / 18–26% with the
   flag off on the same machine.
4. FA4 is on by default on Thor and was left on. Attention is 9.0 ms
   eager vs 3.4 ms cuBLAS on 5090 — a different operator, not a slower
   instance of the same one.
5. NVFP4 activation quantization on Thor is `kernel_quantize_fp4_sfa_vec`
   (4.98 ms), not the 5090 `cast_fp16_to_bf16` + `quantize_bf16_to_nvfp4`
   + `cast_bf16_to_fp16` sequence (5.98 ms).
6. Defaults that match 5090: `text_trim`, VAE in graph, `x0` = 24,
   `vec_rms_norm` on. Default that does not: FA4.
