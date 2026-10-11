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
  Served options as resolved on this machine (defaults were not changed to
  match 5090): `text_trim=True` (prompt "pick up the red block and place it
  in the tray" → `x0` = 24, `a0` = 416, `num_action` = 64, `total` = 480),
  VAE inside the CUDA graph (`vae_graph_input=(2,224,224)`), FA4 at both
  attention sites (`use_fa4` in the profile is `None` / auto; construction
  enables FA4 — `flashrt_fa4` / `FlashAttentionForwardSm100` is in the
  kernel dump). `vec_rms_norm` off (default; the QK RMSNorm kernel is
  `rms_norm_kernel<__half>`, not the warp-per-row `rms_norm_fp16_vec`).
- GEMM M: backbone text stream 24, image stream 392, single-stream 416;
  ActionDiT 64.
- Precisions: `fp16`, `fp8` (dynamic-scale `Fp8Linear`), `nvfp4`. ImageWAM
  has no `bf16` precision. Each precision ran in its own process.
- Machine: Jetson AGX Thor Developer Kit, SM 11.0, torch 2.9.1+cu130,
  CUDA 13.0, FlashRT `286dd82`. Power mode MAXN (`nvpmodel -q`). GPU
  clocks pinned (read-only): GPC 1.575 GHz, NVD 1.692 GHz, `emc_locked`
  unknown. No `nvpmodel -m` / `jetson_clocks` / sysfs write. GPU exclusive
  during the timed runs (no other compute process on `nvidia-smi`).
  Environment: `FlashRT/env.sh` + in-tree `.venv` (this Thor does not use
  the 5090 conda env).

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
  | fp16 | [4504, 4504, 4504] (equal) | 159.22 | 168.26 | 172.54 | −5.4% | +2.5% |
  | fp8 | [6974, 6974, 6974] (equal) | 133.44 | 137.25 | 140.47 | −2.8% | +2.3% |
  | nvfp4 | [4929, 4929, 4929] (equal) | 107.93 | 106.65 | 110.51 | +1.2% | +3.6% |

  Launch counts are integers. Structural floors hold: ActionDiT double
  GEMM 250 / 200 / 200 (floor 200); backbone single GEMM 40 + 3 entry
  (floor 40). fp8 ActionDiT double quant/cast = absmax = GEMM = 200.
  VAE encode is 8.76 / 8.77 / 8.64 ms with 25 conv launches in every
  precision. The module × operator tables use the eager grid; shares
  are of that eager total, which now tracks `infer()` wall to a few
  percent.
- No GEMM-peak or DRAM-ceiling measurement on this Thor in this run.
  Achieved TFLOPS and weight GB/s below reuse the 5090 doc's layer-shape
  inventory (same architecture, same LIBERO M). No compute-bound vs
  bandwidth-bound call is made from those rates.
- Scripts: `benchmarks/imagewam_module_operator_profile.py`,
  `benchmarks/imagewam_module_operator_share.py`. Raw JSON is local
  (`logs/1011_thor_profile_rerun/`, not in the repository).

## Wall time (ms)

| precision | `infer()` P50 | prefill graph | denoise-loop graph |
|---|---:|---:|---:|
| fp16 | 168.26 | 56.95 | 102.26 |
| fp8 | 137.25 | 47.77 | 80.29 |
| nvfp4 | 106.65 | 38.59 | 59.13 |

Remainder (`infer()` − prefill − denoise) is 9.05 / 9.19 / 8.93 ms.
`vae_stage_p50` from the script is ~0.08–0.10 ms because the VAE runs
inside the full infer graph, not in `stage_images` alone. A previous
valid wall measurement on `876b209` (167.16 / 137.77 / 106.05) matches
this rerun within 1.1 ms.

## GPU ms per `infer()` by module and operator (launches)

Attention = FA4 (`FlashAttentionForwardSm100`) + the memcpy that
`attn.run` issues around it. ActionDiT rows are summed over 10 denoise
steps. Totals are the eager kernel total.

### fp16

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.15 (5) | - | 0.33 (1) | 3.98 (25) | 2.37 (68) | 1.94 (29) | 8.76 |
| backbone double (5 blocks) | 11.25 (45) | - | 0.21 (10) | - | 2.30 (42) | 1.26 (54) | 15.02 |
| backbone single (20 blocks) + entry | 26.96 (43) | - | 0.83 (40) | - | 7.91 (60) | 6.56 (140) | 42.26 |
| ActionDiT double (5 blocks × 10 steps) | 11.38 (250) | - | 1.52 (100) | - | 3.40 (210) | 1.61 (320) | 17.91 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 49.53 (630) | 0.01 (10) | 6.08 (400) | - | 12.46 (600) | 7.18 (1420) | 75.26 |
| **total** | **99.26** | **0.01** | **8.96** | **3.98** | **28.44** | **18.56** | **159.22** |

### fp8

| module | GEMM | quant/cast | absmax | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.16 (5) | - | - | 0.33 (1) | 4.01 (25) | 2.35 (68) | 1.94 (29) | 8.77 |
| backbone double (5 blocks) | 9.34 (45) | 0.43 (40) | 0.33 (40) | 0.21 (10) | - | 2.34 (42) | 1.25 (94) | 13.92 |
| backbone single (20 blocks) + entry | 16.46 (43) | 1.35 (40) | 0.68 (40) | 0.83 (40) | - | 7.96 (60) | 6.47 (180) | 33.76 |
| ActionDiT double (5 blocks × 10 steps) | 6.19 (200) | 0.53 (200) | 0.98 (200) | 1.59 (100) | - | 3.40 (210) | 1.82 (520) | 14.60 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 32.45 (430) | 1.54 (410) | 2.20 (400) | 6.11 (400) | - | 12.45 (600) | 7.50 (1820) | 62.38 |
| **total** | **64.59** | **3.85** | **4.19** | **9.07** | **4.01** | **28.50** | **18.97** | **133.44** |

"absmax" is `absmax_kernel` (the 5090 "other" / "absmax + memset" column).
Memset around it is 0.27 ms and is omitted from the cells above.

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.15 (5) | - | 0.33 (1) | 3.90 (25) | 2.34 (68) | 1.93 (29) | 8.64 |
| backbone double (5 blocks) | 6.08 (40) | 0.51 (40) | 0.21 (10) | - | 2.32 (42) | 1.14 (54) | 10.27 |
| backbone single (20 blocks) + entry | 12.24 (43) | 1.36 (40) | 0.83 (40) | - | 8.01 (60) | 6.50 (140) | 28.95 |
| ActionDiT double (5 blocks × 10 steps) | 6.14 (200) | 0.84 (200) | 1.56 (100) | - | 3.38 (210) | 1.61 (320) | 13.53 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 18.67 (430) | 2.26 (410) | 6.10 (400) | - | 12.42 (600) | 7.07 (1420) | 46.52 |
| **total** | **43.28** | **4.98** | **9.03** | **3.90** | **28.48** | **18.25** | **107.93** |

"quant/cast" in nvfp4 is `kernel_quantize_fp4_sfa_vec` plus the ActionDiT
`cast_fp32_fp16` on the Euler path.

## Share of eager kernel total by module and operator (%)

Same runs. Each cell is the cell's eager GPU time divided by that
precision's eager kernel total (fp16 159.22, fp8 133.44, nvfp4 107.93 ms).
Those totals track `infer()` wall to a few percent (see Method).

### fp16

| module | GEMM | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | 0.2 | 2.5 | 1.5 | 1.2 | 5.5 |
| backbone double (5 blocks) | 7.1 | 0.1 | - | 1.4 | 0.8 | 9.4 |
| backbone single (20 blocks) + entry | 16.9 | 0.5 | - | 5.0 | 4.1 | 26.5 |
| ActionDiT double (5 blocks × 10 steps) | 7.1 | 1.0 | - | 2.1 | 1.0 | 11.2 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 31.1 | 3.8 | - | 7.8 | 4.5 | 47.3 |
| **total** | **62.3** | **5.6** | **2.5** | **17.9** | **11.7** | **100** |

### fp8

| module | GEMM | quant/cast | absmax | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | - | 0.2 | 3.0 | 1.8 | 1.5 | 6.6 |
| backbone double (5 blocks) | 7.0 | 0.3 | 0.3 | 0.2 | - | 1.8 | 0.9 | 10.4 |
| backbone single (20 blocks) + entry | 12.3 | 1.0 | 0.5 | 0.6 | - | 6.0 | 4.8 | 25.3 |
| ActionDiT double (5 blocks × 10 steps) | 4.6 | 0.4 | 0.7 | 1.2 | - | 2.6 | 1.4 | 10.9 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 24.3 | 1.2 | 1.6 | 4.6 | - | 9.3 | 5.6 | 46.7 |
| **total** | **48.4** | **2.9** | **3.1** | **6.8** | **3.0** | **21.4** | **14.2** | **100** |

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | 0.3 | 3.6 | 2.2 | 1.8 | 8.0 |
| backbone double (5 blocks) | 5.6 | 0.5 | 0.2 | - | 2.2 | 1.1 | 9.5 |
| backbone single (20 blocks) + entry | 11.3 | 1.3 | 0.8 | - | 7.4 | 6.0 | 26.8 |
| ActionDiT double (5 blocks × 10 steps) | 5.7 | 0.8 | 1.4 | - | 3.1 | 1.5 | 12.5 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 17.3 | 2.1 | 5.7 | - | 11.5 | 6.6 | 43.1 |
| **total** | **40.1** | **4.6** | **8.4** | **3.6** | **26.4** | **16.9** | **100** |

## GEMM achieved rates

FLOPs and weight bytes from the same layer shapes used in
`docs/imagewam_rtx5090_profile.md` (all 20 single-stream layers counted
as full layers). Weight bytes per parameter: fp16 2, fp8 1, nvfp4 0.5625.
No Thor peak is available, so there is no "% of GEMM peak" / "% of DRAM"
column.

| module | precision | GEMM ms | TFLOPS | weight GB/s |
|---|---|---:|---:|---:|
| backbone double | fp16 | 11.25 | 45 | 218 |
| backbone double | fp8 | 9.34 | 55 | 131 |
| backbone double | nvfp4 | 6.08 | 84 | 113 |
| backbone single | fp16 | 26.96 | 76 | 182 |
| backbone single | fp8 | 16.46 | 124 | 149 |
| backbone single | nvfp4 | 12.24 | 167 | 113 |
| ActionDiT double | fp16 | 11.38 | 14 | 221 |
| ActionDiT double | fp8 | 6.19 | 26 | 203 |
| ActionDiT double | nvfp4 | 6.14 | 26 | 115 |
| ActionDiT single | fp16 | 49.53 | 13 | 203 |
| ActionDiT single | fp8 | 32.45 | 20 | 155 |
| ActionDiT single | nvfp4 | 18.67 | 34 | 152 |

## Bottleneck per part

Only claims that the tables themselves support. No Thor roofline.

| part | share of eager fp16 / fp8 / nvfp4 | what the tables show |
|---|---|---|
| ActionDiT single-stream GEMM | 31% / 24% / 17% | Largest GEMM cell in every precision; M = 64, 10 steps |
| backbone single-stream GEMM | 17% / 12% / 11% | Second GEMM cell; M = 416 |
| norm | 18% / 21% / 26% | `rms_norm_kernel` (default, `vec_rms_norm` off), 28.4–28.5 ms in every precision. This is the kernel OPT-032 candidate 5 replaces |
| elementwise | 12% / 14% / 17% | Launch-heavy: 1,400+ `at::native` copies plus SiLU-GLU / RoPE |
| attention (FA4) | 6% / 7% / 8% | 9.0 ms in every precision; FA4, not the 5090 cuBLAS QK^T+softmax+PV chain |
| VAE encode | 6% / 7% / 8% | 8.64–8.77 ms, 25 conv; precision-independent |
| nvfp4 quant/cast | – / – / 4.6% | 4.98 ms, mostly `kernel_quantize_fp4_sfa_vec` |
| fp8 absmax | – / 3.1% / – | `absmax_kernel`, 680 launches |

Fastest served tier on this GPU for this configuration: `nvfp4`
(`infer()` P50 106.65 ms). That ranking is wall time.

## Same-cell comparison to RTX 5090

5090 numbers from `docs/imagewam_rtx5090_profile.md` (eager ms and % of
that machine's eager total). Thor numbers from the tables above. Wall
time is `infer()` P50. Configuration differs only where the default
profile does: Thor FA4 on, 5090 cuBLAS attention; both have VAE in graph,
`text_trim`, `x0` = 24, `vec_rms_norm` off.

### Wall time (ms)

| precision | Thor `infer()` | 5090 `infer()` | Thor / 5090 |
|---|---:|---:|---:|
| fp16 | 168.26 | 36.14 | 4.7× |
| fp8 | 137.25 | 27.34 | 5.0× |
| nvfp4 | 106.65 | 31.68 | 3.4× |

5090's fastest tier is fp8; Thor's is nvfp4.

### Eager GEMM ms (same module)

| module | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| backbone double | 11.25 | 3.50 | 9.34 | 1.41 | 6.08 | 1.10 |
| backbone single | 26.96 | 10.39 | 16.46 | 3.39 | 12.24 | 2.32 |
| ActionDiT double | 11.38 | 2.19 | 6.19 | 1.42 | 6.14 | 2.38 |
| ActionDiT single | 49.53 | 7.69 | 32.45 | 4.55 | 18.67 | 7.48 |

### Eager norm, attention and VAE ms

| part | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| norm total | 28.44 | 3.73 | 28.50 | 3.77 | 28.48 | 3.75 |
| attention total | 8.96 | 3.38 | 9.07 | 3.39 | 9.03 | 3.40 |
| VAE encode total | 8.76 | 3.10 | 8.77 | 3.10 | 8.64 | 3.11 |

Thor default QK RMSNorm is ~7.6× the 5090 norm time at the same
`vec_rms_norm=off` (28.5 vs 3.7 ms). Attention is FA4 vs cuBLAS, so the
ms are not the same operator. VAE is precision-independent on both
machines.

### Share of eager total, GEMM and norm (%)

| cell | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM total | 62.3 | 65.1 | 48.4 | 41.6 | 40.1 | 41.8 |
| norm total | 17.9 | 10.2 | 21.4 | 14.4 | 26.4 | 11.7 |
| ActionDiT single GEMM | 31.1 | 21.1 | 24.3 | 17.6 | 17.3 | 23.5 |
| backbone single GEMM | 16.9 | 28.4 | 12.3 | 13.1 | 11.3 | 7.3 |

## Conclusions

1. Served wall time on Thor, LIBERO config, default profile: nvfp4 106.65
   ms, fp8 137.25 ms, fp16 168.26 ms. nvfp4 is the fastest of the three
   (5090's fastest was fp8).
2. Kernels per iteration are equal in every cell (4504 / 6974 / 4929),
   and eager totals track `infer()` P50 and graph-replay kernel time to a
   few percent.
3. With `vec_rms_norm` off, norm is 18–26% of eager time and 28.4–28.5 ms
   in every precision, vs ~3.7 ms / 10–14% on 5090 at the same flag. The
   kernel is `rms_norm_kernel`.
4. FA4 is on by default on Thor and was left on. Attention is 9.0 ms
   eager vs 3.4 ms cuBLAS on 5090 — a different operator, not a slower
   instance of the same one.
5. NVFP4 activation quantization on Thor is `kernel_quantize_fp4_sfa_vec`
   (4.98 ms), not the 5090 `cast_fp16_to_bf16` + `quantize_bf16_to_nvfp4`
   + `cast_bf16_to_fp16` sequence (5.98 ms).
6. Defaults that match 5090: `text_trim`, VAE in graph, `x0` = 24,
   `vec_rms_norm` off. Default that does not: FA4.
