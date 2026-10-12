# ImageWAM on RTX 5090: module x operator profile

GPU time per `infer()` of the served ImageWAM LIBERO configuration on RTX
5090, split by model module and operator class, for `fp16`, `fp8` and
`nvfp4`, with the bottleneck of each part, and how prefill and the denoise
loop scale with the visual-token count and the denoise-step count.

## Configuration

- Workload: `ImageWAMWorkload.libero()` — 2 views 224x224, action horizon
  64, `action_dim` 7, proprio 8, text padded to 512, 10 denoise steps.
- Weights: real LIBERO checkpoint (`ImageWAM-FLUX.2-4B-LIBERO`), real
  FLUX.2 AE, Qwen3-4B text encoder; built with `load_imagewam` (default
  profile). Served options: `text_trim=True` (prompt "pick up the red block
  and place it in the tray" → `x0` = 24), VAE inside the CUDA graph
  (`vae_graph_input=(2,224,224)`), cuBLAS attention chain (no FA4 on SM120),
  `vec_rms_norm` on (frontend default: warp-per-row QK RMSNorm
  `rms_norm_fp16_vec`).
- GEMM M: backbone text stream 24, image stream 392, single-stream 416;
  ActionDiT 64.
- Precisions: ImageWAM has no `bf16` precision; the 16-bit tier is `fp16`
  (backbone residual in bf16). `fp8` is the dynamic-scale tier (`Fp8Linear`,
  cuBLASLt); the static tiers need a calibration file that is not on this
  machine. `nvfp4` is `Nvfp4LinearSm120` (SM120 CUTLASS ping-pong, FP16↔BF16
  casts + per-call activation quantization).
- Machine: RTX 5090, driver 595.91.07, torch 2.14.0+cu130, FlashRT
  `74b5efe`, SM clock not locked, GPU not shared during the runs.

## Method

- Wall time: `infer()` CUDA-event P50 (warmup 20, 100 iterations);
  prefill and denoise loop captured as separate CUDA graphs with the
  arguments `run_eager()` uses.
- Attribution: `run_eager()` under `torch.profiler`, every module function
  of `pipeline_thor.py` (`_double_stream_layer`, `_single_stream_layer[_kv_only]`,
  `_action_double_layer`, `_action_single_layer`, `imagewam_denoise_step`,
  `imagewam_prefill`, `imagewam_denoise_loop`), the attention backend's
  `run` and the VAE stage's `run` wrapped in a profiler range with a device
  sync at entry and exit, so each kernel lies inside one innermost range.
  A throwaway profiler session runs first (CUPTI initialization can drop
  kernel records in the first session). Operator class by kernel name;
  kernels inside `attn.run` are "attention" (QK^T GEMM + softmax + PV
  GEMM).
- Cross-check: kernels per profiled iteration were equal in every cell
  (fp16 4865, fp8 7585, nvfp4 6570); no kernel unattributed.

  | precision | eager kernels | graph-replay kernels | `infer()` P50 |
  |---|---:|---:|---:|
  | fp16 | 35.15 | 35.19 | 34.80 |
  | fp8 | 24.67 | 24.47 | 25.98 |
  | nvfp4 | 30.52 | 30.74 | 30.35 |

  Graph-replay kernel time equals `infer()` wall time to within 0.4 ms
  (fp16, nvfp4) and 1.5 ms (fp8): the GPU is not idle between kernels.
- Ceilings measured on the same GPU: dense GEMM 8192³ — fp16 225.5,
  fp8 601.6, nvfp4 1217.8 TFLOPS; DRAM 1528 GB/s (1 GiB device copy). No
  hardware counters (`RmProfilingAdminOnly=1`): bottlenecks below come from
  achieved FLOP/s and bytes/s against these ceilings and from per-launch
  duration.
- Scripts: `benchmarks/imagewam_module_operator_profile.py` (timing and
  attribution, one JSON per precision), `benchmarks/imagewam_module_operator_share.py`
  (ms and share tables from that JSON), `benchmarks/imagewam_token_step_scaling.py`
  (token-count and step-count cells). The ceiling measurement and the
  roofline arithmetic are not in the repository; the raw JSON is not in the
  repository.

## Wall time (ms)

| precision | `infer()` P50 | prefill graph | denoise-loop graph | `infer()` P50, `vec_rms_norm` off |
|---|---:|---:|---:|---:|
| fp16 | 34.80 | 15.60 | 15.98 | 36.14 |
| fp8 | 25.98 | 7.29 | 15.33 | 27.34 |
| nvfp4 | 30.35 | 6.89 | 20.20 | 31.68 |

The last column is the same configuration with `vec_rms_norm` off
(`7ca7c99`, same machine, separate processes): the flag saves 1.3–1.4 ms
per `infer()`.

## GPU ms per `infer()` by module and operator (launches)

Attention = cuBLAS QK^T + softmax + PV. The backbone entry projection is
merged into "backbone single"; the ActionDiT input/output projections and
Euler update into "ActionDiT single". ActionDiT rows are summed over 10
denoise steps. Totals are the eager kernel total (input staging, 0.01 ms,
not shown).

### fp16

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.03 (2) | - | 0.20 (1) | 2.40 (30) | 0.31 (68) | 0.14 (31) | 3.09 |
| backbone double (5 blocks) | 3.49 (55) | - | 0.15 (15) | - | 0.18 (42) | 0.14 (54) | 3.96 |
| backbone single (20 blocks) | 10.39 (63) | - | 0.59 (60) | - | 0.26 (60) | 0.65 (140) | 11.89 |
| ActionDiT double (5 blocks × 10 steps) | 2.18 (300) | - | 0.49 (150) | - | 0.42 (210) | 0.43 (320) | 3.52 |
| ActionDiT single (20 blocks × 10 steps) | 7.73 (630) | 0.01 (10) | 1.95 (600) | - | 1.07 (600) | 1.92 (1420) | 12.68 |
| **total** | **23.82** | **0.01** | **3.38** | **2.40** | **2.24** | **3.28** | **35.15** |

### fp8

| module | GEMM | quant/cast | absmax + memset | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.03 (2) | - | - | 0.20 (1) | 2.40 (30) | 0.31 (68) | 0.14 (31) | 3.09 |
| backbone double (5 blocks) | 1.40 (55) | 0.07 (40) | 0.09 (80) | 0.15 (15) | - | 0.18 (42) | 0.17 (94) | 2.06 |
| backbone single (20 blocks) | 3.42 (63) | 0.15 (40) | 0.13 (80) | 0.60 (60) | - | 0.26 (60) | 0.62 (180) | 5.20 |
| ActionDiT double (5 blocks × 10 steps) | 1.42 (300) | 0.17 (200) | 0.32 (400) | 0.49 (150) | - | 0.42 (210) | 0.58 (520) | 3.41 |
| ActionDiT single (20 blocks × 10 steps) | 4.61 (630) | 0.40 (410) | 0.63 (800) | 1.95 (600) | - | 1.07 (600) | 2.23 (1820) | 10.90 |
| **total** | **10.89** | **0.80** | **1.18** | **3.40** | **2.40** | **2.24** | **3.75** | **24.67** |

"absmax + memset" is the dynamic-scale `absmax_kernel` and its memset.

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.03 (2) | - | 0.21 (1) | 2.40 (30) | 0.31 (68) | 0.14 (31) | 3.09 |
| backbone double (5 blocks) | 1.10 (40) | 0.50 (120) | 0.15 (15) | - | 0.18 (42) | 0.14 (54) | 2.07 |
| backbone single (20 blocks) | 2.34 (43) | 1.21 (120) | 0.59 (60) | - | 0.25 (60) | 0.51 (140) | 4.90 |
| ActionDiT double (5 blocks × 10 steps) | 2.38 (200) | 1.08 (600) | 0.50 (150) | - | 0.42 (210) | 0.42 (320) | 4.80 |
| ActionDiT single (20 blocks × 10 steps) | 7.50 (430) | 3.17 (1210) | 1.96 (600) | - | 1.08 (600) | 1.95 (1420) | 15.65 |
| **total** | **13.34** | **5.97** | **3.40** | **2.40** | **2.24** | **3.16** | **30.52** |

"quant/cast" in nvfp4 is `cast_fp16_to_bf16` + `quantize_bf16_to_nvfp4_swizzled`
+ `cast_bf16_to_fp16` around every NVFP4 GEMM.

## Share of `infer()` GPU time by module and operator (%)

Same runs. Each cell is the cell's GPU time divided by the eager kernel
total of that precision (fp16 35.15, fp8 24.67, nvfp4 30.52 ms); the
kernel total is within 1.5 ms of `infer()` wall time, so the shares read
as shares of inference time. Rows merged as in the tables above.

### fp16

| module | GEMM | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | 0.6 | 6.8 | 0.9 | 0.4 | 8.8 |
| backbone double (5 blocks) | 9.9 | 0.4 | - | 0.5 | 0.4 | 11.3 |
| backbone single (20 blocks) | 29.6 | 1.7 | - | 0.7 | 1.8 | 33.8 |
| ActionDiT double (5 blocks × 10 steps) | 6.2 | 1.4 | - | 1.2 | 1.2 | 10.0 |
| ActionDiT single (20 blocks × 10 steps) | 22.0 | 5.5 | - | 3.1 | 5.5 | 36.1 |
| **total** | **67.8** | **9.6** | **6.8** | **6.4** | **9.3** | **100** |

### fp8

| module | GEMM | quant/cast | absmax + memset | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | - | 0.8 | 9.7 | 1.3 | 0.6 | 12.5 |
| backbone double (5 blocks) | 5.7 | 0.3 | 0.4 | 0.6 | - | 0.7 | 0.7 | 8.4 |
| backbone single (20 blocks) | 13.9 | 0.6 | 0.5 | 2.4 | - | 1.1 | 2.5 | 21.1 |
| ActionDiT double (5 blocks × 10 steps) | 5.8 | 0.7 | 1.3 | 2.0 | - | 1.7 | 2.3 | 13.8 |
| ActionDiT single (20 blocks × 10 steps) | 18.7 | 1.6 | 2.6 | 7.9 | - | 4.3 | 9.0 | 44.2 |
| **total** | **44.1** | **3.2** | **4.8** | **13.8** | **9.7** | **9.1** | **15.2** | **100** |

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | 0.7 | 7.9 | 1.0 | 0.5 | 10.1 |
| backbone double (5 blocks) | 3.6 | 1.7 | 0.5 | - | 0.6 | 0.5 | 6.8 |
| backbone single (20 blocks) | 7.7 | 4.0 | 1.9 | - | 0.8 | 1.7 | 16.1 |
| ActionDiT double (5 blocks × 10 steps) | 7.8 | 3.5 | 1.6 | - | 1.4 | 1.4 | 15.7 |
| ActionDiT single (20 blocks × 10 steps) | 24.6 | 10.4 | 6.4 | - | 3.5 | 6.4 | 51.3 |
| **total** | **43.7** | **19.6** | **11.1** | **7.9** | **7.3** | **10.4** | **100** |

## GEMM against the measured ceilings

FLOPs and weight bytes from the layer shapes (all 20 single-stream layers
counted as full layers; backbone entry and ActionDiT enc/head GEMMs
excluded). Weight bytes per parameter: fp16 2, fp8 1, nvfp4 0.5625.

| module | precision | GEMM ms | TFLOPS | % of GEMM peak | weight GB/s | % of DRAM |
|---|---|---:|---:|---:|---:|---:|
| backbone double | fp16 | 3.49 | 146 | 65% | 703 | 46% |
| backbone double | fp8 | 1.40 | 364 | 60% | 874 | 57% |
| backbone double | nvfp4 | 1.10 | 466 | 38% | 630 | 41% |
| backbone single | fp16 | 10.35 | 197 | 87% | 474 | 31% |
| backbone single | fp8 | 3.38 | 604 | ≈100% | 726 | 48% |
| backbone single | nvfp4 | 2.30 | 887 | 73% | 600 | 39% |
| ActionDiT double | fp16 | 2.18 | 74 | 33% | 1153 | 75% |
| ActionDiT double | fp8 | 1.42 | 113 | 19% | 884 | 58% |
| ActionDiT double | nvfp4 | 2.38 | 68 | 6% | 298 | 19% |
| ActionDiT single | fp16 | 7.67 | 84 | 37% | 1312 | 86% |
| ActionDiT single | fp8 | 4.55 | 141 | 24% | 1105 | 72% |
| ActionDiT single | nvfp4 | 7.44 | 87 | 7% | 381 | 25% |

## Bottleneck per part

| part | share of fp16 / fp8 / nvfp4 time | bottleneck | evidence |
|---|---|---|---|
| backbone single-stream GEMM | 30% / 14% / 8% | **compute** | fp16 87% and fp8 ≈100% of the measured GEMM peak at M = 416 |
| backbone double-stream GEMM | 10% / 6% / 4% | compute, diluted by the M = 24 text stream | 60–65% of peak; the text-stream GEMMs at M = 24 run far below it |
| backbone NVFP4 overhead | – / – / 6% | **FP16↔BF16 casts + activation quantization** | 1.71 ms quant/cast on 3.44 ms of NVFP4 GEMM; the NVFP4 GEMM itself reaches 73% of its peak |
| ActionDiT GEMM (M = 64, 10 steps) | 28% / 24% / 32% | **DRAM bandwidth (weight streaming)** for fp16/fp8; **kernel efficiency** for nvfp4 | fp16 reads its 1.26 GB of weights per step at 75–86% of DRAM bandwidth, fp8 at 58–72%; nvfp4 reaches only 19–25% (128-row tiles at M = 64, under-filled N = 1024 shapes) |
| ActionDiT NVFP4 overhead | – / – / 14% | casts + quantization, launch-dominated | 4.25 ms over 1810 launches, 1.8–2.6 µs each |
| attention (cuBLAS chain) | 10% / 14% / 11% | **launch / latency** in ActionDiT | 750 launches per infer at 3.3 µs average; QK^T, softmax, PV as separate kernels |
| norm | 6% / 9% / 7% | **launch / latency** | ActionDiT 810 launches at 1.8–2.0 µs: `rms_norm_fp16_vec_warp_kernel` 560 launches = 0.89 ms, `gate_res_ada_layer_norm_kernel` 300 launches = 0.71 ms |
| elementwise | 9% / 15% / 10% | **launch / latency** | 1740 launches in ActionDiT at 1.1–1.4 µs; 1060 per infer are `at::native` copy kernels (Q/K/V slice copies) |
| fp8 dynamic scale | – / 5% / – | latency | `absmax_kernel` + memset, 1200 launches |
| VAE encode | 9% / 13% / 10% | conv compute in bf16, unchanged by precision | cuDNN fprop kernels 2.40 ms of 3.09 ms |

## Scaling with visual-token count and denoise-step count

Scope: wall time only; action quality is not measured. Same deployment
entry, machine, checkpoint, prompt (`x0` = 24) and timing method as above
(`infer()`, prefill graph and denoise-loop graph P50 over 100 iterations,
one process per cell). Actions are finite in every cell. These cells were
measured at FlashRT `7ca7c99` with `vec_rms_norm` off, the frontend
default at that commit; the absolute times are therefore about 1.3 ms
above the `vec_rms_norm`-on tables above, and the relative changes are
what this section reports.

- Visual-token count is set through the view width with the view height
  fixed at 224: image tokens = (224/16) × (2·W/16), W ∈ {224, 144, 112,
  80} → 392, 252, 196, 140 tokens (100%, 64%, 50%, 36% kept). This
  reproduces the GEMM and attention shapes of a backbone input with that
  many image tokens; it is not a pruning method, and a pruning method's
  token-selection cost is not included.
- A narrower view also shrinks the VAE encode, which pruning after the
  VAE would not. "Remainder" is `infer()` − prefill − denoise (VAE encode
  plus input staging) as measured; "end-to-end at full VAE" replaces it
  with the W = 224 remainder of the same precision (3.21 fp16, 3.32 fp8,
  3.25 nvfp4 ms).
- The denoise-step count is set through `num_steps` at W = 224.

### Visual-token count (10 steps, ms)

| precision | image tokens (kept) | prefill graph | denoise-loop graph | `infer()` | remainder | end-to-end at full VAE | vs 392 tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| fp16 | 392 (100%) | 16.10 | 16.79 | 36.10 | 3.21 | 36.10 | – |
| fp16 | 252 (64%) | 10.70 | 16.84 | 29.66 | 2.12 | 30.75 | −15% |
| fp16 | 196 (50%) | 10.17 | 16.70 | 28.80 | 1.93 | 30.08 | −17% |
| fp16 | 140 (36%) | 7.63 | 16.47 | 25.58 | 1.48 | 27.31 | −24% |
| fp8 | 392 (100%) | 7.81 | 16.27 | 27.40 | 3.32 | 27.40 | – |
| fp8 | 252 (64%) | 6.01 | 16.31 | 24.51 | 2.20 | 25.64 | −6% |
| fp8 | 196 (50%) | 5.20 | 16.22 | 23.43 | 2.01 | 24.74 | −10% |
| fp8 | 140 (36%) | 4.55 | 15.86 | 21.95 | 1.55 | 23.73 | −13% |
| nvfp4 | 392 (100%) | 7.43 | 21.10 | 31.79 | 3.25 | 31.79 | – |
| nvfp4 | 252 (64%) | 5.89 | 21.09 | 29.16 | 2.18 | 30.23 | −5% |
| nvfp4 | 196 (50%) | 5.36 | 21.05 | 28.35 | 1.94 | 29.66 | −7% |
| nvfp4 | 140 (36%) | 4.85 | 20.77 | 27.12 | 1.50 | 28.87 | −9% |

- Prefill falls less than the token count: at 36% of the image tokens, fp8
  prefill is 58% of its 392-token time. The 24 text tokens are unchanged,
  and at smaller M the backbone GEMMs move from compute-bound toward the
  weight-read floor (3.68 B GEMM parameters, ≈2.4 ms at fp8 and 1528 GB/s).
- The denoise loop changes by at most 2.5%: ActionDiT GEMMs run at M = 64
  regardless of the image-token count, and image tokens enter the denoise
  loop only as attention K/V.

### Denoise-step count (392 image tokens, ms)

| precision | steps | prefill graph | denoise-loop graph | `infer()` | per step |
|---|---:|---:|---:|---:|---:|
| fp16 | 10 | 16.10 | 16.79 | 36.10 | 1.68 |
| fp16 | 5 | 16.15 | 8.51 | 27.87 | 1.70 |
| fp8 | 10 | 7.81 | 16.27 | 27.40 | 1.63 |
| fp8 | 5 | 7.82 | 8.14 | 19.23 | 1.63 |
| nvfp4 | 10 | 7.43 | 21.10 | 31.79 | 2.11 |
| nvfp4 | 5 | 7.44 | 10.55 | 21.23 | 2.11 |

The denoise loop is linear in the step count (per-step time equal at 5 and
10 steps to within 0.02 ms); halving the steps cuts `infer()` by 23% (fp16),
30% (fp8), 33% (nvfp4).

## Conclusions

1. Backbone (prefill) GEMMs are compute-bound and already near the
   measured peak in fp16 and fp8; fp8 halves prefill (15.6 → 7.3 ms graph).
   NVFP4 runs the backbone GEMM itself 1.40× faster than fp8 (3.44 vs
   4.82 ms), but its casts and per-call quantization (1.71 ms) cancel most
   of that: prefill graph 6.89 ms vs 7.29 ms.
2. ActionDiT GEMMs are weight-bandwidth-bound at M = 64: every denoise step
   re-reads 1.26 GB (fp16) of weights, more than L2 holds. fp8 cuts the
   GEMM time 1.6×, not 2×; nvfp4 reaches only a quarter of DRAM bandwidth
   and adds 4.25 ms of casts/quantization, so the denoise loop is slower
   in nvfp4 (20.2 ms) than in fp16 (16.0 ms).
3. The precision-independent part — VAE 3.1, attention 3.4, norm 2.2,
   elementwise 3.2–3.8 ms, about 12 ms — is half of the fp8 time. In
   ActionDiT it is launch/latency-bound: about 3,300 kernels per infer at
   1–3 µs each, of which 1,060 are slice copies.
4. Fastest tier on this GPU for this configuration: `fp8` (25.98 ms).
5. In fp8, attention, norm, elementwise and the dynamic-scale kernels
   together take 43% of inference time, about as much as all GEMMs (44%);
   ActionDiT takes 58%, and within ActionDiT the non-GEMM kernels take
   more time than the GEMMs.
6. Fewer image tokens shorten only prefill: keeping 36% of the image
   tokens cuts fp8 end-to-end time by 13% at full VAE cost. The denoise
   loop scales linearly with the step count at 1.63 ms per step (fp8).
