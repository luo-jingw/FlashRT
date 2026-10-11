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
  `vec_rms_norm` off (default).
- GEMM M: backbone text stream 24, image stream 392, single-stream 416;
  ActionDiT 64.
- Precisions: ImageWAM has no `bf16` precision; the 16-bit tier is `fp16`
  (backbone residual in bf16). `fp8` is the dynamic-scale tier (`Fp8Linear`,
  cuBLASLt); the static tiers need a calibration file that is not on this
  machine. `nvfp4` is `Nvfp4LinearSm120` (SM120 CUTLASS ping-pong, FP16↔BF16
  casts + per-call activation quantization).
- Machine: RTX 5090, driver 595.91.07, torch 2.14.0+cu130, FlashRT
  `7ca7c99`, SM clock not locked, GPU not shared during the runs.

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
  Operator class by kernel name; kernels inside `attn.run` are "attention"
  (QK^T GEMM + softmax + PV GEMM).
- Cross-check: kernel time under graph replay vs eager attribution: 36.32
  vs 36.72 ms (fp16), 25.67 vs 26.21 ms (fp8), 31.58 vs 32.12 ms (nvfp4);
  no kernel unattributed. Graph-replay kernel time equals `infer()` wall
  time to within 0.1 ms (fp16, nvfp4) and 1.7 ms (fp8): the GPU is not idle
  between kernels.
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

| precision | `infer()` P50 | prefill graph | denoise-loop graph |
|---|---:|---:|---:|
| fp16 | 36.14 | 16.11 | 16.83 |
| fp8 | 27.34 | 7.78 | 16.26 |
| nvfp4 | 31.68 | 7.41 | 21.07 |

## GPU ms per `infer()` by module and operator (launches)

Attention = cuBLAS QK^T + softmax + PV. ActionDiT rows are summed over 10
denoise steps.

### fp16

| module | GEMM | quant/cast | attention | conv | norm | elementwise | other | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.03 | - | 0.21 | 2.41 | 0.31 | 0.14 | - | 3.10 |
| backbone double (5 blocks) | 3.50 | - | 0.15 | - | 0.30 | 0.14 | - | 4.09 |
| backbone single (20 blocks) | 10.39 | - | 0.59 | - | 0.69 | 0.65 | - | 12.32 |
| backbone entry | 0.04 | - | - | - | - | - | - | 0.04 |
| ActionDiT double (5 blocks × 10 steps) | 2.19 (300) | - | 0.49 (150) | - | 0.61 (210) | 0.43 (320) | - | 3.71 |
| ActionDiT single (20 blocks × 10 steps) | 7.69 (600) | - | 1.94 (600) | - | 1.82 (600) | 1.90 (1400) | - | 13.36 |
| ActionDiT enc/head/Euler | 0.06 | 0.01 | - | - | - | 0.02 | - | 0.09 |
| **total** | **23.90** | **0.01** | **3.38** | **2.41** | **3.73** | **3.28** | **0.01** | **36.72** |

### fp8

| module | GEMM | quant/cast | attention | conv | norm | elementwise | other | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.03 | - | 0.21 | 2.41 | 0.31 | 0.14 | - | 3.10 |
| backbone double (5 blocks) | 1.41 | 0.07 | 0.15 | - | 0.30 | 0.17 | 0.09 | 2.18 |
| backbone single (20 blocks) | 3.39 | 0.15 | 0.59 | - | 0.72 | 0.62 | 0.13 | 5.62 |
| backbone entry | 0.04 | - | - | - | - | - | - | 0.04 |
| ActionDiT double (5 blocks × 10 steps) | 1.42 (300) | 0.17 (200) | 0.49 (150) | - | 0.61 (210) | 0.58 (520) | 0.32 (400) | 3.59 |
| ActionDiT single (20 blocks × 10 steps) | 4.55 (600) | 0.39 (400) | 1.96 (600) | - | 1.83 (600) | 2.21 (1800) | 0.64 (800) | 11.58 |
| ActionDiT enc/head/Euler | 0.06 | 0.01 | - | - | - | 0.02 | - | 0.09 |
| **total** | **10.89** | **0.80** | **3.39** | **2.41** | **3.77** | **3.75** | **1.19** | **26.21** |

"other" in fp8 is the dynamic-scale `absmax_kernel` plus its memset.

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | other | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.03 | - | 0.21 | 2.42 | 0.31 | 0.14 | - | 3.11 |
| backbone double (5 blocks) | 1.10 | 0.51 | 0.15 | - | 0.30 | 0.14 | - | 2.20 |
| backbone single (20 blocks) | 2.32 | 1.22 | 0.59 | - | 0.70 | 0.51 | - | 5.33 |
| backbone entry | 0.04 | - | - | - | - | - | - | 0.04 |
| ActionDiT double (5 blocks × 10 steps) | 2.38 (200) | 1.08 (600) | 0.50 (150) | - | 0.61 (210) | 0.43 (320) | - | 4.99 |
| ActionDiT single (20 blocks × 10 steps) | 7.48 (400) | 3.16 (1200) | 1.95 (600) | - | 1.83 (600) | 1.92 (1400) | - | 16.35 |
| ActionDiT enc/head/Euler | 0.06 | 0.01 | - | - | - | 0.02 | - | 0.09 |
| **total** | **13.41** | **5.98** | **3.40** | **2.42** | **3.75** | **3.16** | **0.01** | **32.12** |

"quant/cast" in nvfp4 is `cast_fp16_to_bf16` + `quantize_bf16_to_nvfp4_swizzled`
+ `cast_bf16_to_fp16` around every NVFP4 GEMM.

## Share of `infer()` GPU time by module and operator (%)

Same runs as the tables above. Each cell is the cell's GPU time divided by
the eager kernel total of that precision (fp16 36.72, fp8 26.21, nvfp4
32.12 ms); the kernel total is within 2% of `infer()` wall time, so the
shares read as shares of inference time. The backbone entry projection
(0.1–0.2%) is merged into "backbone single"; the ActionDiT input/output
projections and Euler update (0.2–0.3%) are merged into "ActionDiT single".
ActionDiT rows are summed over 10 denoise steps.

### fp16

| module | GEMM | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | 0.6 | 6.6 | 0.9 | 0.4 | 8.4 |
| backbone double (5 blocks) | 9.5 | 0.4 | - | 0.8 | 0.4 | 11.1 |
| backbone single (20 blocks) | 28.4 | 1.6 | - | 1.9 | 1.8 | 33.6 |
| ActionDiT double (5 blocks × 10 steps) | 6.0 | 1.3 | - | 1.7 | 1.2 | 10.1 |
| ActionDiT single (20 blocks × 10 steps) | 21.1 | 5.3 | - | 5.0 | 5.3 | 36.6 |
| **total** | **65.1** | **9.2** | **6.6** | **10.2** | **8.9** | **100** |

### fp8

| module | GEMM | quant/cast | absmax + memset | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | - | 0.8 | 9.2 | 1.2 | 0.5 | 11.8 |
| backbone double (5 blocks) | 5.4 | 0.3 | 0.4 | 0.6 | - | 1.1 | 0.7 | 8.3 |
| backbone single (20 blocks) | 13.1 | 0.6 | 0.6 | 2.3 | - | 2.8 | 2.4 | 21.6 |
| ActionDiT double (5 blocks × 10 steps) | 5.4 | 0.7 | 1.3 | 1.9 | - | 2.3 | 2.2 | 13.7 |
| ActionDiT single (20 blocks × 10 steps) | 17.6 | 1.5 | 2.4 | 7.5 | - | 7.0 | 8.5 | 44.5 |
| **total** | **41.6** | **3.0** | **4.5** | **13.0** | **9.2** | **14.4** | **14.3** | **100** |

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | 0.6 | 7.5 | 1.0 | 0.4 | 9.7 |
| backbone double (5 blocks) | 3.4 | 1.6 | 0.5 | - | 0.9 | 0.4 | 6.8 |
| backbone single (20 blocks) | 7.3 | 3.8 | 1.8 | - | 2.2 | 1.6 | 16.7 |
| ActionDiT double (5 blocks × 10 steps) | 7.4 | 3.4 | 1.5 | - | 1.9 | 1.3 | 15.5 |
| ActionDiT single (20 blocks × 10 steps) | 23.5 | 9.9 | 6.1 | - | 5.7 | 6.1 | 51.2 |
| **total** | **41.8** | **18.6** | **10.6** | **7.5** | **11.7** | **9.8** | **100** |

## GEMM against the measured ceilings

FLOPs and weight bytes from the layer shapes (all 20 single-stream layers
counted as full layers). Weight bytes per parameter: fp16 2, fp8 1,
nvfp4 0.5625.

| module | precision | GEMM ms | TFLOPS | % of GEMM peak | weight GB/s | % of DRAM |
|---|---|---:|---:|---:|---:|---:|
| backbone double | fp16 | 3.50 | 146 | 65% | 700 | 46% |
| backbone double | fp8 | 1.41 | 363 | 60% | 871 | 57% |
| backbone double | nvfp4 | 1.10 | 462 | 38% | 625 | 41% |
| backbone single | fp16 | 10.39 | 197 | 87% | 472 | 31% |
| backbone single | fp8 | 3.39 | 602 | ≈100% | 724 | 47% |
| backbone single | nvfp4 | 2.32 | 881 | 72% | 596 | 39% |
| ActionDiT double | fp16 | 2.19 | 74 | 33% | 1149 | 75% |
| ActionDiT double | fp8 | 1.42 | 113 | 19% | 886 | 58% |
| ActionDiT double | nvfp4 | 2.38 | 68 | 6% | 297 | 19% |
| ActionDiT single | fp16 | 7.69 | 84 | 37% | 1309 | 86% |
| ActionDiT single | fp8 | 4.55 | 142 | 24% | 1106 | 72% |
| ActionDiT single | nvfp4 | 7.48 | 86 | 7% | 379 | 25% |

## Bottleneck per part

| part | share of fp16 / fp8 / nvfp4 time | bottleneck | evidence |
|---|---|---|---|
| backbone single-stream GEMM | 28% / 13% / 7% | **compute** | fp16 87% and fp8 ≈100% of the measured GEMM peak at M = 416 |
| backbone double-stream GEMM | 10% / 5% / 3% | compute, diluted by the M = 24 text stream | 60–65% of peak; the text-stream GEMMs at M = 24 run far below it |
| backbone NVFP4 overhead | – / – / 5% | **FP16↔BF16 casts + activation quantization** | 1.73 ms quant/cast on 3.42 ms of NVFP4 GEMM; the NVFP4 GEMM itself reaches 72% of its peak |
| ActionDiT GEMM (M = 64, 10 steps) | 27% / 23% / 31% | **DRAM bandwidth (weight streaming)** for fp16/fp8; **kernel efficiency** for nvfp4 | fp16 reads its 1.26 GB of weights per step at 75–86% of DRAM bandwidth, fp8 at 58–72%; nvfp4 reaches only 19–25% (128-row tiles at M = 64, under-filled N = 1024 shapes) |
| ActionDiT NVFP4 overhead | – / – / 13% | casts + quantization, launch-dominated | 4.24 ms over 1800 launches, 1.8–2.6 µs each |
| attention (cuBLAS chain) | 9% / 13% / 11% | **launch / latency** in ActionDiT | 750 launches per infer at 3.3 µs average; QK^T, softmax, PV as separate kernels |
| norm | 10% / 14% / 12% | **latency** in ActionDiT; per-head QK RMSNorm kernel efficiency | `rms_norm_kernel` 560 launches = 2.4 ms; 2.9–3.0 µs per launch at ActionDiT shapes; 11.5 µs per launch at backbone shapes |
| elementwise | 9% / 14% / 10% | **launch / latency** | 1720 launches in ActionDiT at 1.1–1.4 µs; 1060 per infer are `at::native` copy kernels (Q/K/V slice copies) |
| fp8 dynamic scale | – / 5% / – | latency | `absmax_kernel` + memset, 1200 launches |
| VAE encode | 8% / 12% / 10% | conv compute in bf16, unchanged by precision | cuDNN fprop kernels 2.41 ms of 3.10 ms |

## Scaling with visual-token count and denoise-step count

Scope: wall time only; action quality is not measured. Same deployment
entry, machine, checkpoint, prompt (`x0` = 24) and timing method as above
(`infer()`, prefill graph and denoise-loop graph P50 over 100 iterations,
one process per cell). Actions are finite in every cell.

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
   measured peak in fp16 and fp8; fp8 halves prefill (16.1 → 7.8 ms graph).
   NVFP4 runs the backbone GEMM itself 1.40× faster than fp8 (3.42 vs
   4.80 ms), but its casts
   and per-call quantization (1.73 ms) cancel most of that: prefill graph
   7.41 ms vs 7.78 ms.
2. ActionDiT GEMMs are weight-bandwidth-bound at M = 64: every denoise step
   re-reads 1.26 GB (fp16) of weights, more than L2 holds. fp8 cuts the
   GEMM time 1.6×, not 2×; nvfp4 reaches only a quarter of DRAM bandwidth
   and adds 4.24 ms of casts/quantization, so the denoise loop is slower
   in nvfp4 (21.1 ms) than in fp16 (16.8 ms).
3. The precision-independent part — VAE 3.1, attention 3.4, norm 3.7,
   elementwise 3.2–3.8 ms, about 13.5 ms — is half of the fp8 time. In
   ActionDiT it is launch/latency-bound: about 3,000 kernels per infer at
   1–3 µs each, of which 1,060 are slice copies.
4. Fastest tier on this GPU for this configuration: `fp8` (27.34 ms).
5. In fp8, attention, norm, elementwise and the dynamic-scale kernels
   together take 46% of inference time, more than all GEMMs (42%);
   ActionDiT takes 58%, and within ActionDiT the non-GEMM kernels take
   more time than the GEMMs.
6. Fewer image tokens shorten only prefill: keeping 36% of the image
   tokens cuts fp8 end-to-end time by 13% at full VAE cost. The denoise
   loop scales linearly with the step count at 1.63 ms per step (fp8).
