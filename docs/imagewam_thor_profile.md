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
  CUDA 13.0, FlashRT `876b209`. Power mode MAXN (`nvpmodel -q`). GPU
  clocks pinned (read-only): GPC 1.575 GHz, NVD 1.692 GHz, `emc_locked`
  unknown. No `nvpmodel -m` / `jetson_clocks` / sysfs write. GPU exclusive
  during the timed runs (no other compute process on `nvidia-smi`).
  Environment: `FlashRT/env.sh` + in-tree `.venv` (this Thor does not use
  the 5090 conda env). Shared machine: occupancy was checked before each
  grid; a concurrent fp16 `--names` from another session was allowed to
  finish first and was not stopped.

## Method

Experiment grid (one variable): precision ∈ {fp16, fp8, nvfp4}. Everything
else fixed as above. Cells were enumerated before the timed rerun; the
first fp16 `--names` dump (another session) was used only to inspect
kernel names.

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
  `flash_rt::fp4::kernel_quantize_fp4_sfa_vec` is quant/cast (~4 ms) and
  not attention. `absmax_kernel` stays "other", as on 5090. After the
  regex change the whole grid was rerun. Unattributed kernel time: 0.000 ms
  on all three cells.
- Cross-check: graph-replay kernel total vs `infer()` P50 vs eager kernel
  total (ms):

  | precision | graph-replay kernels | `infer()` P50 | eager kernels | graph − P50 | P50 − eager |
  |---|---:|---:|---:|---:|---:|
  | fp16 | 172.37 | 167.16 | 109.55 | +5.2 | +57.6 |
  | fp8 | 140.41 | 137.77 | 74.25 | +2.6 | +63.5 |
  | nvfp4 | 109.85 | 106.05 | 89.79 | +3.8 | +16.3 |

  Graph-replay kernel time tracks `infer()` wall (within 3–5 ms): the GPU
  is not idle between kernels under the served graph. Eager attribution
  is 16–64 ms below wall. That gap is a Thor fact of this method (5090
  had eager ≈ wall to 2%). The module × operator tables below use the
  eager grid, so shares are shares of the eager kernel total, not of
  `infer()` wall. Do not read them as shares of the 167 / 138 / 106 ms
  numbers.
- fp8 VAE encode in the eager grid is 5.83 ms (17 conv launches) against
  8.74 / 8.78 ms (25 conv launches) in fp16 / nvfp4. The VAE is
  precision-independent; the fp8 cell is flagged, not renormalized.
- No GEMM-peak or DRAM-ceiling measurement on this Thor in this run.
  Achieved TFLOPS and weight GB/s below reuse the 5090 doc's layer-shape
  inventory (same architecture, same LIBERO M). No compute-bound vs
  bandwidth-bound call is made from those rates.
- Scripts: `benchmarks/imagewam_module_operator_profile.py`,
  `benchmarks/imagewam_module_operator_share.py`. Raw JSON is local
  (`logs/1011_thor_profile/`, not in the repository).

## Wall time (ms)

| precision | `infer()` P50 | prefill graph | denoise-loop graph |
|---|---:|---:|---:|
| fp16 | 167.16 | 57.32 | 100.91 |
| fp8 | 137.77 | 47.84 | 80.89 |
| nvfp4 | 106.05 | 38.23 | 58.89 |

Remainder (`infer()` − prefill − denoise) is 8.93 / 9.04 / 8.92 ms.
`vae_stage_p50` from the script is ~0.08 ms because the VAE runs inside
the full infer graph, not in `stage_images` alone.

## GPU ms per `infer()` by module and operator (launches)

Attention = FA4 (`FlashAttentionForwardSm100`) + the memcpy that
`attn.run` issues around it. ActionDiT rows are summed over 10 denoise
steps. Totals are the eager kernel total.

### fp16

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.16 (5) | - | 0.33 (1) | 3.95 (25) | 2.37 (68) | 1.94 (29) | 8.74 |
| backbone double (5 blocks) | 7.54 (30) | - | 0.14 (7) | - | 1.55 (28) | 0.84 (38) | 10.07 |
| backbone single (20 blocks) + entry | 18.12 (30) | - | 0.55 (27) | - | 5.32 (40) | 4.47 (93) | 28.46 |
| ActionDiT double (5 blocks × 10 steps) | 7.55 (167) | - | 1.03 (67) | - | 2.27 (140) | 1.07 (213) | 11.92 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 33.25 (420) | 0.01 (7) | 4.04 (267) | - | 8.30 (400) | 4.75 (947) | 50.34 |
| **total** | **66.62** | **0.01** | **6.09** | **3.95** | **19.80** | **13.08** | **109.55** |

### fp8

| module | GEMM | quant/cast | absmax | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.10 (3) | - | - | 0.22 (1) | 2.63 (17) | 1.57 (45) | 1.31 (19) | 5.83 |
| backbone double (5 blocks) | 6.22 (30) | 0.29 (27) | 0.22 (27) | 0.14 (7) | - | 1.54 (28) | 0.84 (63) | 9.26 |
| backbone single (20 blocks) + entry | 10.90 (29) | 0.90 (27) | 0.45 (27) | 0.57 (27) | - | 5.29 (40) | 4.38 (120) | 22.49 |
| ActionDiT double (5 blocks × 10 steps) | 3.08 (100) | 0.26 (100) | 0.47 (100) | 0.77 (50) | - | 1.70 (105) | 0.92 (260) | 7.24 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 15.11 (206) | 0.73 (197) | 1.05 (192) | 2.90 (192) | - | 5.96 (288) | 3.60 (873) | 29.42 |
| **total** | **35.41** | **2.18** | **2.20** | **4.60** | **2.63** | **16.05** | **11.04** | **74.25** |

"absmax" is `absmax_kernel` (the 5090 "other" / "absmax + memset" column).
Memset around it is 0.13 ms and is omitted from the cells above.

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.15 (5) | - | 0.33 (1) | 3.99 (25) | 2.36 (68) | 1.96 (29) | 8.78 |
| backbone double (5 blocks) | 6.08 (40) | 0.51 (40) | 0.22 (10) | - | 2.33 (42) | 1.15 (54) | 10.30 |
| backbone single (20 blocks) + entry | 12.02 (43) | 1.36 (40) | 0.85 (40) | - | 7.92 (60) | 6.44 (140) | 28.59 |
| ActionDiT double (5 blocks × 10 steps) | 4.32 (141) | 0.58 (141) | 1.12 (71) | - | 2.41 (148) | 1.13 (226) | 9.55 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 13.09 (301) | 1.59 (287) | 4.26 (280) | - | 8.67 (420) | 4.94 (994) | 32.56 |
| **total** | **35.65** | **4.04** | **6.78** | **3.99** | **23.70** | **15.62** | **89.79** |

"quant/cast" in nvfp4 is `kernel_quantize_fp4_sfa_vec` (501 launches,
4.03 ms) plus the ActionDiT `cast_fp32_fp16` on the Euler path.

## Share of eager kernel total by module and operator (%)

Same runs. Each cell is the cell's eager GPU time divided by that
precision's eager kernel total (fp16 109.55, fp8 74.25, nvfp4 89.79 ms).
These are **not** shares of `infer()` wall (see Method).

### fp16

| module | GEMM | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | 0.3 | 3.6 | 2.2 | 1.8 | 8.0 |
| backbone double (5 blocks) | 6.9 | 0.1 | - | 1.4 | 0.8 | 9.2 |
| backbone single (20 blocks) + entry | 16.5 | 0.5 | - | 4.9 | 4.1 | 26.0 |
| ActionDiT double (5 blocks × 10 steps) | 6.9 | 0.9 | - | 2.1 | 1.0 | 10.9 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 30.3 | 3.7 | - | 7.6 | 4.3 | 46.0 |
| **total** | **60.8** | **5.6** | **3.6** | **18.1** | **11.9** | **100** |

### fp8

| module | GEMM | quant/cast | absmax | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.1 | - | - | 0.3 | 3.5 | 2.1 | 1.8 | 7.8 |
| backbone double (5 blocks) | 8.4 | 0.4 | 0.3 | 0.2 | - | 2.1 | 1.1 | 12.5 |
| backbone single (20 blocks) + entry | 14.7 | 1.2 | 0.6 | 0.8 | - | 7.1 | 5.9 | 30.3 |
| ActionDiT double (5 blocks × 10 steps) | 4.1 | 0.4 | 0.6 | 1.0 | - | 2.3 | 1.2 | 9.8 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 20.3 | 1.0 | 1.4 | 3.9 | - | 8.0 | 4.8 | 39.6 |
| **total** | **47.7** | **2.9** | **3.0** | **6.2** | **3.5** | **21.6** | **14.9** | **100** |

### nvfp4

| module | GEMM | quant/cast | attention | conv | norm | elementwise | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAE encode | 0.2 | - | 0.4 | 4.4 | 2.6 | 2.2 | 9.8 |
| backbone double (5 blocks) | 6.8 | 0.6 | 0.2 | - | 2.6 | 1.3 | 11.5 |
| backbone single (20 blocks) + entry | 13.4 | 1.5 | 0.9 | - | 8.8 | 7.2 | 31.8 |
| ActionDiT double (5 blocks × 10 steps) | 4.8 | 0.6 | 1.2 | - | 2.7 | 1.3 | 10.6 |
| ActionDiT single (20 blocks × 10 steps) + enc/head/Euler | 14.6 | 1.8 | 4.7 | - | 9.7 | 5.5 | 36.3 |
| **total** | **39.7** | **4.5** | **7.6** | **4.4** | **26.4** | **17.4** | **100** |

## GEMM achieved rates

FLOPs and weight bytes from the same layer shapes used in
`docs/imagewam_rtx5090_profile.md` (all 20 single-stream layers counted
as full layers). Weight bytes per parameter: fp16 2, fp8 1, nvfp4 0.5625.
No Thor peak is available, so there is no "% of GEMM peak" / "% of DRAM"
column.

| module | precision | GEMM ms | TFLOPS | weight GB/s |
|---|---|---:|---:|---:|
| backbone double | fp16 | 7.54 | 68 | 325 |
| backbone double | fp8 | 6.22 | 82 | 197 |
| backbone double | nvfp4 | 6.08 | 84 | 113 |
| backbone single | fp16 | 18.12 | 113 | 271 |
| backbone single | fp8 | 10.90 | 188 | 225 |
| backbone single | nvfp4 | 12.02 | 170 | 115 |
| ActionDiT double | fp16 | 7.55 | 21 | 333 |
| ActionDiT double | fp8 | 3.08 | 53 | 408 |
| ActionDiT double | nvfp4 | 4.32 | 38 | 164 |
| ActionDiT single | fp16 | 33.25 | 19 | 303 |
| ActionDiT single | fp8 | 15.11 | 43 | 333 |
| ActionDiT single | nvfp4 | 13.09 | 49 | 216 |

## Bottleneck per part

Only claims that the tables themselves support. No Thor roofline.

| part | share of eager fp16 / fp8 / nvfp4 | what the tables show |
|---|---|---|
| ActionDiT single-stream GEMM | 30% / 20% / 15% | Largest GEMM cell in every precision; M = 64, 10 steps |
| backbone single-stream GEMM | 17% / 15% / 13% | Second GEMM cell; M = 416 |
| norm | 18% / 22% / 26% | `rms_norm_kernel` (default, `vec_rms_norm` off). 15.4–18.9 ms of QK RMSNorm in the `--names` dump. This is the kernel OPT-032 candidate 5 replaces |
| elementwise | 12% / 15% / 17% | Launch-heavy: 700+ `at::native` copies plus SiLU-GLU / RoPE |
| attention (FA4) | 6% / 6% / 8% | FA4, not the 5090 cuBLAS QK^T+softmax+PV chain |
| VAE encode | 8% / 8% / 10% | conv 2.6–4.0 ms of 5.8–8.8 ms; precision-independent except the flagged fp8 cell |
| nvfp4 quant/cast | – / – / 4.5% | 4.04 ms, mostly `kernel_quantize_fp4_sfa_vec` |
| fp8 absmax | – / 3% / – | `absmax_kernel`, 345 launches |

Fastest served tier on this GPU for this configuration: `nvfp4`
(`infer()` P50 106.05 ms). That ranking is wall time, not the eager
tables.

## Same-cell comparison to RTX 5090

5090 numbers from `docs/imagewam_rtx5090_profile.md` (eager ms and % of
that machine's eager total). Thor numbers from the tables above. Wall
time is `infer()` P50. Configuration differs only where the default
profile does: Thor FA4 on, 5090 cuBLAS attention; both have VAE in graph,
`text_trim`, `x0` = 24, `vec_rms_norm` off.

### Wall time (ms)

| precision | Thor `infer()` | 5090 `infer()` | Thor / 5090 |
|---|---:|---:|---:|
| fp16 | 167.16 | 36.14 | 4.6× |
| fp8 | 137.77 | 27.34 | 5.0× |
| nvfp4 | 106.05 | 31.68 | 3.3× |

5090's fastest tier is fp8; Thor's is nvfp4.

### Eager GEMM ms (same module)

| module | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| backbone double | 7.54 | 3.50 | 6.22 | 1.41 | 6.08 | 1.10 |
| backbone single | 18.12 | 10.39 | 10.90 | 3.39 | 12.02 | 2.32 |
| ActionDiT double | 7.55 | 2.19 | 3.08 | 1.42 | 4.32 | 2.38 |
| ActionDiT single | 33.25 | 7.69 | 15.11 | 4.55 | 13.09 | 7.48 |

### Eager norm and attention ms

| part | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| norm total | 19.80 | 3.73 | 16.05 | 3.77 | 23.70 | 3.75 |
| attention total | 6.09 | 3.38 | 4.60 | 3.39 | 6.78 | 3.40 |
| VAE encode total | 8.74 | 3.10 | 5.83 | 3.10 | 8.78 | 3.11 |

Thor default QK RMSNorm is ~5–6× the 5090 norm time at the same
`vec_rms_norm=off`. Attention is FA4 vs cuBLAS, so the ms are not the
same operator.

### Share of eager total, GEMM and norm (%)

| cell | fp16 Thor | fp16 5090 | fp8 Thor | fp8 5090 | nvfp4 Thor | nvfp4 5090 |
|---|---:|---:|---:|---:|---:|---:|
| GEMM total | 60.8 | 65.1 | 47.7 | 41.6 | 39.7 | 41.8 |
| norm total | 18.1 | 10.2 | 21.6 | 14.4 | 26.4 | 11.7 |
| ActionDiT single GEMM | 30.3 | 21.1 | 20.3 | 17.6 | 14.6 | 23.5 |
| backbone single GEMM | 16.5 | 28.4 | 14.7 | 13.1 | 13.4 | 7.3 |

## Conclusions

1. Served wall time on Thor, LIBERO config, default profile: nvfp4 106.05
   ms, fp8 137.77 ms, fp16 167.16 ms. nvfp4 is the fastest of the three
   (5090's fastest was fp8).
2. Graph-replay kernel time matches `infer()` P50 to a few milliseconds.
   Eager module attribution does not: it is 16–64 ms short. Tables are
   eager; do not treat their % as % of wall.
3. With `vec_rms_norm` off, norm is 18–26% of eager time (16–24 ms), vs
   ~3.7 ms / 10–14% on 5090 at the same flag. The kernel is
   `rms_norm_kernel` (560-class launch count in the names dump).
4. FA4 is on by default on Thor and was left on. Attention is 4.6–6.8 ms
   eager vs 3.4 ms cuBLAS on 5090 — a different operator, not a slower
   instance of the same one.
5. NVFP4 activation quantization on Thor is `kernel_quantize_fp4_sfa_vec`
   (4.04 ms), not the 5090 `cast_fp16_to_bf16` + `quantize_bf16_to_nvfp4`
   + `cast_bf16_to_fp16` sequence (5.98 ms).
6. Defaults that match 5090: `text_trim`, VAE in graph, `x0` = 24,
   `vec_rms_norm` off. Default that does not: FA4.
