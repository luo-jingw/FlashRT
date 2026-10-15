# ImageWAM on RTX 5090: one ActionDiT step at M = 64k action rows

Time of one ActionDiT denoise step when k denoise timesteps share one
forward (GEMM M = 64k rows, k = 1…10), for `fp16`, `fp8` and `nvfp4`. This
is the per-sweep cost of parallel-in-time sampling (Picard iteration /
ParaDiGMS) over a window of k steps; the serial sampler pays 10 × t(k = 1).

## Configuration

- Weights: `ImageWAM-FLUX.2-4B-LIBERO`, built with `load_imagewam` (default
  profile: `text_trim`, `vec_rms_norm`, `fuse_qkv_norm_rope`, cuBLAS
  attention chain), with `ImageWAMWorkload.libero()` except
  `action_horizon = 64k` and the checkpoint structure's
  `max_action_horizon` raised to 64k.
- Input: LIBERO-10 task 0 with the official prompt template
  (`x0` = 40, `a0` = 432), episode 11 frame 0, initial noise
  `0.01 · N(0, 1)` with seed 0. Prefill runs once before the step is timed.
- Machine: RTX 5090, driver 595.91.07, torch 2.14.0+cu130, FlashRT
  `3d3c1f5`, SM clock not locked, GPU not shared.
- Script: `benchmarks/imagewam_batched_m_step_bench.py PRECISION 1,…,10 OUT`.

## Method

- Step time: `imagewam_denoise_step` (step 0) captured as one CUDA graph,
  CUDA-event P50 over 200 replays after 20 warmup replays. Cross-check at
  k = 1: the 10-step loop graph is 14.55 / 13.85 / 18.76 ms against
  10 × step 14.56 / 13.75 / 18.76 ms (fp16 / fp8 / nvfp4).
- Split: the same step run eagerly under `torch.profiler` (5 iterations
  after a throwaway session), kernels classified as in
  `benchmarks/imagewam_module_operator_profile.py`; every kernel inside
  `attn.run` counts as attention (QKᵀ GEMM + softmax + PV GEMM). Eager
  kernel totals match the graph step time to within 0.1 ms.
- Kernels per step are equal across the profiled iterations of every cell:
  fp16 274, fp8 514, nvfp4 424, except fp16 k = 7 (269) and fp8 k = 7, 9
  (509).
- The 64k rows run as one sequence: every action row attends to the
  prefix and to all 64k action rows, and all rows share one AdaLN
  modulation. A batched step over k timesteps attends to the prefix and
  its own 64 rows and takes one modulation per 64-row group. The
  attention column is therefore this layout's cost; `k × attn(k=1)` is the
  cost of k separate 64-row attentions.

## Results (ms per step)

| precision | k | M | step | step / step(k=1) | attention | k × attn(k=1) | GEMM | quant/cast | norm | elementwise | other | step − attention | (step − attention) / k=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 1 | 64 | 1.456 | 1.00 | 0.27 | 0.27 | 0.99 | – | 0.13 | 0.07 | 0.00 | 1.19 | 1.00 |
| fp16 | 2 | 128 | 1.806 | 1.24 | 0.39 | 0.53 | 1.11 | – | 0.14 | 0.09 | 0.00 | 1.42 | 1.19 |
| fp16 | 3 | 192 | 2.344 | 1.61 | 0.49 | 0.80 | 1.55 | – | 0.16 | 0.10 | 0.00 | 1.85 | 1.56 |
| fp16 | 4 | 256 | 2.957 | 2.03 | 0.67 | 1.07 | 1.96 | – | 0.18 | 0.12 | 0.00 | 2.29 | 1.92 |
| fp16 | 5 | 320 | 3.552 | 2.44 | 0.75 | 1.34 | 2.39 | – | 0.21 | 0.14 | 0.00 | 2.80 | 2.36 |
| fp16 | 6 | 384 | 3.950 | 2.71 | 0.96 | 1.60 | 2.52 | – | 0.24 | 0.16 | 0.00 | 2.99 | 2.52 |
| fp16 | 7 | 448 | 4.823 | 3.31 | 1.42 | 1.87 | 2.96 | – | 0.24 | 0.18 | 0.00 | 3.40 | 2.86 |
| fp16 | 8 | 512 | 5.616 | 3.86 | 1.53 | 2.14 | 3.61 | – | 0.26 | 0.20 | 0.00 | 4.09 | 3.44 |
| fp16 | 9 | 576 | 6.058 | 4.16 | 1.63 | 2.41 | 3.84 | – | 0.29 | 0.22 | 0.00 | 4.43 | 3.73 |
| fp16 | 10 | 640 | 6.696 | 4.60 | 2.08 | 2.67 | 4.03 | – | 0.30 | 0.24 | 0.00 | 4.62 | 3.88 |
| fp8 | 1 | 64 | 1.375 | 1.00 | 0.27 | 0.27 | 0.60 | 0.06 | 0.13 | 0.12 | 0.09 | 1.11 | 1.00 |
| fp8 | 2 | 128 | 1.662 | 1.21 | 0.39 | 0.54 | 0.71 | 0.07 | 0.15 | 0.13 | 0.11 | 1.27 | 1.15 |
| fp8 | 3 | 192 | 1.862 | 1.35 | 0.49 | 0.81 | 0.79 | 0.08 | 0.16 | 0.15 | 0.12 | 1.37 | 1.24 |
| fp8 | 4 | 256 | 2.152 | 1.56 | 0.67 | 1.08 | 0.85 | 0.10 | 0.18 | 0.17 | 0.13 | 1.48 | 1.34 |
| fp8 | 5 | 320 | 2.521 | 1.83 | 0.75 | 1.35 | 1.06 | 0.11 | 0.20 | 0.19 | 0.14 | 1.77 | 1.60 |
| fp8 | 6 | 384 | 2.845 | 2.07 | 0.96 | 1.62 | 1.09 | 0.12 | 0.22 | 0.21 | 0.15 | 1.88 | 1.70 |
| fp8 | 7 | 448 | 3.530 | 2.57 | 1.43 | 1.89 | 1.28 | 0.14 | 0.23 | 0.23 | 0.15 | 2.10 | 1.90 |
| fp8 | 8 | 512 | 3.772 | 2.74 | 1.53 | 2.16 | 1.35 | 0.15 | 0.25 | 0.24 | 0.17 | 2.24 | 2.03 |
| fp8 | 9 | 576 | 4.061 | 2.95 | 1.65 | 2.43 | 1.41 | 0.16 | 0.28 | 0.26 | 0.17 | 2.42 | 2.18 |
| fp8 | 10 | 640 | 4.793 | 3.48 | 2.10 | 2.70 | 1.69 | 0.18 | 0.29 | 0.28 | 0.18 | 2.70 | 2.44 |
| nvfp4 | 1 | 64 | 1.876 | 1.00 | 0.27 | 0.27 | 1.00 | 0.43 | 0.13 | 0.07 | 0.00 | 1.61 | 1.00 |
| nvfp4 | 2 | 128 | 2.225 | 1.19 | 0.39 | 0.54 | 0.99 | 0.50 | 0.14 | 0.09 | 0.00 | 1.84 | 1.14 |
| nvfp4 | 3 | 192 | 2.534 | 1.35 | 0.49 | 0.81 | 1.07 | 0.61 | 0.16 | 0.10 | 0.00 | 2.04 | 1.27 |
| nvfp4 | 4 | 256 | 2.787 | 1.49 | 0.67 | 1.08 | 1.03 | 0.69 | 0.17 | 0.12 | 0.00 | 2.12 | 1.32 |
| nvfp4 | 5 | 320 | 3.094 | 1.65 | 0.75 | 1.34 | 1.12 | 0.77 | 0.20 | 0.14 | 0.00 | 2.34 | 1.46 |
| nvfp4 | 6 | 384 | 3.443 | 1.83 | 0.96 | 1.61 | 1.12 | 0.89 | 0.21 | 0.16 | 0.00 | 2.48 | 1.54 |
| nvfp4 | 7 | 448 | 4.093 | 2.18 | 1.42 | 1.88 | 1.20 | 0.97 | 0.23 | 0.18 | 0.00 | 2.67 | 1.66 |
| nvfp4 | 8 | 512 | 4.371 | 2.33 | 1.53 | 2.15 | 1.21 | 1.10 | 0.26 | 0.20 | 0.00 | 2.84 | 1.77 |
| nvfp4 | 9 | 576 | 4.667 | 2.49 | 1.63 | 2.42 | 1.24 | 1.19 | 0.28 | 0.21 | 0.00 | 3.04 | 1.89 |
| nvfp4 | 10 | 640 | 5.239 | 2.79 | 2.09 | 2.69 | 1.25 | 1.27 | 0.29 | 0.23 | 0.00 | 3.15 | 1.96 |

The split columns are eager-run times; "step − attention" uses the graph
step time.

## Break-even sweep count

A Picard window of k = 10 covering all steps beats the serial sampler of
the same precision when sweeps × t(k=10) < 10 × t(k=1):

| precision | 10 × t(k=1) | t(k=10) | break-even sweeps | break-even sweeps against serial fp8 (13.75 ms) |
|---|---:|---:|---:|---:|
| fp16 | 14.56 | 6.70 | 2.2 | 2.1 |
| fp8 | 13.75 | 4.79 | 2.9 | 2.9 |
| nvfp4 | 18.76 | 5.24 | 3.6 | 2.6 |

## Observations

1. The step does not stay flat in M: t(640) / t(64) is 4.60 (fp16), 3.48
   (fp8), 2.79 (nvfp4). Per evaluated timestep, a k = 10 step costs 0.67 /
   0.48 / 0.52 ms against 1.46 / 1.38 / 1.88 ms at k = 1.
2. GEMM growth from k = 1 to 10: fp16 4.1× (0.99 → 4.03 ms), fp8 2.8×
   (0.60 → 1.69 ms), nvfp4 1.25× (1.00 → 1.25 ms). The NVFP4 GEMM is close to
   flat over M = 64…640; its activation quantization and casts grow 3.0×
   (0.43 → 1.27 ms) and equal the GEMM time at k = 10.
3. Attention grows 7.8× (0.27 → 2.1 ms in every precision) and is the
   largest single part at k = 10 in fp8 and nvfp4. In this layout the
   attention is below k × attn(k=1) for every k ≥ 2.
4. Norm and elementwise kernels grow 2.2–3.4× from k = 1 to 10; their
   launch count does not change with M.
EOF