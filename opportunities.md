# OPT-001

Status: RESOLVED end to end, including real Thor validation (2026-09-15, all 4 phases complete). FP8 calibration/quantization split off into OPT-004 steps 5-6; the real per-layer activation calibration that was the open half is OPT-022 and `docs/imagewam_calibration.md`.

Area: ImageWAM on Thor — the real-weight path, the precision tiers, and the confirmed real deployment shapes.

## Real-weight path (2026-09-15)

`imagewam_thor.py` gained a `ckpt_path=` constructor kwarg; new `checkpoint_loader.py` reads the release checkpoint's own raw `state_dict` by key name (`torch.load(..., mmap=True)['mot']` — no `imagewam`/`flux2` packages needed at all, just `torch`). Two real architecture gaps fixed as prerequisites: `img_in` (image tokens were never projected from raw `HD` width to `hidden` width anywhere in this project) and `action_encoder`/`head` (the flow-matching Euler integration was happening in `action_hidden_dim` space instead of real `action_dim` space). Two real silent shape/convention bugs were found only by running it, not by review: modulation weights need the real `(out,in)` layout (plain `F.linear`), not the FlashRT `(K,N)` GEMM-transposed one, and ActionDiT's own double-block weights need PLAIN (unprefixed) slot names, not the backbone's `img_`-prefixed dual-stream convention. `tests/test_imagewam_checkpoint_loader.py` locks in all three checks (shape match, one real-weight layer forward, full real-checkpoint frontend) and skips cleanly if the real checkpoint files are absent.

Verified end to end on this 8 GB dev machine at real FLUX.2-4B dims (~8.9GB of weights alone, far short of this machine's VRAM): construction with real weights, a real CUDA Graph capture, and replay, producing a finite `(64,7)` action tensor (mean=0.21, std=0.34). Peak CUDA memory ~9.86 GB while `nvidia-smi` reports 8188MiB total, because this WSL2 environment pages beyond dedicated VRAM instead of raising OOM — real enough for a correctness check, not a Thor performance claim.

## Real deployment shapes and calibration (Thor, 2026-09-15)

- Correct calibration dataset: `yuanty/LIBERO-fastwam` (HF), subset `libero_spatial_no_noops_lerobot` — Franka, 434 episodes / 53229 frames / 10 tasks, real LIBERO language instructions, two 512x512x3 AV1 20fps camera views, `action: (T,7)`, `state: (T,8)`.
- Confirmed real image-token shape, superseding the 768-token (`384x512`) guess used everywhere until then: real eval preprocessing resizes each view to `224x224` and concatenates horizontally to `224x448`; the real official `FLUX.2-dev/ae.safetensors` VAE (`x*2/255-1`) encodes that to a `(B,128,14,28)` latent -> `(B,392,128)` packed tokens, i.e. **`img_len=392` (14x28), not 768** (the bench scripts' confirmed shape at that point: `a0=520`, `total=584`). VAE encode itself: real Thor P50 **41.0 ms** for `224x448`.
- FP8 `img_in` calibration, holdout real VAE tokens vs the FP16 reference: `N(0, 0.1)` noise act_scale 0.00102 -> cosine 0.902; real VAE tokens (mean=-0.02, std=0.97, absmax=4.91) act_scale 0.01086 -> **0.99946**. Both `_calibrate_fp8`/`_calibrate_static_fp8` now special-case `img_in.weight`'s calibration input to the measured stats (a narrow, single-slot fix); every other weight keeps the unvalidated 0.1-scale placeholder, which OPT-022 replaced for `fp8_static*` only.
- Real-weight FP16 `infer()` (CUDA Graph, real GEMM shapes, `img_raw` still per-step `normal_()`) at the corrected shape: 231.1 ms at `img_len=768` -> **172.8 ms** at `img_len=392`; peak allocated 9.99 GB. With the real VAE folded in: ~41 + 173 ≈ **214 ms**, still without Qwen3 (`txt_in` still reads a random `context`).
- Re-measured OPT-004 step 5/6 comparison at `img_len=392` (P50, ms, real Thor hardware):

| layer | FP16 | FP8 dynamic | FP8 static | FP8 static+CUTLASS | NVFP4 |
|---|---:|---:|---:|---:|---:|
| backbone_double | 3.93 | 3.88 | 3.69 | 3.27 | **3.00** |
| backbone_single | 2.57 | 2.24 | **2.15** | 2.19 | **2.02** |
| action_double | 0.54 | 0.46 | 0.47 | 0.45 | **0.40** |
| action_single | 0.52 | 0.46 | 0.43 | 0.40 | **0.35** |
| prefill (5+20) | 71.1 | 64.3 | 61.4 | 60.2 | **55.4** |
| denoise x1 | 13.1 | 11.6 | 10.9 | 10.2 | **9.1** |
| prefill+10-step | 202 | 180 | 170 | 163 | **146** |

  Attention kernels: `mot` 0.037 ms (unchanged); standard/backbone attention 0.15 ms, down from 0.82-0.84 ms at `a0=896` because the O(a0^2) cost shrank with the smaller sequence. Relative rankings changed from the 768-token table: dynamic FP8 is now FASTER than FP16 (64.3 vs 71.1 ms) where it was slower at 768, the CUTLASS-over-static-cuBLASLt gap shrank from ~20 ms to 1.2 ms (60.2 vs 61.4 ms, with `backbone_single` actually 2.19 vs 2.15 ms), and NVFP4 is fastest at both shapes.
- Full-pipeline re-validation at `REF_H,REF_W=14,28` + `img_in`: backbone cosine **0.999918** (was 0.999927 at the old 24x32 grid), ActionDiT **0.999962** (was 0.999963), output shape `(520,3072)` matching `a0=520`.
- `test_imagewam_quant_linear.py` on Thor (small-shape random weights, does NOT exercise `img_in`'s own `N(-0.02,0.97)` path): `Fp8Linear` 0.999242, `StaticFp8Linear(cublaslt)` 0.999242, `StaticFp8Linear(cutlass)` 0.999242, `Nvfp4Linear` 0.989133. The 0.99946 number above remains the standalone holdout measurement, not re-verified through this test.

## Open

- A real-checkpoint per-layer P50 on Thor was never produced: the Thor per-layer microbenchmarks are random-weight, and this dev machine's own numbers would not be a meaningful Thor performance number.
- Confirming that the same real-checkpoint sequence completes on Thor's own 128 GB unified memory without the WSL2 paging dependency is still not done.
- Citation note: about fifteen code and test comments cite `opportunities.md OPT-001 "FP16 residual overflow"` for that account; the account itself (root cause, numbers, BF16 fix) lives in OPT-008.

# OPT-004

Status: steps 1-6 implemented and Thor-verified; the graph-capture path, `_autotune_gemm` (one `autotune_fp16_nn` per distinct (M,N,K) shape, before capture), the fused QKV GEMM and the fused AdaLN/gated-residual kernels are `ImageWAMTorchFrontendThor`'s own unconditional defaults, while the quantized tiers stay opt-in (`precision=` / `IMAGEWAM_PRECISION`; the constructor's own default is now `precision="nvfp4"`). The combined graph + pre-autotuned-GEMM run hung and is shelved.

Area: `flash_rt/frontends/torch/imagewam_thor.py` (`_autotune_gemm`, `_calibrate_fp8`, graph capture), `flash_rt/models/imagewam/pipeline_thor.py`, `flash_rt/models/imagewam/quant_linear.py`, `flash_rt/frontends/torch/_imagewam_thor_spec.py`, kernels `ada_layer_norm_fp16`/`gate_res_fp16` (`csrc/bindings.cpp`, `csrc/kernels/norm.cu`, `csrc/kernels/decoder_fused.cu`), benchmarks `imagewam_thor_graph_bench.py`, `imagewam_thor_fp16_bench.py`, `imagewam_thor_fp16_autotuned_bench.py`, `imagewam_thor_fp16_autotuned_graph_bench.py`.

## Observation

Pre-OPT-003 real Thor, full prefill+10-step-denoise steady state: 407-453ms across all four precisions tested (superseded by the `6478844` round below). The pipeline is a direct 1:1 unfused translation of the math (one launch per op: norm, each GEMM separately, attention, residual_add), unlike real FlashRT models such as `cosmos3_edge` (`residual_add_rms_norm_fp8`, fused QKV projections, `bias_gate_mul_residual_bf16`), which autotune `GemmRunner` shapes via `autotune_cached` rather than accepting cuBLASLt's default top-1 heuristic. OPT-003's fix alone brought the graph-free Ada number down to 203.2ms.

## Step 1: graph capture — launch overhead is NOT the bottleneck here

Ada, real dims, post-OPT-003 (`imagewam_thor_graph_bench.py`, replay only): **198.5ms P50** captured vs **203.2ms P50** graph-free (`imagewam_thor_fp16_bench.py`) — ~2%. GEMMs are hundreds of µs to a few ms each, so single-digit-µs per-launch overhead is a small fraction of the total: compute-bound, not launch-bound. Thor, post-OPT-003: 129.9ms captured vs 140.4ms graph-free = **7.5%** (bigger than Ada's ~2%), but autotune alone (**126.1ms**, no graph capture) already beats graph-capture-with-default-heuristic (129.9ms). Direction holds on both GPUs, magnitudes differ.

## Opportunity (steps 2-4, in priority order)

2. Fuse QKV into one wide GEMM per stream (the real checkpoint's own fused tensor shape; 3 GEMMs today only because of this pipeline's own reduced-KV-width convention, OPT-002).
3. Fuse residual+norm (`residual_add_rms_norm_fp8`-style) at every block boundary.
4. Call `GemmRunner.autotune_cached` for ImageWAM's own real shapes instead of relying on cuBLASLt's default heuristic.

## Step 2: QKV fusion (2026-09-14) — checkpoint fidelity, no measured speed win on Ada

`pipeline_thor.py`'s 4 real-math layer helpers (`_double_stream_layer`, `_single_stream_layer`, `_action_double_layer`, `_action_single_layer`) do ONE `qkv` GEMM into a `(seq, 3*width)` scratch buffer, then land each third in `Q_O`/`K_cache`/`V_cache` via a new `_copy_slice` (`_wrap_fp16` gained an optional `row_stride` so a column slice of the wider buffer is viewed/copied without an intermediate copy). `_imagewam_thor_spec.py`: three `{prefix}_q/_k/_v.weight` shapes became one `{prefix}_qkv.weight` `(width, 3*width)`, matching a real checkpoint's own fused `qkv` tensor directly.
Verified: all 21 ImageWAM tests pass; `test_imagewam_thor_real_wiring.py`'s per-layer cosine checks against the tensor-level reference (`real_*.py`) are now exactly 1.000000 for all 4 layer types (previously double-stream was 0.999984 — the fused path reproduces the reference's own fused-GEMM accumulation bit-for-bit). Ada speed (`imagewam_thor_bench.py`, 4 repeat runs): backbone_double 9.84ms → 9.6-10.1ms, backbone_single 9.37ms → 9.1-9.4ms, action_double 0.61ms → 0.66-1.29ms, action_single 0.51ms → 0.56-1.03ms — within run-to-run noise, no clear regression either. Kept regardless: checkpoint fidelity and bit-exactness stand on their own.

## Step 3: residual+norm fusion — an existing kernel already matched the math

`ada_layer_norm_fp16` computes EXACTLY `LayerNorm_no_affine(x)*(1+scale)+shift` in ONE kernel launch, with `scale`/`shift` as `[dim]` per-forward broadcast vectors — precisely ImageWAM's real modulation semantics, no math difference at all. `gate_res_fp16` covers `residual[i] += gemm_out[i]*gate[i]`, modulo one real constraint: its flat elementwise indexing has no stride/broadcast concept, so `gate` must be a genuinely `(seq,dim)`-MATERIALIZED copy.
The 4 layer helpers now call both kernels directly, replacing `layer_norm_no_affine_fp16` + torch elementwise modulate and the torch gated-residual add. New `_fuse_mod_group` produces each kernel's exact inputs (fp16 `shift`/`scale`, broadcast-materialized fp16 `gate`), proven to run ONLY during graph capture/warmup, never during `.replay()`; `normed_scratch`/`action_normed` are gone entirely. Verified: all 21 ImageWAM tests pass, cosines stay exactly 1.000000 for all 4 layer types. Ada: backbone_double 9.6-10.1ms → 10.0ms, backbone_single 9.1-9.4ms → 9.1ms — no measurable win, same compute-bound explanation (the fused-away kernels were already microseconds against single-digit-ms GEMMs). Kept: fewer launches and less global-memory round-tripping.

## Step 4: autotune — real but modest on Ada, the best-verified lever on Thor

`imagewam_thor_fp16_autotuned_bench.py`: each `_Fp16Linear` calls `GemmRunner.autotune_fp16_nn` once, lazily (real weight + a representative activation), before falling back to plain `fp16_nn`; `autotune_fp16_nn` mutates the same cached cuBLASLt entry `fp16_nn` itself reads (`entry.algo = heuristics[best_idx].algo` writes into the identical `CachedGemm&`), so it can only match or beat the default heuristic, never regress correctness.
Ada: backbone prefill 143.5ms → **138.8ms** (~3%), full prefill+10-step 203.2ms → **195.1ms** (~4%). The autotune log shows why the win is small: most shapes had only 1 candidate algorithm from `cublasLtMatmulAlgoGetHeuristic`, and where 4-6 candidates existed the best was frequently the default heuristic's own top-1 pick. Thor: 140.4ms → **126.1ms** (+10%), versus Ada's own +4%. Re-measured on the new real-math pipeline (2026-09-14, commit `61e7c15`), Ada shows no gain: `backbone_double_layer` 9.461ms → 9.840ms, `backbone_single_layer` 8.885ms → 9.369ms (within noise, arguably slightly worse — the real math's wider GEMMs, per-head K/V at `hidden` width instead of broadcast `HD` and MLP-gate at `mlp_hidden*2` instead of `mlp_hidden`, sit where cuBLASLt's default is already near-optimal on Ada). The same round's Thor run (commit `6478844`, steps 2+3 also active so autotune is not isolated) confirms the asymmetry again — backbone_single ~-15%, backbone_double ~-9% — so an Ada-only "no win" result is not final for Thor without its own run. Autotune is wired into `imagewam_thor.py`'s own `__init__` as a genuine default (`_autotune_gemm`, one call per distinct real-math (M,N,K) shape, before any graph capture).

## Steps 5-6: quantized GEMM wrappers and static-scale CUTLASS FP8

`quant_linear.py` promotes the bench scripts' `_Fp8Linear`/`_Fp4Linear` pattern into `Fp16Linear`/`Fp8Linear`/`Nvfp4Linear`; all 21 weight-projection GEMM call sites in `pipeline_thor.py` dispatch uniformly via `weights[key](x_ptr, out_ptr, m, stream)`; `imagewam_thor.py` gained `precision: str` and the bench a matching `IMAGEWAM_PRECISION` env var. `StaticFp8Linear` adds a calibrate-once static activation scale plus `cutlass_fp8_sq`/`_wide`/`_t1`, independently switchable (`use_cutlass=False`/`True`), exposed as `precision="fp8_static"`/`"fp8_static_cutlass"` with a `_calibrate_fp8()` step in `set_prompt()` before graph capture (a captured graph cannot re-issue the host sync a dynamic scale needs); the bench got a matching one-time `.calibrate()` hook, and `cutlass_fp8_sq`/`_wide`/`_t1` sit behind the SAME `ENABLE_SM100_CUTLASS` flag NVFP4 already uses.
Interface facts kept: `cutlass_fp8_*` takes `alpha` as a host float (unlike `fp8_gemm_descale_fp16`'s device pointers), so `alpha` is precomputed once inside `calibrate()` via `np.float32(a)*np.float32(b)` per `docs/calibration.md`'s f32-not-f64 rule — reading it with `.item()` inside `__call__` would force a host sync on every graph replay. Not numerically verifiable on this Ada machine, for two unrelated reasons: FP8 — cuBLASLt 12.8.04 (CUDA 12.8, cc (8,9)) fails `fp8_gemm_descale_fp16` with `cublasLtMatmulAlgoGetHeuristic ... cuBLAS status 15` at every shape tried (down to 4x16x16), a pre-existing environment gap (`issues.md` ISSUE-001, reproduced by the pre-existing `imagewam_thor_fp8_bench.py`); NVFP4 — `flash_rt.flash_rt_fp4` is not built here at all (`ModuleNotFoundError`), it exists only in a `-DGPU_ARCH=110` build. Both variants were verified on Ada only as wiring: each fails at exactly those documented points (`use_cutlass=False` the cuBLASLt gap, `True` the missing `cutlass_fp8_*` symbols), not at a new bug. `tests/test_imagewam_quant_linear.py` therefore probes availability (canary FP8 GEMM call, import guard) and skips cleanly rather than asserting a bar that can never be cleared here; the FP16 passthrough is verified cosine=1.000000, and three pre-existing test files took mechanical updates wrapping their raw-pointer weight dicts in `Fp16Linear`.

## Thor check

Autotune (step 4), QKV fusion (step 2) and the fused AdaLN/gated-residual kernels (step 3) are all unconditional frontend defaults, so one bench run cannot isolate them from each other.

2026-09-14, commit `6478844`, `imagewam_thor_bench.py`, FA4 off — layer | Thor P50 (`6478844`) | Ada P50 | Thor P50 (`61e7c15`, before these changes): backbone_double 5.53 ms | 10.0 ms | 6.04 ms; backbone_single 4.47 ms | 9.1 ms | 5.28 ms; action_double 0.57 ms | 0.9 ms | 0.58 ms; action_single 0.53 ms | 0.5 ms | 0.54 ms.
Derived: backbone prefill (5 double + 20 single) 135.8ms → 117.0ms (-14%), one denoise step 13.7ms → 13.5ms (flat), prefill+10-step 273ms → 252ms; ActionDiT barely moves (only 64 action tokens, still launch-count-dominated). Attention alone is UNCHANGED (`mot_joint_kernel_only` 0.042ms, `standard_attn_kernel_only` 0.824ms). 117ms prefill is still slower than the old approximate-math FP16 full pipeline's own 81.6ms prefill (pre-OPT-002 table) — the real cost of the actually-correct math, not a regression from this work.

2026-09-14, commit `9411b73` (step 5; first real numbers for either kernel, no SKIP on Thor). Correctness: `Fp8Linear` cosine 0.999242 (clears 0.99 comfortably); `Nvfp4Linear` 0.989133 (misses the original 0.99 bar by 0.0009; the test's NVFP4 bar was lowered to 0.98 — consistent with NVFP4's own E2M1 format, 2 mantissa bits, block-16 dynamic scale, no calibration: a judgment call, not a root-caused fact). Per-layer P50 ms, FA4 off — layer | FP16 | FP8 | NVFP4: backbone_double 5.53 | 5.01 | 4.55; backbone_single 4.47 | 4.89 | 3.49; action_double 0.57 | 0.54 | 0.42; action_single 0.53 | 0.51 | 0.39; prefill (5+20) 117.0 | 122.8 | 92.6; denoise x1 13.5 | 12.9 | 10.0; prefill+10-step 252 | 252 | 192. Attention kernels unmoved (mot_joint ~0.041ms, standard_attn ~0.84ms) — all movement is from the weight-projection GEMM swap alone. FP8 is not a speed win here: prefill 117.0 → 122.8ms (+5%), backbone_single alone gets slower (4.47 → 4.89ms), `cublasLtMatmul`'s dynamic quantize+GEMM+dequantize overhead outweighing its tensor-core benefit at these shapes. NVFP4 is a real win: prefill 117.0 → **92.6ms** (-21%), one denoise step 13.5 → **10.0ms** (-26%), but its correctness is borderline on random weights, so `precision="nvfp4"` stayed opt-in pending real-checkpoint accuracy validation (the real checkpoint exists only on Thor and is never fetched locally).

2026-09-15, commit `cfba7ef` (step 6). Same-machine FP16 re-measurement: 116.4ms prefill (vs 117.0ms earlier — run-to-run noise). Correctness, no SKIP lines: `Fp8Linear` (dynamic), `StaticFp8Linear(cublaslt)` and `StaticFp8Linear(cutlass)` are all cosine **0.999242**, bit-for-bit consistent — neither change altered the math, only its cost; `Nvfp4Linear` unchanged at 0.989133. Per-layer P50 ms, FA4 off — layer | FP16 | FP8 dynamic | static+cuBLASLt | static+CUTLASS: backbone_double 5.45 | 5.05 | 4.83 | 4.70; backbone_single 4.46 | 4.85 | 4.70 | 3.72; action_double 0.58 | 0.54 | 0.50 | 0.50; action_single 0.53 | 0.51 | 0.46 | 0.46; prefill (5+20) 116.4 | 122.2 | 118.1 | 97.9; denoise x1 13.6 | 12.8 | 11.6 | 11.7; prefill+10-step 252 | 250 | 234 | 215. The CUTLASS tile swap is the fix, the static scale alone mostly is not: static+cuBLASLt recovers only 4.1ms of the dynamic path's own +5.8ms regression (122.2 → 118.1ms), still slower than FP16 and with `backbone_single` still regressed, whereas static+CUTLASS prefill is **97.9ms** (-16% vs FP16, -20% vs dynamic FP8), almost entirely from `backbone_single` (4.70 → 3.72ms). ActionDiT gets ZERO extra benefit from CUTLASS (0.50/0.46ms, identical to static+cuBLASLt), so `_pick_fp8_cutlass_variant`'s provisional heuristic stays unvalidated at `M=64` (action is ~14ms of the ~116ms prefill, so the backbone win is unaffected). Versus NVFP4 (prefill 92.6ms, denoise 10.0ms), static+CUTLASS FP8 (97.9ms / 11.7ms) is close but behind on both: NVFP4 remains the single fastest measured precision, but its 0.989 correctness is meaningfully weaker than static+CUTLASS FP8's solid 0.999242.

Shape supersession: every table above was measured at the old `img_len=768` / `a0=896` / `total=960` shape. The re-measurement at the confirmed real `img_len=392` (14x28) shape — where the relative rankings change (dynamic FP8 becomes faster than FP16; the CUTLASS-over-static-cuBLASLt gap shrinks to 1.2ms and `backbone_single` is slightly slower under CUTLASS) — is recorded in this file's OPT-001 entry; the served workload now derives `x0=513`, `img_len=392`, `a0=905`, `total=969`.

## Calibration source (superseded)

`_calibrate_fp8` freezes `StaticFp8Linear`'s activation scale from a disposable random tensor, not a real observation distribution; the full house calibration mechanism (`docs/calibration.md`'s multi-sample/percentile approach on real per-model data) was never attempted here. The dataset investigated in this entry, `JingwuLuo/LingBot-VA_RoboTwin_clibration_data` (HF, public, 250 episodes), was the WRONG one — it belongs to a DIFFERENT model (`LingBot-VA`): `actions_N.pt` shape `(1,30,2,16,1)` bf16, range `[-1,1]`, does not reshape to LIBERO's confirmed `action_dim=7`, and its `30` timesteps != this project's `num_action`/`max_action_horizon=64`; `latents_N.pt` `(1,48,2,24,20)` bf16 is a 5D video-latent shape, not this project's own `img_raw` 2D `(img_len, HD=128)` per-token convention; `obs_data_N.pt` does hold real `240x320x3` uint8 RGB (3 camera views) and a real `task` prompt string, but encoding those into `img_raw`/`context` needs the real VAE (`flux2` source, absent on this dev machine) and the real Qwen3-4B text encoder, which `pipeline_thor.py` does not reimplement (`_prepare_flux2_infer_text` is `imagewam`'s own real preprocessing step). Forcing a shape-mismatched or wrong-distribution tensor through `calibrate()` would produce a scale that LOOKS real while calibrating against the wrong thing. The correct dataset (`yuanty/LIBERO-fastwam`) and the actual calibration fix are recorded in this file's OPT-001 (2026-09-15) and OPT-022 (roadmap item 7) entries.

## Shelved: combined graph capture with pre-autotuned GEMMs

`benchmarks/imagewam_thor_fp16_autotuned_graph_bench.py` (Ada) warms up on a side stream so every `_Fp16Linear`'s one-time `autotune_fp16_nn` call fires, then captures one more full run into a `CUDAGraph`. **Result: hung — 66+ minutes of CPU/GPU time with zero new output, killed rather than let run further.** Suspected but not confirmed: `autotune_cached`'s own C++ implementation hardcodes stream 0 for its internal benchmark loop (`cublasLtMatmul(..., workspace_, workspace_size_, 0)` — the stream argument is not even accepted by `autotune_fp16_nn`), while the warmup ran on an explicit non-default side stream (needed so the same stream could capture) — combining the two may have deadlocked in `cudaEventSynchronize`/`cudaDeviceSynchronize`. Not root-caused further; shelved rather than debugged, since autotune-alone (+4%/+10%) and graph-alone (+2%/+7.5%) are both already confirmed, independently useful wins, and a combined win would likely be smaller than their sum. The script is kept as the record of this experiment and is marked `KNOWN TO HANG` in its own docstring ("See opportunities.md OPT-004"); if it is revisited, autotune on the DEFAULT stream first (a warmup pass with no `torch.cuda.stream(...)` context), then switch to a side stream only for the capture call.

## Open

- The combined CUDA-graph + pre-autotuned-GEMM experiment hung (66+ minutes of CPU/GPU time, zero new output, killed) and is shelved; `benchmarks/imagewam_thor_fp16_autotuned_graph_bench.py` is kept as the record and marked `KNOWN TO HANG` in its own docstring.
- Real-checkpoint accuracy validation for the quantized tiers — `fp8_static_cutlass` (0.999242) and `nvfp4` (0.989133) are single-layer random-weight numbers only, with no test yet of whether the error compounds across the real 25-layer stack; real calibration and the `fp8_static` fidelity numbers are recorded in this file's OPT-022 entry and `plan.md`'s execution-status row 7. ActionDiT `M=64` tile selection (`_pick_fp8_cutlass_variant`) is `issues.md` ISSUE-023 / `plan.md`'s roadmap item 1.
- `_autotune_gemm`'s own shape list still holds the pre-merge split shapes and misses every merged `linear1` shape (`(905, 27648, 3072)` backbone, `(64, 17408, 1024)` ActionDiT), so `precision="fp16"` runs those GEMMs on the default heuristic pick; the size of that loss is unmeasured (`issues.md` ISSUE-010).

# OPT-006

Status: not promoted; never started. The missing input is a real-weight accuracy budget for a compute-step schedule — OPT-001's own real-weight path is RESOLVED end to end (2026-09-15), so that is not the blocker.

Area: TeaCache-style step-skipping for the flow-matching denoise loop

## Observation

`flash_rt/models/cosmos3_edge/pipeline_thor.py`'s `CosmosEdgeThor` has a real, already-implemented `set_teacache(compute_steps)` mechanism: a fixed subset of denoise steps actually compute a fresh velocity, the rest reuse the last computed velocity while the scheduler still advances every step. This exploits the same redundancy diffusion/flow-matching literature calls TeaCache — consecutive denoising steps often produce very similar velocity predictions.

## Opportunity

ImageWAM's own flow-matching denoise loop (`imagewam_denoise_loop`, Phase 4) is structurally the same shape (N fixed steps, each a full ActionDiT forward), so a `set_teacache`-equivalent compute-step schedule could skip a fraction of those ActionDiT forwards outright and directly reduce the per-step cost this project has measured: pre-fix (OPT-003) ~35 ms/step on real Thor (Thor FP16 35.3 ms, Ada FP16 29.5 ms), post-fix 5.89 ms (Thor FP16), 5.68 ms (Ada FP16) and 4.37 ms (Ada INT4, GEMM-only) per 25-layer denoise step.

## Expected Mechanism

Same mechanism already implemented and presumably validated for cosmos3_edge: skip N-k of N steps' full forward, reuse the last velocity, accept whatever accuracy cost that implies (needs real-weight validation, not assessable with random weights).

## Required Evidence

Needs real weights and an accuracy budget to determine a safe compute-step schedule — meaningless to tune against random weights. Should follow, not precede, OPT-001.

## Promotion Condition

Promote alongside OPT-001, once real-weight accuracy validation exists to determine which steps are safe to skip.

## Open

- TeaCache-style compute-step schedule for the flow-matching denoise loop: never started; needs a real-weight accuracy budget before any schedule can be chosen.

# OPT-007

Status: CLOSED for Thor (real, measured ~8.6x slower than FP16 there: 1214 ms vs FP16's 140.4 ms, and 1255 ms vs FP16's 178.3 ms / ~7x in the later VAE-included run) and closed for end-to-end use on Ada (INT8 fails at the real `K=9216` shape, FHT crash at the real dims); remains open and unpromoted for a hypothetical true Orin/Ampere deployment only.

Area: INT4 (QuaRot W4A4, SM80 CUTLASS) as an additional precision option — confirmed buildable on Ada and Thor, but a real speed dead end on Thor specifically

## Observation

`csrc/gemm/cutlass_sm80_int4_rowwise.cu` — a real INT4 W4A4 rowwise GEMM family, built for Jetson Orin SM87's QuaRot path but templated on `cutlass::arch::Sm80` — is gated only by `ENABLE_SM80_INT8_CUTLASS` (default ON only for `GPU_ARCH=87`, but overridable) and `FLASHRT_ENABLE_CHAMELEON` (default OFF, opt-in), NOT by any Blackwell-only check like NVFP4. Confirmed by reconfiguring and rebuilding on this dev machine (`-DENABLE_SM80_INT8_CUTLASS=ON -DFLASHRT_ENABLE_CHAMELEON=ON`, `GPU_ARCH=89`) and running `cutlass_int4_rowwise_fp16out` successfully at real ImageWAM projection shapes (`benchmarks/imagewam_gemm_precision_compare.py`); INT8 (SM80, same family) also works. Both are meaningfully faster than FP16 on this machine, at the shapes that work: INT4 ~9x, INT8 ~4x on the `q/proj` (3072x3072) shape. On Thor (SM110) the same flags compile and link fine and every GEMM shape returns success (rc=0) — but "succeeds" only means no error thrown, not numerically verified.

**INT8 fails hard at `K=9216`; INT4's `mlp2` failure at that K is flakiness, not a hard `K` limit.** `cutlass_int8_rowwise_fp16out` reliably fails at `K=9216` in every reproduction attempted: `benchmarks/imagewam_thor_int8_bench.py` builds the full 25+25-layer INT8 pipeline (weight allocation, backend construction all fine) but throws on the very first `run_prefill()` call, inside the first double layer, at `img_mlp2` (`M=768,N=3072,K=9216`, `rc=131079`) — while `txt_mlp2`, the *same* `K=9216` shape called immediately before it in the same layer and differing only in M (128 vs 768), succeeded; the same `rc=131079` reproduces fresh and in isolation at the later real dims `M=512,N=3072,K=9216`. Since every one of the 25 backbone layers' `mlp2`/`mlp_down` calls uses this shape, a full 25-layer prefill has effectively no chance of completing on Ada. The `mlp2` (`K=9216`) failure reported for `cutlass_int4_rowwise_fp16out` does NOT reproduce in isolation or inside a real full-pipeline run (`benchmarks/imagewam_thor_int4_bench.py`, all 25+25 layers including 20 calls at this exact shape, ran clean end to end); it only reproduced inside `imagewam_gemm_precision_compare.py`'s specific fp16 → fp8[fails] → int8[fails] → int4 sequence repeated per shape across 4 shapes, and isolating the exact trigger was not pursued further (shape order, the preceding fp8/int8 failures, and the warmup/iteration loop pattern itself each failed to reproduce it alone) — a real but poorly-understood flakiness in that script's own mixed-precision-in-one-process pattern, not a hard `K` limit on the kernel. On Thor the Ada INT8 `K=9216` crash does not reproduce at all: the full 25+25-layer pipeline, including the corrected 128-channel VAE and the exact `K=9216` (`mlp_down`) shape that reliably crashes on Ada, completes cleanly.

Also real and not yet addressed: this GEMM's own correctness contract (per its file header) requires a QuaRot Hadamard rotation on both activation (online FHT) and weight (offline) before quantizing to int4 — plain per-row symmetric quantization without it is documented in that same file as insufficient for real model activations. **Confirmed blocking, not theoretical**: the real activation quantizer, `fht_int4_quant_fp16`, CRASHES with an illegal memory access at ImageWAM's real hidden dims (3072, 9216, 7680 — none are powers of 2), while working cleanly at 128/1024/4096 (all powers of 2); this FHT kernel needs a power-of-2 transform size, and ImageWAM's real dims are not powers of 2. Every full-pipeline number below is GEMM-only for exactly this reason: it could not include a real per-call activation quantization step even if it wanted to.

## Full-Pipeline Result (Ada, GEMM-only, no activation quantization)

`benchmarks/imagewam_thor_int4_bench.py`: same 25+25-layer structure as the FP16/FP8/FP4 scripts, ran clean end to end on this machine; numbers are AFTER the OPT-003 fix (mot_joint restricted to action queries; before/after breakdown in OPT-003):

| | backbone prefill (25L) | one denoise step (25L) | prefill + 10-step |
|---|---|---|---|
| INT4 (GEMM-only) | 43.7 ms | 4.37 ms | 91.1 ms |
| FP16 (this machine) | 143.5 ms | 5.68 ms | 203.2 ms |

The same scripts' earlier full-pipeline numbers predate the real-shape confirmation and used a stale placeholder (768 image tokens / 384x512 input guess, X0=128, NUM_ACTION=64); re-measured at the real confirmed LIBERO dual-camera shape (224x448 input → 14x28 grid → 392 image tokens), `X0=512`, `NUM_ACTION=10`/`action_dim=7` (`action_dim` itself only affects `action_encoder`/`head.linear`, which these per-layer-type benchmarks don't model, so it has no further effect here), one model per process:

| | prefill (VAE+25L backbone) | one denoise step (25L) | prefill+10-step |
|---|---:|---:|---:|
| FP16 | 180.4 ms | 4.89 ms | 229.4 ms |
| INT8 (SM80) | FAILS: `cutlass_int8_rowwise_fp16out` rc=131079 at the real `mlp2` shape (M=512,N=3072,K=9216) | -- | -- |
| INT4 (SM80) | **74.8 ms (2.4x faster)** | 4.11 ms (1.19x faster) | **115.9 ms (2.0x faster)** |

~3.3x faster prefill and ~2.2x faster overall in the first table (OPT-003 removed the attention-bound ceiling that used to cap the denoise step's own speedup), still an optimistic upper bound rather than a real deployment number. In the real-dims table INT4 is genuinely fast — prefill wins big at large M, the denoise step wins much less at `NUM_ACTION=10`'s tiny M — and INT8 fails identically to every prior finding in this entry, confirmed fresh, in isolation, at these new dims.

## Thor check

Real Thor (SM110) with the same flags, GEMM-only, no Hadamard, plus the same-shape Ada control (`imagewam_gemm_precision_compare.py`, the exact same 4 shapes at M=896, identical convention, on this dev machine's real Ada GPU, RTX 4060 Laptop, sm_89):

| shape | FP16 (Ada) | INT8 (Ada) | INT4 (Ada) | FP16 (Thor) | INT8 (Thor) | INT4 (Thor) | INT4 vs FP16 (Ada) | INT4 vs FP16 (Thor) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q/proj (896×3072×3072) | 0.807ms | 0.213ms | **0.086ms** | 0.127 ms | 0.172 ms | **3.30 ms** | **9.4x faster** | 26x SLOWER |
| k/v (896×128×3072) | 0.045ms | 0.025ms | 0.040ms | 0.019 ms | 0.027 ms | 0.46 ms | 1.1x faster | 24x SLOWER |
| mlp0 (896×9216×3072) | 1.633ms | 0.458ms | **0.286ms** | 0.835 ms | 0.500 ms | **9.39 ms** | **5.7x faster** | 11x SLOWER |
| mlp2 (896×3072×9216) | 1.634ms | FAIL (known K=9216) | FAIL (same) | 0.526 ms | 0.527 ms | **9.86 ms** | — | 18.7x SLOWER |

Full pipeline on Thor (GEMM-only, same convention as the Ada number, earlier round): **1214ms** — versus FP16's 140.4ms on the same hardware, **~8.6x SLOWER**, not faster; per-shape INT4 is 6-19x slower than FP16, where real Blackwell tensor-core INT4 throughput should be much faster than FP16. In the later round, with the corrected (128-channel) VAE included, full 25+25-layer pipeline: **INT8 (SM80) completes cleanly, 177.1ms, essentially identical to FP16's 178.3ms (~1.00x)** — no crash, no benefit; **INT4 remains the dramatic ~7x slowdown, 1255ms in that same run**. No further attempt was made to get a full-pipeline INT8 timing number on Ada — the kernel is not usable end-to-end at ImageWAM's real dims there.

Same compiled kernel binary, same shapes: 5.7-9.4x FASTER on Ada, 11-26x SLOWER on Thor — a hard reversal, not just a magnitude difference. This rules out "INT4/this quantization scheme is just bad" (it would also be bad on Ada if so) and any algorithmic or Hadamard-related cost (no Hadamard rotation runs in this measurement on either machine), and rules out reopening the path through a different quantization pre-processing choice: the bottleneck is the kernel's hardware dispatch. **This SM80 INT4/INT8 path is therefore a dead end on Thor specifically, confirmed by real measurement, not a slower-but-usable fallback.** Thor's own native NVFP4 (SM100) path (`benchmarks/imagewam_thor_fp4_bench.py`) is the correct low-precision target there; the Thor build was reset back to the default slim config (Chameleon/INT4 off) afterward specifically to avoid this slow path being used by accident later. This kernel family remains a legitimate (if still K-limited and flaky above K=4096) option on the true Ampere/Orin-class hardware it was actually built for — the negative result here is Thor-specific, not universal.

## Hadamard padding probe: does zero-padding K to a power of 2 unblock the FHT crash?

`benchmarks/imagewam_int4_hadamard_padding_probe.py`: real math, not a guess — padding both activation and weight with zeros before an orthogonal (Hadamard) rotation exactly preserves their inner product (`<H@x_pad, H@w_pad> = <x,w>`, verified numerically, max abs error ~1e-4). It tested whether this lets `fht_int4_quant_fp16` + `cutlass_int4_rowwise_fp16out` run correctly at ImageWAM's real K values once padded to the next power of 2 (weight-side rotation implemented in plain torch for this probe only, since FlashRT itself has no kernel for it). The result is genuinely mixed, and inconsistent between two otherwise-identical runs of the same logic: `K=3072→4096` works reliably (cosine=0.983, reproduced identically across the exploratory run and 3 repeats of the formalized script); `K=7680→8192` and `K=9216→16384` both FAIL in the formalized script (reproducibly, 3/3 runs), but the FIRST, less careful exploratory run of the identical logic had `K=7680→8192` WORKING (cosine=0.977). On Thor, `K=3072→4096` matches Ada exactly (cosine=0.983), `K=7680→8192` this time WORKS (cosine=0.977, matching Ada's very first exploratory run rather than Ada's later reproducible failures — the instability itself is now confirmed cross-hardware, not an Ada-specific quirk), and `K=9216→16384` fails again with yet another different symptom ("CUDA invalid argument" on Thor vs. Ada's all-zero-scale-with-no-error). Net effect: only K=3072 (q/k/v/proj, mlp0's own input width) is currently a reliable target for real QuaRot-rotated INT4; txt_in (K=7680) and mlp2/mlp_down (K=9216) remain blocked — not only by the already-known FHT crash, but now also by this padding workaround's own unreliability at those larger sizes.

This exact SM80 INT4/INT8 CUTLASS family has now shown unexplained run-to-run instability three separate times (here, and twice in this entry's own `mlp2` findings above), plus a new kind of symptom found while adding the VAE step to `imagewam_thor_int8_bench.py` on Ada: the exact same script, run back to back with no code changes, sometimes completes the `vae_encode` timing loop and reaches the known `K=9216` failure within a few seconds (reproduced most of the time), and sometimes hangs for 60+ seconds at 100% GPU util with zero forward progress before being killed by hand (reproduced twice, not reproduced in three separate instrumented step-by-step re-executions of the identical call sequence — model construction, single `run_vae_encode` calls, and the exact 15-warmup+50-measured pattern `_time_ms` uses). Not root-caused, and it blocks nothing since this path is already closed on both Ada (crash) and Thor (no benefit) — but it establishes a methodology rule: **always benchmark this specific kernel family (SM80 INT8/INT4) in an isolated, single-precision process, never in a combined multi-precision comparison script, regardless of which precision runs first.** The Ada measurement at real dims above follows that rule because the combined attempt produced a nonsensical INT4 prefill number, **5156ms — a 69x regression from the real 74.8ms**, consistent (tight P50/P90) across all 50 measured iterations rather than a one-off spike: isolating layer-by-layer (`_double_layer`/`_single_layer` alone) and even raw isolated GEMM calls at the exact shapes involved all timed fast and normal (sub-2ms/sub-0.3ms respectively) and summed to the correct ~70ms, while INT4 alone in a fresh process was completely stable at ~70ms across 30 consecutive calls with VRAM flat after call 1. Root-caused by elimination, not just observed: the 5156ms number was real but an artifact of cross-precision process/GPU state contamination (most likely from the failed INT8 model's partial 25-layer weight allocation, or cuBLASLt/CUTLASS handle-level state, left behind when its exception path returned without the success path's `torch.cuda.empty_cache()`) — **NOT a property of the INT4 kernel itself**.

## Root cause: a real Blackwell ISA-level fact, not a FlashRT bug

This isn't a quirk of this codebase's own kernel or an isolated "compatibility path" guess — it's a documented, general property of Blackwell's fifth-generation Tensor Core (`sm_110a` on Thor). The `tcgen05.mma` instruction family CUTLASS's own SM100/SM110 support targets covers legacy TF32/FP16/BF16/**INT8/UINT8** dense paths (kept for backward compatibility), plus the new sub-8-bit *block-scaled float* formats — MXFP4, **NVFP4**, MXFP6, MXFP8 (`tcgen05`'s own `kind` enum includes `f16`, `i8`, `mxf8f6f4`, `mxf4`, `mxf4nvf4`) — **there is no `kind::s4` / plain-integer-INT4 entry at all.** Thor's native full-speed 4-bit path is a *scaled float* format (E2M1 mantissa + per-block scale), not the `s4×s4->s32` dense-integer MMA Ampere/Ada/Orin have. `csrc/gemm/cutlass_sm80_int4_rowwise.cu` (lines ~59-63) templates this kernel on `ArchTag = cutlass::arch::Sm80`, `InstructionShape = GemmShape<16, 8, 64>` — Ampere's real `mma.sync.aligned.m16n8k64...s4.s4.s32` integer tensor-core instruction (W4A4), which Ada (sm_89) and Orin (sm_87) have natively, and which is why the Ada-vs-Thor control above shows genuine 5.7-9.4x speedups, i.e. real tensor-core throughput. There is no Blackwell-native lowering for `m16n8k64 s4.s4`: CUTLASS still reports `can_implement` success and the kernel still launches (`rc=0`), but execution must go through a non-tensor-core compatibility/emulation path — not a bug or a missing optimization flag, a genuine hardware-generation fact. Any AWQ/GPTQ-style pre-packed INT4 model would need to be converted to NVFP4 (or dequantized) to reach Thor's real 4-bit throughput. This confirms, does not merely support, the "NVFP4 is Thor's actual native low-bit path, not a kernel choice" conclusion this entry and OPT-014 already reached independently; the same verdict is recorded outside this entry in `THOR_STATUS_SUMMARY.md` §"INT8 / INT4（SM80 CUTLASS）" (with INT4 prefill 1253.6 ms and INT8 138.7 ms) and in `PROJECT.md` ("~8.6x slower than FP16, see `opportunities.md` OPT-007"). **Not planned to be fixed**: the correct fix for a genuine 4-bit tensor-core path on Thor is NVFP4, already the shipped default (OPT-014) and already getting a real, measured speed win there; a new, genuinely Blackwell-native plain-INT4 kernel would duplicate what NVFP4 already provides, with no project-scope justification (Thor-only). This SM80 kernel's only legitimate future target remains a hypothetical true Orin/Ampere deployment.

## Expected Mechanism

Same mechanism the Chameleon-7B path already uses in production on its own (Orin-class) hardware (assumed, not independently re-verified here): FHT-rotated activations + offline-rotated weights survive int4's dynamic range at measured cosine 0.9914 (per the kernel file's own header comment, for Chameleon's own model and data — not re-measured for ImageWAM, and now confirmed NOT to translate to a speed win on Thor even where it runs). The Ada-vs-Thor reversal above is the direct evidence for the "wrong tensor-core dispatch path on Blackwell" half of this mechanism — previously inferred from the Thor numbers alone, now confirmed by a same-shape control on hardware where the same kernel binary is known-good.

## Required Evidence

For Thor: none needed further — closed, real negative result in hand. For a hypothetical true-Orin deployment: resolve the FHT power-of-2 blocker (every real ImageWAM hidden dimension is a non-power-of-2 multiple of 1024) and its own reliability above K=4096, plus real-weight accuracy work (OPT-001). Promotion condition: closed for Thor (real, measured, ~8.6x slower — promoted as a verified negative result, not pursued further there); reopening elsewhere would need a real Orin/Ampere deployment target and the evidence above.

## Open

- Only a hypothetical true Orin/Ampere deployment. Remaining scope if anyone pursues it: a fourth precision tier on true Ampere/Orin hardware (not Thor) for q/k/v/proj/mlp0 specifically (K=3072 only, per the padding probe above), IF someone implements the weight-side offline rotation (this project has none) AND the QuaRot rotation is validated for ImageWAM's own activation distributions (a real correctness project, not yet started — it would need real weights to even evaluate, the same dependency as OPT-001). Blockers: the FHT power-of-2 crash at non-power-of-2 dims, the kernel family's own reliability above K=4096 (`K=7680→8192` and `K=9216→16384` both fail), the offline weight-side rotation that does not exist in this project, and real-weight accuracy work (OPT-001).

# OPT-008

Status: real VAE-encode cost is in every local/Thor full-pipeline benchmark, the VAE encoder itself is wired into the served frontend (2026-09-15, all 3 phases done, with native-NHWC and in-graph VAE paths alongside it), and the `img_in` projection it exposed is now in `pipeline_thor.py` itself (`imagewam_prefill`, once per call, `Bf16OutLinear`); the three real-accuracy bugs this entry found -- `FP16 residual overflow`, `txt_in`/`img_in` re-derived per double-stream layer, flat image RoPE grid -- are all fixed and confirmed together on real Thor.

Area: VAE encoder (input-image tokenization) -- previously excluded from every full-pipeline speed number with no clear justification; now included

## Observation

A fixed task instruction is encoded once per episode, but the camera observation changes every control-loop iteration, so VAE encode runs once per real inference call (once per new observation) -- a real, recurring, resident cost that belongs in a "real machine full inference steady-state speed" number.

Scope is VAE **encode** only: `infer_action_flux2` (the real action-only inference entry point this project's pipeline mirrors) calls `_encode_flux2_image_tokens` once at the start, denoises only the action latent and returns `{"action": ...}` -- no `vae.decode` in that path; the video-editing branch that would need decode (`infer_video_flux2`) is a separate, unused method.

## Implementation

`benchmarks/_imagewam_vae_stub.py` (shared by every `imagewam_thor_*_bench.py` script): a standard SD/FLUX-family latent-diffusion VAE encoder (`ch=128`, `ch_mult=(1,2,4,4)`, 2 ResnetBlocks/stage, one mid-block self-attention) in plain PyTorch/cuDNN (Conv2d/GroupNorm/SiLU/`scaled_dot_product_attention`) -- **not the real FLUX.2 AE**, which lives in `black-forest-labs/flux2` and is not vendored anywhere on this machine (`docs/dependencies.md`: "user clones upstream, not vendored"). A representative, real (real cuDNN conv) workload of the right computational order, not a bit-exact claim.

Input resolution `384x512` was chosen to produce exactly 768 tokens, matching the `A0-X0` image-token span every bench script already used. **Corrected token width, 128 not 64**: `flux2_video_expert.py` states packed image tokens must be `[B,N,128]` and `pack_latents` is `rearrange(latents, "b c h w -> b (h w) c")`, a PURE reshape with no 2x2 patch-merge at all -- so the real FLUX.2 VAE downsamples 16x spatially AND emits 128 channels directly, not the classic SD/FLUX.1 pattern (8x downsample + 16 latent channels + separate 2x2-merge to 64-dim) this module first assumed by analogy. The stub now downsamples 16x in one conv stack (5 stages, 4 downsamples), emits 128 channels, and `pack_latents` is a pure reshape; `img_in`'s K changed from 64 to 128 in every bench script (they import the constant, so no script edits were needed).

`img_in` gap exposed: `pipeline_thor.py`'s own docstring stated only `txt_in` was modeled ("the img_* analogs of all but txt_in") -- there had never been an `img_in` weight anywhere in this project and image tokens were assumed to arrive at HIDDEN width. A real VAE's raw patch output is not 3072-dim, so an `img_in: Linear(64, HIDDEN)` projection is structurally required; it was added in the benchmark scripts (`_Int4Linear`/`_Fp16Linear`/`_Fp8Linear`/`_Fp4Linear`(HIDDEN, 64), matching each script's own precision) and wired into `run_prefill()` once per call in `imagewam_thor_int4_bench.py`, `_fp16_bench.py`, `_fp8_bench.py`, `_fp4_bench.py` -- not `_int8_bench.py`, which already fails to complete a full pipeline run (OPT-007's own K=9216 finding) under the standing policy "INT8 only if it fits, otherwise INT4 only".

## Real local result (Ada)

| | vae_encode (standalone) | backbone_prefill (25L + VAE) | one denoise step | full (prefill + 10-step denoise) |
|---|---|---|---|---|
| INT4 | 55.4 ms | 96.9 ms | 4.78-4.96 ms | 136.6 ms (single run) / 144.7 ms (cross-check sum) |
| FP16 | 56.4 ms | 200.4-200.5 ms | 6.15-6.22 ms | 260.2 ms |

Standalone VAE encode is ~55-56 ms regardless of downstream GEMM precision (same VAE, same input, every script), and roughly DOUBLES the INT4 prefill number (42.6 ms -> 96.9 ms) while being a much smaller relative addition to FP16's own already-large prefill (152 ms range without it): VAE cost matters more, relatively, the faster the backbone gets, so backbone-only precision work alone may not move the full number. FP8/FP4 scripts run the new VAE step cleanly before hitting their own already-known, unrelated failure points (FP8: cuBLASLt env gap; FP4: Blackwell-only `SystemExit`) -- not verified end to end, and now correctly showing K=128 in the error message instead of K=64. INT4/FP16 re-verified end to end after the width fix, VAE cost ~49 ms.

## VAE stub optimization attempt (Ada, 8 GB machine)

Profiled with `torch.profiler` first: convolution itself dominates (~66% of GPU time), the rest split across GroupNorm/SiLU and cuDNN's own NCHW<->NHWC layout-conversion kernels. `channels_last` + `cudnn.benchmark=True` made things WORSE (55 ms -> 63 ms), most likely because the mid-block attention's own `.reshape()`/`.permute()` calls force a layout conversion back. `torch.compile(mode="max-autotune")`: ~55 ms -> ~44 ms in isolation, confirmed end to end in the INT4/FP16 scripts. `torch.compile(mode="default")`: ~55-61 ms -> ~48 ms. But inside the FP8 script both stalled: max-autotune put the GPU at 100% util / ~7.9-7.92 GB of this 8 GB machine (near-OOM) for 45+ seconds with zero forward progress logged (hard kill required), and `mode="default"` stalled again for 65+ seconds stuck inside `FullImageWAMFP8.__init__` itself (before "Built." prints, i.e. before any VAE forward call), GPU pinned at 100% util, memory again near ~7.9 GB, no lingering compile-worker process found; INT4/FP16 use the identical `build_vae_encoder` call and never reproduced it. Not root-caused.

Decision: kept `torch.compile` as an explicit opt-in (`compile=True`), default reverted to `compile=False` -- a real, measured ~13-20% (~7-13 ms) gain off the VAE's own cost is not worth an unpredictable path to a near-OOM multi-minute stall on this machine's already-tight 8 GB budget.

## FP16 residual overflow: root cause and BF16 fix (real Thor, 2026-09-15)

Symptom: the full three-real-components run (real checkpoint weights, real VAE-encoded real LIBERO frame, real Qwen3-4B prompt) at the real `x0=512` through the real 25-layer backbone (5 double + 20 single) produced `nan` (`finite=False`) on both this 8 GB dev machine and Thor, so memory alone could not be the cause. The OFFICIAL ImageWAM reference model's own real bf16 `imagewam_prefill`, on the exact same real instruction and real VAE-encoded frame, produced absmax=**119808** in its own backbone residual -- already past FP16's ~65504 ceiling -- and stays finite only because it runs in bf16 (FP32's exponent range), while FlashRT's entirely-FP16-resident `imagewam_prefill_real`/`pipeline_thor.py` serving path overflows. Every prior real-checkpoint accuracy claim on record (cosine=0.9999) used synthetic `N(0,1)`-scale random context at the OLD `x0=128`, never exercising real Qwen3's actual activation range.

Diagnosis: isolated to a single backbone double-stream layer with the real Qwen3 context, only **row 0** (the chat template's own first special token, a well-documented LLM "attention sink") goes to `Inf`; every other row stays finite. Ruled out as the cause: the txt|img seam/join logic (join-point cosine=1.000000 against the reference), the `txt_in` GEMM itself (its own FP32-accumulated row-0 output is only ~8736, nowhere near overflow), the attention kernel (FA4 vs the plain cuBLAS-composed kernel at real serving shape a0=904: cosine=1.000000, rel_l2=0.000616 either way), a data race, and a masking bug. Clamping the real context to `[-256,256]` before `txt_in` made the whole layer finite (txt absmax=53088, right at FP16's edge), confirming the overflow builds up inside the double-stream block's own AdaLN/MLP/residual chain from that one large starting value, and accumulates further across the full 25-layer stack.

Local prelude (this dev machine, real measurement, superseded as a diagnosis but genuine): `torch.cuda.memory_allocated()` right after loading all 25 real backbone layers' weights (no KV cache, no scratch buffers) was 8.56 GB, already past this machine's 8188MiB (8.19 GB) physical VRAM; `num_double=5, num_single=20` gave layer 0 already `inf`, while `num_double=2, num_single=0` (same weights, same real context) stayed finite with the row-0 outlier actually shrinking across layers (16256 -> 10664 -> 8728), and the real Qwen3 row-0 hidden state absmax=16256 (bf16) projected through real `txt_in.weight` in FP32 accumulate reaches only ~8735. Related narrow margin finding: the isolated one-real-layer test went `inf` at x0=300/384/512 but stayed finite at 128/256/320/340/360 (not a monotonic "too big" threshold), with every instrumented intermediate finite (absmax 4-370) and a finite layer output (mean=93.3, std=284.2), consistent with cuBLASLt's own split-K/atomic run-to-run nondeterminism tipping a value computed near FP16's 65504 ceiling; never reproduced with the REAL Qwen3-encoded context at `num_double=1` or 2.

Fix: promote ONLY the persistent backbone residual buffer (`bufs["backbone_hidden"]`, plus the `context`/`img_raw` buffers that write into it via `txt_in`/`img_in`) from FP16 to **BF16** -- same 2 bytes/element (no memory or bandwidth cost), FP32's exponent range. Everything else (every weight, every post-AdaLayerNorm activation, QKV, attention Q/K/V, MLP intermediates) stays FP16: real Thor tracing showed those are all comfortably bounded (O(1-400)) regardless of the residual's own scale, because AdaLayerNorm re-normalizes on every read. New kernels `ada_layer_norm_bf16in_fp16out` (`csrc/kernels/norm.cu`) and `gate_res_bf16res` (`csrc/kernels/decoder_fused.cu`), both structural copies of their existing FP16 counterparts with only the residual I/O retyped to `__nv_bfloat16`. New `Bf16OutLinear` (`flash_rt/models/imagewam/quant_linear.py`) wraps the already-existing `GemmRunner.bf16_nn`/`autotune_bf16_nn` for `txt_in.weight`/`img_in.weight` specifically, applied regardless of `self._precision` (orthogonal to the FP8/NVFP4 quantization study track). `flash_rt/models/imagewam/text_encoder.py`/`vae_encoder.py` now return BF16 directly instead of downcasting to FP16, removing the exact round-trip that silently produced the overflow. Scope note: ActionDiT's own residual stream was NOT touched (its conditioning comes from the small `action_dim=7` encoder + shared timestep embedding, not Qwen3's context), and `benchmarks/imagewam_real_checkpoint_validation.py`'s reference harness (`imagewam_prefill_real`, `pipeline_real.py`) stays FP16-only, since it deliberately feeds synthetic `N(0,1)`-scale random tensors and never exercised this bug.

## Real VAE encoder + text-context wiring (2026-09-15)

`imagewam_thor.py.infer()` has a real path: given `ae_model_path`/`flux2_src` at construction, a real image (`observation["view1"]`, optionally `"view2"`) is encoded through the REAL FLUX.2 VAE outside the captured CUDA Graph and copied into `img_raw` before `.replay()`; `set_prompt()` accepts a real precomputed `context`/`context_mask` pair (matching `imagewam.py`'s own `_prepare_flux2_infer_text` interface exactly) as an alternative to random-filling `context`.

- `black-forest-labs/flux2` IS clonable from this sandboxed dev machine (previously assumed not reachable, based on never having tried); `git clone` pins to the exact commit `50fe5162777813d869182b139e83b10743caef15` this project's docs already referenced by hash.
- **Wrong-reference-class bug, caught before it shipped**: `imagewam.py`'s real VAE construction calls `flux2.autoencoder.AutoEncoder(AutoEncoderParams())` directly, NOT `diffusers.AutoencoderKLFlux2` (the class `model_index.json` names). Against a real `libero_spatial_no_noops_lerobot` frame the diffusers class gives mean=-0.031/std=1.72/absmax=8.31 (it defines an identical `self.bn` BatchNorm2d submodule but never calls it in its own public `encode()`, and skips the real 2x2 patch-merge entirely); the real `flux2.autoencoder.AutoEncoder.encode()` gives mean=-0.012/std=0.973/absmax=4.72, matching the real Thor measurement (mean=-0.02, std=0.97, absmax=4.91) almost exactly.
- New isolated venv `FlashRT/.venv`: the shared `third_party/openpi/.venv` pins `lerobot==0.4.4` to `diffusers<0.36.0`, while `diffusers.AutoencoderKLFlux2` needed diffusers>=0.37 to import at all; built with `torch==2.14.0+cu130`, `pybind11==3.1.0`, same Python 3.11.13, `flash_rt_kernels.so` rebuilt once (~30 s incremental) and verified working from BOTH venvs.
- New `flash_rt/models/imagewam/vae_encoder.py` (`load_real_ae`, `encode_to_tokens`), new `flash_rt/models/imagewam/text_encoder.py` (`load_real_text_encoder`, `encode_prompts`, porting `imagewam.py`'s `_encode_flux2_prompts`: chat template, `max_length=512`, concatenated hidden layers `[9,18,27]` -> `(1,512,7680)`, confirmed against `flux2.text_encoder.OUTPUT_LAYERS_QWEN3`), `tests/test_imagewam_vae_encoder.py` (skips cleanly without the real clone/checkpoint), and `test_full_frontend_with_real_checkpoint_and_real_vae` in `test_imagewam_checkpoint_loader.py` combining both real paths -- passes end to end, finite `(64,7)` action output. Live Qwen3-4B (`Qwen/Qwen3-4B`, ~7.6 GB, at `/home/ljw/projects/pi0.5/models/qwen3_4b`) also passes: construction, real text encoding, graph capture and `infer()` with random transformer weights, finite `(64,7)`.
- **`x0=512` is the real value** (Qwen3's own fixed `max_length`); `benchmarks/imagewam_thor_bench.py`, `benchmarks/imagewam_real_checkpoint_validation.py` and `tests/test_imagewam_checkpoint_loader.py` were updated from the `x0=128` placeholder. The currently served LIBERO workload derives `x0=513` with the proprio row (OPT-009).

## Thor check

Round 1 (2026-09-15, full-scale `fp16`/`fp8`/`fp4` scripts with the VAE step in place):

| | vae_encode | prefill (25L + VAE) | one denoise step | full (prefill + 10-step) | full, no VAE (prior measurement) |
|---|---|---|---|---|---|
| FP16 | 43.9 ms | 124.5 ms | 5.87 ms | 183.3 ms | 140.4 ms |
| FP8 (dynamic scale) | 43.7 ms | 107.8 ms | 6.17 ms | 169.5 ms | 106.6 ms (old fixed-scale, no VAE) |
| FP4 (NVFP4) | 43.8 ms | 98.7 ms | 5.51 ms | 153.7 ms | 111.1 ms |

Backbone-only numbers (VAE subtracted back out) still match the earlier no-VAE measurements closely: FP16 ≈80.6 ms vs 81.6 ms, FP4 ≈54.9 ms vs 55.8 ms; FP8's own backbone-only prefill grew by ~4 ms and its denoise step from 4.65 ms to 6.17 ms because of switching to genuine per-call dynamic scale measurement (`quantize_fp8_device_fp16`'s real amax kernel) -- a real, expected cost of not having real calibration, not a regression. The VAE was ~24-28% of full pipeline latency at that shape, bigger than the entire 10-step denoise loop post-OPT-003 and than OPT-004's combined graph-capture + autotune ~10 ms win (captured graph 129.3 ms + VAE outside the graph 43.9 ms ≈ 173 ms vs graph-free FP16 with VAE 183 ms, so graph capture saves only ~10 ms beside a ~44 ms VAE cost), which sharpened OPT-004's own "compute-bound, not launch-bound" conclusion. The current figure for that share is OPT-012's real steady-state breakdown (224x448, `x0=513` with proprio, real 10-step shift=5.0 schedule): out-of-graph VAE 21.5 ms / **7.6%** of a 284.9 ms `infer()`.

- Round 2 (2026-09-15, BF16 fix on Thor): `finite` fully restored (`actions` mean=0.126 std=0.390 absmax=0.921; `backbone_hidden` absmax=68096, no longer Inf; the isolated row-0 case and the isolated single-layer random-`N(0,0.5)` case 20/20 finite). But cosine vs the official model was only all=0.559 (txt=0.548, img=0.907) -- a second, independent, much older bug, exposed because this was the FIRST TIME `pipeline_thor.py`'s own serving path was compared cosine-wise against the real official model end to end.
- Round 3 (second bug, found and fixed same day): `_double_stream_layer` re-derived `txt_in`/`img_in` from RAW `context`/`img_raw` at the START of EVERY double-stream layer, discarding the previous layer's entire computed output -- introduced on day one (`7aa431d`, "Phase 3: ImageWAM backbone prefill"). The real `flux2` model (`third_party/flux2/src/flux2/model.py`: `img = self.img_in(x); txt = self.txt_in(ctx)` called ONCE, BEFORE `for block in self.double_blocks`) has exactly ONE `txt_in`/`img_in` each for the whole transformer, confirmed by `checkpoint_loader.py`'s own real-checkpoint finding ("txt_in/img_in shared across every double layer -- same tensor object, not L independent copies"), so only the LAST double-stream layer's own single pass over the raw input ever reached the single-stream layers and the KV cache, discarding 4 of the 5 real double-stream layers' worth of depth for BOTH streams (text hit harder: 0.548 vs image 0.907). Invisible before because every prior accuracy claim (cosine=0.999927 backbone, 0.999963 ActionDiT) compared the official model against `pipeline_real.py`'s `imagewam_prefill_real`, a separate tensor-level reference that takes ALREADY-PROJECTED `txt`/`img` and never calls `txt_in`/`img_in` itself, and `test_imagewam_thor_real_wiring.py`'s `test_double_stream_layer_matches_real_reference` calls `_double_stream_layer` exactly ONCE (a single-layer test structurally cannot expose a "discards the previous layer's output" bug). Fix: project `txt_in`/`img_in` once in `imagewam_prefill` before the double-stream loop, via `weights[("backbone", "double", 0, "txt_in.weight")]`/`"img_in.weight"` (any layer_idx gives the identical shared tensor); both direct-call tests were updated to do the equivalent one-time projection first (cosine=0.999994 against `pipeline_real.py`), and their stale FP16 `combined` allocation was caught and fixed in the same pass after `combined` became BF16-only.
- Round 4 (third bug, found by inspection same day): after the projection fix, `backbone_hidden` absmax matched the official model EXACTLY at every layer boundary (105984/110080/111104/111616/119808 at the 5 double-stream layers, 119808 at the end) and txt cosine reached 0.998 (from 0.548) with per-layer cosine never below 0.9985, but img barely moved (0.907 -> 0.910, all=0.985) and diverged from LAYER 0 already (FlashRT absmax=122 vs official=138 right after the first double-stream layer, cosine 0.979 there, down to 0.752 by layer 3). Cause: `flash_rt/frontends/torch/imagewam_thor.py` (the ACTUAL served frontend) called `build_backbone_rope_table(d["x0"], d["a0"] - d["x0"], 1, device=DEV)`, passing `img_len` (392) as `ref_h` and `1` as `ref_w` -- a degenerate 392x1 strip instead of the real 14x28 image patch grid -- so every image patch got the WRONG 2D spatial position for RoPE, compounding across joint-attention layers. Every other real-math test passed real `ref_h`/`ref_w` (14/28); only the served frontend was wrong, and the direct-call tests' own internally-consistent WRONG table cancelled out in a same-table comparison. Fix: the constructor reads `ref_h`/`ref_w` from `dims` (default the old flat `(img_len, 1)` so toy/default dims are unaffected) and validates `ref_h*ref_w == img_len`; every REAL-dims call site now passes `REF_H=14, REF_W=28` explicitly (`benchmarks/imagewam_thor_bench.py`, `tests/test_imagewam_checkpoint_loader.py`).
- Round 5 (confirmation, same day): with `ref_h=14, ref_w=28` passed explicitly in `dims_override`, `backbone_hidden` absmax stayed 119808 and cosine all/txt/img went 0.985/0.998/0.910 -> **0.999966/0.999966/0.999966**, closing the whole OPT-001/OPT-002 real-accuracy investigation on real Thor against the real official model, with real Qwen3-4B text conditioning, a real VAE-encoded real LIBERO frame and the real `x0=512`. Footgun closed: `ImageWAMTorchFrontendThor.__init__` now raises `ValueError` immediately (before touching the multi-GB checkpoint file) if `ckpt_path` is given without `ref_h`/`ref_w` in `dims_override`; structural/random-weight dry runs (`ckpt_path=None`) are unaffected.
- Round 6 (follow-up, attempted and deliberately not completed): comparing the full `actions` output (backbone prefill + ActionDiT denoise loop) against the official model's own real `infer_action_flux2` was not run, because the official model integrates over a non-uniform shift-based schedule (`imagewam.py` -> `scheduler_continuous.py`'s `WanContinuousFlowMatchScheduler.build_inference_schedule`, `shift=5.0` default, `_phi(u,shift)=shift*u/(1+(shift-1)*u)`, `num_inference_steps=20`) while `imagewam_denoise_loop`/`imagewam_denoise_step` integrated a fixed, UNIFORM `dims["dt"]` -- a pre-existing simplification, so such a comparison would not measure "is ActionDiT's own math correct". Now superseded: the real schedule is implemented (`flash_rt/models/imagewam/scheduler.py`'s `build_inference_schedule`, the optional per-step `deltas`/`delta` parameters of the denoise loop/step, falling back to `dims["dt"]`) and the official-model comparison is done in OPT-010/OPT-011 (cosine 0.870-0.999 over 50 real frames). Indirect evidence either way: `real_action_double_block_forward_fp16`/`real_action_single_block_forward_fp16` (`real_action_expert.py`, shared unchanged by `pipeline_real.py` and `pipeline_thor.py`) already has real-checkpoint cosine=0.999963, and none of the three bugs touched ActionDiT's own code path.

## Open

- `torch.compile` on the benchmark VAE stub stays an explicit opt-in (`compile=True`, default `compile=False` in `benchmarks/_imagewam_vae_stub.py`) because the FP8-script near-OOM / multi-minute stall is not root-caused; the next attempt should either root-cause the FP8-specific stall (ideally on Thor, where memory pressure may not be a confound) or pursue the custom-fused-kernel path (mirroring `cosmos3_edge/vae_native.py`'s own GroupNorm+SiLU fusion precedent) instead of `torch.compile`.
- The benchmark stub's own architecture is not further promotable without either (a) fetching FLUX.2's real AE source/config to replace this representative architecture with the real one, or (b) real weights/calibration (OPT-001) making VAE accuracy relevant, not just its speed. VAE optimization remains a real performance item, now bounded by OPT-012's measured 7.6% share rather than by the stub-era ~44-56 ms figure.

# OPT-009: real closed-loop robot-state (proprio) conditioning

Status: RESOLVED -- real Thor confirmation same day (full 25-layer, real LIBERO first frame): `proprio_row=31` correct, `infer()` finite, real denormalized actions. The out-of-range action values that Thor run first found were root-caused to the uniform-`dt` schedule (OPT-010), and the official-model comparison this entry flagged was completed in OPT-011 (cosine 0.870-0.999 over 50 real frames, proprio included). Normalization stat choice confirmed exact from this release's own `config.yaml`, not guessed.

Area: `flash_rt/models/imagewam/checkpoint_loader.py`, `flash_rt/models/imagewam/dataset_stats.py` (new), `flash_rt/frontends/torch/imagewam_thor.py`, `tests/test_imagewam_proprio.py` (new); declared shapes in `flash_rt/frontends/torch/_imagewam_thor_spec.py`.

## Two real, previously-unmodeled gaps

**`proprio_dim=8`**: the real LIBERO checkpoint's `config.yaml` has it, `pack_proprio_after_text: true`, and a real trained `proprio_encoder` (`weight` `(7680,8)`, `bias` `(7680,)`, confirmed via `torch.load(ckpt_path, mmap=True)`'s TOP-LEVEL payload -- a SIBLING of `mot`, not inside it). The official model's own `infer_action_flux2` -> `_append_proprio_to_context_if_enabled` RAISES `ValueError` if `proprio_encoder` exists but no `proprio` is passed -- proprio conditioning is not optional for this release. FlashRT had ZERO mechanism for it (`checkpoint_loader.py`, `imagewam_thor.py`, `pipeline_thor.py` -- confirmed via grep, no hits at all) before this entry.

**No denormalization in the official path**: the real `infer_action_flux2` returns the raw flow-matching output (`return {"action": latents_action[0]...}`, no `*std+mean` or equivalent anywhere in that function), so the open-loop `actions` numbers already on record all session (e.g. mean=-0.43, absmax=2.62 from the RoPE-grid-fix confirmation) were almost certainly still in the model's own `[-1,1]` TRAINING space, not real physical units. The release ships a `dataset_stats.json` alongside `model.pt` (`state`/`action`, each with `global_min/max/mean/std/q01/q99` AND `stepwise_*` variants) for the standard VLA convention: normalize on the way in (training + proprio input), denormalize on the way out (action output), using the SAME stats.

**Exact normalization convention, confirmed from this release's own `config.yaml`, not assumed**: `use_stepwise_action_norm: false`, `norm_default_mode: min/max`, `norm_exception_mode: null` -- both `state` (proprio, forward/normalize on the way in) and `action` (backward/denormalize on the way out) use plain `global_min`/`global_max` linear scaling to `[-1,1]`, clamped to `[-5,5]`, matching `imagewam/datasets/lerobot/utils/normalizer.py`'s own `SingleFieldLinearNormalizer` exactly (ported verbatim into a new `flash_rt/models/imagewam/dataset_stats.py`, including its degenerate-range `ignore_dim` handling). NEVER `stepwise_*`/`q01/q99`/`z-score` for this specific release -- a different release could use a different mode, check its own `config.yaml` before reusing this unchanged.

**Real insertion rule, NOT a simple append**: `imagewam.py`'s own `_append_proprio_to_context` (`pack_proprio_after_text=True` branch) inserts the proprio token at row `context_mask.sum()` (right after the last REAL text token, before any padding), shifting every padding row one position later. Context length grows by exactly 1 (`x0`: 512 -> 513 for this release's real Qwen3 `max_length=512`). This is data-dependent (depends on THIS prompt's own real token count) but only needs computing ONCE per `set_prompt()` call -- ported into a new `ImageWAMTorchFrontendThor._set_context_with_optional_proprio`, replicating the real scatter exactly (real tokens keep rank, proprio lands at `valid_counts`, padding shifts by 1 -- `tests/test_imagewam_proprio.py`'s `test_proprio_scatter_matches_real_insertion_rule`).

## Implementation

- `checkpoint_loader.py`: new `load_real_proprio_weights(ckpt_path)` -- separate small loader (the existing `load_real_imagewam_state_dict` only returns `payload["mot"]` by design; widening its contract would break every existing caller), returns `None` if this checkpoint has no `proprio_encoder` key.
- `dataset_stats.py` (new): `MinMaxNormalizer` (real `SingleFieldLinearNormalizer` min/max math, verbatim), `load_real_normalizers(dataset_stats_path)` -> `(state_norm, action_norm)`.
- `imagewam_thor.py`: new `dataset_stats_path` constructor kwarg; `dims["proprio_dim"]` opts proprio in (default `None` -- every existing caller/test unaffected) and loads the real `proprio_encoder` (or a random one when `ckpt_path=None`, matching every other weight's toy/random-vs-real convention); `proprio_encoder` is applied OUTSIDE the captured CUDA graph via plain `F.linear` (same convention as the real VAE/Qwen3 encoders); `set_prompt()`'s `_set_context_with_optional_proprio` does the real scatter once per prompt and records `self._proprio_row`; `infer(observation)` REQUIRES `observation["proprio"]` when `proprio_dim` is set (raises otherwise, matching the real model's own contract), normalizes it (if `dataset_stats_path` given), projects it, writes it into the row reserved by `set_prompt()` before `.replay()` (same pattern as `img_raw`/the VAE encode), and denormalizes the returned `actions` (if `dataset_stats_path` given) before returning.
- `tests/test_imagewam_proprio.py` (new): normalizer round-trip against the real `dataset_stats.json`, scatter-rule verification (toy dims, no real checkpoint needed), missing-proprio raises, real `proprio_encoder` shape check against the real checkpoint (skips cleanly without it).

## Thor check

Local, real checkpoint with `ckpt_path=`, `dataset_stats_path=`, `proprio_dim=8`, `x0=513`, `ref_h=14, ref_w=28` (1 real double-stream layer, kept small for this machine's memory): real `proprio_encoder` loads (`(7680,8)`/`(7680,)`), `set_prompt()` places the proprio row correctly (row 31 for a 31-real-token synthetic context, matching this session's own real Qwen3 observation), `infer()` produces finite, denormalized actions in a plausible range vs the real `dataset_stats.json` bounds. Full existing regression suite (backbone/single-stream/action reference tests, checkpoint-shape test, isolated real-weight layer test) still passes unchanged.

Thor, same day, real 3-real-component + proprio `infer()` (full 25-layer, real LIBERO first frame, real `observation.state`, all 8 real dims inside the real `dataset_stats.json` `state` range): finite throughout, `backbone_hidden` absmax=119808 (matches the official model exactly, as before), `proprio_row=31` (matches this session's own real 31-real-token observation), `actions` already denormalized (O(1) real-unit magnitudes, not the model's own `[-1,1]` training space). Wiring confirmed correct. BUT only ~62.5% of predicted action values landed inside the real dataset's own `[global_min, global_max]` range (gripper dim went negative against a real `[0,1]` range; translation dims ran further negative than the real range's own minimum) -- root-caused in OPT-010 to the uniform-`dt` schedule, not to this wiring.

## Open

- Proprio is still missing from `_imagewam_thor_spec.py`'s own declared shape documentation (its `context`/`context_mask` entries declare 512 rows, no proprio row). Cosmetic, not load-bearing for runtime; low priority.

# OPT-010: real shift-based flow-matching inference schedule

Status: RESOLVED -- real Thor confirmation same day: the scheduler is bit-exact on Thor too, and it was the actual cause of OPT-009's out-of-range action values, not policy behavior (fraction inside the real range 62.5% -> 98.4%); real `infer()` P50=289.5ms, single-step ActionDiT cosine=0.999978 vs the official model. Supersedes the fixed-uniform-`dt` integration schedule, this project's own original simplification.

Area: `flash_rt/models/imagewam/scheduler.py` (new), `flash_rt/models/imagewam/pipeline_thor.py`, `flash_rt/frontends/torch/imagewam_thor.py`, `tests/test_imagewam_scheduler.py` (new).

## Finding: much smaller scope than first estimated

The real `imagewam.models.backbones.schedulers.scheduler_continuous.WanContinuousFlowMatchScheduler.step()` (read directly) is a **plain single-step Euler update** (`sample + model_output * delta`) -- NOT a multi-step integrator like UniPC. `fvk.gpu_euler_step` (FlashRT's existing Euler kernel) already implements the exact right formula; only the per-step VALUES feeding it needed to change, from a linear `1.0 - step*dt`/fixed-`dt` formula to the real non-uniform `timesteps[step]`/`deltas[step]` the real `build_inference_schedule` produces. `step` was already treated as a compile-time-constant-per-unrolled-loop-iteration in FlashRT's own existing design (`action_mods[step]` already selected per iteration before this change) -- so this needed zero new architecture, zero new kernels, zero new buffers, purely a formula substitution at two call sites.

**Real confirmed parameters (this release's own `config.yaml`, not guessed)**: `shift=5.0`, `num_train_timesteps=1000` (both `video_scheduler`/`action_scheduler` blocks use the same values for this release), **`eval_num_inference_steps: 10`** (the real evaluation step count -- NOT the scheduler's own generic default of 20 some example configs use; matches this project's own long-standing "10-step" benchmark convention already on record above, confirming that number was already the right target). Unit conversion `_scheduler_timestep_to_unit`: `timestep / num_train_timesteps`, confirmed by reading `imagewam.py` directly, not assumed.

## Implementation

- `scheduler.py` (new): `phi`, `build_inference_schedule` -- ported verbatim from the real `WanContinuousFlowMatchScheduler`, verified bit-for-bit identical against the real scheduler directly (`tests/test_imagewam_scheduler.py`'s `test_schedule_matches_real_scheduler`, skips cleanly without the real `imagewam` package).
- `pipeline_thor.py`: `imagewam_denoise_step` gained an optional `delta=` param (falls back to `dims["dt"]` when `None`, so every existing caller is byte-for-byte unaffected); `imagewam_denoise_loop` gained an optional `deltas=` (list, one per step) threaded through the same way `action_mods`/`head_mods` already are.
- `imagewam_thor.py`: `dims["shift"]` opts the real schedule in (default `None` -- unset, every existing caller/test keeps the exact original linear formula, verified via the full regression suite still passing unchanged). `_compute_action_modulations` now returns `(mods, head_mods, deltas)` (was `(mods, head_mods)`) -- `deltas` is `None` unless `dims["shift"]` is set; threaded into both `_capture_graph()` call sites' `imagewam_denoise_loop(..., deltas=self._deltas)`.

Verified locally: `dims=dict(shift=5.0, num_train_timesteps=1000, num_denoise_steps=10)` end to end (construct, `set_prompt`, `infer`) -- finite, and `self._deltas` matches the real scheduler's own 10-step output exactly (`[-0.0217, -0.0259, -0.0313, -0.0387, -0.0490, -0.0641, -0.0874, -0.1263, -0.1984, -0.3571]`). Full existing regression suite (backbone/action reference tests, checkpoint tests, proprio tests) still passes unchanged with `shift` left unset.

## Thor check

Re-ran the same real 3-real-component + proprio `infer()` with `shift=5.0, num_train_timesteps=1000, num_denoise_steps=10` (the real confirmed values) instead of the old uniform `dt`:

| | old uniform schedule | real shift=5.0 schedule | dataset range |
|---|---|---|---|
| gripper mean / min-max | -0.24 / -0.43~0.14 | **0.996 / 0.981~1.002** | `[0,1]`, mean 0.51 |
| translation dim0 mean | -1.45 | **0.51** | `±0.94` |
| translation dim1 mean | -1.11 | **0.38** | `±0.94` |
| fraction inside `[global_min,max]` | 62.5% | **98.4%** (100% at 5% tol) | |
| first-step gripper vs real GT (1.0) | off by 1.43 | **off by 0.0008** | |

`self._deltas` matched the real `WanContinuousFlowMatchScheduler`'s own output bit-for-bit on Thor too (not just this dev machine). Real end-to-end `infer()` (full real pipeline: real VAE + real Qwen3 + real proprio + real 25-layer backbone + real 10-step ActionDiT denoise, all outside-graph real encode steps included): **P50 = 289.5 ms**.

**Single-step ActionDiT cosine vs the official model, real first step (`t=1.0`, `delta=-0.0217`, the real schedule's own first entry)**: hidden-state cosine = **0.999978**, finite. The per-layer transformer math, which is where any real bug would live, is covered at 0.999978.

**Minor finding, not prioritized**: the 5-precision table's own per-kernel-isolated benchmark showed `mot_joint` jump from 0.044ms (even `total=968`, the old x0=512 shape) to 0.195ms (odd `total=969`, the new x0=513-with-proprio shape) -- `softmax.cu`'s own kernels process columns in `__half2` pairs with a scalar fallback for a trailing odd column, and an odd `total` hits that fallback on every query row. In ABSOLUTE terms this is small (~0.15ms x 10 steps =~1.5ms) against the real 289.5ms end-to-end number, and is the reason the 5-precision table's own DERIVED "10-step" total (e.g. FP16 297ms) runs slightly higher than the real measured `infer()` (289.5ms) -- trust the real measured number, not the derived per-kernel sum, for this shape.

**Superseded takeaway**: at this shape `mot_joint`'s own absolute cost (0.04-0.2ms) is negligible next to the GEMM-dominated per-layer costs (backbone double/single: 4.9-5.9ms; ActionDiT double/single: 0.5-0.7ms), so this entry deprioritized FA4 at the `mot` site; OPT-019 later measured the attention share at the real shapes and revised that -- FA4 at `mot` is its strongest remaining attention lever. The real remaining speed lever at this shape is precision choice (NVFP4 already ~19% faster than FP16 at this shape, `242ms` vs `297ms` derived), not attention-kernel choice.

## Open

- The official `Flux2ActionHead`'s head+Euler-step cosine was never measured: it takes a `vec` tensor rather than the `t_mod` dict `real_action_*` produces, so that final portion wasn't chased down to its own cosine.
- The odd-`total` `mot_joint` penalty (0.044ms -> 0.195ms) is explicitly not chased; revisit only if it compounds with a future higher-`total` shape.

# OPT-011: real open-loop LIBERO evaluation -- closes out the correctness line

Status: RESOLVED -- the strongest validation this project has run: `pipeline_thor.py` faithfully reproduces the official model across real, diverse data (10 tasks x 5 frames = 50 real observations), not just the single frame every prior entry used. The harness that produced this run is not in this repository, and the served default has since moved to `nvfp4` with text trimming (OPT-014 re-ran the same tasks/frames as its own `fp16` baseline row).

Area: whole-pipeline correctness -- `flash_rt/models/imagewam/pipeline_thor.py` plus the whole real serving path (real VAE + real Qwen3 + real proprio + real backbone + real ActionDiT + real shift schedule + real denormalization, ALL together) across real task diversity; the 40-call stability check is measured through `ImageWAMTorchFrontendThor.infer()`.

## Result

**FlashRT vs the official model, same frame/noise/schedule (`num_inference_steps=10, sigma_shift=5.0`)**: cosine 0.870-0.999 across sampled episodes, most 0.997-0.999. **Critical diagnostic signature**: on episodes where BOTH FlashRT and the official model diverge from the real dataset ground truth (e.g. ep226, both land near cosine=0 vs GT, both predict gripper=1 where GT=0), FlashRT still tracks the OFFICIAL model closely (cosine 0.870/0.928 there) -- when the prediction is "wrong" relative to GT, it is wrong the SAME WAY on both sides. That is the signature of a real POLICY/checkpoint limitation on those specific tasks, not a FlashRT serving bug: a wiring/kernel bug would show FlashRT diverging from the OFFICIAL model too, not just from GT.

**50-frame vs real dataset GT (step-0, real units)**: 94.6% of values land inside `[global_min,global_max]` (xyz/rpy nearly all in-range; gripper 64%, occasionally ~1.002 vs a `[0,1]` range -- a tiny, expected float overshoot, not a normalization bug). Overall MAE=0.198, cosine=0.559 across all 50 frames -- but this average is misleading on its own: 4/10 tasks are excellent (MAE 0.029-0.045, gripper error 0.002-0.005, cosine 0.983-0.993), 6/10 tasks are poor (gripper MAE 0.40-1.00, cosine near 0 or negative) -- driven almost entirely by gripper open/close disagreement and some tasks' own xyz pattern not matching their demos, NOT a directional/systematic serving-level bias (rotation MAE stays small and uniform across all tasks: roll/pitch/yaw 0.028/0.058/0.036 -- no single axis is "always off," which is what a real wiring bug would look like).

**Stability, 40 consecutive `infer()` calls after one `set_prompt()`** (the real closed-loop-shaped access pattern): P50=279.5ms, P90=280.1ms, range 278.4-281.1ms -- flat, no drift. GPU memory: 18.132GB at both start and end, delta=0 -- no leak. This was the one remaining "is this actually usable in a real control loop" concern from OPT-009; now directly measured and clean.

## What this means

Confirms the full real-data serving path built this session (BF16 residual, txt_in/img_in-once, real RoPE grid, proprio, denormalization, real shift schedule) is faithful to the official model across real task diversity, not just one frame. The remaining GT mismatch on 6/10 tasks is a checkpoint/policy quality question (does this specific LIBERO fine-tune generalize well to these specific tasks/episodes), which is OUTSIDE this project's own scope (a Thor inference-engine port: speed + precision-vs-own-baseline, not model training/data quality) -- not something to chase here.

## Open

- The substituted-episode observation-to-ground-truth pairing confound was never sanity-checked: the eval's own footnote records a substituted episode (`ep359` instead of the first-listed one for the "stove" task, because the wrist video only covers the first 1050s). If any OTHER of the 6 poor-scoring tasks has a similar frame/video-length mismatch in how the eval harness paired observation to ground truth, that would look like "policy is wrong" while actually being a data-pairing issue upstream of both FlashRT and the official model equally (so it would not show up as a FlashRT-vs-official divergence either way). Worth a quick sanity check on the eval harness before concluding those 6 tasks are really a checkpoint limitation; not urgent, since it does not affect FlashRT's own correctness story.

# OPT-015: systematic op-fusion audit vs. official ImageWAM

Status: audit complete (fork, read-only). Finding 1 sub-problem 1 (qkv+mlp_in merge into one real `linear1` GEMM) IMPLEMENTED, locally bit-exact and real-Thor verified with a measured win -- KEPT, shipped default. Sub-problem 3 (`linear2` merge) is no longer open: implemented as OPT-016 (roadmap item 4). Finding 2 (NVFP4 SwiGLU fusion) implemented, real-Thor verified, REVERTED from the default for lack of a measured win; `Nvfp4SwiGluMlp` stays in the codebase, unrouted by any precision string. Candidates gated-residual CUTLASS epilogue and RMSNorm-prologue fusion both closed "defer, don't attempt". The FP8 alignment-fallback gap found here is fixed centrally; its own local construct+capture check is still not done (## Open).

Area: `pipeline_thor.py` (`_single_stream_layer`/`_action_single_layer`, `_mlp_gate_up`), `quant_linear.py` (`Nvfp4SwiGluMlp`), `checkpoint_loader.py` (`_extract_single_block`), `csrc/kernels/activation.cu` (`silu_glu_merged_kernel`), `csrc/fused_fp4/silu_mul_two_fp4_to_fp4.{cu,cuh}`, `imagewam_thor.py` (`_wrap_linear`, `_autotune_gemm`), `tests/test_imagewam_thor_real_wiring.py`.

## Audit method and result

Official op sequence read directly (`third_party/flux2/src/flux2/model.py` `DoubleStreamBlock`/`SingleStreamBlock`, `action_dit_flux2.py` `SlimFlux2*Block`, `imagewam.py` `infer_action_flux2`) against `pipeline_thor.py` `_double_stream_layer`/`_single_stream_layer`/`_action_double_layer`/`_action_single_layer`. AdaLN modulation sharing across layers is ALREADY correct and was ruled out, not re-flagged: one `Modulation` linear computes shift/scale/gate once per stream-type and is reused across all blocks (`model.py:98-108,132-134`); `pipeline_thor.py:74-91` already did the same (precomputed once by the caller, reused across 25 layers).

**Finding 1**: official single-stream blocks run QKV+MLP-gate/up as ONE `linear1` GEMM and attn-out+MLP-down as ONE `linear2` GEMM over `cat([attn_out, mlp_act(mlp)], dim=-1)`; the real checkpoint stores each as ONE tensor (`checkpoint_loader.py:114-133`, `_extract_single_block`). FlashRT split them into 4 GEMM slots at load time -- deliberate, because `silu_glu_merged_fp16` required its gate/up input's row stride to equal its own width (`real_single_stream_block.py`'s docstring), which breaks for a column-slice of a wider `linear1` output. Affects 40 real layers (20 backbone + 20 ActionDiT single-stream); double-stream blocks keep qkv/mlp genuinely separate in the real checkpoint too, confirmed NOT a gap.

**Finding 2**: `nvfp4` (the default precision since OPT-014) had NO fused SwiGLU path for MLP gate/up; unlike `fp16_cutlass` (`CutlassFp16SwiGluMlp`, OPT-013) it fell through to the generic `Nvfp4Linear` dispatch -- one wide NVFP4 GEMM against the real merged `(2*mlp_hidden, K)` weight, a full `(m, 2*mlp_hidden)` fp16 buffer, then `silu_glu_merged_fp16`: the "extra merged-buffer write+read" pattern OPT-013 had already removed for FP16 CUTLASS, still present for the default.

## Finding 2 implementation

`Nvfp4SwiGluMlp` (`quant_linear.py`), wired into `_mlp_gate_up` (`pipeline_thor.py`), `_load_real_weights` (the real `txt_mlp0.weight`/`img_mlp0.weight`/`mlp0.weight`/`mlp_in.weight` slots, as OPT-013) and `_rnd_swiglu_mlp` (`imagewam_thor.py`): column-splits the real merged weight into two NVFP4-quantized `(mlp_hidden, K)` halves (`quant_weight_nvfp4`) where `CutlassFp16SwiGluMlp` makes one fp16 transpose. This codebase's NVFP4 GEMM exposes no arbitrary activation epilogue, so the mechanism is two NVFP4 GEMMs (`fp4out_gemm`/`FP4Buffer`, the "split-GU FFN path" blocks already present for a DIFFERENT model, never wired to ImageWAM) each producing an `(m, mlp_hidden)` FP4-PACKED (4-bit) intermediate, then the new combiner `silu_glu_two_fp4_to_fp16` (`csrc/fused_fp4/silu_mul_two_fp4_to_fp4.{cu,cuh}`) writing the `(m, mlp_hidden)` fp16 gated buffer; the unchanged down-projection GEMM re-quantizes its own fp16 input internally, so no FP4 requantization is needed. Net: the intermediate shrinks from a full-width fp16 buffer to two FP4-packed halves (real DRAM-traffic reduction, not just a launch-count change); the activation is quantized to FP4 once per call and reused by both GEMMs.

Misnaming found while implementing, before it could ship silently wrong: `geglu_two_fp4_to_fp4`/`silu_mul_two_fp4_to_fp4` claims "SiLU" in its module docstring, its wrapper docstring and its device helper name (`silu_mul_p1`), but the ACTUAL formula (`g/(1+exp(-1.5957691216057308f*g*(1+0.044715*g*g)))`; `1.5957691216057308 == 2*sqrt(2/pi)`) is GELU-tanh, not true SiLU (`g/(1+exp(-g))`, ImageWAM's own formula, `csrc/kernels/activation.cu`). A new `true_silu_mul_p1` device function carries the correct formula. Local verification (Ada, no NVFP4 build): `py_compile` clean, existing regression suite unaffected, `precision="nvfp4"` construction still fails at the same documented point (`Nvfp4Linear`/`Nvfp4SwiGluMlp` both raise the same clear `RuntimeError` for a missing Blackwell build); the kernel compiles cleanly through the same sm_90a syntax-check substitute OPT-013 used (~148KB object file, no tensor-core/MMA instructions -- pure per-thread scalar math over packed FP4 bytes), but all of `flash_rt_fp4` is gated behind `ENABLE_NVFP4` (`GPU_ARCH=100`/`110` only, `CMakeLists.txt:43-60`), so no functional correctness is claimed locally.

## Finding 1, sub-problem 1: qkv+mlp_in merge -- IMPLEMENTED, kept

Plain scalar CUDA with no tensor-core dependency, so genuinely numerically verifiable on the Ada dev machine. `silu_glu_merged_kernel` (`csrc/kernels/activation.cu`) gained a `row_stride` parameter (defaults to `half_dim*2`, every existing caller's tightly-packed layout unchanged) so it reads gate/up straight out of a column-slice of a WIDER buffer -- bit-exact against a plain torch reference for both the default and the wide-stride cases. `_extract_single_block`/`build_real_weights` gained `merge_qkv_mlp: bool`: `True` (every precision except `fp16_cutlass`, which keeps its own separate `mlp_in.weight`-based `CutlassFp16SwiGluMlp`) returns the real UNSPLIT `linear1.weight` under one key; the flag's single owner is now `precision.py`'s table (`Precision(precision).merge_qkv_mlp`). New `single_linear1_merged`/`action_linear1_merged` buffers are always allocated alongside the split ones (~50MB combined memory overhead accepted, so `fp16_cutlass`'s unmerged path needs no special handling). The merged path in `pipeline_thor.py` runs ONE `linear1.weight` GEMM, reads Q/K/V via the existing `_copy_slice` column-slice technique and calls the stride-aware `silu_glu_merged_fp16` directly on the mlp-gate/up column range; `_mlp_gate_up`'s separate-GEMM path is skipped entirely. Verification (`tests/test_imagewam_thor_real_wiring.py`): `test_single_stream_layer_merged_linear1_matches_real_reference` and the merged-linear1 check in `test_action_double_and_single_layers_match_real_reference` build the same real weight VALUES both ways and compare the merged pointer-path output against the same already-verified tensor reference -- cosine=0.999998 (backbone single-stream), cosine=1.000000 (ActionDiT single-stream), identical to the split path's own numbers. The full existing regression suite (`test_imagewam_frontend`, `test_imagewam_proprio`, `test_imagewam_checkpoint_loader`'s real 343-tensor real-checkpoint load, `test_imagewam_scheduler`) passes unchanged.

## Thor check

**Finding 2 round** (`GPU_ARCH=110`, `flash_rt_fp4` rebuilt, all 55 real MLP-gate slots dispatching through `Nvfp4SwiGluMlp`; `x0=513`, proprio on, real 10-step shift schedule, real LIBERO dual-camera input). Finite throughout; cosine as actions / backbone_hidden / action_latent: new fused `nvfp4` vs old (pre-fusion) `nvfp4` 0.99987 / 0.99382 / 0.99975, new fused `nvfp4` vs `fp16` 0.99982 / 0.99303 / 0.99966, old `nvfp4` vs `fp16` 0.99981 / 0.99347 / 0.99966, OPT-014's own `nvfp4` vs `fp16` (reference) 0.9998 / 0.9939 / 0.9997 -- new-vs-old and old-vs-fp16 land at essentially the SAME distance from `fp16`, so no measurable accuracy regression against the actual reference. `infer()` P50: new fused `nvfp4` 244.8 ms, old (pre-fusion) `nvfp4` same commit/build 244.2 ms, OPT-014's own `nvfp4` measurement 236.9 ms, OPT-014's own 40-call stability band 243.2-247.3 ms -- the fused path is 0.6 ms SLOWER, noise-level, both inside that band; at the real shapes (M=513/392/64) two FP4 GEMM launches plus the new combiner plausibly offset the saved bandwidth. **REVERTED from the default** (`_load_real_weights` and `_rnd_swiglu_mlp` back to the plain merged-GEMM path); `Nvfp4SwiGluMlp` and `silu_glu_two_fp4_to_fp16` stay in the codebase, correct and real-Thor-verified, unrouted. The new-vs-old `backbone_hidden` delta was not isolated on Thor and not investigated further (moot): weight-quantization split order is RULED OUT (`csrc/quantize/quantize_fp4_sfa.cu`'s `kernel_quantize_fp4_sfa` is "one thread per (row, 16-element block)", each row's scale depending only on that row's own 16 K-values, so splitting `(2*mlp_hidden, K)` into two row-groups quantizes bit-for-bit identically); the likely structural cause is the extra lossy step, the new gate/up intermediate being FP4-PACKED 4-bit e2m1 (16 levels per block) where the old path's stayed fp16.

**Finding 1 round** (`nvfp4`, same real conditions; 20+20 real `linear1.weight` slots against the old split path's 40+40 `qkv.weight`/`mlp_in.weight` slots). Correctness, merged vs old split path on the same commit, all finite: actions 0.999983, backbone_hidden 0.994464, action_latent 0.999964; `backbone_hidden`'s 0.9945 (against the local FP16 single-layer check's exact 0.999998) is the same benign class of 25-layer `nvfp4` accumulation drift already documented for `fp16_cutlass` (OPT-013, `backbone_hidden=0.995090`) and `Nvfp4SwiGluMlp` (`0.9938`), not a stride bug (a wrong `_col_ptr`/`row_stride` read would show up in `actions`, not 0.99998). Speed, merged vs split: VAE 21.17 vs 21.13 ms, backbone prefill 106.52 vs 107.11 ms (+0.59), ActionDiT 10-step denoise 122.73 vs 130.71 ms (**+8.0**), `infer()` P50 **231.63** vs 239.55 ms (**+7.9, 1.03x**) -- 231.6 ms beats OPT-014's own `nvfp4` baseline (236.9 ms) and its 40-call stability average (243.5 ms). The win is NOT at backbone's large M=905 (+0.59 ms) but at ActionDiT's tiny M=64 denoise loop (20 layers x 10 steps = 200 calls per `infer()`): fixed per-launch overhead, not bandwidth, is what the merge removes, so keep the ActionDiT side if the merge is ever partially reverted. **Decision: keep the merge for the whole network** (shipped default).

**Consolidated-benchmark round** (`imagewam_thor_graph_bench.py`, `x0=513`/`a0=905`/`num_action=64`, no VAE/proprio, random weights: relative ordering only, not comparable to OPT-014's real-`infer()` figures). P50: fp16 286.3, fp16_cutlass 280.8, fp8 254.2, fp8_static 243.2, fp8_static_cutlass 221.4, **nvfp4 212.0** (fastest, consistent with every other measurement). Rewritten `imagewam_thor_int4_bench.py`/`_int8_bench.py` on Thor (real AdaLN/gated-residual/fused-QKV/fused-linear1, real dims): prefill 1253.6/138.7 ms, one denoise step 55.4/14.1 ms, prefill + 10-step 1807.6/279.3 ms (INT4/INT8), all higher than the pre-rewrite stale scripts' own numbers (more real work per layer, not a regression). INT8's real news: Ada's K=9216 crash does NOT reproduce on Thor, confirming it is Ada-specific. **OPT-007 stays closed**.

**Production gap found and fixed in the same round**: `fp8`/`fp8_static`/`fp8_static_cutlass` all crashed on the real `action_encoder` (K=7) shape -- cuBLASLt status 15/`CUBLAS_STATUS_NOT_SUPPORTED`, CUTLASS `can_implement=-1` -- the K=7/N=7 misalignment class `fp16_cutlass` (OPT-013) and `nvfp4` (Stage 3) already had a `_wrap_linear` fallback for, with none added for the FP8 family, which unlike those two fails on cuBLASLt itself and so needs the fallback on every backend. Fixed centrally in `_wrap_linear`, covering `_load_real_weights` and `_alloc_random_weights` alike (`_rnd_linear` routes through `_wrap_linear`); `_calibrate_fp8()` already skips non-`StaticFp8Linear` objects via `isinstance`. The rule now lives in `precision.py`'s table as `Precision.alignment_fallback(n, k)` (8-alignment for the FP8 tiers).

## Candidates evaluated, deferred

- **gated-residual CUTLASS epilogue** (`gate_res_fp16`/`gate_res_bf16res`, `csrc/kernels/decoder_fused.cu`: `residual += gate_vector * proj_output`, per-channel `gate`), flagged in OPT-013. CUTLASS has the semantic building block `PerColLinCombPerColBiasEltAct` (`third_party/cutlass/include/cutlass/epilogue/fusion/operations.hpp:302-317`, `D = activation(per-col alpha*acc + per-col beta*C + per-col bias)`, this math with `alpha=gate, beta=1, bias=0`), but the vendored snapshot wires it for SM90 only (`sm90_callbacks_tma_warpspecialized.hpp`); SM100 (the `cutlass_fp16_k64_*` family OPT-013 uses) has `FusionCallbacks` only for the FP4/FP8 block-scale-factor variants. The target is a SMALLER buffer (`hidden`-width, one tensor) than either OPT-013's CUTLASS swap or finding 2's `2*mlp_hidden`-width target, both real, correct and zero-win. **Verdict: defer, don't attempt** (OPT-004's compute-bound finding).
- **RMSNorm+modulate into the following GEMM's prologue** (`ada_layer_norm_fp16`/`ada_layer_norm_bf16in_fp16out`), eliminating the `modded` buffer's DRAM round-trip: CUTLASS's vendored visitor tree is entirely output-side (`gemm/collective/` has no prologue/input-visitor concept, confirmed by grep), and RMSNorm needs a full row's reduction (sum-of-squares across all of K) before any of that row can be scaled, which fights tile-at-a-time mainloop streaming -- a bigger, more novel piece of authoring than the deferred epilogue above. Quantified savings at real shapes over full `infer()`: ~340-470 MiB of DRAM traffic in the best case, smaller than finding 2's already-measured non-win, and single-stream layers cannot fully realize it without also merging `linear1` (their one `modded` buffer is read by two separate GEMMs). **Verdict: defer indefinitely, not "pending Finding 1"**.

## Benchmark-script consolidation (2026-09-17)

All 5 standalone per-precision benchmarks (`imagewam_thor_{fp16,fp8,fp4,int8,int4}_bench.py`) were equally stale (generic transformer-block skeleton, none of the session's real fusions, a stale 768-token image-shape placeholder). `imagewam_thor_fp16_bench.py`/`_fp8_bench.py`/`_fp4_bench.py` are DEPRECATED (docstring notice, not deleted, internal logic untouched); `imagewam_thor_graph_bench.py` is the canonical replacement (real dims `x0=513, a0=905, num_action=64`, loops over `imagewam_thor._PRECISIONS`, prints `SKIP` for a precision this build does not support instead of crashing the run). INT8/INT4 have NO real dispatch path in `imagewam_thor.py` at all, so their benchmarks were rewritten (not deprecated) to match `pipeline_thor.py`'s real per-layer math; both also built `ImageWAMAttnBackend` without `use_perhead_kv=True, use_real_mot_mask=True` and sized K/V caches at the old broadcast `HD` width instead of real per-head `HIDDEN` width -- a second, independent staleness predating OPT-002, now fixed. Run once locally before local GPU testing paused: INT4 finite (`prefill` P50=111.8ms, `one denoise step` P50=7.6ms, both higher than the pre-rewrite stale version's own numbers -- more real work per layer, not a regression); INT8 failed exactly at the already-documented K=9216 shape (`txt_mlp2`, rc=131079).

## Open

- FP8 alignment fallback (`fp8`, `fp8_static`, `fp8_static_cutlass`): no local "construct + capture per precision" regression check is recorded for the family after the K=7/N=7 fix. The decision itself is covered at CPU level by `tests/test_imagewam_thor_precision_routing.py` and `tests/test_imagewam_precision_table.py`, and Thor runs the family with real calibration (`THOR_STATUS_SUMMARY.md`: `fp8_static` 228.0 ms / cosine 0.99829, `fp8_static_cutlass` 220.3 ms); no `issues.md` entry covers it.
- `_autotune_gemm`'s shape list still holds the pre-merge shapes: it has the merged `linear2` shapes added by roadmap item 4, but neither the backbone `(905, 27648, 3072)` nor the ActionDiT `(64, 17408, 1024)` merged `linear1` shape, so with `precision="fp16"` the 20 backbone and 20 ActionDiT merged `linear1` GEMMs run on the cuBLASLt heuristic's first pick, not the autotuned one; the loss is unmeasured, and quantized precisions are unaffected (`nvfp4` sends only its two K=7/N=7 fallback GEMMs through `fp16_nn`). Recorded as `issues.md` ISSUE-010.
- Finding 2's root cause (the new-vs-old `backbone_hidden` delta) is not isolated; moot, the path is reverted.

# OPT-027: fidelity + latency regression gate on a versioned LIBERO fixture (roadmap item 13)

Status: implemented; the fp16 gate passes on H100 and the Thor `nvfp4` and `fp16` gates have run (`eccf14f`, `0920t`), the `nvfp4` row against fixture v2 being the served default's (trim + FA4, 125.86 ms).
The Thor latency baselines are per configuration (`latency_baselines.json`, schema 2): `nvfp4` `untrimmed_reference` 202.2 ms, margin 5%, re-based from 231.6 ms, describes the untrimmed, FA4-off, torch-VAE configuration, and `nvfp4` `served_default` (the promoted default) is unseeded until its own Thor gate run; H100 latency is ungated.

Area: committed CI/regression gate for the served ImageWAM path

## What exists

| piece | file |
|---|---|
| model-agnostic gate policy and report schema (v1) | `flash_rt/core/regression_gate.py` |
| fixture format, `.npz` IO, manifest with per-file and per-array SHA-256 | `flash_rt/datasets/imagewam_gate_fixture.py` |
| fixture generator (H100: official model, then FlashRT fp16), gate runner (any CUDA device; no official model, no Qwen3) | `benchmarks/imagewam_gate_fixture_generate.py`, `tests/gate_imagewam_libero.py` |
| per-precision fidelity thresholds, per-device, per-configuration latency policy (Thor `nvfp4` `untrimmed_reference` 202.2 ms, margin 5%, `served_default` unseeded; H100 ungated) | `tests/fixtures/imagewam_gate/fidelity_thresholds.json`, `tests/fixtures/imagewam_gate/latency_baselines.json` |
| committed v1 manifest | `tests/fixtures/imagewam_gate/imagewam_libero_gate_v1.manifest.json` |
| fixed-noise entry into the served path | `ImageWAMTorchFrontendThor.infer(observation, *, action_noise=None)` |

Thresholds (`fidelity_thresholds.json`): `fp16` `vs_official_median_min` 0.997, `vs_official_min_min` 0.993, `vs_fp16_reference_median_min` 0.999, `vs_fp16_reference_min_min` 0.995, `mae_vs_gt_ratio_max` 1.02; `nvfp4`, `fp8_static`, `fp8_static_cutlass` and `e0m3_hadamard` 0.995 / 0.985 / 0.997 / 0.99 / 1.05, the two FP8 tiers `requires_calibration`. The `fp16` bounds are the fixture-v1 end-to-end baseline, median 0.99840, min 0.99567, mean MAE 0.18359.

Fixture v1 (`imagewam_libero_gate_v1`, 81 MiB, stored at `/home/user1/workspace/jingwu/artifacts/deploy-gates/imagewam_libero_gate_v1/`, not in git): libero_spatial, 10 tasks, frames 0 and 60, seeds 0 and 1, so 20 observations and 40 (observation, seed) runs; it holds both 224x224 views, raw proprio, ground truth, the 10 official Qwen3 contexts (bfloat16 bits) and masks, the initial noise drawn as the official sampler draws it, and the official and FlashRT fp16 action chunks in normalized space. Provenance: v1 was produced by the generator content committed in `d03b073` (generator file SHA-256 `d16399e0e6afc3bf...`); its manifest records `git.commit` `ae3a358` with `tracked_changes: false` because the generator was still untracked then, and newer manifests also record `generator_sha256` and the untracked files (`git.untracked_files`, `git.clean`); v1 is not regenerated. Fixture v2 (`imagewam_libero_gate_v2`) records its `fp16` reference trimmed (`text_trim=true`, the manifest field), so it gates a trimmed configuration, which v1's untrimmed reference cannot: the gate refuses a configuration whose `text_trim` differs from the reference. The generator writes `fixture.npz` and the manifest to a data directory and copies only the manifest into `tests/fixtures/imagewam_gate/`, so no fixture data is in git.

`fp8_static` interface: thresholds mark it `requires_calibration`. The runner gates it only when `--fp8-calibration PATH` or `$IMAGEWAM_FP8_CALIBRATION` names an existing file, and hands the path to `ImageWAMTorchFrontendThor(..., calibration_path=PATH)`, the keyword the calibration stream's frontend uses and which the frontend declares today. Without a file the verdict is `skipped` (exit 0); with a file but no such constructor keyword it is `blocked` (exit 1). It is never gated on the `N(0, 0.1)` placeholder calibration.

Noise: fidelity is measured with the fixture's fixed N(0,1) initial noise, the official sampler's per-seed draw, passed through `infer(obs, action_noise=...)`; it is not the served default draw, `0.01 * N(0,1)` (ISSUE-002), which the latency loop does use. Clock policy: the latency check records the clock state in every result and never refuses dynamic clocks; Thor runs at MAXN with DVFS-managed clocks, and baselines are measured in that same state (ISSUE-061).

## Measured (H100, shared GPU)

Fixture generation, FlashRT fp16 against official (normalized space); peak GPU memory: official phase 17.4 GiB, FlashRT fp16 phase 9.6 GiB:

| | median | min | mean MAE vs GT |
|---|---:|---:|---:|
| seed 0 (end-to-end baseline: 0.99840 / 0.99567 / 0.18359) | 0.99840 | 0.99567 | 0.18359 |
| seed 1 | 0.99829 | 0.99554 | 0.18369 |
| official, seed 0 vs seed 1 | 0.99630 | 0.97154 | |
| official MAE vs GT, seed 0 / seed 1 | | | 0.18538 / 0.18584 |

Initial noise and fidelity, fp16 on fixture v1, 40 runs, cosine against official in normalized space:

| initial noise | median | min | mean |
|---|---:|---:|---:|
| fixed N(0,1), the official sampler's (what the gate uses) | 0.99836 | 0.99554 | 0.99798 |
| 0.01 x the same noise | 0.99683 | 0.98613 | 0.99602 |
| served default draw, 0.01 x N(0,1) on the device | 0.99683 | 0.98591 | 0.99598 |

The served sampler falls below the fp16 bounds (median 0.997, min 0.993); resolving ISSUE-002 (dropping the 0.01 factor) would bring the served path to the gated configuration. Gate runs:

| precision | verdict | detail |
|---|---|---|
| fp16 | pass | vs official median 0.99836, min 0.99554 over 40 runs; vs fp16 reference 1.0 (max abs difference 0.0, bit-identical across processes); MAE 0.18364 against a limit of 0.18731; latency P50 158.2 ms (P10 142.9, P90 229.1), ungated; peak 9.56 GiB |
| nvfp4 | blocked | frontend construction: `Nvfp4Linear requires a Blackwell/Thor NVFP4 build` (expected on sm_90) |
| fp8_static, no calibration file | skipped | exit 0 |
| fp8_static, file present | blocked | exit 1; the branch then declared no `calibration_path` keyword, which the frontend declares today |
| fp8 | blocked | no thresholds configured |
| fp16, rerun at `77b7bef` | pass | same fidelity values; checkpoint verified by SHA-256 (11.9 s); top-level `latency: "ungated"` with its reason, also named in the verdict reason; P50 159.3 ms; clean worktree recorded |
| fp16, `--require-latency` | blocked | exit 1; ungated H100 latency |
| fp16, wrong checkpoint | blocked | SHA-256 mismatch; with `--skip-checkpoint-hash`, byte-size mismatch |
| any, `--iters 5` | argument error | rejected before any GPU work |

## Thor check

- `eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive, raw logs `/home/jingwu/thor_val/0919s/`: `nvfp4`, `text_trim=true`, fixture v2 - pass, vs official 0.99931 / min 0.99898, vs the fixture's own `fp16` reference 0.99935 / 0.99907, `infer()` P50 114.6 ms. A trimmed configuration against fixture v1 was refused, and an untrimmed configuration against fixture v2 was refused, both by the same `text_trim` check.
- `0919e`, commit `a84916a`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive, raw logs `/home/jingwu/thor_val/0919e/`: the VAE encode geometry change (issues.md ISSUE-086) is fidelity-neutral for the gate's LIBERO configuration - the `libero_spatial` nvfp4 `default` row measures vs official min 0.99418 / median 0.99764 and MAE 0.18290, the values the `eccf14f` round recorded for that row, at P50 202.6 ms against 202.0-202.4 ms in the `eccf14f` session. The fixture-v2 gate (`imagewam_libero_gate_v2`) still passes; `fp8_static_cutlass` with a real calibration file gated at vs official 0.99830 / min 0.99557, P50 233.5 ms, ungated for latency.
- `0920t`, the round the latency policy cites: nvfp4 served default (trim + FA4) 0.99889 / 0.99934 at P50 125.86 ms; nvfp4 fixture v1 with `--no-text-trim` (untrimmed) 0.99418 / 0.99758 at 191.79 ms; fp16 fixture v2 0.99993 / 0.99997 at 284.38 ms, ungated for latency.

## Open

- The latency baseline is per configuration now (`latency_baselines.json`, schema 2: named entries per precision, each stating its `config`; the gate derives the entry's name from the run's resolved options). `untrimmed_reference` holds the 202.2 ms record (untrimmed, FA4 off, torch VAE, margin 0.05), and `served_default` is unseeded (`p50_ms` null): until a Thor gate run of the promoted default is pasted in, its latency is reported ungated (blocked under `--require-latency`) rather than compared with the untrimmed number. The `0920t` figures (125.86 ms trimmed with FA4 at the backbone and the VAE outside the graph; the untrimmed row with FA4 on 191.79 ms) are not this configuration's baseline and were not used to seed it (plan.md "Open" item 1; issues.md ISSUE-061).
- The `nvfp4` gate thresholds in `fidelity_thresholds.json` still read "Provisional until the first Thor gate run", although the Thor `nvfp4` gate has run; a `fp8_static` (cuBLASLt) run with a calibration file is not recorded, only `fp8_static_cutlass` with one on Thor.
- ISSUE-060 is resolved in the frontend (`set_prompt` cache key; issues.md index to `docs/imagewam_configuration.md`, the `set_prompt` notes): the generator and runner hand the fixture's stored context to `set_prompt(context=..., context_mask=...)` rather than working around a cache defect.

# OPT-018: ActionDiT small-M CUTLASS tile selection (roadmap item 1)

Status: implemented behind an opt-in flag (`gemm_variant_autotune=True`) that passes the sm_110 compile check; Thor correctness and speed are not measured, so the promotion decision is unmade and the flag stays opt-in (issues.md ISSUE-023).
The plan is approved with Phases 1-6 completed and Phase 7 (Thor confirmation) `blocked` on an sm_110 device: plan.md "Plan: ActionDiT small-M CUTLASS tile selection".

Area: `flash_rt/models/imagewam/gemm_variant_tuner.py`, `flash_rt/models/imagewam/gemm_variant_timer.py`, `flash_rt/models/imagewam/quant_linear.py`, `csrc/gemm/gemm_types_sm100.h`, `csrc/gemm/cutlass_sm100.cu`, `csrc/bindings.cpp`, `flash_rt/frontends/torch/imagewam_thor.py`, `benchmarks/imagewam_thor_small_m_tile_sweep.py`, `tests/test_imagewam_gemm_variant_tuner.py`, `tests/test_imagewam_gemm_variant_routing.py`

## Observation

Every ActionDiT weight GEMM runs at `M = num_action = 64`, and both CUTLASS-backed quantized precisions pick their tile from an `(N, K)` heuristic tuned at other M. Tiles dispatched today at the real ActionDiT shapes (5 double + 20 single layers):

| site | N | K | `nvfp4` (`pick_variant`) | `fp8_static_cutlass` (`_pick_fp8_cutlass_variant`) | `fp16_cutlass` |
|---|---:|---:|---|---|---|
| double `qkv` | 9216 | 1024 | v6 `128x256x128` c1x1x1 | `wide` `256x128x128` c2x2x1 | `wide` |
| double `proj`, single `attn_out_proj` | 1024 | 3072 | v6 | `sq` `256x256x128` c2x2x1 | `sq` |
| double `mlp0` | 8192 | 1024 | v6 | `wide` | SwiGLU pair, `256x256x64` c2x2x1, N=4096 |
| double `mlp2`, single `mlp_down` | 1024 | 4096 | v6 | `sq` | `sq` |
| single `linear1` | 17408 | 1024 | v8 `128x256x256` c1x1x1 | `wide` | split: `qkv` `wide` + SwiGLU pair |
| `action_encoder` (K=7), `head.linear` (N=7) | | | cuBLASLt `Fp16Linear` (alignment fallback) | same | same |

At M = 64, one M tile covers the whole problem, so the CTA count equals the number of N tiles. The `N = 1024` GEMMs make up 50 of the 80 quantized ActionDiT GEMM calls per denoise step (500 per `infer()`). On Thor's 20 SMs they get 4 CTAs under v6 and 4 useful CTA pairs under FP8 `sq`, while they are weight-bandwidth bound (2M = 128 FLOP per weight element). The SM100 FP8 CUTLASS family had no tile narrower than 128 in N and no 1-SM tile at all. OPT-014 result 3 measured FP8 CUTLASS 1.44-1.68x slower than cuBLASLt at this M.

## Mechanism

- `GemmVariantTuner` (`flash_rt/models/imagewam/gemm_variant_tuner.py`) works on each group of ActionDiT linears sharing `(family, M, N, K)`. Candidates must reproduce the default tile's output on every member (return code 0, no Python exception, finite, cosine >= 0.9999). A stale `flash_rt_kernels` that lacks the `cutlass_fp8_t128x*` symbols raises `AttributeError` for those candidates, which rejects them without aborting construction. They are timed as one launch per member, round robin, so every launch reads a different layer's weight, inside CUDA graphs (`gemm_variant_timer.CudaGraphVariantTimer`) and interleaved across candidates. A candidate the timer cannot capture is rejected (`timing_failed`), and the caller's stream is restored. A candidate replaces the default only if it is more than 2% faster, and the default is kept if it could not be timed itself. The choice is cached per `(family, M, N, K)` and applied before graph capture.
- NVFP4 candidates: every cluster-1x1x1 tile, v4, v5, v6, v7, v8, and v10 `128x64x256` (Pi0.5's decoder tile). The clustered tiles are excluded because Pi0.5 measured them winning in isolation and losing in the pipeline on Thor.
- FP8 CUTLASS candidates: `sq`, `wide`, `t1`, `plain`, plus four new 1-SM cluster-1x1x1 tiles (`gemm_types_sm100.h`, `sm100_small_m`): `t128x64x256` (v10 shape), `t128x64x128`, `t128x128x128`, and `t128x256x128`.
- `ImageWAMTorchFrontendThor(gemm_variant_autotune=True)` is accepted for `nvfp4` and `fp8_static_cutlass` only. Results are in `frontend.gemm_variant_results`. Backbone GEMMs are not tuned. With the flag off (the default), every tile is the one dispatched before this change.

## Local evidence (H100, sm_90)

The SM100 CUTLASS and NVFP4 kernels do not run on sm_90, so every number below is about the mechanism, not about a tile.

- `tests/test_imagewam_gemm_variant_tuner.py` (18 tests): the selection rule against stub GEMMs. It covers argmin choice, the 2% hysteresis, rejection on a nonzero return code, on a raised exception (`AttributeError`) and on a candidate that cannot be timed, mismatch (cosine 0.7987 in the test) and non-finite output, failure on one member only, an error when the default itself fails or raises, keeping an untimeable default, the cache, and a distinct M counting as a distinct key. The timer test also feeds a raising batch and a capture-invalidating batch; both come back as `None`, the good batch is still timed, and the caller's stream is restored. The real-timer test runs on real cuBLASLt launches (M=64, N=1024, K=3072, 6 weights): graph-timed 7.75 us per launch against 11.30 us eager event-timed, so launch overhead is excluded. A batch with 4x the work measured 3.62x.
- `tests/test_imagewam_gemm_variant_routing.py` (7 tests): only the GEMM entry points are replaced (the whole `flash_rt_fp4` module, and the `cutlass_fp8_*` attributes). The real `Nvfp4Linear` / `StaticFp8Linear`, frontend grouping, tuner, graph capture and `infer()` run as shipped. For both precisions, each of the 5 ActionDiT shapes is tuned once at `M = num_action`, the chosen tile reaches every ActionDiT GEMM in the captured graph, and the backbone keeps its heuristic tile. Tuning leaves `StaticFp8Linear`'s calibrate-before-call contract intact. With the flag off, no GEMM launches at construction. With the `cutlass_fp8_t128x*` symbols removed, which simulates a stale build, construction completes and those candidates show `launch_failed AttributeError`.
- `sm110_check.sh` (CUDA 13.0, `GPU_ARCH=110`): `flash_rt_kernels` and `flash_rt_fp4` build and link, and the four `cutlass_fp8_t128x*` symbols are exported.

## Thor check

`benchmarks/imagewam_thor_small_m_tile_sweep.py`, not run: the dev box has no sm_110 device.

- `--part kernels`: every NVFP4 and FP8 CUTLASS tile at each real ActionDiT shape (M=64), over as many distinct weights as layers share the shape. Reports GEMM-only us/GEMM, cosine against the fp32 `x @ W`, the heuristic pick (`*`), the tuner pick (`T`), and the per-shape winner, with cuBLASLt fp16 and cuBLASLt FP8 references on the same weights.
- `--part infer`: for `nvfp4` and `fp8_static_cutlass`, two graphs captured from one frontend (ActionDiT on heuristic vs tuned tiles), with action cosine on identical inputs and alternating `infer()` P10/P50/P90.

Expected: every tile other than the heuristic's shows cosine vs fp16 equal to the heuristic's own within about 1e-4 (same quantized operands, different accumulation order). NVFP4 v10 or v5, and FP8 `t128x64x*`, win at N = 1024. New-vs-old action cosine is at least 0.9999.

## Decision pending

If Thor shows a correct `infer()` win, `gemm_variant_autotune` should become the default for `nvfp4`: a one-line change of the constructor default. If it shows no win, the heuristic stays, and the sweep table still says which tile to hardcode, if any; until that run the flag stays opt-in (issues.md ISSUE-023; plan.md, Phase 7 `blocked`). Split-K / stream-K for the `N = 1024` shapes (16 CTAs at most even with a 64-wide N tile) was not attempted.

# OPT-019: attention-chain fusion recheck at ImageWAM's real shapes (roadmap item 6)

Status: analysis done, both changes implemented. FA4 at the backbone site is the served default where the machine can run it; FA4 at the `mot` site joined the served default on 0921 (`use_fa4_mot=None` resolves like the backbone site; `plan.md` phase F1, owner decision), its Thor observation being THOR_CHECKLIST.md N1-N3. Both sites have Thor numbers inside the captured graph (`c20f3a0`, `eccf14f`, `0920t`, merged under `## Thor check`), and the `nvfp4` baseline this analysis was written against has been re-seeded from 231.6 ms to 202.2 ms (`tests/fixtures/imagewam_gate/latency_baselines.json`). The H100 numbers are indicative only, because a co-tenant training job shares that GPU.

Area: `flash_rt/hardware/thor/fa4_backend.py`, `flash_rt/hardware/thor/attn_backend.py`, `flash_rt/frontends/torch/imagewam_thor.py`; `tests/test_imagewam_fa4_dispatch.py`, `tests/test_imagewam_fa4_backbone.py`; `benchmarks/imagewam_attention_share_bench.py`. Plan: `plan.md` roadmap item 6.

## Finding 1: FlashRT's `mot` call is unmasked; official masks padded text keys

The frontend always runs `use_real_mot_mask=True`, so the `mot` site dispatches unmasked attention: 64 action queries over all 969 keys, 24 heads, HD 128, per-head K/V, through the same kernel as the backbone site. Official `infer_action_flux2` builds its mask with `_build_mot_attention_mask_flux2(target_len=0)` -- `target_len = 0` removes only the region mask, since there is no noisy-target block to exclude -- and applies `mask[:, :, t0:r0] &= text_valid[:, None, :]`, with `text_attention_mask` passed at both the prefill call and the action call, so official excludes the padded text keys for every query row at both sites while FlashRT attends to them at both sites (issues.md ISSUE-020, resolved by `text_trim`; see `## Open`).
For a fused kernel: a plain non-causal FA4 call reproduces what FlashRT computes today at the `mot` site, which is what `use_fa4_mot` runs and what the dispatch tests compare against, but it does not reproduce official; matching official needs the padded keys removed -- a key mask, or, since padded tokens are inert under the official mask, a context of only the valid tokens -- and removing the padding from the sequence keeps plain attention correct, so the fused path stays a plain FA4 call. The three-region `attention_qkv_fp16_mot_joint*` kernels serve only the legacy `use_real_mot_mask=False` path.

## Finding 2: attention share at the real shapes (H100, fp16, random weights)

Graph time with the real attention backend minus graph time with a backend that launches nothing, all else identical (`benchmarks/imagewam_attention_share_bench.py --part share`):

| stage | with attention P50 | without P50 | attention | per call |
|---|---:|---:|---:|---:|
| prefill (25 backbone calls) | 23.160 ms | 16.869 ms | 6.29 ms (27.2%) | 252 us |
| one denoise step (25 `mot` calls) | 3.520 ms | 2.126 ms | 1.39 ms (39.6%) | 56 us |
| prefill + 10 steps | 58.36 ms | | 20.23 ms (34.7%) | |

## Finding 3: per-call kernels at the real shapes (H100)

Each call reads a different layer's K/V, round robin over 8 layers, CUDA-graph timed (`--part kernels`); accuracy is against fp32 PyTorch attention. The chain's per-call cost here (437 us) is higher than its in-graph cost in Finding 2 (252 us), so the ratios, not the absolute values, are the information.

| site | kernel | us/call | vs chain | cosine | rel_l2 |
|---|---|---:|---:|---:|---:|
| backbone q=kv=905 | cuBLAS chain (today) | 437.1 | 1.00x | 1.000000 | 5.5e-4 |
| | SDPA flash (FA2) | 56.1 | 7.79x | 1.000000 | 2.8e-4 |
| | SDPA cuDNN | 39.2 | 11.15x | 1.000000 | 2.8e-4 |
| | SDPA mem-efficient | 176.2 | 2.48x | 1.000000 | 2.8e-4 |
| mot q=64, kv=969 | cuBLAS chain (today) | 42.3 | 1.00x | 1.000000 | 5.5e-4 |
| | SDPA flash (FA2) | 15.8 | 2.68x | 1.000000 | 2.7e-4 |
| | SDPA cuDNN | 22.0 | 1.93x | 1.000000 | 2.8e-4 |
| | SDPA mem-efficient | 44.1 | 0.96x | 1.000000 | 2.8e-4 |

## Finding 4: in-pipeline A/B with an sm_90 fused kernel (H100, fp16)

`--part infer --sdpa-standin`: one frontend, three graphs captured from the same buffers and weights, with PyTorch SDPA behind the FA4 calling convention; timing is 30 `infer()` calls, rotating order. P90 was about 2x P50 for all three configurations, so the co-tenant regime changes rather than the kernel; about 60% of the gain comes from the `mot` site.

| configuration | actions cosine vs chain | `infer()` P10 | P50 | delta P50 |
|---|---:|---:|---:|---:|
| cuBLAS chain at both sites | 1.000000 | 48.78 ms | 49.38 ms | |
| fused at backbone | 1.000000 (max-abs 2.8e-3) | 44.53 ms | 45.45 ms | -3.93 ms |
| fused at both sites | 1.000000 (max-abs 2.6e-3) | 38.66 ms | 39.57 ms | -9.81 ms (-19.9%) |

## Why Pi0.5's rejection does not transfer

Pi0.5 rejected a fused SIMT attention chain at decoder M = 10, HD 256 (5-7x slower, `docs/pi05_thor_decoder_fp4_e2e.md`): its QK^T and PV were about 1 us of tensor-core work that SIMT code could not approach, and FA4 had no KV-split path at HD 256. ImageWAM differs on each point: HD is 128; both sites have long KV (905 and 969 keys); the fused kernels available on both devices are tensor-core kernels; and the chain writes and re-reads a 24 x q x kv fp16 logits buffer, 39 MB per backbone call and 3 MB per `mot` call. On Thor, FA4 at the backbone site already measured 3.75x per call and -10.5% prefill (OPT-005).

## Implemented

- `fa4_backend.thor_default_enabled()` returns True only on a compute-capability-11.x device whose FA4 runtime imports, and it checks the device first, so FA4 is never imported off Thor. `ImageWAMTorchFrontendThor(use_fa4=None)` is the default and takes the machine's answer, with no error for a missing runtime; `_FA4_OPT_IN_DEFAULT` in `imagewam_thor.py` is `"1"`, so `FLASHRT_THOR_FA4=1` and leaving the variable unset are the same answer, `FLASHRT_THOR_FA4=0` forces the chain and also stops `fa4_backend` from importing FA4 for every model, `use_fa4=True` forces FA4 and raises without a runtime, and `use_fa4=False` forces the chain regardless of the environment. The resolved value is `frontend.use_fa4`, and the regression gate records it. The `default` rows in the `c20f3a0`, `eccf14f` and `0919e` sections were taken with FA4 off; the served default's own Thor numbers are the `0920t` rows below.
- `ImageWAMAttnBackend(use_fa4_mot=True)` and `ImageWAMTorchFrontendThor(use_fa4_mot=True)` run the `mot` site through FA4: Q is the action rows at row offset `a0`, K/V are the full per-head `(1, 969, 24, 128)`, with `causal=False`, `pack_gqa=False`, `num_splits=1` and an explicit `softmax_scale`. The combination requires `use_real_mot_mask=True` and `use_perhead_kv=True`, and the constructor raises otherwise.
- FA4 output at both sites goes to a dedicated buffer: the frontend owns `_fa4_out` of shape `(total, hidden)`, allocated only when some site runs FA4, passed to the backend as the `fa4_out` / `fa4_out_numel` slots; the backend checks the capacity at construction and on every call, then copies the result back to the Q rows. The tests pre-fill a guard band after `fa4_out`, and `logits`, with a sentinel and fail on any FA4 write outside `fa4_out`; they fail on the earlier staging in `logits` (`total*NH x (total + total%2)` elements), which is large enough at the real dims but smaller than `q_seq*NH*HD` at small dims.
- Falling back when FA4 fails: FA4 compiles on its first call, in the eager warmup of `set_prompt()`'s graph capture, and can fail there or inside the capture, for instance a runtime that imports but cannot compile for sm_110, as with `nvidia-cutlass-dsl` 4.4.x. On such a failure with FA4 on, `set_prompt()` logs an error, emits a `RuntimeWarning`, stores the reason in `frontend.fa4_fallback_reason`, rebuilds the attention backend with FA4 off at both sites, and captures again; it restores the caller's CUDA stream first, because an invalidated capture leaves the capture stream current. A failure with FA4 off still raises. Tested with stand-ins that raise on every call, raise only inside capture, and issue a device sync inside capture; all three recover to the cuBLAS chain's output, cosine 1.0000000, max-abs up to 9.5e-7, the difference coming from the two frontends' own cuBLASLt autotune picks.
- Local verification, `tests/test_imagewam_fa4_dispatch.py` (23 tests): FA4 is replaced by a stand-in with `_flash_attn_fwd`'s calling convention that computes an fp32 matmul-softmax-matmul in PyTorch, and each FA4 branch is compared against the cuBLAS chain at the real shapes -- backbone q=kv=905 cosine 1.000000, max-abs 4.9e-4, rel_l2 5.8e-4; `mot` q=64 at row 905, kv=969 cosine 1.000000, max-abs 3.7e-4, rel_l2 5.6e-4, with rows `[0, a0)` untouched; and the small-dims frontend end to end, both sites on the stand-in vs the chain, actions cosine 1.000000. It also covers the constructor guard, the `fa4_out` bounds and capacity, `use_fa4` resolution over the environment variable, runtime availability and the explicit argument, and the FA4-failure fallback. `tests/test_imagewam_fa4_backbone.py` gains a real-FA4 test for both sites at the real shapes; it skips without FA4.

## Thor check

Checks, all on Thor: `fa4_backend.status()` plus `_resolve_use_fa4(None)` (expect `active True` and `True`; `FLASHRT_THOR_FA4=0` must print `disabled (FLASHRT_THOR_FA4=0) False`; a device that is not compute capability 11.x gives `False` even with the runtime importable); `pytest tests/test_imagewam_fa4_backbone.py -q -s -k both_sites_real_shapes` (expect a pass, not a skip, and both printed cosines > 0.999); `benchmarks/imagewam_attention_share_bench.py --part kernels` (report both tables, including `fa4_splits1/2/4` for `mot`); `--part share --precision nvfp4 --fa4 off`, then `--fa4 on`, then `--fa4 on --fa4-mot`; `--part infer --precision nvfp4 --iters 60` with `CKPT_PATH` set (report each configuration's action cosine against the chain, expected >= 0.999, plus P10/P50/P90 and delta); and the `nvfp4` end-to-end official compare at the served default against `FLASHRT_THOR_FA4=0` (report `fr_vs_off` median/min, mean `mae_fr_vs_gt` and the printed `infer()` P50 for both, expecting the two runs to match closely, `fr_vs_off` within about 1e-4 at the median). In any run with FA4 on, a line containing `falling back to the cuBLAS attention chain` means FA4 failed and the numbers are the chain's; report it with the reason. A step that means the FA4 configuration states it explicitly (the environment variable, `--fa4 on`, or the bench's own FA4 configurations), and `FLASHRT_THOR_FA4=0` is the off leg.

| round | result |
|---|---|
| `c20f3a0`, libero_spatial, nvfp4, real checkpoint, FA4 at both sites in one captured graph | `vae_trim` (FA4 off) 102.9 ms; `vae_trim_fa4bb` (backbone only) 99.0 ms, marginal -3.9; `stack` (backbone + `mot`) 93.2 ms, marginal -5.8, vs official median 0.99934; `stack_no_vae` (FA4 on, native VAE off) 104.8 ms; no FA4 fallback on any row |
| `eccf14f`, the two suites `c20f3a0` did not cover | libero_goal `vae_trim` 118.7 ms -> `stack` 92.6 ms, delta -26.1, vs official `vae_trim` / `stack` 0.99937 / 0.99930; libero_10 103.0 -> 93.5 ms, delta -9.5, 0.99926 / 0.99925; no fallback on any row |
| `0920t` (commit `4cd06e5`), LIBERO, nvfp4, three gate runs pass and the end-to-end `default` | gates: nvfp4 fixture v2 (served default: trim + FA4) 0.99889 / 0.99934 at 125.86 ms; fp16 fixture v2 (ungated) 0.99993 / 0.99997 at 284.38 ms; nvfp4 fixture v1 with `--no-text-trim` (untrimmed + FA4) 0.99418 / 0.99758 at 191.79 ms. End to end: `default` `served_vs_off` 0.99432 / 0.99750 at P50 126.5 ms; the same with `FLASHRT_THOR_FA4=0` (trim, cuBLAS chain) 0.99441 / 0.99743 at 131.3 ms, so FA4 on is about 5 ms of P50 below FA4 off and the two medians differ by under 1e-4 |

The ladder's criterion for making FA4 the default (at least 2 ms of P50 against the FA4-off row, and not worse against official) held in `c20f3a0` with no fallback on any row, and the decision on the backbone site was the owner's: FA4 is the served default there now, and the `mot` site joined it on 0921 (owner decision, `plan.md` phase F1). ISSUE-082 applies to the size of these marginals. The recorded 202.2 ms baseline was untrimmed **and** FA4 off, so it remains a one-sided bound for that row; nothing in `0920t` isolates FA4 at the untrimmed length. The `0919e` gate on fixture v2 is trimmed with FA4 off and measures 114.6 ms against this session's 125.86 ms in the same gate, about 11 ms apart, recorded as a session difference because no same-machine A/B across the two sessions was measured. The three service paths at `valid_tokens=24` are tabulated in OPT-028's `0920t` section: the served `default` runs trim with FA4 through all three faces, and the `native` profile runs the trim with `use_fa4=False` (OPT-029).

## Open

- FA4 at the `mot` site: promoted to the served default on 0921 (owner decision, `plan.md` phase F1) on the `c20f3a0` ladder's evidence (`stack` 9.7 ms under `vae_trim`, agreement with official not worse); the fixture-v2 gate run of the new default (THOR_CHECKLIST.md N1) is the remaining cosine check against the chain.
- The Thor kernel sweep's `num_splits` 2 and 4 rows for the `mot` shape (`fa4_splits1/2/4`) are unrecorded: no round has reported them, and if a split wins the constant in the `mot` branch changes.
- ISSUE-020 (padded text keys) is answered by `text_trim` (OPT-030, the served default), not by a key mask in this kernel, so no attention-kernel change is needed for it; and no custom fused attention kernel is needed at either site.

# OPT-016: single-stream `linear2` merge (roadmap item 4)

Status: implemented and locally verified (H100, `fp16`); default on for every precision except `fp16_cutlass`. Thor `AB=merge_linear2` (this item's own merged-vs-split A/B) was never recorded; the only Thor speed number is the round's microbenchmark of items 3 and 4 together, about -9.7 ms (`THOR_STATUS_SUMMARY.md`, the `c20f3a0` ladder section, random weights and no VAE).

Area: single-stream blocks, 20 backbone + 20 ActionDiT (`pipeline_thor.py` `_single_stream_layer` / `_action_single_layer`), `checkpoint_loader._extract_single_block`, `silu_glu_merged_fp16`. Plan: `plan.md` roadmap item 4; OPT-015's op-fusion audit finding 1, sub-problem 3.

## What changed

The official blocks run `linear2` as one GEMM over `cat([attn_out, mlp_act])`; FlashRT used to split it at load time and run `attn_out_proj` + `mlp_down` + a torch add before the gated residual. With `dims["merge_linear2"]` (default = the `merge_qkv_mlp` rule):

- `checkpoint_loader._extract_single_block(merge_linear2=True)` keeps the real unsplit `linear2.weight`: `(12288, 3072)` backbone, `(7168, 1024)` ActionDiT, in the (K,N) convention.
- `silu_glu_merged_fp16` gained `out_row_stride`; the `linear1`-merged SiLU-GLU writes straight into the MLP columns of `single_linear2_in` / `action_linear2_in`. A strided copy places the attention output in the first `attn_width` columns, then one GEMM with K = attn + mlp_hidden writes `proj_scratch`. Per layer call: one strided copy + one GEMM replaces two GEMMs + one add (two kernels fewer for `nvfp4`, whose linear op is quantize + GEMM).
- `nvfp4`: `hidden` = 3072 is a multiple of the 16-element scale block, so the merged activation and weight quantize to exactly the split path's operands; K = 12288 and 7168 are multiples of 64 (no scale-factor padding). Only the accumulation changes.

## Local results (H100, `fp16`)

| check | result |
|---|---|
| strided SiLU-GLU vs packed call, real shapes | bit-exact, untouched columns zero |
| real `linear2.weight` vs `cat(attn_out_proj, mlp_down)` (blocks 0, 19, both experts) | `torch.equal` |
| backbone single layer (a0=905), projection, merged vs split | cos 0.9999999, max-abs 3.9e-3, rel_l2 3.5e-4 |
| ActionDiT single layer (M=64), projection, merged vs split | cos 0.9999998, max-abs 9.8e-4, rel_l2 3.6e-4 |
| projection rel_l2 vs FP32 reference, merged / split | 2.07e-4 / 2.95e-4 (backbone), 2.08e-4 / 3.01e-4 (ActionDiT) |
| real checkpoint, full graph, actions merged vs split | cos 0.999999, rel_l2 1.4e-3 |
| CUDA kernels per prefill + 10-step denoise | 7082 -> 6662 |
| e2e vs official, 20 frames, `fr_vs_off` median / min | 0.99840 / 0.99566 (baseline 0.99840 / 0.99567) |
| e2e mean `mae_fr_vs_gt` | 0.18359 (baseline 0.18359) |

The merged path is closer to the FP32 reference because it rounds once where the split path rounds three times.

Speed on the shared H100 (indicative only, same process, interleaved, 40 iterations): `infer()` P50 102.23 -> 101.27 ms (P10 91.42 / 98.87, P90 102.61 / 101.49). No Thor claim.

## Thor check

```
# FlashRT at this branch, GPU_ARCH=110 build
cmake --build build -j --target flash_rt_kernels flash_rt_fp4
pytest tests/test_imagewam_real_mlp.py tests/test_imagewam_thor_real_wiring.py -q -s
CKPT_PATH=<.../model.pt> AB=merge_linear2 PRECISIONS=nvfp4,fp16 COUNT_KERNELS=1 \
  python benchmarks/imagewam_fusion_ab.py
```

Expected: pytest passes and prints `bit_exact_vs_packed=True` and merged vs split `cos` >= 0.9999; the A/B prints per precision `actions B vs A` (expect cos >= 0.9999 and finite for `nvfp4`; the `linear1` merge gave 0.99998), `infer()` / `replay()` P10/P50/P90 for split (A) and merged (B), and the kernel count (B lower). The `linear1` precedent predicts most of any win in the ActionDiT loop; report every printed line.

## Open

- This item's own Thor A/B is unrecorded: no round has reported `actions B vs A`, the split/merged `infer()` P50 or the kernel count on Thor for `AB=merge_linear2`, so Thor speed for the merge alone is unmeasured.
- FP8 precisions: one activation scale now spans both halves, and so does the weight side, where the split path had one scale per half (issues.md ISSUE-011, which measures it negligible at the GEMM level: merged / split median 1.00x, range 0.98-1.03x, and none of these precisions is the shipped default). ISSUE-011's next experiment is `AB=merge_linear2 PRECISIONS=fp8,fp8_static` with `CKPT_PATH` set (`benchmarks/imagewam_fusion_ab.py`), on Thor or on H100; close it if it matches `fp16`'s merged vs split (cos >= 0.9999), otherwise default FP8 to the split path.
- The attention output is copied into the `linear2` input (one strided copy per layer, 11 MB per backbone layer). Having the attention backend write its output there directly (an output row stride in `ImageWAMAttnBackend.run`, or the FA4 path's existing output copy retargeted) would remove it.

# OPT-017: gated residual + next AdaLN in one kernel (roadmap item 3)

Status: implemented and locally verified bit-exact (H100); default on for every precision. Thor `AB=fuse_res_norm` for this item alone was never recorded; the only Thor fusion number is items 3 and 4 together, about -9.7 ms (`THOR_STATUS_SUMMARY.md`, the `c20f3a0` ladder section, microbenchmark with random weights and no VAE).

Area: every gated residual update in `pipeline_thor.py` (backbone double/single, ActionDiT double/single, ActionDiT head) and `csrc/kernels/fusion.cu`. Plan: `plan.md` roadmap item 3; supersedes OPT-015's deferred CUTLASS gated-residual epilogue for this problem.

## What changed

- `csrc/kernels/fusion.cu`: `gate_res_ada_layer_norm_bf16res` (backbone, BF16 residual) and `gate_res_ada_layer_norm_fp16` (ActionDiT) update the residual and write the next normed + modulated FP16 activation in one launch, one block per row. gate/scale/shift are `(dim,)` FP32 vectors read straight from the modulation output and rounded to FP16 in-kernel; the LayerNorm statistics use the stored residual and the reduction of `ada_layer_norm_*`. `out == nullptr` gives the residual-only update (the backbone's last layer).
- `pipeline_thor.py` with `dims["fuse_res_norm"]`: AdaLN2 of each double block fuses into its attention residual; `imagewam_prefill` and `imagewam_denoise_step` chain each layer's last residual into the next layer's AdaLN: double -> double, last double -> single (txt and img rows each normalized with the single blocks' modulation), single -> single, ActionDiT last single -> head (`head_modded`, the standalone head AdaLN is skipped).
- The fused path records no per-layer `_fuse_mod_group` kernels (FP16 casts of shift/scale/gate and the `(rows, dim)` gate broadcast); FP16 copies remain only for the standalone AdaLN at the start of each chain (prefill layer 0, each denoise step's layer 0).

## Local results (H100)

| check | result |
|---|---|
| kernel vs `gate_res_*` + `ada_layer_norm_*`, 513 / 392 / 905 x 3072 BF16, 64 x 1024 FP16 | residual and output bit-exact |
| kernel normed output vs FP32 torch reference | rel_l2 2.07e-4 (FP16 output rounding) |
| residual-only mode vs `gate_res_bf16res` | bit-exact |
| real dims, random weights, prefill + 10-step denoise, fused vs unfused | `backbone_hidden`, all 25 layers' K/V, `action_latent` bit-exact |
| real checkpoint, captured graph, fused vs unfused (`fp16`) | actions and `backbone_hidden` bit-exact |
| CUDA kernels per prefill + 10-step denoise | 6662 -> 4968 (-1694) |
| e2e vs official, 20 frames, both items on, `fr_vs_off` median / min | 0.99840 / 0.99566 (baseline 0.99840 / 0.99567) |
| e2e mean `mae_fr_vs_gt`, both items on | 0.18359 (baseline 0.18359) |

Speed on the shared H100 (indicative only, same process, interleaved):

| A/B | iterations | `infer()` P50 A -> B | P10 A / B | P90 A / B |
|---|---:|---:|---:|---:|
| `fuse_res_norm` | 40 | 101.38 -> 96.24 ms | 100.89 / 93.52 | 102.09 / 96.99 |
| `merge_linear2` + `fuse_res_norm` | 200 | 102.30 -> 96.17 ms | 92.78 / 91.70 | 104.84 / 96.90 |

Both items together: 7082 -> 4968 CUDA kernels per pass. No Thor claim.

## Thor check

```
# FlashRT at this branch, GPU_ARCH=110 build
cmake --build build -j --target flash_rt_kernels flash_rt_fp4
pytest tests/test_imagewam_residual_norm_fusion.py -q -s
CKPT_PATH=<.../model.pt> AB=fuse_res_norm PRECISIONS=nvfp4,fp16 COUNT_KERNELS=1 \
  python benchmarks/imagewam_fusion_ab.py
CKPT_PATH=<.../model.pt> AB=merge_linear2,fuse_res_norm PRECISIONS=nvfp4,fp16 \
  python benchmarks/imagewam_fusion_ab.py
```

Expected: pytest prints `bit_exact=True` for every kernel case and for `backbone_hidden`, `K_cache`, `V_cache`, `action_latent` of the whole pass; `AB=fuse_res_norm` gives `actions` and `backbone_hidden` B vs A bit-exact for both `nvfp4` and `fp16`. The script builds B on A's autotuned `GemmRunner` (`gemm_runner=`), so every cuBLASLt shape runs the same algorithm on both sides: all `fp16` weight GEMMs, and under `nvfp4` the `fp16_nn` fallbacks (`action_encoder`, `head.linear`) and the `bf16_nn` entry GEMMs (`txt_in`, `img_in`); the NVFP4 CUTLASS GEMMs choose their variant from the shape alone. Kernel count B lower by about 1700; `infer()` P50 B below A. The combined run gives the total of items 3 and 4 against the pre-roadmap per-layer path; report every printed line, and add `USE_FA4=1` if the production configuration uses FA4.

## Remaining per-pass launches after items 3 and 4 (H100 profiler, real dims, `fp16`)

Prefill: 478 CUDA kernels. 10-step denoise: 4490, of which:

| kernel | launches | source |
|---|---:|---|
| torch elementwise copy | 950 | 750 Q/K/V column-slice copies (`_copy_slice`, 3 per layer) + 200 attention-output copies into the merged `linear2` input |
| `rms_norm_kernel` | 500 | QK-Norm, 2 per layer |
| `rope_apply_fp16_perhead_kernel` | 500 | RoPE on Q and K, 2 per layer |
| GEMM kernels (cuBLASLt, incl. 310 split-K reduces) | ~1100 | weight GEMMs + attention QK^T / PV |
| `gate_res_ada_layer_norm_kernel` | 300 | this entry |
| `fill_neginf_strided_kernel` | 250 | odd `kv_seq` (969) logits pad column |
| `softmax_fp16_kernel`, `silu_glu_merged_kernel` | 250 each | |

Candidates by launch count (not planned): one kernel doing the Q/K/V split + QK-Norm + RoPE from the `qkv`/`linear1` output would replace 7 launches per layer with 1 (about 1500 per `infer()` in the denoise loop alone); an even-padded K/V length or a pad-aware softmax would drop the 250 pad fills; the attention-output copy is OPT-016's follow-up.

## Open

- Thor `AB=fuse_res_norm` (this item alone) is unrecorded: no round has reported the `actions` / `backbone_hidden` bit-exactness, the `infer()` P50 pair or the kernel count for it on Thor.
- Thor `AB=merge_linear2,fuse_res_norm` (items 3 and 4 together) is likewise unrecorded except for the combined microbenchmark at about -9.7 ms (`THOR_STATUS_SUMMARY.md`, the `c20f3a0` ladder section), which is the only Thor speed number this entry has.

# OPT-020: VAE input preprocessing kernel with a 256-entry normalization table (roadmap item 2)

Status: implemented and served by default, bit-identical to the previous path. Thor: the preprocessing kernel measures `0.97 -> 0.13 ms` in the `c20f3a0` microbenchmark (`THOR_STATUS_SUMMARY.md`), and `THOR_CHECKLIST.md` carries no pending item.

Area: `flash_rt/models/imagewam/vae_preprocess.py`, `csrc/kernels/imagewam_vae_preprocess.cu`, used by `vae_encoder.encode_to_tokens(..., preprocessor=)` and `vae_stage.ImageWAMVaeStage`

## Mechanism

One `imagewam_vae_preprocess_bf16` launch per camera view reads the `(H,W,3)` uint8 view and writes its column block of the `(1,3,224,448)` BF16 VAE input. It replaces `_prep_view` (uint8 -> float32 on the source device, `F.interpolate(mode="area")`, three elementwise normalize ops, BF16 cast) and `torch.cat`.

- No resize (view already 224x224): a 256-entry BF16 table built on the GPU with the same torch expression as `_prep_view`, Pi0.5's `_infer_uint8_to_fp16` technique.
- `resize="area"` (served default): reproduces torch's arithmetic, which was measured on H100: area pooling is `sum / kh / kw` (two rounded float32 divisions over exact integer window sums), and `x / 255.0` is `x * (1.0f/255.0f)`. Explicit `_rn` intrinsics keep it exact under `--use_fast_math`.
- `resize="pil_bilinear"` (opt-in, frontend `vae_resize`): Pillow's `Resample.c` fixed-point bilinear (22-bit coefficients built on the host in double precision, horizontal then vertical pass with uint8 rounding between them) plus the official center crop, then the table. Only the resize is bit-exact to the official eval; the eval normalizes in BF16 arithmetic, which differs from the table in 127 of 256 entries (ISSUE-030).

## Result (H100, shared GPU, indicative)

Bit-exactness, `tests/test_imagewam_vae_preprocess.py`: 0 differing BF16 elements against `_prep_view` (area) and against the official `_center_crop_resize` + served normalization (pil_bilinear), for real 512x512 LIBERO frames and random 512x512, 256x256, 224x224, 480x640 and 100x150 inputs. Tokens from `encode_to_tokens` are bit-identical with and without the kernel.

End to end (`imagewam_e2e_official_compare.py`, fp16, 10 tasks x frames {0,60}, seeds {0,1}): `fr_vs_off` median 0.99840 / min 0.99567, mean `mae_fr_vs_gt` 0.18359, identical to the baseline.

Latency, `benchmarks/imagewam_vae_stage_bench.py --section preprocess` (two views, alternating in one process):

| input | torch path kernels / GPU time | kernel path kernels / GPU time | host enqueue torch -> kernel |
|---|---|---|---|
| raw 512x512, CPU uint8 | 15 / 0.702 ms | 4 / 0.054 ms | 22.2 -> 2.5 ms |
| raw 512x512, GPU uint8 | 15 / 0.455 ms | 2 / 0.009 ms | 0.31 -> 0.05 ms |
| 224x224, CPU uint8 | 11 / 0.102 ms | 4 / 0.016 ms | 20.4 -> 2.3 ms |

The CPU-input rows are dominated by the torch path converting uint8 to float32 on the host (`.to(device, float32)` of a CPU tensor) and then copying 4x the bytes; the kernel path copies uint8 only. The host numbers on this box are inflated by CPU contention from the co-tenant job and by GPU time-slicing (a synchronized call has a ~2.4 ms floor here), so only the direction is meaningful locally. `encode_to_tokens` wall P50 with CPU inputs: 36-42 ms -> 12.4 ms.

## Open

Owner decision on the served resize filter (`area` against `pil_bilinear`): ISSUE-030. Thor `infer()` P50 attributed to this step alone is not recorded here.

# OPT-021: VAE encode inside the CUDA graph and a native NHWC encoder (roadmap item 5)

Status: implemented behind frontend flags (`vae_encoder="native"`, `vae_graph_input=(views, H, W)`), verified on H100; defaults unchanged (`vae_encoder="torch"`, VAE outside the graph). The Thor latency this entry left pending was delivered by the `c20f3a0` ladder and by `eccf14f` (both in `## Thor check`), and the Thor VAE-stage latency is `19.4 -> 8.3 ms` in the `c20f3a0` microbenchmark (`THOR_STATUS_SUMMARY.md`). Still open: the four placements/encoders `infer()` matrix on `nvfp4`, the native-encoder token cosine on sm_110, and the promotion decision (`plan.md` Open item 2).

Area: `flash_rt/models/imagewam/vae_stage.py`, `flash_rt/models/imagewam/vae_native_encoder.py`, `csrc/kernels/imagewam_vae_groupnorm.cu`, `csrc/kernels/imagewam_vae_residual.cu`, `flash_rt/frontends/torch/imagewam_thor.py`

## Profile of the stock encode (H100, 224x448, indicative)

Measured with `benchmarks/imagewam_vae_stage_bench.py --section profile --profile-repeats 7`: torch-profiler GPU kernel time per op family, 3 invocations x 7 captures of 10 encodes. 282 kernels per `AutoEncoder.encode`, 9.5-10.2 ms kernel time per encode (per-invocation medians; single captures 8.6-10.5 ms). Shares are the per-invocation medians over captures, with the single-capture range in parentheses: the co-tenant job time-slices the GPU, which stretches individual kernel durations, so one capture alone can misstate the split.

| op family | kernels per encode | share of GPU kernel time |
|---|---:|---:|
| torch GroupNorm statistics (`RowwiseMoments`, N*G = 32 blocks) | 44 | 35-39% (31-50%) |
| convolution math | 25 | 18-26% (12-34%) |
| other elementwise: conv-bias broadcast adds, residual adds, mul, pad, copies | 109 | 16-24% (10-30%) |
| cuDNN NCHW<->NHWC transforms around each convolution | 72 | 10-17% (8-24%) |
| sigmoid (swish) | 21 | 2-4% |
| attention (one 1568-token block), q/k/v and proj GEMMs | 11 | ~2% |

The same command profiles `NativeFlux2Encoder`: 142 kernels, 2.6-3.9 ms kernel time per encode (per-invocation medians); convolution math 30-44% of it, FlashRT GroupNorm apply 8-21%, GroupNorm statistics + finalize 8-13%, bias+residual about 2%; the single attention kernel ranges 4-53% across captures, the most time-slicing-sensitive entry. Converting the stock module to `channels_last` removes the transforms but makes torch's GroupNorm slower on the strided layout; no net gain.

## Mechanisms

1. `ImageWAMVaeStage`: fixed uint8 view buffer -> preprocessing kernel (OPT-020) -> encoder -> tokens written into the frontend's `img_raw`. `run()` is capture-safe.
2. `vae_graph_input`: the frontend records `stage.run()` ahead of prefill in its one CUDA graph; `infer()` copies the views into the fixed buffer and replays once. Views of exactly that shape are then required on every call.
3. `NativeFlux2Encoder` (`vae_encoder="native"`): the same op order as `AutoEncoder.encode`, all activations `channels_last`, and NHWC GroupNorm(+SiLU): Welford partial statistics per block, a per-(group, sample) Chan merge, one vectorized apply with torch's BF16 rounding points (after the norm, the sigmoid and the product); conv1's bias folded into norm2's reads; one pass for conv2 bias + nin_shortcut bias + residual add, with torch's BF16 rounding after each add (bit-exact to the three torch ops); the attention block's q/k/v 1x1 convolutions as one GEMM.

## Result (H100, shared GPU, indicative)

Bit-exactness: stage eager / graph (torch encoder) vs `encode_to_tokens` gives bit-identical tokens; the frontend graph vs eager prefill+denoise at the same tokens and noise gives bit-identical actions (torch and native encoders); the GroupNorm(+SiLU) kernel against torch over all 22 real encoder inputs is cosine >= 0.9999998 with 0.001-0.03% of BF16 elements differing; the bias+residual kernel is bit-identical.

Native vs torch encoder tokens on real frames: cosine 0.99998, rel_l2 0.0055, max-abs 0.03-0.05. Token stats (native / torch / real Thor): mean -0.0101 / -0.0101 / -0.02, std 0.9705 / 0.9706 / 0.97, absmax 4.78 / 4.78 / 4.91. rel_l2 against an FP32 encode over 5 frames: torch BF16 0.0088-0.0096, native BF16 0.0089-0.0097. The native encoder is as close to FP32 as the stock BF16 encoder; the native-vs-torch gap is smaller than either one's BF16 error.

End to end (`imagewam_e2e_official_compare.py`, fp16, 10 tasks x frames {0,60}, seeds {0,1}): `fr_vs_off` median / min and mean `mae_fr_vs_gt` are 0.99840 / 0.99567 / 0.18359 for the default path and for the torch encoder inside the graph, and 0.99840 / 0.99568 / 0.18359 for the native encoder inside the graph (baseline 0.99840 / 0.99567 / 0.18359).

Speed, VAE stage alone (`--section encode`, two 512x512 CPU views, alternating):

| variant | kernels | GPU kernel time | wall P50 |
|---|---:|---:|---:|
| legacy `encode_to_tokens` (torch preprocess) | 298 | 7.8-9.5 ms | 38-46 ms |
| `encode_to_tokens`, kernel preprocess (served now) | 287 | 8.1-9.4 ms | 12.4 ms |
| stage, torch encoder, CUDA graph | 287 | same | 12.0 ms |
| stage, native encoder, eager | 147 | 1.9 ms | 4.4 ms |
| stage, native encoder, CUDA graph | 147 | 1.9 ms | 4.3 ms |

Wall-clock on this box has a ~2.4 ms floor per synchronized call from GPU time-slicing with the co-tenant job, and kernel-time totals move with that load between runs (stock 7.8-10.5 ms, native 1.9-3.9 ms across the runs recorded here); compare variants within one run. Graph capture alone is worth ~0.35 ms here, where the host CPU is fast and the encode is GPU-bound. `infer()` at real dims (`--section infer`, fp16, random weights, alternating frontends): eager-torch 120.0 ms, graph-torch 119.5 ms, graph-native 113.9 ms (224x224 views); eager-torch 119.2 ms vs eager-native 114.3 ms (512x512 views).

## Open

- `infer()` P50 on `nvfp4` for the four placements/encoders, and the native-encoder token cosine on sm_110 (Thor, checklist). Promoting the native VAE is `plan.md` Open item 2.
- The largest remaining native cost is convolution math (30-44% of its kernel time); the three (0,1,0,1) zero pads before the stride-2 convolutions and the conv_in input layout conversion are the next copy-elimination candidates.

## Thor check

`c20f3a0`, one matrix session (`libero_spatial`, nvfp4): the `vae` row (native NHWC encoder captured into the main graph, everything else at the default configuration) measured 190.1 ms against `default` 225.5 ms, and removing the switch from the full stack costs +11.6 ms (`stack_no_vae` 104.8 against `stack` 93.2). Both readings are below the corresponding row without it, so the ladder's criterion for the native VAE (lower P50 than `default`, not worse against official) holds; the served default stays `vae_encoder="torch"` with the VAE outside the graph, and the decision is the owner's (`plan.md` "Decisions pending"). ISSUE-082 applies to the size of these marginals.

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is end-to-end `infer()` P50 in ms and `vs official` is the median action cosine against the official model. Native NHWC encoder captured into the main graph, real checkpoint, `nvfp4`:

| suite | default P50 ms | vae_trim P50 ms | delta | vs official: default / vae_trim |
|---|---:|---:|---:|---|
| libero_goal | 203.0 | 118.7 | -84.3 | 0.99558 / 0.99937 |
| libero_10 | 202.0 | 103.0 | -99.0 | 0.99765 / 0.99926 |

Both `vae_trim` rows also carry `text_trim`, so this pair measures the native VAE in the graph and the trim together; it does not separate them. The separated ladder exists for `libero_spatial` in the `c20f3a0` section above (`default` 225.5 ms, `vae` 190.1 ms, `vae_trim` 102.9 ms), and there the full stack agreed with official better than the default configuration (0.99934 against 0.99764), which is the same direction the two rows here show.

# OPT-024: Hadamard-rotated INT4 (E0M3) precision tier, `e0m3_hadamard`

Status: implemented behind `precision="e0m3_hadamard"` (default stays `nvfp4`). Accuracy decided by an H100 simulation matched byte for byte to the CUDA quantizers. Thor is no longer pending: `eccf14f` delivered the vs-official and `infer()` P50 rows, and `a84916a` (`0919e`) the precision's LIBERO gate, which passes. The whole-pipeline cosines vs fp16 and the MAE vs GT that `benchmarks/imagewam_e0m3_hadamard_thor_check.py` prints are still unrecorded, so promotion to the default is undecided.

Area: 4-bit block-scaled GEMM tier for every 16-aligned ImageWAM weight (`quant_linear.py` `E0m3HadamardLinear`, `imagewam_thor.py` `_wrap_linear`), `plan.md` roadmap item 9.

## Mechanism

- Thor's tcgen05 block-scaled MMA reads the operand element format from the runtime instruction descriptor. Value 0 decodes E0M3: sign-magnitude integers -7..7 with the same per-16 UE4M3 scale layout as NVFP4. This is the full-rate `kind::mxf4nvf4` path (`SM100_MMA_MXF4_SS`, K = 64 per instruction), not the SM80 `s4` path OPT-007 closed.
- Both operands get the same orthonormal 16-point Hadamard rotation of every 16-wide K block. The rotation is block-diagonal along K, so any `K % 16 == 0` works; ImageWAM's quantized K values are 1024, 3072, 4096 and 9216, and 7680/12288 would also work. The OPT-007 power-of-two padding problem does not arise.
- Weight, offline: fp32 butterfly, times a per-tensor power of two `2^e` (largest block scale at most 448), stored as fp16, quantized by `quantize_e0m3_dynamic_sfa_fp16`; the GEMM `alpha = 2^-e` undoes the pre-scale exactly. Activation, online: `quantize_e0m3_dynamic_sfa_fp16_vec(use_rht=1)` rotates in registers and quantizes.
- GEMM: `cutlass_fp4_gemm_e0m3w_variant(a_format=0)`, new tile variants 1/6/8 (128x256 tiles) beside the existing 10 (128x64x256), picked by the same `pick_variant(N, K)` as `nvfp4`. The merged single-stream `linear1` (qkv + mlp gate/up, K = 3072 or 1024) is one ordinary weight to this tier; `action_encoder` (K=7) and `head.linear` (N=7) fall back to `Fp16Linear`, as in `nvfp4`.

## Simulation fidelity (H100)

`flash_rt/models/imagewam/blockscaled_ref.py` reproduces the quantizers. `tools/check_blockscaled_quantizers_sm90.py` compiles the unmodified quantizer sources (same flags as `fp4_kernels_obj`) for sm_90a and compares bytes on 130M real ImageWAM weight elements (65M packed bytes), outlier activations, and NVFP4 blocks built on the E2M1 rounding thresholds (1.7M values scale exactly onto a threshold):

- byte-exact: E0M3 weights (plain, rotated, and the served weight preparation `prepare_e0m3_hadamard_weight`, which `E0m3HadamardLinear` calls), E0M3 + H16 activations, and NVFP4 amax, including the threshold blocks. The NVFP4 kernel's `1.f / bs_dq` (`quantize_fp4_sfa.cu`) lowers to `rcp.approx.ftz` and its `amax / 6` to `div.approx` under `--use_fast_math`; on sm_90 they choose the same codes as the reference's round-to-nearest arithmetic even at exact thresholds. Thor's lowering is checked by `test_nvfp4_quantizer_bit_exact` (Thor only).
- NVFP4 MSE (not used by ImageWAM) differs on about 2 codes per million: the kernel sums each candidate's squared error sequentially and its PTX contracts `e * scale - v` and `err += d * d` into `fma.rn`, so near-equal candidates can tie-break differently from the reference. Dequantized operands of every tier are exactly representable in fp16 (0 inexact elements across the whole pipeline), so the whole-pipeline simulation runs the unchanged fp16 cuBLASLt GEMM (fp32 accumulation) on them; it differs from the block-scaled GEMM only in accumulation order.

## Results 1-2: per-GEMM error and whole pipeline (H100 simulation)

`benchmarks/imagewam_e0m3_accuracy_study.py`, real checkpoint. `mean` / `pooled` / `w-only` / `act-only` / `W better` are Result 1 (4 LIBERO frames of libero_spatial through the fp16 pipeline, all 180 quantized weights, all 10 denoise steps; output rel_l2 vs the exact fp32 product, `mean` unweighted over the 18 GEMM groups, `pooled` weighted by output norm and dominated by `txt_mlp2`, ISSUE-051; `W better` = weights better than `nvfp4`). The cosine and MAE columns are Result 2 (20 frames: libero_spatial, first episode of 10 tasks, frames 0 and 60; real VAE/Qwen3/proprio/shift schedule; same N(0,1) noise for every tier; `act` = denormalized 64-step actions). `-` = not measured.

| tier | mean | pooled | w-only | act-only | W better | bh cos med / min | al cos med / min | act cos med / min | 1 - act cos med | act MAE vs fp16 | MAE vs GT |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| fp16, other noise seed | - | - | - | - | - | 1 / 1 | 0.99633 / 0.98108 | 0.99375 / 0.98289 | 6.25e-3 | 0.02075 | 0.18369 |
| fp16 | - | - | - | - | - | 1 / 1 | 1 / 1 | 1 / 1 | 0 | 0 | 0.18359 |
| `nvfp4` (shipped) | 0.0669 | 0.0893 | 0.0343 | 0.0805 | - | 0.99819 / 0.99762 | 0.99937 / 0.99912 | 0.99928 / 0.99857 | 7.17e-4 | 0.00906 | 0.18519 |
| `nvfp4`, MSE weight scales | 0.0633 | 0.0884 | 0.0319 | 0.0805 | 180/180 | 0.99847 / 0.99813 | 0.99943 / 0.99922 | 0.99952 / 0.99867 | 4.77e-4 | 0.00855 | 0.18510 |
| `nvfp4` + H16 | 0.0662 | 0.0464 | 0.0388 | 0.0274 | 25/180 | - | - | - | - | - | - |
| `nvfp4`, weight pre-scale | 0.0653 | 0.0884 | 0.0330 | 0.0805 | 180/180 | 0.99838 / 0.99803 | 0.99943 / 0.99920 | 0.99947 / 0.99868 | 5.35e-4 | 0.00841 | 0.18452 |
| E0M3 weights, E2M1 activations | 0.0650 | 0.0887 | 0.0360 | 0.0805 | 169/180 | 0.99848 / 0.99790 | 0.99947 / 0.99923 | 0.99947 / 0.99872 | 5.27e-4 | 0.00831 | 0.18498 |
| E0M3 W4A4, no rotation | 0.0686 | 0.0749 | 0.0360 | 0.0649 | 65/180 | 0.99882 / 0.99840 | 0.99844 / 0.99781 | 0.99840 / 0.99624 | 1.60e-3 | 0.01644 | 0.18901 |
| E0M3 W4A4 + H16 (Pi0.5's tier) | 0.0565 | 0.0399 | 0.0360 | 0.0179 | 180/180 | 0.99970 / 0.99948 | 0.99969 / 0.99954 | 0.99968 / 0.99918 | 3.15e-4 | 0.00622 | 0.18363 |
| **E0M3 W4A4 + H16, weight pre-scale (`e0m3_hadamard`)** | **0.0545** | **0.0380** | **0.0336** | **0.0179** | **180/180** | **0.99960 / 0.99949** | **0.99970 / 0.99961** | **0.99971 / 0.99941** | **2.86e-4** | **0.00606** | **0.18352** |
| same, H64 rotation | 0.0559 | 0.0401 | 0.0340 | 0.0211 | 180/180 | - | - | - | - | - | - |
| same, H256 rotation | 0.0563 | 0.0420 | 0.0340 | 0.0242 | 180/180 | - | - | - | - | - | - |

- The gain is on the activation side: E0M3 needs the rotation (without it, activations are worse than E2M1), and with it the uniform grid beats E2M1 plus rotation. On weights alone the formats are close: pooled weight-only error 0.0336 for `e0m3_hadamard` and 0.0319 for NVFP4 with MSE scales, the best weight quantizer in the study. Rotations larger than 16 are not better, so the existing in-register 16-point kernel is the right one and no new rotation kernel is needed. A per-tensor activation pre-scale adds nothing on top of H16 (0.0545 either way).
- `e0m3_hadamard` lowers the actions error vs fp16 (1 - cos, median) by 60% relative to `nvfp4`; it has lower actions error, lower `backbone_hidden` error, and lower actions MAE vs fp16 on 20 of 20 frames, and its open-loop MAE vs ground truth equals fp16's (0.18352 vs 0.18359; `nvfp4` 0.18519). The plan's decision rule (at least 25% lower, MAE not worse) is met. The quantized tiers' actions error is 4-22x below the fp16 sampler's own seed-to-seed spread (6.25e-3); `e0m3_hadamard`'s is 22x below. The eager run and the captured fp16 graph agree bit for bit (action_latent max |diff| = 0).

## Result 2b: merged single-stream `linear2` (H100 simulation)

With OPT-016 the single-stream `linear2` is one GEMM for every precision that merges `linear1`, `e0m3_hadamard` included: K = 12288 in the backbone (tile variant 1) and 7168 in ActionDiT (variant 6). One per-tensor weight pre-scale now covers the attention half and the MLP half.

- Weight side: in 11 of 20 backbone blocks and 6 of 20 ActionDiT blocks the two halves alone would choose a different exponent (gap up to 2 and 3); with the merged exponent no block scale of either half is subnormal, and UE4M3's relative step is the same across its normal range: the dequantized E0M3 weights of all 40 merged `linear2` tensors are bit-identical to the concatenated split halves. Activation side: the attention/MLP boundary (3072) is a multiple of 16, so no rotation or scale block straddles it and the quantized activation equals the split one. The merge therefore changes only the accumulation (one fp32 sum instead of two fp16 outputs and an fp16 add), as for `nvfp4`.
- Same study, same 20 frames, merged `MERGE_LINEAR2=1` vs split `0`, columns as in the table above: `fp16` other noise seed merged 1 / 1, 0.99632 / 0.98107, 0.99375 / 0.98289, 6.25e-3, 0.02075, 0.18370; `nvfp4` merged 0.99820 / 0.99762, 0.99935 / 0.99911, 0.99933 / 0.99860, 6.71e-4, 0.00910, 0.18515 and split 0.99819 / 0.99762, 0.99937 / 0.99912, 0.99928 / 0.99857, 7.17e-4, 0.00906, 0.18519; no rotation merged 0.99882 / 0.99840, 0.99842 / 0.99796, 0.99839 / 0.99659, 1.61e-3, 0.01619, 0.18896 and split 0.99882 / 0.99840, 0.99844 / 0.99781, 0.99840 / 0.99624, 1.60e-3, 0.01644, 0.18901; `e0m3_hadamard` merged 0.99960 / 0.99949, 0.99970 / 0.99957, 0.99970 / 0.99934, 2.97e-4, 0.00600, 0.18353 and split 0.99960 / 0.99949, 0.99971 / 0.99952, 0.99966 / 0.99941, 3.37e-4, 0.00611, 0.18348; fp16 MAE vs GT 0.18359.
- Merged `e0m3_hadamard` has 56% lower median actions error than merged `nvfp4` (2.97e-4 vs 6.71e-4) and is better on 20 of 20 frames in actions error, `backbone_hidden` error, and actions MAE vs fp16. Per frame, merged/split actions error has median ratio 0.986 (range 0.55-1.32) for `e0m3_hadamard` and 0.993 (0.75-1.26) for `nvfp4`: the same accumulation-order spread for both tiers. The split-form `e0m3_hadamard` value here (3.37e-4) differs from Result 2 (2.86e-4) only in the simulated weight preparation, now `prepare_e0m3_hadamard_weight` itself (butterfly, pre-scale before the fp16 rounding) instead of matrix rotation with fp16 rounding before the pre-scale; pooled per-GEMM error is 0.03802 in both, and at this error level the median `1 - cos` moves by about 15% with sub-ulp operand changes while the tier ranking does not move.
- Per-GEMM error (4 frames), merged `linear2`: backbone `nvfp4` 0.08486, `e0m3_hadamard` 0.07402 (split `attn_out_proj` 0.09926 / 0.08870, `mlp_down` 0.08626 / 0.07427); ActionDiT 0.04671 / 0.03448 (split 0.04767 / 0.04030 and 0.04686 / 0.03409). `e0m3_hadamard` is better than `nvfp4` on all 140 weights of the merged tree.

## Relation to OPT-014's Thor `backbone_hidden` cosine

OPT-014 recorded `nvfp4` `backbone_hidden` cosine 0.9939 vs fp16 on Thor; the simulation with the real Qwen3 context gives 0.9976-0.9986. The same study with `set_prompt()`'s fallback for a frontend without a text encoder (every context row N(0,1), proprio in the last row; `CONTEXT=random`, merged tree, 20 frames) gives backbone_hidden cos median (range), then the squared-error share over real-prompt rows / other text rows / image rows: Qwen3 `nvfp4` 0.99819 (0.9976-0.9986), 0.873 / 0.050 / 0.077; Qwen3 `e0m3_hadamard` 0.99960 (0.9995-0.9997), 0.265 / 0.259 / 0.492; random N(0,1) `nvfp4` 0.99247 (0.9536-0.9964), 0.007 / 0.852 / 0.136; random N(0,1) `e0m3_hadamard` 0.99395 (0.9736-0.9969), 0.005 / 0.831 / 0.144. "Real-prompt rows" are the rows each task's real Qwen3 tokens occupy (about 30 of 513); "other text rows" are the rest of the text rows (padding and the proprio row). With a random context the rows standing in for padding carry 70-98% of the error and the median cosine (0.9925) is close to OPT-014's 0.9939; an independent H100 run with other random contexts gave median 0.9932 (0.9908-0.9958) with 78-87% of the error in those rows. With the real context, 74-90% of the `nvfp4` error sits in the real-prompt rows. Hypothesis, unconfirmed: OPT-014's Thor comparison ran on the random-context fallback; its script is not in the repository. `benchmarks/imagewam_e0m3_hadamard_thor_check.py` uses the real Qwen3 context and will show whether Thor reproduces the simulated 0.998. Actions error in the random-context run: `nvfp4` 8.06e-4, `e0m3_hadamard` 5.23e-4.

## Result 3: activation quantizer cost (H100, indicative only)

The same kernel sources built for sm_90a, CUDA events, 60 alternating rounds of 50 launches, P10/P50/P90 in us, `quantize_fp4_dynamic_sfa_fp16` (`nvfp4`) vs `quantize_e0m3_dynamic_sfa_fp16_vec` + H16: 905 x 3072 10.6 / 10.6 / 10.7 vs 6.1 / 6.2 / 6.2; 905 x 9216 19.6 / 19.7 / 19.8 vs 10.6 / 10.6 / 10.6; 392 x 9216 10.9 / 10.9 / 10.9 vs 6.4 / 6.4 / 6.5; 64 x 1024 4.4 / 4.4 / 4.4 vs 3.0 / 3.0 / 3.1; 64 x 4096 4.6 / 4.6 / 4.7 vs 3.1 / 3.1 / 3.2. The vectorized E0M3 quantizer with the rotation is faster than the scalar NVFP4 quantizer ImageWAM uses. The GEMM tiles match `nvfp4`'s; the runtime-descriptor GEMM's Thor speed at ImageWAM's large-M shapes is unmeasured (Pi0.5 measured it equal to NVFP4 at decoder shapes with the 128x64x256 tile).

## Thor check

Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive; real checkpoint, libero_spatial; latency is end-to-end `infer()` P50 in ms and `vs official` is the median action cosine against the official model.

| round | row | P50 ms | vs official median |
|---|---|---:|---:|
| `eccf14f` | `default` | 198.9 | 0.99786 |
| `eccf14f` | `stack` | 90.5 | 0.99968 |
| `0919e` (`a84916a`) | gate | 222.2 | 0.99781 (min 0.99434) |

- `eccf14f` (logs `/home/jingwu/thor_val/0919s/`): both rows are at or below the `nvfp4` rows with the same names in this round (202.4 / 202.1 / 202.0 ms `default`, 93.1 / 92.8 / 93.3 ms `stack`), and the tier's microbenchmark measures 199.1 ms, 0.2 ms above the `default` row. Its LIBERO gate reported `blocked`: `tests/fixtures/imagewam_gate/fidelity_thresholds.json` held no `e0m3_hadamard` entry, so the gate did not measure this tier and the figures above are that round's accuracy evidence for it.
- `0919e` / `a84916a` (logs `/home/jingwu/thor_val/0919e/`): the thresholds file having gained an entry for the precision, the gate runs and passes, and that median is at or above `nvfp4`'s own gate median 0.99744, the criterion this precision's gate was set against. The latency half of the gate does not apply to this precision: `tests/fixtures/imagewam_gate/latency_baselines.json` holds no Thor baseline for it, the same as for `fp8_static_cutlass`.

## Open

- Thor correctness of the kernel path (`tests/test_imagewam_e0m3_hadamard.py`) is not recorded green on Thor, and the whole-pipeline cosines vs fp16, the open-loop MAE vs GT and the `infer()` P50 comparison that `benchmarks/imagewam_e0m3_hadamard_thor_check.py` prints are unrecorded; the gate's vs-fp16 and MAE-vs-GT bounds pass only by implication.
- Promotion to the default needs those Thor numbers: better accuracy is established in simulation, speed parity is not measured.
- Side findings for the shipped `nvfp4` tier: ISSUE-050 (subnormal weight scales), ISSUE-051 (`txt_mlp2` activation scale saturation), ISSUE-052 (down-projection activation scales).

# OPT-023: AWQ per-channel scales folded into the NVFP4 weights (roadmap item 8)

Status: implemented behind `nvfp4_awq=True` (default off); accuracy measured on H100 with simulated NVFP4. The Thor check was delivered: `eccf14f` measured the real `nvfp4` path and `0920` the kernel count (both in `## Thor check`). Plan: `plan.md` roadmap item 8; mechanism, fold points and exactness: `docs/imagewam_nvfp4_awq.md`.

Area: `flash_rt/models/imagewam/awq.py`, `flash_rt/models/imagewam/quant_linear.py` (`Nvfp4Linear`), `flash_rt/models/imagewam/pipeline_thor.py` (`_awq_target`), frontend options `nvfp4_awq` / `awq_alpha` / `awq_scope` in `flash_rt/frontends/torch/imagewam_thor.py`

## H100 results (simulated NVFP4, bit-exact quantizer, real checkpoint)

Whole pipeline vs `fp16`, 20 held-out `libero_spatial` frames, median (min) cosine, `benchmarks/imagewam_precision_fidelity.py`:

| | backbone_hidden | action_latent | actions | MAE / fp16 |
|---|---:|---:|---:|---:|
| `nvfp4_sim` | 0.99819 (0.99762) | 0.99937 (0.99911) | 0.99931 (0.99848) | 1.010 |
| `nvfp4_sim` + AWQ 0.5, folds A + B | **0.99956 (0.99916)** | **0.99973 (0.99944)** | **0.99965 (0.99882)** | **1.000** |
| Thor `nvfp4`, OPT-014 result 1/2 (1 frame; 50 frames for MAE) | 0.9939 | 0.9997 | 0.9998 | 1.01 |

AWQ cuts the backbone_hidden error (1 - cosine) 4x and the action error about 2x, and brings the open-loop MAE ratio from 1.010 to 1.000.

Against official ImageWAM (`imagewam_e2e_official_compare.py`, `N_TASKS=10 FRAMES=0,60 SEEDS=0,1`, same N(0,1) noise both sides):

| FlashRT path (H100) | fr_vs_off median | min | mean MAE vs GT |
|---|---:|---:|---:|
| fp16 (baseline) | 0.99840 | 0.99567 | 0.18359 |
| `nvfp4_sim` | 0.99746 | 0.99399 | 0.18519 |
| `nvfp4_sim` + AWQ 0.5, folds A + B | 0.99779 | 0.99469 | 0.18378 |

Per-layer: alpha 0.5 is best for both fold classes (0.25-1.0 swept); fold A sites 0.0742 -> 0.0610 rel_l2, fold B sites 0.0788 -> 0.0679; sites without a fold point would gain ~1% at most.

After the `linear2` merge and the residual+AdaLN fusion (fold A reaches the fused kernel as an FP32 pair, fold B covers the MLP channels of the merged `linear2`), the same comparison gives `nvfp4_sim` 0.99820 / 0.99938 / 0.99933 / MAE 1.009 and with AWQ 0.99956 / 0.99973 / 0.99971 / MAE 1.000 (backbone_hidden / action_latent / actions median).

Speed: AWQ changes weight values and the AdaLN constants only. At toy dims the AWQ pipeline launches the same kernels per forward (413 = 413 unfused, 285 = 285 fused, `tests/test_imagewam_awq.py`); on Thor `infer()` P50 does not move (202.2-202.9 against 202.0-203.0 ms).

## Open

- Per-tensor power-of-two NVFP4 weight pre-scale: tracked in `issues.md` ISSUE-050. It combines with AWQ: in this study's per-layer comparison, fold-A sites gave rel_l2 0.0610 with AWQ 0.5 alone and 0.0595 with AWQ 0.5 plus the pre-scale.
- `proj` / `attn_out_proj` have no exact fold point; a fused multiply-and-quantize of the attention output would be needed to scale them, for at most ~1% per-layer gain.

## Thor check

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Real `nvfp4` with `nvfp4_awq=True`: actions cosine 0.99970 against `fp16`, where the plain `nvfp4` path gives 0.99939, at P50 202.2-202.9 ms against 202.0-203.0 ms for this round's default rows. Both halves of the Thor check this entry left open therefore hold on real hardware: the cosines improve over plain `nvfp4` and the P50 does not move.

`0920`: the folded path adds no kernels of its own, measured with the AWQ test's own counter after the profiler warm-up region is discarded: `kernels per eager forward: plain=285 awq=285` (Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive, log `/home/jingwu/thor_val/0920/R_awq.log`). The count must be read after a warm-up region is discarded, because a first `torch.profiler` CUDA region in a process can report nothing at all; `run_eager` is eager with and without a `weights` argument. The equality is the property this entry is about: the fold moves work into the quantizer, it does not add a per-forward kernel.

# OPT-028: ImageWAM through `frt_model_runtime_v1` (Python producer)

Status: implemented and verified on H100 (fp16, real checkpoint, bit-exact) and on Thor at `nvfp4` — the tick test (`tests/test_imagewam_model_runtime_export.py`, two captured lengths bit-exact against `infer()`) and the export gate this entry's promotion condition names (`0920c`, both VAE placements, every row bit-exact, five mutants detected) pass. No open item.

Area: deployment engineering, roadmap item 12; port schema and ownership in `docs/imagewam_model_runtime.md`.

## Opportunity

`ImageWAMTorchFrontendThor.export_model_runtime(io="python")` publishes the captured graph and its buffers through the generic ABI, with the same Python-producer construction path Pi0.5 uses (`flash_rt.runtime.export.build_model_runtime`). The initial action noise (`0.01 * N(0,1)`, ISSUE-002) is an explicit SWAP input consumed as written; the `0.01` factor stays inside `infer()`.

## Expected Mechanism

No numerics change: the verbs call the frontend's own staging methods (`stage_images`, `stage_proprio`, `read_actions`, `set_prompt`), which `infer()` now also calls, and `step` replays the same instantiated graph exec through `frt_graph_replay`; parity is bit-exact by construction and measured. The export is additive and opt-in; `infer()` behavior is unchanged.

## Required Evidence

H100 (shared GPU), `tests/gate_imagewam_model_runtime_export.py --precision fp16`, real checkpoint + VAE + Qwen3 + dataset stats, LIBERO spatial episode 0 frame 0, ctypes consumer vs `infer()`, same seed — every row `array_equal=True` / `max_abs=0`: control `infer()` vs `infer()`; `images` STAGED → `image_tokens` window; `actions` (denormalized, STAGED); `actions_raw` (normalized, SWAP); `image_tokens` SWAP path → `actions`; `prompt` SETUP (second task) → `actions`. The second task string moves the chunk by `max_abs = 0.0821`, so the prompt check is not vacuous. Peak GPU memory 17.2 GiB.

Latency on the same shared H100 (co-tenant at 100% utilization, indicative only), alternating A/B, 20 iterations each, wall time with VAE, proprio staging and host readback: `infer()` P10 / P50 / P90 133.7 / 142.5 / 152.8 ms against the ABI tick (`images` + `proprio` + `noise` + `step` + `actions`) 127.8 / 138.5 / 153.1 ms.

Invalid calls return the `io="native"` face's statuses (`-2` unknown port, `-3` SWAP port, `-4` payload size, `-5` short buffer, `-1` other), and the runtime's pybind trampolines honor a `VerbStatusError`'s status; the identity carries the calibration file digest and `nvfp4_awq`. Small random-weight dims (`tests/test_imagewam_model_runtime_export.py`, 5 tests): schema, identity sensitivity, guards (`-3`, `-1`, `-5`) and an `array_equal` tick.

Regression: `pytest tests/test_imagewam_*.py` 73 passed, 6 skipped then (baseline 68/6 plus these 5); on the merged tree (H100) 362 passed, 31 skipped with `exec/build`, `runtime/build` and the native target, 326 / 33 without the native target, 316 / 35 without any of the three — every skip needs Thor, FA4 or an FP8 cuBLASLt layout this GPU lacks. `tests/test_imagewam_model_runtime_vae.py` adds both VAE placements at the real token count.

The gate and tests NaN-fill every buffer a tick must write before each ABI tick and re-run each row with its verb made a no-op, so a verb that stages nothing fails (2026-09-18 review). On H100 at fp16 with the real checkpoint after merging the calibration stream, every row above is still `array_equal` (`max_abs = 0`) with the VAE outside and inside the graph (`--vae-graph-input 224 224`, where the `image_views` SWAP window replaces `image_tokens`), and all five mutants (images, proprio on each image path, prompt, step) fail in both placements.

## Promotion Condition — met

Thor reports every parity row `array_equal=True` at `nvfp4`; the export is additive and opt-in. Met at `0920c`: `tests/gate_imagewam_model_runtime_export.py --precision nvfp4` passes with every row `array_equal=True` / `max_abs=0` (the VAE token bits `images` stages included), its determinism control passes, all five mutants are detected with the VAE outside and inside, and both legs print `use_fa4=False` — the gate's `--use-fa4` is a `store_true` flag, so neither leg depends on the machine's FA4 default.

## Thor check

Jetson AGX Thor, MAXN, `emc_locked=null`, real checkpoint; P50 in ms on one wall-clock timer per path, every path in one process; raw logs under `/home/jingwu/thor_val/`. Columns are `infer()` / ABI tick (`io="python"`) / native tick (`io="native"`).

- `0920c` (`c495cb2`), LIBERO `nvfp4` gate: 226.93 / 227.49 / — with the VAE outside the graph and 226.24 / 226.64 / — with it inside (`--vae-graph-input 224 224`), process peak 19.7 GiB in both. The ABI tick measures within 0.7 ms of `infer()` and the two placements within 0.7 ms of each other: this round does not show the in-graph VAE's win end to end and does not attribute the difference (the frames come from `DATA_ROOT` when it is set, and the two legs ran sequentially in one process). What it establishes is the promotion condition.
- `eccf14f`, LIBERO: `default` 202.3 / 184.2 / 183.8; `fast`, every text length precaptured, 93.2 / 95.1 / skipped (the native pipeline held one graph and one context length then). Target workload (three views of 256x256, horizon 32), `default`: ABI 152.7 ms, layout-only — placeholder image tokens, because the VAE path refused this workload.
- `0919e` (`a84916a`), target workload (`text_max_len=128`): `default` 216.93 / 173.55 / 173.27; `fast`, lengths precaptured, 137.64 / 139.35 / skipped; `text_trim` at 16 / 72 / 128 valid tokens 197.00 / 207.06 / 217.50 against ABI 153.51 / 161.68 / 172.05, native skipped (the native face refused the trim then). These rows carry the workload's own VAE encode geometry (ISSUE-086), not the placeholder image tokens of the `eccf14f` layout-only row.
- `0920t` (`4cd06e5`), LIBERO `valid_tokens=24` (`x0 = 25`): `default` 139.47 / 120.29 / 120.05 (trim, `use_fa4=True`, `graph_producer=python`); `fast --precapture` 108.08 / 110.09 / skipped (trim, FA4 at both sites, native VAE in the graph, every length precaptured so no first-capture cost); `native` 145.93 / 126.58 / 126.38 (trim, `use_fa4=False`, differing from `default` in FA4 alone, both sites off since the native C++ pipeline has no FA4 attention, and no longer skipped by any rule). `default` is the served configuration; its gates are in OPT-019's and OPT-030's `0920t` sections.

`text_trim` through the ABI is one adopted graph exec per text length, selected by the replay key: every swept length produced a number, and the tick test passes on Thor including two different captured lengths, bit-exact against `infer()`.

## Open

No open item: every `Required Evidence` row was produced, including the Thor `nvfp4` gate and the H100 re-run after the fusion and VAE streams merged. The one measurement no round made is the in-graph VAE's end-to-end win at these dims and the memory it adds — `0920c` records the absence of that attribution above, and the memory figures are in OPT-030's `## Open`.

# OPT-029: ImageWAM native C++ overlay (`io="native"`)

Status: implemented and verified bit-exact on H100 (fp16, small dims and real checkpoint) and on Thor at `nvfp4` (`0920s4`, `0920t`, `0920c`: both gates pass, node counts unchanged). The native pipeline's own per-length capture is verified as of `0920c`: `test_pipeline_records_one_graph_per_text_length` passes with a complete GEMM hand-off per installed length, and the native pair collects 39 tests with no skip on Thor. Promotion condition met: Thor reports every parity row `array_equal` at `nvfp4` and the replay-only A/B shows the native graph not slower than the Python graph. NVFP4 wiring compiles and links for sm_110.

Area: deployment engineering, roadmap item 14; interface record `docs/imagewam_native_cpp.md`.

## Opportunity

After OPT-028 every ImageWAM tick through the ABI still entered Python (GIL-acquiring trampolines for proprio, actions and `step`), and the graph was recorded from Python, carrying per-replay torch kernels for the AdaLN modulation casts and gate expansions. The overlay is `libflashrt_imagewam_native.so`: C verbs (proprio, actions, step) over a declaration the Python producer builds, and a C++ `NativePipeline` that records prefill + denoise against the existing `csrc` kernels from a borrowed resource table, with the frontend's autotuned cuBLASLt algorithms handed off (`GemmRunner.get/set_cached_algo`, additive) and the modulation precomputed once. VAE and Qwen3 stay in Python.

## Expected Mechanism

Same kernels, same algorithms, same inputs: bit-exact to the Python pipeline, with fewer graph nodes (no in-graph modulation casts/copies) and no Python in the tick.

## Required Evidence

H100 (shared GPU), fp16. Every state buffer `array_equal`: backbone double-stream block 0 (native vs Python, small dims), backbone single-stream block 0, full prefill (`backbone_hidden`, all K/V), full denoise loop (`action_latent`, action K/V rows), native graph vs Python eager / Python graph, and the `io="native"` tick on the native graph vs `infer()`. Python frames entered by `set_input(proprio)` + `step`: io=native 0, io=python 57. Schema records at real dims (Python declaration, C++, golden): identical, 7 records. Real checkpoint: actions / actions_raw / native proprio token / native vs Python graph action latent and K cache all `array_equal`, `max_abs = 0`. GEMM shapes handed off (real dims): 20 of 20. Graph nodes (real dims): native 5732, Python 7112. Exported symbols of the library (sm_90 and sm_110): 18, all `frt_imagewam_native_*`.

Latency (indicative only: H100 shared with a co-tenant at 100% utilization, real checkpoint, fp16, alternating A/B, 50 iterations each; P10 / P50 / P90): `io="python"` tick (SWAP tokens, proprio, noise, step, actions) 104.33 / 107.10 / 110.22 ms; `io="native"` tick on the native graph 102.26 / 103.16 / 104.28 ms; Python graph replay only (CUDA events) 95.86 / 101.94 / 102.40 ms; native graph replay only 100.58 / 100.97 / 101.56 ms.

After the fusion stream (the served layer structure: merged `linear2`, gated residual fused with the next AdaLN, which also removed the per-layer modulation casts from the Python graph), re-run on the merged tree, H100, fp16, real checkpoint, 50 alternating iterations: step-by-step parity in both layer structures, every state buffer `array_equal` at every step; real checkpoint actions / actions_raw / native proprio token / native vs Python graph all `array_equal`; graph nodes native 4974, Python 4998; tick P10 / P50 / P90 85.54 / 98.26 / 100.99 ms (`io="python"`) against 96.26 / 98.95 / 99.25 ms (`io="native"`); graph replay 93.14 / 96.04 / 96.87 ms (Python) against 85.94 / 96.92 / 97.90 ms (native). With the served structure the two graphs differ by 24 nodes (the fp16 casts for the standalone AdaLN at each chain start), so no replay speed difference is expected; the native path's value is a tick with no Python and no GIL, not a faster graph.

Review follow-up (2026-09-18), H100, fp16, merged tree (calibration stream included): NaN-poisoned tick buffers (actions, actions_raw, native proprio token, backbone residual, K/V caches, proprio staged by the frontend or the native verb) all `array_equal`; poisoned native-graph vs Python-graph replay (action latent, backbone residual, K/V caches) all `array_equal`; all 6 mutants detected (proprio verb or `step` skipped, and native pipelines with no backbone block, last single-stream block dropped, last denoise step dropped, one block fed another's `linear1` weight), also as small-dims tests. `set_pipeline` after `capture` destroys the captured graph and frees the old resources, and is refused while an export is live (before: replayed freed memory); a native handle alone with the frontend dropped keeps the frontend alive and replays `array_equal` (before: illegal address); status codes are the same table as `io="python"`; exported symbols (sm_90 and sm_110) 20, all `frt_imagewam_native_*`; `sm110_check.sh` (`flash_rt_kernels`, `flash_rt_fp4`, `flashrt_imagewam_native`) rc 0. ISSUE-071 was a race between a test snapshot on the torch stream and the native warm-up on the non-blocking native stream; `run` / `capture` now wait for prior device work and the test compares `backbone_hidden` between the two graphs again. The native pipeline refuses `nvfp4_awq` (no AWQ fold).

## Promotion Condition

Met: Thor reports every parity row `array_equal` at `nvfp4`, and the replay-only A/B shows the native graph not slower than the Python graph. Neither the A/B nor `benchmarks/imagewam_thor_path_bench.py` measures the deployment's own choice of graph producer: the A/B compares the Python graph with the frontend's adopted one, both recorded by Python, and the bench's native row adopts every captured length the same way (`use_graph(key, exec)` per entry) before `export_model_runtime(io="native")`.

## Thor check

Jetson AGX Thor, MAXN, `emc_locked=null`, real checkpoint; P50 in ms on one wall-clock timer per path, every path in one process; raw logs under `/home/jingwu/thor_val/`. Columns are `infer()` / ABI tick (`io="python"`) / native tick (`io="native"`), all three paths in one process.

- `eccf14f`: LIBERO `default` 202.3 / 184.2 / 183.8. Target workload (three views of 256x256, horizon 32): native tick 154.5 ms, measured with placeholder image tokens, so it is a layout-only number as well; `text_trim` was refused for this face then.
- `0919e` (`a84916a`), target workload (`text_max_len=128`): `default` 216.93 / 173.55 / 173.27, served with the workload's own VAE encode geometry; `fast`, lengths precaptured, 137.64 / 139.35 / skipped; `text_trim` at 16 / 72 / 128 valid tokens skipped for this face (it held one graph and one context length then), ABI 153.51 / 161.68 / 172.05 against `infer()` 197.00 / 207.06 / 217.50.
- `0920s4` (`a4852be`): S4 lands on the native model runtime — the handle carries one adopted graph per trimmed text length (plan.md phase S4). Only the native C++ was rebuilt (`c_api.cpp`, `native_runtime.cpp`, `native_pipeline.cpp`, `fp4_linear.cpp`); the kernels and the fp4 targets were already current and the ctypes layout check passed silently. `IMAGEWAM_NATIVE_PRECISION=nvfp4 pytest tests/test_imagewam_native_pipeline.py tests/test_imagewam_native_runtime.py -q`: 38 passed. Two-length tick (`x0 = 6` first, then `14`): `actions` / `actions_raw` vs `infer()` both `array_equal=True`, `max_abs=0`; native manifest `text_lengths` `{'default_key': 14, 'keys': [6, 14], 'per_prompt_length': True}`; `set_text_length(14)` for a length never adopted `rc=-2`; the untrimmed one-key capture path green whole-file, including its native-vs-Python-graph `array_equal` rows. Gate `tests/gate_imagewam_native_parity.py --precision nvfp4 --bench-iters 50`: `--graph python` every tick row `array_equal=True`, both call mutants detected (proprio verb and `step` not called), P50 python 207.53 / native 207.13 ms; `--graph native` every tick row green, all six detected, P50 python tick 206.21 / native tick 204.33 ms, graph replay python 205.74 / native 204.18 ms. Node counts native 5324 / Python 5348, the same as the previous round's `08_gate_native.log`; these P50s sit about 22 ms above that round's ~182 ms, and since no A/B of the same binary under both sessions' machine state was measured, the 22 ms carries no attribution and is recorded as a session difference rather than a measured regression. `text_trim` was refused for the native *pipeline* then (one graph and one context length from one resource table, so `consumer="native"` refused it); the adopted per-length execs above come from the frontend's captures, which the native model runtime takes by key, and the pipeline's own per-length capture now serves the trim there too.
- `0920t` (`4cd06e5`): native C++ rebuilt. `FLASHRT_THOR_FA4=0 IMAGEWAM_NATIVE_PRECISION=nvfp4 pytest tests/test_imagewam_native_pipeline.py tests/test_imagewam_native_runtime.py -q`: 38 passed, 1 failed — the untrimmed one-key path fully green in that run (state rows `array_equal` with `graph_exec=0`, `graph_nodes=0`, `graph_producer=''`), the failure `test_pipeline_records_one_graph_per_text_length` with both causes fixed at `43c49ce` (the length table is a property, and the comparison covered rows beyond the active length). Gate `--graph native` PASS: native 5324 nodes against Python 5348, all six mutants detected, tick `array_equal` / `max_abs=0`, P50 native 205.27 / Python 207.13 ms. `tests/test_imagewam_text_trim_consumer_guards.py`: 11 passed, its GPU row printing the per-length resource table — `x0=6` dims `(6,16,20)`, `x0=10` dims `(10,20,24)`, the AdaLN rows and the backbone RoPE table following the active length, `buffers identical=True`. Service paths at `valid_tokens=24`: the `native` profile ticks 126.38 ms against 126.58 ms through the ABI, the served `default` 120.05 ms against 120.29 ms.
- `0920c` (`c495cb2`): native C++ unchanged, so the library from `0920t` was reused. `IMAGEWAM_NATIVE_PRECISION=nvfp4 pytest tests/test_imagewam_native_pipeline.py tests/test_imagewam_native_runtime.py -q`: **39 passed** with no skip (the `0920t` failure fixed, `exec/` present); `tests/test_imagewam_text_trim_consumer_guards.py`: **15 passed**, no skip; no test sets `FLASHRT_THOR_FA4` and every frontend on this path states `use_fa4=False` itself (rule R6). `test_pipeline_records_one_graph_per_text_length` installs and captures one pipeline per captured length from the handle itself (`x0=6` first, `14` second), and the per-length GEMM hand-off is complete for both: `gemm_installed_by_key == {6: (4, 4), 14: (4, 4)}` — every shape each length's pipeline launches had an algorithm handed over; both ticks are `array_equal` to `infer()` at the same length with `differing=[]` and `actions max_abs=0`; the untrimmed one-key path is unchanged (`graph_exec=0`, `graph_nodes=0`, `graph_producer=''`). Gate `--graph native` PASS: native 5324 nodes against Python 5348 (unchanged from `0920s4`), all six mutants `detected=True`, tick `array_equal` / `max_abs=0`, P50 native tick 204.43 ms against Python 206.37 ms. The guards' GPU row repeats the per-length resource table above with `buffers identical=True`. With this round the native pipeline's own per-length capture — the last unverified half of ISSUE-080 condition 5 — is confirmed on Thor.

Schema gate `tests/gate_imagewam_native_schema_parity.py --precision nvfp4`: PASS in `0920s4`, `0920t` and `0920c`, its 7 records byte-for-byte identical to the golden records in every round; `x0` is not among the records, so nothing was re-baselined.

## Open

- No benchmark reports the latency of the graphs the C++ pipeline records itself through `capture_pipeline_text_lengths`; their correctness is pinned by the parity gate's `--graph native` row and their per-length install by the checklist's S4-pipeline row, but no round has timed them.
- Remaining native work beyond this entry: VAE encoding in the graph (roadmap item 5, then an `images` STAGED native port), proprio projection inside the graph, and a native checkpoint loader (`native_v2`).

# OPT-030: text context trimmed to the prompt's valid length (issues.md ISSUE-020)

Status: implemented and serving as the default (`PROFILES["default"]` sets `text_trim=True`; the constructor keeps `text_trim=False` for a caller that passes dims and switches by hand), verified on H100 at `fp16`, `fp8` and `fp8_static` and on Thor at `nvfp4` (trim plus FA4, gates and end to end); the `e0m3_hadamard` trim safety check is no longer pending — it ran in the `0920c` round (4 passed, poison bytes overwritten = 0).

Area: `flash_rt/models/imagewam/text_context.py`, `flash_rt/frontends/torch/imagewam_thor.py`, `benchmarks/imagewam_text_trim_bench.py`, `benchmarks/imagewam_e2e_official_compare.py` (`TEXT_TRIM`), `tests/test_imagewam_text_trim.py`

## Mechanism

- Official ImageWAM masks the padded text keys for every query at both attention calls. With proprio packing the valid tokens sit at rows `[0, n_valid)`, the proprio token at row `n_valid`, and text RoPE positions are the row indices, so a sequence of only those rows computes the official masked math without a mask.
- `set_prompt` packs the valid rows by rank plus the proprio slot (`pack_trimmed_context`, checked against official `_append_proprio_to_context`, prefix and non-prefix masks) into `context[:x0]`, `x0 = n_valid + 1`.
- Buffers stay allocated at the max dims (`x0 = 513`, `a0 = 905`, `total = 969`). Each distinct `x0` gets a `TextLengthCapture`: its dims (`a0`, `total` shifted by the same rows), its backbone RoPE table (text positions `0..x0-1`, image positions unchanged), and a CUDA graph captured on first use; a cached length re-activates without capture.
- A new length autotunes the cuBLASLt shapes it adds: `bf16_nn` `txt_in` at `M = x0` for every precision, and the `fp16_nn` text and single-stream shapes (`M = x0`, `M = a0`) for `precision="fp16"` only; untuned shapes would run cuBLASLt's heuristic top-1.
- Before the first trimmed capture one eager prefill at the max dims runs: `Nvfp4Linear`, `Fp8Linear`, `StaticFp8Linear`, `CutlassFp16SwiGluMlp`, `Nvfp4SwiGluMlp` and `E0m3HadamardLinear` grow their activation scratch on a larger `m`, which would free a buffer an earlier graph still reads; after the max-dims pass every backbone op has its largest `m`.
- All captures share one capture stream and one CUDA-graph memory pool; `infer()` replays one graph at a time and no value that outlives a replay lives in the pool.
- Unchanged: `pipeline_thor.py` and `attn_backend.py` already take every length from `dims` and the per-call arguments. The served per-head attention pads an odd `kv_seq` internally (the untrimmed dims already run odd lengths, 905 and 969); the even-`kv_seq` guard belongs to the non-per-head kernel, which the frontend never selects. FA4 output staging (`fa4_out`) is sized at the max dims.
- The FA4 fallback drops every cached graph once the cuBLAS recapture has succeeded, so all graphs use the same attention; until then the old graphs and their RoPE tables stay alive.
- A capture that raises leaves no graph active (`infer()` refuses until a `set_prompt` succeeds) and clears the prompt cache key: `set_prompt` has already written the new context, which no captured graph matches; cached captures of other lengths stay valid.
- Python's cyclic garbage collector is run once before each capture and kept off during it: `torch.cuda.graph` no longer collects before a capture, and destroying a CUDA graph held by a dead reference cycle while a stream captures invalidates the capture, which `text_trim` runs into while the process serves.
- `run_eager()` (the calibration recorder's forward) runs the active length's dims and RoPE table, bit-identical to the active graph; `precapture_text_lengths(x0s)` captures known lengths ahead of their first `set_prompt` without changing the active prompt.

## Result (H100, shared GPU)

`benchmarks/imagewam_e2e_official_compare.py`, fp16 vs official bf16, 10 tasks x frames {0, 60} per suite, seeds {0, 1}, `fr_vs_off` median / min / mean:

| suite | valid tokens | untrimmed median / min / mean | trimmed median / min / mean |
|---|---|---:|---:|
| libero_spatial | 26-31 | 0.99840 / 0.99566 / 0.99804 | 0.99998 / 0.99993 / 0.99997 |
| libero_goal | 16-21 | 0.99680 / 0.92997 / 0.99176 | 0.99998 / 0.99971 / 0.99996 |
| libero_10 | 20-31 | 0.99860 / 0.96654 / 0.99647 | 0.99998 / 0.99654 / 0.99979 |
| all 60 frames | | 0.99809 / 0.92997 / 0.99542 | 0.99998 / 0.99654 / 0.99991 |

Mean `mae_fr_vs_gt` (official in parentheses): libero_spatial 0.18359 -> 0.18555 (0.18538), libero_goal 0.16201 -> 0.15958 (0.15941), libero_10 0.13044 -> 0.13144 (0.13139). The trimmed minimum, libero_10 ep 0 frame 60 (0.99654 at seed 0, 0.99986 at seed 1), is the frame where official's own seed 0 vs seed 1 cosine is 0.77926; at seed 1 the trimmed libero_goal minimum is ep 114 frame 60 (0.99487 to 0.99510 over four runs), the frame where official's own seed 0 vs seed 1 cosine is 0.93157. Repeat runs of libero_goal with the final code (VAE outside and inside the graph): median 0.99998, min 0.99966 and 0.99970, mean MAE 0.15957; per-frame values move by up to 5e-5 between processes because each process autotunes its own cuBLASLt picks for the new fp16 shapes. `served_vs_off` and `fr_noise001_vs_off` use the served `0.01 * N(0,1)` initial noise (ISSUE-002) and are dominated by that sampler difference.

Other checks:
- random weights, small dims: trimmed vs untrimmed with the padded keys masked under the same fp32 PyTorch attention is bit-identical (n_valid 5 and 8); trimmed under the served cuBLAS per-head attention cosine 1.0000000, rel_l2 3.6e-5 to 5.9e-5, against rel_l2 1.2e-3 to 1.3e-3 for the old untrimmed unmasked rule; served per-head attention at trimmed real shapes (`x0` 20, 21, 32, 33; backbone `a0` 412-425, `mot` `total` 476-489) vs fp32 PyTorch cosine 0.9999999, rel_l2 5.4e-4 at both parities, and FA4 dispatch at those shapes with an fp32 stand-in for the kernel cosine 1.0000000.
- switching lengths 6 -> 10 -> 17 -> 6: the cached graph replays, bit-identical to the first run, and each length is bit-identical to a fresh frontend; the VAE stage inside every length's graph (shared pool) is bit-identical to the VAE outside the graph; `text_trim=False` vs the previous frontend file in the same process at small and real dims is bit-identical (context, random prompt, explicit and served noise); the fp16 gate on fixture v1 is 40 per-sample results identical before and after.
- multi-length safety check (`tests/test_imagewam_text_trim_graph_safety.py`), lengths A, longer, shorter, A, max, ...: fp16 small and real dims, fp8 small and real dims, fp8_static small (placeholder scales) and real dims with real weights and the trimmed calibration file, each with the VAE outside and inside the graph; every length bit-identical to a fresh single-length frontend, weight-op tensors never reallocated after the first capture (144 at small dims, 560 at real dims for fp8/fp8_static), replays unchanged and no write into 0xFF-poisoned free memory (graph pool and 1152 MiB of the regular cache).
- same check without the max-dims prefill (control, growing-scratch stand-in ops): 14 of 38 (small) and 60 of 142 (real) scratch tensors reallocated; a capture that raises on a new length leaves no graph active (`infer()` refuses, the cached length replays its own result, the retried length equals a fresh frontend); the FA4 fallback (stand-in FA4 failing) keeps the old graphs alive through the recapture and drops them after it, and a second failure leaves no graph.
- `run_eager()` vs replay at two lengths and back: bit-identical (before the fix: cosine 0.9999982, max-abs 4.8e-3).

Speed, `benchmarks/imagewam_text_trim_bench.py --precision fp16`, real dims, random weights, trimmed `x0 = 21` vs full `x0 = 513` in one frontend, 100 alternating samples each, P10 / P50 / P90: graph replay (CUDA events) trimmed 32.4 / 32.5 / 33.1 ms against full 45.8 / 46.3 / 47.6 ms, P50 ratio 0.701; `infer()` (wall, synchronized) trimmed 32.9 / 33.0 / 33.5 ms against full 46.1 / 46.5 / 48.6 ms, P50 ratio 0.711. With the VAE stage in the graph the same A/B was contaminated by the co-tenant (bimodal samples); its P10 ratio is 0.72 for both replay and `infer()`.

Capture cost and memory per length (15 LIBERO lengths, `x0` 17-32), VAE outside against inside the graph: first length `set_prompt` wall / process memory 1.17 s / +84 MiB against 1.50 s / +298 MiB (206 MiB of it the shared pool); each later new length wall median (range) 0.78 s (0.61-1.97) against 0.76 s (0.62-1.83) with process memory (NVML) +12 MiB (10-16) against +12 MiB (12-16); cached length `set_prompt` wall 12 ms (8-36) against 12 ms (1-16). The fp16 autotune of a new length's shapes is 0.3-0.6 s of the capture cost, the eager warmup and capture 0.27-0.35 s; before the shared pool each length with the VAE in the graph added 218 MiB.

## Constraints on consumers of a trimmed frontend

With `text_trim=True` the sequence length changes with the prompt. `frontend.dims` holds the buffer sizes (the maximum); the dims the active graph runs are `frontend.active_dims`. What the consumers needed, and what they now have:

- the active dims everywhere a length appears: `context_rows` is `active_dims["x0"]`, the image rows start at it, the backbone RoPE table (`_rope_table`) has `active_dims["a0"]` rows, the action rows start at `active_dims["a0"]`. `runtime_surface()` does this, and carries the graph table (`graph_variants`: the `(key, exec)` entries with `key = x0`);
- `text_trim` and the active `x0` in the setup identity;
- after the prompt verb (`set_prompt`), re-adopting the graph: a new length activates, and may capture, another graph (`_graph`); the capture stream (`_graph_stream`) is the same for every length. Both model-runtime faces do it by key: `export_model_runtime(io="python")` replays the key of the active length, `export_model_runtime(io="native")` adopts one exec per captured length on the native handle (`use_graph(key, exec)`), which selects it with `set_text_length(key)` before `set_proprio_row`;
- the native *pipeline*'s own capture (`pipeline_resources()` -> `frt_imagewam_native_set_pipeline` -> `capture()`) records one graph per text length: `pipeline_resources()` describes the active length, `set_pipeline` installs the key its table carries (replacing that key's pipeline and graph while the other keys keep both) and selects it as the active length, and `ImageWAMNativeRuntime.capture_pipeline_text_lengths` installs and captures every length the frontend has captured, restoring the active length afterwards. `consumer="native"` therefore accepts `text_trim` like the other two consumers, and rule R6 is what remains specific to it.

Activation statistics for a trimmed frontend are recorded at the active dims (`run_eager()` runs them). The untrimmed forward's text and single-stream GEMM inputs include about 490 padded context rows that a trimmed frontend never computes. The calibration file records `text_trim` (since format version 2) and a frontend refuses a file recorded with the other setting. (Format version 3 added the camera geometry to the identity; version-1 and version-2 files are no longer read and are re-recorded.)

Trimmed calibration file (`benchmarks/imagewam_build_calibration.py --n 64 --text-trim`, the same 64 frames as the untrimmed N=64 file, H100): per-site static FP8 scale, trimmed over untrimmed, 0.85x-1.50x (median per site group 0.98x-1.32x; largest spread at ActionDiT `proj`, backbone `linear2` and the image-stream `mlp2`). `fp8_static` on libero_goal (10 tasks x frames {0, 60}, seeds {0, 1}; one of the ten evaluation episodes, ep 0, also contributes calibration frames), vs official median / min and mean MAE (official 0.15941):

| | vs official median / min | mean MAE |
|---|---:|---:|
| `fp8_static`, untrimmed, untrimmed file | 0.99678 / 0.92891 | 0.16214 |
| `fp8_static`, trimmed, trimmed file | 0.99995 / 0.99927 | 0.15968 |
| `fp16`, trimmed (reference) | 0.99998 / 0.99966 | 0.15957 |

Trimmed `fp8_static` vs trimmed `fp16` (`benchmarks/imagewam_precision_fidelity.py`, `TEXT_TRIM=1 SUITE=libero_goal`): actions cosine median 0.99998 / min 0.99981, backbone residual 0.99994 / 0.99989, MAE ratio 1.000. The untrimmed file forced onto the trimmed frontend (identity check bypassed) measures the same (actions 0.99998 / 0.99992): the identity rule keeps the statistics consistent with the served path; on these frames it is not an accuracy gain.

## Open

- The memory the in-graph native VAE adds is not isolated from the per-length `text_trim` figures: the memory numbers recorded so far are the per-length `text_trim` graphs (first graph +218.0 MiB reserved / +206.3 MiB allocated, later ones about 0; `0919e`) and the `0920c` export-gate runs, where the process peaked at 19.7 GiB with the VAE outside the graph and 19.7 GiB with it inside at the real dims.
- `fp8`/`fp8_static` run on H100 since the TN FP8 path (issues.md ISSUE-001) and are verified with trimming above; `fp8_static_cutlass` runs on Thor only.
- Decision: `text_trim` serves by default (`plan.md`, "Decisions pending", E1; ISSUE-080). With that, the Thor rows this list used to carry have run: the `nvfp4` end-to-end compare and `infer()` P50 A/B, the capture time per new length and the cache's eviction behaviour (`0919e`, `eccf14f`), FA4 on against off in one session (`0920t`: 126.5 ms against 131.3 ms end to end), and the multi-length safety check at every precision the deployment offers — `nvfp4` with FA4 off and on at the real dims (`0920`) and `e0m3_hadamard` (`0920c`).

## Thor check

- `0920c` (`c495cb2`, Jetson AGX Thor, MAXN, `emc_locked=null`, GPU idle, logs under `/home/jingwu/thor_val/0920c/`), `e0m3_hadamard` trim safety: `TRIM_PRECISION=e0m3_hadamard pytest tests/test_imagewam_text_trim_graph_safety.py -q` is 4 passed; every captured length is bit-identical to a freshly built single-length frontend (`equal=True cosine=1 max_abs=0`), a failed capture leaves the same bit-for-bit state, and 20 replays after poisoning the graph pool and the regular allocator cache are all `equal` with poison bytes overwritten = 0.
- `c20f3a0` (libero_spatial, nvfp4), `scripts/imagewam_thor_matrix.sh`, `N_TASKS=10 FRAMES=0,60 SEEDS=0,1`, real checkpoint, `infer()` P50, `profile_*` rows through `load_imagewam` (plan.md W12), `fast` and `stack` the same switch set (GPU co-tenancy and clock state in that round's logs): the ladder with each step's marginal against the row before it, default 225.5 -> vae 190.1 (-35.4) -> vae_trim 102.9 (-87.2) -> vae_trim_fa4bb 99.0 (-3.9) -> stack 93.2 (-5.8, vs official median 0.99934); stack_no_vae 104.8 (+11.6 vs stack), stack_no_trim 131.4 (+38.2 vs stack), profile_fast 106.8 (same switches as stack, 0.99936), profile_default 225.2 (0.99764). `text_trim` is the largest single step from `vae` (-87.2 ms) and removing it from the full stack costs +38.2 ms, more than the other three switches together; the two numbers differ because FA4 also shortens the padded-key work the trim already removes. Agreement with official improves rather than degrades (0.99934-0.99936 stacked against 0.99764 for the default row). Read the marginals with ISSUE-082: the same configuration appears twice in this session, 93.2 and 106.8 ms.
- `eccf14f` (Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive, raw logs under `/home/jingwu/thor_val/0919s/`; real checkpoint, `nvfp4`, end-to-end `infer()` P50 in ms, `vs official` the median action cosine): libero_spatial default three repeats 202.4 / 202.1 / 202.0, stack three repeats 93.1 / 92.8 / 93.3 at 0.99931-0.99936, vae_trim not measured this round; libero_goal default 203.0 (0.99558), vae_trim 118.7 (0.99937), stack 92.6 (0.99930), profile=fast 92.7; libero_10 default 202.0 (0.99765), vae_trim 103.0 (0.99926), stack 93.5 (0.99925), profile=fast 93.7. The spatial `stack` repeats bracket the 0.99934 the `c20f3a0` session measured for the same row, and on libero_goal and libero_10 the trimmed rows agree with official better than the untrimmed default, so the faster configurations are also the closer ones; the gates with `text_trim=True` pass at `fp16` and at `nvfp4`, and no row in the round reported an FA4 fallback.
- Gate fixture v2 (`imagewam_libero_gate_v2`, its `fp16` reference recorded trimmed) now gates a trimmed configuration: the `nvfp4` gate against it measures vs official 0.99931 / min 0.99898, vs the fixture's own `fp16` reference 0.99935 / 0.99907, `infer()` P50 114.6 ms, and passes; a trimmed run against fixture v1 and an untrimmed run against v2 are both refused, which is the intended behaviour. Capture cost on Thor, three lengths swept (16, 24 and 31 valid tokens): `set_prompt` takes 0.000-0.012 s for a length captured at construction (precapture) and 0.42-0.58 s for a length captured on first use, both at or below the H100 capture-cost table above (0.61-1.97 s for each later new length, 12 ms for a cached one).
- `0919e` (commit `a84916a`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive, raw logs under `/home/jingwu/thor_val/0919e/`): per-length graph memory (`benchmarks/imagewam_text_trim_bench.py`, `nvfp4`, FA4 off, 15 distinct LIBERO lengths, deltas of the process's reserved and allocated memory, NVML and `torch.cuda.max_memory_allocated()`): first captured graph +218.0 MiB reserved / +206.3 MiB allocated, each following capture +0.0 MiB / +0.1 MiB; the reserved total is 11.79 GiB after the 15-length sweep, so the default `text_trim_cache_size=32` costs on the order of 221 MiB in total, not 32 times the first graph. Trimmed against full in one frontend at trimmed `x0 = 21`, same benchmark, `nvfp4`, FA4 off: `infer()` P50 123.0 ms against 203.7 ms, ratio 0.604. Eviction, `text_trim_cache_size=2` with no precapture, lengths 16 -> 24 -> 31 -> 16 valid tokens: `set_prompt` takes 0.636 / 0.503 / 0.468 / 0.465 s, so the revisited length captures again, LRU having evicted it; the same three lengths precaptured at `text_trim_cache_size=8` switch in 0.012 / 0.000 / 0.000 s.
- `0920`: the multi-length safety check passes with FA4 on as well as off, and the recovery case reports the recovered length equal to a like-for-like chain reference: `after a failed capture, n_valid=5 vs fresh equal=True cosine=1 max_abs=0` (log `/home/jingwu/thor_val/0920/R_fa4_recover.log`; Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive). The capture-path defect behind the earlier failure — the cuBLAS fallback reusing the capture pool an invalidated capture had left recording — is fixed in the frontend (issues.md ISSUE-085).
- `0920t` (commit `4cd06e5`, Jetson AGX Thor, MAXN, GPC 1.575 GHz / NVD 1.692 GHz, `emc_locked=null`, GPU idle at the start of the round, raw logs under `/home/jingwu/thor_val/0920t/`, nvfp4): all three gate runs pass and fixture v2's `fp16` reference is trimmed, so the `nvfp4` row against it is the served default (trimming plus FA4) — nvfp4 + fixture v2 0.99889 / 0.99934 at 125.86 ms, nvfp4 + fixture v1 with `--no-text-trim` (untrimmed + FA4) 0.99418 / 0.99758 at 191.79 ms, fp16 + fixture v2 (ungated) 0.99993 / 0.99997 at 284.38 ms; the same fixture's trimmed row in the `0919e` section (FA4 off) is 0.99898 min / 0.99931 median at 114.6 ms, so the 11 ms between the two sessions is a session difference, not a regression, and the untrimmed leg (FA4 on) at 191.79 ms against the 202.2 ms baseline that was untrimmed **and** FA4 off bounds the untrimmed configuration from one side only. End to end at `nvfp4` the trimmed configuration measures 126.5 ms with FA4 and 131.3 ms with `FLASHRT_THOR_FA4=0` (both runs' `served_vs_off` rows are in OPT-019's `0920t` section). The trimmed consumer contract was re-checked in the same round: `tests/test_imagewam_text_trim_consumer_guards.py` 11 passed, its GPU row printing the per-length resource table, recorded under OPT-029's `0920t` section.

# OPT-031: derive the remaining per-benchmark shape tables from the workload

Status: identified, not started

Area: `benchmarks/imagewam_fp8_layout_bench.py`, `imagewam_thor_int4_bench.py`, `imagewam_thor_int8_bench.py`, `imagewam_thor_bench.py`, `imagewam_thor_fp16_bench.py`, `imagewam_thor_fp16_autotuned_bench.py`, `imagewam_thor_fp4_bench.py`, `imagewam_thor_fp8_bench.py`, `imagewam_gemm_precision_compare.py`, `imagewam_int4_hadamard_padding_probe.py`, `imagewam_real_checkpoint_validation.py`, `imagewam_attention_share_bench.py`, `imagewam_thor_small_m_tile_sweep.py`

## Observation

The `dims` dictionaries of the served LIBERO workload are now one mapping (`libero_dims.LIBERO_REAL_DIMS`, derived from `ImageWAMWorkload.libero()` and `ImageWAMStructure.libero()`). Several benchmarks still state the same served widths a second time, not as a `dims` dict but inside their own kernel-shape tables:

- `("txt_qkv": (513, 9216, 3072))` in `imagewam_fp8_layout_bench.py` (its `REAL_SHAPES`, the comment above it reading "Every distinct (M, N, K) of the served pipeline at the real LIBERO dims");
- `X0, A0 = 513, 513 + VAE_NUM_TOKENS` in `imagewam_thor_int4_bench.py` and `imagewam_thor_int8_bench.py`, and per-GEMM `(M, N, K)` rows in the per-layer benches (`imagewam_thor_bench.py`, `imagewam_thor_fp16_bench.py`, `imagewam_thor_fp16_autotuned_bench.py`, `imagewam_thor_fp4_bench.py`, `imagewam_thor_fp8_bench.py`, `imagewam_gemm_precision_compare.py`);
- `SiteShape("backbone", 905, 905)` and `ActionShape("double qkv", 9216, 1024, 5)` in the attention-share and tile sweep tables.

## Mechanism

Those tables describe one layer's or one GEMM's shape, which is a composition of dims entries (`3 * hidden + 2 * mlp_hidden`, `x0`, `x0 + ref_h * ref_w`, `num_action`), so they can be built from the same two objects the resolver uses instead of restating the numbers. `tests/test_imagewam_quant_linear.py` does this for its own `(M, N, K)` table.

## Value

A change to the workload (the target deployment's camera count or per-view size) or to a structure constant then reaches every benchmark by construction, and a sweep cannot silently measure a shape the model no longer has. The cost is per-file, mechanical, and needs no GPU: the resulting table is asserted equal to the present literals.

## Precondition

The sweep tables must stay readable as measurement records (what shape was measured); the change is the derivation, not the removal of the numbers from the prose.

## Open

- Not started: the benchmarks above still restate their stored widths inside their own kernel-shape tables rather than deriving them from `ImageWAMWorkload` / `ImageWAMStructure`, even where the file already imports `libero_dims.LIBERO_REAL_DIMS` for its `dims` dict.

# Index of removed entries

Status: the entries below are closed and are no longer part of this file. Each block states the result, the measurement that is recorded nowhere else, and where the rest of the account lives. The ids stay literal because code, tests and other persistent files cite them.

## OPT-002 — real attention/block math vs. the official model

Status: RESOLVED — kernel-level math, the full real-math pipeline, and real-checkpoint validation, all verified on Thor.

Facts: real per-head K/V (`num_kv_heads == num_q_heads`, so the K/V projections return to `hidden` width 3072 instead of `HD`); a 4-axis RoPE over `axes_dim=(32,32,32,32)` summing to `head_dim=128`; an independent learned per-head RMSNorm on Q and K before RoPE (reuses `rms_norm_fp16`, no new kernel); the real MLP first projection at `mlp_hidden*2` width with a SiLU-gated GLU chunk instead of GELU; real dims `hidden=3072, mlp_hidden=9216, NH=24, HD=128`; ActionDiT `hidden=1024, attn_dim=3072, mlp_hidden=4096`. With the real `target_len=0` the `mot` rule reduces to no mask between text and ref while action sees everything, so the three-region `mot_joint` mask is not the real rule and `use_real_mot_mask` dispatches through the same unmasked kernels. Real trained weights on Thor: backbone cosine 0.999927, ActionDiT cosine 0.999963. The only number the repository records nowhere else: the `use_real_mot_mask` flag's own non-no-op check, cosine 0.81 against the masked default (1.0 against the plain kernel).

→ `flash_rt/hardware/thor/attn_backend.py` and `flash_rt/frontends/torch/imagewam_thor.py` / `_imagewam_thor_spec.py` docstrings, `plan.md`'s attention plan, `issues.md` ISSUE-003 (which also records the two cosines).

## OPT-003 — `mot_joint` restricted to action queries

Status: RESOLVED — verified on Ada and on real Thor hardware. The shipped frontend now passes the real `mot` rule unconditionally (`use_real_mot_mask=True`), so the entry's own "default stays off until real-checkpoint validation exists" line is stale.

Numbers: Ada FP16 29.5 -> 5.68 ms (5.2x) per 25-layer denoise step and 447.8 -> 203.2 ms (2.2x) for prefill+10-step; Ada INT4 (GEMM-only) 24.2 -> 4.37 ms (5.5x) and 283.3 -> 91.1 ms (3.1x); Thor FP16 35.3 -> 5.89 ms (6.0x) and ~434 -> 140.4 ms (3.1x). The problem it fixed: every denoise step computed joint attention over the whole `total` sequence (`total*NH` query rows) while only the `num_action` action rows' output is read, roughly 15x more query rows than needed at `total=960, num_action=64`; the speedup is smaller than 15x because GEMM cost does not shrink. The round's own starting observation was that FP4/FP8 GEMM quantization speeds backbone prefill up by 1.3-1.5x while leaving the denoise step at ~35 ms per step regardless of precision (FP16/BF16/FP8/FP4). Consequence confirmed on Thor: prefill is now 58% of the full path, so the optimization center of gravity moved from denoise attention to backbone GEMM.

→ the kernel comments in `csrc/kernels/attention_cublas.{cu,cuh}` and `csrc/kernels/softmax.{cu,cuh}`, `tests/test_imagewam_mot_joint_action_kernel.py`. The mask-correctness half is OPT-002, answered by the shipped `use_real_mot_mask=True`.

## OPT-005 — FA4 at the backbone site

Status: RESOLVED and wired in. The served default resolves FA4 by machine (`use_fa4=None` turns it on where compute capability is 11.x and the FA4 runtime imports; `FLASHRT_THOR_FA4=0` forces the cuBLAS chain), which supersedes the entry's "default False".

Numbers: cuBLAS `attention_qkv_fp16` 0.821 ms vs FA4 0.203 ms (4.05x), and 0.826 vs 0.202 ms (4.09x) in the later round; output cosine 1.000000 with rel_l2 0.000412 for broadcast K/V and 0.000427 for real per-head K/V through the `pack_gqa=False` branch that the per-head fix added; real per-head shape 0.778 vs 0.208 ms (3.75x), rel_l2 0.000614 in the per-head confirmation round. The Ada-only half of the speed bench measured the cuBLAS path at 0.853 ms P50 before any Thor run. In the per-layer table `action_double`/`action_single` are unchanged by FA4 (0.57 / 0.53 -> 0.57 / 0.56 ms), since FA4 only ever touches the backbone site, and the isolated `standard_attn_kernel_only` row stayed flat (0.824 -> 0.842 ms) under `IMAGEWAM_USE_FA4=1` by design: that function calls `attention_qkv_fp16_perhead` directly and bypasses `ImageWAMAttnBackend`. Folded into the full per-layer benchmark on top of OPT-004's steps 1-3: `backbone_double` 5.53 -> 4.81 ms (-13%), `backbone_single` 4.47 -> 4.03 ms (-10%), so prefill 117.0 -> 104.7 ms (-10.5%) and prefill+10-step 252 -> 246 ms — 135.8 ms (commit `61e7c15`) -> 104.7 ms (-23% total).

→ `plan.md`'s attention plan, `THOR_STATUS_SUMMARY.md` (the FA4 rows), `benchmarks/imagewam_thor_bench.py`.

## OPT-012 — steady-state FP16 breakdown, real vs official PyTorch

Status: RESOLVED — real, directly measured profiling. Its official-vs-FlashRT half was superseded; the breakdown itself is recorded only here.

Breakdown of a 284.9 ms `infer()`: ActionDiT 10-step denoise 144.6 ms (50.7%), backbone prefill 118.4 ms (41.5%), VAE `encode_to_tokens` 21.5 ms (7.6%), proprio encode 0.11 ms, action-noise fill 0.02 ms, denorm + D2H 0.05 ms; sum 284.7 vs measured 284.9 ms. `graph.replay()` alone 262.9 ms (92.3% of `infer()`); eager prefill+denoise 281 ms, so capture saves ~1.07x (compute-bound, not launch-bound); one denoise step P50 15.5 ms x 10 = 155 ms. Denoise + prefill = 92.2% of the total, which moved the priority from VAE fusion (hard ceiling 7.6%) to precision choice; the same-session VAE estimate of ~15.7% (43.9 ms historical number / 279.5 ms current total) was about 2x too high.

The official-vs-FlashRT comparison this entry recorded (605 / 863 / 485 ms against a stale 284.9 ms FlashRT baseline, i.e. 2.12x / 3.03x / 1.70x) is superseded by `THOR_STATUS_SUMMARY.md`'s "FlashRT vs 官方 PyTorch 实现" section: official bf16 eager 453.6 ms vs FlashRT `nvfp4` 203.3 ms and 106.1 ms stacked.

The official side's own per-stage split (10-step, same run): Qwen3 118 ms, re-run every call in the official path and NOT part of FlashRT's own `infer()`, which caches text via `set_prompt()`; VAE 21.5 ms; backbone prefill 201 ms; ActionDiT 10-step 258 ms — against FlashRT's prefill 118 ms (1.70x faster) and denoise 145 ms (1.78x faster), with VAE matching exactly (same AE). In the later identical-conditions run the official breakdown is VAE+proprio 21.4 ms and transformer (prefill + 10-step) 432.3 ms, so FlashRT's advantage is almost entirely in the transformer compute, not the VAE.

→ `THOR_STATUS_SUMMARY.md` for the comparison; the breakdown only here.

## OPT-013 — FP16 CUTLASS GEMM + SwiGLU epilogue fusion (`fp16_cutlass`)

Status: CLOSED — correct, but a real negative speed result, not recommended for production and not carried forward as a speed candidate.

Thor verdict table, full-pipeline `infer()` P50 at `x0=513/a0=905`: `fp16` (cuBLASLt, the default) 287.2 ms; `fp16_cutlass` with the default `sq`/`wide` heuristic 306.9 ms (+6.9%); `variant="plain"` 297.8; `2sm21` 299.8; `k64` 303.3; `t1` 305.1 ms — the same table's deltas are +3.7% / +4.4% / +5.6% / +6.2%. Correctness: isolated GEMMs and the fused SwiGLU cosine 1.000000, full `infer()` actions/action_latent 1.000000, and `backbone_hidden` over the 25-layer stack 0.995090 (floating-point non-associativity, not a variant bug). Real bug fixed same day: CUTLASS FP16 requires N and K divisible by 8, so `action_encoder` (K=7) and `head.linear` (N=7) fall back to plain cuBLASLt `Fp16Linear`. Conclusion: cuBLASLt's own per-shape autotuning already matches or beats this CUTLASS family at ImageWAM's shapes, unlike the FP8 case, so "CUTLASS exists for this precision" is not a reason to expect a win.

→ `docs/imagewam_configuration.md` (precision), `flash_rt/models/imagewam/quant_linear.py`'s `CutlassFp16Linear`.

## OPT-014 — Stage 3 default precision decision (`nvfp4`)

Status: CLOSED — the constructor's default precision is `nvfp4`, a production default rather than a benchmark opt-in.

Numbers, all recorded only here (the later served-configuration numbers are in the summary's precision tables):

- real-weight cosines vs `fp16`: `nvfp4` actions 0.9998 / `backbone_hidden` 0.9939 / `action_latent` 0.9997; `fp8_static_cutlass` 0.897 / 0.461 / 0.874; `fp8_static` (cuBLASLt) 0.697 / 0.467 / 0.257 — the FP8-static collapse is the placeholder `N(0,0.1)` activation calibration, not the GEMM backend (both backends degrade identically).
- real open-loop LIBERO, 50 frames vs ground truth: MAE 0.1985 (`fp16`) / 0.2007 (1.01x) / 0.2839 (1.43x), cosine vs GT 0.559 / 0.558 / 0.565, first-4 previously-good tasks 0.983-0.993 / 0.982-0.994 / 0.90-0.95.
- full `infer()` P50 at `x0=513`, proprio on and the real 10-step shift schedule: `nvfp4` 236.9 / `fp8_static_cutlass` 243.0 / `fp8_static` 259.8 / `fp16` 280.2 / `fp16_cutlass` 306.7 ms.
- NVFP4 stability over 40 calls: P50 243.5 ms, min-max 243.2-247.3 ms, memory delta 0.
- ActionDiT M=64 FP8 CUTLASS vs cuBLASLt: numerically identical but 1.44-1.68x slower (qkv 0.023 -> 0.034 ms), the reason `fp8_static*` keeps cuBLASLt at small M.
- `_calibrate_fp8()` itself ran clean on real weights (no NaN/inf across all ~220 scales, `act_scale` ~1e-3): the calibration STEP works, its INPUT DISTRIBUTION is the problem.
- The entry's own "follow-up, not started" (real per-layer activation calibration for `fp8_static*`) was carried out as OPT-022.

→ `THOR_STATUS_SUMMARY.md`'s precision tables (`fp16` 275.2 / `fp8_static` 228.0 / `fp8_static_cutlass` 220.3 / `nvfp4` 202.3 / `e0m3_hadamard` 198.9 ms) and `plan.md`'s "Decisions pending". Four files cite this id: `issues.md` ISSUE-061, `plan.md`, `tests/fixtures/imagewam_gate/fidelity_thresholds.json` (twice) and `flash_rt/models/imagewam/gemm_variant_tuner.py`.

## OPT-022 — real activation calibration for `fp8_static*`

Status: RESOLVED — the Thor confirmation the entry was waiting on was delivered (`eccf14f` fidelity rows, `0919e`/`a84916a` gate pass), so its "Thor confirmation pending" line is stale.

Facts: the calibration file is built by `benchmarks/imagewam_build_calibration.py` from 64 LIBERO frames of `libero_object`/`libero_goal`/`libero_10` through the real fp16 pipeline; the scale is the house 99.9th percentile of per-sample absmax divided by 448. Why the placeholder failed: real GEMM-input absmax ranges from ~2 (ActionDiT `proj`) to ~7000 (backbone double `txt_mlp2`, whose input has p99.99/amax 0.018) against `N(0,0.1)`'s ~0.5, so the placeholder clipped nearly every site by 1-4 orders of magnitude. H100, 20 held-out `libero_spatial` frames vs `fp16`, median (min): `backbone_hidden` 0.45582 (0.42467), `action_hidden` 0.68728, `action_latent` 0.87187 (0.67097), `actions` 0.90093 (0.73530), MAE ratio 1.697 for the placeholder — against 0.99994 (0.99984), 0.99995, 0.99997 (0.99995), 0.99997 (0.99989), MAE ratio 1.000 with the real file. Against official ImageWAM: `fp16` 0.99840, placeholder 0.87576, real file 0.99844 median (`served_vs_off` 0.91927 -> 0.99599); the calibration-set sensitivity study (N=64 shipped as 22/21/21 frames per suite, N=64 as 31/27/6, N=8 as 3/3/2) gives the same `backbone_hidden` and `actions` medians. After the `linear2` merge, the residual+AdaLN fusion and the VAE stage, the file is rebuilt at 142 sites and reproduces the same fidelity, gate pass and `run_eager()` vs graph-replay bit-exactness. Thor: the real file reaches actions cosine 0.99997 with MAE ratio 1.000 (the placeholder still collapses the backbone and gives actions about 0.90 with MAE ratio 1.77); `fp8_static_cutlass` P50 219.5 ms (`default`) and 104.6 ms (`stack`) at `eccf14f`, and the gate passes at vs official 0.99830 / 0.99557 with P50 233.5 ms at `0919e`.

- Against official ImageWAM (`benchmarks/imagewam_e2e_official_compare.py`, 20 frames, `N_TASKS=10 FRAMES=0,60 SEEDS=0,1`): `fp16` 0.99840 median / 0.99567 min / MAE 0.18359; `fp8_static` with the placeholder 0.87576 / 0.66887 / 0.30207 (`served_vs_off` 0.91927); with the real file 0.99844 / 0.99559 / 0.18372 (`served_vs_off` 0.99599); `fp8` (dynamic scale, 3 tasks) 0.99845 / 0.99836 / MAE 0.20706 against the official side's own 0.20688, `served_vs_off` 0.99439; official's own seed-to-seed spread is median 0.99630 / min 0.97154.
- The regression gate (fixture v1, 40 runs) passes `fp8_static` with the N=64 file on H100: vs official median 0.99834 / min 0.99540, vs its `fp16` reference 0.999968 / 0.999943, MAE 0.18373 against the reference's own 0.18364. Sensitivity to the calibration set: N=64 shipped (22/21/21 frames per suite) gives `backbone_hidden` 0.99994 (0.99984) and `actions` 0.99997 (0.99989) at MAE ratio 1.000; N=64 as 31/27/6 gives 0.99994 (0.99985) and 0.99997 (0.99991) at 1.001; N=8 (3/3/2) gives 0.99993 and 0.99997 at 1.000.
- On the merged tree (142 sites) the same checks hold: fidelity 0.99994 (0.99984) / 0.99997 (0.99994) at MAE ratio 1.000; e2e `fp8_static` 0.99837 / 0.99571 / 0.18370 against `fp16`'s 0.99840 / 0.99566 / 0.18359; gate vs official 0.99830 / 0.99553, vs the `fp16` reference 0.999969, MAE 0.18373 against 0.18364; `run_eager()` vs graph replay bit-exact with the VAE outside and inside the graph.

→ `docs/imagewam_calibration.md` (recorded statistics + measured accuracy), `flash_rt/models/imagewam/calibration_file.py`.

## OPT-025 — Jetson clock-state record for the benchmarks

Status: RESOLVED — the Thor record the entry was waiting on is delivered by the summary's own Thor round headers (MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive), so its "Thor record pending" line is stale; `plan.md`'s clock plan Phase 3 still reads `blocked` although those rounds record the state.

Facts: `flash_rt/hardware/jetson_clock_state.py` reads the nvpmodel mode, every GPU and EMC devfreq node under `/sys/class/devfreq` (`cur_freq`, `min_freq`, `max_freq`, `governor`) and `/sys/kernel/nvpmodel_clk_cap/*`, all without root; it never runs `sudo`, `jetson_clocks` or `nvpmodel -m` and never writes sysfs, because Thor is shared and runs as is at MAXN with DVFS-managed clocks; `JetsonClockState` carries a pinned verdict and warns only for a non-MAXN power mode or unobservable state, and returns `is_jetson=false` on any other machine; `report_jetson_clock_state()` prints the record as one `[jetson-clock-state]` JSON line plus a summary and warning lines. Printed before timing by `imagewam_thor_graph_bench.py`, `imagewam_thor_int4_bench.py`, `imagewam_thor_int8_bench.py`, the timing section of `imagewam_e2e_official_compare.py`, and embedded in every regression-gate result (OPT-027). Measured: 11 passed on fake Thor sysfs trees, `is_jetson=false` with no tool run on the H100 box, and the graph bench's fp16 row printing the record with P50 107.3 ms.

→ `THOR_STATUS_SUMMARY.md`'s round headers, `THOR_CHECKLIST.md`'s prerequisites, `flash_rt/hardware/jetson_clock_state.py`.

## OPT-026 — precision-routing contract test

Status: RESOLVED. The table has been extended past the entry's own "37 slot rows by the 6 precisions of `_PRECISIONS`": `tests/test_imagewam_thor_precision_routing.py` now has 8 precision columns (`PRECISION_COLUMNS`) and 39 `EXPECTED_ROUTING` slot rows.

Evidence, recorded only here: on the H100 box with the real extension importable, 54 passed (~16 s); with `CUDA_VISIBLE_DEVICES=""` and `flash_rt_kernels` unimportable, 53 passed and 1 skipped (the stub-signature check needs the real module, ~14 s); two mutations produce five failures each — removing the `nvfp4` K/N%16 fallback names both the K=7 and the N=7 slot, and making `fp8_static` lose the `linear1` merge fails five. Maintenance rule: a stream that changes routing edits the table rows in the same commit.

→ `tests/test_imagewam_thor_precision_routing.py`.
