# Inference optimization methodology (Pi0.5, ImageWAM)

Chinese version: [`inference_optimization_methodology_zh.md`](inference_optimization_methodology_zh.md).

How FlashRT takes a robot policy from a reference implementation to a fast
served runtime, written as an ordered procedure. Every step cites where it
was applied to Pi0.5 or ImageWAM and what it measured. The order matters:
each phase assumes the previous one holds.

| Phase | Question it answers | Exit condition |
|---|---|---|
| 0. Reference and gates | What is "correct"? | A fixture, a reference output and a numeric gate exist |
| 1. Static execution | Can the whole call be replayed? | One (or a few) CUDA graphs, no host work inside |
| 2. Measurement | Where does the time go, and why? | A module × operator × precision table with a bottleneck per cell |
| 3. Remove work | What computation is never needed? | Dead and repeated work removed, algorithm-level savings checked |
| 4. Fusion | Which kernels are launch- or traffic-bound? | Fused kernels A/B-tested on every target device |
| 5. Precision | Which tier is fastest at acceptable accuracy, per device? | Tier chosen per device with calibration and gates passing |
| 6. GEMM and attention backends | Is each shape on its best kernel? | Per-shape kernel choice measured at real M |
| 7. Task-level validation | Does the policy still succeed? | Closed-loop success matches the reference within its confidence interval |

## Phase 0: reference, fixture, gates

- Run the official implementation once and keep its outputs as the
  reference. Compare against an FP32 (or official BF16) reference, never
  against the runtime itself (`docs/calibration.md` §10).
- Fix the inputs: a versioned fixture of real frames, prompts, proprio and
  initial noise (`flash_rt/datasets/imagewam_gate_fixture.py`,
  `tests/gate_imagewam_libero.py`; the Pi0.5 FP4 harness uses an
  8-observation fixture with matched noise seeds,
  `docs/pi05_thor_decoder_fp4_e2e.md`).
- Gates used in this repository:
  - Pi0.5 bring-up: action cosine ≥ 0.995 vs the reference; deployment
    threshold 0.998 (`docs/adding_new_model.md` §3, `docs/calibration.md` §10).
  - Pi0.5 FP4: action cosine ≥ 0.999 global, ≥ 0.995 per sample
    (`docs/pi05_thor_decoder_fp4_e2e.md`).
  - Refactors that must not change numerics are checked bit-exact
    (`torch.equal`), e.g. the exec contract (`docs/exec_contract.md` §8.2),
    the native C++ producers (`docs/pi05_native_cpp.md`), and ImageWAM
    fusions such as `fuse_res_norm` and `qkv_split_norm_rope_fp16`.
- Bit-exactness is defined against the current default path, not a legacy
  one. `qkv_split_norm_rope_fp16` was first bit-exact with the old
  `rms_norm_fp16` only; once `vec_rms_norm` became the default it was
  rewritten to match the new reduction order (`opportunities.md` OPT-032).

## Phase 1: static execution

Goal: the whole `infer()` is a graph replay plus input staging.

- Capture the full call as CUDA graph(s). Pi0.5 captures SigLIP, encoder
  and the 10-step decoder (≈2,840 nodes vs ≈21,000 in the TensorRT/Myelin
  engine, `docs/optimization-details.md` §1.6); ImageWAM captures prefill
  and the 10-step denoise loop, and the VAE stage can go inside the graph
  (`vae_graph_input`, OPT-021).
- Capture rules (`docs/kernel_fusion.md` §6, `docs/adding_new_model.md` §2):
  pointers and ints only; every buffer allocated at load; no allocation,
  sync, `.cpu()` or data-dependent Python branches inside the captured
  region; warmup shapes equal to served shapes; scalar timesteps built on
  the device (`docs/pi05_scalar_capture.md`).
- Precompute everything that does not depend on the observation:
  per-step AdaRMSNorm / AdaLN modulation and time embeddings at
  `set_prompt` (Pi0.5 removed 370 dense GEMMs from the graph,
  `docs/optimization-details.md` §1.4; ImageWAM precomputes
  `action_mods` / `head_mods` per step), text encoding once per prompt.
- Variable-length inputs: either one graph per exact length with a cache
  (Pi0.5 `state_prompt_mode="exact"`; ImageWAM `text_trim=True`, OPT-030)
  or one max-length graph with masking (Pi0.5 `state_prompt_mode="fixed"`,
  FA2 `seqused`). Precapture known lengths so serving never captures
  (`precapture_text_lengths`).
- Zero-copy handoff between stages: producers write the buffer the next
  stage reads (`docs/exec_contract.md` §5b).
- Thor: graph instantiation varies by about ±2 ms; Pi0.5 recaptures until a
  fast schedule is found (`docs/optimization-details.md` §1.6).

## Phase 2: measurement before optimization

- Build a grid: model module × operator class × precision × device. The
  ImageWAM profiles do this with `benchmarks/imagewam_module_operator_profile.py`
  (`docs/imagewam_rtx5090_profile.md`, `docs/imagewam_thor_profile.md`,
  `docs/imagewam_profile_compare.html`).
- Cross-check attribution: graph-replay kernel time must equal `infer()`
  wall time within a small margin (ImageWAM 5090: within 0.2–1.8 ms), and
  kernels per iteration must be equal across profiled iterations (a
  throwaway first profiler session avoids dropped CUPTI records).
- Measure the ceilings on the same device (dense GEMM peak per precision,
  DRAM bandwidth) and classify each cell:

  | Bottleneck | Signature | ImageWAM example |
  |---|---|---|
  | compute | GEMM near measured peak | 5090 backbone single-stream GEMM, fp8 ≈100% of peak at M = 416 |
  | weight bandwidth | GEMM time ≈ weight bytes / DRAM bandwidth | 5090 ActionDiT GEMM at M = 64, fp16 75–84% of DRAM bandwidth |
  | kernel efficiency | far from both ceilings | 5090 NVFP4 ActionDiT GEMM at 19–25% of bandwidth |
  | launch / latency | many 1–3 µs kernels | 5090 ActionDiT non-GEMM kernels, ≈1,800 per infer |
  | bytes moved by small kernels | elementwise kernel time ≈ bytes / achievable bandwidth | Thor `silu_glu_merged` 5.7 ms (10× the 5090) |

- Record per-device conclusions separately: the same kernel can be
  launch-bound on RTX 5090 and bandwidth-bound on Thor.

## Phase 3: remove work

Algorithm- and structure-level removal usually beats kernel tuning.

- Compute the shared prefix once and let the iterative part read its KV
  (ImageWAM: the action expert reads the backbone's per-layer prefix KV;
  Pi0.5: the decoder reads the encoder's K/V cache).
- Drop computation nothing reads: ImageWAM's last backbone block only
  needs K/V (`docs/imagewam_last_block_kv_only.md`); text trimmed to its
  valid length (OPT-030).
- Measure how the cost scales with the loop count and token count before
  touching kernels (`docs/imagewam_rtx5090_profile.md`: the denoise loop
  is linear in steps; fewer image tokens shorten only prefill).
- Check the structure of the iterative sampler before assuming every step
  is needed. For the ImageWAM LIBERO checkpoint the velocity Jacobian is
  ≈ (1/t)·I (`benchmarks/imagewam_velocity_jacobian_study.py`), the
  prediction x̂₀ = x − t·v barely depends on x, and evaluating the action
  expert at grid points {0, 9} of the 10-step grid matches the 10-step
  result within 0.0006 action MAE (`benchmarks/imagewam_few_step_study.py`).
  Closed loop: LIBERO-10 48/50 for both (fp8, 50 episodes); RoboTwin 2.0
  (official torch policy, 50 tasks × 10 episodes, `demo_clean`) 94.2% vs
  94.6% on the 446 seeds both configurations evaluated, McNemar p = 0.83.
- Parallel-in-time sampling (Picard / ParaDiGMS) trades serial steps for
  wider batches; it pays only if a batched step at M = 64·k costs well
  under k serial steps (`docs/imagewam_rtx5090_batched_m_step.md`: 2.8–4.6×
  at k = 10 on RTX 5090).

## Phase 4: fusion

- Target launch-bound and traffic-bound regions found in Phase 2.
- Fusions used in this repository:
  - norm + quantize, residual + norm + quantize (Pi0.5
    `rms_norm_fp8_static`, `residual_add_rms_norm_fp8_noweight_fp16`,
    `gate_res_adarms_fp8_static`; ImageWAM `gate_res_ada_layer_norm_fp16`,
    `fuse_res_norm`, OPT-017);
  - Q/K/V split + per-head norm + RoPE (Pi0.5 `qkv_split_rope_kvcache_fp16`;
    ImageWAM `qkv_split_norm_rope_fp16`, OPT-032);
  - gated MLP activations (`gate_geglu_merged_fp8_fp16`, `silu_glu_merged_fp16`);
  - merged GEMMs (ImageWAM single-stream `linear1` qkv + mlp_in, `linear2`
    merge, OPT-015/016);
  - folding norm weights into the next GEMM at load (Pi0.5 `(1+w)` fold,
    no-weight norm ≈15% faster, `docs/kernel_fusion.md`);
  - vectorized loads and packed stores (1.4–2.3× per kernel,
    `docs/optimization-details.md`).
- A fusion must be measured on every target device. `qkv_split_norm_rope_fp16`
  with one 256-thread block per (row, head) saved 1.0–1.7 ms on RTX 5090
  and lost 7.0–9.6 ms on Thor; the warp-per-(row, head) rewrite saved
  2.0–2.7 ms on 5090 and 7.3–8.3 ms on Thor (OPT-032). Launch count alone
  does not predict the result; the fused kernel's own efficiency does.
- Pi0.5's TensorRT path showed that fusions which change graph topology
  can lose (einsum→matmul +5.18 ms, a RoPE plugin +10 ms;
  `docs/kernel_fusion.md` §4–§5, `docs/optimization-details.md` §6).

## Phase 5: precision

- Default tier per device is decided by measurement, not by format width:
  ImageWAM fastest is fp8 on RTX 5090 (23.85 ms) and nvfp4 on Thor
  (80.22 ms); on RTX 5090 nvfp4 (28.23 ms) is slower than fp8.
- FP8: static per-tensor activation scales from calibration remove the
  per-call absmax/scale kernels (Pi0.5: ≈630 quantize kernels,
  `docs/optimization-details.md` §1.2). Calibrate on real data with several
  samples and a percentile (Pi0.5: p99.9, N = 8 on RTX, 64 on Thor; real-data
  recalibration took LIBERO Spatial from 91.8% to 98.2%,
  `docs/calibration.md`). Compute alpha in float32, keep weight-spec order
  identical (`docs/calibration.md` §2.3).
- FP4 (NVFP4): only where the kernel reaches its throughput at the served
  shapes; use AWQ where accuracy needs it (`docs/imagewam_nvfp4_awq.md`,
  Pi0.5 encoder FFN AWQ α 0.8). Account for its overhead: on RTX 5090 the
  ImageWAM ActionDiT spends 0.43 ms per step on fp16→bf16 casts, activation
  quantization and bf16→fp16 casts (1.27 ms at M = 640).
- Some tiers are rejected for accuracy, permanently (Pi0.5 static INT8 in
  vision: cosine 0.991 → 0.282).

## Phase 6: GEMM and attention backends

- Choose kernels per shape at the served M. Small M is the usual problem
  for policies (action tokens: ImageWAM M = 64, Pi0.5 decoder M ≈ 10).
- Know the hardware's kernel constraints before tuning. CUTLASS 4.4.2's
  SM120 block-scaled (NVFP4) mainloop requires M and N tiles of at least
  128 (scale-factor TMA), so narrow tiles (128×8 / 128×16) and M = 64 tiles
  do not compile; output-1024 shapes then give 8 tiles on 170 SMs. Per-shape
  cooperative / stream-K / split-K selection cut the ActionDiT NVFP4 GEMM
  per step from 955 to 780 µs (`docs/imagewam_rtx5090_batched_m_step.md`,
  OPT-038). Stream-K / split-K need their workspace zeroed before every
  launch, or the second launch hangs.
- Tiles that win in isolation can lose in the pipeline (Pi0.5 Thor cluster
  tiles +2.2 to +14 ms, `docs/pi05_thor_decoder_fp4_e2e.md`): confirm
  end to end.
- Autotune with care: on SM89 a bad FP8 NT candidate can poison the
  context, so that autotune is disabled (`pipeline_rtx.py`).
- Attention backend per device and site: FA2 on RTX for Pi0.5, FA4 on Thor
  (both sites for ImageWAM), cuBLAS QKᵀ/softmax/PV chain elsewhere; measure
  at real sequence lengths (OPT-019).

## Phase 7: task-level validation

- Any change that alters numerics (precision tier, fewer evaluations,
  calibration) is validated in closed loop, not only by cosine.
- Run configurations on the same episodes and initial states, compare
  paired outcomes, and report confidence intervals; state the number of
  episodes. RoboTwin evaluation seeds pass a stochastic expert check, so
  two configurations share only part of their seeds; compare on the shared
  seeds as well as overall.

## Benchmark protocol

- CUDA-event P50 over ≥100 iterations after warmup; report P10/P90.
  Short warmups read high on RTX 5090 while clocks ramp
  (`docs/exec_contract.md` §8.3); Pi0.5 FP4 on Thor uses 300 warmups.
- A/B in one process, alternating ABBA, reusing the autotuned GEMM
  algorithms (`benchmarks/imagewam_fusion_ab.py`). Thor shows clock
  regimes about 3 ms apart and up to 21 ms cross-process variation
  (`issues.md` ISSUE-095): compare within a session, repeat in independent
  processes.
- One precision per process when comparing tiers; record commit, driver,
  clock state and GPU exclusivity with every number; check that no other
  job uses the GPU.
- Time kernels in isolation inside a CUDA graph (host enqueue cost
  otherwise dominates 10 µs kernels).

## Pitfalls collected from both models

- Comparing against the runtime's own output instead of the reference hides
  scale bugs (a scale-order bug dropped cosine to 0.933; float64 alpha to
  0.9878; `docs/calibration.md` §4). Tactic noise moves cosine by ≤0.002
  in both directions; a scale bug is consistently low (§6.1).
- A fused kernel that inherits an inefficient structure loses on
  bandwidth-limited devices (OPT-032).
- Library wheels pulled into a venv can shadow the base cuBLAS (ImageWAM
  fp8 GEMM failed with status 15; `PROJECT.md`).
- Padding in block-scaled buffers must be zero-initialized (Pi0.5 NVFP4 NaNs,
  `docs/pi05_thor_decoder_fp4_e2e.md`).
- "Fewer kernels" and "lower precision" are not speed by themselves; the
  measured end-to-end time on each device decides.
