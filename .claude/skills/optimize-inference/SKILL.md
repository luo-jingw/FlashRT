---
name: optimize-inference
description: Take a model or a new device from reference implementation to a fast served runtime in eight ordered phases (gates, static execution, measurement, removing work, fusion, precision, kernel selection, task-level validation), each with an exit condition and a recorded artifact.
---

# Trigger

Use when:
- a model is being brought up or sped up on a device;
- a decision is needed on one optimization (fuse or not, change precision
  tier, change a GEMM or attention kernel, fewer denoising evaluations).

Do not use for a single kernel bug fix with a known correct outcome —
use `work`. Measurements on Jetson Thor go through `thor-round`.

# Inputs

Read:
- `AGENTS.md`, `PROJECT.md`;
- `docs/inference_optimization_methodology.md` (the method this skill
  executes; its sections hold the examples and the numbers);
- the model's existing docs, profiles and `opportunities.md` / `issues.md`
  entries;
- the active phase in `plan.md` if the work is part of an approved plan.

# Preconditions

- The official reference implementation runs, or a stored reference
  output exists.
- The target device and precision tiers are named.

# Procedure

Run the phases in order. A phase starts only when the previous phase's
exit condition holds. Phases 2, 4, 5 and 6 are run as `experiment`
grids: enumerate the cells first, run all of them, report one table.

## Phase 0: reference and gates

- Fix a versioned fixture: real inputs, prompts, proprio, initial noise.
- Store the reference outputs (official FP32/BF16), never the runtime's own.
- Define the numeric gate (cosine / MAE thresholds) and which changes must
  be bit-exact against the current default path.
- Exit: the gate runs as a command and the current runtime passes it.

## Phase 1: static execution

- Capture `infer()` as CUDA graph(s); allocate every buffer at load;
  precompute observation-independent tensors at prompt time; handle
  variable lengths with per-length graphs or one masked max-length graph.
- Exit: steady-state `infer()` is input staging plus graph replay; graph
  kernel time equals wall time within a stated margin.

## Phase 2: measurement

- Profile module × operator × precision on the device; measure the
  device's GEMM peak and DRAM bandwidth; classify each cell's bottleneck
  (compute, weight bandwidth, kernel efficiency, launch, small-kernel bytes).
- Check kernel counts are equal across profiled iterations.
- Exit: a profile table with a bottleneck per cell, written to `docs/`.

## Phase 3: remove work

- Look for computation nothing reads, work repeated per step that could be
  shared, and loop counts larger than the result needs (for samplers:
  measure the velocity field's structure and the error of fewer
  evaluations against the full schedule).
- Exit: each candidate measured (time saved, numeric change) and either
  adopted behind a gate or recorded in `opportunities.md`.

## Phase 4: fusion

- Fuse only regions Phase 2 marked launch- or traffic-bound.
- A/B each fusion in one process (ABBA order, shared autotune) on every
  target device; check bit-exactness or the gate.
- Exit: a fusion is default only on devices where it measured faster.

## Phase 5: precision

- Per device, compare tiers end to end; for static FP8 calibrate on real
  data (several samples, percentile); include cast and quantization
  kernels in the cost of low-bit tiers.
- Exit: the default tier per device is the fastest one that passes the gate.

## Phase 6: GEMM and attention backends

- For each GEMM shape at the served M, list the kernel variants the
  device and library support (tile limits first), measure them in
  isolation inside a CUDA graph, then confirm the chosen set end to end.
- Exit: per-shape choice recorded with its end-to-end effect.

## Phase 7: task-level validation

- Any numeric change is checked in closed loop: same episodes and initial
  states for each configuration, paired comparison, confidence interval,
  episode count stated.
- Exit: success rate matches the reference within its interval, or the
  change is rejected.

# Benchmark rules

- CUDA-event P50 over ≥100 iterations after warmup, with P10/P90.
- One precision per process when comparing tiers; record commit, driver,
  clocks and GPU exclusivity; check `nvidia-smi` for other jobs first.
- Repeat device comparisons in independent processes when the device
  shows cross-process variance.

# Stop Conditions

Stop and report when:
- no reference or gate exists for the change;
- timing is not stable enough to separate the A/B difference;
- a change is slower on any target device and no per-device switch exists;
- the change requires a model or interface not covered by the approved plan.

# Completion

- Profiles and verified results in `docs/`;
- candidates not taken in `opportunities.md`;
- unexplained observations in `issues.md`;
- environment facts discovered in `PROJECT.md`.
