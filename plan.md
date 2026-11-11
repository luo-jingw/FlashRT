# Plan

Plan Status: completed

# Problem

## Current

## Problem

## Goal

# Structure

## Modules

## Responsibilities

## State Ownership

# Interface

## Interfaces

## Inputs

## Outputs

## State Changes

# Flow

# Code Mapping

## Modules

## Interfaces

## State

# Implementation

## Phase 1 — Weight declaration with a random-init path

Phase Status: completed

## Phase 2 — Joint attention kernel

Phase Status: completed

## Phase 3 — Encode-once and backbone prefill

Phase Status: completed

## Phase 4 — Denoise loop and CUDA Graph capture

Phase Status: completed

## Phase 5 — Frontend and text-context caching

Phase Status: completed

# Roadmap: Pi0.5-derived optimization tracks

Roadmap Status: approved. All 14 items and the ISSUE-020 text-trim fix
are implemented, verified on H100 and merged into `roadmap/integration`;
the Thor rounds are listed in "Execution status" below and
`THOR_CHECKLIST.md` carries no pending item. An item whose Thor step is
still open keeps its own `# Plan: <item>` section later in this file;
every item's results are recorded in the `opportunities.md` entry named
in "Execution status" and per round in `THOR_STATUS_SUMMARY.md`.

## Problem

Goal: track the 14-item optimization inventory as a roadmap -- this
section is an index + dependency map, not a substitute for a real
per-item plan.

## Structure: three independent tracks

- **Speed track** -- items that directly reduce real `infer()` latency.
- **Accuracy track** -- items that improve quantization/calibration
  quality (some also gate future speed/accuracy work, e.g. real
  calibration data is a prerequisite for a fair AWQ/`fp8_static` trial).
- **Deployment-engineering track** -- architecture/reliability/
  maintainability, not raw speed.

Tracks are independent of each other. Only 3 real dependency edges
exist across all 14 items; the other 11 have zero prerequisites and
can start in any order, in parallel, immediately.

## The 14 items

| id | item | track | cost | depends on | note |
|---|---|---|---|---|---|
| 1 | ActionDiT small-M CUTLASS tile retry (Pi0.5's proven "v10" `128x64x256` tile) | speed | hours | none | opportunities.md OPT-014's own `M=64` CUTLASS-slower-than-cuBLASLt regression |
| 2 | Image normalization LUT (256-entry FP16, precomputed) | speed | hours | none | Pi0.5's own proven technique, "bit-identical" per its docs |
| 3 | Gated-residual + next-layer-norm fusion (one elementwise kernel) | speed | hours -- 1-2 days | none | Pi0.5's own real, working equivalent -- supersedes the previously-deferred CUTLASS-epilogue idea for this same problem (opportunities.md OPT-013's deferred gated-residual epilogue) |
| 4 | `linear2` merge (attn_out_proj + mlp_down) | speed | days | none | opportunities.md OPT-015 op-fusion audit sub-problem 3, implemented as OPT-016 |
| 5 | VAE port to FlashRT kernel style + in-graph capture | speed | weeks, standalone | none | opportunities.md OPT-008; prerequisite for in-graph VAE, not for anything else here |
| 6 | Attention-chain fusion feasibility recheck at ImageWAM's own real shapes | analysis | hours, analysis only | none | Pi0.5 rejected this at `M=10` (5-7x slower); ImageWAM's shapes (`M~905` backbone, `M=64` ActionDiT) differ, worth re-checking, not assuming the same verdict |
| 7 | Real calibration data pipeline (replace `N(0,0.1)` placeholder in `_calibrate_fp8()`) | accuracy | days | none | hub node -- unlocks items 8 and 13 |
| 8 | AWQ per-channel scale folded into NVFP4 weights | accuracy | +2-3 days | 7 | may help ImageWAM MORE than it helped Pi0.5 (ImageWAM's merged-`linear1` GEMMs are more bandwidth-shaped than Pi0.5's own compute-bound QKV case, where Pi0.5 shipped AWQ disabled) |
| 9 | Hadamard-rotated INT4 (E0M3) new precision tier | accuracy | 1-2 weeks, standalone | none | NOT the same dead SM80 kernel OPT-007 already closed -- real native SM100 block-scaled path (same layout family as `nvfp4`); the only item here that could beat `nvfp4` on accuracy |
| 10 | Jetson clock-locking check for benchmark scripts | deployment | hours | none | Pi0.5's own devfreq/nvpmodel check; ImageWAM benchmarks currently have none |
| 11 | Precision-routing contract test (stubbed, no GPU) | deployment | hours -- 1 day | none | mirrors Pi0.5's `test_pi05_thor_fp4_routing.py` pattern against ImageWAM's own `_PRECISIONS`/`_wrap_linear` |
| 12 | ABI integration (`frt_model_runtime_v1`, `io="python"` producer mode) | deployment | 1-3 days | none | zero C++, zero new kernels -- exposes ImageWAM's existing `self._graph`/buffers through the same generic ABI Pi0.5's Python producer already uses |
| 13 | Fidelity + latency CI/regression gate harness | deployment | days | 7 | gate logic (cosine thresholds, `p50<baseline-margin`, JSON result schema) is generic/copyable from Pi0.5's own harness; content needs real calibration data + a real LIBERO fixture format; ready-made baseline: the round that built it recorded 231.6 ms, since re-seeded to 202.2 ms |
| 14 | Native C++ overlay (`io="native"`/`"native_v2"`) | deployment | 2-4 weeks | 12 | ports existing Python orchestration to C++ against ImageWAM's EXISTING kernels, not a kernel rewrite; the item that actually removes the Python/GIL dependency from the hot path |

## Dependency graph

```
7 (real calibration data) --> 8  (AWQ)
7 (real calibration data) --> 13 (CI/regression harness)
12 (ABI integration)      --> 14 (native C++ overlay)
```

Every other item (1, 2, 3, 4, 5, 6, 9, 10, 11, 12) has zero
prerequisites and can start immediately, in parallel.

## Execution status

H100 (sm_90, shared GPU) unless stated otherwise. "State" says whether
the served configuration changed. Each row's full results are in the
`opportunities.md` entry it names; the per-round Thor records are in
`THOR_STATUS_SUMMARY.md`.

| id | state | record |
|---|---|---|
| 1 | Per-shape measured tile choice for the ActionDiT NVFP4/FP8 CUTLASS GEMMs, plus four 1-SM FP8 small-M tiles, compiled for sm_110. Off (`gemm_variant_autotune=True`); correctness and speed unmeasured on Thor. | OPT-018 |
| 2 | Fused uint8→BF16 VAE preprocessing kernel with a 256-entry table; bit-exact to the previous path. On. | OPT-020 |
| 3 | Gated residual fused with the following AdaLN, including across layer boundaries; bit-exact over the whole pass. On. | OPT-017 |
| 4 | Single-stream `attn_out_proj` + `mlp_down` merged into one `linear2` GEMM; the NVFP4 operands are identical to the split path, only the accumulation order changes. On, except `fp16_cutlass`. | OPT-016 |
| 5 | VAE stage capturable in the main graph, plus a native NHWC encoder with fused GroupNorm(+SiLU): the stage takes 4.3 ms on H100 against about 12 ms for the torch encoder. Off (`vae_encoder="native"`, `vae_graph_input`). | OPT-021 |
| 6 | FA4 backbone and `mot` attention with a dedicated output buffer and a fallback to cuBLAS on failure. The backbone site is the served default where the machine can run FA4 (`use_fa4=None` resolves it; `FLASHRT_THOR_FA4=0` forces the chain); the `mot` site resolves the same way since 0921 (`use_fa4_mot=None`; it was opt-in before phase F1). Thor measured it at the served shapes: the `c20f3a0` ladder's `vae_trim_fa4bb` row is 99.0 against `vae_trim`'s 102.9 ms, and the `0920t` end-to-end pair is 126.5 against 131.3 ms with FA4 forced off, with no row falling back. | OPT-019 |
| 7 | Real `fp8_static` calibration from 64 LIBERO frames (142 sites); against fp16, actions cos is 0.99997 with the real calibration against 0.90 with the placeholder, and it resolves ISSUE-001 with a TN FP8 layout on sm_89/sm_90. Opt-in (`calibration_path=`). | OPT-022 |
| 8 | AWQ per-channel scales folded into NVFP4 weights; in simulation the backbone cos goes from 0.99820 to 0.99956. Off (`nvfp4_awq=True`). | OPT-023 |
| 9 | `e0m3_hadamard` precision tier; simulated 1 − actions cos is 2.97e-4 against 6.71e-4 for `nvfp4`. Off (`precision="e0m3_hadamard"`). | OPT-024 |
| 10 | Jetson clock-state probe, printed by the ImageWAM benchmarks. On (reporting only). | OPT-025 |
| 11 | CPU-only precision-routing contract test, eight precision columns. n/a. | OPT-026 |
| 12 | `frt_model_runtime_v1` export (`io="python"`); bit-exact to `infer()`, with parity gates carrying mutation tests. n/a. | OPT-028 |
| 13 | LIBERO fidelity and latency gate: fixture v1, runner, per-device baselines; fp16 passes on H100. n/a. | OPT-027 |
| 14 | Native C++ overlay (`io="native"`); bit-exact, runs a tick without Python, and its latency equals `io="python"`'s (4974 against 4998 graph nodes at fp16 on H100; the Thor `nvfp4` runs measure 5324 against 5348, `0920t`). n/a. | OPT-029 |
| ISSUE-020 | `text_trim`: each prompt runs at its valid text length, which reproduces official's masked attention; against official on libero_goal (fp16) the median/min goes from 0.99680/0.92997 to 0.99998/0.99971, and H100 `infer()` is about 30% faster. Off (`text_trim=True`). | OPT-030, ISSUE-080 |

Rounds that closed items: items 2-5 and 7-14 plus the ISSUE-020 fix
merged into `roadmap/integration` on the H100 numbers above; Thor
`eccf14f` closed item 13's like-for-like gate (202.2 ms) and re-seeded
the latency baseline (231.6 -> 202.2 ms), `a84916a`/`0919e` closed the
target workload and the `fp8_static`/`e0m3_hadamard` gates, `0920s4` and
`0920t` closed the native per-length graphs and measured the served
default, and `0920c` closed the last three checklist rows. Items 1 and 6
keep their Thor steps open (ISSUE-023).

The fp16 served default matches the baseline end to end:
`fr_vs_off` 0.99840/0.99566 and MAE 0.18359 on libero_spatial.

Decisions that depend on Thor results or the owner:

- Default precision: `nvfp4`, `nvfp4_awq`, `e0m3_hadamard`, or
  `fp8_static(_cutlass)` with a calibration file.
- Default-on for `text_trim` (conditions in ISSUE-080), FA4, the native VAE
  encoder, and the tile tuner.
- The initial-noise scale in `infer()` (ISSUE-002) and the served resize
  filter (ISSUE-030).
- Gate fixture v2 with a trimmed fp16 reference, and re-seeding the Thor
  latency baseline (ISSUE-061, ISSUE-080).

## Thor validation checklist

`THOR_CHECKLIST.md` carries no pending item: the last three rows ran in
the `0920c` round, and every measured conclusion is in
`THOR_STATUS_SUMMARY.md`. `scripts/imagewam_thor_validation.sh` runs the
steps; its header lists the build, environment and artifact-bundle
prerequisites, it writes one log per command plus `SUMMARY.txt`, and
`STEPS="..."` selects steps. The bundle (`thor_bundle`: gate fixture v1
plus calibration files, with `SHA256SUMS`) is built on the H100 dev box.
The script's comments state what each step checks; its header names this
list as the place that states what each step decides:

| step | decides |
|---|---|
| 0 | provenance of every number |
| 1 | correctness of every Thor-only kernel path |
| 2 | whether items 2-4 stay default-on; new `nvfp4` latency baseline |
| 3 | `text_trim` default (ISSUE-080) |
| 4 | default precision; FP8 layout on Thor |
| 5 | FA4 default |
| 6 | native VAE and in-graph defaults |
| 7 | tile tuner default, or a fixed tile |
| 8 | Thor readiness of items 12 and 14 |

## Not planned (confirmed, not cost-gated)

- Multi-subgraph stage-splitting (Pi0.5's RTC-prefix-reuse/VJP-guided-
  denoising scheduling machinery, `flash_rt/subgraphs/pi05/`) --
  ImageWAM has no incremental-replanning requirement today; out of
  scope for lack of a real need, not because it's expensive.
- RMSNorm-into-GEMM prologue fusion -- CUTLASS has no prologue-fusion
  mechanism at all (confirmed earlier this session); Pi0.5 doesn't do
  this either. Dead, unchanged by this investigation.

# Plan: Jetson clock-locking check for the benchmark scripts (roadmap item 10)

Plan Status: completed. Phases 1-3 are completed: the record is captured as the
machine is in every Thor round (the round headers of `THOR_STATUS_SUMMARY.md`
carry `MAXN` / `GPC 1.575 GHz` / `emc_locked=null`), and `THOR_CHECKLIST.md`'s
prerequisites run the probe each round.

## Problem

### Current

The ImageWAM benchmark entry points printed latency without recording the
Jetson power and clock state it was measured under, so no ImageWAM latency on
record, including the shipped `nvfp4` P50 of 231.6 ms, carried a clock record.
Pi0.5's `tests/bench_pi05_decoder_fp4_e2e.py` (`machine_state()`) is the model
it follows: it refuses to run unless `nvpmodel -q` reports MAXN and the
`gpu-gpc-0`/`gpu-nvd-0` devfreq nodes have `min_freq == max_freq == cur_freq`,
and it writes that state into its result JSON.

### Goal

A read-only helper that reports the nvpmodel mode and the GPU/EMC devfreq state
without root, warns only for a non-MAXN mode or unobservable state, and never
changes machine state (no `sudo`, `jetson_clocks` or `nvpmodel -m`): Thor is
shared and runs as is, at MAXN with DVFS-managed clocks. Every listed benchmark
prints the record once before timing. Unit tests cover the pinned, dynamic,
missing-nvpmodel, EMC and non-Jetson cases; on Thor the record is captured as
the machine is. Results: `opportunities.md` OPT-025.

## Structure

| module | responsibility | state owned |
|---|---|---|
| `flash_rt/hardware/jetson_clock_state.py` | read the sysfs tree and `nvpmodel -q` under an injectable root and command runner; derive the pinned verdict; print the record | none (reads only) |
| benchmark entry points | call `report_jetson_clock_state()` once before timing | none |
| `tests/test_jetson_clock_state.py` | fake sysfs trees in `tmp_path`, fake command runner | none |

The helper is stdlib-only (no torch, no extension), so it imports on
any machine.

## Interface

```python
# flash_rt/hardware/jetson_clock_state.py
@dataclass(frozen=True)
class DevfreqNode:            # one /sys/class/devfreq/<name> entry
    name: str; kind: str      # "gpu" | "emc"
    cur_hz: int | None; min_hz: int | None; max_hz: int | None
    governor: str | None
    locked: bool              # cur == min == max, all readable

@dataclass(frozen=True)
class ToolQuery:              # one external tool invocation
    available: bool; text: str

@dataclass(frozen=True)
class JetsonClockState:
    is_jetson: bool; platform: str
    nvpmodel: ToolQuery; nvpmodel_mode: str | None; nvpmodel_mode_id: int | None
    gpu: tuple[DevfreqNode, ...]; emc: tuple[DevfreqNode, ...]
    clock_caps_hz: tuple[tuple[str, int], ...]   # /sys/kernel/nvpmodel_clk_cap/*
    gpu_locked: bool; emc_locked: bool | None; power_mode_max: bool | None
    locked: bool; warnings: tuple[str, ...]
    def to_dict(self) -> dict[str, object]

class CommandRunner(Protocol):
    def run(self, argv: Sequence[str]) -> ToolQuery

class JetsonClockProbe:
    def __init__(self, root: Path = Path("/"), runner: CommandRunner | None = None) -> None
    def read(self) -> JetsonClockState

def report_jetson_clock_state(probe: JetsonClockProbe | None = None,
                              emit: Callable[[str], None] = print) -> JetsonClockState
```

Lock verdict: `gpu_locked` = at least one GPU devfreq node and every GPU
node locked. `emc_locked` = `None` when no EMC devfreq node is visible,
else every EMC node locked. `power_mode_max` = `None` when `nvpmodel` is
unavailable, else the mode name starts with `MAXN`. `locked` = Jetson,
`gpu_locked`, and neither `emc_locked` nor `power_mode_max` is `False`.
A non-MAXN power mode or unobservable state adds a warning line;
dynamic clocks at MAXN are recorded without a warning.

## Flow

`JetsonClockProbe.read()` returns `is_jetson=False` with no tool calls when
neither `/etc/nv_tegra_release` nor `/proc/device-tree/{model,compatible}`
exists. On a Jetson it runs `nvpmodel -q` through the runner (timeout, never
`sudo`), reads `cur_freq`/`min_freq`/`max_freq`/`governor` from
`/sys/class/devfreq/*` (names containing `gpu`, or the Tegra GPU ids
`gp10b/gv11b/ga10b/gb10b`, are GPU; names containing `emc` are EMC) plus
`/sys/kernel/nvpmodel_clk_cap/*`, and derives the verdict.
`report_jetson_clock_state()` prints `[jetson-clock-state] <json>` plus one
`[jetson-clock-state] WARNING: ...` line per warning, and returns the record.

## Code Mapping

| item | file |
|---|---|
| probe, record types, runner protocol, reporter | `flash_rt/hardware/jetson_clock_state.py` (new) |
| unit tests | `tests/test_jetson_clock_state.py` (new) |
| wiring | `benchmarks/imagewam_thor_graph_bench.py`, `benchmarks/imagewam_e2e_official_compare.py` (timing section), `benchmarks/imagewam_thor_int4_bench.py`, `benchmarks/imagewam_thor_int8_bench.py`; the item-13 gate runner embeds the record in its result JSON |

## Implementation Phases

### Phase 1 — helper and unit tests

Phase Status: completed

`jetson_clock_state.py` with the interface above, plus `tests/test_jetson_clock_state.py`; verified by the x86 pytest run on fake sysfs trees (11 passed, OPT-025) and the `is_jetson=false` record printed on the H100 box.

### Phase 2 — wire into the benchmark entry points

Phase Status: completed

Every benchmark in Code Mapping prints the record once before timing (`python -m py_compile` on each); verified by the `[jetson-clock-state]` line the graph bench's fp16 row prints on H100 (OPT-025).

### Phase 3 — Thor handoff

Phase Status: completed

The record captured as the machine is, through the owner's run of `python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state; report_jetson_clock_state()"`; no file changes. Run in every Thor round since `c20f3a0`, with the state recorded beside that round's numbers.

# Plan: Fidelity and latency regression gate harness (roadmap item 13)

Plan Status: completed. Phases 1-5 are completed: the Thor gates ran in the
`eccf14f` round (`nvfp4` 202.2 ms, pass) and in the `0920t` round (three rows,
the served default among them), which is what Phase 5's handoff was for.

## Problem

### Current

No ImageWAM gate was committed: fidelity was checked ad hoc with
`benchmarks/imagewam_e2e_official_compare.py`, which loads the official bf16
model (Qwen3-4B included) next to FlashRT in the same process (about 34GB) and
needs `av`/`pandas` for LIBERO decoding, and latency claims (`nvfp4` 231.6 ms,
`opportunities.md` OPT-015) were prose, with no machine-readable baseline and no
pass/fail rule. Pi0.5's harness (`tests/bench_pi05_decoder_fp4_e2e.py`) supplies
the generic pieces: per-sample cosine thresholds, `p50` against a regression
baseline, and a versioned result JSON with the clock state.

### Goal

A versioned LIBERO fixture: real preprocessed observations (two 224x224 views,
proprio, prompt), the official Qwen3 context and mask, fixed initial action
noise, official reference actions, and FlashRT `fp16` reference actions; the
data lives under `/home/user1/workspace/jingwu/artifacts/deploy-gates/`, git
holds the generator and a manifest with checksums, and gating on Thor needs
neither the official model nor Qwen3. A gate runner that, for one precision,
checks fidelity against the official reference and the FlashRT `fp16`
reference, and latency against a per-device baseline JSON (Thor `nvfp4` seeded
at 231.6 ms; H100 latency ungated). An `fp8_static` slot that activates when a
calibration file exists, with the hand-off interface below and no dependency on
the calibration stream's code. Measurable: on H100, `fp16` against the fixture
reproduces the end-to-end baseline (`fr_vs_off` median 0.99840, min 0.99567;
mean `mae_fr_vs_gt` 0.18359) and passes. Roadmap item 13 lists item 7 (real
calibration data) as a dependency; only the `fp8_static` gate needs it.

## Structure

| module | responsibility | state owned |
|---|---|---|
| `flash_rt/core/regression_gate.py` | model-agnostic gate policy: thresholds, latency baseline, latency summary, per-check results, report schema | none |
| `flash_rt/datasets/imagewam_gate_fixture.py` | fixture arrays, `.npz` IO, manifest with per-file and per-array SHA-256, verification | fixture format version |
| `benchmarks/imagewam_gate_fixture_generate.py` | produce a fixture on H100: reuses `imagewam_e2e_official_compare.py` for LIBERO loading, preprocessing and the official model; then the FlashRT `fp16` reference through the served `infer()` | fixture data on disk |
| `tests/gate_imagewam_libero.py` | gate runner CLI: load and verify fixture, run one precision through served `infer()`, evaluate, write result JSON | result JSON |
| `tests/fixtures/imagewam_gate/fidelity_thresholds.json` | per-precision fidelity thresholds, `requires_calibration` flag | thresholds |
| `tests/fixtures/imagewam_gate/latency_baselines.json` | per-device latency baselines and gating switch | baselines |
| `tests/fixtures/imagewam_gate/<name>.manifest.json` | committed manifest of the generated fixture | fixture identity |
| `flash_rt/frontends/torch/imagewam_thor.py` | `infer(observation, *, action_noise=None)`: optional explicit initial noise; default behavior unchanged | action latent buffer (unchanged owner) |
| `tests/test_imagewam_regression_gate.py` | CPU tests for gate policy, fixture round trip and tamper detection, committed config files | none |

The runner goes through the served `infer()` rather than re-implementing
its body, so later changes to `infer()` (for example an in-graph VAE)
are gated as served.

## Interface

```python
# flash_rt/core/regression_gate.py
RESULT_SCHEMA_VERSION = 1
@dataclass(frozen=True) class CosineSummary:   median: float; minimum: float; count: int
@dataclass(frozen=True) class FidelityThresholds:
    vs_official_median_min: float; vs_official_min_min: float
    vs_fp16_reference_median_min: float; vs_fp16_reference_min_min: float
    mae_vs_gt_ratio_max: float; requires_calibration: bool
@dataclass(frozen=True) class LatencySummary:   p10/p50/p90/min/max_ms, iters, group_medians_ms
    @classmethod from_samples(samples_ms) -> LatencySummary
@dataclass(frozen=True) class LatencyBaseline:  p50_ms: float | None (None = unseeded); margin: float; source: str; config: Mapping
@dataclass(frozen=True) class DeviceLatencyPolicy: device: str; gated: bool; reason: str; baselines: dict[str, dict[str, LatencyBaseline]]  # precision -> configuration name -> entry
@dataclass(frozen=True) class GateCheck:        name: str; status: "pass"|"fail"|"ungated"|"skipped"; value; limit; detail
class FidelityGate:  evaluate(vs_official, vs_fp16_reference, mae_mean, reference_mae_mean, all_finite) -> list[GateCheck]
class LatencyGate:   evaluate(precision, summary, configuration, resolved_config=None) -> GateCheck   # p50 < baseline*(1+margin) of that configuration's entry; seed_entry(...) -> paste-ready record
@dataclass class GateReport: checks + context; verdict ("pass"|"fail"|"skipped"|"blocked"); to_dict()

# flash_rt/datasets/imagewam_gate_fixture.py
FIXTURE_FORMAT_VERSION = 1
@dataclass class ImageWAMGateFixture:  view1 (N,224,224,3) u8, view2, state (N,8) f32, task_index (N,),
    episode, frame, gt_actions (N,H,7) f32 (NaN-padded), gt_len (N,), prompts (T,),
    context_bf16_bits (T,L,D) u16, context_mask (T,L) bool, seeds (S,),
    noise (N,S,H,7) f32, official_actions (N,S,H,7) f32 (normalized),
    fp16_reference_actions (N,S,H,7) f32 (normalized),
    text_trim bool  (whether both references were recorded trimmed; a gate run refuses a mismatch)
class GateFixtureStore:  save(fixture, directory, metadata) -> FixtureManifest; load(directory, manifest) -> ImageWAMGateFixture
@dataclass class FixtureManifest: name, format_version, files{name: sha256,bytes}, arrays{name: shape,dtype,sha256}, metadata

# fp8_static hand-off (tests/gate_imagewam_libero.py)
FP8_CALIBRATION_ENV = "IMAGEWAM_FP8_CALIBRATION"      # or --fp8-calibration PATH
FP8_CALIBRATION_FRONTEND_KWARG = "calibration_path"   # the calibration stream's keyword
```

`fp8_static` contract: a precision whose thresholds say
`requires_calibration` is gated only when a calibration file path is
given and exists; otherwise the report verdict is `skipped` with the
reason. When the file exists, the runner passes its path to
`ImageWAMTorchFrontendThor(..., calibration_path=<path>)` only if the
constructor declares that keyword explicitly; if not, the verdict is
`blocked`, naming the missing keyword. The runner never runs
`fp8_static` on the placeholder `N(0, 0.1)` calibration. The file's
SHA-256 goes into the report.

Exit codes: 0 for `pass` and `skipped`, 1 for `fail` and `blocked`.

## Flow

Generator (H100, once per fixture version): `load_samples()` from the end-to-end
script (env `SUITE`, `N_TASKS`, `FRAMES`, `SEEDS`), both views
`center_crop_resize`d to 224x224, the official model's `_prepare_flux2_infer_text`
per task, the official sampler's per-seed noise draw (CPU generator, bf16 round
trip) and `infer_action_flux2(..., seed)` for the official normalized actions;
then FlashRT `fp16` (real checkpoint, AE, dataset stats, no Qwen3) with
`set_prompt(context)` and `infer(obs, action_noise=noise)`;
`GateFixtureStore.save` writes `fixture.npz` and the manifest.

Runner (any CUDA device): verify the fixture against the committed manifest,
resolve the precision's thresholds and the `fp8_static` contract, read the clock
state (item 10) and the device policy; per sample and seed run served
`infer(obs, action_noise=noise)` and measure cosine in normalized action space
against the official and `fp16` references and MAE against ground truth in real
units; latency is served `infer(obs)` with default noise, warmup then timed
iterations (`time.perf_counter` around each call, `infer()` synchronizes);
evaluate, write `result.json`, print one `__IMAGEWAM_GATE__ <json>` line, exit.

## Code Mapping

| item | file |
|---|---|
| gate policy | `flash_rt/core/regression_gate.py` (new) |
| fixture format | `flash_rt/datasets/imagewam_gate_fixture.py` (new) |
| generator | `benchmarks/imagewam_gate_fixture_generate.py` (new) |
| runner | `tests/gate_imagewam_libero.py` (new) |
| configs and manifest | `tests/fixtures/imagewam_gate/*.json` (new) |
| explicit noise hook | `flash_rt/frontends/torch/imagewam_thor.py` (`infer`) |
| unit tests | `tests/test_imagewam_regression_gate.py` (new) |
| fixture data | `/home/user1/workspace/jingwu/artifacts/deploy-gates/imagewam_libero_gate_v1/` (not in git) |

## Implementation Phases

### Phase 1 — gate policy and fixture format, CPU tests

Phase Status: completed

`regression_gate.py`, `imagewam_gate_fixture.py`, the two config JSON files and the unit tests in Code Mapping; verified by the CPU pytest run, including the tamper test (flip one byte of a fixture array) failing verification.

### Phase 2 — explicit initial-noise hook in `infer()`

Phase Status: completed

`infer(observation, *, action_noise=None)` in `flash_rt/frontends/torch/imagewam_thor.py`, default path unchanged; verified on H100 fp16 with the real checkpoint, where `infer(obs, action_noise=n)` matches the end-to-end script's `flashrt_infer_with_noise` bit-exactly after denormalization.

### Phase 3 — fixture generator, fixture v1 on H100

Phase Status: completed

`imagewam_libero_gate_v1` (libero_spatial, 10 tasks, frames 0 and 60, seeds 0 and 1) generated with `tests/fixtures/imagewam_gate/imagewam_libero_gate_v1.manifest.json` committed; verified by the generator's per-sample official-vs-fp16 cosine summary reproducing the end-to-end baseline (OPT-027).

### Phase 4 — gate runner, real fp16 gate on H100

Phase Status: completed

`tests/gate_imagewam_libero.py`; the real `fp16` run passes fidelity with latency ungated, `fp8_static` without a calibration file reports `skipped`, and `nvfp4` on H100 fails at construction with the existing clear NVFP4 build error (OPT-027).

### Phase 5 — Thor handoff

Phase Status: completed

The fixture copied and its checksums verified, then `nvfp4` and `fp16` run, with the result JSON reported; no file changes. Both precisions ran on Thor: `nvfp4` against fixture v1 in the `eccf14f` round (202.2 ms, pass) and three rows against fixture v2 in the `0920t` round (the served default's `nvfp4` row among them).

# Plan: ActionDiT small-M CUTLASS tile selection (roadmap item 1)

Plan Status: completed. Phases 1-7 are completed; Phase 7 ran on Thor in the
`1021` round (opportunities.md OPT-018). `gemm_variant_autotune` is the served
`nvfp4` default since that round (owner decision, "Decisions pending").

## Problem

### Current

Every ActionDiT weight GEMM runs at `M = num_action = 64`, and the tile for the
two CUTLASS-backed quantized precisions comes from an `(N, K)`-only heuristic
tuned at other shapes: `Nvfp4Linear` calls
`flash_rt.executors.fp4_utils.fp4_gemm` without a variant, so the
`pick_variant(N, K)` table applies (calibrated for Pi0.5's encoder at
`M = 968`), and
`StaticFp8Linear(use_cutlass=True)` uses `_pick_fp8_cutlass_variant(N, K)`, which
picks `wide` when `N >= 4K` and `sq` otherwise. Inventory at the real ActionDiT
shapes (`M = 64`, `action_hidden_dim = 1024`, `action_attn_width = 3072`,
`action_mlp_hidden = 4096`, `action_dim = 7`, 5 double and 20 single layers):

| site | N | K | calls per step | `nvfp4` | `fp8_static_cutlass` | `fp16_cutlass` |
|---|---:|---:|---:|---|---|---|
| double `qkv` | 9216 | 1024 | 5 | v6 `128x256x128` c1x1x1 | `wide` `256x128x128` c2x2x1 | `wide` |
| double `proj` | 1024 | 3072 | 5 | v6 | `sq` `256x256x128` c2x2x1 | `sq` |
| double `mlp0` (merged gate/up) | 8192 | 1024 | 5 | v6 | `wide` | SwiGLU pair `k64_silu` + `k64_mul_aux` `256x256x64` c2x2x1 at N=4096 |
| double `mlp2` | 1024 | 4096 | 5 | v6 | `sq` | `sq` |
| single `linear1` (qkv + gate/up) | 17408 | 1024 | 20 | v8 `128x256x256` c1x1x1 | `wide` | not merged: `qkv` `wide` + SwiGLU pair at N=4096 |
| single `attn_out_proj` | 1024 | 3072 | 20 | v6 | `sq` | `sq` |
| single `mlp_down` | 1024 | 4096 | 20 | v6 | `sq` | `sq` |
| `action_encoder` | 1024 | 7 | 1 | cuBLASLt `Fp16Linear` (alignment fallback) | same | same |
| `head.linear` | 7 | 1024 | 1 | cuBLASLt `Fp16Linear` (alignment fallback) | same | same |

At `M = 64` one output tile row covers the whole M extent, so the CTA count
equals the number of N tiles, and these GEMMs are weight-bandwidth bound
(arithmetic intensity 2M = 128 FLOP per weight element): on Thor's 20 SMs the
`N = 1024` GEMMs (50 calls per denoise step, 500 per `infer()`) launch 4 CTAs
under v6 and 4 useful CTA pairs under FP8 `sq`. FP8 CUTLASS has no tile narrower
than 128 in N and no 1-SM (cluster 1x1x1) tile at all, and measured 1.44-1.68x
slower than cuBLASLt at this M (opportunities.md OPT-014, result 3). Pi0.5's
decoder (`M = 10`) runs the narrow-N v10 tile (`128x64x256`, cluster 1x1x1) for
all four projections (`docs/pi05_thor_decoder_fp4_e2e.md`, "Decoder v10 Tiles"),
and v10 is already instantiated in `cutlass_fp4_gemm_variants.cu` but ImageWAM
never selects it.

### Problem

No ActionDiT GEMM tile choice is measured at `M = 64`; the existing choices are
extrapolated from other shapes, and FP8 CUTLASS has no small-M tile to choose.

### Measurable goal

A per-shape tile choice for the ActionDiT GEMMs, measured once at construction
on the device the frontend runs on and cached per `(family, M, N, K)`, with the
current heuristic pick among the candidates and a candidate required to
reproduce that pick's output. Plus FP8 small-M 1-SM tiles with the Pi0.5 v10
tile `128x64x256` as template, selection logic unit-tested with the kernels
stubbed, and a Thor script that sweeps every candidate at every real ActionDiT
shape (cosine against fp16, per-shape winner) with an `infer()` A/B of old vs
new selection on `nvfp4` and `fp8_static_cutlass`. The shipped default selection
stays unchanged (opt-in flag) until Thor confirms correctness and speed.

## Structure

- NEW `flash_rt/models/imagewam/gemm_variant_tuner.py`: the selection policy
  (candidate filtering, correctness gate, timing comparison, hysteresis against
  the incumbent) and the per-frontend result cache; defines
  `VariantTunableGemm`, `VariantTimer` and the result dataclasses, no CUDA code.
- NEW `flash_rt/models/imagewam/gemm_variant_timer.py`: the device timing
  mechanism (`CudaGraphVariantTimer`: CUDA-graph capture of a launch batch,
  replay timed with CUDA events).
- `flash_rt/models/imagewam/quant_linear.py`: `Nvfp4Linear` and
  `StaticFp8Linear(use_cutlass=True)` implement `VariantTunableGemm`, each
  linear owning its own current variant, whose default equals today's heuristic
  pick, so `__call__` is unchanged until a variant is set.
- `csrc/gemm/gemm_types_sm100.h`, `csrc/gemm/cutlass_sm100.cu`,
  `csrc/bindings.cpp`: four new FP8 1-SM tiles (cluster 1x1x1): `t128x64x256`
  (v10 template), `t128x64x128`, `t128x128x128`, `t128x256x128`.
- `flash_rt/frontends/torch/imagewam_thor.py`: the decision to tune
  (`gemm_variant_autotune: bool = False`), the grouping of ActionDiT linears by
  shape, and the tuner instance and its results (`gemm_variant_results`).
- NEW `benchmarks/imagewam_thor_small_m_tile_sweep.py`: Thor sweep and `infer()`
  A/B; NEW `tests/test_imagewam_gemm_variant_tuner.py`: stubbed selection tests
  (CPU) and a real-timer test (any CUDA GPU).

State ownership:

| state | owner |
|---|---|
| current tile variant of one linear | that `Nvfp4Linear` / `StaticFp8Linear` instance |
| tuning results cache `(family, M, N, K) -> result` | the `GemmVariantTuner` instance owned by the frontend |
| whether tuning runs | frontend constructor argument |

## Interface

```python
# flash_rt/models/imagewam/gemm_variant_tuner.py
@dataclass(frozen=True) class GemmShape:  m: int; n: int; k: int

@dataclass(frozen=True)
class VariantMeasurement:
    variant: str
    us_per_gemm: float | None      # None: not timed (rejected)
    cosine_vs_default: float | None
    status: str                     # "ok" | "launch_failed rc=.." | "mismatch" | "nonfinite"

@dataclass(frozen=True)
class VariantTuneResult:
    family: str; shape: GemmShape; members: int
    default_variant: str; chosen_variant: str
    measurements: tuple[VariantMeasurement, ...]

class VariantTunableGemm(Protocol):
    family: str; n: int; k: int
    default_variant: str; variant: str
    def candidate_variants(self) -> tuple[str, ...]: ...
    def set_variant(self, variant: str) -> None: ...
    def prepare_tuning_input(self, x_ptr: int, m: int, stream: int) -> None: ...
    def launch_variant(self, variant: str, out_ptr: int, m: int, stream: int) -> int: ...

class VariantTimer(Protocol):
    def us_per_launch(self, batches: Sequence[Callable[[int], None]],
                      launches_per_batch: int) -> tuple[float | None, ...]: ...   # None: batch raised

class GemmVariantTuner:
    def __init__(self, timer: VariantTimer, *, device: str = "cuda",
                 min_gain: float = 0.02, cosine_floor: float = 0.9999): ...
    def tune(self, members: Sequence[VariantTunableGemm], m: int) -> VariantTuneResult: ...
    def results(self) -> tuple[VariantTuneResult, ...]: ...

# flash_rt/models/imagewam/gemm_variant_timer.py
class CudaGraphVariantTimer:
    def __init__(self, *, reps: int = 4, samples: int = 15, warmup: int = 3): ...
    def us_per_launch(self, batches: Sequence[Callable[[int], None]],
                      launches_per_batch: int) -> tuple[float | None, ...]: ...

# flash_rt/frontends/torch/imagewam_thor.py
class ImageWAMTorchFrontendThor:
    def __init__(..., gemm_variant_autotune: bool = False, ...): ...
    gemm_variant_results: tuple[VariantTuneResult, ...]
```

Selection rule, per group of linears sharing `(family, M, N, K)`: stage the
same random input into every member; launch every candidate once per member
eagerly, recording `launch_failed` for a nonzero return code or a raised Python
exception and `nonfinite`/`mismatch` when its output against the default
variant's on the same member is non-finite or below `cosine_floor` (the default
variant failing or raising is an error); time each survivor as one launch per
member, round robin, so each launch reads a different layer's weight instead of
a warm L2, with the batch captured in one CUDA graph so launch overhead is
excluded (`timing_failed` when the timer cannot capture it); keep the default
unless the winner is faster by more than `min_gain` (2%) or the default itself
could not be timed; apply the choice to every member and cache it.

## Flow

```
ImageWAMTorchFrontendThor.__init__(gemm_variant_autotune=True, precision in {nvfp4, fp8_static_cutlass})
  -> _load_real_weights / _alloc_random_weights      (linears built on default variants)
  -> _tune_action_dit_gemm_variants(d)
       group self._weights[("action_dit", ...)] implementing VariantTunableGemm by (family, n, k)
       for each group: self._gemm_tuner.tune(members, m=d["num_action"])
         -> member.prepare_tuning_input / launch_variant       (eager checks)
         -> CudaGraphVariantTimer.us_per_launch                (graph-timed batches)
         -> member.set_variant(chosen)
       self.gemm_variant_results = self._gemm_tuner.results()
  -> set_prompt(): _calibrate_fp8 (unchanged), _capture_graph (captures chosen variants)
```

Tuning never calls `__call__`, so `StaticFp8Linear`'s
calibrate-before-call contract is unaffected.

## Code Mapping

| item | file |
|---|---|
| `GemmShape`, `VariantMeasurement`, `VariantTuneResult`, `VariantTunableGemm`, `VariantTimer`, `GemmVariantTuner` | `flash_rt/models/imagewam/gemm_variant_tuner.py` |
| `CudaGraphVariantTimer` | `flash_rt/models/imagewam/gemm_variant_timer.py` |
| protocol implementation | `flash_rt/models/imagewam/quant_linear.py` |
| FP8 1-SM tiles | `csrc/gemm/gemm_types_sm100.h`, `csrc/gemm/cutlass_sm100.cu`, `csrc/bindings.cpp` |
| flag, grouping, tuner ownership | `flash_rt/frontends/torch/imagewam_thor.py` |
| Thor sweep + A/B | `benchmarks/imagewam_thor_small_m_tile_sweep.py` |
| tests | `tests/test_imagewam_gemm_variant_tuner.py` |

## Implementation Phases

### Phase 1: tuner and timer

Phase Status: completed

Selection policy and device timer, independent of any kernel (`gemm_variant_tuner.py`, `gemm_variant_timer.py`, `tests/test_imagewam_gemm_variant_tuner.py`); verified by the stubbed tests (argmin choice, hysteresis, launch failure, mismatch rejection, nonfinite rejection, default failing, every candidate failing, the cache, group application) and an H100 real-timer test over two real sm_90 kernels, checked against a direct CUDA-event measurement.

### Phase 2: FP8 1-SM small-M tiles

Phase Status: completed

`cutlass_fp8_t128x64x256`, `_t128x64x128`, `_t128x128x128`, `_t128x256x128` exported under `ENABLE_SM100_CUTLASS` in `csrc/gemm/gemm_types_sm100.h`, `csrc/gemm/cutlass_sm100.cu`, `csrc/bindings.cpp`; verified now by `sm110_check.sh` passing with the sm_90 build unaffected, correctness and speed being Thor checklist items.

### Phase 3: quant_linear protocol implementation

Phase Status: completed

`Nvfp4Linear` and `StaticFp8Linear(use_cutlass=True)` expose `family`, `default_variant`, `variant`, `candidate_variants()`, `set_variant()`, `prepare_tuning_input()` and `launch_variant()` with default behavior unchanged (`flash_rt/models/imagewam/quant_linear.py`); verified by the unchanged regression suite and the same `RuntimeError` on H100 construction.

### Phase 4: frontend wiring

Phase Status: completed

The `gemm_variant_autotune` flag, grouping ActionDiT linears by shape and `gemm_variant_results` (`flash_rt/frontends/torch/imagewam_thor.py`, tests); verified by a routing test with stubbed linear classes (only ActionDiT groups tuned, `m = num_action`, one tune per distinct shape, the chosen variant applied to every member, nothing changed with the flag off).

### Phase 5: Thor sweep and A/B script, handoff

Phase Status: completed

`benchmarks/imagewam_thor_small_m_tile_sweep.py`, with results and the Thor checklist in `opportunities.md` OPT-018; verified by its non-Thor paths on H100 (argument parsing, shape table, cuBLASLt fp16 reference timing) printing SKIP for families this build lacks.

### Phase 6: candidates that raise are rejected

Phase Status: completed

A Python exception from a candidate's launch, such as an `AttributeError` for a `cutlass_fp8_t128x*` symbol missing from a stale build, rejects that candidate instead of aborting construction, and a batch the timer cannot capture is rejected as `timing_failed`.
Verified by stub tests (a raising candidate, a raising default still an error, an untimeable candidate, an untimeable default kept), the timer test's raising and capture-invalidating batches, and a routing test with the `t128x*` symbols removed.

### Phase 7: Thor confirmation

Phase Status: completed

The Thor checklist in `opportunities.md` OPT-018: the tile sweep must show correct outputs for every tile, and the `infer()` A/B of heuristic vs tuned tiles must show action cosine >= 0.9999 and a P50 delta.
Result (`1021` Thor round, THOR_STATUS_SUMMARY.md): every tile matches the heuristic tile's cosine; the tuned graph is bit-identical in actions and 1.98 ms faster for `nvfp4` (beyond the P10–P90 width), 0.46 ms for `fp8_static_cutlass` (within it). Tuned picks at N = 1024: v7, v7, v10. The flag is the served `nvfp4` default since this round (owner decision, "Decisions pending").

# Plan: attention-chain fusion recheck at ImageWAM's real shapes (roadmap item 6)

Plan Status: approved. Phases 1-6 are completed; Phase 7 (Thor confirmation,
which the FA4 switch stays opt-in until) is blocked on an sm_110 device and an
FA4 runtime (`issues.md` ISSUE-023).

## Problem

### Current

Both attention sites run a cuBLAS-composed chain in `ImageWAMAttnBackend.run()`
(`flash_rt/hardware/thor/attn_backend.py`): a strided-batched QK^T GEMM into a
`logits` buffer, a softmax kernel, then a strided-batched PV GEMM
(`fvk.attention_qkv_fp16_perhead`, `csrc/kernels/attention_cublas.cuh`). The
frontend always constructs the backend with `use_perhead_kv=True` and
`use_real_mot_mask=True`.

- `"backbone"` site: prefill self-attention, 25 layers, `q = kv = a0 = 905`
  tokens, 24 heads, HD 128, no mask. FA4 is wired in as `use_fa4=True`
  (opportunities.md OPT-005) and measured on Thor at cosine 1.000000 against the
  cuBLAS chain, 3.75x per call at the real per-head shape and -10.5% prefill in
  the per-layer benchmark. The frontend default is `use_fa4=False`, because the
  previous dev box had no FA4 runtime and `use_fa4=True` raises when the runtime
  is missing.
- `"mot"` site: ActionDiT joint attention, 25 layers x 10 steps = 250 calls per
  `infer()`, `q = 64` action queries over `kv = total = 969` keys. With
  `use_real_mot_mask=True`, the rule the frontend always uses, the call is
  unmasked attention through the same `attention_qkv_fp16_perhead`, while
  upstream `_build_mot_attention_mask_flux2` with `target_len = 0` still
  excludes padded text keys for every query row, at the prefill call and at the
  action call (issues.md ISSUE-020; the mask analysis, including that
  `pipeline_thor.py` models it at neither site, is in `opportunities.md` OPT-019
  Finding 1). FA4 has never been evaluated here.

Pi0.5 rejected a fused SIMT attention chain at decoder `M = 10`, HD 256, as
5-7x slower (`docs/pi05_thor_decoder_fp4_e2e.md`) because its QK^T/PV GEMMs are
about 1 us of tensor-core work and FA4 has no KV-split path at HD 256; why that
verdict does not transfer here is in `opportunities.md` OPT-019.

### Problem

Nobody has measured which share of ImageWAM prefill and denoise time attention
takes at the real shapes. The Thor FA4 win is not on by default. The `mot`
site's fused-kernel eligibility has never been evaluated.

### Measurable goal

- H100, indicative only: the attention share of prefill and of one denoise step
  at the real shapes, measured in-graph as graph time with the real attention
  minus graph time with attention removed, and per-call cuBLAS chain vs fused
  kernels available on sm_90 (PyTorch SDPA flash / cuDNN / mem-efficient) at
  both sites' shapes, with cosine against the cuBLAS chain.
- A recommendation with evidence per site.
- A Thor FA4 switch for `"backbone"` that resolves to the cuBLAS chain when the
  FA4 runtime is missing or the device is not Thor, staying opt-in
  (`FLASHRT_THOR_FA4=1`) until Thor confirms FA4 at the served shapes, plus an
  opt-in FA4 path for `"mot"`; dispatch verified locally against the cuBLAS
  chain with a reference-backed FA4 stand-in, and FA4 itself on the Thor
  checklist.

## Structure

- NEW `benchmarks/imagewam_attention_share_bench.py`: the measurements (in-graph
  attention share; per-call chain vs fused kernels; on Thor it also times FA4
  per call, with `num_splits` swept for the `mot` shape).
- `flash_rt/hardware/thor/attn_backend.py`: `ImageWAMAttnBackend` owns per-site
  kernel dispatch and gains `use_fa4_mot: bool` for the `"mot"` site FA4 branch.
  That branch is valid only with `use_real_mot_mask=True` and
  `use_perhead_kv=True`, FlashRT's unmasked per-head rule, and the constructor
  rejects any other combination.
- `flash_rt/hardware/thor/fa4_backend.py`: owns FA4 availability and gains
  `thor_default_enabled() -> bool`, true only on an sm_11x device with an active
  FA4 runtime.
- `flash_rt/frontends/torch/imagewam_thor.py`: owns the FA4 output buffer, the
  default, and the FA4-failure fallback in `set_prompt()`
  (`_capture_graph_or_fall_back`): on an exception during warmup or capture with
  FA4 on, it logs, warns, records `fa4_fallback_reason`, rebuilds the backend
  with FA4 off, and captures again. `use_fa4: bool | None = None` resolves,
  through `_resolve_use_fa4`, to False unless `FLASHRT_THOR_FA4=1`, in which case
  it resolves to `fa4_backend.thor_default_enabled()`; an explicit `True` still
  requires the runtime and an explicit `False` forces the cuBLAS chain. It also
  gains `use_fa4_mot: bool = False`, passed through.
- Tests: `tests/test_imagewam_fa4_dispatch.py` (new) checks the backend's FA4
  branches for both sites against the cuBLAS chain, with FA4 replaced by a
  stand-in that has FA4's `_flash_attn_fwd` signature and computes attention as
  an fp32 matmul-softmax-matmul in PyTorch, and checks the default resolution;
  `tests/test_imagewam_fa4_backbone.py` gains a real-FA4 `mot` case that skips
  without FA4.

State ownership:

| state | owner |
|---|---|
| FA4 on/off per site | `ImageWAMAttnBackend` instance (`_use_fa4`, `_use_fa4_mot`) |
| default resolution | frontend `_resolve_use_fa4` (`FLASHRT_THOR_FA4`, then `fa4_backend.thor_default_enabled()`) |
| FA4 runtime availability | `fa4_backend` module |
| FA4 output buffer `_fa4_out` | frontend (passed to the backend as `fa4_out` slots) |
| `fa4_fallback_reason` | frontend |

## Interface

```python
# flash_rt/hardware/thor/fa4_backend.py
def thor_default_enabled() -> bool: ...   # sm_11x device AND FA4 runtime active

# flash_rt/hardware/thor/attn_backend.py
class ImageWAMAttnBackend:
    def __init__(self, spec, ctx, *, backbone_slots: dict, mot_slots: dict,
                 use_fa4: bool = False, use_perhead_kv: bool = False,
                 use_real_mot_mask: bool = False, use_fa4_mot: bool = False): ...

# flash_rt/frontends/torch/imagewam_thor.py
class ImageWAMTorchFrontendThor:
    def __init__(..., use_fa4: bool | None = None, use_fa4_mot: bool = False, ...): ...
    use_fa4: bool        # resolved value, read-only after construction
    use_fa4_mot: bool
```

FA4 `"mot"` call: Q `(1, q_seq, NH, HD)` at row offset `a0` of `Q_O`,
K/V `(1, kv_seq, NH, HD)` per layer, `causal=False`, `pack_gqa=False`,
`num_splits=1`. Output goes to the dedicated FA4 output buffer, then
is copied back to the Q rows. This is the same pattern as the
`"backbone"` per-head branch.

FA4 output buffer: a site that runs FA4 gets `"fa4_out"` (fp16 pointer)
and `"fa4_out_numel"` in its slots. The capacity must be at least that
site's `max_q_seq * NH * HD`, checked at construction and on every call.
The frontend owns one `(total, hidden)` buffer for both sites, allocated
only when some site runs FA4. `logits` is sized for the cuBLAS chain's
score matrix (`total*NH x total`), which at small dims is smaller than
`q_seq*NH*HD`, so FA4 output never goes there.

## Flow

```
frontend __init__(use_fa4=None)
  -> use_fa4 = FLASHRT_THOR_FA4 == "1" and fa4_backend.thor_default_enabled()   # opt-in
  -> ImageWAMAttnBackend(..., use_fa4=use_fa4, use_fa4_mot=use_fa4_mot)
prefill:  attn.run("backbone", ...) -> FA4 if use_fa4 else attention_qkv_fp16_perhead
denoise:  attn.run("mot", ...)      -> FA4 if use_fa4_mot else attention_qkv_fp16_perhead
```

## Code Mapping

| item | file |
|---|---|
| measurements | `benchmarks/imagewam_attention_share_bench.py` |
| `thor_default_enabled` | `flash_rt/hardware/thor/fa4_backend.py` |
| `use_fa4_mot` dispatch | `flash_rt/hardware/thor/attn_backend.py` |
| default resolution, `use_fa4_mot` pass-through | `flash_rt/frontends/torch/imagewam_thor.py` |
| dispatch tests | `tests/test_imagewam_fa4_dispatch.py`, `tests/test_imagewam_fa4_backbone.py` |
| results, recommendation | `opportunities.md` OPT-019 |

## Implementation Phases

### Phase 1: measurement

Phase Status: completed

Attention share (H100) and per-call chain vs fused kernels at the real shapes (`benchmarks/imagewam_attention_share_bench.py`); verified by the printed P10/P50/P90 for graphs with and without attention, per stage, and the per-call medians per kernel with cosine against the cuBLAS chain (opportunities.md OPT-019 Findings 2-3).

### Phase 2: Thor FA4 switch (opt-in), opt-in FA4 for `mot`

Phase Status: completed

`use_fa4=None` resolution (opt-in through `FLASHRT_THOR_FA4=1`) and `use_fa4_mot`, in `fa4_backend.py`, `attn_backend.py`, `imagewam_thor.py` and tests; verified by dispatch tests with an fp32 matmul stand-in matching the cuBLAS chain at the real shapes (cosine, max-abs, rel_l2), the resolution resolving to False on H100, and an fp16 end-to-end quick run matching the baseline.

### Phase 3: recommendation and Thor handoff

Phase Status: completed

OPT-019 with evidence, recommendation and Thor checks (`opportunities.md`); the per-site numbers are that entry's Findings 2-4.

### Phase 4: FA4 back to opt-in

Phase Status: completed

`use_fa4=None` resolves to the cuBLAS chain on every device and `FLASHRT_THOR_FA4=1` opts in, making FA4 the default a one-line change (`_FA4_OPT_IN_DEFAULT`) in `imagewam_thor.py`; verified by `tests/test_imagewam_fa4_dispatch.py` resolution tests covering every combination of the environment variable, runtime availability and explicit argument.

### Phase 5: dedicated FA4 output buffer

Phase Status: completed

FA4 output goes to `fa4_out` slots, which the frontend owns as `(total, hidden)`, instead of `logits`, which overruns at small dims (`attn_backend.py`, `imagewam_thor.py`, FA4 tests, FA4 benches); verified by guard-band tests after `fa4_out`, and on `logits`, at (a0, total) = (8, 12), (8, 24) and (905, 969), plus a frontend range check, both failing on the old staging, with capacity checked at construction.

### Phase 6: fall back to the cuBLAS chain when FA4 fails

Phase Status: completed

An FA4 failure during `set_prompt()`'s warmup or capture logs, warns, records `fa4_fallback_reason`, rebuilds the backend without FA4 and captures again (`imagewam_thor.py`, `tests/test_imagewam_fa4_dispatch.py`); verified by stand-ins that fail at first call, inside capture, and by invalidating the capture, all recovering to the chain's output with the caller's stream restored, while a failure with FA4 off still raises.

### Phase 7: Thor confirmation

Phase Status: blocked

The Thor checklist in `opportunities.md` OPT-019, with FA4 explicitly opted in: the real-FA4 real-shape test at 905/905 and 64/969, kernel timings, an nvfp4 end-to-end official compare with FA4 on vs off, and an `infer()` A/B with FA4 on vs off.
Blocker: no sm_110 device and no FA4 runtime on the dev box (`issues.md` ISSUE-023).

# Plan: configuration consolidation (workload / precision / profile)

Plan Status: completed. W0-W12, T1-T5 and S1-S4 finished with the `0920c`
round (ISSUE-080 resolved); the follow-up phases F1-F3 (the served default
promoted to the fastest configuration, the latency baselines per
configuration, the workload in the calibration identity) finished with the
`0921` Thor round (THOR_STATUS_SUMMARY.md). What that round left open is
ISSUE-087 (the full pytest run's capture-state cascade), not a phase.

## Problem

### Current

`ImageWAMThorFrontend.__init__` takes about twenty parameters
(`imagewam_thor.py:199`), and the per-model shape lives in a 19-key
`dims` dict (`_DEFAULT_DIMS`, `imagewam_thor.py:166`, toy defaults) that
callers override with `dims_override`. Two kinds of knobs are mixed in
one surface:

- **Workload** (what the deployment serves): camera count and image size
  (through `ref_h/ref_w`), text length (`x0`), action horizon
  (`num_action`), proprio dim, denoise steps and schedule shift. Today
  these are entered as derived numbers (`x0`, `a0`, `total`, `img_len`,
  `dt`) that must agree with each other; nothing checks that they do
  except `ref_h * ref_w == img_len` (`imagewam_thor.py:485`).
- **Optimisation switches** that grew from parallel roadmap branches:
  `precision` (8 values, `_PRECISIONS`, `imagewam_thor.py:117`),
  `text_trim`, `use_fa4` (env `FLASHRT_THOR_FA4`), `use_fa4_mot`,
  `vae_encoder`, `vae_graph_input`, `nvfp4_awq` (+ `awq_alpha`,
  `awq_scope`), `gemm_variant_autotune`, `gemm_runner`, and the fusion
  flags (`merge_qkv_mlp`, `merge_linear2`).

Consequences observed in this repository:

- 20 files carry a literal copy of the LIBERO dims (`grep x0=513`:
  benchmarks, gates and tests). Only 3 import the canonical
  `flash_rt/models/imagewam/libero_dims.py` (`LIBERO_REAL_DIMS`).
- The served defaults are decided by an env var plus constructor flags;
  the configuration that is fastest on Thor (`text_trim` + FA4 backbone +
  FA4 mot + native VAE in graph, 106.1 ms vs 203.3 ms default, THOR
  results in opportunities.md) needs four switches that no single entry
  point sets.
- `text_trim=True` is refused by `runtime_surface()`, `pipeline_resources()`
  and `export_model_runtime()` (`_refuse_text_trim`,
  `imagewam_thor.py:1779`), because the ABI describes one graph at the
  maximum dims (issues.md ISSUE-080, conditions 4, 5, 6).
- Illegal combinations are rejected in scattered places
  (`gemm_variant_autotune` vs precision at `imagewam_thor.py:226`,
  `vae_graph_input` needing `ae_model_path` at `:255`, `nvfp4_awq` needing
  a calibration file, native runtime not supporting AWQ), not by one
  resolver, so the set of legal configurations is not written down.

Facts that constrain the design (checked in source and in the checkpoint
config, not assumed):

- Backbone widths (`hidden`, `HD`, `NH`, `mlp_hidden`,
  `joint_attention_dim`, layer counts) are read from checkpoint tensor
  shapes and key counts, not from `config.yaml`. `config.yaml` carries the
  action-expert dims, `max_action_horizon` (64), the noise schedule shifts
  and `proprio_dim`. There is no denoise step count in the served config
  (`eval_num_inference_steps: 10` is a training-time eval setting), and
  `data.train.context_len: 128` is not the served text length (512), so
  `num_steps` and `text_max_len` belong to the workload only.
- `benchmarks/imagewam_e2e_official_compare.py:REAL_DIMS` and
  `libero_dims.LIBERO_REAL_DIMS` describe one workload: two 224x224 views
  -> `ref_h=14, ref_w=28`, `x0=513` (512 text + 1 proprio), `a0=905`
  (`x0 + 392`), `total=969` (`a0 + 64`), `proprio_dim=8`, `shift=5.0`,
  10 steps.
- `precapture_text_lengths` already exists (`imagewam_thor.py:1504`);
  ISSUE-080 condition 6 is only the bounded per-length cache and its use
  at startup.
- `vae_graph_input=(num_views, in_h, in_w)` is fully determined by the
  workload; it is a derived value today entered by hand.
- `tests/test_imagewam_thor_precision_routing.py` already pins the
  weight-to-linear-class routing per precision (`EXPECTED_ROUTING`, CPU
  only). A profile test belongs there, not in a new file.

### Problem

There is no single object that says what workload is served, and no
single place that maps a named deployment profile to a legal set of
switches. Deployment-relevant knobs (sequence length, action dimension,
ODE loop) are entered as derived integers, optimisation switches are
independent booleans, and the best configuration is not reachable by name.

### Measurable goal

- One immutable `ImageWAMWorkload` carries the workload; `x0`, `a0`,
  `total`, `img_len`, `ref_h/ref_w`, `dt` and `vae_graph_input` are
  derived from it and validated (inconsistent input raises before any
  allocation).
- One `resolve_config(...)` returns `(dims, options)` or raises a single
  error naming the illegal combination; every check currently scattered
  in `__init__` is a rule in it.
- Named profiles (`default`, `fast`, plus the precision tiers) map to
  option sets and are pinned by a CPU-only test; changing a profile
  without changing the test fails.
- The `LIBERO_REAL_DIMS` dict is produced from
  `ImageWAMWorkload.libero()` and the 20 literal copies import it;
  none redefines `x0/a0/total`.
- With `ImageWAMWorkload.libero()` and the profile that reproduces today's
  defaults, `infer()` outputs are bit-identical to the current frontend
  (fp16 and nvfp4), and the resolved `effective_config` line matches.
- Thor: the matrix (`scripts/imagewam_thor_matrix.sh`) rows reproduce
  their recorded P50 within run-to-run noise when built through the new
  entry point, and the target-workload table (THOR_CHECKLIST.md D) can be
  entered as a workload instead of hand-edited dims.

Not a goal: changing any kernel, precision numerics or graph capture
order; removing a switch that the experts need (they move, they are not
deleted); making `text_trim` default (that stays with ISSUE-080).

## Structure

| Module | Responsibility | State it owns |
|---|---|---|
| `Workload` (`flash_rt/models/imagewam/workload.py`, new) | Workload description, derivation of sequence layout, validation | The workload fields and the derived layout (`x0`, `img_len`, `a0`, `total`, `ref_h`, `ref_w`, `dt`) |
| `StructureConstants` (`flash_rt/models/imagewam/structure.py`, new) | Model structure read once: backbone widths from checkpoint tensor shapes, action-expert dims and schedule from `config.yaml` | Backbone/action-expert dims; none of the workload |
| `PrecisionSpec` (`flash_rt/models/imagewam/precision.py`, new) | Enum of precision tiers with their properties: needs calibration, alignment rule, native-runtime support, whether tile autotune applies, fallback class per shape | The precision property table (replaces `_PRECISIONS`, `_NVFP4_PRECISIONS`, `_STATIC_FP8_PRECISIONS`, `_VARIANT_TUNED_PRECISIONS`, `_wrap_linear` fallback conditions) |
| `Options` + `resolve_config` (`flash_rt/models/imagewam/config_resolver.py`, new) | Named profiles; the single rule set for illegal combinations; produce `(dims, Options)` and the `effective_config` string | The profile table and the rule set |
| Frontend (`imagewam_thor.py`) | Build buffers, weights and graphs from resolved `(dims, Options)` | Buffers, graphs, weights (unchanged) |
| ABI / native identity (`runtime_export.py`, `runtime_surface.py`, `pipeline_resources.py`, `native_*.py`, `calibration_file.py`) | Carry the workload as part of the model identity so a runtime and a calibration file are checked against one description | Runtime identity and calibration identity (`calibration_file.py:identity_dims`, `validate_for`) |
| Public entry (`flash_rt/frontends/torch/imagewam_thor.py`, `load_imagewam(workload, profile=..., **expert)`) | The small surface a deployment uses | None |

Ownership rules:

- The workload is owned by `Workload`. Nothing else stores `x0`, `a0`,
  `total`, `dt` or `vae_graph_input` as an independent input; they are
  read from it.
- Structure constants are owned by `StructureConstants`; the workload
  never contains them, and they are never typed by hand outside toy test
  dims.
- Legality of a combination is owned by `resolve_config`; the frontend
  constructor keeps only assertions for internal invariants.
- Optimisation switches are split into three tiers (the public entry
  exposes only the first):
  1. Deployment surface: `Workload`, `precision`, `profile`, calibration
     path.
  2. Profile-controlled (set by a named profile, overridable by an
     expert): `text_trim`, FA4 (backbone and mot as one tier),
     `vae_encoder`/`vae_graph_input` (derived from the workload when the
     native VAE is selected), `nvfp4_awq`.
  3. Expert/diagnostic (constructor only, not on the public entry):
     `gemm_variant_autotune`, `gemm_runner`, `awq_alpha`, `awq_scope`, the
     fusion flags (`merge_qkv_mlp`, `merge_linear2`; kept as A/B switches
     for `benchmarks/imagewam_fusion_ab.py`).

## Interface

`flash_rt/models/imagewam/workload.py`:

```
@dataclass(frozen=True)
class ImageWAMWorkload:
    num_views: int
    image_h: int; image_w: int          # per view, pixels
    text_max_len: int                   # tokens padded to (512 for LIBERO)
    action_horizon: int                 # <= structure.max_action_horizon
    action_dim: int                     # output dim after de-normalisation
    proprio_dim: int
    num_steps: int                      # denoise steps
    shift: float                        # schedule shift
    num_train_timesteps: int = 1000

    @staticmethod
    def libero() -> "ImageWAMWorkload"  # 2 x 224x224, 512, 64, 8, 10, 5.0
    def layout(self, structure) -> "SequenceLayout"
    # SequenceLayout(x0, img_len, a0, total, ref_h, ref_w, dt); raises on
    # non-multiple-of-patch image size, horizon > structure limit, any
    # dimension <= 0.
    def vae_graph_input(self) -> tuple[int, int, int]   # (num_views, h, w)
```

`flash_rt/models/imagewam/structure.py`:

```
@dataclass(frozen=True)
class ImageWAMStructure:            # backbone + action expert, no workload
    hidden, HD, NH, mlp_hidden, joint_attention_dim,
    num_layers_double, num_layers_single,
    action_hidden_dim, action_attn_width, action_mlp_hidden,
    action_num_layers_double, action_num_layers_single,
    max_action_horizon, patch_stride
    @staticmethod
    def from_checkpoint(ckpt_path) -> "ImageWAMStructure"
    @staticmethod
    def toy() -> "ImageWAMStructure"   # the current _DEFAULT_DIMS values
```

`flash_rt/models/imagewam/precision.py`:

```
class Precision(str, Enum): FP16, FP16_CUTLASS, FP8, NVFP4, FP8_STATIC,
    FP8_STATIC_CUTLASS, E0M3_HADAMARD, NVFP4_SIM
    needs_calibration: bool         # fp8_static*
    supports_native_runtime: bool
    supports_awq: bool              # nvfp4 family
    supports_tile_autotune: bool
    alignment_fallback(n, k) -> Fp16 | native   # today's _wrap_linear rules
```

`flash_rt/models/imagewam/config_resolver.py`:

```
@dataclass(frozen=True)
class ImageWAMOptions:
    precision: Precision
    text_trim: bool
    use_fa4: bool; use_fa4_mot: bool
    vae_encoder: Literal["torch", "native"]
    vae_graph_input: tuple[int,int,int] | None
    nvfp4_awq: bool
    calibration_path: str | None
    # expert tier
    gemm_variant_autotune: bool; gemm_runner: object | None
    awq_alpha: float; awq_scope: str
    merge_qkv_mlp: bool; merge_linear2: bool

PROFILES: dict[str, ProfileSpec]    # "default", "fast", ...

def resolve_config(workload, structure, *, profile="default",
                   precision=None, calibration_path=None, ae_model_path=None,
                   **expert) -> ResolvedConfig
# ResolvedConfig(dims: dict, options: ImageWAMOptions, effective_config: str)
# Raises ConfigError("<rule id>: <combination> ...") on an illegal set.
```

Rules (each has an id, a test, and moves an existing check):

| Rule | Combination | Today's location |
|---|---|---|
| R1 | `Precision.needs_calibration` without `calibration_path`. Behaviour change vs today: the constructor only logs a warning and uses N(0, 0.1) placeholder scales (`_calibrate_fp8`); `resolve_config` raises unless `allow_placeholder_calibration=True`, the old constructor path keeps the warning | `_calibrate_fp8` path / `calibration_file.py` |
| R2 | `gemm_variant_autotune` with a precision that does not support it | `imagewam_thor.py:226` |
| R3 | native VAE in graph without `ae_model_path`/`flux2_src` | `imagewam_thor.py:255` |
| R4 | `nvfp4_awq` with a non-AWQ precision or without calibration | AWQ setup |
| R5 | *retired in S2 for the ABI face and in S4 for the native pipeline*: with every consumer carrying one graph per trimmed length, the refusal has no referent. `text_trim` is legal for `consumer="infer"`, `"abi"` and `"native"` | was `_refuse_text_trim`, removed |
| R6 | native runtime with AWQ | native runtime |
| R7 | workload layout inconsistent (`ref_h * ref_w != img_len`, horizon above structure limit) | `imagewam_thor.py:485` |
| R8 | calibration file identity mismatch vs the workload and `text_trim` | `calibration_file.py:validate_for` |

Frontend entry (`imagewam_thor.py`), added beside the existing
constructor (which keeps its signature, so existing callers, tests and
benchmarks do not break):

```
ImageWAMThorFrontend.from_config(resolved: ResolvedConfig, *, ckpt_path, ...)
def load_imagewam(ckpt_path, workload, *, profile="default",
                  precision=None, calibration_path=None, **expert)
```

W0 addendum: names frozen while executing W6-W12, each one a plan edit:

- `ImageWAMStructure.libero()` (`structure.py`), beside `toy()` and
  `from_checkpoint()`. The structure constants of the real
  `ImageWAM-FLUX.2-4B-LIBERO` release (`hidden=3072`, `HD=128`, `NH=24`,
  `mlp_hidden=9216`, `joint_attention_dim=7680`, 5 double + 20 single
  backbone layers and the same for the action expert,
  `action_hidden_dim=1024`, `action_attn_width=3072`,
  `action_mlp_hidden=4096`, `max_action_horizon=64`,
  `patch_stride=16`), needed because `LIBERO_REAL_DIMS` mixes structure
  and workload and the resolver takes the two separately. It is a
  constant table, pinned two ways: against `LIBERO_REAL_DIMS`' own
  structure entries, and against `from_checkpoint()` when a checkpoint
  is configured.
- `ImageWAMTorchFrontendThor.from_config(resolved: ResolvedConfig, *,
  workload: ImageWAMWorkload | None = None, **kwargs)`. A classmethod
  that splits `ResolvedConfig` into the constructor's arguments through
  `frontend_kwargs_from_config(resolved, **kwargs)`: dims (through
  `dims_override=resolved.frontend_dims()`) and every option in
  `resolved.options`; `**kwargs` are the path arguments the resolver does
  not see (`ckpt_path`, `ae_model_path`, `flux2_src`,
  `qwen3_model_spec`, `dataset_stats_path`, `vae_resize`). It constructs
  the same object the equivalent constructor call builds, and records what
  it was built from: `workload`, and the resolved configuration readable
  back as the `resolved_config` property. Both are `None` for a frontend
  the constructor built from hand-passed dims.
- `frontend_kwargs_from_config(resolved, *, ckpt_path=None, ...) -> dict`
  (module level in `imagewam_thor.py`): the one place that maps a resolved
  configuration onto constructor arguments, so a CPU-only test pins the
  mapping without building a frontend.
- `load_imagewam(ckpt_path, workload, *, structure=None, profile="default",
  precision=None, calibration_path=None, ae_model_path=None,
  flux2_src=None, qwen3_model_spec=None, dataset_stats_path=None,
  consumer="infer", allow_placeholder_calibration=False,
  vae_resize="area", **expert)` in
  `imagewam_thor.py`, the deployment entry. `structure=None` reads
  `ImageWAMStructure.from_checkpoint(ckpt_path)`; a `structure` is
  required when `ckpt_path` is `None` (random-weight run). It calls
  `resolve_config(...)` and `from_config(...)`, so every illegal
  combination raises the resolver's `ConfigError` before anything is
  allocated. `**expert` carries the expert tier (`EXPERT_KEYS`) only; the
  tier-1 arguments above are named.
- S3 (`imagewam_thor.py`, `config_resolver.py`):
  `evict_lru(captures, *, active_x0, limit) -> tuple[int, ...]` is the pure
  eviction decision (least recently used first, never the active capture, may
  return fewer keys than the excess); `text_trim_cache_size` is a constructor
  keyword and an expert option (default 32, rule V1 for a non-int or `< 1`);
  `precapture_text_lengths` is a keyword of `from_config` and
  `load_imagewam` taking `x0` values (valid tokens + 1 with proprio, the units
  of the `captured_text_lengths` property) and capturing them once at
  construction.
- Runtime identity: `runtime_surface` `setup_identity` describes the
  workload explicitly, as `workload.<field>` entries beside the existing
  `dims.<key>` entries. The field list is
  `runtime_surface.WORKLOAD_IDENTITY_FIELDS` (`num_views`, `image_h`,
  `image_w`, `text_max_len`, `action_horizon`, `action_dim`,
  `proprio_dim`, `num_steps`, `shift`) and
  `runtime_surface.workload_identity(workload)` renders the pairs, so the
  ABI identity, the native path and the tests read one list. The entries
  are additive and descriptive: they invalidate no stored calibration file.
  (`calibration_file.IDENTITY_DIM_KEYS` kept its key set in W10; phase F3
  later added `num_views`, `image_h` and `image_w` to it, format version 3.)

State transitions: `Workload` + `Structure` -> `resolve_config` ->
`ResolvedConfig` (immutable) -> frontend construction. No state is
mutated after resolution; `effective_config` is the resolved value, not a
re-read of the frontend attributes, so the compare script, the matrix
script and the ABI identity print the same string.

## Flow

```
Deployment:
  ImageWAMWorkload(...)            ImageWAMStructure.from_checkpoint(ckpt)
              \                       /
               resolve_config(workload, structure, profile, precision, ...)
                       |  (rules R1..R8; ConfigError on illegal)
                       v
                ResolvedConfig(dims, options, effective_config)
                       |
        ImageWAMThorFrontend.from_config(...)   # thin: build buffers/graphs
                       |
        infer()  |  export_model_runtime(identity=workload+options)  |  native
```

Runtime inference flow, the graph, the per-length `text_trim` graph cache
and `set_prompt` are unchanged.

Existing callers: `dims_override=` remains accepted by the constructor;
`from_config` is the new path. A test asserts that
`resolve_config(ImageWAMWorkload.libero(), ...)` produces a `dims` dict
equal to `LIBERO_REAL_DIMS`, which is what makes the switch of the 20
files a mechanical change.

## Code Mapping

| Item | File |
|---|---|
| Workload, layout, validation | `flash_rt/models/imagewam/workload.py` (new) |
| Structure constants | `flash_rt/models/imagewam/structure.py` (new); reads via `checkpoint_loader.py:load_real_imagewam_state_dict` and the checkpoint `config.yaml` |
| Precision enum and property table | `flash_rt/models/imagewam/precision.py` (new); absorbs the tuples at `imagewam_thor.py:117-130` and the `_wrap_linear` fallback rules (`imagewam_thor.py:754`) |
| Options, profiles, rules R1-R8, `effective_config` | `flash_rt/models/imagewam/config_resolver.py` (new) |
| Canonical LIBERO dims | `flash_rt/models/imagewam/libero_dims.py` (becomes a thin view over `ImageWAMWorkload.libero()`) |
| Routing / profile / rules tests | `tests/test_imagewam_thor_precision_routing.py` (extended: profile table, rule ids), `tests/test_imagewam_workload.py` (new, CPU: layout derivation, validation, `libero()` equals `LIBERO_REAL_DIMS`) |
| Frontend construction | `flash_rt/frontends/torch/imagewam_thor.py` (`from_config`, `load_imagewam`; constructor kept) |
| Hand-copied dims to replace | `benchmarks/imagewam_e2e_official_compare.py`, `imagewam_thor_graph_bench.py`, `imagewam_attention_share_bench.py`, `imagewam_fusion_ab.py`, `imagewam_e0m3_accuracy_study.py`, `imagewam_text_trim_bench.py`, `imagewam_vae_stage_bench.py`, `imagewam_e0m3_hadamard_thor_check.py`, `imagewam_thor_small_m_tile_sweep.py`; `tests/gate_imagewam_model_runtime_export.py`, `gate_imagewam_native_parity.py`, `gate_imagewam_native_schema_parity.py`, `test_imagewam_residual_norm_fusion.py`, `test_imagewam_fa4_backbone.py`, `test_imagewam_fa4_dispatch.py`, `test_imagewam_thor_real_wiring.py`, `test_imagewam_text_trim_graph_safety.py`, `test_imagewam_quant_linear.py`, `test_imagewam_text_trim.py`, `test_imagewam_thor_precision_routing.py` (toy dims stay toy and are marked as such) |
| Derived flags | `vae_graph_input` from `workload.vae_graph_input()`; `use_fa4_mot` folded into the FA4 tier; tuner/AWQ knobs and fusion flags stay constructor-only |
| ABI / native identity | `flash_rt/models/imagewam/runtime_export.py`, `runtime_surface.py`, `pipeline_resources.py`, `native_resources.py`, `native_runtime.py`, `calibration_file.py` (`identity_dims`, `validate_for`); C++ side under `cpp/models/imagewam` reads the same identity fields |
| Effective-config string | produced by `config_resolver.py`; consumed by `benchmarks/imagewam_e2e_official_compare.py`, `scripts/imagewam_thor_matrix.sh` (`parse_log`), `scripts/imagewam_thor_validation.sh` |
| Thor checklist / status | `THOR_CHECKLIST.md`, `THOR_STATUS_SUMMARY.md`, `opportunities.md` |

## Implementation Phases

`W` items are code; `T` items are Thor runs; `S` is a side track that does
not block the rest.

```
W0 freeze the interface (Workload / Structure / Precision / Options / rule ids)
 +--> W1 Workload: layout derivation + validation
 +--> W2 Structure constants (checkpoint tensors + config.yaml)
 +--> W3 Precision enum + property table
 +--> W4 resolve_config: profiles + rules R1-R8 (pure Python; needs W1, W3 types)
 +--> W5 tests: workload, profile table, rule ids (CPU; extends the routing test)
 join -> W6 frontend: build dims from ResolvedConfig (from_config, constructor kept)
      -> W7 frontend: wire Precision + resolver (replace tuples / fallback rules)
           +--> W8 hand-copied dims -> import
           +--> W9 derive / demote flags (vae_graph_input, fa4_mot, tuner, fusion)
           +--> W10 ABI + native identity carries the workload (runtime_export,
                surface, pipeline_resources, native_*, calibration_file)
           join -> W11 public entry load_imagewam + expert/constructor split
                     (needs T4: preset contents)
                -> W12 Thor validation on the new entry

Thor line (independent of W0-W10; T4 feeds W11 and the profile contents):
 T1 THOR_CHECKLIST A (data trustworthy) -> T2 B (missing data)
   -> T3 C (configuration matrix) -> T4 decide preset contents
   -> T5 FA4 / native VAE default decision (THOR_CHECKLIST C criteria)

Side track S (S1-S4 completed: the R5 refusal fell in S2 for the ABI face
and in S4 for the native pipeline):
 S1 ISSUE-080 condition 4: fixture v2 with trim (fp16 reference)
 S2 ISSUE-080 condition 5: runtime surface / ABI / native for per-length graphs
 S3 ISSUE-080 condition 6: bounded per-length graph cache + startup use of
    `precapture_text_lengths`
```

Parallelism: W1, W2, W3 are independent files after W0; W4 needs the W1 and
W3 types but not their implementations, so it starts once W0 is frozen, and
W5 runs when W1-W4 land. W6 and W7 both edit the frontend constructor
(`imagewam_thor.py`) and are serial, one owner: all logic lives in the new
pure-Python modules and the constructor change is only "take resolved dims
and options". W8, W9 and W10 touch disjoint files after W7 and run in
parallel — W8 is mechanical (import instead of literal), W9 changes which
flags are public, W10 changes the identity that ABI and calibration files
check. W11 waits for T4 because the `fast` profile contents are a measured
decision, while the mechanism (W0-W10) does not. T1-T5 run on Thor
independently of the code work and are the source of the profile contents.

### Phase W0: interface freeze
Phase Status: completed
This section's Interface and rule table are the contract; changing a name
here is a plan edit. Files: `plan.md` only.

### Phase W1: `ImageWAMWorkload` and layout
Phase Status: completed
`flash_rt/models/imagewam/workload.py` (new: `ImageWAMWorkload`,
`SequenceLayout`): derive `x0`, `img_len`, `a0`, `total`, `ref_h`, `ref_w`,
`dt` and `vae_graph_input` and reject inconsistent input. Views concatenate
along the width (`vae_stage.py`, `pipeline_real.py`, `rope.py:build_img_ids`),
so LIBERO's two 224x224 views give 14 x 28 and the derivation must reproduce
513/905/969 and 14 x 28 exactly for `libero()`. Observation:
`tests/test_imagewam_workload.py` (CPU): `libero()` equals `LIBERO_REAL_DIMS`
and invalid inputs raise.

### Phase W2: `ImageWAMStructure`
Phase Status: completed
`flash_rt/models/imagewam/structure.py` (new): backbone dims from checkpoint
tensor shapes, the rest from `config.yaml`, `toy()` for tests, reads only
through `checkpoint_loader.py`.

### Phase W3: `Precision` property table
Phase Status: completed
`flash_rt/models/imagewam/precision.py` (new): precision properties as data.
The alignment rules reproduce today's `_wrap_linear` fallbacks (fp16_cutlass
n/k % 8, nvfp4 % 16, fp8/fp8_static/fp8_static_cutlass % 8 -> fp16 linear).

### Phase W4: `resolve_config`, profiles, rules
Phase Status: completed
`flash_rt/models/imagewam/config_resolver.py` (new): rules R1-R8 with ids in
one function, plus an `effective_config` string identical in format to the
compare-script line. Observation: one test per rule id (legal and illegal
case); the `default` profile reproduces today's constructor defaults.

### Phase W5: tests
Phase Status: completed
`tests/test_imagewam_thor_precision_routing.py` (extended) and
`tests/test_imagewam_workload.py` (new) pin the profiles and the rules;
CPU-only, no GPU needed.

### Phase W6: frontend builds dims from `ResolvedConfig`
Phase Status: completed
`from_config` builds the frontend from resolved dims and options, while the
constructor signature and `dims_override` behaviour stay
(`flash_rt/frontends/torch/imagewam_thor.py`). Observation: `libero()` through
`from_config` is bit-identical to the old constructor with `LIBERO_REAL_DIMS`
(fp16 and nvfp4, existing `test_imagewam_thor_real_wiring.py` plus a new
equality check).

### Phase W7: frontend uses `Precision` and the resolver
Phase Status: completed
The precision tuples and the scattered checks the resolver now owns are
deleted, and `_wrap_linear` reads `Precision` properties, with
`test_imagewam_thor_precision_routing.py` results unchanged
(`EXPECTED_ROUTING` still holds).

### Phase W8: replace hand-copied dims
Phase Status: completed
The 20 literal copies import the workload/`libero_dims` (files in Code
Mapping); toy dims stay toy and are labelled. Observation:
`grep -rE "x0\s*=\s*513"` returns only `libero_dims.py` and the workload test,
and every touched script still imports with its `--help`/collect step working.

### Phase W9: derive and demote flags
Phase Status: completed
`vae_graph_input` derived from the workload, `use_fa4_mot` folded into the
FA4 tier of a profile, and the tuner, AWQ tuning and fusion flags
(`benchmarks/imagewam_fusion_ab.py`) left constructor-only. Observation:
existing callers still work and the resolved `effective_config` shows them.

### Phase W10: ABI and native identity carry the workload
Phase Status: completed
`identity` (ABI) and the calibration file identity include the workload
fields, so a runtime and a calibration file for a different workload are
rejected by name (files in Code Mapping). Observation:
`gate_imagewam_model_runtime_export.py`, `gate_imagewam_native_schema_parity.py`;
a mismatched-workload calibration file is refused with the differing field
named.

### Phase W11: public entry and constructor split
Phase Status: completed
`load_imagewam` is the deployment entry and the expert switches reach
`resolve_config` through `**expert` rather than positional constructor
arguments; the `fast` profile contents come from T4 (files:
`imagewam_thor.py`, `config_resolver.py`, `THOR_STATUS_SUMMARY.md` options
table). Observation: the profile test pins the contents and the compare
script builds its frontend through `load_imagewam`.

### Phase W12: Thor validation on the new entry
Phase Status: completed
Goal: the recorded numbers reproduce through the new path and the target
workload runs as a `Workload`. Observation: matrix rows `default` and `stack`
within run-to-run noise of the recorded 203.3 / 106.1 ms (same commit
conditions, GPU exclusivity recorded), `vs official` not below the recorded
values, and the target-workload table filled. Rounds: `0919e` fixed and
verified the VAE encode geometry blocker (ISSUE-086) and served the target
workload on all three paths, with the LIBERO rows fidelity-identical to the
previous round; `eccf14f` reproduced the recorded numbers through the new
entry, showed the previous session's 225 ms and 13.6 ms spread to be that
session's state rather than a scope difference (ISSUE-082 resolved) and
re-based the latency gate's Thor baseline (231.6 -> 202.2 ms, closing item
E2); `c20f3a0` passed both identity checks (`effective_config` character for
character, the nine `workload.<field>` entries, ABI bit-exact against
`infer()`, native schema against the golden records; C1-C3 closed) and its
`default` rows (225.2 / 225.5 ms against the recorded 203.3 ms gate number)
left the like-for-like baseline to the open owner decision in "Decisions
pending" 1. Numbers: `THOR_STATUS_SUMMARY.md` (`0919e`, `eccf14f`, `c20f3a0`);
the switch ladder and the FA4 / native-VAE criteria: `opportunities.md`
OPT-019, OPT-021, OPT-030.

### Phase T1-T5: Thor line
Phase Status: completed
The Thor-only rows: the lettered checklist sections A-E ran and were removed
as their conclusions were recorded (T1-T3 the matrix, the switch ladder and
the target-workload table; T4 the preset contents in "Decisions pending"; T5
the FA4 and native-VAE default criteria). The line's last three rows ran in
the `0920c` round — the native pipeline's own per-length capture, the
`frt_model_runtime_v1` export gate at `nvfp4` and the `e0m3_hadamard`
trim-safety check — all three green, so `THOR_CHECKLIST.md` carries no
pending item and ISSUE-080 is resolved. Files: `THOR_CHECKLIST.md` (finished
items removed), `opportunities.md`, `issues.md`.

### Phase S1: a gate fixture whose `fp16` reference is recorded trimmed
Phase Status: completed
ISSUE-080 condition 4: fixture v1's `fp16` reference is untrimmed, so a
trimmed `fp16` run measures 0.99837 / 0.99579 against it, below the fp16
bounds 0.999 / 0.995, while being closer to official (0.99998 / 0.99992) — a
trimmed configuration cannot pass the gate until a fixture recorded trimmed
exists. Files: `benchmarks/imagewam_gate_fixture_generate.py` (the `TEXT_TRIM`
/ `--text-trim` switch, the reference built through `load_imagewam`),
`flash_rt/datasets/imagewam_gate_fixture.py` (`ImageWAMGateFixture.text_trim`,
`FixtureManifest.text_trim`, absent means untrimmed so v1 still loads),
`tests/gate_imagewam_libero.py` (`--text-trim`; a fixture recorded with the
other value is refused before the checkpoint hash),
`tests/fixtures/imagewam_gate/fidelity_thresholds.json` (the like-for-like
rule stated, numbers unchanged), `tests/test_imagewam_regression_gate.py`.
Observation: the CPU round trip through `save`/`load` and the v1 manifest, the
refusal of either mismatch, and the generator's imports resolving against the
compare script (the check that caught the generator's own `REAL_DIMS`
breakage); on Thor the v2 fixture was generated (`text_trim=true`) and the
trimmed nvfp4 gate passes against it, with a trimmed run against v1 and an
untrimmed run against v2 both refused — numbers in
`THOR_STATUS_SUMMARY.md` `a84916a`. The fixture's manifest still has to be
committed (checklist item E2).

### Phase S2: the ABI carries one graph per trimmed length
Phase Status: completed
ISSUE-080 condition 5: the exec layer already models a graph as a `ShapeKey ->
graph-exec` variant table with an LRU cap, so a trimmed frontend's per-length
graphs become keys of one declared graph and `step` replays the key of the
length the prompt set, which is what lifts rule R5 for `consumer="abi"`; the
native pipeline kept its refusal until its own phase. Files:
`flash_rt/models/imagewam/runtime_surface.py`, `runtime_export.py`,
`flash_rt/frontends/torch/imagewam_thor.py`,
`flash_rt/models/imagewam/config_resolver.py` (the R5 scope), their tests.
Interface: `runtime_surface()` exposes `graph_variants` (`active_key`, the
`(key, graph_exec)` entries ascending, `per_prompt_length`), the `ShapeKey` is
the context length `x0`, `graph_variant_plan` turns the table into the
declaration's `default_key` / `keys` / `max_variants` (pure, CPU-tested), the
manifest records `text_lengths`, and `step` replays `active_dims["x0"]` after
`frt_graph_has_variant`, refusing an uncaptured length by name.
`setup_identity` gains the `text_trim` pair, so the export fingerprint of an
existing untrimmed deployment changes (no artifact compatibility is required
at this stage). Observation: the pure table/plan seam and the resolver's R5
scope are covered on CPU; on Thor the multi-length check passes (the ABI tick
at two captured lengths is bit-exact against `infer()`, and the `guards` 7
tests pass) and the three-path numbers are in `THOR_STATUS_SUMMARY.md`
`eccf14f`.

### Phase S4: the native model runtime carries one graph per trimmed length
Phase Status: completed
ISSUE-080 condition 5's remaining half: the ABI face served trimmed prompts
while the native C++ pipeline refused them explicitly at both entry points
(`ImageWAMNativeRuntime.create`, `export_model_runtime(io="native")`). What it
needed, all of it C-ABI and host work and none of it Python-only:
`NativeRuntime` held one `graph_`, one `owned_graph_` for the `capture()` path
and one `context_rows_` (used by `set_proprio_row`'s bound and
`set_pipeline`'s dims check), so the keyed variant table has to replace them
(`frt_graph` already offers adopt/has_variant/replay); the host needs a
visible key (`use_graph(key, exec)` / `has_variant(key)` /
`set_text_length(key)`, wired into the prompt/proprio verb path, since C++
cannot see the Python prompt); `native_resources.build_io_config` hands the
per-key `context_rows`; and `native_schema.cpp` plus the schema-parity gate
gain the new records. Files: `cpp/models/imagewam/**`,
`flash_rt/models/imagewam/native_runtime.py`, `native_resources.py`,
`runtime_export.py`, the native gates. Interface: `frt_imagewam_io_config`
declares the deployment's text lengths (`num_text_lengths` / `text_lengths`,
`context_rows` = the active one), and the handle gained `use_graph(key, exec)`,
`has_variant(key)`, `variant_exec(key)`, `set_text_length(key)` and
`text_length`; `step`, `set_proprio_row` and `set_pipeline` resolve against the
active key. The key is `x0`, the same space `GraphVariants` uses, and adoption
is refused while a model runtime over the handle is live — so
`set_text_length` and `set_proprio_row` are the two calls that stay legal on
the hot path. `export_model_runtime(io="native")` adopts the handle's exec per
captured length and records the same `text_lengths` manifest table as the ABI
face.

**R5 is lifted**: the native pipeline carries one resource table and one graph
per text length. `pipeline_resources()` describes the ACTIVE length (sequence
dims and RoPE table active, buffers max-size and shared); the handle installs
one pipeline per key (`set_pipeline` installs the key its config carries,
replaces only that key's pipeline and graph, and makes it the active text
length; `gemm_shapes` / `set_gemm_algo` / `run` / `capture` all resolve
against the active key); `ImageWAMNativeRuntime.capture_pipeline_text_lengths(source)`
activates each of `source.captured_text_lengths` (a property), installs its
table and captures its graph, then restores the active length; `graph_producer`
follows the active key. Rule R5 is removed from `config_resolver.py` and from
the table above; the `native` profile is the native consumer's set — the served
`default`'s switches with FA4 explicitly off, contents otherwise equal to
`default`. Observation:
`tests/test_imagewam_native_runtime.py::test_native_tick_matches_infer_at_every_captured_length`
(two captured lengths, the shorter ticked first, `array_equal` to `infer()` in
the actions and the action latent), the native manifest's length table, and
CPU-only pins for the io config's table and the handle's key plumbing over a
stubbed library; on Thor the per-length resource table and the install loop's
call sequence are pinned over a stub frontend and handle and the 13-file list
stays green. The round numbers (`0920s4`, `0920c`: `array_equal` with
`max_abs=0`, `text_lengths`, `set_text_length` returning `-2` for an uncaptured
length, node counts native 5324 / Python 5348, the schema gate's seven records
line-identical to the golden file, both graph producers and all six mutants
green, and the pipeline's own two-length install and capture) are in
`THOR_STATUS_SUMMARY.md` and `opportunities.md` OPT-029, together with the
22 ms P50 difference from the earlier `08_gate_native` round, which is recorded
there as a session difference, not a measured regression.

### Phase S3: a bounded per-length graph cache, precaptured at construction
Phase Status: completed
ISSUE-080 condition 6: a trimmed frontend captured one graph per distinct
length with no bound, and `precapture_text_lengths` existed but nothing called
it. Files: `flash_rt/frontends/torch/imagewam_thor.py` (`evict_lru`, the
`text_trim_cache_size` constructor keyword, `_touch_capture` /
`_store_capture`, the recency updates, the `precapture_text_lengths` keyword on
`from_config`/`load_imagewam`), `flash_rt/models/imagewam/config_resolver.py`
(`text_trim_cache_size` in `ImageWAMOptions` and `EXPERT_KEYS`, default 32,
rule V1), `tests/test_imagewam_text_trim_cache.py`. Observation: `evict_lru`
(order, the active capture never dropped, a bound of 1, a cache whose only
entry is active), the resolver option and its V1 cases, the constructor
validation before any allocation, and the precapture keyword reaching
`precapture_text_lengths` once through a mocked constructor; the Thor switch,
eviction and memory timings are in `THOR_STATUS_SUMMARY.md` `a84916a`. Two
decisions made while implementing: `precapture_text_lengths` refuses a request
with more distinct lengths than the bound (its refill loop would otherwise
recapture what eviction had just dropped, forever), and the cache may stay
above the bound when the only remaining candidate is the active capture (that
is what keeps the replayable graph alive). The memory measurement corrected
this phase's own assumption: the first captured graph costs +218.0 MiB
reserved / +206.3 MiB allocated and every following one +0.0 / +0.1 MiB (the
captures share the pool), so 15 lengths sit under one graph's fixed cost and
the default bound of 32 is on the order of 221 MiB, not 32 x 218 MiB.

### Phase F1: the served default is the fastest configuration
Phase Status: completed
- Goal: `resolve_config(profile="default")` resolves to text_trim + FA4 at
  both sites + the native VAE inside the graph, degrading where the machine
  or the inputs cannot carry it. `use_fa4_mot` becomes tri-state (`None`
  resolves through the same machine rule as the backbone site), the profile
  table's `vae_encoder`/`vae_graph` accept an "auto" that means "native and
  in the graph when `ae_model_path` is given", and the native consumer
  resolves every auto value to off.
- Modified files: `flash_rt/models/imagewam/config_resolver.py` (`PROFILES`,
  `ProfileSpec`, `ImageWAMOptions`, `resolve_config`, `format_effective_config`),
  `flash_rt/frontends/torch/imagewam_thor.py` (`use_fa4_mot` resolved by
  `_resolve_use_fa4`), `scripts/imagewam_thor_matrix.sh` (every flag row states
  its VAE and both FA4 sites: a row that left one unstated would take the new
  default), `benchmarks/imagewam_gate_fixture_generate.py` (the fp16
  reference is pinned to the cuBLAS chain and the torch VAE, whatever the
  default runs), `benchmarks/imagewam_thor_path_bench.py` (docs),
  `tests/test_imagewam_config_resolver.py`, `tests/test_imagewam_frontend_from_config.py`.
- Affected modules: the deployment entry `load_imagewam`; the constructor's own
  defaults are unchanged (untrimmed, FA4 mot off, torch VAE).
- Observation: CPU tests pin the profile contents, the auto resolution, the
  native-consumer resolution and the `effective_config` line. On Thor: the
  default's own gate run and matrix row (FA4 at both sites, no fallback, the
  VAE inside the graph), and the three risks (FA4 first-call compile, an
  FA4 capture failure falling back, the in-graph VAE's fixed input).
- Observed (`0921`, `de0ef51`): the gate on the default ran three times with
  FA4 at both sites, the native VAE in the graph and no fallback
  (105.42 / 105.93 / 105.66 ms); the flag-row ladder and the profile rows
  reproduce (`stack` 94.4, `profile_default` 93.1, `profile_fast` 93.3 ms);
  `FLASHRT_THOR_FA4=0` degrades without an error (103.2 ms); the first FA4
  compile costs 2.89 s once; the target workload runs on `infer()` and ABI.

### Phase F2: latency baselines per configuration
Phase Status: completed
- Goal: a baseline judges only a run of the configuration it describes. The
  gate names the configuration from what the frontend resolved, and an
  unseeded or unknown configuration is ungated with a paste-ready seed record.
- Modified files: `tests/fixtures/imagewam_gate/latency_baselines.json`
  (schema 2: `served_default`, unseeded, and `untrimmed_reference`, the
  202.2 ms record), `flash_rt/core/regression_gate.py` (`LatencyPolicyTable`,
  `LatencyGate`, seeding), `tests/gate_imagewam_libero.py` (builds through
  `load_imagewam`, `--profile`, `--override`), `scripts/imagewam_thor_validation.sh`
  (its gate rows state the untrimmed-reference switches),
  `tests/test_imagewam_regression_gate.py`.
- Observation: CPU tests for lookup, seeding and naming. On Thor: the gate on
  the default seeds `served_default`; the fidelity checks against fixture v2
  (recorded with the plain chain and the torch VAE) must still pass with FA4
  at both sites and the native VAE.
- Observed (`0921`): it passed (vs official 0.99931-0.99933) and seeded
  `served_default` at 105.42 ms; the three repeats' spread is 0.51 ms, ten
  times smaller than the 5% margin. `untrimmed_reference` still gates its own
  configuration (202.38 ms).

### Phase F3: the calibration identity carries the workload
Phase Status: completed
- Goal: two workloads with equal `ref_h`/`ref_w` but different camera
  geometry (2 x 224x224 vs 4 x 224x112) no longer share a calibration
  identity.
- Modified files: `flash_rt/models/imagewam/calibration_file.py` (format
  version 3, only version 3 read), `flash_rt/models/imagewam/config_resolver.py`
  (`_dims` adds `num_views`, `image_h`, `image_w`),
  `benchmarks/imagewam_build_calibration.py`, `tests/test_imagewam_calibration_file.py`,
  `tests/test_imagewam_workload.py`, docs.
- Affected: a frontend built by hand (`dims_override` without a workload) has
  no camera keys and is refused a version-3 file with a diff that names them;
  it must be built through `load_imagewam` or carry the keys in `dims_override`.
  The bundle's calibration files are refused until re-recorded.
- Observation: CPU tests reproduce the collision and the refusals. On Thor:
  re-record the two bundle files and load them.
- Observed (`0921`): both files re-recorded as version 3 with identity
  `2/224/224`, loaded by the trimmed `fp8_static_cutlass` `stack` row (vs
  official median 0.99994), the version-2 files refused.

## Execution record

Observed on the development machine that ran W1-W11 (WSL2,
`torch.cuda.is_available()` is `False`, `nvidia-smi` reports the GPU blocked
by the operating system), so every observation here is a CPU-side contract
check: nothing that needs a CUDA device was run and no latency or accuracy
number is claimed.

- `tests/test_imagewam_workload.py`, `test_imagewam_structure.py` and
  `test_imagewam_precision_table.py` pin the workload, structure and precision
  tables (W1-W3, W5); `tests/test_imagewam_config_resolver.py` covers one legal
  and one illegal case per rule id (W4).
- `tests/test_imagewam_frontend_from_config.py` checks that
  `frontend_kwargs_from_config` maps every option onto a constructor keyword
  the constructor declares and that `from_config` passes exactly that mapping,
  without constructing the frontend because it allocates CUDA (W6, W9). The
  bit-identical fp16/nvfp4 comparison of `from_config` against the old
  constructor is the W12 Thor item.
- `test_imagewam_precision_table.py` pins the frontend's four precision tuples
  as comprehensions over `Precision` and checks by `ast` that no precision
  literal survives in `_wrap_linear`, with `EXPECTED_ROUTING` unchanged (W7).
- `tests/test_imagewam_public_entry.py` covers `workload_identity()`'s nine
  `workload.<field>` pairs, the `load_imagewam` signature, `ConfigError` before
  construction and the structure read only when given (W10, W11); the identity
  of a real exported runtime is the W12 Thor item.
- W2's addendum read the real checkpoint on this machine:
  `from_checkpoint(<local model.pt>) == ImageWAMStructure.libero()` with no
  missing or extra key.
- W8: `grep -rnE "x0\s*=\s*513"` over `benchmarks/`, `tests/` and `flash_rt/`
  returns four hits — the literal pin table in `tests/test_imagewam_workload.py`
  and three docstrings — so no module redefines the dims. The benchmark modules
  import and their argparse `--help` runs; three of them need `pandas`, absent
  from this machine's `.venv`.

The CPU test set used for every phase above, and the directory-wide collection
count, are recorded once in `PROJECT.md` under the CPU-only work note; that
record is the current one and is not repeated here. Outside ImageWAM,
whole-directory collection still reports errors from modules whose extensions
are not built or installed on this machine (`flash_rt.flash_rt_fp4`,
`_flashrt_exec`, `ml_dtypes`).

The Thor rounds after `c20f3a0` (`eccf14f`, the `0919e` round at `a84916a`,
and the `0920`, `0920s4`, `0920t` and `0920c` rounds) ran the rest of the
line, each leaving `THOR_CHECKLIST.md` as its conclusions were recorded; the
last three rows cleared in `0920c`, so nothing is pending there. The S-track
design is recorded in `issues.md` ISSUE-080 conditions 4, 5 and 6.

## Decisions pending (owner)

Decided (owner, after the `1021` Thor round): `gemm_variant_autotune` is on by default for `precision="nvfp4"` on the Python consumers (profile field `gemm_variant_autotune=None`, auto: nvfp4 and not the native consumer; the `native` profile states it off). Thor `1021` measured −1.98 ms per `infer()` with bit-identical actions; tuned tiles v7 / v7 / v10 for the three N = 1024 ActionDiT shapes. `fp8_static_cutlass` stays on the heuristic (no gain beyond noise); the native pipeline was not measured. The frontend constructor default stays `False` for callers that pass switches by hand (opportunities.md OPT-018).

Decided (owner, after the `c20f3a0` Thor round):

- `libero_dims.py` stays: one import for the benchmarks, gates and tests,
  with its literal values pinned by `tests/test_imagewam_workload.py`.
- No compatibility with previously produced artifacts. This is a
  development stage: a calibration file, a runtime export or a gate fixture
  that no longer matches is re-produced, not migrated. It follows that a
  format or identity change may be made directly.
- The target workload does not get a latency budget and no latency
  pass/fail: the runs record numbers. (Whether `tests/fixtures/imagewam_gate/latency_baselines.json`
  keeps its pass/fail role for the LIBERO gate is a separate question, item
  E2.)
- The service paths are compared as configurations of their own (Python
  `infer()`, ABI, native) rather than one path being picked.

- `text_trim` is the served default, not an option: the deployment's
  instructions are short, so the padded text block is overhead on every
  call, and trimming also removes the padding-key deviation from the
  official model (ISSUE-020). It is what the Python `infer()` path runs
  with; the ABI and native paths get it as soon as they carry a graph per
  trimmed length (the S2 phase below), and the regression gate needs a
  fixture whose `fp16` reference was recorded trimmed (S1).
- The per-length graphs are the exact trimmed lengths, precaptured at
  startup from the lengths the deployment declares, with a bounded cache
  (S3). 16-token buckets are not needed for a fixed instruction set: they
  would bound the graph count for an open-ended one, and they cost a mask
  over the in-bucket padding rows at both attention sites (the official
  model masks padded text keys, so an unmasked bucketed run changes the
  softmax), including in FA4, which has no mask parameter today. The
  comparison is recorded in ISSUE-080.
- Framework first: the configuration surface (`load_imagewam`, `Workload`,
  `Structure`, profiles) must serve any deployment, not one checkpoint. A
  specific target checkpoint is an input, not a design constraint;
  `proprio_dim` and the camera count come from the deployment and are
  validated against the checkpoint's own weights at construction.
- Two workloads are first-class and both stay measured: the served LIBERO
  configuration (`ImageWAMWorkload.libero()`, two 224x224 views, 512 text
  tokens, horizon 64) and the target (`TARGET_WORKLOAD`: three views of
  256x256, instructions of 16-128 tokens, horizon 32, 10 steps, proprio 8).
  `benchmarks/imagewam_thor_path_bench.py --workload libero|target` measures
  either one on all three service paths.

Decided after the `0920` re-run (E1, `text_trim` as the served default):

- **`default` trims; FA4 and the native VAE stay in `fast`.** The option is
  the owner's, on the measured basis: trimming is the largest single step
  (LIBERO `infer()` about 202 ms untrimmed against about 115 ms trimmed, and
  agreement with official improves rather than degrades on every suite), while
  FA4 and the native VAE pass their criteria with smaller margins and change
  more (FA4 compiles on first use and can fall back; the in-graph VAE changes
  the graph). `fast` keeps the three together.
- **The `native` profile** carries the native consumer's set: the served
  `default`'s switches with `use_fa4` stated off (the native pipeline has no
  FA4 attention) and everything else equal to `default`. It was the way around
  R5 until S4; R5 is gone, so it is the name for that consumer's
  configuration. The served `default` is trim + FA4 left to the machine.
- **FA4 is part of the served default, as auto**: `FLASHRT_THOR_FA4`'s default
  is now `"1"`, so `use_fa4=None` means FA4 wherever `thor_default_enabled()`
  holds (compute capability 11.x plus an importable FA4 runtime) and the cuBLAS
  chain elsewhere, never raising for a missing runtime. `FLASHRT_THOR_FA4=0`
  forces the chain and an explicit `use_fa4` still wins. The ActionDiT site
  (`use_fa4_mot`) stays off. This is the "fastest configuration that always
  constructs": FA4 and the in-graph native VAE both measured faster, but FA4
  can be absent and the in-graph VAE needs `ae_model_path` (rule R3), so the
  latter stays in `fast`.
- **The regression gate's defaults follow the served configuration**: fixture
  v2 and trimming on, with `--no-text-trim` (plus the v1 manifest) selecting
  the untrimmed reference, which `scripts/imagewam_thor_validation.sh`'s rows
  now state explicitly so their recorded numbers keep their meaning.
- The profiles therefore mean **the served configuration**, while the
  constructor keeps its own historical defaults (untrimmed) for a caller that
  passes dims by hand. That divergence is deliberate.
- Consequences recorded rather than solved: the configuration matrix's flag
  rows now state their trim explicitly (otherwise the `default` row would
  silently become the `vae_trim` row). Superseded for the gate by F2: the
  regression gate now builds through `load_imagewam` and its baseline is per
  configuration.

Answered in S4: the native pipeline takes option (a), one pipeline install per
length with the handle's owned graphs surviving `set_pipeline` (a per-key
pipeline table in `native_runtime.{h,cpp}`). Rule R5 is gone; the `native`
profile remains the name for that consumer's set because it is the only named
set that turns FA4 off.

Decided (owner, 0921):

- **The served default is the fastest configuration**, so a run on the device
  needs the fewest changes (F1). Profile `default` is nvfp4 + `text_trim` +
  FA4 at the backbone and the `mot` site + the native VAE encoder inside the
  graph, where "auto" degrades instead of raising: FA4 at either site is used
  where the machine can run it and falls back to the cuBLAS chain at capture
  time otherwise; the VAE goes native and into the graph exactly when an
  autoencoder path is given (without one there is no VAE stage). Both
  criteria of the `c20f3a0` round were met for the two promoted switches.
  The three risks that came with promoting them (FA4's first-call compile,
  an FA4 capture failure, the in-graph VAE's fixed input shape) are what
  the F1 Thor rows observe. `fast` stays as the same switches stated
  explicitly (it raises where `default` degrades); `native` stays as the
  native consumer's set, and `default` with `consumer="native"` resolves to
  it.
- **The latency baseline is per configuration** (F2): the gate names the
  configuration a run resolved to and compares against that configuration's
  entry; the untrimmed record keeps its own entry, and the served default's
  entry is seeded from its own gate run.
- **The calibration file's identity carries the workload's camera geometry**
  (`num_views`, `image_h`, `image_w`) (F3): format version 3, no reader for
  earlier versions, the bundle files re-recorded.

Open:

One decision, not a Thor run. Phases F1-F3 are code-complete and wait for
their Thor rows in `THOR_CHECKLIST.md`; the target workload below is the only
question left for the owner.

1. The target-workload declaration (`TARGET_WORKLOAD` in
   `benchmarks/_imagewam_workload_cli.py`: `num_views=3`, `image_h=image_w=256`,
   `action_horizon=32`, `action_dim=7`, `proprio_dim=8`, `num_steps=10`,
   `shift=5.0`, instruction tokens 16-128). The workload serves on all three
   paths: `0919e` measured `infer()` 216.93 / ABI 173.55 / native 173.27 ms
   with `default`, and a trimmed sweep at 16, 72 and 128 valid tokens measured
   197.00 / 207.06 / 217.50 ms on `infer()` and 153.51 / 161.68 / 172.05 ms on
   the ABI. Still open: whether 128 padded tokens is the deployment's own
   count (ISSUE-083 resolves how it is encoded, not what it should be), and
   the checkpoint, calibration file and graph memory budget that go with that
   workload — the per-length memory figures recorded so far are LIBERO's.
   With those settled the workload has latency evidence but no fidelity
   evidence: every `vs official` number recorded so far is LIBERO's.


# Plan: the final result tables (LIBERO and RoboTwin standard configurations)

Plan Status: R1 and R3 completed; the LIBERO table exists (`docs/imagewam_results.md`,
session `0921d_final`, int8 and int4 rows left blank by decision). R2 (the RoboTwin
declaration), R4 for RoboTwin and the RoboTwin session are pending the owner's
declaration.

## Problem

### Current

The project's numbers are spread over sessions and documents, and the two
ends of the comparison are not produced by one protocol:

- The official torch baseline (bf16 eager, 453.6 ms on Thor at LIBERO) was
  measured in another session; no script in the repository times the official
  implementation (`benchmarks/imagewam_e2e_official_compare.py` builds it only
  to compare actions).
- FlashRT's own rows come from different rounds (`eccf14f`, `c20f3a0`,
  `0920t`, ...), several of them in machine states that differ by more than the
  effects being compared (ISSUE-082), so a ratio across rounds is not a result.
- The frontend has no int8 or int4 tier. The int8 and int4 numbers come from
  `benchmarks/imagewam_thor_int8_bench.py` / `imagewam_thor_int4_bench.py`,
  which time the GEMMs with random packed operands and no activation
  quantization (an upper bound, OPT-007); on Thor int4 has no tensor-core path
  (`tcgen05.mma` has no integer 4-bit kind) and measured slower than fp16.
- FlashRT's non-quantized tier is `fp16`; the official model runs bf16. There
  is no bf16 GEMM tier in the frontend.
- The RoboTwin workload is not declared anywhere in the repository (the only
  checkpoint here is the LIBERO one), and the benchmarks that carry a workload
  are LIBERO-shaped (`_imagewam_workload_cli.WORKLOADS`: `libero`, `target`).
  RoboTwin runs 30 denoise steps against LIBERO's 10.

### Problem

There is no single, formatted record of "official torch vs FlashRT at
fp16 / fp8 / fp4 / int8 / int4, steady state, on each standard workload".

### Measurable goal

Two tables, one per workload, each row measured in ONE Thor session with the
same timed-call boundary, recorded in one JSON document and rendered from it:

| Row | What runs |
|---|---|
| `official_torch` | the official implementation, bf16 eager |
| `flashrt_fp16` | `precision="fp16"` |
| `flashrt_fp8` | `precision="fp8_static_cutlass"` (real calibration for that workload) |
| `flashrt_fp4` | `precision="nvfp4"` |
| `flashrt_int8` | the SM80 INT8 CUTLASS bench, scope `gemm_only` |
| `flashrt_int4` | the SM80 INT4 CUTLASS bench, scope `gemm_only` |

The timed call is camera frames + proprio (text context already encoded) to the
de-normalised action chunk on the host, in steady state. FlashRT rows use the
served `default` profile of `load_imagewam` (the fastest configuration) so a
row is "what a deployment runs", with `effective_config` recorded per row.

## Structure

| Module | Responsibility | State it owns |
|---|---|---|
| `benchmarks/imagewam_result_table.py` (R1) | the schema, the checker, the renderer | the row set, the standard step counts, the rules |
| `docs/imagewam_results.json` | the record set | every measured number, its session, its configuration |
| `docs/imagewam_results.md` | the rendered tables | nothing: generated, never edited |
| workload presets (`benchmarks/_imagewam_workload_cli.py`, R2) | the named workloads the benches run | the `robotwin` workload once declared |
| official bench (R3) | the official implementation's steady-state latency | nothing |
| int8 / int4 benches (R4) | GEMM-only timings at the workload's shapes | nothing |
| results driver (R5) | one session: every row of a table, then the import into the JSON | the session id |

Ownership rules: the JSON is the only place a table's numbers live; the
Markdown is derived; a workload's fields are declared once (the table's
`workload` block equals the preset the benches ran); a ratio is only computed
by the renderer, and only inside one session.

## Interface

Schema `schema_version` 1 (`benchmarks/imagewam_result_table.py`, whose module
docstring is the field list). Table level: `workload`, `checkpoint`,
`boundary`, `session`, `measurement` (device, commit, date, `gpu_exclusive`,
clock state, warmup, iters), `rows`. Row level: `id`, `status`
(`measured` / `not_measured` / `not_supported`), `scope` (`full_infer` /
`gemm_only`), `reason`, `session`, `config` (`effective_config`,
calibration), `latency` (p10 / p50 / p90 / n), `fidelity` (source, cosine vs
official median and min, MAE vs ground truth, n) or null, `note`.

Commands: `skeleton`, `check <json>`, `render <json>`.

Rules (checked): the six rows once, in order; `num_steps` equals the table's
standard (libero 10, robotwin 30); a measured row has ordered percentiles and
n >= 1, a complete workload, a measurement block and an identified checkpoint;
a FlashRT row records its `effective_config`; a `gemm_only` row has no
fidelity and no ratio (marked †); a ratio against the official row exists only
when both rows carry the same session (otherwise ‡).

## Flow

```
declare the workload (R2) -> one Thor session:
   official bench (R3) -> FlashRT fp16, fp8, fp4 through load_imagewam (R5)
   -> int8, int4 GEMM benches (R4)
 -> importer fills docs/imagewam_results.json (rows, session, config, fidelity)
 -> check -> render -> docs/imagewam_results.md
```

## Code Mapping

| Item | File |
|---|---|
| schema, checker, renderer, skeleton | `benchmarks/imagewam_result_table.py`, `tests/test_imagewam_result_table.py`, `docs/imagewam_results.json`, `docs/imagewam_results.md` |
| `robotwin` workload, `VALID_TOKENS` entry | `benchmarks/_imagewam_workload_cli.py` |
| official steady-state bench | `benchmarks/imagewam_official_torch_bench.py` (new) |
| int8 / int4 at a workload and step count | `benchmarks/imagewam_thor_int8_bench.py`, `imagewam_thor_int4_bench.py`, `imagewam_thor_graph_bench.py` |
| FlashRT rows | `benchmarks/imagewam_thor_path_bench.py` (latency, `--workload`, `--profile default`), `benchmarks/imagewam_e2e_official_compare.py` + `scripts/imagewam_thor_matrix.sh` (latency and fidelity where data exists) |
| fp8 calibration for a workload | `benchmarks/imagewam_build_calibration.py` (LIBERO data loader today) |
| session driver and importer | `scripts/imagewam_thor_results.sh`, `benchmarks/imagewam_result_table.py import` (new) |

## Implementation Phases

### Phase R1: the schema, checker and renderer
Phase Status: completed
- Goal: the record format and its rules exist and are tested before any
  number is entered.
- Modified files: `benchmarks/imagewam_result_table.py`,
  `tests/test_imagewam_result_table.py`, `docs/imagewam_results.json` (the
  skeleton: every row `not_measured`), `docs/imagewam_results.md`.
- Observation: `tests/test_imagewam_result_table.py` (CPU, 13 tests).

### Phase R2: the RoboTwin workload
Phase Status: pending (owner input)
- Goal: `robotwin` is a named workload with `num_steps=30`.
- Needs from the owner (none of it is in the repository): camera count and
  image size, valid instruction token range and the padded length,
  `action_horizon`, `action_dim`, `proprio_dim`, `shift`, the RoboTwin
  checkpoint (its backbone widths must match; `ImageWAMStructure.from_checkpoint`
  reads them) and where its observation data lives.
- Modified files: `benchmarks/_imagewam_workload_cli.py`, the `robotwin`
  `workload` block of `docs/imagewam_results.json`.

### Phase R3: the official torch steady-state bench
Phase Status: completed (code); its Thor run is THOR_CHECKLIST.md L1
- Goal: a script times the official implementation at a workload and step
  count over exactly the timed call above (image encode + proprio + transformer
  with the text context passed in), with warmup, device sync and percentiles.
  It replaces the one-off 453.6 ms.
- Modified files: `benchmarks/imagewam_official_torch_bench.py` (new). It takes the
  workload flags of `_imagewam_workload_cli.py`, so the step count and shape of
  every table come from the named workload. `benchmarks/imagewam_thor_path_bench.py`
  gained `--calibration` so the fp8 row can run through it.

### Phase R4: int8 and int4 at a workload
Phase Status: pending for RoboTwin; the LIBERO row needs nothing (the benches
are written for the LIBERO dims and run unchanged, THOR_CHECKLIST.md L4)
- Goal: the two benches take `--workload` and the step count, and emit the
  same latency record; their rows stay `gemm_only`.
- Open decision: whether a real full-pipeline INT8 tier (activation
  quantization inside the frontend; INT8 has a `tcgen05` legacy path on Thor)
  is worth building so the int8 row can be `full_infer`. int4 cannot be
  (no hardware path, OPT-007), so its row is `gemm_only` either way.
- Modified files: `benchmarks/imagewam_thor_int8_bench.py`,
  `imagewam_thor_int4_bench.py`, `imagewam_thor_graph_bench.py`.

### Phase R5: the session driver and importer
Phase Status: decided against
- Decision: no driver script and no log importer. The Thor has no push
  credentials and its reports come back as text, so the checklist's L rows are
  the driver (one session, fixed log names, the numbers to report) and the
  credentialed side enters them into the JSON, where `check` and `render` guard
  the format. A log parser over five scripts' different outputs would be
  the part most likely to be wrong.
- One schema consequence: the int8 and int4 benches print the prefill and one
  denoise step, not percentiles of the composed call, so a `gemm_only` row
  records `latency.p50_ms` and `components` (prefill, step, steps) with
  p50 = prefill + steps x step, checked.

### Phase R6: the Thor session
Phase Status: pending
- `THOR_CHECKLIST.md` L0-L6 is the LIBERO session (with a bracketing repeat of
  the official and fp4 rows to bound drift inside the session); the RoboTwin
  session is written once R2 is answered. Points it must state: fp8 needs
  a calibration recorded on the workload's own data (the identity carries
  `num_denoise_steps`, `shift` and the camera geometry); 30 steps put about
  three times the ActionDiT kernels in one graph, so the capture time and the
  graph memory are observed on the RoboTwin rows; the official row runs at 30
  steps too.


# Plan: wire OPT-032's kernel-fusion candidates into the served pipeline

Plan Status: proposed

## Problem

### Current

opportunities.md OPT-032 has four kernel-fusion candidates authored and
bit-exact tested (candidate 1: fused QKV split + QK-RMSNorm + RoPE;
candidate 3: fused gated-residual + AdaLN emitting NVFP4 + SFA directly;
candidate 6: fused Euler-step + next-step cast; candidate 8: last
backbone block computing only its exported K/V), none wired into
`pipeline_thor.py`, `csrc/bindings.cpp` or `CMakeLists.txt`. Candidate 4
(the vectorized activation quantizer) is wired
(`flash_rt/executors/fp4_utils.py`).

### Problem

Candidates 1, 3, 8 all touch the same four per-layer functions in
`pipeline_thor.py` (`_double_stream_layer`, `_single_stream_layer`,
`_action_double_layer`, `_action_single_layer`), so wiring them is
inherently serial, one owner, one fusion at a time -- the OPT-015/016/017
precedent this project already follows for exactly this reason. This
machine cannot compile or run the production `flash_rt_kernels`
extension (no working local build), so every wired change here is
validated by a CPU-only structural/compile check plus a NEW Thor-only
bit-exact test added alongside it; the actual pass/fail is a Thor A/B,
not a local one.

### Measurable goal

Each candidate lands behind its own `dims` flag, default off (opt-in
until Thor confirms it, matching `text_trim`/`gemm_variant_autotune`/
`nvfp4_awq`'s own rollout), with the flag off producing byte-identical
code to before (regression-proof) and the flag on producing a
`torch.equal` match against the unfused path at real production shapes
(not cosine -- the kernels are already proven bit-exact in isolation, so
anything short of bit-exact in the WIRED path is an argument/offset bug,
not a numerics difference).

## Structure

| Module | Responsibility |
|---|---|
| `csrc/kernels/fused_qkv_norm_rope/`, `fused_norm_fp4/`, `fused_step_boundary/` | the fused kernels (opportunities.md OPT-032), unowned by this plan -- already authored and tested |
| `CMakeLists.txt`, `csrc/bindings.cpp`, `csrc/kernels/kernels.h` | production build/binding surface: one entry per fused kernel, additive only |
| `flash_rt/models/imagewam/pipeline_thor.py` | the four per-layer functions; each fusion flag is read once per function and both branches (fused/unfused) call the SAME downstream code |
| `tests/test_imagewam_thor_real_wiring.py` | the Thor-only bit-exact A/B per fusion, at real dims, added alongside the wiring |

## Interface

`dims` flags (read via `dims.get(...)`, default `False`, not yet exposed
as a constructor kwarg or `config_resolver.py` expert key -- only
`dims_override` until Thor confirms a candidate is worth promoting):

- `fuse_qkv_norm_rope` (candidate 1): read in `_double_stream_layer`,
  `_single_stream_layer`, `_action_double_layer`, `_action_single_layer`.
  Calls `fvk.qkv_split_norm_rope_fp16(qkv, q_norm, k_norm, rope_table,
  Q_out, K_out, V_out, rows, NH, HD, hidden, src_row_stride,
  q_col_offset, k_col_offset, v_col_offset, dst_row_stride, eps, stream)`
  in place of the 3 `_copy_slice` + 2 `rms_norm_fp16` + 2
  `rope_apply_fp16_perhead` calls it replaces. Double-stream calls it
  TWICE (once per txt/img stream, each with its own row-offset
  `rope_table` pointer into the shared table and its own row range of
  the shared `Q_O`/`K_cache`/`V_cache` destination) -- proven equivalent
  to the reference's single joint-RoPE call by the kernel's own authoring
  test (`test_double_stream_split_call_matches_joint_rope`).
- `fuse_ada_norm_fp4` (candidate 3, not yet wired): will replace
  `_fused_gate_res` + the following `Nvfp4Linear.__call__`'s internal
  quantize with one call; needs a new `Nvfp4Linear` method that accepts
  pre-quantized (packed, sfa) scratch instead of an fp16 input, since the
  fusion kernel now produces the GEMM's activand quantized operand
  directly (`quant_linear.py`, not yet designed).
- `fuse_step_boundary` (candidate 6, low priority): will replace the
  per-step trailing `gpu_euler_step` + the next step's leading
  `gpu_cast_fp32_to_fp16` with one call, in `imagewam_denoise_step`/
  `imagewam_denoise_loop`; measured gain is small (0.03-0.09 ms total,
  opportunities.md OPT-032), deprioritized.
- candidate 8 (last-block K/V-only) is not a per-layer flag: it is a
  new function called from `imagewam_prefill`'s loop for exactly
  `site_layer_idx == num_layers_single - 1` (the bijection
  `attn_backend.py:get_slot_ptrs` established, `docs/imagewam_last_block_kv_only.md`),
  a different, non-overlapping integration point from 1/3's per-layer flags.

## Flow

```
pipeline_thor.py's per-layer functions (unchanged call sites, dims-gated):
  dims.get("fuse_qkv_norm_rope") -> fvk.qkv_split_norm_rope_fp16(...)
                                     or the 7-call unfused sequence
  (candidate 3, 6, 8: same pattern, not yet wired)
        |
tests/test_imagewam_thor_real_wiring.py (Thor only): flag off vs on,
  same inputs, torch.equal on the layer's output
        |
Thor rebuild (ENABLE_SM100_CUTLASS) -> the new test -> the existing
  imagewam_fusion_ab.py-style whole-graph A/B -> THOR_CHECKLIST.md X4/X5
  for the measured kernel-count and latency win
```

## Code Mapping

| Task | Files |
|---|---|
| Candidate 1 wiring (single-stream, backbone + ActionDiT) | `flash_rt/models/imagewam/pipeline_thor.py` (`_single_stream_layer`, `_action_single_layer`), `csrc/bindings.cpp`, `CMakeLists.txt`, `csrc/kernels/kernels.h`, `tests/test_imagewam_thor_real_wiring.py` |
| Candidate 1 wiring (double-stream) | `pipeline_thor.py` (`_double_stream_layer`, `_action_double_layer`) |
| Candidate 3 wiring | `pipeline_thor.py` (`_fused_gate_res` and its call sites), `quant_linear.py` (`Nvfp4Linear` pre-quantized-input method) |
| Candidate 8 wiring | `pipeline_thor.py` (`imagewam_prefill`'s loop, a new last-layer function per `docs/imagewam_last_block_kv_only.md`) |
| Candidate 6 wiring | `pipeline_thor.py` (`imagewam_denoise_step`, `imagewam_denoise_loop`); low priority |

## Implementation Phases

### Phase 1: candidate 1, single-stream (backbone + ActionDiT)
Phase Status: completed, Thor-confirmed (`0922d`: `1 passed`, all three comparisons `torch.equal`/`bit_exact=True`, `max_abs=0`)
- Goal: `fuse_qkv_norm_rope` wired into `_single_stream_layer` and
  `_action_single_layer`, both the `merge_qkv_mlp` and split `qkv.weight`
  branches; the shared post-branch RMSNorm+RoPE block is skipped when the
  flag is on.
- Modified files: `CMakeLists.txt` (new kernel source), `csrc/kernels/kernels.h`
  (new header), `csrc/bindings.cpp` (new binding, additive), `pipeline_thor.py`,
  `tests/test_imagewam_thor_real_wiring.py` (new bit-exact A/B test at real
  dims, both `merge_qkv_mlp` values).
- Observation: local CPU regression suite unaffected (`py_compile`,
  `tests/test_imagewam_thor_precision_routing.py` and the config-resolver/
  workload/structure/precision-table tests, 205 passed).
- Thor (`0922`, `thor_val/0922`): rebuild succeeded (`qkv_split_norm_rope_fp16`
  symbol present), but the new wiring test FAILED the bit-exact check
  (`cos=0.9993150 max_abs=1.625`) -- traced to a bug in the TEST itself, not
  (as far as this trace shows) the wiring: `torch.manual_seed` was reset too
  late in `run_single_stream`/`run_action_single`, after some weights/inputs
  were already drawn, so the two compared runs did not share identical
  inputs. Fixed (seed reset moved to the top of each helper, before any
  draw); also renamed the test so `pytest -k fuse_qkv_norm_rope` actually
  collects it (it did not, `0922`: the name had "fused", not "fuse").
- Next: Thor rerun of the fixed test (`python -m pytest
  tests/test_imagewam_thor_real_wiring.py -k fuse_qkv_norm_rope -q -s`) is
  the real bit-exact confirmation, still pending; then the whole-graph
  `imagewam_fusion_ab.py`-style A/B (its own `FLAGS` do not yet include this
  one, `0922`) and THOR_CHECKLIST.md X4/X5 re-run for the measured
  kernel-count and latency change. X5 ran ahead of the wiring confirmation
  (`0922`, without the flag): backbone GEMM is near compute-bound
  (bb.single 164.9 TFLOPs, bb.double 82.3), ActionDiT GEMM is not
  (act.single 32.6 TFLOPs, act.double 24.1) and carries a high short-kernel
  share (25-35% under 25us) -- consistent with candidate 1 (targets the
  small per-row kernels, not the GEMMs) mattering more on the ActionDiT
  side.
- Thor (`0922b`): with the seed bug fixed, the test crashed differently --
  `_action_single_layer(fuse_qkv_norm_rope=True)` hit an illegal memory
  access. Reproduced locally (Ada) and diagnosed precisely (issues.md
  ISSUE-090): NOT a bug in the kernel or the wiring -- a pre-existing
  `GemmRunner` fragility, reproduced identically with the flag OFF on both
  compared calls, triggered by the test's own pattern of rebuilding fresh
  `Fp16Linear` weights for each of the two compared calls against one
  shared `GemmRunner`. Fixed by building weights/attention/input ONCE and
  replaying both `fuse_qkv_norm_rope` values against them (matching
  `test_single_stream_linear2_merged_vs_split_real_shapes`'s own established
  pattern). Verified locally, 3 stable runs: bit-exact for backbone (both
  `merge_qkv_mlp` values) and ActionDiT. No known production code path hits
  ISSUE-090's trigger (weights are built once at frontend construction,
  never rebuilt during `infer()`), so it is not a blocker, only noted.
- Thor (`0922d`, HEAD `e727f91`): rerun of the corrected test, `1 passed,
  6 deselected, 1.40s`. All three comparisons `torch.equal`/`bit_exact=True`,
  `max_abs=0`: backbone `merge_qkv_mlp=True`, backbone `merge_qkv_mlp=False`,
  ActionDiT single. MAXN, `emc_locked=null`, GPU idle, no rebuild needed.
  Phase 1 is fully confirmed; candidate 1's single-stream wiring is closed.

### Phase 2: candidate 1, double-stream
Phase Status: completed, Thor-confirmed (`0922e`: `2 passed`, all five comparisons `torch.equal`/`bit_exact=True`, `max_abs=0`, including backbone double-stream and ActionDiT double)
- Goal: the same flag in `_double_stream_layer`/`_action_double_layer`,
  two calls per block (txt, img), each with its own row-offset RoPE
  table pointer and destination row range.
- Modified files: `pipeline_thor.py`, `tests/test_imagewam_thor_real_wiring.py`.
- `_double_stream_layer`: `fused_qkv` calls `fvk.qkv_split_norm_rope_fp16`
  TWICE, once for the txt rows (rope_table at its base pointer) and once
  for the img rows (rope_table offset by `x0` rows via `_ptr_offset`),
  replacing each stream's own 3-copy+2-rms_norm and skipping the shared
  joint `rope_apply_fp16_perhead` pair that used to follow both streams.
  `_action_double_layer` is a single-call site, same pattern as
  `_action_single_layer`'s non-merged branch.
- New test: `test_double_stream_and_action_double_fuse_qkv_norm_rope_bit_exact_at_real_shapes`,
  same build-once/replay-both-values structure as Phase 1's test
  (issues.md ISSUE-090). Verified locally (Ada, JIT-bound kernel onto
  `flash_rt.flash_rt_kernels`), stable over 3 runs: `torch.equal`,
  `max_abs=0`, for both the backbone double-stream layer and the
  ActionDiT double layer.
- A real bug surfaced and was fixed during this verification, but it
  was in the NEW TEST's own weight construction, not the wiring: the
  ActionDiT double layer's `proj.weight` `Fp16Linear` was built with
  its `n`/`k` arguments swapped (`Fp16Linear(gemm, proj_w.data_ptr(),
  aaw, ahd)` instead of `(..., ahd, aaw)`), so `key("proj.weight")`
  ran a GEMM with the wrong output width, overrunning the
  `action_proj_scratch` buffer by (aaw-ahd) columns. `compute-sanitizer
  --tool memcheck` found 0 errors (the overrun stayed inside the same
  cudaMalloc'd allocator segment, so nothing looked illegal to it,
  matching ISSUE-090's own note about redzone padding not catching a
  marginal overrun) while a bit-exact fused-vs-unfused comparison
  still caught it, because the corruption differed between the two
  code paths depending on what else was allocated nearby.  Root-caused
  by tracing every `GemmRunner.fp16_nn` call's own (M,N,K) and a
  zero-copy snapshot of its output buffer for both compared runs
  side by side, which isolated the divergence to this one GEMM.
  Fixed by swapping the two arguments back; re-verified bit-exact 3x.
- Thor (`0922e`, HEAD `c5898a2`): `2 passed, 6 deselected, 1.47s`. All
  five comparisons `torch.equal`/`bit_exact=True`, `max_abs=0`:
  backbone single `merge_qkv_mlp=True`/`False`, ActionDiT single (all
  three re-confirming Phase 1), backbone double-stream, ActionDiT
  double (Phase 2's own new coverage). MAXN, `emc_locked=null`, GPU
  idle, no rebuild needed. Phase 2 is fully confirmed.

### Phase 3: candidate 8, last backbone block K/V-only
Phase Status: completed, Thor-confirmed (`0922h`: `torch.equal`/
`bit_exact=True`, `max_abs=0` for both K and V)
- Goal: `imagewam_prefill`'s loop calls a new, narrower function for
  the LAST single-stream layer instead of the full
  `_single_stream_layer`, per `docs/imagewam_last_block_kv_only.md`'s
  design (already bit-exact/ULP-bounded tested at real dims on this dev
  machine, against the read-only, unmodified `_single_stream_layer`).
- Modified files, three: `checkpoint_loader.py` (`_extract_single_block`
  gets a `kv_only: bool` param; when set, ALSO returns
  `"linear1_kv.weight"`, a one-time column slice of the real, RAW
  `linear1.weight` tensor down to its K,V columns
  `[attn_dim, 3*attn_dim)`, on the (out,in) tensor before transpose --
  `"linear1.weight"` itself is still ALSO returned unconditionally, so
  the AWQ-fold / dev-path bookkeeping below has both keys to choose
  from; `build_real_weights` gets a `last_layer_kv_only` param, passing
  `kv_only=True` only for `L == num_single - 1`, backbone only --
  ActionDiT's blocks are unaffected, this candidate is backbone-only
  per the design doc's own scope); `imagewam_thor.py` (`_alloc_buffers`:
  new `"single_kv_merged"` buffer, `(a0, 2*hidden)`, always allocated;
  `_alloc_random_weights`: the dev-path counterpart, adding
  `"linear1_kv.weight"` for the last backbone single layer when
  `dims["last_layer_kv_only"]`; `_load_real_weights` needed NO change --
  its wrap loop is already generic over `raw.items()`, so the new key
  flows through `_wrap_linear` automatically); `pipeline_thor.py` (new
  function `_single_stream_layer_kv_only`; `imagewam_prefill`'s loop
  dispatches to it for the last single-stream layer when
  `dims["last_layer_kv_only"]`, and its own `single_linear1(i)` helper
  -- used for AWQ-fold A on the CROSS-LAYER `next_norm` target feeding
  the last layer -- now targets `"linear1_kv.weight"` instead of
  `"linear1.weight"` for that one layer, since that is the real
  consumer once this flag is on; getting this wrong would fold the
  AdaLN for a GEMM that never runs).
- Real bug found and fixed while verifying locally: the first draft of
  `_single_stream_layer_kv_only` dropped the `rope_apply_fp16_perhead`
  call on K (kept `rms_norm_fp16` but not the RoPE step that follows
  it in the design doc's own prototype) -- caught immediately by the
  new wiring test (`cos=0.7366`, partial mismatch, not the O(1e3)/NaN
  shape of a pointer-dtype bug like ISSUE-091, but still definitively
  wrong), fixed by adding the missing call and threading `rope_table`
  through the function's own signature and both call sites.
- New test: `tests/test_imagewam_thor_real_wiring.py::test_single_stream_kv_only_matches_full_block_at_real_shape`,
  comparing `_single_stream_layer_kv_only`'s K/V output against the
  full, unmodified `_single_stream_layer`'s own K/V, at real production
  shape (`a0=905`), `linear1_kv.weight` built as the SAME column slice
  of the SAME `linear1.weight` tensor (mirroring
  `checkpoint_loader.py`'s own slice exactly, not two independently
  drawn tensors). Runs for real on this Ada machine (only `Fp16Linear`,
  no NVFP4 needed) -- `torch.equal` for both K and V, stable over 3
  runs. Also a standalone CPU-only sanity check of
  `_extract_single_block(kv_only=True)`'s column-slice math against a
  synthetic state_dict (not added as a committed test, ad hoc during
  verification), confirming the K/V columns it extracts match the
  corresponding columns of the full `linear1.weight` exactly.
- Not attempted: AWQ interaction beyond the one fold-target fix above
  is unverified end-to-end (no test exercises `last_layer_kv_only`
  together with an active AWQ plan); the design doc's own correctness
  section only validates the plain FP16 path. NVFP4/FP8 precision for
  `linear1_kv.weight` itself also untested (the generic `_wrap_linear`
  path would pick whatever `self._precision` says, same as any other
  weight, but this specific weight's behavior under a quantized
  precision has not been separately checked).
- Thor (`0922h`, HEAD `11e8869`, no rebuild needed): `torch.equal`/
  `bit_exact=True`, `max_abs=0` for both K and V. MAXN,
  `emc_locked=null`, GPU idle. Phase 3 is fully confirmed.

### Phase 4: candidate 3, fused AdaLN + NVFP4 direct quantize
Phase Status: completed for Round 1's scope, Thor-confirmed (`0922g`:
`1 passed`, `torch.equal`/`bit_exact=True`, `max_abs=0`); Round 2
(remaining call sites) pending
- Goal: design the `Nvfp4Linear` pre-quantized-input method, then wire
  `_fused_gate_res`'s NVFP4 callers.
- Interface designed and implemented:
  - `Nvfp4Linear.gemm_prequantized(out_ptr, m, stream)` (`quant_linear.py`):
    runs the GEMM directly against `self.scratch`, skipping `_quant_act`.
    The caller must have already written valid `scratch.packed`/`.sfa`
    for `m` (a matching `_ensure_scratch(m)` plus an external write).
  - `AdaLNTarget` (`pipeline_thor.py`) gains an optional `lin` field:
    the GEMM object consuming this AdaLN's output. `_awq_target`
    (which already receives that object as its first argument) now
    always attaches it, AWQ-scaled or not.
  - `_fused_gate_res` gains an opt-in `fp4_direct: bool = False` kwarg.
    When `fp4_direct` and `isinstance(target.lin, Nvfp4Linear)`, it
    calls `target.lin`'s own `_fvk_fp4.gate_res_ada_layer_norm_fp4_sfa_bf16res`/
    `_fp16res` (`csrc/kernels/fused_norm_fp4/`), writing NVFP4+SFA
    straight into `target.lin.scratch`, instead of the plain FP16 AdaLN
    it otherwise writes to `target.out_ptr` -- no intermediate FP16
    buffer materialized for that consumer. `target.out_ptr` holds no
    valid data in this branch. Default off, so every existing caller
    (none of which pass `fp4_direct`) is unaffected even if `target.lin`
    happens to be an `Nvfp4Linear` -- the flag, not the object's type,
    gates the new behavior.
- Wiring, Round 1 scope: `_single_stream_layer`'s own `merge_qkv_mlp=True`
  `linear1.weight` consumer only (the real deployment default, the
  single biggest/widest GEMM in that function) -- `dims["fuse_res_norm_fp4"]`
  (opt-in, default off). When set and `input_normed` and the layer's
  own `linear1.weight` is an `Nvfp4Linear`, calls `.gemm_prequantized(...)`
  instead of `linear1(modded, ...)`; the tail `_fused_gate_res` call
  passes `fp4_direct=fp4_direct`. NOT yet wired: `_double_stream_layer`'s
  four targets, `_action_double_layer`/`_action_single_layer`'s targets,
  `imagewam_denoise_step`'s head target, and `_single_stream_layer`'s
  non-merged (`qkv.weight`/`mlp_in.weight`) branch -- Round 2, deferred,
  same incremental-expansion pattern Phase 1 (single-stream only) used
  before Phase 2 (double-stream).
- Modified files: `quant_linear.py`, `pipeline_thor.py`, `CMakeLists.txt`
  (added `csrc/kernels/fused_norm_fp4/fused_norm_fp4.cu` to the
  `ENABLE_SM100_CUTLASS`-gated `fp4_kernels_obj` sources -- this kernel
  was standalone/JIT-only before this phase), `csrc/fp4_bindings.cpp`
  (added the production `flash_rt_fp4` module bindings for
  `gate_res_ada_layer_norm_fp4_sfa_bf16res`/`_fp16res`, matching the
  JIT binding's own established arg order exactly).
- Validation: `tests/test_imagewam_fuse_res_norm_fp4_dispatch.py` (new,
  CPU-only, 8/8 passed locally): the dispatch/call-order contract in
  `_fused_gate_res` and `_awq_target`, using a real `Nvfp4Linear`
  instance built via `object.__new__` (so `isinstance` holds without
  running its Blackwell-only `__init__`, which this machine cannot).
  `tests/test_imagewam_thor_real_wiring.py::test_single_stream_fuse_res_norm_fp4_direct_bit_exact_at_real_shapes`
  (new): a real two-layer single-stream chain, `torch.equal` between
  `fuse_res_norm_fp4=False`/`True`, at real production shape -- cannot
  run on this dev machine (Ada, sm_89; `Nvfp4Linear` itself needs a
  Blackwell/Thor NVFP4 build, `pytest.importorskip("flash_rt.flash_rt_fp4")`),
  not yet run anywhere. This is the real confirmation still needed
  (THOR_CHECKLIST.md X10): the fused kernel's own numerics are already
  Thor-confirmed against the unfused pair (X6), so this test's job,
  like Phase 1/2's own wiring tests, is to catch a pointer/argument
  wiring bug, not a numerics one.
- Thor (`0922f`): the real wiring test found exactly that kind of bug --
  `fuse_res_norm_fp4=True` produced a BF16 residual with O(1e3)
  values/NaN, not the O(1) the unfused control path gave. Root-caused
  by reading the kernel's own C++ signature directly: `gate`/`scale`/
  `shift`/`inv_s` are `const __half*` there (confirmed against
  `tests/test_fused_norm_fp4_kernel.py`'s own FP16 reference
  construction), NOT the `const float*` `_mod_vec_ptr` supplies (that
  helper's contract matches the OLDER, still-live
  `gate_res_ada_layer_norm_bf16res`, which really does take FP32 and
  round internally -- a different kernel, a different contract).
  Fixed in `_fused_gate_res`'s `fp4_direct` branch: round `gate`/
  `scale`/`shift`/`awq_inv_s` to fresh FP16 tensors before passing
  their pointers (issues.md ISSUE-091 has the full account, including
  a regression test that decodes the actual bytes reaching the kernel
  call). Local CPU dispatch suite re-verified 9/9 passing after the
  fix.
- Thor (`0922g`, HEAD `3056b03`, no rebuild needed -- the kernel itself
  was unchanged, only the Python call site): `1 passed, 8 deselected,
  1.40s`. `torch.equal`/`bit_exact=True`, `max_abs=0` for the two-layer
  single-stream chain. MAXN, `emc_locked=null`, GPU idle. Phase 4
  Round 1 is fully confirmed.
- Round 2 (this same round, after X10): extended the identical
  `merge_qkv_mlp=True` -> `linear1.weight` pattern to
  `_action_single_layer` (ActionDiT's own single-stream chain).
  **Important documented gap, in BOTH Round 1 and Round 2**: the
  dispatch only covers the i>=1 same-function chain (weight_layer_idx
  i's tail write feeding i+1's own consumption, both inside ONE call
  to `_single_stream_layer`/`_action_single_layer`). Neither round
  covers the double->single BOUNDARY (weight_layer_idx=0,
  `input_normed=True` there too whenever `num_layers_double>0` /
  `action_num_layers_double>0` -- true for the real LIBERO structure --
  fed by `_double_stream_layer`'s/`_action_double_layer`'s own tail,
  neither of which is wired for `fp4_direct` yet). Turning
  `dims["fuse_res_norm_fp4"]` on for a real full-model run before
  those two functions are ALSO wired would read `linear1`'s scratch
  before anything populates it -- a real, silent-corruption-shaped
  risk, not just an unoptimized path, so this is called out explicitly
  in both functions' own code comments, not left implicit. This flag
  must stay off in any full-model configuration until that boundary is
  closed.
  New test: `test_action_single_fuse_res_norm_fp4_direct_bit_exact_at_real_shapes`,
  same construction as the backbone one, `pytest.importorskip`'d here
  (same reason). Local CPU dispatch tests unaffected (9/9); Thor
  confirmation: Thor (`0922h`, HEAD `11e8869`, no rebuild needed):
  `3 passed, 8 deselected, 1.46s`, all `torch.equal`/`bit_exact=True`,
  `max_abs=0` (X10's backbone chain re-confirmed, plus this new
  ActionDiT chain). Round 2 is fully confirmed.
- Round 3 (closing the double->single boundary, same round): attempted
  and REVERTED after finding a genuine kernel limitation, not a
  wiring mistake. `fused_norm_fp4.cu`'s `launch()` computes its CUTLASS
  tile-interleaved SFA layout fresh from `seq_len` every call
  (`tile_atom_to_shape_SFA(make_shape(seq_len, ...))`), with no
  row-offset/total-seq-len parameter. Candidate 1's kernel could be
  called twice (once for txt's x0 rows, once for img's img_len rows)
  to jointly populate one combined a0-row destination because its
  outputs (Q/K/V, plain row-major) tolerate an external row offset;
  this kernel's SFA output cannot -- two calls at the same base
  pointer would each compute their own small tile layout, not two
  slices of one larger tiled layout, corrupting `_single_stream_layer`'s
  single combined `linear1.weight` consumer. Closing this needs either
  a kernel change (an explicit row-offset parameter) or a
  consumer-side redesign (two separate GEMMs instead of Round 1's
  single combined one) -- real design work, not attempted (a Round 4).
  The attempted `_double_stream_layer` edits (consumption dispatch for
  `txt_qkv.weight`/`img_qkv.weight`, `fp4_direct` on the txt_gate2/
  img_gate2 tail calls) were written, found unsound before being run
  anywhere, and reverted in the same local pass -- nothing unsound was
  pushed or handed to Thor.

### Phase 5: candidate 6, step-boundary Euler+cast
Phase Status: pending, low priority
- Goal: fuse the trailing Euler step with the next step's leading cast
  (9 of 10 pairs; the last step has no following cast).
- Modified files: `pipeline_thor.py` (`imagewam_denoise_step`,
  `imagewam_denoise_loop`).
- Note: measured gain is small (0.03-0.09 ms total, OPT-032); do this
  only if Phases 1-4 leave capacity.

# Plan: ImageWAM on Jetson Orin (rtx_sm87), registration only

Plan Status: completed

## Problem

`ImageWAMTorchFrontendThor` was constructible on Thor (`thor`) and RTX
5090 (`rtx_sm120`, dispatched internally via `self._arch` inside
`_wrap_linear`), but `flash_rt.hardware`'s dispatch table had no
`("imagewam", "torch", "rtx_sm87")` entry and `_SM87_ALLOWED` did not
list `imagewam` -- construction on Orin was rejected at the API layer
before any model code ran. Goal: make the frontend constructible on
Orin, restricted to a precision that is actually safe there (`fp16`,
cuBLAS-only, no FP8/FP4 tensor cores needed) -- not a full Orin
precision-tier project (that is OPT-034, not part of this plan).

## Structure / Interface

No new module or class. `ImageWAMTorchFrontendThor` is reused as-is on
all three architectures; the per-arch difference stays inside the class
(`self._arch`, already the established pattern from the NVFP4-on-5090
work), not as separate frontend classes per hardware.

## Code Mapping

- `flash_rt/hardware/__init__.py`: added `("imagewam", "torch",
  "rtx_sm87")` to `_PIPELINE_MAP` (same class as `thor`/`rtx_sm120`) and
  to `_SM87_ALLOWED`; updated the ImageWAM dispatch comment (was
  "Thor only", now stale).
- `flash_rt/frontends/torch/imagewam_thor.py`: `__init__` raises
  `ValueError` immediately after `self._arch = detect_arch()` if
  `self._arch == "rtx_sm87"` and `self._precision != "fp16"` --
  `_wrap_linear` has no SM87 branch for any other precision, so every
  other precision would otherwise reach a kernel launch built for a
  different architecture instead of failing at construction.

## Implementation Phases

### Phase 1: dispatch registration + fp16-only guard
Phase Status: completed
- Verified: `flash_rt.hardware._PIPELINE_MAP[("imagewam", "torch",
  "rtx_sm87")]` resolves; `resolve_pipeline_class`/`load_model()` no
  longer reject this combination. Not verified on real Orin hardware
  (none available on this dev machine) -- the change is structural
  (dispatch table + an explicit precision guard), not a numerically
  validated Orin run. First real Orin construction should confirm `fp16`
  actually builds and runs end-to-end before anything is called
  "supported" in a deployment doc.

## Related, not done here

- OPT-034 (opportunities.md): a real Orin INT8 precision tier for
  ImageWAM, reusing the existing SM80-family INT8 CUTLASS kernel
  (OPT-007) the way Pi0.5's own Orin path already uses INT8 -- not
  started.
- ISSUE-093 (issues.md): `("imagewam", "torch", "rtx_sm120")` has never
  been in `_PIPELINE_MAP` either -- every RTX 5090 ImageWAM benchmark
  this project has run so far constructs `ImageWAMTorchFrontendThor`
  directly, bypassing `flash_rt.load_model()`. Found while fixing the
  Orin gap; not fixed here (out of this plan's scope, and nothing
  currently depends on the public API path for 5090).

# Plan: Pi0.5 NVFP4 encoder on RTX 5090 (SM120)

Plan Status: completed

## Problem

`Pi05TorchFrontendRtx` has no NVFP4 tier on RTX 5090: every large GEMM is
FP8 (cuBLASLt) or BF16, and `benchmark_results/pi05_flashrt_rtx5090_nvfp4.json`
is `blocked`. At the cross-hardware target shape (3 views, chunk 30, 10
steps, synthetic weights) the FP8 row is `infer()` P50 17.21 ms.

Measured on this RTX 5090 (synthetic weights, CUDA-graph replay of each
stage, median of 100 after 20 warmup):

| stage | FP8 GPU time |
|---|---:|
| vision (SigLIP, 27 layers) | 3.04 ms |
| encoder (Gemma-2B, 18 layers, M = 768 + prompt length) | 7.16 ms |
| decoder (Gemma-300M, 18 layers x 10 steps, M = chunk_size = 30) | 7.35 ms |

Per-GEMM time, us per launch, weights round-robined across distinct
copies inside one CUDA graph (not L2-resident); the SM120 NVFP4 kernels
are `fp4_w4a16_gemm_sm120_bf16out{,_widen,_pingpong}` (W4A4 despite the
name), activation pre-quantized:

| stage | GEMM | M | N | K | FP8 | NVFP4 plain | pingpong | widen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| decoder | qkv | 30 | 2560 | 1024 | 2.89 | 7.88 | 8.33 | 13.11 |
| decoder | o | 30 | 1024 | 2048 | 3.21 | 11.13 | 11.80 | 20.48 |
| decoder | gate_up | 30 | 8192 | 1024 | 6.95 | 8.42 | 8.85 | 13.46 |
| decoder | down | 30 | 1024 | 4096 | 4.25 | 18.06 | 19.16 | 35.01 |
| encoder | qkv | 788 | 2560 | 2048 | 17.71 | 12.89 | 13.29 | 20.83 |
| encoder | o | 788 | 2048 | 2048 | 13.43 | 12.24 | 12.39 | 20.69 |
| encoder | gate_up | 788 | 32768 | 2048 | 175.64 | 106.47 | 112.14 | 142.32 |
| encoder | down | 788 | 2048 | 16384 | 91.16 | 60.03 | 57.81 | 106.98 |

Unfused NVFP4 activation quantization (`quantize_bf16_to_nvfp4_swizzled`)
costs 6.5 us (K=2048) and 32.0 us (K=16384) at M=788.

Consequences:

- Decoder (M=30): every existing SM120 NVFP4 tile is slower than FP8,
  and the decoder's FP8 activation quantization is already fused into
  its norm kernels. An NVFP4 decoder with these kernels is predicted to
  add about 8 ms. Tile selection does not change this: `plain` is the
  fastest of the three at every decoder shape. The existing small-M
  kernel `fp4_w4a4_mma_sm120_warpsplit_mrows_bf16out` supports M <= 16
  only (2.89/3.41/5.17/4.49 us at M=16 for qkv/o/gate_up/down); M=30
  needs two launches that read the weight twice.
- Encoder (M~788): NVFP4 GEMMs total 189 us per layer vs FP8's 298 us,
  about 1.95 ms over 18 layers before activation-quantization overhead.
- Vision: `down` (K=4304) is rejected by all three tiles
  (`can_implement` status 11); the remaining shapes are within +-2 us of
  FP8. No vision change.

Goal: an opt-in NVFP4 encoder for `Pi05TorchFrontendRtx` on SM120, with
vision and decoder unchanged (FP8), and its measured `infer()` P50
against the 17.21 ms FP8 row. Numerical accuracy on real weights is not
a gate; synthetic weights are the measurement input.

## Structure

- `flash_rt/models/pi05/nvfp4_sm120.py` (new): the SM120 NVFP4 primitive
  used by the Pi0.5 RTX pipeline -- an `Nvfp4WeightSm120` record
  (packed FP4 weight, swizzled scale factors, global scale `alpha`, N, K),
  its one-time quantization from a `(K, N)` BF16 weight, an
  `Nvfp4ActBufferSm120` record (packed activation + swizzled scale
  factors for a fixed `(rows, K)`), and the GEMM dispatch by tile
  variant. Raw-pointer, BF16 in / BF16 out, matching the pipeline's
  existing calling convention.
- `flash_rt/frontends/torch/pi05_rtx.py`: owns the quantized encoder
  weights (`self._nvfp4_weights`), built once at construction.
- `flash_rt/models/pi05/pipeline_rtx.py`: owns the per-pipeline NVFP4
  activation buffers and the NVFP4 encoder layer.

## State Ownership

- NVFP4 encoder weights: `Pi05TorchFrontendRtx._nvfp4_weights`
  (`dict[str, Nvfp4WeightSm120]`, keyed like the FP8 weight names),
  shared read-only with every pipeline through `weights["nvfp4"]`.
- NVFP4 activation buffers: each `Pi05Pipeline` instance (sized by its
  own `encoder_seq_len`).
- The FP8 encoder weights for the four encoder projections are not
  created when the NVFP4 encoder is on; the vision projector stays FP8.

## Interface

- `Pi05TorchFrontendRtx.__init__(..., use_fp4_encoder: bool = False)`.
  Raises `ValueError` when combined with an unsupported configuration:
  a non-SM120 device or a build without the SM120 NVFP4 kernels,
  `use_fp8=False`, the BF16 fallback (`FVK_PI05_RTX_FORCE_BF16` or no
  FP8 support), or any INT8 mode (`FVK_PI05_RTX_FORCE_INT8`,
  `FVK_PI05_RTX_INT8_ENCODER_ONLY`). `set_rl_mode(cfg_enable=True)` and
  `set_batched_mode(enable=True)` raise `ValueError` when it is on.
- `Pi05Pipeline.__init__(..., use_fp4_encoder: bool = False)`, passed
  through `_pipeline_precision_kwargs()`; requires `weights["nvfp4"]`.
- Tile per encoder GEMM is fixed from the measurement above: `pingpong`
  for `down`, `plain` for `qkv`/`o`/`gate_up`.
- `flash_rt.load_model()` routing is not part of this plan.

## Flow

Construction (`use_fp4_encoder=True`): load BF16 checkpoint -> FP8
quantize vision, vision projector and decoder -> NVFP4 quantize encoder
`qkv`/`o`/`gate_up` (gate and up concatenated, N = 32768)/`down` for all
18 layers. `set_prompt()` builds a pipeline that allocates its NVFP4
activation buffers. Calibration runs the pipeline eagerly once (the
NVFP4 encoder needs no scales; the eager run also creates the CUTLASS
per-shape workspace before capture) -> FP8 autotune skips the encoder
shapes -> graph capture.

Encoder layer `i` (NVFP4):
1. RMSNorm(x) -> NVFP4 activation -> qkv GEMM -> QKV split + RoPE.
   (Last layer stops here, as in the FP8 path.)
2. Attention -> NVFP4-quantize attention output -> o GEMM -> `x_norm`.
3. Residual add + RMSNorm -> NVFP4 activation -> gate_up GEMM.
4. GeGLU (merged) -> NVFP4-quantize hidden -> down GEMM -> `x_norm`.
5. Residual add.

## Code Mapping

- `flash_rt/models/pi05/nvfp4_sm120.py`: `Nvfp4WeightSm120`,
  `Nvfp4ActBufferSm120`, `quantize_weight_kn`, `allocate_act_buffer`,
  `gemm_bf16out`.
- `flash_rt/frontends/torch/pi05_rtx.py`: `use_fp4_encoder` validation,
  `_quantize_encoder_nvfp4()`, `_quantize_all_fp8()` skipping the
  encoder projections, `weights["nvfp4"]`, `_pipeline_precision_kwargs()`,
  guards in `set_rl_mode`/`set_batched_mode`.
- `flash_rt/models/pi05/pipeline_rtx.py`: `use_fp4_encoder`,
  `_allocate_fp4_encoder_scratch()`, `_encoder_layer_fp4()`, dispatch
  from `_encoder_layer`, encoder FP8 autotune skipped.
- `tests/test_pi05_rtx_fp4_encoder.py` (new): GPU-free dispatch and
  validation tests.
- `tests/test_pi05_rtx_autotune_scratch.py`: fixture gains
  `use_fp4_encoder = False`.
- `benchmark_results/pi05_flashrt_rtx5090_nvfp4.json`,
  `docs/pi05_imagewam_benchmark_status.md`, `opportunities.md`: results.

## Implementation Phases

### Phase 1: NVFP4 encoder, unfused activation quantization
Phase Status: completed

- Result (RTX 5090, synthetic weights, prompt of 13 tokens so
  `encoder_seq_len` = 781; SM clock not locked, 2.40-2.87 GHz observed,
  so only same-process interleaved A/B numbers are compared): encoder
  stage 7.19 -> 5.70 ms; full-graph replay P50 16.92 -> 15.79 ms;
  `infer()` P50 17.54 -> 16.42 ms (FP8 -> NVFP4 encoder). Cosine of the
  32-dim raw actions vs FP8 under identical noise: 0.99947. Peak
  allocated memory 9.02 -> 8.22 GiB. `tests/test_pi05_rtx_fp4_encoder.py`
  and `tests/test_pi05_rtx_autotune_scratch.py`: 20 passed.

- Steps 1-5 of the Flow above, each activation quantized by
  `quantize_bf16_to_nvfp4_swizzled` into the pipeline's NVFP4 buffer.
- Validation: GPU-free dispatch test (kernel calls, pointers, shapes,
  variants per GEMM; rejected configurations raise); on this RTX 5090
  with synthetic weights: finite `infer()` output, cosine of actions vs
  the FP8 frontend (recorded, not gated), encoder stage GPU time, and
  `infer()` P50 (CUDA event, warmup 20 / iters 100) vs FP8 17.21 ms.

### Phase 2: fused RMSNorm -> NVFP4 activation
Phase Status: completed

- Step 1 uses `rms_norm_to_nvfp4_swizzled_bf16` (layer 0) or
  `residual_add_rms_norm_to_nvfp4_swizzled_bf16` (layers 1-17, folding
  the previous layer's step-5 residual, in place); step 3 uses
  `residual_add_rms_norm_to_nvfp4_swizzled_bf16`. Both compute
  `x * rsqrt(mean(x^2) + eps) * w` per row, the same math as `rms_norm`
  with the pipeline's all-ones weight. Steps 2 and 4 stay unfused (no
  GeGLU -> NVFP4 kernel exists).
- Validation: same measurements as Phase 1, reported side by side.
- Result: the fused kernels' packed FP4 and scale-factor bytes are
  identical to `rms_norm` + `quantize_bf16_to_nvfp4_swizzled` (and the
  in-place residual output identical to `residual_add`) at M=781,
  D=2048. Per call 11.24 -> 8.69 us (residual variant; its `_v2` measured
  9.49 us and is not used). Encoder stage, round robin in one process:
  FP8 6.99 ms, NVFP4 unfused 5.77 ms, NVFP4 fused 5.70 ms.
  `infer()` vs FP8 in the same process: 18.25 -> 17.35 and
  18.46 -> 17.95 ms (P50, two runs; SM clock unlocked). 21 tests passed.

### Phase 3: results
Phase Status: completed

- Update `benchmark_results/pi05_flashrt_rtx5090_nvfp4.json`,
  `docs/pi05_imagewam_benchmark_status.md` and `opportunities.md`
  (decoder small-M NVFP4 kernel as a new entry, with the measurements
  above).
- Result: `benchmark_results/pi05_flashrt_rtx5090_nvfp4.json` is `ok`
  (P50 18.21 ms, fp8 in the same process 19.03 ms);
  `docs/pi05_imagewam_benchmark_status.md` has the interleaved table;
  `opportunities.md` OPT-036 records the decoder measurements.

## Related, not in this plan

- An M <= 32 NVFP4 kernel for the decoder (a two-atom version of
  `warpsplit_mrows`, reading each weight once): `opportunities.md`
  OPT-036.
- `flash_rt.load_model()` routing for the RTX NVFP4 encoder.


# Plan: image input for the Qwen3.6 / Qwen3.8-27B NVFP4 frontend (RTX 5090)

Plan Status: completed

## Problem

Current state:

- `Qwen36TorchFrontendRtx(quant='nvfp4')` runs the Qwen3.5-architecture 27B
  language model (Qwen3.6-27B, Qwen3.8-27B; `Qwen3_5ForConditionalGeneration`)
  text-only. `extract_weights_nvfp4` skips `model.visual.*`; RoPE is 1-D
  (`_rope_cos_table[pos]`, rotary dim 64 = `head_dim` 256 × `partial_rotary_factor`
  0.25, θ = 1e7).
- The checkpoints carry a vision tower (`vision_config`: depth 27, hidden 1152,
  16 heads, patch 16, spatial merge 2, `out_hidden_size` 5120,
  `deepstack_visual_indexes` []), image/video token ids, a `Qwen3VLProcessor`
  config and interleaved MRoPE (`mrope_section` [11, 11, 10]).
- In transformers 5.17, `Qwen3_5VisionModel` subclasses `Qwen3VLVisionModel`
  and only deletes the deepstack mergers; `Qwen3_5Model` inherits
  `Qwen3VLModel.get_rope_index`.
- `Qwen3VlVisionRtx` (`_qwen3_vl_vision_rtx.py`) implements the Qwen3-VL ViT
  from `vision_config` (depth, widths, `deepstack_visual_indexes`,
  `out_hidden_size` all read from config; FP8 GEMMs on SM >= 89) and loads from
  a sharded or single-file checkpoint. `_qwen3_vl_geometry.py` implements the
  processor-side geometry (vision segments, MRoPE position ids, vision RoPE,
  position-embedding interpolation).
- In `qwen36_rtx.py`, every RoPE application (8 `qwen36_partial_rope_qk_bf16`
  call sites, all routes: BF16 prefill walk, long-context chunked prefill,
  decode, verify, MTP) reads cos/sin from `_rope_cos_table` /
  `_rope_sin_table` rows indexed by cache position, and every embedding lookup
  (8 `qwen36_embedding_lookup_bf16` call sites, MTP included) reads
  `self._weights.ptrs['embed_w']` at call time. No CUDA graph is captured in
  the constructor; graphs are captured on first use and replay the table
  memory they were captured with.

Problem: no image input on this frontend.

Goal: `generate(messages)` with images on RTX 5090 for the Qwen3.8-27B
LoRA-merged checkpoint, with (1) ViT output matching HF `Qwen3_5VisionModel`
on real images, (2) MRoPE position ids equal to HF `get_rope_index` and the
cos/sin rows matching HF's rotary embedding, (3) text-only generation
unchanged, (4) measured image TTFT (preprocess, ViT, prefill) and decode
tok/s, (5) end-to-end agreement with an HF reference of the same
checkpoint (its FP8 release) on image prompts at the level measured on
text-only prompts. Video input and the Thor frontend are outside this
plan.

## Structure

- `flash_rt/frontends/torch/qwen36_rtx.py` (`Qwen36TorchFrontendRtx`): the
  language model. Gains two explicit hooks: reserved embedding rows and
  per-request RoPE rows. No change to any forward path.
- `flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py`: allocates the
  embedding table with the reserved rows at load.
- `flash_rt/frontends/torch/_qwen36_mrope.py` (new): builds Qwen3.5 text RoPE
  rows (rotary dim 64, full-width rotate_half layout of `_rope_cos_table`)
  from (3, S) MRoPE position ids.
- `flash_rt/frontends/torch/qwen36_vl_rtx.py` (new,
  `Qwen36VlTorchFrontendRtx`): processor, ViT, image-feature placement,
  RoPE-row construction, generation; composes the two classes below.
- `flash_rt/frontends/torch/_qwen3_vl_vision_rtx.py` (`Qwen3VlVisionRtx`) and
  `_qwen3_vl_geometry.py`: reused; changed only if Phase 1 or 2 finds a
  mismatch.
- `research/topics/wam-inference-acceleration/experiments/qwen38_distill_bench/fp8_to_nvfp4_ct.py`: keeps the
  vision tower (BF16) in the converted checkpoint, so one directory holds the
  language model and the ViT.

## State Ownership

- Embedding table (vocab rows + reserved image rows): `Qwen36TorchFrontendRtx`
  (`_weights.ptrs['embed_w']`); the reserved rows are written only through
  `write_embedding_rows`.
- RoPE tables: `Qwen36TorchFrontendRtx` owns `_rope_cos_table` /
  `_rope_sin_table` and an unmodified copy (`_rope_cos_base` /
  `_rope_sin_base`); the live tables are rewritten in place only through
  `set_rope_rows` / `reset_rope_rows`, never reassigned after construction.
- Per-request image state (token ids after remapping, image features,
  position ids, rope delta): `Qwen36VlTorchFrontendRtx`, for the duration of
  one `generate` call.
- ViT weights: `Qwen3VlVisionRtx`.

## Interface

`flash_rt/frontends/torch/qwen36_rtx.py`:

- `Qwen36TorchFrontendRtx.__init__(..., extra_embedding_rows: int = 0)`:
  NVFP4 only (`ValueError` otherwise). The embedding table has
  `vocab_size + extra_embedding_rows` rows; rows `>= vocab_size` start at zero.
- `embedding_slot_base -> int`: first reserved row id (`vocab_size`).
- `write_embedding_rows(first_slot: int, rows: torch.Tensor) -> None`:
  `rows` (n, 5120) BF16 on the device; `ValueError` if
  `first_slot + n > vocab_size + extra_embedding_rows` or
  `first_slot < vocab_size`.
- `set_rope_rows(start: int, cos: torch.Tensor, sin: torch.Tensor) -> None`:
  writes rows `[start, start + n)` of the live tables in place; `cos`/`sin`
  (n, 64) BF16.
- `reset_rope_rows() -> None`: copies the base tables back into the live
  tables.

`flash_rt/frontends/torch/_qwen36_mrope.py`:

- `text_rope_rows(position_ids: torch.Tensor, *, rotary_dim: int,
  rope_theta: float, mrope_section: tuple[int, int, int]) ->
  tuple[torch.Tensor, torch.Tensor]`: (3, S) long -> (S, rotary_dim) BF16
  cos and sin, interleaved sections, halves duplicated (the
  `_rope_cos_table` layout). Equal to the base table rows when the three axes
  are equal.

`flash_rt/frontends/torch/qwen36_vl_rtx.py`:

- `Qwen36VlTorchFrontendRtx(checkpoint_path: str, *, device: str = 'cuda:0',
  max_seq: int, max_image_tokens: int, max_pixels: int | None = None)`.
  `checkpoint_path` is a converted NVFP4 directory that also holds
  `model.visual.*`, `config.json` and processor files; MTP from
  `FLASHRT_QWEN36_MTP_CKPT_DIR` as today.
- `generate(messages: list, *, max_new_tokens: int) -> VlGeneration`:
  `VlGeneration` dataclass with `token_ids: list[int]`, `text: str`,
  `timings_ms: dict[str, float]` (`preprocess`, `vit`, `prefill`, `decode`),
  `num_image_tokens: int`, `prompt_len: int`. Text-only messages take the
  unchanged text path. `ValueError` when the prompt exceeds `max_seq` or the
  image tokens exceed `max_image_tokens`; video input raises
  `NotImplementedError`.

## Flow

Construction: `Qwen36TorchFrontendRtx(path, quant='nvfp4', max_seq,
extra_embedding_rows=max_image_tokens)` -> base RoPE tables copied after
`_extend_rope_table_to` -> `Qwen3VlVisionRtx(path, config=vision_config)` ->
`AutoProcessor.from_pretrained(path)` with the `max_pixels` cap.

`generate(messages)`:
1. Processor -> `input_ids` (1, S), `pixel_values`, `image_grid_thw`.
2. `_qwen3_vl_geometry.vision_segments` -> image spans; vision RoPE and
   position embeddings per image; `Qwen3VlVisionRtx.forward` per image ->
   (n_i, 5120) features (deepstack list empty).
3. `write_embedding_rows(embedding_slot_base, concat features)`; image-token
   positions of `input_ids` replaced by `embedding_slot_base + j`.
4. `mrope_position_ids` -> (3, S); `text_rope_rows` -> rows for cache
   positions `[0, S)`; delta = `max(position_ids) + 1 - S`; rows for cache
   positions `[S, max_seq)` = base rows `p + delta`;
   `set_rope_rows(0, ...)` for the whole range.
5. `generate_own_speculative_KN_nvfp4(remapped_ids, max_new_tokens)`
   unchanged; prefill, decode, verify and MTP read the rewritten rows and
   the image rows through their existing lookups.
6. Generated ids taken after position S; `reset_rope_rows()`.

## Code Mapping

| Item | File |
|---|---|
| reserved embedding rows, `write_embedding_rows`, `embedding_slot_base` | `flash_rt/frontends/torch/qwen36_rtx.py` |
| table allocation with reserved rows | `flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py` |
| `set_rope_rows`, `reset_rope_rows`, base tables | `flash_rt/frontends/torch/qwen36_rtx.py` |
| `text_rope_rows` | `flash_rt/frontends/torch/_qwen36_mrope.py` (new) |
| `Qwen36VlTorchFrontendRtx`, `VlGeneration` | `flash_rt/frontends/torch/qwen36_vl_rtx.py` (new) |
| ViT, geometry (reused) | `flash_rt/frontends/torch/_qwen3_vl_vision_rtx.py`, `_qwen3_vl_geometry.py` |
| converter keeps `model.visual.*` | `research/topics/wam-inference-acceleration/experiments/qwen38_distill_bench/fp8_to_nvfp4_ct.py` |
| validation scripts and results | `research/topics/wam-inference-acceleration/experiments/qwen38_vision/` (new) |

## Implementation Phases

### Phase 1: ViT parity against HF

Phase Status: completed

- Goal: `Qwen3VlVisionRtx` on the Qwen3.8 vision tower matches HF
  `Qwen3_5VisionModel` (BF16 weights from the same checkpoint).
- Modified files: `research/topics/wam-inference-acceleration/experiments/qwen38_vision/vit_parity.py` (new);
  `_qwen3_vl_vision_rtx.py` / `_qwen3_vl_geometry.py` only on a found
  mismatch; `fp8_to_nvfp4_ct.py` (keep `model.visual.*`).
- Observation: on LIBERO frames at native size and resized to non-square
  sizes (several grids), and two images in one prompt: cosine, max abs and
  relative L2 of the merged features (pooler output) against HF, FP8 and
  BF16-GEMM modes of `Qwen3VlVisionRtx`; ViT latency per image.
- Result (`research/topics/wam-inference-acceleration/experiments/qwen38_vision/vit_parity_5090.json`): no code
  change to the ViT or geometry. Merged features vs HF over six cases (grids
  32x32, 20x28, 30x40, 46x136, two 32x32 images): BF16 GEMMs cosine
  0.99902-0.99976, rel. L2 0.022-0.044; FP8 GEMMs cosine 0.99592-0.99835,
  rel. L2 0.059-0.091, minimum per-token cosine 0.923. Eager latency, 256
  tokens: HF 10.5 ms, FP8 6.65 ms, BF16 10.1 ms; 1564 tokens: 75.9 / 55.4 /
  77.3 ms. Requires `-DFLASHRT_BUILD_QWEN3_VL=ON` (module
  `flash_rt_qwen3_vl_kernels` and `w16a16_gemm_sm120_bf16` in
  `flash_rt_kernels`). The converter keeps `model.visual.*` (27.70 GB).

### Phase 2: MRoPE positions and rows against HF

Phase Status: completed

- Goal: position ids and cos/sin rows equal to HF for image prompts.
- Modified files: `flash_rt/frontends/torch/_qwen36_mrope.py` (new);
  `research/topics/wam-inference-acceleration/experiments/qwen38_vision/mrope_parity.py` (new).
- Observation: `mrope_position_ids` vs HF `Qwen3_5Model.get_rope_index` on
  processor outputs (text-only, one image, two images, image between text):
  exact equality and the rope delta. `text_rope_rows` vs HF
  `Qwen3_5TextRotaryEmbedding` for the same ids: max abs difference; for
  equal axes, equality with `_rope_cos_table` rows.
- Result (`research/topics/wam-inference-acceleration/experiments/qwen38_vision/mrope_parity_5090.json`): on
  five prompts (text only; one image; two images; text-image-text-image-text;
  one 640x480 image; S 20-536, grids 32x32 and 30x40) the position ids equal
  HF's and the deltas agree (0, -240, -480, -480, -280); `text_rope_rows`
  equals HF's rotary output (max abs difference 0 for cos and sin) and, for
  the text-only prompt, the 1-D table bit for bit. No change to
  `_qwen3_vl_geometry.py`.

### Phase 3: language-model hooks

Phase Status: completed

- Goal: reserved embedding rows and per-request RoPE rows with text-only
  behavior unchanged.
- Modified files: `flash_rt/frontends/torch/qwen36_rtx.py`,
  `flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py`;
  `research/topics/wam-inference-acceleration/experiments/qwen38_vision/hook_checks.py` (new).
- Observation, on the Qwen3.8 NVFP4 checkpoint at 1k and 4k context:
  (a) `extra_embedding_rows` 0 vs N: generated tokens identical, GPU memory
  delta reported; (b) reserved rows filled with the embeddings of real
  prompt tokens and those positions remapped to the reserved ids: tokens
  identical to the original prompt; (c) all three position axes shifted by
  a constant through `set_rope_rows`: greedy token agreement and
  first-token logit cosine against the unshifted run; (d) after
  `reset_rope_rows`, tokens identical to a fresh run; (e)
  `context_bench.py` prefill and decode within run-to-run noise.
- Result (`research/topics/wam-inference-acceleration/experiments/qwen38_vision/hook_{baseline,hooks}_5090.json`,
  doc prompts, 64 greedy MTP-spec tokens): (a), (b), (d) identical tokens at
  1k and 4k; reserved 4096 rows cost 0.039 GiB; prefill / decode 76.4 / 712.9
  ms (1k) and 308.7 / 849.1 ms (4k) vs 76.3 / 711.3 and 308.5 / 845.7 ms
  without. (c) a 1000-position shift keeps the first 12 (1k) / 22 (4k) tokens,
  then a coherent different continuation. First-token logits (300-token
  prompts at 8 offsets, BF16 route, the logits the API exposes): shift 1000 vs
  none cosine median 0.986 (min 0.927), top-1 6/8; shift 1 vs none median
  0.986 (min 0.924), top-1 5/8; 1000 vs 1001 median 0.984. A one-position
  shift, which only re-rounds every BF16 table row, moves the logits as much
  as a 1000-position shift, so the spread is the model's sensitivity to the
  BF16 RoPE table's rounding, not a position-magnitude effect. This spread is
  the noise floor for Phase 5.

### Phase 4: image frontend

Phase Status: completed

- Goal: `Qwen36VlTorchFrontendRtx.generate` with images.
- Modified files: `flash_rt/frontends/torch/qwen36_vl_rtx.py` (new);
  `research/topics/wam-inference-acceleration/experiments/qwen38_vision/vl_bench.py` (new).
- Observation: answers for LIBERO frames and a few non-robot images printed
  verbatim; timings per stage (preprocess, ViT, prefill, decode) and image
  token counts at two `max_pixels` settings; GPU memory peak. No numerical
  end-to-end reference in this phase.
- Result (`research/topics/wam-inference-acceleration/experiments/qwen38_vision/vl_bench_route{512,1}*`): the
  frontend runs text-only and image prompts. With the default long-route
  threshold (512 tokens), single-image prompts (S 277-283) take the BF16
  one-token-at-a-time prefill: 19.7-20.2 s. With
  `FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ=1` every prompt takes the chunked
  FP8-KV route: one LIBERO frame (256 image tokens) preprocess 3.4 / ViT 7.0 /
  inject 0.5 / prefill 32.7 ms; two frames (512) 4.3 / 13.5 / 0.5 / 51.7 ms;
  FlashRT.png full resolution (1564) 9.2 / 56.2 / 0.8 / 123.4 ms; a chart at
  full resolution (6596) 36.4 / 622 / 1.1 / 545 ms. Decode (128 tokens, MTP)
  1.12-1.46 s. GPU allocated 20.83 GiB after construction (4096 reserved rows
  and the ViT), peak 22.74 GiB at 6619 prompt tokens. Answers: the logo is
  read as "FlashRT"; the full-resolution chart answer reproduces its Chinese
  legend text; LIBERO answers name stove / moka pot / frying pan on the BF16
  route; on the chunked route the single-frame answer is a degenerate JSON
  echo and the wrist answer repeats one phrase, and the capped chart prompt
  yields a tool call. Whether these come from the model or from the pipeline
  is the Phase 5 question.

### Phase 5: end-to-end comparison against HF

Phase Status: completed

- Goal: image-prompt outputs of `Qwen36VlTorchFrontendRtx` (NVFP4) agree
  with HF `Qwen3_5ForConditionalGeneration` on the FP8 checkpoint as closely
  as text-only outputs do; the gap on text-only prompts is the NVFP4-vs-FP8
  quantization floor.
- Reference: the FP8 checkpoint
  (`qwen38-27b-master-full-gpt55-first-recovery-fp8`, 30.4 GB of weights) in
  transformers 5.17 on the same RTX 5090, run in its own process before the
  FlashRT run (both do not fit at once); layers offloaded to CPU memory if a
  prompt does not fit. Requires `accelerate` in the `flashrt` environment.
- Modified files: `research/topics/wam-inference-acceleration/experiments/qwen38_vision/hf_reference.py`
  (new; saves first-step logits and greedy tokens per prompt),
  `research/topics/wam-inference-acceleration/experiments/qwen38_vision/e2e_compare.py` (new).
- Observation, same prompt set for both: text-only prompts and image prompts
  (LIBERO frames, non-robot images, two images in one prompt); per prompt
  first-token logit cosine, top-1 / top-5 agreement, length of the identical
  greedy prefix over 64 tokens; HF peak memory and whether offload was used.
- Result (`research/topics/wam-inference-acceleration/experiments/qwen38_vision/e2e_compare*_5090.json`): the
  FP8 checkpoint does not fit the RTX 5090 in transformers (load stops at
  30.66 GiB allocated), and transformers' FP8 matmul needs the `kernels`
  package; the reference therefore dequantizes the FP8 weights to BF16 at
  load (`FineGrainedFP8Config(dequantize=True)`) with 24 GiB on the GPU and
  the rest in CPU memory (peak 24.39 GiB, 2.6 s per generated token). Prompt
  token ids equal on every prompt. First-token logits, FlashRT vs reference:

  | prompts | prompt tokens | cosine | top-1 | top-5 overlap |
  |---|---|---|---|---|
  | text, short (3) | 19-39 | 0.978-0.986 | 2/3 | 1.0 |
  | text, length-matched (3) | 279-1579 | 0.946-0.968 | 3/3 | 0.6-0.8 |
  | image, FP8 ViT (6) | 245-1582 | 0.892-0.951 | 6/6 | 0.6-1.0 |
  | image, BF16-GEMM ViT (6) | 245-1582 | 0.871-0.965 | 6/6 | 0.6-0.8 |

  Greedy outputs share 0-5 leading tokens on every prompt, text and image,
  and agree in content (logo read as "FlashRT" by both; both name the stove
  and its knob or plate; both hallucinate on the capped chart). Image
  cosines sit below length-matched text (median 0.92 vs 0.96) inside the
  spread a one-position RoPE re-rounding produces (Phase 3, min 0.924), and
  switching the ViT between FP8 and BF16 GEMMs moves single prompts by up to
  0.05 without raising the median; no image-specific deviation is found.

## Related, not in this plan

- Video input (per-frame timestamps in the prompt).
- `qwen36_thor.py` with the same hooks.

# Plan: host-side decoupling for a planner and an executor in one process (RTX 5090)

Plan Status: completed

## Problem

Current state (measured 2026-11-03, `research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/`,
`research/topics/wam-inference-acceleration/reports/planner_executor_colocation_5090.md`):

- A Qwen3.8-27B NVFP4 planner (`Qwen36VlTorchFrontendRtx`) and the ImageWAM fp8 executor
  (`ImageWAMTorchFrontendThor`) in one process, two Python threads, miss 80-92% of the
  executor's 50 ms deadlines under `prio` and `green<P>`. The executor's GPU work is only
  10-20% slower; its thread waits 24-118 ms for the GIL before staging its inputs.
- `fvk.cuda_read_i32_sync` (`csrc/bindings.cpp`) runs `cudaMemcpyAsync` + `cudaStreamSynchronize`
  with the GIL held. The planner calls it once per speculative-decode cycle
  (`qwen36_rtx.py`, two call sites) and waits there for the whole verify forward
  (20-30 ms); 632 of 683 py-spy samples of the planner thread holding the GIL are in it.
- The long-context chunked prefill launches its kernels from a Python loop unless
  `FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1`.
- All 17 `torch.cuda.graph(...)` captures in `qwen36_rtx.py`, the ViT capture in
  `_qwen3_vl_vision_rtx.py` and the ImageWAM capture in `imagewam_thor.py` use the default
  global capture mode. The planner captures graphs lazily on first use, also inside a
  running co-location window (46-64 → 86-110 graphs); a CUDA call from the other thread
  during such a capture fails it with `cudaErrorStreamCaptureUnsupported`.
- Research-only monkeypatches in `colocation_bench.py` (`PLANNER_READBACK_NOGIL`,
  `CAPTURE_THREAD_LOCAL`) remove the first and third problem; with them and the prefill
  graph, `prio` short and `green25` / `green50` short and long miss 0-0.1%.
- ISSUE-104: constructing the Qwen3.6 frontend while a side stream is current gives garbage
  output; `_quant_bf16_lin_proj` copies on the current stream and quantizes on stream 0.
- ISSUE-103: the ImageWAM fp8 tick is 28.2 ms against 23.85 ms documented, with the
  current `build_full/` kernel library.

Problem: the host-side fixes live only as monkeypatches in a research script; FlashRT itself
still blocks the GIL in the readback, captures in global mode, and builds wrong weights under
a side stream.

Goal: with the unmodified FlashRT frontends (no research monkeypatch), (1) the planner's
per-cycle readback does not hold the GIL, (2) no capture fails because of the other thread,
(3) side-stream construction gives the same output and weights as default-stream
construction, (4) the co-location cells that met the deadline with the monkeypatches meet it
again, with planner output unchanged, (5) the cause of the executor's 28.2 ms is measured.

## Structure

- `csrc/bindings.cpp` (`flash_rt_kernels` module): `cuda_read_i32_sync` releases the GIL while
  it waits. No other binding changes.
- `flash_rt/frontends/torch/qwen36_rtx.py`, `_qwen3_vl_vision_rtx.py`, `imagewam_thor.py`:
  every graph capture passes `capture_error_mode='thread_local'`. Capture order, pools and
  streams unchanged.
- `flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py`: load-time conversion kernels run on
  the current torch stream instead of stream 0.
- `research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/`: measurement only; the monkeypatch flags stay,
  default off.

## State Ownership

- The GIL during the readback: released by the binding for the duration of the copy and
  stream synchronize only; no Python object is touched while released.
- Graph capture state: each capturing frontend, per capture; the mode is fixed at each call site.
- Weight buffers at load: `extract_weights_nvfp4`, produced on the stream current at
  construction.

## Interface

- `flash_rt_kernels.cuda_read_i32_sync(src: int, stream: int = 0) -> int`: signature and result
  unchanged; the GIL is released between entry and return of the CUDA calls.
- `torch.cuda.graph(g, ..., capture_error_mode='thread_local')` at every capture site listed above.
- `_quant_bf16_lin_proj` and the other load-time helpers in `_qwen36_rtx_nvfp4_weights.py` that
  launch with a `stream` argument: the default becomes the current torch stream's handle.

## Flow

Planner thread, per decode cycle: draft chain graph replay → verify graph replay →
`cuda_read_i32_sync` (GIL released while waiting) → accept. Executor thread, per tick: wait for
the release time → stage inputs (needs the GIL; obtains it while the planner waits in the
readback) → graph replay on its stream. A planner graph captured during the window captures
in thread-local mode, so the executor thread's CUDA calls neither fail nor enter it.

## Code Mapping

- Readback: `csrc/bindings.cpp` (`cuda_read_i32_sync`); rebuilt into
  `flash_rt/flash_rt_kernels.cpython-312-x86_64-linux-gnu.so` from `build_full/`.
- Capture mode: `flash_rt/frontends/torch/qwen36_rtx.py` (17 sites),
  `flash_rt/frontends/torch/_qwen3_vl_vision_rtx.py` (1), `flash_rt/frontends/torch/imagewam_thor.py`
  (`_record_graph`).
- Load stream: `flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py`.
- Measurements: `research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/` (cells, README), report and page
  `research/topics/wam-inference-acceleration/reports/planner_executor_colocation_5090.{md,html}`; issues ISSUE-103 / ISSUE-104.

## Implementation

### Phase 1 — Readback outside the GIL

Phase Status: completed

Result: the binding drains the stream with the GIL released, then copies. Copying first (the first attempt) left the
executor's graph launch at 37.5 ms p50 and 92.4% misses: a device-to-pageable-host copy enqueued behind pending work
waits inside the driver and stalls the other thread's launches. With drain-then-copy, `prio` short with only
`CAPTURE_THREAD_LOCAL=1`: executor 36.9 / 46.8 ms p50 / p99, 0% misses, staging 0.5 ms (the monkeypatched cell: 36.9 /
46.6, 0%); planner alone output CRC unchanged in both contexts (`cells/hostfix/`). One planner call in 63 gave a
different output under co-location, as with the monkeypatch (ISSUE-105).

- Goal: `cuda_read_i32_sync` releases the GIL while it copies and synchronizes.
- Modified files: `csrc/bindings.cpp`; rebuild `flash_rt_kernels` in `build_full/`.
- Observation: planner alone (short / long, whole device) output CRC identical to the
  pre-change run, call ms within run-to-run spread; co-location `prio` short with
  `PLANNER_READBACK_NOGIL=0`, `CAPTURE_THREAD_LOCAL=1`: executor miss rate and input-staging
  wait, against the monkeypatched cell (0%, 0.5 ms).

### Phase 2 — Thread-local graph capture

Phase Status: completed

Result: 12 windows with both monkeypatches off (`prio`, `green25`, short and long, one under py-spy): no capture error;
planner alone CRC unchanged. `prio` short 0% misses (p99 46.7-47.0 ms; 13.5% under py-spy), `green25` long 0% (p99 36.0),
`prio` long 44-46% (prefill fills the SMs, as with the monkeypatches). `green25` short without the prefill graph misses
2.4-3.6% (p99 53-56 ms), as the monkeypatched cell without it (2.3%); Phase 5 therefore runs short context with the
prefill graph too. Differing planner outputs coincide with in-window captures (ISSUE-105); a few executor ticks differ
by up to 4.9e-3 under `prio` only (ISSUE-106).

- Goal: captures in the planner and executor frontends use thread-local capture mode.
- Modified files: `qwen36_rtx.py`, `_qwen3_vl_vision_rtx.py`, `imagewam_thor.py`.
- Observation: planner alone and executor alone outputs bit-identical to Phase 1; co-location
  `prio` and `green25`, short and long, with both monkeypatch flags off: no capture error in
  three 60 s windows each, one of them under py-spy sampling (which triggered the failure
  before); planner graph counts at window start and end.

### Phase 3 — Load-time kernels on the current stream (ISSUE-104)

Phase Status: completed

Result: before the change a side-stream construction left 48 of 1 482 weight buffers different (most all zero) and
gave garbage; after it the weight digest and all three outputs equal the default-stream construction (ISSUE-104
resolved).

- Goal: constructing the frontend under a side stream gives the same weights and output.
- Modified files: `_qwen36_rtx_nvfp4_weights.py`.
- Observation: `stream_check.py` with side-stream construction: output text equal to the
  default-stream construction; checksums of every quantized weight buffer equal.

### Phase 4 — Executor tick time by kernel build (ISSUE-103)

Phase Status: completed

Result: the kernel library does not explain it (all six build pairs 28.0-28.2 ms). The documented profile script now
gives 28.17 ms with the whole difference in cuBLAS GEMMs; the env loads conda's cuBLAS 13.4.1.3, and preloading
13.6.1.10 (the version PROJECT.md recorded) gives 24.4 ms (ISSUE-103). Phase 5 keeps the current environment so that
it compares with the monkeypatched cells under the same libraries.

- Goal: measure whether the kernel library explains 28.2 ms against 23.85 ms.
- Modified files: none (experiment); `.so` files swapped from `tmp/so_backup_slim/` and
  `tmp/so_backup_premerge/` in one session, current build restored afterwards.
- Observation: `exec_capture_check.py` tick ms (100 ticks) per build, SM clock per run. The
  result goes to ISSUE-103; a build change, if any, is a separate plan.

### Phase 5 — Co-location with framework fixes only

Phase Status: completed

Result: with both monkeypatches off, every compared cell reproduces the monkeypatched one (p99 within 0.1-1.3 ms,
same miss rates, planner call within 0-50 ms): prio short 0.1 %, green25 short / long 0.1 / 0 %, green50 short / long
0 / 0.1 %, coop short / long 0 %, prio long 44.4 % (no SM limit). The 0.1 % are single ticks during the planner's first
call of the window (`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/README.md`, "Host-side fixes in FlashRT").

- Goal: re-measure the co-location cells with the unmodified frontends.
- Modified files: research cells, README, report and page.
- Observation: short and long × `prio`, `green25`, `green50`, `coop`, both monkeypatch flags
  off, `FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1` in both contexts except `coop` (eager 256-token chunks; prio and green25
  long are the Phase 2 windows): executor p50 / p99 / miss, planner call ms and output CRC, against the monkeypatched
  cells.

# Plan: replay of the VoLoAgent Full-loop master workload with the ImageWAM executor (RTX 5090)

Plan Status: completed

## Problem

Current state:

- Co-location has been measured under a stress workload only: executor every 50 ms, planner back-to-back with a
  602 / 8 110-token prompt and 64 output tokens (`research/topics/wam-inference-acceleration/experiments/colocation_qwen27b_5090/`).
- The agent harness in `/home/jingwu/workspace/bear/code` runs VoLoAgent's Full loop (`scripts/volo_tool/eval_agent.sh`:
  `--mode subgoal --failure-monitor vlm --recovery-mode replan_tools --check-interval 80 --use-front-camera`):
  - the evaluation client executes 8 actions per policy query at 15 Hz (`scripts/volo_tool/robolab_run.py:193`), so
    one query every 533 ms of sim time;
  - the proxy runs the master call synchronously before the policy query
    (`third_party/VoLoAgent/vlm_orchestrator/proxy.py:614`, `:828`); the sim waits;
  - master calls: one subgoal decomposition at episode start; a monitoring decision every 80 env steps
    (`failure_handlers/vlm.py:430-439`); a replan right after a `replan` decision or a `next` on the last subgoal, at
    most 3 per episode (`strategies/subgoal_base.py:56`, `:1222-1285`); each call is one system and one user message,
    no history, temperature 0, `max_tokens` 512 (replan 1024), JSON reply;
  - images at full resolution, JPEG: decomposition = front 864×480 + exterior 1280×720 + wrist; monitor = BEFORE
    (episode start) and NOW of the same three, 6 images; replan = BEFORE and NOW front;
  - the policy's instruction is the current subgoal, so it changes on each advance and replan.
- The harness's recorded calls (`<OUT>/orchestrator/*/vlm_calls.jsonl`) are not available on this machine; the wrist
  resolution the master receives (224×224 or 1280×720) is not determinable from the code.
- `Qwen36VlTorchFrontendRtx.generate` decodes `max_new_tokens` tokens and truncates at the first EOS afterwards.
- `planner_qwen27b.Planner` builds the frontend with `max_image_tokens=1024`; a monitor call has 2 698 (wrist 224) or
  4 330 (wrist 1280×720) image tokens (32 px per token after the 2×2 merge).
- The co-located executor runs with its text encoder freed (6.30 instead of 22.96 GB); a new instruction cannot be
  encoded on the GPU in that state.

Problem: the planner's and executor's timing under the harness's frequencies and call contents is unmeasured.

Goal: measured, with ImageWAM LIBERO fp8 as the executor and zero-shot Qwen3.8-27B NVFP4 as the master:
(1) per call type: prompt, image and output tokens; ViT, prefill, decode and call ms alone;
(2) per episode under the harness's synchronous protocol: model time per episode against its 15 Hz sim time;
(3) under a real-time schedule at the same frequencies (executor every 533 ms, master calls concurrent): executor
latency against one control step (66.7 ms) and one chunk (533 ms), master decision latency in env steps, for in-process
`prio`, `green50` and `coop`;
(4) the cost of switching the executor's instruction.

## Structure

- `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/volo_calls.py`: the master call contents. System prompts and user-message
  templates copied verbatim from VoLoAgent with source references; image preparation to the harness resolutions from
  LIBERO frames (resized to the target height, centered on a black canvas of the target size).
- `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/record_calls.py`: the Full-loop decision logic (decompose, monitor every 80
  steps, replan on `replan` or a last-subgoal `next`, at most 3) run on the planner alone; writes one calls file per
  episode in the shape of the harness's `vlm_calls.jsonl` (messages, reply).
- `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/replay_bench.py`: replays a calls file with the executor in the same
  process: `sync` (one thread: master call, then executor query) and `realtime` (executor thread on a 15 Hz clock,
  planner thread starting each call at its step's time) with sharing `prio`, `green50` or `coop`.
- `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/run_replay.sh`, `README.md`, `calls/`, `cells/`.
- Reused unchanged: `colocation_qwen27b_5090/planner_qwen27b.py` (checkpoint paths, environment settings,
  `PlannerStreams`), `colocation_qwen27b_5090/exec_slim.py` (`slim_executor`), `colocation_qwen27b_5090/coop_gate.py`
  (`CoopGate`, `ExecutorClock`, `install`), `colocation_5090/green_partition.py` (`GreenPartition`),
  `benchmarks/_imagewam_libero_frames.py`.
- No FlashRT source changes.

## State Ownership

- Episode decision state (subgoal list, current index, replan count, step of the last check): `record_calls.py`, one
  episode at a time; the replay reads only the calls file.
- Calls files: written once by `record_calls.py`, read-only afterwards.
- Executor instruction: set once per replay process before the executor is slimmed; switching is measured separately
  (Phase 4).

## Interface

- `volo_calls.py`:
  - `CallSpec` (dataclass): `kind: Literal["decompose", "monitor", "replan"]`, `step: int`, `system: str`,
    `user_parts: list[TextPart | ImagePart]`, `max_tokens: int`.
  - `HarnessFrames` (dataclass): `front` 864×480, `exterior` 1280×720, `wrist` 224×224 or 1280×720 (PIL images).
  - `harness_frames(episode: int, step: int, wrist: int) -> HarnessFrames`.
  - `decompose_call(task: str, f: HarnessFrames) -> CallSpec`;
    `monitor_call(task: str, subgoals: list[str], idx: int, before: HarnessFrames, now: HarnessFrames, step: int) -> CallSpec`;
    `replan_call(task: str, original: list[str], before: HarnessFrames, now: HarnessFrames, step: int) -> CallSpec`.
- `python record_calls.py EPISODE WRIST OUT_JSONL`: one row per call: `kind`, `step`, `messages` (image paths),
  `max_tokens`, `output_text`, `output_tokens` (before EOS), `parse_ok`, `decision`, alone timings.
- `python replay_bench.py {sync|realtime} SHARE CALLS_JSONL OUT_JSON` (`SHARE` in `none`, `prio`, `green50`, `coop`):
  per-call timings, per-release executor records, summary.
- A replayed call runs `fe.generate(messages, max_new_tokens=output_tokens + 1)`, so its decode length equals the
  recorded reply through EOS.

## Flow

- Record: planner alone, whole device. Per LIBERO-10 episode (800 env steps, frames taken from the episode video at
  `step × (n_frames − 1) / 800`): decompose at step 0 → parse subgoals (failure → the instruction as the only subgoal,
  as VoLoAgent does) → at steps 80, 160, …, 800: monitor → parse the decision → `next` advances the subgoal; `replan`,
  or `next` on the last subgoal, issues the replan call at the same step (count < 3) and replaces the subgoals → rows
  written in call order.
- Sync replay: query points every 8 steps; at a point that holds recorded calls, run them in order, then the executor
  query. Totals: master ms, executor ms per episode, against 800 / 15 = 53.3 s of sim time.
- Real-time replay: executor released every 8 / 15 s (533.3 ms) on the wall clock; planner thread starts each call at
  `step / 15` s after the episode start or when the previous call returns, whichever is later. Executor latency is
  measured from release to completion; decision latency = call end − its step's time, in env steps.

## Code Mapping

- Call contents: `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/volo_calls.py`.
- Recording: `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/record_calls.py` → `calls/ep<E>_wrist<W>.jsonl` (+ images).
- Replay: `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/replay_bench.py` → `cells/<mode>_<share>_ep<E>_wrist<W>.json`.
- Grid runner: `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/run_replay.sh`.
- Results: `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/README.md`; a section in
  `research/topics/wam-inference-acceleration/reports/planner_executor_colocation_5090.{md,html}`; `issues.md` / `opportunities.md` for open items.

## Implementation Phases

### Phase 1 — Call contents and recorded calls

Phase Status: completed

- Goal: the three call types built from VoLoAgent's texts; recorded calls for LIBERO-10 episodes 0, 43 and 188 (three tasks) × wrist
  {224, 1280×720}; alone timings.
- Modified files: new `volo_calls.py`, `record_calls.py`.
- Observation: per call: prompt / image / output tokens, `parse_ok`, decision; ViT / prefill / decode / call ms; per
  call type, the decode time at the `max_tokens` cap (what the frontend spends without an EOS stop).

### Phase 2 — Synchronous replay

Phase Status: completed

- Goal: the harness's protocol: master call, then executor query, in one thread.
- Modified files: new `replay_bench.py` (`sync`), `run_replay.sh`.
- Observation: per episode: master ms (by call kind), executor ms, call count, against 53.3 s of sim time; per-call ms
  with the executor resident against the alone timings.

### Phase 3 — Real-time replay

Phase Status: completed

- Goal: the same calls and frequencies on a wall-clock schedule, executor and planner concurrent.
- Modified files: `replay_bench.py` (`realtime`), `run_replay.sh`.
- Observation: per cell (`prio`, `green50`, `coop` × wrist {224, 1280×720}, episodes 0, 43, 188): executor p50 / p99 / max
  latency and misses against 66.7 ms and 533 ms; master call ms and decision latency in env steps; free memory.
- `coop`: the executor loop calls `ExecutorClock.stop()` (`colocation_qwen27b_5090/coop_gate.py`) after an episode's
  last query, so planner calls still running then are not gated against a release that never comes.

### Phase 4 — Executor instruction switch; results

Phase Status: completed

- Goal: the cost of giving the executor a new subgoal; results recorded.
- Modified files: `replay_bench.py` (an `instruction` measurement), README, report and page, `issues.md`,
  `opportunities.md`.
- Observation: `set_prompt` ms for the decomposed subgoals with the text encoder resident, the executor's memory with
  and without the encoder, and whether the planner, encoder and executor fit together in 32 GB.

## Related, not in this plan

- An EOS stop in `Qwen36VlTorchFrontendRtx.generate` (needed for serving calls of unknown length).
- OpenAI-compatible and openpi-protocol servers around the FlashRT frontends; closed-loop runs in LIBERO or RoboLab.
- Replaying the harness's own `vlm_calls.jsonl` once available.

# Plan: image input for the 27B planner on Jetson Thor, and the planner's draft length per call

Plan Status: completed

## Problem

Current state:

- `Qwen36VlTorchFrontendRtx.__init__` builds `Qwen36TorchFrontendRtx` and `Qwen3VlVisionRtx` with its defaults (FP8
  block-128 linears, FA2 attention). On Thor FA2 is not built and the Qwen3-VL kernel module has no FP8 block-128 ViT
  GEMM (`qwen3_vl_thor` builds the tower with `fp8=False, attention_backend='sdpa'`).
- `Qwen36TorchFrontendThor` subclasses the RTX frontend, forwards its constructor arguments (so
  `extra_embedding_rows`), and reads the parent's `_rope_cos_table` / `_rope_sin_table` in its K-row forward and MTP
  tail; `write_embedding_rows`, `set_rope_rows` and `reset_rope_rows` are inherited.
- `generate` calls `generate_own_speculative_KN_nvfp4` with its default K = 6, i.e. the prompt-length table: K = 6 for
  decompose (2 126-2 130 tokens), K = 3 for monitor (4 072-4 083); on the master, K = 3 decodes decompose in 218 ms
  against 241 ms (OPT-039).

Problem: recorded harness calls with images cannot run on Thor; the draft length cannot be chosen per call.

Goal: (1) the image frontend on Thor with the same `generate` interface; per recorded call on Thor: ViT, prefill,
decode and call ms, spec cycles, tokens per forward, reply equal to the RTX reply of the same model; (2) a per-call
draft length.

## Structure

- `flash_rt/frontends/torch/qwen36_vl_rtx.py`, `Qwen36VlTorchFrontendRtx`: the request flow (processor, one ViT call
  per image, embedding-row injection, MRoPE rows, speculative generation) and two construction methods that choose
  the language model and the ViT.
- `flash_rt/frontends/torch/qwen36_vl_thor.py` (new), `Qwen36VlTorchFrontendThor(Qwen36VlTorchFrontendRtx)`: overrides
  the two construction methods only.
- `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/thor/planner_calls_vl.py` (new): the recorded calls on either frontend.

## State Ownership

- Embedding table, RoPE tables, KV caches: the language-model frontend (`qwen36_rtx.py`); the image frontend changes
  them only through `write_embedding_rows`, `set_rope_rows`, `reset_rope_rows` (unchanged).
- Draft length: requested by the caller of `generate`; the language model's `_long_tq_effective_k` decides the K used.

## Interface

- `Qwen36VlTorchFrontendRtx._build_language_model(self, checkpoint_path: str, device: str, max_seq: int) ->
  Qwen36TorchFrontendRtx` and `_build_vision(self, checkpoint_path: str, device: str, config: dict) ->
  Qwen3VlVisionRtx`, called once from `__init__`.
- `Qwen36VlTorchFrontendRtx.generate(messages: list, *, max_new_tokens: int, spec_k: int = 6,
  chat_template_kwargs: dict | None = None) -> VlGeneration`; `spec_k` is passed as `K`: below 6 it caps the
  prompt-length table, 6 or more takes the table.
- `Qwen36VlTorchFrontendThor`: `_build_language_model` returns `Qwen36TorchFrontendThor(..., quant='nvfp4',
  extra_embedding_rows=max_image_tokens)`; `_build_vision` returns `Qwen3VlVisionRtx(..., fp8=False,
  attention_backend='sdpa')`.
- `python research/topics/wam-inference-acceleration/experiments/harness_replay_5090/thor/planner_calls_vl.py {rtx|thor}` (env `CK`, `OUT`) →
  `$OUT/planner_calls_vl_{rtx|thor}.json`: per (call, spec_k) a `CallRecord`.

## Flow

`planner_calls_vl.py` builds the frontend for the given hardware from `$CK/qwen38-27b-nvfp4-official` (MTP head from
`$CK/qwen38-27b-fp8-official`) → for each recorded call (ep0, ep43, wrist 224) and `spec_k` in (6, 3): one warm-up
and one timed `generate` with `max_new_tokens` = the official model's reply through EOS (`cells/mtp_tail_base.json`)
→ the timed call's stage times, spec counters and reply against the reference.

## Code Mapping

- `flash_rt/frontends/torch/qwen36_vl_rtx.py`: construction methods, `spec_k`.
- `flash_rt/frontends/torch/qwen36_vl_thor.py`: the Thor class.
- `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/thor/planner_calls_vl.py`, `thor/planner_calls_vl_rtx.{json,log}`,
  `smoke/vit_thor_config_check.{py,log}`.
- Records: `docs/qwen36_vl.md`, `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/README.md`, `THOR_CHECKLIST.md` (H3),
  `opportunities.md` OPT-039.

## Implementation Phases

### Phase 1 — Construction methods and `spec_k` on the RTX image frontend

Phase Status: completed

- Modified files: `qwen36_vl_rtx.py`.
- Observation: the 22 recorded calls on the 5090 with the official model: replies equal to the references at both
  `spec_k`; effective K 6 / 3 for decompose, 3 / 3 for monitor; decode ms per call kind.

### Phase 2 — Thor image frontend and the call script

Phase Status: completed

- Modified files: new `qwen36_vl_thor.py`, `thor/planner_calls_vl.py`, `smoke/vit_thor_config_check.py`.
- Observation: the script's RTX run (`thor/planner_calls_vl_rtx.json`) as the Thor denominator; the Thor ViT
  configuration against the RTX default on this GPU (merged-feature cosine).

### Phase 3 — Thor run

Phase Status: completed

- Thor row H3 in `THOR_CHECKLIST.md` (after H3a has converted the checkpoint).
- Observation: the class names the script prints; per call kind ViT / prefill / decode / call ms, tokens per forward
  and same reply against `thor/planner_calls_vl_rtx.json`; peak memory.
- Result (THOR_STATUS_SUMMARY.md round `1104_h3`): the frontend loads as `Qwen36VlTorchFrontendThor` /
  `Qwen36TorchFrontendThor` with the MTP head and runs all 44 calls; monitor 6.97-7.02 s per call (ViT 0.27, prefill
  4.36, decode 2.31-2.35 s), decompose 5.17-5.24 s; 0 of 44 replies equal the RTX references; peak memory not
  reported. The reply difference and the decode-per-cycle gap between H3a and H3 are issues.md ISSUE-112; the Thor
  tail fc is opportunities.md OPT-039 candidate 5.

# Plan: Thor monitor-call latency, decision quality and co-location

Plan Status: approved

Rationale, evidence and the expected effect of every item: `research/topics/long-horizon-planner-executor/thor_rationale.md`.

## Problem

- Current state (Jetson Thor, official Qwen3.8-27B NVFP4, monitor calls of the VoLoAgent harness, round
  `1110_h7h4bh8`): from call start to decision available 6.40 s on the original path (above the 5.33 s monitor
  interval) and 2.61 s with the prefix capsule and stop at action closure; prefill is about 2.0 s of the 2.61 s, and
  55 % of the prefill is the `native` linear-attention backend that `Qwen36TorchFrontendThor` pins because the WY
  backend drifts on SM110. All 20 Thor reference decisions are `continue`; Thor replies are not reproducible run to
  run (issues.md ISSUE-114). The 27B planner has not been measured co-located with the executor on the Thor.
- Goal: on the Thor, decision available below the monitor interval with the smallest robot wait, executor calls
  within their deadline under co-location, and intervention decisions equal to the original path.

## Structure

| Module | Responsibility | State it owns |
|---|---|---|
| `flash_rt/frontends/torch/qwen36_thor.py`, `qwen36_rtx.py` | linear-attention prefill backend selection and kernels | the backend choice (`FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND` default per frontend) |
| `research/topics/long-horizon-planner-executor/experiments/` | probes, call sets and their results | measured results only |
| `THOR_CHECKLIST.md` | Thor rows | undone rows |

The co-located executor, the sharing modes and the shared-board lock are owned by the co-location measurement on
the Thor; this plan delivers the planner process it drives.

## Interface

- WY stage probe: `python gdn_wy_stage_drift.py {rtx|thor} OUT_JSON` -> per layer and per WY stage, cos and max abs
  difference against `native`, and prefill ms per backend.
- Repeat probe: `python reply_repeat.py {rtx|thor} OUT_JSON CALLS_JSONL...` -> per call and repetition, reply tokens,
  first differing token position, top-2 logit margin there, decision.
- Intervention call set: `calls/*.jsonl` in the `harness_replay_5090/calls/` format, recorded from the closed-loop
  failure episodes; consumed by `monitor_call/monitor_call_bench.py`.
- Planner process: `python planner_process.py thor {original|fast} OUT_JSONL CALLS_JSONL...` -> one line per call with
  call start, decision available, call end (monotonic clock), decision.

## Flow

H10 runs the existing `decode_cycle_drift.py` on the Thor. The three probes run on the 5090 first (denominator), then
in one Thor round (H9, H11, H12). The WY fix is defined from H9's result. The planner process is handed to the
co-location measurement.

## Code Mapping

- H10: `research/topics/wam-inference-acceleration/experiments/harness_replay_5090/thor/decode_cycle_drift.py`
  (unchanged).
- H12: new `research/topics/long-horizon-planner-executor/experiments/thor_repeatability/reply_repeat.py`.
- H9: new `research/topics/long-horizon-planner-executor/experiments/gdn_wy_thor/gdn_wy_stage_drift.py`.
- H11: new `research/topics/long-horizon-planner-executor/experiments/sim_closed_loop/record_intervention_calls.py`
  and its `calls/`; `monitor_call/monitor_call_bench.py` (unchanged).
- Planner process: new `research/topics/long-horizon-planner-executor/experiments/colocation_thor/planner_process.py`.
- WY fix: `csrc/kernels/` and `flash_rt/frontends/torch/qwen36_thor.py`, files fixed after H9.
- Records: `THOR_STATUS_SUMMARY.md`, issues.md ISSUE-112 / ISSUE-114, `opportunities.md` OPT-040 candidate 5, the
  experiment READMEs, `thor_rationale.md`.

## Implementation Phases

### Phase 1 — H10: decode after 180 s and 300 s of idle

Phase Status: active

- Modified files: `THOR_CHECKLIST.md`.
- Observation: first and second generation ms per cycle after each gap against the steady value; copy GB/s.

### Phase 2 — H12 repeat probe

Phase Status: pending

- Modified files: new `thor_repeatability/reply_repeat.py`.
- Observation: the 5090 run (expected identical repetitions) as the denominator; Thor row H12.

### Phase 3 — H9 WY stage probe

Phase Status: pending

- Modified files: new `gdn_wy_thor/gdn_wy_stage_drift.py`.
- Observation: the 5090 run (per-stage cos against `native`, prefill ms per backend); Thor row H9.

### Phase 4 — H11 intervention call set

Phase Status: pending

- Modified files: new `sim_closed_loop/record_intervention_calls.py`, `sim_closed_loop/calls/`.
- Observation: number of calls per reference decision; the 5090 `monitor_call_bench` run on the set; Thor row H11.

### Phase 5 — Thor round H9, H11, H12

Phase Status: pending

- Modified files: `THOR_CHECKLIST.md`; records.
- Observation: the rows' numbers as defined in the checklist.

### Phase 6 — planner process for the co-location measurement

Phase Status: pending

- Modified files: new `colocation_thor/planner_process.py`.
- Observation: a 5090 run of both arms alone (L per call, decisions); the Thor co-location numbers come from the
  co-location measurement.

### Phase 7 — WY backend on SM110

Phase Status: pending

- Modified files: defined from Phase 5's H9 result (return to plan if it needs a kernel interface change).
- Observation: per-layer cos against `native` on the Thor; monitor prefill ms; H8 decisions.
