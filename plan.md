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

Plan Status: approved. Phases 1-6 are completed; Phase 7 (Thor confirmation)
is blocked: no sm_110 device on the dev box, `gemm_variant_autotune` remains
an opt-in flag, OPT-018's Thor check is unmeasured, and `issues.md`
ISSUE-023 tracks it.

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

Phase Status: blocked

The Thor checklist in `opportunities.md` OPT-018: the tile sweep must show correct outputs for every tile, and the `infer()` A/B of heuristic vs tuned tiles must show action cosine >= 0.9999 and a P50 delta.
Blocker: no sm_110 device on the dev box; SM100 CUTLASS and NVFP4 kernels do not run on sm_90, which only compile-checks them (`sm110_check.sh`), recorded as `issues.md` ISSUE-023.

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

Plan Status: completed for W0-W12, T1-T5 and S1-S4 (the `0920c` round ran
the line's last three Thor rows and ISSUE-080 is resolved). The follow-up
phases F1-F3 (the served default promoted to the fastest configuration, the
latency baselines per configuration, the workload in the calibration
identity) have their code and CPU tests in place; their Thor observations
are the pending rows of `THOR_CHECKLIST.md`.

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
Phase Status: code complete; Thor observation pending (THOR_CHECKLIST.md)
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

### Phase F2: latency baselines per configuration
Phase Status: code complete; Thor observation pending (THOR_CHECKLIST.md)
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

### Phase F3: the calibration identity carries the workload
Phase Status: code complete; Thor observation pending (THOR_CHECKLIST.md)
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

Plan Status: R1 completed (schema, checker, renderer, skeleton); R2-R6 pending
the owner's RoboTwin declaration and a Thor session.

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
Phase Status: pending
- Goal: a script times the official implementation at a workload and step
  count over exactly the timed call above (image encode + proprio + transformer
  with the text context passed in), with warmup, device sync and percentiles.
  It replaces the one-off 453.6 ms.
- Modified files: `benchmarks/imagewam_official_torch_bench.py` (new).

### Phase R4: int8 and int4 at a workload
Phase Status: pending
- Goal: the two benches take `--workload` and the step count, and emit the
  same latency record; their rows stay `gemm_only`.
- Open decision: whether a real full-pipeline INT8 tier (activation
  quantization inside the frontend; INT8 has a `tcgen05` legacy path on Thor)
  is worth building so the int8 row can be `full_infer`. int4 cannot be
  (no hardware path, OPT-007), so its row is `gemm_only` either way.
- Modified files: `benchmarks/imagewam_thor_int8_bench.py`,
  `imagewam_thor_int4_bench.py`, `imagewam_thor_graph_bench.py`.

### Phase R5: the session driver and importer
Phase Status: pending
- Goal: one script runs every row of a table in one process family, writes the
  raw logs, and an importer turns the logs into rows of the JSON (session id,
  `effective_config`, percentiles, fidelity).
- Modified files: `scripts/imagewam_thor_results.sh` (new),
  `benchmarks/imagewam_result_table.py` (`import` command).

### Phase R6: the Thor session
Phase Status: pending
- The next `THOR_CHECKLIST.md` describes it. Points it must state: fp8 needs
  a calibration recorded on the workload's own data (the identity carries
  `num_denoise_steps`, `shift` and the camera geometry); 30 steps put about
  three times the ActionDiT kernels in one graph, so the capture time and the
  graph memory are observed on the RoboTwin rows; the official row runs at 30
  steps too.
