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
are implemented, verified on H100 and merged into `roadmap/integration`.
Thor validation is pending; see "Execution status" and "Thor validation
checklist" below. An item whose Thor step is still open keeps its own
`# Plan: <item>` section later in this file; every item's results are
recorded in the `opportunities.md` entry named in "Execution status".

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
| 4 | `linear2` merge (attn_out_proj + mlp_down) | speed | days | none | opportunities.md OPT-015 op-fusion audit sub-problem 3, not started |
| 5 | VAE port to FlashRT kernel style + in-graph capture | speed | weeks, standalone | none | opportunities.md OPT-008; prerequisite for in-graph VAE, not for anything else here |
| 6 | Attention-chain fusion feasibility recheck at ImageWAM's own real shapes | analysis | hours, analysis only | none | Pi0.5 rejected this at `M=10` (5-7x slower); ImageWAM's shapes (`M~905` backbone, `M=64` ActionDiT) differ, worth re-checking, not assuming the same verdict |
| 7 | Real calibration data pipeline (replace `N(0,0.1)` placeholder in `_calibrate_fp8()`) | accuracy | days | none | hub node -- unlocks items 8 and 13 |
| 8 | AWQ per-channel scale folded into NVFP4 weights | accuracy | +2-3 days | 7 | may help ImageWAM MORE than it helped Pi0.5 (ImageWAM's merged-`linear1` GEMMs are more bandwidth-shaped than Pi0.5's own compute-bound QKV case, where Pi0.5 shipped AWQ disabled) |
| 9 | Hadamard-rotated INT4 (E0M3) new precision tier | accuracy | 1-2 weeks, standalone | none | NOT the same dead SM80 kernel OPT-007 already closed -- real native SM100 block-scaled path (same layout family as `nvfp4`); the only item here that could beat `nvfp4` on accuracy |
| 10 | Jetson clock-locking check for benchmark scripts | deployment | hours | none | Pi0.5's own devfreq/nvpmodel check; ImageWAM benchmarks currently have none |
| 11 | Precision-routing contract test (stubbed, no GPU) | deployment | hours -- 1 day | none | mirrors Pi0.5's `test_pi05_thor_fp4_routing.py` pattern against ImageWAM's own `_PRECISIONS`/`_wrap_linear` |
| 12 | ABI integration (`frt_model_runtime_v1`, `io="python"` producer mode) | deployment | 1-3 days | none | zero C++, zero new kernels -- exposes ImageWAM's existing `self._graph`/buffers through the same generic ABI Pi0.5's Python producer already uses |
| 13 | Fidelity + latency CI/regression gate harness | deployment | days | 7 | gate logic (cosine thresholds, `p50<baseline-margin`, JSON result schema) is generic/copyable from Pi0.5's own harness; content needs real calibration data + a real LIBERO fixture format; ready-made baseline: this session's own real 231.6ms `nvfp4` `infer()` number |
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

Measured on H100 (sm_90, shared GPU). Each item's OPT entry holds the
details. "Default" says whether the served configuration changed.

| id | result | default | record |
|---|---|---|---|
| 1 | Per-shape measured tile choice for ActionDiT NVFP4/FP8 CUTLASS GEMMs, plus four 1-SM FP8 small-M tiles. Compiled for sm_110; correctness and speed not yet measured on Thor. | off (`gemm_variant_autotune=True`) | OPT-018 |
| 2 | Fused uint8→BF16 VAE preprocessing kernel with a 256-entry table. Bit-exact to the previous path. | on | OPT-020 |
| 3 | Gated residual fused with the following AdaLN, including across layer boundaries. Bit-exact over the whole pass. Together with item 4, cuts kernels per pass from 7082 to 4968. | on | OPT-017 |
| 4 | Single-stream `attn_out_proj` + `mlp_down` merged into one `linear2` GEMM. NVFP4 operands are identical to the split path; only the accumulation order changes. | on, except `fp16_cutlass` | OPT-016 |
| 5 | VAE stage capturable in the main graph, plus a native NHWC encoder with fused GroupNorm(+SiLU). VAE stage takes 4.3 ms on H100, against about 12 ms for the torch encoder. | off (`vae_encoder="native"`, `vae_graph_input`) | OPT-021 |
| 6 | FA4 backbone and `mot` attention with a dedicated output buffer and a fallback to cuBLAS on failure. Not run on Thor at the served shapes. | off (`FLASHRT_THOR_FA4=1`, `use_fa4_mot=True`) | OPT-019 |
| 7 | Real `fp8_static` calibration from 64 LIBERO frames (142 sites). Against fp16, actions cos is 0.99997 with the real calibration, vs 0.90 with the placeholder. Resolves ISSUE-001 with a TN FP8 layout on sm_89/sm_90. | opt-in (`calibration_path=`) | OPT-022 |
| 8 | AWQ per-channel scales folded into NVFP4 weights. In simulation, backbone cos goes from 0.99820 to 0.99956. | off (`nvfp4_awq=True`) | OPT-023 |
| 9 | `e0m3_hadamard` precision tier. Simulated 1 − actions cos is 2.97e-4, against 6.71e-4 for `nvfp4`. | off (`precision="e0m3_hadamard"`) | OPT-024 |
| 10 | Jetson clock-state probe, printed by the ImageWAM benchmarks. | on (reporting only) | OPT-025 |
| 11 | CPU-only precision-routing contract test. Eight precision columns. | n/a | OPT-026 |
| 12 | `frt_model_runtime_v1` export (`io="python"`). Bit-exact to `infer()`; parity gates carry mutation tests. | n/a | OPT-028 |
| 13 | LIBERO fidelity and latency gate: fixture v1, runner, per-device baselines. fp16 passes on H100. | n/a | OPT-027 |
| 14 | Native C++ overlay (`io="native"`). Bit-exact; runs a tick without Python. Latency is equal to `io="python"` (4974 vs 4998 graph nodes). | n/a | OPT-029 |
| ISSUE-020 | `text_trim`: each prompt runs at its valid text length, which reproduces official's masked attention. On libero_goal (fp16), vs official, median/min goes from 0.99680/0.92997 to 0.99998/0.99963. H100 `infer()` is about 30% faster. | off (`text_trim=True`) | OPT-030, ISSUE-080 |

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

`scripts/imagewam_thor_validation.sh` runs all steps. Its header lists
the build, environment and artifact-bundle prerequisites. It writes one log
per command and `SUMMARY.txt`. `STEPS="..."` selects steps. The bundle
(`thor_bundle`: gate fixture v1 plus calibration files, with `SHA256SUMS`)
is built on the H100 dev box.

| step | checks | decides |
|---|---|---|
| 0 | commit, clock state, bundle checksums | provenance of every number |
| 1 | full `pytest` suite; Thor-only tests run instead of skipping | correctness of every Thor-only kernel path |
| 2 | served `nvfp4` default vs official; fusion A/B (items 3, 4); LIBERO gate for `nvfp4` and fp16; VAE preprocessing kernel | whether items 2-4 stay default-on; new `nvfp4` latency baseline |
| 3 | `text_trim` off/on on three suites; speed and capture cost; multi-length graph safety at `nvfp4` and `e0m3_hadamard` | `text_trim` default (ISSUE-080) |
| 4 | `e0m3_hadamard` vs `nvfp4`; NVFP4 with and without AWQ; `fp8_static(_cutlass)` with real and placeholder calibration; FP8 TN vs NN; `fp8_static` gate; trimmed `fp8_static_cutlass` | default precision; FP8 layout on Thor |
| 5 | FA4 runtime, real-shape correctness, end to end, attention A/B, FA4 with `text_trim` | FA4 default |
| 6 | VAE encode latency (torch/native, eager/graph); `infer()` A/B; native in-graph end to end | native VAE and in-graph defaults |
| 7 | small-M tile sweep and `infer()` A/B | tile tuner default, or a fixed tile |
| 8 | ABI export and native parity gates with mutants at `nvfp4` | Thor readiness of items 12 and 14 |

## Not planned (confirmed, not cost-gated)

- Multi-subgraph stage-splitting (Pi0.5's RTC-prefix-reuse/VJP-guided-
  denoising scheduling machinery, `flash_rt/subgraphs/pi05/`) --
  ImageWAM has no incremental-replanning requirement today; out of
  scope for lack of a real need, not because it's expensive.
- RMSNorm-into-GEMM prologue fusion -- CUTLASS has no prologue-fusion
  mechanism at all (confirmed earlier this session); Pi0.5 doesn't do
  this either. Dead, unchanged by this investigation.

# Plan: Jetson clock-locking check for the benchmark scripts (roadmap item 10)

Plan Status: approved

## Problem

### Current

The ImageWAM benchmark entry points (`benchmarks/imagewam_thor_graph_bench.py`,
the timing section of `benchmarks/imagewam_e2e_official_compare.py`,
`benchmarks/imagewam_thor_int4_bench.py`, `benchmarks/imagewam_thor_int8_bench.py`)
print latency without recording the Jetson power and clock state it was
measured under. Pi0.5's end-to-end benchmark
(`tests/bench_pi05_decoder_fp4_e2e.py`, `machine_state()`) refuses to run
unless `nvpmodel -q` reports MAXN and the `gpu-gpc-0`/`gpu-nvd-0`
devfreq nodes have `min_freq == max_freq == cur_freq`, and it writes that
state into its result JSON. No ImageWAM latency on record, including the
shipped `nvfp4` P50 of 231.6 ms, carries a clock record.

### Goal

One reusable, read-only helper that reads the nvpmodel mode and
GPU/EMC devfreq `cur/min/max/governor` from sysfs without root; prints
and returns a structured record (power mode; clocks pinned or dynamic);
warns only for a non-MAXN mode or unobservable state; and returns an
explicit "not a Jetson" record on any other machine. It never changes
machine state (no `sudo`, `jetson_clocks` or `nvpmodel -m`): Thor is
shared and runs as is, at MAXN with DVFS-managed clocks. Every listed
benchmark prints this record once before timing. Measurable: unit tests
on x86 against a fake sysfs tree cover the pinned, dynamic,
missing-nvpmodel, EMC and non-Jetson cases; on Thor the record is
captured as the machine is.

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

1. `JetsonClockProbe.read()` checks `/etc/nv_tegra_release` and
   `/proc/device-tree/{model,compatible}`. Neither present: return a
   record with `is_jetson=False` and no tool calls.
2. On a Jetson: run `nvpmodel -q` through the runner (timeout, never
   `sudo`); list `/sys/class/devfreq/*`; classify
   names containing `gpu` (and the Tegra GPU ids `gp10b/gv11b/ga10b/gb10b`)
   as GPU and names containing `emc` as EMC; read `cur_freq`, `min_freq`,
   `max_freq`, `governor`; read `/sys/kernel/nvpmodel_clk_cap/*`.
3. Derive the verdict and warnings.
4. `report_jetson_clock_state()` prints `[jetson-clock-state] <json>`
   plus one `[jetson-clock-state] WARNING: ...` line per warning, and
   returns the record.

## Code Mapping

| item | file |
|---|---|
| probe, record types, runner protocol, reporter | `flash_rt/hardware/jetson_clock_state.py` (new) |
| unit tests | `tests/test_jetson_clock_state.py` (new) |
| wiring | `benchmarks/imagewam_thor_graph_bench.py`, `benchmarks/imagewam_e2e_official_compare.py` (timing section), `benchmarks/imagewam_thor_int4_bench.py`, `benchmarks/imagewam_thor_int8_bench.py`; the item-13 gate runner embeds the record in its result JSON |

## Implementation Phases

### Phase 1 — helper and unit tests

Phase Status: completed

Goal: `jetson_clock_state.py` with the interface above.
Modified files: `flash_rt/hardware/jetson_clock_state.py`, `tests/test_jetson_clock_state.py`.
Observation method: pytest on x86 with fake sysfs trees; the record
printed on this H100 box (expected `is_jetson=false`).

### Phase 2 — wire into the benchmark entry points

Phase Status: completed

Goal: every listed benchmark prints the record once before timing.
Modified files: the four benchmarks listed in Code Mapping.
Observation method: `python -m py_compile` on each; run the graph bench
on H100 far enough to see the record line (fp16 row).

### Phase 3 — Thor handoff

Phase Status: blocked

Goal: Thor checklist entry: the record captured as the machine is.
Modified files: none (checklist in the stream report).
Observation method: owner's Thor run.
Blocker: needs the Thor hardware. The checklist command is
`python -c "from flash_rt.hardware.jetson_clock_state import
report_jetson_clock_state; report_jetson_clock_state()"`, run as the
machine is; on the shared H100 the record is `is_jetson=false`.

# Plan: Fidelity and latency regression gate harness (roadmap item 13)

Plan Status: approved

## Problem

### Current

The repository has no committed gate for ImageWAM. Fidelity is checked
ad hoc with `benchmarks/imagewam_e2e_official_compare.py`, which loads
the official bf16 model (Qwen3-4B included) next to FlashRT in the same
process (about 34GB) and needs `av`/`pandas` for LIBERO decoding, none
of which a Thor gate run should depend on. Latency claims (`nvfp4` 231.6 ms, `opportunities.md`
OPT-015) are recorded in prose, with no machine-readable baseline and no
pass/fail rule. Pi0.5's harness (`tests/bench_pi05_decoder_fp4_e2e.py`)
has the generic pieces: per-sample cosine thresholds,
`p50` against a regression baseline, and a versioned result JSON with the
clock state. Roadmap item 13 lists item 7 (real calibration data) as a
dependency; only the `fp8_static` gate needs it.

### Goal

1. A versioned LIBERO fixture: real preprocessed observations (two
   224x224 views, proprio, prompt), the official Qwen3 context and mask,
   fixed initial action noise, official reference actions, and FlashRT
   `fp16` reference actions. Data lives under
   `/home/user1/workspace/jingwu/artifacts/deploy-gates/`; git holds the
   generator and a manifest with checksums. Gating on Thor needs neither
   the official model nor Qwen3.
2. A gate runner that, for one precision, checks fidelity against the
   official reference and the FlashRT `fp16` reference, and latency
   against a per-device baseline JSON (Thor `nvfp4` seeded at 231.6 ms;
   H100 latency ungated).
3. An `fp8_static` slot that activates when a calibration file exists,
   with a documented hand-off interface and no dependency on the
   calibration stream's code.

Measurable: on H100, `fp16` against the fixture reproduces the
end-to-end baseline (`fr_vs_off` median 0.99840, min 0.99567; mean
`mae_fr_vs_gt` 0.18359) and passes.

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
@dataclass(frozen=True) class LatencyBaseline:  p50_ms: float; margin: float; source: str
@dataclass(frozen=True) class DeviceLatencyPolicy: device: str; gated: bool; reason: str; baselines: dict[str, LatencyBaseline]
@dataclass(frozen=True) class GateCheck:        name: str; status: "pass"|"fail"|"ungated"|"skipped"; value; limit; detail
class FidelityGate:  evaluate(vs_official, vs_fp16_reference, mae_mean, reference_mae_mean, all_finite) -> list[GateCheck]
class LatencyGate:   evaluate(precision, summary) -> GateCheck   # p50 < baseline*(1+margin)
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

Generator (H100, once per fixture version):
1. `load_samples()` from the end-to-end script (env `SUITE`, `N_TASKS`,
   `FRAMES`, `SEEDS`); `center_crop_resize` both views to 224x224.
2. Official model: per task `_prepare_flux2_infer_text` gives context and
   mask; per sample and seed, noise emulated exactly as the official
   sampler draws it (CPU generator, bf16 round trip), then
   `infer_action_flux2(..., seed)` gives the official normalized actions.
3. Free the official model; construct FlashRT `fp16` (real checkpoint,
   AE, dataset stats, no Qwen3); per sample and seed, `set_prompt(context)`
   and `infer(obs, action_noise=noise)`; store the renormalized actions.
4. `GateFixtureStore.save` writes `fixture.npz` and the manifest.

Runner (any CUDA device):
1. Load and verify the fixture against the committed manifest.
2. Resolve fidelity thresholds for the precision; apply the `fp8_static`
   contract above.
3. Read the clock state (item 10) and the device policy.
4. Construct the frontend (no Qwen3); per sample and seed run served
   `infer(obs, action_noise=noise)`; cosine in normalized action space
   against the official and `fp16` references; MAE against ground truth
   in real units.
5. Latency: served `infer(obs)` with default noise, warmup then timed
   iterations (`time.perf_counter` around each call; `infer()`
   synchronizes).
6. Evaluate, write `result.json`, print one `__IMAGEWAM_GATE__ <json>`
   line, exit.

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

Goal: `regression_gate.py`, `imagewam_gate_fixture.py`, the two config
JSON files, unit tests.
Modified files: those files and `tests/test_imagewam_regression_gate.py`.
Observation method: pytest on CPU; tamper test (flip one byte of a
fixture array) must fail verification.

### Phase 2 — explicit initial-noise hook in `infer()`

Phase Status: completed

Goal: `infer(observation, *, action_noise=None)`; default path
unchanged.
Modified files: `flash_rt/frontends/torch/imagewam_thor.py`.
Observation method: on H100 fp16 real checkpoint, `infer(obs,
action_noise=n)` equals the end-to-end script's
`flashrt_infer_with_noise` (bit-exact after denormalization);
regression suite unchanged.

### Phase 3 — fixture generator, fixture v1 on H100

Phase Status: completed

Goal: `imagewam_libero_gate_v1` (libero_spatial, 10 tasks, frames 0 and
60, seeds 0 and 1) generated; manifest committed.
Modified files: `benchmarks/imagewam_gate_fixture_generate.py`,
`tests/fixtures/imagewam_gate/imagewam_libero_gate_v1.manifest.json`.
Observation method: generator prints per-sample official-vs-fp16
cosine; its summary must reproduce the end-to-end baseline.

### Phase 4 — gate runner, real fp16 gate on H100

Phase Status: completed

Goal: `tests/gate_imagewam_libero.py`; real `fp16` run on H100 passes
fidelity with latency ungated; `fp8_static` without a calibration file
reports `skipped`; `nvfp4` on H100 fails at construction with the
existing clear NVFP4 build error.
Modified files: `tests/gate_imagewam_libero.py`.
Observation method: result JSON values next to the end-to-end baseline.

### Phase 5 — Thor handoff

Phase Status: blocked

Goal: Thor checklist: copy fixture, verify checksums, run `nvfp4` and
`fp16`, report result JSON.
Modified files: none.
Observation method: owner's Thor run.
Blocker: needs the Thor hardware and a copy of the v1 fixture there.

# Plan: ActionDiT small-M CUTLASS tile selection (roadmap item 1)

Plan Status: approved

## Problem

### Current

Every ActionDiT weight GEMM runs at `M = num_action = 64`. The tile
variant for the two CUTLASS-backed quantized precisions is chosen by
an `(N, K)`-only heuristic that was tuned for other shapes:

- `nvfp4` (shipped default): `Nvfp4Linear` calls
  `flash_rt.executors.fp4_utils.fp4_gemm` without a variant, so
  `pick_variant(N, K)` applies. That table was calibrated for Pi0.5's
  encoder at `M = 968`.
- `fp8_static_cutlass`: `StaticFp8Linear(use_cutlass=True)` uses
  `_pick_fp8_cutlass_variant(N, K)`, which picks `wide` when
  `N >= 4K` and `sq` otherwise. It is a provisional guess ported
  from backbone shapes.

Inventory at the real ActionDiT shapes (`M = 64`,
`action_hidden_dim = 1024`, `action_attn_width = 3072`,
`action_mlp_hidden = 4096`, `action_dim = 7`, 5 double and 20 single
layers):

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

At `M = 64` one output tile row covers the whole M extent, so the CTA
count equals the number of N tiles. On Thor's 20 SMs, the `N = 1024`
GEMMs launch 4 CTAs under v6 (N tile 256), and 4 useful CTA pairs
under FP8 `sq` with a 2x2 cluster. Those GEMMs are weight-bandwidth
bound (arithmetic intensity 2M = 128 FLOP per weight element), so a
launch that occupies 4 of 20 SMs cannot reach DRAM bandwidth. There
are 50 such calls per denoise step and 500 per `infer()`. FP8 CUTLASS
has no tile narrower than 128 in N and no 1-SM (cluster 1x1x1) tile at
all. On Thor it measured 1.44-1.68x slower than cuBLASLt at this M
(opportunities.md OPT-014, result 3).

Pi0.5 runs its decoder (`M = 10`) on the narrow-N v10 tile
(`128x64x256`, cluster 1x1x1) for all four projections
(`docs/pi05_thor_decoder_fp4_e2e.md`, "Decoder v10 Tiles"). v10 is
already instantiated in `cutlass_fp4_gemm_variants.cu`, but ImageWAM
never selects it.

### Problem

No ActionDiT GEMM tile choice is measured at `M = 64`. The existing
choices are extrapolated from other shapes, and FP8 CUTLASS has no
small-M tile to choose.

### Measurable goal

- A per-shape tile choice for the ActionDiT GEMMs, made by a one-time
  measurement at construction on the device the frontend runs on,
  cached per `(family, M, N, K)`. The candidate set includes the
  current heuristic pick, and a candidate must reproduce the current
  pick's output before it can be selected.
- FP8 small-M 1-SM tiles, with the Pi0.5 v10 tile `128x64x256` as the
  template.
- Selection logic unit-tested with the kernels stubbed.
- A Thor script that sweeps every candidate at every real ActionDiT
  shape, reports cosine against fp16 and the per-shape winner, and
  runs an `infer()` A/B of old vs new selection on `nvfp4` and
  `fp8_static_cutlass`.
- The shipped default selection stays unchanged (opt-in flag) until
  Thor confirms correctness and speed.

## Structure

- NEW `flash_rt/models/imagewam/gemm_variant_tuner.py`: owns the
  selection policy (candidate filtering, correctness gate, timing
  comparison, hysteresis against the incumbent) and the per-frontend
  result cache. Defines the `VariantTunableGemm` and `VariantTimer`
  protocols and the result dataclasses. No CUDA code.
- NEW `flash_rt/models/imagewam/gemm_variant_timer.py`: owns the
  device timing mechanism (`CudaGraphVariantTimer`: CUDA-graph
  capture of a launch batch, replay timed with CUDA events).
- `flash_rt/models/imagewam/quant_linear.py`: `Nvfp4Linear` and
  `StaticFp8Linear(use_cutlass=True)` implement `VariantTunableGemm`.
  Each linear owns its own current variant. The default variant equals
  today's heuristic pick, so `__call__` is unchanged until a variant is
  set.
- `csrc/gemm/gemm_types_sm100.h`, `csrc/gemm/cutlass_sm100.cu`,
  `csrc/bindings.cpp`: four new FP8 1-SM tiles (cluster 1x1x1):
  `t128x64x256` (v10 template), `t128x64x128`, `t128x128x128`,
  `t128x256x128`.
- `flash_rt/frontends/torch/imagewam_thor.py`: owns the decision to
  tune (`gemm_variant_autotune: bool = False`), the grouping of
  ActionDiT linears by shape, and the tuner instance and its results
  (`gemm_variant_results`).
- NEW `benchmarks/imagewam_thor_small_m_tile_sweep.py`: Thor sweep
  and `infer()` A/B.
- NEW `tests/test_imagewam_gemm_variant_tuner.py`: stubbed selection
  tests (CPU) and a real-timer test (any CUDA GPU).

State ownership:

| state | owner |
|---|---|
| current tile variant of one linear | that `Nvfp4Linear` / `StaticFp8Linear` instance |
| tuning results cache `(family, M, N, K) -> result` | the `GemmVariantTuner` instance owned by the frontend |
| whether tuning runs | frontend constructor argument |

## Interface

```python
# flash_rt/models/imagewam/gemm_variant_tuner.py
@dataclass(frozen=True)
class GemmShape:
    m: int
    n: int
    k: int

@dataclass(frozen=True)
class VariantMeasurement:
    variant: str
    us_per_gemm: float | None      # None: not timed (rejected)
    cosine_vs_default: float | None
    status: str                     # "ok" | "launch_failed rc=.." | "mismatch" | "nonfinite"

@dataclass(frozen=True)
class VariantTuneResult:
    family: str
    shape: GemmShape
    members: int
    default_variant: str
    chosen_variant: str
    measurements: tuple[VariantMeasurement, ...]

class VariantTunableGemm(Protocol):
    family: str
    n: int
    k: int
    default_variant: str
    variant: str
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

Selection rule, per group of linears sharing `(family, M, N, K)`:

1. Stage the same random input into every member.
2. For every candidate, launch it once per member eagerly. Record
   `launch_failed` for a nonzero return code or a raised Python
   exception, and `nonfinite` or `mismatch` when its output against the
   default variant's output on the same member is non-finite or below
   `cosine_floor`. The default variant failing or raising is an error.
3. Time each surviving candidate as one launch per member, round
   robin, so each launch reads a different layer's weight and the
   timing does not run from a warm L2. The batch is captured in one
   CUDA graph so launch overhead is excluded.
   A candidate the timer cannot capture is `timing_failed`.
4. Choose the fastest candidate. Keep the default unless the winner is
   faster by more than `min_gain` (2%), or if the default itself could
   not be timed.
5. Apply the choice to every member and cache it.

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

- Goal: selection policy and device timer, independent of any kernel.
- Files: `gemm_variant_tuner.py`, `gemm_variant_timer.py`,
  `tests/test_imagewam_gemm_variant_tuner.py`.
- Observation: stubbed tests cover argmin choice, hysteresis, launch
  failure, mismatch rejection, nonfinite rejection, default failing,
  every candidate failing, the cache, and group application. On H100,
  a real-timer test tunes two real sm_90 kernels, and its printed
  per-launch times are compared with a direct CUDA-event measurement.

### Phase 2: FP8 1-SM small-M tiles

Phase Status: completed

- Goal: `cutlass_fp8_t128x64x256`, `_t128x64x128`, `_t128x128x128`,
  `_t128x256x128` exported under `ENABLE_SM100_CUTLASS`.
- Files: `csrc/gemm/gemm_types_sm100.h`, `csrc/gemm/cutlass_sm100.cu`,
  `csrc/bindings.cpp`.
- Observation: `sm110_check.sh` passes, and the sm_90 build is
  unaffected. Correctness and speed are Thor checklist items.

### Phase 3: quant_linear protocol implementation

Phase Status: completed

- Goal: `Nvfp4Linear` and `StaticFp8Linear(use_cutlass=True)` expose
  `family`, `default_variant`, `variant`, `candidate_variants()`,
  `set_variant()`, `prepare_tuning_input()`, and `launch_variant()`.
  Default behavior stays unchanged.
- Files: `flash_rt/models/imagewam/quant_linear.py`.
- Observation: the regression suite is unchanged. Construction on
  H100 still raises the same `RuntimeError`.

### Phase 4: frontend wiring

Phase Status: completed

- Goal: `gemm_variant_autotune` flag, grouping ActionDiT linears by
  shape, and `gemm_variant_results`.
- Files: `flash_rt/frontends/torch/imagewam_thor.py`, tests.
- Observation: a routing test with stubbed linear classes confirms
  that only ActionDiT groups are tuned, with `m = num_action`, one
  tune per distinct shape, and the chosen variant applied to every
  member. With the flag off, nothing changes. The fp16 path and the
  regression suite are unchanged.

### Phase 5: Thor sweep and A/B script, handoff

Phase Status: completed

- Goal: `benchmarks/imagewam_thor_small_m_tile_sweep.py`, plus
  results and Thor checklist in `opportunities.md` OPT-018.
- Observation: the script's non-Thor paths (argument parsing, shape
  table, cuBLASLt fp16 reference timing) run on H100 and print SKIP
  for families this build lacks.

### Phase 6: candidates that raise are rejected

Phase Status: completed

- Goal: a Python exception from a candidate's launch, such as an
  `AttributeError` for a `cutlass_fp8_t128x*` symbol missing from a
  stale build, rejects that candidate instead of aborting construction.
  A batch the timer cannot capture is rejected as `timing_failed`.
- Files: `gemm_variant_tuner.py`, `gemm_variant_timer.py`, tests, both
  benchmarks.
- Observation: stub tests cover a raising candidate, a raising default
  (still an error), an untimeable candidate, and an untimeable default
  (kept). The timer test feeds a raising batch and a
  capture-invalidating batch. A routing test removes the `t128x*`
  symbols and construction completes.

### Phase 7: Thor confirmation

Phase Status: blocked

- Goal: the Thor checklist in opportunities.md OPT-018. The tile
  sweep must show correct outputs for every tile, and the
  `infer()` A/B of heuristic vs tuned tiles must show action cosine
  >= 0.9999 and a P50 delta.
- Blocker: no sm_110 device on the dev box. SM100 CUTLASS and NVFP4
  kernels do not run on sm_90, which only compile-checks them
  (`sm110_check.sh`). Recorded as issues.md ISSUE-023.

# Plan: attention-chain fusion recheck at ImageWAM's real shapes (roadmap item 6)

Plan Status: approved

## Problem

### Current

Both attention sites run a cuBLAS-composed chain in
`ImageWAMAttnBackend.run()` (`flash_rt/hardware/thor/attn_backend.py`):
a strided-batched QK^T GEMM into a `logits` buffer, a softmax kernel,
then a strided-batched PV GEMM (`fvk.attention_qkv_fp16_perhead`,
`csrc/kernels/attention_cublas.cuh`). The frontend always constructs
the backend with `use_perhead_kv=True, use_real_mot_mask=True`.

- `"backbone"` site: prefill self-attention, 25 layers, `q = kv = a0 =
  905` tokens, 24 heads, HD 128, no mask. FA4 is wired in as
  `use_fa4=True` (opportunities.md OPT-005). On Thor it measured
  cosine 1.000000 against the cuBLAS chain, 3.75x per call at the real
  per-head shape, and -10.5% prefill in the per-layer benchmark. The
  frontend default is `use_fa4=False`, because the previous dev box had
  no FA4 runtime and `use_fa4=True` raises when the runtime is missing.
- `"mot"` site: ActionDiT joint attention, 25 layers x 10 steps = 250
  calls per `infer()`, `q = 64` action queries over `kv = total = 969`
  keys. With `use_real_mot_mask=True`, the rule the frontend always
  uses, the call is unmasked attention through the same
  `attention_qkv_fp16_perhead`. Upstream `_build_mot_attention_mask_flux2`
  with `target_len = 0` removes only the region mask, but it still
  excludes padded text keys for every query row, at the prefill call
  and at the action call. `pipeline_thor.py` models that mask at
  neither site (issues.md ISSUE-020). FA4 has never been evaluated
  here.
- Pi0.5 rejected a fused SIMT attention chain at decoder `M = 10`,
  HD 256, as 5-7x slower (`docs/pi05_thor_decoder_fp4_e2e.md`). At that
  shape the QK^T/PV GEMMs are about 1 us of tensor-core work, and FA4
  has no KV-split path at HD 256.

### Problem

Nobody has measured which share of ImageWAM prefill and denoise time
attention takes at the real shapes. The Thor FA4 win is not on by
default. The `mot` site's fused-kernel eligibility has never been
evaluated.

### Measurable goal

- H100, indicative only: the attention share of prefill and of one
  denoise step at the real shapes, measured in-graph as graph time with
  the real attention minus graph time with attention removed. Also
  per-call cuBLAS chain vs fused kernels available on sm_90 (PyTorch
  SDPA flash / cuDNN / mem-efficient) at both sites' shapes, with
  cosine against the cuBLAS chain.
- Recommendation with evidence per site.
- If cheap: a Thor FA4 switch for `"backbone"` that resolves to the
  cuBLAS chain when the FA4 runtime is missing or the device is not
  Thor. It stays opt-in (`FLASHRT_THOR_FA4=1`) until Thor confirms FA4
  at the served shapes. Also an opt-in FA4 path for `"mot"`. Dispatch
  logic verified locally against the cuBLAS chain with a
  reference-backed FA4 stand-in. FA4 itself goes on the Thor
  checklist.

## Structure

- NEW `benchmarks/imagewam_attention_share_bench.py`: owns the
  measurements (in-graph attention share; per-call chain vs fused
  kernels; on Thor it also times FA4 per call, with `num_splits` swept
  for the `mot` shape).
- `flash_rt/hardware/thor/attn_backend.py`: `ImageWAMAttnBackend`
  owns per-site kernel dispatch. It gains `use_fa4_mot: bool` for the
  `"mot"` site FA4 branch. That branch is valid only with
  `use_real_mot_mask=True` and `use_perhead_kv=True`, FlashRT's
  unmasked per-head rule, and the constructor rejects any other
  combination.
- `flash_rt/hardware/thor/fa4_backend.py`: owns FA4 availability. It
  gains `thor_default_enabled() -> bool`, true only on an sm_11x
  device with an active FA4 runtime.
- `flash_rt/frontends/torch/imagewam_thor.py`: owns the FA4 output
  buffer, and the FA4-failure fallback in `set_prompt()`
  (`_capture_graph_or_fall_back`): on an exception during warmup or
  capture with FA4 on, it logs, warns, records `fa4_fallback_reason`,
  rebuilds the backend with FA4 off, and captures again. It also owns
  the default.
  `use_fa4: bool | None = None` resolves, through `_resolve_use_fa4`, to
  False unless `FLASHRT_THOR_FA4=1`. With the variable set, it resolves
  to `fa4_backend.thor_default_enabled()`. An explicit `True` still
  requires the runtime, and an explicit `False` forces the cuBLAS
  chain. It also gains `use_fa4_mot: bool = False`, passed through.
- Tests: `tests/test_imagewam_fa4_dispatch.py` (new) checks the
  backend's FA4 branches for both sites against the cuBLAS chain, with
  FA4 replaced by a stand-in that has FA4's `_flash_attn_fwd`
  signature and computes attention as an fp32 matmul-softmax-matmul in
  PyTorch. It also checks the default resolution.
  `tests/test_imagewam_fa4_backbone.py` gains a real-FA4 `mot` case
  that skips without FA4.

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

- Goal: attention share (H100) and per-call chain vs fused kernels at
  real shapes.
- Files: `benchmarks/imagewam_attention_share_bench.py`.
- Observation: printed P10/P50/P90 for graphs with and without
  attention, per stage. Per-call medians for each kernel, with cosine
  against the cuBLAS chain.

### Phase 2: Thor FA4 switch (opt-in), opt-in FA4 for `mot`

Phase Status: completed

- Goal: `use_fa4=None` resolution (opt-in through `FLASHRT_THOR_FA4=1`),
  `use_fa4_mot`.
- Files: `fa4_backend.py`, `attn_backend.py`, `imagewam_thor.py`,
  tests.
- Observation: dispatch tests show the FA4 branches, with an fp32
  matmul stand-in, matching the cuBLAS chain at real shapes (cosine,
  max-abs, rel_l2). Resolution resolves to False on H100. The
  regression count is unchanged apart from the new tests. An fp16
  end-to-end quick run matches the baseline, since the default
  resolves to off on H100.

### Phase 3: recommendation and Thor handoff

Phase Status: completed

- Goal: OPT-019 with evidence, recommendation, and Thor checks.
- Files: `opportunities.md`.

### Phase 4: FA4 back to opt-in

Phase Status: completed

- Goal: `use_fa4=None` resolves to the cuBLAS chain on every device.
  `FLASHRT_THOR_FA4=1` opts in, and making FA4 the default is a
  one-line change (`_FA4_OPT_IN_DEFAULT`).
- Files: `imagewam_thor.py`, `tests/test_imagewam_fa4_dispatch.py`.
- Observation: resolution tests cover every combination of the
  environment variable, runtime availability, and explicit argument.

### Phase 5: dedicated FA4 output buffer

Phase Status: completed

- Goal: FA4 output goes to `fa4_out` slots, which the frontend owns as
  `(total, hidden)`, instead of `logits`, which overruns at small dims.
- Files: `attn_backend.py`, `imagewam_thor.py`, FA4 tests, FA4 benches.
- Observation: guard-band tests after `fa4_out`, and on `logits`, at
  (a0, total) = (8, 12), (8, 24), and (905, 969), plus a frontend range
  check. Both fail on the old staging. Capacity is checked at
  construction.

### Phase 6: fall back to the cuBLAS chain when FA4 fails

Phase Status: completed

- Goal: an FA4 failure during `set_prompt()`'s warmup or capture logs,
  warns, records `fa4_fallback_reason`, rebuilds the backend without
  FA4, and captures again.
- Files: `imagewam_thor.py`, `tests/test_imagewam_fa4_dispatch.py`.
- Observation: stand-ins that fail at first call, inside capture, and
  by invalidating the capture all recover to the chain's output with
  the caller's stream restored. A failure with FA4 off still raises.

### Phase 7: Thor confirmation

Phase Status: blocked

- Goal: the Thor checklist in opportunities.md OPT-019, with FA4
  explicitly opted in:
  - the real-FA4 real-shape test at 905/905 and 64/969;
  - kernel timings;
  - an nvfp4 end-to-end official compare with FA4 on vs off;
  - an `infer()` A/B with FA4 on vs off.
- Blocker: no sm_110 device and no FA4 runtime on the dev box (issues.md
  ISSUE-023).

# Plan: configuration consolidation (workload / precision / profile)

Plan Status: approved

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
| R5 | `text_trim=True` together with a consumer that needs one fixed graph (`runtime_surface`, `pipeline_resources`, ABI export, native runtime) | `_refuse_text_trim` (`imagewam_thor.py:1779`); stays a refusal until ISSUE-080 condition 5 |
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
  of `captured_text_lengths()`) and capturing them once at construction.
- Runtime identity: `runtime_surface` `setup_identity` describes the
  workload explicitly, as `workload.<field>` entries beside the existing
  `dims.<key>` entries. The field list is
  `runtime_surface.WORKLOAD_IDENTITY_FIELDS` (`num_views`, `image_h`,
  `image_w`, `text_max_len`, `action_horizon`, `action_dim`,
  `proprio_dim`, `num_steps`, `shift`) and
  `runtime_surface.workload_identity(workload)` renders the pairs, so the
  ABI identity, the native path and the tests read one list. The entries
  are additive and descriptive: no stored calibration file is invalidated
  by them, and `calibration_file.IDENTITY_DIM_KEYS` keeps its current key
  set (a new key there would refuse every file recorded before it, which is
  an owner decision, not part of W10).

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

Work items are grouped into a dependency graph so independent ones can be
done in parallel. `W` items are code; `T` items are Thor runs; `S` is a
side track that does not block the rest.

```
W0 freeze the interface (this section: Workload / Structure /
   Precision / Options / rule ids)
 |
 +--> W1 Workload: layout derivation + validation ---------------+
 +--> W2 Structure constants (checkpoint tensors + config.yaml) -+
 +--> W3 Precision enum + property table -----------------------+
 +--> W4 resolve_config: profiles + rules R1-R8 (pure Python) ---+  (needs W1, W3 types)
 +--> W5 tests: workload, profile table, rule ids (CPU) --------+   (extends routing test)
                                                                 |
                                   join                          v
                                    W6 frontend: build dims from ResolvedConfig
                                     |     (from_config, constructor kept)
                                     v
                                    W7 frontend: wire Precision + resolver
                                     |     (replace tuples / fallback rules)
                        +------------+-------------+
                        v            v             v
                       W8 hand-    W9 derive /    W10 ABI + native identity
                       copied      demote flags   carries the workload
                       dims        (vae_graph_    (runtime_export, surface,
                       -> import   input, fa4_mot, pipeline_resources,
                                   tuner, fusion) native_*, calibration_file)
                        |            |             |
                        +------------+-------------+
                                     v
                                    W11 public entry load_imagewam +
                                        expert-constructor split
                                        (needs T4: preset contents)
                                     v
                                    W12 Thor validation on the new entry

Thor line (independent of W0-W10; T4 feeds W11 and the profile contents):
 T1 THOR_CHECKLIST A (data trustworthy) -> T2 B (missing data)
   -> T3 C (configuration matrix) -> T4 decide preset contents
   -> T5 FA4 / native VAE default decision (THOR_CHECKLIST C criteria)

Side track S (blocks R5 becoming a non-refusal, not the rest):
 S1 ISSUE-080 condition 4: fixture v2 with trim (fp16 reference)
 S2 ISSUE-080 condition 5: runtime surface / ABI / native for per-length graphs
 S3 ISSUE-080 condition 6: bounded per-length graph cache + startup use of
    `precapture_text_lengths`
```

Parallelism:

- W1, W2, W3 are independent files and can proceed in parallel after W0.
  W4 needs the types from W1 and W3 but not their implementations, so it
  starts once W0 is frozen. W5 tests are written against W0 and run when
  W1-W4 land.
- W6 and W7 both edit the frontend constructor (`imagewam_thor.py`) and
  are serial, one owner. To keep the seam small, all logic lives in the
  new pure-Python modules; the constructor change is only "take resolved
  dims and options".
- W8, W9, W10 touch disjoint files after W7 and run in parallel. W8 is
  mechanical (import instead of literal), W9 changes which flags are
  public, W10 changes the identity that ABI and calibration files check.
- W11 waits for T4 because the `fast` profile contents are a measured
  decision; the mechanism (W0-W10) does not wait for it.
- T1-T5 run on Thor independently of the code work; they are the source
  of the profile contents.

### Phase W0: interface freeze
Phase Status: completed
- Goal: this section's Interface and rule table are the contract; any
  change to a name here is a plan edit.
- Modified files: `plan.md` only.
- Observation: review of this section.

### Phase W1: `ImageWAMWorkload` and layout
Phase Status: completed
- Goal: derive `x0`, `img_len`, `a0`, `total`, `ref_h`, `ref_w`, `dt`,
  `vae_graph_input` from the workload; reject inconsistent input.
- First step: confirm from `vae_stage.py`, `pipeline_real.py` and
  `rope.py:build_img_ids` how multiple views form the `ref_h x ref_w`
  grid (LIBERO's two 224x224 views give 14 x 28, so views concatenate
  along the width); the derivation must reproduce 513/905/969 and 14 x 28
  exactly for `libero()`.
- Modified files: `flash_rt/models/imagewam/workload.py` (new).
- New structures: `ImageWAMWorkload`, `SequenceLayout`.
- Affected modules: none yet (not wired).
- Observation: `tests/test_imagewam_workload.py` (CPU): `libero()` layout
  equals `LIBERO_REAL_DIMS`; invalid inputs raise.

### Phase W2: `ImageWAMStructure`
Phase Status: completed
- Goal: one place that reads backbone dims from checkpoint tensor shapes
  and the rest from `config.yaml`, with `toy()` for tests.
- Modified files: `flash_rt/models/imagewam/structure.py` (new).
- Affected modules: `checkpoint_loader.py` (reads only).
- Observation: on the local checkpoint, `from_checkpoint()` matches the
  backbone values in `LIBERO_REAL_DIMS`; `toy()` matches `_DEFAULT_DIMS`.

### Phase W3: `Precision` property table
Phase Status: completed
- Goal: precision properties as data. The alignment rules reproduce
  today's `_wrap_linear` fallbacks (fp16_cutlass n/k % 8, nvfp4 % 16,
  fp8/fp8_static/fp8_static_cutlass % 8 -> fp16 linear).
- Modified files: `flash_rt/models/imagewam/precision.py` (new).
- Observation: unit test compares the property table with
  `_PRECISIONS`, `_NVFP4_PRECISIONS`, `_STATIC_FP8_PRECISIONS`,
  `_VARIANT_TUNED_PRECISIONS`.

### Phase W4: `resolve_config`, profiles, rules
Phase Status: completed
- Goal: rules R1-R8 in one function with ids; `effective_config` string
  identical in format to the current compare-script line.
- Modified files: `flash_rt/models/imagewam/config_resolver.py` (new).
- Observation: one test per rule id (legal and illegal case); the
  `default` profile reproduces today's constructor defaults.

### Phase W5: tests
Phase Status: completed
- Goal: pin profiles and rules.
- Modified files: `tests/test_imagewam_thor_precision_routing.py`
  (extend), `tests/test_imagewam_workload.py` (new).
- Observation: CPU-only, no GPU needed; run locally.

### Phase W6: frontend builds dims from `ResolvedConfig`
Phase Status: completed
- Goal: `from_config` constructs the frontend from resolved dims and
  options; the constructor signature and behaviour with `dims_override`
  stay.
- Modified files: `flash_rt/frontends/torch/imagewam_thor.py`.
- Observation: `libero()` through `from_config` is bit-identical to the
  old constructor with `LIBERO_REAL_DIMS` (fp16 and nvfp4, existing
  `test_imagewam_thor_real_wiring.py` plus a new equality check).

### Phase W7: frontend uses `Precision` and the resolver
Phase Status: completed
- Goal: delete the precision tuples and scattered checks the resolver now
  owns; `_wrap_linear` reads `Precision` properties.
- Modified files: `flash_rt/frontends/torch/imagewam_thor.py`.
- Observation: `test_imagewam_thor_precision_routing.py` unchanged
  results (`EXPECTED_ROUTING` still holds).

### Phase W8: replace hand-copied dims
Phase Status: completed
- Goal: the 20 literal copies import the workload/`libero_dims`; toy dims
  stay toy and are labelled.
- Modified files: the list in Code Mapping.
- Observation: `grep -rE "x0\s*=\s*513"` returns only `libero_dims.py` and
  the workload test; every touched script still imports and its
  `--help`/collect step works.

### Phase W9: derive and demote flags
Phase Status: completed
- Goal: `vae_graph_input` derived from the workload; `use_fa4_mot` folded
  into the FA4 tier of a profile; tuner, AWQ tuning and fusion flags
  remain constructor-only.
- Modified files: `imagewam_thor.py`, `config_resolver.py`,
  `benchmarks/imagewam_fusion_ab.py` (uses the fusion flags).
- Observation: existing callers of these flags still work; resolved
  `effective_config` shows them.

### Phase W10: ABI and native identity carry the workload
Phase Status: completed
- Goal: `identity` (ABI) and the calibration file identity include the
  workload fields, so a runtime and a calibration file for a different
  workload are rejected by name.
- Modified files: `runtime_export.py`, `runtime_surface.py`,
  `pipeline_resources.py`, `native_resources.py`, `native_runtime.py`,
  `calibration_file.py`, `cpp/models/imagewam` (identity fields only).
- Observation: `gate_imagewam_model_runtime_export.py` and
  `gate_imagewam_native_schema_parity.py`; a mismatched-workload
  calibration file is refused with the differing field named.

### Phase W11: public entry and constructor split
Phase Status: completed
- Goal: `load_imagewam(ckpt_path, workload, profile=..., precision=...,
  calibration_path=...)` is the deployment entry; expert switches go
  through `**expert` into `resolve_config`, not through positional
  constructor arguments. The `fast` profile contents come from T4.
- Modified files: `imagewam_thor.py`, `config_resolver.py`,
  `THOR_STATUS_SUMMARY.md` (options table).
- Observation: profile test pins the contents; the compare script builds
  its frontend through `load_imagewam`.

### Phase W12: Thor validation on the new entry
Phase Status: completed
- Closed: the blocker was the VAE encode geometry (ISSUE-086), fixed and
  verified on Thor at `0919e` — the target workload serves on all three paths
  (`default` `infer()` 216.93 / ABI 173.55 / native 173.27 ms; `profile=fast`
  with precaptured lengths 137.64 / 139.35 / native refused by R5; the trimmed
  sweep at 16 / 72 / 128 valid tokens 197.00 / 207.06 / 217.50 ms on `infer()`
  and 153.51 / 161.68 / 172.05 ms on the ABI), and the LIBERO rows are
  fidelity-identical to the previous round (`libero_spatial` nvfp4 `default`:
  vs official min 0.99418 / median 0.99764, MAE 0.18290, P50 202.6 ms).
- Result on Thor at `eccf14f` (Jetson AGX Thor, MAXN, GPC 1.575 GHz,
  `emc_locked=null`, GPU exclusive, logs under `/home/jingwu/thor_val/0919s/`):
  the recorded numbers reproduce through the new entry. The gate measures
  202.2 ms nvfp4 (pass) and the end-to-end `default` row 202.4 / 202.1 /
  202.0 ms over three repeats on libero_spatial, against the recorded
  203.3 ms gate and 202.3 ms e2e — the previous round's 225 ms and 13.6 ms
  spread were that session's state, not a scope difference (ISSUE-082
  resolved). The same three repeats spread 0.4 ms (`default`) and 0.5 ms
  (`stack`), and no row fell back from FA4.
- The recorded `stack` / `fast` value of 106.1 ms is likewise session-stale:
  the same switch set measures 92.6-93.7 ms across libero_goal and
  libero_10, and 92.8-93.3 ms over three repeats on libero_spatial. The
  latency gate's Thor baseline is re-based on the new session
  (`tests/fixtures/imagewam_gate/latency_baselines.json`, 231.6 -> 202.2 ms),
  which closes item E2.
- Goal: the recorded numbers reproduce through the new path, and the
  target workload (THOR_CHECKLIST D) runs as a `Workload`.
- Modified files: `scripts/imagewam_thor_matrix.sh`,
  `benchmarks/imagewam_e2e_official_compare.py`, `THOR_CHECKLIST.md`.
- Observation: matrix rows `default` and `stack` within run-to-run noise
  of the recorded 203.3 / 106.1 ms (same commit conditions, GPU
  exclusivity recorded), `vs official` not below recorded values; the
  target workload table filled.
- Result on Thor at `c20f3a0` (Jetson AGX Thor, MAXN, DVFS-managed
  clocks, logs under `/home/jingwu/thor_val/c20f3a0`):
  - The two identity checks pass. `effective_config` is character for
    character what `config_resolver.format_effective_config` produces for
    the same resolved configuration, for `default` and for `fast`; the
    exported runtime identity carries all nine `workload.<field>` entries
    with `ImageWAMWorkload.libero()`'s values, its ABI is bit-exact
    against `infer()`, and the native schema matches the golden records.
    `THOR_CHECKLIST.md`'s C1-C3 items are closed and removed.
  - `profile=fast` measured 106.8 ms (`vs official` median 0.99936)
    against the recorded 106.1 ms, and the ladder's `stack` row (the same
    switches) 93.2 ms; no row fell back from FA4. `profile=default`
    measured 225.2 ms and the ladder's `default` row 225.5 ms, against a
    recorded 203.3 ms that is a gate number (the frontend alone) while
    the matrix is the end-to-end compare; the same session's
    frontend-only ABI path measured 226 ms. Whether the default path is
    slower on this commit is therefore not decided by this round
    (ISSUE-082), and the ladder's switch marginals carry an in-session
    spread of 13.6 ms between two instances of the same configuration.
  - The switch ladder and the FA4 / native-VAE criterion outcomes are
    recorded in `opportunities.md` OPT-019, OPT-021 and OPT-030. Section
    C's 2 ms working threshold and the gate baseline need the two
    measurements item E2 now spells out.
  - Still open in this phase: the like-for-like `default` baseline, and
    running the target workload of `THOR_CHECKLIST.md` section D as an
    `ImageWAMWorkload`.

### Phase T1-T5: Thor line
Phase Status: pending
- Goal: T1-T3 are THOR_CHECKLIST sections A, B, C; T4 records the preset
  contents in `plan.md` "Decisions"; T5 applies the FA4 and native-VAE
  default criteria in THOR_CHECKLIST C.
- Modified files: `THOR_CHECKLIST.md` (finished items removed),
  `opportunities.md`, `issues.md`.
- Observation: matrix CSV/MD per THOR_CHECKLIST.

### Phase S1: a gate fixture whose `fp16` reference is recorded trimmed
Phase Status: completed
- Goal: ISSUE-080 condition 4. Fixture v1's `fp16` reference is untrimmed, so
  a trimmed `fp16` run measures 0.99837 / 0.99579 against it, below the fp16
  bounds 0.999 / 0.995, while being closer to official (0.99998 / 0.99992):
  a trimmed configuration cannot pass the gate until a fixture recorded
  trimmed exists.
- Modified files: `benchmarks/imagewam_gate_fixture_generate.py` (the
  `TEXT_TRIM` / `--text-trim` switch, the reference built through
  `load_imagewam`), `flash_rt/datasets/imagewam_gate_fixture.py`
  (`ImageWAMGateFixture.text_trim`, `FixtureManifest.text_trim`, absent means
  untrimmed so v1 still loads), `tests/gate_imagewam_libero.py` (`--text-trim`;
  a fixture recorded with the other value is refused before the checkpoint
  hash), `tests/fixtures/imagewam_gate/fidelity_thresholds.json` (the
  like-for-like rule stated, numbers unchanged),
  `tests/test_imagewam_regression_gate.py`.
- Observation: CPU round trip through `save`/`load` and the v1 manifest, the
  refusal of either mismatch, and the generator's imports resolving against
  the compare script (the check that caught the generator's own `REAL_DIMS`
  breakage). On Thor the v2 fixture was generated (`text_trim=true`) and the
  trimmed nvfp4 gate passes against it: vs official 0.99931 / min 0.99898,
  vs the fixture's fp16 reference 0.99935 / 0.99907, P50 114.6 ms; a trimmed
  run against v1 and an untrimmed run against v2 are both refused. The
  fixture's manifest still has to be committed (checklist item E2).

### Phase S2: the ABI carries one graph per trimmed length
Phase Status: completed
- Goal: ISSUE-080 condition 5. The exec layer already models a graph as a
  `ShapeKey -> graph-exec` variant table with an LRU cap, so a trimmed
  frontend's per-length graphs become keys of one declared graph, and `step`
  replays the key of the length the prompt set. This is what lifts rule R5
  for `consumer="abi"`; the native pipeline keeps its refusal until its own
  phase.
- Modified files: `flash_rt/models/imagewam/runtime_surface.py`,
  `runtime_export.py`, `flash_rt/frontends/torch/imagewam_thor.py`,
  `flash_rt/models/imagewam/config_resolver.py` (the R5 scope), their tests.
- Interface: `runtime_surface()` exposes `graph_variants` (`active_key`, the
  `(key, graph_exec)` entries ascending, `per_prompt_length`), the `ShapeKey`
  is the context length `x0`, `graph_variant_plan` turns the table into the
  declaration's `default_key` / `keys` / `max_variants` (pure, CPU-tested),
  the manifest records `text_lengths`, and `step` replays
  `active_dims["x0"]` after `frt_graph_has_variant`, refusing an uncaptured
  length by name. `setup_identity` gains the `text_trim` pair, so the export
  fingerprint of an existing untrimmed deployment changes (no artifact
  compatibility is required at this stage).
- Observation: the pure table/plan seam and the resolver's R5 scope are
  covered on CPU. On Thor the multi-length check passes (the ABI tick at two
  captured lengths is bit-exact against `infer()`, and `guards` 7 tests
  pass), and the served numbers come out: LIBERO `default` `infer()` 202.3 ms
  against ABI 184.2 ms and native 183.8 ms in one process, and with
  `profile=fast` plus precaptured lengths `infer()` 93.2 ms against ABI
  95.1 ms. The native face keeps refusing trimmed prompts; its own phase is
  S4.

### Phase S4: the native pipeline carries one graph per trimmed length
Phase Status: pending
- Goal: ISSUE-080 condition 5's remaining half. The ABI face serves trimmed
  prompts now; the native C++ pipeline refuses them explicitly at both entry
  points (`ImageWAMNativeRuntime.create`, `export_model_runtime(io="native")`).
- What it needs (assessed while implementing S2, all of it C-ABI + host work,
  none of it Python-only): `NativeRuntime` holds one `graph_`, one
  `owned_graph_` for the `capture()` path and one `context_rows_` (used by
  `set_proprio_row`'s bound and `set_pipeline`'s dims check), so the keyed
  variant table has to replace them (`frt_graph` already offers
  adopt/has_variant/replay); the host needs a visible key
  (`use_graph(key, exec)` / `has_variant(key)` / `set_text_length(key)`, wired
  into the prompt/proprio verb path, since C++ cannot see the Python prompt);
  `native_resources.build_io_config` hands the per-key `context_rows`; and
  `native_schema.cpp` plus the schema-parity gate gain the new records.
- Modified files: `cpp/models/imagewam/**`, `flash_rt/models/imagewam/native_runtime.py`,
  `native_resources.py`, `runtime_export.py`, the native gates.
- Observation: a native tick at two prompt lengths, bit-exact against
  `infer()`, next to the ABI row that S2 already added.

### Phase S3: a bounded per-length graph cache, precaptured at construction
Phase Status: completed
- Goal: ISSUE-080 condition 6; a trimmed frontend captured one graph per
  distinct length with no bound, and `precapture_text_lengths` existed but
  nothing called it.
- Modified files: `flash_rt/frontends/torch/imagewam_thor.py` (`evict_lru`,
  the `text_trim_cache_size` constructor keyword, `_touch_capture` /
  `_store_capture`, the recency updates, the `precapture_text_lengths` keyword
  on `from_config`/`load_imagewam`), `flash_rt/models/imagewam/config_resolver.py`
  (`text_trim_cache_size` in `ImageWAMOptions` and `EXPERT_KEYS`, default 32,
  rule V1), `tests/test_imagewam_text_trim_cache.py`.
- Observation: `evict_lru` (order, the active capture never dropped, a bound
  of 1, a cache whose only entry is active), the resolver option and its V1
  cases, the constructor validation before any allocation, and the precapture
  keyword reaching `precapture_text_lengths` once through a mocked
  constructor. On Thor a precaptured length switches in 0.000-0.012 s where a
  length captured on first use takes 0.42-0.58 s, over three swept lengths
  (16, 24, 31 valid tokens), and eviction behaves as designed: with the bound
  at 2 and no precapture, revisiting an evicted length captures again
  (0.636 / 0.503 / 0.468 / 0.465 s for 16 -> 24 -> 31 -> 16). The memory
  measurement corrected this phase's own assumption: the first captured graph
  costs +218.0 MiB reserved / +206.3 MiB allocated and every following one
  +0.0 / +0.1 MiB (the captures share the pool), so 15 lengths sit under one
  graph's fixed cost and the default bound of 32 is on the order of 221 MiB,
  not 32 x 218 MiB.
- Two decisions made while implementing: `precapture_text_lengths` refuses a
  request with more distinct lengths than the bound (its refill loop would
  otherwise recapture what eviction had just dropped, forever), and the cache
  may stay above the bound when the only remaining candidate is the active
  capture (that is what keeps the replayable graph alive).

## Execution record

Observed on the development machine that ran W1-W11 (WSL2, `torch.cuda.is_available()`
is `False`, `nvidia-smi` reports the GPU blocked by the operating system).
Every observation below is therefore a CPU-side contract check; nothing that
needs a CUDA device has been run, and no latency or accuracy number is
claimed here.

| Phase | Observation | Result |
|---|---|---|
| W0 | this section reviewed, names frozen | completed; the addendum above holds the names added while executing |
| W1, W2, W3, W5 | earlier session | committed before this record; `tests/test_imagewam_workload.py`, `tests/test_imagewam_structure.py`, `tests/test_imagewam_precision_table.py` |
| W2 (addendum) | `ImageWAMStructure.libero()` against the real checkpoint | `from_checkpoint(/home/ljw/projects/pi0.5/models/imagewam_flux2_4b_libero/model.pt) == ImageWAMStructure.libero()`, `missing/extra` none; the checkpoint exists on this machine, so this is a real read, not a skip |
| W4 | one legal and one illegal case per rule id | `tests/test_imagewam_config_resolver.py` |
| W6 | `frontend_kwargs_from_config` maps every option onto the constructor keyword the constructor declares; `from_config` passes exactly that mapping | `tests/test_imagewam_frontend_from_config.py` (the frontend itself is not constructed: it allocates CUDA). The bit-identical fp16/nvfp4 comparison of `from_config` against the old constructor is part of the W12 Thor run |
| W7 | the frontend's four precision tuples are comprehensions over `Precision`, `_wrap_linear` takes its fallback from `alignment_fallback`, no precision literal survives in it | `tests/test_imagewam_precision_table.py` (values pinned literally, delegation checked by `ast` over the frontend source); `tests/test_imagewam_thor_precision_routing.py` `EXPECTED_ROUTING` unchanged |
| W8 | no served-dims literal outside `libero_dims.py` and its tests | `grep -rnE "x0\s*=\s*513"` over `benchmarks/`, `tests/`, `flash_rt/` returns four hits: the literal pin table in `tests/test_imagewam_workload.py` and three docstrings; no module redefines the dims. Benchmark modules import and their argparse `--help` runs (three of them need `pandas`, absent from this machine's `.venv`) |
| W9 | `vae_graph_input` and `use_fa4_mot` come from the resolved options | `tests/test_imagewam_frontend_from_config.py` |
| W10 | `workload_identity()` renders the nine `workload.<field>` pairs; `runtime_surface()` adds them for a frontend built from a resolved configuration | `tests/test_imagewam_public_entry.py`; the identity on a real exported runtime is the W12 Thor item |
| W11 | `load_imagewam` signature, `ConfigError` before construction, structure read only when given | `tests/test_imagewam_public_entry.py` |
| W12 | run on Thor at `c20f3a0`; the identity checks pass and `fast`/`stack` reproduce their recorded values, `default`'s like-for-like baseline is open | see the phase's own "Result on Thor" block, `THOR_CHECKLIST.md` item E2 and ISSUE-082 |

The CPU test set used for every phase above, green at the last commit of
this work (242 passed, 3 skipped; the skips are the CUDA guards):

```
.venv/bin/python -m pytest tests/test_imagewam_workload.py tests/test_imagewam_structure.py \
  tests/test_imagewam_config_resolver.py tests/test_imagewam_precision_table.py \
  tests/test_imagewam_thor_precision_routing.py tests/test_imagewam_text_trim_consumer_guards.py \
  tests/test_imagewam_frontend_from_config.py tests/test_imagewam_public_entry.py -q
```

`pytest tests/test_imagewam_*.py --collect-only -q` collects 542 without
errors; the five directory-wide collection errors are pre-existing and
outside ImageWAM (`flash_rt.flash_rt_fp4`, `_flashrt_exec`, `ml_dtypes` are
not built or installed on this machine).

Not done, and why:

- T1-T5 and the rest of W12 are Thor runs; the Thor is a separate shared
  machine. `THOR_CHECKLIST.md` carries the commands, each item's 判据 and
  where its conclusion goes, so one pass covers what is still pending. The
  `c20f3a0` round closed the new-entry checks and the nvfp4 `libero_spatial`
  ladder; sections A and B, the remaining C rows, D and E2 are not run.
- S1-S3 are untouched: S1 needs the fixture data regenerated on a GPU (the
  generator's own `fp16` reference), S2 and S3 change the runtime surface
  and the capture cache, whose only observation is a captured graph. Their
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

Open:

- Whether `text_trim` becomes the default of the `default` profile for the
  Python `infer()` and ABI paths. Everything it needs is verified: the
  multi-length safety check passes with FA4 off (4 tests), the ABI tick at two
  captured lengths is bit-exact, the trimmed gate fixture exists and passes,
  the cache is bounded and precaptured, and the per-length memory cost is one
  shared pool. The technical blockers left are the FA4-**on** recovery test
  (ISSUE-085) and the native face's own phase S4. Changing a profile is a plan
  edit and is the owner's call.
- Profile contents (T4): keep `fast` as `text_trim` + FA4 (backbone and mot)
  + native VAE in graph. The FA4 criterion was met in the `c20f3a0` round
  (`stack` 9.7 ms below `vae_trim`, agreement with official not worse) and
  so was the native-VAE criterion; `text_trim` remains the largest single
  step on the LIBERO workload.
- Nothing is blocking `text_trim` as the default now; what is left is code
  in the S phases: S1 (a gate fixture whose `fp16` reference is recorded
  trimmed, so a trimmed configuration can pass the regression gate), S2 (the
  ABI and the native pipeline carry one graph per trimmed length, which also
  lifts rule R5), S3 (a bounded per-length cache filled by
  `precapture_text_lengths` at startup). With `text_max_len=128` the trimmed
  sequence saves at most 112 of about 897 backbone rows, so the gain there is
  a fraction of the LIBERO one (ISSUE-083) — the trim is still the right
  default, it is just a smaller number on that workload.
- Whether the calibration file's identity gains the workload fields
  (`num_views`, `image_h`, `image_w`): with no compatibility requirement
  this is a format version bump and a re-recorded file. It closes the one
  collision the dims-derived identity has — two workloads with the same
  `ref_h`/`ref_w` (e.g. two 224x224 views and four 224x112 views) have the
  same dims but different VAE inputs.
- The target-workload table in `THOR_CHECKLIST.md` section D: `num_views=3`,
  `image_h=image_w=256`, `action_horizon=32`, `action_dim=7`,
  `proprio_dim=8`, `num_steps=10`, `shift=5.0`; instruction tokens 16-128,
  and the buffer length they imply (`text_max_len`) is the open field
  (ISSUE-083). The checkpoint and calibration files and the graph memory
  budget are still open.
