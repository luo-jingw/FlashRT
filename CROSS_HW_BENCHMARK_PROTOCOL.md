# Cross-hardware Pi0.5 / ImageWAM benchmark protocol

Result schema: `docs/benchmark_result_schema.json` (JSON Schema draft-07).
One JSON file per (model, implementation, hardware, precision) run,
validating against that schema. Read it before running anything -- the
`status` and `config.matches_target` fields exist specifically so a run
that cannot hit the frozen target shape (below) still produces a valid,
comparable result instead of a silent gap.

## Frozen target config

| Field | Value |
|---|---|
| Cameras | 3 |
| Image size | 224x224 |
| ODE / denoise steps | 10 |
| action_dim | 32 |
| action_horizon | 30 |
| Weights | random (no real checkpoint) |

Every run should aim at this exact shape. Where it cannot yet be
produced, run at whatever real shape is available now, set
`config.matches_target: false`, and say what differs in `notes` --
do not wait for every model to reach the frozen shape before recording
anything.

## Per (model, implementation) status

| Model | Implementation | Synthetic-shape construction | Status |
|---|---|---|---|
| ImageWAM | FlashRT | `ImageWAMTorchFrontendThor(checkpoint_dir=None, dims_override={...}, num_views=3, precision=...)` | ready now |
| ImageWAM | official (torch) | `imagewam_official_torch_bench.py` -- confirm it accepts the same shape override before assuming it does | check before running |
| Pi0.5 | FlashRT | needs a generated synthetic checkpoint directory (`scripts/gen_synthetic_pi05_checkpoint.py`) | in progress, not landed yet |
| Pi0.5 | official (torch) | same generated checkpoint, if the loader format is shared (to be confirmed by the same script's own investigation) | in progress, not landed yet |

This file will be updated with exact Pi0.5 commands once the generator
script lands. Do not block ImageWAM's rows on that.

## ImageWAM — ready now

```python
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

frontend = ImageWAMTorchFrontendThor(
    checkpoint_dir=None,                 # random weights, no calibration
    dims_override={
        "action_dim": 32,
        "num_action": 30,                # this is ImageWAM's action_horizon field
        "num_denoise_steps": 10,         # default is 2, must be set explicitly
    },
    num_views=3,
    precision="fp16",                    # or bf16/fp8_static/fp8_static_cutlass/nvfp4;
                                          # Orin (rtx_sm87) accepts fp16 only, see OPT-034
)
```

Time `infer()` (or whatever this frontend's existing benchmark harness
already calls -- check `benchmarks/imagewam_thor_graph_bench.py` for the
established warmup/CUDA-event pattern rather than writing a new one)
at `warmup=20, iters=100` on Thor/RTX5090 (CUDA event) and `warmup=10,
iters=50` on Orin (wall clock is acceptable there if CUDA events are
inconvenient -- record `timing.protocol` accordingly either way).

Official torch baseline: `benchmarks/imagewam_official_torch_bench.py`
already exists and is what every "official (bf16)" row in
`docs/pi05_imagewam_benchmark_status.md` came from -- check whether it
takes the same `action_dim`/`num_action`/`num_denoise_steps` override
before assuming it does; if it only runs the real LIBERO shape, say so
in that run's `notes` and set `config.matches_target: false`.

## Where results go

Write each run's JSON to `benchmark_results/<model>_<impl>_<hardware>_<precision>.json`
(new directory, not committed by default -- these are raw per-device
outputs, gather them and send back rather than committing from each
edge device). Example filename: `imagewam_flashrt_thor_nvfp4.json`.

## Example instance (fabricated numbers, for shape only)

```json
{
  "schema_version": "1.0",
  "model": "imagewam",
  "implementation": "flashrt",
  "hardware": "thor",
  "precision": "nvfp4",
  "status": "ok",
  "config": {
    "num_views": 3,
    "image_size": 224,
    "num_flow_steps": 10,
    "action_dim": 32,
    "action_horizon": 30,
    "text_tokens": null,
    "random_weights": true,
    "checkpoint": null,
    "matches_target": true
  },
  "timing": {
    "protocol": "cuda_event",
    "warmup": 20,
    "iters": 100,
    "p50_ms": 108.0,
    "p10_ms": 107.6,
    "p90_ms": 108.4,
    "min_ms": 107.3
  },
  "environment": {
    "commit": "c4be567",
    "gpu_name": "Jetson AGX Thor",
    "compute_capability": "11.0",
    "clock_locked": true,
    "gpu_exclusive": true,
    "cuda_version": "13.0",
    "torch_version": "2.10.0",
    "date": "2026-09-28"
  },
  "correctness": {
    "cos_vs_official_median": null,
    "cos_vs_official_min": null
  },
  "notes": "random weights -> no official reference to compare against, correctness fields intentionally null."
}
```

Note: `correctness` is null here because random weights have no
meaningful "official" reference to compare against -- that is expected
and valid for every random-weights row, not a gap. It is only required
to be non-null for a run against a real checkpoint where an official
reference exists.

## Sending results back

Gather every JSON file produced across every device into one directory
and send the whole set back (as files, or pasted). Do not summarize or
average them before sending -- the schema exists so raw results can be
ingested directly into the standing comparison tables
(`docs/pi05_imagewam_benchmark_status.md`) without re-deriving
methodology each time, which is the recurring cost this protocol is
meant to remove.
