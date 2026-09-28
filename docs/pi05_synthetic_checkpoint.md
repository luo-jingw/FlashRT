# Pi0.5 Synthetic Checkpoint Generator

`scripts/gen_synthetic_pi05_checkpoint.py` produces a random-weight,
shape-correct Pi0.5 checkpoint directory for `Pi05TorchFrontendThor`
(`flash_rt/frontends/torch/pi05_thor.py`) and `Pi05TorchFrontendRtx`
(`flash_rt/frontends/torch/pi05_rtx.py`), without a real trained checkpoint.
Weight values are random (`torch.randn * 0.02`, finite FP16); tensor names,
shapes, and dtypes match exactly what both frontends' loaders read.

## Checkpoint format both frontends actually load

Neither frontend reads a `config.json`. Each reads exactly:

- `model.safetensors` — raw HuggingFace-style keys under
  `paligemma_with_expert.paligemma.*` (SigLIP vision tower + PaliGemma
  encoder + shared token embedding) and `paligemma_with_expert.gemma_expert.*`
  (the Gemma-300M action-expert decoder), plus top-level
  `action_in_proj`/`action_out_proj`/`time_mlp_in`/`time_mlp_out`. This is
  confirmed identical between `Pi05TorchFrontendThor._load_weights` (via the
  declarative `_pi05_thor_spec.py` / `_thor_spec_common.py` weight specs) and
  `Pi05TorchFrontendRtx.convert_pi05_safetensors` — both read the same key
  set at the same shapes. OpenPI's own PyTorch eager reference (the official
  baseline throughout this repo's benchmark docs, e.g.
  `docs/pi05_thor_decoder_fp4_e2e.md`) loads the same `model.safetensors`
  file, per that document's own statement that FlashRT and the official
  reference "load the same converted weights".
- `norm_stats.json` (openpi schema: `{"actions": {"q01": [...], "q99":
  [...]}, "state": {...}}`) — read by `flash_rt.core.utils.norm_stats.
  load_norm_stats`; only `actions.q01`/`actions.q99` are actually consumed
  (by `unnormalize_actions`).
- The PaliGemma SentencePiece tokenizer (`paligemma_tokenizer.model`) is a
  separate, global asset resolved outside `checkpoint_dir` entirely (see
  `flash_rt/utils/paligemma_tokenizer.py`) — the generator does not produce
  it.

## Fixed architecture constants (not checkpoint- or config-driven)

`action_dim`, the SigLIP/encoder/decoder layer counts and widths, and the
PaliGemma vocabulary size are Python-level constants both frontends hardcode
directly, not values a checkpoint or config file selects:

| Constant | Value | Source |
|---|---:|---|
| `ACTION_DIM` | 32 | `flash_rt/models/pi05/pipeline_rtx.py` (module constant, "Fixed Pi0.5 model dimensions"); `Pi05TorchFrontendThor._load_weights` hardcodes the same value directly in buffer shapes (`self._ae_action_f32 = torch.empty(Sa, 32, ...)`, `self._g_noise = torch.zeros(Sa, 32, ...)`) and via `action_in_proj`/`action_out_proj`'s on-disk shape |
| SigLIP layers / width | 27 / 1152 / 4304 | `VIS_L` / `VIS_D` / `VIS_H` |
| Encoder layers / width | 18 / 2048 / 16384 | `ENC_L` / `ENC_D` / `ENC_H` (GQA: 8 query heads, 1 KV head, head_dim 256) |
| Decoder layers / width | 18 / 1024 / 4096 | `DEC_L` / `DEC_D` / `DEC_H` (GQA: 8 query heads, 1 KV head, head_dim 256) |
| PaliGemma vocabulary | 257152 | Confirmed via `sentencepiece.SentencePieceProcessor(...).GetPieceSize()` on the real `paligemma_tokenizer.model`; matches the same hardcoded fallback in `flash_rt/frontends/torch/pi0fast.py` |

The generator therefore only supports `--action-dim 32` and raises for any
other value.

## `chunk_size` / `num_flow_steps`: confirmed field names and which frontend honors which value

- **`Pi05TorchFrontendRtx`**: real constructor kwargs `chunk_size` (default
  `CHUNK_SIZE = 10`, attribute `self.chunk_size`) and `num_steps` (default
  `NUM_STEPS_DEFAULT = 10`, attribute `self._num_steps`). Both are fully
  independent of checkpoint content — no `model.safetensors` tensor shape
  depends on either; only runtime buffers and the precomputed decoder-style
  tables (`_precompute_decoder_styles`) scale with them. Passing
  `chunk_size=30, num_steps=10` to the constructor is sufficient; nothing in
  the checkpoint needs to change.
- **`Pi05TorchFrontendThor`**: hardcodes `Sa = 10` (action horizon) and
  `steps = 10` (denoising step count) as literal values inside
  `_load_weights` (attributes `self.Sa`, `self.steps`). Its `__init__`
  signature has no `chunk_size` or `num_steps` parameter, and
  `flash_rt.api.load_model()` only forwards `num_steps` when
  `"num_steps" in inspect.signature(pipe_cls).parameters`, which is `False`
  for this class. There is therefore no checkpoint-side or kwarg-side path
  to run `Pi05TorchFrontendThor` at a chunk size or step count other than
  10 today. The `chunk_size` constructor argument on
  `Pi05ThorPipeline.bind_runtime_export` (`flash_rt/models/pi05/
  pipeline_thor.py`, `bind_runtime_export`) is unrelated metadata for the
  model-runtime export container, not the actual decoder action-chunk
  length.
- **`num_views`**: a real constructor kwarg on both classes, confirmed not
  stored in checkpoint content — every view is processed through the same
  shared vision-tower weights, so no tensor shape scales with it.

The generator accepts `--chunk-size` and `--num-flow-steps` flags and
records the request in an informational `synthetic_checkpoint_metadata.json`
sidecar (not read by either loader), but a value other than 10 only takes
effect by passing it to `Pi05TorchFrontendRtx`'s constructor directly —
`Pi05TorchFrontendThor` silently ignores both and always runs at
chunk_size=10 / num_flow_steps=10, regardless of what the checkpoint or its
metadata sidecar say.

## Usage

```bash
python scripts/gen_synthetic_pi05_checkpoint.py \
    --out /tmp/pi05_synthetic_checkpoint \
    --action-dim 32 --chunk-size 30 --num-flow-steps 10 \
    --num-views 3 --seed 0
```

Produces, under `--out`:

- `model.safetensors` (810 tensors at the constants above, FP16, ~6.3 GiB)
- `norm_stats.json`
- `synthetic_checkpoint_metadata.json` (informational only)

`Pi05TorchFrontendRtx(out_dir, num_views=3, chunk_size=30, num_steps=10)`
constructs from this directory at the requested shape.
`Pi05TorchFrontendThor(out_dir, num_views=3)` constructs from the same
directory but always runs at chunk_size=10 / num_flow_steps=10 per the table
above, regardless of the `--chunk-size 30` used to generate it.

## Verification status

Confirmed on this project's dev machine (RTX 4060 Laptop, Ada SM89, 8 GB
VRAM):

- The generated `model.safetensors` round-trips completely off-GPU: all 810
  keys open, every shape matches the table above, every tensor is finite.
- `Pi05TorchFrontendRtx` reads the file correctly through weight loading and
  FP8 quantization on real CUDA hardware (confirmed by construction
  progressing past both stages before failing at an unrelated, pre-existing
  attention-kernel build gap).
- `Pi05TorchFrontendThor` constructs successfully end to end (including its
  cuBLAS attention fallback).

Full `set_prompt()` + `infer()` output finiteness was **not** confirmed on
this machine for either frontend — both hit pre-existing, unrelated
build/hardware gaps in this specific fork's compiled kernels before
reaching that call. See `issues.md` ISSUE-094 for the exact failure points
and what a real Thor/RTX box with a full kernel build would need to confirm.
