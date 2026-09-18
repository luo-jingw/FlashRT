# ImageWAM Model Runtime (`frt_model_runtime_v1`, `io="python"`)

`ImageWAMTorchFrontendThor.export_model_runtime()` publishes the captured
ImageWAM graph through the generic model-runtime ABI
([`model_runtime_api.md`](model_runtime_api.md)). The producer is the
Python frontend; any consumer that speaks the ABI (a C++/Rust host loop,
a capsule/state host) drives the model without Python-level knowledge of
ImageWAM.

## Ownership

- `flash_rt/frontends/torch/imagewam_thor.py` owns the weights, the
  captured graph, its capture stream, every device buffer, and the
  per-tick staging operations: `stage_images` (the configured
  preprocessing kernel and VAE encoder into `img_raw`, or, with
  `vae_graph_input`, a copy into the graph's uint8 view buffer),
  `stage_proprio` (state normalization and `proprio_encoder` projection
  into the context row), `read_actions` (action denormalization and
  readback). `infer()` is these operations plus the initial noise
  (`action_noise`, or `0.01 * N(0,1)`) and one graph replay.
- `flash_rt/models/imagewam/runtime_surface.py` declares the interface
  between the two: `ImageWAMRuntimeSurface` (graph exec, stream, the
  `img_raw` / `context` / `action_latent` windows, setup identity) and the
  `ImageWAMRuntimeSource` Protocol.
- `flash_rt/models/imagewam/runtime_export.py` builds the runtime: one
  exec context wrapping the capture stream, the torch graph exec adopted
  (not owned) as graph `infer`, the device windows wrapped (not owned),
  ports, one stage, one region, identity, and the Python verbs. The
  verbs call the frontend's staging operations on the capture stream.
  The runtime anchors the frontend for its lifetime.

## Port schema

Declaration order is the port index. Ports whose condition is false are
omitted; the others keep this relative order.

| port | dir | update | modality | dtype | shape | window | declared when |
|---|---|---|---|---|---|---|---|
| `images` | in | STAGED | IMAGE | u8 | (views, H, W, 3) | none | VAE loaded |
| `image_tokens` | in | SWAP | TENSOR | bf16 | (392, 128) | `img_raw` | VAE outside the graph |
| `image_views` | in | SWAP | IMAGE | u8 | (views, H, W, 3) | `views_u8` | VAE inside the graph |
| `proprio` | in | STAGED | STATE | f32 | (8,) | none | `dims["proprio_dim"]` set |
| `noise` | in | SWAP | TENSOR | f32 | (64, 7) | `action_latent` | always |
| `actions` | out | STAGED | ACTION | f32 | (64, 7) | none | always |
| `actions_raw` | out | SWAP | TENSOR | f32 | (64, 7) | `action_latent` | always |
| `prompt` | in | SETUP | TEXT | u8 | (-1,) | none | Qwen3 loaded |

Shapes are for the LIBERO release (`img_len=392`, `HD=128`,
`num_action=64`, `action_dim=7`, `proprio_dim=8`); they follow the
frontend's `dims`. `(views, H, W)` is `(2, 224, 224)` with the VAE outside
the graph and the frontend's `vae_graph_input` with the VAE inside it.

- `images`: `views` `frt_image_view`, RGB8, `W x H`, any row stride, in
  view order `view1` (agent view), `view2` (wrist view). `set_input` runs
  `stage_images`. With the VAE outside the graph the tokens land in
  `img_raw`, which is also exposed raw as `image_tokens` for a host that
  brings its own VAE tokens. With the VAE inside the graph the frames land
  in the graph's uint8 view buffer, which is also exposed raw as
  `image_views`; the graph writes `img_raw` itself, so `image_tokens` is
  not declared.
- `proprio`: raw robot state, f32. `set_input` applies the dataset
  `state` min/max normalization (when `dataset_stats_path` was given),
  the `proprio_encoder` linear, and writes the row `set_prompt()` reserved.
- `noise`: the initial action latent, the ABI form of
  `infer(..., action_noise=)`. The graph reads it exactly as written and
  integrates it in place, so after `step` the same window holds the
  normalized chunk (`actions_raw`). The host rewrites it before every
  `step`. Without `action_noise`, `infer()` fills it with
  `0.01 * N(0,1)`; the ABI applies no scale of its own (see `issues.md`
  ISSUE-002 for that factor).
- `actions`: what `infer()` returns, the chunk denormalized with the
  dataset `action` min/max when loaded, else the raw latent. The identity
  record `action_denormalized` states which.
- `prompt`: SETUP, legal only outside a tick. `set_input` runs
  `set_prompt(text)`: Qwen3 encoding outside the graph into the context
  buffer. It never recaptures (the graph already exists at export).
- Stage plan: one GRAPH stage `infer` (backbone prefill plus the 10-step
  denoise loop). Region: `rollout_boundary` = the `action_latent` window.
- Buffers: `img_raw` (input), `context` (input, state), `action_latent`
  (input, output), and `views_u8` (input) with the VAE inside the graph.
- Identity: `model=imagewam`, the frontend class, `precision`, the
  resolved `use_fa4` / `use_fa4_mot`, `calibration` (first 16 hex chars
  of the calibration file's SHA-256, or `none`), `nvfp4_awq` (with
  `awq_alpha` / `awq_scope` when on), `vae_resize`, `vae_encoder`,
  `vae_graph_input`, every `dims` entry, `io`, `graph_producer`, `views`,
  `vae_in_graph`, `has_vae`, `has_text_encoder`, `proprio_dim`,
  `action_denormalized`, then caller pairs (production callers pass a
  weights digest).

## Verbs and ordering

`set_input` / `get_output` / `step` are Python callables behind the
builder's GIL-acquiring trampolines, callable from any host thread. A
tick is not atomic: calls on one runtime must not overlap (the host
serializes set_input, step and get_output, one tick at a time), and
`last_error` is per runtime, valid until its next verb call.
STAGED verbs and `step` run on the exported stream (`streams[0]`, the
capture stream). A host that writes or reads a SWAP window enqueues the
copy on that stream (`native_handle`) or synchronizes it first. Status
codes, the same on the `io="native"` face: `-2` unknown port index, `-3`
`set_input` or `get_output` on a SWAP port, `-4` wrong payload size (or
image geometry), `-5` short `get_output` buffer (`written` = needed
bytes), `-1` any other invalid call (a stream that is not the exported
one, a malformed image view); the reason is in `last_error`. The verbs
raise `VerbStatusError` to set these codes.

## Build and use

```bash
cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$(which python) \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build exec/build -j
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$(which python) \
  -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build runtime/build -j
```

```python
fe = ImageWAMTorchFrontendThor(precision=..., dims_override=..., ckpt_path=...,
                               ae_model_path=..., flux2_src=...,
                               qwen3_model_spec=..., dataset_stats_path=...)
fe.set_prompt("pick up the black bowl ...")          # captures the graph
rt = fe.export_model_runtime(identity={"weights_sha256": digest})
# hand rt.ptr (frt_model_runtime_v1*) to the native consumer; rt.release() when done
```

## Verification

Before every ABI tick the tests and the gate NaN-fill each buffer the
tick must write or refresh (`img_raw` or the in-graph uint8 views, the
proprio row or the whole context, K/V caches, `Q_O`, backbone residual,
action latent), and each parity row is re-run with the verb it exercises
made a no-op (`tests/_helpers/imagewam_abi_checks.py`); every such mutant
must make its row fail.

- `tests/test_imagewam_model_runtime_export.py`: small random-weight
  dims. Schema, identity (including sensitivity to caller pairs), status
  codes equal to the `io="native"` face's (`EXPECTED_STATUSES`), and a
  poisoned ctypes consumer tick that is `array_equal` to `infer()`; the
  proprio-verb, `step` and unwritten-`image_tokens` mutants fail. Skips
  when `exec/build` or `runtime/build` is missing.
- `tests/test_imagewam_model_runtime_vae.py`: the real AE at the real
  token count (random-weight backbone), both VAE placements; the poisoned
  `images` STAGED path and (VAE inside the graph) the `image_views` SWAP
  path are `array_equal` to `infer()`, and an `images` no-op fails.
- `tests/gate_imagewam_model_runtime_export.py`: real checkpoint, VAE,
  Qwen3 and dataset stats. On H100 at `fp16` with a real LIBERO frame,
  with the VAE outside and inside the graph (`--vae-graph-input 224 224`),
  the poisoned ABI tick is bit-identical to `infer()` for the VAE tokens
  the `images` port stages, the denormalized `actions`, `actions_raw`,
  the `image_tokens` / `image_views` SWAP path, and the `prompt` SETUP
  path (all `array_equal`, `max_abs = 0`). All five mutants are detected
  in both placements (images, proprio on each image path, prompt, step).
  On Thor the gate runs with `--precision nvfp4` and has not been run yet.
