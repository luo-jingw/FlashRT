# ImageWAM Native C++ Runtime (`io="native"`)

`libflashrt_imagewam_native.so` is the native half of ImageWAM's
`frt_model_runtime_v1` face. It supplies the hot-path verbs as C
functions and records the backbone prefill and the ActionDiT denoise loop
in C++ against the existing `csrc` kernels, so a tick runs with no Python
and no GIL. The Python frontend remains the setup producer. The generic
ABI is in [`model_runtime_api.md`](model_runtime_api.md); the Python face
of the same model is in [`imagewam_model_runtime.md`](imagewam_model_runtime.md).

## Ownership

| state | owner |
|---|---|
| weights (incl. NVFP4 packed weights and scales), activation scratch, K/V caches, `img_raw` / `context` / `action_latent`, every pipeline buffer | `ImageWAMTorchFrontendThor` |
| fp16 AdaLN modulation tensors, transposed proprio weight | the Python handoff objects (`pipeline_resources()` result, `native_resources.py`), kept alive by `ImageWAMNativeRuntime` |
| CUDA stream, the captured graph execs (one per text length the handle recorded or adopted), `GemmRunner` and cuBLAS/cuBLASLt handles and workspaces, proprio staging scratch, host copies of the min/max constants | the native handle (`frt_imagewam_native`, refcounted) |

The native library never frees a borrowed pointer. `ImageWAMNativeRuntime`
holds the runtime surface, whose `owner` is the frontend, and the source
of its installed pipeline; the model runtime anchors the frontend through
the declaration's Python owner and retains the native handle through the
verb override. Borrowed memory therefore outlives every verb call, with or
without an export. The handle holds one native pipeline and one
graph per text length: `set_pipeline` installs the pipeline of the key its
resource table carries and selects that key as the active text length —
replacing that key's pipeline and destroying the graph captured for it,
while every other key keeps both. `gemm_shapes`, `set_gemm_algo`, `run` and
`capture` all resolve against the active text length. While a model runtime
over the handle is live, `use_graph` (adoption), `set_pipeline` and
`capture` are refused; `set_text_length` and `set_proprio_row` stay legal,
because they carry the prompt change on the hot path. The native-owned
capture path therefore serves `text_trim` like the adopted table above:
`pipeline_resources()` describes the ACTIVE length, and
`capture_pipeline_text_lengths` installs and captures every length the
frontend has captured, restoring the active length afterwards. No rule
refuses `text_trim`; R6 is what remains specific to the native consumer.
The Python graph and the native graph share the same buffers and must not
run concurrently.

Threading: calls on one handle (verbs, setup calls, `last_error`) must not
overlap; the host serializes them, one tick at a time, from any thread.
The handle has no locks: the verbs share its stream, proprio staging
scratch and error string. `last_error` stays valid until the next call on
the handle. Reference counting is thread-safe. `run` and `capture` first
wait for all prior device work (`cudaDeviceSynchronize`), because the
native stream is non-blocking and not ordered after work other streams
queued on the shared buffers; they must not run while another thread
captures a CUDA graph in global mode. The same contract is in `c_api.h`.

What stays in Python: checkpoint loading and quantization, GEMM autotune,
AdaLN modulation and RoPE precompute, VAE image encoding, Qwen3 prompt
encoding, and building the declaration. The VAE runs outside the native
graph: `io="native"` and the native pipeline refuse a frontend built with
`vae_graph_input`, and the host supplies VAE tokens through
`image_tokens`. The native pipeline also refuses `nvfp4_awq` (it has no
AWQ input-scale fold) and precisions whose weights it has no linear
descriptor for (`nvfp4_sim`, `fp8*`, `e0m3_hadamard`). FA4 (`use_fa4` /
`use_fa4_mot`) is not available to the native pipeline, which uses the cuBLAS-decomposed per-head attention
(`attention_qkv_fp16_perhead`). Precisions (the merged single-stream
`linear1` path): `fp16`, verified on H100; `nvfp4`, compiled for sm_110
(SM100-class CUTLASS builds) and verified on Thor at both captured lengths
(`0920s4`, `0920t`). Both
layer structures are supported: the served one (single-stream `linear2`
as one GEMM, each gated residual fused with the following AdaLN) and the
split/unfused one (`merge_linear2` / `fuse_res_norm` off).

## Module map

| file | responsibility |
|---|---|
| `cpp/models/imagewam/include/flashrt/cpp/models/imagewam/c_api.h` | C ABI: IO config, pipeline config (resource table), handle lifetime, verbs, setup calls |
| `cpp/models/imagewam/src/native_runtime.{h,cpp}` | verbs, declaration check, graph ownership, the graph variant table (one exec per text length), eager segments, capture |
| `cpp/models/imagewam/src/native_pipeline.{h,cpp}` | prefill / denoise recording in `pipeline_thor.py` order |
| `cpp/models/imagewam/src/fp4_linear.{h,cpp}` | NVFP4 linear (SM100-class builds) |
| `cpp/models/imagewam/src/proprio_projection.{h,cpp}` | cached cuBLASLt bf16 GEMM + bias for the proprio token |
| `cpp/models/imagewam/src/io_transforms.{h,cpp}` | host min/max normalization, bf16 rounding, action denormalization (no FP contraction) |
| `cpp/models/imagewam/src/native_schema.{h,cpp}` | the port / region / stage records the verbs implement |
| `cpp/models/imagewam/imagewam_native.cmake` | the `flashrt_imagewam_native` target, included from the root `CMakeLists.txt` |
| `flash_rt/models/imagewam/pipeline_resources.py` | `ImageWAMPipelineResources`, the frontend → native resource interface |
| `flash_rt/models/imagewam/native_library.py` | ctypes mirror of `c_api.h` (layout-checked at load) |
| `flash_rt/models/imagewam/native_resources.py` | fills the config structs from a surface / resources |
| `flash_rt/models/imagewam/native_runtime.py` | `ImageWAMNativeRuntime`, the Python owner of a handle |

## Port schema (`io="native"`)

| port | dir | update | modality | dtype | shape | window |
|---|---|---|---|---|---|---|
| `image_tokens` | in | SWAP | TENSOR | bf16 | (392, 128) | `img_raw` |
| `proprio` | in | STAGED | STATE | f32 | (8,) | none |
| `noise` | in | SWAP | TENSOR | f32 | (64, 7) | `action_latent` |
| `actions` | out | STAGED | ACTION | f32 | (64, 7) | none |
| `actions_raw` | out | SWAP | TENSOR | f32 | (64, 7) | `action_latent` |

Buffers `img_raw`, `context`, `action_latent`; region `rollout_boundary`
(the `action_latent` window); one GRAPH stage `infer` whose graphs are the
handle's, one per text length `x0` (the variant key), with the C `step`
replaying the handle's active length. Identity adds `io=native` and
`graph_producer`, who recorded the active text length's graph: `python`
when the native verbs replay the frontend's exec for it, `native` when
`capture` recorded it. `images` and `prompt` are not declared:
their transforms (VAE, Qwen3) run in Python. The prompt is set through the
frontend in setup, followed by
`ImageWAMNativeRuntime.set_text_length(x0)` and then
`set_proprio_row(runtime_surface().proprio_row)`; a length the handle holds
no graph for is refused by both (`-2`), and lengths are adopted
(`use_graph(key, exec)`) before the export, because adoption is refused
while a model runtime over the handle is live.
The handle's own capture path fills the same table: each length gets one
installed pipeline and one graph this handle recorded, so `graph_producer`
is `native` instead of `python`. The lengths a handle can
serve are declared when it is created (`frt_imagewam_io_config`'s
`num_text_lengths` / `text_lengths`), so a deployment declares or
precaptures them first.
The canonical records are pinned in
`tests/data/imagewam_native_schema.records`.

Verbs:

- `set_input(proprio)`: host normalization in fp32 (`x*scale`, then
  `+offset`, clamp to ±5), bf16 rounding, asynchronous copy, then one
  cuBLASLt bf16 GEMM with bias epilogue (descriptors and algorithm
  created at setup) written into the context row, on the native stream.
- `get_output(actions)`: device-to-host copy on the native stream,
  synchronize, host `(x - offset) / scale`.
- `step`: `cudaGraphLaunch` of the active text length's graph on the native stream; `-2` when the handle holds no graph for it.
- Status codes, the same as on the `io="python"` face: `-2` unknown
  port or a text length with no adopted/captured graph, `-3` SWAP port
  passed to `set_input` / `get_output`, `-4` wrong proprio payload size,
  `-5` short output buffer, `-1` a stream that is not the native stream or
  a proprio write before `set_proprio_row`; the message is in
  `last_error`. `tests/_helpers/imagewam_abi_checks.py`
  (`EXPECTED_STATUSES`) pins the table for both faces.

## Native pipeline

`ImageWAMNativeRuntime.set_pipeline(frontend)` passes
`frontend.pipeline_resources()`. One table describes one context length,
the active one: its sequence dims (`x0`, `a0`, `total`) and its backbone
RoPE table, while the pipeline buffers are the ones the frontend allocated
for the longest declared length and every length's table points at them.
`set_pipeline` installs that key's pipeline and makes it the active text
length. The table carries dimensions and the two layer-structure
flags, every pipeline buffer, the attention pointers (shared `Q_O`,
per-layer K/V at base + layer × stride, logits), RoPE tables, each
weight as a linear descriptor (`fp16_nn`, `bf16_nn` or NVFP4 packed
weight + scales + activation scratch + the CUTLASS tile the op launches),
the QK-norm and bias pointers, and every AdaLN site in two forms: fp16
shift/scale and materialized gate built once with `pipeline_thor`'s own
`_fuse_mod_group` / `_fuse_mod_pair` (the unfused path and the
standalone AdaLN that starts each chain), and the FP32 modulation chunks
the fused `gate_res_ada_layer_norm_*` kernel reads. With `fuse_res_norm`
the native pipeline follows the same chain as `imagewam_prefill` /
`imagewam_denoise_step`: each block's last gated residual writes the
next block's AdaLN, and the last ActionDiT block writes the head's.

For every `fp16_nn` / `bf16_nn` shape the pipeline launches
(`frt_imagewam_native_gemm_shapes`), the frontend's `GemmRunner`
hands over the cuBLASLt algorithm it selected (autotuned at construction)
through `GemmRunner.get_cached_algo` / `set_cached_algo`, so both
pipelines launch the same GEMM kernels. The `csrc` kernels are compiled
into the library from the same sources with the `flash_rt_kernels` CUDA
flags; on SM100-class builds the `flash_rt_fp4` objects are linked in.
`capture()` runs one eager warm-up, then records prefill + denoise on the
native stream (thread-local capture mode) into a graph the handle owns for
the ACTIVE text length, whose pipeline must be installed (`-1` when none is).
The library links with `--no-undefined` and exports only
`frt_imagewam_native_*`.

## Build

```bash
cmake --build build --target flashrt_imagewam_native     # writes flash_rt/libflashrt_imagewam_native.so
```

plus `exec/build` and `runtime/build` as in
[`imagewam_model_runtime.md`](imagewam_model_runtime.md). The target is
`EXCLUDE_FROM_ALL`; it builds for `GPU_ARCH=90` and `GPU_ARCH=110`.

```python
native = ImageWAMNativeRuntime.create(fe.runtime_surface())   # declares fe's text lengths
native.set_pipeline(fe)          # resource table + GEMM algorithm hand-off (one key)
native.capture()                 # native graph of the active text length (the one-key path)
# or, per captured length, one installed pipeline + one graph this handle
# records, which is what serves text_trim: native.capture_pipeline_text_lengths(fe)
# or, per captured length, adopt the frontend's graphs instead of capturing:
#   for key, exec in fe.runtime_surface().graph_variants.entries:
#       native.use_graph(key, exec)
rt = fe.export_model_runtime(io="native", native=native, identity={...})
# hand rt.ptr to the host; set_input/get_output/step are C functions.
# Per prompt: frontend set_prompt, then native.set_text_length(x0), then
# set_proprio_row(surface.proprio_row).
```

## Verification

On H100, fp16. Before every tick and every graph replay the tests and
gates NaN-fill each buffer the tick must write (`img_raw`, the proprio
row, K/V caches, `Q_O`, backbone residual, action latent), and each
parity row is re-run against mutants that must make it fail.

- `tests/test_imagewam_native_pipeline.py` (small random dims, both
  layer structures): one backbone double-stream block, one single-stream
  block, the full prefill, the full denoise loop, and the captured native
  graph are `array_equal` to the Python pipeline in every state buffer,
  and the native graph to the Python graph, `backbone_hidden` included.
  The poisoned `io="native"` tick on the native graph is `array_equal` to
  `infer()` in the actions, the backbone residual and the K/V caches.
  Native pipelines built from a mutated resource table (no backbone
  block, last single-stream block dropped, last denoise step dropped, one
  block fed another block's `linear1` weight) all fail that tick. Also:
  a second `set_pipeline` for a key leaves no graph for that key and frees
  the replaced pipeline's resources; a native handle alone keeps the frontend
  alive (without it the replay faults); an AWQ frontend is refused.
  `test_pipeline_records_one_graph_per_text_length` installs and captures one
  pipeline per captured length of a `text_trim=True` frontend (`x0` 6 and 14),
  records the manifest's `text_lengths` table, and the poisoned `io="native"`
  tick at either length — the shorter one first, which is not the export's
  default key — is `array_equal` to `infer()` at that same length with
  `graph_producer=native`.
- `tests/test_imagewam_text_trim_consumer_guards.py`: the per-length resource
  table and the install loop are pinned without a GPU, over a stub frontend
  and the stub native handle: `pipeline_resources()` describes the active
  length (sequence dims, AdaLN row counts and the backbone RoPE table follow
  it) while the buffers stay the maximal ones, and
  `capture_pipeline_text_lengths`'s call sequence is `set_pipeline` +
  `gemm_shapes` + `set_gemm_algo` + `capture` per length, ascending, with the
  source's active length restored. A real trimmed frontend on that path needs
  a GPU.
- `tests/test_imagewam_native_runtime.py`: the Python declaration's
  records equal the C++ records; the poisoned tick matches `infer()` with
  proprio staged by the frontend or by the native verb, and fails when
  the consumer skips the proprio verb or `step`; a tick through the C
  verbs enters no Python function (the `io="python"` face enters 57);
  status codes equal the `io="python"` face's (`EXPECTED_STATUSES`);
  `use_graph` / `set_pipeline` / `capture` are refused while exported,
  and `set_text_length` is not (it carries the prompt change);
  `test_native_tick_matches_infer_at_every_captured_length` ticks at two
  captured lengths (the shorter first) and each is `array_equal` to
  `infer()` in the actions and the action latent, with the manifest's
  `text_lengths` table; `set_text_length` returns `-2` for a length with no
  graph.
- `tests/gate_imagewam_native_schema_parity.py`: at the real dims the
  Python declaration, the C++ records and the golden file are identical.
- `tests/gate_imagewam_native_parity.py --graph native`: real checkpoint,
  served layer structure. The native graph has 4974 nodes (the Python
  graph 4998) at `fp16` here; the same gate at `nvfp4` on Thor measures
  5324 against 5348 (`0920s4`, `0920t`), so the counts belong to the
  precision and the binary they were measured with. With proprio staged by
  the frontend or by the native verb,
  the poisoned `io="native"` tick's actions, `actions_raw`, native
  proprio token, backbone residual and K/V caches are `array_equal` to
  `infer()`, and a poisoned native-graph replay equals a Python-graph
  replay in the action latent, backbone residual and K/V caches. All six
  mutants are detected (proprio verb or `step` not called, and the four
  resource-table mutants). Without `--graph native` (the native verbs
  replay the Python graph) the same tick rows and the two call mutants
  pass.

Thor (`nvfp4`): `sm110_check.sh` builds `flashrt_imagewam_native` for
sm_110. The io config carries the text-length table and `use_graph` takes
the key, so a **rebuild is required**; a stale library fails at load in
`native_library._check_layout` ("config struct sizes differ from the
ctypes mirror; rebuild the library"). The parity tests and gates above run
there with `IMAGEWAM_NATIVE_PRECISION=nvfp4` and `--precision nvfp4`
(plan.md, "Native C++ overlay", Thor checklist) and pass, with the node
counts unchanged (`0920s4`, `0920t`). The native pipeline's own per-length
capture is the one row of this surface still waiting for its Thor re-run.
