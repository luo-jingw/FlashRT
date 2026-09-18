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
| CUDA stream, captured graph exec, `GemmRunner` and cuBLAS/cuBLASLt handles and workspaces, proprio staging scratch, host copies of the min/max constants | the native handle (`frt_imagewam_native`, refcounted) |

The native library never frees a borrowed pointer. `ImageWAMNativeRuntime`
holds the runtime surface, whose `owner` is the frontend, and the source
of its installed pipeline; the model runtime anchors the frontend through
the declaration's Python owner and retains the native handle through the
verb override. Borrowed memory therefore outlives every verb call, with or
without an export. Replacing the pipeline (`set_pipeline`) destroys the
graph captured from the previous one; while a model runtime over the
handle is live, `use_graph`, `set_pipeline` and `capture` are refused.
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
`image_tokens`. FA4 (`use_fa4` / `use_fa4_mot`) is not available to the
native pipeline, which uses the cuBLAS-decomposed per-head attention
(`attention_qkv_fp16_perhead`). Supported precisions: `fp16` and `nvfp4`
(the merged single-stream `linear1` path), with either layer structure:
the served one (single-stream `linear2` as one GEMM, each gated residual
fused with the following AdaLN) or the split/unfused one
(`merge_linear2` / `fuse_res_norm` off).

## Module map

| file | responsibility |
|---|---|
| `cpp/models/imagewam/include/flashrt/cpp/models/imagewam/c_api.h` | C ABI: IO config, pipeline config (resource table), handle lifetime, verbs, setup calls |
| `cpp/models/imagewam/src/native_runtime.{h,cpp}` | verbs, declaration check, graph ownership, eager segments, capture |
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
(the `action_latent` window); one GRAPH stage `infer`. The declaration's
stream is the native stream and its graph is the one the native `step`
replays. Identity adds `io=native` and `graph_producer` (`python` when the
native verbs replay the frontend's graph, `native` after `capture`).
`images` and `prompt` are not declared: their transforms (VAE, Qwen3) run
in Python. The prompt is set through the frontend in setup, followed by
`ImageWAMNativeRuntime.set_proprio_row(runtime_surface().proprio_row)`.
The canonical records are pinned in
`tests/data/imagewam_native_schema.records`.

Verbs:

- `set_input(proprio)`: host normalization in fp32 (`x*scale`, then
  `+offset`, clamp to ±5), bf16 rounding, asynchronous copy, then one
  cuBLASLt bf16 GEMM with bias epilogue (descriptors and algorithm
  created at setup) written into the context row, on the native stream.
- `get_output(actions)`: device-to-host copy on the native stream,
  synchronize, host `(x - offset) / scale`.
- `step`: `cudaGraphLaunch` on the native stream.
- Status codes, the same as on the `io="python"` face: `-2` unknown
  port, `-3` SWAP port passed to `set_input` / `get_output`, `-4` wrong
  proprio payload size, `-5` short output buffer, `-1` a stream that is
  not the native stream or a proprio write before `set_proprio_row`; the
  message is in `last_error`. `tests/_helpers/imagewam_abi_checks.py`
  (`EXPECTED_STATUSES`) pins the table for both faces.

## Native pipeline

`ImageWAMNativeRuntime.set_pipeline(frontend)` passes
`frontend.pipeline_resources()`: dimensions and the two layer-structure
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
native stream (thread-local capture mode) into a graph the handle owns.
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
native = ImageWAMNativeRuntime.create(fe.runtime_surface())
native.set_pipeline(fe)          # resource table + GEMM algorithm hand-off
native.capture()                 # native graph
rt = fe.export_model_runtime(io="native", native=native, identity={...})
# hand rt.ptr to the host; set_input/get_output/step are C functions
```

## Verification

On H100, fp16:

- `tests/test_imagewam_native_pipeline.py` (small random dims, both
  layer structures): one backbone double-stream block, one single-stream
  block, the full prefill, the full denoise loop, and the captured native
  graph are `array_equal` to the Python pipeline in every state buffer
  (`backbone_hidden`, K/V caches, `Q_O`, `action_latent`); the
  `io="native"` tick on the native graph is `array_equal` to `infer()`.
- `tests/test_imagewam_native_runtime.py`: the Python declaration's
  records equal the C++ records; a tick through the C verbs enters no
  Python function (the `io="python"` face enters 57); status codes.
- `tests/gate_imagewam_native_schema_parity.py`: at the real dims the
  Python declaration, the C++ records and the golden file are identical.
- `tests/gate_imagewam_native_parity.py --graph native`: with the real
  checkpoint and the served layer structure, the native graph (4974
  nodes; the Python graph has 4998) and the `io="native"` tick, including
  the native proprio token, are `array_equal` to the Python graph and to
  `infer()`.
