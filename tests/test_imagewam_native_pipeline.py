"""ImageWAM native C++ pipeline vs the Python pipeline, step by step.

Small random-weight dims, fp16 (IMAGEWAM_NATIVE_PRECISION=nvfp4 on Thor).
The native pipeline (cpp/models/imagewam/src/native_pipeline.cpp) records
the same kernels as flash_rt/models/imagewam/pipeline_thor.py over the
frontend's own buffers and weights, with the frontend's autotuned GEMM
algorithms handed off.
Each check restores one input state, runs the Python function eagerly,
restores again, runs the native segment, and compares every state buffer
bit for bit: one backbone block of each type, the full prefill, the full
denoise loop, then the captured native graph against the Python graph
and through the `io="native"` model runtime against `infer()`, with every
buffer the tick writes NaN-filled first. Native pipelines built from a
mutated resource table (no backbone block, last single-stream block or
denoise step dropped, one block fed another block's weight) must fail that
tick. A second `set_pipeline` for a key drops the graph captured for that
key (one pipeline and one graph per text length), and a native handle alone
keeps its frontend alive. With `text_trim=True` the handle's own capture path
carries one pipeline and one graph per captured length
(`capture_pipeline_text_lengths`), and the tick at either length is
`array_equal` to `infer()` at that length — the native mirror of
`test_imagewam_native_runtime.py::test_native_tick_matches_infer_at_every_captured_length`,
with the graphs this handle recorded instead of the frontend's.
Runs the served layer structure (merged single-stream `linear2`,
gated residual fused with the next AdaLN) and the split/unfused one.
Every frontend here states `use_fa4=False`: the native pipeline has no FA4
attention (rule R6), so this path is the cuBLAS chain on every machine and
nothing in this file depends on `FLASHRT_THOR_FA4` or on what the machine
can run.
Skips when exec/, runtime/ or the native library is not built.
"""
import ctypes
import gc
import json
import os
import weakref

import numpy as np
import pytest
import torch

pytest.importorskip("flash_rt.runtime.exec", exc_type=ImportError)
pytest.importorskip("flash_rt.runtime.export", exc_type=ImportError)

import flash_rt.flash_rt_kernels as fvk  # noqa: E402
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor  # noqa: E402
from flash_rt.models.imagewam import native_library as nl  # noqa: E402
from flash_rt.models.imagewam import pipeline_thor  # noqa: E402
from flash_rt.models.imagewam.native_runtime import ImageWAMNativeError, ImageWAMNativeRuntime  # noqa: E402
from _helpers.imagewam_abi_checks import (  # noqa: E402
    PIPELINE_MUTATIONS, MutatedPipelineSource, bits, poison_tick_state)
from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path  # noqa: E402

try:
    LIBRARY = nl.ImageWAMNativeLibrary()
except ImportError as e:  # pragma: no cover - depends on the local build
    pytest.skip(str(e), allow_module_level=True)

PROPRIO_DIM = 8
# fp16 locally; the Thor checklist sets IMAGEWAM_NATIVE_PRECISION=nvfp4.
PRECISION = os.environ.get("IMAGEWAM_NATIVE_PRECISION", "fp16")
STATE_NAMES = ("backbone_hidden", "K_cache", "V_cache", "Q_O", "context", "img_raw", "action_latent")
# Buffers a tick leaves behind besides the actions.
TICK_STATE = ("backbone_hidden", "K_cache", "V_cache")
# (name, dims_override): the frontend default, and the path with both
# layer-structure fusions off.
LAYER_STRUCTURES = (
    ("served", {}),
    ("split_unfused", {"merge_linear2": False, "fuse_res_norm": False}),
)


class Harness:
    """One frontend + native pipeline over the same buffers, with a saved
    input state both sides start from."""

    def __init__(self, dims_override: dict):
        self.fe = ImageWAMTorchFrontendThor(precision=PRECISION, use_fa4=False,
                                            dims_override={"proprio_dim": PROPRIO_DIM, **dims_override})
        self.fe.set_prompt("pick up the red cup")
        self.native = ImageWAMNativeRuntime.create(self.fe.runtime_surface(), LIBRARY)
        self.native.set_pipeline(self.fe)
        self.stream = torch.cuda.Stream()
        fe = self.fe
        torch.manual_seed(7)
        fe._backbone_hidden.copy_(torch.randn(fe._backbone_hidden.shape, device="cuda").to(torch.bfloat16))
        fe._context.normal_()
        fe._img_raw.normal_()
        fe._action_latent.normal_().mul_(0.01)
        self.base = self.snapshot()

    def tensors(self) -> dict:
        return {name: getattr(self.fe, f"_{name}") for name in STATE_NAMES}

    def snapshot(self) -> dict:
        """Copies of the state buffers, complete on return (the native
        stream is not ordered after the torch stream the copies run on)."""
        torch.cuda.synchronize()
        state = {k: v.detach().clone() for k, v in self.tensors().items()}
        torch.cuda.synchronize()
        return state

    def restore(self) -> None:
        for k, v in self.tensors().items():
            v.copy_(self.base[k])
        torch.cuda.synchronize()

    def python(self, fn) -> dict:
        self.restore()
        with torch.cuda.stream(self.stream):
            fn(self.stream.cuda_stream)
        return self.snapshot()

    def native_run(self, segment: int, index: int = 0) -> dict:
        self.restore()
        self.native.run(segment, index)
        return self.snapshot()


def compare(label: str, a: dict, b: dict, names=STATE_NAMES) -> bool:
    ok = True
    for k in names:
        eq = torch.equal(a[k], b[k])
        ok &= eq
        if not eq:
            print(f"  {label}: {k} differs, max_abs={(a[k].float() - b[k].float()).abs().max().item():.3g}")
    print(f"{label}: {'all state buffers array_equal' if ok else 'MISMATCH'} ({', '.join(names)})")
    return ok


@pytest.fixture(scope="module", params=LAYER_STRUCTURES, ids=[name for name, _ in LAYER_STRUCTURES])
def h(request):
    harness = Harness(request.param[1])
    print(f"layer structure {request.param[0]}: merge_linear2={harness.fe.dims['merge_linear2']} "
          f"fuse_res_norm={harness.fe.dims['fuse_res_norm']}")
    yield harness
    harness.native.close()


def test_gemm_algorithms_handed_off(h):
    print(f"native GEMM shapes: {len(h.native.gemm_shapes)}, algorithms handed off: "
          f"{h.native.gemm_algos_installed}")
    assert h.native.gemm_algos_installed == len(h.native.gemm_shapes) > 0
    kind, m, n, k = h.native.gemm_shapes[0]
    assert len(h.fe.gemm_algo(kind, m, n, k)) == 64


def test_single_stream_block(h):
    d, fe = h.fe.dims, h.fe
    py = h.python(lambda s: pipeline_thor._single_stream_layer(
        fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, d, 0, d["num_layers_double"], s, fe._attn,
        fe._mod_single, fe._rope_table.data_ptr()))
    assert compare("backbone single-stream block 0", py, h.native_run(nl.SEGMENT_SINGLE_LAYER, 0))


def test_double_stream_block(h):
    d, fe = h.fe.dims, h.fe
    py = h.python(lambda s: pipeline_thor._double_stream_layer(
        fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, d, 0, s, fe._attn, fe._mod_txt, fe._mod_img,
        fe._rope_table.data_ptr()))
    assert compare("backbone double-stream block 0", py, h.native_run(nl.SEGMENT_DOUBLE_LAYER, 0))


def _python_prefill(fe, s):
    pipeline_thor.imagewam_prefill(
        fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s, attn=fe._attn,
        mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
        rope_table=fe._rope_table.data_ptr())


def _python_denoise(fe, s):
    pipeline_thor.imagewam_denoise_loop(
        fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s, attn=fe._attn,
        action_mods=fe._action_mods, head_mods=fe._head_mods,
        action_rope_table=fe._action_rope_table.data_ptr(), deltas=fe._deltas)


def test_prefill(h):
    py = h.python(lambda s: _python_prefill(h.fe, s))
    assert compare("prefill", py, h.native_run(nl.SEGMENT_PREFILL))


def test_denoise(h):
    py = h.python(lambda s: _python_denoise(h.fe, s))
    assert compare("denoise loop", py, h.native_run(nl.SEGMENT_DENOISE))


def test_native_graph(h):
    fe = h.fe
    eager = h.python(lambda s: (_python_prefill(fe, s), _python_denoise(fe, s)))
    h.restore()
    fe._graph.replay()
    python_graph = h.snapshot()
    h.native.capture()
    print(f"native graph nodes: {h.native.graph_nodes}")
    h.restore()
    rt = fe.export_model_runtime(io="native", native=h.native)
    consumer = ModelRuntimeConsumer(rt.ptr, exec_library_path())
    try:
        consumer.step()
        consumer.sync()
        native_graph = h.snapshot()
    finally:
        consumer.close()
        rt.release()
    assert compare("native graph vs Python eager (prefill + denoise)", eager, native_graph)
    assert compare("native graph vs Python graph", python_graph, native_graph)


def _infer_reference(fe) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """infer() on fixed proprio and noise: the reference outputs, and the
    image tokens, noise and proprio a consumer must send to reproduce it.

    The state buffers are NaN-poisoned (`poison_tick_state`) BEFORE the
    reference runs, exactly as `_poisoned_native_tick` poisons them before
    the native tick: both sides then start from the same baseline, which is
    what makes the whole-buffer comparison below honest. A graph recorded
    for a trimmed length writes only the rows of its own sequence
    (`backbone_hidden` up to its `a0`, the K/V caches up to its `total`), so
    the rows beyond them hold the baseline on both sides instead of whatever
    the previous tick or capture left there — and a side that wrote one of
    those rows anyway, or wrote it differently, still fails the comparison
    against the other side's baseline. The reference's own outputs do not
    move: `infer()` writes `img_raw`, the proprio row and the action latent
    in full, and the graph writes the rows it owns."""
    proprio = np.linspace(-0.4, 0.5, PROPRIO_DIM, dtype=np.float32)
    poison_tick_state(fe)
    torch.manual_seed(3)
    noise = torch.empty_like(fe._action_latent).normal_().mul_(0.01)
    actions = fe.infer({"proprio": proprio}, action_noise=noise)["actions"]
    torch.cuda.synchronize()
    ref = {"actions": actions, **{k: bits(getattr(fe, f"_{k}")) for k in TICK_STATE}}
    return ref, bits(fe._img_raw), noise.cpu().numpy(), proprio


def _poisoned_native_tick(fe, native, tokens, noise, proprio) -> dict:
    """One io="native" tick after NaN-filling every buffer it must write."""
    rt = fe.export_model_runtime(io="native", native=native)
    consumer = ModelRuntimeConsumer(rt.ptr, exec_library_path())
    try:
        assert "graph_producer=native" in rt.identity
        poison_tick_state(fe)
        consumer.write_swap("image_tokens", tokens)
        consumer.set_input("proprio", proprio.tobytes())
        consumer.write_swap("noise", noise)
        consumer.step()
        out = {"actions": consumer.get_output("actions", np.float32, (fe.dims["num_action"], fe.dims["action_dim"]))}
        torch.cuda.synchronize()
        out.update({k: bits(getattr(fe, f"_{k}")) for k in TICK_STATE})
        return out
    finally:
        consumer.close()
        rt.release()


def _differing(out: dict, ref: dict) -> list[str]:
    """Buffers that differ over their whole contents between the two poisoned
    sides (`_infer_reference` poisons before `infer()`, `_poisoned_native_tick`
    before the native tick)."""
    return [k for k in ref if not np.array_equal(out[k], ref[k])]


def test_native_face_tick_matches_infer(h):
    fe = h.fe
    if h.native.graph_producer != "native":
        h.native.capture()
    ref, tokens, noise, proprio = _infer_reference(fe)
    out = _poisoned_native_tick(fe, h.native, tokens, noise, proprio)
    differing = _differing(out, ref)
    print(f"poisoned io=native tick on the native graph vs infer(): differing={differing} "
          f"actions max_abs={np.abs(out['actions'] - ref['actions']).max():.3g}")
    assert differing == []


@pytest.mark.parametrize("mutation", PIPELINE_MUTATIONS)
def test_pipeline_mutant_fails_the_tick(h, mutation):
    fe = h.fe
    ref, tokens, noise, proprio = _infer_reference(fe)
    mutant = ImageWAMNativeRuntime.create(fe.runtime_surface(), LIBRARY)
    try:
        mutant.set_pipeline(MutatedPipelineSource(fe, mutation))
        mutant.capture()
        out = _poisoned_native_tick(fe, mutant, tokens, noise, proprio)
    finally:
        mutant.close()
    differing = _differing(out, ref)
    print(f"native pipeline mutant {mutation}: differing={differing}")
    assert differing, f"mutant {mutation} passed the tick"


def test_set_pipeline_drops_the_captured_graph(h):
    """The captured graph records the pipeline's GEMM handles and resource
    pointers: installing the pipeline of a key again destroys that key's
    graph, so the old resources can be freed and nothing can replay the graph
    over them. Refused while a model runtime over the handle is live."""
    fe = h.fe
    native = ImageWAMNativeRuntime.create(fe.runtime_surface(), LIBRARY)
    try:
        native.set_pipeline(fe)
        native.capture()
        key = native.text_length
        rt = fe.export_model_runtime(io="native", native=native)
        with pytest.raises(ImageWAMNativeError) as exc:
            native.set_pipeline(fe)
        print(f"set_pipeline while exported: {exc.value}")
        assert native.graph_exec and native.graph_producer == "native"
        rt.release()

        first = weakref.ref(native._pipeline_handoffs[key])
        native.set_pipeline(fe)
        gc.collect()
        print(f"after a second set_pipeline: graph_exec={native.graph_exec} graph_nodes={native.graph_nodes} "
              f"graph_producer={native.graph_producer!r} first pipeline's resources released={first() is None}")
        assert native.graph_exec == 0 and native.graph_nodes == 0 and native.graph_producer == ""
        assert first() is None
        with pytest.raises(ValueError):
            fe.export_model_runtime(io="native", native=native)

        native.capture()
        ref, tokens, noise, proprio = _infer_reference(fe)
        assert _differing(_poisoned_native_tick(fe, native, tokens, noise, proprio), ref) == []
    finally:
        native.close()


def _replay_native_graph(native, tensors) -> np.ndarray:
    """Seeded inputs, one launch of the native graph, the action latent."""
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    context, img_raw, latent = tensors
    torch.manual_seed(11)
    context.normal_()
    img_raw.normal_()
    latent.normal_().mul_(0.01)
    torch.cuda.synchronize()
    assert cudart.cudaGraphLaunch(native.graph_exec, native.stream) == 0
    assert cudart.cudaStreamSynchronize(native.stream) == 0
    return latent.cpu().numpy().copy()


def test_native_handle_keeps_the_frontend_alive():
    """The captured graph reads the frontend's weights and buffers; the
    handle must keep the frontend alive when nothing else refers to it."""
    fe = ImageWAMTorchFrontendThor(precision=PRECISION, use_fa4=False,
                                   dims_override={"proprio_dim": PROPRIO_DIM})
    fe.set_prompt("pick up the red cup")
    native = ImageWAMNativeRuntime.create(fe.runtime_surface(), LIBRARY)
    native.set_pipeline(fe)
    native.capture()
    tensors = (fe._context, fe._img_raw, fe._action_latent)
    ref = _replay_native_graph(native, tensors)
    alive = weakref.ref(fe)
    del fe
    gc.collect()
    torch.cuda.empty_cache()
    out = _replay_native_graph(native, tensors)
    print(f"frontend alive with only the native handle referring to it: {alive() is not None}; "
          f"replay array_equal={np.array_equal(out, ref)}")
    assert alive() is not None and np.array_equal(out, ref)
    native.close()
    del native
    gc.collect()
    assert alive() is None, "closing the native handle must release the frontend"


def test_pipeline_resources_refuses_awq(h, monkeypatch):
    """The native pipeline has no AWQ input-scale fold (pipeline_thor.py
    folds it into the AdaLN operands), so an AWQ frontend is refused."""
    monkeypatch.setattr(h.fe, "_nvfp4_awq", True)
    with pytest.raises(ValueError, match="AWQ") as exc:
        h.fe.pipeline_resources()
    print(f"pipeline_resources() with nvfp4_awq: {exc.value}")


# -- text_trim=True: one native pipeline and one graph per text length -----

TRIM_TEXT_ROWS = 16
# `x0 = valid + 1` (the proprio row), `a0 = x0 + 10` and
# `total = a0 + num_action` keep the image and action blocks at their default
# lengths (the rows test_imagewam_native_runtime.py's trimmed fixture uses).
TRIM_DIMS = dict(x0=TRIM_TEXT_ROWS + 1, a0=TRIM_TEXT_ROWS + 1 + 10, total=TRIM_TEXT_ROWS + 1 + 10 + 4,
                 proprio_dim=PROPRIO_DIM)
TRIM_LENGTHS = (5, 13)          # valid text tokens, i.e. x0 = 6 and 14


def _set_trimmed_prompt(fe: ImageWAMTorchFrontendThor, valid: int) -> None:
    """One precomputed prompt of `valid` real text tokens; the trimmed
    context length is `valid + 1` (the proprio row)."""
    mask = torch.zeros(TRIM_TEXT_ROWS, dtype=torch.bool)
    mask[:valid] = True
    torch.manual_seed(valid)
    fe.set_prompt(context=torch.randn(TRIM_TEXT_ROWS, 64).to(torch.bfloat16), context_mask=mask)


def test_pipeline_records_one_graph_per_text_length():
    """The native pipeline's own capture path, one length at a time: a
    `text_trim=True` frontend with two captured lengths gets one installed
    pipeline and one graph this handle recorded per length
    (`capture_pipeline_text_lengths`), the lengths are declared in the
    manifest, and the poisoned `io="native"` tick at EITHER length — the
    shorter one first — is `array_equal` to `infer()` at that same length.

    The ticks run at the shorter length first, which is not the export-time
    default key, so a tick replaying the default key would fail here. Every
    graph is the handle's own, so `graph_producer` is `native`.

    The state buffers are compared whole, and both sides start from the same
    poisoned baseline: `_infer_reference` poisons the state before `infer()`,
    `_poisoned_native_tick` poisons it before the native tick. That is the
    per-length contract — a graph recorded for `x0` writes the rows of that
    length's own sequence (`backbone_hidden` up to its `a0`, `K_cache` and
    `V_cache` up to its `total`), so at `x0=6` the rows past them are outside
    what the length's graph owns and hold the baseline on both sides instead
    of the previous tick's values. What the length does own is still compared
    exactly: the active span bit for bit, and `actions` — that length's own
    output — in full. The untrimmed one-key rows above are unchanged: that
    frontend's single graph IS the longest length, so its tick writes those
    buffers in full (no row is left at any baseline)."""
    fe = ImageWAMTorchFrontendThor(precision=PRECISION, use_fa4=False, dims_override=dict(TRIM_DIMS),
                                   text_trim=True)
    keys = tuple(valid + 1 for valid in TRIM_LENGTHS)
    for valid in TRIM_LENGTHS:
        _set_trimmed_prompt(fe, valid)
    surface = fe.runtime_surface()
    assert tuple(entry.key for entry in surface.graph_variants.entries) == keys
    native = ImageWAMNativeRuntime.create(surface, LIBRARY)
    try:
        assert native.capture_pipeline_text_lengths(fe) == keys
        print(f"native pipelines: {native.gemm_algos_installed} algorithms handed off for the active "
              f"x0={native.text_length} ({len(native.gemm_shapes)} GEMM shapes: "
              f"{native.gemm_shapes[:3]}...), graph {native.graph_nodes} nodes")
        assert all(native.has_variant(key) for key in keys)
        assert native.graph_producer == "native"
        assert native.text_length == fe.active_dims["x0"] == keys[-1]

        rt = fe.export_model_runtime(io="native", native=native)
        try:
            manifest = json.loads(rt.manifest)
            print(f"trimmed native manifest graphs={manifest['graphs']} "
                  f"text_lengths={manifest['text_lengths']}")
            assert manifest["text_lengths"] == {"default_key": keys[-1], "keys": list(keys),
                                                "per_prompt_length": True}
            assert manifest["graphs"][0] == {"name": "infer", "default_key": keys[-1],
                                             "keys": list(keys), "stream": "main"}
        finally:
            rt.release()

        for valid in (TRIM_LENGTHS[0], TRIM_LENGTHS[1]):
            key = valid + 1
            _set_trimmed_prompt(fe, valid)
            shifted = fe.runtime_surface()
            # The prompt changed: the length reaches the handle before the
            # proprio row it bounds is set again.
            native.set_text_length(key)
            native.set_proprio_row(shifted.proprio_row)
            assert native.text_length == key
            assert native.graph_exec == native.variant_exec(key)
            ref, tokens, noise, proprio = _infer_reference(fe)
            out = _poisoned_native_tick(fe, native, tokens, noise, proprio)
            differing = _differing(out, ref)
            print(f"x0={key}: graph {native.graph_nodes} nodes vs infer(), differing={differing} "
                  f"actions max_abs={np.abs(out['actions'] - ref['actions']).max():.3g}")
            assert differing == [], f"x0={key}: {differing}"
    finally:
        native.close()
