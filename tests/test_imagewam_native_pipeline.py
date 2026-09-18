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
and through the `io="native"` model runtime against `infer()`.
Skips when exec/, runtime/ or the native library is not built.
"""
import os

import numpy as np
import pytest
import torch

pytest.importorskip("flash_rt.runtime.exec", exc_type=ImportError)
pytest.importorskip("flash_rt.runtime.export", exc_type=ImportError)

import flash_rt.flash_rt_kernels as fvk  # noqa: E402
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor  # noqa: E402
from flash_rt.models.imagewam import native_library as nl  # noqa: E402
from flash_rt.models.imagewam import pipeline_thor  # noqa: E402
from flash_rt.models.imagewam.native_runtime import ImageWAMNativeRuntime  # noqa: E402
from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path  # noqa: E402

try:
    LIBRARY = nl.ImageWAMNativeLibrary()
except ImportError as e:  # pragma: no cover - depends on the local build
    pytest.skip(str(e), allow_module_level=True)

PROPRIO_DIM = 8
# fp16 locally; the Thor checklist sets IMAGEWAM_NATIVE_PRECISION=nvfp4.
PRECISION = os.environ.get("IMAGEWAM_NATIVE_PRECISION", "fp16")
STATE_NAMES = ("backbone_hidden", "K_cache", "V_cache", "Q_O", "context", "img_raw", "action_latent")


class Harness:
    """One frontend + native pipeline over the same buffers, with a saved
    input state both sides start from."""

    def __init__(self):
        self.fe = ImageWAMTorchFrontendThor(precision=PRECISION, dims_override={"proprio_dim": PROPRIO_DIM})
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
        torch.cuda.synchronize()
        return {k: v.detach().clone() for k, v in self.tensors().items()}

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


@pytest.fixture(scope="module")
def h():
    harness = Harness()
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
    assert compare("native graph vs Python graph", python_graph, native_graph,
                   names=("K_cache", "V_cache", "Q_O", "action_latent"))


def test_native_face_tick_matches_infer(h):
    fe = h.fe
    if h.native.graph_producer != "native":
        h.native.capture()
    rt = fe.export_model_runtime(io="native", native=h.native)
    consumer = ModelRuntimeConsumer(rt.ptr, exec_library_path())
    try:
        assert "graph_producer=native" in rt.identity
        chunk = (fe.dims["num_action"], fe.dims["action_dim"])
        proprio = np.linspace(-0.4, 0.5, PROPRIO_DIM, dtype=np.float32)
        torch.manual_seed(3)
        ref = fe.infer({"proprio": proprio})["actions"]
        torch.manual_seed(3)
        tokens = torch.empty_like(fe._img_raw).normal_()
        noise = torch.empty_like(fe._action_latent).normal_().mul_(0.01)
        consumer.write_swap("image_tokens", tokens.view(torch.int16).cpu().numpy())
        consumer.set_input("proprio", proprio.tobytes())
        consumer.write_swap("noise", noise.cpu().numpy())
        consumer.step()
        actions = consumer.get_output("actions", np.float32, chunk)
        print(f"io=native tick on the native graph vs infer(): array_equal={np.array_equal(actions, ref)} "
              f"max_abs={np.abs(actions - ref).max():.3g}")
        assert np.array_equal(actions, ref)
    finally:
        consumer.close()
        rt.release()
