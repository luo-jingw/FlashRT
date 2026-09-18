"""ImageWAM `io="native"` model runtime: C verbs from libflashrt_imagewam_native.

Small random-weight dims (fp16, or IMAGEWAM_NATIVE_PRECISION; proprio +
dataset stats). Covers the schema (Python declaration records == C++
records), the native verbs over
the Python-captured graph, a tick that enters no Python frame, and the
status codes. Skips when exec/, runtime/ or the native library is not
built (docs/imagewam_native_cpp.md).
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("flash_rt.runtime.exec", exc_type=ImportError)
pytest.importorskip("flash_rt.runtime.export", exc_type=ImportError)

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor  # noqa: E402
from flash_rt.models.imagewam.native_library import ImageWAMNativeLibrary  # noqa: E402
from flash_rt.models.imagewam.native_runtime import ImageWAMNativeError, ImageWAMNativeRuntime  # noqa: E402
from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path  # noqa: E402

try:
    LIBRARY = ImageWAMNativeLibrary()
except ImportError as e:  # pragma: no cover - depends on the local build
    pytest.skip(str(e), allow_module_level=True)

PROPRIO_DIM = 8
# fp16 locally; the Thor checklist sets IMAGEWAM_NATIVE_PRECISION=nvfp4.
PRECISION = os.environ.get("IMAGEWAM_NATIVE_PRECISION", "fp16")
SEED = 99


@pytest.fixture(scope="module")
def frontend(tmp_path_factory):
    stats = {
        "state": {"default": {"global_min": [-1.0 - 0.1 * i for i in range(PROPRIO_DIM)],
                              "global_max": [1.0 + 0.2 * i for i in range(PROPRIO_DIM)]}},
        "action": {"default": {"global_min": [-0.5 - 0.05 * i for i in range(7)],
                               "global_max": [0.5 + 0.1 * i for i in range(7)]}},
    }
    path = tmp_path_factory.mktemp("imagewam_stats") / "dataset_stats.json"
    path.write_text(json.dumps(stats))
    fe = ImageWAMTorchFrontendThor(precision=PRECISION, dims_override={"proprio_dim": PROPRIO_DIM},
                                   dataset_stats_path=str(path))
    fe.set_prompt("pick up the red cup")
    return fe


@pytest.fixture(scope="module")
def native_runtime(frontend):
    surface = frontend.runtime_surface()
    native = ImageWAMNativeRuntime.create(surface, LIBRARY)
    native.use_graph(surface.graph_exec)
    mr = frontend.export_model_runtime(io="native", native=native,
                                       identity={"test": "imagewam_native_runtime"})
    consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
    yield native, mr, consumer
    consumer.close()
    mr.release()
    native.close()


def _records(identity: str) -> list[str]:
    return [line for line in identity.splitlines() if line.startswith(("region:", "port:", "stage:"))]


def _draw(frontend, seed):
    """infer()'s own draws, in order: img_raw.normal_(), then 0.01 * noise."""
    torch.manual_seed(seed)
    tokens = torch.empty_like(frontend._img_raw).normal_()
    noise = torch.empty_like(frontend._action_latent).normal_().mul_(0.01)
    return tokens, noise


def test_schema_records_match(native_runtime):
    native, mr, consumer = native_runtime
    python_records = _records(mr.identity)
    native_records = native.schema_records()
    print("\n".join(native_records))
    assert python_records == native_records
    assert [p.name for p in consumer.ports] == ["image_tokens", "proprio", "noise", "actions", "actions_raw"]
    assert "io=native" in mr.identity and "graph_producer=python" in mr.identity
    assert consumer.stream_handle == native.stream


def test_native_tick_matches_infer(frontend, native_runtime):
    _, _, consumer = native_runtime
    d = frontend.dims
    chunk = (d["num_action"], d["action_dim"])
    proprio = np.linspace(-0.6, 0.8, PROPRIO_DIM, dtype=np.float32)
    torch.manual_seed(SEED)
    ref = frontend.infer({"proprio": proprio})["actions"]
    ref_raw = frontend._action_latent.detach().cpu().numpy().copy()
    row = frontend.runtime_surface().proprio_row
    ref_token = frontend._context[row].detach().clone()
    tokens, noise = _draw(frontend, SEED)

    # 1. the proprio token infer() staged is still in the context row
    consumer.write_swap("image_tokens", tokens.view(torch.int16).cpu().numpy())
    consumer.write_swap("noise", noise.cpu().numpy())
    consumer.step()
    a1 = consumer.get_output("actions", np.float32, chunk)
    raw1 = consumer.read_swap("actions_raw", np.float32, chunk)

    # 2. the native verb stages proprio (C++ normalization + cuBLASLt projection)
    frontend._context[row].zero_()
    torch.cuda.synchronize()
    consumer.set_input("proprio", proprio.tobytes())
    consumer.sync()
    native_token = frontend._context[row].detach().clone()
    consumer.write_swap("noise", noise.cpu().numpy())
    consumer.step()
    a2 = consumer.get_output("actions", np.float32, chunk)

    token_max = (native_token.float() - ref_token.float()).abs().max().item()
    print(f"python-staged proprio: actions array_equal={np.array_equal(a1, ref)} "
          f"actions_raw array_equal={np.array_equal(raw1, ref_raw)}")
    print(f"native proprio token: array_equal={torch.equal(native_token, ref_token)} max_abs={token_max:.3g}; "
          f"actions array_equal={np.array_equal(a2, ref)} max_abs={np.abs(a2 - ref).max():.3g}")
    assert np.array_equal(a1, ref) and np.array_equal(raw1, ref_raw)
    assert torch.equal(native_token, ref_token)
    assert np.array_equal(a2, ref)


def _python_frames_during_tick(consumer: ModelRuntimeConsumer, proprio: np.ndarray) -> list[str]:
    """Python functions entered while the C verbs set_input(proprio) and
    step run (argument set-up happens before profiling starts)."""
    set_input, step = consumer._m.verbs.set_input, consumer._m.verbs.step
    self_, port = consumer._m.self_, consumer.port("proprio").index
    data, nbytes = proprio.ctypes.data, proprio.nbytes
    frames = []

    def profile(frame, event, arg):
        if event == "call":
            frames.append(frame.f_code.co_qualname)

    sys.setprofile(profile)
    try:
        set_input(self_, port, data, nbytes, -1)
        step(self_)
    finally:
        sys.setprofile(None)
    consumer.sync()
    return frames


def test_native_tick_enters_no_python(frontend, native_runtime):
    _, _, consumer = native_runtime
    tokens, noise = _draw(frontend, SEED)
    proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
    consumer.write_swap("image_tokens", tokens.view(torch.int16).cpu().numpy())
    consumer.write_swap("noise", noise.cpu().numpy())
    native_frames = _python_frames_during_tick(consumer, proprio)

    python_face = frontend.export_model_runtime(io="python")
    python_consumer = ModelRuntimeConsumer(python_face.ptr, exec_library_path())
    try:
        python_consumer.write_swap("noise", noise.cpu().numpy())
        python_frames = _python_frames_during_tick(python_consumer, proprio)
    finally:
        python_consumer.close()
        python_face.release()
    print(f"python frames entered by set_input(proprio) + step: io=native {len(native_frames)}, "
          f"io=python {len(python_frames)} (e.g. {python_frames[:3]})")
    assert native_frames == []
    assert python_frames, "control: the Python face must enter Python"


def test_status_codes(native_runtime):
    native, _, consumer = native_runtime
    rc = consumer.set_input_status("noise", b"\0" * 16)
    print(f"set_input(noise SWAP) rc={rc}: {consumer.last_error()}")
    assert rc == -3
    rc = consumer.set_input_status("proprio", b"\0" * 12)
    print(f"set_input(proprio, 12 bytes) rc={rc}: {consumer.last_error()}")
    assert rc == -4
    rc, _, _ = consumer.get_output_status("actions_raw", 1024)
    print(f"get_output(actions_raw SWAP) rc={rc}: {consumer.last_error()}")
    assert rc == -3
    need = consumer.port("actions").nbytes
    rc, _, written = consumer.get_output_status("actions", need - 4)
    print(f"get_output(actions, short) rc={rc} written={written}")
    assert rc == -5 and written == need
    with pytest.raises(ImageWAMNativeError) as exc:
        native.set_proprio_row(10_000)
    print(f"set_proprio_row(10000): {exc.value}")


def test_bind_rejects_the_python_face(frontend, native_runtime):
    native, _, _ = native_runtime
    python_face = frontend.export_model_runtime(io="python")
    try:
        with pytest.raises(ImageWAMNativeError) as exc:
            native.bind_declaration(python_face.ptr)
        print(f"bind_declaration(io=python face): {exc.value}")
    finally:
        python_face.release()
