"""ImageWAM `io="native"` model runtime: C verbs from libflashrt_imagewam_native.

Small random-weight dims (fp16, or IMAGEWAM_NATIVE_PRECISION; proprio +
dataset stats). Covers the schema (Python declaration records == C++
records), the native verbs over
the Python-captured graph (every buffer a tick writes NaN-filled first;
the tick must fail when the consumer skips the proprio verb or `step`), a
tick that enters no Python frame, the status codes, and setup calls
refused while the model runtime is live. Skips when exec/, runtime/ or the native library is not
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
from _helpers.imagewam_abi_checks import bits, poison_tick_state  # noqa: E402
from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path  # noqa: E402

try:
    LIBRARY = ImageWAMNativeLibrary()
except ImportError as e:  # pragma: no cover - depends on the local build
    pytest.skip(str(e), allow_module_level=True)

PROPRIO_DIM = 8
# fp16 locally; the Thor checklist sets IMAGEWAM_NATIVE_PRECISION=nvfp4.
PRECISION = os.environ.get("IMAGEWAM_NATIVE_PRECISION", "fp16")
SEED = 99
# Buffers a tick leaves behind besides the actions.
TICK_STATE = ("backbone_hidden", "K_cache", "V_cache")


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
    """Seeded image tokens and 0.01-scaled noise for a tick."""
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


@pytest.fixture(scope="module")
def reference(frontend):
    """infer() on fixed proprio and noise, and what a consumer sends to
    reproduce it (image tokens, noise, proprio)."""
    proprio = np.linspace(-0.6, 0.8, PROPRIO_DIM, dtype=np.float32)
    torch.manual_seed(SEED)
    noise = torch.empty_like(frontend._action_latent).normal_().mul_(0.01)
    actions = frontend.infer({"proprio": proprio}, action_noise=noise)["actions"]
    torch.cuda.synchronize()
    row = frontend.runtime_surface().proprio_row
    ref = {"actions": actions, "actions_raw": frontend._action_latent.cpu().numpy().copy(),
           "token": bits(frontend._context[row]), **{k: bits(getattr(frontend, f"_{k}")) for k in TICK_STATE}}
    return ref, bits(frontend._img_raw), noise.cpu().numpy(), proprio


def _tick(frontend, consumer, reference, *, proprio_by: str, step: bool = True) -> dict:
    """One tick after NaN-filling every buffer it must write; proprio is
    staged by the frontend ("python"), the native verb ("native") or not
    at all ("none")."""
    ref, tokens, noise, proprio = reference
    poison_tick_state(frontend)
    if proprio_by == "python":
        frontend.stage_proprio(proprio)
        torch.cuda.synchronize()
    consumer.write_swap("image_tokens", tokens)
    if proprio_by == "native":
        consumer.set_input("proprio", proprio.tobytes())
    consumer.write_swap("noise", noise)
    if step:
        consumer.step()
    chunk = ref["actions"].shape
    out = {"actions": consumer.get_output("actions", np.float32, chunk),
           "actions_raw": consumer.read_swap("actions_raw", np.float32, chunk)}
    torch.cuda.synchronize()
    out["token"] = bits(frontend._context[frontend.runtime_surface().proprio_row])
    out.update({k: bits(getattr(frontend, f"_{k}")) for k in TICK_STATE})
    return out


def _differing(out: dict, ref: dict) -> list[str]:
    return [k for k in ref if not np.array_equal(out[k], ref[k])]


@pytest.mark.parametrize("proprio_by", ["python", "native"])
def test_native_tick_matches_infer(frontend, native_runtime, reference, proprio_by):
    """proprio staged by the frontend (the token infer() stages), or by the
    native verb (C++ normalization + cuBLASLt projection)."""
    _, _, consumer = native_runtime
    ref = reference[0]
    out = _tick(frontend, consumer, reference, proprio_by=proprio_by)
    differing = _differing(out, ref)
    print(f"poisoned tick, proprio staged by {proprio_by}: differing={differing} "
          f"actions max_abs={np.abs(out['actions'] - ref['actions']).max():.3g}")
    assert differing == []


@pytest.mark.parametrize("mutant", ["proprio verb not called", "step not called"])
def test_skipped_verb_fails_the_tick(frontend, native_runtime, reference, mutant):
    _, _, consumer = native_runtime
    if mutant == "proprio verb not called":
        out = _tick(frontend, consumer, reference, proprio_by="none")
    else:
        out = _tick(frontend, consumer, reference, proprio_by="native", step=False)
    differing = _differing(out, reference[0])
    print(f"mutant {mutant}: differing={differing}")
    assert differing


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


def test_setup_refused_while_exported(frontend, native_runtime):
    """The live model runtime adopted the current graph exec: replacing the
    graph or the pipeline under it is refused."""
    native, _, _ = native_runtime
    graph_exec = frontend.runtime_surface().graph_exec
    for what, call in (("use_graph", lambda: native.use_graph(graph_exec)),
                       ("set_pipeline", lambda: native.set_pipeline(frontend)),
                       ("capture", native.capture)):
        with pytest.raises(ImageWAMNativeError) as exc:
            call()
        print(f"{what} while exported: {exc.value}")
        assert exc.value.status == -1 and "release it first" in str(exc.value)
    assert native.graph_producer == "python" and native.graph_exec == graph_exec
