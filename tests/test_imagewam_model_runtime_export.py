"""ImageWAM `frt_model_runtime_v1` Python producer (io="python").

Small random-weight dims (`imagewam_thor._DEFAULT_DIMS` plus proprio and
dataset stats), fp16. A ctypes consumer drives the runtime only through
the C ABI (tests/_helpers/model_runtime_consumer.py) and must reproduce
`frontend.infer()` bit for bit for the same image tokens, proprio and
initial noise. Skips when the exec/ and runtime/ native modules are not
built (see docs/imagewam_model_runtime.md).
"""
import json

import numpy as np
import pytest
import torch

pytest.importorskip("flash_rt.runtime.exec", exc_type=ImportError)
pytest.importorskip("flash_rt.runtime.export", exc_type=ImportError)

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor  # noqa: E402
from _helpers.model_runtime_consumer import (  # noqa: E402
    ModelRuntimeConsumer,
    exec_library_path,
)

PROPRIO_DIM = 8
SEED = 1234


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
    fe = ImageWAMTorchFrontendThor(precision="fp16", dims_override={"proprio_dim": PROPRIO_DIM},
                                   dataset_stats_path=str(path))
    fe.set_prompt("pick up the red cup")
    return fe


@pytest.fixture(scope="module")
def runtime(frontend):
    mr = frontend.export_model_runtime(identity={"test": "imagewam_model_runtime"})
    consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
    yield mr, consumer
    consumer.close()
    mr.release()


def _bf16_bytes(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).cpu().numpy()


def test_port_schema(frontend, runtime):
    mr, consumer = runtime
    d = frontend.dims
    img_len, chunk = d["a0"] - d["x0"], (d["num_action"], d["action_dim"])
    names = [p.name for p in consumer.ports]
    print(f"ports={names} stages={consumer.n_stages} fingerprint=0x{consumer.fingerprint:016x}")
    assert names == ["image_tokens", "proprio", "noise", "actions", "actions_raw"]
    assert consumer.n_stages == 1
    p = {x.name: x for x in consumer.ports}
    # (update, direction, dtype, shape): SWAP=0 STAGED=1; IN=0 OUT=1; U8=0 F32=1 BF16=3
    assert (p["image_tokens"].update, p["image_tokens"].dtype, p["image_tokens"].shape) == (0, 3, (img_len, 128))
    assert p["image_tokens"].nbytes == img_len * 128 * 2 and p["image_tokens"].required
    assert (p["proprio"].update, p["proprio"].dtype, p["proprio"].shape) == (1, 1, (PROPRIO_DIM,))
    assert p["proprio"].buffer == 0
    assert (p["noise"].update, p["noise"].direction, p["noise"].dtype, p["noise"].shape) == (0, 0, 1, chunk)
    assert (p["actions"].update, p["actions"].direction, p["actions"].shape) == (1, 1, chunk)
    assert p["actions"].buffer == 0 and p["actions"].nbytes == chunk[0] * chunk[1] * 4
    assert (p["actions_raw"].update, p["actions_raw"].direction) == (0, 1)
    assert p["actions_raw"].buffer == p["noise"].buffer, "noise and actions_raw share the in-place window"
    ident = mr.identity
    for record in ("model=imagewam", "precision=fp16", "io=python", "action_denormalized=True",
                   "proprio_dim=8", "has_vae=False"):
        assert record in ident, (record, ident)


def test_identity_sensitivity(frontend, runtime):
    mr, _ = runtime
    same = frontend.export_model_runtime(identity={"test": "imagewam_model_runtime"})
    other = frontend.export_model_runtime(identity={"test": "other"})
    try:
        print(f"fingerprints: base=0x{mr.fingerprint:016x} same=0x{same.fingerprint:016x} "
              f"other=0x{other.fingerprint:016x}")
        assert same.fingerprint == mr.fingerprint
        assert other.fingerprint != mr.fingerprint
    finally:
        same.release()
        other.release()


def test_unknown_io_face_rejected(frontend):
    with pytest.raises(ValueError, match="io face"):
        frontend.export_model_runtime(io="native")


def test_abi_tick_matches_infer_bit_exact(frontend, runtime):
    _, consumer = runtime
    d = frontend.dims
    chunk = (d["num_action"], d["action_dim"])
    proprio = np.linspace(-0.7, 0.9, PROPRIO_DIM, dtype=np.float32)

    torch.manual_seed(SEED)
    ref_actions = frontend.infer({"proprio": proprio})["actions"]
    ref_raw = frontend._action_latent.detach().cpu().numpy().copy()
    ref_tokens = frontend._img_raw.detach().clone()
    torch.manual_seed(SEED)
    again = frontend.infer({"proprio": proprio})["actions"]
    assert np.array_equal(again, ref_actions), "infer() is not deterministic; parity check is meaningless"

    # Reproduce infer()'s own draws in the same order: img_raw.normal_(),
    # then action_latent.normal_().mul_(0.01).
    torch.manual_seed(SEED)
    tokens = torch.empty_like(frontend._img_raw).normal_()
    noise = torch.empty_like(frontend._action_latent).normal_().mul_(0.01)
    assert torch.equal(tokens, ref_tokens)

    consumer.write_swap("image_tokens", _bf16_bytes(tokens))
    consumer.set_input("proprio", proprio.tobytes())
    consumer.write_swap("noise", noise.cpu().numpy())
    consumer.step()
    abi_actions = consumer.get_output("actions", np.float32, chunk)
    abi_raw = consumer.read_swap("actions_raw", np.float32, chunk)

    max_abs = float(np.max(np.abs(abi_actions - ref_actions)))
    raw_max_abs = float(np.max(np.abs(abi_raw - ref_raw)))
    print(f"actions: shape={abi_actions.shape} array_equal={np.array_equal(abi_actions, ref_actions)} "
          f"max_abs={max_abs:.3g}; actions_raw: array_equal={np.array_equal(abi_raw, ref_raw)} "
          f"max_abs={raw_max_abs:.3g}; |actions|_max={np.abs(ref_actions).max():.4f}")
    assert np.array_equal(abi_actions, ref_actions)
    assert np.array_equal(abi_raw, ref_raw)


def test_staged_and_swap_guards(frontend, runtime):
    _, consumer = runtime
    rc = consumer.set_input_status("noise", b"\0" * 16)
    print(f"set_input(noise SWAP) rc={rc}")
    assert rc == -3

    rc = consumer.set_input_status("proprio", b"\0" * 12)
    err = consumer.last_error()
    print(f"set_input(proprio, 12 bytes) rc={rc} last_error={err.splitlines()[0]!r}")
    assert rc == -1 and "proprio payload must be 32 bytes" in err

    rc, _, _ = consumer.get_output_status("actions_raw", 1024)
    print(f"get_output(actions_raw SWAP) rc={rc} last_error={consumer.last_error().splitlines()[0]!r}")
    assert rc == -1

    need = consumer.port("actions").nbytes
    rc, _, written = consumer.get_output_status("actions", need - 4)
    print(f"get_output(actions, capacity={need - 4}) rc={rc} written={written}")
    assert rc == -5 and written == need
