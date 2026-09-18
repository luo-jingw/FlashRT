"""``ImageWAMTorchFrontendThor.infer(..., action_noise=...)``.

The regression gate drives the served ``infer()`` with a fixed initial
action latent. This checks, at the frontend's small default dims with
random weights (fp16, runs on any CUDA GPU), that:

* ``infer(obs, action_noise=n)`` is bit-identical to writing ``n`` into
  the action-latent buffer and replaying the graph directly (the
  end-to-end script's ``flashrt_infer_with_noise`` path);
* the default path (``action_noise=None``) is unchanged: it draws
  ``0.01 * N(0,1)`` from the same device RNG stream as before;
* a wrongly shaped ``action_noise`` is rejected.

The random image input (no VAE at these dims) is made repeatable by
reseeding the device RNG before each call.
"""
from __future__ import annotations

import pytest
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

SEED = 1234


@pytest.fixture(scope="module")
def frontend() -> ImageWAMTorchFrontendThor:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    fe = ImageWAMTorchFrontendThor(precision="fp16")
    fe.set_prompt("pick up the black bowl")
    return fe


def _manual_replay(fe: ImageWAMTorchFrontendThor, noise: torch.Tensor | None) -> torch.Tensor:
    torch.manual_seed(SEED)
    fe._img_raw.normal_()
    if noise is None:
        fe._action_latent.normal_()
        fe._action_latent.mul_(0.01)
    else:
        fe._action_latent.copy_(noise)
    fe._graph.replay()
    torch.cuda.synchronize()
    return fe._action_latent.detach().cpu().clone()


def _served(fe: ImageWAMTorchFrontendThor, noise: torch.Tensor | None) -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.from_numpy(fe.infer({}, action_noise=noise)["actions"])


def test_fixed_noise_matches_direct_replay_bit_exactly(frontend):
    d = frontend.dims
    noise = torch.randn(d["num_action"], d["action_dim"], generator=torch.Generator().manual_seed(0)).cuda()
    served = _served(frontend, noise)
    direct = _manual_replay(frontend, noise)
    again = _served(frontend, noise)
    print(f"max |served - direct| = {(served - direct).abs().max().item():.3e}, "
          f"max |served - again| = {(served - again).abs().max().item():.3e}")
    assert torch.equal(served, direct)
    assert torch.equal(served, again)
    assert torch.isfinite(served).all()


def test_default_noise_path_is_unchanged(frontend):
    served = _served(frontend, None)
    direct = _manual_replay(frontend, None)
    assert torch.equal(served, direct)


def test_fixed_noise_changes_the_output(frontend):
    d = frontend.dims
    noise = torch.randn(d["num_action"], d["action_dim"], generator=torch.Generator().manual_seed(1)).cuda()
    assert not torch.equal(_served(frontend, noise), _served(frontend, None))


def test_wrong_noise_shape_is_rejected(frontend):
    d = frontend.dims
    with pytest.raises(ValueError, match="action_noise shape"):
        frontend.infer({}, action_noise=torch.zeros(d["num_action"] + 1, d["action_dim"], device="cuda"))
