"""`text_encoder.py` correctness against real Qwen3-4B (real VAE +
text-context wiring plan's own deferred "Live Qwen3" item, closed
2026-09-15).

**Skips cleanly if the real Qwen3-4B checkpoint isn't present** --
following this project's own established skip pattern, inverted for a
local RESOURCE. Defaults to this dev machine's own actual download
location (`/home/ljw/projects/pi0.5/models/qwen3_4b`) -- override via
the `QWEN3_MODEL_SPEC` env var (an HF repo id or a local snapshot
directory) for a different machine.
"""
import os

import torch

DEV = "cuda"

_QWEN3_SPEC = os.environ.get("QWEN3_MODEL_SPEC", "/home/ljw/projects/pi0.5/models/qwen3_4b")
_AVAILABLE = os.path.isdir(_QWEN3_SPEC) or _QWEN3_SPEC.count("/") <= 1  # local dir OR a bare "org/repo" HF id


def test_encode_prompts_real_shape():
    if not os.path.isdir(_QWEN3_SPEC):
        import pytest
        pytest.skip(f"Qwen3 checkpoint not present at {_QWEN3_SPEC} on this machine "
                    f"(set QWEN3_MODEL_SPEC to a local snapshot dir or leave unset to "
                    f"download 'Qwen/Qwen3-4B' from the Hub)")

    from flash_rt.models.imagewam.text_encoder import encode_prompts, load_real_text_encoder

    model, tokenizer = load_real_text_encoder(_QWEN3_SPEC)
    context, mask = encode_prompts(model, tokenizer, [
        "pick up the black bowl between the plate and the ramekin and place it on the plate"])

    assert context.shape == (1, 512, 7680), f"expected (1,512,7680), got {tuple(context.shape)}"
    assert mask.shape == (1, 512)
    assert context.dtype == torch.bfloat16, (
        "opportunities.md OPT-001 'FP16 residual overflow': context must stay BF16, "
        "not FP16 -- see text_encoder.py's own encode_prompts docstring")
    assert mask.dtype == torch.bool
    assert torch.isfinite(context).all(), "real Qwen3 encode produced NaN/Inf"
    num_real = mask.sum().item()
    assert 0 < num_real < 512, f"expected some real tokens and some padding, got {num_real}/512"
    print(f"PASS: context shape={tuple(context.shape)}, real tokens={num_real}/512, "
          f"mean={context.float().mean().item():.4f} std={context.float().std().item():.4f}")


if __name__ == "__main__":
    if not os.path.isdir(_QWEN3_SPEC):
        print(f"SKIPPED: Qwen3 checkpoint not present at {_QWEN3_SPEC}")
    else:
        test_encode_prompts_real_shape()
        print("PASS")
