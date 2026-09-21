"""Shared workload and observation construction for the ImageWAM benchmark
entry points (`benchmarks/imagewam_thor_path_bench.py`).

One place that turns a command line into an `ImageWAMWorkload` and builds
the two inputs a served tick needs: a random observation and a random text
context. The workload is the deployment's own description of what is served
(plan.md "Plan: configuration consolidation", W1); every sequence dim a
benchmark used to type by hand (`x0`, `img_len`, `a0`, `total`, `ref_h`,
`ref_w`, `dt`) is derived from it by `ImageWAMWorkload.layout`.

`WORKLOADS` holds two named workloads: `libero` (`ImageWAMWorkload.libero()`,
the served `ImageWAM-FLUX.2-4B-LIBERO` release) and `target`
(`TARGET_WORKLOAD`). `--workload` picks one and the nine `--<field>` flags
override individual fields on top of it; an unset flag keeps the named
workload's own value, so a run without overrides measures exactly the named
configuration.

`random_observation` returns host frames and a proprio vector, and
`random_context` returns the `(context, mask)` pair `set_prompt(context=...,
context_mask=...)` takes, built the way
`benchmarks/imagewam_text_trim_bench.py:_context_and_mask` builds it. Both
are random by design: they stand in for a camera and a text encoder, so a
benchmark that only reports latency needs no LIBERO data and no Qwen3.
"""
from __future__ import annotations

import argparse
from typing import TypeVar

import numpy as np
import torch

from flash_rt.models.imagewam.workload import ImageWAMWorkload

# The device and dtype the frontend's own context buffer holds
# (`imagewam_thor.DEV` / `BF16`): `set_prompt` casts to both, so building the
# context there allocates the served buffers and nothing else.
CUDA_DEVICE = "cuda"
BF16 = torch.bfloat16

# The deployment's candidate target configuration (a candidate, not a
# verified fact): three views at 256x256, a 32-step horizon, 128 padded text
# tokens. `plan.md`'s open list carries the same candidate and the field
# still open; pass `--text-max-len 512` when the deployment confirms a longer
# instruction set (`max_length` here is what the frontend pads the encoder to,
# ISSUE-083). Nothing here is measured or confirmed -- the bench that
# consumes it (`imagewam_thor_path_bench.py`) reports latency only.
TARGET_WORKLOAD = ImageWAMWorkload(
    num_views=3, image_h=256, image_w=256, text_max_len=128, action_horizon=32,
    action_dim=7, proprio_dim=8, num_steps=10, shift=5.0)

TARGET = "target"
LIBERO = "libero"

WORKLOADS: dict[str, ImageWAMWorkload] = {
    LIBERO: ImageWAMWorkload.libero(),
    TARGET: TARGET_WORKLOAD,
}

# A representative valid instruction length per named workload, for a bench
# that has no real prompts (`random_context`'s `valid_tokens`). LIBERO's
# served instructions carry 16-31 valid tokens; the target's are 16-128, so
# the short end of each is the representative one. A trim comparison sweeps
# this (`imagewam_thor_path_bench.py --valid-tokens`).
VALID_TOKENS: dict[str, int] = {LIBERO: 24, TARGET: 32}

_T = TypeVar("_T")


def _override(value: _T | None, default: _T) -> _T:
    """`value` when the caller set it, else the named workload's own field."""
    return default if value is None else value


def add_workload_args(parser: argparse.ArgumentParser) -> None:
    """Add `--workload` and its nine field overrides to `parser`.

    Each override is `None` by default, which keeps the named workload's own
    value; `--workload target --num-views 2` is therefore "`TARGET_WORKLOAD`
    with two views" and nothing else changes.
    """
    parser.add_argument("--workload", default=LIBERO, choices=tuple(WORKLOADS),
                        help=f"named workload: {LIBERO} = ImageWAMWorkload.libero(), "
                             f"{TARGET} = TARGET_WORKLOAD")
    parser.add_argument("--num-views", type=int, default=None, help="camera views (default: the workload's)")
    parser.add_argument("--image-h", type=int, default=None, help="per view, pixels")
    parser.add_argument("--image-w", type=int, default=None, help="per view, pixels")
    parser.add_argument("--text-max-len", type=int, default=None,
                        help="text tokens padded to; the sequence's x0 is this + 1 (the proprio row)")
    parser.add_argument("--action-horizon", type=int, default=None, help="actions per chunk")
    parser.add_argument("--action-dim", type=int, default=None, help="action dim after de-normalisation")
    parser.add_argument("--proprio-dim", type=int, default=None, help="proprio vector width")
    parser.add_argument("--num-steps", type=int, default=None, help="denoise steps")
    parser.add_argument("--shift", type=float, default=None, help="noise-schedule shift")


def workload_from_args(args: argparse.Namespace) -> ImageWAMWorkload:
    """The `ImageWAMWorkload` `args` describes: the named workload with every
    override that was actually passed applied."""
    base = WORKLOADS[args.workload]
    return ImageWAMWorkload(
        num_views=_override(args.num_views, base.num_views),
        image_h=_override(args.image_h, base.image_h),
        image_w=_override(args.image_w, base.image_w),
        text_max_len=_override(args.text_max_len, base.text_max_len),
        action_horizon=_override(args.action_horizon, base.action_horizon),
        action_dim=_override(args.action_dim, base.action_dim),
        proprio_dim=_override(args.proprio_dim, base.proprio_dim),
        num_steps=_override(args.num_steps, base.num_steps),
        shift=_override(args.shift, base.shift),
    )


def random_observation(workload: ImageWAMWorkload, seed: int) -> dict[str, object]:
    """One seeded random `(H, W, 3)` uint8 frame per view and a float32
    proprio vector of `workload.proprio_dim`, keyed the way the frontend's
    observation path reads them.

    The keys are `view1` ... `view<num_views>` plus `proprio`, which is
    `runtime_export.view_names(workload.num_views)`'s order and what
    `ImageWAMTorchFrontendThor`'s observation path reads
    (`observation_views(observation, num_views)`, the frontend's own
    `num_views`): one frame per view of the workload, so a three-view
    workload gets three keys and the frontend encodes all three.
    """
    rng = np.random.default_rng(seed)
    observation: dict[str, object] = {}
    for i in range(workload.num_views):
        observation[f"view{i + 1}"] = np.ascontiguousarray(
            rng.integers(0, 256, (workload.image_h, workload.image_w, 3), dtype=np.uint8))
    observation["proprio"] = rng.uniform(-0.5, 0.5, workload.proprio_dim).astype(np.float32)
    return observation


def view_frames(observation: dict[str, object], num_views: int) -> list[np.ndarray]:
    """`observation`'s camera frames in view order (`view1` first), as the
    `image_views` SWAP window stacks them."""
    return [np.ascontiguousarray(observation[f"view{i + 1}"]) for i in range(num_views)]


def random_context(workload: ImageWAMWorkload, joint_attention_dim: int, seed: int,
                   *, valid_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """`(context, mask)` for `set_prompt(context=..., context_mask=...)`:
    `context` is a BF16 CUDA `(text_max_len, joint_attention_dim)` tensor and
    `mask` a bool `(text_max_len,)`, both on the current CUDA device.

    `text_max_len` rows is what the frontend requires of a precomputed
    context: `dims["x0"]` is `text_max_len + 1`, and
    `_set_context_with_optional_proprio` raises unless the context length is
    exactly that when `dims["proprio_dim"]` is set.

    `valid_tokens`: how many of those rows are marked valid (`None` = all of
    them). This is what a `text_trim` comparison turns on: with the trim the
    active sequence is `valid_tokens + 1` rows, without it the full
    `text_max_len + 1`, so an all-valid mask measures the trim as a no-op.
    """
    if valid_tokens is None:
        valid_tokens = workload.text_max_len
    if not 0 < valid_tokens <= workload.text_max_len:
        raise ValueError(f"valid_tokens={valid_tokens} must be in 1..{workload.text_max_len}")
    generator = torch.Generator().manual_seed(seed)
    context = torch.randn(workload.text_max_len, joint_attention_dim, generator=generator
                          ).to(device=CUDA_DEVICE, dtype=BF16)
    mask = torch.zeros(workload.text_max_len, dtype=torch.bool, device=CUDA_DEVICE)
    mask[:valid_tokens] = True
    return context, mask


__all__ = [
    "CUDA_DEVICE", "BF16", "LIBERO", "TARGET", "TARGET_WORKLOAD", "WORKLOADS",
    "add_workload_args", "random_context", "random_observation", "view_frames", "workload_from_args",
]
