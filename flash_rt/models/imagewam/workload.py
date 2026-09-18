"""ImageWAM workload: what a deployment serves, and the sequence layout
derived from it (plan.md, "configuration consolidation", W1).

`ImageWAMWorkload` holds only the served quantities (cameras, image size,
text length, action horizon, proprio dim, denoise loop). `layout()` derives
what the frontend used to take as hand-entered, mutually dependent
numbers (`x0`, `img_len`, `a0`, `total`, `ref_h`, `ref_w`, `dt`) and
rejects inconsistent input.

Derivation (checked against `vae_stage.py`, `vae_encoder.py`, `rope.py`):
each view is `image_h x image_w`; the views are concatenated along the
width, so the image is `image_h x (num_views * image_w)`; the VAE plus its
2x2 patch merge divide both sides by `patch_stride` (16), giving the
`ref_h x ref_w` token grid the image RoPE is built on. The backbone
sequence is `[text | proprio | image | action]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class SequenceLayout:
    """Row layout of the backbone sequence for one workload."""
    x0: int          # text rows + 1 proprio row
    img_len: int     # image tokens (ref_h * ref_w)
    a0: int          # x0 + img_len, first action row
    total: int       # a0 + action horizon
    ref_h: int       # image token grid height
    ref_w: int       # image token grid width (views side by side)
    dt: float        # ODE step, 1 / num_steps


def _positive_int(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name}={value!r} must be an integer")
    if value <= 0:
        raise ValueError(f"{name}={value} must be > 0")


@dataclass(frozen=True)
class ImageWAMWorkload:
    num_views: int
    image_h: int                 # per view, pixels (the size the VAE encodes)
    image_w: int
    text_max_len: int            # tokens padded to (512 for LIBERO)
    action_horizon: int          # <= structure.max_action_horizon
    action_dim: int              # output dim after de-normalisation
    proprio_dim: int
    num_steps: int               # denoise steps
    shift: float                 # schedule shift
    num_train_timesteps: int = 1000

    def __post_init__(self) -> None:
        for name in ("num_views", "image_h", "image_w", "text_max_len", "action_horizon",
                     "action_dim", "proprio_dim", "num_steps", "num_train_timesteps"):
            _positive_int(name, getattr(self, name))
        if not self.shift > 0:
            raise ValueError(f"shift={self.shift!r} must be > 0")

    @staticmethod
    def libero() -> "ImageWAMWorkload":
        """`ImageWAM-FLUX.2-4B-LIBERO`: two 224x224 views, 512 text tokens,
        64-step horizon, 8-dim proprio, 10 steps, shift 5.0."""
        return ImageWAMWorkload(
            num_views=2, image_h=224, image_w=224, text_max_len=512, action_horizon=64,
            action_dim=7, proprio_dim=8, num_steps=10, shift=5.0)

    def layout(self, structure) -> SequenceLayout:
        """Sequence layout under `structure` (any object exposing
        `patch_stride` and `max_action_horizon`)."""
        stride = structure.patch_stride
        _positive_int("structure.patch_stride", stride)
        if self.image_h % stride:
            raise ValueError(f"image_h={self.image_h} must be a multiple of the patch stride {stride}")
        if self.image_w % stride:
            raise ValueError(f"image_w={self.image_w} must be a multiple of the patch stride {stride}")
        if self.action_horizon > structure.max_action_horizon:
            raise ValueError(f"action_horizon={self.action_horizon} exceeds the structure limit "
                             f"max_action_horizon={structure.max_action_horizon}")
        ref_h = self.image_h // stride
        ref_w = self.num_views * self.image_w // stride
        img_len = ref_h * ref_w
        x0 = self.text_max_len + 1
        a0 = x0 + img_len
        return SequenceLayout(x0=x0, img_len=img_len, a0=a0, total=a0 + self.action_horizon,
                              ref_h=ref_h, ref_w=ref_w, dt=1.0 / self.num_steps)

    def vae_graph_input(self) -> tuple[int, int, int]:
        """`(num_views, in_h, in_w)` of the in-graph VAE stage's uint8 view
        buffer, for views delivered at the workload's image size."""
        return (self.num_views, self.image_h, self.image_w)
