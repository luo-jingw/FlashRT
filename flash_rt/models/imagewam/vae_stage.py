"""Fixed-address ImageWAM VAE stage, capturable into a CUDA graph
(roadmap item 5, `plan.md`).

`ImageWAMVaeStage` owns the stage's device state at fixed addresses:

- `views_u8`: `(num_views, in_h, in_w, 3)` uint8, the camera views;
- `image`: `(1, 3, out_h, num_views * out_w)` BF16, the VAE input, each
  view encoded at `VaeStageSpec.encode_hw` (the views' own `in_h x in_w`
  by default: the workload's per-view size);

and writes the encoder's tokens into a caller-owned `img_raw`
`(img_len, 128)` BF16 buffer (the frontend's backbone input), in the
same row-major `(h, w)` token order `vae_encoder.encode_to_tokens`
produces. `stage()` copies new views in; `run()` issues preprocessing
(`VaePreprocessor`), `encoder.encode`, and the token write on the
current CUDA stream. `run()` reads only the fixed buffers, so it can be
recorded once into a CUDA graph and replayed after every `stage()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from flash_rt.models.imagewam.vae_preprocess import VaePreprocessor

# FLUX.2 AutoEncoder: 8x conv downsampling, then a 2x2 patch merge.
VAE_SPATIAL_FACTOR = 16
VAE_TOKEN_DIM = 128


class VaeEncoder(Protocol):
    """`flux2.autoencoder.AutoEncoder` and `NativeFlux2Encoder` both
    satisfy this: `(1,3,H,W)` BF16 image -> `(1,128,H/16,W/16)` BF16."""

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class VaeStageSpec:
    """Static input shape of the stage: `num_views` camera views of
    `in_h x in_w` uint8 RGB, concatenated horizontally into the one image
    the encoder takes, `encode_hw` per view."""
    num_views: int
    in_h: int
    in_w: int
    out_hw: tuple[int, int] | None = None

    @property
    def encode_hw(self) -> tuple[int, int]:
        """The per-view size each view is encoded at: the views' own
        `in_h x in_w` unless the caller pinned `out_hw`.

        The views this stage stages are the workload's cameras, delivered
        at the workload's `image_h x image_w`, and that is the size
        `ImageWAMWorkload.layout` divides by the patch stride to get the
        token grid the graph's `img_raw`, the image RoPE and the VAE's own
        output all carry. The encode size therefore follows those two
        fields -- a fixed 224x224 grid mismatches every workload but
        LIBERO's. `out_hw` stays for a caller that stages a different
        delivered size and resizes (`benchmarks/imagewam_vae_stage_bench.py`
        encodes 512x512 LIBERO frames at 224x224).
        """
        if self.out_hw is None:
            return (self.in_h, self.in_w)
        return self.out_hw

    @property
    def latent_hw(self) -> tuple[int, int]:
        """The VAE's token grid: each view contributes
        `encode_hw / VAE_SPATIAL_FACTOR`, the views side by side along the
        width."""
        h, w = self.encode_hw
        return (h // VAE_SPATIAL_FACTOR, self.num_views * w // VAE_SPATIAL_FACTOR)

    @property
    def img_len(self) -> int:
        h, w = self.latent_hw
        return h * w


class ImageWAMVaeStage:
    """Fixed-address VAE stage: staged uint8 views -> BF16 image ->
    encoder -> tokens in `img_raw`."""

    def __init__(self, encoder: VaeEncoder, preprocessor: VaePreprocessor,
                 spec: VaeStageSpec, img_raw: torch.Tensor) -> None:
        if spec.num_views < 1:
            raise ValueError(f"num_views={spec.num_views} must be >= 1")
        if tuple(preprocessor.out_hw) != tuple(spec.encode_hw):
            raise ValueError(f"preprocessor.out_hw={preprocessor.out_hw} != spec.encode_hw={spec.encode_hw}")
        if spec.encode_hw[0] % VAE_SPATIAL_FACTOR or spec.encode_hw[1] % VAE_SPATIAL_FACTOR:
            raise ValueError(f"spec.encode_hw={spec.encode_hw} must be divisible by {VAE_SPATIAL_FACTOR}")
        expected = (spec.img_len, VAE_TOKEN_DIM)
        if tuple(img_raw.shape) != expected or img_raw.dtype != torch.bfloat16 or not img_raw.is_contiguous():
            raise ValueError(f"img_raw must be contiguous BF16 {expected} for {spec}, got "
                             f"{tuple(img_raw.shape)} {img_raw.dtype}")
        self.spec = spec
        self._encoder = encoder
        self._preprocessor = preprocessor
        device = img_raw.device
        out_h, out_w = spec.encode_hw
        self.views_u8 = torch.zeros(spec.num_views, spec.in_h, spec.in_w, 3, dtype=torch.uint8, device=device)
        self.image = torch.zeros(1, 3, out_h, spec.num_views * out_w, dtype=torch.bfloat16, device=device)
        self._view_slices = [self.views_u8[i] for i in range(spec.num_views)]
        lat_h, lat_w = spec.latent_hw
        self._tokens_nhwc = img_raw.view(1, lat_h, lat_w, VAE_TOKEN_DIM)
        preprocessor.prepare(spec.in_h, spec.in_w)

    def stage(self, views: Sequence[torch.Tensor]) -> None:
        """Copies `views` (each `(in_h, in_w, 3)` uint8, CPU or CUDA)
        into the fixed `views_u8` buffer on the current stream. Every
        view is validated before any is copied, so a rejected call leaves
        the buffer unchanged."""
        if len(views) != self.spec.num_views:
            raise ValueError(f"expected {self.spec.num_views} views, got {len(views)}")
        shape = (self.spec.in_h, self.spec.in_w, 3)
        for i, v in enumerate(views):
            if v.dtype != torch.uint8 or tuple(v.shape) != shape:
                raise ValueError(f"view {i} must be uint8 {shape} (the captured VAE input shape), "
                                 f"got {tuple(v.shape)} {v.dtype}")
        for i, v in enumerate(views):
            self._view_slices[i].copy_(v)

    @torch.no_grad()
    def run(self) -> None:
        """Preprocess -> encode -> tokens into `img_raw`, on the current stream."""
        self._preprocessor.run(self._view_slices, self.image, torch.cuda.current_stream().cuda_stream)
        z = self._encoder.encode(self.image)
        self._tokens_nhwc.copy_(z.permute(0, 2, 3, 1))
