"""Versioned ImageWAM activation-calibration file.

Holds, for every GEMM input site of the served pipeline (site names from
`activation_recorder.site_name`, e.g. `backbone.single.3.linear1.weight`),
statistics recorded on real LIBERO observations through the real fp16
pipeline, reduced across samples with the house reducer
(`flash_rt.core.calibration.accumulate_amax`, default percentile 99.9):

- `amax`: per-tensor |x| max -> the static FP8 activation scale
  `amax / 448` (float32, same as `quantize_fp8_device_fp16`'s
  `compute_scale_kernel`);
- `channel_amax`: per-input-channel |x| max (length K), the AWQ statistic;
- `abs_percentiles`: the 99 / 99.9 / 99.99th percentiles of |x|
  (diagnostics);
- `sample_absmax`: each sample's own amax before the reduction.

Identity: the checkpoint (`flash_rt.core.quant.calibrator._checkpoint_hash`:
SHA-256 of the first 64KB + file size, first 16 hex chars), every dims
entry that changes GEMM shapes or activation distributions
(`IDENTITY_DIM_KEYS`), and `text_trim`: whether the statistics were
recorded with the text context trimmed to the prompt's valid tokens
(`ImageWAMTorchFrontendThor(text_trim=True)`, issues.md ISSUE-020).
Untrimmed text and single-stream GEMM inputs include about 490 padded
context rows that a trimmed frontend never computes. A frontend refuses a
file whose identity differs from its own.

The dims include the workload's camera geometry: `num_views`, `image_h`,
`image_w`. The sequence layout alone does not fix it (two 224x224 views and
four 224x112 views both give `ref_h x ref_w = 14 x 28`, the same `x0` and
`a0`), yet the VAE input, and so the image-token statistics, differ. A
frontend supplies the three keys only when its dims come from a workload:
build it through `load_imagewam` / `ImageWAMTorchFrontendThor.from_config`,
or include them in `dims_override` (`libero_dims.LIBERO_REAL_DIMS` has
them). A frontend built by hand without them cannot validate a file, and
the error says so.

Format version: 3 (the workload identity above). There is no reader for
earlier files: version 1 (no `text_trim`) and version 2 (no camera
geometry) predate the workload identity, `load_calibration` refuses them,
and they are re-recorded with `benchmarks/imagewam_build_calibration.py`
rather than migrated.

On disk: one safetensors file. Arrays are tensors named
`<site>.channel_amax` / `<site>.sample_absmax`; everything else is JSON
in the metadata entry `imagewam_calibration`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from flash_rt.core.calibration import accumulate_amax
from flash_rt.core.quant.calibrator import _checkpoint_hash
from flash_rt.models.imagewam.activation_recorder import ABS_PERCENTILES, SampleStats

FORMAT_NAME = "imagewam_activation_calibration"
FORMAT_VERSION = 3
# Versions `load_calibration` reads: the current one only. Files of an
# earlier version are refused, not migrated (see `load_calibration`).
SUPPORTED_VERSIONS = (3,)
FP8_E4M3_MAX = 448.0
DEFAULT_PERCENTILE = 99.9
IDENTITY_DIM_KEYS = (
    "hidden", "HD", "NH", "mlp_hidden", "joint_attention_dim", "x0", "a0",
    "num_layers_double", "num_layers_single", "action_hidden_dim", "action_attn_width",
    "action_mlp_hidden", "num_action", "action_dim", "action_num_layers_double",
    "action_num_layers_single", "num_denoise_steps", "shift", "num_train_timesteps",
    "proprio_dim", "ref_h", "ref_w", "merge_qkv_mlp", "merge_linear2",
    # The workload's camera geometry (`ImageWAMWorkload.num_views/image_h/image_w`).
    "num_views", "image_h", "image_w",
)


@dataclass
class SiteCalibration:
    amax: float
    abs_percentiles: np.ndarray   # (len(ABS_PERCENTILES),) float32
    channel_amax: np.ndarray      # (K,) float32
    sample_absmax: np.ndarray     # (num_samples,) float32
    rows: int

    def fp8_act_scale(self) -> float:
        """Static FP8 E4M3 activation scale, `amax / 448` in float32 (IEEE
        divide), floored at 1e-12 as `compute_scale_kernel`
        (`csrc/kernels/quantize.cu`) does. The kernel is built with
        `--use_fast_math`, so its device result can differ from this one by one
        float32 ULP (seen on Thor: 0.0093122218 against 0.0093122208)."""
        s = np.float32(self.amax) / np.float32(FP8_E4M3_MAX)
        return float(max(s, np.float32(1e-12)))


@dataclass
class ImageWAMCalibration:
    version: int
    checkpoint_id: str
    checkpoint_size: int
    dims: dict
    percentile: float
    frames: list[tuple[str, int, int]]   # (suite, episode, frame)
    noise: str
    sites: dict[str, SiteCalibration]
    text_trim: bool

    def validate_for(self, *, checkpoint_path: str, dims: dict, text_trim: bool) -> None:
        """Raise `ValueError` unless this file was built for exactly this
        checkpoint, these dims and this `text_trim` setting.

        `dims` must carry every `IDENTITY_DIM_KEYS` entry, the camera
        geometry (`num_views`, `image_h`, `image_w`) included: a frontend
        built by hand without them is refused (the diff shows the file's
        value against `None`)."""
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"calibration file version {self.version} not in {SUPPORTED_VERSIONS}")
        if bool(text_trim) != self.text_trim:
            raise ValueError(f"calibration file was recorded with text_trim={self.text_trim}, the frontend "
                             f"runs text_trim={bool(text_trim)} (the text and single-stream GEMM inputs differ)")
        ckpt_id, ckpt_size = checkpoint_identity(checkpoint_path)
        if (ckpt_id, ckpt_size) != (self.checkpoint_id, self.checkpoint_size):
            raise ValueError(f"calibration file is for checkpoint {self.checkpoint_id} "
                             f"({self.checkpoint_size} bytes), not {checkpoint_path} "
                             f"({ckpt_id}, {ckpt_size} bytes)")
        want = identity_dims(dims)
        if want != self.dims:
            diff = {k: (self.dims.get(k), want.get(k)) for k in sorted(set(want) | set(self.dims))
                    if self.dims.get(k) != want.get(k)}
            missing = sorted(k for k in diff if k not in want)
            hint = ""
            if missing:
                hint = (f"; the frontend's dims do not carry {missing}: build the frontend through "
                        f"load_imagewam / ImageWAMTorchFrontendThor.from_config, or include those keys "
                        f"in dims_override")
            raise ValueError(f"calibration file dims differ (file, frontend): {diff}{hint}")


def checkpoint_identity(checkpoint_path: str) -> tuple[str, int]:
    return _checkpoint_hash(checkpoint_path), os.path.getsize(checkpoint_path)


def identity_dims(dims: dict) -> dict:
    return {k: dims[k] for k in IDENTITY_DIM_KEYS if k in dims}


def build_calibration(samples: list[SampleStats], *, percentile: float, checkpoint_path: str,
                      dims: dict, frames: list[tuple[str, int, int]], noise: str,
                      text_trim: bool) -> ImageWAMCalibration:
    """Reduce per-sample statistics across samples with the house
    percentile reducer (`accumulate_amax`: linear interpolation along the
    sample axis; 100.0 is the plain max). `text_trim`: the recording
    frontend's setting. With it, a site's `rows` (sample 0's) varies with
    the prompt length for the text and single-stream sites."""
    if not samples:
        raise ValueError("build_calibration needs at least one sample")
    names = sorted(samples[0].sites)
    for i, s in enumerate(samples):
        if sorted(s.sites) != names:
            raise ValueError(f"sample {i} recorded a different site set")
    per_sample_amax = [np.array([s.sites[n].absmax for n in names], dtype=np.float32) for s in samples]
    amax = accumulate_amax(per_sample_amax, percentile=percentile)
    sites = {}
    for j, n in enumerate(names):
        pct = accumulate_amax([s.sites[n].abs_percentiles for s in samples], percentile=percentile)
        ch = accumulate_amax([s.sites[n].channel_amax for s in samples], percentile=percentile)
        sites[n] = SiteCalibration(
            amax=float(amax[j]), abs_percentiles=pct.astype(np.float32),
            channel_amax=ch.astype(np.float32),
            sample_absmax=np.array([s.sites[n].absmax for s in samples], dtype=np.float32),
            rows=samples[0].sites[n].rows)
    ckpt_id, ckpt_size = checkpoint_identity(checkpoint_path)
    return ImageWAMCalibration(version=FORMAT_VERSION, checkpoint_id=ckpt_id, checkpoint_size=ckpt_size,
                               dims=identity_dims(dims), percentile=float(percentile),
                               frames=[tuple(f) for f in frames], noise=noise, sites=sites,
                               text_trim=bool(text_trim))


def save_calibration(cal: ImageWAMCalibration, path: str) -> None:
    tensors = {}
    site_meta = {}
    for n, s in cal.sites.items():
        tensors[f"{n}.channel_amax"] = torch.from_numpy(np.ascontiguousarray(s.channel_amax, dtype=np.float32))
        tensors[f"{n}.sample_absmax"] = torch.from_numpy(np.ascontiguousarray(s.sample_absmax, dtype=np.float32))
        site_meta[n] = {"amax": s.amax, "rows": s.rows,
                        "abs_percentiles": [float(v) for v in s.abs_percentiles]}
    meta = {
        "format": FORMAT_NAME, "version": cal.version,
        "checkpoint_id": cal.checkpoint_id, "checkpoint_size": cal.checkpoint_size,
        "dims": cal.dims, "percentile": cal.percentile,
        "abs_percentile_levels": list(ABS_PERCENTILES),
        "frames": [list(f) for f in cal.frames], "noise": cal.noise, "sites": site_meta,
        "text_trim": cal.text_trim,
    }
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata={"imagewam_calibration": json.dumps(meta)})
    os.replace(tmp, path)


def load_calibration(path: str) -> ImageWAMCalibration:
    """Read a version-3 file. A file of an earlier version predates the
    workload identity and raises `ValueError` naming its version; it is
    re-recorded, not migrated."""
    with safe_open(path, framework="pt") as f:
        raw = (f.metadata() or {}).get("imagewam_calibration")
        if raw is None:
            raise ValueError(f"{path}: not an ImageWAM calibration file (no metadata)")
        meta = json.loads(raw)
        if meta.get("format") != FORMAT_NAME:
            raise ValueError(f"{path}: format {meta.get('format')!r} != {FORMAT_NAME!r}")
        version = int(meta["version"])
        if version < FORMAT_VERSION:
            raise ValueError(f"{path}: calibration file version {version} predates the workload identity "
                             f"(camera geometry num_views/image_h/image_w, format version {FORMAT_VERSION}) "
                             f"and is not read; re-record it with benchmarks/imagewam_build_calibration.py")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"{path}: calibration file version {version} not in {SUPPORTED_VERSIONS}")
        sites = {}
        for n, sm in meta["sites"].items():
            sites[n] = SiteCalibration(
                amax=float(sm["amax"]),
                abs_percentiles=np.asarray(sm["abs_percentiles"], dtype=np.float32),
                channel_amax=f.get_tensor(f"{n}.channel_amax").numpy(),
                sample_absmax=f.get_tensor(f"{n}.sample_absmax").numpy(),
                rows=int(sm["rows"]))
    return ImageWAMCalibration(
        version=version, checkpoint_id=meta["checkpoint_id"],
        checkpoint_size=int(meta["checkpoint_size"]), dims=meta["dims"],
        percentile=float(meta["percentile"]), frames=[tuple(f) for f in meta["frames"]],
        noise=meta["noise"], sites=sites, text_trim=bool(meta["text_trim"]))
