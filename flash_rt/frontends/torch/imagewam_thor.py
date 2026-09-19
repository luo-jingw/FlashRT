"""ImageWAM (FLUX.2-4B variant) Thor torch frontend.

**Rewritten 2026-09-14** alongside `pipeline_thor.py`'s own real-math
rewrite (opportunities.md OPT-002) -- weights here are still
random-initialized (see `checkpoint_dir` below), but the MATH they now
feed is real: per-head K/V, real 4-axis RoPE, QK-Norm, real AdaLN
modulation, real SiLU-gated-GLU MLP widths, and the real (no-mask)
attention rule, confirmed against the real trained checkpoint on Thor
(see `benchmarks/imagewam_real_checkpoint_validation.py`). This is now
the confirmed target for real Thor deployment (`PROJECT.md`'s
"Confirmed end goal"), following `_template/frontend.py`'s STEP 1-6
shape -- adapted where the real, working `CosmosEdgeThor`
(`flash_rt/models/cosmos3_edge/pipeline_thor.py`) precedent differs
from the generic template (plain `torch.cuda.Tensor` + `.data_ptr()`
throughout, not the template's `CudaBuffer` ctypes wrapper).

`checkpoint_dir` is accepted for interface parity with every other
FlashRT frontend but still unused: every shape here is random-filled
from `dims`, never loaded from a real checkpoint. Real checkpoint
loading needs the actual `imagewam`/`flux2` Python packages (only
available on Thor -- see `PROJECT.md`'s "Real checkpoint testing
happens ONLY on Thor" note) and would reuse
`benchmarks/imagewam_real_checkpoint_validation.py`'s own
`extract_*_weights` functions (already verified against the real
checkpoint, cosine=0.9999+) rather than re-deriving weight extraction
here -- tracked as still-open in `opportunities.md` OPT-002.

`context_mask` (declared as an input shape in `_imagewam_thor_spec.py`)
places the proprio row, and with `text_trim=True` it also sets the
sequence length: official ImageWAM masks the padded text keys for every
query, and the trimmed sequence (valid tokens + proprio row only)
computes that same math with no mask (`text_context.py`,
issues.md ISSUE-020). With `text_trim=False` every context row is
attended to, padding included.

AdaLN modulation and RoPE tables are precomputed ONCE here (backbone's
own conditioning timestep is fixed, ActionDiT's varies per denoise
step but `step` is itself a compile-time constant during graph
capture -- see `pipeline_thor.py`'s own module docstring) and passed
into the captured graph as small, fixed-address read-only buffers --
never recomputed per replay.

Dims default to a small, deliberately-not-real-FLUX.2-4B-size
structural test scale (this machine's own 8GB GPU headroom, per
PROJECT.md) -- pass `dims_override` for Thor-scale testing. `HD=128`
is NOT a free "keep it small" parameter here (unlike the other dims):
real 4-axis RoPE (`axes_dim=(32,32,32,32)`, opportunities.md OPT-002)
sums to a fixed 128, so every default/override dims dict below keeps
`HD=128` and only shrinks `NH`/`hidden`/`mlp_hidden`/sequence lengths.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor import fa4_backend
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.config_resolver import ResolvedConfig
from flash_rt.models.imagewam.gemm_variant_timer import CudaGraphVariantTimer
from flash_rt.models.imagewam.gemm_variant_tuner import GemmVariantTuner, VariantTuneResult
from flash_rt.models.imagewam.pipeline_resources import (
    ActionDoubleLayerResource,
    ActionStepResource,
    AdaLNResource,
    AttentionResource,
    DoubleLayerResource,
    ImageWAMPipelineResources,
    LinearResource,
    PipelineBuffers,
    PipelineDims,
    SingleLayerResource,
    linear_resource,
)
from flash_rt.models.imagewam.pipeline_thor import (
    fp16_adaln_operands,
    fp16_adaln_shift_scale,
    imagewam_denoise_loop,
    imagewam_prefill,
)
from flash_rt.models.imagewam.pipeline_real import (
    compute_action_head_modulation,
    compute_action_modulation,
    compute_shared_modulation,
)
from flash_rt.models.imagewam.quant_linear import (
    Bf16OutLinear,
    CutlassFp16Linear,
    CutlassFp16SwiGluMlp,
    E0m3HadamardLinear,
    Fp8Linear,
    Fp16Linear,
    Nvfp4Linear,
    SimNvfp4Linear,
    StaticFp8Linear,
)
from flash_rt.models.imagewam.rope import build_action_rope_table, build_backbone_rope_table
from flash_rt.models.imagewam.text_context import pack_trimmed_context, trimmed_sequence_dims
from flash_rt.models.imagewam.precision import Precision
from flash_rt.models.imagewam.runtime_surface import ImageWAMRuntimeSurface, workload_identity
from flash_rt.models.imagewam.vae_preprocess import RESIZE_MODES, VaePreprocessor
from flash_rt.models.imagewam.vae_stage import ImageWAMVaeStage, VaeStageSpec

if TYPE_CHECKING:
    # Type-only: `runtime.export` loads the built runtime module on import,
    # and the frontend works without it.
    from flash_rt.models.imagewam.native_runtime import ImageWAMNativeRuntime
    from flash_rt.models.imagewam.structure import ImageWAMStructure
    from flash_rt.models.imagewam.workload import ImageWAMWorkload
    from flash_rt.runtime.export import ModelRuntime

# The four precision sets below are VIEWS of `precision.py`'s
# `Precision` / `PROPERTIES` table (plan W7): which precision needs
# calibration, takes `awq_inv_s`, or has a switchable CUTLASS tile is
# stated there once, and the module-level names stay because other
# modules and tests import them.
_PRECISIONS = tuple(p.value for p in Precision)
# `nvfp4_sim`: NVFP4 numerics emulated with fp16 GEMMs (SimNvfp4Linear,
# bit-exact quantizer), for accuracy work on GPUs without Blackwell FP4.
# Not a fast path.
_NVFP4_PRECISIONS = tuple(p.value for p in Precision if p.supports_awq)
# OPT-004 step 6 (plan.md): the two `StaticFp8Linear` variants need a
# one-time calibration call in set_prompt() before graph capture (see
# _calibrate_fp8 below) -- everything else needs no such step.
_STATIC_FP8_PRECISIONS = tuple(p.value for p in Precision if p.needs_calibration)
# Precisions whose GEMMs run a switchable CUTLASS tile (roadmap item 1,
# plan.md "Plan: ActionDiT small-M CUTLASS tile selection"): the only ones
# `gemm_variant_autotune=True` applies to.
_VARIANT_TUNED_PRECISIONS = tuple(p.value for p in Precision if p.supports_tile_autotune)
# FA4 at the "backbone" site stays opt-in until Thor confirms it at the served
# shapes, inside the captured graph, end to end (opportunities.md OPT-019).
# `use_fa4=None` resolves to False unless this environment variable is "1"
# (then FA4 is used exactly when `fa4_backend.thor_default_enabled()` holds).
# Making FA4 the default is a one-line change: the "0" below becomes "1".
_FA4_OPT_IN_ENV = "FLASHRT_THOR_FA4"
_FA4_OPT_IN_DEFAULT = "0"

logger = logging.getLogger(__name__)

# Roadmap item 5 (plan.md): which real VAE encoder runs.
#   "torch"  -- flux2.autoencoder.AutoEncoder.encode, NCHW (default).
#   "native" -- NativeFlux2Encoder: NHWC convolutions + the FlashRT
#               GroupNorm(+SiLU) / bias+residual kernels; same math,
#               near-exact tokens (vae_native_encoder.py).
# Where it runs is set by `vae_graph_input`: None -> outside the CUDA
# graph, once per infer(); (num_views, H, W) -> inside the main graph
# (`ImageWAMVaeStage` over a fixed uint8 view buffer of that shape).
_VAE_ENCODERS = ("torch", "native")
# Stage 3 default precision decision (opportunities.md, real Thor
# checklist against real checkpoint weights + real open-loop LIBERO
# data): nvfp4 is the fastest AND closest to fp16/GT (actions
# cosine=0.9998 vs fp16, open-loop MAE 1.01x fp16's own). fp8_static*
# was measured with the N(0,0.1) placeholder activation scales
# (backbone_hidden cosine ~0.46); with a real calibration file
# (calibration_path=, opportunities.md OPT-022) it tracks fp16 on H100.
# `nvfp4` is a real production default here, not just a benchmark-only
# opt-in.

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16
F32 = torch.float32
_LOG = logging.getLogger(__name__)

_DEFAULT_DIMS = dict(
    hidden=256, HD=128, NH=2, mlp_hidden=384, joint_attention_dim=64,
    x0=3, a0=8, num_layers_double=2, num_layers_single=3,
    action_hidden_dim=128, action_attn_width=256, action_mlp_hidden=192,
    action_dim=7,  # OPT-001: real LIBERO 7-DoF action width
    num_action=4, total=12,
    action_num_layers_double=2, action_num_layers_single=3,
    dt=0.5, num_denoise_steps=2,
)

# The camera geometry of a frontend that NAMES NO WORKLOAD: a caller that
# passes `dims_override` by hand. LIBERO's own two 224x224 views, which is
# what `runtime_surface()` reported and what the observation path took
# before the workload object existed; a workload-resolved frontend
# (`from_config` / `load_imagewam`) reports its own instead.
_NO_WORKLOAD_VIEW_SHAPE = (2, 224, 224)


def observation_views(observation: dict, num_views: int) -> list:
    """The `num_views` camera frames of an `infer()` observation, in view
    order: `[observation["view1"], ..., observation[f"view{num_views}"]]`.

    `num_views` is the served workload's camera count
    (`ImageWAMTorchFrontendThor.num_views`). A missing key raises
    `ValueError` naming it and the expected count -- an observation carrying
    fewer views than the workload serves is a mismatch, not a shorter run.
    """
    views = []
    for i in range(num_views):
        key = f"view{i + 1}"
        if key not in observation:
            last = f"view{num_views}"
            raise ValueError(f"observation[{key!r}] is missing: this frontend's workload has "
                             f"num_views={num_views}, so infer() needs observation['view1'] ... "
                             f"observation[{last!r}]")
        views.append(observation[key])
    return views


def frontend_kwargs_from_config(resolved: ResolvedConfig, *, ckpt_path: str | None = None,
                               ae_model_path: str | None = None, flux2_src: str | None = None,
                               qwen3_model_spec: str | None = None,
                               dataset_stats_path: str | None = None,
                               vae_resize: str = "area") -> dict:
    """The `ImageWAMTorchFrontendThor.__init__` keyword arguments for
    `resolved`, as one dict.

    This is the ONE place that maps a resolved configuration onto the
    constructor's arguments: `dims_override=resolved.frontend_dims()`
    (the resolver's `dims` plus the two merge flags a `dims_override`
    carries) and one entry per field of `resolved.options`
    (`ImageWAMOptions`), each under the constructor's own keyword name --
    `precision` as the plain string `str(resolved.options.precision)`.
    Because the mapping lives here, a CPU-only test pins it against the
    constructor without building a frontend.

    `vae_graph_input` and `use_fa4_mot` are the resolver's own derived
    values (`ImageWAMWorkload.vae_graph_input()` through the profile's
    `vae_graph`, and the profile's FA4-mot tier); nothing is re-derived
    and no switch is added that the constructor does not already have.

    The arguments the resolver never sees -- the checkpoint and encoder
    paths (`ckpt_path`, `ae_model_path`, `flux2_src`,
    `qwen3_model_spec`, `dataset_stats_path`) and `vae_resize` -- are
    passed through here, so the caller is
    `cls(**frontend_kwargs_from_config(resolved, ckpt_path=...))`
    (`ImageWAMTorchFrontendThor.from_config`).

    A `ResolvedConfig` holds plain data: this allocates nothing, loads
    no checkpoint and touches no CUDA device.
    """
    options = resolved.options
    return {
        "dims_override": resolved.frontend_dims(),
        "precision": str(options.precision),
        "text_trim": options.text_trim,
        "use_fa4": options.use_fa4,
        "use_fa4_mot": options.use_fa4_mot,
        "vae_encoder": options.vae_encoder,
        "vae_graph_input": options.vae_graph_input,
        "nvfp4_awq": options.nvfp4_awq,
        "calibration_path": options.calibration_path,
        "awq_alpha": options.awq_alpha,
        "awq_scope": options.awq_scope,
        "gemm_variant_autotune": options.gemm_variant_autotune,
        "gemm_runner": options.gemm_runner,
        "ckpt_path": ckpt_path,
        "ae_model_path": ae_model_path,
        "flux2_src": flux2_src,
        "qwen3_model_spec": qwen3_model_spec,
        "dataset_stats_path": dataset_stats_path,
        "vae_resize": vae_resize,
    }


@dataclass(frozen=True)
class TextLengthCapture:
    """One captured forward at one text-context length (`text_trim`).

    `dims`: the frontend's dims with this length's `x0/a0/total` (the
    frontend's own `dims` object for the max length); `rope_table`:
    `(a0, 128)` FP16 backbone RoPE for these rows; `graph`: the captured
    VAE stage (if any) + prefill + denoise loop over the frontend's
    max-size buffers.
    """
    dims: dict
    rope_table: torch.Tensor
    graph: torch.cuda.CUDAGraph


class ImageWAMTorchFrontendThor:
    """Thor frontend for ImageWAM's real-math (random-weight) dry run.

    Naming rule: `<Model><Framework>Frontend<Hardware>` per
    `docs/adding_new_model.md` §0 rule 2.
    """

    def __init__(self, checkpoint_dir=None, *, dims_override: dict | None = None,
                 use_fa4: bool | None = None, precision: str = "nvfp4",
                 ckpt_path: str | None = None,
                 ae_model_path: str | None = None, flux2_src: str | None = None,
                 qwen3_model_spec: str | None = None,
                 dataset_stats_path: str | None = None,
                 calibration_path: str | None = None,
                 nvfp4_awq: bool = False, awq_alpha: float = 0.5, awq_scope: str = "adaln+down",
                 gemm_variant_autotune: bool = False,
                 use_fa4_mot: bool = False,
                 gemm_runner: object | None = None,
                 vae_resize: str = "area",
                 vae_encoder: str = "torch",
                 vae_graph_input: tuple[int, int, int] | None = None,
                 text_trim: bool = False,
                 **kwargs):
        del checkpoint_dir, kwargs
        if precision not in _PRECISIONS:
            raise ValueError(f"precision={precision!r} -- must be one of {_PRECISIONS}")
        # issues.md ISSUE-020: run each prompt at its valid text length
        # (`x0 = n_valid + 1` with proprio) instead of the padded
        # `dims["x0"]`, one CUDA graph per distinct length over the same
        # max-size buffers. Opt-in until Thor confirms it (OPT-030).
        self._text_trim = bool(text_trim)
        # Roadmap item 1: measure the CUTLASS tile per ActionDiT GEMM shape
        # (M = num_action) at construction instead of using the (N, K)
        # heuristic. Opt-in until Thor confirms it (opportunities.md OPT-018).
        if gemm_variant_autotune and precision not in _VARIANT_TUNED_PRECISIONS:
            raise ValueError(
                f"gemm_variant_autotune=True applies to {_VARIANT_TUNED_PRECISIONS}, "
                f"got precision={precision!r}")
        self._gemm_tuner: GemmVariantTuner | None = None
        self.gemm_variant_results: tuple[VariantTuneResult, ...] = ()
        # Roadmap item 6 (opportunities.md OPT-019): resolved attention
        # kernel choice, fixed for this frontend's lifetime.
        self.use_fa4: bool = self._resolve_use_fa4(use_fa4)
        self.use_fa4_mot: bool = bool(use_fa4_mot)
        # Set when FA4 failed during warmup or capture and the frontend
        # fell back to the cuBLAS chain (see `_capture_graph_or_fall_back`).
        self.fa4_fallback_reason: str | None = None
        # Real VAE + text-context wiring plan: independent of ckpt_path
        # (OPT-001) -- one loads real transformer weights, this loads a
        # real image encoder. Loaded here (once), used inside infer().
        if (ae_model_path is None) != (flux2_src is None):
            raise ValueError("ae_model_path and flux2_src must be given together, or not at all")
        self._ae = None
        self._vae_encoder = None
        self._vae_pre = None
        self._vae_stage = None
        # Recorded for the runtime export's identity (image -> token numerics).
        self._vae_setup = (("vae_resize", vae_resize), ("vae_encoder", vae_encoder),
                           ("vae_graph_input", str(vae_graph_input)))
        if vae_resize not in RESIZE_MODES:
            raise ValueError(f"vae_resize={vae_resize!r} -- must be one of {RESIZE_MODES}")
        if vae_encoder not in _VAE_ENCODERS:
            raise ValueError(f"vae_encoder={vae_encoder!r} -- must be one of {_VAE_ENCODERS}")
        if vae_graph_input is not None and ae_model_path is None:
            raise ValueError("vae_graph_input (VAE inside the CUDA graph) needs ae_model_path/flux2_src")
        if vae_encoder != "torch" and ae_model_path is None:
            raise ValueError(f"vae_encoder={vae_encoder!r} selects a real VAE encoder and needs "
                             f"ae_model_path/flux2_src")
        if vae_resize != "area" and ae_model_path is None:
            raise ValueError(f"vae_resize={vae_resize!r} configures the real VAE preprocessing and needs "
                             f"ae_model_path/flux2_src")
        if ae_model_path is not None:
            from flash_rt.models.imagewam.vae_encoder import load_real_ae
            self._ae = load_real_ae(ae_model_path, flux2_src)
            self._vae_encoder = self._ae
            if vae_encoder == "native":
                from flash_rt.models.imagewam.vae_native_encoder import NativeFlux2Encoder
                self._vae_encoder = NativeFlux2Encoder(self._ae)
            # Roadmap item 2 (plan.md): fused uint8 -> BF16 preprocessing
            # kernel. `vae_resize="area"` is bit-identical to the former
            # `_prep_view` path (the served default); `"pil_bilinear"`
            # reproduces the official LIBERO eval's PIL center-crop resize
            # bit-exactly (issues.md ISSUE-030).
            self._vae_pre = VaePreprocessor(resize=vae_resize)
        # Live Qwen3 text encoding (real VAE + text-context wiring
        # plan's own deferred item, closed once real Qwen3-4B weights
        # were downloaded -- see opportunities.md). Independent of
        # ae_model_path/ckpt_path; loaded here once, used in set_prompt().
        self._qwen3 = None
        if qwen3_model_spec is not None:
            from flash_rt.models.imagewam.text_encoder import load_real_text_encoder
            self._qwen3 = load_real_text_encoder(qwen3_model_spec)
        # OPT-004 step 5 (plan.md): every weight-projection GEMM in
        # pipeline_thor.py dispatches through weights[key](...), a
        # callable built here by _rnd_linear -- "nvfp4" requires a
        # Blackwell/Thor build (flash_rt.flash_rt_fp4) and will raise a
        # clear RuntimeError from Nvfp4Linear's own constructor on any
        # other machine (see quant_linear.py's own module docstring),
        # not here.
        self._precision = precision
        self._keepalive = []
        self.dims = dict(_DEFAULT_DIMS)
        if dims_override:
            self.dims.update(dims_override)
        # op-fusion audit finding 1 (opportunities.md): merge single-
        # stream blocks' real fused `linear1` (qkv+mlp-gate/up) into ONE
        # GEMM instead of the historical qkv.weight/mlp_in.weight split
        # -- every precision except `fp16_cutlass`, which already has
        # its OWN separate fused mlp-gate/up mechanism
        # (`CutlassFp16SwiGluMlp`) that needs `mlp_in.weight` as its own
        # standalone tensor, not a slice of a wider linear1 buffer (that
        # would need a stride-aware weight-loading path this class
        # doesn't have -- not attempted, `fp16_cutlass` keeps the old
        # split unchanged). `linear2` is governed by `merge_linear2` below.
        self.dims["merge_qkv_mlp"] = Precision(precision).merge_qkv_mlp
        # Roadmap item 4 (plan.md "single-stream linear2 merge"): run the
        # single-stream blocks' real `linear2` (attn_out_proj+mlp_down)
        # as ONE GEMM over `[attn_out | mlp_act]`, like the official
        # block. Same precision rule as `merge_qkv_mlp` (the merged path
        # writes the SiLU-GLU output straight into the `linear2` input
        # buffer, which only the merged-`linear1` path does);
        # `dims_override={"merge_linear2": False}` selects the split
        # path for A/B measurement.
        self.dims.setdefault("merge_linear2", self.dims["merge_qkv_mlp"])
        if self.dims["merge_linear2"] and not self.dims["merge_qkv_mlp"]:
            raise ValueError(
                f"merge_linear2=True needs merge_qkv_mlp=True (precision={precision!r} keeps "
                f"the split linear1 path)")
        # Roadmap item 3 (plan.md "gated-residual + next-AdaLN fusion"):
        # every gated residual update also emits the AdaLN that follows
        # it, in one kernel that reads the FP32 modulation directly (no
        # per-layer modulation cast/broadcast kernels). Bit-identical to
        # the unfused path, so on for every precision;
        # `dims_override={"fuse_res_norm": False}` selects the unfused
        # path for A/B measurement.
        self.dims.setdefault("fuse_res_norm", True)
        d = self.dims
        if d["action_attn_width"] != d["hidden"]:
            raise ValueError(
                f"action_attn_width ({d['action_attn_width']}) must equal hidden "
                f"({d['hidden']}) -- required for mot_joint attention (both experts' "
                f"Q/K/V must land in the same per-head geometry)")
        if d["HD"] != 128:
            raise ValueError(
                f"HD={d['HD']} -- real 4-axis RoPE (axes_dim=(32,32,32,32)) sums to a "
                f"fixed 128; HD is not a free structural-test parameter")
        if ckpt_path is not None and ("ref_h" not in d or "ref_w" not in d):
            # `ref_h`/`ref_w`: the REAL image RoPE needs the actual 2D
            # patch grid (14x28 for the real confirmed 224x448 input,
            # NOT a flat (img_len, 1) placeholder) -- bug found
            # 2026-09-15 via a real Thor cosine comparison against the
            # official model (opportunities.md OPT-002's correction):
            # this silent fallback previously cost 0.09 of cosine
            # similarity (img=0.91 vs 0.999966 with the real grid)
            # without ever crashing. Checked here, BEFORE loading the
            # (multi-GB) checkpoint below, so a real-checkpoint caller
            # that forgot this fails fast, not after a slow load.
            # Structural/random-weight dry runs (ckpt_path=None) keep
            # the silent flat-grid default below -- no accuracy claim
            # there to silently break.
            raise ValueError(
                "ckpt_path given without ref_h/ref_w in dims_override -- the real "
                "image RoPE grid (14x28 for the real confirmed 224x448 input) must "
                "be passed explicitly for real-checkpoint accuracy")

        # Real activation calibration (calibration_file.py): loaded and
        # identity-checked here, before the multi-GB checkpoint load. Used
        # by _calibrate_fp8() (static FP8 scales) and, with nvfp4_awq, by
        # _load_real_weights() (AWQ channel statistics, awq.py).
        self._nvfp4_awq = nvfp4_awq
        self._awq_alpha = float(awq_alpha)
        self._awq_scope = awq_scope
        if nvfp4_awq:
            if precision not in _NVFP4_PRECISIONS:
                raise ValueError(f"nvfp4_awq applies to {_NVFP4_PRECISIONS}, not precision={precision!r}")
            if calibration_path is None or ckpt_path is None:
                raise ValueError("nvfp4_awq needs calibration_path (per-channel activation "
                                 "statistics) and ckpt_path")
        self._calibration = None
        # First 16 hex chars of the calibration file's SHA-256, for the
        # runtime identity ("none" without a file).
        self._calibration_digest = "none"
        if calibration_path is not None:
            if precision not in _STATIC_FP8_PRECISIONS and not nvfp4_awq:
                raise ValueError(f"calibration_path is used by {_STATIC_FP8_PRECISIONS} and by "
                                 f"nvfp4_awq, not precision={precision!r}")
            if ckpt_path is None:
                raise ValueError("calibration_path requires ckpt_path (a calibration file is "
                                 "tied to one real checkpoint)")
            from flash_rt.models.imagewam.calibration_file import load_calibration
            self._calibration = load_calibration(calibration_path)
            self._calibration.validate_for(checkpoint_path=ckpt_path, dims=d, text_trim=self._text_trim)
            with open(calibration_path, "rb") as f:
                self._calibration_digest = hashlib.sha256(f.read()).hexdigest()[:16]

        self._ctx = fvk.FvkContext()
        # `gemm_runner`: an already-autotuned `fvk.GemmRunner` from another
        # frontend. Both then run the same cuBLASLt algorithm per shape,
        # which an A/B of two frontends needs for a bit-exact comparison
        # (`benchmarks/imagewam_fusion_ab.py`); the autotune shape set does
        # not depend on dims flags, so one autotune covers both. A frontend
        # autotunes only a runner it owns, including the shapes a new
        # trimmed text length adds (`_activate_text_length`).
        self._owns_gemm = gemm_runner is None
        self._tuned_gemm_shapes: set[tuple[str, int, int, int]] = set()
        if gemm_runner is None:
            self._gemm = fvk.GemmRunner()
            self._autotune_gemm(d)
        else:
            self._gemm = gemm_runner

        # OPT-001 (plan.md): ckpt_path switches every weight/modulation
        # source from random to the real checkpoint's own tensors --
        # precision selection (self._precision) is UNCHANGED and
        # ORTHOGONAL either way, same as OPT-004 step 5's own design.
        real_mod = None
        if ckpt_path is not None:
            from flash_rt.models.imagewam.checkpoint_loader import (
                build_real_modulation_weights, load_real_imagewam_state_dict,
            )
            sd = load_real_imagewam_state_dict(ckpt_path)
            self._weights = self._load_real_weights(d, sd)
            real_mod = build_real_modulation_weights(sd)
            del sd
        else:
            self._weights = self._alloc_random_weights(d)
        if gemm_variant_autotune:
            self._tune_action_dit_gemm_variants(d)

        # Real closed-loop robot-state conditioning (opportunities.md,
        # found 2026-09-15 scoping real closed-loop testing): a plain
        # biased Linear(proprio_dim -> joint_attention_dim), applied
        # OUTSIDE the captured graph in infer() (same convention as the
        # real VAE/Qwen3 encoders -- see that method's own docstring),
        # NOT through Fp16Linear/quant_linear.py. `dims["proprio_dim"]`
        # opts this in; omitted (None) by default so every existing
        # caller/test (no proprio conditioning) is unaffected.
        self._proprio_dim = d.get("proprio_dim")
        self._proprio_w = None
        self._proprio_b = None
        if self._proprio_dim is not None:
            if ckpt_path is not None:
                from flash_rt.models.imagewam.checkpoint_loader import load_real_proprio_weights
                pe = load_real_proprio_weights(ckpt_path)
                if pe is None:
                    raise ValueError(
                        f"dims['proprio_dim']={self._proprio_dim} given but {ckpt_path} has no "
                        f"top-level 'proprio_encoder' key -- this checkpoint was not trained "
                        f"with proprio conditioning")
                w, b = pe
                if tuple(w.shape) != (d["joint_attention_dim"], self._proprio_dim):
                    raise ValueError(
                        f"real proprio_encoder.weight shape {tuple(w.shape)} != "
                        f"(joint_attention_dim={d['joint_attention_dim']}, proprio_dim={self._proprio_dim})")
                self._proprio_w = self._own(w.to(DEV, dtype=BF16))
                self._proprio_b = self._own(b.to(DEV, dtype=BF16))
            else:
                self._proprio_w = self._own(
                    torch.randn(d["joint_attention_dim"], self._proprio_dim, dtype=BF16, device=DEV) * 0.02)
                self._proprio_b = self._own(torch.zeros(d["joint_attention_dim"], dtype=BF16, device=DEV))

        # Real min/max normalization (dataset_stats.json), same
        # closed-loop scope: `state` normalizes real proprio INTO the
        # model's [-1,1] space (infer()'s own input side); `action`
        # denormalizes the model's flow-matching output back OUT to
        # real units (infer()'s own return value). Omitted by default
        # -- every existing caller keeps getting the model's own raw
        # (still-normalized-space) action_latent, unchanged.
        self._state_norm = None
        self._action_norm = None
        if dataset_stats_path is not None:
            from flash_rt.models.imagewam.dataset_stats import load_real_normalizers
            self._state_norm, self._action_norm = load_real_normalizers(dataset_stats_path, device=DEV)

        self._bufs = self._alloc_buffers(d)
        if vae_graph_input is not None:
            # Roadmap item 5 (plan.md): the VAE stage reads a fixed uint8
            # view buffer and writes straight into img_raw, so
            # _capture_graph() records it ahead of prefill and infer()
            # does one replay.
            nv, in_h, in_w = (int(v) for v in vae_graph_input)
            self._vae_stage = ImageWAMVaeStage(
                self._vae_encoder, self._vae_pre, VaeStageSpec(num_views=nv, in_h=in_h, in_w=in_w),
                self._img_raw)
        # `ref_h`/`ref_w`: the REAL image RoPE needs the actual 2D patch
        # grid (14x28 for the real confirmed 224x448 input, NOT a flat
        # (img_len, 1) "392x1" placeholder) -- see the ckpt_path check
        # above for the full account (opportunities.md OPT-002's
        # correction). Defaults to the OLD flat placeholder so the toy/
        # default dims (no real 2D image structure) are unaffected.
        img_len = d["a0"] - d["x0"]
        ref_h = d.get("ref_h", img_len)
        ref_w = d.get("ref_w", 1)
        if ref_h * ref_w != img_len:
            raise ValueError(
                f"ref_h*ref_w ({ref_h}*{ref_w}={ref_h * ref_w}) must equal img_len "
                f"(a0-x0={img_len}) -- every image patch needs exactly one RoPE position")
        self._ref_hw = (ref_h, ref_w)
        self._max_rope_table = self._own(build_backbone_rope_table(
            d["x0"], ref_h, ref_w, device=DEV))
        self._action_rope_table = self._own(build_action_rope_table(d["num_action"], device=DEV))
        self._mod_txt, self._mod_img, self._mod_single = self._compute_backbone_modulation(
            d, real_mod=real_mod["backbone"] if real_mod else None)
        self._action_mods, self._head_mods, self._deltas = self._compute_action_modulations(
            d, real_mod=real_mod["action"] if real_mod else None)

        num_layers = d["num_layers_double"] + d["num_layers_single"]
        HD, hidden = d["HD"], d["hidden"]
        # num_layers/num_heads/head_dim must match THIS frontend's own
        # dims, not make_imagewam_attention_spec's real-FLUX.2-4B
        # defaults (24/128/25) -- see that function's own docstring
        # for the real bug this fixes (an unparameterized spec here
        # silently caused an out-of-bounds attention read/write at any
        # non-real dims, found via tests/test_imagewam_thor_real_wiring.py).
        spec = make_imagewam_attention_spec(
            max_prefix_seq=d["a0"], max_total_seq=d["total"],
            num_layers=num_layers, num_heads=d["NH"], head_dim=HD)
        # Real per-head K/V: (num_layers, total, hidden) -- NOT the old
        # broadcast-K/V (num_layers, total, HD) shape (opportunities.md
        # OPT-002).
        self._K_cache = self._own(torch.zeros(num_layers, d["total"], hidden, dtype=FP16, device=DEV))
        self._V_cache = self._own(torch.zeros(num_layers, d["total"], hidden, dtype=FP16, device=DEV))
        self._Q_O = self._own(torch.zeros(d["total"], hidden, dtype=FP16, device=DEV))
        self._logits = self._own(
            torch.zeros(d["total"] * d["NH"], d["total"] + (d["total"] % 2), dtype=FP16, device=DEV))
        layer_stride = self._K_cache[0].numel() * 2
        # FA4 output staging (OPT-019): FA4 cannot write over its own Q
        # input. `(total, hidden)` holds either site's `q_seq * NH * HD`;
        # `logits` (sized for the cuBLAS chain's scores) is too small for
        # that at small dims. Allocated only when some site runs FA4.
        self._fa4_out = None
        if self.use_fa4 or self.use_fa4_mot:
            self._fa4_out = self._own(torch.zeros(d["total"], hidden, dtype=FP16, device=DEV))
        common = {
            "Q_O": self._Q_O.data_ptr(), "K": self._K_cache.data_ptr(),
            "V": self._V_cache.data_ptr(), "logits": self._logits.data_ptr(),
            "scale": 1.0 / (HD ** 0.5),
        }
        if self._fa4_out is not None:
            common.update(fa4_out=self._fa4_out.data_ptr(), fa4_out_numel=self._fa4_out.numel())
        self._attn_spec = spec
        self._attn_slots = {"backbone": dict(common), "mot": dict(common, layer_stride=layer_stride)}
        self._attn = self._build_attn_backend()

        # The active capture: the graph `infer()` replays, the dims it
        # runs, and its backbone RoPE table. Set only by
        # `_activate_text_length` (and `_capture_graph`, which fills
        # `_graph` for the dims/table already made active). With
        # `text_trim=False` the dims are always `self.dims` itself.
        self._graph = None
        self._active_dims = self.dims
        self._rope_table = self._max_rope_table
        # One `TextLengthCapture` per captured context length `x0`.
        self._captures: dict[int, TextLengthCapture] = {}
        # `text_trim`: set once the max-dims eager prefill that sizes every
        # lazily grown GEMM scratch has run (`_capture_graph`).
        self._scratch_reserved = False
        # Set once `_calibrate_fp8` has run (before the first capture only).
        self._fp8_calibrated = False
        # Capture stream and CUDA-graph memory pool shared by every capture
        # of this frontend (created at the first capture, `_capture_graph`).
        # The runtime export also runs ABI replay and staging verbs on the
        # stream (see runtime_export.py).
        self._graph_stream: torch.cuda.Stream | None = None
        self._graph_pool: tuple[int, int] | None = None
        # Cache key of the live-Qwen3 and random `set_prompt` paths; a
        # precomputed `context` is always applied (issues.md ISSUE-060).
        self._current_prompt = None
        # Row index inside self._context where the proprio token lives
        # for the CURRENTLY captured prompt -- computed once in
        # set_prompt() (depends on that prompt's own real token count),
        # reused by every infer() call until the next set_prompt().
        self._proprio_row = None
        # The workload this frontend serves, when it was built from a
        # resolved configuration (`from_config` / `load_imagewam`); `None`
        # for a caller that passed dims by hand, which has no workload to
        # name. The runtime identity reports it as `workload.<field>`
        # (`runtime_surface.workload_identity`); nothing else reads it.
        self._workload: ImageWAMWorkload | None = None
        # The `ResolvedConfig` this frontend was built from, on the same
        # construction path (`resolved_config`). `None` for a caller that
        # passed dims by hand.
        self._resolved: ResolvedConfig | None = None

    @property
    def resolved_config(self) -> ResolvedConfig | None:
        """The `ResolvedConfig` this frontend was built from, or `None` when
        it was built by the constructor with hand-passed dims.

        It is the resolved configuration, not a re-read of the frontend's
        attributes: `config_resolver.format_effective_config` on
        `resolved_config.options`, with the runtime-resolved `use_fa4`,
        `use_fa4_mot` and `fa4_fallback_reason`, is the same line the
        resolver produced for this run.
        """
        return self._resolved

    def _input_view_shape(self) -> tuple[int, int, int]:
        """`(num_views, H, W)` of the uint8 camera frames this frontend's
        staged image paths take, in order of authority:

        1. the resolved workload's own `vae_graph_input()` -- the frontend
           was built by `from_config`/`load_imagewam` and serves that
           workload's cameras;
        2. the in-graph VAE stage's `(spec.num_views, spec.in_h, spec.in_w)`
           -- built from that same `vae_graph_input()`, so it agrees with
           1 whenever both exist;
        3. `_NO_WORKLOAD_VIEW_SHAPE` -- a caller that passed `dims_override`
           by hand names no workload, and took two 224x224 views before the
           workload object existed.

        `view_shape` (`runtime_surface()`) and `num_views` both read it, so
        the declared frame shape and the observation's own keys cannot
        disagree.
        """
        if self._workload is not None:
            return self._workload.vae_graph_input()
        if self._vae_stage is not None:
            return (self._vae_stage.spec.num_views, self._vae_stage.spec.in_h, self._vae_stage.spec.in_w)
        return _NO_WORKLOAD_VIEW_SHAPE

    @property
    def num_views(self) -> int:
        """The camera views this frontend serves: `observation_views` reads
        `view1` ... `view<num_views>` from an observation, `stage_images`
        takes exactly that many views, and `runtime_surface().view_shape[0]`
        declares them. Resolved by `_input_view_shape` (the workload's own
        count when this frontend was built for one). Read-only.
        """
        return self._input_view_shape()[0]

    @classmethod
    def from_config(cls, resolved: ResolvedConfig, *, workload: ImageWAMWorkload | None = None,
                    **kwargs) -> ImageWAMTorchFrontendThor:
        """Build the frontend for a resolved configuration.

        `dims_override` and every optimisation switch come from
        `frontend_kwargs_from_config(resolved)` (that function is the
        mapping, and holds the argument list); `**kwargs` are the path
        arguments the resolver does not see (`ckpt_path`,
        `ae_model_path`, `flux2_src`, `qwen3_model_spec`,
        `dataset_stats_path`, `vae_resize`). The constructor itself is
        unchanged: an existing caller that passes `dims_override=` and the
        old arguments builds exactly the same object.

        `workload`: the workload `resolved` was resolved for. It is recorded
        on the frontend (`runtime_surface()` reports it as the
        `workload.<field>` identity entries) and is not re-derived from
        `dims`. Both it and `resolved` are readable back as
        `self._workload` and `resolved_config`.
        """
        fe = cls(**frontend_kwargs_from_config(resolved, **kwargs))
        fe._workload = workload
        fe._resolved = resolved
        return fe

    @staticmethod
    def _resolve_use_fa4(use_fa4: bool | None) -> bool:
        """`use_fa4` constructor argument -> the backbone-site FA4 choice.

        - `True`: FA4; `ImageWAMAttnBackend` raises if the runtime is missing.
        - `False`: the cuBLAS chain.
        - `None` (default): opt-in. False unless `FLASHRT_THOR_FA4=1`; with
          it, FA4 exactly when `fa4_backend.thor_default_enabled()` holds
          (compute capability 11.x and an importable FA4 runtime), so it
          never raises for a missing runtime. FA4 has been measured on Thor
          at `a0=896` in the per-layer bench (OPT-005), not yet at the
          served shapes or end to end (OPT-019).
        """
        if use_fa4 is not None:
            return bool(use_fa4)
        if os.environ.get(_FA4_OPT_IN_ENV, _FA4_OPT_IN_DEFAULT) != "1":
            return False
        return fa4_backend.thor_default_enabled()

    def _build_attn_backend(self) -> ImageWAMAttnBackend:
        """The attention backend for this frontend's own buffers, with
        the current `self.use_fa4` / `self.use_fa4_mot` choice."""
        return ImageWAMAttnBackend(
            self._attn_spec, self._ctx,
            backbone_slots=dict(self._attn_slots["backbone"]),
            mot_slots=dict(self._attn_slots["mot"]),
            # OPT-002: real per-head K/V + the real "mot" rule (no region
            # mask), both confirmed against the real trained checkpoint
            # (benchmarks/imagewam_real_checkpoint_validation.py).
            use_perhead_kv=True, use_real_mot_mask=True,
            # OPT-005 / OPT-019: FA4 for the "backbone" site, resolved by
            # `_resolve_use_fa4` (opt-in; see `_FA4_OPT_IN_ENV`).
            use_fa4=self.use_fa4,
            # OPT-019: FA4 for the "mot" site. Opt-in until Thor confirms it.
            use_fa4_mot=self.use_fa4_mot,
        )

    def _own(self, t: torch.Tensor) -> torch.Tensor:
        """Keep a buffer tensor alive for the frontend's own lifetime.

        A bare `torch.zeros(...).data_ptr()` expression drops the only
        Python reference the instant that expression finishes --
        PyTorch's caching allocator is then free to hand the same
        memory to the next allocation, silently corrupting an
        already-stored pointer. Every buffer here goes through this
        helper for that reason.
        """
        self._keepalive.append(t)
        return t

    def _autotune_gemm(self, d: dict, *, fp16_nn_shapes: bool = True) -> None:
        """Autotune `GemmRunner.fp16_nn` once per distinct (M,N,K) shape
        this frontend's own real math uses (OPT-004 step 4,
        opportunities.md), at construction time -- before any weight/
        buffer allocation even, so it never touches this frontend's
        own real pointers (uses disposable zero-filled scratch of the
        right shape/dtype instead; autotune only times candidate
        cuBLASLt algorithms, it does not need meaningful values).

        `autotune_fp16_nn(x_ptr, w_ptr, out_ptr, m, n, k, num_algos)`
        real-benchmarks up to `num_algos` candidates and writes the
        winner into the SAME per-(op,M,N,K) cache `fp16_nn` reads from
        afterward (confirmed by reading `csrc/gemm/gemm_runner.cu`
        directly -- see `benchmarks/imagewam_thor_fp16_autotuned_bench.py`'s
        own docstring for the full account) -- this can only match or
        beat the default heuristic's own top-1 pick, never regress
        (identical math, only the algorithm choice differs). Every
        layer of the same type shares an identical shape (confirmed
        elsewhere in this codebase, e.g. `imagewam_thor_bench.py`'s own
        docstring: "every layer of the same type has identical shapes
        and therefore identical steady-state cost"), so exactly one
        autotune call per distinct shape below covers every layer of
        that type -- not one call per layer.

        Shapes already tuned by this frontend (`self._tuned_gemm_shapes`)
        are skipped, so a trimmed text length (`text_trim`) tunes only the
        shapes it adds: the text GEMMs at `M = x0` and the single-stream
        GEMMs at `M = a0`. `fp16_nn_shapes=False` tunes the `bf16_nn`
        shapes only (`txt_in`/`img_in`, which run on cuBLASLt at every
        precision).
        """
        hidden, mlp_hidden, HD = d["hidden"], d["mlp_hidden"], d["HD"]
        joint_attention_dim = d["joint_attention_dim"]
        x0, a0 = d["x0"], d["a0"]
        img_len = a0 - x0
        ahd, aaw, amh, num_action = (
            d["action_hidden_dim"], d["action_attn_width"],
            d["action_mlp_hidden"], d["num_action"])
        action_dim = d["action_dim"]

        # txt_in/img_in are BF16-in/out (`Bf16OutLinear`, OPT-001 "FP16
        # residual overflow") -- autotuned separately via autotune_bf16_nn
        # below, a distinct GemmRunner cache key from fp16_nn's.
        bf16_shapes = {
            (x0, hidden, joint_attention_dim),      # txt_in
            (img_len, hidden, HD),                    # img_in (OPT-001/OPT-008)
        }
        tuned_any = False
        for m, n, k in bf16_shapes:
            if ("bf16", m, n, k) in self._tuned_gemm_shapes:
                continue
            x = torch.zeros(m, k, dtype=BF16, device=DEV)
            w = torch.zeros(k, n, dtype=BF16, device=DEV)
            out = torch.zeros(m, n, dtype=BF16, device=DEV)
            self._gemm.autotune_bf16_nn(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, 16)
            self._tuned_gemm_shapes.add(("bf16", m, n, k))
            tuned_any = True

        shapes = {
            (x0, 3 * hidden, hidden),                 # txt_qkv (OPT-004 step 2, fused)
            (x0, hidden, hidden),                      # txt_proj
            (x0, mlp_hidden * 2, hidden),               # txt_mlp0
            (x0, hidden, mlp_hidden),                   # txt_mlp2
            (img_len, 3 * hidden, hidden),              # img_qkv (fused)
            (img_len, hidden, hidden),                  # img_proj
            (img_len, mlp_hidden * 2, hidden),          # img_mlp0
            (img_len, hidden, mlp_hidden),               # img_mlp2
            (a0, 3 * hidden, hidden),                     # single qkv (fused)
            (a0, hidden, hidden),                          # single attn_out_proj
            (a0, mlp_hidden * 2, hidden),                   # single mlp_in
            (a0, hidden, mlp_hidden),                        # single mlp_down
            (a0, hidden, hidden + mlp_hidden),                # single linear2 (merged, roadmap item 4)
            (num_action, 3 * aaw, ahd),                       # action qkv (fused)
            (num_action, ahd, aaw),                            # action proj/attn_out_proj
            (num_action, amh * 2, ahd),                         # action mlp0/mlp_in
            (num_action, ahd, amh),                              # action mlp2/mlp_down
            (num_action, ahd, aaw + amh),                         # action linear2 (merged, roadmap item 4)
            (num_action, ahd, action_dim),                        # action_encoder (OPT-001)
            (num_action, action_dim, ahd),                         # head.linear (OPT-001)
        }
        for m, n, k in shapes:
            if not fp16_nn_shapes or ("fp16", m, n, k) in self._tuned_gemm_shapes:
                continue
            x = torch.zeros(m, k, dtype=FP16, device=DEV)
            w = torch.zeros(k, n, dtype=FP16, device=DEV)
            out = torch.zeros(m, n, dtype=FP16, device=DEV)
            self._gemm.autotune_fp16_nn(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, 16)
            self._tuned_gemm_shapes.add(("fp16", m, n, k))
            tuned_any = True
        if tuned_any:
            torch.cuda.synchronize()

    def _rnd_linear(self, n: int, k: int):
        """Real GEMM (K,N) convention: `n` = output width, `k` = input
        width, stored as (k, n) (a real checkpoint's own (out,in)
        `nn.Linear` weight would need `.t().contiguous()` at load time
        -- see `CosmosEdgeThor`'s own precedent). Returns a linear-op
        OBJECT (`Fp16Linear`/`Fp8Linear`/`Nvfp4Linear`, selected by
        `self._precision`), NOT a raw pointer -- OPT-004 step 5
        (`plan.md`): `pipeline_thor.py`'s own weight-projection call
        sites are `weights[key](x_ptr, out_ptr, m, stream)` uniformly.
        The real FP16 weight is always materialized first (quantized
        classes read it once, at construction, to build their own
        quantized copy) and kept alive via `self._own` regardless of
        which precision ultimately uses it.
        """
        w = self._own(torch.randn(k, n, dtype=FP16, device=DEV) * 0.02)
        return self._wrap_linear(w, n, k)

    def _rnd_bf16out_linear(self, n: int, k: int) -> Bf16OutLinear:
        """`_rnd_linear`'s counterpart for `txt_in.weight`/`img_in.weight`
        specifically -- see `Bf16OutLinear`'s own docstring (opportunities.md
        OPT-001 "FP16 residual overflow"). Applied regardless of
        `self._precision`, unlike `_rnd_linear` -> `_wrap_linear`."""
        w = self._own(torch.randn(k, n, dtype=BF16, device=DEV) * 0.02)
        return Bf16OutLinear(self._gemm, w.data_ptr(), n, k)

    def _rnd_swiglu_mlp(self, n: int, k: int):
        """`_rnd_linear`'s counterpart for the merged MLP gate/up
        projection (`{txt,img}_mlp0.weight`/`mlp0.weight`/`mlp_in.weight`,
        `n=2*mlp_hidden`) -- opportunities.md OPT-013. Only
        `precision=="fp16_cutlass"` gets the fused `CutlassFp16SwiGluMlp`;
        every other precision falls back to the plain `_rnd_linear`
        (one wide GEMM, `pipeline_thor.py`'s own `_mlp_gate_up` helper
        then does the separate `silu_glu_merged_fp16` step as before).
        `Nvfp4SwiGluMlp` (op-fusion audit finding 2) was wired here too
        and real-Thor-verified: numerically fine (no regression vs the
        plain path relative to fp16) but NO measured `infer()` speed win
        at ImageWAM's real shapes (opportunities.md OPT-015) -- reverted
        back to the plain path for the actual shipped default; the class
        and its `silu_glu_two_fp4_to_fp16` kernel stay in the codebase,
        documented, not wired to any precision string."""
        if not Precision(self._precision).fused_swiglu_mlp:
            return self._rnd_linear(n, k)
        mlp_hidden = n // 2
        w = self._own(torch.randn(k, n, dtype=FP16, device=DEV) * 0.02)
        return CutlassFp16SwiGluMlp(w.data_ptr(), mlp_hidden, k)

    def _wrap_linear(self, w: torch.Tensor, n: int, k: int, *, awq_inv_s: torch.Tensor | None = None):
        """Wrap an already-materialized `(k,n)` fp16 CUDA weight tensor
        in the `self._precision`-selected linear-op object -- the part
        of `_rnd_linear` that's shared with `_load_real_weights` (OPT-001),
        which sources `w` from the real checkpoint instead of
        `torch.randn`. `w` must already be `self._own`'d by the caller
        (except for `SimNvfp4Linear`, which keeps its own copy).
        `awq_inv_s`: the AWQ input scale `w` already carries (NVFP4
        precisions only, `awq.py`).

        Which precision falls back to the plain fp16 linear, and at which
        alignment, is `precision.py`'s table:
        `Precision(self._precision).alignment_fallback(n, k)` is True when
        the native GEMM of this precision cannot take the shape (both `n`
        and `k` must be multiples of the tier's alignment); the per-class
        notes below say why each tier has that width.
        """
        prec = Precision(self._precision)
        if awq_inv_s is not None and not prec.supports_awq:
            raise ValueError(f"awq_inv_s given for precision={self._precision!r}")
        fallback = prec.alignment_fallback(n, k)
        if prec is Precision.FP16:
            return Fp16Linear(self._gemm, w.data_ptr(), n, k)
        if prec is Precision.FP16_CUTLASS:
            # Real Thor finding, opportunities.md OPT-013: CUTLASS FP16
            # requires N/K divisible by 8 (`can_implement` fails
            # otherwise, confirmed on real hardware) -- action_dim=7
            # (real LIBERO width) makes action_encoder/head.linear
            # structurally incompatible with every tile variant. Fall
            # back to the plain cuBLASLt path for just these
            # mis-aligned shapes rather than crashing `set_prompt()`'s
            # graph capture -- cuBLASLt has no such alignment
            # requirement. This precision tier is NOT currently
            # recommended anyway (real Thor measurement: every tile
            # variant tried is slower than cuBLASLt's own default, see
            # that entry), so this fallback is about not crashing on
            # an already-not-recommended option, not about chasing
            # speed for these two tiny GEMMs.
            if fallback:
                return Fp16Linear(self._gemm, w.data_ptr(), n, k)
            return CutlassFp16Linear(w.data_ptr(), n, k)
        if prec is Precision.FP8:
            # Real Thor finding: FP8 (both cuBLASLt's own heuristic
            # search, status 15/CUBLAS_STATUS_NOT_SUPPORTED, AND CUTLASS's
            # can_implement) rejects action_encoder (K=7)/head.linear
            # (N=7) -- unlike fp16_cutlass/nvfp4 below, where cuBLASLt
            # itself tolerated the misaligned shape and only the
            # CUTLASS-specific path needed a fallback, FP8 needs it on
            # EVERY backend for this precision family. Same 8-alignment
            # threshold as fp16_cutlass.
            if fallback:
                return Fp16Linear(self._gemm, w.data_ptr(), n, k)
            return Fp8Linear(w.data_ptr(), n, k)
        if prec is Precision.NVFP4:
            # Real Thor finding (opportunities.md, Stage 3 checklist):
            # NVFP4 requires K divisible by 16 (`Nvfp4Linear`'s own
            # constructor check), and the real underlying CUTLASS
            # block-scaled kernel also rejects action_encoder (K=7) and
            # head.linear (N=7) -- same structurally-misaligned pair
            # fp16_cutlass hit. Fall back to plain fp16 for these two
            # tiny, FLOPs-negligible GEMMs rather than crashing
            # set_prompt()'s graph capture; every other real weight in
            # the model is a multiple of 16 already.
            if fallback:
                return Fp16Linear(self._gemm, w.data_ptr(), n, k)
            return Nvfp4Linear(w.data_ptr(), n, k, awq_inv_s=awq_inv_s)
        if prec is Precision.E0M3_HADAMARD:
            # opportunities.md OPT-024: E0M3 weights and activations with a
            # per-16 Hadamard rotation on both. Same block-scaled operand
            # layout as nvfp4, so the same K%16/N%16 requirement and the
            # same fallback for action_encoder (K=7) and head.linear (N=7).
            # The merged single-stream linear1 is one ordinary (K, N)
            # weight here; the rotation runs along K only.
            if fallback:
                return Fp16Linear(self._gemm, w.data_ptr(), n, k)
            return E0m3HadamardLinear(w.data_ptr(), n, k)
        if prec is Precision.NVFP4_SIM:
            # Same K=7/N=7 fallback as "nvfp4" above, so both precisions
            # quantize exactly the same set of GEMMs.
            if fallback:
                return Fp16Linear(self._gemm, w.data_ptr(), n, k)
            return SimNvfp4Linear(self._gemm, w.data_ptr(), n, k, awq_inv_s=awq_inv_s)
        if prec is Precision.FP8_STATIC:
            # Same K=7/N=7 FP8 alignment gap as the "fp8" branch above --
            # _calibrate_fp8() already skips non-StaticFp8Linear objects
            # (isinstance check), so this fallback needs no other
            # special-casing.
            if fallback:
                return Fp16Linear(self._gemm, w.data_ptr(), n, k)
            return StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False)
        if prec is Precision.FP8_STATIC_CUTLASS:
            if fallback:
                return Fp16Linear(self._gemm, w.data_ptr(), n, k)
            return StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=True)
        raise ValueError(f"unknown precision {self._precision!r}")  # pragma: no cover -- validated in __init__

    @staticmethod
    def _is_variant_tunable(lin: object) -> bool:
        return isinstance(lin, Nvfp4Linear) or (isinstance(lin, StaticFp8Linear) and lin.use_cutlass)

    def _tune_action_dit_gemm_variants(self, d: dict) -> None:
        """Roadmap item 1 (plan.md "Plan: ActionDiT small-M CUTLASS tile
        selection"): group every ActionDiT linear with a switchable
        CUTLASS tile by `(family, N, K)` and let one `GemmVariantTuner`
        measure and apply a tile per group at `M = num_action` (see
        `gemm_variant_tuner.py` for the rule). Runs before `set_prompt()`,
        so the captured graph uses the chosen tiles. Backbone GEMMs are
        not tuned. Results land in `self.gemm_variant_results`."""
        groups: dict[tuple[str, int, int], list] = {}
        seen: set[int] = set()
        for key, lin in self._weights.items():
            if key[0] != "action_dit" or not self._is_variant_tunable(lin) or id(lin) in seen:
                continue
            seen.add(id(lin))
            groups.setdefault((lin.family, lin.n, lin.k), []).append(lin)
        tuner = GemmVariantTuner(CudaGraphVariantTimer())
        for members in groups.values():
            tuner.tune(members, d["num_action"])
        self._gemm_tuner = tuner
        self.gemm_variant_results = tuner.results()

    def _calibrate_fp8(self, d: dict) -> None:
        """Freeze every `StaticFp8Linear` activation scale ONCE, here,
        before `_capture_graph()` -- a captured CUDA Graph replays
        identical kernel launches forever, so the scale must already be
        fixed by the time capture starts (see `StaticFp8Linear`'s own
        docstring for the ordering contract this calls into). No-op for
        every other precision.

        With `calibration_path=` (the normal case): each site's scale is
        the real calibration file's `amax / 448`
        (`calibration_file.py`, built by
        `benchmarks/imagewam_build_calibration.py` from real LIBERO
        observations through the real fp16 pipeline). Every
        `StaticFp8Linear` must have a site in the file.

        Without it: a PLACEHOLDER. Each `StaticFp8Linear` is calibrated
        on `N(0, 0.1)` noise of its site's shape (`m` = `a0` for backbone
        sites, `num_action` for ActionDiT sites), which does not match
        real activations: on Thor it collapsed `backbone_hidden` cosine
        vs `fp16` to ~0.46 (`opportunities.md` OPT-014). Logged as a
        warning.
        """
        if self._precision not in _STATIC_FP8_PRECISIONS:
            return
        if self._calibration is not None:
            from flash_rt.models.imagewam.activation_recorder import site_name
            missing = []
            for key, lin in self._weights.items():
                if not isinstance(lin, StaticFp8Linear):
                    continue
                site = self._calibration.sites.get(site_name(key))
                if site is None or site.channel_amax.shape[0] != lin.k:
                    missing.append(site_name(key))
                    continue
                lin.set_activation_scale(site.fp8_act_scale())
            if missing:
                raise ValueError(f"calibration file has no (or a wrong-width) site for "
                                 f"{len(missing)} StaticFp8Linear weights, e.g. {missing[:3]}")
            torch.cuda.synchronize()
            return
        _LOG.warning(
            "ImageWAM precision=%r: NO calibration file given -- every FP8 activation "
            "scale is a PLACEHOLDER measured on N(0, 0.1) noise, not real data "
            "(on Thor this collapsed backbone_hidden cosine vs fp16 to ~0.46, "
            "opportunities.md OPT-014). Build a real file with "
            "benchmarks/imagewam_build_calibration.py and pass calibration_path=.",
            self._precision)
        a0, num_action = d["a0"], d["num_action"]
        scratch_by_shape: dict[tuple, torch.Tensor] = {}
        for key, lin in self._weights.items():
            if not isinstance(lin, StaticFp8Linear):
                continue
            m = a0 if key[0] == "backbone" else num_action
            shape = (m, lin.k)
            x = scratch_by_shape.get(shape)
            if x is None:
                x = torch.randn(m, lin.k, dtype=FP16, device=DEV) * 0.1
                scratch_by_shape[shape] = x
            lin.calibrate(x.data_ptr(), m, 0)
        torch.cuda.synchronize()

    def _rnd_norm_scale(self, HD: int) -> int:
        """QK-Norm scale, real-checkpoint-typical positive bias (avoids
        near-zero/huge random values that would make a random-weight
        dry run's own NaN/Inf check meaningless for unrelated reasons)."""
        return self._own((torch.randn(HD, dtype=torch.float32, device=DEV).abs() + 0.5).to(FP16)).data_ptr()

    def _alloc_random_weights(self, d: dict) -> dict:
        hidden, HD, mlp_hidden = d["hidden"], d["HD"], d["mlp_hidden"]
        joint_attention_dim = d["joint_attention_dim"]
        weights = {}
        for L in range(d["num_layers_double"]):
            weights[("backbone", "double", L, "txt_in.weight")] = self._rnd_bf16out_linear(hidden, joint_attention_dim)
            weights[("backbone", "double", L, "img_in.weight")] = self._rnd_bf16out_linear(hidden, HD)
            for prefix in ("txt", "img"):
                weights[("backbone", "double", L, f"{prefix}_qkv.weight")] = self._rnd_linear(3 * hidden, hidden)
                weights[("backbone", "double", L, f"{prefix}_proj.weight")] = self._rnd_linear(hidden, hidden)
                weights[("backbone", "double", L, f"{prefix}_mlp0.weight")] = self._rnd_swiglu_mlp(mlp_hidden * 2, hidden)
                weights[("backbone", "double", L, f"{prefix}_mlp2.weight")] = self._rnd_linear(hidden, mlp_hidden)
                weights[("backbone", "double", L, f"{prefix}_query_norm")] = self._rnd_norm_scale(HD)
                weights[("backbone", "double", L, f"{prefix}_key_norm")] = self._rnd_norm_scale(HD)
        for L in range(d["num_layers_single"]):
            if self.dims.get("merge_qkv_mlp"):
                weights[("backbone", "single", L, "linear1.weight")] = self._rnd_linear(
                    3 * hidden + 2 * mlp_hidden, hidden)
            else:
                weights[("backbone", "single", L, "qkv.weight")] = self._rnd_linear(3 * hidden, hidden)
                weights[("backbone", "single", L, "mlp_in.weight")] = self._rnd_swiglu_mlp(mlp_hidden * 2, hidden)
            if self.dims.get("merge_linear2"):
                weights[("backbone", "single", L, "linear2.weight")] = self._rnd_linear(hidden, hidden + mlp_hidden)
            else:
                weights[("backbone", "single", L, "attn_out_proj.weight")] = self._rnd_linear(hidden, hidden)
                weights[("backbone", "single", L, "mlp_down.weight")] = self._rnd_linear(hidden, mlp_hidden)
            weights[("backbone", "single", L, "query_norm")] = self._rnd_norm_scale(HD)
            weights[("backbone", "single", L, "key_norm")] = self._rnd_norm_scale(HD)

        ahd, aaw, amh = d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"]
        action_dim = d["action_dim"]
        # OPT-001: real action_encoder (WITH bias, the one biased weight
        # in this project) / head, once per denoise step, not per layer
        # -- see imagewam_denoise_step's own docstring.
        weights[("action_dit", "shared", 0, "action_encoder.weight")] = self._rnd_linear(ahd, action_dim)
        weights[("action_dit", "shared", 0, "action_encoder.bias")] = self._own(
            (torch.randn(ahd, dtype=torch.float32, device=DEV) * 0.02).to(FP16)).data_ptr()
        weights[("action_dit", "shared", 0, "head.linear.weight")] = self._rnd_linear(action_dim, ahd)
        for L in range(d["action_num_layers_double"]):
            weights[("action_dit", "double", L, "qkv.weight")] = self._rnd_linear(3 * aaw, ahd)
            weights[("action_dit", "double", L, "proj.weight")] = self._rnd_linear(ahd, aaw)
            weights[("action_dit", "double", L, "mlp0.weight")] = self._rnd_swiglu_mlp(amh * 2, ahd)
            weights[("action_dit", "double", L, "mlp2.weight")] = self._rnd_linear(ahd, amh)
            weights[("action_dit", "double", L, "query_norm")] = self._rnd_norm_scale(HD)
            weights[("action_dit", "double", L, "key_norm")] = self._rnd_norm_scale(HD)
        for L in range(d["action_num_layers_single"]):
            if self.dims.get("merge_qkv_mlp"):
                weights[("action_dit", "single", L, "linear1.weight")] = self._rnd_linear(
                    3 * aaw + 2 * amh, ahd)
            else:
                weights[("action_dit", "single", L, "qkv.weight")] = self._rnd_linear(3 * aaw, ahd)
                weights[("action_dit", "single", L, "mlp_in.weight")] = self._rnd_swiglu_mlp(amh * 2, ahd)
            if self.dims.get("merge_linear2"):
                weights[("action_dit", "single", L, "linear2.weight")] = self._rnd_linear(ahd, aaw + amh)
            else:
                weights[("action_dit", "single", L, "attn_out_proj.weight")] = self._rnd_linear(ahd, aaw)
                weights[("action_dit", "single", L, "mlp_down.weight")] = self._rnd_linear(ahd, amh)
            weights[("action_dit", "single", L, "query_norm")] = self._rnd_norm_scale(HD)
            weights[("action_dit", "single", L, "key_norm")] = self._rnd_norm_scale(HD)
        return weights

    def _load_real_weights(self, d: dict, sd: dict) -> dict:
        """OPT-001 (plan.md): real checkpoint counterpart to
        `_alloc_random_weights` -- same structural shape (same 4-tuple
        keys), sourced from `checkpoint_loader.build_real_weights`
        instead of `torch.randn`. `sd` is the checkpoint's own flat
        state_dict (`checkpoint_loader.load_real_imagewam_state_dict`'s
        own return value), already loaded once by the caller.

        Norm scales and `action_encoder.bias` stay raw CUDA pointers
        (same convention as `_rnd_norm_scale`); every other value goes
        through `_wrap_linear` so precision selection stays uniform and
        orthogonal to weight source, exactly like `_rnd_linear`.
        """
        from flash_rt.models.imagewam.checkpoint_loader import build_real_weights

        raw = build_real_weights(
            sd, num_double=d["num_layers_double"], num_single=d["num_layers_single"],
            action_num_double=d["action_num_layers_double"], action_num_single=d["action_num_layers_single"],
            action_attn_width=d["action_attn_width"], merge_qkv_mlp=d.get("merge_qkv_mlp", False),
            merge_linear2=d.get("merge_linear2", False))

        awq_plans = self._plan_awq(d, raw) if self._nvfp4_awq else {}

        weights = {}
        seen_shared: dict[int, object] = {}  # id(cpu tensor) -> wrapped/ptr, for shared txt_in/img_in
        for key, t in raw.items():
            slot = key[-1]
            cpu_id = id(t)
            if cpu_id in seen_shared:
                weights[key] = seen_shared[cpu_id]
                continue
            if slot.endswith("_norm") or slot == "action_encoder.bias":
                tg = self._own(t.to(DEV))
                value = tg.data_ptr()
            elif key[0] == "backbone" and slot in ("txt_in.weight", "img_in.weight"):
                # OPT-001 "FP16 residual overflow" fix -- see Bf16OutLinear's
                # own docstring. `t` is already the real checkpoint weight,
                # FP16 (checkpoint_loader.py's own uniform convention) --
                # upcasting FP16->BF16 here loses nothing meaningful (this
                # weight's own absmax is ~0.26, comfortably exact in FP16
                # already; the only reason for BF16 is the OUTPUT range,
                # not this weight's own precision).
                n, k = t.shape[1], t.shape[0]
                tg = self._own(t.to(DEV, dtype=BF16).contiguous())
                value = Bf16OutLinear(self._gemm, tg.data_ptr(), n, k)
            elif Precision(self._precision).fused_swiglu_mlp and slot in (
                    "txt_mlp0.weight", "img_mlp0.weight", "mlp0.weight", "mlp_in.weight"):
                # opportunities.md OPT-013: fused SwiGLU gate/up (see
                # CutlassFp16SwiGluMlp's own docstring) -- `t` is the
                # real checkpoint's own merged (K, 2*mlp_hidden) weight,
                # split internally, same real trained values.
                n, k = t.shape[1], t.shape[0]
                tg = self._own(t.to(DEV).contiguous())
                value = CutlassFp16SwiGluMlp(tg.data_ptr(), n // 2, k)
            else:
                n, k = t.shape[1], t.shape[0]  # already (K,N) convention, see checkpoint_loader._w
                tg = t.to(DEV).contiguous()
                inv_s = None
                plan = awq_plans.get(key)
                if plan is not None:
                    from flash_rt.models.imagewam.awq import apply_awq_plan
                    tg = apply_awq_plan(tg, plan)
                    inv_s = (1.0 / plan.input_scale) if plan.fold_input else None
                value = self._wrap_linear(tg, n, k, awq_inv_s=inv_s)
                if not isinstance(value, SimNvfp4Linear):
                    self._own(tg)
            weights[key] = value
            seen_shared[cpu_id] = value
        return weights

    def _plan_awq(self, d: dict, raw: dict) -> dict:
        """AWQ plan (`awq.plan_awq`) for every NVFP4-quantized weight of
        the real checkpoint, from the calibration file's per-channel
        activation statistics: fold A for AdaLN-fed GEMMs, plus fold B
        (down projections via the preceding up columns) when
        `awq_scope="adaln+down"`."""
        from flash_rt.models.imagewam.activation_recorder import site_name
        from flash_rt.models.imagewam.awq import plan_awq
        eligible = [key for key, t in raw.items()
                    if not (key[-1].endswith("_norm") or key[-1] == "action_encoder.bias"
                            or key[-1] in ("txt_in.weight", "img_in.weight"))
                    and t.shape[0] % 16 == 0 and t.shape[1] % 16 == 0]
        channel_amax = {}
        for key in eligible:
            site = self._calibration.sites.get(site_name(key))
            if site is None or site.channel_amax.shape[0] != raw[key].shape[0]:
                raise ValueError(f"calibration file has no (or a wrong-width) site for {site_name(key)}")
            channel_amax[key] = torch.from_numpy(site.channel_amax).to(DEV)
        return plan_awq(eligible, channel_amax, d, alpha=self._awq_alpha, scope=self._awq_scope)

    def _alloc_buffers(self, d: dict) -> dict:
        hidden, mlp_hidden, x0, a0, HD = d["hidden"], d["mlp_hidden"], d["x0"], d["a0"], d["HD"]
        img_len = a0 - x0
        joint_attention_dim = d["joint_attention_dim"]
        ahd, aaw, amh, num_action = (
            d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"], d["num_action"])
        action_dim = d["action_dim"]
        z = lambda *shape: self._own(torch.zeros(*shape, dtype=FP16, device=DEV))
        # BF16, not FP16 -- OPT-001 "FP16 residual overflow" (opportunities.md):
        # real Qwen3-4B text conditioning, once projected through the real
        # trained backbone at real x0=512, legitimately drives this
        # persistent residual buffer to ~120000 in magnitude, which FP16
        # (max ~65504) cannot represent. `context`/`img_raw` need the same
        # dtype since they feed `combined` directly via `txt_in`/`img_in`
        # (`Bf16OutLinear`, see that class's own docstring).
        self._context = self._own(torch.zeros(x0, joint_attention_dim, dtype=BF16, device=DEV))
        self._backbone_hidden = self._own(torch.zeros(a0, hidden, dtype=BF16, device=DEV))
        self._img_raw = self._own(torch.zeros(img_len, HD, dtype=BF16, device=DEV))
        # OPT-001: real action_dim width (e.g. 7), not action_hidden_dim
        # -- see imagewam_denoise_step's own docstring.
        self._action_latent = self._own(torch.zeros(num_action, action_dim, dtype=F32, device=DEV))
        return {
            "context": self._context.data_ptr(),
            "backbone_hidden": self._backbone_hidden.data_ptr(),
            "img_raw": self._img_raw.data_ptr(),
            "modded_scratch": z(a0, hidden).data_ptr(),
            "txt_qkv_merged": z(x0, 3 * hidden).data_ptr(),
            "img_qkv_merged": z(img_len, 3 * hidden).data_ptr(),
            "single_qkv_merged": z(a0, 3 * hidden).data_ptr(),
            "action_qkv_merged": z(num_action, 3 * aaw).data_ptr(),
            # op-fusion audit finding 1: single-stream blocks' real
            # fused linear1 (qkv+mlp-gate/up in ONE GEMM) output, used
            # instead of single_qkv_merged/single_mlp_merged (and their
            # action_dit counterparts) when merge_qkv_mlp is set. Both
            # buffer sets are always allocated (a real but small, ~50MB
            # combined, memory overhead) so `fp16_cutlass`'s own
            # unmerged path keeps working unchanged.
            "single_linear1_merged": z(a0, 3 * hidden + 2 * mlp_hidden).data_ptr(),
            "action_linear1_merged": z(num_action, 3 * aaw + 2 * amh).data_ptr(),
            # Roadmap item 4: merged single-stream linear2 GEMM input,
            # `[attn_out | mlp_act]` side by side (used when
            # merge_linear2 is set; always allocated, like the linear1
            # buffers above).
            "single_linear2_in": z(a0, hidden + mlp_hidden).data_ptr(),
            "action_linear2_in": z(num_action, aaw + amh).data_ptr(),
            "action_latent_fp16": z(num_action, action_dim).data_ptr(),
            "velocity": z(num_action, action_dim).data_ptr(),
            "head_modded": z(num_action, ahd).data_ptr(),
            "txt_mlp_merged": z(x0, mlp_hidden * 2).data_ptr(),
            "txt_mlp_gated": z(x0, mlp_hidden).data_ptr(),
            "img_mlp_merged": z(img_len, mlp_hidden * 2).data_ptr(),
            "img_mlp_gated": z(img_len, mlp_hidden).data_ptr(),
            "single_mlp_merged": z(a0, mlp_hidden * 2).data_ptr(),
            "single_mlp_gated": z(a0, mlp_hidden).data_ptr(),
            "proj_scratch": z(a0, hidden).data_ptr(),
            "proj_scratch2": z(a0, hidden).data_ptr(),
            "action_latent": self._action_latent.data_ptr(),
            "action_hidden": z(num_action, ahd).data_ptr(),
            "action_modded": z(num_action, ahd).data_ptr(),
            "action_proj_scratch": z(num_action, ahd).data_ptr(),
            "action_proj_scratch2": z(num_action, ahd).data_ptr(),
            "action_mlp_merged": z(num_action, amh * 2).data_ptr(),
            "action_mlp_gated": z(num_action, amh).data_ptr(),
        }

    def _compute_backbone_modulation(self, d: dict, *, real_mod: dict | None = None):
        """Backbone's own AdaLN modulation, computed ONCE: real
        inference always conditions the reference/context encode on a
        FIXED timestep=0 (confirmed against the real checkpoint run,
        `benchmarks/imagewam_real_checkpoint_validation.py`'s own
        `video_timestep = torch.zeros(1)`), so this never needs
        recomputing per replay -- see pipeline_thor.py's own docstring.

        `real_mod`: OPT-001, the real `mod_w` dict from
        `checkpoint_loader.build_real_modulation_weights()["backbone"]`
        -- when given, used INSTEAD of random weights (moved to CUDA
        here, same as every other real-weight tensor in this file).
        """
        hidden = d["hidden"]
        if real_mod is not None:
            mod_w = {k: self._own(v.to(DEV)) for k, v in real_mod.items()}
        else:
            mod_w = {
                "time_in_w1": self._own(torch.randn(hidden, 256, dtype=torch.float32, device=DEV) * 0.02),
                "time_in_w2": self._own(torch.randn(hidden, hidden, dtype=torch.float32, device=DEV) * 0.02),
                "mod_double_txt": self._own(torch.randn(6 * hidden, hidden, dtype=torch.float32, device=DEV) * 0.02),
                "mod_double_img": self._own(torch.randn(6 * hidden, hidden, dtype=torch.float32, device=DEV) * 0.02),
                "mod_single": self._own(torch.randn(3 * hidden, hidden, dtype=torch.float32, device=DEV) * 0.02),
            }
        timestep = self._own(torch.zeros(1, dtype=torch.float32, device=DEV))
        mod_txt, mod_img, mod_single = compute_shared_modulation(timestep, mod_w, hidden)
        for group in (mod_txt[0], mod_txt[1], mod_img[0], mod_img[1], mod_single):
            for t in group:
                self._own(t)
        return mod_txt, mod_img, mod_single

    def _compute_action_modulations(self, d: dict, *, real_mod: dict | None = None):
        """ActionDiT's own AdaLN modulation, ONE tuple PER DENOISE STEP:
        its conditioning timestep changes every step, but `step` is
        itself a compile-time Python constant during CUDA Graph
        capture, so every step's own modulation is ALSO a compile-time
        constant -- precomputed here, once, never recomputed per
        replay.

        Timestep schedule: `d["shift"]` set -> the REAL non-uniform
        shift-based schedule (opportunities.md OPT-009's follow-up,
        `scheduler.build_inference_schedule`, ported verbatim from the
        real `WanContinuousFlowMatchScheduler` -- confirmed real LIBERO
        release values `shift=5.0`, `num_train_timesteps=1000`,
        `eval_num_inference_steps=10` in that release's own
        `config.yaml`). `d["shift"]` unset (default) -> this project's
        ORIGINAL fixed-uniform `action_timestep = 1.0 - step*dt`
        simplification, unchanged -- every existing caller/test is
        unaffected. Also returns `deltas` (`None` in the unset case) --
        the caller threads it into `imagewam_denoise_loop`'s own
        `deltas=` for the matching per-step Euler step size.

        `real_mod`: OPT-001, the real `mod_w` dict from
        `checkpoint_loader.build_real_modulation_weights()["action"]`
        (includes `head_adaln`) -- when given, used instead of random.
        """
        ahd = d["action_hidden_dim"]
        if real_mod is not None:
            mod_w = {k: self._own(v.to(DEV)) for k, v in real_mod.items()}
        else:
            mod_w = {
                "time_in_w1": self._own(torch.randn(ahd, 256, dtype=torch.float32, device=DEV) * 0.02),
                "time_in_w2": self._own(torch.randn(ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
                "mod_double": self._own(torch.randn(6 * ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
                "mod_single": self._own(torch.randn(3 * ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
                # OPT-001: head's own AdaLN modulation (shift/scale only, no
                # gate -- see adaln.head_modulation's own docstring).
                "head_adaln": self._own(torch.randn(2 * ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
            }
        shift = d.get("shift")
        deltas_out = None
        if shift is not None:
            from flash_rt.models.imagewam.scheduler import build_inference_schedule
            num_train_timesteps = d.get("num_train_timesteps", 1000)
            timesteps, deltas = build_inference_schedule(
                d["num_denoise_steps"], shift=shift, num_train_timesteps=num_train_timesteps, device=DEV)
            action_timesteps = (timesteps / num_train_timesteps).tolist()
            deltas_out = deltas.tolist()
        else:
            dt = d["dt"]
            action_timesteps = [1.0 - step * dt for step in range(d["num_denoise_steps"])]
        mods, head_mods = [], []
        for step, action_timestep in enumerate(action_timesteps):
            timestep = self._own(torch.full((1,), action_timestep, dtype=torch.float32, device=DEV))
            mod_double, mod_single = compute_action_modulation(timestep, mod_w, ahd)
            for t in mod_double[0]:
                self._own(t)
            for t in mod_double[1]:
                self._own(t)
            for t in mod_single:
                self._own(t)
            mods.append((mod_double, mod_single))
            head_shift, head_scale = compute_action_head_modulation(timestep, mod_w, ahd)
            self._own(head_shift)
            self._own(head_scale)
            head_mods.append((head_shift, head_scale))
        return mods, head_mods, deltas_out

    def _capture_graph(self) -> None:
        """Warm up and capture the active length (`self._active_dims`,
        `self._rope_table`) into `self._graph`.

        Every capture of this frontend uses one capture stream and one
        CUDA-graph memory pool, so the graphs of different text lengths
        share the temporaries their captures allocate (the VAE stage's
        activations, about 200 MB at 224x448, are the large ones). This
        is safe because `infer()` replays one graph at a time, to
        completion, and every value that outlives a replay lives in the
        frontend's own buffers, not in the pool.

        Python's cyclic garbage collector does not run during the capture:
        it runs once just before it, and is disabled until the capture
        ends. A dead reference cycle holding a CUDA graph (another
        frontend, for example) would otherwise be destroyed whenever an
        allocation triggers a collection, and destroying a CUDA graph
        while a stream captures invalidates the capture. `text_trim`
        captures new lengths while the process serves, so a collection
        can fall inside a capture at any time."""
        dims = self._active_dims
        if self._graph_stream is None:
            self._graph_stream = torch.cuda.Stream()
            self._graph_pool = torch.cuda.graph_pool_handle()
        s = self._graph_stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            if self._text_trim and not self._scratch_reserved:
                self._reserve_scratch_at_max_dims(s.cuda_stream)
            for _ in range(2):
                if self._vae_stage is not None:
                    self._vae_stage.run()
                imagewam_prefill(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                  dims, stream=s.cuda_stream, attn=self._attn,
                                  mod_txt=self._mod_txt, mod_img=self._mod_img,
                                  mod_single=self._mod_single, rope_table=self._rope_table.data_ptr())
                imagewam_denoise_loop(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                       dims, stream=s.cuda_stream, attn=self._attn,
                                       action_mods=self._action_mods, head_mods=self._head_mods,
                                       action_rope_table=self._action_rope_table.data_ptr(),
                                       deltas=self._deltas)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        gc.collect()
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            self._record_graph(graph, dims, s)
        finally:
            if gc_enabled:
                gc.enable()
        self._graph = graph

    def _record_graph(self, graph: torch.cuda.CUDAGraph, dims: dict, s: torch.cuda.Stream) -> None:
        """The capture itself (`_capture_graph`): VAE stage, prefill and
        denoise loop for `dims` on stream `s`, into the shared pool."""
        with torch.cuda.graph(graph, pool=self._graph_pool, stream=s):
            if self._vae_stage is not None:
                self._vae_stage.run()
            imagewam_prefill(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                              dims, stream=s.cuda_stream, attn=self._attn,
                              mod_txt=self._mod_txt, mod_img=self._mod_img,
                              mod_single=self._mod_single, rope_table=self._rope_table.data_ptr())
            imagewam_denoise_loop(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                   dims, stream=s.cuda_stream, attn=self._attn,
                                   action_mods=self._action_mods, head_mods=self._head_mods,
                                   action_rope_table=self._action_rope_table.data_ptr(),
                                   deltas=self._deltas)

    def _reserve_scratch_at_max_dims(self, stream: int) -> None:
        """`text_trim`, before the first capture: one eager prefill at the
        max dims.

        `Nvfp4Linear`, `Fp8Linear`, `StaticFp8Linear`,
        `CutlassFp16SwiGluMlp`, `Nvfp4SwiGluMlp` and `E0m3HadamardLinear`
        size their activation scratch by the largest `m` they have been
        called with and
        reallocate it for a larger one, which would free a buffer an
        already captured graph reads. After this pass every backbone
        weight op has seen its largest `m` (`x0` or `a0` of the max dims),
        so no later length reallocates. ActionDiT ops always run at
        `m = num_action`. The pass writes only rows that every captured
        graph recomputes before reading.
        """
        imagewam_prefill(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                          self.dims, stream=stream, attn=self._attn,
                          mod_txt=self._mod_txt, mod_img=self._mod_img,
                          mod_single=self._mod_single, rope_table=self._max_rope_table.data_ptr())
        self._scratch_reserved = True

    def _capture_graph_or_fall_back(self) -> None:
        """`_capture_graph()`, falling back to the cuBLAS chain when FA4
        fails.

        FA4 compiles on its first call, during `_capture_graph`'s eager
        warmup, and can fail there or during capture: an FA4 runtime can
        import and then fail to compile for sm_110 (see `fa4_backend`),
        or a kernel can be rejected inside stream capture. If any site
        runs FA4 and warmup or capture raises, this logs an error, emits
        a `RuntimeWarning`, records the reason in
        `self.fa4_fallback_reason`, rebuilds the attention backend with
        FA4 off at both sites, and captures again. A failure with FA4 off,
        or a second failure after the fallback, propagates; after a second
        failure the frontend holds no graph (`_graph` is None and the
        per-length cache is empty).

        An invalidated capture (for example a device sync inside it)
        makes `torch.cuda.graph`'s exit raise before it restores the
        caller's stream, so the current stream is restored here first.
        """
        caller_stream = torch.cuda.current_stream()
        try:
            self._capture_graph()
            return
        except Exception as exc:  # FA4 compile/launch/capture errors are not one exception type
            torch.cuda.set_stream(caller_stream)
            if not (self.use_fa4 or self.use_fa4_mot):
                raise
            reason = f"{type(exc).__name__}: {exc}"
        message = (f"ImageWAM FA4 attention failed during warmup/capture "
                   f"(use_fa4={self.use_fa4}, use_fa4_mot={self.use_fa4_mot}): {reason} -- "
                   f"falling back to the cuBLAS attention chain at both sites")
        logger.error(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        self.fa4_fallback_reason = reason
        self.use_fa4 = False
        self.use_fa4_mot = False
        self._attn = self._build_attn_backend()
        torch.cuda.synchronize()
        try:
            self._capture_graph()
        except BaseException:
            torch.cuda.set_stream(caller_stream)
            self._graph = None
            self._captures.clear()
            raise
        # Graphs of other text lengths captured with FA4 are dropped, so
        # every graph this frontend replays uses the same attention. They
        # (and their RoPE tables) stay alive until the replacement graph
        # exists; after a second failure no graph is left at all.
        self._captures.clear()

    def _set_context_with_optional_proprio(self, text_ctx: torch.Tensor, text_mask: torch.Tensor) -> None:
        """`text_ctx`: `(text_len, joint_attention_dim)` BF16 -- the
        real (or precomputed) Qwen3 text context, BEFORE proprio.
        `text_mask`: `(text_len,)` bool, `1` for real tokens.

        `self._proprio_dim is None`: `text_ctx` must already be exactly
        `dims["x0"]` rows -- copied in directly, unchanged behavior.

        `self._proprio_dim` set: replicates `imagewam.py`'s own real
        `_append_proprio_to_context` (`pack_proprio_after_text=True`
        branch) EXACTLY -- real tokens keep their rank, the proprio
        slot lands at row `valid_counts` (right after the last real
        token), padding shifts one row later to make room. Real tokens'
        own RoPE positions are unaffected (same indices either way);
        the proprio slot's position must match the real model's
        placement for its own RoPE position to be correct -- found
        while scoping real closed-loop testing, opportunities.md.
        `dims["x0"]` must equal `text_ctx`'s own length + 1 (validated
        below). The proprio ROW ITSELF is left zero here -- `infer()`
        overwrites it with the real `proprio_encoder(proprio)` output
        every call, since proprio (unlike the text prompt) changes
        every control step.
        """
        x0 = self.dims["x0"]
        if self._proprio_dim is None:
            if text_ctx.shape[0] != x0:
                raise ValueError(f"context length {text_ctx.shape[0]} != dims['x0']={x0}")
            self._context.copy_(text_ctx)
            return
        text_len = text_ctx.shape[0]
        if x0 != text_len + 1:
            raise ValueError(
                f"dims['x0']={x0} must equal the text context length ({text_len}) + 1 "
                f"(the proprio slot) when dims['proprio_dim'] is set")
        valid_counts = int(text_mask.sum().item())
        self._proprio_row = valid_counts
        self._context.zero_()
        self._context[:valid_counts].copy_(text_ctx[:valid_counts])
        self._context[valid_counts + 1:x0].copy_(text_ctx[valid_counts:text_len])

    def _write_trimmed_context(self, text_ctx: torch.Tensor, text_mask: torch.Tensor) -> int:
        """`text_trim`: the valid text rows packed by rank, then the
        proprio slot (when `dims["proprio_dim"]` is set), into
        `self._context[:x0]`; the rows after them are zeroed. Returns the
        active `x0` (`n_valid + 1` with proprio, `n_valid` without)."""
        packed = pack_trimmed_context(text_ctx, text_mask, proprio_slot=self._proprio_dim is not None)
        x0 = int(packed.rows.shape[0])
        if x0 > self.dims["x0"]:
            raise ValueError(f"{packed.n_valid} valid text tokens need x0={x0} rows, more than "
                             f"dims['x0']={self.dims['x0']}")
        self._context.zero_()
        self._context[:x0].copy_(packed.rows)
        self._proprio_row = packed.proprio_row
        return x0

    def _write_context(self, text_ctx: torch.Tensor, text_mask: torch.Tensor) -> int:
        """Writes a prompt's context rows and returns the context length
        `x0` the prompt runs at: `dims["x0"]` with `text_trim=False`
        (every row, padding included), the valid length with
        `text_trim=True`."""
        if self._text_trim:
            return self._write_trimmed_context(text_ctx, text_mask)
        self._set_context_with_optional_proprio(text_ctx, text_mask)
        return self.dims["x0"]

    def _capture_text_length(self, x0: int) -> TextLengthCapture:
        """Captures a graph for context length `x0` and returns its record
        (not yet cached; `_active_dims`/`_rope_table`/`_graph` hold the new
        length afterwards).

        The new length gets its sequence dims (`trimmed_sequence_dims`),
        its backbone RoPE table (text positions `0..x0-1`, image positions
        unchanged), the static FP8 calibration before the first capture
        only, the GEMM autotune of the shapes it adds (only on a
        `GemmRunner` this frontend owns; the `fp16_nn` weight-GEMM shapes
        only for `precision="fp16"`, the one precision whose backbone
        weight GEMMs run on `fp16_nn`), and a capture."""
        dims = trimmed_sequence_dims(self.dims, x0)
        if dims is self.dims:
            rope_table = self._max_rope_table
        else:
            rope_table = build_backbone_rope_table(x0, *self._ref_hw, device=DEV)
        self._active_dims, self._rope_table = dims, rope_table
        if not self._fp8_calibrated:
            self._calibrate_fp8(self.dims)
            self._fp8_calibrated = True
        if self._owns_gemm:
            self._autotune_gemm(dims, fp16_nn_shapes=Precision(self._precision).fp16_nn_backbone_gemm)
        self._capture_graph_or_fall_back()
        return TextLengthCapture(dims=dims, rope_table=rope_table, graph=self._graph)

    def _invalidate_active_graph(self) -> None:
        """After a failed capture: the context rows may already hold a
        prompt no captured graph matches, so `infer()` refuses until a
        `set_prompt` succeeds, and the next `set_prompt` never returns
        early on the old cache key. Cached captures stay valid."""
        self._graph = None
        self._current_prompt = None

    def _activate_text_length(self, x0: int) -> None:
        """Makes the capture for context length `x0` the one `infer()`
        replays, capturing it first if this length has none
        (`_capture_text_length`). If that capture raises, no graph is
        active afterwards (`_invalidate_active_graph`) and the exception
        propagates; the cached captures of other lengths, with their RoPE
        tables, are kept (the FA4 fallback's own rule aside)."""
        capture = self._captures.get(x0)
        if capture is None:
            try:
                capture = self._capture_text_length(x0)
            except BaseException:
                self._invalidate_active_graph()
                raise
            self._captures[x0] = capture
        self._active_dims, self._rope_table, self._graph = capture.dims, capture.rope_table, capture.graph

    def precapture_text_lengths(self, x0s: Sequence[int]) -> None:
        """`text_trim`: captures a graph for every context length in `x0s`
        (valid text tokens + 1 with proprio, as `captured_text_lengths`
        reports) that has none, so the first `set_prompt` of a prompt
        with that length only switches graphs. The context rows and the
        active length are unchanged afterwards. If a capture raises, no
        graph is active afterwards, as for a failed `set_prompt`."""
        if not self._text_trim:
            raise ValueError("precapture_text_lengths needs text_trim=True")
        active_x0 = None if self._graph is None else int(self._active_dims["x0"])
        wanted = [int(x0) for x0 in x0s]
        try:
            # An FA4 fallback during one capture drops the graphs captured
            # before it (at most once: FA4 is off afterwards), so the
            # missing set is recomputed after every capture.
            missing = [x0 for x0 in wanted if x0 not in self._captures]
            while missing:
                self._captures[missing[0]] = self._capture_text_length(missing[0])
                missing = [x0 for x0 in wanted if x0 not in self._captures]
        except BaseException:
            self._invalidate_active_graph()
            raise
        if active_x0 is None:
            self._graph = None
        else:
            self._activate_text_length(active_x0)

    @property
    def active_dims(self) -> dict:
        """A copy of the dims the active graph runs (`x0/a0/total` of the
        current prompt with `text_trim=True`; `self.dims` otherwise)."""
        return dict(self._active_dims)

    @property
    def captured_text_lengths(self) -> tuple[int, ...]:
        """The context length `x0` of every cached capture, ascending."""
        return tuple(sorted(self._captures))

    def _text_max_length(self) -> int:
        """The padded text length this frontend's own context implies: the
        number of rows `_set_context_with_optional_proprio` requires of the
        encoder's output, `dims["x0"] - 1` when `dims["proprio_dim"]` is set
        (the proprio row takes one row of the context), else `dims["x0"]`.
        LIBERO (`x0=513`, `proprio_dim=8`) therefore encodes at 512, FLUX.2's
        own `MAX_LENGTH`."""
        if self._proprio_dim is None:
            return self.dims["x0"]
        return self.dims["x0"] - 1

    def set_prompt(self, prompt_text: str | None = None, *,
                    context: torch.Tensor | None = None,
                    context_mask: torch.Tensor | None = None) -> None:
        """Random-fills the text-context input (default), OR loads a
        real precomputed `context`/`context_mask` pair, OR (if this
        frontend was constructed with `qwen3_model_spec=`) live-encodes
        `prompt_text` through the real Qwen3 text encoder -- then
        captures the graph. Matches `imagewam.py`'s own real
        `_prepare_flux2_infer_text`: a raw prompt XOR a precomputed
        `context`/`context_mask` pair, never both.

        A precomputed `context` is applied on every call (issues.md
        ISSUE-060). The live-Qwen3 and random paths return early when
        called again with the same `prompt_text`.

        `context_mask` places the proprio row (`dims["proprio_dim"]`
        set, see `_set_context_with_optional_proprio`). With
        `text_trim=True` it also sets the context length: the valid
        tokens and the proprio row only (`_write_trimmed_context`), with
        one graph per distinct length, captured on first use. With
        `text_trim=False` the attention reads every context row,
        padding included. The random path always uses `dims["x0"]`.
        """
        if prompt_text is not None and context is not None:
            raise ValueError("set_prompt: prompt_text and context are mutually exclusive "
                              "(matches imagewam.py's own _prepare_flux2_infer_text)")
        cache_key = (prompt_text, context is not None)
        if context is None and cache_key == self._current_prompt:
            return
        if context is not None:
            if context_mask is None:
                raise ValueError("set_prompt(context=...) requires context_mask too "
                                  "(matches imagewam.py's own _prepare_flux2_infer_text)")
            x0 = self._write_context(
                context.to(device=DEV, dtype=BF16), context_mask.to(device=DEV, dtype=torch.bool))
        elif self._qwen3 is not None and prompt_text is not None:
            from flash_rt.models.imagewam.text_encoder import encode_prompts
            model, tokenizer = self._qwen3
            # The context width this frontend's dims imply, not the encoder
            # module's own 512 default: a workload with text_max_len=128
            # needs a (128, ...) context (ISSUE-083).
            real_context, real_mask = encode_prompts(model, tokenizer, [prompt_text],
                                                      max_length=self._text_max_length())
            x0 = self._write_context(real_context[0].to(device=DEV, dtype=BF16), real_mask[0].to(device=DEV))
        else:
            self._context.normal_()
            if self._proprio_dim is not None:
                # No real context/mask on this structural path -- no
                # "real token count" to place the proprio slot after,
                # so it goes at the very last row. Arbitrary, no
                # accuracy claim on this path either way (matches
                # every other random-fill branch in this class).
                self._proprio_row = self.dims["x0"] - 1
            x0 = self.dims["x0"]
        self._activate_text_length(x0)
        self._current_prompt = cache_key

    def infer(self, observation: dict, *, action_noise: torch.Tensor | None = None) -> dict:
        """Replay the captured graph with a new observation.

        `action_noise`: optional `(num_action, action_dim)` initial action
        latent for the flow-matching sampler, copied in as given. `None`
        (the default) keeps the served sampler: `0.01 * N(0,1)` drawn on
        the device (issues.md ISSUE-002). The regression gate
        (`tests/gate_imagewam_libero.py`) passes the fixture's fixed noise
        here so every run of the served path starts from the same latent.

        `observation` random-fills `img_raw` by default (unchanged
        placeholder, standing in for whatever a real VAE would have
        produced) UNLESS this frontend was constructed with
        `ae_model_path=`/`flux2_src=` AND `observation` carries a real
        `view1` ... `view<num_views>` camera frame -- then the real VAE
        (`vae_encoder.encode_to_tokens`, one encode of every view
        concatenated horizontally) runs OUTSIDE the captured graph (plain
        PyTorch/`flux2`-dependent code has no business being captured) and
        its result is copied into `img_raw` before `.replay()`.
        `backbone_hidden`'s own image rows are WRITTEN by
        `img_in.weight` inside the graph itself either way, not filled
        directly here. `num_views` is the served workload's camera count
        (`observation_views` raises when a `view*` key is missing).

        `observation["proprio"]` -- real closed-loop robot-state
        conditioning (opportunities.md, found 2026-09-15), REQUIRED
        (raises `ValueError`, matching the real model's own
        `_append_proprio_to_context_if_enabled`) when this frontend was
        constructed with `dims["proprio_dim"]` set. Unlike the text
        prompt (fixed per `set_prompt()` episode), proprio genuinely
        changes every control step -- normalized via the real
        `dataset_stats.json` `state` min/max (if `dataset_stats_path`
        was given at construction; raw otherwise, no accuracy claim),
        projected through the real `proprio_encoder` OUTSIDE the graph
        (plain `F.linear`, same convention as the VAE/Qwen3 encoders),
        and copied into the row `set_prompt()` already reserved for it
        (`self._proprio_row`) -- BEFORE `.replay()`, same pattern as
        `img_raw`.

        `vae_graph_input` given (roadmap item 5, plan.md): the VAE stage
        (preprocessing kernel, encoder, token write into `img_raw`) is
        part of the captured graph, so this method only copies
        `view1` ... `view<num_views>` (shape fixed by `vae_graph_input`)
        into the stage's fixed uint8 buffer, stages proprio and noise, and
        replays once. Views are then required on every call.
        `vae_encoder` selects the encoder in both placements.
        """
        if self._graph is None:
            raise RuntimeError("call set_prompt() before infer()")
        self.stage_inputs(observation, noise=action_noise)
        self._graph.replay()
        torch.cuda.synchronize()
        return {"actions": self.read_actions()}

    @property
    def weights(self) -> dict:
        """The weight dict `pipeline_thor.py` dispatches through (keys
        `(model, stream, layer, slot)`, values linear-op objects or raw
        pointers). Read-only use: `run_eager()` takes a wrapped copy."""
        return self._weights

    def stage_inputs(self, observation: dict, noise: torch.Tensor | None = None) -> None:
        """Write one observation into the persistent input buffers -- the
        pre-replay half of `infer()`: `img_raw` (real VAE when available,
        else random), the proprio row of `context`, and the initial action
        latent. With `vae_graph_input` the views go into the VAE stage's
        fixed uint8 buffer instead (the stage encodes them inside the
        graph, or in `run_eager()`), and are required on every call.
        The views of an observation with a real VAE are
        `observation_views(observation, self.num_views)`: `view1` ...
        `view<num_views>` of the workload this frontend serves.

        `noise`: `(num_action, action_dim)` initial action latent, copied
        in as given (`infer()`'s `action_noise`). `None` keeps the served
        sampler (`0.01 * N(0,1)`, `issues.md` ISSUE-002); the calibration
        builder and the end-to-end checks pass the official sampler's
        unscaled `N(0,1)` noise."""
        if self._vae_stage is not None:
            # VAE inside the graph: it encodes whatever the fixed view
            # buffer holds, so the views are required every call.
            if "view1" not in observation:
                raise ValueError("with vae_graph_input (VAE in the graph) infer()/stage_inputs() need "
                                 f"observation['view1'] ... observation['view{self.num_views}'] every call")
            self.stage_images(*observation_views(observation, self.num_views))
        elif self._ae is not None and "view1" in observation:
            self.stage_images(*observation_views(observation, self.num_views))
        else:
            self._img_raw.normal_()
        if self._proprio_dim is not None:
            proprio = observation.get("proprio")
            if proprio is None:
                raise ValueError(
                    "infer(observation=...) requires observation['proprio'] when "
                    "dims['proprio_dim'] is set (matches imagewam.py's own "
                    "_append_proprio_to_context_if_enabled)")
            self.stage_proprio(proprio)
        if noise is None:
            self._action_latent.normal_()
            self._action_latent.mul_(0.01)
        else:
            if tuple(noise.shape) != tuple(self._action_latent.shape):
                raise ValueError(
                    f"action_noise shape {tuple(noise.shape)} != action latent "
                    f"{tuple(self._action_latent.shape)}")
            self._action_latent.copy_(noise)

    def run_eager(self, weights: dict | None = None) -> None:
        """Prefill + the full denoise loop on the staged buffers, eagerly
        (no CUDA Graph), on the current stream -- same kernels, same
        buffers, same order as the captured graph (including the VAE
        stage when `vae_graph_input` put it in the graph). `weights` (default:
        this frontend's own) must have the same keys; the calibration
        builder passes recording wrappers around the real weights
        (`activation_recorder.py`), which a graph replay would bypass.
        `set_prompt()` must have run (it stages the text context). Runs
        the active prompt's dims and RoPE table (`active_dims`: the
        trimmed length with `text_trim=True`), as the active graph does."""
        if self._current_prompt is None:
            raise RuntimeError("call set_prompt() before run_eager()")
        w = self._weights if weights is None else weights
        stream = torch.cuda.current_stream().cuda_stream
        if self._vae_stage is not None:
            # With vae_graph_input the VAE encode is part of the graph, so
            # stage_inputs() only filled the stage's uint8 view buffer;
            # run the stage here, as the graph would, to fill img_raw.
            self._vae_stage.run()
        dims = self._active_dims
        imagewam_prefill(self._ctx, fvk, self._gemm, self._bufs, w, dims, stream=stream,
                         attn=self._attn, mod_txt=self._mod_txt, mod_img=self._mod_img,
                         mod_single=self._mod_single, rope_table=self._rope_table.data_ptr())
        imagewam_denoise_loop(self._ctx, fvk, self._gemm, self._bufs, w, dims, stream=stream,
                              attn=self._attn, action_mods=self._action_mods, head_mods=self._head_mods,
                              action_rope_table=self._action_rope_table.data_ptr(),
                              deltas=self._deltas)

    # -- per-tick staging operations ------------------------------------
    # Shared by infer() and the model-runtime verbs
    # (flash_rt/models/imagewam/runtime_export.py). Each runs its torch
    # ops on the caller's current stream.

    def stage_images(self, *views: torch.Tensor) -> None:
        """Stage `self.num_views` `(H,W,3)` uint8 camera views (the served
        workload's cameras), in view order. With the VAE inside the graph
        (`vae_graph_input`) the views are copied into the stage's fixed uint8
        buffer and the next replay encodes them; otherwise the configured
        preprocessing kernel and VAE encoder run now (outside the graph) and
        the tokens are written into `img_raw`.

        A view count other than `self.num_views` raises: `img_raw` and the
        captured graph are sized for that count, so a shorter or longer
        sequence of views is a mismatch, not a narrower run.
        """
        if self._ae is None:
            raise RuntimeError("stage_images requires ae_model_path/flux2_src at construction")
        if self._vae_stage is not None:
            # The stage validates the count against its own spec.num_views.
            self._vae_stage.stage([torch.as_tensor(v) for v in views])
            return
        if len(views) != self.num_views:
            raise ValueError(f"stage_images takes this frontend's {self.num_views} views (num_views of "
                             f"the workload it serves, runtime_surface().view_shape), got {len(views)}")
        from flash_rt.models.imagewam.vae_encoder import encode_to_tokens
        tokens = encode_to_tokens(self._ae, [torch.as_tensor(v) for v in views],
                                  preprocessor=self._vae_pre, encoder=self._vae_encoder)
        self._img_raw.copy_(tokens[0].to(dtype=BF16))

    def stage_proprio(self, proprio: np.ndarray) -> None:
        """Normalize raw robot state with the dataset `state` min/max (when
        loaded), project it through the real `proprio_encoder` (outside the
        graph), and write it into the context row `set_prompt()` reserved."""
        if self._proprio_dim is None:
            raise RuntimeError("stage_proprio requires dims['proprio_dim'] at construction")
        proprio_t = torch.as_tensor(proprio, dtype=torch.float32, device=DEV).reshape(1, self._proprio_dim)
        if self._state_norm is not None:
            proprio_t = self._state_norm.forward(proprio_t)
        proprio_tok = torch.nn.functional.linear(
            proprio_t.to(dtype=BF16), self._proprio_w, self._proprio_b)
        self._context[self._proprio_row].copy_(proprio_tok[0])

    def read_actions(self) -> np.ndarray:
        """The `(num_action, action_dim)` f32 chunk after replay: denormalized
        with the dataset `action` min/max when loaded, else the raw latent.
        Blocks on the current stream (device-to-host copy)."""
        actions = self._action_latent.detach()
        if self._action_norm is not None:
            actions = self._action_norm.backward(actions)
        return actions.cpu().numpy()

    # -- runtime export ---------------------------------------------------

    def _refuse_text_trim(self, what: str) -> None:
        """`runtime_surface()` / `pipeline_resources()` describe one graph at
        `self.dims`; with `text_trim` the graph, the context length and the
        RoPE table change with every new prompt length (opportunities.md
        OPT-030 lists what per-length support needs)."""
        if self._text_trim:
            raise ValueError(f"{what} does not support text_trim=True: it describes one graph at the max dims, "
                             f"while a trimmed frontend runs one graph per prompt length (active_dims); "
                             f"construct with text_trim=False")

    def runtime_surface(self) -> ImageWAMRuntimeSurface:
        """The captured graph and its device windows, for the runtime export.
        Not available with `text_trim=True` (`ValueError`)."""
        self._refuse_text_trim("runtime_surface()")
        if self._graph is None:
            raise RuntimeError("call set_prompt() before runtime_surface()")
        d = self.dims
        setup = [("pipeline", type(self).__name__), ("precision", self._precision),
                 ("use_fa4", str(self.use_fa4)), ("use_fa4_mot", str(self.use_fa4_mot)),
                 ("calibration", self._calibration_digest), ("nvfp4_awq", str(self._nvfp4_awq))]
        if self._nvfp4_awq:
            setup.extend((("awq_alpha", str(self._awq_alpha)), ("awq_scope", self._awq_scope)))
        setup.extend(self._vae_setup)
        if self._workload is not None:
            # The workload the graph was built for, beside the dims it
            # produced (`runtime_surface.workload_identity`).
            setup.extend(workload_identity(self._workload))
        setup.extend((f"dims.{k}", str(d[k])) for k in sorted(d))
        return ImageWAMRuntimeSurface(
            graph_exec=int(self._graph.raw_cuda_graph_exec()),
            stream=self._graph_stream,
            img_raw=self._img_raw,
            context=self._context,
            action_latent=self._action_latent,
            img_len=d["a0"] - d["x0"],
            token_dim=d["HD"],
            num_action=d["num_action"],
            action_dim=d["action_dim"],
            proprio_dim=self._proprio_dim,
            has_vae=self._ae is not None,
            has_text_encoder=self._qwen3 is not None,
            action_denormalized=self._action_norm is not None,
            setup_identity=tuple(setup),
            context_rows=d["x0"],
            context_width=d["joint_attention_dim"],
            proprio_row=self._proprio_row,
            proprio_weight=self._proprio_w,
            proprio_bias=self._proprio_b,
            state_scale=None if self._state_norm is None else self._state_norm.scale,
            state_offset=None if self._state_norm is None else self._state_norm.offset,
            action_scale=None if self._action_norm is None else self._action_norm.scale,
            action_offset=None if self._action_norm is None else self._action_norm.offset,
            view_shape=self._input_view_shape(),
            views_u8=None if self._vae_stage is None else self._vae_stage.views_u8,
            owner=self,
        )

    def pipeline_resources(self) -> ImageWAMPipelineResources:
        """Everything the native C++ pipeline needs to record the same
        prefill and denoise loop as `pipeline_thor.py` over this frontend's
        buffers and weights (`flash_rt/models/imagewam/pipeline_resources.py`).
        Every AdaLN site carries the fp16 form `pipeline_thor`'s
        `fp16_adaln_operands` / `fp16_adaln_shift_scale` build (the unfused path and the
        standalone AdaLN that starts each chain) and the FP32 modulation
        chunks the fused gated residual + next AdaLN kernel reads
        (`dims["fuse_res_norm"]`). Not available with `text_trim=True`
        (`ValueError`)."""
        self._refuse_text_trim("pipeline_resources()")
        if self._graph is None:
            raise RuntimeError("call set_prompt() before pipeline_resources()")
        if self.use_fa4 or self.use_fa4_mot:
            raise ValueError("the native pipeline has no FA4 attention; use_fa4/use_fa4_mot must be off")
        if self._vae_stage is not None:
            raise ValueError("the native pipeline has no VAE stage; construct with vae_graph_input=None "
                             "(the VAE then runs outside the graph and feeds image_tokens)")
        if self._nvfp4_awq:
            raise ValueError("the native pipeline has no AWQ input-scale fold; construct with nvfp4_awq=False")
        d = self.dims
        if not d.get("merge_qkv_mlp"):
            raise ValueError("the native pipeline records the merged single-stream linear1 only "
                             f"(precision {self._precision!r} uses the split path)")
        img_len = d["a0"] - d["x0"]
        hidden, ahd, na = d["hidden"], d["action_hidden_dim"], d["num_action"]

        def adaln(group: tuple[torch.Tensor, torch.Tensor, torch.Tensor], rows: int,
                  dim: int) -> AdaLNResource:
            shift, scale, gate = group
            return AdaLNResource(*fp16_adaln_operands(shift, scale, gate, rows, dim),
                                 shift_f32=shift, scale_f32=scale, gate_f32=gate)

        def head_adaln(pair: tuple[torch.Tensor, torch.Tensor]) -> AdaLNResource:
            shift, scale = pair
            return AdaLNResource(*fp16_adaln_shift_scale(shift, scale), None,
                                 shift_f32=shift, scale_f32=scale, gate_f32=None)

        def single_layer(stack: str, L: int) -> SingleLayerResource:
            merged = bool(d.get("merge_linear2"))
            return SingleLayerResource(
                linear1=lin(stack, "single", L, "linear1.weight"),
                attn_out_proj=None if merged else lin(stack, "single", L, "attn_out_proj.weight"),
                mlp_down=None if merged else lin(stack, "single", L, "mlp_down.weight"),
                linear2=lin(stack, "single", L, "linear2.weight") if merged else None,
                query_norm=int(weight(stack, "single", L, "query_norm")),
                key_norm=int(weight(stack, "single", L, "key_norm")))

        def weight(stack: str, block: str, layer: int, slot: str) -> object:
            """One `self._weights` entry: a linear op, or a norm/bias pointer."""
            return self._weights[(stack, block, layer, slot)]

        def lin(stack: str, block: str, layer: int, slot: str) -> LinearResource:
            return linear_resource(weight(stack, block, layer, slot))

        steps = []
        for step in range(d["num_denoise_steps"]):
            mod_double, mod_single = self._action_mods[step]
            steps.append(ActionStepResource(
                double1=adaln(mod_double[0], na, ahd), double2=adaln(mod_double[1], na, ahd),
                single=adaln(mod_single, na, ahd),
                head=head_adaln(self._head_mods[step]),
                delta=float(d["dt"] if self._deltas is None else self._deltas[step])))
        return ImageWAMPipelineResources(
            dims=PipelineDims(
                hidden=hidden, head_dim=d["HD"], num_heads=d["NH"], mlp_hidden=d["mlp_hidden"],
                joint_attention_dim=d["joint_attention_dim"], x0=d["x0"], a0=d["a0"], total=d["total"],
                num_action=na, action_dim=d["action_dim"], action_hidden_dim=ahd,
                action_attn_width=d["action_attn_width"], action_mlp_hidden=d["action_mlp_hidden"],
                num_double=d["num_layers_double"], num_single=d["num_layers_single"],
                action_num_double=d["action_num_layers_double"],
                action_num_single=d["action_num_layers_single"],
                num_steps=d["num_denoise_steps"], merge_linear2=bool(d.get("merge_linear2")),
                fuse_res_norm=bool(d.get("fuse_res_norm")), eps=1e-6),
            buffers=PipelineBuffers(**{name: int(self._bufs[name])
                                       for name in PipelineBuffers.__dataclass_fields__}),
            attention=AttentionResource(
                q_o=self._Q_O.data_ptr(), k_cache=self._K_cache.data_ptr(),
                v_cache=self._V_cache.data_ptr(), logits=self._logits.data_ptr(),
                kv_layer_stride_bytes=self._K_cache[0].numel() * self._K_cache.element_size(),
                scale=1.0 / (d["HD"] ** 0.5), rope_table=self._rope_table.data_ptr(),
                action_rope_table=self._action_rope_table.data_ptr()),
            txt_in=lin("backbone", "double", 0, "txt_in.weight"),
            img_in=lin("backbone", "double", 0, "img_in.weight"),
            action_encoder=lin("action_dit", "shared", 0, "action_encoder.weight"),
            head_linear=lin("action_dit", "shared", 0, "head.linear.weight"),
            action_encoder_bias=int(weight("action_dit", "shared", 0, "action_encoder.bias")),
            txt_mod1=adaln(self._mod_txt[0], d["x0"], hidden),
            txt_mod2=adaln(self._mod_txt[1], d["x0"], hidden),
            img_mod1=adaln(self._mod_img[0], img_len, hidden),
            img_mod2=adaln(self._mod_img[1], img_len, hidden),
            single_mod=adaln(self._mod_single, d["a0"], hidden),
            double_layers=tuple(DoubleLayerResource(
                **{slot: lin("backbone", "double", L, f"{slot}.weight")
                   for slot in ("txt_qkv", "img_qkv", "txt_proj", "img_proj",
                                "txt_mlp0", "img_mlp0", "txt_mlp2", "img_mlp2")},
                **{slot: int(weight("backbone", "double", L, slot))
                   for slot in ("txt_query_norm", "txt_key_norm", "img_query_norm", "img_key_norm")})
                for L in range(d["num_layers_double"])),
            single_layers=tuple(single_layer("backbone", L) for L in range(d["num_layers_single"])),
            action_double_layers=tuple(ActionDoubleLayerResource(
                qkv=lin("action_dit", "double", L, "qkv.weight"),
                proj=lin("action_dit", "double", L, "proj.weight"),
                mlp0=lin("action_dit", "double", L, "mlp0.weight"),
                mlp2=lin("action_dit", "double", L, "mlp2.weight"),
                query_norm=int(weight("action_dit", "double", L, "query_norm")),
                key_norm=int(weight("action_dit", "double", L, "key_norm")))
                for L in range(d["action_num_layers_double"])),
            action_single_layers=tuple(single_layer("action_dit", L)
                                       for L in range(d["action_num_layers_single"])),
            steps=tuple(steps),
        )

    def gemm_algo(self, kind: int, m: int, n: int, k: int) -> bytes | None:
        """The cuBLASLt algorithm this frontend's GemmRunner uses for one
        `bf16_nn` (kind 0) / `fp16_nn` (kind 1) shape, or None if that shape
        has not been planned; the native pipeline installs it so both run the
        same kernel."""
        return self._gemm.cached_algo(kind, m, n, k)

    def export_model_runtime(self, *, identity: Mapping[str, str] | None = None, io: str = "python",
                             native: ImageWAMNativeRuntime | None = None) -> ModelRuntime:
        """Package the captured graph as an `frt_model_runtime_v1`. See
        `flash_rt.models.imagewam.runtime_export.export_model_runtime`.
        Needs the exec/ and runtime/ native modules (built separately);
        `io="native"` also needs `native` (an `ImageWAMNativeRuntime`).
        Not available with `text_trim=True` (`runtime_surface()` raises
        `ValueError`)."""
        from flash_rt.models.imagewam.runtime_export import export_model_runtime
        return export_model_runtime(self, identity=identity, io=io, native=native)


def load_imagewam(ckpt_path: str | None, workload: ImageWAMWorkload, *,
                  structure: ImageWAMStructure | None = None, profile: str = "default",
                  precision: str | Precision | None = None,
                  calibration_path: str | None = None, ae_model_path: str | None = None,
                  flux2_src: str | None = None, qwen3_model_spec: str | None = None,
                  dataset_stats_path: str | None = None, consumer: str = "infer",
                  allow_placeholder_calibration: bool = False,
                  vae_resize: str = "area",
                  **expert) -> ImageWAMTorchFrontendThor:
    """The deployment entry: resolve the configuration, then build the frontend.

    `workload` says what is served (`ImageWAMWorkload`); `ckpt_path` is the
    checkpoint the structure (and, with `ckpt_path`, the weights) come from.
    `structure=None` reads `ImageWAMStructure.from_checkpoint(ckpt_path)`, so
    the widths are the checkpoint's own; passing `structure` is for a caller
    that already read it, and it is required when `ckpt_path` is `None`
    (random-weight dry runs).

    `profile` names the option set (`config_resolver.PROFILES`); `precision`
    overrides the profile's; `calibration_path` is the static-FP8 / AWQ
    statistics file; `consumer` is what the configuration will be used for
    (`"infer"`, `"abi"`, `"native"`), which is what makes the ABI and native
    refusals (`text_trim`, AWQ, FA4, the VAE stage) apply.
    `allow_placeholder_calibration` accepts the N(0, 0.1) placeholder scales
    for a static-FP8 precision without a calibration file; the default
    refuses it (rule R1).

    Every legality decision is `resolve_config`'s: an illegal combination
    raises its `ConfigError` (message `<rule id>: ...`) before the frontend
    is constructed, so nothing is allocated for a configuration that cannot
    run. `**expert` carries the expert tier (`config_resolver.EXPERT_KEYS`).
    """
    from flash_rt.models.imagewam.config_resolver import resolve_config
    from flash_rt.models.imagewam.structure import ImageWAMStructure

    if structure is None:
        if ckpt_path is None:
            raise ValueError(
                "load_imagewam needs ckpt_path (the structure is read from the checkpoint) or an "
                "explicit structure= for a random-weight run")
        structure = ImageWAMStructure.from_checkpoint(ckpt_path)
    resolved = resolve_config(workload, structure, profile=profile, precision=precision,
                              calibration_path=calibration_path, ae_model_path=ae_model_path,
                              consumer=consumer,
                              allow_placeholder_calibration=allow_placeholder_calibration,
                              **expert)
    return ImageWAMTorchFrontendThor.from_config(
        resolved, workload=workload, ckpt_path=ckpt_path, ae_model_path=ae_model_path,
        flux2_src=flux2_src, qwen3_model_spec=qwen3_model_spec,
        dataset_stats_path=dataset_stats_path, vae_resize=vae_resize)
