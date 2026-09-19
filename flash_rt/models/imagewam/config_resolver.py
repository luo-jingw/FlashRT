"""ImageWAM configuration resolver: profiles + the single rule set for
illegal combinations (plan.md "Plan: configuration consolidation", W4).

`resolve_config(workload, structure, ...)` turns a workload (what is
served), a structure (what the checkpoint is), a named profile and optional
overrides into a `ResolvedConfig(dims, options, effective_config)`, or
raises one `ConfigError` whose message starts with the id of the rule that
was violated. Nothing here builds a buffer, loads a weight or touches a GPU.

Import contract: this module imports without torch and without the compiled
kernels. Its only runtime dependency is the leaf `precision.py`
(standard library only). The workload and structure objects are used by
attribute access only (`workload.layout(structure)`, `structure.hidden`,
...), so a caller does not even need `structure.py` (which reads a
checkpoint through torch) to be importable. `calibration_file.py` (torch,
safetensors) is imported lazily, inside rule R8, and only when a
calibration file exists on disk. The package `__init__` of
`flash_rt.models.imagewam` imports the whole pipeline; when this file is
loaded by path (`importlib.util.spec_from_file_location`, as
tests/test_imagewam_config_resolver.py does in one subprocess test) there is
no package, and `precision.py` is loaded by path from the same directory.

Profiles (`PROFILES`) map a name to option values. `default` reproduces
today's constructor defaults of `ImageWAMTorchFrontendThor` (pinned against
the constructor source by the test). `fast` is PROVISIONAL, see its
`ProfileSpec.description`. `precision=` overrides the profile's precision;
`**expert` overrides individual options (`EXPERT_KEYS`).

Rules (each moves a check that exists today; ids are stable, tests pin
them):

    R1  static-FP8 precision without a calibration file
    R2  gemm_variant_autotune with a precision that has no switchable tile
    R3  a real VAE encoder / VAE in the graph without ae_model_path
    R4  nvfp4_awq with a non-AWQ precision or without a calibration file
    R5  text_trim with a consumer whose path carries no per-length graph
        (native; the ABI carries one graph per captured length)
    R6  native consumer with something the native pipeline does not carry
        (AWQ, a precision it cannot describe, FA4, the VAE stage)
    R7  workload layout inconsistent with the structure
    R8  calibration file identity differs from the resolved dims / text_trim

and, for existing checks the plan's table does not list:

    R9  merge flags inconsistent with the precision (merge_linear2 needs
        merge_qkv_mlp; the frontend derives merge_qkv_mlp from precision)
    R10 calibration_path given to a precision/option that does not use it
    R11 structure invariants the frontend asserts (HD == 128,
        action_attn_width == hidden)
    V1  a value outside its domain (unknown profile, precision, vae_encoder,
        consumer, expert key, or a non-bool switch)

`effective_config`: one line, `effective_config key=value ...`, in exactly
the key order and formatting of the line
`benchmarks/imagewam_e2e_official_compare.py` prints
(`EFFECTIVE_CONFIG_FIELDS`), so `scripts/imagewam_thor_matrix.sh:parse_log`
parses it unchanged. `use_fa4` may be unresolved at this point (see
`ImageWAMOptions.use_fa4`); it is printed as `auto`. `fa4_fallback_reason`
is a runtime value (FA4 can fall back to the cuBLAS chain at capture time),
so the resolver prints `None`; the frontend / compare script calls
`format_effective_config(options, use_fa4=fe.use_fa4, use_fa4_mot=...,
fa4_fallback_reason=fe.fa4_fallback_reason)` with what actually ran.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if __package__:
    from .precision import Precision
else:  # loaded by file path: there is no package, and its __init__ is heavy
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _Path

    def _load_leaf(name: str):
        key = f"_imagewam_{name}_standalone"
        mod = _sys.modules.get(key)
        if mod is None:
            spec = _ilu.spec_from_file_location(key, _Path(__file__).with_name(f"{name}.py"))
            mod = _ilu.module_from_spec(spec)
            _sys.modules[key] = mod  # dataclasses / enum need the module registered
            spec.loader.exec_module(mod)
        return mod

    Precision = _load_leaf("precision").Precision

if TYPE_CHECKING:
    from .structure import ImageWAMStructure
    from .workload import ImageWAMWorkload

VAE_ENCODERS = ("torch", "native")     # imagewam_thor._VAE_ENCODERS
CONSUMERS = ("infer", "abi", "native")
# `infer`: the frontend's own infer() (any option). `abi`: runtime_surface()
# / export_model_runtime(io="python"). `native`: pipeline_resources() / the
# native runtime / export_model_runtime(io="native").


class ConfigError(ValueError):
    """An illegal configuration. The message is `"<rule id>: <combination>"`
    and `.rule` is the id (`"R1"` ... `"R11"`, `"V1"`)."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"{rule}: {message}")
        self.rule = rule


@dataclass(frozen=True)
class ImageWAMOptions:
    """Every optimisation switch of the frontend, resolved.

    `use_fa4`: `None` means "resolve at construction", exactly what
    `ImageWAMTorchFrontendThor._resolve_use_fa4(None)` does today (False
    unless `FLASHRT_THOR_FA4=1`, then FA4 when the hardware/runtime allow);
    it depends on the machine that constructs the frontend, so it is not
    decided here. `True` / `False` are explicit.

    `vae_graph_input`: `(num_views, in_h, in_w)` when the VAE runs inside
    the CUDA graph (derived from the workload), else `None`.

    The last block is the expert tier: constructor-only switches that a
    deployment does not set.
    """
    precision: Precision
    text_trim: bool
    use_fa4: bool | None
    use_fa4_mot: bool
    vae_encoder: str
    vae_graph_input: tuple[int, int, int] | None
    nvfp4_awq: bool
    calibration_path: str | None
    # expert tier
    gemm_variant_autotune: bool
    gemm_runner: object | None
    awq_alpha: float
    awq_scope: str
    merge_qkv_mlp: bool
    merge_linear2: bool


@dataclass(frozen=True)
class ProfileSpec:
    """A named set of profile-controlled options (tier 2 of the plan).

    `vae_graph`: run the VAE inside the CUDA graph; the resolver then sets
    `vae_graph_input` from `workload.vae_graph_input()`.
    """
    name: str
    description: str
    precision: Precision
    text_trim: bool
    use_fa4: bool | None
    use_fa4_mot: bool
    vae_encoder: str
    vae_graph: bool
    nvfp4_awq: bool


PROFILES: dict[str, ProfileSpec] = {
    "default": ProfileSpec(
        name="default",
        description=(
            "Today's constructor defaults of ImageWAMTorchFrontendThor: nvfp4, no text trim, "
            "FA4 backbone as the frontend resolves it (opt-in through FLASHRT_THOR_FA4), no FA4 "
            "mot, torch VAE encoder outside the graph, no AWQ."),
        precision=Precision.NVFP4, text_trim=False, use_fa4=None, use_fa4_mot=False,
        vae_encoder="torch", vae_graph=False, nvfp4_awq=False),
    "fast": ProfileSpec(
        name="fast",
        description=(
            "PROVISIONAL. text_trim + FA4 backbone + FA4 mot + native VAE encoder inside the "
            "CUDA graph (vae_graph_input from the workload). Measured on Thor at 106.1 ms vs "
            "203.3 ms for the default configuration (THOR_STATUS_SUMMARY.md, nvfp4). The owner "
            "has not approved it as the default (plan.md 'Decisions pending', T4/T5): it stays "
            "opt-in by name. text_trim is refused with the native consumer (rule R5); the abi and "
            "infer consumers serve it. It needs ae_model_path (rule R3); use_fa4=True "
            "raises at construction when the FA4 runtime is missing (unlike the env-auto "
            "default)."),
        precision=Precision.NVFP4, text_trim=True, use_fa4=True, use_fa4_mot=True,
        vae_encoder="native", vae_graph=True, nvfp4_awq=False),
}

# Options an expert may override per call (`resolve_config(**expert)`).
# `vae_graph` (bool) is the profile's VAE-in-graph switch; the resolved
# `vae_graph_input` is derived from it and the workload, never typed.
EXPERT_KEYS = (
    "use_fa4", "use_fa4_mot", "text_trim", "vae_encoder", "vae_graph", "nvfp4_awq",
    "awq_alpha", "awq_scope", "gemm_variant_autotune", "gemm_runner",
    "merge_qkv_mlp", "merge_linear2",
)
_BOOL_KEYS = ("use_fa4_mot", "text_trim", "vae_graph", "nvfp4_awq", "gemm_variant_autotune",
              "merge_qkv_mlp", "merge_linear2")

# Key order of the `effective_config` line of
# benchmarks/imagewam_e2e_official_compare.py (pinned by the test).
EFFECTIVE_CONFIG_FIELDS = ("precision", "text_trim", "vae_encoder", "vae_graph", "use_fa4",
                           "use_fa4_mot", "fa4_fallback_reason", "calibration", "awq")


def format_effective_config(options: ImageWAMOptions, *, use_fa4: bool | None = None,
                            use_fa4_mot: bool | None = None,
                            fa4_fallback_reason: str | None = None) -> str:
    """The `effective_config` line for `options`.

    `use_fa4` / `use_fa4_mot`: the runtime-resolved values (the frontend's
    `fe.use_fa4` / `fe.use_fa4_mot`); `None` falls back to `options`, and an
    unresolved `options.use_fa4` prints `auto`. `fa4_fallback_reason`:
    `fe.fa4_fallback_reason` (`None` when FA4 did not fall back, and the
    resolver's own placeholder).
    """
    fa4 = options.use_fa4 if use_fa4 is None else use_fa4
    mot = options.use_fa4_mot if use_fa4_mot is None else use_fa4_mot
    values = {
        "precision": options.precision,
        "text_trim": options.text_trim,
        "vae_encoder": options.vae_encoder,
        "vae_graph": options.vae_graph_input is not None,
        "use_fa4": "auto" if fa4 is None else fa4,
        "use_fa4_mot": mot,
        "fa4_fallback_reason": fa4_fallback_reason,
        "calibration": options.calibration_path,
        "awq": options.nvfp4_awq,
    }
    return "effective_config " + " ".join(f"{k}={values[k]}" for k in EFFECTIVE_CONFIG_FIELDS)


@dataclass(frozen=True)
class ResolvedConfig:
    """`dims`: the frontend's dims dict for this workload and structure
    (workload/structure keys only). `options`: every switch, resolved.
    `effective_config`: the resolved line, see `format_effective_config`."""
    dims: dict
    options: ImageWAMOptions
    effective_config: str

    def frontend_dims(self) -> dict:
        """`dims` plus the two merge flags: what the frontend holds as
        `self.dims` after its own precision rule (and what a calibration
        file identity is computed from). The frontend derives
        `merge_qkv_mlp` itself and takes `merge_linear2` from
        `dims_override`, so this dict is a valid `dims_override`."""
        return dict(self.dims, merge_qkv_mlp=self.options.merge_qkv_mlp,
                    merge_linear2=self.options.merge_linear2)


# -- resolution ------------------------------------------------------------


def _bool(key: str, value) -> bool:
    if not isinstance(value, bool):
        raise ConfigError("V1", f"{key}={value!r} must be a bool")
    return value


def _dims(workload, structure, lay) -> dict:
    """`imagewam_thor._DEFAULT_DIMS` / `libero_dims.LIBERO_REAL_DIMS` keys.

    `action_dim` is not in `LIBERO_REAL_DIMS`: the frontend supplies it from
    `_DEFAULT_DIMS` (7), and it reads `dims["action_dim"]` (head output
    width, action normalisation, runtime surface). `merge_qkv_mlp`,
    `merge_linear2` and `fuse_res_norm` are also frontend-filled, see
    `ResolvedConfig.frontend_dims`.
    """
    return dict(
        hidden=structure.hidden, HD=structure.HD, NH=structure.NH, mlp_hidden=structure.mlp_hidden,
        joint_attention_dim=structure.joint_attention_dim,
        x0=lay.x0, a0=lay.a0,
        num_layers_double=structure.num_layers_double, num_layers_single=structure.num_layers_single,
        action_hidden_dim=structure.action_hidden_dim, action_attn_width=structure.action_attn_width,
        action_mlp_hidden=structure.action_mlp_hidden,
        action_dim=workload.action_dim,
        num_action=workload.action_horizon, total=lay.total,
        action_num_layers_double=structure.action_num_layers_double,
        action_num_layers_single=structure.action_num_layers_single,
        dt=lay.dt, num_denoise_steps=workload.num_steps,
        ref_h=lay.ref_h, ref_w=lay.ref_w, proprio_dim=workload.proprio_dim,
        shift=workload.shift, num_train_timesteps=workload.num_train_timesteps,
    )


def _check_calibration_identity(path: str, frontend_dims: dict, text_trim: bool) -> None:
    """R8. Same comparison as `ImageWAMCalibration.validate_for` (format
    version, `text_trim`, identity dims), minus the checkpoint hash: the
    resolver does not see the checkpoint file, so the frontend's own
    `validate_for` still checks that part. Only runs when the file exists;
    `calibration_file` (torch, safetensors) is imported here."""
    from flash_rt.models.imagewam.calibration_file import (
        SUPPORTED_VERSIONS, identity_dims, load_calibration,
    )

    try:
        cal = load_calibration(path)
    except Exception as e:  # unreadable / not a calibration file: name the path
        raise ConfigError("R8", f"cannot read calibration file {path}: {e}") from e
    if cal.version not in SUPPORTED_VERSIONS:
        raise ConfigError("R8", f"calibration file version {cal.version} not in {SUPPORTED_VERSIONS}")
    if bool(text_trim) != cal.text_trim:
        raise ConfigError(
            "R8", f"calibration file {path} was recorded with text_trim={cal.text_trim}, the "
                  f"configuration runs text_trim={bool(text_trim)} (the text and single-stream GEMM "
                  f"inputs differ)")
    want = identity_dims(frontend_dims)
    if want != cal.dims:
        diff = {k: (cal.dims.get(k), want.get(k)) for k in set(want) | set(cal.dims)
                if cal.dims.get(k) != want.get(k)}
        raise ConfigError("R8", f"calibration file {path} dims differ (file, configuration): {diff}")


def resolve_config(workload: "ImageWAMWorkload", structure: "ImageWAMStructure", *,
                   profile: str = "default", precision: str | Precision | None = None,
                   calibration_path: str | None = None, ae_model_path: str | None = None,
                   consumer: str = "infer", allow_placeholder_calibration: bool = False,
                   **expert: Any) -> ResolvedConfig:
    """Resolve `(workload, structure, profile, overrides)` to a `ResolvedConfig`.

    `precision`: overrides the profile's. `calibration_path`: static-FP8 /
    AWQ statistics file. `ae_model_path`: the FLUX.2 autoencoder (a real VAE
    encoder needs it, rule R3; it is not stored in the options).
    `consumer`: what the configuration will be used for (`CONSUMERS`); the
    native consumer describes one fixed graph (rules R5, R6), while the ABI
    consumer carries one graph per captured text length.
    `allow_placeholder_calibration`: permit a static-FP8 precision without a
    calibration file (the old constructor path logs a warning and uses
    N(0, 0.1) placeholder scales; this resolver refuses it by default,
    rule R1). `**expert`: option overrides, keys in `EXPERT_KEYS`.

    Raises `ConfigError` (a `ValueError`) whose message starts with the rule
    id.
    """
    # -- V1: domains ---------------------------------------------------------
    if profile not in PROFILES:
        raise ConfigError("V1", f"unknown profile {profile!r}; known: {tuple(PROFILES)}")
    if consumer not in CONSUMERS:
        raise ConfigError("V1", f"unknown consumer {consumer!r}; must be one of {CONSUMERS}")
    unknown = sorted(set(expert) - set(EXPERT_KEYS))
    if unknown:
        raise ConfigError("V1", f"unknown expert option(s) {unknown}; known: {EXPERT_KEYS}")
    spec = PROFILES[profile]
    try:
        prec = Precision(spec.precision if precision is None else precision)
    except ValueError:
        raise ConfigError("V1", f"precision={precision!r} -- must be one of "
                                f"{tuple(p.value for p in Precision)}") from None
    for key in _BOOL_KEYS:
        if key in expert:
            _bool(key, expert[key])
    if "use_fa4" in expert and expert["use_fa4"] is not None:
        _bool("use_fa4", expert["use_fa4"])
    vae_encoder = expert.get("vae_encoder", spec.vae_encoder)
    if vae_encoder not in VAE_ENCODERS:
        raise ConfigError("V1", f"vae_encoder={vae_encoder!r} -- must be one of {VAE_ENCODERS}")
    awq_scope = expert.get("awq_scope", "adaln+down")
    if not isinstance(awq_scope, str):
        raise ConfigError("V1", f"awq_scope={awq_scope!r} must be a str")

    # -- R7: workload layout against the structure ----------------------------
    try:
        lay = workload.layout(structure)
    except ValueError as e:
        raise ConfigError("R7", f"workload layout inconsistent with the structure: {e}") from e
    if lay.ref_h * lay.ref_w != lay.a0 - lay.x0:  # imagewam_thor.py: ref_h*ref_w == img_len
        raise ConfigError("R7", f"ref_h*ref_w ({lay.ref_h}*{lay.ref_w}={lay.ref_h * lay.ref_w}) must "
                                f"equal img_len (a0-x0={lay.a0 - lay.x0})")

    # -- R11: structure invariants the frontend asserts ------------------------
    if structure.action_attn_width != structure.hidden:
        raise ConfigError(
            "R11", f"action_attn_width ({structure.action_attn_width}) must equal hidden "
                   f"({structure.hidden}) -- required for mot_joint attention (both experts' Q/K/V "
                   f"must land in the same per-head geometry)")
    if structure.HD != 128:
        raise ConfigError(
            "R11", f"HD={structure.HD} -- real 4-axis RoPE (axes_dim=(32,32,32,32)) sums to a fixed "
                   f"128; HD is not a free parameter")

    text_trim = expert.get("text_trim", spec.text_trim)
    nvfp4_awq = expert.get("nvfp4_awq", spec.nvfp4_awq)
    use_fa4 = expert.get("use_fa4", spec.use_fa4)
    use_fa4_mot = expert.get("use_fa4_mot", spec.use_fa4_mot)
    vae_graph = expert.get("vae_graph", spec.vae_graph)
    gemm_variant_autotune = expert.get("gemm_variant_autotune", False)
    vae_graph_input = workload.vae_graph_input() if vae_graph else None

    # -- R2: switchable-tile autotune ------------------------------------------
    if gemm_variant_autotune and not prec.supports_tile_autotune:
        tuned = tuple(p.value for p in Precision if p.supports_tile_autotune)
        raise ConfigError("R2", f"gemm_variant_autotune=True applies to {tuned}, got precision={prec.value!r}")

    # -- R3: a real VAE needs the autoencoder ----------------------------------
    if vae_graph_input is not None and ae_model_path is None:
        raise ConfigError("R3", "vae_graph_input (VAE inside the CUDA graph) needs ae_model_path/flux2_src")
    if vae_encoder != "torch" and ae_model_path is None:
        raise ConfigError("R3", f"vae_encoder={vae_encoder!r} selects a real VAE encoder and needs "
                                f"ae_model_path/flux2_src")

    # -- R9: merge flags ---------------------------------------------------------
    # The frontend sets dims["merge_qkv_mlp"] from the precision (a
    # dims_override value is overwritten); merge_linear2 defaults to it.
    merge_qkv_mlp = expert.get("merge_qkv_mlp", prec.merge_qkv_mlp)
    if merge_qkv_mlp != prec.merge_qkv_mlp:
        raise ConfigError(
            "R9", f"merge_qkv_mlp={merge_qkv_mlp} is not selectable for precision={prec.value!r}: "
                  f"the frontend derives it from the precision (merged={prec.merge_qkv_mlp})")
    merge_linear2 = expert.get("merge_linear2", merge_qkv_mlp)
    if merge_linear2 and not merge_qkv_mlp:
        raise ConfigError(
            "R9", f"merge_linear2=True needs merge_qkv_mlp=True (precision={prec.value!r} keeps the "
                  f"split linear1 path)")

    # -- R4: AWQ ------------------------------------------------------------------
    if nvfp4_awq:
        if not prec.supports_awq:
            awq_precs = tuple(p.value for p in Precision if p.supports_awq)
            raise ConfigError("R4", f"nvfp4_awq applies to {awq_precs}, not precision={prec.value!r}")
        if calibration_path is None:
            raise ConfigError("R4", "nvfp4_awq needs calibration_path (per-channel activation statistics)")

    # -- R1: static FP8 needs real scales ---------------------------------------
    # `needs_calibration` is soft in the old constructor (a placeholder
    # N(0, 0.1) scale and a warning); this resolver is the strict path.
    if prec.needs_calibration and calibration_path is None and not allow_placeholder_calibration:
        raise ConfigError(
            "R1", f"precision={prec.value!r} needs calibration_path (static activation scales); pass "
                  f"allow_placeholder_calibration=True to accept the N(0, 0.1) placeholder scales")

    # -- R10: a calibration file nothing would use --------------------------------
    if calibration_path is not None and not prec.accepts_calibration_path(nvfp4_awq=nvfp4_awq):
        static = tuple(p.value for p in Precision if p.needs_calibration)
        raise ConfigError("R10", f"calibration_path is used by {static} and by nvfp4_awq, not "
                                 f"precision={prec.value!r}")

    options = ImageWAMOptions(
        precision=prec, text_trim=text_trim, use_fa4=use_fa4, use_fa4_mot=use_fa4_mot,
        vae_encoder=vae_encoder, vae_graph_input=vae_graph_input, nvfp4_awq=nvfp4_awq,
        calibration_path=calibration_path,
        gemm_variant_autotune=gemm_variant_autotune, gemm_runner=expert.get("gemm_runner"),
        awq_alpha=float(expert.get("awq_alpha", 0.5)), awq_scope=awq_scope,
        merge_qkv_mlp=merge_qkv_mlp, merge_linear2=merge_linear2)
    dims = _dims(workload, structure, lay)

    # -- R8: calibration identity (only when the file is there to read) -----------
    if calibration_path is not None and os.path.isfile(calibration_path):
        _check_calibration_identity(
            calibration_path, dict(dims, merge_qkv_mlp=merge_qkv_mlp, merge_linear2=merge_linear2),
            text_trim)

    # -- R5 / R6: consumers that describe one fixed graph -------------------------
    if consumer == "native" and text_trim:
        raise ConfigError(
            "R5", f"text_trim=True with consumer={consumer!r}: the native pipeline replays one graph at "
                  f"one context length and has no per-length variant table, while a trimmed frontend runs "
                  f"one graph per prompt length (ISSUE-080 condition 5). consumer='abi' serves a trimmed "
                  f"frontend: runtime_surface() and the export carry one graph per captured length, "
                  f"keyed by that length")
    if consumer == "native":
        # pipeline_resources() / export_model_runtime(io="native") refusals.
        if nvfp4_awq:
            raise ConfigError("R6", "the native pipeline has no AWQ input-scale fold; nvfp4_awq must be off")
        if not prec.supports_native_runtime:
            native = tuple(p.value for p in Precision if p.supports_native_runtime)
            raise ConfigError("R6", f"the native pipeline does not describe precision={prec.value!r} "
                                    f"(supported: {native})")
        # use_fa4=None is decided at construction; pipeline_resources() still refuses then.
        if use_fa4 is True or use_fa4_mot:
            raise ConfigError("R6", "the native pipeline has no FA4 attention; use_fa4/use_fa4_mot must be off")
        if vae_graph_input is not None:
            raise ConfigError("R6", "the native pipeline has no VAE stage; the VAE must run outside the "
                                    "graph (vae_graph_input=None)")

    return ResolvedConfig(dims=dims, options=options,
                          effective_config=format_effective_config(options))


__all__ = [
    "CONSUMERS", "EFFECTIVE_CONFIG_FIELDS", "EXPERT_KEYS", "PROFILES", "VAE_ENCODERS", "ConfigError",
    "ImageWAMOptions", "Precision", "ProfileSpec", "ResolvedConfig", "format_effective_config",
    "resolve_config",
]
