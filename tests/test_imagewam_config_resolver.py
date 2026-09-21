"""`resolve_config` (flash_rt/models/imagewam/config_resolver.py), CPU only.

Pins, against the frontend SOURCE (`ast`, no frontend import):

* every named profile trims, and the `native` profile is the native
  consumer's set: the served default's switches with FA4 pinned off;
* the profiles are the SERVED configuration while
  `ImageWAMTorchFrontendThor.__init__` keeps its historical untrimmed defaults,
  and the divergence between the served profiles and those defaults is exactly
  the `text_trim` switch (taken with it: the explicitly stated `use_fa4` of
  `native`);
* every `dims` key the frontend and pipeline read with a string subscript is
  either produced by the resolver or filled by the frontend itself;
* the `effective_config` line has the key order of the compare script's
  print and survives `scripts/imagewam_thor_matrix.sh:parse_log` (the shell
  function is extracted from the script and run with bash).

One legal and one illegal case per rule id. R8 uses a tiny synthetic
calibration file (`sites={}`; only the identity is read).
"""
from __future__ import annotations

import ast
import dataclasses
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from flash_rt.models.imagewam import config_resolver as cr
from flash_rt.models.imagewam.config_resolver import (
    EFFECTIVE_CONFIG_FIELDS, EXPERT_KEYS, PROFILES, ConfigError, ImageWAMOptions, Precision,
    format_effective_config, resolve_config,
)
from flash_rt.models.imagewam.libero_dims import LIBERO_REAL_DIMS
from flash_rt.models.imagewam.structure import VAE_PATCH_STRIDE, ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "flash_rt/frontends/torch/imagewam_thor.py"
PIPELINE_SRC = ROOT / "flash_rt/models/imagewam/pipeline_thor.py"
COMPARE_SRC = ROOT / "benchmarks/imagewam_e2e_official_compare.py"
MATRIX_SH = ROOT / "scripts/imagewam_thor_matrix.sh"
RESOLVER_SRC = ROOT / "flash_rt/models/imagewam/config_resolver.py"

_STRUCTURE_KEYS = ("hidden", "HD", "NH", "mlp_hidden", "joint_attention_dim", "num_layers_double",
                   "num_layers_single", "action_hidden_dim", "action_attn_width", "action_mlp_hidden",
                   "action_num_layers_double", "action_num_layers_single")
LIBERO_STRUCT = ImageWAMStructure(
    **{k: LIBERO_REAL_DIMS[k] for k in _STRUCTURE_KEYS},
    max_action_horizon=LIBERO_REAL_DIMS["num_action"], patch_stride=VAE_PATCH_STRIDE)
LIBERO = ImageWAMWorkload.libero()
AE = "/models/flux2/ae.safetensors"          # never opened: only its presence matters (R3)
CAL = "/nonexistent/calibration.safetensors"  # R8 reads a calibration file only if it exists


def resolve(**kw) -> cr.ResolvedConfig:
    return resolve_config(LIBERO, LIBERO_STRUCT, **kw)


def resolve_native(**kw) -> cr.ResolvedConfig:
    """The set a native consumer resolves by name: the served default's
    switches with `use_fa4` stated off (rule R6 refuses FA4 on that path)."""
    return resolve_config(LIBERO, LIBERO_STRUCT, profile="native", **kw)


def rule_of(**kw) -> str:
    with pytest.raises(ConfigError) as ei:
        resolve(**kw)
    assert str(ei.value).startswith(ei.value.rule + ":"), str(ei.value)
    return ei.value.rule


# -- source readers ----------------------------------------------------------


def _frontend_init_defaults() -> dict:
    tree = ast.parse(FRONTEND_SRC.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ImageWAMTorchFrontendThor")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    a = init.args
    return {arg.arg: ast.literal_eval(d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None}


def _subscript_keys(path: Path) -> set[str]:
    """String keys of `d["k"]`, `dims["k"]`, `self.dims["k"]`, `self._active_dims["k"]`."""
    names = {"d", "dims"}
    attrs = {"dims", "_active_dims", "active_dims"}
    keys = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            continue
        v = node.value
        if (isinstance(v, ast.Name) and v.id in names) or (isinstance(v, ast.Attribute) and v.attr in attrs):
            keys.add(node.slice.value)
    return keys


# -- profiles ----------------------------------------------------------------


def test_native_profile_is_the_native_consumers_set():
    """`native` is what a native caller resolves: every switch the
    constructor defaults to, except `text_trim` (the profiles carry the served
    configuration and trim) and `use_fa4` (stated off instead of left to the
    environment: rule R6 refuses FA4 there, and `False` keeps the
    `FLASHRT_THOR_FA4` opt-in from switching it on behind the caller's back)."""
    fd = _frontend_init_defaults()
    o = resolve_native().options
    assert o.precision == fd["precision"] == "nvfp4"
    assert o.text_trim is True and fd["text_trim"] is False
    assert o.text_trim_cache_size == fd["text_trim_cache_size"] == 32
    assert o.use_fa4 is False                    # the profile states it; the constructor leaves it None
    assert o.use_fa4_mot is fd["use_fa4_mot"] is False
    assert o.vae_encoder == fd["vae_encoder"] == "torch"
    assert o.vae_graph_input is fd["vae_graph_input"] is None
    assert o.nvfp4_awq is fd["nvfp4_awq"] is False
    assert o.calibration_path is fd["calibration_path"] is None
    assert o.gemm_variant_autotune is fd["gemm_variant_autotune"] is False
    assert o.gemm_runner is fd["gemm_runner"] is None
    assert o.awq_alpha == fd["awq_alpha"] == 0.5
    assert o.awq_scope == fd["awq_scope"] == "adaln+down"
    assert o.merge_qkv_mlp is True and o.merge_linear2 is True   # frontend: precision != "fp16_cutlass"
    # the native consumer accepts it: no R6
    assert resolve_native(consumer="native").options.text_trim is True


def test_the_served_default_is_the_native_set_with_the_auto_switches_left_open():
    """The profiles carry what is SERVED. `default` and `native` differ only in
    the switches the native consumer cannot carry: FA4 at both sites (auto in
    `default`, stated off in `native`) and, given an autoencoder, the VAE
    stage. Both trim. For the native consumer `default` resolves to exactly
    the `native` profile's options."""
    served, native = resolve().options, resolve_native().options
    assert served.text_trim is native.text_trim is True
    assert served.use_fa4 is None and served.use_fa4_mot is None
    assert native.use_fa4 is False and native.use_fa4_mot is False
    other = [f.name for f in dataclasses.fields(ImageWAMOptions)
             if f.name not in ("use_fa4", "use_fa4_mot")
             and getattr(served, f.name) != getattr(native, f.name)]
    assert other == [], f"the profiles differ in more than the FA4 sites: {other}"
    assert resolve(consumer="native").options == native
    assert resolve(consumer="native", ae_model_path=AE).options == native   # no VAE stage there


def test_the_default_is_the_fastest_configuration_the_inputs_allow():
    """Auto values: with an autoencoder the default runs the native VAE inside
    the graph (vae_graph_input from the workload); without one there is no VAE
    stage. Explicit values are never overridden."""
    o = resolve(ae_model_path=AE).options
    assert (o.text_trim, o.use_fa4, o.use_fa4_mot) == (True, None, None)
    assert o.vae_encoder == "native" and o.vae_graph_input == LIBERO.vae_graph_input() == (2, 224, 224)
    assert o.precision is Precision.NVFP4 and o.nvfp4_awq is False
    bare = resolve().options
    assert bare.vae_encoder == "torch" and bare.vae_graph_input is None
    # explicit values win over the auto ones
    x = resolve(ae_model_path=AE, vae_encoder="torch").options
    assert x.vae_encoder == "torch" and x.vae_graph_input is None
    x = resolve(ae_model_path=AE, vae_graph=False).options
    assert x.vae_encoder == "native" and x.vae_graph_input is None
    x = resolve(ae_model_path=AE, use_fa4=False, use_fa4_mot=False).options
    assert (x.use_fa4, x.use_fa4_mot) == (False, False)
    # ... and an explicit FA4 request the native consumer cannot carry still raises
    assert rule_of(use_fa4_mot=True, consumer="native") == "R6"
    assert rule_of(ae_model_path=AE, vae_graph=True, consumer="native") == "R6"


def test_the_profiles_diverge_from_the_constructor_defaults():
    """`ImageWAMTorchFrontendThor.__init__` keeps the historical defaults for a
    caller that passes dims and switches by hand (untrimmed, FA4 mot off, torch
    VAE outside the graph); the profiles carry the served configuration. The
    divergence is deliberate, so it is pinned rather than assumed."""
    fd = _frontend_init_defaults()
    assert fd["text_trim"] is False and fd["use_fa4"] is None and fd["use_fa4_mot"] is False
    assert fd["vae_encoder"] == "torch" and fd["vae_graph_input"] is None
    served = resolve().options
    assert served.text_trim is True
    assert served.use_fa4_mot is None                  # auto in the profile, off in the constructor
    # everything else the constructor defaults to, the served profile does too
    for key in ("precision", "text_trim_cache_size", "vae_encoder", "nvfp4_awq",
                "calibration_path", "gemm_variant_autotune", "gemm_runner", "awq_alpha", "awq_scope"):
        assert getattr(served, key) == fd[key], key
    assert served.vae_graph_input is fd["vae_graph_input"] is None   # no autoencoder given
    assert served.use_fa4 is fd["use_fa4"] is None    # still env-resolved, not stated
    assert served.merge_qkv_mlp is True and served.merge_linear2 is True


def test_merge_flags_follow_the_frontend_precision_rule():
    """`dims["merge_qkv_mlp"] = precision != "fp16_cutlass"`, merge_linear2 defaults to it."""
    for p in Precision:
        o = resolve(precision=p, allow_placeholder_calibration=True).options
        assert o.merge_qkv_mlp is o.merge_linear2 is (p != Precision.FP16_CUTLASS), p


def test_use_fa4_none_is_the_frontends_env_resolution():
    tree = ast.parse(FRONTEND_SRC.read_text())
    consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id in ("_FA4_OPT_IN_ENV", "_FA4_OPT_IN_DEFAULT")}
    # The served default: the environment variable unset means the machine's own
    # answer (`fa4_backend.thor_default_enabled()`); only an explicit "0" forces
    # the cuBLAS chain. This file reads the source; the behavioural pin is
    # tests/test_imagewam_frontend_from_config.py::test_use_fa4_none_resolves_on_the_machine.
    assert consts == {"_FA4_OPT_IN_ENV": "FLASHRT_THOR_FA4", "_FA4_OPT_IN_DEFAULT": "1"}
    assert "def _resolve_use_fa4" in FRONTEND_SRC.read_text()


def test_profile_table_and_fast_contents():
    assert tuple(PROFILES) == ("default", "fast", "native")
    assert cr.CONSUMERS == ("infer", "abi", "native")
    assert PROFILES["default"].precision == "nvfp4"
    assert PROFILES["default"].text_trim is True
    assert (PROFILES["default"].use_fa4, PROFILES["default"].use_fa4_mot) == (None, None)
    assert (PROFILES["default"].vae_encoder, PROFILES["default"].vae_graph) == ("auto", None)
    assert PROFILES["native"].precision == "nvfp4"
    assert PROFILES["native"].text_trim is True and PROFILES["native"].use_fa4 is False
    fast = resolve(profile="fast", ae_model_path=AE).options
    assert fast.precision == "nvfp4"
    assert fast.text_trim is True
    assert fast.use_fa4 is True and fast.use_fa4_mot is True
    assert fast.vae_encoder == "native"
    assert fast.vae_graph_input == LIBERO.vae_graph_input() == (2, 224, 224)
    assert fast.nvfp4_awq is False
    # `fast` is `default` stated explicitly: with an autoencoder the two differ only
    # in how the FA4 sites are stated (auto vs True)
    d = resolve(ae_model_path=AE).options
    assert [f.name for f in dataclasses.fields(ImageWAMOptions)
            if getattr(d, f.name) != getattr(fast, f.name)] == ["use_fa4", "use_fa4_mot"]
    assert "raises" in PROFILES["fast"].description
    # the native profile names the consumer it is for, and what it states off
    ndesc = PROFILES["native"].description
    assert "native" in ndesc and "FA4" in ndesc and 'consumer="native"' in ndesc
    assert resolve_config.__kwdefaults__["profile"] == "default"


def test_default_profile_resolves_trimmed_for_the_serving_consumers():
    """The served default: text_trim on, and the two consumers that carry one
    graph per length serve it. `fast` is unchanged by this (it already
    trimmed)."""
    served = resolve().options
    assert served.text_trim is True and served.use_fa4 is None
    assert served.vae_encoder == "torch" and served.vae_graph_input is None
    assert resolve(consumer="infer").options.text_trim is True
    assert resolve(consumer="abi").options.text_trim is True
    fast = resolve(profile="fast", ae_model_path=AE, consumer="abi").options
    assert (fast.text_trim, fast.use_fa4, fast.use_fa4_mot) == (True, True, True)
    assert fast.vae_encoder == "native" and fast.vae_graph_input == (2, 224, 224)


def test_fast_vae_graph_input_follows_the_workload():
    w = dataclasses.replace(LIBERO, num_views=3, image_h=256, image_w=192)
    r = resolve_config(w, LIBERO_STRUCT, profile="fast", ae_model_path=AE)
    assert r.options.vae_graph_input == (3, 256, 192)
    assert (r.dims["ref_h"], r.dims["ref_w"]) == (16, 36)


def test_precision_override():
    assert resolve(precision="fp16").options.precision is Precision.FP16
    assert resolve(precision=Precision.FP8).options.precision is Precision.FP8
    fast = resolve(profile="fast", precision="fp16", ae_model_path=AE)
    assert fast.options.precision is Precision.FP16 and fast.options.text_trim is True
    assert "precision=fp16 " in fast.effective_config


def test_expert_override_and_unknown_key():
    o = resolve(use_fa4=True, use_fa4_mot=True, gemm_variant_autotune=True, awq_alpha=0.7,
                awq_scope="all", merge_linear2=False, gemm_runner=object).options
    assert (o.use_fa4, o.use_fa4_mot, o.gemm_variant_autotune) == (True, True, True)
    assert (o.awq_alpha, o.awq_scope, o.merge_linear2, o.gemm_runner) == (0.7, "all", False, object)
    f = resolve(profile="fast", ae_model_path=AE, text_trim=False, vae_encoder="torch", vae_graph=False,
                use_fa4=None).options
    assert f.text_trim is False and f.vae_encoder == "torch" and f.vae_graph_input is None
    assert f.use_fa4 is None
    with pytest.raises(ConfigError, match=r"^V1: .*'use_fa5'") as ei:
        resolve(use_fa5=True)
    assert ei.value.rule == "V1"
    for key in ("use_fa4", "text_trim", "gemm_runner", "merge_qkv_mlp"):
        assert key in EXPERT_KEYS
    assert set(EXPERT_KEYS) >= {f.name for f in dataclasses.fields(ImageWAMOptions)
                                if f.name not in ("precision", "calibration_path", "vae_graph_input")}


def test_options_are_frozen():
    o = resolve().options
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.text_trim = False


# -- dims ----------------------------------------------------------------------


def test_libero_dims_equal_libero_real_dims():
    """`LIBERO_REAL_DIMS` is the resolver's own mapping for the LIBERO
    workload and structure (`libero_dims.py`), so the two are the same key
    set with the same values; nothing is added on either side."""
    r = resolve()
    assert r.dims == LIBERO_REAL_DIMS
    assert set(r.dims) == set(LIBERO_REAL_DIMS)
    assert r.dims["action_dim"] == 7
    assert r.dims["dt"] == LIBERO_REAL_DIMS["dt"] and r.dims["shift"] == LIBERO_REAL_DIMS["shift"]


def test_every_frontend_dims_key_is_resolved_or_frontend_filled():
    """Required string-subscript keys of the frontend and pipeline == resolver
    dims plus the flags the frontend fills itself (`merge_qkv_mlp`,
    `merge_linear2`; `fuse_res_norm` is `dims.setdefault(..., True)`)."""
    consumed = _subscript_keys(FRONTEND_SRC) | _subscript_keys(PIPELINE_SRC)
    frontend_filled = {"merge_qkv_mlp", "merge_linear2", "fuse_res_norm"}
    missing = consumed - set(resolve().dims) - frontend_filled
    assert not missing, f"frontend reads dims keys the resolver does not produce: {sorted(missing)}"
    # and the canonical LIBERO dims carry every one of those keys
    assert (consumed - frontend_filled) - set(LIBERO_REAL_DIMS) == set()


def test_frontend_dims_adds_merge_flags():
    r = resolve(merge_linear2=False)
    assert r.frontend_dims() == dict(r.dims, merge_qkv_mlp=True, merge_linear2=False)
    assert "merge_linear2" not in r.dims


def test_dims_from_the_real_checkpoint_structure():
    ckpt = os.environ.get("CKPT_PATH", "/home/ljw/projects/pi0.5/models/imagewam_flux2_4b_libero/model.pt")
    if not Path(ckpt).is_file():
        pytest.skip(f"no ImageWAM checkpoint at {ckpt}")
    r = resolve_config(LIBERO, ImageWAMStructure.from_checkpoint(ckpt))
    assert r.dims == dict(LIBERO_REAL_DIMS, action_dim=7)


# -- rules: one legal and one illegal case each ---------------------------------


def test_R1_static_fp8_needs_calibration():
    assert rule_of(precision="fp8_static") == "R1"
    assert rule_of(precision="fp8_static_cutlass") == "R1"
    assert resolve(precision="fp8_static", calibration_path=CAL).options.calibration_path == CAL
    ph = resolve(precision="fp8_static", allow_placeholder_calibration=True)
    assert ph.options.calibration_path is None
    resolve(precision="fp16")  # other precisions need none
    resolve(precision="nvfp4")


def test_R2_gemm_variant_autotune_needs_a_tiled_precision():
    assert rule_of(precision="fp16", gemm_variant_autotune=True) == "R2"
    assert rule_of(precision="fp8", gemm_variant_autotune=True) == "R2"
    assert resolve(precision="nvfp4", gemm_variant_autotune=True).options.gemm_variant_autotune
    assert resolve(precision="fp8_static_cutlass", calibration_path=CAL,
                   gemm_variant_autotune=True).options.gemm_variant_autotune


def test_R3_real_vae_needs_the_autoencoder():
    assert rule_of(profile="fast") == "R3"                       # native + VAE in graph
    assert rule_of(vae_encoder="native", vae_graph=False) == "R3"  # real encoder outside the graph
    assert rule_of(vae_graph=True) == "R3"                       # torch encoder in the graph
    assert resolve(profile="fast", ae_model_path=AE).options.vae_graph_input == (2, 224, 224)
    # native encoder with an autoencoder: inside the graph unless vae_graph=False says otherwise
    assert resolve(vae_encoder="native", ae_model_path=AE).options.vae_graph_input == (2, 224, 224)
    assert resolve(vae_encoder="native", vae_graph=False, ae_model_path=AE).options.vae_graph_input is None
    resolve()                                                    # torch encoder outside the graph: no AE needed


def test_R4_awq_needs_an_awq_precision_and_a_calibration_file():
    assert rule_of(precision="fp16", nvfp4_awq=True, calibration_path=CAL) == "R4"
    assert rule_of(precision="fp8_static_cutlass", nvfp4_awq=True, calibration_path=CAL) == "R4"
    assert rule_of(nvfp4_awq=True) == "R4"                       # nvfp4, no calibration
    assert resolve(nvfp4_awq=True, calibration_path=CAL).options.nvfp4_awq
    assert resolve(precision="nvfp4_sim", nvfp4_awq=True, calibration_path=CAL).options.nvfp4_awq


def test_text_trim_is_legal_for_every_consumer():
    """All three consumers carry one graph per captured text length — the
    Python `infer()` path, the ABI face's declaration and the native C++
    pipeline (`pipeline_resources()` describes the active length and the
    handle installs one pipeline per length) — so no combination of
    `text_trim=True` with a consumer is refused. R6 is all that remains
    specific to the native consumer."""
    for consumer in cr.CONSUMERS:
        assert resolve(text_trim=True, consumer=consumer).options.text_trim, consumer
        assert resolve(consumer=consumer).options.text_trim, consumer           # the served default
        assert resolve_native(consumer=consumer).options.text_trim, consumer     # the native set
    fast = resolve(profile="fast", ae_model_path=AE)
    assert fast.options.text_trim
    for consumer in ("infer", "abi"):
        assert resolve_config(LIBERO, LIBERO_STRUCT, profile="fast", ae_model_path=AE,
                              consumer=consumer).options.text_trim, consumer
    # `fast`'s FA4 is what the native consumer is refused for, not its trim
    with pytest.raises(ConfigError, match=r"^R6: ") as ei:
        resolve_config(LIBERO, LIBERO_STRUCT, profile="fast", ae_model_path=AE, consumer="native")
    assert "FA4" in str(ei.value), str(ei.value)


def test_R6_native_consumer_limits():
    # the native consumer's own set resolves, so R6 is what is reached
    assert rule_of(profile="native", nvfp4_awq=True, calibration_path=CAL, consumer="native") == "R6"
    for p in ("fp16_cutlass", "fp8", "e0m3_hadamard", "nvfp4_sim"):
        assert rule_of(profile="native", precision=p, consumer="native") == "R6", p
    assert rule_of(profile="native", use_fa4=True, consumer="native") == "R6"
    assert rule_of(profile="native", use_fa4_mot=True, consumer="native") == "R6"
    assert rule_of(profile="native", vae_encoder="native", vae_graph=True, ae_model_path=AE,
                   consumer="native") == "R6"
    assert resolve_native(precision="nvfp4", consumer="native").options.precision is Precision.NVFP4
    assert resolve_native(precision="fp16", consumer="native").options.precision is Precision.FP16
    # the same options are fine for the ABI (python io) and for infer()
    assert resolve(nvfp4_awq=True, calibration_path=CAL, consumer="abi").options.nvfp4_awq
    assert resolve(precision="fp8", consumer="abi").options.precision is Precision.FP8
    assert resolve_native(use_fa4=False, use_fa4_mot=False, consumer="native").options.use_fa4 is False
    assert resolve_native(consumer="native").options.use_fa4 is False   # the profile states it off
    assert resolve().options.use_fa4 is None                           # the served default leaves it to the env
    # the trim is not what the native consumer is refused for: the served
    # default resolves for it, and FA4 on top of it is the R6 case
    assert resolve(consumer="native").options.text_trim is True
    assert rule_of(use_fa4=True, consumer="native") == "R6"


def test_R7_layout_inconsistent_with_the_structure():
    assert rule_of_workload(dataclasses.replace(LIBERO, image_w=225)) == "R7"          # patch stride
    assert rule_of_workload(dataclasses.replace(LIBERO, image_h=100)) == "R7"
    assert rule_of_workload(dataclasses.replace(LIBERO, action_horizon=65)) == "R7"    # > max_action_horizon
    with pytest.raises(ConfigError, match=r"^R7: .*max_action_horizon"):
        resolve_config(LIBERO, ImageWAMStructure.toy())                                # toy horizon is 4
    assert resolve_config(dataclasses.replace(LIBERO, action_horizon=64), LIBERO_STRUCT).dims["num_action"] == 64


def rule_of_workload(workload) -> str:
    with pytest.raises(ConfigError) as ei:
        resolve_config(workload, LIBERO_STRUCT)
    return ei.value.rule


def _write_calibration(path: str, dims: dict, *, text_trim: bool) -> None:
    pytest.importorskip("safetensors")
    from flash_rt.models.imagewam.calibration_file import (
        FORMAT_VERSION, ImageWAMCalibration, identity_dims, save_calibration,
    )
    save_calibration(ImageWAMCalibration(
        version=FORMAT_VERSION, checkpoint_id="0" * 16, checkpoint_size=1, dims=identity_dims(dims),
        percentile=99.9, frames=[], noise="test", sites={}, text_trim=text_trim), path)


def test_R8_calibration_identity(tmp_path):
    ok = str(tmp_path / "ok.safetensors")
    base = resolve(precision="fp8_static", calibration_path=CAL)
    _write_calibration(ok, base.frontend_dims(), text_trim=True)
    r = resolve(precision="fp8_static", calibration_path=ok)               # legal: identity matches
    assert r.options.calibration_path == ok

    # text_trim mismatch: file recorded trimmed, configuration untrimmed
    with pytest.raises(ConfigError, match=r"^R8: .*text_trim") as ei:
        resolve(precision="fp8_static", calibration_path=ok, text_trim=False)
    assert ei.value.rule == "R8"

    # dims mismatch: a file recorded for another workload (different text length -> x0/a0)
    other = str(tmp_path / "other.safetensors")
    w = dataclasses.replace(LIBERO, text_max_len=256)
    _write_calibration(other, resolve_config(w, LIBERO_STRUCT, precision="fp8_static",
                                             calibration_path=CAL).frontend_dims(), text_trim=True)
    with pytest.raises(ConfigError, match=r"^R8: .*dims differ.*'x0'") as ei:
        resolve(precision="fp8_static", calibration_path=other)
    assert ei.value.rule == "R8"

    # merge flags are part of the identity: a file recorded with the split linear2
    split = str(tmp_path / "split.safetensors")
    _write_calibration(split, resolve(precision="fp8_static", calibration_path=CAL, merge_linear2=False)
                       .frontend_dims(), text_trim=True)
    with pytest.raises(ConfigError, match=r"^R8: .*merge_linear2"):
        resolve(precision="fp8_static", calibration_path=split)
    assert resolve(precision="fp8_static", calibration_path=split, merge_linear2=False)

    # a file that is not a calibration file
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    with pytest.raises(ConfigError, match=r"^R8: cannot read"):
        resolve(precision="fp8_static", calibration_path=str(junk))

    # AWQ reads the same file: same identity check
    with pytest.raises(ConfigError, match=r"^R8: "):
        resolve(nvfp4_awq=True, calibration_path=other)
    assert resolve(nvfp4_awq=True, calibration_path=ok).options.nvfp4_awq


def test_R9_merge_flags_follow_the_precision():
    assert rule_of(precision="fp16_cutlass", merge_linear2=True) == "R9"
    assert rule_of(precision="nvfp4", merge_qkv_mlp=False) == "R9"
    assert rule_of(precision="fp16_cutlass", merge_qkv_mlp=True) == "R9"
    assert not resolve(precision="nvfp4", merge_linear2=False).options.merge_linear2      # A/B split linear2
    o = resolve(precision="fp16_cutlass").options
    assert (o.merge_qkv_mlp, o.merge_linear2) == (False, False)
    assert resolve(precision="fp16_cutlass", merge_linear2=False).options.merge_linear2 is False


def test_R10_calibration_path_must_be_used():
    assert rule_of(precision="fp16", calibration_path=CAL) == "R10"
    assert rule_of(calibration_path=CAL) == "R10"                # nvfp4 without AWQ
    assert resolve(precision="fp8_static", calibration_path=CAL).options.calibration_path == CAL
    assert resolve(calibration_path=CAL, nvfp4_awq=True).options.calibration_path == CAL


def test_R11_structure_invariants():
    with pytest.raises(ConfigError, match=r"^R11: .*HD=64"):
        resolve_config(LIBERO, dataclasses.replace(LIBERO_STRUCT, HD=64))
    with pytest.raises(ConfigError, match=r"^R11: action_attn_width"):
        resolve_config(LIBERO, dataclasses.replace(LIBERO_STRUCT, action_attn_width=2048))
    assert resolve_config(LIBERO, LIBERO_STRUCT)
    toy = ImageWAMStructure.toy()
    assert resolve_config(dataclasses.replace(LIBERO, action_horizon=4), toy).dims["hidden"] == toy.hidden


def test_V1_value_domains():
    for kw in (dict(profile="turbo"), dict(precision="int3"), dict(consumer="cloud"),
               dict(vae_encoder="onnx"), dict(text_trim=1), dict(use_fa4="yes"), dict(awq_scope=3),
               dict(text_trim_cache_size=0), dict(text_trim_cache_size="32")):
        assert rule_of(**kw) == "V1", kw


# -- effective_config --------------------------------------------------------------


def _compare_effective_config_call() -> ast.Call:
    """The compare script's `effective_config` print. Since W12 the script
    calls the resolver's own `format_effective_config` on the resolved
    configuration, with the runtime-resolved FA4 values, so the line it prints
    cannot drift from `EFFECTIVE_CONFIG_FIELDS`."""
    for node in ast.walk(ast.parse(COMPARE_SRC.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "format_effective_config"):
            return node
    raise AssertionError("the compare script does not print format_effective_config(...)")


def test_effective_config_keeps_the_compare_scripts_format():
    call = _compare_effective_config_call()
    assert isinstance(call.args[0], ast.Attribute) and call.args[0].attr == "options", ast.dump(call)
    kwargs = {k.arg for k in call.keywords}
    assert {"use_fa4", "use_fa4_mot", "fa4_fallback_reason"} <= kwargs, sorted(kwargs)
    # the key order of the one line every log carries, pinned literally: the
    # matrix script's parse_log and the ABI identity both read this order
    assert tuple(EFFECTIVE_CONFIG_FIELDS) == ("precision", "text_trim", "vae_encoder", "vae_graph",
                                              "use_fa4", "use_fa4_mot", "fa4_fallback_reason",
                                              "calibration", "awq")
    line = resolve().effective_config
    assert "\n" not in line
    assert line == ("effective_config precision=nvfp4 text_trim=True vae_encoder=torch vae_graph=False "
                    "use_fa4=auto use_fa4_mot=auto fa4_fallback_reason=None calibration=None awq=False")
    assert resolve(ae_model_path=AE).effective_config == (
        "effective_config precision=nvfp4 text_trim=True vae_encoder=native vae_graph=True "
        "use_fa4=auto use_fa4_mot=auto fa4_fallback_reason=None calibration=None awq=False")
    assert resolve(profile="native").effective_config == (
        "effective_config precision=nvfp4 text_trim=True vae_encoder=torch vae_graph=False "
        "use_fa4=False use_fa4_mot=False fa4_fallback_reason=None calibration=None awq=False")


def test_effective_config_takes_runtime_values():
    o = resolve(profile="fast", ae_model_path=AE, precision="fp16").options
    assert format_effective_config(o).startswith(
        "effective_config precision=fp16 text_trim=True vae_encoder=native vae_graph=True use_fa4=True "
        "use_fa4_mot=True fa4_fallback_reason=None calibration=None awq=False")
    ran = format_effective_config(o, use_fa4=False, use_fa4_mot=False, fa4_fallback_reason="capture failed")
    assert " use_fa4=False use_fa4_mot=False fa4_fallback_reason=capture failed calibration=None " in ran
    cal = resolve(nvfp4_awq=True, calibration_path=CAL)
    assert cal.effective_config.endswith(f"calibration={CAL} awq=True")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_effective_config_round_trips_through_parse_log(tmp_path):
    src = MATRIX_SH.read_text()
    m = re.search(r"^parse_log\(\) \{\n.*?^\}\n", src, re.S | re.M)
    assert m, "parse_log() not found in scripts/imagewam_thor_matrix.sh"
    fn = tmp_path / "parse_log.sh"
    fn.write_text(m.group(0))

    def parse(effective: str) -> list[str]:
        log = tmp_path / "run.log"
        log.write_text("\n".join([
            "fr_vs_off              min=0.99887 median=0.99933 mean=0.99930",
            "mae_fr_vs_gt           min=0.10000 median=0.18600 mean=0.19000",
            "infer() P50=106.1ms min=105.0ms (shared GPU, not a perf number)",
            effective, "peak GPU mem: 12.3 GiB", ""]))
        out = subprocess.run(["bash", "-c", 'source "$1"; parse_log "$2"', "_", str(fn), str(log)],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out.split(",")

    o = resolve(profile="fast", ae_model_path=AE).options
    runtime = format_effective_config(o, use_fa4=True, use_fa4_mot=True)
    assert parse(runtime) == ["0.99887", "0.99933", "0.18600", "106.1", "True", "True", "None"]
    fell = format_effective_config(o, use_fa4=False, use_fa4_mot=True,
                                   fa4_fallback_reason="FA4 failed, using cuBLAS")
    assert parse(fell) == ["0.99887", "0.99933", "0.18600", "106.1", "False", "True", "FA4 failed; using cuBLAS"]
    # the resolver's own (pre-construction) line parses too
    assert parse(resolve().effective_config)[3:] == ["106.1", "auto", "auto", "None"]


# -- import contract -----------------------------------------------------------------


def test_config_resolver_loads_by_path_without_torch():
    """No package, no torch, no compiled kernels: `precision.py` and the
    workload are loaded by path and `resolve_config` runs."""
    prog = textwrap.dedent(f"""
        import importlib.util, sys
        sys.modules["torch"] = None                    # `import torch` now raises ImportError
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
        cr = load("cr_by_path", {str(RESOLVER_SRC)!r})
        wl = load("wl_by_path", {str(ROOT / 'flash_rt/models/imagewam/workload.py')!r})
        from collections import namedtuple
        S = namedtuple("S", {" ".join(_STRUCTURE_KEYS) + " max_action_horizon patch_stride"!r})
        s = S(**{{k: v for k, v in zip(S._fields, {[LIBERO_REAL_DIMS[k] for k in _STRUCTURE_KEYS] + [64, 16]!r})}})
        r = cr.resolve_config(wl.ImageWAMWorkload.libero(), s)
        assert r.dims["x0"] == 513 and r.options.precision == "nvfp4", r
        try:
            cr.resolve_config(wl.ImageWAMWorkload.libero(), s, precision="fp8_static")
        except cr.ConfigError as e:
            assert str(e).startswith("R1: "), e
        else:
            raise AssertionError("R1 not raised")
        assert sys.modules["torch"] is None and "flash_rt" not in sys.modules
        print("ok")
    """)
    res = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0 and res.stdout.strip() == "ok", res.stdout + res.stderr
