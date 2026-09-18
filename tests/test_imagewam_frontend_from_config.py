"""`ImageWAMTorchFrontendThor.from_config` and `frontend_kwargs_from_config`.

CPU only: no CUDA, no compiled extension, no frontend construction. The
resolver (`resolve_config`) and the mapping
(`frontend_kwargs_from_config`, `imagewam_thor.py`) are pure data
transformations, so this file pins what a `ResolvedConfig` turns into
without a GPU:

* the mapping names exactly the constructor's own keywords, one entry per
  `ImageWAMOptions` field, and the two merge flags land in `dims_override`
  (the constructor has no `merge_*` keyword: it derives `merge_qkv_mlp`
  from the precision and takes `merge_linear2` from `dims_override`);
* `dims_override` carries the LIBERO layout the resolver derived
  (`x0=513`, `a0=905`, `total=969`, `ref_h=14`, `ref_w=28`,
  `num_action=64`, `dt=0.1`);
* `vae_graph_input` and `use_fa4_mot` are the resolver's derived values,
  not new switches;
* an illegal combination raises `ConfigError` before the constructor is
  reached;
* `from_config` is that mapping and nothing else (its `__init__` call is
  captured with `unittest.mock`, which keeps the classmethod covered
  without a GPU).

The structure below is the literal `ImageWAMStructure` for the real
`ImageWAM-FLUX.2-4B-LIBERO` release, not `ImageWAMStructure.libero()`, so
this file does not depend on that constant table. `ImageWAMWorkload.libero()`
is the workload the resolver derives the layout from.
"""
from __future__ import annotations

import inspect
from dataclasses import fields
from unittest import mock

import pytest
import torch

from flash_rt.frontends.torch.imagewam_thor import (
    ImageWAMTorchFrontendThor,
    frontend_kwargs_from_config,
)
from flash_rt.models.imagewam.config_resolver import (
    ConfigError,
    ImageWAMOptions,
    ResolvedConfig,
    resolve_config,
)
from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload

LIBERO_STRUCTURE = ImageWAMStructure(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    action_num_layers_double=5, action_num_layers_single=20,
    max_action_horizon=64, patch_stride=16,
)
LIBERO_WORKLOAD = ImageWAMWorkload.libero()
AE = "/models/flux2/ae.safetensors"            # never opened: R3 checks presence only
FLUX2 = "/models/flux2/src"
CALIBRATION = "/nonexistent/calibration.safetensors"   # R8 reads a file only if it exists

# Constructor parameters the resolved path never sets: `self` and `**kwargs`
# are not arguments, and `checkpoint_dir` is the legacy positional parameter
# the constructor accepts for interface parity and deletes (`del
# checkpoint_dir, kwargs`).
NOT_A_RESOLVED_ARGUMENT = ("self", "checkpoint_dir")

# Where every `ImageWAMOptions` field lands in the constructor call. The two
# merge flags are `dims_override` entries (`ResolvedConfig.frontend_dims`),
# not keywords of their own.
OPTION_TARGETS: dict[str, str] = {
    "precision": "precision",
    "text_trim": "text_trim",
    "use_fa4": "use_fa4",
    "use_fa4_mot": "use_fa4_mot",
    "vae_encoder": "vae_encoder",
    "vae_graph_input": "vae_graph_input",
    "nvfp4_awq": "nvfp4_awq",
    "calibration_path": "calibration_path",
    "gemm_variant_autotune": "gemm_variant_autotune",
    "gemm_runner": "gemm_runner",
    "awq_alpha": "awq_alpha",
    "awq_scope": "awq_scope",
    "merge_qkv_mlp": "dims_override.merge_qkv_mlp",
    "merge_linear2": "dims_override.merge_linear2",
}

# Constructor keyword names a `ResolvedConfig` may set.
CONSTRUCTOR_KEYWORDS = tuple(
    name for name, parameter in
    inspect.signature(ImageWAMTorchFrontendThor.__init__).parameters.items()
    if name not in NOT_A_RESOLVED_ARGUMENT
    and parameter.kind is not inspect.Parameter.VAR_KEYWORD)


def resolved(**kw) -> ResolvedConfig:
    return resolve_config(LIBERO_WORKLOAD, LIBERO_STRUCTURE, **kw)


def fast_resolved() -> ResolvedConfig:
    return resolved(profile="fast", ae_model_path=AE)


# -- the mapping names the constructor's keywords ----------------------------


def test_mapping_names_exactly_the_constructor_keywords():
    """One entry per constructor keyword, and nothing the constructor does
    not accept: the mapping cannot drift from `__init__`."""
    kwargs = frontend_kwargs_from_config(resolved())
    assert set(kwargs) == set(CONSTRUCTOR_KEYWORDS)
    assert "dims_override" in kwargs


def test_every_option_field_has_a_target():
    assert set(OPTION_TARGETS) == {f.name for f in fields(ImageWAMOptions)}


@pytest.mark.parametrize("field_name", sorted(OPTION_TARGETS))
def test_option_field_reaches_its_target(field_name):
    """The value under the target is the option's own value (identity for
    `gemm_runner`, `None` for an unset path)."""
    resolution = resolved()
    options = resolution.options
    kwargs = frontend_kwargs_from_config(resolution)
    where = OPTION_TARGETS[field_name]
    got = (kwargs["dims_override"][where.split(".", 1)[1]] if where.startswith("dims_override.")
           else kwargs[where])
    expected = getattr(options, field_name)
    if field_name == "precision":
        # a plain string, not the `Precision` member: the constructor's own
        # `precision` domain (`_PRECISIONS`) is a tuple of strings
        assert type(got) is str
        assert got == str(expected) == "nvfp4"
    elif field_name == "gemm_runner":
        assert got is expected
    else:
        assert got == expected


def test_default_profile_maps_to_the_constructor_defaults():
    """The `default` profile through the mapping equals `__init__`'s own
    defaults (the resolved path and the legacy path agree)."""
    kwargs = frontend_kwargs_from_config(resolved())
    defaults = {name: parameter.default for name, parameter in
                inspect.signature(ImageWAMTorchFrontendThor.__init__).parameters.items()
                if name not in NOT_A_RESOLVED_ARGUMENT
                and parameter.kind is not inspect.Parameter.VAR_KEYWORD}
    assert kwargs["precision"] == defaults["precision"] == "nvfp4"
    assert kwargs["text_trim"] is defaults["text_trim"] is False
    assert kwargs["use_fa4"] is defaults["use_fa4"] is None   # resolved at construction
    assert kwargs["use_fa4_mot"] is defaults["use_fa4_mot"] is False
    assert kwargs["vae_encoder"] == defaults["vae_encoder"] == "torch"
    assert kwargs["vae_graph_input"] is defaults["vae_graph_input"] is None
    assert kwargs["nvfp4_awq"] is defaults["nvfp4_awq"] is False
    assert kwargs["calibration_path"] is defaults["calibration_path"] is None
    assert kwargs["awq_alpha"] == defaults["awq_alpha"] == 0.5
    assert kwargs["awq_scope"] == defaults["awq_scope"] == "adaln+down"
    assert kwargs["gemm_variant_autotune"] is defaults["gemm_variant_autotune"] is False
    assert kwargs["gemm_runner"] is defaults["gemm_runner"] is None
    assert kwargs["vae_resize"] == defaults["vae_resize"] == "area"


def test_path_arguments_are_passed_through():
    """The arguments the resolver never sees arrive unchanged, and their
    defaults are the constructor's."""
    paths = dict(ckpt_path="/ckpt/model.pt", ae_model_path=AE, flux2_src=FLUX2,
                 qwen3_model_spec="Qwen/Qwen3-4B", dataset_stats_path="/stats/dataset_stats.json",
                 vae_resize="pil_bilinear")
    kwargs = frontend_kwargs_from_config(resolved(), **paths)
    for name, value in paths.items():
        assert kwargs[name] == value
    assert {name: kwargs[name] for name in paths} == paths
    default_kwargs = frontend_kwargs_from_config(resolved())
    assert {name: default_kwargs[name] for name in paths} == {
        "ckpt_path": None, "ae_model_path": None, "flux2_src": None,
        "qwen3_model_spec": None, "dataset_stats_path": None, "vae_resize": "area"}


def test_calibration_path_is_the_resolvers():
    """`calibration_path` is not a path argument of this function: it comes
    from `resolved.options` (rule R1/R10 checked it), so a static-FP8
    configuration carries its own file."""
    resolution = resolved(precision="fp8_static", calibration_path=CALIBRATION)
    kwargs = frontend_kwargs_from_config(resolution, ckpt_path="/ckpt/model.pt")
    assert kwargs["precision"] == "fp8_static"
    assert kwargs["calibration_path"] == CALIBRATION == resolution.options.calibration_path


# -- dims_override -----------------------------------------------------------


def test_dims_override_carries_the_libero_layout():
    resolution = resolved()
    kwargs = frontend_kwargs_from_config(resolution)
    dims = kwargs["dims_override"]
    assert dims == resolution.frontend_dims()
    assert (dims["x0"], dims["a0"], dims["total"]) == (513, 905, 969)
    assert (dims["ref_h"], dims["ref_w"]) == (14, 28)
    assert dims["ref_h"] * dims["ref_w"] == dims["a0"] - dims["x0"]
    assert dims["num_action"] == 64
    assert dims["dt"] == pytest.approx(0.1)
    assert dims["action_dim"] == 7 and dims["proprio_dim"] == 8


def test_dims_override_carries_the_merge_flags():
    """The merged single-stream GEMMs are `dims_override` entries: the
    constructor derives `merge_qkv_mlp` from the precision and reads
    `merge_linear2` from `dims_override`. `merge_qkv_mlp` itself is not
    selectable against the precision (rule R9)."""
    dims = frontend_kwargs_from_config(resolved())["dims_override"]
    assert dims["merge_qkv_mlp"] is True and dims["merge_linear2"] is True
    cutlass = frontend_kwargs_from_config(resolved(precision="fp16_cutlass"))["dims_override"]
    assert cutlass["merge_qkv_mlp"] is False and cutlass["merge_linear2"] is False
    split = frontend_kwargs_from_config(resolved(merge_linear2=False))["dims_override"]
    assert split["merge_qkv_mlp"] is True and split["merge_linear2"] is False
    with pytest.raises(ConfigError) as excinfo:
        resolved(merge_qkv_mlp=False)
    assert excinfo.value.rule == "R9"


# -- derived flags (W9) ------------------------------------------------------


def test_fast_profile_carries_the_derived_flags():
    """`vae_graph_input` is the workload's own `(num_views, h, w)` and
    `use_fa4_mot` the profile's, both resolved; neither is re-derived here."""
    resolution = fast_resolved()
    options = resolution.options
    assert options.vae_graph_input == LIBERO_WORKLOAD.vae_graph_input() == (2, 224, 224)
    kwargs = frontend_kwargs_from_config(resolution, ae_model_path=AE, flux2_src=FLUX2)
    assert kwargs["vae_graph_input"] == (2, 224, 224)
    assert kwargs["vae_graph_input"] == options.vae_graph_input
    assert kwargs["vae_encoder"] == "native"
    assert kwargs["use_fa4"] is True and kwargs["use_fa4_mot"] is True
    assert kwargs["use_fa4_mot"] == options.use_fa4_mot


def test_no_vae_graph_input_without_the_vae_in_graph_profile():
    """The default profile runs the VAE outside the graph: `None`, and no
    `(num_views, h, w)` is invented for it."""
    kwargs = frontend_kwargs_from_config(resolved())
    assert kwargs["vae_graph_input"] is None
    assert kwargs["use_fa4_mot"] is False


# -- illegal combinations ----------------------------------------------------


ILLEGAL = [
    (dict(precision="fp8_static"), "R1"),
    (dict(text_trim=True, consumer="native"), "R5"),
    (dict(gemm_variant_autotune=True, precision="fp16"), "R2"),
]


@pytest.mark.parametrize("overrides, rule", [(o, r) for o, r in ILLEGAL],
                         ids=[r for _, r in ILLEGAL])
def test_illegal_combination_raises_before_the_constructor(overrides, rule):
    """The resolver refuses the combination; nothing was built -- the
    constructor is patched to fail this test if `from_config` reaches it."""
    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__",
                           side_effect=AssertionError("the constructor was reached")):
        with pytest.raises(ConfigError) as excinfo:
            ImageWAMTorchFrontendThor.from_config(
                resolve_config(LIBERO_WORKLOAD, LIBERO_STRUCTURE, **overrides))
    assert excinfo.value.rule == rule
    assert str(excinfo.value).startswith(f"{rule}: ")


# -- CPU-only mapping, and `from_config` is that mapping ---------------------


def test_mapping_returns_on_a_cpu_only_machine():
    """The mapping is plain data: it returns here (no GPU, no compiled
    extension) and creates no CUDA context."""
    if not torch.cuda.is_available():
        assert not torch.cuda.is_initialized()
    kwargs = frontend_kwargs_from_config(resolved())
    assert isinstance(kwargs, dict) and kwargs["dims_override"]["x0"] == 513
    if not torch.cuda.is_available():
        assert not torch.cuda.is_initialized()


def test_from_config_passes_exactly_the_mapping_kwargs():
    """`from_config(resolved, **paths)` is `cls(**frontend_kwargs_from_config(
    resolved, **paths))`, with the real `__init__` replaced by a recorder."""
    resolution = resolved()
    paths = dict(ckpt_path="/ckpt/model.pt", ae_model_path=AE, flux2_src=FLUX2,
                 qwen3_model_spec="Qwen/Qwen3-4B", dataset_stats_path="/stats/dataset_stats.json",
                 vae_resize="pil_bilinear")
    calls: list[dict] = []

    def record(self, **kwargs) -> None:
        calls.append(dict(kwargs))

    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", record):
        frontend = ImageWAMTorchFrontendThor.from_config(resolution, **paths)

    assert isinstance(frontend, ImageWAMTorchFrontendThor)
    assert len(calls) == 1
    expected = frontend_kwargs_from_config(resolution, **paths)
    assert calls[0] == expected
    assert set(calls[0]) == set(CONSTRUCTOR_KEYWORDS)
    assert calls[0]["dims_override"]["x0"] == 513 and calls[0]["ckpt_path"] == paths["ckpt_path"]


def test_from_config_uses_the_resolvers_derived_flags():
    """The `fast` profile through `from_config`, again without building."""
    calls: list[dict] = []

    def record(self, **kwargs) -> None:
        calls.append(dict(kwargs))

    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", record):
        ImageWAMTorchFrontendThor.from_config(fast_resolved(), ae_model_path=AE, flux2_src=FLUX2)

    assert calls[0]["vae_graph_input"] == (2, 224, 224)
    assert calls[0]["vae_encoder"] == "native"
    assert calls[0]["use_fa4"] is True and calls[0]["use_fa4_mot"] is True
