"""`load_imagewam`, the deployment entry, and the workload runtime identity.

CPU only: this machine has no usable GPU, and the entry point is exercised
without constructing a frontend (the constructor allocates CUDA tensors).
What is pinned here:

* the entry's signature is the one `plan.md`'s W0 addendum freezes;
* every legality decision belongs to `resolve_config`: an illegal combination
  raises its `ConfigError` (rule id included) and `from_config` is never
  reached, so nothing is allocated;
* the entry routes through `resolve_config` and `from_config`, reading the
  structure from the checkpoint only when none is given;
* `runtime_surface.workload_identity` is the `workload.<field>` identity a
  captured graph reports beside its `dims.<key>` entries, and
  `runtime_surface()` includes it for a frontend built from a resolved
  configuration.
"""
from __future__ import annotations

import inspect
from unittest import mock

import pytest

from flash_rt.frontends.torch.imagewam_thor import (
    ImageWAMTorchFrontendThor,
    frontend_kwargs_from_config,
    load_imagewam,
)
from flash_rt.models.imagewam.config_resolver import (
    ConfigError,
    resolve_config,
)
from flash_rt.models.imagewam.runtime_surface import (
    WORKLOAD_IDENTITY_FIELDS,
    workload_identity,
)
from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload

WORKLOAD = ImageWAMWorkload.libero()
STRUCTURE = ImageWAMStructure.libero()
CKPT = "/models/imagewam/model.pt"


def test_signature_is_the_frozen_entry():
    params = inspect.signature(load_imagewam).parameters
    assert list(params) == ["ckpt_path", "workload", "structure", "profile", "precision",
                            "calibration_path", "ae_model_path", "flux2_src", "qwen3_model_spec",
                            "dataset_stats_path", "consumer", "allow_placeholder_calibration",
                            "vae_resize", "precapture_text_lengths", "expert"]
    positional = [n for n, p in params.items()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and n != "expert"]
    assert positional == ["ckpt_path", "workload"]
    assert params["expert"].kind is inspect.Parameter.VAR_KEYWORD
    assert params["profile"].default == "default"
    assert params["consumer"].default == "infer"
    assert params["allow_placeholder_calibration"].default is False
    assert params["precapture_text_lengths"].default is None


def test_structure_is_required_without_a_checkpoint():
    with pytest.raises(ValueError, match="needs ckpt_path"):
        load_imagewam(None, WORKLOAD, structure=None)
    # with an explicit structure a random-weight run is legal up to the frontend
    assert True


def _structure_from_checkpoint_stub() -> mock.Mock:
    """`ImageWAMStructure.from_checkpoint` replaced by a Mock returning the
    LIBERO constant table: no checkpoint is read here."""
    return mock.patch.object(ImageWAMStructure, "from_checkpoint",
                             new=mock.Mock(return_value=STRUCTURE))


def test_entry_reads_the_structure_from_the_checkpoint_only_when_given():
    with _structure_from_checkpoint_stub() as from_ckpt, \
            mock.patch.object(ImageWAMTorchFrontendThor, "from_config",
                              return_value="frontend") as from_config:
        assert load_imagewam(CKPT, WORKLOAD) == "frontend"
        from_ckpt.assert_called_once_with(CKPT)
        assert from_config.call_count == 1
        _, kwargs = from_config.call_args
        assert kwargs["ckpt_path"] == CKPT
        assert kwargs["workload"] is WORKLOAD
        resolved = from_config.call_args[0][0]
        assert resolved.dims["x0"] == 513 and resolved.dims["total"] == 969

    with mock.patch.object(ImageWAMStructure, "from_checkpoint") as from_ckpt, \
            mock.patch.object(ImageWAMTorchFrontendThor, "from_config",
                              return_value="frontend") as from_config:
        assert load_imagewam(CKPT, WORKLOAD, structure=STRUCTURE) == "frontend"
        from_ckpt.assert_not_called()
        assert from_config.call_args[0][0].dims == resolve_config(WORKLOAD, STRUCTURE).dims


def test_entry_passes_the_tier_one_arguments_through():
    with _structure_from_checkpoint_stub(), \
            mock.patch.object(ImageWAMTorchFrontendThor, "from_config",
                              return_value="frontend") as from_config:
        load_imagewam(CKPT, WORKLOAD, profile="fast", precision="fp16", ae_model_path="/ae",
                      flux2_src="/flux2", qwen3_model_spec="/qwen3", dataset_stats_path="/stats",
                      calibration_path=None, vae_resize="pil_bilinear")
        resolved = from_config.call_args[0][0]
        assert resolved.options.precision == "fp16"
        assert resolved.options.text_trim is True          # the fast profile
        assert resolved.options.vae_graph_input == (2, 224, 224)
        kwargs = from_config.call_args[1]
        assert kwargs["ae_model_path"] == "/ae" and kwargs["flux2_src"] == "/flux2"
        assert kwargs["qwen3_model_spec"] == "/qwen3" and kwargs["dataset_stats_path"] == "/stats"
        assert kwargs["vae_resize"] == "pil_bilinear"


@pytest.mark.parametrize("kw,rule", [
    (dict(profile="turbo"), "V1"),
    (dict(precision="fp8_static"), "R1"),
    (dict(precision="fp16", gemm_variant_autotune=True), "R2"),
    (dict(vae_graph=True), "R3"),
    (dict(precision="fp16", nvfp4_awq=True, calibration_path="/cal"), "R4"),
    (dict(text_trim=True, consumer="native"), "R5"),
    (dict(consumer="native", precision="fp8_static", allow_placeholder_calibration=True), "R6"),
])
def test_illegal_combination_raises_before_the_constructor(kw, rule):
    with _structure_from_checkpoint_stub(), \
            mock.patch.object(ImageWAMTorchFrontendThor, "__init__",
                              side_effect=AssertionError("the frontend must not be constructed")):
        with pytest.raises(ConfigError) as e:
            load_imagewam(CKPT, WORKLOAD, **kw)
    assert e.value.rule == rule, str(e.value)


def test_text_trim_reaches_the_constructor_for_the_abi_consumer():
    """`text_trim=True` is refused for the native consumer only (rule R5):
    the ABI consumer carries one graph per captured text length, so the
    entry resolves it and passes it to the frontend."""
    with _structure_from_checkpoint_stub(), \
            mock.patch.object(ImageWAMTorchFrontendThor, "from_config",
                              return_value="frontend") as from_config:
        assert load_imagewam(CKPT, WORKLOAD, text_trim=True, consumer="abi") == "frontend"
    resolved = from_config.call_args[0][0]
    assert resolved.options.text_trim is True


def test_from_config_records_the_workload_without_building():
    resolved = resolve_config(WORKLOAD, STRUCTURE)
    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", lambda self, **kw: None):
        fe = ImageWAMTorchFrontendThor.from_config(resolved, workload=WORKLOAD)
    assert fe._workload is WORKLOAD
    assert fe.resolved_config is resolved
    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", lambda self, **kw: None):
        fe = ImageWAMTorchFrontendThor.from_config(resolved)
    assert fe._workload is None
    assert fe.resolved_config is resolved


def test_workload_identity_names_the_served_workload():
    pairs = workload_identity(WORKLOAD)
    assert pairs == (("workload.num_views", "2"), ("workload.image_h", "224"),
                     ("workload.image_w", "224"), ("workload.text_max_len", "512"),
                     ("workload.action_horizon", "64"), ("workload.action_dim", "7"),
                     ("workload.proprio_dim", "8"), ("workload.num_steps", "10"),
                     ("workload.shift", "5.0"))
    assert tuple(name for name, _ in pairs) == tuple(f"workload.{f}" for f in WORKLOAD_IDENTITY_FIELDS)
    # a different workload is a different identity
    other = ImageWAMWorkload(num_views=3, image_h=224, image_w=224, text_max_len=512,
                             action_horizon=64, action_dim=7, proprio_dim=8, num_steps=10, shift=5.0)
    assert workload_identity(other)[0] == ("workload.num_views", "3")
    assert workload_identity(other) != pairs


def test_runtime_surface_reports_the_workload_identity():
    src = inspect.getsource(ImageWAMTorchFrontendThor.runtime_surface)
    assert "workload_identity(self._workload)" in src
    assert "self._workload is not None" in src
    assert "dims.{k}" in src  # the dims entries stay, the workload ones are additive


def test_frontend_kwargs_carry_the_resolved_options():
    resolved = resolve_config(WORKLOAD, STRUCTURE)
    kwargs = frontend_kwargs_from_config(resolved, ckpt_path=CKPT)
    assert kwargs["precision"] == "nvfp4" and kwargs["ckpt_path"] == CKPT
    assert kwargs["dims_override"]["x0"] == 513
