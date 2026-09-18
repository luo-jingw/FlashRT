"""`ImageWAMStructure` (flash_rt/models/imagewam/structure.py), CPU only.

* `toy()` equals the structure-related values of the frontend's
  `_DEFAULT_DIMS`. The frontend needs a GPU context and the compiled
  kernels to import, so the dict is read out of the module SOURCE with
  `ast` (the actual literal, not a copy in this file).
* `from_checkpoint()` on the real ImageWAM-FLUX.2-4B-LIBERO release equals
  the structure keys of `libero_dims.LIBERO_REAL_DIMS`. The release is
  looked up at `$CKPT_PATH` (the same variable the other real-checkpoint
  tests use) or the default dev-machine path; the test skips when absent.
  Only tensor shapes are read (mmap), the ~9 GB file is not loaded.
* `from_checkpoint()` on small synthetic checkpoints (shapes only) covers
  the file/directory forms, the toy round trip and the error paths.
"""
from __future__ import annotations

import ast
import os
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import yaml

from flash_rt.models.imagewam.libero_dims import LIBERO_HORIZON, LIBERO_REAL_DIMS
from flash_rt.models.imagewam.structure import VAE_PATCH_STRIDE, ImageWAMStructure

REPO = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO / "flash_rt" / "frontends" / "torch" / "imagewam_thor.py"
VAE_STAGE_SRC = REPO / "flash_rt" / "models" / "imagewam" / "vae_stage.py"

_DEFAULT_CKPT = "/home/ljw/projects/pi0.5/models/imagewam_flux2_4b_libero/model.pt"
_CKPT_PATH = os.environ.get("CKPT_PATH", _DEFAULT_CKPT)

# Fields that are `dims` keys; the remaining two (`max_action_horizon`,
# `patch_stride`) bound / derive workload fields and are not `dims` keys.
_DIMS_KEYS = tuple(f.name for f in fields(ImageWAMStructure)
                   if f.name not in ("max_action_horizon", "patch_stride"))


def _frontend_default_dims() -> dict:
    """`_DEFAULT_DIMS = dict(k=<literal>, ...)` from the frontend source."""
    tree = ast.parse(FRONTEND_SRC.read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "_DEFAULT_DIMS"):
            call = node.value
            assert isinstance(call, ast.Call) and call.func.id == "dict"
            return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    raise AssertionError(f"_DEFAULT_DIMS not found in {FRONTEND_SRC}")


def _module_int_constant(path: Path, name: str) -> int:
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


# ── toy ─────────────────────────────────────────────────────────────────


def test_toy_equals_frontend_default_dims():
    default = _frontend_default_dims()
    toy = ImageWAMStructure.toy()
    for key in _DIMS_KEYS:
        assert getattr(toy, key) == default[key], key
    # _DEFAULT_DIMS has no horizon limit; the only horizon it holds is the
    # one the random-weight path allocates.
    assert toy.max_action_horizon == default["num_action"]


def test_patch_stride_is_the_vae_factor():
    assert VAE_PATCH_STRIDE == _module_int_constant(VAE_STAGE_SRC, "VAE_SPATIAL_FACTOR") == 16
    assert ImageWAMStructure.toy().patch_stride == 16


def test_structure_is_frozen_and_validated():
    toy = ImageWAMStructure.toy()
    with pytest.raises(Exception):
        toy.hidden = 1  # type: ignore[misc]
    kw = {f.name: getattr(toy, f.name) for f in fields(toy)}
    kw["NH"] = 0
    with pytest.raises(ValueError, match="NH"):
        ImageWAMStructure(**kw)


# ── synthetic checkpoints (shapes only) ─────────────────────────────────


def _zeros(*shape: int) -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.bfloat16)


def _mot_state_dict(s: ImageWAMStructure, *, action_dim: int = 7) -> dict:
    """The `mot` keys `from_checkpoint` reads, at `s`'s shapes."""
    v = "mixtures.video.transformer"
    sd = {
        f"{v}.txt_in.weight": _zeros(s.hidden, s.joint_attention_dim),
        f"{v}.img_in.weight": _zeros(s.hidden, s.HD),
    }
    for i in range(s.num_layers_double):
        for side in ("img", "txt"):
            sd[f"{v}.double_blocks.{i}.{side}_attn.qkv.weight"] = _zeros(3 * s.hidden, s.hidden)
            sd[f"{v}.double_blocks.{i}.{side}_attn.norm.query_norm.scale"] = _zeros(s.HD)
            sd[f"{v}.double_blocks.{i}.{side}_mlp.0.weight"] = _zeros(2 * s.mlp_hidden, s.hidden)
            sd[f"{v}.double_blocks.{i}.{side}_mlp.2.weight"] = _zeros(s.hidden, s.mlp_hidden)
    for i in range(s.num_layers_single):
        sd[f"{v}.single_blocks.{i}.linear1.weight"] = _zeros(3 * s.hidden + 2 * s.mlp_hidden, s.hidden)
        sd[f"{v}.single_blocks.{i}.linear2.weight"] = _zeros(s.hidden, s.hidden + s.mlp_hidden)
    a = "mixtures.action"
    sd[f"{a}.action_encoder.weight"] = _zeros(s.action_hidden_dim, action_dim)
    for i in range(s.action_num_layers_double):
        sd[f"{a}.double_blocks.{i}.img_attn.qkv.weight"] = _zeros(3 * s.action_attn_width, s.action_hidden_dim)
        sd[f"{a}.double_blocks.{i}.img_mlp.0.weight"] = _zeros(2 * s.action_mlp_hidden, s.action_hidden_dim)
        sd[f"{a}.double_blocks.{i}.img_mlp.2.weight"] = _zeros(s.action_hidden_dim, s.action_mlp_hidden)
    for i in range(s.action_num_layers_single):
        sd[f"{a}.single_blocks.{i}.linear1.weight"] = _zeros(
            3 * s.action_attn_width + 2 * s.action_mlp_hidden, s.action_hidden_dim)
    return sd


def _config(s: ImageWAMStructure) -> dict:
    return {"model": {"action_dit_config": {
        "action_dim": 7,
        "hidden_dim": s.action_hidden_dim,
        "num_heads": s.action_attn_width // s.HD,
        "attn_head_dim": s.HD,
        "num_layers_double": s.action_num_layers_double,
        "num_layers_single": s.action_num_layers_single,
        "mlp_ratio": s.action_mlp_hidden / s.action_hidden_dim,
        "max_action_horizon": s.max_action_horizon,
    }}}


def _write(dirpath: Path, sd: dict | None = None, cfg: dict | bool | None = None,
           *, structure: ImageWAMStructure | None = None) -> Path:
    s = structure or ImageWAMStructure.toy()
    dirpath.mkdir(parents=True, exist_ok=True)
    torch.save({"mot": _mot_state_dict(s) if sd is None else sd, "step": 0}, dirpath / "model.pt")
    if cfg is not False:
        (dirpath / "config.yaml").write_text(yaml.safe_dump(_config(s) if cfg is None else cfg))
    return dirpath / "model.pt"


def test_synthetic_round_trip_file_and_directory(tmp_path):
    toy = ImageWAMStructure.toy()
    model_pt = _write(tmp_path / "ck", structure=toy)
    assert ImageWAMStructure.from_checkpoint(model_pt) == toy
    assert ImageWAMStructure.from_checkpoint(str(model_pt)) == toy
    assert ImageWAMStructure.from_checkpoint(model_pt.parent) == toy


def test_action_dims_fall_back_to_tensor_shapes(tmp_path):
    toy = ImageWAMStructure.toy()
    cfg = _config(toy)
    for k in ("hidden_dim", "num_heads", "attn_head_dim", "mlp_ratio", "num_layers_double",
              "num_layers_single"):
        del cfg["model"]["action_dit_config"][k]
    model_pt = _write(tmp_path / "ck", cfg=cfg, structure=toy)
    assert ImageWAMStructure.from_checkpoint(model_pt) == toy


def test_missing_config_yaml_raises(tmp_path):
    model_pt = _write(tmp_path / "ck", cfg=False)
    with pytest.raises(ValueError, match="config.yaml"):
        ImageWAMStructure.from_checkpoint(model_pt)
    with pytest.raises(ValueError, match="config.yaml"):
        ImageWAMStructure.from_checkpoint(model_pt.parent)


def test_missing_model_pt_raises(tmp_path):
    with pytest.raises(ValueError, match="model.pt"):
        ImageWAMStructure.from_checkpoint(tmp_path)


def test_config_without_action_dit_config_raises(tmp_path):
    model_pt = _write(tmp_path / "ck", cfg={"model": {"proprio_dim": 8}})
    with pytest.raises(ValueError, match="action_dit_config"):
        ImageWAMStructure.from_checkpoint(model_pt)


def test_missing_max_action_horizon_raises(tmp_path):
    cfg = _config(ImageWAMStructure.toy())
    del cfg["model"]["action_dit_config"]["max_action_horizon"]
    model_pt = _write(tmp_path / "ck", cfg=cfg)
    with pytest.raises(ValueError, match="max_action_horizon"):
        ImageWAMStructure.from_checkpoint(model_pt)


@pytest.mark.parametrize("missing", [
    "mixtures.video.transformer.txt_in.weight",
    "mixtures.video.transformer.double_blocks.0.img_attn.norm.query_norm.scale",
    "mixtures.video.transformer.double_blocks.0.img_mlp.2.weight",
    "mixtures.video.transformer.single_blocks.0.linear1.weight",
    "mixtures.action.action_encoder.weight",
    "mixtures.action.double_blocks.0.img_attn.qkv.weight",
])
def test_missing_tensor_key_raises_naming_it(tmp_path, missing):
    sd = _mot_state_dict(ImageWAMStructure.toy())
    del sd[missing]
    model_pt = _write(tmp_path / "ck", sd=sd)
    with pytest.raises(ValueError, match="missing") as e:
        ImageWAMStructure.from_checkpoint(model_pt)
    assert missing in str(e.value)


def test_missing_mot_payload_raises(tmp_path):
    (tmp_path / "ck").mkdir()
    torch.save({"model": {}}, tmp_path / "ck" / "model.pt")
    (tmp_path / "ck" / "config.yaml").write_text(yaml.safe_dump(_config(ImageWAMStructure.toy())))
    with pytest.raises(ValueError, match="mot"):
        ImageWAMStructure.from_checkpoint(tmp_path / "ck")


def test_config_that_disagrees_with_tensors_raises(tmp_path):
    cfg = _config(ImageWAMStructure.toy())
    cfg["model"]["action_dit_config"]["num_layers_single"] += 1
    model_pt = _write(tmp_path / "ck", cfg=cfg)
    with pytest.raises(ValueError, match="action_num_layers_single"):
        ImageWAMStructure.from_checkpoint(model_pt)


def test_inconsistent_backbone_shapes_raise(tmp_path):
    sd = _mot_state_dict(ImageWAMStructure.toy())
    key = "mixtures.video.transformer.single_blocks.0.linear2.weight"
    sd[key] = _zeros(*sd[key].shape[:1], sd[key].shape[1] + 1)
    model_pt = _write(tmp_path / "ck", sd=sd)
    with pytest.raises(ValueError, match="linear2"):
        ImageWAMStructure.from_checkpoint(model_pt)


# ── the real release ────────────────────────────────────────────────────

_real_missing = not Path(_CKPT_PATH).is_file() or not (Path(_CKPT_PATH).parent / "config.yaml").is_file()
_real_reason = (f"real ImageWAM-FLUX.2-4B-LIBERO release not found at {_CKPT_PATH} "
                f"(model.pt + config.yaml; set CKPT_PATH to override)")


@pytest.mark.skipif(_real_missing, reason=_real_reason)
def test_real_checkpoint_equals_libero_real_dims():
    s = ImageWAMStructure.from_checkpoint(_CKPT_PATH)
    for key in _DIMS_KEYS:
        assert getattr(s, key) == LIBERO_REAL_DIMS[key], key
    assert s.max_action_horizon == LIBERO_HORIZON
    # Two 224x224 views -> the 14 x 28 latent grid LIBERO_REAL_DIMS pins.
    assert s.patch_stride == 16
    assert (224 // s.patch_stride, 2 * 224 // s.patch_stride) == (
        LIBERO_REAL_DIMS["ref_h"], LIBERO_REAL_DIMS["ref_w"])


@pytest.mark.skipif(_real_missing, reason=_real_reason)
def test_real_checkpoint_directory_form_equals_file_form():
    assert (ImageWAMStructure.from_checkpoint(Path(_CKPT_PATH).parent)
            == ImageWAMStructure.from_checkpoint(_CKPT_PATH))
