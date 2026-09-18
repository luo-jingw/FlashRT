"""Model structure of an ImageWAM checkpoint: backbone + action expert,
no workload (plan.md "Plan: configuration consolidation", phase W2).

`ImageWAMStructure` is the one place that reads the per-model widths
that today are typed by hand into a `dims` dict
(`imagewam_thor._DEFAULT_DIMS`, `libero_dims.LIBERO_REAL_DIMS`, the
benchmarks' `REAL_DIMS`):

* Backbone widths (`hidden`, `HD`, `NH`, `mlp_hidden`,
  `joint_attention_dim`, layer counts) are read from the tensor SHAPES
  and key counts of the checkpoint's `mot` state_dict. `config.yaml`
  does not carry them (it names the base model only through a
  `variant: klein-base-4b` string), so the tensors are the only source.
* Action-expert dims and `max_action_horizon` come from
  `model.action_dit_config` of the `config.yaml` that sits next to
  `model.pt`. Where a value can also be read off a tensor it is
  cross-checked, and a disagreement raises (a wrong `config.yaml` next
  to a checkpoint must not silently produce wrong dims).
* `patch_stride` is a constant of the FLUX.2 VAE, not a checkpoint
  value (see `VAE_PATCH_STRIDE`).

Only tensor shapes are read. `model.pt` is opened with
`torch.load(..., mmap=True, map_location="cpu")` (through
`checkpoint_loader.load_real_imagewam_state_dict`), so the ~9 GB of
weights are never materialised; no GPU and no compiled extension are
needed.

Every problem with the inputs (missing `config.yaml`, missing
`model.pt`, a missing tensor key, a shape that disagrees with another
tensor or with the config) raises `ValueError` naming the file, key or
field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

from flash_rt.models.imagewam.checkpoint_loader import load_real_imagewam_state_dict

# FLUX.2 AutoEncoder: 8x conv downsampling, then a 2x2 patch merge, so one
# latent patch covers 16x16 pixels (vae_stage.py:26-27, `VAE_SPATIAL_FACTOR
# = 16`; `VaeStageSpec.latent_hw` divides the per-view 224x224 by it to get
# the 14x14 latent grid, two views wide 14x28). vae_stage imports the
# compiled kernels extension, so this module keeps its own copy;
# tests/test_imagewam_structure.py pins the two equal.
VAE_PATCH_STRIDE = 16

_VIDEO = "mixtures.video.transformer"
_ACTION = "mixtures.action"


@dataclass(frozen=True)
class ImageWAMStructure:
    """Backbone and action-expert dims of one ImageWAM model.

    Field names are the `dims` keys of `imagewam_thor._DEFAULT_DIMS` /
    `LIBERO_REAL_DIMS`, except `max_action_horizon` and `patch_stride`
    which are not `dims` keys (they bound and derive workload fields).
    """
    hidden: int
    HD: int
    NH: int
    mlp_hidden: int
    joint_attention_dim: int
    num_layers_double: int
    num_layers_single: int
    action_hidden_dim: int
    action_attn_width: int
    action_mlp_hidden: int
    action_num_layers_double: int
    action_num_layers_single: int
    max_action_horizon: int
    patch_stride: int

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"ImageWAMStructure.{f.name} must be a positive int, got {v!r}")

    @staticmethod
    def toy() -> "ImageWAMStructure":
        """The random-weight dry-run dims: the structure-related values of
        `imagewam_thor._DEFAULT_DIMS` (pinned equal by
        tests/test_imagewam_structure.py).

        `_DEFAULT_DIMS` has no `max_action_horizon`; the only horizon it
        holds is `num_action=4`, which is what the random-weight path
        allocates, so that is the toy limit. `patch_stride` is the VAE
        constant.
        """
        return ImageWAMStructure(
            hidden=256, HD=128, NH=2, mlp_hidden=384, joint_attention_dim=64,
            num_layers_double=2, num_layers_single=3,
            action_hidden_dim=128, action_attn_width=256, action_mlp_hidden=192,
            action_num_layers_double=2, action_num_layers_single=3,
            max_action_horizon=4, patch_stride=VAE_PATCH_STRIDE,
        )

    @staticmethod
    def libero() -> "ImageWAMStructure":
        """The structure constants of the real `ImageWAM-FLUX.2-4B-LIBERO`
        release, as a constant table (not read from a checkpoint):

            hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
            num_layers_double=5, num_layers_single=20,
            action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
            action_num_layers_double=5, action_num_layers_single=20,
            max_action_horizon=64, patch_stride=16

        These are the structure-related entries of
        `libero_dims.LIBERO_REAL_DIMS`, which carries them beside the
        served workload and sequence layout; `patch_stride` is the FLUX.2
        VAE constant, as in `toy()`. Pinned two ways by
        tests/test_imagewam_structure.py: field by field against that
        literal table, and through `from_checkpoint(ckpt_path) ==
        libero()` on the real checkpoint when one is configured.
        """
        return ImageWAMStructure(
            hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
            num_layers_double=5, num_layers_single=20,
            action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
            action_num_layers_double=5, action_num_layers_single=20,
            max_action_horizon=64, patch_stride=VAE_PATCH_STRIDE,
        )

    @staticmethod
    def from_checkpoint(ckpt_path) -> "ImageWAMStructure":
        """Read the structure of the checkpoint at `ckpt_path`.

        `ckpt_path` is the `model.pt` file (what the frontend takes as
        `ckpt_path=`) or the directory holding it; `config.yaml` is read
        from the same directory.
        """
        model_pt, config_yaml = _locate(ckpt_path)
        cfg = _read_action_config(config_yaml)
        sd = load_real_imagewam_state_dict(str(model_pt))
        shapes = _ShapeReader(sd, model_pt)
        try:
            return _build(shapes, cfg, config_yaml)
        finally:
            del shapes, sd


# ── file location and config ──────────────────────────────────────────────


def _locate(ckpt_path) -> tuple[Path, Path]:
    p = Path(ckpt_path)
    if p.is_dir():
        model_pt, root = p / "model.pt", p
    else:
        model_pt, root = p, p.parent
    if not model_pt.is_file():
        raise ValueError(f"ImageWAM checkpoint not found: {model_pt}")
    config_yaml = root / "config.yaml"
    if not config_yaml.is_file():
        raise ValueError(
            f"config.yaml not found next to {model_pt} (expected {config_yaml}); it is the only "
            f"source of model.action_dit_config.max_action_horizon")
    return model_pt, config_yaml


def _read_action_config(config_yaml: Path) -> dict:
    import yaml

    try:
        with open(config_yaml, "r", encoding="utf-8") as f:
            root = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"{config_yaml} is not valid YAML: {e}") from e
    if not isinstance(root, dict):
        raise ValueError(f"{config_yaml}: top level must be a mapping, got {type(root).__name__}")
    model = root.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("action_dit_config"), dict):
        raise ValueError(f"{config_yaml}: missing model.action_dit_config mapping")
    return model["action_dit_config"]


# ── tensor shapes ─────────────────────────────────────────────────────────


class _ShapeReader:
    """Shape lookups on the `mot` state_dict, with errors naming the key."""

    def __init__(self, sd: dict, source: Path) -> None:
        self._sd = sd
        self._source = source

    def shape(self, key: str) -> tuple[int, ...]:
        try:
            t = self._sd[key]
        except KeyError:
            raise ValueError(f"{self._source}: expected tensor key {key!r} is missing") from None
        return tuple(int(s) for s in t.shape)

    def dim(self, key: str, axis: int, ndim: int) -> int:
        s = self.shape(key)
        if len(s) != ndim:
            raise ValueError(f"{self._source}: tensor {key!r} has shape {s}, expected {ndim} dims")
        return s[axis]

    def count_blocks(self, prefix: str, probe: str) -> int:
        """Number of `{prefix}.{i}.{probe}` keys; indices must be 0..n-1."""
        pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.{re.escape(probe)}$")
        idx = sorted(int(m.group(1)) for k in self._sd if (m := pat.match(k)))
        if not idx:
            raise ValueError(f"{self._source}: no tensors matching {prefix}.<i>.{probe}")
        if idx != list(range(len(idx))):
            raise ValueError(f"{self._source}: block indices under {prefix} are not contiguous "
                             f"from 0: {idx[:8]}...")
        return len(idx)


def _agree(name: str, from_config, from_tensor, config_yaml: Path, tensor_key: str):
    """Value of one structural field known from config and/or a tensor."""
    if from_config is None and from_tensor is None:
        raise ValueError(f"cannot determine {name}: absent from {config_yaml} and no tensor")
    if from_config is not None and from_tensor is not None and from_config != from_tensor:
        raise ValueError(
            f"{name}: {config_yaml} says {from_config} but tensor {tensor_key} implies "
            f"{from_tensor}; config.yaml does not belong to this checkpoint")
    return from_config if from_config is not None else from_tensor


def _cfg_int(cfg: dict, key: str, config_yaml: Path):
    v = cfg.get(key)
    if v is None:
        return None
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(f"{config_yaml}: model.action_dit_config.{key} must be an int, got {v!r}")
    return v


def _build(sh: _ShapeReader, cfg: dict, config_yaml: Path) -> ImageWAMStructure:
    # ── backbone: tensors only ────────────────────────────────────────────
    txt_in = f"{_VIDEO}.txt_in.weight"  # (hidden, joint_attention_dim), real (out,in)
    hidden = sh.dim(txt_in, 0, 2)
    joint_attention_dim = sh.dim(txt_in, 1, 2)
    img_in = f"{_VIDEO}.img_in.weight"
    if sh.dim(img_in, 0, 2) != hidden:
        raise ValueError(f"{img_in} out width {sh.dim(img_in, 0, 2)} != {txt_in} out width {hidden}")

    blk0 = f"{_VIDEO}.double_blocks.0"
    HD = sh.dim(f"{blk0}.img_attn.norm.query_norm.scale", 0, 1)
    qkv_out = sh.dim(f"{blk0}.img_attn.qkv.weight", 0, 2)
    if qkv_out % (3 * HD):
        raise ValueError(f"{blk0}.img_attn.qkv.weight out width {qkv_out} is not 3*NH*HD (HD={HD})")
    NH = qkv_out // (3 * HD)
    if NH * HD != hidden:
        raise ValueError(f"backbone attention width NH*HD = {NH}*{HD} != hidden {hidden} "
                         f"(from {blk0}.img_attn.qkv.weight)")
    mlp_hidden = sh.dim(f"{blk0}.img_mlp.2.weight", 1, 2)
    if sh.dim(f"{blk0}.img_mlp.0.weight", 0, 2) != 2 * mlp_hidden:
        raise ValueError(f"{blk0}.img_mlp.0.weight out width != 2*mlp_hidden ({2 * mlp_hidden}; "
                         f"gate+up from {blk0}.img_mlp.2.weight)")
    single0 = f"{_VIDEO}.single_blocks.0"
    if sh.dim(f"{single0}.linear1.weight", 0, 2) != 3 * hidden + 2 * mlp_hidden:
        raise ValueError(f"{single0}.linear1.weight out width != 3*hidden + 2*mlp_hidden "
                         f"({3 * hidden + 2 * mlp_hidden})")
    if sh.dim(f"{single0}.linear2.weight", 1, 2) != hidden + mlp_hidden:
        raise ValueError(f"{single0}.linear2.weight in width != hidden + mlp_hidden "
                         f"({hidden + mlp_hidden})")
    num_layers_double = sh.count_blocks(f"{_VIDEO}.double_blocks", "img_attn.qkv.weight")
    num_layers_single = sh.count_blocks(f"{_VIDEO}.single_blocks", "linear1.weight")

    # ── action expert: config.yaml, cross-checked against tensors ─────────
    a0 = f"{_ACTION}.double_blocks.0"
    enc = f"{_ACTION}.action_encoder.weight"
    action_hidden_dim = _agree(
        "action_hidden_dim", _cfg_int(cfg, "hidden_dim", config_yaml), sh.dim(enc, 0, 2),
        config_yaml, enc)

    heads, head_dim = _cfg_int(cfg, "num_heads", config_yaml), _cfg_int(cfg, "attn_head_dim", config_yaml)
    if head_dim is not None and head_dim != HD:
        raise ValueError(f"{config_yaml}: attn_head_dim {head_dim} != backbone head dim {HD}; "
                         f"mot joint attention needs one per-head geometry")
    qkv_key = f"{a0}.img_attn.qkv.weight"
    qkv_out_a = sh.dim(qkv_key, 0, 2)
    if qkv_out_a % 3:
        raise ValueError(f"{qkv_key} out width {qkv_out_a} is not 3*attn_width")
    action_attn_width = _agree(
        "action_attn_width", heads * head_dim if heads and head_dim else None, qkv_out_a // 3,
        config_yaml, qkv_key)

    mlp_ratio = cfg.get("mlp_ratio")
    mlp_from_cfg = None
    if mlp_ratio is not None:
        if isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, (int, float)):
            raise ValueError(f"{config_yaml}: model.action_dit_config.mlp_ratio must be a number, "
                             f"got {mlp_ratio!r}")
        prod = action_hidden_dim * mlp_ratio
        if prod != int(prod):
            raise ValueError(f"{config_yaml}: hidden_dim {action_hidden_dim} * mlp_ratio {mlp_ratio} "
                             f"is not an integer")
        mlp_from_cfg = int(prod)
    mlp_key = f"{a0}.img_mlp.2.weight"
    action_mlp_hidden = _agree(
        "action_mlp_hidden", mlp_from_cfg, sh.dim(mlp_key, 1, 2), config_yaml, mlp_key)

    n_ad = sh.count_blocks(f"{_ACTION}.double_blocks", "img_attn.qkv.weight")
    n_as = sh.count_blocks(f"{_ACTION}.single_blocks", "linear1.weight")
    action_num_layers_double = _agree(
        "action_num_layers_double", _cfg_int(cfg, "num_layers_double", config_yaml), n_ad,
        config_yaml, f"{_ACTION}.double_blocks.*")
    action_num_layers_single = _agree(
        "action_num_layers_single", _cfg_int(cfg, "num_layers_single", config_yaml), n_as,
        config_yaml, f"{_ACTION}.single_blocks.*")

    # No tensor carries the horizon; config.yaml is its only source.
    max_action_horizon = _cfg_int(cfg, "max_action_horizon", config_yaml)
    if max_action_horizon is None:
        raise ValueError(f"{config_yaml}: model.action_dit_config.max_action_horizon is missing "
                         f"(no checkpoint tensor carries it)")

    return ImageWAMStructure(
        hidden=hidden, HD=HD, NH=NH, mlp_hidden=mlp_hidden,
        joint_attention_dim=joint_attention_dim,
        num_layers_double=num_layers_double, num_layers_single=num_layers_single,
        action_hidden_dim=action_hidden_dim, action_attn_width=action_attn_width,
        action_mlp_hidden=action_mlp_hidden,
        action_num_layers_double=action_num_layers_double,
        action_num_layers_single=action_num_layers_single,
        max_action_horizon=max_action_horizon, patch_stride=VAE_PATCH_STRIDE,
    )
