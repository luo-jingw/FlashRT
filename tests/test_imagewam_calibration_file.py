"""Activation recording, calibration-file reduction/round trip, identity
checks (format version 3: the workload's camera geometry is part of the
identity, earlier versions are refused), and
`StaticFp8Linear.set_activation_scale` (roadmap item 7).

Small dims, random weights; the real build is
`benchmarks/imagewam_build_calibration.py`.
"""
import dataclasses
import json
import os
import tempfile

import numpy as np
import pytest
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.core.calibration import accumulate_amax
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.activation_recorder import (
    ABS_PERCENTILES, ActivationRecorder, site_name,
)
from flash_rt.models.imagewam.calibration_file import (
    FORMAT_VERSION, IDENTITY_DIM_KEYS, SUPPORTED_VERSIONS, build_calibration, identity_dims,
    load_calibration, save_calibration,
)
from flash_rt.models.imagewam.config_resolver import resolve_config
from flash_rt.models.imagewam.libero_dims import LIBERO_REAL_DIMS
from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload
from flash_rt.models.imagewam.quant_linear import Bf16OutLinear, Fp16Linear, StaticFp8Linear

DEV = "cuda"
FP16 = torch.float16


def test_recorder_matches_torch_reference():
    torch.manual_seed(0)
    gemm = fvk.GemmRunner()
    k, n = 96, 64
    w = (torch.randn(k, n, device=DEV) * 0.02).to(FP16)
    key = ("backbone", "single", 0, "linear1.weight")
    weights = {key: Fp16Linear(gemm, w.data_ptr(), n, k), ("x", 0): 12345}
    rec = ActivationRecorder()
    wrapped = rec.wrap(weights)
    assert wrapped[("x", 0)] == 12345

    xs = [torch.randn(40, k, device=DEV).to(FP16) * s for s in (1.0, 3.0)]
    rec.begin_sample()
    for x in xs:
        out_w = torch.zeros(40, n, dtype=FP16, device=DEV)
        out_ref = torch.zeros(40, n, dtype=FP16, device=DEV)
        wrapped[key](x.data_ptr(), out_w.data_ptr(), 40, 0)
        weights[key](x.data_ptr(), out_ref.data_ptr(), 40, 0)
        torch.cuda.synchronize()
        assert torch.equal(out_w, out_ref), "recording wrapper changed the GEMM output"
    stats = rec.end_sample().sites[site_name(key)]

    ax = [x.float().abs() for x in xs]
    q = torch.tensor([p / 100 for p in ABS_PERCENTILES], device=DEV)
    ref_pct = torch.maximum(torch.quantile(ax[0].flatten(), q), torch.quantile(ax[1].flatten(), q))
    ref_ch = torch.maximum(ax[0].amax(0), ax[1].amax(0))
    print(f"absmax {stats.absmax:.5f} vs {max(a.max().item() for a in ax):.5f}; "
          f"pct {stats.abs_percentiles} vs {ref_pct.cpu().numpy()}; calls={stats.calls}")
    assert stats.calls == 2 and stats.rows == 40
    assert stats.absmax == max(a.max().item() for a in ax)
    assert np.array_equal(stats.channel_amax, ref_ch.cpu().numpy())
    assert np.array_equal(stats.abs_percentiles, ref_pct.cpu().numpy())


def test_recorder_covers_every_gemm_site_in_frontend():
    """Toy-dims eager run: every fp16 GEMM site is recorded, ActionDiT
    sites once per denoise step, txt_in/img_in (BF16, never quantized)
    not at all."""
    fe = ImageWAMTorchFrontendThor(precision="fp16")
    fe.set_prompt()
    fe.stage_inputs({}, noise=torch.randn(fe.dims["num_action"], fe.dims["action_dim"], device=DEV))
    rec = ActivationRecorder()
    rec.begin_sample()
    fe.run_eager(rec.wrap(fe.weights))
    stats = rec.end_sample().sites
    expected = {site_name(k) for k, v in fe.weights.items()
                if not isinstance(v, (int, Bf16OutLinear))}
    print(f"{len(stats)} sites recorded, {len(expected)} expected")
    assert set(stats) == expected
    steps = fe.dims["num_denoise_steps"]
    for name, s in stats.items():
        assert s.calls == (steps if name.startswith("action_dit") else 1), name
        assert np.isfinite(s.absmax) and s.absmax > 0, name


def _fake_samples(names_k, n_samples, rng):
    from flash_rt.models.imagewam.activation_recorder import SampleStats, SiteStats
    out = []
    for _ in range(n_samples):
        sites = {}
        for name, k in names_k.items():
            ch = rng.random(k).astype(np.float32) * 5
            sites[name] = SiteStats(absmax=float(ch.max()),
                                    abs_percentiles=np.sort(rng.random(3).astype(np.float32)),
                                    channel_amax=ch, calls=1, rows=7)
        out.append(SampleStats(sites=sites))
    return out


def test_build_save_load_round_trip_and_identity():
    rng = np.random.default_rng(0)
    names_k = {"backbone.single.0.linear1.weight": 32, "action_dit.double.1.mlp2.weight": 48}
    samples = _fake_samples(names_k, 9, rng)
    dims = dict(hidden=256, x0=3, a0=8, num_action=4, merge_qkv_mlp=True, shift=5.0, dt=0.5,
                num_views=2, image_h=224, image_w=224)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "model.pt")
        with open(ckpt, "wb") as f:
            f.write(os.urandom(100_000))
        cal = build_calibration(samples, percentile=99.9, checkpoint_path=ckpt, dims=dims,
                                frames=[("libero_goal", 3, 10)], noise="test", text_trim=False)
        for name in names_k:
            per = [np.array([s.sites[name].absmax], dtype=np.float32) for s in samples]
            assert cal.sites[name].amax == float(accumulate_amax(per, percentile=99.9)[0])
            ref_ch = accumulate_amax([s.sites[name].channel_amax for s in samples], percentile=99.9)
            assert np.array_equal(cal.sites[name].channel_amax, ref_ch)
            scale = cal.sites[name].fp8_act_scale()
            assert scale == float(np.float32(cal.sites[name].amax) / np.float32(448.0))
        path = os.path.join(tmp, "cal.safetensors")
        save_calibration(cal, path)
        back = load_calibration(path)
        assert back.checkpoint_id == cal.checkpoint_id and back.dims == identity_dims(dims)
        assert back.version == FORMAT_VERSION == 3 and SUPPORTED_VERSIONS == (3,)
        assert (back.dims["num_views"], back.dims["image_h"], back.dims["image_w"]) == (2, 224, 224)
        assert "dt" not in back.dims
        for name in names_k:
            a, b = cal.sites[name], back.sites[name]
            assert a.amax == b.amax and a.rows == b.rows
            assert np.array_equal(a.channel_amax, b.channel_amax)
            assert np.array_equal(a.sample_absmax, b.sample_absmax)
            assert np.array_equal(a.abs_percentiles, b.abs_percentiles)
        back.validate_for(checkpoint_path=ckpt, dims=dims, text_trim=False)
        with pytest.raises(ValueError, match="dims differ"):
            back.validate_for(checkpoint_path=ckpt, dims=dict(dims, x0=4), text_trim=False)
        with pytest.raises(ValueError, match="text_trim"):
            back.validate_for(checkpoint_path=ckpt, dims=dims, text_trim=True)
        other = os.path.join(tmp, "other.pt")
        with open(other, "wb") as f:
            f.write(os.urandom(100_000))
        with pytest.raises(ValueError, match="checkpoint"):
            back.validate_for(checkpoint_path=other, dims=dims, text_trim=False)
        print(f"round trip OK: {len(back.sites)} sites, {os.path.getsize(path)} bytes")


def test_static_fp8_set_activation_scale_equals_calibrate():
    torch.manual_seed(0)
    m, n, k = 64, 256, 512
    w = (torch.randn(k, n, device=DEV) * 0.02).to(FP16)
    x = torch.randn(m, k, device=DEV).to(FP16)
    a = StaticFp8Linear(w.data_ptr(), n, k)
    a.calibrate(x.data_ptr(), m, 0)
    # `calibrate()` takes its scale from `compute_scale_kernel` (amax / 448 on the device, built with
    # --use_fast_math, which is not the IEEE divide numpy does: 1 ULP apart on Thor,
    # 0.0093122218 vs 0.0093122208). The calibration file's scale is the numpy value, so the two agree
    # to one float32 ULP, and `set_activation_scale` itself is pinned with the device's own scale below.
    host_scale = float(np.float32(x.float().abs().max().item()) / np.float32(448.0))
    assert host_scale == pytest.approx(a.act_scale.item(), rel=2.5e-7)
    b = StaticFp8Linear(w.data_ptr(), n, k)
    b.set_activation_scale(a.act_scale.item())
    oa = torch.zeros(m, n, dtype=FP16, device=DEV)
    ob = torch.zeros(m, n, dtype=FP16, device=DEV)
    a(x.data_ptr(), oa.data_ptr(), m, 0)
    b(x.data_ptr(), ob.data_ptr(), m, 0)
    torch.cuda.synchronize()
    print(f"act_scale calibrate={a.act_scale.item():.8g} set={b.act_scale.item():.8g} "
          f"outputs bit-exact={torch.equal(oa, ob)}")
    assert a.act_scale.item() == b.act_scale.item()
    assert torch.equal(oa, ob)
    with pytest.raises(RuntimeError, match="after __call__"):
        b.set_activation_scale(1.0)


def _write_v3_file(tmp: str, dims: dict, *, text_trim: bool = False, name: str = "cal.safetensors"):
    """A version-3 file for a fresh random checkpoint file; returns
    `(checkpoint_path, calibration_path)`."""
    ckpt = os.path.join(tmp, "model.pt")
    if not os.path.exists(ckpt):
        with open(ckpt, "wb") as f:
            f.write(os.urandom(100_000))
    samples = _fake_samples({"backbone.single.0.linear1.weight": 32}, 3, np.random.default_rng(1))
    cal = build_calibration(samples, percentile=99.9, checkpoint_path=ckpt, dims=dims,
                            frames=[("libero_goal", 3, 10)], noise="test", text_trim=text_trim)
    path = os.path.join(tmp, name)
    save_calibration(cal, path)
    return ckpt, path


def _rewrite_metadata(src: str, dst: str, edit) -> None:
    """Copy the safetensors file `src` to `dst`, with the JSON metadata
    entry passed through `edit(meta)` (in place)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(src, framework="pt") as f:
        meta = json.loads(f.metadata()["imagewam_calibration"])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    edit(meta)
    save_file(tensors, dst, metadata={"imagewam_calibration": json.dumps(meta)})


# Two workloads whose layouts coincide: 2 views of 224x224 and 4 views of
# 224x112 both give a 14 x 28 latent grid (x0, a0, ref_h, ref_w equal).
_LAYOUT = dict(hidden=256, x0=3, a0=395, num_action=4, ref_h=14, ref_w=28, merge_qkv_mlp=True)
_TWO_VIEWS = dict(_LAYOUT, num_views=2, image_h=224, image_w=224)
_FOUR_VIEWS = dict(_LAYOUT, num_views=4, image_h=224, image_w=112)


def test_identity_includes_the_camera_geometry():
    assert {"num_views", "image_h", "image_w"} <= set(IDENTITY_DIM_KEYS)
    assert identity_dims(_TWO_VIEWS) != identity_dims(_FOUR_VIEWS)
    # Every identity key except the two merge flags (the frontend fills
    # those) comes from the resolver's dims, the camera geometry included.
    assert set(IDENTITY_DIM_KEYS) - set(LIBERO_REAL_DIMS) == {"merge_qkv_mlp", "merge_linear2"}


def test_same_layout_different_camera_geometry_is_refused():
    """The collision this identity closes: a file recorded for 2 x 224x224
    views must not load for 4 x 224x112 views, though every layout dim
    (`x0`, `a0`, `ref_h`, `ref_w`) is equal."""
    assert all(_TWO_VIEWS[k] == _FOUR_VIEWS[k] for k in ("x0", "a0", "ref_h", "ref_w"))
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, path = _write_v3_file(tmp, _TWO_VIEWS)
        cal = load_calibration(path)
        cal.validate_for(checkpoint_path=ckpt, dims=_TWO_VIEWS, text_trim=False)
        with pytest.raises(ValueError, match="dims differ") as e:
            cal.validate_for(checkpoint_path=ckpt, dims=_FOUR_VIEWS, text_trim=False)
        msg = str(e.value)
        assert "num_views': (2, 4)" in msg and "image_w': (224, 112)" in msg
        assert "image_h" not in msg and "'x0'" not in msg   # image_h and the layout are equal
        # And the other way round.
        _, path4 = _write_v3_file(tmp, _FOUR_VIEWS, name="cal4.safetensors")
        with pytest.raises(ValueError, match="num_views"):
            load_calibration(path4).validate_for(checkpoint_path=ckpt, dims=_TWO_VIEWS, text_trim=False)


def test_resolved_workloads_with_one_layout_have_distinct_identities():
    """The same collision through the resolver's own dims: the workload's
    `num_views`/`image_h`/`image_w` reach `identity_dims`."""
    structure = ImageWAMStructure.libero()
    two = resolve_config(ImageWAMWorkload.libero(), structure).dims
    four = resolve_config(dataclasses.replace(ImageWAMWorkload.libero(), num_views=4, image_w=112),
                          structure).dims
    assert all(two[k] == four[k] for k in ("x0", "a0", "total", "ref_h", "ref_w"))
    assert (four["num_views"], four["image_h"], four["image_w"]) == (4, 224, 112)
    assert identity_dims(two) != identity_dims(four)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, path = _write_v3_file(tmp, two)
        cal = load_calibration(path)
        cal.validate_for(checkpoint_path=ckpt, dims=two, text_trim=False)
        with pytest.raises(ValueError, match="num_views"):
            cal.validate_for(checkpoint_path=ckpt, dims=four, text_trim=False)


def test_frontend_dims_without_the_camera_geometry_are_refused():
    """A frontend built by hand (`dims_override` without a workload) has no
    `num_views`/`image_h`/`image_w`; `identity_dims` then omits them and a
    version-3 file, which carries them, is refused with a diff naming the
    missing keys (file value, `None`) and how to fix the frontend."""
    hand_built = {k: v for k, v in _TWO_VIEWS.items() if k not in ("num_views", "image_h", "image_w")}
    assert set(identity_dims(hand_built)).isdisjoint({"num_views", "image_h", "image_w"})
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, path = _write_v3_file(tmp, _TWO_VIEWS)
        with pytest.raises(ValueError, match="dims differ") as e:
            load_calibration(path).validate_for(checkpoint_path=ckpt, dims=hand_built, text_trim=False)
        msg = str(e.value)
        for key, value in (("num_views", 2), ("image_h", 224), ("image_w", 224)):
            assert f"'{key}': ({value}, None)" in msg, msg
        assert "load_imagewam" in msg and "from_config" in msg and "dims_override" in msg
        # Adding the keys to dims_override is the other way out.
        load_calibration(path).validate_for(checkpoint_path=ckpt, dims=dict(hand_built, num_views=2,
                                                                          image_h=224, image_w=224),
                                            text_trim=False)


def test_text_trim_identity():
    """Version 3 records `text_trim`; the frontend's setting must match."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, path = _write_v3_file(tmp, _TWO_VIEWS, text_trim=True)
        back = load_calibration(path)
        assert back.version == 3 and back.text_trim is True
        back.validate_for(checkpoint_path=ckpt, dims=_TWO_VIEWS, text_trim=True)
        with pytest.raises(ValueError, match="text_trim"):
            back.validate_for(checkpoint_path=ckpt, dims=_TWO_VIEWS, text_trim=False)


@pytest.mark.parametrize("version", [1, 2])
def test_files_before_the_workload_identity_are_refused(version):
    """No compatibility with earlier artifacts: a version-1 or -2 file is
    refused by `load_calibration` with a message that names its version and
    the way to record a new one."""
    def as_old(meta):
        meta["version"] = version
        for k in ("num_views", "image_h", "image_w"):
            del meta["dims"][k]
        if version == 1:
            del meta["text_trim"]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt, path = _write_v3_file(tmp, _TWO_VIEWS)
        old = os.path.join(tmp, f"v{version}.safetensors")
        _rewrite_metadata(path, old, as_old)
        with pytest.raises(ValueError) as e:
            load_calibration(old)
        msg = str(e.value)
        assert f"version {version}" in msg and "predates the workload identity" in msg
        assert "benchmarks/imagewam_build_calibration.py" in msg and "re-record" in msg
        # The current file is unaffected.
        load_calibration(path).validate_for(checkpoint_path=ckpt, dims=_TWO_VIEWS, text_trim=False)


def test_unknown_future_version_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        _, path = _write_v3_file(tmp, _TWO_VIEWS)
        future = os.path.join(tmp, "v4.safetensors")
        _rewrite_metadata(path, future, lambda meta: meta.update(version=4))
        with pytest.raises(ValueError, match=r"version 4 not in \(3,\)"):
            load_calibration(future)
