"""Activation recording, calibration-file reduction/round trip, identity
checks, and `StaticFp8Linear.set_activation_scale` (roadmap item 7).

Small dims, random weights; the real build is
`benchmarks/imagewam_build_calibration.py`.
"""
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
    build_calibration, identity_dims, load_calibration, save_calibration,
)
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
    dims = dict(hidden=256, x0=3, a0=8, num_action=4, merge_qkv_mlp=True, shift=5.0, dt=0.5)
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
    b = StaticFp8Linear(w.data_ptr(), n, k)
    b.set_activation_scale(float(np.float32(x.float().abs().max().item()) / np.float32(448.0)))
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


def test_text_trim_identity_and_version_1_files():
    """Version 2 records `text_trim`; a version-1 file (no such entry,
    recorded untrimmed) loads as `text_trim=False`; either way the
    frontend's setting must match."""
    import json

    from safetensors import safe_open
    from safetensors.torch import save_file

    rng = np.random.default_rng(1)
    samples = _fake_samples({"backbone.single.0.linear1.weight": 32}, 3, rng)
    dims = dict(hidden=256, x0=3, a0=8, num_action=4, merge_qkv_mlp=True)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "model.pt")
        with open(ckpt, "wb") as f:
            f.write(os.urandom(100_000))
        trimmed = build_calibration(samples, percentile=99.9, checkpoint_path=ckpt, dims=dims,
                                    frames=[("libero_goal", 3, 10)], noise="test", text_trim=True)
        path = os.path.join(tmp, "trimmed.safetensors")
        save_calibration(trimmed, path)
        back = load_calibration(path)
        assert back.version == 2 and back.text_trim is True
        back.validate_for(checkpoint_path=ckpt, dims=dims, text_trim=True)
        with pytest.raises(ValueError, match="text_trim"):
            back.validate_for(checkpoint_path=ckpt, dims=dims, text_trim=False)

        # A version-1 file: the same content, the version-1 metadata (no text_trim entry).
        with safe_open(path, framework="pt") as f:
            meta = json.loads(f.metadata()["imagewam_calibration"])
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta["version"] = 1
        del meta["text_trim"]
        v1_path = os.path.join(tmp, "v1.safetensors")
        save_file(tensors, v1_path, metadata={"imagewam_calibration": json.dumps(meta)})
        v1 = load_calibration(v1_path)
        assert v1.version == 1 and v1.text_trim is False
        v1.validate_for(checkpoint_path=ckpt, dims=dims, text_trim=False)
        with pytest.raises(ValueError, match="text_trim"):
            v1.validate_for(checkpoint_path=ckpt, dims=dims, text_trim=True)
        print("version 2 records text_trim; version 1 loads as text_trim=False; mismatches refused")
