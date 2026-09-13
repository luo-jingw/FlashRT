#!/usr/bin/env python
"""ImageWAM Thor structural-dry-run speed benchmark (plan.md, all phases).

Measures steady-state per-layer-type latency at ImageWAM's REAL
per-head geometry (NH=24, HD=128, backbone hidden=3072/mlp=9216,
ActionDiT hidden=1024/attn_width=3072/mlp=4096 -- from
`_imagewam_thor_spec.py`'s confirmed FLUX.2-klein-4B/ActionDiT config
values), then derives whole-pipeline totals by simple arithmetic
(layer count x per-layer time). This is NOT accuracy validation --
random weights, explicitly out of scope per PROJECT.md -- and NOT a
Thor number: this runs on the project's own Ada (sm_89) 8GB GPU.

Deliberately does NOT allocate the full 25-layer weight set at once
(~5.5GB of random FP16 weights at real dims) -- this machine has ~6.7GB
free on an 8GB shared laptop GPU. Instead, each layer TYPE (backbone
double-stream, backbone single-stream, ActionDiT double-stream,
ActionDiT single-stream) is benchmarked in isolation with a 1-layer
attention spec, called repeatedly at layer_idx=0 -- valid because
every layer of the same type has identical shapes and therefore
identical steady-state cost (confirmed by GemmRunner's own
per-(op,M,N,K) cache: repeated identical-shape calls hit the same
cached cuBLASLt algorithm).

Sequence lengths are a representative choice, not a confirmed real
ImageWAM deployment value (that number was never established in this
project's research) -- kept at total=960 (a0=896 + num_action=64,
64 = max_action_horizon from _imagewam_thor_spec.py) to stay under
softmax_mot_joint_fp16's confirmed 1024-column ceiling (see
csrc/kernels/softmax.cu: SM_MAX_COLS=1024 -- columns beyond that are
silently never read by the single-warp-per-row reduction loop, which
would make a >1024 timing number meaningless: the kernel would be
doing LESS work than a correct implementation needs, not more).
"""
from __future__ import annotations

import statistics
import time

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.pipeline_thor import (
    _action_double_layer,
    _action_single_layer,
    _double_stream_layer,
    _single_stream_layer,
)

DEV = "cuda"
FP16 = torch.float16

# Real confirmed dims (_imagewam_thor_spec.py).
HIDDEN, HD, NH, MLP_HIDDEN, JOINT_ATTN_DIM = 3072, 128, 24, 9216, 7680
ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH, ACTION_MLP_HIDDEN = 1024, 3072, 4096
NUM_DOUBLE, NUM_SINGLE = 5, 20
MAX_ACTION_HORIZON = 64

# Representative (not confirmed-real) sequence lengths, see module docstring.
X0, A0 = 128, 896
NUM_ACTION = MAX_ACTION_HORIZON
TOTAL = A0 + NUM_ACTION  # 960, under the 1024 softmax ceiling

WARMUP, ITERS = 15, 50

_keepalive = []


def _rand(*shape):
    t = torch.randn(*shape, dtype=FP16, device=DEV)
    _keepalive.append(t)
    return t


def _zeros(*shape):
    t = torch.zeros(*shape, dtype=FP16, device=DEV)
    _keepalive.append(t)
    return t


def _ones(*shape):
    t = torch.ones(*shape, dtype=FP16, device=DEV)
    _keepalive.append(t)
    return t


def _time_ms(fn, warmup=WARMUP, iters=ITERS) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    return p50, p90, statistics.mean(times)


def _make_1layer_backend(*, kind: str):
    """A 1-layer AttentionSpec/backend for isolated per-layer-type timing."""
    spec = make_imagewam_attention_spec(max_prefix_seq=A0, max_total_seq=TOTAL)
    # Shrink both sites to 1 layer -- this micro-benchmark only ever
    # calls layer_idx=0.
    spec.sites["backbone"].num_layers = 1
    spec.sites["mot"].num_layers = 1
    ctx = fvk.FvkContext()
    max_seq = A0 if kind == "backbone" else TOTAL
    K_cache = _zeros(1, max_seq, HD)
    V_cache = _zeros(1, max_seq, HD)
    Q_O = _zeros(max_seq, HIDDEN)
    logits = _zeros(max_seq * NH, max_seq + (max_seq % 2))
    backend = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
        },
        mot_slots={
            "Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
            "layer_stride": K_cache[0].numel() * 2,
        },
    )
    return ctx, backend


def bench_backbone_double():
    ctx, attn = _make_1layer_backend(kind="backbone")
    gemm = fvk.GemmRunner()
    weights = {}
    weights[("backbone", "double", 0, "txt_in")] = _rand(JOINT_ATTN_DIM, HIDDEN).data_ptr()
    for prefix in ("txt", "img"):
        weights[("backbone", "double", 0, f"{prefix}_q")] = _rand(HIDDEN, HIDDEN).data_ptr()
        weights[("backbone", "double", 0, f"{prefix}_k")] = _rand(HIDDEN, HD).data_ptr()
        weights[("backbone", "double", 0, f"{prefix}_v")] = _rand(HIDDEN, HD).data_ptr()
        weights[("backbone", "double", 0, f"{prefix}_proj")] = _rand(HIDDEN, HIDDEN).data_ptr()
        weights[("backbone", "double", 0, f"{prefix}_mlp0")] = _rand(HIDDEN, MLP_HIDDEN).data_ptr()
        weights[("backbone", "double", 0, f"{prefix}_mlp2")] = _rand(MLP_HIDDEN, HIDDEN).data_ptr()
    dims = dict(hidden=HIDDEN, HD=HD, mlp_hidden=MLP_HIDDEN,
                joint_attention_dim=JOINT_ATTN_DIM, x0=X0, a0=A0)
    bufs = {
        "context": _rand(X0, JOINT_ATTN_DIM).data_ptr(),
        "backbone_hidden": _rand(A0, HIDDEN).data_ptr(),
        "normed_scratch": _zeros(A0, HIDDEN).data_ptr(),
        "norm_ones": _ones(HIDDEN).data_ptr(),
        "txt_mlp_hidden": _zeros(X0, MLP_HIDDEN).data_ptr(),
        "img_mlp_hidden": _zeros(A0 - X0, MLP_HIDDEN).data_ptr(),
        "proj_scratch": _zeros(A0, HIDDEN).data_ptr(),
    }

    def run():
        _double_stream_layer(ctx, fvk, gemm, bufs, weights, dims, 0, 0, attn)

    return _time_ms(run)


def bench_backbone_single():
    ctx, attn = _make_1layer_backend(kind="backbone")
    gemm = fvk.GemmRunner()
    weights = {
        ("backbone", "single", 0, "q"): _rand(HIDDEN, HIDDEN).data_ptr(),
        ("backbone", "single", 0, "k"): _rand(HIDDEN, HD).data_ptr(),
        ("backbone", "single", 0, "v"): _rand(HIDDEN, HD).data_ptr(),
        ("backbone", "single", 0, "mlp_in"): _rand(HIDDEN, MLP_HIDDEN).data_ptr(),
        ("backbone", "single", 0, "attn_out_proj"): _rand(HIDDEN, HIDDEN).data_ptr(),
        ("backbone", "single", 0, "mlp_down"): _rand(MLP_HIDDEN, HIDDEN).data_ptr(),
    }
    dims = dict(hidden=HIDDEN, HD=HD, mlp_hidden=MLP_HIDDEN, a0=A0)
    bufs = {
        "backbone_hidden": _rand(A0, HIDDEN).data_ptr(),
        "normed_scratch": _zeros(A0, HIDDEN).data_ptr(),
        "norm_ones": _ones(HIDDEN).data_ptr(),
        "single_mlp_hidden": _zeros(A0, MLP_HIDDEN).data_ptr(),
        "proj_scratch": _zeros(A0, HIDDEN).data_ptr(),
    }

    def run():
        _single_stream_layer(ctx, fvk, gemm, bufs, weights, dims, 0, 0, 0, attn)

    return _time_ms(run)


def bench_action_double():
    ctx, attn = _make_1layer_backend(kind="mot")
    gemm = fvk.GemmRunner()
    weights = {
        ("action_dit", "double", 0, "q"): _rand(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH).data_ptr(),
        ("action_dit", "double", 0, "k"): _rand(ACTION_HIDDEN_DIM, HD).data_ptr(),
        ("action_dit", "double", 0, "v"): _rand(ACTION_HIDDEN_DIM, HD).data_ptr(),
        ("action_dit", "double", 0, "proj"): _rand(ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM).data_ptr(),
        ("action_dit", "double", 0, "mlp0"): _rand(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN).data_ptr(),
        ("action_dit", "double", 0, "mlp2"): _rand(ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM).data_ptr(),
    }
    dims = dict(action_hidden_dim=ACTION_HIDDEN_DIM, action_attn_width=ACTION_ATTN_WIDTH,
                HD=HD, action_mlp_hidden=ACTION_MLP_HIDDEN, x0=X0, a0=A0,
                num_action=NUM_ACTION, total=TOTAL)
    bufs = {
        "action_hidden": _rand(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_normed": _zeros(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_norm_ones": _ones(ACTION_HIDDEN_DIM).data_ptr(),
        "action_proj_scratch": _zeros(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_mlp_hidden": _zeros(NUM_ACTION, ACTION_MLP_HIDDEN).data_ptr(),
    }

    def run():
        _action_double_layer(ctx, fvk, gemm, bufs, weights, dims, 0, 0, 0, attn)

    return _time_ms(run)


def bench_action_single():
    ctx, attn = _make_1layer_backend(kind="mot")
    gemm = fvk.GemmRunner()
    weights = {
        ("action_dit", "single", 0, "q"): _rand(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH).data_ptr(),
        ("action_dit", "single", 0, "k"): _rand(ACTION_HIDDEN_DIM, HD).data_ptr(),
        ("action_dit", "single", 0, "v"): _rand(ACTION_HIDDEN_DIM, HD).data_ptr(),
        ("action_dit", "single", 0, "mlp_in"): _rand(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN).data_ptr(),
        ("action_dit", "single", 0, "attn_out_proj"): _rand(ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM).data_ptr(),
        ("action_dit", "single", 0, "mlp_down"): _rand(ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM).data_ptr(),
    }
    dims = dict(action_hidden_dim=ACTION_HIDDEN_DIM, action_attn_width=ACTION_ATTN_WIDTH,
                HD=HD, action_mlp_hidden=ACTION_MLP_HIDDEN, x0=X0, a0=A0,
                num_action=NUM_ACTION, total=TOTAL)
    bufs = {
        "action_hidden": _rand(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_normed": _zeros(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_norm_ones": _ones(ACTION_HIDDEN_DIM).data_ptr(),
        "action_proj_scratch": _zeros(NUM_ACTION, ACTION_HIDDEN_DIM).data_ptr(),
        "action_mlp_hidden": _zeros(NUM_ACTION, ACTION_MLP_HIDDEN).data_ptr(),
    }

    def run():
        _action_single_layer(ctx, fvk, gemm, bufs, weights, dims, 0, 0, 0, attn)

    return _time_ms(run)


def bench_mot_joint_kernel_only():
    ctx = fvk.FvkContext()
    q = _rand(TOTAL * NH, HD)
    k = _rand(TOTAL, HD)
    v = _rand(TOTAL, HD)
    total_pad = TOTAL + (TOTAL % 2)
    logits = _zeros(TOTAL * NH, total_pad)
    out = _zeros(TOTAL * NH, HD)
    scale = 1.0 / (HD ** 0.5)

    def run():
        fvk.attention_qkv_fp16_mot_joint(ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                          logits.data_ptr(), out.data_ptr(),
                                          TOTAL, NH, HD, A0, A0, scale, 0)

    return _time_ms(run)


def bench_standard_attn_kernel_only():
    ctx = fvk.FvkContext()
    q = _rand(A0 * NH, HD)
    k = _rand(A0, HD)
    v = _rand(A0, HD)
    logits = _zeros(A0 * NH, A0)
    out = _zeros(A0 * NH, HD)
    scale = 1.0 / (HD ** 0.5)

    def run():
        fvk.attention_qkv_fp16(ctx, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                logits.data_ptr(), out.data_ptr(),
                                A0, A0, NH, HD, scale, 0)

    return _time_ms(run)


def main():
    print(f"Dims: hidden={HIDDEN} HD={HD} NH={NH} mlp_hidden={MLP_HIDDEN} "
          f"| action_hidden_dim={ACTION_HIDDEN_DIM} action_attn_width={ACTION_ATTN_WIDTH} "
          f"action_mlp_hidden={ACTION_MLP_HIDDEN}")
    print(f"Seq: x0={X0} a0={A0} num_action={NUM_ACTION} total={TOTAL} "
          f"(representative, not a confirmed real ImageWAM deployment length)")
    print(f"Warmup={WARMUP} iters={ITERS}, CUDA-event timing, P50/P90/mean in ms\n")

    results = {}
    for name, fn in [
        ("backbone_double_layer", bench_backbone_double),
        ("backbone_single_layer", bench_backbone_single),
        ("action_double_layer", bench_action_double),
        ("action_single_layer", bench_action_single),
        ("mot_joint_kernel_only", bench_mot_joint_kernel_only),
        ("standard_attn_kernel_only", bench_standard_attn_kernel_only),
    ]:
        p50, p90, mean = fn()
        results[name] = p50
        print(f"{name:28s} P50={p50:8.3f}  P90={p90:8.3f}  mean={mean:8.3f}")
        _keepalive.clear()
        torch.cuda.empty_cache()

    backbone_total = NUM_DOUBLE * results["backbone_double_layer"] + \
        NUM_SINGLE * results["backbone_single_layer"]
    action_step_total = NUM_DOUBLE * results["action_double_layer"] + \
        NUM_SINGLE * results["action_single_layer"]

    print(f"\nDerived (layer_count x per-layer P50, NOT a single measured run):")
    print(f"  backbone prefill total ({NUM_DOUBLE} double + {NUM_SINGLE} single): "
          f"{backbone_total:.2f} ms")
    print(f"  one ActionDiT denoise step ({NUM_DOUBLE} double + {NUM_SINGLE} single): "
          f"{action_step_total:.2f} ms")
    for n_steps in (1, 4, 10):
        print(f"  prefill + {n_steps}-step denoise loop: "
              f"{backbone_total + n_steps * action_step_total:.2f} ms")


if __name__ == "__main__":
    main()
