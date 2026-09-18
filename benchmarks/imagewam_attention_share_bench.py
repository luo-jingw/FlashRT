#!/usr/bin/env python
"""ImageWAM attention cost at the real shapes (roadmap item 6, plan.md
"Plan: attention-chain fusion recheck at ImageWAM's real shapes";
opportunities.md OPT-019).

`--part share`: in-graph attention share. One frontend at the real dims
(random weights, speed only). Four CUDA graphs are captured from it:
prefill and one denoise step, each with the frontend's real attention
backend and with a proxy that hands out the same slot pointers but
launches nothing. Every other kernel is identical, so on-minus-off is
the attention time inside the graph. The four graphs are replayed
interleaved and timed with CUDA events.

`--part kernels`: per-call attention kernels at the two sites' real
shapes, with real per-head K/V (24 heads, HD 128):
  backbone  q = kv = 905          (prefill self-attention, 25 calls per infer)
  mot       q = 64, kv = 969      (ActionDiT joint attention, unmasked
                                   real rule, 250 calls per infer)
Rows: the cuBLAS chain the pipeline runs today
(`attention_qkv_fp16_perhead`: QK^T GEMM -> logits -> softmax -> PV GEMM),
PyTorch SDPA with each fused backend this device supports (flash, cuDNN,
mem-efficient), and on Thor FA4 (`fa4_backend.fa4_fwd`, `num_splits`
swept). Each call reads a different layer's K/V (round robin over
`--layers` copies) and is CUDA-graph timed
(`gemm_variant_timer.CudaGraphVariantTimer`). Accuracy is reported
against an fp32 PyTorch reference: cosine, max-abs, rel_l2.

`--part infer` (needs an FA4 runtime, i.e. Thor): one frontend, one CUDA
graph per attention configuration, captured from the same buffers and
weights with a different `ImageWAMAttnBackend`: `chain` (cuBLAS chain at
both sites), `fa4_backbone` (FA4 at "backbone", the Thor default), and
`fa4_both` (FA4 at both sites, `use_fa4_mot=True`). Reports action cosine
and max-abs against `chain` on identical inputs, then `infer()`
P10/P50/P90 per configuration, rotating the order every iteration. Set
`CKPT_PATH` for real weights (random otherwise). On a device without FA4
(sm_90), `--sdpa-standin` puts PyTorch SDPA (its default fused backend)
behind the same `fa4_fwd` calling convention, so the same in-pipeline
A/B runs with an sm_90 fused kernel; those numbers say what a fused
kernel does to the pipeline on that device, not what FA4 does on Thor.

Usage:
  python benchmarks/imagewam_attention_share_bench.py                      # H100: fp16
  python benchmarks/imagewam_attention_share_bench.py --precision nvfp4    # Thor: shipped default
  python benchmarks/imagewam_attention_share_bench.py --part kernels
  CKPT_PATH=... python benchmarks/imagewam_attention_share_bench.py --part infer --precision nvfp4
  --fa4 auto|on|off and --fa4-mot select the frontend's FA4 settings for --part share.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend
from flash_rt.models.imagewam.gemm_variant_timer import CudaGraphVariantTimer
from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_step, imagewam_prefill

FP16 = torch.float16
NH, HD = 24, 128
REAL_DIMS = dict(
    hidden=3072, HD=HD, NH=NH, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=64, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=1.0 / 10, num_denoise_steps=10,
)


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


class _AttentionRemoved:
    """Attention-backend stand-in: real slot pointers, no attention launch."""

    def __init__(self, real) -> None:
        self._real = real

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        return self._real.get_slot_ptrs(site, layer_idx)

    def run(self, site: str, layer_idx: int, q_seq: int, *, kv_seq: int | None = None, stream: int = 0,
            state_nk: int | None = None, x0: int | None = None, a0: int | None = None) -> int:
        return 0


def _capture(fn: Callable[[int], None]) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn(s.cuda_stream)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn(s.cuda_stream)
    torch.cuda.synchronize()
    return g


def _frontend(precision: str, **kw):
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

    dims = dict(REAL_DIMS)
    ckpt = os.environ.get("CKPT_PATH")
    if ckpt:
        dims.update(ref_h=14, ref_w=28)
        kw["ckpt_path"] = ckpt
    return ImageWAMTorchFrontendThor(precision=precision, dims_override=dims, **kw)


def run_share(precision: str, fa4: str, fa4_mot: bool, iters: int) -> None:
    use_fa4 = {"auto": None, "on": True, "off": False}[fa4]
    try:
        fe = _frontend(precision, use_fa4=use_fa4, use_fa4_mot=fa4_mot)
    except RuntimeError as e:
        print(f"share: SKIP precision={precision} ({str(e)[:100]})")
        return
    fe.set_prompt("attention share")  # calibration (fp8_static*) + the frontend's own capture
    d = fe.dims
    real, removed = fe._attn, _AttentionRemoved(fe._attn)
    print(f"\n=== in-graph attention share, precision={precision}, backbone FA4="
          f"{getattr(real, '_use_fa4', False)}, mot FA4={getattr(real, '_use_fa4_mot', False)} ===")

    def prefill(attn):
        def fn(stream: int) -> None:
            imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, d, stream=stream, attn=attn,
                             mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
                             rope_table=fe._rope_table.data_ptr())
        return fn

    def step(attn):
        mod_double, mod_single = fe._action_mods[0]

        def fn(stream: int) -> None:
            imagewam_denoise_step(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, d, 0, stream=stream,
                                  attn=attn, mod_double=mod_double, mod_single=mod_single,
                                  head_mod=fe._head_mods[0], action_rope_table=fe._action_rope_table.data_ptr(),
                                  delta=None if fe._deltas is None else fe._deltas[0])
        return fn

    graphs = {
        "prefill_on": _capture(prefill(real)), "prefill_off": _capture(prefill(removed)),
        "step_on": _capture(step(real)), "step_off": _capture(step(removed)),
    }
    for g in graphs.values():
        for _ in range(3):
            g.replay()
    torch.cuda.synchronize()
    times: dict[str, list[float]] = {k: [] for k in graphs}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    names = list(graphs)
    for i in range(iters):
        for name in (names if i % 2 == 0 else names[::-1]):
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end))
    for name in names:
        t = times[name]
        print(f"{name:12s} P10={_pct(t, 0.1):8.3f}ms P50={_pct(t, 0.5):8.3f}ms P90={_pct(t, 0.9):8.3f}ms")
    n_layers = d["num_layers_double"] + d["num_layers_single"]
    for stage, calls in (("prefill", n_layers), ("step", n_layers)):
        on, off = _pct(times[f"{stage}_on"], 0.5), _pct(times[f"{stage}_off"], 0.5)
        print(f"{stage}: attention = {on - off:.3f}ms of {on:.3f}ms ({100 * (on - off) / on:.1f}%), "
              f"{1000 * (on - off) / calls:.1f}us per call over {calls} calls")
    steps = d["num_denoise_steps"]
    pre_on, pre_off = _pct(times["prefill_on"], 0.5), _pct(times["prefill_off"], 0.5)
    st_on, st_off = _pct(times["step_on"], 0.5), _pct(times["step_off"], 0.5)
    tot_on = pre_on + steps * st_on
    tot_att = (pre_on - pre_off) + steps * (st_on - st_off)
    print(f"prefill + {steps} steps: attention = {tot_att:.2f}ms of {tot_on:.2f}ms "
          f"({100 * tot_att / tot_on:.1f}%)")
    del graphs, fe
    torch.cuda.empty_cache()


@dataclass(frozen=True)
class SiteShape:
    site: str
    q: int
    kv: int


SITES = (SiteShape("backbone", 905, 905), SiteShape("mot", 64, 969))


def _accuracy(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    o, r = out.float().flatten(), ref.float().flatten()
    cos = float(o @ r / (o.norm() * r.norm() + 1e-12))
    return cos, float((o - r).abs().max()), float((o - r).norm() / (r.norm() + 1e-12))


def run_kernels(layers: int, samples: int) -> None:
    from flash_rt.hardware.thor import fa4_backend
    from torch.nn.attention import SDPBackend, sdpa_kernel

    ctx = fvk.FvkContext()
    timer = CudaGraphVariantTimer(reps=4, samples=samples, warmup=3)
    scale = 1.0 / HD ** 0.5
    fa4 = fa4_backend.fa4_fwd()
    print(f"\n=== per-call attention kernels, real per-head K/V, NH={NH} HD={HD}, "
          f"{layers} K/V layers round robin; FA4: {fa4_backend.status()} ===")
    for shape in SITES:
        torch.manual_seed(0)
        q = torch.randn(shape.q, NH * HD, dtype=FP16, device="cuda")
        ks = [torch.randn(shape.kv, NH * HD, dtype=FP16, device="cuda") for _ in range(layers)]
        vs = [torch.randn(shape.kv, NH * HD, dtype=FP16, device="cuda") for _ in range(layers)]
        logits = torch.empty(shape.q * NH, shape.kv + (shape.kv % 2), dtype=FP16, device="cuda")
        out = torch.empty(shape.q, NH * HD, dtype=FP16, device="cuda")

        def bhsd(t: torch.Tensor, s: int) -> torch.Tensor:
            return t.view(1, s, NH, HD).transpose(1, 2)

        qf, kf, vf = bhsd(q, shape.q).float(), bhsd(ks[0], shape.kv).float(), bhsd(vs[0], shape.kv).float()
        ref = torch.softmax((qf @ kf.transpose(-1, -2)) * scale, dim=-1) @ vf  # (1, NH, q, HD)
        ref = ref.transpose(1, 2).reshape(shape.q, NH * HD)

        rows: list[tuple[str, Callable[[int], None], Callable[[], torch.Tensor]]] = []

        def chain(stream: int, layer: int) -> None:
            fvk.attention_qkv_fp16_perhead(ctx, q.data_ptr(), ks[layer].data_ptr(), vs[layer].data_ptr(),
                                            logits.data_ptr(), out.data_ptr(), shape.q, shape.kv, NH, HD,
                                            scale, stream)

        def chain_once() -> torch.Tensor:
            chain(0, 0)
            torch.cuda.synchronize()
            return out.clone()

        rows.append(("cublas_chain", lambda s: [chain(s, i) for i in range(layers)], chain_once))

        for name, backend in (("sdpa_flash", SDPBackend.FLASH_ATTENTION),
                              ("sdpa_cudnn", SDPBackend.CUDNN_ATTENTION),
                              ("sdpa_efficient", SDPBackend.EFFICIENT_ATTENTION)):
            def sdpa(stream: int, layer: int, backend=backend) -> torch.Tensor:
                with sdpa_kernel([backend]):
                    return F.scaled_dot_product_attention(bhsd(q, shape.q), bhsd(ks[layer], shape.kv),
                                                          bhsd(vs[layer], shape.kv), scale=scale)

            def sdpa_once(sdpa=sdpa) -> torch.Tensor:
                o = sdpa(0, 0)
                torch.cuda.synchronize()
                return o.transpose(1, 2).reshape(shape.q, NH * HD)

            def sdpa_batch(stream: int, sdpa=sdpa) -> None:
                for i in range(layers):
                    sdpa(stream, i)

            rows.append((name, sdpa_batch, sdpa_once))

        if fa4 is not None:
            for splits in ((1,) if shape.site == "backbone" else (1, 2, 4)):
                def fa4_call(stream: int, layer: int, splits=splits) -> None:
                    fa4(q.view(1, shape.q, NH, HD), ks[layer].view(1, shape.kv, NH, HD),
                        vs[layer].view(1, shape.kv, NH, HD), causal=False, num_splits=splits,
                        pack_gqa=False, out=out.view(1, shape.q, NH, HD))

                def fa4_once(fa4_call=fa4_call) -> torch.Tensor:
                    fa4_call(0, 0)
                    torch.cuda.synchronize()
                    return out.clone()

                def fa4_batch(stream: int, fa4_call=fa4_call) -> None:
                    # FA4 launches on PyTorch's current stream, which the timer
                    # has already set to `stream`.
                    for i in range(layers):
                        fa4_call(stream, i)

                rows.append((f"fa4_splits{splits}", fa4_batch, fa4_once))

        print(f"\n--- {shape.site}: q={shape.q} kv={shape.kv} ---")
        print(f"{'kernel':16s} {'us/call':>9s} {'vs chain':>9s} {'cos_fp32':>9s} {'max_abs':>9s} {'rel_l2':>9s}")
        ok_rows = []
        for name, batch, once in rows:
            try:
                acc = _accuracy(once(), ref)
                ok_rows.append((name, batch, acc))
            except RuntimeError as e:
                print(f"{name:16s} SKIP ({str(e).splitlines()[0][:80]})")
        times = timer.us_per_launch([b for _, b, _ in ok_rows], layers)
        chain_us = times[0]
        for (name, _, (cos, mx, rl2)), us in zip(ok_rows, times):
            print(f"{name:16s} {us:9.2f} {chain_us / us:9.3f} {cos:9.6f} {mx:9.2e} {rl2:9.2e}")
        del ks, vs
        torch.cuda.empty_cache()


def _sdpa_as_fa4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool, num_splits: int,
                 pack_gqa: bool, out: torch.Tensor, softmax_scale: float | None = None):
    """PyTorch SDPA behind FA4's `_flash_attn_fwd` calling convention
    ((B, S, H, D) tensors, `out=`), for `--sdpa-standin` only."""
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       is_causal=causal, scale=softmax_scale)
    out.copy_(o.transpose(1, 2))
    return out, None


def run_infer(precision: str, warmup: int, iters: int, sdpa_standin: bool) -> None:
    from flash_rt.hardware.thor import fa4_backend

    print(f"\n=== infer() A/B over attention configurations, precision={precision} ===")
    if sdpa_standin:
        print("fused kernel: PyTorch SDPA stand-in (NOT FA4)")
        fa4_backend.fa4_fwd = lambda: _sdpa_as_fa4
    if fa4_backend.fa4_fwd() is None:
        print(f"SKIP: no FA4 runtime ({fa4_backend.status()})")
        return
    try:
        fe = _frontend(precision, use_fa4=False)
    except RuntimeError as e:
        print(f"SKIP precision={precision} ({str(e)[:100]})")
        return
    fe.set_prompt("attention A/B")
    base = fe._attn
    configs = {"chain": (False, False), "fa4_backbone": (True, False), "fa4_both": (True, True)}
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name, (use_fa4, use_fa4_mot) in configs.items():
        fe._attn = ImageWAMAttnBackend(
            base._spec, fe._ctx, backbone_slots=dict(base._slots["backbone"]),
            mot_slots=dict(base._slots["mot"]), use_fa4=use_fa4, use_perhead_kv=True,
            use_real_mot_mask=True, use_fa4_mot=use_fa4_mot)
        fe._capture_graph()
        graphs[name] = fe._graph

    gen = torch.Generator(device="cuda").manual_seed(0)
    img = torch.randn(fe._img_raw.shape, generator=gen, device="cuda").to(fe._img_raw.dtype)
    noise = torch.randn(fe._action_latent.shape, generator=gen, device="cuda").to(fe._action_latent.dtype)
    outs = {}
    for name, g in graphs.items():
        fe._img_raw.copy_(img)
        fe._action_latent.copy_(noise)
        g.replay()
        torch.cuda.synchronize()
        outs[name] = fe._action_latent.detach().float().clone()
    for name in graphs:
        o, r = outs[name].flatten(), outs["chain"].flatten()
        cos = float(o @ r / (o.norm() * r.norm() + 1e-12))
        print(f"actions {name:13s} vs chain: cosine={cos:.6f} max_abs={(o - r).abs().max().item():.3e} "
              f"finite={bool(torch.isfinite(o).all())}")

    names = list(graphs)
    obs: dict = {}
    for _ in range(warmup):
        for name in names:
            fe._graph = graphs[name]
            fe.infer(obs)
    torch.cuda.synchronize()
    times: dict[str, list[float]] = {n: [] for n in names}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(iters):
        order = names[i % len(names):] + names[:i % len(names)]
        for name in order:
            fe._graph = graphs[name]
            start.record()
            fe.infer(obs)
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end))
    chain_p50 = _pct(times["chain"], 0.5)
    for name in names:
        t = times[name]
        p50 = _pct(t, 0.5)
        print(f"infer() {name:13s} P10={_pct(t, 0.1):8.2f}ms P50={p50:8.2f}ms P90={_pct(t, 0.9):8.2f}ms "
              f"delta vs chain {p50 - chain_p50:+7.2f}ms")
    del graphs, fe
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description="ImageWAM attention cost at the real shapes")
    ap.add_argument("--part", choices=("share", "kernels", "infer", "all"), default="all")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--fa4", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--fa4-mot", action="store_true")
    ap.add_argument("--sdpa-standin", action="store_true",
                    help="--part infer on a device without FA4: PyTorch SDPA stands in for FA4")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--samples", type=int, default=25)
    args = ap.parse_args()
    print(f"device: {torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
          f"torch={torch.__version__}")
    if args.part in ("kernels", "all"):
        run_kernels(args.layers, args.samples)
    if args.part in ("share", "all"):
        run_share(args.precision, args.fa4, args.fa4_mot, args.iters)
    if args.part in ("infer", "all"):
        run_infer(args.precision, args.warmup, args.iters, args.sdpa_standin)


if __name__ == "__main__":
    main()
