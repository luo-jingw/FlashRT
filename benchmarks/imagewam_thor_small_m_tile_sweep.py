#!/usr/bin/env python
"""ActionDiT small-M GEMM tile sweep and `infer()` A/B (Thor, sm_110).

Roadmap item 1 (plan.md, "Plan: ActionDiT small-M CUTLASS tile
selection"; opportunities.md OPT-018). Two parts:

`--part kernels` -- every ActionDiT GEMM shape the served pipeline runs
at M = 64 (merged single-stream linear1 and linear2):

  site                                   N      K   layers sharing it
  double qkv                          9216   1024    5
  double proj                         1024   3072    5
  double mlp0 (merged gate/up)        8192   1024    5
  double mlp2                         1024   4096    5
  single linear1 (qkv + gate/up)     17408   1024   20
  single linear2 (attn out + down)    1024   7168   20

  For each shape it builds as many distinct random weights as real layers
  share the shape, then for every NVFP4 variant (all `cutlass_fp4_gemm_variant`
  indices) and every FP8 CUTLASS variant (`quant_linear.FP8_CUTLASS_VARIANTS`,
  including the 1-SM small-M tiles) it reports:
    - us/GEMM: the GEMM launch alone (activation quantization, identical
      across a family's variants, is excluded), CUDA-graph timed, one
      launch per weight round robin (cold weights, as in the pipeline),
      interleaved across variants, median;
    - cos_fp16: cosine of layer 0's output against `x @ W` in fp32;
    - which variant the (N, K) heuristic picks today (`*`) and which one
      `GemmVariantTuner` (the `gemm_variant_autotune=True` rule) picks (`T`).
  cuBLASLt fp16 (`fp16`) and cuBLASLt FP8 (`fp8_static`'s
  `fp8_gemm_descale_fp16`, or `_tn` off Blackwell) are timed on the same weights as references.

`--part infer` -- for `nvfp4` and `fp8_static_cutlass`: one frontend
  built with `gemm_variant_autotune=True` at the real dims, two CUDA graphs
  captured from it (ActionDiT on the heuristic tiles, and on the tuned
  tiles), same buffers and weights. Reports the action cosine between the
  two graphs on identical inputs, then `infer()` P10/P50/P90 for each,
  alternating old/new every iteration in the same process.

Real weights: set `CKPT_PATH` (same layout as
`benchmarks/imagewam_real_checkpoint_validation.py`). Without it the
frontend uses random weights, which is fine for speed but makes the
action cosine a random-weight number.

Usage on Thor:
  python benchmarks/imagewam_thor_small_m_tile_sweep.py                # both parts
  python benchmarks/imagewam_thor_small_m_tile_sweep.py --part kernels
  CKPT_PATH=... python benchmarks/imagewam_thor_small_m_tile_sweep.py --part infer --iters 60

On a build without SM100 CUTLASS / NVFP4 (e.g. sm_90) every quantized
family prints SKIP; only the cuBLASLt fp16 reference runs.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.imagewam.gemm_variant_timer import CudaGraphVariantTimer
from flash_rt.models.imagewam.gemm_variant_tuner import GemmVariantTuner
from flash_rt.models.imagewam.libero_dims import LIBERO_REAL_DIMS
from flash_rt.models.imagewam.quant_linear import (
    FP8_CUTLASS_VARIANTS,
    Nvfp4Linear,
    StaticFp8Linear,
)

FP16 = torch.float16
# The ActionDiT GEMM M this sweep runs at -- the served action horizon.
M = LIBERO_REAL_DIMS["num_action"]
# The served dims minus the entries this structural run leaves at the
# frontend's own defaults: `proprio_dim` (proprio conditioning), `shift` and
# `num_train_timesteps` (the real timestep schedule). `ref_h`/`ref_w` (the
# real 2D image RoPE grid) are added by `_frontend()` only when a checkpoint
# is given.
_DEFAULTED_DIM_KEYS = ("ref_h", "ref_w", "proprio_dim", "shift", "num_train_timesteps")
REAL_DIMS = {key: value for key, value in dict(LIBERO_REAL_DIMS, num_action=M).items()
             if key not in _DEFAULTED_DIM_KEYS}
FP8_TILES = {
    "sq": "256x256x128 c2x2x1", "wide": "256x128x128 c2x2x1", "t1": "128x256x128 c2x1x1 2SM",
    "plain": "256x128x64 c2x2x1", "t128x64x256": "128x64x256 c1x1x1 (v10 shape)",
    "t128x64x128": "128x64x128 c1x1x1", "t128x128x128": "128x128x128 c1x1x1",
    "t128x256x128": "128x256x128 c1x1x1",
}


@dataclass(frozen=True)
class ActionShape:
    site: str
    n: int
    k: int
    layers: int


SHAPES = (
    ActionShape("double qkv", 9216, 1024, 5),
    ActionShape("double proj", 1024, 3072, 5),
    ActionShape("double mlp0", 8192, 1024, 5),
    ActionShape("double mlp2", 1024, 4096, 5),
    ActionShape("single linear1", 17408, 1024, 20),
    ActionShape("single linear2", 1024, 7168, 20),
)


@dataclass(frozen=True)
class Row:
    family: str
    variant: str
    tile: str
    us: float | None
    cos_fp16: float | None
    status: str


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _sweep_family(family: str, lins: list, variants: list[str], tiles: dict[str, str],
                  x: torch.Tensor, ref0: torch.Tensor, timer: CudaGraphVariantTimer) -> list[Row]:
    n = lins[0].n
    for lin in lins:
        lin.prepare_tuning_input(x.data_ptr(), M, 0)
    torch.cuda.synchronize()
    out = torch.empty(M, n, dtype=FP16, device="cuda")
    ok: list[str] = []
    rows: dict[str, Row] = {}
    for v in variants:
        out.zero_()
        try:
            rc = lins[0].launch_variant(v, out.data_ptr(), M, 0)
            torch.cuda.synchronize()
        except Exception as e:  # e.g. a kernel symbol this build does not export
            rows[v] = Row(family, v, tiles.get(v, ""), None, None, f"raised {type(e).__name__}")
            continue
        if rc != 0:
            rows[v] = Row(family, v, tiles.get(v, ""), None, None, f"rc={rc:#x}")
            continue
        finite = bool(torch.isfinite(out).all())
        rows[v] = Row(family, v, tiles.get(v, ""), None, _cos(out, ref0), "ok" if finite else "nonfinite")
        if finite:
            ok.append(v)

    def make(v: str):
        def batch(stream: int) -> None:
            for lin in lins:
                lin.launch_variant(v, out.data_ptr(), M, stream)
        return batch

    times = timer.us_per_launch([make(v) for v in ok], len(lins))
    for v, t in zip(ok, times):
        r = rows[v]
        rows[v] = Row(r.family, r.variant, r.tile, t, r.cos_fp16, r.status if t is not None else "timing_failed")
    return [rows[v] for v in variants]


def _reference_rows(weights: list[torch.Tensor], x: torch.Tensor, ref0: torch.Tensor,
                    timer: CudaGraphVariantTimer) -> list[Row]:
    n, k = weights[0].shape[1], weights[0].shape[0]
    rows: list[Row] = []
    gemm = fvk.GemmRunner()
    xs = torch.zeros(M, k, dtype=FP16, device="cuda")
    ws = torch.zeros(k, n, dtype=FP16, device="cuda")
    out = torch.empty(M, n, dtype=FP16, device="cuda")
    gemm.autotune_fp16_nn(xs.data_ptr(), ws.data_ptr(), out.data_ptr(), M, n, k, 16)
    gemm.fp16_nn(x.data_ptr(), weights[0].data_ptr(), out.data_ptr(), M, n, k, 0)
    torch.cuda.synchronize()
    cos16 = _cos(out, ref0)

    def fp16_batch(stream: int) -> None:
        for w in weights:
            gemm.fp16_nn(x.data_ptr(), w.data_ptr(), out.data_ptr(), M, n, k, stream)

    (t16,) = timer.us_per_launch([fp16_batch], len(weights))
    rows.append(Row("cublaslt", "fp16", "cuBLASLt autotuned", t16, cos16, "ok"))

    try:
        lins = [StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False) for w in weights]
        for lin in lins:
            lin.calibrate(x.data_ptr(), M, 0)
        lins[0](x.data_ptr(), out.data_ptr(), M, 0)
        torch.cuda.synchronize()
        cos8 = _cos(out, ref0)

        for lin in lins[1:]:
            lin(x.data_ptr(), out.data_ptr(), M, 0)  # stages each op's quantized activation
        torch.cuda.synchronize()

        # Each op's own cuBLASLt entry point: the weight storage (w_f8)
        # depends on the op's layout (NN on Blackwell, TN below; see
        # quant_linear.fp8_cublaslt_layout), and only that function reads it.
        def fp8_batch(stream: int) -> None:
            for lin in lins:
                lin._gemm_fn(lin.act_f8.data_ptr(), lin.w_f8.data_ptr(), out.data_ptr(),
                             M, n, k, lin.act_scale.data_ptr(), lin.w_scale.data_ptr(), stream)

        label = f"cuBLASLt FP8 ({lins[0].layout.upper()}, fp8_gemm_descale_fp16{'' if lins[0].layout == 'nn' else '_tn'})"
        (t8,) = timer.us_per_launch([fp8_batch], len(lins))
        rows.append(Row("cublaslt", "fp8_static", label, t8, cos8, "ok"))
    except RuntimeError as e:
        rows.append(Row("cublaslt", "fp8_static", "cuBLASLt FP8 (fp8_gemm_descale_fp16[_tn])", None, None,
                        f"SKIP {str(e)[:60]}"))
    return rows


def run_kernels(timer: CudaGraphVariantTimer) -> list[dict]:
    summary: list[dict] = []
    for shape in SHAPES:
        torch.manual_seed(0)
        weights = [torch.randn(shape.k, shape.n, dtype=FP16, device="cuda") * 0.02
                   for _ in range(shape.layers)]
        x = torch.randn(M, shape.k, dtype=FP16, device="cuda")
        ref0 = (x.float() @ weights[0].float())
        print(f"\n=== {shape.site}: M={M} N={shape.n} K={shape.k} ({shape.layers} layers) ===")
        rows = _reference_rows(weights, x, ref0, timer)
        picks: dict[str, tuple[str, str]] = {}

        try:
            nv = [Nvfp4Linear(w.data_ptr(), shape.n, shape.k) for w in weights]
            import flash_rt.flash_rt_fp4 as fvk_fp4
            count = int(fvk_fp4.cutlass_fp4_gemm_num_variants())
            tiles = {f"v{i}": fvk_fp4.cutlass_fp4_gemm_variant_name(i) for i in range(count)}
            rows += _sweep_family("nvfp4", nv, [f"v{i}" for i in range(count)], tiles, x, ref0, timer)
            tuned = GemmVariantTuner(timer).tune(nv, M)
            picks["nvfp4"] = (nv[0].default_variant, tuned.chosen_variant)
            del nv
        except RuntimeError as e:
            rows.append(Row("nvfp4", "-", "", None, None, f"SKIP {str(e)[:60]}"))

        try:
            f8 = [StaticFp8Linear(w.data_ptr(), shape.n, shape.k, use_cutlass=True) for w in weights]
            rows += _sweep_family("fp8_cutlass", f8, list(FP8_CUTLASS_VARIANTS), FP8_TILES, x, ref0, timer)
            tuned = GemmVariantTuner(timer).tune(f8, M)
            picks["fp8_cutlass"] = (f8[0].default_variant, tuned.chosen_variant)
            del f8
        except RuntimeError as e:
            rows.append(Row("fp8_cutlass", "-", "", None, None, f"SKIP {str(e)[:60]}"))

        base = {fam: next((r.us for r in rows if r.family == fam and r.variant == d), None)
                for fam, (d, _) in picks.items()}
        print(f"{'family':12s} {'variant':14s} {'tile':44s} {'us/GEMM':>9s} {'vs dflt':>8s} "
              f"{'cos_fp16':>9s}  status")
        for r in rows:
            mark = ""
            if r.family in picks:
                d, t = picks[r.family]
                mark = ("*" if r.variant == d else "") + ("T" if r.variant == t else "")
            us = f"{r.us:9.2f}" if r.us is not None else f"{'-':>9s}"
            b = base.get(r.family)
            rel = f"{b / r.us:8.3f}" if (b is not None and r.us) else f"{'':>8s}"
            cs = f"{r.cos_fp16:9.6f}" if r.cos_fp16 is not None else f"{'-':>9s}"
            print(f"{r.family:12s} {(r.variant + mark):14s} {r.tile[:44]:44s} {us} {rel} {cs}  {r.status}")
        for fam in ("nvfp4", "fp8_cutlass"):
            timed = [r for r in rows if r.family == fam and r.us is not None]
            if not timed:
                continue
            best = min(timed, key=lambda r: r.us)
            d, t = picks.get(fam, ("?", "?"))
            b = base.get(fam)
            vs = f"{b:.2f}us, {b / best.us:.3f}x" if b is not None else "not timed"
            print(f"winner {fam}: {best.variant} {best.us:.2f}us (heuristic {d}: {vs}); tuner picks {t}")
            summary.append(dict(site=shape.site, n=shape.n, k=shape.k, family=fam, heuristic=d,
                                heuristic_us=None if b is None else round(b, 3), winner=best.variant,
                                winner_us=round(best.us, 3), tuner=t))
        ref_fp16 = next(r for r in rows if r.variant == "fp16")
        summary.append(dict(site=shape.site, n=shape.n, k=shape.k, family="cublaslt_fp16",
                            us=None if ref_fp16.us is None else round(ref_fp16.us, 3)))
        del weights
        torch.cuda.empty_cache()
    print("\nSUMMARY_JSON " + json.dumps(summary))
    return summary


def _frontend(precision: str):
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

    dims = dict(REAL_DIMS)
    kwargs = {}
    ckpt = os.environ.get("CKPT_PATH")
    if ckpt:
        dims.update(ref_h=LIBERO_REAL_DIMS["ref_h"], ref_w=LIBERO_REAL_DIMS["ref_w"])
        kwargs["ckpt_path"] = ckpt
    return ImageWAMTorchFrontendThor(precision=precision, dims_override=dims,
                                     gemm_variant_autotune=True, **kwargs), bool(ckpt)


def _set_action_variants(fe, tuned: bool) -> int:
    changed = 0
    for key, lin in fe._weights.items():
        if key[0] == "action_dit" and fe._is_variant_tunable(lin):
            chosen = None
            for r in fe.gemm_variant_results:
                if (r.family, r.shape.n, r.shape.k) == (lin.family, lin.n, lin.k):
                    chosen = r.chosen_variant
            target = chosen if tuned else lin.default_variant
            changed += int(target != lin.default_variant)
            lin.set_variant(target)
    return changed


def run_infer(precision: str, warmup: int, iters: int) -> None:
    print(f"\n=== infer() A/B, precision={precision} ===")
    try:
        fe, real = _frontend(precision)
    except RuntimeError as e:
        print(f"SKIP ({type(e).__name__}: {str(e)[:100]})")
        return
    for r in fe.gemm_variant_results:
        print("  " + r.summary())
    print(f"weights: {'real checkpoint' if real else 'random'}")
    fe.set_prompt("tile sweep")  # calibrates (fp8_static_cutlass) and captures with the tuned tiles
    graph_new = fe._graph
    n_changed = _set_action_variants(fe, tuned=False)
    fe._capture_graph()
    graph_old = fe._graph
    _set_action_variants(fe, tuned=True)
    print(f"ActionDiT linears whose tile differs between the two graphs: {n_changed}")

    gen = torch.Generator(device="cuda").manual_seed(0)
    img = torch.randn(fe._img_raw.shape, generator=gen, device="cuda").to(fe._img_raw.dtype)
    noise = torch.randn(fe._action_latent.shape, generator=gen, device="cuda").to(fe._action_latent.dtype)
    outs = {}
    for name, g in (("old", graph_old), ("new", graph_new)):
        fe._img_raw.copy_(img)
        fe._action_latent.copy_(noise)
        g.replay()
        torch.cuda.synchronize()
        outs[name] = fe._action_latent.detach().clone()
    finite = bool(torch.isfinite(outs["old"]).all() and torch.isfinite(outs["new"]).all())
    diff = (outs["new"].float() - outs["old"].float()).abs().max().item()
    print(f"actions new vs old (same inputs): cosine={_cos(outs['new'], outs['old']):.6f} "
          f"max_abs={diff:.3e} finite={finite}")

    obs: dict = {}
    for _ in range(warmup):
        for g in (graph_old, graph_new):
            fe._graph = g
            fe.infer(obs)
    torch.cuda.synchronize()
    times: dict[str, list[float]] = {"old": [], "new": []}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(iters):
        order = (("old", graph_old), ("new", graph_new)) if i % 2 == 0 else (("new", graph_new), ("old", graph_old))
        for name, g in order:
            fe._graph = g
            start.record()
            fe.infer(obs)
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end))
    fe._graph = graph_new
    for name in ("old", "new"):
        t = times[name]
        print(f"infer() {name}: P10={_pct(t, 0.1):.2f}ms P50={_pct(t, 0.5):.2f}ms P90={_pct(t, 0.9):.2f}ms "
              f"(n={len(t)})")
    p50_old, p50_new = _pct(times["old"], 0.5), _pct(times["new"], 0.5)
    print(f"P50 delta new-old = {p50_new - p50_old:+.2f}ms ({p50_old / p50_new:.4f}x)")
    del fe
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=("kernels", "infer", "all"), default="all")
    ap.add_argument("--precisions", default="nvfp4,fp8_static_cutlass")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--samples", type=int, default=25, help="timer replays per variant (kernels part)")
    args = ap.parse_args()
    print(f"device: {torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()} "
          f"torch={torch.__version__}")
    if args.part in ("kernels", "all"):
        run_kernels(CudaGraphVariantTimer(reps=4, samples=args.samples, warmup=3))
    if args.part in ("infer", "all"):
        for precision in args.precisions.split(","):
            run_infer(precision.strip(), args.warmup, args.iters)


if __name__ == "__main__":
    main()
