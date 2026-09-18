#!/usr/bin/env python
"""A/B `pipeline_thor.py` dims flags through the real served frontend.

Builds two `ImageWAMTorchFrontendThor` instances at the real dims that
differ only in the `dims` flags named by `AB` (A = flags False, B =
flags True), captures both CUDA graphs, then reports, per precision:

1. Correctness: identical text context, image tokens and initial action
   noise into both, one graph replay each; `action_latent` (the final
   actions, normalized space) and `backbone_hidden` (the backbone's
   final residual) of B against A: cosine, max-abs, rel_l2, bit-exact.
2. Speed: `infer()` and `graph.replay()` latency, A and B interleaved in
   the same process (ABBA order), CUDA events, P10/P50/P90 per side.
3. Kernel count (`COUNT_KERNELS=1`): CUDA kernel launches of one eager
   (uncaptured) prefill + denoise pass per side, from `torch.profiler`
   -- the same launches the captured graph replays.

Flags (`AB=`, one or a comma list toggled together):
- `merge_linear2`: single-stream `linear2` as one GEMM (roadmap item 4,
  plan.md "single-stream `linear2` merge").
- `fuse_res_norm`: gated residual fused with the next AdaLN (roadmap
  item 3, plan.md "gated-residual + next-AdaLN fusion").

Env:
- `AB` (required): the flag(s) to toggle.
- `PRECISIONS` (default `nvfp4,fp16`): comma list from `_PRECISIONS`.
- `CKPT_PATH` (optional): real `model.pt`. Unset -> random weights drawn
  from the same seed for both sides. The merged `linear2` weight is drawn
  as one tensor, so with random weights the `merge_linear2` sides hold
  different weight values and only the speed numbers are comparable;
  its correctness comparison needs `CKPT_PATH`.
- `ITERS` (default 50), `WARMUP` (default 10), `COUNT_KERNELS` (default 0).
- `USE_FA4` (default 0): `use_fa4=True` for the "backbone" attention site
  (Thor only; match the production configuration being compared).

Speed on a shared GPU is indicative only; Thor numbers come from Thor.
"""
from __future__ import annotations

import os

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch.imagewam_thor import _PRECISIONS, ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_loop, imagewam_prefill

REAL_DIMS = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=64, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=1.0 / 10, num_denoise_steps=10,
    ref_h=14, ref_w=28, shift=5.0, num_train_timesteps=1000,
)
FLAGS = ("merge_linear2", "fuse_res_norm")
BF16 = torch.bfloat16
DEV = "cuda"


def _stats(b: torch.Tensor, a: torch.Tensor) -> str:
    a_, b_ = a.float().flatten(), b.float().flatten()
    cos = (a_ @ b_ / (a_.norm() * b_.norm() + 1e-12)).item()
    max_abs = (a_ - b_).abs().max().item()
    rel_l2 = ((a_ - b_).norm() / (a_.norm() + 1e-12)).item()
    return (f"cos={cos:.6f} max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} "
            f"bit_exact={torch.equal(a, b)} finite={bool(torch.isfinite(b.float()).all())}")


def _pcts(ts: list[float]) -> str:
    p10, p50, p90 = np.percentile(np.asarray(ts), [10, 50, 90])
    return f"P10={p10:8.2f}  P50={p50:8.2f}  P90={p90:8.2f} ms"


def _build(precision: str, flags: list[str], value: bool, ckpt: str | None,
           use_fa4: bool) -> ImageWAMTorchFrontendThor:
    torch.manual_seed(0)
    dims = dict(REAL_DIMS, **{flag: value for flag in flags})
    return ImageWAMTorchFrontendThor(precision=precision, dims_override=dims, ckpt_path=ckpt, use_fa4=use_fa4)


def _replay_with(fe: ImageWAMTorchFrontendThor, img: torch.Tensor, noise: torch.Tensor):
    fe._img_raw.copy_(img)
    fe._action_latent.copy_(noise)
    fe._graph.replay()
    torch.cuda.synchronize()
    return fe._action_latent.clone(), fe._backbone_hidden.clone()


def _count_kernels(fe: ImageWAMTorchFrontendThor) -> int:
    s = torch.cuda.current_stream()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s.cuda_stream,
                         attn=fe._attn, mod_txt=fe._mod_txt, mod_img=fe._mod_img,
                         mod_single=fe._mod_single, rope_table=fe._rope_table.data_ptr())
        imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s.cuda_stream,
                              attn=fe._attn, action_mods=fe._action_mods, head_mods=fe._head_mods,
                              action_rope_table=fe._action_rope_table.data_ptr(), deltas=fe._deltas)
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
               and not e.name.startswith(("Memcpy", "Memset")))


def run(precision: str, flags: list[str], ckpt: str | None, iters: int, warmup: int, count_kernels: bool,
        use_fa4: bool) -> None:
    names = "+".join(flags)
    print(f"\n=== {precision}: A = {names} off, B = {names} on "
          f"({'real checkpoint' if ckpt else 'random weights'}, use_fa4={use_fa4}) ===", flush=True)
    a = _build(precision, flags, False, ckpt, use_fa4)
    b = _build(precision, flags, True, ckpt, use_fa4)
    x0, jad = REAL_DIMS["x0"], REAL_DIMS["joint_attention_dim"]
    g = torch.Generator(device=DEV).manual_seed(1)
    ctx = torch.randn(x0, jad, generator=g, device=DEV).to(BF16)
    mask = torch.ones(x0, dtype=torch.bool, device=DEV)
    for fe in (a, b):
        fe.set_prompt(context=ctx, context_mask=mask)
    img = torch.randn(REAL_DIMS["a0"] - x0, REAL_DIMS["HD"], generator=g, device=DEV).to(BF16)
    noise = torch.randn(REAL_DIMS["num_action"], a.dims["action_dim"], generator=g, device=DEV)

    act_a, hid_a = _replay_with(a, img, noise)
    act_b, hid_b = _replay_with(b, img, noise)
    note = "" if (ckpt or "merge_linear2" not in flags) else "  (different random weights per side, see docstring)"
    print(f"actions         B vs A: {_stats(act_b, act_a)}{note}")
    print(f"backbone_hidden B vs A: {_stats(hid_b, hid_a)}{note}")

    obs: dict = {}
    for _ in range(warmup):
        a.infer(obs)
        b.infer(obs)
    t_inf = {"A": [], "B": []}
    t_rep = {"A": [], "B": []}
    for i in range(iters):
        order = (("A", a), ("B", b)) if i % 2 == 0 else (("B", b), ("A", a))
        for name, fe in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fe.infer(obs)
            e.record()
            e.synchronize()
            t_inf[name].append(s.elapsed_time(e))
            s.record()
            fe._graph.replay()
            e.record()
            e.synchronize()
            t_rep[name].append(s.elapsed_time(e))
    for name in ("A", "B"):
        print(f"infer()  {name}: {_pcts(t_inf[name])}")
    for name in ("A", "B"):
        print(f"replay() {name}: {_pcts(t_rep[name])}")
    d50 = np.median(t_inf["A"]) - np.median(t_inf["B"])
    print(f"infer() P50 A - B = {d50:+.2f} ms")

    if count_kernels:
        print(f"CUDA kernels per prefill+denoise pass: A={_count_kernels(a)} B={_count_kernels(b)}")
    del a, b
    torch.cuda.empty_cache()


def main() -> None:
    flags = [f for f in os.environ.get("AB", "").split(",") if f]
    if not flags or any(f not in FLAGS for f in flags):
        raise SystemExit(f"set AB to one or more (comma-separated) of {FLAGS}")
    precisions = os.environ.get("PRECISIONS", "nvfp4,fp16").split(",")
    for p in precisions:
        if p not in _PRECISIONS:
            raise SystemExit(f"unknown precision {p!r}; one of {_PRECISIONS}")
    ckpt = os.environ.get("CKPT_PATH") or None
    iters = int(os.environ.get("ITERS", "50"))
    warmup = int(os.environ.get("WARMUP", "10"))
    count_kernels = os.environ.get("COUNT_KERNELS", "0") == "1"
    use_fa4 = os.environ.get("USE_FA4", "0") == "1"
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}, AB={flags}, "
          f"precisions={precisions}, iters={iters}")
    for p in precisions:
        run(p, flags, ckpt, iters, warmup, count_kernels, use_fa4)


if __name__ == "__main__":
    main()
