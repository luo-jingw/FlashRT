"""Per-layer simulated NVFP4 error, with and without AWQ, on held-out frames.

The real fp16 pipeline (real checkpoint, VAE, Qwen3, proprio, shift
schedule) runs eagerly on held-out LIBERO frames (the end-to-end
harness's libero_spatial evaluation frames; the calibration file comes
from other suites). At every NVFP4-eligible GEMM (N and K divisible by
16) a probe takes the real fp16 input `x` and compares, against the fp32
reference `x @ W`, the NVFP4 GEMM each variant would compute
(`nvfp4_sim.fake_quant_nvfp4` on both operands along K, fp32 matmul):

  nvfp4            the shipped quantizer
  awq a=<alpha>    x / s and W * s, s = awq_scale(channel_amax, alpha)
                   from the calibration file, applied at EVERY site
                   (isolated per-layer effect; the served folds cover a
                   subset, see awq.py)
  gscale           W * 2^e before quantization and 2^-e after the GEMM
                   (a per-tensor power-of-two weight scale, e the largest
                   integer with max|W| * 2^e <= 6 * 448): lifts the weight
                   block scales out of E4M3's subnormal range
  awq a=<a>+gscale both

The pipeline itself stays fp16, so each site sees its true fp16 input.
Reported per site group: rel_l2 = sqrt(sum ||y - y_ref||^2 / sum ||y_ref||^2)
over all calls and frames, and the share of weight blocks whose E4M3
scale is subnormal or zero.

Env: CKPT_PATH, FLUX2_AE_MODEL_PATH, FLUX2_SRC, QWEN3_MODEL_SPEC, DATA_ROOT,
CALIBRATION (required), N_TASKS (5), FRAMES ("0,60"), ALPHAS ("0.25,0.5,0.75,1.0").
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.activation_recorder import site_name
from flash_rt.models.imagewam.awq import awq_scale
from flash_rt.models.imagewam.calibration_file import load_calibration
from flash_rt.models.imagewam.libero_dims import LIBERO_HORIZON, LIBERO_REAL_DIMS
from flash_rt.models.imagewam.libero_frames import evaluation_frames, load_frame
from flash_rt.models.imagewam.nvfp4_sim import E2M1_MAX, E4M3_MAX, fake_quant_nvfp4, quantize_nvfp4
from flash_rt.models.imagewam.quant_linear import Fp16Linear

DEV = "cuda"
BF16 = torch.bfloat16
CKPT = os.environ["CKPT_PATH"]
N_TASKS = int(os.environ.get("N_TASKS", "5"))
FRAMES = tuple(int(x) for x in os.environ.get("FRAMES", "0,60").split(","))
ALPHAS = tuple(float(x) for x in os.environ.get("ALPHAS", "0.25,0.5,0.75,1.0").split(","))
FOLD_A = {"txt_qkv.weight", "img_qkv.weight", "txt_mlp0.weight", "img_mlp0.weight",
          "linear1.weight", "qkv.weight", "mlp0.weight"}
FOLD_B = {"txt_mlp2.weight", "img_mlp2.weight", "mlp_down.weight", "mlp2.weight"}


def _view(ptr: int, m: int, k: int) -> torch.Tensor:
    interface = {"data": (int(ptr), False), "shape": (int(m), int(k)), "typestr": "<f2", "version": 3}
    return torch.as_tensor(type("_V", (), {"__cuda_array_interface__": interface})(), device=DEV)


def pow2_weight_exponent(w: torch.Tensor) -> int:
    return int(math.floor(math.log2(E2M1_MAX * E4M3_MAX / w.float().abs().max().item())))


class Variant:
    def __init__(self, name: str, alpha: float | None, gscale: bool):
        self.name, self.alpha, self.gscale = name, alpha, gscale


class _Probe:
    """Wraps one fp16 GEMM: runs it unchanged, then accumulates each
    variant's simulated NVFP4 error on the same input."""

    def __init__(self, inner: Fp16Linear, name: str, channel_amax: torch.Tensor,
                 variants: list[Variant], acc: dict):
        self.inner, self.name, self.k, self.n = inner, name, inner.k, inner.n
        self.w = _view(inner.weight_ptr, inner.k, inner.n)  # (K, N)
        self.s = {v.alpha: awq_scale(channel_amax, v.alpha) for v in variants if v.alpha is not None}
        self.variants, self.acc = variants, acc

    def _weight_fq(self, v: Variant) -> tuple[torch.Tensor, float]:
        w = self.w.float()
        if v.alpha is not None:
            w = w * self.s[v.alpha].unsqueeze(1)
        w16 = w.to(torch.float16)
        post = 1.0
        if v.gscale:
            e = pow2_weight_exponent(w16)
            w16 = (w16.float() * 2.0 ** e).to(torch.float16)
            post = 2.0 ** -e
        return fake_quant_nvfp4(w16.t().contiguous()).float(), post  # (N, K)

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        self.inner(x_ptr, out_ptr, m, stream)
        x = _view(x_ptr, m, self.k)
        y_ref = x.float() @ self.w.float()
        ref2 = y_ref.pow(2).sum().item()
        for v in self.variants:
            xs = x if v.alpha is None else (x.float() / self.s[v.alpha]).to(torch.float16)
            w_fq, post = self._weight_fq(v)
            y = (fake_quant_nvfp4(xs).float() @ w_fq.t()) * post
            d = self.acc.setdefault((self.name, v.name), [0.0, 0.0])
            d[0] += (y - y_ref).pow(2).sum().item()
            d[1] += ref2


def group_of(name: str) -> str:
    p = name.split(".")
    return ".".join(p[:2] + p[3:])


def main() -> None:
    cal = load_calibration(os.environ["CALIBRATION"])
    variants = [Variant("nvfp4", None, False)]
    variants += [Variant(f"awq a={a}", a, False) for a in ALPHAS]
    variants += [Variant("gscale", None, True), Variant("awq a=0.5+gscale", 0.5, True)]

    fe = ImageWAMTorchFrontendThor(
        precision="fp16", dims_override=dict(LIBERO_REAL_DIMS), ckpt_path=CKPT,
        ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
        qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
        dataset_stats_path=os.path.join(os.path.dirname(CKPT), "dataset_stats.json"))
    acc: dict = {}
    weights = dict(fe.weights)
    subnormal: dict[str, list[float]] = {}
    for key, lin in fe.weights.items():
        if isinstance(lin, Fp16Linear) and lin.n % 16 == 0 and lin.k % 16 == 0:
            name = site_name(key)
            ch = torch.from_numpy(cal.sites[name].channel_amax).to(DEV)
            weights[key] = _Probe(lin, name, ch, variants, acc)
            w_nk = _view(lin.weight_ptr, lin.k, lin.n).t().contiguous()
            _, sc = quantize_nvfp4(w_nk)
            _, sc_g = quantize_nvfp4((w_nk.float() * 2.0 ** pow2_weight_exponent(w_nk)).to(torch.float16))
            subnormal[name] = [(sc.float() < 2 ** -6).float().mean().item(),
                               (sc_g.float() < 2 ** -6).float().mean().item()]
    refs = evaluation_frames(os.environ["DATA_ROOT"], n_tasks=N_TASKS, frames=FRAMES)
    print(f"{len(weights)} weights, {len(subnormal)} NVFP4 sites probed; {len(refs)} held-out frames; "
          f"variants: {[v.name for v in variants]}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i, ref in enumerate(refs):
            fr = load_frame(os.environ["DATA_ROOT"], ref, LIBERO_HORIZON)
            fe.set_prompt(fr.task)
            g = torch.Generator(device="cpu").manual_seed(i)
            noise = torch.randn((1, LIBERO_HORIZON, 7), generator=g).to(DEV, BF16).float()[0]
            fe.stage_inputs({"view1": torch.from_numpy(fr.view1), "view2": torch.from_numpy(fr.view2),
                             "proprio": fr.state}, noise=noise)
            fe.run_eager(weights)
            torch.cuda.synchronize()
            print(f"frame {i + 1}/{len(refs)} done ({time.time() - t0:.0f}s)", flush=True)

    sites = sorted(subnormal)
    rel = {(s, v.name): math.sqrt(acc[(s, v.name)][0] / acc[(s, v.name)][1]) for s in sites for v in variants}
    groups: dict[str, list[str]] = {}
    for s in sites:
        groups.setdefault(group_of(s), []).append(s)
    header = f"{'site group (fold)':46s} {'n':>3s} {'subn w':>7s} {'subn g':>7s} " + " ".join(
        f"{v.name:>14s}" for v in variants)
    print("\nper-layer rel_l2 vs fp32 reference (median over the group's layers)")
    print(header)
    for gname, members in sorted(groups.items()):
        slot = gname.split(".")[-2] + ".weight"
        fold = "A" if slot in FOLD_A else ("B" if slot in FOLD_B else "-")
        sub_w = np.median([subnormal[s][0] for s in members])
        sub_g = np.median([subnormal[s][1] for s in members])
        vals = " ".join(f"{np.median([rel[(s, v.name)] for s in members]):14.5f}" for v in variants)
        print(f"{gname + ' (' + fold + ')':46s} {len(members):3d} {sub_w:7.1%} {sub_g:7.1%} {vals}")
    print("\nsite-weighted mean rel_l2 by fold class")
    for label, cls in (("fold A (AdaLN-fed)", FOLD_A), ("fold B (down proj)", FOLD_B),
                       ("no fold (attn out)", None)):
        members = [s for s in sites if (s.split(".")[-2] + ".weight" in cls if cls else
                                        s.split(".")[-2] + ".weight" not in FOLD_A | FOLD_B)]
        vals = " ".join(f"{np.mean([rel[(s, v.name)] for s in members]):14.5f}" for v in variants)
        print(f"{label:46s} {len(members):3d} {'':7s} {'':7s} {vals}")


if __name__ == "__main__":
    main()
