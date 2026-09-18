"""FP8 cuBLASLt NN vs TN layout, at every real ImageWAM GEMM shape
(`issues.md` ISSUE-001).

`StaticFp8Linear(use_cutlass=False)` runs `fp8_gemm_descale_fp16` (NN,
weight stored (K,N)) on Blackwell and `fp8_gemm_descale_fp16_tn` (TN,
weight stored (N,K)) on sm_89/sm_90. Blackwell supports both, so on Thor
this measures whether TN could replace NN. On a GPU without NN support
only TN is timed.

Per shape: both layouts built from the same fp16 weight and calibrated
on the same activation; timed alternately (NN batch, TN batch, ...) with
CUDA events; P10/P50/P90 per call over all batches; plus TN-vs-NN output
cosine / max-abs / bit-exactness.

Env: ITERS (per batch, default 50), ROUNDS (alternations, default 20).
"""
from __future__ import annotations

import os

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk  # noqa: F401  (loads the extension)
from flash_rt.models.imagewam.quant_linear import StaticFp8Linear

DEV = "cuda"
FP16 = torch.float16
ITERS = int(os.environ.get("ITERS", "50"))
ROUNDS = int(os.environ.get("ROUNDS", "20"))

REAL_SHAPES = {
    "txt_qkv": (513, 9216, 3072), "txt_proj": (513, 3072, 3072),
    "txt_mlp0": (513, 18432, 3072), "txt_mlp2": (513, 3072, 9216),
    "img_qkv": (392, 9216, 3072), "img_proj": (392, 3072, 3072),
    "img_mlp0": (392, 18432, 3072), "img_mlp2": (392, 3072, 9216),
    "single_linear1": (905, 27648, 3072), "single_attn_out": (905, 3072, 3072),
    "single_mlp_down": (905, 3072, 9216),
    "action_qkv": (64, 9216, 1024), "action_proj": (64, 1024, 3072),
    "action_mlp0": (64, 8192, 1024), "action_mlp2": (64, 1024, 4096),
    "action_linear1": (64, 17408, 1024),
}
# Calls per infer() at the real dims (5 double + 20 single backbone
# layers once; 5 double + 20 single ActionDiT layers x 10 steps). The
# ActionDiT single-stream attn_out_proj / mlp_down share the double
# stream's proj / mlp2 shapes.
CALLS_PER_INFER = {
    "txt_qkv": 5, "txt_proj": 5, "txt_mlp0": 5, "txt_mlp2": 5,
    "img_qkv": 5, "img_proj": 5, "img_mlp0": 5, "img_mlp2": 5,
    "single_linear1": 20, "single_attn_out": 20, "single_mlp_down": 20,
    "action_qkv": 50, "action_proj": 250, "action_mlp0": 50, "action_mlp2": 250,
    "action_linear1": 200,
}


def _build(w: torch.Tensor, x: torch.Tensor, n: int, k: int, m: int, layout: str):
    try:
        lin = StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False, layout=layout)
        lin.calibrate(x.data_ptr(), m, 0)
        out = torch.zeros(m, n, dtype=FP16, device=DEV)
        lin(x.data_ptr(), out.data_ptr(), m, 0)
        torch.cuda.synchronize()
        return lin, out
    except RuntimeError as e:
        print(f"  layout {layout}: unavailable ({e})")
        return None, None


def _time_batch(lin, x_ptr: int, out_ptr: int, m: int) -> list[float]:
    ts = []
    for _ in range(ITERS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        lin(x_ptr, out_ptr, m, 0)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)  # us
    return ts


def main() -> None:
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()} "
          f"ITERS={ITERS} ROUNDS={ROUNDS}")
    total = {"nn": 0.0, "tn": 0.0}
    for name, (m, n, k) in REAL_SHAPES.items():
        w = (torch.randn(k, n, device=DEV) * 0.02).to(FP16).contiguous()  # (K,N) project convention
        x = torch.randn(m, k, device=DEV).to(FP16)
        built = {lay: _build(w, x, n, k, m, lay) for lay in ("nn", "tn")}
        live = [lay for lay in ("nn", "tn") if built[lay][0] is not None]
        for lay in live:  # warmup
            _time_batch(built[lay][0], x.data_ptr(), built[lay][1].data_ptr(), m)
        samples = {lay: [] for lay in live}
        for _ in range(ROUNDS):
            for lay in live:
                samples[lay] += _time_batch(built[lay][0], x.data_ptr(), built[lay][1].data_ptr(), m)
        row = f"{name:16s} M={m:4d} N={n:5d} K={k:4d}"
        for lay in live:
            p10, p50, p90 = np.percentile(samples[lay], [10, 50, 90])
            row += f" | {lay} P50={p50:7.1f}us P10={p10:7.1f} P90={p90:7.1f}"
            total[lay] += p50 * CALLS_PER_INFER.get(name, 0)
        if len(live) == 2:
            o_nn, o_tn = built["nn"][1].float(), built["tn"][1].float()
            cos = (o_nn.flatten() @ o_tn.flatten() / (o_nn.norm() * o_tn.norm())).item()
            row += (f" | tn/nn={np.median(samples['tn']) / np.median(samples['nn']):.3f}"
                    f" cos={cos:.6f} maxabs={(o_nn - o_tn).abs().max().item():.4f}"
                    f" bitexact={torch.equal(built['nn'][1], built['tn'][1])}")
        print(row, flush=True)
    for lay in ("nn", "tn"):
        if total[lay]:
            print(f"sum over one infer()'s FP8 GEMM calls (P50 x calls), {lay}: {total[lay] / 1e3:.2f} ms")


if __name__ == "__main__":
    main()
