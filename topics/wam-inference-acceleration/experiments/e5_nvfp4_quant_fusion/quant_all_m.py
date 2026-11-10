"""Per-step NVFP4 activation-side cost of the ImageWAM ActionDiT at M = 64k (k = 1..10).

Per GEMM shape (count per denoise step: double x5 qkv/proj/mlp0/mlp2,
single x20 linear1/linear2), CUDA-graph timing (200 launches, median of 5):
  production : cast_fp16_to_bf16 + quantize_bf16_to_nvfp4_swizzled (+ for the
               norm-fed inputs qkv/mlp0/linear1 the gate_res_ada_layer_norm_fp16
               that produces them) and cast_bf16_to_fp16 on the GEMM output;
  prototype  : proto_quant_fp16_nvfp4 (proj/mlp2/linear2), the fused
               proto_gate_res_ada_ln_nvfp4 (qkv/mlp0/linear1), no output cast
               (GEMM writes fp16, measured equal in time to bf16 output).
The norm kernel is counted on both sides so the difference is the saving.
usage: python quant_all_m.py LIB OUT_JSON
"""
import ctypes
import json
import sys

import numpy as np
import torch

from flash_rt import flash_rt_kernels as fvk

L = ctypes.CDLL(sys.argv[1])
L.proto_quant_fp16_nvfp4.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
L.proto_gate_res_ada_ln_nvfp4.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                                                   ctypes.c_void_p]
dev = "cuda"
SHAPES = {"qkv": (9216, 1024, True, 5), "proj": (1024, 3072, False, 5), "mlp0": (8192, 1024, True, 5),
          "mlp2": (1024, 4096, False, 5), "linear1": (17408, 1024, True, 20), "linear2": (1024, 7168, False, 20)}


def gtime(fn, reps: int = 200) -> float:
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(s.cuda_stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn(s.cuda_stream)
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(5):
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / reps)
    return float(np.median(ts))


out = {}
for k in range(1, 11):
    M = 64 * k
    prod_tot, proto_tot, per = 0.0, 0.0, {}
    for name, (N, K, normfed, cnt) in SHAPES.items():
        x = (torch.randn(M, K, device=dev) * 3).half()
        xb = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
        pk = torch.empty(M, K // 2, device=dev, dtype=torch.uint8)
        sf = torch.empty(fvk.nvfp4_sf_swizzled_bytes(M, K), device=dev, dtype=torch.uint8)
        yb = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        y16 = torch.empty(M, N, device=dev, dtype=torch.float16)
        if normfed:
            proj = torch.randn(M, K, device=dev).half(); gate = torch.randn(K, device=dev) * 0.5
            res = torch.randn(M, K, device=dev).half(); sc = torch.randn(K, device=dev) * 0.2
            sh = torch.randn(K, device=dev) * 0.2

            def prod(st):
                fvk.gate_res_ada_layer_norm_fp16(proj.data_ptr(), gate.data_ptr(), res.data_ptr(), sc.data_ptr(),
                                                 sh.data_ptr(), x.data_ptr(), M, K, 1e-6, st)
                fvk.cast_fp16_to_bf16(x.data_ptr(), xb.data_ptr(), M * K, st)
                fvk.quantize_bf16_to_nvfp4_swizzled(xb.data_ptr(), pk.data_ptr(), sf.data_ptr(), M, K, st)
                fvk.cast_bf16_to_fp16(yb.data_ptr(), y16.data_ptr(), M * N, st)

            def proto(st):
                L.proto_gate_res_ada_ln_nvfp4(proj.data_ptr(), gate.data_ptr(), res.data_ptr(), sc.data_ptr(),
                                              sh.data_ptr(), x.data_ptr(), pk.data_ptr(), sf.data_ptr(), M, K, 1e-6, st)
        else:
            def prod(st):
                fvk.cast_fp16_to_bf16(x.data_ptr(), xb.data_ptr(), M * K, st)
                fvk.quantize_bf16_to_nvfp4_swizzled(xb.data_ptr(), pk.data_ptr(), sf.data_ptr(), M, K, st)
                fvk.cast_bf16_to_fp16(yb.data_ptr(), y16.data_ptr(), M * N, st)

            def proto(st):
                L.proto_quant_fp16_nvfp4(x.data_ptr(), pk.data_ptr(), sf.data_ptr(), M, K, st)
        a, b = gtime(prod), gtime(proto)
        per[name] = [round(a, 3), round(b, 3)]
        prod_tot += cnt * a
        proto_tot += cnt * b
    out[k] = {"M": M, "per_call_us": per, "per_step_us": [round(prod_tot, 1), round(proto_tot, 1)],
              "saving_us": round(prod_tot - proto_tot, 1)}
    print(k, M, out[k]["per_step_us"], "saving", out[k]["saving_us"], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
