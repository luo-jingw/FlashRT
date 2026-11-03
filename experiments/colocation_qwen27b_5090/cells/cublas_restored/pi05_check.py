"""Pi0.5 RTX frontend: actions and latency for one fixed observation, for comparing two checkouts.

usage: python pi05_merge_check.py CKPT {default,fp4enc} OUT_NPZ
"""
import sys
import time

import numpy as np
import torch

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

ckpt, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
fe = Pi05TorchFrontendRtx(ckpt, num_views=2, use_fp4_encoder=(mode == "fp4enc"))
fe.set_prompt("put both the alphabet soup and the tomato sauce in the basket")
rng = np.random.default_rng(0)
obs = {"image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
       "wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
       "state": rng.standard_normal(8).astype(np.float32)}
torch.manual_seed(0)
fe.calibrate_with_real_data([obs])
acts = []
for i in range(3):
    torch.manual_seed(1234)
    acts.append(np.asarray(fe.infer(obs)["actions"], dtype=np.float32))
for _ in range(20):
    fe.infer(obs)
ts = []
for _ in range(50):
    torch.cuda.synchronize()
    t = time.perf_counter()
    fe.infer(obs)
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t) * 1e3)
np.savez(out, actions=np.stack(acts), latency_ms=np.array(ts))
print(f"{mode}: actions {acts[0].shape}, repeat max|diff| {float(np.abs(acts[1] - acts[0]).max()):.3g}, "
      f"latency p50 {np.percentile(ts, 50):.2f} ms p90 {np.percentile(ts, 90):.2f} ms")
