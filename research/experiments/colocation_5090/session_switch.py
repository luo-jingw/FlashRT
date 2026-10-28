"""R3a: what one ImageWAM session owns, and what switching between sessions costs on one frontend.

A session = one robot / episode with its own instruction. Its persistent state in the served frontend is the text
context written by set_prompt (rows x joint_attention_dim, BF16) plus the proprio row index; everything else in the
graph is recomputed every call. Measured: the context bytes; the time of set_prompt with a new instruction (live
Qwen3 text encode), of set_prompt with a cached context (copy only), and of one infer(); then N sessions served
round-robin on one frontend (switch + infer per tick, batch 1), giving the per-session rate the GPU sustains.

usage: python session_switch.py OUT_JSON
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
from _imagewam_libero_frames import center_crop_resize  # noqa: E402

import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
from flash_rt.models.imagewam.text_encoder import encode_prompts  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

OUT = sys.argv[1]
TASKS = ["put both the alphabet soup and the tomato sauce in the basket", "turn on the stove and put the moka pot on it",
         "put the black bowl in the bottom drawer of the cabinet and close it", "pick up the book and place it in the back compartment of the caddy",
         "put both moka pots on the stove", "put the white mug on the left plate and put the yellow and white mug on the right plate",
         "put both the cream cheese box and the butter in the basket", "put the yellow and white mug in the microwave and close it"]
PREFIX = "A video recorded from a robot's point of view executing the following instruction: "

fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision="fp8",
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"], dataset_stats_path=os.environ["DATASET_STATS_PATH"])
rng = np.random.default_rng(0)
frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
obs = {"view1": torch.from_numpy(center_crop_resize(frame, 224, 224)), "view2": torch.from_numpy(center_crop_resize(frame, 224, 224)),
       "proprio": np.zeros(8, dtype=np.float32)}


def ms(fn) -> float:
    torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3


model, tok = fe._qwen3
ctx = []
for t in TASKS:
    c, m = encode_prompts(model, tok, [PREFIX + t], max_length=fe._text_max_length())
    ctx.append((c[0].to(torch.bfloat16).cuda(), m[0].cuda()))
fe.set_prompt(PREFIX + TASKS[0])
for _ in range(5):
    fe.infer(obs)

encode_ms = [ms(lambda t=t: fe.set_prompt(PREFIX + t)) for t in TASKS[1:]] + [ms(lambda: fe.set_prompt(PREFIX + TASKS[0]))]
cached_ms = []
for i in range(40):
    c, m = ctx[i % len(ctx)]
    cached_ms.append(ms(lambda c=c, m=m: fe.set_prompt(context=c, context_mask=m)))
infer_ms = [ms(lambda: fe.infer(obs)) for _ in range(40)]

rr = []
for n in (1, 2, 4, 8):
    lat = []
    t0 = time.perf_counter()
    k = 0
    while time.perf_counter() - t0 < 10.0:
        c, m = ctx[k % n]
        t = time.perf_counter()
        if n > 1:
            fe.set_prompt(context=c, context_mask=m)
        fe.infer(obs)
        lat.append((time.perf_counter() - t) * 1e3)
        k += 1
    el = time.perf_counter() - t0
    rr.append({"sessions": n, "ticks": k, "tick_ms_p50": float(np.median(lat)), "aggregate_hz": k / el,
               "per_session_hz": k / el / n})
    print(rr[-1], flush=True)

res = {"precision": "fp8", "gpu": torch.cuda.get_device_name(),
       "context_shape": list(fe._context.shape), "context_bytes": fe._context.numel() * fe._context.element_size(),
       "set_prompt_live_encode_ms_p50": float(np.median(encode_ms)), "set_prompt_cached_context_ms_p50": float(np.median(cached_ms)),
       "infer_ms_p50": float(np.median(infer_ms)), "round_robin": rr,
       "weights_GB_allocated": torch.cuda.memory_allocated() / 1e9, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
json.dump(res, open(OUT, "w"), indent=1)
print({k: v for k, v in res.items() if k != "round_robin"})
