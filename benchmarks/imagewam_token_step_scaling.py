"""ImageWAM visual-token-count and denoise-step-count timing: same deployment entry and
timing method as imagewam_module_operator_profile.py, with the image token count set by
the view width (img_len = (224/16) * (2*W/16)) and the step count by num_steps.
usage: python benchmarks/imagewam_token_step_scaling.py PRECISION WIDTH STEPS OUT_JSON"""
import dataclasses
import json
import os
import sys

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk
import flash_rt.frontends.torch.imagewam_thor as F
import flash_rt.models.imagewam.pipeline_thor as P
from flash_rt.models.imagewam.workload import ImageWAMWorkload

PRECISION, W, STEPS, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
wl = dataclasses.replace(ImageWAMWorkload.libero(), image_w=W, num_steps=STEPS)
fe = F.load_imagewam(os.environ["CKPT_PATH"], wl, precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
fe.set_prompt("pick up the red block and place it in the tray")
rng = np.random.default_rng(0)
obs = {"view1": rng.integers(0, 255, (224, W, 3), dtype=np.uint8),
       "view2": rng.integers(0, 255, (224, W, 3), dtype=np.uint8),
       "proprio": rng.standard_normal(8).astype(np.float32)}
views = (torch.as_tensor(obs["view1"]), torch.as_tensor(obs["view2"]))
for _ in range(3):
    out = fe.infer(obs)
torch.cuda.synchronize()
act = np.asarray(out["actions"])
ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def timed(fn):
    ev0.record(); fn(); ev1.record(); ev1.synchronize()
    return ev0.elapsed_time(ev1)


s = torch.cuda.Stream()


def graph_of(fn):
    with torch.cuda.stream(s):
        fn(s.cuda_stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn(s.cuda_stream)
    return g


def prefill_fn(st):
    P.imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe._active_dims, stream=st,
                       attn=fe._attn, mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
                       rope_table=fe._rope_table.data_ptr())


def denoise_fn(st):
    P.imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe._active_dims, stream=st,
                            attn=fe._attn, action_mods=fe._action_mods, head_mods=fe._head_mods,
                            action_rope_table=fe._action_rope_table.data_ptr(), deltas=fe._deltas)


g_pre, g_den = graph_of(prefill_fn), graph_of(denoise_fn)
for _ in range(20):
    fe.infer(obs); g_pre.replay(); g_den.replay()
torch.cuda.synchronize()
T = {"infer": [], "prefill": [], "denoise": [], "vae": []}
for _ in range(100):
    T["infer"].append(timed(lambda: fe.infer(obs)))
    T["prefill"].append(timed(g_pre.replay))
    T["denoise"].append(timed(g_den.replay))
    T["vae"].append(timed(lambda: fe.stage_images(*views)))
res = {"precision": PRECISION, "image_w": W, "steps": STEPS,
       "active_dims": {k: fe._active_dims[k] for k in ("x0", "a0", "total", "num_action")},
       "actions_shape": list(act.shape), "actions_finite": bool(np.isfinite(act).all()),
       **{f"{k}_p50_ms": float(np.median(v)) for k, v in T.items()}}
print(json.dumps(res), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
