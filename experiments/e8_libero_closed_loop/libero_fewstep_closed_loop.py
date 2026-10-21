"""LIBERO closed-loop success of the ImageWAM ActionDiT with fewer denoise evaluations.

Configs (same frontend, same initial noise per policy call):
  serial10  served `fe.infer()` (CUDA graph, 10 Euler steps);
  sub_0_7_9, sub_0_9  prefill via `fe.run_eager()` on the staged observation,
            then Euler over the subset S of the served 10-step grid from the
            same initial latent: from grid point s to the next point of S (or
            t = 0), velocity from `imagewam_denoise_step` with step s's own
            conditioning, step size t_s - t_next.
Every config runs the same episodes (task, init state) in the same order.
Preprocessing / gripper mapping as in libero_e2e_latency.py
(eval_libero_single.py); num_steps_wait 30, replan 10, seed 7.

usage: python libero_fewstep_closed_loop.py PRECISION SUITE EPISODES_PER_TASK CONFIGS(comma) OUT_JSON
"""
import json
import os
import sys
import time

import numpy as np
import torch
from experiments.libero.libero_utils import (LIBERO_ENV_RESOLUTION, get_libero_dummy_action, get_libero_env,
                                             get_libero_image, quat2axisangle)
from libero.libero import benchmark
from PIL import Image

from flash_rt import flash_rt_kernels as fvk
import flash_rt.frontends.torch.imagewam_thor as F
import flash_rt.models.imagewam.pipeline_thor as P
from flash_rt.models.imagewam.pipeline_thor import _wrap_fp16
from flash_rt.models.imagewam.workload import ImageWAMWorkload

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
PRECISION, SUITE, EPT, CONFIGS, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(","), sys.argv[5]
SEED, NUM_STEPS_WAIT, REPLAN = 7, 30, 10
MAX_STEPS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700}[SUITE]
SUBSETS = {"sub_0_7_9": [0, 7, 9], "sub_0_9": [0, 9], "sub_0_5_9": [0, 5, 9]}


def center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left, top = max((rw - width) // 2, 0), max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def preprocess(obs: dict) -> dict:
    imgs = get_libero_image(obs)
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"])).astype(np.float32)
    return {"view1": torch.from_numpy(center_crop_resize(imgs["image"], 224, 224).copy()),
            "view2": torch.from_numpy(center_crop_resize(imgs["wrist_image"], 224, 224).copy()),
            "proprio": state}


def postprocess(actions: np.ndarray) -> np.ndarray:
    a = actions.copy()
    a[:, -1] = np.sign(-(a[:, -1] * 2 - 1))
    return a


fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
NA, AD = fe.dims["num_action"], fe.dims["action_dim"]
vel_buf = _wrap_fp16(fe._bufs["velocity"], NA, AD)
deltas = [float(x) for x in fe._deltas]
tgrid = [1.0 - sum(abs(d) for d in deltas[:j]) for j in range(len(deltas))] + [0.0]


def velocity(x: torch.Tensor, j: int) -> torch.Tensor:
    fe._action_latent.copy_(x)
    mod_double, mod_single = fe._action_mods[j]
    P.imagewam_denoise_step(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe._active_dims, j,
                            stream=torch.cuda.current_stream().cuda_stream, attn=fe._attn,
                            mod_double=mod_double, mod_single=mod_single, head_mod=fe._head_mods[j],
                            action_rope_table=fe._action_rope_table.data_ptr(), delta=deltas[j])
    return vel_buf.float().clone()


def policy(o: dict, noise: torch.Tensor, config: str) -> np.ndarray:
    if config == "serial10":
        return np.asarray(fe.infer(o, action_noise=noise)["actions"])
    S = SUBSETS[config]
    fe.stage_inputs(o, noise=noise)
    fe.run_eager()
    x = noise.clone()
    for i, s in enumerate(S):
        t_next = tgrid[S[i + 1]] if i + 1 < len(S) else 0.0
        x = x - (tgrid[s] - t_next) * velocity(x, s)
    fe._action_latent.copy_(x)
    return np.asarray(fe.read_actions()).copy()


suite = benchmark.get_benchmark_dict()[SUITE]()
results = {c: [] for c in CONFIGS}
t0 = time.time()
for tid in range(suite.n_tasks):
    task = suite.get_task(tid)
    init_states = suite.get_task_init_states(tid)
    env, desc = get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)
    fe.set_prompt(DEFAULT_PROMPT.format(task=desc))
    for ep in range(EPT):
        for config in CONFIGS:
            gen = torch.Generator(device="cuda").manual_seed(1000 * tid + ep)
            env.reset()
            obs = env.set_init_state(init_states[ep])
            pending, done, t, calls = [], False, 0, 0
            while t < MAX_STEPS + NUM_STEPS_WAIT:
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    t += 1
                    continue
                if not pending:
                    noise = 0.01 * torch.randn(NA, AD, generator=gen, device="cuda")
                    act = postprocess(policy(preprocess(obs), noise, config))
                    if not np.isfinite(act).all():
                        raise RuntimeError(f"non-finite actions {config} task {tid} ep {ep}")
                    pending = act[:REPLAN].tolist()
                    calls += 1
                obs, _, done, _ = env.step(pending.pop(0))
                t += 1
                if done:
                    break
            results[config].append({"task": tid, "episode": ep, "success": bool(done),
                                    "env_steps": t - NUM_STEPS_WAIT, "calls": calls})
        line = " ".join(f"{c}={'S' if results[c][-1]['success'] else 'F'}" for c in CONFIGS)
        print(f"task {tid} ep {ep}: {line}  {time.time() - t0:.0f}s", flush=True)
    env.close()

summary = {c: {"successes": sum(r["success"] for r in results[c]), "episodes": len(results[c]),
               "per_task": [sum(r["success"] for r in results[c] if r["task"] == t) for t in range(suite.n_tasks)],
               "mean_env_steps_success": float(np.mean([r["env_steps"] for r in results[c] if r["success"]] or [0]))}
           for c in CONFIGS}
json.dump({"precision": PRECISION, "suite": SUITE, "episodes_per_task": EPT, "seed": SEED, "replan": REPLAN,
           "subsets": SUBSETS, "tgrid": tgrid, "summary": summary, "results": results}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1))
