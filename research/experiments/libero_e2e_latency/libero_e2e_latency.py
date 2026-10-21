"""End-to-end policy latency of FlashRT ImageWAM in the LIBERO simulator (closed loop).

Per policy call, wall clock (time.perf_counter) of:
  pre    raw env obs dict -> model input: get_libero_image (180 deg rotation),
         PIL center-crop-resize 256->224 of both cameras, 8-dim proprio
         (eef pos + axis-angle + gripper qpos);
  infer  fe.infer(obs) -> numpy actions (H2D, VAE + prefill + denoise graphs,
         D2H, denormalization);
  post   gripper mapping (dataset 0=close/1=open -> LIBERO -1=open/+1=close)
         and binarization;
  e2e    pre + infer + post.
Per task: set_prompt wall time (Qwen3 encoding of DEFAULT_PROMPT.format(task)),
active x0, first call (includes graph capture for a new text length) reported
separately; steady-state stats exclude each task's first call.
Preprocessing and gripper mapping follow
tmp/ImageWAM/experiments/libero/eval_libero_single.py; num_steps_wait 30,
replan_steps 10, seed 7.

usage: python libero_e2e_latency.py PRECISION SUITE EPISODES_PER_TASK OUT_JSON
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

import flash_rt.frontends.torch.imagewam_thor as F
from flash_rt.models.imagewam.workload import ImageWAMWorkload

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
PRECISION, SUITE, EPT, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
SEED, NUM_STEPS_WAIT, REPLAN = 7, 30, 10
MAX_STEPS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700}[SUITE]


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


t0 = time.time()
fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
build_s = time.time() - t0
opts = fe._resolved.options
print(f"built {PRECISION} in {build_s:.0f}s text_trim={opts.text_trim} use_fa4={fe.use_fa4} "
      f"vec_rms_norm={fe.dims.get('vec_rms_norm')} fuse_qkv_norm_rope={fe.dims.get('fuse_qkv_norm_rope')}", flush=True)

suite = benchmark.get_benchmark_dict()[SUITE]()
tasks, steady = [], {"pre": [], "infer": [], "post": [], "e2e": [], "env_step": []}
for tid in range(suite.n_tasks):
    task = suite.get_task(tid)
    init_states = suite.get_task_init_states(tid)
    env, desc = get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)
    s = time.perf_counter()
    fe.set_prompt(DEFAULT_PROMPT.format(task=desc))
    torch.cuda.synchronize()
    set_prompt_ms = 1000 * (time.perf_counter() - s)
    first_ms, succ, n_calls = None, 0, 0
    for ep in range(EPT):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        pending, done, t = [], False, 0
        while t < MAX_STEPS + NUM_STEPS_WAIT:
            if t < NUM_STEPS_WAIT:
                obs, _, done, _ = env.step(get_libero_dummy_action())
                t += 1
                continue
            if not pending:
                c0 = time.perf_counter()
                o = preprocess(obs)
                c1 = time.perf_counter()
                raw = fe.infer(o)["actions"]
                c2 = time.perf_counter()
                act = postprocess(np.asarray(raw))
                c3 = time.perf_counter()
                if not np.isfinite(act).all():
                    raise RuntimeError(f"non-finite actions task {tid} t={t}")
                ms = [1000 * (c1 - c0), 1000 * (c2 - c1), 1000 * (c3 - c2), 1000 * (c3 - c0)]
                if first_ms is None:
                    first_ms = ms
                else:
                    for k, v in zip(("pre", "infer", "post", "e2e"), ms):
                        steady[k].append(v)
                n_calls += 1
                pending = act[:REPLAN].tolist()
            e0 = time.perf_counter()
            obs, _, done, _ = env.step(pending.pop(0))
            steady["env_step"].append(1000 * (time.perf_counter() - e0))
            t += 1
            if done:
                break
        succ += int(done)
    env.close()
    rec = {"task_id": tid, "task": desc, "x0": int(fe._active_dims["x0"]), "set_prompt_ms": round(set_prompt_ms, 1),
           "first_call_ms": [round(v, 2) for v in first_ms], "calls": n_calls, "successes": succ, "episodes": EPT}
    tasks.append(rec)
    print(json.dumps(rec), flush=True)


def stats(x: list) -> dict:
    a = np.asarray(x)
    return {"n": int(a.size), "mean": round(float(a.mean()), 3), "p50": round(float(np.median(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3), "p99": round(float(np.percentile(a, 99)), 3),
            "max": round(float(a.max()), 3)}


summary = {"precision": PRECISION, "suite": SUITE, "episodes_per_task": EPT, "seed": SEED, "replan_steps": REPLAN,
           "num_steps_wait": NUM_STEPS_WAIT, "max_steps": MAX_STEPS, "build_s": round(build_s, 1),
           "gpu": torch.cuda.get_device_name(), "steady": {k: stats(v) for k, v in steady.items()},
           "successes": sum(t["successes"] for t in tasks), "episodes": EPT * len(tasks), "tasks": tasks}
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps(summary["steady"]), f"success {summary['successes']}/{summary['episodes']}", flush=True)
