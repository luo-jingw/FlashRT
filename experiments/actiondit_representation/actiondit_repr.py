"""Collect ImageWAM ActionDiT pre-head representations on LIBERO-10 dataset frames.

For each of the 10 libero_10 tasks, 5 episodes drawn at random (seed 0); in
each episode every STRIDE-th frame (0, STRIDE, 2*STRIDE, ...) is one
open-loop `infer()` on the dataset's two camera frames + proprio state,
prompt = official DEFAULT_PROMPT.format(task=...). After each replay the
last denoise step's buffers are read:

- `res`:  `bufs["action_hidden"]`, the ActionDiT residual stream after the
          last block, (num_action=64, action_hidden_dim=1024) fp16;
- `head`: `bufs["head_modded"]`, the head's AdaLN output = input of the
          final `head.linear` that produces the velocity.

Initial action latent per sample: 0.01 * N(0, 1) (the served sampler's
distribution) from a generator seeded with the sample index.

usage: python actiondit_repr.py PRECISION STRIDE OUT_NPZ [EPS_PER_TASK SEED EXCLUDE_JSON]
(EXCLUDE_JSON: a previous run's .json; its picked episodes are not drawn.)
"""
import json
import os
import sys
import time

import av
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
from _imagewam_libero_frames import _episode_parquet, _episode_video, _read_jsonl, center_crop_resize  # noqa: E402

import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
from flash_rt.models.imagewam.pipeline_thor import _wrap_fp16  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

PRECISION, STRIDE, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
EPS_PER_TASK = int(sys.argv[4]) if len(sys.argv) > 4 else 5
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
EXCLUDE = ({e for v in json.load(open(sys.argv[6]))["picks"].values() for e in v} if len(sys.argv) > 6 else set())
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROOT = os.path.join(os.environ["DATA_ROOT"], "libero_10_no_noops_lerobot")


def decode(path: str) -> list:
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


tasks = {t["task_index"]: t["task"] for t in _read_jsonl(f"{ROOT}/meta/tasks.jsonl")}
by_task: dict[int, list[int]] = {}
for ep in _read_jsonl(f"{ROOT}/meta/episodes.jsonl"):
    e = int(ep["episode_index"])
    ti = int(pd.read_parquet(_episode_parquet(ROOT, e), columns=["task_index"])["task_index"].iloc[0])
    by_task.setdefault(ti, []).append(e)
rng = np.random.default_rng(SEED)
picks = {ti: sorted(rng.choice([e for e in eps if e not in EXCLUDE], EPS_PER_TASK, replace=False).tolist())
         for ti, eps in sorted(by_task.items())}
print("episodes per task:", {ti: len(v) for ti, v in sorted(by_task.items())})
print("picked:", picks, flush=True)

fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
na, ahd = fe.dims["num_action"], fe.dims["action_hidden_dim"]
res_buf = _wrap_fp16(fe._bufs["action_hidden"], na, ahd)
head_buf = _wrap_fp16(fe._bufs["head_modded"], na, ahd)
ad = fe.dims["action_dim"]
vel_buf = _wrap_fp16(fe._bufs["velocity"], na, ad)
head_lin = fe.weights[("action_dit", "shared", 0, "head.linear.weight")]
head_w = _wrap_fp16(head_lin.weight_ptr, head_lin.k, head_lin.n).float().cpu().numpy()  # (K=1024, N=7)
dt_last = float(fe._deltas[-1]) if fe._deltas is not None else float(fe.dims["dt"])
print(f"built {PRECISION}: num_action={na} action_hidden_dim={ahd} vec_rms_norm={fe.dims.get('vec_rms_norm')} "
      f"fuse_qkv_norm_rope={fe.dims.get('fuse_qkv_norm_rope')}", flush=True)

res, head, acts, meta, vel, lat = [], [], [], [], [], []
t0 = time.time()
for ti, eps in picks.items():
    fe.set_prompt(DEFAULT_PROMPT.format(task=tasks[ti]))
    for e in eps:
        df = pd.read_parquet(_episode_parquet(ROOT, e))
        v1 = decode(_episode_video(ROOT, "observation.images.image", e))
        v2 = decode(_episode_video(ROOT, "observation.images.wrist_image", e))
        n = min(len(df), len(v1), len(v2))
        for fr in range(0, n, STRIDE):
            g = torch.Generator(device="cuda").manual_seed(len(meta))
            noise = 0.01 * torch.randn(na, fe.dims["action_dim"], generator=g, device="cuda")
            obs = {"view1": torch.from_numpy(center_crop_resize(v1[fr], 224, 224)),
                   "view2": torch.from_numpy(center_crop_resize(v2[fr], 224, 224)),
                   "proprio": np.asarray(df["observation.state"].iloc[fr], dtype=np.float32)}
            a = fe.infer(obs, action_noise=noise)["actions"]
            res.append(res_buf.cpu().numpy().copy())
            head.append(head_buf.cpu().numpy().copy())
            acts.append(a)
            vel.append(vel_buf.float().cpu().numpy().copy())
            lat.append(fe._action_latent.float().cpu().numpy().copy())
            meta.append((ti, e, fr, n))
        print(f"task {ti} episode {e}: {n} frames, samples so far {len(meta)}, {time.time() - t0:.0f}s", flush=True)

res, head, acts = np.stack(res), np.stack(head), np.stack(acts)
for k, v in (("res", res), ("head", head)):
    if not np.isfinite(v.astype(np.float32)).all():
        raise RuntimeError(f"non-finite values in {k}")
np.savez(OUT, res=res, head=head, actions=acts, meta=np.array(meta, dtype=np.int64),
         velocity=np.stack(vel), latent=np.stack(lat), head_w=head_w, dt_last=dt_last,
         tasks=np.array([tasks[i] for i in sorted(tasks)]))
json.dump({"precision": PRECISION, "stride": STRIDE, "eps_per_task": EPS_PER_TASK, "seed": SEED, "picks": picks,
           "n_samples": len(meta), "shape": list(res.shape)}, open(OUT.replace(".npz", ".json"), "w"), indent=1)
print(f"saved {OUT}: res {res.shape} head {head.shape} actions {acts.shape}")
