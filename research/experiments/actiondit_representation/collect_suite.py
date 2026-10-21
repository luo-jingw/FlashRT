"""Collect ImageWAM ActionDiT head representations on one LIBERO suite (open loop, dataset frames).

For every task of SUITE, EPS_PER_TASK episodes (-1 = all; otherwise drawn at
random with SEED, sorted); in each episode every STRIDE-th frame is one
`infer()` on the dataset's two camera frames + proprio state, prompt =
official DEFAULT_PROMPT.format(task=...). After each replay (last denoise step):
`head` = `bufs["head_modded"]` (64 x 1024 fp16, input of the bias-free
head.linear), `velocity` (64 x 7), `latent` (final normalized action, f32),
`actions` (denormalized). Initial latent: 0.01 * N(0,1) from a generator
seeded with NOISE_OFFSET + global sample index (served sampler's distribution).
EPISODE_FILE (optional): a JSON list of episode indices; only those are run
(overrides EPS_PER_TASK/SEED).

usage: python collect_suite.py PRECISION SUITE STRIDE EPS_PER_TASK SEED NOISE_OFFSET OUT_NPZ [EPISODE_FILE]
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

PRECISION, SUITE, STRIDE, EPT, SEED, NOFF, OUT = (sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                                                  int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
EP_FILE = sys.argv[8] if len(sys.argv) > 8 else None
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROOT = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")


def decode(path: str) -> list:
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


tasks = {t["task_index"]: t["task"] for t in _read_jsonl(f"{ROOT}/meta/tasks.jsonl")}
by_task: dict[int, list[int]] = {}
for ep in _read_jsonl(f"{ROOT}/meta/episodes.jsonl"):
    e = int(ep["episode_index"])
    ti = int(pd.read_parquet(_episode_parquet(ROOT, e), columns=["task_index"])["task_index"].iloc[0])
    by_task.setdefault(ti, []).append(e)
if EP_FILE:
    keep = set(json.load(open(EP_FILE)))
    picks = {ti: sorted(e for e in eps if e in keep) for ti, eps in sorted(by_task.items())}
elif EPT < 0:
    picks = {ti: sorted(eps) for ti, eps in sorted(by_task.items())}
else:
    rng = np.random.default_rng(SEED)
    picks = {ti: sorted(rng.choice(eps, min(EPT, len(eps)), replace=False).tolist()) for ti, eps in sorted(by_task.items())}
print(SUITE, "episodes per task:", {ti: len(v) for ti, v in picks.items()}, flush=True)

fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
na, ahd, ad = fe.dims["num_action"], fe.dims["action_hidden_dim"], fe.dims["action_dim"]
head_buf = _wrap_fp16(fe._bufs["head_modded"], na, ahd)
vel_buf = _wrap_fp16(fe._bufs["velocity"], na, ad)
head_lin = fe.weights[("action_dit", "shared", 0, "head.linear.weight")]
head_w = _wrap_fp16(head_lin.weight_ptr, head_lin.k, head_lin.n).float().cpu().numpy()
dt_last = float(fe._deltas[-1]) if fe._deltas is not None else float(fe.dims["dt"])

n_total = sum((len(pd.read_parquet(_episode_parquet(ROOT, e), columns=["task_index"])) + STRIDE - 1) // STRIDE
              for eps in picks.values() for e in eps)
head = np.empty((n_total, na, ahd), dtype=np.float16)
vel = np.empty((n_total, na, ad), dtype=np.float32)
lat = np.empty((n_total, na, ad), dtype=np.float32)
acts = np.empty((n_total, na, ad), dtype=np.float32)
meta = []
t0 = time.time()
for ti, eps in picks.items():
    fe.set_prompt(DEFAULT_PROMPT.format(task=tasks[ti]))
    for e in eps:
        df = pd.read_parquet(_episode_parquet(ROOT, e))
        v1 = decode(_episode_video(ROOT, "observation.images.image", e))
        v2 = decode(_episode_video(ROOT, "observation.images.wrist_image", e))
        n = len(df)
        if len(v1) < n or len(v2) < n:
            raise RuntimeError(f"episode {e}: {n} rows but {len(v1)}/{len(v2)} video frames")
        for fr in range(0, n, STRIDE):
            i = len(meta)
            g = torch.Generator(device="cuda").manual_seed(NOFF + i)
            noise = 0.01 * torch.randn(na, ad, generator=g, device="cuda")
            obs = {"view1": torch.from_numpy(center_crop_resize(v1[fr], 224, 224)),
                   "view2": torch.from_numpy(center_crop_resize(v2[fr], 224, 224)),
                   "proprio": np.array(df["observation.state"].iloc[fr], dtype=np.float32)}
            acts[i] = fe.infer(obs, action_noise=noise)["actions"]
            head[i] = head_buf.cpu().numpy()
            vel[i] = vel_buf.float().cpu().numpy()
            lat[i] = fe._action_latent.cpu().numpy()
            meta.append((ti, e, fr, n))
    print(f"{SUITE} task {ti}: {len(eps)} episodes, samples {len(meta)}/{n_total}, {time.time() - t0:.0f}s", flush=True)

assert len(meta) == n_total, (len(meta), n_total)
if not np.isfinite(head.astype(np.float32)).all():
    raise RuntimeError("non-finite head representation")
np.savez(OUT, head=head, velocity=vel, latent=lat, actions=acts, meta=np.array(meta, dtype=np.int64),
         head_w=head_w, dt_last=dt_last, suite=SUITE, tasks=np.array([tasks[i] for i in sorted(tasks)]))
json.dump({"precision": PRECISION, "suite": SUITE, "stride": STRIDE, "eps_per_task": EPT, "seed": SEED,
           "noise_offset": NOFF, "picks": picks, "n_samples": n_total}, open(OUT.replace(".npz", ".json"), "w"), indent=1)
print(f"saved {OUT}: head {head.shape}, {time.time() - t0:.0f}s")
