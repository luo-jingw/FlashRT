"""Effective rank of the flattened ActionDiT representations.

Each sample's (64, 1024) representation is flattened to one 65536-vector;
the samples form X (N x 65536). Singular values from the Gram matrix
X X^T (float64). Effective rank (Roy & Vetterli 2007):
erank = exp(-sum p_i log p_i), p_i = s_i / sum s. Also reported: the
participation ratio (sum s^2)^2 / sum s^4 and the count of singular
values needed for 90% / 99% of the energy (sum s^2). "centered" subtracts
the mean vector of the set first.

usage: python erank.py NPZ OUT_JSON
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
from _imagewam_libero_frames import _episode_parquet  # noqa: E402

D = np.load(sys.argv[1])
meta = D["meta"]  # (task, episode, frame, n_frames)
ROOT = os.path.join(os.environ["DATA_ROOT"], "libero_10_no_noops_lerobot")


def spectrum(x: np.ndarray, center: bool) -> dict:
    X = torch.from_numpy(x.reshape(len(x), -1)).to("cuda", torch.float64)
    if center:
        X = X - X.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(X @ X.T).clamp_min(0).flip(0)
    s = ev.sqrt()
    p = s / s.sum()
    nz = p[p > 0]
    erank = float(torch.exp(-(nz * nz.log()).sum()))
    e = ev / ev.sum()
    cum = torch.cumsum(e, 0)
    return {"n": len(x), "erank": round(erank, 2), "participation_ratio": round(float(ev.sum() ** 2 / (ev ** 2).sum()), 2),
            "k90": int((cum < 0.90).sum()) + 1, "k99": int((cum < 0.99).sum()) + 1,
            "top5_energy": [round(float(v), 4) for v in e[:5]]}


out = {"file": sys.argv[1], "n_samples": int(len(meta)), "dim": int(np.prod(D["res"].shape[1:]))}
for key in ("res", "head"):
    x = D[key].astype(np.float32)
    r = {"all": {c: spectrum(x, c == "centered") for c in ("uncentered", "centered")}}
    r["per_task"] = {int(t): {c: spectrum(x[meta[:, 0] == t], c == "centered") for c in ("uncentered", "centered")}
                     for t in np.unique(meta[:, 0])}
    r["per_episode_centered_erank"] = {int(e): spectrum(x[meta[:, 1] == e], True)["erank"] for e in np.unique(meta[:, 1])}
    r["norm_mean"] = float(np.linalg.norm(x.reshape(len(x), -1), axis=1).mean())
    out[key] = r

# sanity: predicted actions vs the dataset's ground truth (first 10 steps of each chunk)
maes = []
cache = {}
for i, (t, e, fr, n) in enumerate(meta):
    if e not in cache:
        cache[e] = np.stack(pd.read_parquet(_episode_parquet(ROOT, int(e)))["action"].to_numpy()).astype(np.float32)
    gt = cache[e][fr:fr + 10]
    maes.append(np.abs(D["actions"][i][:len(gt), :6] - gt[:, :6]).mean())
out["sanity_action_mae_first10_xyzrpy"] = float(np.mean(maes))
json.dump(out, open(sys.argv[2], "w"), indent=1)
for key in ("res", "head"):
    a = out[key]["all"]
    print(key, "all uncentered", a["uncentered"], "\n    all centered  ", a["centered"])
    print("   per-task centered erank:", {t: v["centered"]["erank"] for t, v in out[key]["per_task"].items()})
    pe = list(out[key]["per_episode_centered_erank"].values())
    print(f"   per-episode centered erank: mean {np.mean(pe):.2f} min {np.min(pe):.2f} max {np.max(pe):.2f}")
print("action MAE (first 10 steps, xyz+rpy):", round(out["sanity_action_mae_first10_xyzrpy"], 4))
