"""Is the generating basis shared across the 10 LIBERO-10 tasks?

Representation: `head` (64 x 1024 flattened). TRAIN = the 50-episode set,
TEST = the 20 held-out episodes (2 per task).

1. Reconstruction of task t's TEST samples, h_k = base + V_k V_k^T (h - base):
   - all10:      base/V from TRAIN of all 10 tasks;
   - loto:       base/V from TRAIN of the other 9 tasks (task t never seen);
   - loto+own:   V from the other 9 tasks, base = task t's own TRAIN mean;
   - own:        base/V from task t's own TRAIN (5 episodes).
   Error = ||h - h_k|| / ||h - mu_all10|| (same denominator for every variant).
   Action column: MAE (xyz+rpy) vs the model's own actions when the last
   denoise step's head representation is replaced by h_k (exact: the head is
   a bias-free linear W, last Euler step latent += dt_last * v).
2. Subspace overlap between per-task top-k subspaces (each centered on its own
   task mean): ||U_i^T U_j||_F^2 / k; ceiling = within-task overlap between
   two disjoint episode halves (episodes {0,1} vs {2,3,4} of the 5).

usage: python basis_shared.py TRAIN_NPZ HELDOUT_NPZ OUT_JSON
"""
import json
import os
import sys

import numpy as np
import torch

from flash_rt.models.imagewam.dataset_stats import load_real_normalizers

TR, HO = np.load(sys.argv[1]), np.load(sys.argv[2])
DEV = "cuda"
_, anorm = load_real_normalizers(os.environ["DATASET_STATS_PATH"], device=DEV)
W = torch.from_numpy(TR["head_w"]).to(DEV, torch.float64)
DT = float(TR["dt_last"])
KS = [8, 32, 64, 128]
X = torch.from_numpy(TR["head"].reshape(len(TR["head"]), -1)).to(DEV, torch.float64)
Y = torch.from_numpy(HO["head"].reshape(len(HO["head"]), -1)).to(DEV, torch.float64)
tt, th = TR["meta"][:, 0], HO["meta"][:, 0]
et = TR["meta"][:, 1]
mu_all = X.mean(0, keepdim=True)


def pcs(x: torch.Tensor, kmax: int):
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    ev, U = torch.linalg.eigh(xc @ xc.T)
    ev, U = ev.flip(0).clamp_min(0), U.flip(1)
    r = min(kmax, int((ev > ev[0] * 1e-12).sum()))
    return mu, (xc.T @ U[:, :r]) / ev[:r].sqrt()


def recon_err(y, base, V, k, idx):
    k = min(k, V.shape[1])
    rec = base + ((y - base) @ V[:, :k]) @ V[:, :k].T
    err = float((y - rec).norm() / (y - mu_all).norm())
    lat = torch.from_numpy(HO["latent"][idx]).to(DEV, torch.float64)
    dv = (rec - y).reshape(len(y), 64, -1) @ W
    a = anorm.backward((lat + DT * dv).float()).cpu().numpy()
    mae = float(np.abs(a[..., :6] - HO["actions"][idx][..., :6]).mean())
    return round(err, 4), round(mae, 4)


mu10, V10 = pcs(X, 256)
res = {}
for t in range(10):
    idx = np.where(th == t)[0]
    y = Y[idx]
    mu9, V9 = pcs(X[torch.from_numpy(tt != t).to(DEV)], 256)
    mut, Vt = pcs(X[torch.from_numpy(tt == t).to(DEV)], 256)
    res[t] = {k: {"all10": recon_err(y, mu10, V10, k, idx), "loto": recon_err(y, mu9, V9, k, idx),
                  "loto+own_base": recon_err(y, mut, V9, k, idx), "own": recon_err(y, mut, Vt, k, idx)}
              for k in KS}

summary = {k: {v: [round(float(np.mean([res[t][k][v][j] for t in range(10)])), 4) for j in (0, 1)]
               for v in ("all10", "loto", "loto+own_base", "own")} for k in KS}

overlap = {}
for k in (8, 32):
    U = {t: pcs(X[torch.from_numpy(tt == t).to(DEV)], k)[1] for t in range(10)}
    M = np.array([[float((U[i].T @ U[j]).norm() ** 2 / k) for j in range(10)] for i in range(10)])
    ceil = []
    for t in range(10):
        eps = sorted(set(et[tt == t]))
        a = pcs(X[torch.from_numpy(np.isin(et, eps[:2])).to(DEV)], k)[1]
        b = pcs(X[torch.from_numpy(np.isin(et, eps[2:])).to(DEV)], k)[1]
        ceil.append(float((a.T @ b).norm() ** 2 / k))
    off = M[~np.eye(10, dtype=bool)]
    overlap[k] = {"cross_task_mean": round(float(off.mean()), 3), "cross_task_min": round(float(off.min()), 3),
                  "cross_task_max": round(float(off.max()), 3), "within_task_split_half_mean": round(float(np.mean(ceil)), 3),
                  "random_k_over_D": k / X.shape[1], "matrix": M.round(3).tolist()}

json.dump({"per_task": {str(t): {str(k): v for k, v in r.items()} for t, r in res.items()},
           "mean_over_tasks": summary, "overlap": overlap}, open(sys.argv[3], "w"), indent=1)
print("mean over tasks: [repr err, action MAE vs model]")
for k in KS:
    print(k, summary[k])
print("loto repr err per task at k=32:", {t: res[t][32]["loto"][0] for t in range(10)})
print("all10 repr err per task at k=32:", {t: res[t][32]["all10"][0] for t in range(10)})
for k, o in overlap.items():
    print(f"overlap k={k}: cross-task mean {o['cross_task_mean']} (min {o['cross_task_min']}, max {o['cross_task_max']}), "
          f"within-task split-half {o['within_task_split_half_mean']}, random {o['random_k_over_D']:.1e}")
