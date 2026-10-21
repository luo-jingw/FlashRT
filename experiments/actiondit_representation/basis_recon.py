"""Can held-out samples be written as base + weighted sum of a basis built from the 50 episodes?

Basis (from TRAIN = the 50-episode set, each sample's (64, 1024) representation
flattened to 65536): base = mean vector mu; generating basis = the top-k
principal directions V_k of the centered TRAIN matrix. A sample h is
reconstructed as h_k = mu + V_k V_k^T (h - mu).

Representation metric: relative error ||h - h_k|| / ||h - mu|| over the set
(Frobenius, = sqrt(1 - explained variance)).

Action metric (head representation only): the head is AdaLN + a bias-free
linear W (1024 -> 7), and the last Euler step is latent += dt_last * v, so
replacing h by h_k changes the final normalized action by
dt_last * (h_k - h) W exactly; denormalized with the dataset action min/max.
Reported against the model's own actions (MAE over xyz+rpy, gripper sign
agreement) and against the dataset ground truth (first 10 steps, xyz+rpy).

Baseline: the same base + top-k construction directly on the action chunks
(64 x 7 flattened) of TRAIN.

usage: python basis_recon.py TRAIN_NPZ HELDOUT_NPZ OUT_JSON
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
from _imagewam_libero_frames import _episode_parquet  # noqa: E402
from flash_rt.models.imagewam.dataset_stats import load_real_normalizers  # noqa: E402

TR, HO = np.load(sys.argv[1]), np.load(sys.argv[2])
ROOT = os.path.join(os.environ["DATA_ROOT"], "libero_10_no_noops_lerobot")
DEV = "cuda"
_, anorm = load_real_normalizers(os.environ["DATASET_STATS_PATH"], device=DEV)
W = torch.from_numpy(TR["head_w"]).to(DEV, torch.float64)  # (1024, 7)
DT = float(TR["dt_last"])
KS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1386]


def gt_chunks(meta: np.ndarray) -> list:
    cache, out = {}, []
    for t, e, fr, n in meta:
        if e not in cache:
            cache[e] = np.stack(pd.read_parquet(_episode_parquet(ROOT, int(e)))["action"].to_numpy()).astype(np.float32)
        out.append(cache[e][fr:fr + 10])
    return out


def basis(x: torch.Tensor):
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    ev, U = torch.linalg.eigh(xc @ xc.T)
    ev, U = ev.flip(0).clamp_min(0), U.flip(1)
    keep = ev > ev[0] * 1e-12
    s = ev[keep].sqrt()
    V = (xc.T @ U[:, keep]) / s  # (D, r), orthonormal columns
    return mu, V


def checks(D, name: str) -> dict:
    h = torch.from_numpy(D["head"]).to(DEV, torch.float64)
    v = h @ W
    vb = torch.from_numpy(D["velocity"]).to(DEV, torch.float64)
    a = anorm.backward(torch.from_numpy(D["latent"]).to(DEV)).cpu().numpy()
    return {f"{name}_velocity_rel_err": float((v - vb).norm() / vb.norm()),
            f"{name}_denorm_max_abs": float(np.abs(a - D["actions"]).max())}


def evaluate(key: str, mu, V, D, gts) -> list:
    n = len(D[key])
    x = torch.from_numpy(D[key].reshape(n, -1)).to(DEV, torch.float64)
    xc = x - mu
    c = xc @ V
    base_norm = xc.norm()
    lat = torch.from_numpy(D["latent"]).to(DEV, torch.float64)
    acts = D["actions"]
    rows = []
    for k in KS:
        k = min(k, V.shape[1])
        rec = mu + c[:, :k] @ V[:, :k].T
        row = {"k": k, "repr_rel_err": round(float((x - rec).norm() / base_norm), 4)}
        if key == "head":
            dv = (rec - x).reshape(n, 64, -1) @ W
            a_rec = anorm.backward((lat + DT * dv).float()).cpu().numpy()
            row["act_mae_vs_model"] = round(float(np.abs(a_rec[..., :6] - acts[..., :6]).mean()), 4)
            row["gripper_sign_agree"] = round(float((np.sign(a_rec[..., 6]) == np.sign(acts[..., 6])).mean()), 4)
            row["act_mae_vs_gt"] = round(float(np.mean([np.abs(a_rec[i, :len(g), :6] - g[:, :6]).mean()
                                                        for i, g in enumerate(gts)])), 4)
        rows.append(row)
    return rows


out = {"n_train": int(len(TR["head"])), "n_heldout": int(len(HO["head"])), "dt_last": DT}
out.update(checks(TR, "train")); out.update(checks(HO, "heldout"))
gts_tr, gts_ho = gt_chunks(TR["meta"]), gt_chunks(HO["meta"])
out["model_act_mae_vs_gt"] = {
    "train": round(float(np.mean([np.abs(TR["actions"][i, :len(g), :6] - g[:, :6]).mean() for i, g in enumerate(gts_tr)])), 4),
    "heldout": round(float(np.mean([np.abs(HO["actions"][i, :len(g), :6] - g[:, :6]).mean() for i, g in enumerate(gts_ho)])), 4)}
for key in ("head", "res"):
    xt = torch.from_numpy(TR[key].reshape(len(TR[key]), -1)).to(DEV, torch.float64)
    mu, V = basis(xt)
    out[key] = {"rank": int(V.shape[1]), "train": evaluate(key, mu, V, TR, gts_tr),
                "heldout": evaluate(key, mu, V, HO, gts_ho)}
    del xt, mu, V
    torch.cuda.empty_cache()

# baseline: base + top-k directly on action chunks
at = torch.from_numpy(TR["actions"].reshape(len(TR["actions"]), -1)).to(DEV, torch.float64)
mu_a, V_a = basis(at)
ah = torch.from_numpy(HO["actions"].reshape(len(HO["actions"]), -1)).to(DEV, torch.float64)
rows = []
for k in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 448]:
    k = min(k, V_a.shape[1])
    rec = (mu_a + ((ah - mu_a) @ V_a[:, :k]) @ V_a[:, :k].T).reshape(-1, 64, 7).cpu().numpy()
    rows.append({"k": k, "act_mae_vs_model": round(float(np.abs(rec[..., :6] - HO["actions"][..., :6]).mean()), 4),
                 "gripper_sign_agree": round(float((np.sign(rec[..., 6]) == np.sign(HO["actions"][..., 6])).mean()), 4)})
out["action_pca_heldout"] = rows
json.dump(out, open(sys.argv[3], "w"), indent=1)

print({k: v for k, v in out.items() if k not in ("head", "res", "action_pca_heldout")})
print("k | head repr err train / heldout | heldout act MAE vs model | gripper agree | act MAE vs GT | res repr err heldout")
for i, k in enumerate(KS):
    ht, hh, rh = out["head"]["train"][i], out["head"]["heldout"][i], out["res"]["heldout"][i]
    print(f"{hh['k']:5d} | {ht['repr_rel_err']:.4f} / {hh['repr_rel_err']:.4f} | {hh['act_mae_vs_model']:.4f} | "
          f"{hh['gripper_sign_agree']:.4f} | {hh['act_mae_vs_gt']:.4f} | {rh['repr_rel_err']:.4f}")
print("action-PCA baseline (heldout):", [(r["k"], r["act_mae_vs_model"], r["gripper_sign_agree"]) for r in rows])
