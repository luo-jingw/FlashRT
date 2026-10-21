"""Is the ActionDiT generating basis shared across tasks? Full-scale statistics.

Inputs (collect_suite.py outputs): libero_10_all.npz (all episodes),
libero_{spatial,object,goal}_20.npz, libero_10_fold0_noise2.npz (fold-0
episodes again with different initial noise), libero10_folds.json.
Representation: `head` (64 x 1024 -> 65536). Reconstruction of h with a
basis (base mu, top-k principal directions V_k of the centered basis set):
h_k = mu + V_k V_k^T (h - mu). Episode-level error: ||H_e - H_e,k||_F /
||H_e - mu_ref||_F over the episode's frames, mu_ref = mean of the fold's
LIBERO-10 training set (same denominator for every method).

A. LIBERO-10, 5-fold CV by episode (stratified by task). Per test episode of
   task t: bases all10 / loto (other 9 tasks) / own (task t), at full size
   and matched size (all10 / loto subsampled by episode to task t's training
   size). Paired per-episode differences: mean, 95% bootstrap CI (episodes
   resampled, 10000 draws), Wilcoxon signed-rank p.
   Action column: last-denoise-step replacement, exact through the bias-free
   head (MAE xyz+rpy vs the model's own actions, per episode).
B. Subspace overlap ||U_i^T U_j||_F^2 / k between per-task subspaces, each
   from a random half of the task's episodes (matched size); within-task =
   the two halves of the same task; 20 random splits.
C. Cross-suite: each other suite's episodes reconstructed with (i) a
   LIBERO-10 basis (all episodes, and subsampled to the suite's training
   size) vs (ii) the suite's own basis, 5-fold CV by episode inside the suite.
D. Noise floor: fold-0 episodes, ||H_seedA - H_seedB|| / ||H_seedA - mu_ref||.
E. Effective rank (Roy & Vetterli) of the centered matrix at a fixed N = 1000
   samples drawn by whole episodes, 50 bootstrap draws, per suite.

usage: python shared_basis_stats.py DIR OUT_JSON
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import wilcoxon

from flash_rt.models.imagewam.dataset_stats import load_real_normalizers

D, OUT = sys.argv[1], sys.argv[2]
DEV = "cuda"
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
KMAX = max(KS)
rng = np.random.default_rng(0)
_, anorm = load_real_normalizers(os.environ["DATASET_STATS_PATH"], device=DEV)


def load(name):
    z = np.load(f"{D}/{name}.npz")
    return {"X": torch.from_numpy(z["head"].reshape(len(z["head"]), -1)).to(DEV, torch.float32),
            "meta": z["meta"], "lat": z["latent"], "act": z["actions"], "W": z["head_w"], "dt": float(z["dt_last"])}


L10 = load("libero_10_all")
W = torch.from_numpy(L10["W"]).to(DEV, torch.float32)
DT = L10["dt"]
folds = {int(k): v for k, v in json.load(open(f"{D}/libero10_folds.json")).items()}
fold_of = np.array([folds[int(e)] for e in L10["meta"][:, 1]])


def basis(x: torch.Tensor, kmax: int = KMAX):
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    ev, U = torch.linalg.eigh((xc @ xc.T).double())
    ev, U = ev.flip(0).clamp_min(0), U.flip(1)
    r = min(kmax, int((ev > ev[0] * 1e-10).sum()))
    V = (xc.T @ U[:, :r].float()) / ev[:r].sqrt().float()
    return mu, V


def recon_errors(y, mu, V, mu_ref, ks=KS):
    """(len(ks),) Frobenius error of y's reconstruction relative to ||y - mu_ref||, plus the rec tensors' dv."""
    c = (y - mu) @ V
    den = (y - mu_ref).norm()
    out = []
    for k in ks:
        k = min(k, V.shape[1])
        rec = mu + c[:, :k] @ V[:, :k].T
        out.append(rec)
    return [float((y - r).norm() / den) for r in out], out


def action_mae(y, rec, lat, act):
    dv = (rec - y).reshape(len(y), 64, -1) @ W
    a = anorm.backward(torch.from_numpy(lat).to(DEV) + DT * dv).cpu().numpy()
    return float(np.abs(a[..., :6] - act[..., :6]).mean())


def episodes_upto(cands: np.ndarray, sizes: dict, n: int) -> list:
    """Random whole episodes from `cands` until their sample count reaches n."""
    perm = rng.permutation(cands)
    out, tot = [], 0
    for e in perm:
        if tot >= n:
            break
        out.append(e)
        tot += sizes[e]
    return out


def paired(a, b):
    """mean(a - b), 95% bootstrap CI over episodes, Wilcoxon p."""
    d = np.asarray(a) - np.asarray(b)
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_wilcoxon": p, "n": int(len(d))}


def ci(v):
    v = np.asarray(v)
    boots = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": int(len(v))}


out = {"n_samples": {"libero_10": int(len(L10["meta"]))}, "ks": KS}

# ---------------- A. LIBERO-10 cross-validation ----------------
meta = L10["meta"]
task, ep = meta[:, 0], meta[:, 1]
sizes = {e: int((ep == e).sum()) for e in np.unique(ep)}
methods = ["all10", "loto", "own", "all10_matched", "loto_matched"]
rows = []  # one per test episode
for f in range(5):
    tr = fold_of != f
    Xtr = L10["X"][torch.from_numpy(tr).to(DEV)]
    mu_ref = Xtr.mean(0, keepdim=True)
    B_all = basis(Xtr)
    for t in range(10):
        own_mask = tr & (task == t)
        oth_mask = tr & (task != t)
        n_own = int(own_mask.sum())
        B = {"all10": B_all,
             "loto": basis(L10["X"][torch.from_numpy(oth_mask).to(DEV)]),
             "own": basis(L10["X"][torch.from_numpy(own_mask).to(DEV)])}
        m_all = np.isin(ep, episodes_upto(np.unique(ep[tr]), sizes, n_own))
        m_oth = np.isin(ep, episodes_upto(np.unique(ep[oth_mask]), sizes, n_own))
        B["all10_matched"] = basis(L10["X"][torch.from_numpy(m_all).to(DEV)])
        B["loto_matched"] = basis(L10["X"][torch.from_numpy(m_oth).to(DEV)])
        for e in np.unique(ep[(fold_of == f) & (task == t)]):
            idx = np.where(ep == e)[0]
            y = L10["X"][torch.from_numpy(idx).to(DEV)]
            row = {"fold": f, "task": t, "episode": int(e), "n": len(idx)}
            for m in methods:
                errs, recs = recon_errors(y, *B[m], mu_ref)
                row[m] = errs
                row[m + "_act"] = [action_mae(y, r, L10["lat"][idx], L10["act"][idx]) for r in recs]
            rows.append(row)
    print(f"A: fold {f} done, {len(rows)} test episodes", flush=True)

A = {"n_test_episodes": len(rows), "mean_error": {}, "mean_action_mae": {}, "paired": {}}
for m in methods:
    A["mean_error"][m] = [ci([r[m][i] for r in rows]) for i in range(len(KS))]
    A["mean_action_mae"][m] = [ci([r[m + "_act"][i] for r in rows]) for i in range(len(KS))]
for a, b in [("loto", "all10"), ("own", "all10"), ("own", "all10_matched"), ("loto_matched", "all10_matched"),
             ("own", "loto_matched")]:
    A["paired"][f"{a}-{b}"] = [paired([r[a][i] for r in rows], [r[b][i] for r in rows]) for i in range(len(KS))]
A["paired_action"] = {f"{a}-{b}": [paired([r[a + '_act'][i] for r in rows], [r[b + '_act'][i] for r in rows])
                                   for i in range(len(KS))] for a, b in [("loto", "all10"), ("own", "all10_matched")]}
out["A_libero10_cv"] = A
torch.cuda.empty_cache()

# ---------------- B. subspace overlap, matched size ----------------
eps_by_task = {t: np.unique(ep[task == t]) for t in range(10)}
B_res = {k: {"within": [], "cross": []} for k in (8, 16, 32, 64)}
for r in range(20):
    halves = {}
    for t in range(10):
        p = rng.permutation(eps_by_task[t])
        h = len(p) // 2
        halves[t] = [basis(L10["X"][torch.from_numpy(np.isin(ep, p[:h])).to(DEV)], 64)[1],
                     basis(L10["X"][torch.from_numpy(np.isin(ep, p[h:2 * h])).to(DEV)], 64)[1]]
    for k in B_res:
        for t in range(10):
            B_res[k]["within"].append(float((halves[t][0][:, :k].T @ halves[t][1][:, :k]).norm() ** 2 / k))
            for s in range(10):
                if s != t:
                    B_res[k]["cross"].append(float((halves[t][0][:, :k].T @ halves[s][1][:, :k]).norm() ** 2 / k))
    print(f"B: split {r} done", flush=True)
out["B_overlap"] = {k: {"within": ci(v["within"]), "cross": ci(v["cross"]), "random_k_over_D": k / L10["X"].shape[1]}
                    for k, v in B_res.items()}
torch.cuda.empty_cache()

# ---------------- C. cross-suite ----------------
C = {}
B_l10_full = basis(L10["X"])
l10_eps = np.unique(ep)
for name in ("libero_spatial_20", "libero_object_20", "libero_goal_20"):
    S = load(name)
    s_ep = S["meta"][:, 1]
    s_sizes = {e: int((s_ep == e).sum()) for e in np.unique(s_ep)}
    ueps = rng.permutation(np.unique(s_ep))
    s_fold = {e: i % 5 for i, e in enumerate(ueps)}
    sf = np.array([s_fold[e] for e in s_ep])
    rows_c = []
    for f in range(5):
        tr = sf != f
        Xtr = S["X"][torch.from_numpy(tr).to(DEV)]
        mu_ref = Xtr.mean(0, keepdim=True)
        B_own = basis(Xtr)
        B_l10m = basis(L10["X"][torch.from_numpy(np.isin(ep, episodes_upto(l10_eps, sizes, int(tr.sum())))).to(DEV)])
        for e in np.unique(s_ep[sf == f]):
            idx = np.where(s_ep == e)[0]
            y = S["X"][torch.from_numpy(idx).to(DEV)]
            rows_c.append({"own": recon_errors(y, *B_own, mu_ref)[0],
                           "libero10_full": recon_errors(y, *B_l10_full, mu_ref)[0],
                           "libero10_matched": recon_errors(y, *B_l10m, mu_ref)[0]})
    C[name] = {"n_samples": int(len(s_ep)), "n_test_episodes": len(rows_c),
               "mean_error": {m: [ci([r[m][i] for r in rows_c]) for i in range(len(KS))] for m in rows_c[0]},
               "paired": {f"libero10_{v}-own": [paired([r[f'libero10_{v}'][i] for r in rows_c], [r['own'][i] for r in rows_c])
                                               for i in range(len(KS))] for v in ("full", "matched")}}
    out["n_samples"][name] = int(len(s_ep))
    del S
    torch.cuda.empty_cache()
    print(f"C: {name} done", flush=True)
out["C_cross_suite"] = C

# ---------------- D. noise floor ----------------
N2 = load("libero_10_fold0_noise2")
key = {(int(e), int(fr)): i for i, (t, e, fr, n) in enumerate(meta)}
idx_a = np.array([key[(int(e), int(fr))] for t, e, fr, n in N2["meta"]])
mu_ref = L10["X"][torch.from_numpy(fold_of != 0).to(DEV)].mean(0, keepdim=True)
nf = []
for e in np.unique(N2["meta"][:, 1]):
    m = N2["meta"][:, 1] == e
    ya = L10["X"][torch.from_numpy(idx_a[m]).to(DEV)]
    yb = N2["X"][torch.from_numpy(np.where(m)[0]).to(DEV)]
    nf.append(float((ya - yb).norm() / (ya - mu_ref).norm()))
out["D_noise_floor"] = ci(nf)
del N2

# ---------------- E. effective rank at fixed N ----------------
def erank(x):
    xc = x - x.mean(0, keepdim=True)
    s = torch.linalg.eigvalsh((xc @ xc.T).double()).clamp_min(0).sqrt()
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


E = {}
for name in ("libero_10_all", "libero_spatial_20", "libero_object_20", "libero_goal_20"):
    S = L10 if name == "libero_10_all" else load(name)
    s_ep = S["meta"][:, 1]
    s_sizes = {e: int((s_ep == e).sum()) for e in np.unique(s_ep)}
    vals = []
    for _ in range(50):
        pick = np.isin(s_ep, episodes_upto(np.unique(s_ep), s_sizes, 1000))
        idx = np.where(pick)[0][:1000]
        vals.append(erank(S["X"][torch.from_numpy(idx).to(DEV)]))
    E[name] = ci(vals)
out["E_erank_N1000"] = E

json.dump(out, open(OUT, "w"), indent=1)
print("done")
