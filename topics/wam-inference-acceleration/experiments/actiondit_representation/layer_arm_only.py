"""ActionDiT per-layer token structure with the gripper direction projected out (arm motion only).

Per (step, layer), pooled tokens X (N*64, 1024). Gripper direction
u = normalized (mean of tokens whose predicted gripper > 0.5) - (mean of the
rest), computed on all tokens of that (step, layer); X' = X - (X u) u^T.
Checks on X': |corr| of PC1..PC5 with the predicted gripper value, and
PC1 two-cluster gap / within-cluster sd (split by gripper).

On X' (same definitions as layer_lowrank.py): erank / k90 / k99 per task,
pooled, pooled same-N; covariance ellipsoid leave-one-episode-out coverage
and out-of-subspace variance (own / pooled / other task), k in KS.
Arm decodability: leave-one-episode-out over all 15 episodes; PCA top-k of
the training tokens' X', least-squares (with intercept) from the k
coordinates to the token's own predicted arm action (dims 0..5, each
standardized on the training tokens); held-out R^2 averaged over the 6 dims.
k in (8, 32, 128) and a ridge fit on all 1024 dims (lambda = 1e-3 * trace/1024).
3-D view (step 9): PC1-PC3 of X', 400 tokens per task, arm-speed per token.

usage: python layer_arm_only.py IN_NPZ OUT_JSON
"""
import json
import sys

import numpy as np
import torch

D = np.load(sys.argv[1], mmap_mode="r")
DEV = "cuda"
STEPS = [int(s) for s in D["steps"]]
NAMES = [str(n) for n in D["layer_names"]]
META = np.asarray(D["meta"])
TASKS = [int(t) for t in D["task_ids"]]
KS = [8, 32, 128]
ACT = np.asarray(D["actions"])  # (N, 64, 7)
tok_task = np.repeat(META[:, 0], 64)
tok_ep = np.repeat(META[:, 1], 64)
grip = ACT[..., 6].reshape(-1)
arm = ACT[..., :6].reshape(-1, 6)
open_ = grip > 0.5
rng = np.random.default_rng(0)


def erank_stats(x: torch.Tensor) -> dict:
    xc = x - x.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(xc.T @ xc).flip(0).clamp_min(0)
    s = ev.sqrt(); p = s / s.sum(); p = p[p > 0]
    cum = torch.cumsum(ev, 0) / ev.sum()
    return {"erank": float(torch.exp(-(p * p.log()).sum())), "k90": int((cum < 0.90).sum()) + 1,
            "k99": int((cum < 0.99).sum()) + 1, "top1": float(ev[0] / ev.sum())}


class Ellipsoid:
    def __init__(self, x: torch.Tensor, k: int):
        self.mu = x.mean(0, keepdim=True)
        xc = x - self.mu
        ev, U = torch.linalg.eigh(xc.T @ xc / (x.shape[0] - 1))
        self.lam, self.U = ev.flip(0)[:k].clamp_min(1e-12), U.flip(1)[:, :k]
        self.r = float(torch.quantile(self.radius(x), 0.95))

    def radius(self, x: torch.Tensor) -> torch.Tensor:
        c = (x - self.mu) @ self.U
        return ((c ** 2) / self.lam).sum(1).sqrt()

    def score(self, x: torch.Tensor) -> tuple[float, float]:
        xc = x - self.mu
        c = xc @ self.U
        return float((xc.norm() ** 2 - c.norm() ** 2) / xc.norm() ** 2), float((self.radius(x) <= self.r).float().mean())


def r2_heldout(Xtr, Ytr, Xte, Yte, k):
    mu = Xtr.mean(0, keepdim=True)
    if k is None:
        A = Xtr - mu
        lam = 1e-3 * float((A * A).sum()) / A.shape[0] / A.shape[1]
        Bte = Xte - mu
    else:
        A0 = Xtr - mu
        U = torch.linalg.eigh(A0.T @ A0)[1].flip(1)[:, :k]
        A, Bte, lam = A0 @ U, (Xte - mu) @ U, 0.0
    ym, ys = Ytr.mean(0, keepdim=True), Ytr.std(0, keepdim=True)
    Yn = (Ytr - ym) / ys
    G = A.T @ A + lam * A.shape[0] * torch.eye(A.shape[1], device=DEV, dtype=A.dtype)
    W = torch.linalg.solve(G, A.T @ Yn)
    pred = Bte @ W
    yt = (Yte - ym) / ys
    ss_res = ((yt - pred) ** 2).sum(0)
    ss_tot = ((yt - yt.mean(0, keepdim=True)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).cpu().numpy()


armT = torch.from_numpy(arm.astype(np.float64)).to(DEV)
gripT = torch.from_numpy(grip.astype(np.float64)).to(DEV)
eps_all = sorted(set(tok_ep.tolist()))
out = {"steps": STEPS, "layers": NAMES, "task_ids": [str(t) for t in TASKS], "ks": KS, "per": {}}
for s in STEPS:
    T = D[f"tokens_step{s}"]
    for li, name in enumerate(NAMES):
        X = torch.from_numpy(np.asarray(T[:, li]).reshape(-1, 1024).astype(np.float32)).to(DEV, torch.float64)
        om = torch.from_numpy(open_).to(DEV)
        u = X[om].mean(0) - X[~om].mean(0)
        u = u / u.norm()
        Xp = X - (X @ u)[:, None] * u[None, :]
        rec = {}
        # checks
        xc = Xp - Xp.mean(0, keepdim=True)
        ev, U = torch.linalg.eigh(xc.T @ xc); ev, U = ev.flip(0), U.flip(1)
        P5 = xc @ U[:, :5]
        g = gripT - gripT.mean()
        rec["grip_corr_pc1_5"] = [round(abs(float((P5[:, j] - P5[:, j].mean()) @ g / ((P5[:, j] - P5[:, j].mean()).norm() * g.norm()))), 3)
                                  for j in range(5)]
        p1 = P5[:, 0].cpu().numpy()
        wsd = np.sqrt(open_.mean() * p1[open_].var() + (~open_).mean() * p1[~open_].var())
        rec["pc1_gap_over_within_sd"] = round(float(abs(p1[open_].mean() - p1[~open_].mean()) / wsd), 2)
        rec["removed_var_share"] = round(float(((X - X.mean(0)) @ u).pow(2).sum() / (X - X.mean(0)).pow(2).sum()), 4)
        # erank
        rec["task"] = {str(t): erank_stats(Xp[torch.from_numpy(tok_task == t).to(DEV)]) for t in TASKS}
        rec["pooled"] = erank_stats(Xp)
        n_task = min(int((META[:, 0] == t).sum()) for t in TASKS)
        sub_samples = np.concatenate([rng.choice(np.where(META[:, 0] == t)[0], n_task // 3, replace=False) for t in TASKS])
        sub_tok = (sub_samples[:, None] * 64 + np.arange(64)[None, :]).reshape(-1)
        rec["pooled_sub"] = erank_stats(Xp[torch.from_numpy(sub_tok).to(DEV)])
        # ellipsoids
        rec["ellipsoid"] = {}
        for k in KS:
            own, pooled, other = [], [], []
            for t in TASKS:
                tm = tok_task == t
                for e in sorted(set(tok_ep[tm].tolist())):
                    te = torch.from_numpy(tm & (tok_ep == e)).to(DEV)
                    tr = torch.from_numpy(tm & (tok_ep != e)).to(DEV)
                    trp = torch.from_numpy(tok_ep != e).to(DEV)
                    own.append(Ellipsoid(Xp[tr], k).score(Xp[te]))
                    pooled.append(Ellipsoid(Xp[trp], k).score(Xp[te]))
                et = Ellipsoid(Xp[torch.from_numpy(tm).to(DEV)], k)
                other += [et.score(Xp[torch.from_numpy(tok_task == v).to(DEV)]) for v in TASKS if v != t]
            rec["ellipsoid"][str(k)] = {"own": np.mean(own, 0).round(4).tolist(), "pooled": np.mean(pooled, 0).round(4).tolist(),
                                        "other": np.mean(other, 0).round(4).tolist()}
        # arm decodability
        r2 = {str(k): [] for k in KS}; r2["full"] = []
        for e in eps_all:
            te = torch.from_numpy(tok_ep == e).to(DEV); tr = ~te
            for k in KS:
                r2[str(k)].append(r2_heldout(Xp[tr], armT[tr], Xp[te], armT[te], k))
            r2["full"].append(r2_heldout(Xp[tr], armT[tr], Xp[te], armT[te], None))
        rec["arm_r2"] = {k: [round(float(x), 4) for x in np.mean(v, 0)] for k, v in r2.items()}
        rec["arm_r2_mean"] = {k: round(float(np.mean(v)), 4) for k, v in r2.items()}
        if s == STEPS[-1]:
            P3 = (xc @ U[:, :3]).cpu().numpy()
            speed = np.linalg.norm(arm[:, :3], axis=1)
            v3 = {"share": [round(float(x), 4) for x in (ev[:3] / ev.sum()).cpu().numpy()], "pts": {}, "grip": {}, "speed": {}}
            for t in TASKS:
                idx = rng.choice(np.where(tok_task == t)[0], 400, replace=False)
                v3["pts"][str(t)] = np.round(P3[idx], 2).tolist()
                v3["grip"][str(t)] = open_[idx].astype(int).tolist()
                v3["speed"][str(t)] = np.round(speed[idx], 3).tolist()
            rec["view3d"] = v3
        out["per"][f"{s}/{name}"] = rec
        e32 = rec["ellipsoid"]["32"]
        print(f"step {s} {name:9s} removed {rec['removed_var_share']:.3f} gripcorr {rec['grip_corr_pc1_5']} gap {rec['pc1_gap_over_within_sd']} | "
              f"erank task {[round(rec['task'][str(t)]['erank'], 1) for t in TASKS]} pooled {rec['pooled']['erank']:.1f} "
              f"sub {rec['pooled_sub']['erank']:.1f} k99 {rec['pooled']['k99']} | k32 own {e32['own']} pooled {e32['pooled']} "
              f"other {e32['other']} | armR2 {rec['arm_r2_mean']}", flush=True)
        del X, Xp, xc
        torch.cuda.empty_cache()
json.dump(out, open(sys.argv[2], "w"))
print("saved", sys.argv[2])
