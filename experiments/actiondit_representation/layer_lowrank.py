"""Per-layer low-rank structure of ImageWAM ActionDiT action tokens + covariance ellipsoids.

Input: collect_layers.py npz (tokens_step{s}: (N, L+1, 64, 1024) fp16).
Unit = one action token (a 1024-d residual-stream vector); a sample gives 64.

Per (step, layer):
  erank / k90 / k99 / top-1 share of the centered token cloud, for each task
  alone, for the 3 tasks pooled, and for the pooled set subsampled to one
  task's token count (same-N comparison). erank = exp(H(p)), p = sigma/sum(sigma)
  (singular values of the centered matrix, Roy & Vetterli); kXX = smallest k
  whose eigenvalues (sigma^2) reach XX% of the variance.
  pos_share: fraction of the variance explained by the 64 per-position means.
  Sample-level erank (each sample's 64x1024 flattened) per task / pooled-
  subsampled, for reference (N-limited: <= 75 per task).

Ellipsoid E_k(mu, U_k, Lambda_k, r) = {mu + U_k z : z^T Lambda_k^-1 z <= r^2},
r = 95th percentile of the training tokens' in-subspace Mahalanobis radius.
Held out by episode (5-fold leave-one-episode-out inside each task):
  resid  = sum ||(I - U_k U_k^T)(x - mu)||^2 / sum ||x - mu||^2 over held-out
           tokens (variance outside the k-dim subspace);
  cover  = fraction of held-out tokens with Mahalanobis radius <= r (0.95 if
           the ellipsoid generalizes);
  for ellipsoids fit on: own task (4 train episodes), pooled 3 tasks (train
  episodes of all 3), other task (all episodes of each other task, averaged).
2-D view: pooled top-2 PCs of per-sample mean tokens, per-task 2-sigma
ellipse; plus 600 random tokens per task.

usage: python layer_lowrank.py IN_NPZ OUT_JSON
"""
import json
import sys

import numpy as np
import torch

D = np.load(sys.argv[1])
DEV = "cuda"
STEPS = [int(s) for s in D["steps"]]
NAMES = [str(n) for n in D["layer_names"]]
META = D["meta"]
TASKS = [int(t) for t in D["task_ids"]]
KS = [8, 32, 128]
rng = np.random.default_rng(0)
task_of, ep_of = META[:, 0], META[:, 1]


def eig_stats(x: torch.Tensor) -> dict:
    xc = x - x.mean(0, keepdim=True)
    if xc.shape[0] > xc.shape[1]:
        ev = torch.linalg.eigvalsh(xc.T @ xc).flip(0).clamp_min(0)
    else:
        ev = torch.linalg.eigvalsh(xc @ xc.T).flip(0).clamp_min(0)
    s = ev.sqrt()
    p = s / s.sum()
    p = p[p > 0]
    cum = torch.cumsum(ev, 0) / ev.sum()
    return {"erank": float(torch.exp(-(p * p.log()).sum())),
            "pr": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "k90": int((cum < 0.90).sum()) + 1, "k99": int((cum < 0.99).sum()) + 1,
            "top1": float(ev[0] / ev.sum()), "spectrum": (ev / ev.sum())[:256].cpu().numpy()}


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
        resid = float((xc.norm() ** 2 - c.norm() ** 2) / xc.norm() ** 2)
        cover = float((self.radius(x) <= self.r).float().mean())
        return resid, cover


def tokens(arr: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr[idx].astype(np.float32)).to(DEV, torch.float64).reshape(-1, arr.shape[-1])


out = {"steps": STEPS, "layers": NAMES, "tasks": [str(t) for t in D["tasks"]], "task_ids": TASKS, "ks": KS,
       "n_samples": {str(t): int((task_of == t).sum()) for t in TASKS}, "per": {}}
for s in STEPS:
    T = D[f"tokens_step{s}"]  # (N, L+1, 64, 1024)
    for li, name in enumerate(NAMES):
        A = T[:, li]  # (N, 64, 1024)
        rec = {"task": {}, "pooled": None, "pooled_sub": None, "sample": {}, "ellipsoid": {}}
        n_task = min((task_of == t).sum() for t in TASKS)
        X = {t: tokens(A, np.where(task_of == t)[0]) for t in TASKS}
        Xall = torch.cat([X[t] for t in TASKS])
        for t in TASKS:
            rec["task"][str(t)] = {k: v for k, v in eig_stats(X[t]).items() if k != "spectrum"}
        st = eig_stats(Xall)
        rec["pooled"] = {k: v for k, v in st.items() if k != "spectrum"}
        rec["spectrum_pooled"] = st["spectrum"][:128].round(6).tolist()
        rec["spectrum_task"] = {str(t): eig_stats(X[t])["spectrum"][:128].round(6).tolist() for t in TASKS}
        sub = np.concatenate([rng.choice(np.where(task_of == t)[0], n_task // len(TASKS), replace=False) for t in TASKS])
        rec["pooled_sub"] = {k: v for k, v in eig_stats(tokens(A, sub)).items() if k != "spectrum"}
        # position-mean share
        xa = torch.from_numpy(A.astype(np.float32)).to(DEV, torch.float64)
        tot = ((xa - xa.mean((0, 1), keepdim=True)) ** 2).sum()
        pm = xa.mean(0, keepdim=True)
        rec["pos_share"] = float(((pm - xa.mean((0, 1), keepdim=True)) ** 2).sum() * xa.shape[0] / tot)
        del xa
        # sample-level (flattened) erank
        for t in TASKS:
            idx = np.where(task_of == t)[0]
            rec["sample"][str(t)] = eig_stats(torch.from_numpy(A[idx].reshape(len(idx), -1).astype(np.float32))
                                              .to(DEV, torch.float64))["erank"]
        rec["sample"]["pooled_sub"] = eig_stats(torch.from_numpy(A[sub].reshape(len(sub), -1).astype(np.float32))
                                                .to(DEV, torch.float64))["erank"]
        # ellipsoids, leave-one-episode-out
        for k in KS:
            own, pooled, other = [], [], []
            for t in TASKS:
                eps = sorted(set(ep_of[task_of == t]))
                for e in eps:
                    te = np.where((task_of == t) & (ep_of == e))[0]
                    tr = np.where((task_of == t) & (ep_of != e))[0]
                    trp = np.where(~np.isin(ep_of, [e]) & np.isin(task_of, TASKS)
                                   & ~((task_of == t) & (ep_of == e)))[0]
                    xt = tokens(A, te)
                    own.append(Ellipsoid(tokens(A, tr), k).score(xt))
                    pooled.append(Ellipsoid(tokens(A, trp), k).score(xt))
                ell_t = Ellipsoid(X[t], k)
                other += [ell_t.score(X[u]) for u in TASKS if u != t]
            rec["ellipsoid"][str(k)] = {
                "own": np.mean(own, 0).round(4).tolist(), "pooled": np.mean(pooled, 0).round(4).tolist(),
                "other": np.mean(other, 0).round(4).tolist()}
        # 2-D view (step 9 only, for the figure)
        if s == STEPS[-1]:
            ms = torch.from_numpy(A.mean(1).astype(np.float32)).to(DEV, torch.float64)
            mc = ms - ms.mean(0, keepdim=True)
            _, _, Vh = torch.linalg.svd(mc, full_matrices=False)
            P2 = Vh[:2].T
            pts = (mc @ P2).cpu().numpy()
            ell = {}
            for t in TASKS:
                q = pts[task_of == t]
                ell[str(t)] = {"mean": q.mean(0).round(4).tolist(), "cov": np.cov(q.T).round(6).tolist()}
            tok2 = {}
            Pt = None
            mu_all = Xall.mean(0, keepdim=True)
            _, _, Vt = torch.linalg.svd(Xall[torch.randperm(Xall.shape[0], device=DEV)[:8000]] - mu_all,
                                        full_matrices=False)
            Pt = Vt[:2].T
            for t in TASKS:
                sel = torch.randperm(X[t].shape[0], device=DEV)[:600]
                tok2[str(t)] = ((X[t][sel] - mu_all) @ Pt).cpu().numpy().round(3).tolist()
            rec["view2d"] = {"sample_means": pts.round(4).tolist(), "sample_task": task_of.tolist(),
                             "ellipse": ell, "tokens": tok2}
        out["per"][f"{s}/{name}"] = rec
        e32 = rec["ellipsoid"]["32"]
        print(f"step {s} {name:9s} erank task {[round(rec['task'][str(t)]['erank'], 1) for t in TASKS]} "
              f"pooled {rec['pooled']['erank']:.1f} sub {rec['pooled_sub']['erank']:.1f} "
              f"k90 {rec['pooled']['k90']} top1 {rec['pooled']['top1']:.3f} pos {rec['pos_share']:.3f} | "
              f"k32 own {e32['own']} pooled {e32['pooled']} other {e32['other']}", flush=True)
        torch.cuda.empty_cache()
json.dump(out, open(sys.argv[2], "w"))
print("saved", sys.argv[2])
