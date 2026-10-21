"""Robustness of the per-layer token erank: what the dominant direction is.
Per (step, layer), pooled tokens: erank raw; erank after removing PC1;
erank of per-channel standardized tokens (correlation matrix); share of PC1's
loading energy on its top-8 channels; channel variance concentration (top-8
channels' share of total variance). usage: python layer_robust.py IN_NPZ OUT_JSON"""
import json, sys
import numpy as np, torch
D = np.load(sys.argv[1]); out = {}
def erank_ev(ev):
    s = ev.clamp_min(0).sqrt(); p = s / s.sum(); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))
for s in [int(x) for x in D["steps"]]:
    T = D[f"tokens_step{s}"]
    for li, name in enumerate([str(n) for n in D["layer_names"]]):
        x = torch.from_numpy(T[:, li].reshape(-1, T.shape[-1]).astype(np.float32)).cuda().double()
        xc = x - x.mean(0, keepdim=True)
        C = xc.T @ xc / (x.shape[0] - 1)
        ev, U = torch.linalg.eigh(C); ev, U = ev.flip(0), U.flip(1)
        var = C.diagonal()
        sd = var.clamp_min(1e-12).sqrt()
        evz = torch.linalg.eigvalsh(C / sd[:, None] / sd[None, :]).flip(0)
        u1 = U[:, 0] ** 2
        r = {"erank": erank_ev(ev), "erank_minus_pc1": erank_ev(ev[1:]), "erank_standardized": erank_ev(evz),
             "pc1_top8_channel_share": float(u1.sort(descending=True).values[:8].sum()),
             "top8_channel_var_share": float(var.sort(descending=True).values[:8].sum() / var.sum()),
             "pc1_share": float(ev[0] / ev.sum())}
        out[f"{s}/{name}"] = r
        if name in ("encoder", "double0", "double4", "single0", "single9", "single19"):
            print(s, name, {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open(sys.argv[2], "w"))
