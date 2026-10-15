#!/usr/bin/env python
"""Local linear structure of the ImageWAM ActionDiT velocity field and Newton-type Picard sampling.

Velocity v_j(x): one served denoise step at step j from action latent x
((64, 7) f32, flattened to 448), read from `bufs["velocity"]` (served
kernels, step j's own AdaLN modulation and delta). Samples: LIBERO-10 dataset
frames; a J-train set (EPS_TRAIN episodes per task) and a disjoint test set
(EPS_TEST episodes per task), frame stride STRIDE.

Part 1 (initialization spread): per test sample, N_INIT initial latents
  sigma * N(0, 1) for sigma in SIGMAS; relative spread of v_0 across inits
  (||v_i - mean|| / ||mean||, mean over i) and MAE (xyz + rpy) of the serial
  10-step actions between inits.
Part 2 (Jacobian): J_j = dv_j/dx at the serial trajectory point x_j (served
  noise), central differences along the 448 unit directions, step EPS; for
  J-train samples at every step j. Reported: singular-value effective rank
  (exp entropy of normalized singular values), k90 / k99 of sum sigma^2,
  c_j = trace(J_j) / 448, ||J_j - c_j I||_F / ||J_j||_F, effective rank of
  J_j - c_j I, estimation noise ||J(EPS) - J(EPS2)||_F / ||J(EPS)||_F, and
  linear-prediction error ||v(x+d) - v(x) - J d|| / ||v(x+d) - v(x)|| for
  random d of RMS sizes DELTAS (also for d = difference of two N(0,1)
  initial latents at step 0).
Part 3 (Newton-Picard, served noise): full-window iteration
  Delta_0 = 0, Delta_{j+1} = (I + d_j A_j) Delta_j + r_j,
  r_j = x_j + d_j v_j(x_j) - x_{j+1},  X <- X + Delta,
  with A_j = 0 (plain Picard), c_j I (scalar, mean over J-train), mean J-train
  Jacobian, or the sample's own exact J_j at its serial trajectory (upper
  bound; first N_ORACLE test samples). Per sweep s = 1..10: MAE of the final
  actions vs serial. Every variant evaluates the 10 velocities once per sweep.

usage: python benchmarks/imagewam_velocity_jacobian_study.py PRECISION OUT_JSON
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

from flash_rt import flash_rt_kernels as fvk  # noqa: E402
import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
import flash_rt.models.imagewam.pipeline_thor as P  # noqa: E402
from flash_rt.models.imagewam.pipeline_thor import _wrap_fp16  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

PRECISION, OUT = sys.argv[1], sys.argv[2]
SUITE, SEED = "libero_10", 1
EPS_TRAIN, EPS_TEST, STRIDE = 1, 2, 60
SIGMAS, N_INIT = [0.01, 1.0], 8
EPS, EPS2 = 0.03, 0.1
DELTAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_ORACLE = 20
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROOT = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")
DEV = "cuda"


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
split = {}
for ti, eps in sorted(by_task.items()):
    pick = rng.choice(eps, EPS_TRAIN + EPS_TEST, replace=False).tolist()
    split[ti] = (sorted(pick[:EPS_TRAIN]), sorted(pick[EPS_TRAIN:]))

fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
NA, AD = fe.dims["num_action"], fe.dims["action_dim"]
DIM = NA * AD
NSTEP = fe.dims["num_denoise_steps"]
vel_buf = _wrap_fp16(fe._bufs["velocity"], NA, AD)
deltas = [float(x) for x in fe._deltas] if fe._deltas is not None else [float(fe.dims["dt"])] * NSTEP


def velocity(x: torch.Tensor, j: int) -> torch.Tensor:
    fe._action_latent.copy_(x.view(NA, AD))
    mod_double, mod_single = fe._action_mods[j]
    P.imagewam_denoise_step(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe._active_dims, j,
                            stream=torch.cuda.current_stream().cuda_stream, attn=fe._attn,
                            mod_double=mod_double, mod_single=mod_single, head_mod=fe._head_mods[j],
                            action_rope_table=fe._action_rope_table.data_ptr(), delta=deltas[j])
    return vel_buf.float().reshape(-1).clone()


def actions_of(x: torch.Tensor) -> np.ndarray:
    fe._action_latent.copy_(x.view(NA, AD))
    return np.asarray(fe.read_actions()).copy()


def serial(x0: torch.Tensor) -> list[torch.Tensor]:
    xs = [x0.clone()]
    for j in range(NSTEP):
        xs.append(xs[-1] + deltas[j] * velocity(xs[-1], j))
    return xs


def jacobian(x: torch.Tensor, j: int, eps: float) -> torch.Tensor:
    J = torch.empty(DIM, DIM, device=DEV)
    for i in range(DIM):
        e = torch.zeros(DIM, device=DEV)
        e[i] = eps
        J[:, i] = (velocity(x + e, j) - velocity(x - e, j)) / (2 * eps)
    return J


def spec(M: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(M.double())
    p = s / s.sum()
    p = p[p > 0]
    cum = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
    return {"erank": float(torch.exp(-(p * p.log()).sum())), "k90": int((cum < 0.9).sum()) + 1,
            "k99": int((cum < 0.99).sum()) + 1, "sigma_max": float(s[0])}


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a[..., :6] - b[..., :6]).mean())


def newton_picard(x0: torch.Tensor, A: list | None, ref: np.ndarray) -> list[float]:
    X = [x0.clone() for _ in range(NSTEP + 1)]
    I = torch.eye(DIM, device=DEV)
    out = []
    for _ in range(NSTEP):
        vs = [velocity(X[j], j) for j in range(NSTEP)]
        D = [torch.zeros(DIM, device=DEV)]
        for j in range(NSTEP):
            r = X[j] + deltas[j] * vs[j] - X[j + 1]
            if A is None:
                D.append(D[-1] + r)
            else:
                D.append((I + deltas[j] * A[j]) @ D[-1] + r)
        X = [X[j] + D[j] for j in range(NSTEP + 1)]
        out.append(mae(actions_of(X[NSTEP]), ref))
    return out


def frames(eps: list[int]) -> list[tuple]:
    out = []
    for e in eps:
        df = pd.read_parquet(_episode_parquet(ROOT, e))
        v1 = decode(_episode_video(ROOT, "observation.images.image", e))
        v2 = decode(_episode_video(ROOT, "observation.images.wrist_image", e))
        for fr in range(0, len(df), STRIDE):
            out.append((e, fr, {"view1": torch.from_numpy(center_crop_resize(v1[fr], 224, 224)),
                                "view2": torch.from_numpy(center_crop_resize(v2[fr], 224, 224)),
                                "proprio": np.array(df["observation.state"].iloc[fr], dtype=np.float32)}))
    return out


t0 = time.time()
gen = torch.Generator(device=DEV).manual_seed(SEED)
noise = lambda sigma: sigma * torch.randn(DIM, generator=gen, device=DEV)

# ---------------- Part 2a: J-train ----------------
Jtrain = [[] for _ in range(NSTEP)]
part2 = []
for ti in sorted(split):
    fe.set_prompt(DEFAULT_PROMPT.format(task=tasks[ti]))
    for e, fr, obs in frames(split[ti][0])[:2]:
        x0 = noise(0.01)
        fe.stage_inputs(obs, noise=x0.view(NA, AD))
        fe.run_eager()
        xs = serial(x0)
        rec = {"task": ti, "episode": e, "frame": fr, "steps": []}
        for j in range(NSTEP):
            J = jacobian(xs[j], j, EPS)
            Jtrain[j].append(J)
            c = float(torch.trace(J)) / DIM
            st = {"j": j, "J": spec(J), "c": c, "offscalar_rel": float((J - c * torch.eye(DIM, device=DEV)).norm() / J.norm()),
                  "J_minus_cI": spec(J - c * torch.eye(DIM, device=DEV))}
            if j in (0, 4, 9):
                J2 = jacobian(xs[j], j, EPS2)
                st["eps_noise_rel"] = float((J - J2).norm() / J.norm())
                v0 = velocity(xs[j], j)
                lin = {}
                for dsz in DELTAS:
                    errs = []
                    for _ in range(4):
                        d = torch.randn(DIM, generator=gen, device=DEV) * dsz
                        dv = velocity(xs[j] + d, j) - v0
                        errs.append(float((dv - J @ d).norm() / dv.norm()))
                    lin[str(dsz)] = float(np.mean(errs))
                if j == 0:
                    errs = []
                    for _ in range(4):
                        d = noise(1.0) - noise(1.0)
                        dv = velocity(xs[0] + d, 0) - v0
                        errs.append(float((dv - J @ d).norm() / dv.norm()))
                    lin["init_difference_sigma1"] = float(np.mean(errs))
                st["linear_pred_rel_err"] = lin
            rec["steps"].append(st)
        part2.append(rec)
        print(f"J-train task {ti} ep {e} fr {fr}: erank {[round(s['J']['erank'], 1) for s in rec['steps']]} "
              f"c {[round(s['c'], 2) for s in rec['steps']]} offscalar {[round(s['offscalar_rel'], 2) for s in rec['steps']]} "
              f"{time.time() - t0:.0f}s", flush=True)
c_mean = [float(np.mean([float(torch.trace(J)) / DIM for J in Jtrain[j]])) for j in range(NSTEP)]
J_mean = [torch.stack(Jtrain[j]).mean(0) for j in range(NSTEP)]
Jmean_spread = [float(np.mean([float((J - J_mean[j]).norm() / J.norm()) for J in Jtrain[j]])) for j in range(NSTEP)]
del Jtrain
torch.cuda.empty_cache()

# ---------------- Part 1 + Part 3: test ----------------
part1, part3 = [], []
n_test = 0
for ti in sorted(split):
    fe.set_prompt(DEFAULT_PROMPT.format(task=tasks[ti]))
    for e, fr, obs in frames(split[ti][1]):
        fe.stage_inputs(obs, noise=torch.zeros(NA, AD, device=DEV))
        fe.run_eager()
        p1 = {"task": ti, "episode": e, "frame": fr}
        for sigma in SIGMAS:
            inits = [noise(sigma) for _ in range(N_INIT)]
            v0 = torch.stack([velocity(x, 0) for x in inits])
            vm = v0.mean(0)
            acts = [actions_of(serial(x)[-1]) for x in inits]
            pair = [mae(acts[a], acts[b]) for a in range(N_INIT) for b in range(a + 1, N_INIT)]
            p1[str(sigma)] = {"v0_rel_spread": float(((v0 - vm).norm(dim=1) / vm.norm()).mean()),
                              "action_pair_mae": float(np.mean(pair))}
        part1.append(p1)
        x0 = noise(0.01)
        xs = serial(x0)
        ref = actions_of(xs[-1])
        p3 = {"task": ti, "episode": e, "frame": fr,
              "picard": newton_picard(x0, None, ref),
              "scalar": newton_picard(x0, [c_mean[j] * torch.eye(DIM, device=DEV) for j in range(NSTEP)], ref),
              "mean_J": newton_picard(x0, J_mean, ref)}
        if n_test < N_ORACLE:
            Jown = [jacobian(xs[j], j, EPS) for j in range(NSTEP)]
            p3["own_J"] = newton_picard(x0, Jown, ref)
            p3["mean_J_rel_err_vs_own"] = [float((J_mean[j] - Jown[j]).norm() / Jown[j].norm()) for j in range(NSTEP)]
            del Jown
        part3.append(p3)
        n_test += 1
        print(f"test {n_test} task {ti} ep {e} fr {fr}: v0 spread {p1['0.01']['v0_rel_spread']:.4f}/{p1['1.0']['v0_rel_spread']:.3f} "
              f"action pair MAE {p1['0.01']['action_pair_mae']:.5f}/{p1['1.0']['action_pair_mae']:.4f} | "
              f"sweep3 picard {p3['picard'][2]:.4f} scalar {p3['scalar'][2]:.4f} meanJ {p3['mean_J'][2]:.4f} "
              f"own {p3.get('own_J', [0, 0, -1])[2]:.4f} {time.time() - t0:.0f}s", flush=True)


def m(xs: list) -> float:
    return round(float(np.mean(xs)), 6)


summary = {
    "precision": PRECISION, "n_jtrain": len(part2), "n_test": len(part3), "eps": EPS, "deltas_schedule": deltas,
    "c_mean": [round(c, 4) for c in c_mean], "J_train_rel_spread_from_mean": [round(x, 4) for x in Jmean_spread],
    "J": {str(j): {"erank": m([r["steps"][j]["J"]["erank"] for r in part2]),
                   "k90": m([r["steps"][j]["J"]["k90"] for r in part2]),
                   "k99": m([r["steps"][j]["J"]["k99"] for r in part2]),
                   "offscalar_rel": m([r["steps"][j]["offscalar_rel"] for r in part2]),
                   "J_minus_cI_erank": m([r["steps"][j]["J_minus_cI"]["erank"] for r in part2]),
                   "J_minus_cI_k99": m([r["steps"][j]["J_minus_cI"]["k99"] for r in part2])} for j in range(NSTEP)},
    "eps_noise_rel": {str(j): m([r["steps"][j]["eps_noise_rel"] for r in part2]) for j in (0, 4, 9)},
    "linear_pred_rel_err": {str(j): {k: m([r["steps"][j]["linear_pred_rel_err"][k] for r in part2])
                                     for k in part2[0]["steps"][j]["linear_pred_rel_err"]} for j in (0, 4, 9)},
    "init_spread": {str(s): {"v0_rel_spread": m([p[str(s)]["v0_rel_spread"] for p in part1]),
                             "action_pair_mae": m([p[str(s)]["action_pair_mae"] for p in part1])} for s in SIGMAS},
    "newton_picard_mae_per_sweep": {v: [m([p[v][s] for p in part3 if v in p]) for s in range(NSTEP)]
                                    for v in ("picard", "scalar", "mean_J", "own_J")},
    "mean_J_rel_err_vs_own": [m([p["mean_J_rel_err_vs_own"][j] for p in part3 if "mean_J_rel_err_vs_own" in p])
                              for j in range(NSTEP)],
}
json.dump({"summary": summary, "part1": part1, "part2": part2, "part3": part3}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1))
