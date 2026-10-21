"""Effective rank of ImageWAM ActionDiT tokens and their attention Q/K/V under one
fixed instruction, across the visual conditions of the training demos.

Per task (LIBERO instruction = one fixed text prompt), EPS random training
episodes, every STRIDE-th frame; each frame = one open-loop eager infer() with
served noise 0.01*N(0,1) seeded by sample index. At denoise steps STEPS:
  h      residual stream after each ActionDiT layer (64 x 1024)
  Q,K,V  the 64 action rows of the "mot" attention slot, read right before
         attn.run (post QK-RMSNorm and RoPE, i.e. what attention consumes;
         64 x 3072 = 24 heads x 128)
Rows are tokens (n_frames x 64). Statistics are accumulated on the GPU as
Gram matrices of (x - ref_pos) in fp32, ref = first frame's tokens per
position (limits cancellation), plus per-position sums in fp64:
  "all"      covariance around the global mean (token position + visual)
  "visual"   covariance around each token position's own mean (only what
             changes with the frame: image, proprio, noise)
  per-head   mean over 24 heads of the 128 x 128 diagonal blocks ("visual")
erank = exp(entropy of s/sum s), s = sqrt(eigenvalues) (as erank.py); also
participation ratio and k90 / k99 (eigen-energy counts).

usage: python qkv_rank.py PRECISION SUITE TASKS(comma) EPS STRIDE SEED STEPS(comma) OUT_JSON
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
import flash_rt.models.imagewam.pipeline_thor as P  # noqa: E402
from flash_rt.models.imagewam.pipeline_thor import _ptr_offset, _wrap_fp16  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

PRECISION, SUITE = sys.argv[1], sys.argv[2]
TASKS = [int(t) for t in sys.argv[3].split(",")]
EPS, STRIDE, SEED = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
STEPS = [int(s) for s in sys.argv[7].split(",")]
OUT = sys.argv[8]
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ROOT = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")


def decode(path: str) -> list:
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


class Acc:
    """Gram of (x - ref) over rows, per-position sums, for one (step, layer, kind)."""

    def __init__(self, d: int, npos: int) -> None:
        self.G = torch.zeros(d, d, device="cuda", dtype=torch.float32)
        self.S = torch.zeros(npos, d, device="cuda", dtype=torch.float64)
        self.ref = None
        self.n = 0  # frames

    def add(self, x: torch.Tensor) -> None:  # x: (npos, d) fp16
        x = x.float()
        if self.ref is None:
            self.ref = x.clone()
        y = x - self.ref
        self.G.addmm_(y.T, y)
        self.S += y.double()
        self.n += 1

    def covs(self) -> tuple[torch.Tensor, torch.Tensor]:
        G = self.G.double()
        rows = self.n * self.S.shape[0]
        mu = self.S.sum(0) / rows
        c_all = (G - rows * torch.outer(mu, mu)) / rows
        c_vis = (G - self.S.T @ self.S / self.n) / rows
        return c_all, c_vis


def spec(C: torch.Tensor) -> dict:
    ev = torch.linalg.eigvalsh(0.5 * (C + C.T)).clamp_min(0).flip(0)
    s = ev.sqrt()
    p = s / s.sum()
    p = p[p > 0]
    e = ev / ev.sum()
    cum = torch.cumsum(e, 0)
    return {"erank": round(float(torch.exp(-(p * p.log()).sum())), 1),
            "pr": round(float(ev.sum() ** 2 / (ev ** 2).sum()), 1),
            "k90": int((cum < 0.90).sum()) + 1, "k99": int((cum < 0.99).sum()) + 1,
            "var": float(ev.sum())}


def head_spec(C: torch.Tensor, nh: int, hd: int) -> dict:
    r = [spec(C[h * hd:(h + 1) * hd, h * hd:(h + 1) * hd]) for h in range(nh)]
    return {k: round(float(np.mean([x[k] for x in r])), 1) for k in ("erank", "pr", "k90", "k99")}


tasks = {t["task_index"]: t["task"] for t in _read_jsonl(f"{ROOT}/meta/tasks.jsonl")}
by_task: dict[int, list[int]] = {}
for ep in _read_jsonl(f"{ROOT}/meta/episodes.jsonl"):
    e = int(ep["episode_index"])
    ti = int(pd.read_parquet(_episode_parquet(ROOT, e), columns=["task_index"])["task_index"].iloc[0])
    by_task.setdefault(ti, []).append(e)
rng = np.random.default_rng(SEED)
picks = {ti: sorted(rng.choice(by_task[ti], min(EPS, len(by_task[ti])), replace=False).tolist()) for ti in TASKS}
print("picks", picks, flush=True)

fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
d = fe.dims
na, ahd = d["num_action"], d["action_hidden_dim"]
ND, NS, NSTEP = d["action_num_layers_double"], d["action_num_layers_single"], d["num_denoise_steps"]
L = ND + NS
W, NH, HD = d["action_attn_width"], d["NH"], d["HD"]
names = [f"double{i}" for i in range(ND)] + [f"single{i}" for i in range(NS)]
print(f"ActionDiT {ND}+{NS} layers, {NSTEP} steps, tokens {na}x{ahd}, attn {NH}x{HD}", flush=True)
hid = _wrap_fp16(fe._bufs["action_hidden"], na, ahd)

state = {"layer_calls": 0, "attn_calls": 0, "acc": None}
orig_double, orig_single = P._action_double_layer, P._action_single_layer
orig_run = fe._attn.run


def wrapped_layer(orig):
    def f(*a, **k):
        orig(*a, **k)
        c = state["layer_calls"]
        step, li = divmod(c, L)
        if step in STEPS:
            torch.cuda.current_stream().synchronize()
            state["acc"][(step, li, "h")].add(hid)
        state["layer_calls"] += 1
    return f


def wrapped_run(site, layer_idx, *a, **k):
    if site == "mot":
        c = state["attn_calls"]
        step, li = divmod(c, L)
        if step in STEPS:
            torch.cuda.current_stream().synchronize()
            ptrs = fe._attn.get_slot_ptrs("mot", layer_idx)
            for kind in ("Q", "K", "V"):
                # live action-row offset: set_prompt trims the text, so the static fe.dims["a0"] is stale
                state["acc"][(step, li, kind)].add(_wrap_fp16(_ptr_offset(ptrs[kind], k["a0"], W), na, W))
        state["attn_calls"] += 1
    return orig_run(site, layer_idx, *a, **k)


out = {"precision": PRECISION, "suite": SUITE, "picks": {str(k): v for k, v in picks.items()}, "stride": STRIDE,
       "steps": STEPS, "layers": names, "tasks": {}}
t0 = time.time()
for ti, eps in picks.items():
    fe.set_prompt(DEFAULT_PROMPT.format(task=tasks[ti]))
    state["acc"] = {(s, li, kind): Acc(ahd if kind == "h" else W, na)
                    for s in STEPS for li in range(L) for kind in ("h", "Q", "K", "V")}
    nfr = 0
    for e in eps:
        df = pd.read_parquet(_episode_parquet(ROOT, e))
        v1 = decode(_episode_video(ROOT, "observation.images.image", e))
        v2 = decode(_episode_video(ROOT, "observation.images.wrist_image", e))
        for fr in range(0, len(df), STRIDE):
            g = torch.Generator(device="cuda").manual_seed(nfr)
            noise = 0.01 * torch.randn(na, d["action_dim"], generator=g, device="cuda")
            obs = {"view1": torch.from_numpy(center_crop_resize(v1[fr], 224, 224)),
                   "view2": torch.from_numpy(center_crop_resize(v2[fr], 224, 224)),
                   "proprio": np.array(df["observation.state"].iloc[fr], dtype=np.float32)}
            fe.stage_inputs(obs, noise=noise)
            state["layer_calls"] = state["attn_calls"] = 0
            P._action_double_layer, P._action_single_layer = wrapped_layer(orig_double), wrapped_layer(orig_single)
            fe._attn.run = wrapped_run
            try:
                fe.run_eager()
            finally:
                P._action_double_layer, P._action_single_layer = orig_double, orig_single
                fe._attn.run = orig_run
            torch.cuda.synchronize()
            if state["layer_calls"] != L * NSTEP or state["attn_calls"] != L * NSTEP:
                raise RuntimeError(f"calls layer={state['layer_calls']} attn={state['attn_calls']} != {L * NSTEP}")
            nfr += 1
        print(f"task {ti} ep {e}: frames {nfr} {time.time() - t0:.0f}s", flush=True)
    res = {"instruction": tasks[ti], "n_frames": nfr, "n_tokens": nfr * na, "steps": {}}
    for s in STEPS:
        rs = {}
        for li, nm in enumerate(names):
            rl = {}
            for kind in ("h", "Q", "K", "V"):
                acc = state["acc"][(s, li, kind)]
                if not torch.isfinite(acc.G).all():
                    raise RuntimeError(f"non-finite Gram {s} {nm} {kind}")
                c_all, c_vis = acc.covs()
                rl[kind] = {"all": spec(c_all), "visual": spec(c_vis)}
                if kind != "h":
                    rl[kind]["visual_per_head"] = head_spec(c_vis, NH, HD)
            rs[nm] = rl
        res["steps"][str(s)] = rs
    out["tasks"][str(ti)] = res
    del state["acc"]
    torch.cuda.empty_cache()
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"task {ti} done: {nfr} frames, {time.time() - t0:.0f}s", flush=True)
print(f"saved {OUT}")
