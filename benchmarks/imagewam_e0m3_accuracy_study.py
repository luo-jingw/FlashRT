"""Simulated accuracy of 4-bit block-scaled tiers on real ImageWAM (H100).

Question: is E0M3 (uniform INT4) with a per-16 Hadamard rotation more
accurate than the shipped `nvfp4` tier on ImageWAM's real weights and
activations? Thor-only kernels cannot run here, so every tier is
simulated with `flash_rt.models.imagewam.blockscaled_ref`, which
reproduces the repository's quantizers (element grid, UE4M3 per-16
scale rule, rotation, fp16 rounding of the rotated weight). A block
scaled GEMM equals an fp32-accumulated GEMM of the dequantized
operands, and both dequantized operands are exactly representable in
fp16 (checked at run time), so the simulation runs the unchanged fp16
FlashRT GEMM (cuBLASLt, fp32 accumulation) on them.

Two passes over real LIBERO frames (real checkpoint, real VAE, real
Qwen3 context, real proprio, 10-step shift schedule, N(0,1) noise):

1. Per-GEMM (`PROBE_FRAMES` frames): the pipeline runs exactly in fp16;
   every quantizable GEMM call additionally evaluates each tier on the
   same input and weight, in fp32, and accumulates the output error
   relative to the exact fp32 product (also split into weight-only and
   activation-only error). Also records activation block-amax
   statistics (UE4M3 subnormal and zero scales).
2. Whole pipeline (all frames, `PIPE_TIERS`): quantizable weights are
   overwritten in place by their fake-quantized value, activations are
   fake-quantized before each GEMM; `backbone_hidden`, `action_latent`
   (normalized actions) and denormalized actions are compared with the
   fp16 run on the same noise, and actions with ground truth.

Tier names: `nvfp4` (shipped: E2M1 weights and activations, amax/6),
`nvfp4_mse` (MSE weight scales), `e0m3w` (E0M3 weights, E2M1
activations), `e0m3` (E0M3 both), `*_h<N>` (block-diagonal Hadamard of
size N along K on both operands; `h16` is the kernel's rotation),
`*_gs` (per-tensor power-of-two weight pre-scale so block scales use
UE4M3's normal range, undone through the GEMM `alpha`), `*_ags` (the
same for the activation, from each call's own amax).

Env: SUITE (libero_spatial), N_TASKS (10), FRAMES ("0,60"),
PROBE_FRAMES (4), PROBE_TIERS, PIPE_TIERS (comma lists), MERGE_LINEAR2
("1"/"0" forces the merged/split single-stream `linear2`; unset keeps the
frontend default), CONTEXT ("qwen3", or "random" for `set_prompt()`'s
N(0,1) fallback), OUT (json path).
Needs the variables of /home/user1/workspace/jingwu/imagewam_env.sh.
Peak GPU memory is about 22 GB.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.environ["FLUX2_SRC"] + "/src")
sys.path.insert(0, os.environ["FLUX2_SRC"])

from _imagewam_libero_frames import LiberoFrame, load_libero_frames  # noqa: E402

import flash_rt.flash_rt_kernels as fvk  # noqa: E402
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor  # noqa: E402
from flash_rt.models.imagewam.blockscaled_ref import (  # noqa: E402
    BLOCK,
    E0M3_MAX,
    E2M1_MAX,
    UE4M3_MAX,
    dequantize_blocks,
    fake_quantize,
    fwht16_butterfly,
    prepare_e0m3_hadamard_weight,
    quantize_blocks,
    rotate_k_blocks,
)
from flash_rt.models.imagewam.libero_dims import (  # noqa: E402
    LIBERO_HORIZON as HORIZON,
    LIBERO_REAL_DIMS as REAL_DIMS,
    LIBERO_SHIFT as SHIFT,
    LIBERO_STEPS as STEPS,
)
from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_loop, imagewam_prefill  # noqa: E402
from flash_rt.models.imagewam.quant_linear import Fp16Linear  # noqa: E402
from flash_rt.models.imagewam.text_encoder import encode_prompts  # noqa: E402
from flash_rt.models.imagewam.vae_encoder import encode_to_tokens  # noqa: E402

DEV = "cuda"
BF16 = torch.bfloat16
FP16 = torch.float16
CKPT = os.environ["CKPT_PATH"]
STATS = os.path.join(os.path.dirname(CKPT), "dataset_stats.json")
SUITE = os.environ.get("SUITE", "libero_spatial")
N_TASKS = int(os.environ.get("N_TASKS", "10"))
FRAMES = [int(x) for x in os.environ.get("FRAMES", "0,60").split(",")]
PROBE_FRAMES = int(os.environ.get("PROBE_FRAMES", "4"))
# "1"/"0" forces the merged/split single-stream linear2; unset keeps the
# frontend default (merged whenever linear1 is merged).
MERGE_LINEAR2 = os.environ.get("MERGE_LINEAR2")
# "qwen3" (default): real Qwen3 context per task. "random": `set_prompt()`'s
# fallback without a text encoder (context filled with N(0,1), proprio in
# the last row), seeded per task so every tier sees the same values.
CONTEXT = os.environ.get("CONTEXT", "qwen3")
OUT = os.environ.get("OUT", "/home/user1/workspace/jingwu/artifacts/hadamard-int4/accuracy_study.json")


@dataclass(frozen=True)
class Tier:
    """One simulated precision: weight/activation element format, weight
    scale rule, rotation block along K (0 = none), whether the weight gets
    a per-tensor power-of-two pre-scale, and whether the activation gets
    one (computed from each call's own amax: the best case a static,
    calibrated per-GEMM scale could reach)."""

    name: str
    w_fmt: str
    w_rule: str
    a_fmt: str
    rot: int
    w_gs: bool
    a_gs: bool = False


def _tier(name: str) -> Tier:
    base = name.split("_")[0]
    parts = name.split("_")[1:]
    rot = 0
    w_gs = False
    a_gs = False
    w_rule = "amax"
    for p in parts:
        if p.startswith("h") and p[1:].isdigit():
            rot = int(p[1:])
        elif p == "gs":
            w_gs = True
        elif p == "ags":
            a_gs = True
        elif p == "mse":
            w_rule = "mse"
        else:
            raise ValueError(f"unknown tier suffix {p!r} in {name!r}")
    if base == "nvfp4":
        return Tier(name, "e2m1", w_rule, "e2m1", rot, w_gs, a_gs)
    if base == "e0m3w":
        return Tier(name, "e0m3", w_rule, "e2m1", rot, w_gs, a_gs)
    if base == "e0m3":
        return Tier(name, "e0m3", w_rule, "e0m3", rot, w_gs, a_gs)
    raise ValueError(f"unknown tier {name!r}")


PROBE_TIERS = [_tier(t) for t in os.environ.get(
    "PROBE_TIERS",
    "nvfp4,nvfp4_mse,nvfp4_h16,nvfp4_gs,e0m3w,e0m3,e0m3_h16,e0m3_gs,e0m3_h16_gs,e0m3_h64_gs,e0m3_h256_gs"
).split(",")]
PIPE_TIERS = [_tier(t) for t in os.environ.get(
    "PIPE_TIERS", "nvfp4,nvfp4_mse,e0m3w,e0m3,e0m3_h16,nvfp4_gs,e0m3_h16_gs").split(",")]


# ── simulated operands ──────────────────────────────────────────────

def _chunked(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, rows: int = 4096) -> torch.Tensor:
    return torch.cat([fn(x[i:i + rows]) for i in range(0, x.shape[0], rows)], 0)


def _rotate(x: torch.Tensor, block: int) -> torch.Tensor:
    """fp32 rotation; the 16-point case uses the kernels' butterfly."""
    return fwht16_butterfly(x) if block == 16 else rotate_k_blocks(x, block)


def sim_weight(w_kn: torch.Tensor, tier: Tier) -> torch.Tensor:
    """(K,N) fp16 weight -> dequantized (N,K) float32 in the tier's
    (possibly rotated) domain, divided back by the pre-scale.

    The served `e0m3_hadamard` tier (E0M3, H16, weight pre-scale, amax
    rule) uses `prepare_e0m3_hadamard_weight` itself. Other tiers follow
    the same order: rotate in fp32, pre-scale by a power of two, round to
    fp16, quantize."""
    w_nk = w_kn.t().contiguous()
    if tier.w_fmt == "e0m3" and tier.rot == 16 and tier.w_gs and tier.w_rule == "amax":
        w_in, alpha = prepare_e0m3_hadamard_weight(w_nk)
        return _chunked(lambda w: dequantize_blocks(quantize_blocks(w, "e0m3")) * alpha, w_in)
    qmax = E2M1_MAX if tier.w_fmt == "e2m1" else E0M3_MAX
    w_rot = _chunked(lambda w: _rotate(w.float(), tier.rot) if tier.rot else w.float(), w_nk)
    g = 1.0
    if tier.w_gs:
        g = 2.0 ** math.floor(math.log2(UE4M3_MAX * qmax / float(w_rot.abs().max())))
    return _chunked(lambda w: dequantize_blocks(quantize_blocks((w * g).half(), tier.w_fmt, tier.w_rule)) / g,
                    w_rot)


def sim_act(x: torch.Tensor, tier: Tier) -> torch.Tensor:
    """(M,K) activation -> dequantized float32 in the tier's domain
    (dynamic amax scales, as every activation quantizer uses)."""
    xf = x.float()
    if tier.rot:
        xf = _rotate(xf, tier.rot)
    if tier.a_gs:
        qmax = E2M1_MAX if tier.a_fmt == "e2m1" else E0M3_MAX
        g = 2.0 ** math.floor(math.log2(UE4M3_MAX * qmax / max(float(xf.abs().max()), 1e-30)))
        return fake_quantize(xf * g, tier.a_fmt, "amax") / g
    return fake_quantize(xf, tier.a_fmt, "amax")


def act_block_stats(x: torch.Tensor, rot: int) -> dict:
    xf = x.float()
    if rot:
        xf = _rotate(xf, rot)
    amax = xf.reshape(xf.shape[0], -1, BLOCK).abs().amax(-1)
    return dict(
        n=amax.numel(),
        sub_e2m1=int((amax / E2M1_MAX < 2.0 ** -6).sum()),
        sub_e0m3=int((amax / E0M3_MAX < 2.0 ** -6).sum()),
        zero_e0m3=int((amax / E0M3_MAX < 2.0 ** -10).sum()),
        sat_e2m1=int((amax > UE4M3_MAX * E2M1_MAX).sum()),
        absmax=float(amax.max()),
    )


# ── pointer views and linear wrappers ───────────────────────────────

def view_fp16(ptr: int, rows: int, cols: int) -> torch.Tensor:
    iface = {"data": (int(ptr), False), "shape": (int(rows), int(cols)), "typestr": "<f2", "version": 3}
    owner = type("_Fp16View", (), {"__cuda_array_interface__": iface})()
    return torch.as_tensor(owner, device=DEV)


def group_of(key: tuple[str, str, int, str]) -> str:
    model, kind, _, slot = key
    return f"{'bb' if model == 'backbone' else 'ad'}.{kind}.{slot.replace('.weight', '')}"


class ProbeLinear:
    """Runs the real fp16 GEMM, then scores one tier on the same input."""

    def __init__(self, inner: Fp16Linear, key: tuple[str, str, int, str], tier: Tier,
                 study: "ProbeStudy") -> None:
        self.inner, self.key, self.tier, self.study = inner, key, tier, study

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        self.inner(x_ptr, out_ptr, m, stream)
        self.study.observe(self.key, self.tier, view_fp16(x_ptr, m, self.inner.k), self.inner)


class FakeQuantLinear:
    """Fake-quantizes the activation, then runs the fp16 GEMM against the
    already fake-quantized weight buffer."""

    def __init__(self, inner: Fp16Linear, tier: Tier, study: "PipelineStudy") -> None:
        self.inner, self.tier, self.study = inner, tier, study

    def __call__(self, x_ptr: int, out_ptr: int, m: int, stream: int = 0) -> None:
        x = view_fp16(x_ptr, m, self.inner.k)
        xq32 = sim_act(x, self.tier)
        xq = xq32.half()
        self.study.act_inexact += int((xq.float() != xq32).sum())
        self.inner.gemm.fp16_nn(xq.data_ptr(), self.inner.weight_ptr, out_ptr, m, self.inner.n, self.inner.k, stream)


# ── frontend driving ────────────────────────────────────────────────

class Runner:
    def __init__(self, frames: list[LiberoFrame]) -> None:
        t = time.time()
        dims = dict(REAL_DIMS)
        if MERGE_LINEAR2 is not None:
            dims["merge_linear2"] = MERGE_LINEAR2 == "1"
        self.fe = ImageWAMTorchFrontendThor(
            precision="fp16", dims_override=dims, ckpt_path=CKPT,
            ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
            qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"], dataset_stats_path=STATS)
        print(f"fp16 frontend constructed in {time.time() - t:.1f}s", flush=True)
        fe = self.fe
        q_model, q_tok = fe._qwen3
        self.ctx = {}
        for task in sorted({f.task for f in frames}):
            c, msk = encode_prompts(q_model, q_tok, [task])
            self.ctx[task] = (c[0].clone(), msk[0].clone())
        fe._qwen3 = None
        del q_model, q_tok
        torch.cuda.empty_cache()
        self.tokens = []
        for f in frames:
            tok = encode_to_tokens(fe._ae, [torch.from_numpy(f.view1.copy()), torch.from_numpy(f.view2.copy())],
                                   preprocessor=fe._vae_pre, encoder=fe._vae_encoder)
            self.tokens.append(tok[0].to(BF16).clone())
        self.frames = frames
        self.keepalive = {t.data_ptr(): t for t in fe._keepalive}
        self.quant_keys = [k for k, v in fe._weights.items()
                           if isinstance(v, Fp16Linear) and v.n % 16 == 0 and v.k % 16 == 0]
        self.orig_linears = {k: fe._weights[k] for k in self.quant_keys}
        self.cur_task = None
        # First set_prompt() captures the fp16 CUDA graph; do it while the
        # plain Fp16Linear objects are still installed.
        self.prepare(0, noise_for(0))
        print(f"quantizable GEMM weights: {len(self.quant_keys)}; "
              f"fp16-fallback: {[k[-1] for k, v in fe._weights.items() if isinstance(v, Fp16Linear) and k not in self.orig_linears]}",
              flush=True)

    def weight_tensor(self, key: tuple[str, str, int, str]) -> torch.Tensor:
        lin = self.orig_linears[key]
        w = self.keepalive[lin.weight_ptr]
        assert tuple(w.shape) == (lin.k, lin.n), (key, w.shape)
        return w

    def set_linears(self, factory: Callable[[tuple[str, str, int, str], Fp16Linear], object]) -> None:
        for k in self.quant_keys:
            self.fe._weights[k] = factory(k, self.orig_linears[k])

    def n_valid(self, i: int) -> int:
        """Real token count of frame `i`'s prompt (Qwen3 mask)."""
        return int(self.ctx[self.frames[i].task][1].sum())

    def prepare(self, i: int, noise: torch.Tensor) -> None:
        fe, f = self.fe, self.frames[i]
        if f.task != self.cur_task:
            fe._current_prompt = None
            if CONTEXT == "random":
                torch.cuda.manual_seed(1000 + sorted(self.ctx).index(f.task))
                fe.set_prompt()
            else:
                c, msk = self.ctx[f.task]
                fe.set_prompt(context=c, context_mask=msk)
            self.cur_task = f.task
        fe._img_raw.copy_(self.tokens[i])
        p = fe._state_norm.forward(torch.as_tensor(f.state, device=DEV).reshape(1, -1))
        tok = torch.nn.functional.linear(p.to(BF16), fe._proprio_w, fe._proprio_b)
        fe._context[fe._proprio_row].copy_(tok[0])
        fe._action_latent.copy_(noise)

    def run_eager(self) -> None:
        fe = self.fe
        s = torch.cuda.current_stream().cuda_stream
        imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s, attn=fe._attn,
                         mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
                         rope_table=fe._rope_table.data_ptr())
        imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s, attn=fe._attn,
                              action_mods=fe._action_mods, head_mods=fe._head_mods,
                              action_rope_table=fe._action_rope_table.data_ptr(), deltas=fe._deltas)
        torch.cuda.synchronize()

    def outputs(self) -> dict:
        fe = self.fe
        al = fe._action_latent.detach().float().clone()
        return dict(bh=fe._backbone_hidden.detach().float().cpu(), al=al.cpu(),
                    act=fe._action_norm.backward(al).float().cpu())


def noise_for(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, HORIZON, 7), generator=g, dtype=torch.float32).to(DEV, BF16).float()[0]


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


# ── pass 1: per-GEMM error on real activations ──────────────────────

class ProbeStudy:
    def __init__(self, runner: Runner) -> None:
        self.r = runner
        self.err = {}      # (group, tier) -> [err2_total, err2_w_only, err2_a_only, ref2]
        self.per_key = {}  # (key, tier) -> [err2, ref2]
        self.act = {}      # (group, rot) -> summed stats
        self.wcache = {}

    def observe(self, key: tuple[str, str, int, str], tier: Tier, x: torch.Tensor, lin: Fp16Linear) -> None:
        w_kn = self.r.weight_tensor(key)
        if key not in self.wcache:
            self.wcache[key] = sim_weight(w_kn, tier).half()
        wq = self.wcache[key].float()                      # (N,K), tier domain
        xf = x.float()
        w_ref = w_kn.float()
        y_ref = xf @ w_ref                                   # exact fp32 product
        xq = sim_act(x, tier)
        xr = _rotate(xf, tier.rot) if tier.rot else xf
        y_t = xq @ wq.t()
        y_w = xr @ wq.t()
        y_a = xq @ (_rotate(w_ref.t().contiguous(), tier.rot).t() if tier.rot else w_ref)
        ref2 = float((y_ref.double() ** 2).sum())
        e = [float(((y - y_ref).double() ** 2).sum()) for y in (y_t, y_w, y_a)]
        g = group_of(key)
        acc = self.err.setdefault((g, tier.name), [0.0, 0.0, 0.0, 0.0])
        for i in range(3):
            acc[i] += e[i]
        acc[3] += ref2
        pk = self.per_key.setdefault((key, tier.name), [0.0, 0.0])
        pk[0] += e[0]
        pk[1] += ref2
        for rot in (0, 16):
            if (g, rot) not in self.act or self.act[(g, rot)]["_tier"] == tier.name:
                st = act_block_stats(x, rot)
                cur = self.act.setdefault((g, rot), dict(_tier=tier.name, n=0, sub_e2m1=0, sub_e0m3=0,
                                                         zero_e0m3=0, sat_e2m1=0, absmax=0.0))
                for k in ("n", "sub_e2m1", "sub_e0m3", "zero_e0m3", "sat_e2m1"):
                    cur[k] += st[k]
                cur["absmax"] = max(cur["absmax"], st["absmax"])

    def run(self, n_frames: int) -> None:
        r = self.r
        for tier in PROBE_TIERS:
            t = time.time()
            self.wcache = {}
            r.set_linears(lambda k, lin: ProbeLinear(lin, k, tier, self))
            for i in range(n_frames):
                r.prepare(i, noise_for(0))
                r.run_eager()
            self.wcache = {}
            torch.cuda.empty_cache()
            print(f"[probe] {tier.name}: {time.time() - t:.1f}s", flush=True)
        r.set_linears(lambda k, lin: lin)

    def report(self) -> dict:
        groups = sorted({g for g, _ in self.err})
        names = [t.name for t in PROBE_TIERS]
        print("\n=== per-GEMM output rel_l2 vs exact fp32 product (total; real activations, "
              f"{PROBE_FRAMES} frames, all layers/steps pooled) ===")
        print(f"{'group':28s}" + "".join(f"{n:>13s}" for n in names))
        table = {}
        for g in groups:
            row = []
            for n in names:
                e = self.err[(g, n)]
                row.append(math.sqrt(e[0] / e[3]))
                table.setdefault(g, {})[n] = dict(total=math.sqrt(e[0] / e[3]), w_only=math.sqrt(e[1] / e[3]),
                                                   a_only=math.sqrt(e[2] / e[3]))
            print(f"{g:28s}" + "".join(f"{v:13.5f}" for v in row))
        overall = {}
        for n in names:
            tot = [sum(self.err[(g, n)][i] for g in groups) for i in range(4)]
            overall[n] = dict(total=math.sqrt(tot[0] / tot[3]), w_only=math.sqrt(tot[1] / tot[3]),
                              a_only=math.sqrt(tot[2] / tot[3]))
        for part in ("total", "w_only", "a_only"):
            print(f"{'ALL (' + part + ')':28s}" + "".join(f"{overall[n][part]:13.5f}" for n in names))
        # How many individual GEMM weights each tier beats nvfp4 on.
        keys = sorted({k for k, _ in self.per_key}, key=str)
        wins = {}
        for n in names:
            wins[n] = sum(1 for k in keys if self.per_key[(k, n)][0] < self.per_key[(k, "nvfp4")][0])
        print(f"{'#weights better than nvfp4':28s}" + "".join(f"{wins[n]:>13d}" for n in names)
              + f"   (of {len(keys)})")
        print("\n=== activation block statistics (fraction of per-16 blocks) ===")
        print(f"{'group':28s}{'rot':>5s}{'sub(amax/6)':>13s}{'sub(amax/7)':>13s}{'zero(e0m3)':>12s}"
              f"{'sat(>2688)':>12s}{'absmax':>10s}")
        act = {}
        for (g, rot), st in sorted(self.act.items()):
            n = st["n"]
            act[f"{g}|h{rot}"] = {k: v for k, v in st.items() if k != "_tier"}
            print(f"{g:28s}{rot:5d}{st['sub_e2m1'] / n:13.4f}{st['sub_e0m3'] / n:13.4f}"
                  f"{st['zero_e0m3'] / n:12.5f}{st['sat_e2m1'] / n:12.6f}{st['absmax']:10.1f}")
        return dict(groups=table, overall=overall, wins_vs_nvfp4=wins, n_weights=len(keys), act_stats=act)


# ── pass 2: whole-pipeline fake quantization ────────────────────────

class PipelineStudy:
    def __init__(self, runner: Runner) -> None:
        self.r = runner
        self.act_inexact = 0
        self.cpu_orig = {k: runner.weight_tensor(k).detach().cpu() for k in runner.quant_keys}

    def load_weights(self, tier: Tier | None) -> int:
        inexact = 0
        for k in self.r.quant_keys:
            w = self.r.weight_tensor(k)
            w.copy_(self.cpu_orig[k].to(DEV))
            if tier is not None:
                wq32 = sim_weight(w, tier).t().contiguous()
                wq = wq32.half()
                inexact += int((wq.float() != wq32).sum())
                w.copy_(wq)
                del wq32, wq
        torch.cuda.synchronize()
        return inexact

    def run(self) -> dict:
        r = self.r
        n = len(r.frames)
        ref, seed1 = [], []
        self.load_weights(None)
        r.set_linears(lambda k, lin: lin)
        for i in range(n):
            r.prepare(i, noise_for(0))
            r.run_eager()
            ref.append(r.outputs())
            r.prepare(i, noise_for(1))
            r.run_eager()
            seed1.append(r.outputs())
        # Eager vs the captured fp16 graph on the last frame: same kernels.
        r.prepare(n - 1, noise_for(0))
        r.fe._graph.replay()
        torch.cuda.synchronize()
        graph_vs_eager = float((r.fe._action_latent.float().cpu() - ref[-1]["al"]).abs().max())
        print(f"[pipe] fp16 graph replay vs eager, action_latent max|diff| = {graph_vs_eager:.3e}", flush=True)

        rows = {"fp16_seed1": [self._metrics(seed1[i], ref[i], i) for i in range(n)]}
        rows["fp16"] = [self._metrics(ref[i], ref[i], i) for i in range(n)]
        inexact = {}
        for tier in PIPE_TIERS:
            t = time.time()
            inexact[tier.name] = self.load_weights(tier)
            self.act_inexact = 0
            r.set_linears(lambda k, lin: FakeQuantLinear(lin, tier, self))
            out = []
            for i in range(n):
                r.prepare(i, noise_for(0))
                r.run_eager()
                out.append(self._metrics(r.outputs(), ref[i], i))
            rows[tier.name] = out
            inexact[tier.name] = (inexact[tier.name], self.act_inexact)
            print(f"[pipe] {tier.name}: {time.time() - t:.1f}s  non-fp16-exact weight/act elements: "
                  f"{inexact[tier.name]}", flush=True)
        r.set_linears(lambda k, lin: lin)
        self.load_weights(None)
        return self._report(rows, graph_vs_eager, inexact)

    def _metrics(self, o: dict, ref: dict, i: int) -> dict:
        """Cosines and MAEs, plus the share of the `backbone_hidden` squared
        error in the rows the real prompt's tokens occupy
        (`[0, n_valid)`), the remaining text rows (`[n_valid, x0)`,
        padding and the proprio row), and the image rows."""
        f = self.r.frames[i]
        gt = torch.from_numpy(f.gt)
        n_gt = min(len(gt), HORIZON)
        row_err = ((o["bh"] - ref["bh"]).double() ** 2).sum(-1)
        tot = float(row_err.sum())
        nv, x0 = self.r.n_valid(i), self.r.fe.dims["x0"]
        share = (lambda a, b: float(row_err[a:b].sum()) / tot) if tot > 0 else (lambda a, b: float("nan"))
        return dict(
            bh_cos=cos(o["bh"], ref["bh"]), al_cos=cos(o["al"], ref["al"]), act_cos=cos(o["act"], ref["act"]),
            act_mae_vs_fp16=float((o["act"] - ref["act"]).abs().mean()),
            mae_vs_gt=float((o["act"][:n_gt] - gt[:n_gt]).abs().mean()),
            bh_err_valid_text=share(0, nv), bh_err_pad_text=share(nv, x0), bh_err_image=share(x0, row_err.numel()),
            finite=bool(torch.isfinite(o["act"]).all()))

    @staticmethod
    def _report(rows: dict, graph_vs_eager: float, inexact: dict) -> dict:
        print("\n=== whole pipeline vs fp16 (same N(0,1) noise), "
              f"{len(next(iter(rows.values())))} frames ===")
        hdr = (f"{'tier':14s}{'bh_cos med':>11s}{'bh min':>9s}{'al_cos med':>11s}{'al min':>9s}"
               f"{'act_cos med':>12s}{'act min':>9s}{'1-act med':>11s}{'MAE vs fp16':>12s}{'MAE vs GT':>11s}")
        print(hdr)
        summary = {}
        for name, rs in rows.items():
            a = {k: np.array([x[k] for x in rs]) for k in rs[0]}
            s = dict(bh_cos_median=float(np.median(a["bh_cos"])), bh_cos_min=float(a["bh_cos"].min()),
                     al_cos_median=float(np.median(a["al_cos"])), al_cos_min=float(a["al_cos"].min()),
                     act_cos_median=float(np.median(a["act_cos"])), act_cos_min=float(a["act_cos"].min()),
                     act_err_median=float(np.median(1 - a["act_cos"])),
                     act_mae_vs_fp16_mean=float(a["act_mae_vs_fp16"].mean()),
                     mae_vs_gt_mean=float(a["mae_vs_gt"].mean()), all_finite=bool(a["finite"].all()),
                     bh_err_share_median={k: float(np.median(a[k])) for k in
                                          ("bh_err_valid_text", "bh_err_pad_text", "bh_err_image")})
            summary[name] = s
            print(f"{name:14s}{s['bh_cos_median']:11.5f}{s['bh_cos_min']:9.5f}{s['al_cos_median']:11.5f}"
                  f"{s['al_cos_min']:9.5f}{s['act_cos_median']:12.5f}{s['act_cos_min']:9.5f}"
                  f"{s['act_err_median']:11.2e}{s['act_mae_vs_fp16_mean']:12.5f}{s['mae_vs_gt_mean']:11.5f}")
        print("\nbackbone_hidden squared-error share by rows (median over frames): "
              "real-prompt token rows / other text rows (padding, proprio) / image rows")
        for name, sm in summary.items():
            sh = sm["bh_err_share_median"]
            print(f"{name:14s}{sh['bh_err_valid_text']:8.3f}{sh['bh_err_pad_text']:8.3f}{sh['bh_err_image']:8.3f}")
        return dict(summary=summary, per_frame=rows, graph_vs_eager=graph_vs_eager, inexact=inexact)


@torch.no_grad()
def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    frames = load_libero_frames(os.environ["DATA_ROOT"], SUITE, N_TASKS, FRAMES, HORIZON)
    print(f"frames: {len(frames)} ({SUITE}, frames {FRAMES}); probe frames: {min(PROBE_FRAMES, len(frames))}",
          flush=True)
    runner = Runner(frames)
    print(f"merge_linear2={runner.fe.dims.get('merge_linear2')} context={CONTEXT}", flush=True)
    result = dict(suite=SUITE, frames=[(f.episode, f.frame) for f in frames], context=CONTEXT,
                  merge_linear2=bool(runner.fe.dims.get("merge_linear2")),
                  probe_tiers=[t.name for t in PROBE_TIERS], pipe_tiers=[t.name for t in PIPE_TIERS])
    if PROBE_FRAMES > 0:
        probe = ProbeStudy(runner)
        probe.run(min(PROBE_FRAMES, len(frames)))
        result["probe"] = probe.report()
    if PIPE_TIERS:
        result["pipeline"] = PipelineStudy(runner).run()
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
