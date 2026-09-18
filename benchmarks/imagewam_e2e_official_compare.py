"""End-to-end: FlashRT served frontend (real checkpoint, VAE, Qwen3, proprio,
shift schedule) vs official ImageWAM `infer_action_flux2` on real LIBERO frames.

Both sides get the same preprocessed 224x448 image (official PIL center-crop
resize), the same official Qwen3 context/mask, the same normalized proprio and
the same N(0,1) initial action noise. Reported per frame:

  ctx_cos              FlashRT Qwen3 encoding vs official context
  fr_vs_off            FlashRT vs official normalized actions, same noise
  off_seed0_vs_seed1   official vs itself across seeds (sampling spread)
  fr_noise001_vs_off   FlashRT with 0.01*noise (what infer() uses) vs official
  served_vs_off        frontend.infer() as served vs official (denormalized)
  mae_*_vs_gt          denormalized action chunk vs dataset ground truth

The frontend is built by the deployment entry
`flash_rt.frontends.torch.imagewam_thor.load_imagewam` on an
`ImageWAMWorkload` (plan.md "configuration consolidation", W11/W12): one named
`PROFILE` carries the switches, the environment variables below are expert
overrides on top of it, and the structure comes from the checkpoint itself
(`structure=None`), not from a table typed here.

Required env: FLUX2_SRC, CKPT_PATH (dataset_stats.json and config.yaml beside
it), FLUX2_MODEL_PATH, FLUX2_AE_MODEL_PATH, QWEN3_MODEL_SPEC, DATA_ROOT
(LIBERO-fastwam, LeRobot v2.1), and the ImageWAM `src/` on PYTHONPATH.
Optional env:
  PROFILE      default -- the named profile (`config_resolver.PROFILES`).
  SUITE        libero_spatial.
  N_TASKS      10.
  FRAMES       "0,40".
  SEEDS        "0,1".
  PRECISION    fp16 -- overrides the profile's precision.
  CALIBRATION  activation-calibration file for fp8_static* (see
               benchmarks/imagewam_build_calibration.py), and the AWQ
               statistics. The resolver refuses fp8_static* without one
               (rule R1); the N(0, 0.1) placeholder scales stay a
               constructor-only path.
  NVFP4_AWQ    0/1, with PRECISION=nvfp4 or nvfp4_sim and CALIBRATION.
  AWQ_ALPHA    0.5.
  AWQ_SCOPE    adaln+down.

The switch variables below are expert overrides of the profile
(`config_resolver.EXPERT_KEYS`), each applied only when it is set in the
environment: an unset variable leaves the profile's own value in place, so
`PROFILE=fast` alone runs the whole `fast` profile. `use_fa4` is never set
from here -- the profile's own `use_fa4` (`None` in `default`) lets the
frontend resolve the `FLASHRT_THOR_FA4` opt-in exactly as before.

VAE options (roadmap items 2 and 5, plan.md):
  VAE_ENCODER torch | native -- frontend vae_encoder.
  VAE_GRAPH   1: the VAE runs inside the CUDA graph, and the resolver derives
              `vae_graph_input` from the workload ((2, H, W) of the FlashRT
              views, so it follows RAW_VIEWS/RAW_SIZE). 0 or unset: outside
              the graph.
  VAE_RESIZE  area (default) | pil_bilinear -- frontend vae_resize, a
              constructor argument of `load_imagewam`, not a profile option.
  RAW_VIEWS   0 (default): FlashRT gets the same PIL-resized 224x224 views
              as the official side. 1: FlashRT gets the raw 512x512
              dataset frames and resizes them itself (vae_resize), while
              the official side keeps its PIL center-crop resize -- this
              measures the served preprocessing against the official one
              (issues.md ISSUE-030).
  RAW_SIZE    0 (default): dataset frames as stored (512x512). S > 0: every
              frame is first PIL-downscaled (BILINEAR) to SxS for BOTH
              sides, a stand-in for a simulator rendering at SxS (the
              official eval renders at 256x256).

Text context option (issues.md ISSUE-020):
  TEXT_TRIM   1: frontend text_trim=True, the sequence holds only the
              valid tokens and the proprio row (one graph per length);
              the per-frame `x0` column is the context length used. Unset:
              the profile's own text_trim (all 512 padded text rows for
              `default`, trimmed for `fast`).
"""
from __future__ import annotations

import json
import os
import sys
import time

import av
import numpy as np
import pandas as pd
import torch
from PIL import Image

from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state
from flash_rt.models.imagewam.config_resolver import format_effective_config
from flash_rt.models.imagewam.libero_dims import (
    LIBERO_HORIZON as HORIZON,
    LIBERO_SHIFT as SHIFT,
    LIBERO_STEPS as STEPS,
)
from flash_rt.models.imagewam.workload import ImageWAMWorkload

sys.path.insert(0, os.environ["FLUX2_SRC"] + "/src")
sys.path.insert(0, os.environ["FLUX2_SRC"])

DEV = "cuda"
BF16 = torch.bfloat16
CKPT = os.environ["CKPT_PATH"]
CKPT_DIR = os.path.dirname(CKPT)
STATS = os.path.join(CKPT_DIR, "dataset_stats.json")
PROFILE = os.environ.get("PROFILE", "default")
SUITE = os.environ.get("SUITE", "libero_spatial")
N_TASKS = int(os.environ.get("N_TASKS", "10"))
FRAMES = [int(x) for x in os.environ.get("FRAMES", "0,40").split(",")]
PRECISION = os.environ.get("PRECISION", "fp16")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
CALIBRATION = os.environ.get("CALIBRATION") or None
AWQ_KW = (dict(nvfp4_awq=True, awq_alpha=float(os.environ.get("AWQ_ALPHA", "0.5")),
               awq_scope=os.environ.get("AWQ_SCOPE", "adaln+down"))
          if os.environ.get("NVFP4_AWQ", "0") == "1" else {})
VAE_ENCODER = os.environ.get("VAE_ENCODER", "torch")
VAE_GRAPH = os.environ.get("VAE_GRAPH", "0") == "1"
VAE_RESIZE = os.environ.get("VAE_RESIZE", "area")
RAW_VIEWS = os.environ.get("RAW_VIEWS", "0") == "1"
RAW_SIZE = int(os.environ.get("RAW_SIZE", "0"))
TEXT_TRIM = os.environ.get("TEXT_TRIM", "0") == "1"
# FA4_MOT=1 also runs the ActionDiT ("mot") attention through FA4; the
# backbone site is switched by FLASHRT_THOR_FA4=1 as before.
FA4_MOT = os.environ.get("FA4_MOT", "0") == "1"


def expert_overrides() -> dict:
    """The option environment variables that are actually set, as expert
    overrides of the profile (`config_resolver.EXPERT_KEYS`).

    A variable that is not in the environment is absent from the dict, so the
    profile's own value for that switch is what runs: a profile row is the
    profile, not the profile plus this script's defaults. `use_fa4` is not an
    override here (the profile decides, and `None` leaves the
    `FLASHRT_THOR_FA4` opt-in to the frontend).
    """
    expert = {}
    if "TEXT_TRIM" in os.environ:
        expert["text_trim"] = TEXT_TRIM
    if "FA4_MOT" in os.environ:
        expert["use_fa4_mot"] = FA4_MOT
    if "VAE_GRAPH" in os.environ:
        expert["vae_graph"] = VAE_GRAPH
    if "VAE_ENCODER" in os.environ:
        expert["vae_encoder"] = VAE_ENCODER
    expert.update(AWQ_KW)
    return expert


EXPERT = expert_overrides()


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def center_crop_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Official eval_libero_single._center_crop_resize (PIL bilinear + center crop)."""
    pil = Image.fromarray(img)
    sw, sh = pil.size
    scale = max(w / sw, h / sh)
    r = pil.resize((round(sw * scale), round(sh * scale)), resample=Image.BILINEAR)
    rw, rh = r.size
    left, top = (rw - w) // 2, (rh - h) // 2
    return np.asarray(r.crop((left, top, left + w, top + h)))


def read_frame(path: str, idx: int) -> np.ndarray:
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == idx:
                return f.to_ndarray(format="rgb24")
    raise IndexError(idx)


def load_samples():
    root = os.path.join(os.environ["DATA_ROOT"], f"{SUITE}_no_noops_lerobot")
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{root}/meta/tasks.jsonl")}
    eps = [json.loads(l) for l in open(f"{root}/meta/episodes.jsonl")]
    seen, out = set(), []
    for ep in eps:
        e = ep["episode_index"]
        df = pd.read_parquet(f"{root}/data/chunk-000/episode_{e:06d}.parquet")
        ti = int(df["task_index"].iloc[0])
        if ti in seen:
            continue
        seen.add(ti)
        for fr in FRAMES:
            if fr >= len(df):
                continue
            v1 = read_frame(f"{root}/videos/chunk-000/observation.images.image/episode_{e:06d}.mp4", fr)
            v2 = read_frame(f"{root}/videos/chunk-000/observation.images.wrist_image/episode_{e:06d}.mp4", fr)
            if RAW_SIZE:
                v1, v2 = (np.array(Image.fromarray(v).resize((RAW_SIZE, RAW_SIZE), resample=Image.BILINEAR))
                          for v in (v1, v2))
            state = np.asarray(df["observation.state"].iloc[fr], dtype=np.float32)
            gt = np.stack(df["action"].iloc[fr:fr + HORIZON].to_numpy()).astype(np.float32)
            out.append(dict(ep=e, frame=fr, task=tasks[ti], v1=v1, v2=v2, state=state, gt=gt))
        if len(seen) >= N_TASKS:
            break
    return out


def build_official():
    from omegaconf import OmegaConf
    from imagewam.runtime import create_imagewam_flux2_klein
    cfg = OmegaConf.load(os.path.join(CKPT_DIR, "config.yaml")).model
    m = create_imagewam_flux2_klein(
        flux2_model_path=os.environ["FLUX2_MODEL_PATH"], ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"],
        flux2_src_path=os.environ["FLUX2_SRC"], variant=cfg.variant,
        qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"], load_text_encoder=True,
        action_dit_config=cfg.action_dit_config, action_dit_pretrained_path=None,
        proprio_dim=cfg.proprio_dim, video_scheduler=cfg.video_scheduler,
        action_scheduler=cfg.action_scheduler, loss=cfg.loss,
        mot_checkpoint_mixed_attn=cfg.mot_checkpoint_mixed_attn,
        mot_gqa_implementation=cfg.mot_gqa_implementation,
        mot_force_flash_attention=cfg.mot_force_flash_attention,
        pack_proprio_after_text=cfg.pack_proprio_after_text, device=DEV)
    m.load_checkpoint(CKPT)
    return m.eval()


def flashrt_infer_with_noise(fe, v1, v2, state, noise):
    """frontend.infer() body with the initial action noise injected instead of 0.01*randn."""
    if fe._vae_stage is not None:
        fe._vae_stage.stage([torch.from_numpy(v1), torch.from_numpy(v2)])
    else:
        from flash_rt.models.imagewam.vae_encoder import encode_to_tokens
        tokens = encode_to_tokens(fe._ae, torch.from_numpy(v1), torch.from_numpy(v2), preprocessor=fe._vae_pre,
                                  encoder=fe._vae_encoder)
        fe._img_raw.copy_(tokens[0].to(dtype=BF16))
    p = fe._state_norm.forward(torch.as_tensor(state, device=DEV).reshape(1, -1))
    tok = torch.nn.functional.linear(p.to(BF16), fe._proprio_w, fe._proprio_b)
    fe._context[fe._proprio_row].copy_(tok[0])
    fe._action_latent.copy_(noise)
    fe._graph.replay()
    torch.cuda.synchronize()
    return fe._action_latent.detach().clone()


@torch.no_grad()
def main():
    samples = load_samples()
    print(f"samples: {len(samples)} ({SUITE}, frames {FRAMES})", flush=True)
    print(f"PROFILE={PROFILE} overrides={EXPERT} VAE_RESIZE={VAE_RESIZE} RAW_VIEWS={int(RAW_VIEWS)} "
          f"RAW_SIZE={RAW_SIZE}", flush=True)

    # The served workload (plan.md W1): the dims, the sequence layout and the
    # in-graph VAE input are derived from it, not typed here. RAW_VIEWS=1
    # feeds the VAE the raw dataset frames, so their own size is the workload's.
    view_h, view_w = samples[0]["v1"].shape[:2] if RAW_VIEWS else (224, 224)
    workload = ImageWAMWorkload(
        num_views=2, image_h=int(view_h), image_w=int(view_w), text_max_len=512,
        action_horizon=HORIZON, action_dim=7, proprio_dim=8, num_steps=STEPS, shift=SHIFT)

    t = time.time()
    off = build_official()
    print(f"official loaded in {time.time() - t:.1f}s", flush=True)

    from flash_rt.frontends.torch.imagewam_thor import load_imagewam
    t = time.time()
    fe = load_imagewam(
        CKPT, workload, profile=PROFILE, precision=PRECISION, calibration_path=CALIBRATION,
        ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
        qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"], dataset_stats_path=STATS,
        vae_resize=VAE_RESIZE, **EXPERT)
    print(f"flashrt (profile={PROFILE}, {PRECISION}, calibration={CALIBRATION}, awq={AWQ_KW}) "
          f"constructed in {time.time() - t:.1f}s", flush=True)

    rows, cur_task = [], None
    for s in samples:
        # Official text context (shared by both sides) + FlashRT's own Qwen3 encoding check.
        ctx, mask = off._prepare_flux2_infer_text(s["task"], None, None)
        if s["task"] != cur_task:
            from flash_rt.models.imagewam.text_encoder import encode_prompts
            q_model, q_tok = fe._qwen3
            fr_ctx, fr_mask = encode_prompts(q_model, q_tok, [s["task"]])
            n = int(mask[0].sum())
            ctx_cos = cos(fr_ctx[0][:n], ctx[0][:n])
            mask_eq = bool((fr_mask[0].bool().cpu() == mask[0].bool().cpu()).all())
            fe.set_prompt(context=ctx[0], context_mask=mask[0])
            cur_task = s["task"]

        view1 = center_crop_resize(s["v1"], 224, 224)
        view2 = center_crop_resize(s["v2"], 224, 224)
        fr_v1, fr_v2 = (s["v1"], s["v2"]) if RAW_VIEWS else (view1, view2)
        img = np.concatenate([view1, view2], axis=1)
        x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) * (2.0 / 255.0) - 1.0
        p_norm = fe._state_norm.forward(torch.as_tensor(s["state"], device=DEV).reshape(1, -1))

        outs = {}
        for seed in SEEDS:
            o = off.infer_action_flux2(prompt=None, input_image=x, action_horizon=HORIZON, proprio=p_norm,
                                       context=ctx, context_mask=mask, num_inference_steps=STEPS,
                                       sigma_shift=SHIFT, seed=seed)["action"]
            g = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn((1, HORIZON, 7), generator=g, dtype=torch.float32).to(DEV, BF16).float()[0]
            f = flashrt_infer_with_noise(fe, fr_v1, fr_v2, s["state"], noise).cpu()
            f_small = flashrt_infer_with_noise(fe, fr_v1, fr_v2, s["state"], noise * 0.01).cpu()
            outs[seed] = (o, f, f_small)

        served = torch.from_numpy(fe.infer({"view1": torch.from_numpy(fr_v1), "view2": torch.from_numpy(fr_v2),
                                             "proprio": s["state"]})["actions"])
        o0, f0, fs0 = outs[SEEDS[0]]
        denorm = lambda a: fe._action_norm.backward(a.to(DEV)).cpu()
        gt = torch.from_numpy(s["gt"])
        n_gt = min(len(gt), HORIZON)
        row = dict(
            ep=s["ep"], frame=s["frame"], task=s["task"][:48], x0=fe.active_dims["x0"],
            ctx_cos=ctx_cos, mask_eq=mask_eq,
            fr_vs_off=cos(f0, o0),
            fr_vs_off_seed1=cos(outs[SEEDS[-1]][1], outs[SEEDS[-1]][0]),
            off_seed0_vs_seed1=cos(outs[SEEDS[0]][0], outs[SEEDS[-1]][0]),
            fr_noise001_vs_off=cos(fs0, o0),
            served_vs_off=cos(denorm(o0), served),
            mae_off_vs_gt=(denorm(o0)[:n_gt] - gt[:n_gt]).abs().mean().item(),
            mae_fr_vs_gt=(denorm(f0)[:n_gt] - gt[:n_gt]).abs().mean().item(),
            mae_served_vs_gt=(served[:n_gt] - gt[:n_gt]).abs().mean().item(),
            finite=bool(torch.isfinite(f0).all() and torch.isfinite(served).all()),
        )
        rows.append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    df = pd.DataFrame(rows)
    print("\n=== SUMMARY ===")
    for c in ["ctx_cos", "fr_vs_off", "fr_vs_off_seed1", "off_seed0_vs_seed1", "fr_noise001_vs_off",
              "served_vs_off", "mae_off_vs_gt", "mae_fr_vs_gt", "mae_served_vs_gt"]:
        print(f"{c:22s} min={df[c].min():.5f} median={df[c].median():.5f} mean={df[c].mean():.5f}")
    print(f"all finite: {bool(df['finite'].all())}; masks equal: {bool(df['mask_eq'].all())}")

    # Steady-state latency (contaminated by co-tenant GPU load on this box; indicative only).
    report_jetson_clock_state()
    obs = {"view1": torch.from_numpy(fr_v1), "view2": torch.from_numpy(fr_v2), "proprio": s["state"]}
    for _ in range(5):
        fe.infer(obs)
    ts = []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fe.infer(obs); ts.append((time.perf_counter() - t0) * 1e3)
    print(f"infer() P50={np.median(ts):.1f}ms min={min(ts):.1f}ms (shared GPU, not a perf number)")
    # The configuration that actually ran: FA4 can fall back to cuBLAS at capture time. The
    # configuration itself is the one `load_imagewam` resolved for this frontend
    # (`fe.resolved_config`), not a second resolution here.
    print(format_effective_config(fe.resolved_config.options, use_fa4=fe.use_fa4, use_fa4_mot=fe.use_fa4_mot,
                                  fa4_fallback_reason=fe.fa4_fallback_reason))
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
