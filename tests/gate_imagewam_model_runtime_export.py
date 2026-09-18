"""Gate: the real ImageWAM checkpoint driven only through `frt_model_runtime_v1`.

A ctypes consumer (tests/_helpers/model_runtime_consumer.py) uses only the
runtime's C function pointers, its port descriptors, `frt_buffer_dptr` and
the CUDA runtime. For the same camera frames, proprio, prompt and initial
noise it must reproduce `ImageWAMTorchFrontendThor.infer()` bit for bit:

  1. images STAGED + proprio STAGED + noise SWAP -> actions STAGED and
     actions_raw SWAP, vs infer()
  2. image_tokens SWAP (the tokens infer() staged) instead of images; with
     `--vae-graph-input` (VAE inside the graph) the raw frames written to
     the image_views SWAP window instead
  3. prompt SETUP (a second task string), vs set_prompt() + infer()

The initial noise is explicit on both sides: `infer(..., action_noise=)`
and the `noise` SWAP window get the same 0.01 * N(0,1) latent.

plus a determinism control (infer() twice, same seed) and an indicative,
alternating latency A/B of infer() vs one ABI tick.

Build exec/ and runtime/ first (see docs/imagewam_model_runtime.md), then:

    PYTHONPATH=.:$IMAGEWAM_SRC:$FLUX2_SRC/src python tests/gate_imagewam_model_runtime_export.py \
        --precision fp16                              # H100
        --precision nvfp4                             # Thor
        --precision nvfp4 --vae-graph-input 224 224   # Thor, VAE inside the graph

Env: CKPT_PATH (dataset_stats.json beside it), FLUX2_AE_MODEL_PATH (or
AE_MODEL_PATH), FLUX2_SRC, QWEN3_MODEL_SPEC. Optional DATA_ROOT
(LIBERO-fastwam, LeRobot v2.1) supplies a real frame and state; without
it the frames and state are seeded random.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path, make_image_views

HORIZON, STEPS, SHIFT = 64, 10, 5.0
REAL_DIMS = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=HORIZON, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=1.0 / STEPS, num_denoise_steps=STEPS,
    ref_h=14, ref_w=28, proprio_dim=8, shift=SHIFT, num_train_timesteps=1000,
)
EXPECTED_PORTS = ["images", "image_tokens", "proprio", "noise", "actions", "actions_raw", "prompt"]
EXPECTED_PORTS_VAE_IN_GRAPH = ["images", "image_views", "proprio", "noise", "actions", "actions_raw", "prompt"]
PROMPT_A = "pick up the black bowl between the plate and the ramekin and place it on the plate"
PROMPT_B = "pick up the black bowl next to the ramekin and place it on the plate"


def _center_crop_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(img)
    sw, sh = pil.size
    scale = max(w / sw, h / sh)
    r = pil.resize((round(sw * scale), round(sh * scale)), resample=Image.BILINEAR)
    left, top = (r.size[0] - w) // 2, (r.size[1] - h) // 2
    return np.array(r.crop((left, top, left + w, top + h)))


def _real_observation(data_root: str) -> tuple[list[np.ndarray], np.ndarray, str]:
    import av
    import pandas as pd
    root = os.path.join(data_root, "libero_spatial_no_noops_lerobot")
    df = pd.read_parquet(f"{root}/data/chunk-000/episode_000000.parquet")
    frames = []
    for cam in ("image", "wrist_image"):
        with av.open(f"{root}/videos/chunk-000/observation.images.{cam}/episode_000000.mp4") as c:
            frame = next(iter(c.decode(video=0))).to_ndarray(format="rgb24")
        frames.append(np.ascontiguousarray(_center_crop_resize(frame, 224, 224)))
    state = np.asarray(df["observation.state"].iloc[0], dtype=np.float32)
    return frames, state, "libero_spatial episode 0 frame 0"


def _observation(seed: int) -> tuple[list[np.ndarray], np.ndarray, str]:
    data_root = os.environ.get("DATA_ROOT")
    if data_root and os.path.isdir(os.path.join(data_root, "libero_spatial_no_noops_lerobot")):
        return _real_observation(data_root)
    rng = np.random.default_rng(seed)
    frames = [np.ascontiguousarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    state = rng.uniform(-0.5, 0.5, REAL_DIMS["proprio_dim"]).astype(np.float32)
    return frames, state, "seeded random frames"


def _compare(label: str, a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    cos = float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    row = {"check": label, "array_equal": bool(np.array_equal(a, b)),
           "max_abs": float(np.max(np.abs(a64 - b64))), "cos": cos}
    print(f"  {label:<44} array_equal={row['array_equal']!s:<5} max_abs={row['max_abs']:.3g} cos={cos:.8f}")
    return row


def _percentiles(ms: list[float]) -> str:
    q = np.percentile(ms, [10, 50, 90])
    return f"P10={q[0]:.2f} P50={q[1]:.2f} P90={q[2]:.2f} ms (n={len(ms)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--use-fa4", action="store_true")
    ap.add_argument("--vae-graph-input", type=int, nargs=2, metavar=("H", "W"), default=None,
                    help="run the VAE inside the graph for two H x W views (frames must match)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ckpt = os.environ["CKPT_PATH"]
    ae_path = os.environ.get("FLUX2_AE_MODEL_PATH") or os.environ["AE_MODEL_PATH"]
    flux2_src = os.environ["FLUX2_SRC"]
    sys.path.insert(0, flux2_src + "/src")
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
    from flash_rt.models.imagewam.runtime_export import FrtImageView

    frames, state, obs_label = _observation(args.seed)
    t0 = time.time()
    fe = ImageWAMTorchFrontendThor(
        precision=args.precision, use_fa4=args.use_fa4, dims_override=dict(REAL_DIMS), ckpt_path=ckpt,
        ae_model_path=ae_path, flux2_src=flux2_src, qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
        dataset_stats_path=os.path.join(os.path.dirname(ckpt), "dataset_stats.json"),
        vae_graph_input=None if args.vae_graph_input is None else (2, *args.vae_graph_input))
    fe.set_prompt(PROMPT_A)
    in_graph = args.vae_graph_input is not None
    print(f"frontend ({args.precision}, use_fa4={fe.use_fa4}, vae_in_graph={in_graph}) ready in "
          f"{time.time() - t0:.1f}s; observation: {obs_label}")

    obs = {"view1": torch.from_numpy(frames[0]), "view2": torch.from_numpy(frames[1]), "proprio": state}
    views = make_image_views(frames, FrtImageView)
    chunk = (HORIZON, 7)

    mr = fe.export_model_runtime(identity={"gate": "imagewam_model_runtime_export"})
    consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
    rows = []
    try:
        names = [p.name for p in consumer.ports]
        print(f"runtime: ports={names} stages={consumer.n_stages} fingerprint=0x{consumer.fingerprint:016x}")
        if names != (EXPECTED_PORTS_VAE_IN_GRAPH if in_graph else EXPECTED_PORTS) or consumer.n_stages != 1:
            raise AssertionError(f"unexpected schema: {names}, stages={consumer.n_stages}")

        def draw_noise(seed: int) -> torch.Tensor:
            torch.manual_seed(seed)
            return torch.empty_like(fe._action_latent).normal_().mul_(0.01)

        def python_ref(seed: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
            actions = fe.infer(obs, action_noise=draw_noise(seed))["actions"]
            return actions, fe._action_latent.detach().cpu().numpy().copy(), fe._img_raw.detach().clone()

        def abi_tick(noise: torch.Tensor, tokens: torch.Tensor | None) -> tuple[np.ndarray, np.ndarray]:
            if tokens is None:
                consumer.set_input("images", views)
            elif in_graph:
                consumer.write_swap("image_views", np.stack(frames))
            else:
                consumer.write_swap("image_tokens", tokens.contiguous().view(torch.int16).cpu().numpy())
            consumer.set_input("proprio", state.tobytes())
            consumer.write_swap("noise", noise.cpu().numpy())
            consumer.step()
            return (consumer.get_output("actions", np.float32, chunk),
                    consumer.read_swap("actions_raw", np.float32, chunk))

        print("parity (ABI consumer vs frontend.infer()):")
        ref_a, ref_raw, ref_tokens = python_ref(args.seed)
        again, _, _ = python_ref(args.seed)
        rows.append(_compare("control: infer() vs infer(), same seed", again, ref_a))

        abi_a, abi_raw = abi_tick(draw_noise(args.seed), None)
        if in_graph:  # the graph wrote img_raw from the staged uint8 views
            abi_tokens = fe._img_raw.detach().view(torch.int16).cpu().numpy()
        else:
            abi_tokens = consumer.read_swap("image_tokens", np.int16, tuple(ref_tokens.shape))
        rows.append(_compare("images STAGED -> VAE tokens (img_raw)", abi_tokens,
                             ref_tokens.view(torch.int16).cpu().numpy()))
        rows.append(_compare("images STAGED: actions (denormalized)", abi_a, ref_a))
        rows.append(_compare("images STAGED: actions_raw (normalized)", abi_raw, ref_raw))

        swap_a, _ = abi_tick(draw_noise(args.seed), ref_tokens)
        rows.append(_compare("image_views SWAP: actions" if in_graph else "image_tokens SWAP: actions",
                             swap_a, ref_a))

        fe._current_prompt = None
        fe.set_prompt(PROMPT_B)
        ref_b, _, _ = python_ref(args.seed)
        fe.set_prompt(PROMPT_A)
        consumer.set_input("prompt", PROMPT_B.encode())
        abi_b, _ = abi_tick(draw_noise(args.seed), None)
        rows.append(_compare("prompt SETUP (task B): actions", abi_b, ref_b))
        prompt_effect = float(np.max(np.abs(ref_b - ref_a)))
        print(f"  prompt changes the chunk: max_abs(task B - task A) = {prompt_effect:.4f}")
        consumer.set_input("prompt", PROMPT_A.encode())

        finite = bool(np.isfinite(ref_a).all() and np.isfinite(abi_a).all())
        ok = all(r["array_equal"] for r in rows) and finite and prompt_effect > 0

        if args.bench_iters > 0:
            noise = draw_noise(args.seed)

            def tick_python() -> None:
                fe.infer(obs, action_noise=noise)

            def tick_abi() -> None:
                abi_tick(noise, None)

            for _ in range(3):
                tick_python()
                tick_abi()
            ms = {"infer()": [], "ABI tick": []}
            for _ in range(args.bench_iters):
                for label, fn in (("infer()", tick_python), ("ABI tick", tick_abi)):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    ms[label].append((time.perf_counter() - t) * 1e3)
            print("latency, alternating A/B, wall incl. VAE/proprio staging and readback (indicative only):")
            for label, v in ms.items():
                print(f"  {label:<9} {_percentiles(v)}")
            rows.append({"latency_ms": {k: list(v) for k, v in ms.items()}})

        print(f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; finite={finite}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"precision": args.precision, "observation": obs_label, "rows": rows}, f, indent=1)
        print("PASS" if ok else "FAIL", "- ImageWAM model runtime (io=python) vs infer()")
        return 0 if ok else 1
    finally:
        consumer.close()
        mr.release()


if __name__ == "__main__":
    raise SystemExit(main())
