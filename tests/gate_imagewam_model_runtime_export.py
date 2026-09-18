"""Gate: the real ImageWAM checkpoint driven only through `frt_model_runtime_v1`.

A ctypes consumer (tests/_helpers/model_runtime_consumer.py) uses only the
runtime's C function pointers, its port descriptors, `frt_buffer_dptr` and
the CUDA runtime. For the same camera frames, proprio, prompt and initial
noise it must reproduce `ImageWAMTorchFrontendThor.infer()` bit for bit:

  1. images STAGED + proprio STAGED + noise SWAP -> actions STAGED,
     actions_raw SWAP and the VAE tokens, vs infer()
  2. image_tokens SWAP (the tokens infer() staged) instead of images; with
     `--vae-graph-input` (VAE inside the graph) the raw frames written to
     the image_views SWAP window instead
  3. prompt SETUP (a second task string), vs set_prompt() + infer()

Before every ABI tick, every buffer the tick must write or refresh is
NaN-filled (`img_raw` or the in-graph uint8 views, the proprio row or the
whole context, K/V caches, `Q_O`, backbone residual, action latent), so a
row cannot pass on what the reference `infer()` left behind. Unless
`--no-mutants`, each row is then re-run with the verb it exercises made a
no-op (images, proprio, prompt, step); every mutant must make its row
fail, which shows the row can detect a verb that stages nothing.

The initial noise is explicit on both sides: `infer(..., action_noise=)`
and the `noise` SWAP window get the same 0.01 * N(0,1) latent.

Also a determinism control (infer() twice, same noise) and an indicative,
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

from _helpers.imagewam_abi_checks import bits, poison_tick_state, python_step_noop, python_verb_noop
from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path, make_image_views

from flash_rt.models.imagewam.libero_dims import (
    LIBERO_HORIZON as HORIZON,
    LIBERO_REAL_DIMS as REAL_DIMS,
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
    print(f"  {label:<56} array_equal={row['array_equal']!s:<5} max_abs={row['max_abs']:.3g} cos={cos:.8f}")
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
    ap.add_argument("--no-mutants", action="store_true", help="skip the no-op verb mutants")
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

    def draw_noise(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.empty_like(fe._action_latent).normal_().mul_(0.01)

    noise = draw_noise(args.seed)

    def reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = fe.infer(obs, action_noise=noise)["actions"]
        return actions, fe._action_latent.detach().cpu().numpy().copy(), bits(fe._img_raw)

    def restore_prompt_a() -> None:
        fe._current_prompt = None
        fe.set_prompt(PROMPT_A)

    # Rows: each poisons the tick's buffers, drives one ABI tick, and
    # returns (label, abi value, reference value) triples.
    def staged_row(c: ModelRuntimeConsumer) -> list:
        poison_tick_state(fe)
        c.set_input("images", views)
        c.set_input("proprio", state.tobytes())
        c.write_swap("noise", noise.cpu().numpy())
        c.step()
        actions = c.get_output("actions", np.float32, chunk)
        raw = c.read_swap("actions_raw", np.float32, chunk)
        return [("images STAGED + proprio STAGED: actions (denormalized)", actions, ref_a),
                ("images STAGED + proprio STAGED: actions_raw", raw, ref_raw),
                ("images STAGED -> VAE tokens (img_raw bits)", bits(fe._img_raw), ref_tokens)]

    def swap_row(c: ModelRuntimeConsumer) -> list:
        poison_tick_state(fe)
        if in_graph:
            c.write_swap("image_views", np.stack(frames))
        else:
            c.write_swap("image_tokens", ref_tokens)
        c.set_input("proprio", state.tobytes())
        c.write_swap("noise", noise.cpu().numpy())
        c.step()
        label = "image_views SWAP: actions" if in_graph else "image_tokens SWAP: actions"
        return [(label, c.get_output("actions", np.float32, chunk), ref_a)]

    def prompt_row(c: ModelRuntimeConsumer) -> list:
        poison_tick_state(fe, whole_context=True)
        c.set_input("prompt", PROMPT_B.encode())
        c.set_input("images", views)
        c.set_input("proprio", state.tobytes())
        c.write_swap("noise", noise.cpu().numpy())
        c.step()
        result = [("prompt SETUP (task B): actions", c.get_output("actions", np.float32, chunk), ref_b)]
        restore_prompt_a()
        return result

    def compare_rows(triples: list) -> list:
        return [_compare(label, a, b) for label, a, b in triples]

    mr = fe.export_model_runtime(identity={"gate": "imagewam_model_runtime_export"})
    consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
    rows, mutants = [], []
    try:
        names = [p.name for p in consumer.ports]
        print(f"runtime: ports={names} stages={consumer.n_stages} fingerprint=0x{consumer.fingerprint:016x}")
        if names != (EXPECTED_PORTS_VAE_IN_GRAPH if in_graph else EXPECTED_PORTS) or consumer.n_stages != 1:
            raise AssertionError(f"unexpected schema: {names}, stages={consumer.n_stages}")

        fe._current_prompt = None
        fe.set_prompt(PROMPT_B)
        ref_b = fe.infer(obs, action_noise=noise)["actions"]
        restore_prompt_a()
        ref_a, ref_raw, ref_tokens = reference()
        again, _, _ = reference()
        prompt_effect = float(np.max(np.abs(ref_b - ref_a)))

        print("parity (ABI consumer vs frontend.infer(), tick buffers NaN-poisoned before every tick):")
        rows.append(_compare("control: infer() vs infer(), same noise", again, ref_a))
        rows += compare_rows(staged_row(consumer))
        rows += compare_rows(swap_row(consumer))
        rows += compare_rows(prompt_row(consumer))
        print(f"  prompt changes the chunk: max_abs(task B - task A) = {prompt_effect:.4f}")
        finite = bool(np.isfinite(ref_a).all())
        ok = all(r["array_equal"] for r in rows) and finite and prompt_effect > 0

        if not args.no_mutants:
            print("mutants (the verb returns success and does nothing; the row must fail):")
            cases = [("images verb no-op", python_verb_noop("images"), staged_row),
                     ("proprio verb no-op", python_verb_noop("proprio"), staged_row),
                     ("proprio verb no-op, SWAP image path", python_verb_noop("proprio"), swap_row),
                     ("prompt verb no-op", python_verb_noop("prompt"), prompt_row),
                     ("step no-op", python_step_noop(), staged_row)]
            for label, patch, row_fn in cases:
                with patch:
                    mr_m = fe.export_model_runtime()
                c_m = ModelRuntimeConsumer(mr_m.ptr, exec_library_path())
                try:
                    triples = row_fn(c_m)
                finally:
                    c_m.close()
                    mr_m.release()
                    restore_prompt_a()
                detected = not all(np.array_equal(a, b) for _, a, b in triples)
                mutants.append({"mutant": label, "detected": detected})
                print(f"  {label:<56} detected={detected}")
            ok = ok and all(m["detected"] for m in mutants)

        if args.bench_iters > 0:
            def tick_python() -> None:
                fe.infer(obs, action_noise=noise)

            def tick_abi() -> None:
                consumer.set_input("images", views)
                consumer.set_input("proprio", state.tobytes())
                consumer.write_swap("noise", noise.cpu().numpy())
                consumer.step()
                consumer.get_output("actions", np.float32, chunk)

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
                json.dump({"precision": args.precision, "observation": obs_label, "vae_in_graph": in_graph,
                           "rows": rows, "mutants": mutants}, f, indent=1)
        print("PASS" if ok else "FAIL", "- ImageWAM model runtime (io=python) vs infer()")
        return 0 if ok else 1
    finally:
        consumer.close()
        mr.release()


if __name__ == "__main__":
    raise SystemExit(main())
