"""Build ImageWAM's real activation-calibration file (`calibration_file.py`).

Runs the real fp16 pipeline (real checkpoint, VAE, Qwen3 prompt, proprio,
10-step shift schedule) eagerly on N real LIBERO observations and records
every GEMM input (`activation_recorder.py`): absmax, |x| percentiles and
per-channel absmax, max over the calls within a sample (all denoise steps
for ActionDiT sites). The house reducer (`accumulate_amax`, percentile
99.9 by default) combines samples; the static FP8 scale is `amax / 448`.

Calibration frames: house stratified rule (episode x frame position) over
`--suites` (default: libero_object, libero_goal, libero_10), with every
episode of the evaluation set removed (default: the end-to-end harness's
libero_spatial first-episode-per-task frames). Initial action noise per
sample: N(0,1) from `torch.Generator("cpu").manual_seed(i)`, rounded to
bf16 (the official sampler's noise, as in the end-to-end harness).

Required env: CKPT_PATH (dataset_stats.json beside it), FLUX2_AE_MODEL_PATH,
FLUX2_SRC, QWEN3_MODEL_SPEC, DATA_ROOT.

`--text-trim` records with the frontend's `text_trim=True` (the text
context trimmed to the prompt's valid tokens, issues.md ISSUE-020); the
file's identity then records `text_trim=True`, and only a
`text_trim=True` frontend loads it.

The file's identity (`calibration_file.py`, format version 3) is the
checkpoint, the frontend's dims and `text_trim`. The dims include the
workload's camera geometry (`num_views`, `image_h`, `image_w`), which
comes from `libero_dims.LIBERO_REAL_DIMS` through the frontend's `dims`;
the frames staged here are two `VIEW_HW x VIEW_HW` views, and the run
refuses to start if those dims name another geometry. Files of an earlier
format version are not read: re-record them with this script.

  python benchmarks/imagewam_build_calibration.py --out <file>.safetensors [--n 64] [--text-trim]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from flash_rt.core.calibration import format_summary, summarize_amax_dispersion
from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.activation_recorder import ActivationRecorder
from flash_rt.models.imagewam.calibration_file import (
    DEFAULT_PERCENTILE, build_calibration, save_calibration,
)
from flash_rt.models.imagewam.libero_dims import LIBERO_HORIZON, LIBERO_REAL_DIMS

from _imagewam_libero_frames import (  # benchmarks/ helper, on sys.path when run as a script
    CALIBRATION_SUITES, EVAL_SUITE, VIEW_HW, evaluation_frames, load_frame, select_calibration_frames,
)

DEV = "cuda"
BF16 = torch.bfloat16
NOISE_DESC = "N(0,1) per sample, torch.Generator('cpu').manual_seed(sample_index), bf16-rounded"


def sample_noise(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = torch.randn((1, LIBERO_HORIZON, 7), generator=g, dtype=torch.float32)
    return n.to(DEV, BF16).float()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    ap.add_argument("--suites", default=",".join(CALIBRATION_SUITES))
    ap.add_argument("--eval-suite", default=EVAL_SUITE)
    ap.add_argument("--eval-tasks", type=int, default=10)
    ap.add_argument("--eval-frames", default="0,60")
    ap.add_argument("--text-trim", action="store_true",
                    help="record with the text context trimmed to the prompt's valid tokens "
                         "(frontend text_trim=True); the file then serves text_trim=True frontends only")
    args = ap.parse_args()

    data_root = os.environ["DATA_ROOT"]
    ckpt = os.environ["CKPT_PATH"]
    suites = tuple(s for s in args.suites.split(",") if s)
    eval_refs = evaluation_frames(data_root, suite=args.eval_suite, n_tasks=args.eval_tasks,
                                  frames=tuple(int(x) for x in args.eval_frames.split(",")))
    refs = select_calibration_frames(data_root, suites=suites, n=args.n, exclude=eval_refs)
    overlap = {(r.suite, r.episode) for r in refs} & {(r.suite, r.episode) for r in eval_refs}
    if overlap:
        raise RuntimeError(f"calibration and evaluation share episodes: {sorted(overlap)[:5]}")
    by_suite = {s: sum(r.suite == s for r in refs) for s in suites}
    print(f"calibration frames: {len(refs)} from {len({(r.suite, r.episode) for r in refs})} "
          f"episodes {by_suite}; evaluation frames excluded: {len(eval_refs)} ({args.eval_suite})",
          flush=True)

    t0 = time.time()
    fe = ImageWAMTorchFrontendThor(
        precision="fp16", dims_override=dict(LIBERO_REAL_DIMS), ckpt_path=ckpt,
        ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
        qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
        dataset_stats_path=os.path.join(os.path.dirname(ckpt), "dataset_stats.json"),
        text_trim=args.text_trim)
    print(f"fp16 frontend (text_trim={args.text_trim}) constructed in {time.time() - t0:.1f}s", flush=True)
    # The identity recorded below is `fe.dims`; the frames staged below are
    # two VIEW_HW x VIEW_HW views. They must be the same geometry.
    staged = (2, VIEW_HW, VIEW_HW)
    named = tuple(fe.dims.get(k) for k in ("num_views", "image_h", "image_w"))
    if named != staged:
        raise RuntimeError(f"the frontend's dims name (num_views, image_h, image_w)={named} but this script "
                           f"stages {staged}; the calibration file would record the wrong workload identity")

    rec = ActivationRecorder()
    wrapped = rec.wrap(fe.weights)
    samples, tasks = [], set()
    t_run = time.time()
    with torch.no_grad():
        for i, ref in enumerate(refs):
            fr = load_frame(data_root, ref, LIBERO_HORIZON)
            tasks.add(fr.task)
            fe.set_prompt(fr.task)
            fe.stage_inputs({"view1": torch.from_numpy(fr.view1), "view2": torch.from_numpy(fr.view2),
                             "proprio": fr.state}, noise=sample_noise(i))
            rec.begin_sample()
            fe.run_eager(wrapped)
            s = rec.end_sample()
            samples.append(s)
            print(f"[{i + 1}/{len(refs)}] {ref.suite} ep{ref.episode} fr{ref.frame} sites={len(s.sites)} "
                  f"linear1[0].amax={s.sites['backbone.single.0.linear1.weight'].absmax:.2f} "
                  f"act_linear1[0].amax={s.sites['action_dit.single.0.linear1.weight'].absmax:.2f}",
                  flush=True)
    per_sample_s = (time.time() - t_run) / len(refs)
    print(f"recorded {len(refs)} samples, {len(tasks)} distinct prompts, {per_sample_s:.2f}s/sample",
          flush=True)

    cal = build_calibration(samples, percentile=args.percentile, checkpoint_path=ckpt,
                            dims=fe.dims, frames=[(r.suite, r.episode, r.frame) for r in refs],
                            noise=NOISE_DESC, text_trim=args.text_trim)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_calibration(cal, args.out)
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB): {len(cal.sites)} sites, "
          f"checkpoint {cal.checkpoint_id}, percentile {cal.percentile}, text_trim {cal.text_trim}")

    names = sorted(cal.sites)
    per_sample = [np.array([s.sites[n].absmax for n in names], dtype=np.float32) for s in samples]
    final = np.array([cal.sites[n].amax for n in names], dtype=np.float32)
    print(format_summary(summarize_amax_dispersion(per_sample, final)))
    print(f"{'site group':44s} {'n':>4s} {'amax min':>9s} {'median':>9s} {'max':>9s} "
          f"{'p99.99/amax':>11s}")
    groups: dict[str, list[str]] = {}
    for n in names:
        parts = n.split(".")
        groups.setdefault(".".join(parts[:2] + parts[3:]), []).append(n)
    for g, members in sorted(groups.items()):
        a = np.array([cal.sites[n].amax for n in members])
        r = np.array([cal.sites[n].abs_percentiles[2] / max(cal.sites[n].amax, 1e-12) for n in members])
        print(f"{g:44s} {len(members):4d} {a.min():9.3f} {np.median(a):9.3f} {a.max():9.3f} "
              f"{np.median(r):11.4f}")


if __name__ == "__main__":
    main()
