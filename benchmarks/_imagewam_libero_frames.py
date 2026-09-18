"""Real LIBERO frames for ImageWAM accuracy checks and calibration.

Reads the FastWAM-preprocessed LIBERO release (`yuanty/LIBERO-fastwam`,
LeRobot v2.1 layout under `DATA_ROOT`: `<suite>_no_noops_lerobot/{meta,
data,videos}`) with the preprocessing of
`imagewam_e2e_official_compare.py`: both cameras center-cropped and
resized to 224x224 with PIL bilinear (the official
`eval_libero_single._center_crop_resize`), the 8-D proprio state, and
the ground-truth action chunk starting at the frame.

Two frame sets, disjoint by construction:

- evaluation frames (`evaluation_frames`, `load_libero_frames`): the
  first episode of each task of one suite, at fixed frame indices (the
  end-to-end harness's rule);
- calibration frames (`select_calibration_frames`): the house stratified
  rule (`flash_rt.core.calibration.stratified_sample_indices`) over other
  suites, with every evaluation episode excluded.

A benchmark-side helper (needs `av`, `pandas`/`pyarrow`, `Pillow`), not
part of the `flash_rt` package.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import av
import numpy as np
import pandas as pd
from PIL import Image

from flash_rt.core.calibration import stratified_sample_indices

VIEW_HW = 224
EVAL_SUITE = "libero_spatial"
CALIBRATION_SUITES = ("libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class FrameRef:
    """One dataset frame: suite name (without `_no_noops_lerobot`),
    episode index within the suite, frame index within the episode."""
    suite: str
    episode: int
    frame: int


@dataclass
class LiberoFrame:
    """One observation: `view1`/`view2` uint8 `(224, 224, 3)` (agent view,
    wrist view), `state` float32 `(8,)` raw proprio, `gt` float32
    `(<=horizon, 7)` raw actions from this frame on."""
    suite: str
    episode: int
    frame: int
    task: str
    view1: np.ndarray
    view2: np.ndarray
    state: np.ndarray
    gt: np.ndarray


def _suite_root(data_root: str, suite: str) -> str:
    return os.path.join(data_root, f"{suite}_no_noops_lerobot")


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _episode_parquet(root: str, episode: int) -> str:
    return f"{root}/data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"


def _episode_video(root: str, key: str, episode: int) -> str:
    return f"{root}/videos/chunk-{episode // 1000:03d}/{key}/episode_{episode:06d}.mp4"


def center_crop_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Official `eval_libero_single._center_crop_resize` (PIL bilinear
    resize so the image covers `w x h`, then center crop)."""
    pil = Image.fromarray(img)
    sw, sh = pil.size
    scale = max(w / sw, h / sh)
    r = pil.resize((round(sw * scale), round(sh * scale)), resample=Image.BILINEAR)
    rw, rh = r.size
    left, top = (rw - w) // 2, (rh - h) // 2
    return np.array(r.crop((left, top, left + w, top + h)))


def read_video_frame(path: str, idx: int) -> np.ndarray:
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == idx:
                return f.to_ndarray(format="rgb24")
    raise IndexError(f"{path}: frame {idx} out of range")


def load_frame(data_root: str, ref: FrameRef, horizon: int) -> LiberoFrame:
    root = _suite_root(data_root, ref.suite)
    tasks = {t["task_index"]: t["task"] for t in _read_jsonl(f"{root}/meta/tasks.jsonl")}
    df = pd.read_parquet(_episode_parquet(root, ref.episode))
    if not 0 <= ref.frame < len(df):
        raise IndexError(f"{ref}: episode has {len(df)} frames")
    v1 = read_video_frame(_episode_video(root, "observation.images.image", ref.episode), ref.frame)
    v2 = read_video_frame(_episode_video(root, "observation.images.wrist_image", ref.episode), ref.frame)
    return LiberoFrame(
        suite=ref.suite, episode=ref.episode, frame=ref.frame,
        task=tasks[int(df["task_index"].iloc[0])],
        view1=center_crop_resize(v1, VIEW_HW, VIEW_HW), view2=center_crop_resize(v2, VIEW_HW, VIEW_HW),
        state=np.array(df["observation.state"].iloc[ref.frame], dtype=np.float32),
        gt=np.stack(df["action"].iloc[ref.frame:ref.frame + horizon].to_numpy()).astype(np.float32))


def evaluation_frames(data_root: str, *, suite: str = EVAL_SUITE, n_tasks: int = 10,
                      frames: tuple[int, ...] = (0, 60)) -> list[FrameRef]:
    """The end-to-end harness's evaluation rule: the first episode of
    each of the first `n_tasks` tasks of `suite` (episode order), at
    each index in `frames` that the episode reaches."""
    root = _suite_root(data_root, suite)
    out: list[FrameRef] = []
    seen: set[int] = set()
    for ep in _read_jsonl(f"{root}/meta/episodes.jsonl"):
        e = int(ep["episode_index"])
        df = pd.read_parquet(_episode_parquet(root, e), columns=["task_index"])
        ti = int(df["task_index"].iloc[0])
        if ti in seen:
            continue
        seen.add(ti)
        out += [FrameRef(suite, e, fr) for fr in frames if fr < len(df)]
        if len(seen) >= n_tasks:
            break
    return out


def load_libero_frames(data_root: str, suite: str, n_tasks: int, frames: list[int],
                       horizon: int = 64) -> list[LiberoFrame]:
    """`evaluation_frames`, loaded."""
    refs = evaluation_frames(data_root, suite=suite, n_tasks=n_tasks, frames=tuple(frames))
    return [load_frame(data_root, r, horizon) for r in refs]


def select_calibration_frames(data_root: str, *, suites: tuple[str, ...] = CALIBRATION_SUITES,
                              n: int = 64, exclude: list[FrameRef] | None = None) -> list[FrameRef]:
    """`n` calibration frames over `suites`: `n` split evenly across the
    suites (the first `n % len(suites)` suites take one extra), and each
    suite's share stratified by episode x frame position with the house
    rule (`stratified_sample_indices`). Every episode that holds any
    `exclude` frame is removed from the pool first, so calibration and
    evaluation never share an episode."""
    excluded_eps = {(r.suite, r.episode) for r in (exclude or [])}
    out: list[FrameRef] = []
    for s_idx, suite in enumerate(suites):
        share = n // len(suites) + (1 if s_idx < n % len(suites) else 0)
        if share == 0:
            continue
        rows = []
        root = _suite_root(data_root, suite)
        for ep in _read_jsonl(f"{root}/meta/episodes.jsonl"):
            e = int(ep["episode_index"])
            if (suite, e) in excluded_eps:
                continue
            rows += [(e, fr) for fr in range(int(ep["length"]))]
        df = pd.DataFrame(rows, columns=["episode_index", "frame_index"])
        df["index"] = np.arange(len(df))
        df["task_index"] = 0
        picks = stratified_sample_indices(df, n=share)
        sel = df.set_index("index").loc[picks]
        out += [FrameRef(suite, int(r.episode_index), int(r.frame_index)) for r in sel.itertuples()]
    return out
