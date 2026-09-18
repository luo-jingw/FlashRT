"""Real LIBERO frames for ImageWAM accuracy checks.

Same selection and preprocessing as `imagewam_e2e_official_compare.py`:
the first episode of each task in a LIBERO-fastwam suite (LeRobot v2.1
layout under `DATA_ROOT`), the requested frame indices, both cameras
center-cropped and resized to 224x224 with PIL bilinear (the official
`eval_libero_single._center_crop_resize`), the 8-D proprio state, and
the ground-truth action chunk starting at that frame.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import av
import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class LiberoFrame:
    """One observation: `view1`/`view2` uint8 `(224, 224, 3)`, `state`
    float32 `(8,)`, `gt` float32 `(<=horizon, 7)`."""

    episode: int
    frame: int
    task: str
    view1: np.ndarray
    view2: np.ndarray
    state: np.ndarray
    gt: np.ndarray


def center_crop_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Official `eval_libero_single._center_crop_resize` (PIL bilinear + center crop)."""
    pil = Image.fromarray(img)
    sw, sh = pil.size
    scale = max(w / sw, h / sh)
    r = pil.resize((round(sw * scale), round(sh * scale)), resample=Image.BILINEAR)
    rw, rh = r.size
    left, top = (rw - w) // 2, (rh - h) // 2
    return np.asarray(r.crop((left, top, left + w, top + h)))


def read_video_frame(path: str, idx: int) -> np.ndarray:
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == idx:
                return f.to_ndarray(format="rgb24")
    raise IndexError(f"{path}: frame {idx} out of range")


def load_libero_frames(data_root: str, suite: str, n_tasks: int, frames: list[int],
                       horizon: int = 64) -> list[LiberoFrame]:
    root = os.path.join(data_root, f"{suite}_no_noops_lerobot")
    with open(f"{root}/meta/tasks.jsonl") as fh:
        tasks = {json.loads(line)["task_index"]: json.loads(line)["task"] for line in fh}
    with open(f"{root}/meta/episodes.jsonl") as fh:
        episodes = [json.loads(line) for line in fh]
    seen: set[int] = set()
    out: list[LiberoFrame] = []
    for ep in episodes:
        e = ep["episode_index"]
        df = pd.read_parquet(f"{root}/data/chunk-000/episode_{e:06d}.parquet")
        ti = int(df["task_index"].iloc[0])
        if ti in seen:
            continue
        seen.add(ti)
        for fr in frames:
            if fr >= len(df):
                continue
            v1 = read_video_frame(f"{root}/videos/chunk-000/observation.images.image/episode_{e:06d}.mp4", fr)
            v2 = read_video_frame(f"{root}/videos/chunk-000/observation.images.wrist_image/episode_{e:06d}.mp4", fr)
            out.append(LiberoFrame(
                episode=e, frame=fr, task=tasks[ti],
                view1=center_crop_resize(v1, 224, 224), view2=center_crop_resize(v2, 224, 224),
                state=np.asarray(df["observation.state"].iloc[fr], dtype=np.float32),
                gt=np.stack(df["action"].iloc[fr:fr + horizon].to_numpy()).astype(np.float32)))
        if len(seen) >= n_tasks:
            break
    return out
