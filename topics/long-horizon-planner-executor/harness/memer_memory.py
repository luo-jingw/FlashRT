"""MemER-style keyframe memory (MemER, arXiv 2510.20328, §3.2): every keyframe the planner nominates becomes a candidate
frame index; candidates are grouped by 1-D single-linkage clustering (neighbours closer than `merge_s` seconds join a
cluster), each cluster is represented by its lower median, and the latest `max_frames` representatives are the memory.
Written from the paper's description; no code is copied from the MemER repository."""
from __future__ import annotations


class KeyframeMemory:
    def __init__(self, merge_s: float = 2.5, max_frames: int = 8) -> None:
        self.merge_s, self.max_frames = merge_s, max_frames
        self._candidates: list[tuple[float, int]] = []      # (time, history index); duplicates are kept as votes

    def reset(self) -> None:
        self._candidates.clear()

    def add(self, nominated: list[tuple[float, int]]) -> None:
        self._candidates.extend(nominated)

    def frames(self, exclude_from_t: float | None = None) -> list[int]:
        """History indices of the memory frames, in time order; frames at or after `exclude_from_t` (the recent
        window, which the planner sees anyway) are left out."""
        pts = sorted(c for c in self._candidates if exclude_from_t is None or c[0] < exclude_from_t)
        clusters: list[list[tuple[float, int]]] = []
        for p in pts:
            if clusters and p[0] - clusters[-1][-1][0] <= self.merge_s:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        reps = [c[(len(c) - 1) // 2][1] for c in clusters]  # lower median
        return reps[-self.max_frames:]
