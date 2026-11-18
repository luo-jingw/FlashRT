"""Append episode results to a JSONL file, one line per episode."""
from __future__ import annotations

import json
from dataclasses import asdict

from records import EpisodeResult


def append_episode(path: str, result: EpisodeResult, setting: dict[str, object]) -> None:
    """`setting` names the cell this episode belongs to (harness, protocol, injected L, thinking, ...)."""
    with open(path, "a") as f:
        f.write(json.dumps({"setting": setting, "episode": asdict(result)}, ensure_ascii=False) + "\n")
