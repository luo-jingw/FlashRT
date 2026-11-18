"""Zero-shot MemER prompt and reply parsing.

The fine-tuned MemER high level learns the sub-task vocabulary, what a keyframe is, the output format and to predict
slightly ahead; a zero-shot planner gets all four in the prompt (memer_port_study.md §5, "MemER 假定高层已微调的位置").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from PIL import Image

SYSTEM = (
    "You are the high-level planner of a robot. You see images from the robot's cameras and decide which sub-task "
    "the robot should execute now.\n"
    "Sub-tasks (answer with exactly one of these strings):\n{vocab}\n"
    "Images labelled MEMORY are keyframes kept from earlier in the episode: moments where a sub-task started or "
    "finished, or where something later needed was visible (an object being hidden, counted or moved). Images "
    "labelled RECENT are the most recent frames, oldest first, with their time in seconds.\n"
    "Your answer takes about {lead_s:.1f} s to reach the robot, so name the sub-task that should be running then.\n"
    "Reply with JSON only: {{\"current_subtask\": <one sub-task string>, \"keyframe_positions\": [<positions of "
    "RECENT images that are keyframes, 1-indexed, possibly empty>]}}"
)


@dataclass(frozen=True)
class ParsedReply:
    subtask: str | None            # None if missing or not in the vocabulary
    keyframe_positions: list[int]  # 1-indexed positions among the RECENT images
    parse_ok: bool


def stack_cameras(images: dict[str, object], cameras: list[str], width: int = 320, height: int = 180) -> Image.Image:
    """One image per time step: the listed cameras resized and stacked vertically (MemER uses two 320x180 views)."""
    tiles = [_to_pil(images[c]).resize((width, height)) for c in cameras]
    out = Image.new("RGB", (width, height * len(tiles)))
    for i, t in enumerate(tiles):
        out.paste(t, (0, i * height))
    return out


def _to_pil(x: object) -> Image.Image:
    return x if isinstance(x, Image.Image) else Image.fromarray(x).convert("RGB")  # type: ignore[arg-type]


def build_messages(task: str, vocab: list[str], lead_s: float, memory: list[tuple[float, Image.Image]],
                   recent: list[tuple[float, Image.Image]]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [{"type": "text", "text": f"Task: {task}\n"}]
    for t, img in memory:
        content += [{"type": "text", "text": f"MEMORY t={t:.1f}s:"}, {"type": "image", "image": img}]
    for i, (t, img) in enumerate(recent, 1):
        content += [{"type": "text", "text": f"RECENT {i} t={t:.1f}s:"}, {"type": "image", "image": img}]
    content.append({"type": "text", "text": "Which sub-task should the robot execute now?"})
    system = SYSTEM.format(vocab="\n".join(f"- {v}" for v in vocab), lead_s=lead_s)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_reply(text: str, vocab: list[str], n_recent: int) -> ParsedReply:
    m = _JSON_RE.search(text)
    if m is None:
        return ParsedReply(None, [], False)
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ParsedReply(None, [], False)
    sub = obj.get("current_subtask") if isinstance(obj, dict) else None
    pos = obj.get("keyframe_positions", []) if isinstance(obj, dict) else []
    if not isinstance(pos, list):
        return ParsedReply(None, [], False)
    try:
        positions = [int(p) for p in pos]
    except (TypeError, ValueError):
        return ParsedReply(None, [], False)
    positions = [p for p in positions if 1 <= p <= n_recent]
    return ParsedReply(sub if sub in vocab else None, positions, sub in vocab)
