"""Master (VLM) calls of VoLoAgent's Full loop as the agent harness runs it, rebuilt for replay against the FlashRT
planner. System prompts come from `prompts.json` (extract_prompts.py, read from VoLoAgent's source); the user messages
follow VoLoAgent's builders:

  decompose  `SubgoalStrategy._decompose_and_setup` + `_build_image_message` (strategies/subgoal.py:206-225,
             strategies/subgoal_base.py:699-738): preamble text with the hard-coded labels, then front, exterior and
             wrist images; max_tokens 512.
  monitor    `VLMFailureHandler.step` + `_build_check_message` (failure_handlers/vlm.py:451-480,
             strategies/subgoal_base.py:741-819): BEFORE and NOW front / exterior / wrist, each after its label, then
             the task text; max_tokens 512.
  replan     `_recycle` (strategies/subgoal_base.py:3493-3544): BEFORE and NOW front only, then the task text with the
             episode's original subgoal list; max_tokens 1024.

Images: the harness sends full-resolution JPEG (quality 90): front 864x480, exterior 1280x720, wrist 224x224 or
1280x720 (not determinable from the harness code; both are replayed). Here the frames come from a LIBERO-10 episode
video (agentview for front and exterior, wrist camera for wrist, 512x512 at 20 fps), resized to the target height and
centered on a black canvas of the target size, then JPEG-encoded at quality 90 as the harness does.
"""
import io
import json
import os
from dataclasses import dataclass
from typing import Literal

import av
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = json.load(open(os.path.join(HERE, "prompts.json")))
EPISODE_STEPS = 800          # env steps per replayed episode (15 Hz: 53.3 s)
CHECK_INTERVAL = 80          # env steps between monitoring calls
CHUNK = 8                    # env steps per executor query
MAX_REPLANS = 3              # strategies/subgoal_base.py:56
FRONT_WH = (864, 480)
EXTERIOR_WH = (1280, 720)
WRIST_WH = {224: (224, 224), 720: (1280, 720)}
CAMERA_LABELS = ("Front camera", ["External camera", "Wrist camera"])   # strategies/base.py:293-309


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    path: str            # JPEG file of the image as sent


@dataclass
class CallSpec:
    kind: Literal["decompose", "monitor", "replan"]
    step: int
    system: str
    user_parts: list[TextPart | ImagePart]
    max_tokens: int


@dataclass
class HarnessFrames:
    front: str           # JPEG paths
    exterior: str
    wrist: str


def _letterbox(img: Image.Image, wh: tuple[int, int]) -> Image.Image:
    w, h = wh
    side = min(w, h)
    canvas = Image.new("RGB", (w, h))
    canvas.paste(img.resize((side, side), Image.BICUBIC), ((w - side) // 2, (h - side) // 2))
    return canvas


def _jpeg(img: Image.Image, path: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return path


class LiberoEpisode:
    """Frames of one LIBERO-10 episode (lerobot videos), mapped onto EPISODE_STEPS env steps."""

    def __init__(self, data_root: str, episode: int):
        root = os.path.join(data_root, "libero_10_no_noops_lerobot")
        meta = [json.loads(line) for line in open(os.path.join(root, "meta", "episodes.jsonl"))]
        self.task: str = meta[episode]["tasks"][0]
        self.frames = {key: self._decode(f"{root}/videos/chunk-{episode // 1000:03d}/{key}/episode_{episode:06d}.mp4")
                       for key in ("observation.images.image", "observation.images.wrist_image")}

    @staticmethod
    def _decode(path: str) -> list[Image.Image]:
        with av.open(path) as c:
            return [f.to_image() for f in c.decode(video=0)]

    def harness_frames(self, step: int, wrist: int, out_dir: str) -> HarnessFrames:
        agent, wr = self.frames["observation.images.image"], self.frames["observation.images.wrist_image"]
        i = round(step * (len(agent) - 1) / EPISODE_STEPS)
        tag = f"s{step:04d}"
        return HarnessFrames(
            front=_jpeg(_letterbox(agent[i], FRONT_WH), os.path.join(out_dir, f"{tag}_front.jpg")),
            exterior=_jpeg(_letterbox(agent[i], EXTERIOR_WH), os.path.join(out_dir, f"{tag}_exterior.jpg")),
            wrist=_jpeg(_letterbox(wr[i], WRIST_WH[wrist]), os.path.join(out_dir, f"{tag}_wrist{wrist}.jpg")))


def decompose_call(task: str, f: HarnessFrames) -> CallSpec:
    preamble = ("You have 3 camera views.\nImage 1: External camera (full scene).\n"
                "Image 2: Wrist camera (close-up near gripper).\nImage 3: Wrist camera (close-up near gripper).\n"
                "\nUse the external camera as the primary view. The wrist camera shows a partial close-up.\n\n")
    return CallSpec("decompose", 0, PROMPTS["decompose"]["text"],
                    [TextPart(preamble + f'Scene instruction: "{task}"'),
                     ImagePart(f.front), ImagePart(f.exterior), ImagePart(f.wrist)], 512)


def monitor_call(task: str, subgoals: list[str], idx: int, before: HarnessFrames, now: HarnessFrames,
                 step: int) -> CallSpec:
    primary, extra = CAMERA_LABELS
    remaining = subgoals[idx + 1:]
    remaining_str = ("Remaining subgoals after this:\n" + "\n".join(
        f"  {i}. {s}" for i, s in enumerate(remaining, start=idx + 2)) if remaining else "This is the LAST subgoal.")
    text = (f'Overall task: "{task}"\n'
            f'Current subgoal ({idx + 1}/{len(subgoals)}): "{subgoals[idx]}"\n'
            f"{remaining_str}\n\n"
            "Assess the current subgoal status and decide the action.")
    parts: list[TextPart | ImagePart] = [
        TextPart(f"BEFORE — {primary} (scene at episode start):"), ImagePart(before.front),
        TextPart(f"BEFORE — {extra[0]}:"), ImagePart(before.exterior),
        TextPart(f"BEFORE — {extra[1]}:"), ImagePart(before.wrist),
        TextPart(f"NOW — {primary} (current scene):"), ImagePart(now.front),
        TextPart(f"NOW — {extra[0]}:"), ImagePart(now.exterior),
        TextPart(f"NOW — {extra[1]}:"), ImagePart(now.wrist),
        TextPart(text)]
    return CallSpec("monitor", step, PROMPTS["monitor"]["text"], parts, 512)


def replan_call(task: str, original: list[str], before: HarnessFrames, now: HarnessFrames, step: int) -> CallSpec:
    primary, _ = CAMERA_LABELS
    numbered = "\n".join(f"  {i + 1}. {sg}" for i, sg in enumerate(original))
    text = (f'Task: "{task}"\n\nImage 1 is BEFORE (episode start). Image 2 is NOW (current).\n'
            f"\nORIGINAL SUBGOAL LIST (select ONLY from these, using the exact wording):\n{numbered}\n"
            "\nList ALL remaining subgoals to complete the task from the current state.")
    return CallSpec("replan", step, PROMPTS["replan"]["text"],
                    [TextPart(f"BEFORE — {primary} (scene at episode start):"), ImagePart(before.front),
                     TextPart(f"NOW — {primary} (current scene):"), ImagePart(now.front), TextPart(text)], 1024)


def to_messages(call: CallSpec) -> list[dict]:
    """The call as a chat for `Qwen36VlTorchFrontendRtx.generate` (system message, then one user message whose parts
    keep the harness's order)."""
    content = [{"type": "text", "text": p.text} if isinstance(p, TextPart)
               else {"type": "image", "image": Image.open(p.path).convert("RGB")} for p in call.user_parts]
    return [{"role": "system", "content": call.system}, {"role": "user", "content": content}]


def call_to_row(call: CallSpec) -> dict:
    return {"kind": call.kind, "step": call.step, "max_tokens": call.max_tokens, "system": call.system,
            "user_parts": [{"text": p.text} if isinstance(p, TextPart) else {"image": p.path} for p in call.user_parts]}


def call_from_row(row: dict) -> CallSpec:
    parts = [TextPart(p["text"]) if "text" in p else ImagePart(p["image"]) for p in row["user_parts"]]
    return CallSpec(row["kind"], row["step"], row["system"], parts, row["max_tokens"])
