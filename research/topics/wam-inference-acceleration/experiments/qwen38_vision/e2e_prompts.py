"""The prompt set shared by hf_reference.py and e2e_compare.py: chat messages and the processor pixel cap per prompt.

Text-only prompts measure the NVFP4-vs-FP8 floor: three short questions, and three summarization prompts whose
lengths match the image prompts (about 280, 550 and 1580 tokens; excerpts of docs/qwen36_nvfp4.md cut at a token
count). Image prompts use LIBERO frames (two suites), both views of one frame together, FlashRT.png at full resolution
and a chart capped at 262144 pixels.
"""
import os
import sys
from dataclasses import dataclass

from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from _imagewam_libero_frames import _episode_video, _suite_root, read_video_frame  # noqa: E402

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


@dataclass
class E2EPrompt:
    name: str
    kind: str                 # "text" or "image"
    messages: list
    max_pixels: int | None    # processor pixel cap; None keeps the checkpoint default


def _frame(data_root: str, suite: str, key: str, episode: int, idx: int) -> Image.Image:
    return Image.fromarray(read_video_frame(_episode_video(_suite_root(data_root, suite), key, episode), idx))


def _user(*content) -> list:
    return [{"role": "user", "content": [{"type": "image", "image": c} if isinstance(c, Image.Image)
                                         else {"type": "text", "text": c} for c in content]}]


def _doc_excerpt(tokenizer, n_tokens: int) -> str:
    text = open(os.path.join(REPO, "docs", "qwen36_nvfp4.md"), encoding="utf-8").read()
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:n_tokens])


def build_prompts(data_root: str, tokenizer=None) -> list[E2EPrompt]:
    """All prompts; the length-matched text prompts need ``tokenizer`` and are left out without it."""
    agent = _frame(data_root, "libero_10", "observation.images.image", 0, 30)
    wrist = _frame(data_root, "libero_10", "observation.images.wrist_image", 0, 30)
    goal = _frame(data_root, "libero_goal", "observation.images.image", 0, 0)
    logo = Image.open(os.path.join(REPO, "FlashRT.png")).convert("RGB")
    chart = Image.open(os.path.join(REPO, "research/topics/wam-inference-acceleration/experiments/actiondit_representation/preview.png")).convert("RGB")
    return [
        E2EPrompt("text_quantum", "text", _user("Explain quantum entanglement in one short paragraph."), None),
        E2EPrompt("text_robot_plan", "text", _user("A robot must put a bowl in a drawer and close it. "
                                                   "List the steps as a JSON array of strings."), None),
        E2EPrompt("text_summary", "text", _user("Summarize in two sentences: CUDA graphs record a sequence of "
                                                "kernel launches once and replay it later, which removes the CPU "
                                                "launch overhead of each kernel."), None),
        *([E2EPrompt(f"text_doc_{n}", "text", _user("Summarize the following text in one sentence.\n\n"
                                                      + _doc_excerpt(tokenizer, n)), None)
           for n in (260, 525, 1560)] if tokenizer is not None else []),
        E2EPrompt("img_libero10_agent", "image", _user(agent, "Describe the scene. Which objects are on the table?"),
                  None),
        E2EPrompt("img_libero10_wrist", "image",
                  _user(wrist, "This is a robot's wrist camera. What is in front of the gripper?"), None),
        E2EPrompt("img_libero_goal_agent", "image", _user(goal, "List the objects you can see."), None),
        E2EPrompt("img_two_views", "image", _user(agent, wrist, "These are a robot's front camera and wrist camera. "
                                                               "What task is the robot most likely doing?"), None),
        E2EPrompt("img_logo_text", "image", _user(logo, "What text is written in this image?"), None),
        E2EPrompt("img_chart", "image", _user(chart, "What kind of figure is this, and what does it show?"), 262144),
    ]
