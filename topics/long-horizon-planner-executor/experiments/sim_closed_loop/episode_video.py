"""Annotated episode video: agentview and wrist frames side by side, a text band with the current subgoal and the last
planner event; while the planner runs (the synchronous protocol stops the robot), the last frame is held for the call's
wall time, so the video's time axis is the episode's wall time at the control rate.
"""
import textwrap
from dataclasses import dataclass, field

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf")
BAND_H = 96
INK, MUTED, ACCENT, WAIT = (29, 26, 23), (110, 102, 94), (156, 61, 23), (235, 104, 52)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class EpisodeVideo:
    path: str
    fps: int
    frames: int = 0
    _container: av.container.OutputContainer | None = field(default=None, repr=False)
    _stream: av.video.stream.VideoStream | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._container = av.open(self.path, mode="w")
        self._stream = self._container.add_stream("libx264", rate=self.fps)
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": "23"}
        self._font, self._small = _font(15), _font(13)
        self._last: Image.Image | None = None

    def _compose(self, agent: np.ndarray, wrist: np.ndarray, step: int, subgoal: str, idx: int, n: int,
                 event: str, waiting: bool) -> Image.Image:
        h, w = agent.shape[:2]
        img = Image.new("RGB", (2 * w, h + BAND_H), (250, 248, 245))
        img.paste(Image.fromarray(agent), (0, 0))
        img.paste(Image.fromarray(wrist), (w, 0))
        d = ImageDraw.Draw(img)
        d.text((8, h + 6), f"step {step}   subgoal {idx + 1}/{n}", font=self._small, fill=MUTED)
        for k, line in enumerate(textwrap.wrap(subgoal, 60)[:2]):
            d.text((8, h + 24 + 18 * k), line, font=self._font, fill=INK)
        d.text((8, h + 70), event[:80], font=self._small, fill=WAIT if waiting else ACCENT)
        if waiting:
            d.rectangle((0, 0, 2 * w - 1, h - 1), outline=WAIT, width=4)
        return img

    def _write(self, img: Image.Image) -> None:
        frame = av.VideoFrame.from_image(img)
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames += 1

    def step(self, agent: np.ndarray, wrist: np.ndarray, step: int, subgoal: str, idx: int, n: int,
             event: str) -> None:
        img = self._compose(agent, wrist, step, subgoal, idx, n, event, waiting=False)
        self._last = img
        self._write(img)

    def hold(self, agent: np.ndarray, wrist: np.ndarray, step: int, subgoal: str, idx: int, n: int, event: str,
             seconds: float) -> None:
        """The robot stands still for `seconds` (planner call or instruction switch)."""
        img = self._compose(agent, wrist, step, subgoal, idx, n, event, waiting=True)
        for _ in range(max(1, round(seconds * self.fps))):
            self._write(img)

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
