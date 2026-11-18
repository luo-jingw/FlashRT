"""MemER high level as a `Planner`: each call sees the latest RECENT frames (sampled every `recent_stride_s`) and the
keyframe memory, names the current sub-task, and nominates keyframes that join the memory. A reply that cannot be
parsed, or names a sub-task outside the vocabulary, keeps the current instruction (status "keep")."""
from __future__ import annotations

from typing import Callable

from memer_memory import KeyframeMemory
from memer_prompt import build_messages, parse_reply, stack_cameras
from qwen_backend import QwenVlBackend
from records import PlannerResult, PlannerSnapshot


class MemerPlanner:
    def __init__(self, backend: QwenVlBackend, vocab_for_task: Callable[[str], list[str]], cameras: list[str],
                 n_recent: int = 8, recent_stride_s: float = 0.5, lead_s: float = 0.5, thinking: bool = False,
                 max_new_tokens: int = 256, memory: KeyframeMemory | None = None) -> None:
        self.backend, self.vocab_for_task, self.cameras = backend, vocab_for_task, cameras
        self.n_recent, self.stride, self.lead_s = n_recent, recent_stride_s, lead_s
        self.thinking, self.max_new_tokens = thinking, max_new_tokens
        self.memory = memory or KeyframeMemory()
        self.task, self.vocab = "", []

    def reset(self, task: str) -> None:
        self.task, self.vocab = task, self.vocab_for_task(task)
        self.memory.reset()

    def _recent(self, snapshot: PlannerSnapshot) -> list[int]:
        """History indices of the RECENT frames: the latest frame, then one every `stride` seconds back."""
        hist = snapshot.history
        picked: list[int] = []
        target = hist[-1].t
        for i in range(len(hist) - 1, -1, -1):
            if hist[i].t <= target + 1e-9:
                picked.append(i)
                target -= self.stride
                if len(picked) == self.n_recent:
                    break
        return picked[::-1]

    def decide(self, snapshot: PlannerSnapshot) -> PlannerResult:
        hist = snapshot.history
        recent = self._recent(snapshot)
        mem = self.memory.frames(exclude_from_t=hist[recent[0]].t)
        img = lambda i: (hist[i].t, stack_cameras(hist[i].images, self.cameras))  # noqa: E731
        msgs = build_messages(self.task, self.vocab, self.lead_s, [img(i) for i in mem], [img(i) for i in recent])
        g = self.backend.generate(msgs, thinking=self.thinking, max_new_tokens=self.max_new_tokens)
        parsed = parse_reply(g.answer, self.vocab, len(recent))
        self.memory.add([(hist[recent[p - 1]].t, recent[p - 1]) for p in parsed.keyframe_positions])
        return PlannerResult(parsed.subtask, "subtask" if parsed.subtask else "keep", g.answer, parsed.parse_ok,
                             g.prefill_ms + g.vit_ms + g.preprocess_ms, g.decode_ms, g.input_tokens, g.output_tokens)
