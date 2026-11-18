"""Planner call time with thinking on and off, against context length (text-only prompts).

Each prompt is a synthetic robot log of S tokens followed by one planner-style question. The model decodes greedily
until EOS (or a token cap), on the long-context agent path with MTP. With thinking on the script records how many tokens
the model spends before </think> and how long that takes, then the answer. Every prompt runs twice in a row: the first
run records the CUDA graphs for its decode positions and is discarded, the second is timed.

usage (cwd anywhere; PLANNER_CKPT = NVFP4 checkpoint, PLANNER_MTP_DIR = MTP head directory):
  PLANNER_CKPT=DIR PLANNER_MTP_DIR=DIR python thinking_sweep.py {rtx|thor} OUT_JSON CAP LENGTH [LENGTH ...]
"""
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

CKPT = os.environ["PLANNER_CKPT"]
os.environ.update({
    "FLASHRT_QWEN36_MTP_CKPT_DIR": os.environ["PLANNER_MTP_DIR"],
    "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096",
    "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
    "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "1",
    "FLASHRT_QWEN36_TQ_PREFILL_GRAPH": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

import torch  # noqa: E402

from context_sweep import FRONTENDS, LOOKAHEAD_ROWS, MAX_IMAGE_TOKENS, MIN_MAX_SEQ, SPEC_K, log_line  # noqa: E402

QUESTIONS = {
    "next_subtask": ("Based on the robot log above, which subtask should the robot execute next? "
                     'Reply with JSON {"subtask": ..., "reason": ...}.'),
    "drawer_count": ("How many times did the robot start the subtask 'open the drawer' in the log above, and what was "
                     'the status the last time? Reply with JSON {"count": ..., "last_status": ...}.'),
    "replan_check": ("Which subtasks ended with status replan in the last 20 steps of the log above? "
                     'Reply with JSON {"subtasks": [...]}.'),
}


@dataclass
class Cell:
    question: str
    thinking: bool
    context_tokens: int
    prefill_ms: float
    think_tokens: int        # tokens before and including </think>; 0 with thinking off
    think_ms: float          # decode time until the chunk that contains </think>
    answer_tokens: int
    decode_ms: float         # whole decode until EOS or the cap
    hit_cap: bool
    tokens_per_attempt: float
    answer: str


class ThinkingSweep:
    def __init__(self, frontend: str, max_len: int, cap: int) -> None:
        self.cap = cap
        self.fe = FRONTENDS[frontend](CKPT, max_seq=max(MIN_MAX_SEQ, max_len + cap + LOOKAHEAD_ROWS + 64),
                                      max_image_tokens=MAX_IMAGE_TOKENS)
        self.llm = self.fe.llm
        self.tok = self.fe.processor.tokenizer
        self.log_ids = self.tok("".join(log_line(i) for i in range(9000)), add_special_tokens=False)["input_ids"]
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self.stop_ids = sorted({self.tok.convert_tokens_to_ids("<|im_end|>"), int(self.tok.eos_token_id)})

    def prompt_ids(self, target: int, question: str, thinking: bool) -> torch.Tensor:
        kw = {"enable_thinking": thinking}

        def build(n: int) -> list[int]:
            body = self.tok.decode(self.log_ids[:n]) + "\n" + question
            out = self.tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True,
                                               tokenize=True, **kw)
            return list(out["input_ids"] if hasattr(out, "keys") else out)
        overhead = len(build(0))
        ids = build(target - overhead)
        while len(ids) > target:
            n = target - overhead - (len(ids) - target)
            ids = build(n)
            overhead += 1
        assert abs(len(ids) - target) <= 8, (len(ids), target)
        return torch.tensor([ids], device=self.fe.device)

    @torch.inference_mode()
    def run(self, name: str, ids: torch.Tensor, thinking: bool) -> Cell:
        self.llm._spec_attempts = 0
        self.llm._spec_accepts = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.llm.prefill_long_ctx_nvfp4_agent(ids, max_new_tokens=self.cap, K=SPEC_K)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        toks: list[int] = []
        think_tokens, think_ms = 0, 0.0
        for chunk in self.llm.decode_long_ctx_nvfp4_committed_stream(max_new_tokens=self.cap, K=SPEC_K,
                                                                       stop_token_ids=self.stop_ids):
            toks.extend(chunk)
            if thinking and think_tokens == 0 and self.think_end in chunk:
                torch.cuda.synchronize()
                think_ms = (time.perf_counter() - t1) * 1e3
                think_tokens = toks.index(self.think_end) + 1
            if any(t in self.stop_ids for t in chunk):
                break
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if any(t in self.stop_ids for t in toks):
            toks = toks[: min(toks.index(t) for t in self.stop_ids if t in toks) + 1]
        hit_cap = not any(t in self.stop_ids for t in toks)
        att = int(getattr(self.llm, "_spec_attempts", 0))
        answer = self.tok.decode(toks[think_tokens:], skip_special_tokens=True).strip()
        return Cell(name, thinking, int(ids.shape[1]), (t1 - t0) * 1e3, think_tokens, think_ms,
                    len(toks) - think_tokens, (t2 - t1) * 1e3, hit_cap, len(toks) / max(1, att), answer[:300])


def main() -> None:
    frontend, out, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
    lengths = [int(x) for x in sys.argv[4:]]
    sw = ThinkingSweep(frontend, max(lengths), cap)
    cells: list[dict] = []
    for L in lengths:
        for thinking in (False, True):
            for name, q in QUESTIONS.items():
                ids = sw.prompt_ids(L, q, thinking)
                sw.run(name, ids, thinking)          # warm-up: records CUDA graphs for these decode positions
                c = sw.run(name, ids, thinking)
                cells.append(asdict(c))
                print(f"S={c.context_tokens:>6} think={'on ' if thinking else 'off'} {name:<13} prefill {c.prefill_ms:8.0f} ms"
                      f"  think {c.think_tokens:5d} tok {c.think_ms:9.0f} ms  answer {c.answer_tokens:4d} tok"
                      f"  decode {c.decode_ms:9.0f} ms{'  CAP' if c.hit_cap else ''}", flush=True)
                with open(out, "w") as f:
                    json.dump({"frontend": frontend, "ckpt": CKPT, "cap": cap, "spec_k": SPEC_K,
                               "torch": torch.__version__, "device": torch.cuda.get_device_name(), "cells": cells},
                              f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
