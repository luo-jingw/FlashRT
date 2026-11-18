"""Planner prefill and decode time against context length (text-only prompts).

For each context length S the script builds a chat prompt of S tokens (a synthetic episode log followed by a request for
a long summary), prefills it on the long-context agent path and decodes a fixed number of tokens with MTP speculative
decoding (no stop token, so every cell decodes the same count). One untimed warm-up per length records the CUDA graphs
for that length; the timed run follows. Prefill content does not change its cost; decode cost depends on context length
(KV read per step) and on MTP acceptance, which the output records.

usage (cwd anywhere; PLANNER_CKPT = NVFP4 checkpoint, PLANNER_MTP_DIR = MTP head directory):
  PLANNER_CKPT=DIR PLANNER_MTP_DIR=DIR python context_sweep.py {rtx|thor} OUT_JSON LENGTH [LENGTH ...]
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

from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_vl_thor import Qwen36VlTorchFrontendThor  # noqa: E402

FRONTENDS = {"rtx": Qwen36VlTorchFrontendRtx, "thor": Qwen36VlTorchFrontendThor}
CHAT_KWARGS = {"enable_thinking": False}
SPEC_K = 6             # the frontend's prompt-length table, as in monitor_call_bench.py
DECODE_TOKENS = 256    # fixed decode count per cell
WARMUP_DECODE = DECODE_TOKENS  # CUDA graphs are recorded lazily per decode position
LOOKAHEAD_ROWS = 32    # qwen36_vl_rtx._SPEC_LOOKAHEAD_ROWS
MAX_IMAGE_TOKENS = 64  # text-only prompts; the frontend still sizes an image-row table
QUESTION = ("Summarize the robot log above in detail. For every subtask, list the steps where it started and ended, "
            "the status changes, and the reason for each replan.")


@dataclass
class Cell:
    context_tokens: int
    prefill_ms: float
    decode_ms: float
    decode_tokens: int
    decode_ms_per_token: float
    spec_attempts: int
    spec_accepts: int
    tokens_per_attempt: float


def log_line(i: int) -> str:
    subtasks = ["pick up the red block", "place it in the bowl", "open the drawer", "move the can next to the pot",
                "press the bell", "lift the pot", "put the hammer on the block", "close the drawer"]
    statuses = ["continue", "continue", "continue", "next", "replan"]
    return (f"Step {i * 8}: subtask '{subtasks[i % len(subtasks)]}', gripper {'open' if i % 3 else 'closed'}, "
            f"object at ({(i * 37) % 100}, {(i * 53) % 100}), status {statuses[i % len(statuses)]}.\n")


class Sweep:
    def __init__(self, frontend: str, max_len: int) -> None:
        self.fe = FRONTENDS[frontend](CKPT, max_seq=max_len + DECODE_TOKENS + LOOKAHEAD_ROWS + 64,
                                      max_image_tokens=MAX_IMAGE_TOKENS)
        self.llm = self.fe.llm
        self.tok = self.fe.processor.tokenizer
        self.log_ids = self.tok("".join(log_line(i) for i in range(9000)), add_special_tokens=False)["input_ids"]

    def prompt_ids(self, target: int) -> torch.Tensor:
        """Chat prompt of exactly `target` tokens: a log prefix cut to fit, then the question."""
        def build(n: int) -> list[int]:
            body = self.tok.decode(self.log_ids[:n]) + "\n" + QUESTION
            out = self.tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True,
                                               tokenize=True, **CHAT_KWARGS)
            return list(out["input_ids"] if hasattr(out, "keys") else out)
        overhead = len(build(0))
        ids = build(target - overhead)
        while len(ids) > target:   # detokenize/retokenize can merge tokens; trim the log until it fits
            n = target - overhead - (len(ids) - target)
            ids = build(n)
            overhead += 1
        assert abs(len(ids) - target) <= 8, (len(ids), target)
        return torch.tensor([ids], device=self.fe.device)

    @torch.inference_mode()
    def run(self, ids: torch.Tensor, budget: int) -> Cell:
        self.llm._spec_attempts = 0
        self.llm._spec_accepts = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.llm.prefill_long_ctx_nvfp4_agent(ids, max_new_tokens=budget, K=SPEC_K)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        toks: list[int] = []
        for chunk in self.llm.decode_long_ctx_nvfp4_committed_stream(max_new_tokens=budget, K=SPEC_K):
            toks.extend(chunk)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        att = int(getattr(self.llm, "_spec_attempts", 0))
        acc = int(getattr(self.llm, "_spec_accepts", 0))
        dec = (t2 - t1) * 1e3
        return Cell(int(ids.shape[1]), (t1 - t0) * 1e3, dec, len(toks), dec / max(1, len(toks)), att, acc,
                    len(toks) / max(1, att))


def main() -> None:
    frontend, out = sys.argv[1], sys.argv[2]
    lengths = [int(x) for x in sys.argv[3:]]
    sw = Sweep(frontend, max(lengths))
    cells: list[dict] = []
    for L in lengths:
        ids = sw.prompt_ids(L)
        sw.run(ids, WARMUP_DECODE)
        c = sw.run(ids, DECODE_TOKENS)
        cells.append(asdict(c))
        print(f"S={c.context_tokens:>7}  prefill {c.prefill_ms:>10.1f} ms  decode {c.decode_ms_per_token:6.2f} ms/token"
              f"  ({c.decode_tokens} tokens, {c.tokens_per_attempt:.2f} per attempt)", flush=True)
        with open(out, "w") as f:
            json.dump({"frontend": frontend, "ckpt": CKPT, "decode_tokens": DECODE_TOKENS, "spec_k": SPEC_K,
                       "torch": torch.__version__, "device": torch.cuda.get_device_name(), "cells": cells}, f, indent=1)


if __name__ == "__main__":
    main()
