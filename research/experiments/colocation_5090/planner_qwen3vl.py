"""Qwen3-VL-2B-Instruct as a small local planner: one call = one image + a short planning prompt -> NEW_TOKENS tokens.

Used three ways:
  as a library (Planner) by colocation_bench.py, in-process (serial / stream-priority modes);
  as a worker process:  python planner_qwen3vl.py worker HZ DURATION_S OUT_JSON   (HZ 0 = back-to-back);
  PLANNER_MODEL overrides the checkpoint directory (default $FLASHRT_ROOT/tmp/checkpoints/Qwen3-VL-2B-Instruct);
  PLANNER_CTX_EXTRA=N pads the prompt with about N extra text tokens (a long planner context);
  as a cost sweep (R2): python planner_qwen3vl.py sweep OUT_JSON
Every call records wall-clock start / end and the prefill (time to first token) and decode times, CUDA-event timed
on the planner's own stream.
"""
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList, Qwen3VLForConditionalGeneration

MODEL = os.environ.get("PLANNER_MODEL", os.path.join(os.environ["FLASHRT_ROOT"], "tmp/checkpoints/Qwen3-VL-2B-Instruct"))
NEW_TOKENS = 32
IMAGE_PX = 448
PROMPT = ("You are the high-level planner of a robot arm. The overall task is: put both the alphabet soup and the "
          "tomato sauce in the basket. Look at the image, say which sub-task is done, and give the next sub-task "
          "as one short imperative sentence.")


@dataclass
class CallRecord:
    t_start: float
    t_end: float
    prefill_ms: float
    decode_ms: float
    prompt_tokens: int


class Planner:
    def __init__(self, stream: torch.cuda.Stream | None = None):
        self.stream = stream or torch.cuda.Stream()
        self.proc = AutoProcessor.from_pretrained(MODEL)
        with torch.cuda.stream(self.stream):
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
        rng = np.random.default_rng(0)
        self.image = Image.fromarray(rng.integers(0, 255, (IMAGE_PX, IMAGE_PX, 3), dtype=np.uint8))

    def inputs(self, extra_text_tokens: int = int(os.environ.get("PLANNER_CTX_EXTRA", "0")), n_images: int = 1) -> dict:
        text = PROMPT + (" history:" + " step" * extra_text_tokens if extra_text_tokens else "")
        content = [{"type": "image", "image": self.image} for _ in range(n_images)] + [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        x = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt")
        return {k: v.cuda() for k, v in x.items()}

    @torch.inference_mode()
    def call(self, x: dict) -> CallRecord:
        t0 = time.time()
        with torch.cuda.stream(self.stream):
            e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            first = []

            class FirstToken(LogitsProcessor):  # runs once per generated token; the first call follows the prefill
                def __call__(self, input_ids, scores):
                    if not first:
                        e1.record(torch.cuda.current_stream())
                        first.append(1)
                    return scores

            e0.record(self.stream)
            self.model.generate(**x, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False,
                                logits_processor=LogitsProcessorList([FirstToken()]))
            e2.record(self.stream)
            e2.synchronize()
        return CallRecord(t0, time.time(), e0.elapsed_time(e1), e1.elapsed_time(e2), int(x["input_ids"].shape[1]))


def worker(hz: float, duration: float, out: str) -> None:
    p = Planner()
    x = p.inputs()
    for _ in range(3):
        p.call(x)
    print("READY", flush=True)
    sys.stdin.readline()  # parent says go
    recs, t0 = [], time.time()
    nxt = t0
    while time.time() - t0 < duration:
        if hz > 0:
            now = time.time()
            if now < nxt:
                time.sleep(nxt - now)
            nxt += 1.0 / hz
        recs.append(p.call(x).__dict__)
    json.dump({"hz": hz, "records": recs, "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}, open(out, "w"))


def sweep(out: str) -> None:
    p = Planner()
    rows = []
    for n_img in (1, 2, 4):
        for extra in (0, 1000, 4000, 16000, 32000):
            x = p.inputs(extra, n_img)
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                p.call(x)
            r = [p.call(x) for _ in range(5)]
            rows.append({"images": n_img, "prompt_tokens": r[0].prompt_tokens,
                         "prefill_ms": float(np.median([c.prefill_ms for c in r])),
                         "decode_ms_per_token": float(np.median([c.decode_ms for c in r])) / (NEW_TOKENS - 1),
                         "call_ms": float(np.median([(c.t_end - c.t_start) * 1e3 for c in r])),
                         "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9})
            print(rows[-1], flush=True)
    json.dump({"model": "Qwen3-VL-2B-Instruct bf16 (transformers eager loop)", "new_tokens": NEW_TOKENS,
               "image_px": IMAGE_PX, "gpu": torch.cuda.get_device_name(), "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    else:
        sweep(sys.argv[2])
