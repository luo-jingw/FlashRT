"""Qwen3.8-27B (distilled, NVFP4) as the robot's high-level planner on FlashRT: one call = two LIBERO camera frames
(front + wrist) + a planning prompt [+ a text history] -> NEW_TOKENS greedy tokens with MTP speculative decode.

Language model: `Qwen36VlTorchFrontendRtx` (NVFP4, FP8 KV, chunked prefill route for every prompt). Thinking is off.

Used three ways:
  library   `Planner(streams)` from colocation_bench.py (in-process modes); `streams` puts the planner's main, graph
            and snapshot streams on given streams (green-context partition or plain);
  worker    python planner_qwen27b.py worker HZ DURATION_S OUT_JSON   (HZ 0 = back-to-back; separate process / MPS);
  alone     python planner_qwen27b.py alone SM_COUNT N_CALLS OUT_JSON  (planner alone; SM_COUNT 0 = whole device,
            otherwise a green-context partition of at least SM_COUNT SMs)
Environment: PLANNER_CKPT (NVFP4 dir with model.visual), PLANNER_MTP_DIR, PLANNER_CTX (short | long),
PLANNER_NEW_TOKENS (default 64), PLANNER_LONG_TOKENS (target prompt length of the long context, default 8192).

Per call: wall-clock start / end, and the frontend's stage times (preprocess, ViT, embedding/RoPE inject, prefill
and decode, the last two CUDA-event timed); summarize.py derives the phase windows in wall time backwards from the end.
"""
import json
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ["FLASHRT_ROOT"]
CKPT = os.environ.get("PLANNER_CKPT", os.path.join(REPO, "tmp/checkpoints/qwen27-tmp-nvfp4"))
MTP_DIR = os.environ.get("PLANNER_MTP_DIR", os.path.join(REPO, "tmp/checkpoints/qwen38-27b-fp8-official-mtp"))
CTX = os.environ.get("PLANNER_CTX", "short")
NEW_TOKENS = int(os.environ.get("PLANNER_NEW_TOKENS", "64"))
LONG_TOKENS = int(os.environ.get("PLANNER_LONG_TOKENS", "8192"))
MAX_SEQ = LONG_TOKENS + 256
os.environ.update({
    "FLASHRT_QWEN36_MTP_CKPT_DIR": MTP_DIR,
    "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096",
    "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
    "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from _imagewam_libero_frames import _episode_video, _suite_root, read_video_frame  # noqa: E402

from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402

TASK = "put both the alphabet soup and the tomato sauce in the basket"
PROMPT = ("You are the high-level planner of a robot arm. The overall task is: " + TASK + ". The first image is the "
          "front camera, the second the wrist camera. Say which sub-tasks are done and give the next sub-task as JSON "
          "with keys \"done\" (list of strings) and \"next\" (one short imperative sentence).")
SUBTASKS = ["locate the alphabet soup", "move the gripper above the alphabet soup", "grasp the alphabet soup",
            "lift the alphabet soup", "move the alphabet soup above the basket", "release the alphabet soup",
            "locate the tomato sauce", "grasp the tomato sauce", "move the tomato sauce above the basket",
            "release the tomato sauce"]
CHAT_KWARGS = {"enable_thinking": False}


@dataclass
class PlannerStreams:
    main: torch.cuda.Stream
    graph: torch.cuda.Stream
    snap: torch.cuda.Stream


@dataclass
class CallRecord:
    t_start: float
    t_end: float
    preprocess_ms: float
    vit_ms: float
    inject_ms: float
    prefill_ms: float
    decode_ms: float
    prompt_tokens: int
    image_tokens: int
    new_tokens: int
    route: str
    spec_cycles: int
    spec_accepted: int
    output_crc: int   # crc32 of the generated token ids (same output <=> same crc)

    @property
    def call_ms(self) -> float:
        return (self.t_end - self.t_start) * 1e3


def libero_frames() -> tuple[Image.Image, Image.Image]:
    root = _suite_root(os.environ["DATA_ROOT"], "libero_10")
    return tuple(Image.fromarray(read_video_frame(_episode_video(root, key, 0), 30))
                 for key in ("observation.images.image", "observation.images.wrist_image"))


class Planner:
    def __init__(self, streams: PlannerStreams | None = None):
        self.streams = streams or PlannerStreams(torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream())
        # Construct on the default stream: the loader mixes kernels on the legacy stream 0 with copies on the current
        # stream, so a side current stream during loading races and corrupts the weights (garbage output).
        self.fe = Qwen36VlTorchFrontendRtx(CKPT, max_seq=MAX_SEQ, max_image_tokens=1024)
        torch.cuda.synchronize()
        llm = self.fe.llm
        llm._graph_stream = self.streams.graph  # every graph is captured lazily on first use, so on these streams
        llm._snap_stream = self.streams.snap
        self.front, self.wrist = libero_frames()
        self.last_text = ""

    def messages(self, ctx: str = CTX) -> list:
        text = PROMPT
        if ctx == "long":
            tok = self.fe.processor.tokenizer
            lines, n = [], 0
            while n < LONG_TOKENS - 700:  # two frames and the prompt take about 600 tokens
                i = len(lines)
                lines.append(f"step {i}: executed \"{SUBTASKS[i % len(SUBTASKS)]}\" for {1 + i % 7} s, "
                             f"gripper {'closed' if i % 3 else 'open'}, result: {'success' if i % 5 else 'retry'}.")
                n = len(tok.encode("\n".join(lines), add_special_tokens=False))
            text = "Execution history:\n" + "\n".join(lines) + "\n\n" + PROMPT
        return [{"role": "user", "content": [{"type": "image", "image": self.front},
                                             {"type": "image", "image": self.wrist},
                                             {"type": "text", "text": text}]}]

    @torch.inference_mode()
    def call(self, msgs: list) -> CallRecord:
        t0 = time.time()
        with torch.cuda.stream(self.streams.main):
            r = self.fe.generate(msgs, max_new_tokens=NEW_TOKENS, chat_template_kwargs=CHAT_KWARGS)
        t1 = time.time()
        self.last_text = r.text
        tm = r.timings_ms
        llm = self.fe.llm
        return CallRecord(t0, t1, tm["preprocess"], tm["vit"], tm["inject"], tm["prefill"], tm["decode"],
                          r.prompt_len, r.num_image_tokens, NEW_TOKENS, r.route, int(getattr(llm, "_spec_attempts", -1)),
                          int(getattr(llm, "_spec_accepts", -1)),
                          zlib.crc32(str(r.token_ids).encode()))


def worker(hz: float, duration: float, out: str) -> None:
    p = Planner()
    msgs = p.messages()
    for _ in range(3):
        p.call(msgs)
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
        recs.append(asdict(p.call(msgs)))
    json.dump({"hz": hz, "ctx": CTX, "records": recs, "last_text": p.last_text, "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9},
              open(out, "w"))


def alone(sm_count: int, n_calls: int, out: str) -> None:
    sys.path.insert(0, os.path.join(HERE, "..", "colocation_5090"))
    streams, green = None, None
    if sm_count > 0:
        from green_partition import GreenPartition
        green = GreenPartition(sm_count)
        streams = PlannerStreams(green.part.stream, green.new_stream(0), green.new_stream(0))
    p = Planner(streams)
    rows = {}
    for ctx in ("short", "long"):
        msgs = p.messages(ctx)
        for _ in range(2):
            first = p.call(msgs)
        recs = [p.call(msgs) for _ in range(n_calls)]
        med = {k: float(np.median([getattr(c, k) for c in recs]))
               for k in ("preprocess_ms", "vit_ms", "inject_ms", "prefill_ms", "decode_ms")}
        med["call_ms"] = float(np.median([c.call_ms for c in recs]))
        med["decode_tok_s"] = (NEW_TOKENS - 1) * 1e3 / med["decode_ms"]
        med["spec_cycles"] = float(np.median([c.spec_cycles for c in recs]))
        med["decode_ms_per_cycle"] = med["decode_ms"] / max(med["spec_cycles"], 1)
        rows[ctx] = {"prompt_tokens": first.prompt_tokens, "image_tokens": first.image_tokens, "route": first.route,
                     "median": med, "calls": [asdict(c) for c in recs]}
        print(ctx, sm_count, rows[ctx]["prompt_tokens"], {k: round(v, 1) for k, v in med.items()}, flush=True)
    text = p.fe.generate(p.messages("short"), max_new_tokens=NEW_TOKENS, chat_template_kwargs=CHAT_KWARGS).text
    json.dump({"sm_count_requested": sm_count, "sm_count": None if green is None else green.part.sm_count,
               "new_tokens": NEW_TOKENS, "allocated_GB": torch.cuda.memory_allocated() / 1e9,
               "peak_GB": torch.cuda.max_memory_allocated() / 1e9, "short_output_text": text, "rows": rows},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
    else:
        alone(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
