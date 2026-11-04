"""The FlashRT planner (zero-shot Qwen3.8-27B NVFP4, MTP, thinking off) replaying recorded master calls on given
streams. A replayed call decodes the recorded reply's length through EOS (`output_tokens + 1`), so it costs what a stop
at EOS would; its reply is compared with the recorded one.
"""
import os
import sys
import time
from dataclasses import dataclass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colocation_qwen27b_5090"))
import planner_qwen27b as pq  # noqa: E402  (planner environment: MTP head, long-context route, FP8 KV)

from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402

import volo_calls as vc  # noqa: E402
from record_calls import MAX_IMAGE_TOKENS, MAX_SEQ  # noqa: E402

PlannerStreams = pq.PlannerStreams


@dataclass
class CallRecord:
    kind: str
    step: int
    t_start: float
    t_end: float
    prompt_tokens: int
    image_tokens: int
    output_tokens: int
    same_reply: bool     # reply equals the recorded one
    graphs_captured: int  # planner graphs captured during the call (a new prompt length or decode position)
    preprocess_ms: float
    vit_ms: float
    inject_ms: float
    prefill_ms: float
    decode_ms: float

    @property
    def call_ms(self) -> float:
        return (self.t_end - self.t_start) * 1e3


class ReplayPlanner:
    def __init__(self, streams: PlannerStreams):
        self.streams = streams
        # construct on the default stream, then move the frontend's graph and snapshot streams
        self.fe = Qwen36VlTorchFrontendRtx(pq.CKPT, max_seq=MAX_SEQ, max_image_tokens=MAX_IMAGE_TOKENS)
        torch.cuda.synchronize()
        self.fe.llm._graph_stream = streams.graph
        self.fe.llm._snap_stream = streams.snap

    def graph_count(self) -> int:
        return sum(len(v) for k, v in vars(self.fe.llm).items() if k.startswith("_captured_") and isinstance(v, dict))

    @torch.inference_mode()
    def call(self, row: dict) -> CallRecord:
        msgs = vc.to_messages(vc.call_from_row(row))
        g0 = self.graph_count()
        t0 = time.time()
        with torch.cuda.stream(self.streams.main):
            r = self.fe.generate(msgs, max_new_tokens=min(row["output_tokens"] + 1, row["max_tokens"]),
                                 chat_template_kwargs=pq.CHAT_KWARGS)
        t1 = time.time()
        t = r.timings_ms
        return CallRecord(row["kind"], row["step"], t0, t1, r.prompt_len, r.num_image_tokens, len(r.token_ids),
                          r.text == row["output_text"], self.graph_count() - g0, t["preprocess"], t["vit"],
                          t["inject"], t["prefill"], t["decode"])
