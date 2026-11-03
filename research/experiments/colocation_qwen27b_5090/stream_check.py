"""Planner output correctness by configuration (greedy, PLANNER_NEW_TOKENS tokens, thinking off):
mode "override": planner_qwen27b.Planner (main / graph / snapshot streams replaced by side streams) -- short, long,
short again; mode "plain": the frontend constructed on the default stream with its own streams -- short, long, short
again; mode "side": as "plain", but constructed while a side stream is current (ISSUE-104).
"plain" and "side" also record a digest of the language model's weight buffers (`llm._weights.anchors`): per buffer
the sum of its bytes, and the SHA-1 of that list.

usage: python stream_check.py MODE OUT_JSON
"""
import hashlib
import json
import sys

import torch

import planner_qwen27b as pq
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx


def byte_sum(t: torch.Tensor) -> int:
    b = t.detach().contiguous().view(-1).view(torch.uint8)
    return sum(int(c.sum(dtype=torch.int64)) for c in b.split(1 << 26))  # 64 Mi-byte chunks: int64 temp <= 512 MiB


def weight_digest(fe: Qwen36VlTorchFrontendRtx) -> dict:
    sums = [byte_sum(t) for t in fe.llm._weights.anchors]
    return {"n_buffers": len(sums), "byte_sums_sha1": hashlib.sha1(json.dumps(sums).encode()).hexdigest(),
            "byte_sums": sums}


mode, out = sys.argv[1], sys.argv[2]
res = {"mode": mode}
if mode == "override":
    p = pq.Planner()
else:
    class Bare(pq.Planner):
        def __init__(self):
            self.fe = Qwen36VlTorchFrontendRtx(pq.CKPT, max_seq=pq.MAX_SEQ, max_image_tokens=1024)
            self.front, self.wrist = pq.libero_frames()
    if mode == "side":
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            p = Bare()
        side.synchronize()
    else:
        p = Bare()
    torch.cuda.synchronize()
    res["weights"] = weight_digest(p.fe)
run = lambda ctx: p.fe.generate(p.messages(ctx), max_new_tokens=pq.NEW_TOKENS, chat_template_kwargs=pq.CHAT_KWARGS)
for i, ctx in enumerate(("short", "long", "short")):
    r = run(ctx)
    res[f"{i}_{ctx}"] = {"prompt_len": r.prompt_len, "route": r.route, "text": r.text}
print(json.dumps({k: (v if k != "weights" else {kk: vv for kk, vv in v.items() if kk != "byte_sums"})
                  for k, v in res.items()}, indent=1, ensure_ascii=False), flush=True)
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
