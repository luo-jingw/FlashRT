"""Planner output correctness by configuration (greedy, PLANNER_NEW_TOKENS tokens, thinking off):
mode "override": planner_qwen27b.Planner (main / graph / snapshot streams replaced by side streams) -- short, long,
short again; mode "plain": the frontend on the default stream with its own streams -- short, long, short again.

usage: python stream_check.py MODE OUT_JSON
"""
import json
import sys

import planner_qwen27b as pq
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx

mode, out = sys.argv[1], sys.argv[2]
if mode == "override":
    p = pq.Planner()
    run = lambda ctx: p.fe.generate(p.messages(ctx), max_new_tokens=pq.NEW_TOKENS, chat_template_kwargs=pq.CHAT_KWARGS)
else:
    class Bare(pq.Planner):
        def __init__(self):
            self.fe = Qwen36VlTorchFrontendRtx(pq.CKPT, max_seq=pq.MAX_SEQ, max_image_tokens=1024)
            self.front, self.wrist = pq.libero_frames()
    p = Bare()
    run = lambda ctx: p.fe.generate(p.messages(ctx), max_new_tokens=pq.NEW_TOKENS, chat_template_kwargs=pq.CHAT_KWARGS)
res = {}
for i, ctx in enumerate(("short", "long", "short")):
    r = run(ctx)
    res[f"{i}_{ctx}"] = {"prompt_len": r.prompt_len, "route": r.route, "text": r.text}
print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
