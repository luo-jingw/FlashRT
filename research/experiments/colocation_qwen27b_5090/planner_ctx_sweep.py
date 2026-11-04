"""Planner alone (whole device, no executor): call time by prompt length. Two LIBERO frames (512 image tokens) and the
planning prompt, with a text execution history prepended to reach each target length; PLANNER_NEW_TOKENS (64) greedy
tokens; prefill as in `planner_qwen27b.py alone` (the frontend's default eager chunked prefill).

Per target: 2 warm-up calls, then N_CALLS timed calls; medians of the frontend's stage times (preprocess, ViT, inject,
prefill, decode) and of the wall-clock call time.

usage: PLANNER_LONG_TOKENS=16384 python planner_ctx_sweep.py N_CALLS OUT_JSON TARGET [TARGET ...]
       (TARGET 0 = no history, the short context; PLANNER_LONG_TOKENS sets max_seq and must cover the largest target)
"""
import json
import sys
from dataclasses import asdict

import numpy as np
import torch

import planner_qwen27b as pq

STAGES = ("preprocess_ms", "vit_ms", "inject_ms", "prefill_ms", "decode_ms")


def main(n_calls: int, out: str, targets: list[int]) -> None:
    p = pq.Planner()
    rows = []
    for target in targets:
        msgs = p.messages("short") if target == 0 else p.messages("long", long_tokens=target)
        for _ in range(2):
            p.call(msgs)
        recs = [p.call(msgs) for _ in range(n_calls)]
        med = {k: float(np.median([getattr(c, k) for c in recs])) for k in STAGES}
        med["call_ms"] = float(np.median([c.call_ms for c in recs]))
        row = {"target": target, "prompt_tokens": recs[0].prompt_tokens, "image_tokens": recs[0].image_tokens,
               "new_tokens": pq.NEW_TOKENS, "route": recs[0].route, "median": med,
               "output_crcs": sorted({c.output_crc for c in recs}), "calls": [asdict(c) for c in recs]}
        rows.append(row)
        print(row["prompt_tokens"], {k: round(v, 1) for k, v in med.items()}, flush=True)
    json.dump({"n_calls": n_calls, "max_seq": pq.MAX_SEQ, "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
               "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], [int(t) for t in sys.argv[3:]])
