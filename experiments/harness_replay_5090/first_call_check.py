"""Planner output of the first call at a new prompt length against later calls of the same messages, in a fresh process
(greedy, thinking off). The recorded calls show the first call at each new prompt length giving a different, broken
reply; this runs one recorded call N times and prints each output and its timings.

usage: FLASHRT_QWEN36_TQ_PREFILL_GRAPH={0,1} python first_call_check.py CALLS_JSONL ROW_INDEX N MAX_NEW OUT_JSON
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colocation_qwen27b_5090"))
import planner_qwen27b as pq  # noqa: E402

from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402

import volo_calls as vc  # noqa: E402
from record_calls import MAX_IMAGE_TOKENS, MAX_SEQ  # noqa: E402


def main(calls: str, index: int, n: int, max_new: int, out: str) -> None:
    row = [json.loads(line) for line in open(calls)][index]
    msgs = vc.to_messages(vc.call_from_row(row))
    fe = Qwen36VlTorchFrontendRtx(pq.CKPT, max_seq=MAX_SEQ, max_image_tokens=MAX_IMAGE_TOKENS)
    torch.cuda.synchronize()
    runs = []
    for i in range(n):
        r = fe.generate(msgs, max_new_tokens=max_new, chat_template_kwargs=pq.CHAT_KWARGS)
        runs.append({"text": r.text, "tokens": len(r.token_ids), "prompt_len": r.prompt_len, "timings_ms": r.timings_ms,
                     "n_graphs": len(getattr(fe.llm, "_graphs", {}) or {})})
        print(i, r.prompt_len, len(r.token_ids), {k: round(v, 1) for k, v in r.timings_ms.items()}, repr(r.text[:100]),
              flush=True)
    json.dump({"calls": calls, "index": index, "kind": row["kind"],
               "prefill_graph": os.environ.get("FLASHRT_QWEN36_TQ_PREFILL_GRAPH", "0"), "runs": runs},
              open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
