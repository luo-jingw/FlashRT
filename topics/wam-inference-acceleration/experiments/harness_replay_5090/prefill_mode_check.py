"""Planner alone (no executor, no gate) on recorded calls under the prefill settings in the environment
(FLASHRT_QWEN36_TQ_PREFILL_GRAPH, FLASHRT_QWEN36_TQ_PREFILL_CHUNK): each call decoded through EOS
(max_new_tokens = recorded output_tokens + 1), reply compared with the recorded one (recorded with the prefill graph
on, default chunk). Separates the prefill setting from co-location as the cause of a changed reply.

usage: python prefill_mode_check.py OUT_JSON CALLS_JSONL [CALLS_JSONL ...]
"""
import json
import os
import sys

import torch

from planner_replay import PlannerStreams, ReplayPlanner

import planner_replay as pr


def main(out: str, files: list[str]) -> None:
    planner = ReplayPlanner(PlannerStreams(*(torch.cuda.Stream() for _ in range(3))))
    rows = []
    for f in files:
        for line in open(f):
            row = json.loads(line)
            msgs = pr.vc.to_messages(pr.vc.call_from_row(row))
            with torch.inference_mode(), torch.cuda.stream(planner.streams.main):
                r = planner.fe.generate(msgs, max_new_tokens=min(row["output_tokens"] + 1, row["max_tokens"]),
                                        chat_template_kwargs=pr.pq.CHAT_KWARGS)
            rows.append({"file": os.path.basename(f), "kind": row["kind"], "step": row["step"],
                         "same_reply": r.text == row["output_text"], "output_tokens": len(r.token_ids),
                         "recorded_tokens": row["output_tokens"], "text": r.text, "recorded_text": row["output_text"]})
            print(f"{os.path.basename(f)} {row['kind']}@{row['step']}: same {rows[-1]['same_reply']}, "
                  f"{len(r.token_ids)} vs {row['output_tokens']} tokens", flush=True)
    json.dump({"prefill_graph": os.environ.get("FLASHRT_QWEN36_TQ_PREFILL_GRAPH", "0"),
               "prefill_chunk": os.environ.get("FLASHRT_QWEN36_TQ_PREFILL_CHUNK"), "rows": rows},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
