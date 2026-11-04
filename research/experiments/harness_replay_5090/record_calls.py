"""Record the master calls of VoLoAgent's Full loop for replay: the decision logic run on the FlashRT planner alone
(zero-shot Qwen3.8-27B NVFP4, whole device, greedy, thinking off), over a LIBERO-10 episode mapped onto 800 env steps.

Decision logic (VoLoAgent, strategies/subgoal.py and subgoal_base.py, failure_handlers/vlm.py):
  step 0       decompose; unparsable or empty subgoals -> [instruction]
  every 80     monitor; an invalid action or an unparsable reply is "continue"; "next" on a subgoal before the last
               advances it; "replan", or "next" on the last subgoal, issues the replan at the same step (at most 3 per
               episode); a replan reply with done=true or no subgoals is rejected, otherwise its subgoals replace the
               list from index 0. grasp_tool / place_tool are recorded but no tool runs (none is served here).
  end          after step 800, or once a replan reports done after a "next" on the last subgoal.

Per call, two timings: the reply decoded up to the call's max_tokens (the frontend decodes max_new_tokens tokens and
truncates at EOS afterwards), and the reply decoded to its own length (max_new_tokens = output tokens + 1, the median
of two runs) -- the second is what a stop at EOS costs and what the replay uses.

usage: python record_calls.py OUT_DIR EPISODE:WRIST [EPISODE:WRIST ...]     (WRIST 224 or 720)
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colocation_qwen27b_5090"))
import planner_qwen27b as pq  # noqa: E402  (sets the planner's environment: MTP head, long-context route, FP8 KV)

from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402

import volo_calls as vc  # noqa: E402

MAX_SEQ = 8448
MAX_IMAGE_TOKENS = 4608
ACTIONS = {"next", "continue", "replan", "grasp_tool", "place_tool"}


def parse_json(text: str) -> dict:
    """VoLoAgent's vlm/api.py parse_json: whole text, else a fenced block, else the outermost braces; an object only."""
    for candidate in (text,
                      (m.group(1) if (m := re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)) else None),
                      (m.group(0) if (m := re.search(r"\{.*\}", text, re.DOTALL)) else None)):
        if candidate is None:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(text[:200])


def normalize_subgoals(items: list) -> list[str]:
    return [s["instruction"] if isinstance(s, dict) and "instruction" in s else str(s) for s in items]


def stage_ms(r, wall_ms: float) -> dict:
    t = r.timings_ms
    return {"preprocess_ms": t["preprocess"], "vit_ms": t["vit"], "inject_ms": t["inject"],
            "prefill_ms": t["prefill"], "decode_ms": t["decode"], "call_ms": wall_ms}


class Recorder:
    def __init__(self):
        self.fe = Qwen36VlTorchFrontendRtx(pq.CKPT, max_seq=MAX_SEQ, max_image_tokens=MAX_IMAGE_TOKENS)
        torch.cuda.synchronize()

    def _generate(self, msgs: list, max_new: int):
        t0 = time.perf_counter()
        r = self.fe.generate(msgs, max_new_tokens=max_new, chat_template_kwargs=pq.CHAT_KWARGS)
        return r, (time.perf_counter() - t0) * 1e3

    def run(self, call: vc.CallSpec) -> dict:
        msgs = vc.to_messages(call)
        r_cap, ms_cap = self._generate(msgs, call.max_tokens)
        n_out = len(r_cap.token_ids)
        row = vc.call_to_row(call)
        row.update({"output_text": r_cap.text, "output_tokens": n_out, "hit_cap": n_out >= call.max_tokens,
                    "prompt_tokens": r_cap.prompt_len, "image_tokens": r_cap.num_image_tokens,
                    "route": r_cap.route, "timings_cap": stage_ms(r_cap, ms_cap)})
        runs = [self._generate(msgs, min(n_out + 1, call.max_tokens)) for _ in range(2)]
        row["text_equal"] = all(r.text == r_cap.text for r, _ in runs)
        row["timings"] = {k: float(np.median([stage_ms(r, ms)[k] for r, ms in runs])) for k in stage_ms(*runs[0])}
        print(f"  {call.kind:9s} step {call.step:3d}: prompt {row['prompt_tokens']} (image {row['image_tokens']}), "
              f"out {n_out}{' CAP' if row['hit_cap'] else ''}, call {row['timings']['call_ms']:.0f} ms "
              f"(cap {ms_cap:.0f}) | {r_cap.text[:90]!r}", flush=True)
        return row

    def episode(self, episode: int, wrist: int, out_dir: str) -> list[dict]:
        ep = vc.LiberoEpisode(os.environ["DATA_ROOT"], episode)
        img_dir = os.path.join(out_dir, f"ep{episode}_wrist{wrist}")
        os.makedirs(img_dir, exist_ok=True)
        before = ep.harness_frames(0, wrist, img_dir)
        rows = []

        row = self.run(vc.decompose_call(ep.task, before))
        try:
            data = parse_json(row["output_text"])
            subgoals = normalize_subgoals(data.get("subgoals", []))
            if not subgoals:
                raise ValueError("empty subgoals")
            row["parse_ok"] = True
        except ValueError:
            subgoals, row["parse_ok"] = [ep.task], False
        row["subgoals"] = subgoals
        rows.append(row)
        original, idx, replans, done = list(subgoals), 0, 0, False

        for step in range(vc.CHECK_INTERVAL, vc.EPISODE_STEPS + 1, vc.CHECK_INTERVAL):
            now = ep.harness_frames(step, wrist, img_dir)
            row = self.run(vc.monitor_call(ep.task, subgoals, idx, before, now, step))
            try:
                data = parse_json(row["output_text"])
                action = data.get("action", "continue")
                row["parse_ok"] = True
            except ValueError:
                action, row["parse_ok"] = "continue", False
            action = action if action in ACTIONS else "continue"
            row["decision"] = action
            rows.append(row)
            last = idx == len(subgoals) - 1
            if action == "next" and not last:
                idx += 1
            elif (action == "replan" or (action == "next" and last)) and replans < vc.MAX_REPLANS:
                replans += 1
                rrow = self.run(vc.replan_call(ep.task, original, before, now, step))
                try:
                    data = parse_json(rrow["output_text"])
                    new = normalize_subgoals(data.get("subgoals", []))
                    rrow["parse_ok"] = True
                    rrow["done"] = bool(data.get("done", False))
                except ValueError:
                    new, rrow["parse_ok"], rrow["done"] = [], False, False
                rrow["accepted"] = bool(new) and not rrow["done"]
                if rrow["accepted"]:
                    subgoals, idx = new, 0
                rows.append(rrow)
                if rrow["done"] and action == "next" and last:
                    done = True
            if done:
                break
        return rows


def main(out_dir: str, specs: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rec = Recorder()
    for spec in specs:
        episode, wrist = (int(x) for x in spec.split(":"))
        print(f"episode {episode} wrist {wrist}", flush=True)
        rows = rec.episode(episode, wrist, out_dir)
        with open(os.path.join(out_dir, f"ep{episode}_wrist{wrist}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
