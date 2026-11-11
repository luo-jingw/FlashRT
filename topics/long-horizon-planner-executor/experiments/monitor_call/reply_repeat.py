"""Run-to-run repeatability of the planner's monitor replies in one process (Thor row H12).

Per recorded monitor call: REPEATS greedy `generate` calls (the reference path of `monitor_call_bench.py`, OWN_BUDGET
new tokens, cut at EOS), then the top-2 logit margin of the next-token distribution at three positions of the first
repetition's reply, each scored by one prefill of the prompt plus the reply tokens before that position:
  first   : position 0 (the reply's first token; a ```json fence or not);
  action  : the first token of the action value;
  diverge : the first position where any repetition differs from the first one (only if one differs).
The prefill-scored distribution is the long-context prefill path, not the decode path that emitted the tokens; each
score records whether its top-1 token equals the emitted token (`top1_is_emitted`).

usage (cwd anywhere; PLANNER_CKPT, PLANNER_MTP_DIR as for monitor_call_bench.py):
  PLANNER_CKPT=DIR PLANNER_MTP_DIR=DIR python reply_repeat.py {rtx|thor} OUT_JSON CALLS_JSONL...
"""
import json
import re
import sys
from dataclasses import asdict, dataclass

import torch

from monitor_call_bench import (LOOKAHEAD_ROWS, OWN_BUDGET, REPLAY_DIR, SPEC_K, MonitorBench, PreparedCall,
                                episode_label, parse_action)

REPEATS = 5
ACTION_OPEN_RE = re.compile(r'"action"\s*:\s*"[^"]')


@dataclass
class Score:
    kind: str              # first | action | diverge
    position: int          # index into the reply tokens
    emitted: int           # the first repetition's token at that position
    top1: int
    top2: int
    logit1: float
    logit2: float
    margin: float          # logit1 - logit2
    top1_is_emitted: bool


@dataclass
class CallRecord:
    episode: str
    step: int
    prompt_tokens: int
    texts: list[str]               # per repetition
    tokens: list[list[int]]        # per repetition, cut at EOS
    decisions: list[str | None]
    first_diff: list[int]          # per repetition: first position differing from repetition 0 (-1 = identical)
    scores: list[Score]


class RepeatProbe:
    def __init__(self, bench: MonitorBench) -> None:
        self.b = bench

    def reply_tokens(self, pc: PreparedCall) -> list[int]:
        toks, _ = self.b.run_generate(pc, OWN_BUDGET)
        cut = next((i for i, t in enumerate(toks) if t in self.b.fe._eos_token_ids), len(toks))
        return list(toks[:cut])

    @torch.inference_mode()
    def score(self, pc: PreparedCall, reply: list[int], position: int, kind: str) -> Score:
        """Next-token logits after the prompt + reply[:position], from one long-context prefill."""
        ext = torch.cat([pc.input_ids, torch.tensor(reply[:position], dtype=pc.input_ids.dtype,
                                                    device=pc.input_ids.device)])
        feats = self.b.encode(pc, 0)
        mask = ext == self.b.fe._image_token_id
        slot = self.b.llm.embedding_slot_base
        self.b.llm.write_embedding_rows(slot, feats)
        lm_ids = ext.clone()
        lm_ids[mask] = slot + torch.arange(int(mask.sum()), device=ext.device)
        cos, sin = self.b.fe._rope_rows(ext, pc.grid, int(ext.shape[0]) + 1 + LOOKAHEAD_ROWS)
        self.b.llm.set_rope_rows(0, cos, sin)
        self.b.llm.prefill_long_ctx_nvfp4_agent(lm_ids.view(1, -1), max_new_tokens=1, K=SPEC_K)
        logits = self.b.llm._logits_buf[0].float()
        self.b.llm.reset_rope_rows()
        v, i = torch.topk(logits, 2)
        emitted = reply[position]
        return Score(kind, position, emitted, int(i[0]), int(i[1]), float(v[0]), float(v[1]), float(v[0] - v[1]),
                     int(i[0]) == emitted)

    def action_position(self, reply: list[int]) -> int | None:
        for n in range(1, len(reply) + 1):
            if ACTION_OPEN_RE.search(self.b.tok.decode(reply[:n], skip_special_tokens=True)):
                return n - 1
        return None

    def run(self, pc: PreparedCall, episode: str) -> CallRecord:
        reps = [self.reply_tokens(pc) for _ in range(REPEATS)]
        texts = [self.b.tok.decode(r, skip_special_tokens=True) for r in reps]

        def first_diff(a: list[int], b: list[int]) -> int:
            for k in range(min(len(a), len(b))):
                if a[k] != b[k]:
                    return k
            return -1 if len(a) == len(b) else min(len(a), len(b))

        diffs = [first_diff(reps[0], r) for r in reps]
        scores = [self.score(pc, reps[0], 0, "first")]
        pa = self.action_position(reps[0])
        if pa is not None:
            scores.append(self.score(pc, reps[0], pa, "action"))
        div = [d for d in diffs if d >= 0]
        if div and min(div) < len(reps[0]):
            scores.append(self.score(pc, reps[0], min(div), "diverge"))
        return CallRecord(episode, int(pc.row["step"]), int(pc.input_ids.shape[0]), texts, reps,
                          [parse_action(t) for t in texts], diffs, scores)


def main() -> None:
    frontend, out_path, jsonls = sys.argv[1], sys.argv[2], sys.argv[3:]
    bench = MonitorBench(frontend)
    probe = RepeatProbe(bench)
    records: list[CallRecord] = []
    for jl in jsonls:
        ep = episode_label(jl)
        rows = [json.loads(line) for line in open(jl)]
        mon = [r for r in rows if r["kind"] == "monitor"]
        img_sets = [{p["image"] for p in r["user_parts"] if "image" in p} for r in mon]
        before = sorted(set.intersection(*img_sets))
        for r in mon:
            pc = bench.prepare(r, REPLAY_DIR, before)
            rec = probe.run(pc, ep)
            records.append(rec)
            same = sum(1 for d in rec.first_diff if d == -1)
            sc = " ".join(f"{s.kind}@{s.position} margin {s.margin:.3f}{'' if s.top1_is_emitted else ' (top1!=emitted)'}"
                          for s in rec.scores)
            print(f"{ep} step {rec.step}: {same}/{REPEATS} identical, first diffs {rec.first_diff}, "
                  f"decisions {rec.decisions}, {sc}", flush=True)
            with open(out_path, "w") as f:
                json.dump({"frontend": frontend, "repeats": REPEATS, "calls": [asdict(c) for c in records]}, f,
                          indent=1)


if __name__ == "__main__":
    main()
