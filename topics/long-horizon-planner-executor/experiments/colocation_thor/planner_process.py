"""The 27B planner as a standalone process at the VoLoAgent harness's call cadence, for co-location measurements on the
Jetson Thor (an executor process runs beside it; this process does not start or time the executor).

Per recorded episode (`harness_replay_5090/calls/*.jsonl`): the decompose call is due at the episode start, each
monitor call at start + step / 80 x INTERVAL_S (80 control steps at 15 Hz = 5.33 s), a replan call right after the
monitor call of the same step. A call that is due while the previous one runs starts when it ends (one planner
stream, as in the harness). Every call is one long-context prefill on the committed-stream path and a decode stream
read chunk by chunk (one chunk = one verify cycle).

Arms:
  original : every call prefills its whole prompt and decodes to EOS; the decision is available at the end.
  fast     : monitor calls restore the episode's constant prefix from a capsule (system prompt + BEFORE images, cut
             at the 2 048-token prefill chunk; built once at the episode start, after decompose) and stop decoding
             after the cycle that closes the action value; decompose and replan as in `original`.

One JSON line per call: episode, kind, step, due / start / decision / end on the monotonic clock (s since the
process's first episode start) and on the wall clock (time.time(), for alignment with the executor's log), stage ms,
new tokens, decision. The capsule build is one line of kind "capsule".

usage (cwd anywhere; PLANNER_CKPT, PLANNER_MTP_DIR as for ../monitor_call/monitor_call_bench.py):
  PLANNER_CKPT=DIR PLANNER_MTP_DIR=DIR python planner_process.py {rtx|thor} {original|fast} OUT_JSONL CALLS_JSONL...
"""
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "monitor_call"))

from monitor_call_bench import (LOOKAHEAD_ROWS, MAX_SEQ, REPLAY_DIR, SPEC_K, MonitorBench, PreparedCall,  # noqa: E402
                                episode_label, parse_action, sync_ms, to_messages)

import torch  # noqa: E402

INTERVAL_S = 80 / 15.0       # harness_replay_5090/replay_bench.py: monitor every 80 env steps at 15 Hz
BUDGET = {"decompose": 512, "monitor": 512, "replan": 1024}   # volo_calls max_tokens


@dataclass
class CallLine:
    episode: str
    kind: str
    step: int
    arm: str
    due_s: float
    start_s: float
    decision_s: float
    end_s: float
    start_wall: float
    decision_wall: float
    end_wall: float
    vit_ms: float
    restore_ms: float
    prefill_ms: float
    decode_ms: float
    new_tokens: int
    prompt_tokens: int
    decision: str | None


class PlannerProcess:
    def __init__(self, frontend: str, arm: str) -> None:
        assert arm in ("original", "fast"), arm
        self.arm = arm
        self.b = MonitorBench(frontend)
        self.eos = set(self.b.fe._eos_token_ids)
        self.t0 = 0.0

    def now(self) -> float:
        return time.monotonic() - self.t0

    @torch.inference_mode()
    def call(self, pc: PreparedCall, kind: str, cap: tuple[dict, int] | None, stop_at_action: bool) -> dict:
        """One call; returns stage times, tokens, decision and the monotonic / wall time the decision was available."""
        S = int(pc.input_ids.shape[0])
        budget = min(BUDGET[kind], MAX_SEQ - S - LOOKAHEAD_ROWS - 8)
        self.b.fe.processor.apply_chat_template(
            to_messages(pc.row_abs), add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", processor_kwargs={"device": self.b.fe.device}, enable_thinking=False)
        t0 = time.perf_counter()
        first_img = 0 if cap is None else pc.first_now_image
        feats = self.b.encode(pc, first_img)
        vit = sync_ms(t0)
        first_tok = 0 if cap is None else self.b.image_tokens_before(pc, pc.boundary)
        lm_ids = self.b.inject(pc, feats, first_tok, S + budget + LOOKAHEAD_ROWS)
        t1 = time.perf_counter()
        restore = 0.0
        if cap is None:
            self.b.llm.prefill_long_ctx_nvfp4_agent(lm_ids, max_new_tokens=budget, K=SPEC_K)
        else:
            self.b.llm.restore_capsule(cap[0])
            restore = sync_ms(t1)
            t1 = time.perf_counter()
            self.b.llm.append_long_ctx_nvfp4_agent(lm_ids, start_pos=cap[1], max_new_tokens=budget, K=SPEC_K)
        prefill = sync_ms(t1)
        t2 = time.perf_counter()
        toks: list[int] = []
        dec_s = dec_wall = None
        for chunk in self.b.llm.decode_long_ctx_nvfp4_committed_stream(max_new_tokens=budget, K=SPEC_K,
                                                                       stop_token_ids=self.eos):
            toks.extend(chunk)
            if any(t in self.eos for t in chunk):
                break
            if stop_at_action and parse_action(self.b.tok.decode(toks, skip_special_tokens=True)) is not None:
                torch.cuda.synchronize()
                dec_s, dec_wall = self.now(), time.time()
                break
        decode = sync_ms(t2)
        self.b.llm.reset_rope_rows()
        text = self.b.text_of(toks)
        n_new = next((i for i, t in enumerate(toks) if t in self.eos), len(toks))
        return {"vit_ms": vit, "restore_ms": restore, "prefill_ms": prefill, "decode_ms": decode,
                "new_tokens": n_new, "prompt_tokens": S, "decision": parse_action(text),
                "decision_s": dec_s, "decision_wall": dec_wall}

    def episode(self, jsonl: str, out) -> None:
        ep = episode_label(jsonl)
        rows = [json.loads(line) for line in open(jsonl)]
        mon = [r for r in rows if r["kind"] == "monitor"]
        before = sorted(set.intersection(*[{p["image"] for p in r["user_parts"] if "image" in p} for r in mon]))
        start = self.now()
        cap = None
        for r in rows:
            pc = self.b.prepare(r, REPLAY_DIR, before if r["kind"] == "monitor" else [])
            due = start + (r["step"] / 80) * INTERVAL_S
            while self.now() < due:
                time.sleep(min(0.002, due - self.now()))
            t_start, w_start = self.now(), time.time()
            fast_monitor = self.arm == "fast" and r["kind"] == "monitor"
            st = self.call(pc, r["kind"], cap if fast_monitor else None, stop_at_action=fast_monitor)
            t_end, w_end = self.now(), time.time()
            line = CallLine(ep, r["kind"], r["step"], self.arm, due, t_start,
                            st["decision_s"] if st["decision_s"] is not None else t_end, t_end, w_start,
                            st["decision_wall"] if st["decision_wall"] is not None else w_end, w_end,
                            st["vit_ms"], st["restore_ms"], st["prefill_ms"], st["decode_ms"], st["new_tokens"],
                            st["prompt_tokens"], st["decision"])
            out.write(json.dumps(asdict(line)) + "\n")
            out.flush()
            print(f"{ep} {r['kind']}@{r['step']}: start {t_start - due:+.2f} s late, decision after "
                  f"{line.decision_s - t_start:.2f} s, end {t_end - t_start:.2f} s, {st['decision']}", flush=True)
            if self.arm == "fast" and r["kind"] == "decompose":
                pc0 = self.b.prepare(mon[0], REPLAY_DIR, before)
                P = self.b.llm.capsule_aligned_len(pc0.boundary)
                c0 = self.now()
                capsule, ms = self.b.build_capsule(pc0, P, BUDGET["monitor"])
                cap = (capsule, P)
                out.write(json.dumps({"episode": ep, "kind": "capsule", "start_s": c0, "end_s": self.now(),
                                      "build_ms": ms, "prefix_tokens": P}) + "\n")


def main() -> None:
    frontend, arm, out_path, jsonls = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    proc = PlannerProcess(frontend, arm)
    print(f"loaded: {type(proc.b.fe).__name__} {type(proc.b.llm).__name__} arm {arm}", flush=True)
    proc.t0 = time.monotonic()   # graphs are captured lazily: the first calls of the process include captures
    with open(out_path, "w") as out:
        for jl in jsonls:
            proc.episode(jl, out)


if __name__ == "__main__":
    main()
