"""MTP acceptance on the planner by MTP prompt-tail prefill, tail fc kernel and draft length K, for one main model.

Grid (one process = one model) over the synthetic planner prompts (SYNTH_TARGETS, 64 new tokens) and the recorded
harness calls (CALLS_JSONL): every (tail, fc) in TAIL_CONFIGS x K in KS.
  tail = FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL: "0" (the drafter starts without prompt context), "auto" (the
         frontend's row table), "full" (every prompt row);
  fc   = the kernel of the tail's fc projection (rows x 10240 -> 5120), set per instance on `_mtp_tail_fc_matmul`:
         "warp" (the shared small-M kernel `bf16_matmul_qwen36_bf16`, same fma order as the M = 1 path), "cublaslt"
         (`bf16_matmul_cublaslt_bf16` with the 8-candidate autotune);
  K    = FLASHRT_QWEN36_TQ_SPEC_K as in mtp_speedup.py ("table" = the frontend's default plus adaptive K).

Decode budget: a recorded call is first decoded once (tail 0, K table) with the call's max_tokens; the model's own
reply through EOS (output tokens + 1, or max_tokens without EOS) is the decode length and the reference text of every
cell, so each model decodes to its own EOS. A synthetic prompt decodes 64 tokens and its reference is that first
decode's text. Per cell: one warm-up call and one timed call (prefill / decode ms from the frontend's CUDA events,
spec cycles, accepted drafts), the tail rows, and the MTP tail prefill inside the timed call (synchronized wall ms).

usage (cwd = this directory; PLANNER_CKPT selects the main model):
  PLANNER_CKPT=DIR PLANNER_LONG_TOKENS=16384 FLASHRT_QWEN36_TQ_PREFILL_GRAPH=1 \
    python mtp_tail_model.py MODEL_LABEL OUT_JSON CALLS_JSONL [CALLS_JSONL ...]
"""
import json
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass

import torch

from flash_rt import flash_rt_kernels as fvk
from mtp_speedup import MtpBench, PromptSpec, prompts

import mtp_speedup as ms

TAIL_ENV = {"0": "0", "auto": "auto", "full": "65536"}
TAIL_CONFIGS = (("0", "-"), ("auto", "cublaslt"), ("full", "cublaslt"), ("auto", "warp"))
KS = ("table", "1", "3", "5", "7")


@dataclass
class Reference:
    prompt: str
    prompt_tokens: int
    budget: int
    decoded_tokens: int
    eos: bool
    text: str
    same_as_recorded: bool | None


@dataclass
class CellRecord:
    prompt: str
    tail: str
    fc: str
    tail_rows: int
    tail_ms: float
    k: str
    effective_k: int
    prompt_tokens: int
    decoded_tokens: int
    output_tokens: int
    prefill_ms: float
    decode_ms: float
    spec_cycles: int
    spec_accepted: int
    output_crc: int
    same_as_reference: bool


def warp_fc(x_ptr: int, w_ptr: int, out_ptr: int, rows: int, hidden: int, stream: int) -> None:
    fvk.bf16_matmul_qwen36_bf16(x_ptr, w_ptr, out_ptr, rows, hidden, hidden * 2, stream)


def cublaslt_fc(x_ptr: int, w_ptr: int, out_ptr: int, rows: int, hidden: int, stream: int) -> None:
    fvk.bf16_matmul_cublaslt_bf16(x_ptr, w_ptr, out_ptr, rows, hidden, hidden * 2, stream, 8)


FC = {"-": warp_fc, "warp": warp_fc, "cublaslt": cublaslt_fc}


class TailBench(MtpBench):
    def reference(self, spec: PromptSpec, budget: int) -> tuple[PromptSpec, Reference]:
        os.environ["FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL"] = TAIL_ENV["0"]
        self.set_k("table")
        r = self.generate(PromptSpec(spec.name, spec.messages, budget, None))
        eos = len(r.token_ids) < budget
        n = len(r.token_ids) + 1 if eos else budget
        same = None if spec.recorded_text is None else r.text == spec.recorded_text
        return (PromptSpec(spec.name, spec.messages, n, r.text),
                Reference(spec.name, r.prompt_len, budget, n, eos, r.text, same))

    def cell(self, spec: PromptSpec, tail: str, fc: str, k: str) -> CellRecord:
        os.environ["FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL"] = TAIL_ENV[tail]
        self.set_k(k)
        tail_s = []
        inner = self.llm._prefill_mtp_tail_kv_nvfp4

        def timed_tail(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = inner(*a, **kw)
            torch.cuda.synchronize()
            tail_s.append(time.perf_counter() - t0)
            return out

        self.llm._mtp_tail_fc_matmul = FC[fc]
        try:
            self.generate(spec)
            self.llm._prefill_mtp_tail_kv_nvfp4 = timed_tail
            try:
                r = self.generate(spec)
            finally:
                del self.llm._prefill_mtp_tail_kv_nvfp4
        finally:
            del self.llm._mtp_tail_fc_matmul
        L = r.prompt_len
        eff = self.llm._long_tq_effective_k(L, 6, spec.max_new_tokens) if k == "table" else int(k)
        rows = min(self.llm._long_mtp_prefill_tail_for_prompt(L), L - 1)
        return CellRecord(spec.name, tail, fc, rows, sum(tail_s) * 1e3, k, eff, L, spec.max_new_tokens,
                          len(r.token_ids), r.timings_ms["prefill"], r.timings_ms["decode"],
                          int(self.llm._spec_attempts), int(self.llm._spec_accepts),
                          zlib.crc32(str(r.token_ids).encode()), r.text == spec.recorded_text)


def main(label: str, out: str, files: list[str]) -> None:
    bench = TailBench()
    synth, real = prompts(bench, files)
    budgets = {}
    for f in files:
        for line in open(f):
            row = json.loads(line)
            budgets[f"{os.path.basename(f)}:{row['kind']}@{row['step']}"] = int(row["max_tokens"])
    refs, cells = [], []
    for spec in synth + real:
        spec, ref = bench.reference(spec, budgets.get(spec.name, spec.max_new_tokens))
        refs.append(ref)
        print(f"{spec.name} reference: prompt {ref.prompt_tokens}, decode {ref.decoded_tokens} of {ref.budget}, "
              f"eos {ref.eos}, same_as_recorded {ref.same_as_recorded}", flush=True)
        for tail, fc in TAIL_CONFIGS:
            for k in KS:
                c = bench.cell(spec, tail, fc, k)
                cells.append(c)
                print(f"  tail={tail}/{fc} ({c.tail_rows} rows, {c.tail_ms:.1f} ms) K={k}({c.effective_k}): decode "
                      f"{c.decode_ms:.1f} ms, cycles {c.spec_cycles}, accepted {c.spec_accepted}, "
                      f"same {c.same_as_reference}", flush=True)
    json.dump({"model": label, "ckpt": ms.pq.CKPT, "mtp_dir": ms.pq.MTP_DIR, "max_seq": ms.pq.MAX_SEQ,
               "max_image_tokens": ms.MAX_IMAGE_TOKENS,
               "prefill_graph": os.environ.get("FLASHRT_QWEN36_TQ_PREFILL_GRAPH", "0"),
               "tail_env": TAIL_ENV, "tail_configs": TAIL_CONFIGS, "ks": KS,
               "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
               "references": [asdict(r) for r in refs], "cells": [asdict(c) for c in cells]},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
