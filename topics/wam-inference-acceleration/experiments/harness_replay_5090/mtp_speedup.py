"""MTP speculative decode on the planner: the LoRA-merged Qwen3.8-27B master (NVFP4, FP8 KV cache, long-context
route) with the MTP head of the official Qwen3.8-27B-FP8 release, which the merged checkpoint does not contain.

Part "decode": each prompt decoded with draft length K in KS (FLASHRT_QWEN36_TQ_SPEC_K; "table" = the frontend's
default, a K looked up by prompt length plus adaptive K). Prompts: the recorded harness calls (CALLS_JSONL, decoded
through EOS: max_new_tokens = recorded output_tokens + 1) and the synthetic planner prompts of `planner_qwen27b.py` at
SYNTH_TARGETS (64 new tokens). Per (prompt, K): one warm-up call (captures the graphs of new decode positions), then
one timed call: prefill and decode ms (the frontend's CUDA events), spec cycles, accepted drafts, output crc.

Part "graphs": at each synthetic prompt length L, the median replay time of single captured graphs at decode position
L + GRAPH_OFFSET: the main-model forward over Kv tokens (the verify graph; Kv = 1 is one single-token decode step) and
the MTP draft chain of K steps. Also the MTP prompt-tail prefill inside the call's prefill (synchronized wall ms).

usage (cwd = this directory, recorded image paths are relative to it):
  PLANNER_LONG_TOKENS=16384 python mtp_speedup.py OUT_JSON CALLS_JSONL [CALLS_JSONL ...]
"""
import json
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "colocation_qwen27b_5090"))
import planner_qwen27b as pq  # noqa: E402  (planner environment: MTP head, long-context route, FP8 KV)

import volo_calls as vc  # noqa: E402

KS = ("table", "1", "3", "5", "7")
SYNTH_TARGETS = (0, 4096, 8192, 16384)
VERIFY_KV = (1, 2, 4, 6, 8)
CHAIN_K = (1, 3, 5, 7)
GRAPH_OFFSET = 16
REPLAYS = 20
MAX_IMAGE_TOKENS = 4608


@dataclass
class PromptSpec:
    name: str
    messages: list
    max_new_tokens: int
    recorded_text: str | None


@dataclass
class DecodeRecord:
    prompt: str
    k: str
    effective_k: int
    prompt_tokens: int
    output_tokens: int
    prefill_ms: float
    decode_ms: float
    spec_cycles: int
    spec_accepted: int
    output_crc: int
    same_as_recorded: bool | None
    graphs_captured_warmup: int


@dataclass
class GraphRecord:
    prompt_tokens: int
    position: int
    verify_ms: dict      # Kv -> median replay ms of the main-model forward over Kv tokens
    chain_ms: dict       # K -> median replay ms of the K-step MTP draft chain
    mtp_tail_rows: int
    mtp_tail_prefill_ms: float
    prefill_ms: float


class MtpBench:
    def __init__(self):
        self.planner = pq.Planner(max_image_tokens=MAX_IMAGE_TOKENS)
        self.llm = self.planner.fe.llm

    def graph_count(self) -> int:
        return sum(len(v) for k, v in vars(self.llm).items() if k.startswith("_captured_") and isinstance(v, dict))

    @staticmethod
    def set_k(k: str) -> None:
        if k == "table":
            os.environ.pop("FLASHRT_QWEN36_TQ_SPEC_K", None)
        else:
            os.environ["FLASHRT_QWEN36_TQ_SPEC_K"] = k

    @torch.inference_mode()
    def generate(self, spec: PromptSpec):
        with torch.cuda.stream(self.planner.streams.main):
            return self.planner.fe.generate(spec.messages, max_new_tokens=spec.max_new_tokens,
                                            chat_template_kwargs=pq.CHAT_KWARGS)

    def decode(self, spec: PromptSpec, k: str) -> DecodeRecord:
        self.set_k(k)
        g0 = self.graph_count()
        self.generate(spec)
        captured = self.graph_count() - g0
        r = self.generate(spec)
        eff = self.llm._long_tq_effective_k(r.prompt_len, 6, spec.max_new_tokens) if k == "table" else int(k)
        return DecodeRecord(spec.name, k, eff, r.prompt_len, len(r.token_ids), r.timings_ms["prefill"],
                            r.timings_ms["decode"], int(self.llm._spec_attempts), int(self.llm._spec_accepts),
                            zlib.crc32(str(r.token_ids).encode()),
                            None if spec.recorded_text is None else r.text == spec.recorded_text, captured)

    def replay_ms(self, g) -> float:
        gs = self.llm._graph_stream
        times = []
        for _ in range(REPLAYS):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(gs):
                e0.record(gs)
                g.replay()
                e1.record(gs)
            gs.synchronize()
            times.append(e0.elapsed_time(e1))
        return float(np.median(times))

    @torch.inference_mode()
    def graphs(self, spec: PromptSpec) -> GraphRecord:
        """Called right after `spec` was decoded, so the caches hold a valid state for its prompt."""
        tail_s = []
        inner = self.llm._prefill_mtp_tail_kv_nvfp4

        def timed_tail(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = inner(*a, **kw)
            torch.cuda.synchronize()
            tail_s.append(time.perf_counter() - t0)
            return out

        self.set_k("table")
        self.llm._prefill_mtp_tail_kv_nvfp4 = timed_tail
        try:
            r = self.generate(spec)
        finally:
            del self.llm._prefill_mtp_tail_kv_nvfp4
        L = r.prompt_len
        tail_rows = min(self.llm._long_mtp_prefill_tail_for_prompt(L), L - 1)
        pos = L + GRAPH_OFFSET
        with torch.cuda.stream(self.planner.streams.main):
            verify = {kv: self.replay_ms(self.llm._ensure_verify_graph_nvfp4_fp8kv(pos, kv)) for kv in VERIFY_KV}
            chain = {k: self.replay_ms(self.llm._ensure_mtp_chain_graph_nvfp4(pos, k, cache_base_pos=tail_rows
                                                                                + GRAPH_OFFSET))
                     for k in CHAIN_K}
        return GraphRecord(L, pos, verify, chain, tail_rows, sum(tail_s) * 1e3, r.timings_ms["prefill"])


def prompts(bench: MtpBench, files: list[str]) -> tuple[list[PromptSpec], list[PromptSpec]]:
    p = bench.planner
    synth = [PromptSpec(f"synth:{t}", p.messages("short") if t == 0 else p.messages("long", long_tokens=t),
                        pq.NEW_TOKENS, None) for t in SYNTH_TARGETS]
    real = []
    for f in files:
        for line in open(f):
            row = json.loads(line)
            real.append(PromptSpec(f"{os.path.basename(f)}:{row['kind']}@{row['step']}",
                                   vc.to_messages(vc.call_from_row(row)),
                                   min(row["output_tokens"] + 1, row["max_tokens"]), row["output_text"]))
    return synth, real


def main(out: str, files: list[str]) -> None:
    bench = MtpBench()
    synth, real = prompts(bench, files)
    decode, graphs = [], []
    for spec in synth + real:
        for k in KS:
            rec = bench.decode(spec, k)
            decode.append(rec)
            print(f"{spec.name} K={k}({rec.effective_k}): prompt {rec.prompt_tokens}, out {rec.output_tokens}, decode "
                  f"{rec.decode_ms:.1f} ms, cycles {rec.spec_cycles}, accepted {rec.spec_accepted}, crc "
                  f"{rec.output_crc}, same_as_recorded {rec.same_as_recorded}, captured {rec.graphs_captured_warmup}",
                  flush=True)
        if spec.recorded_text is None:
            g = bench.graphs(spec)
            graphs.append(g)
            print(f"graphs @ {g.position}: verify {g.verify_ms}, chain {g.chain_ms}, mtp tail {g.mtp_tail_rows} rows "
                  f"{g.mtp_tail_prefill_ms:.1f} ms of prefill {g.prefill_ms:.1f} ms", flush=True)
    json.dump({"ks": KS, "max_seq": pq.MAX_SEQ, "max_image_tokens": MAX_IMAGE_TOKENS, "ckpt": pq.CKPT,
               "mtp_dir": pq.MTP_DIR, "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
               "decode": [asdict(d) for d in decode], "graphs": [asdict(g) for g in graphs]},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
