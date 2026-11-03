"""Prefill and decode speed of a Qwen3.6-architecture NVFP4 checkpoint on the FlashRT Qwen3.6 NVFP4 frontend, per
prompt length.

Method (as in docs/qwen36_nvfp4.md section 5): a prompt of exactly CTX tokens, greedy MTP speculative
decode of NEW_TOKENS tokens, one same-shape warmup generation, then REPEATS timed generations. Prefill and decode
times are the frontend's own CUDA-event windows (``_long_ctx_prefill_ms`` / ``_long_ctx_decode_ms``); decode tok/s =
(NEW_TOKENS - 1) * 1000 / decode_ms. The frontend is built once with max_seq = max(CTX) + NEW_TOKENS + 64 and the
long-context threshold set below that, so every prompt takes the chunked FP8-KV route; 512 and above route there by
default.

Two prompt kinds: ``repeat`` tiles one paragraph (the docs/qwen36_nvfp4.md prompt style; repetition makes MTP drafts
easy to accept) and ``doc`` takes the first CTX tokens of the repository's Markdown documentation (README.md, then
docs/*.md in name order), which does not repeat.

Also generates one short chat answer first and prints it, as a check that the checkpoint produces text.

usage: python context_bench.py NVFP4_DIR MTP_DIR OUT_JSON [--ctx 1024 4096 8192] [--prompts repeat doc]
                                [--new-tokens 64] [--repeats 3]
"""
import argparse
import glob
import json
import os
import statistics
from dataclasses import asdict, dataclass

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("mtp_dir")
ap.add_argument("out")
ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096, 8192])
ap.add_argument("--prompts", nargs="+", choices=["repeat", "doc"], default=["repeat", "doc"])
ap.add_argument("--new-tokens", type=int, default=64)
ap.add_argument("--repeats", type=int, default=3)
args = ap.parse_args()

MAX_SEQ = max(args.ctx) + args.new_tokens + 64
os.environ.update({
    "FLASHRT_QWEN36_MTP_CKPT_DIR": args.mtp_dir,
    "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": str(min(4096, MAX_SEQ - 1)),
    "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
    "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "512",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

import torch  # noqa: E402

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx  # noqa: E402

PARAGRAPH = ("The robot arm moves the red block onto the blue plate, then returns to its home pose and waits for the "
             "next instruction from the planner. Each step is logged with its timestamp, joint angles and gripper "
             "state so that the episode can be replayed and inspected later. ")


@dataclass
class RunTiming:
    prefill_ms: float
    decode_ms: float
    route: str
    spec_attempts: int
    spec_accepts: int


@dataclass
class CellResult:
    prompt: str
    ctx: int
    new_tokens: int
    route: str
    prefill_ms_median: float
    decode_ms_median: float
    prefill_tok_s: float
    decode_tok_s: float
    spec_attempts: int
    spec_accepts: int
    peak_allocated_gib: float
    runs: list


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def prompt_ids(tok, kind: str, n: int) -> torch.Tensor:
    if kind == "repeat":
        unit = tok.encode(PARAGRAPH, add_special_tokens=False)
        ids = (unit * (n // len(unit) + 1))[:n]
    else:
        files = [os.path.join(REPO, "README.md")] + sorted(glob.glob(os.path.join(REPO, "docs", "*.md")))
        ids = tok.encode("\n\n".join(open(f, encoding="utf-8").read() for f in files), add_special_tokens=False)
        if len(ids) < n:
            raise RuntimeError(f"documentation has {len(ids)} tokens, fewer than {n}")
        ids = ids[:n]
    return torch.tensor([ids], dtype=torch.long, device="cuda")


def timed_run(fe: Qwen36TorchFrontendRtx, ids: torch.Tensor, new_tokens: int) -> tuple[RunTiming, torch.Tensor]:
    out = fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=new_tokens)
    return RunTiming(prefill_ms=float(fe._long_ctx_prefill_ms), decode_ms=float(fe._long_ctx_decode_ms),
                     route=str(getattr(fe, "_long_ctx_route", "?")), spec_attempts=int(fe._spec_attempts),
                     spec_accepts=int(fe._spec_accepts)), out


fe = Qwen36TorchFrontendRtx(args.ckpt, quant="nvfp4", max_seq=MAX_SEQ)
tok = fe._tokenizer
free, total = torch.cuda.mem_get_info()
print(f"loaded: max_seq {fe.max_seq}, long_ctx_mode {fe._long_ctx_mode}, "
      f"mtp {'yes' if fe._weights.ptrs.get('mtp') is not None else 'no'}, "
      f"GPU used {(total - free) / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)

chat = tok.apply_chat_template([{"role": "user", "content": "Explain quantum entanglement in one short paragraph."}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
chat_ids = tok(chat, return_tensors="pt").input_ids.cuda()
chat_out = fe.generate_own_speculative_KN_nvfp4(chat_ids, max_new_tokens=128)
print("chat check:", repr(tok.decode(chat_out[0, chat_ids.shape[1]:].tolist())), flush=True)

results = []
for kind, ctx in [(k, c) for k in args.prompts for c in args.ctx]:
    ids = prompt_ids(tok, kind, ctx)
    timed_run(fe, ids, args.new_tokens)
    torch.cuda.reset_peak_memory_stats()
    runs = []
    for _ in range(args.repeats):
        r, out = timed_run(fe, ids, args.new_tokens)
        runs.append(r)
    pf = statistics.median(r.prefill_ms for r in runs)
    dc = statistics.median(r.decode_ms for r in runs)
    cell = CellResult(prompt=kind, ctx=ctx, new_tokens=args.new_tokens, route=runs[-1].route, prefill_ms_median=pf,
                      decode_ms_median=dc, prefill_tok_s=ctx * 1000 / pf, decode_tok_s=(args.new_tokens - 1) * 1000 / dc,
                      spec_attempts=runs[-1].spec_attempts, spec_accepts=runs[-1].spec_accepts,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, runs=[asdict(r) for r in runs])
    results.append(cell)
    print(f"{kind:6s} ctx {ctx:5d} | route {cell.route} | prefill {pf:8.1f} ms ({cell.prefill_tok_s:7.0f} tok/s) | "
          f"decode {dc:7.1f} ms ({cell.decode_tok_s:6.1f} tok/s) | spec {cell.spec_attempts}/{cell.spec_accepts} | "
          f"peak {cell.peak_allocated_gib:.2f} GiB | runs prefill {[round(r.prefill_ms, 1) for r in runs]} "
          f"decode {[round(r.decode_ms, 1) for r in runs]}", flush=True)
    print("   output tail:", repr(tok.decode(out[0, ctx:].tolist())[:160]), flush=True)

json.dump({"ckpt": args.ckpt, "mtp_dir": args.mtp_dir, "max_seq": MAX_SEQ, "gpu": torch.cuda.get_device_name(),
           "chat_check": tok.decode(chat_out[0, chat_ids.shape[1]:].tolist()),
           "cells": [asdict(c) for c in results]}, open(args.out, "w"), indent=1)
