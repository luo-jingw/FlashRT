"""The official Qwen3.8-27B NVFP4 planner (its own MTP head) on recorded harness calls with images, on the image
frontend given on the command line: `rtx` = Qwen36VlTorchFrontendRtx, `thor` = Qwen36VlTorchFrontendThor (Thor rows
H3, H3b), `rtx_thor_vit` = the RTX language model with the Thor frontend's ViT configuration (BF16 linears, SDPA).
Thor row H3b also ran `thor_cublaslt_tail` (the Thor language model with the RTX frontend's cuBLASLt tail fc in place
of the Thor M-tile kernel, at commit 791b1168); the Thor frontend now inherits that tail fc, so `thor` runs it.
Calls: the rows of calls/ep0_wrist224.jsonl and calls/ep43_wrist224.jsonl (22 calls; images are the recorded JPEGs
under calls/). Each call decodes the official model's own reply through EOS: max_new_tokens and the reference
text come from that model's references in cells/mtp_tail_base.json (RTX 5090). Per call and draft length in SPEC_KS
(passed as `generate(spec_k=...)`: 6 = the language model's prompt-length table, 3 = capped at 3): one warm-up call
(wall ms, which includes the graph captures of a new prompt length) and one timed call (preprocess / ViT / inject
wall ms, prefill / decode CUDA-event ms, spec cycles, decode ms per cycle, accepted drafts, the reply text and whether
it equals the reference). Then one untimed decision call per call at K = 3 with DECISION_BUDGET new tokens (more than
any reference reply, cut at EOS), so that a reply longer than the reference is not truncated; its text is the one
`planner_calls_vl_table.py` parses into the call's decision.

usage (cwd = repository root): CK=<dir with qwen38-27b-nvfp4-official and qwen38-27b-fp8-official> OUT=<dir> \
  python research/topics/wam-inference-acceleration/experiments/harness_replay_5090/thor/planner_calls_vl.py {rtx|thor|rtx_thor_vit}
"""
import json
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass

CK = os.environ["CK"]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALLS = ("calls/ep0_wrist224.jsonl", "calls/ep43_wrist224.jsonl")
REFS = "cells/mtp_tail_base.json"
SPEC_KS = (6, 3)
DECISION_K = 3
DECISION_BUDGET = 160
MAX_SEQ = 16640
MAX_IMAGE_TOKENS = 4608
CHAT_KWARGS = {"enable_thinking": False}
os.environ.update({"FLASHRT_QWEN36_MTP_CKPT_DIR": CK + "/qwen38-27b-fp8-official",
                   "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096", "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
                   "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "1", "FLASHRT_QWEN36_TQ_PREFILL_GRAPH": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from flash_rt.frontends.torch._qwen3_vl_vision_rtx import Qwen3VlVisionRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_vl_thor import Qwen36VlTorchFrontendThor  # noqa: E402


class Qwen36VlRtxWithThorVision(Qwen36VlTorchFrontendRtx):
    """The RTX language model with the ViT configuration of Qwen36VlTorchFrontendThor."""

    def _build_vision(self, checkpoint_path: str, device: str, config: dict) -> Qwen3VlVisionRtx:
        return Qwen3VlVisionRtx(checkpoint_path, device=device, config=config, fp8=False, attention_backend='sdpa')


FRONTENDS = {"rtx": Qwen36VlTorchFrontendRtx, "thor": Qwen36VlTorchFrontendThor,
             "rtx_thor_vit": Qwen36VlRtxWithThorVision}


@dataclass
class Call:
    name: str
    kind: str
    messages: list
    max_new_tokens: int
    reference_text: str


@dataclass
class CallRecord:
    name: str
    kind: str
    spec_k: int
    effective_k: int
    tail_rows: int
    prompt_tokens: int
    image_tokens: int
    max_new_tokens: int
    output_tokens: int
    warmup_wall_ms: float
    wall_ms: float
    preprocess_ms: float
    vit_ms: float
    inject_ms: float
    prefill_ms: float
    decode_ms: float
    spec_cycles: int
    ms_per_cycle: float
    spec_accepted: int
    tok_per_forward: float
    output_crc: int
    same_as_reference: bool
    output_text: str


@dataclass
class DecisionRecord:
    name: str
    kind: str
    max_new_tokens: int
    output_tokens: int
    text: str


def load_calls() -> list[Call]:
    refs = {r["prompt"]: r for r in json.load(open(os.path.join(HERE, REFS)))["references"]}
    calls = []
    for f in CALLS:
        for line in open(os.path.join(HERE, f)):
            row = json.loads(line)
            name = f"{os.path.basename(f)}:{row['kind']}@{row['step']}"
            content = [{"type": "text", "text": p["text"]} if "text" in p
                       else {"type": "image", "image": Image.open(os.path.join(HERE, p["image"])).convert("RGB")}
                       for p in row["user_parts"]]
            messages = [{"role": "system", "content": row["system"]}, {"role": "user", "content": content}]
            calls.append(Call(name, row["kind"], messages, int(refs[name]["decoded_tokens"]), refs[name]["text"]))
    return calls


def timed(fe: Qwen36VlTorchFrontendRtx, call: Call, spec_k: int, max_new_tokens: int) -> tuple[float, object]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fe.generate(call.messages, max_new_tokens=max_new_tokens, spec_k=spec_k, chat_template_kwargs=CHAT_KWARGS)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, r


def main(frontend: str) -> None:
    fe = FRONTENDS[frontend](CK + "/qwen38-27b-nvfp4-official", device="cuda:0", max_seq=MAX_SEQ,
                             max_image_tokens=MAX_IMAGE_TOKENS)
    llm = fe.llm
    print("loaded:", type(fe).__name__, type(llm).__name__, "long_ctx_mode", llm._long_ctx_mode,
          "mtp", llm._weights.ptrs.get("mtp") is not None, flush=True)
    records, decisions = [], []
    for call in load_calls():
        for k in SPEC_KS:
            warm_ms, _ = timed(fe, call, k, call.max_new_tokens)
            wall_ms, r = timed(fe, call, k, call.max_new_tokens)
            L = r.prompt_len
            cyc = int(llm._spec_attempts)
            rec = CallRecord(call.name, call.kind, k, llm._long_tq_effective_k(L, k, call.max_new_tokens),
                             min(llm._long_mtp_prefill_tail_for_prompt(L), L - 1), L, r.num_image_tokens,
                             call.max_new_tokens, len(r.token_ids), warm_ms, wall_ms, r.timings_ms["preprocess"],
                             r.timings_ms["vit"], r.timings_ms["inject"], r.timings_ms["prefill"],
                             r.timings_ms["decode"], cyc, r.timings_ms["decode"] / cyc if cyc else 0.0,
                             int(llm._spec_accepts), (call.max_new_tokens - 1) / cyc if cyc else 0.0,
                             zlib.crc32(str(r.token_ids).encode()), r.text == call.reference_text, r.text)
            records.append(rec)
            print(json.dumps(asdict(rec)), flush=True)
        _, r = timed(fe, call, DECISION_K, DECISION_BUDGET)
        dec = DecisionRecord(call.name, call.kind, DECISION_BUDGET, len(r.token_ids), r.text)
        decisions.append(dec)
        print(json.dumps(asdict(dec)), flush=True)
    json.dump({"frontend": frontend, "gpu": torch.cuda.get_device_name(), "frontend_class": type(fe).__name__,
               "llm_class": type(llm).__name__, "max_seq": MAX_SEQ, "max_image_tokens": MAX_IMAGE_TOKENS,
               "peak_GiB": torch.cuda.max_memory_allocated() / 2**30, "records": [asdict(x) for x in records],
               "decision_k": DECISION_K, "decisions": [asdict(x) for x in decisions]},
              open(os.path.join(os.environ["OUT"], f"planner_calls_vl_{frontend}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
