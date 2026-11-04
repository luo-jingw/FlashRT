"""LM-only time of the Qwen3.8-27B NVFP4 planner (official MTP head) at the harness call shapes, on the frontend
class given on the command line (the 5090 counterpart of Thor row H3a, which runs the same code with
Qwen36TorchFrontendThor). Shapes: decompose 2 126 prompt tokens / 25 new, monitor 4 072 / 53 (wrist 224), monitor
5 711 / 53 (wrist 720). The prompt is the first N tokens of the repository's documentation (README.md, docs/*.md), so
it does not repeat; image content and the ViT are not part of this measurement. Per shape: one warm-up, three timed
generations (the frontend's prefill / decode CUDA-event windows, spec cycles, accepted drafts); ms per cycle depends
on the hardware only, tokens per forward on the text.

usage (cwd = repository root): CK=<dir with qwen38-27b-nvfp4-official and qwen38-27b-fp8-official> OUT=<dir> \
  python research/experiments/harness_replay_5090/thor/planner_call_shapes.py {rtx|thor}
"""
import glob
import json
import os
import statistics
import sys

CK = os.environ["CK"]
SHAPES = [("decompose", 2126, 25), ("monitor_wrist224", 4072, 53), ("monitor_wrist720", 5711, 53)]
MAX_SEQ = max(c for _, c, _ in SHAPES) + 128
os.environ.update({"FLASHRT_QWEN36_MTP_CKPT_DIR": CK + "/qwen38-27b-fp8-official",
                   "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096", "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
                   "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "512", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

import torch  # noqa: E402

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_thor import Qwen36TorchFrontendThor  # noqa: E402

FRONTENDS = {"rtx": Qwen36TorchFrontendRtx, "thor": Qwen36TorchFrontendThor}


def main(frontend: str) -> None:
    fe = FRONTENDS[frontend](CK + "/qwen38-27b-nvfp4-official", device="cuda:0", max_seq=MAX_SEQ, quant="nvfp4")
    tok = fe._tokenizer
    print("loaded: long_ctx_mode", fe._long_ctx_mode, "mtp", fe._weights.ptrs.get("mtp") is not None, flush=True)
    chat = tok.apply_chat_template([{"role": "user", "content": "Explain quantum entanglement in one short paragraph."}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    cid = tok(chat, return_tensors="pt").input_ids.cuda()
    out = fe.generate_own_speculative_KN_nvfp4(cid, max_new_tokens=64)
    print("chat check:", repr(tok.decode(out[0, cid.shape[1]:].tolist())), flush=True)
    files = ["README.md"] + sorted(glob.glob("docs/*.md"))
    doc = tok.encode("\n\n".join(open(f, encoding="utf-8").read() for f in files), add_special_tokens=False)
    rows = []
    for name, ctx, new in SHAPES:
        ids = torch.tensor([doc[:ctx]], dtype=torch.long, device="cuda")
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=new)
        torch.cuda.reset_peak_memory_stats()
        runs = []
        for _ in range(3):
            fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=new)
            runs.append((float(fe._long_ctx_prefill_ms), float(fe._long_ctx_decode_ms), int(fe._spec_attempts),
                         int(fe._spec_accepts)))
        pf = statistics.median(r[0] for r in runs)
        dc = statistics.median(r[1] for r in runs)
        cyc = runs[-1][2]
        rows.append(dict(shape=name, ctx=ctx, new=new, prefill_ms=pf, decode_ms=dc, cycles=cyc, accepted=runs[-1][3],
                         ms_per_cycle=dc / cyc, tok_per_forward=(new - 1) / cyc,
                         route=str(getattr(fe, "_long_ctx_route", "?")),
                         peak_GiB=torch.cuda.max_memory_allocated() / 2**30, runs=runs))
        print(json.dumps(rows[-1]), flush=True)
    json.dump({"frontend": frontend, "gpu": torch.cuda.get_device_name(), "rows": rows},
              open(os.path.join(os.environ["OUT"], f"planner_call_shapes_{frontend}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
