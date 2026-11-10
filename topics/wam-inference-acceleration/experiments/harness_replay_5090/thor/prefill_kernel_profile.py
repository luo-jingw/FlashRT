"""Per-kernel profile of one long-context prefill of the Qwen3.8-27B NVFP4 planner (official MTP head) at the monitor
call's prompt length (4 072 prompt tokens of the repository's documentation), with the prefill graph on or off.

Steps: WARM generations of NEW_TOKENS tokens; TIMED generations whose prefill ms (the frontend's CUDA-event window,
which includes the MTP prompt-tail K/V prefill) are recorded; one generation of one new token under torch.profiler.
Every CUDA kernel of the profiled generation is grouped by name and by category (the first CATEGORIES pattern that
matches the lower-cased name; quantize_fused holds the kernels that quantize to NVFP4 / FP8, including the ones fused
with a norm or an activation). Printed: prefill ms of the timed and profiled generations, the kernel time sum and
the first-to-last kernel span of the profiled generation, the category table, and the TOP kernels by total time.
Output: $OUT/prefill_kernel_profile_<frontend>_<graph|eager>.json (every kernel name with count and total ms).

usage (cwd = repository root): CK=<dir with qwen38-27b-nvfp4-official and qwen38-27b-fp8-official> OUT=<dir> \
  python research/topics/wam-inference-acceleration/experiments/harness_replay_5090/thor/prefill_kernel_profile.py {rtx|thor} {graph|eager}
"""
import glob
import json
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass

CK = os.environ["CK"]
PROMPT_TOKENS = 4072
NEW_TOKENS = 53
MAX_SEQ = PROMPT_TOKENS + 128
WARM = 3
TIMED = 3
TOP = 40
MODEL_PARAMS = 27e9
CATEGORIES = (("gemm", r"gemmuniversal|gemm|cublas|nvjet|matmul|matvec|xmma"),
              ("linear_attention", r"linear_attention|gdn|causal_conv1d|solve_tril|pack_recompute|rms_norm_gated"),
              ("attention", r"flash_fwd|fmha|xqa|softmax|_attn|attention"),
              ("quantize_fused", r"nvfp4|fp8|quant"),
              ("norm", r"norm"),
              ("rope", r"rope|rotary"),
              ("elementwise_copy", r"add_out|sigmoid|silu|concat|split|embedding|argmax|accept|copy|memcpy|memset|fill|"
                                   r"elementwise"))
MODE = sys.argv[2]
os.environ.update({"FLASHRT_QWEN36_MTP_CKPT_DIR": CK + "/qwen38-27b-fp8-official",
                   "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096", "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
                   "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "1",
                   "FLASHRT_QWEN36_TQ_PREFILL_GRAPH": {"graph": "1", "eager": "0"}[MODE],
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_thor import Qwen36TorchFrontendThor  # noqa: E402

FRONTENDS = {"rtx": Qwen36TorchFrontendRtx, "thor": Qwen36TorchFrontendThor}


@dataclass
class KernelStat:
    name: str
    category: str
    count: int
    total_ms: float


def category(name: str) -> str:
    low = name.lower()
    for cat, pattern in CATEGORIES:
        if re.search(pattern, low):
            return cat
    return "other"


def kernel_events(prof: profile) -> list[dict]:
    path = os.path.join(os.environ["OUT"], f"prefill_kernel_trace_{sys.argv[1]}_{MODE}.json")
    prof.export_chrome_trace(path)
    events = json.load(open(path))["traceEvents"]
    return [e for e in events if e.get("cat") == "kernel" and "dur" in e]


def group(events: list[dict]) -> list[KernelStat]:
    stats: dict[str, KernelStat] = {}
    for e in events:
        st = stats.setdefault(e["name"], KernelStat(e["name"], category(e["name"]), 0, 0.0))
        st.count += 1
        st.total_ms += e["dur"] / 1e3
    return sorted(stats.values(), key=lambda s: -s.total_ms)


def main(frontend: str) -> None:
    fe = FRONTENDS[frontend](CK + "/qwen38-27b-nvfp4-official", device="cuda:0", max_seq=MAX_SEQ, quant="nvfp4")
    tok = fe._tokenizer
    print("loaded:", type(fe).__name__, "long_ctx_mode", fe._long_ctx_mode, "mtp",
          fe._weights.ptrs.get("mtp") is not None, "prefill", MODE, flush=True)
    files = ["README.md"] + sorted(glob.glob("docs/*.md"))
    doc = tok.encode("\n\n".join(open(f, encoding="utf-8").read() for f in files), add_special_tokens=False)
    ids = torch.tensor([doc[:PROMPT_TOKENS]], dtype=torch.long, device="cuda")
    for _ in range(WARM):
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=NEW_TOKENS)
    timed = []
    for _ in range(TIMED):
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=NEW_TOKENS)
        timed.append(float(fe._long_ctx_prefill_ms))
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=1)
        torch.cuda.synchronize()
    profiled_ms = float(fe._long_ctx_prefill_ms)
    events = kernel_events(prof)
    stats = group(events)
    kernel_ms = sum(s.total_ms for s in stats)
    span_ms = (max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)) / 1e3 if events else 0.0
    cats: dict[str, list[float]] = {}
    for s in stats:
        c = cats.setdefault(s.category, [0.0, 0])
        c[0] += s.total_ms
        c[1] += s.count
    pf = statistics.median(timed)
    print(f"prefill ms timed {' '.join(f'{x:.1f}' for x in timed)} (median {pf:.1f}), profiled {profiled_ms:.1f}")
    print(f"profiled generation (1 new token): {len(events)} kernels, kernel sum {kernel_ms:.1f} ms, "
          f"first-to-last span {span_ms:.1f} ms; whole-model TFLOP/s at the timed median "
          f"{2 * MODEL_PARAMS * PROMPT_TOKENS / (pf * 1e9):.1f} (2 x 27e9 FLOP per prompt token)")
    print("category kernels total_ms share")
    for cat, (ms, n) in sorted(cats.items(), key=lambda kv: -kv[1][0]):
        print(f"{cat} {n} {ms:.1f} {100 * ms / kernel_ms:.1f}%")
    print(f"top {TOP} kernels: total_ms count mean_us category name")
    for s in stats[:TOP]:
        print(f"{s.total_ms:.2f} {s.count} {1e3 * s.total_ms / s.count:.1f} {s.category} {s.name[:160]}")
    json.dump({"frontend": frontend, "prefill": MODE, "gpu": torch.cuda.get_device_name(),
               "prompt_tokens": PROMPT_TOKENS, "timed_prefill_ms": timed, "profiled_prefill_ms": profiled_ms,
               "kernel_count": len(events), "kernel_ms": kernel_ms, "span_ms": span_ms,
               "kernels": [asdict(s) for s in stats]},
              open(os.path.join(os.environ["OUT"], f"prefill_kernel_profile_{frontend}_{MODE}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
