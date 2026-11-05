"""Time per kernel binding of one long-context prefill of the Qwen3.8-27B NVFP4 planner (official MTP head) at the
monitor call's prompt length (4 072 prompt tokens of the repository's documentation), without a CUDA profiler.

The prefill runs eagerly (prefill graph off). Steps: WARM generations of NEW_TOKENS tokens; TIMED generations whose
prefill ms (the frontend's CUDA-event window, including the MTP prompt-tail K/V prefill) are recorded; then every
callable of `flash_rt.flash_rt_kernels` (and `flash_rt.flash_rt_fa2` where it is built) is replaced by a wrapper that
synchronizes the device before and after the call and adds the wall time to that binding's total, and one generation
of one new token runs. Wrapped times include one synchronization per call; "outside bindings" is the wall time of the
wrapped generation minus the sum of binding times (torch operators, host work, bindings held by reference before the
wrap). Printed: prefill ms, the wrapped generation's wall ms, the category table (first CATEGORIES pattern matching
the binding name) and the TOP bindings. Output: $OUT/prefill_binding_times_<frontend>.json.

usage (cwd = repository root): CK=<dir with qwen38-27b-nvfp4-official and qwen38-27b-fp8-official> OUT=<dir> \
  python research/experiments/harness_replay_5090/thor/prefill_binding_times.py {rtx|thor}
"""
import glob
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass

CK = os.environ["CK"]
PROMPT_TOKENS = 4072
NEW_TOKENS = 53
MAX_SEQ = PROMPT_TOKENS + 128
WARM = 2
TIMED = 2
TOP = 50
MODEL_PARAMS = 27e9
CATEGORIES = (("linear_attention", r"linear_attn|gdn|conv1d|tril|wy_"),
              ("gemm", r"gemm|matmul|matvec|cublas"),
              ("attention", r"attn|attention|fa2|fwd_bf16|xqa|flash|softmax"),
              ("quantize_fused", r"nvfp4|fp8|quant|fp4"),
              ("norm", r"norm"),
              ("rope", r"rope|rotary"),
              ("elementwise_copy", r"add|mul|silu|sigmoid|copy|memcpy|fill|concat|split|embed|argmax|accept|gather"))
os.environ.update({"FLASHRT_QWEN36_MTP_CKPT_DIR": CK + "/qwen38-27b-fp8-official",
                   "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096", "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
                   "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "1", "FLASHRT_QWEN36_TQ_PREFILL_GRAPH": "0",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

import torch  # noqa: E402

from flash_rt import flash_rt_kernels  # noqa: E402
from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_thor import Qwen36TorchFrontendThor  # noqa: E402

try:
    from flash_rt import flash_rt_fa2
    KERNEL_MODULES = (flash_rt_kernels, flash_rt_fa2)
except ImportError:
    KERNEL_MODULES = (flash_rt_kernels,)

FRONTENDS = {"rtx": Qwen36TorchFrontendRtx, "thor": Qwen36TorchFrontendThor}


@dataclass
class BindingStat:
    name: str
    category: str
    count: int
    total_ms: float


class BindingTimer:
    """Replaces the callables of the kernel modules with synchronizing, timing wrappers; restore() undoes it."""

    def __init__(self) -> None:
        self.stats: dict[str, BindingStat] = {}
        self.originals: list[tuple[object, str, object]] = []

    def install(self) -> None:
        for mod in KERNEL_MODULES:
            for name in dir(mod):
                fn = getattr(mod, name)
                if name.startswith("_") or not callable(fn) or isinstance(fn, type):
                    continue
                self.originals.append((mod, name, fn))
                setattr(mod, name, self.wrap(mod.__name__.rsplit(".", 1)[-1] + "." + name, fn))

    def wrap(self, label: str, fn: object):
        stats = self.stats

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            st = stats.setdefault(label, BindingStat(label, category(label), 0, 0.0))
            st.count += 1
            st.total_ms += (time.perf_counter() - t0) * 1e3
            return out
        return timed

    def restore(self) -> None:
        for mod, name, fn in self.originals:
            setattr(mod, name, fn)


def category(name: str) -> str:
    low = name.lower().rsplit(".", 1)[-1]
    for cat, pattern in CATEGORIES:
        if re.search(pattern, low):
            return cat
    return "other"


def main(frontend: str) -> None:
    fe = FRONTENDS[frontend](CK + "/qwen38-27b-nvfp4-official", device="cuda:0", max_seq=MAX_SEQ, quant="nvfp4")
    tok = fe._tokenizer
    print("loaded:", type(fe).__name__, "long_ctx_mode", fe._long_ctx_mode, "mtp",
          fe._weights.ptrs.get("mtp") is not None, "kernel modules", [m.__name__ for m in KERNEL_MODULES], flush=True)
    files = ["README.md"] + sorted(glob.glob("docs/*.md"))
    doc = tok.encode("\n\n".join(open(f, encoding="utf-8").read() for f in files), add_special_tokens=False)
    ids = torch.tensor([doc[:PROMPT_TOKENS]], dtype=torch.long, device="cuda")
    for _ in range(WARM):
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=NEW_TOKENS)
    timed = []
    for _ in range(TIMED):
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=NEW_TOKENS)
        timed.append(float(fe._long_ctx_prefill_ms))
    timer = BindingTimer()
    timer.install()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=1)
        torch.cuda.synchronize()
    finally:
        timer.restore()
    wall_ms = (time.perf_counter() - t0) * 1e3
    stats = sorted(timer.stats.values(), key=lambda s: -s.total_ms)
    bind_ms = sum(s.total_ms for s in stats)
    calls = sum(s.count for s in stats)
    pf = statistics.median(timed)
    print(f"prefill ms timed {' '.join(f'{x:.1f}' for x in timed)} (median {pf:.1f}); whole-model TFLOP/s "
          f"{2 * MODEL_PARAMS * PROMPT_TOKENS / (pf * 1e9):.1f} (2 x 27e9 FLOP per prompt token)")
    print(f"wrapped generation (1 new token): wall {wall_ms:.1f} ms, {calls} binding calls, binding sum "
          f"{bind_ms:.1f} ms, outside bindings {wall_ms - bind_ms:.1f} ms")
    cats: dict[str, list] = {}
    for s in stats:
        c = cats.setdefault(s.category, [0.0, 0])
        c[0] += s.total_ms
        c[1] += s.count
    print("category calls total_ms share_of_binding_sum")
    for cat, (ms, n) in sorted(cats.items(), key=lambda kv: -kv[1][0]):
        print(f"{cat} {n} {ms:.1f} {100 * ms / bind_ms:.1f}%")
    print(f"top {TOP} bindings: total_ms calls mean_us category name")
    for s in stats[:TOP]:
        print(f"{s.total_ms:.2f} {s.count} {1e3 * s.total_ms / s.count:.1f} {s.category} {s.name}")
    json.dump({"frontend": frontend, "gpu": torch.cuda.get_device_name(), "prompt_tokens": PROMPT_TOKENS,
               "timed_prefill_ms": timed, "wrapped_wall_ms": wall_ms, "binding_ms": bind_ms,
               "bindings": [asdict(s) for s in stats]},
              open(os.path.join(os.environ["OUT"], f"prefill_binding_times_{frontend}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
