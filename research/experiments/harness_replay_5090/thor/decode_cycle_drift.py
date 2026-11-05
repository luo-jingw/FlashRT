"""Decode ms per speculative cycle of the Qwen3.8-27B NVFP4 planner (official MTP head) over time in one process, with
the clock readings the machine exposes without root. The monitor call shape (4 072 prompt tokens of the repository's
documentation, 53 new tokens, the prompt-length table's K) is generated back to back for DURATION_S seconds; per
generation one line: seconds since the first generation, prefill and decode ms (the frontend's CUDA-event windows),
spec cycles, decode ms per cycle, the CPU cores' current frequencies (cpufreq, kHz) and every devfreq device's current
frequency (Hz). Output: $OUT/decode_cycle_drift_<frontend>.json.

usage (cwd = repository root): CK=<dir with qwen38-27b-nvfp4-official and qwen38-27b-fp8-official> OUT=<dir> \
  python research/experiments/harness_replay_5090/thor/decode_cycle_drift.py {rtx|thor}
"""
import glob
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

CK = os.environ["CK"]
PROMPT_TOKENS = 4072
NEW_TOKENS = 53
DURATION_S = 300.0
MAX_SEQ = PROMPT_TOKENS + 128
os.environ.update({"FLASHRT_QWEN36_MTP_CKPT_DIR": CK + "/qwen38-27b-fp8-official",
                   "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096", "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
                   "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "1", "FLASHRT_QWEN36_TQ_PREFILL_GRAPH": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

import torch  # noqa: E402

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_thor import Qwen36TorchFrontendThor  # noqa: E402

FRONTENDS = {"rtx": Qwen36TorchFrontendRtx, "thor": Qwen36TorchFrontendThor}


@dataclass
class Sample:
    t_s: float
    prefill_ms: float
    decode_ms: float
    cycles: int
    ms_per_cycle: float
    cpu_khz: list
    devfreq_hz: dict


def read_int(path: str) -> int:
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return -1


def cpu_khz() -> list:
    return [read_int(p) for p in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))]


def devfreq_hz() -> dict:
    return {os.path.basename(os.path.dirname(p)): read_int(p) for p in sorted(glob.glob("/sys/class/devfreq/*/cur_freq"))}


def main(frontend: str) -> None:
    fe = FRONTENDS[frontend](CK + "/qwen38-27b-nvfp4-official", device="cuda:0", max_seq=MAX_SEQ, quant="nvfp4")
    tok = fe._tokenizer
    print("loaded:", type(fe).__name__, "long_ctx_mode", fe._long_ctx_mode, "mtp",
          fe._weights.ptrs.get("mtp") is not None, "devfreq", list(devfreq_hz()), flush=True)
    files = ["README.md"] + sorted(glob.glob("docs/*.md"))
    doc = tok.encode("\n\n".join(open(f, encoding="utf-8").read() for f in files), add_special_tokens=False)
    ids = torch.tensor([doc[:PROMPT_TOKENS]], dtype=torch.long, device="cuda")
    samples = []
    t0 = time.perf_counter()
    while not samples or samples[-1].t_s < DURATION_S:
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=NEW_TOKENS)
        cyc = int(fe._spec_attempts)
        dc = float(fe._long_ctx_decode_ms)
        s = Sample(time.perf_counter() - t0, float(fe._long_ctx_prefill_ms), dc, cyc, dc / cyc if cyc else 0.0,
                   cpu_khz(), devfreq_hz())
        samples.append(s)
        print(json.dumps(asdict(s)), flush=True)
    json.dump({"frontend": frontend, "gpu": torch.cuda.get_device_name(), "prompt_tokens": PROMPT_TOKENS,
               "new_tokens": NEW_TOKENS, "samples": [asdict(x) for x in samples]},
              open(os.path.join(os.environ["OUT"], f"decode_cycle_drift_{frontend}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
