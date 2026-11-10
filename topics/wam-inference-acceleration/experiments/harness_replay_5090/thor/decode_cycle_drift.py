"""Decode ms per speculative cycle of the Qwen3.8-27B NVFP4 planner (official MTP head) over time in one process, with
the clock readings the machine exposes without root and a copy-bandwidth probe. The monitor call shape (4 072 prompt
tokens of the repository's documentation, 53 new tokens, the prompt-length table's K) is generated back to back for
RUN_S seconds; then, for each gap in IDLE_S (a comma-separated list of seconds), the process leaves the GPU idle for
that gap and generates back to back for AFTER_S more seconds. Per generation one line: phase ("run", or "after<gap>s"
for the generations after a gap), seconds since the first generation, prefill and decode ms
(the frontend's CUDA-event windows), spec cycles, decode ms per cycle, the copy probe (device-to-device copy of a
PROBE_BYTES buffer, read + write bytes / time, median of PROBE_REPS, GB/s), the CPU cores' current frequencies
(cpufreq, kHz) and every devfreq device's current frequency (Hz). Output: $OUT/decode_cycle_drift_<frontend>.json.

usage (cwd = repository root): CK=<dir with qwen38-27b-nvfp4-official and qwen38-27b-fp8-official> OUT=<dir> \
  python research/topics/wam-inference-acceleration/experiments/harness_replay_5090/thor/decode_cycle_drift.py {rtx|thor} [RUN_S [IDLE_S [AFTER_S]]]
(defaults 300 0 0; IDLE_S e.g. 300 or 5,15,30,60,120)
"""
import glob
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass

CK = os.environ["CK"]
PROMPT_TOKENS = 4072
NEW_TOKENS = 53
MAX_SEQ = PROMPT_TOKENS + 128
PROBE_BYTES = 1 << 30
PROBE_REPS = 5
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
    phase: str
    t_s: float
    prefill_ms: float
    decode_ms: float
    cycles: int
    ms_per_cycle: float
    copy_gbps: float
    cpu_khz: list
    devfreq_hz: dict


class CopyProbe:
    """Device-to-device copy of PROBE_BYTES; GB/s counts the bytes read and written."""

    def __init__(self) -> None:
        self.src = torch.empty(PROBE_BYTES, dtype=torch.uint8, device="cuda")
        self.dst = torch.empty_like(self.src)

    def gbps(self) -> float:
        rates = []
        for _ in range(PROBE_REPS):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            self.dst.copy_(self.src)
            e1.record()
            e1.synchronize()
            rates.append(2 * PROBE_BYTES / (e0.elapsed_time(e1) * 1e6))
        return statistics.median(rates)


def read_int(path: str) -> int:
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return -1


def cpu_khz() -> list:
    return [read_int(p) for p in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))]


def devfreq_hz() -> dict:
    return {os.path.basename(os.path.dirname(p)): read_int(p) for p in sorted(glob.glob("/sys/class/devfreq/*/cur_freq"))}


def generate_for(fe: Qwen36TorchFrontendRtx, ids: torch.Tensor, probe: CopyProbe, phase: str, t0: float,
                 until_s: float, samples: list) -> None:
    while not samples or samples[-1].phase != phase or samples[-1].t_s < until_s:
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=NEW_TOKENS)
        cyc = int(fe._spec_attempts)
        dc = float(fe._long_ctx_decode_ms)
        s = Sample(phase, time.perf_counter() - t0, float(fe._long_ctx_prefill_ms), dc, cyc, dc / cyc if cyc else 0.0,
                   probe.gbps(), cpu_khz(), devfreq_hz())
        samples.append(s)
        print(json.dumps(asdict(s)), flush=True)


def main(frontend: str, run_s: float, idle_s: list[float], after_s: float) -> None:
    fe = FRONTENDS[frontend](CK + "/qwen38-27b-nvfp4-official", device="cuda:0", max_seq=MAX_SEQ, quant="nvfp4")
    tok = fe._tokenizer
    print("loaded:", type(fe).__name__, "long_ctx_mode", fe._long_ctx_mode, "mtp",
          fe._weights.ptrs.get("mtp") is not None, "devfreq", list(devfreq_hz()),
          "run/idle/after s", run_s, idle_s, after_s, flush=True)
    files = ["README.md"] + sorted(glob.glob("docs/*.md"))
    doc = tok.encode("\n\n".join(open(f, encoding="utf-8").read() for f in files), add_special_tokens=False)
    ids = torch.tensor([doc[:PROMPT_TOKENS]], dtype=torch.long, device="cuda")
    probe = CopyProbe()
    samples: list = []
    t0 = time.perf_counter()
    generate_for(fe, ids, probe, "run", t0, run_s, samples)
    for gap in idle_s if after_s > 0 else []:
        torch.cuda.synchronize()
        print(f"idle {gap:.0f} s", flush=True)
        time.sleep(gap)
        generate_for(fe, ids, probe, f"after{gap:g}s", t0, samples[-1].t_s + gap + after_s, samples)
    json.dump({"frontend": frontend, "gpu": torch.cuda.get_device_name(), "prompt_tokens": PROMPT_TOKENS,
               "new_tokens": NEW_TOKENS, "run_s": run_s, "idle_s": idle_s, "after_s": after_s,
               "probe_bytes": PROBE_BYTES, "samples": [asdict(x) for x in samples]},
              open(os.path.join(os.environ["OUT"], f"decode_cycle_drift_{frontend}.json"), "w"), indent=1)


if __name__ == "__main__":
    args = sys.argv[2:] + ["300", "0", "0"][len(sys.argv[2:]):]
    main(sys.argv[1], float(args[0]), [float(g) for g in args[1].split(",")], float(args[2]))
