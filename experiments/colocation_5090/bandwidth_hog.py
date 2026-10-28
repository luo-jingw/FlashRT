"""A planner stand-in whose only demand is DRAM bandwidth: back-to-back device-to-device copies on one stream.

BandwidthHog(stream).run(t_end) copies a BUF_MB buffer into another until wall-clock t_end and returns the achieved
bandwidth (bytes read + written per second, CUDA-event timed on the hog's stream). Put on a green-context partition
(colocation_bench.py greenhog<P>), it gives the executor's slowdown from bandwidth alone on disjoint SMs, with a
measured demand instead of a derived one.

usage as a probe: python bandwidth_hog.py SM_COUNT   (hog bandwidth on the whole device and on SM_COUNT SMs)
"""
import sys
import time
from dataclasses import dataclass

import torch

BUF_MB = 256
COPIES_PER_EVENT = 8


@dataclass
class HogRecord:
    copies: int
    seconds: float
    GBps: float


class BandwidthHog:
    def __init__(self, stream: torch.cuda.Stream):
        self.stream = stream
        with torch.cuda.stream(stream):
            self.src = torch.empty(BUF_MB << 20, dtype=torch.uint8, device="cuda").fill_(1)
            self.dst = torch.empty_like(self.src)
        stream.synchronize()

    def run(self, t_end: float) -> HogRecord:
        copies, ms = 0, 0.0
        with torch.cuda.stream(self.stream):
            while time.time() < t_end:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(self.stream)
                for _ in range(COPIES_PER_EVENT):
                    self.dst.copy_(self.src)
                e1.record(self.stream)
                e1.synchronize()
                copies += COPIES_PER_EVENT
                ms += e0.elapsed_time(e1)
        return HogRecord(copies, ms / 1e3, 2 * copies * self.src.numel() / (ms / 1e3) / 1e9 if ms else 0.0)


if __name__ == "__main__":
    from green_partition import GreenPartition
    p = GreenPartition(int(sys.argv[1]))
    for name, s in (("device", torch.cuda.Stream()), (f"part {p.part.sm_count} SMs", p.part.stream),
                    (f"rest {p.rest.sm_count} SMs", p.rest.stream)):
        r = BandwidthHog(s).run(time.time() + 3)
        print(f"{name}: {r.GBps:.1f} GB/s ({r.copies} copies of {BUF_MB} MB)")
