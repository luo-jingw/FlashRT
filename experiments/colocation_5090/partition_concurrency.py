"""Do two SM partitions run at the same time, or take turns? A probe without DRAM-bandwidth demand.

Two loads on two streams of one process, each in its own thread:
  compute  back-to-back 2048^3 BF16 matmuls (operands stay in L2; compute-bound), throughput in TFLOP/s;
  long     one LONG_K-deep 8192 x 8192 BF16 matmul (a single long kernel) on the first stream, while the second stream
           runs small 512^3 matmuls one at a time (launch, synchronize); the longest wait of a small matmul is how long
           the long kernel holds it up.
Each load runs on: green contexts (green_partition.py, first stream on PART_SM SMs, second on the rest) and two plain
streams of one context (second stream at high priority). For compute, each stream alone and both together: together
close to alone means the partitions run at the same time; together close to half of alone means they take turns.

usage: python partition_concurrency.py PART_SM OUT_JSON
"""
import json
import sys
import threading
import time

import torch

from green_partition import GreenPartition

PART_SM, OUT = int(sys.argv[1]), sys.argv[2]
SECONDS = 3.0
LONG_K = 65536


def compute(stream: torch.cuda.Stream, t_end: float, out: dict) -> None:
    with torch.cuda.stream(stream):
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        n, ms = 0, 0.0
        while time.time() < t_end:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(stream)
            for _ in range(20):
                a @ a
            e1.record(stream)
            e1.synchronize()
            n, ms = n + 20, ms + e0.elapsed_time(e1)
    out["TFLOPs"] = 2 * 2048 ** 3 * n / (ms / 1e3) / 1e12


def small_waits(stream: torch.cuda.Stream, t_end: float, out: dict) -> None:
    with torch.cuda.stream(stream):
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        waits = []
        while time.time() < t_end:
            t = time.time()
            a @ a
            stream.synchronize()
            waits.append((time.time() - t) * 1e3)
    waits.sort()
    out.update(n=len(waits), p50_ms=waits[len(waits) // 2], max_ms=waits[-1])


def long_kernel(stream: torch.cuda.Stream, out: dict) -> None:
    with torch.cuda.stream(stream):
        a = torch.randn(8192, LONG_K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(LONG_K, 8192, device="cuda", dtype=torch.bfloat16)
        a @ b
        stream.synchronize()
        time.sleep(0.5)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(stream)
        a @ b
        e1.record(stream)
        e1.synchronize()
    out["long_ms"] = e0.elapsed_time(e1)


def together(*jobs: tuple) -> None:
    ts = [threading.Thread(target=f, args=a) for f, *a in jobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


g = GreenPartition(PART_SM)
setups = {"green": (g.part.stream, g.rest.stream),
          "streams": (torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=-1))}
res = {"device_sm": g.device_sm_count, "part_sm": g.part.sm_count, "rest_sm": g.rest.sm_count}
for name, (s1, s2) in setups.items():
    r = {}
    for key, s in (("first_alone", s1), ("second_alone", s2)):
        r[key] = {}
        compute(s, time.time() + SECONDS, r[key])
    r["first_together"], r["second_together"] = {}, {}
    t_end = time.time() + SECONDS
    together((compute, s1, t_end, r["first_together"]), (compute, s2, t_end, r["second_together"]))
    r["small_alone"] = {}
    small_waits(s2, time.time() + 1.0, r["small_alone"])
    r["long"], r["small_during_long"] = {}, {}
    together((long_kernel, s1, r["long"]), (small_waits, s2, time.time() + 4.0, r["small_during_long"]))
    res[name] = r
    print(f"{name}: compute TFLOP/s first {r['first_alone']['TFLOPs']:.1f} alone / {r['first_together']['TFLOPs']:.1f} "
          f"together, second {r['second_alone']['TFLOPs']:.1f} / {r['second_together']['TFLOPs']:.1f}; long kernel "
          f"{r['long']['long_ms']:.0f} ms; small matmul wait p50 / max {r['small_alone']['p50_ms']:.2f} / "
          f"{r['small_alone']['max_ms']:.2f} ms alone, {r['small_during_long']['p50_ms']:.2f} / "
          f"{r['small_during_long']['max_ms']:.1f} ms during the long kernel", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"device {g.device_sm_count} SMs: first {g.part.sm_count}, second {g.rest.sm_count}")
