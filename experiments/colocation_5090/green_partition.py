"""In-process SM partition with CUDA green contexts (driver API, CUDA >= 12.4), for devices or setups without MPS.

GreenPartition(sm_count) splits the device's SMs into one group of at least `sm_count` SMs (rounded up to the
device's SM granularity) and the remainder, makes a green context on each, and exposes one torch stream per group.
Kernels launched on a group's stream (eager launches and CUDA-graph replays alike) run only on that group's SMs;
memory is shared with the rest of the process.

usage as a probe: python green_partition.py SM_COUNT   (prints the split and runs a matmul on each stream)
"""
import sys
from dataclasses import dataclass

import torch
from cuda.bindings import driver as cu


def _ok(ret: tuple) -> object:
    err, *vals = ret if isinstance(ret, tuple) else (ret,)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}")
    return vals[0] if len(vals) == 1 else vals


@dataclass
class GreenGroup:
    sm_count: int
    stream: torch.cuda.ExternalStream


class GreenPartition:
    def __init__(self, sm_count: int, device: int = 0, priority: int = 0):
        torch.cuda.init()
        _ok(cu.cuInit(0))
        dev = _ok(cu.cuDeviceGet(device))
        whole = _ok(cu.cuDeviceGetDevResource(dev, cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
        self.device_sm_count = whole.sm.smCount
        groups, n, rest = _ok(cu.cuDevSmResourceSplitByCount(1, whole, 0, sm_count))
        self._ctx, self.groups = [], []
        for res in (groups[0], rest):
            desc = _ok(cu.cuDevResourceGenerateDesc([res], 1))
            g = _ok(cu.cuGreenCtxCreate(desc, dev, cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
            s = _ok(cu.cuGreenCtxStreamCreate(g, cu.CUstream_flags.CU_STREAM_NON_BLOCKING, priority))
            self._ctx.append(g)
            self.groups.append(GreenGroup(res.sm.smCount, torch.cuda.ExternalStream(int(s))))

    @property
    def part(self) -> GreenGroup:
        return self.groups[0]

    @property
    def rest(self) -> GreenGroup:
        return self.groups[1]


if __name__ == "__main__":
    p = GreenPartition(int(sys.argv[1]))
    print(f"device SMs {p.device_sm_count}; part {p.part.sm_count} SMs, rest {p.rest.sm_count} SMs")
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for name, g in (("part", p.part), ("rest", p.rest)):
        with torch.cuda.stream(g.stream):
            a @ a
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(10):
                a @ a
            e1.record()
        e1.synchronize()
        print(f"{name}: {g.sm_count} SMs, 8192^3 bf16 matmul {e0.elapsed_time(e1) / 10:.2f} ms")
