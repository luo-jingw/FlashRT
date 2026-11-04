"""Tail fc kernel check: the shared warp kernel (fma order of the M = 1 path) against cuBLASLt for the MTP
prompt-tail fc (rows x 10240 -> 5120). Per call (one decompose, one monitor) and tail (auto / full) at K = 3: spec
cycles, accepted drafts, decode ms, tail ms, same output; and the fc output difference on the call's tail rows."""
import json
import sys

import torch

import mtp_tail_model as mt
from flash_rt import flash_rt_kernels as fvk


def main(calls: str) -> None:
    bench = mt.TailBench()
    llm = bench.llm
    _, real = mt.prompts(bench, [calls])
    for spec in real:
        spec, ref = bench.reference(spec, 512)
        for tail in ("auto", "full"):
            for fc in ("warp", "cublaslt"):
                print(json.dumps(mt.asdict(bench.cell(spec, tail, fc, "3"))), flush=True)
    hidden = llm._cfg["hidden_size"]
    w = int(llm._weights.ptrs["mtp"]["fc_w"])
    for rows in (1, 512, 2048):
        x = torch.randn(rows, 2 * hidden, device="cuda", dtype=torch.bfloat16)
        a = torch.empty(rows, hidden, device="cuda", dtype=torch.bfloat16)
        b = torch.empty_like(a)
        s = torch.cuda.current_stream().cuda_stream
        fvk.bf16_matmul_qwen36_bf16(x.data_ptr(), w, a.data_ptr(), rows, hidden, 2 * hidden, s)
        mt.cublaslt_fc(x.data_ptr(), w, b.data_ptr(), rows, hidden, s)
        torch.cuda.synchronize()
        d = (a.float() - b.float()).abs()
        print(f"fc rows {rows}: max |warp - cublaslt| {d.max().item():.4g}, mean {d.mean().item():.4g}, "
              f"bitwise equal {(a == b).float().mean().item() * 100:.1f}%, max |out| {a.abs().max().item():.3g}")


if __name__ == "__main__":
    main(sys.argv[1])
