"""Kernel times of one MTP prompt-tail K/V prefill (rows = 512 / 2048) after a decompose call, torch.profiler."""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

import mtp_tail_model as mt


def main(calls: str) -> None:
    bench = mt.TailBench()
    _, real = mt.prompts(bench, [calls])
    spec, _ = bench.reference(real[0], 64)
    bench.cell(spec, "auto", "warp", "3")
    llm = bench.llm
    hidden = llm._cfg["hidden_size"]
    for rows in (512, 2048):
        h = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, 1000, (rows,), device="cuda", dtype=torch.long)
        with torch.inference_mode():
            llm._prefill_mtp_tail_kv_nvfp4(h, ids, 1, 0)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                llm._prefill_mtp_tail_kv_nvfp4(h, ids, 1, 0)
                torch.cuda.synchronize()
        print(f"rows {rows}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=70))


if __name__ == "__main__":
    main(sys.argv[1])
