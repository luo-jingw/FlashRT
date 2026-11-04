"""The MTP prompt-tail fc (rows x 10240 -> 5120, bf16, random weights) on each kernel the frontends can dispatch:
the Thor M-tile kernel (`bf16_matmul_qwen36_thor_mtp_fc_bf16`, used by Qwen36TorchFrontendThor; it returns non-zero
on a device without its 160 KB of shared memory per block), the shared warp kernel (`bf16_matmul_qwen36_bf16`) and
cuBLASLt with the heuristic's first algorithm (`bf16_matmul_cublaslt_bf16`, max_algos=1, the RTX frontend's
default). Median of 20 timed calls after 3 warm-ups (CUDA events), and the largest absolute difference of each
output from the warp kernel's.

usage (cwd = repository root): python research/experiments/harness_replay_5090/thor/mtp_tail_fc_kernels.py
"""
import torch

from flash_rt import flash_rt_kernels as fvk

H = 5120
ROWS = (512, 2048, 4071)


def median_ms(fn, x: torch.Tensor, o: torch.Tensor, rows: int) -> float:
    for _ in range(3):
        fn(x, o, rows)
    ts = []
    for _ in range(20):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn(x, o, rows)
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[10]


def main() -> None:
    torch.manual_seed(0)
    w = (torch.randn(H, 2 * H, device="cuda") * 0.02).to(torch.bfloat16)
    s = torch.cuda.current_stream().cuda_stream
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())

    def thor_mtile(x: torch.Tensor, o: torch.Tensor, r: int) -> int:
        return fvk.bf16_matmul_qwen36_thor_mtp_fc_bf16(x.data_ptr(), w.data_ptr(), o.data_ptr(), r, H, s)

    def warp(x: torch.Tensor, o: torch.Tensor, r: int) -> None:
        fvk.bf16_matmul_qwen36_bf16(x.data_ptr(), w.data_ptr(), o.data_ptr(), r, H, 2 * H, s)

    def cublaslt_algos1(x: torch.Tensor, o: torch.Tensor, r: int) -> None:
        fvk.bf16_matmul_cublaslt_bf16(x.data_ptr(), w.data_ptr(), o.data_ptr(), r, H, 2 * H, s, 1)

    for rows in ROWS:
        x = torch.randn(rows, 2 * H, device="cuda").to(torch.bfloat16)
        ref = torch.empty(rows, H, device="cuda", dtype=torch.bfloat16)
        warp(x, ref, rows)
        torch.cuda.synchronize()
        parts = []
        o = torch.empty_like(ref)
        rc = thor_mtile(x, o, rows)
        torch.cuda.synchronize()
        if rc == 0:
            diff = (o.float() - ref.float()).abs().max().item()
            parts.append(f"thor_mtile {median_ms(thor_mtile, x, o, rows):.3f} ms (max |diff| {diff:.3e})")
        else:
            parts.append(f"thor_mtile unavailable (rc {rc})")
        parts.append(f"warp {median_ms(warp, x, o, rows):.3f} ms")
        cublaslt_algos1(x, o, rows)
        torch.cuda.synchronize()
        diff = (o.float() - ref.float()).abs().max().item()
        parts.append(f"cublaslt_algos1 {median_ms(cublaslt_algos1, x, o, rows):.3f} ms (max |diff| {diff:.3e})")
        print(f"rows {rows}: " + ", ".join(parts), flush=True)


if __name__ == "__main__":
    main()
