"""Tail fc (rows x 10240 -> 5120, bf16, random weights) on the frontend's default (cuBLASLt, max_algos=1) against the
shared warp kernel and cuBLASLt with the 8-candidate autotune: median of 20 timed calls after 3 warm-ups (CUDA
events). Output: smoke/mtp_tail_fc_default.log."""
import torch

from flash_rt import flash_rt_kernels as fvk

H = 5120


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
    fns = {"warp": lambda x, o, r: fvk.bf16_matmul_qwen36_bf16(x.data_ptr(), w.data_ptr(), o.data_ptr(), r, H, 2 * H, s),
           "cublaslt_algos1": lambda x, o, r: fvk.bf16_matmul_cublaslt_bf16(x.data_ptr(), w.data_ptr(), o.data_ptr(),
                                                                            r, H, 2 * H, s, 1),
           "cublaslt_autotune8": lambda x, o, r: fvk.bf16_matmul_cublaslt_bf16(x.data_ptr(), w.data_ptr(),
                                                                               o.data_ptr(), r, H, 2 * H, s, 8)}
    print(torch.cuda.get_device_name())
    for rows in (512, 2048, 4071):
        x = torch.randn(rows, 2 * H, device="cuda").to(torch.bfloat16)
        o = torch.empty(rows, H, device="cuda", dtype=torch.bfloat16)
        print(f"rows {rows}: " + ", ".join(f"{k} {median_ms(f, x, o, rows):.3f} ms" for k, f in fns.items()))


if __name__ == "__main__":
    main()
