#!/usr/bin/env python3
"""Byte-exact check of the block-scaled quantizers without a Blackwell GPU.

Compiles the unmodified quantizer sources (`quantize_e0m3_sfa.cu`,
`quantize_fp4_sfa.cu`, `fused_fp4/pi05_e0m3_act.cu`) with the flags of
the `fp4_kernels_obj` target (`--use_fast_math`, `-O3`) for the local
GPU, together with `tools/blockscaled_quantizers_shim.cu`, and compares
their packed codes and tile-interleaved scale bytes with
`flash_rt.models.imagewam.blockscaled_ref`:

- weights: real ImageWAM checkpoint tensors (`CKPT_PATH`; 130M elements),
  E0M3 plain, E0M3 after the per-16 Hadamard rotation, E0M3 of the
  served `e0m3_hadamard` weight preparation
  (`prepare_e0m3_hadamard_weight`: butterfly, per-tensor 2^e, fp16),
  NVFP4 amax, NVFP4 MSE;
- activations: random fp16 with outlier columns and near-zero rows,
  E0M3 with and without the rotation (`quantize_e0m3_dynamic_sfa_fp16_vec`)
  and NVFP4;
- NVFP4 on blocks built on the E2M1 rounding thresholds (about half the
  elements scale exactly onto a threshold), in both the SFB and the SFA
  layout: the kernel's reciprocal is `rcp.approx.ftz` and its `amax / 6`
  is `div.approx` under `--use_fast_math`, the reference's are
  round-to-nearest.

Prints the mismatch count of every case; exits 1 if any case other than
NVFP4 MSE mismatches (MSE differs at ties of its sequential,
FMA-contracted error sum). Needs nvcc (sm_89 or newer target), the CUTLASS
checkout, and CUDA torch.

  python tools/check_blockscaled_quantizers_sm90.py --arch 90a
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile

import torch

from flash_rt.models.imagewam.blockscaled_ref import (
    BlockQuantized,
    fwht16_butterfly,
    pack_codes,
    pack_scales,
    prepare_e0m3_hadamard_weight,
    quantize_blocks,
    rotate_k_blocks,
    sf_size_bytes,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["csrc/quantize/quantize_e0m3_sfa.cu", "csrc/quantize/quantize_fp4_sfa.cu",
           "csrc/fused_fp4/pi05_e0m3_act.cu", "tools/blockscaled_quantizers_shim.cu"]
WEIGHT_KEYS = ["mixtures.video.single_blocks.3.linear1.weight", "mixtures.video.double_blocks.1.img_mlp.2.weight",
               "mixtures.action.single_blocks.7.linear2.weight", "mixtures.action.double_blocks.2.img_attn.qkv.weight"]
DEV = "cuda"


def build(arch: str, cutlass: str, out_dir: str) -> str:
    lib = os.path.join(out_dir, "libblockscaled_shim.so")
    cmd = [os.environ.get("NVCC", "nvcc"), "-std=c++17", "-O3", "--use_fast_math", "--expt-relaxed-constexpr",
           f"-gencode=arch=compute_{arch},code=sm_{arch}", "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
           "-Xcompiler", "-fPIC", "-shared", f"-I{ROOT}/csrc", f"-I{cutlass}/include",
           f"-I{cutlass}/tools/util/include", *[os.path.join(ROOT, s) for s in SOURCES], "-o", lib]
    subprocess.run(cmd, check=True)
    return lib


def e2m1_threshold_blocks(rows: int, k: int, seed: int) -> torch.Tensor:
    """Blocks with amax = 6*s (s any UE4M3 value) whose other elements are
    +-t*s, t an E2M1 threshold or grid value; fp16-exact."""
    g = torch.Generator().manual_seed(seed)
    scales = torch.arange(1, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    s = scales[torch.randint(0, scales.numel(), (rows, k // 16), generator=g)]
    levels = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    t = levels[torch.randint(0, levels.numel(), (rows, k // 16, 16), generator=g)]
    t[..., 0] = 6.0
    sign = torch.where(torch.rand(rows, k // 16, 16, generator=g) < 0.5, -1.0, 1.0)
    return (t * sign * s.unsqueeze(-1)).reshape(rows, k).to(torch.float16).to(DEV)


class Shim:
    def __init__(self, path: str) -> None:
        self.lib = ctypes.CDLL(path)
        for fn in ("shim_e0m3_w", "shim_e0m3_vec", "shim_nvfp4", "shim_nvfp4_mse"):
            getattr(self.lib, fn).argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 3

    def run(self, fn: str, x: torch.Tensor, last: int) -> tuple[torch.Tensor, torch.Tensor]:
        n, k = x.shape
        packed = torch.zeros(n, k // 2, dtype=torch.uint8, device=DEV)
        sf = torch.zeros(sf_size_bytes(n, k), dtype=torch.uint8, device=DEV)
        rc = getattr(self.lib, fn)(x.data_ptr(), packed.data_ptr(), sf.data_ptr(), n, k, last)
        if rc != 0:
            raise RuntimeError(f"{fn} rc={rc}")
        return packed, sf


def compare(tag: str, got: tuple[torch.Tensor, torch.Tensor], ref: BlockQuantized) -> int:
    packed, sf = got
    bad_c = int((packed != pack_codes(ref.codes)).sum())
    bad_s = int((sf != pack_scales(ref.scale_bytes)).sum())
    print(f"{tag:64s} codes {bad_c}/{packed.numel()}  scales {bad_s}/{sf.numel()}")
    return bad_c + bad_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="90a")
    ap.add_argument("--cutlass", default=os.path.join(ROOT, "third_party", "cutlass"))
    ap.add_argument("--ckpt", default=os.environ.get("CKPT_PATH"))
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        shim = Shim(build(args.arch, args.cutlass, tmp))
        exact_bad = 0
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)["mot"]
            for key in WEIGHT_KEYS:
                w = sd[key].to(torch.float16).to(DEV).contiguous()
                n, k = w.shape
                wr = rotate_k_blocks(w.float(), 16).half().contiguous()
                wg, alpha = prepare_e0m3_hadamard_weight(w)
                print(f"{key} [{n}, {k}]")
                exact_bad += compare("  weight e0m3", shim.run("shim_e0m3_w", w, 1), quantize_blocks(w, "e0m3"))
                exact_bad += compare("  weight e0m3, H16-rotated", shim.run("shim_e0m3_w", wr, 1),
                                     quantize_blocks(wr, "e0m3"))
                exact_bad += compare(f"  weight e0m3, served prep (alpha={alpha:g})", shim.run("shim_e0m3_w", wg, 1),
                                     quantize_blocks(wg, "e0m3"))
                exact_bad += compare("  weight nvfp4 amax", shim.run("shim_nvfp4", w, 1), quantize_blocks(w, "e2m1"))
                compare("  weight nvfp4 mse (sum-order and fma ties expected)", shim.run("shim_nvfp4_mse", w, 1),
                        quantize_blocks(w, "e2m1", "mse"))
        g = torch.Generator().manual_seed(0)
        for m, k in ((905, 3072), (905, 9216), (64, 1024), (64, 4096)):
            x = torch.randn(m, k, generator=g)
            x[:, torch.randint(0, k, (k // 256,), generator=g)] *= 300.0
            x[:5] *= 1e-3
            x = x.half().to(DEV)
            print(f"activation [{m}, {k}]")
            exact_bad += compare("  e0m3 + H16 (vec)", shim.run("shim_e0m3_vec", x, 1),
                                 quantize_blocks(fwht16_butterfly(x), "e0m3"))
            exact_bad += compare("  e0m3 (vec)", shim.run("shim_e0m3_vec", x, 0), quantize_blocks(x, "e0m3"))
            exact_bad += compare("  nvfp4", shim.run("shim_nvfp4", x, 0), quantize_blocks(x, "e2m1"))
        ties = e2m1_threshold_blocks(4096, 1024, 0)
        print("nvfp4 on E2M1 thresholds [4096, 1024]")
        exact_bad += compare("  SFB layout (weight)", shim.run("shim_nvfp4", ties, 1), quantize_blocks(ties, "e2m1"))
        exact_bad += compare("  SFA layout (activation)", shim.run("shim_nvfp4", ties, 0), quantize_blocks(ties, "e2m1"))
    print(f"mismatches in byte-exact cases: {exact_bad}")
    return 1 if exact_bad else 0


if __name__ == "__main__":
    sys.exit(main())
