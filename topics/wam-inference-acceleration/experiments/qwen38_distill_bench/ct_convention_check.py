"""Check ct_nvfp4.decode against a published compressed-tensors NVFP4 checkpoint.

Reads one MLP projection from unsloth/Qwen3.8-27B-NVFP4 (NVFP4) and the same projection from Qwen/Qwen3.8-27B-FP8
by HTTP range requests, decodes both, and prints their relative error. A matching nibble order and scale convention
gives an error at the NVFP4 quantization level (about 0.1); a swapped nibble order or an inverted global scale gives
an error near or above 1. Also prints the round-trip error of ct_nvfp4.encode on the FP8 weight.

usage: python ct_convention_check.py [TENSOR_PREFIX]
"""
import json
import struct
import sys
import urllib.request

import torch

import ct_nvfp4

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "model.language_model.layers.0.mlp.gate_proj"
DTYPES = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16}


def _get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req, timeout=300).read()


def read_tensors(repo: str, names: list[str]) -> dict[str, torch.Tensor]:
    base = f"https://huggingface.co/{repo}/resolve/main/"
    idx = json.loads(urllib.request.urlopen(base + "model.safetensors.index.json", timeout=60).read())["weight_map"]
    out = {}
    for name in names:
        url = base + idx[name]
        n = struct.unpack("<Q", _get(url, 0, 7))[0]
        meta = json.loads(_get(url, 8, 7 + n))[name]
        s, e = meta["data_offsets"]
        raw = bytearray(_get(url, 8 + n + s, 8 + n + e - 1))
        out[name] = torch.frombuffer(raw, dtype=DTYPES[meta["dtype"]]).view(meta["shape"]).clone()
    return out


ct = read_tensors("unsloth/Qwen3.8-27B-NVFP4",
                  [PREFIX + ".weight_packed", PREFIX + ".weight_scale", PREFIX + ".weight_global_scale"])
fp8 = read_tensors("Qwen/Qwen3.8-27B-FP8", [PREFIX + ".weight", PREFIX + ".weight_scale_inv"])
dev = "cuda"
ref = ct_nvfp4.dequant_fp8_block128(fp8[PREFIX + ".weight"].to(dev), fp8[PREFIX + ".weight_scale_inv"].to(dev))
q = ct_nvfp4.CtNvfp4Weight(packed=ct[PREFIX + ".weight_packed"].to(dev), scale=ct[PREFIX + ".weight_scale"].to(dev),
                           global_scale=ct[PREFIX + ".weight_global_scale"].reshape(1).to(dev))
swapped = ct_nvfp4.CtNvfp4Weight(packed=((q.packed & 0x0F) << 4) | (q.packed >> 4), scale=q.scale,
                                 global_scale=q.global_scale)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


print(f"{PREFIX}: shape {tuple(ref.shape)}, published global_scale {float(q.global_scale):.4g}, "
      f"448*6/amax {ct_nvfp4.global_scale_for(float(ref.abs().max())):.4g}")
print(f"published NVFP4 decoded vs FP8 reference: rel err {rel(ct_nvfp4.decode(q), ref):.4f}")
print(f"same, nibble order swapped:              rel err {rel(ct_nvfp4.decode(swapped), ref):.4f}")
mine = ct_nvfp4.encode(ref, ct_nvfp4.global_scale_for(float(ref.abs().max())))
print(f"ct_nvfp4.encode round trip vs FP8 reference: rel err {rel(ct_nvfp4.decode(mine), ref):.4f}")
print(f"codes identical to published: {float((mine.packed == q.packed).float().mean()):.4f} of bytes, "
      f"scales identical: {float((mine.scale.view(torch.uint8) == q.scale.view(torch.uint8)).float().mean()):.4f}")
