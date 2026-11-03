"""Convert a Qwen3.5-architecture (Qwen3.6 / Qwen3.8 27B) block-128 FP8 checkpoint into the compressed-tensors NVFP4
layout read by the FlashRT Qwen3.6 NVFP4 frontend.

Quant scope (the layout that frontend requires):
  NVFP4   mlp.{gate,up,down}_proj, self_attn.{q,k,v,o}_proj; gate and up share one global scale
  BF16    linear_attn.{in_proj_qkv,in_proj_z,out_proj} (dequantized from FP8); every tensor stored as BF16 is copied,
          the vision tower (``model.visual.*``) included
Skipped: the MTP head (``mtp.*``; the frontend reads it from ``FLASHRT_QWEN36_MTP_CKPT_DIR``). Output:
``model.safetensors`` plus the source directory's config, tokenizer and processor files; ``quantization_config`` is
removed from config.json.

Each NVFP4 weight is encoded from the FP8 weight dequantized to float32, so the result carries FP8 and NVFP4 rounding.
Prints per-kind counts and the mean / max relative error of NVFP4 against the dequantized FP8 weight.

usage: python fp8_to_nvfp4_ct.py SRC_DIR DST_DIR
"""
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import ct_nvfp4

NVFP4_RE = re.compile(r"\.(mlp\.(gate|up|down)_proj|self_attn\.(q|k|v|o)_proj)$")
BF16_RE = re.compile(r"\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$")
SKIP_PREFIXES = ("mtp.",)
SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "vocab.json", "merges.txt", "special_tokens_map.json", "processor_config.json", "preprocessor_config.json",
              "video_preprocessor_config.json")


@dataclass
class ConversionLog:
    nvfp4: int = 0
    fp8_to_bf16: int = 0
    copied: int = 0
    skipped: int = 0
    rel_errs: list[float] = field(default_factory=list)


class SourceCheckpoint:
    """Tensor lookup across the shards of a safetensors checkpoint directory."""

    def __init__(self, src: str) -> None:
        idx_path = os.path.join(src, "model.safetensors.index.json")
        if os.path.isfile(idx_path):
            weight_map = json.load(open(idx_path))["weight_map"]
        else:
            weight_map = {}
            for fn in sorted(f for f in os.listdir(src) if f.endswith(".safetensors")):
                with safe_open(os.path.join(src, fn), framework="pt") as f:
                    weight_map.update({k: fn for k in f.keys()})
        self.weight_map: dict[str, str] = weight_map
        self._handles = {fn: safe_open(os.path.join(src, fn), framework="pt") for fn in set(weight_map.values())}

    def get(self, name: str) -> torch.Tensor:
        return self._handles[self.weight_map[name]].get_tensor(name)


def fp8_weight(src: SourceCheckpoint, base: str, dev: str) -> torch.Tensor:
    return ct_nvfp4.dequant_fp8_block128(src.get(base + ".weight").to(dev), src.get(base + ".weight_scale_inv").to(dev))


def emit_nvfp4(out: dict[str, torch.Tensor], base: str, w: torch.Tensor, gscale: float, log: ConversionLog) -> None:
    q = ct_nvfp4.encode(w, gscale)
    log.rel_errs.append(float((ct_nvfp4.decode(q) - w).norm() / w.norm()))
    out[base + ".weight_packed"] = q.packed.cpu()
    out[base + ".weight_scale"] = q.scale.cpu()
    out[base + ".weight_global_scale"] = q.global_scale.cpu()
    log.nvfp4 += 1


def convert(src_dir: str, dst_dir: str, dev: str = "cuda") -> ConversionLog:
    src = SourceCheckpoint(src_dir)
    log = ConversionLog()
    out: dict[str, torch.Tensor] = {}
    names = sorted(src.weight_map)
    fp8_bases = {n[: -len(".weight")] for n in names if n.endswith(".weight") and n[: -len(".weight")] + ".weight_scale_inv" in src.weight_map}
    done: set[str] = set()
    for name in names:
        if name.startswith(SKIP_PREFIXES):
            log.skipped += 1
            continue
        if name.endswith(".weight_scale_inv"):
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        if base in fp8_bases and NVFP4_RE.search(base):
            if base in done:
                continue
            if base.endswith(("gate_proj", "up_proj")):
                gate, up = re.sub(r"(gate|up)_proj$", "gate_proj", base), re.sub(r"(gate|up)_proj$", "up_proj", base)
                wg, wu = fp8_weight(src, gate, dev), fp8_weight(src, up, dev)
                gscale = ct_nvfp4.global_scale_for(max(float(wg.abs().max()), float(wu.abs().max())))
                emit_nvfp4(out, gate, wg, gscale, log)
                emit_nvfp4(out, up, wu, gscale, log)
                done.update((gate, up))
                del wg, wu
            else:
                w = fp8_weight(src, base, dev)
                emit_nvfp4(out, base, w, ct_nvfp4.global_scale_for(float(w.abs().max())), log)
                done.add(base)
                del w
        elif base in fp8_bases and BF16_RE.search(base):
            out[name] = fp8_weight(src, base, dev).to(torch.bfloat16).cpu()
            log.fp8_to_bf16 += 1
        elif base in fp8_bases:
            raise RuntimeError(f"FP8 tensor outside the conversion scope: {name}")
        else:
            out[name] = src.get(name).contiguous()
            log.copied += 1
    os.makedirs(dst_dir, exist_ok=True)
    save_file(out, os.path.join(dst_dir, "model.safetensors"),
              metadata={"format": "pt", "source": os.path.basename(os.path.normpath(src_dir)),
                        "conversion": "fp8 block128 -> nvfp4 (mlp, full-attn qkvo) / bf16 (linear-attn proj)"})
    for fn in SIDE_FILES:
        p = os.path.join(src_dir, fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst_dir, fn))
    cfg_path = os.path.join(dst_dir, "config.json")
    cfg = json.load(open(cfg_path))
    cfg.pop("quantization_config", None)
    cfg.get("text_config", {}).pop("quantization_config", None)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    return log


if __name__ == "__main__":
    lg = convert(sys.argv[1], sys.argv[2])
    errs = torch.tensor(lg.rel_errs)
    print(f"nvfp4 {lg.nvfp4}, fp8->bf16 {lg.fp8_to_bf16}, copied {lg.copied}, skipped {lg.skipped}; "
          f"nvfp4 rel err vs fp8 mean {float(errs.mean()):.4f} max {float(errs.max()):.4f}")
    print(f"wrote {os.path.join(sys.argv[2], 'model.safetensors')} "
          f"{os.path.getsize(os.path.join(sys.argv[2], 'model.safetensors')) / 1e9:.2f} GB")
