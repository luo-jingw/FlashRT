"""Which arithmetic reproduces the stored NVFP4 codes, and which one reproduces the Thor H8 copy's differences.

`ct_nvfp4.encode` divides by Python scalars (`amax / E2M1_MAX`, `scale / global_scale`) and by a tensor
(`blocks / step`), then buckets |scaled| against the E2M1 midpoints with `bucketize(right=False)`. The FP8 source is
coarse, so many values sit exactly on a midpoint, and a one-ulp change of `step` moves them to the other code. This
script re-encodes every NVFP4 tensor of the official FP8 checkpoint on the CPU with five arithmetic variants and
compares the packed codes of each with the bytes stored in a converted checkpoint:

  exact        every division is a correctly rounded division
  rs           division by a Python scalar is a multiply by its float32 reciprocal (both scalar divisions)
  rs_gs        only `scale / global_scale` uses the float32 reciprocal
  rs_amax      only `amax / E2M1_MAX` uses the float32 reciprocal
  rs_all       as rs, and `blocks / step` is also `blocks * (1 / step)`

Per variant: tensors whose codes differ from the stored ones, by kind (down / gate / up / o / k / q / v). The Thor H8
copy differs from the RTX 5090 copy in 63 tensors: down 20, gate 12, up 12, o 7, k 5, q 4, v 3.

usage: python encode_variants_check.py FP8_SRC_DIR NVFP4_MODEL_SAFETENSORS OUT_TSV
"""
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "wam-inference-acceleration", "experiments", "qwen38_distill_bench"))

import ct_nvfp4  # noqa: E402
from fp8_to_nvfp4_ct import NVFP4_RE, SKIP_PREFIXES, SourceCheckpoint, fp8_weight  # noqa: E402

VARIANTS = ("exact", "rs", "rs_gs", "rs_amax", "rs_all")
KINDS = ("down", "gate", "up", "o", "k", "q", "v")


@dataclass
class TensorRow:
    name: str
    kind: str
    differing_codes: dict[str, int]


def scalar_div(x: torch.Tensor, b: float, reciprocal: bool) -> torch.Tensor:
    if reciprocal:
        return x * (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(b, dtype=torch.float32))
    return x / torch.tensor(b, dtype=torch.float32)


def codes(w: torch.Tensor, gscale: float, variant: str) -> torch.Tensor:
    out_f, in_f = w.shape
    blocks = w.float().view(out_f, in_f // ct_nvfp4.BLOCK, ct_nvfp4.BLOCK)
    amax = blocks.abs().amax(dim=-1)
    r_amax = variant in ("rs", "rs_amax", "rs_all")
    r_gs = variant in ("rs", "rs_gs", "rs_all")
    scale = (scalar_div(amax, ct_nvfp4.E2M1_MAX, r_amax) * torch.tensor(gscale, dtype=torch.float32)
             ).clamp(max=ct_nvfp4.E4M3_MAX).to(torch.float8_e4m3fn)
    step = scalar_div(scale.float(), gscale, r_gs)
    st = step.clamp(min=1e-30).unsqueeze(-1)
    q = blocks * (1.0 / st) if variant == "rs_all" else blocks / st
    scaled = torch.where(step.unsqueeze(-1) > 0, q, torch.zeros_like(blocks))
    mids = torch.tensor(ct_nvfp4._MIDPOINTS, dtype=torch.float32)
    mag = torch.bucketize(scaled.abs().contiguous(), mids, right=False).to(torch.uint8)
    c = (mag | ((scaled < 0) & (mag > 0)).to(torch.uint8) << 3).view(out_f, in_f)
    return c


def unpack(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack([packed & 0x0F, packed >> 4], dim=-1).view(packed.shape[0], -1)


def main() -> None:
    src_dir, ref_path, out_tsv = sys.argv[1:4]
    src = SourceCheckpoint(src_dir)
    ref = safe_open(ref_path, framework="pt")
    names = sorted(n for n in src.weight_map if not n.startswith(SKIP_PREFIXES))
    bases = sorted({n[: -len(".weight")] for n in names if n.endswith(".weight")
                    and n[: -len(".weight")] + ".weight_scale_inv" in src.weight_map
                    and NVFP4_RE.search(n[: -len(".weight")])})
    gscales: dict[str, float] = {}
    rows: list[TensorRow] = []
    for base in bases:
        if base.endswith(("gate_proj", "up_proj")) and base not in gscales:
            g, u = re.sub(r"(gate|up)_proj$", "gate_proj", base), re.sub(r"(gate|up)_proj$", "up_proj", base)
            amax = max(float(fp8_weight(src, g, "cpu").abs().max()), float(fp8_weight(src, u, "cpu").abs().max()))
            gscales[g] = gscales[u] = ct_nvfp4.global_scale_for(amax)
        w = fp8_weight(src, base, "cpu")
        gscale = gscales.get(base, ct_nvfp4.global_scale_for(float(w.abs().max())))
        stored = unpack(ref.get_tensor(base + ".weight_packed"))
        diff = {v: int((codes(w, gscale, v) != stored).sum()) for v in VARIANTS}
        kind = re.search(r"(gate|up|down|q|k|v|o)_proj$", base).group(1)
        rows.append(TensorRow(base, kind, diff))
        print(base, " ".join(f"{v}={diff[v]}" for v in VARIANTS), flush=True)
    with open(out_tsv, "w") as f:
        f.write("name\tkind\t" + "\t".join(VARIANTS) + "\n")
        for r in rows:
            f.write(f"{r.name}\t{r.kind}\t" + "\t".join(str(r.differing_codes[v]) for v in VARIANTS) + "\n")
    print(f"{len(rows)} NVFP4 tensors; tensors whose codes differ from the stored bytes, by kind ({' / '.join(KINDS)}):")
    for v in VARIANTS:
        c = Counter(r.kind for r in rows if r.differing_codes[v] > 0)
        print(f"  {v:8s} total {sum(c.values()):3d}: " + " / ".join(str(c.get(k, 0)) for k in KINDS)
              + f"; differing codes {sum(r.differing_codes[v] for r in rows)}")


if __name__ == "__main__":
    main()
