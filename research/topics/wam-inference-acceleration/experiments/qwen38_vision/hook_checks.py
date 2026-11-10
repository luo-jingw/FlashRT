"""Checks of the Qwen36TorchFrontendRtx embedding-row and RoPE-row hooks on text prompts.

Run twice, in separate processes (two frontends do not fit on the GPU together):

  baseline  extra_embedding_rows=0; greedy MTP-spec generation per context; tokens and timings saved to BASE_JSON.
  hooks     extra_embedding_rows=EXTRA, then per context:
            (a) plain generation, compared with the baseline tokens;
            (b) 256 prompt positions remapped to reserved rows holding those tokens' own embeddings;
            (c) all three RoPE axes shifted by SHIFT through set_rope_rows (text_rope_rows on arange + SHIFT);
            (d) reset_rope_rows, then plain generation again.
            (e) first-token logits (``_logits_buf`` after a one-token generation on the BF16 route, prompts of
            SHORT_LEN tokens at 8 offsets) for every shift in E_SHIFTS against no shift, and between the two
            largest shifts: cosine, top-1 and top-5 agreement. Shifts of 1 and 7 change the rounding of every
            table row while barely changing the angles' magnitude; they calibrate the BF16-table noise floor.
            Reports, against (a): identical greedy prefix length and matching-token fraction over NEW_TOKENS; the
            frontend's prefill / decode ms per run; GPU memory after construction.

Prompt: the first CTX tokens of the repository's Markdown documentation (README.md, then docs/*.md), as in
context_bench.py; frontend settings as in context_bench.py.

usage: python hook_checks.py {baseline,hooks} NVFP4_DIR MTP_DIR BASE_JSON OUT_JSON [--extra 4096] [--shift 1000]
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["baseline", "hooks"])
ap.add_argument("ckpt")
ap.add_argument("mtp_dir")
ap.add_argument("base_json")
ap.add_argument("out")
ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096])
ap.add_argument("--new-tokens", type=int, default=64)
ap.add_argument("--extra", type=int, default=4096)
ap.add_argument("--shift", type=int, default=1000)
ap.add_argument("--short-len", type=int, default=300)
ap.add_argument("--e-shifts", type=int, nargs="+", default=[1, 7, 1000, 1001])
args = ap.parse_args()

MAX_SEQ = max(args.ctx) + args.new_tokens + 64
os.environ.update({
    "FLASHRT_QWEN36_MTP_CKPT_DIR": args.mtp_dir,
    "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": str(min(4096, MAX_SEQ - 1)),
    "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
    "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "512",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

import torch  # noqa: E402

from flash_rt.frontends.torch._qwen36_mrope import text_rope_rows  # noqa: E402
from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))


def doc_ids(tok, n: int) -> torch.Tensor:
    files = [os.path.join(REPO, "README.md")] + sorted(glob.glob(os.path.join(REPO, "docs", "*.md")))
    ids = tok.encode("\n\n".join(open(f, encoding="utf-8").read() for f in files), add_special_tokens=False)[:n]
    return torch.tensor([ids], dtype=torch.long, device="cuda")


def run(fe: Qwen36TorchFrontendRtx, ids: torch.Tensor) -> dict:
    out = fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=args.new_tokens)
    return {"tokens": out[0, ids.shape[1]:].tolist(), "prefill_ms": float(fe._long_ctx_prefill_ms),
            "decode_ms": float(fe._long_ctx_decode_ms)}


def agreement(a: list, b: list) -> dict:
    prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {"identical_prefix": prefix, "match_fraction": sum(x == y for x, y in zip(a, b)) / len(a)}


extra = 0 if args.mode == "baseline" else args.extra
fe = Qwen36TorchFrontendRtx(args.ckpt, quant="nvfp4", max_seq=MAX_SEQ, extra_embedding_rows=extra)
torch.cuda.synchronize()
mem_gib = torch.cuda.memory_allocated() / 2**30
print(f"{args.mode}: extra_embedding_rows {extra}, allocated after construction {mem_gib:.3f} GiB", flush=True)
tok = fe._tokenizer
report = {"mode": args.mode, "extra_embedding_rows": extra, "allocated_gib": mem_gib, "ctx": {}}
base = json.load(open(args.base_json)) if args.mode == "hooks" else None
for ctx in args.ctx:
    ids = doc_ids(tok, ctx)
    run(fe, ids)
    plain = run(fe, ids)
    cell = {"plain": plain}
    if args.mode == "hooks":
        cell["a_vs_baseline"] = agreement(plain["tokens"], base["ctx"][str(ctx)]["plain"]["tokens"])
        p0, n = ctx // 2, 256
        slot = fe.embedding_slot_base
        fe.write_embedding_rows(slot, fe._embed_table[ids[0, p0:p0 + n]].clone())
        remapped = ids.clone()
        remapped[0, p0:p0 + n] = slot + torch.arange(n, device="cuda")
        cell["b_remapped"] = run(fe, remapped)
        cell["b_vs_a"] = agreement(cell["b_remapped"]["tokens"], plain["tokens"])
        rows = ctx + args.new_tokens + 32
        pos = (torch.arange(rows, device="cuda") + args.shift).view(1, -1).expand(3, -1)
        cfg = fe._cfg
        cos, sin = text_rope_rows(pos, rotary_dim=int(cfg["head_dim"] * cfg["partial_rotary_factor"]),
                                  rope_theta=float(cfg["rope_theta"]), mrope_section=(11, 11, 10))
        fe.set_rope_rows(0, cos, sin)
        cell["c_shifted"] = run(fe, ids)
        cell["c_vs_a"] = agreement(cell["c_shifted"]["tokens"], plain["tokens"])
        fe.reset_rope_rows()
        cell["d_after_reset"] = run(fe, ids)
        cell["d_vs_a"] = agreement(cell["d_after_reset"]["tokens"], plain["tokens"])
        print(f"ctx {ctx}: (a) vs baseline {cell['a_vs_baseline']} | (b) remapped {cell['b_vs_a']} | (c) shift "
              f"{args.shift} {cell['c_vs_a']} | (d) reset {cell['d_vs_a']}", flush=True)
    print(f"ctx {ctx}: plain prefill {plain['prefill_ms']:.1f} ms decode {plain['decode_ms']:.1f} ms", flush=True)
    report["ctx"][str(ctx)] = cell
if args.mode == "hooks":
    full = doc_ids(tok, 8 * 4096)
    cfg = fe._cfg
    rd = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    def first_logits(ids: torch.Tensor, shift: int) -> torch.Tensor:
        fe.reset_rope_rows()
        if shift:
            pos = (torch.arange(args.short_len + 32, device="cuda") + shift).view(1, -1).expand(3, -1)
            cos, sin = text_rope_rows(pos, rotary_dim=rd, rope_theta=float(cfg["rope_theta"]),
                                      mrope_section=(11, 11, 10))
            fe.set_rope_rows(0, cos, sin)
        fe.generate_own_speculative_KN_nvfp4(ids, max_new_tokens=1)
        out = fe._logits_buf.float().clone().view(-1)
        fe.reset_rope_rows()
        return out

    def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
        return {"cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                "top1_equal": bool(a.argmax() == b.argmax()),
                "top5_overlap": len(set(a.topk(5).indices.tolist()) & set(b.topk(5).indices.tolist())) / 5}

    pairs = [(0, s) for s in args.e_shifts] + [tuple(sorted(args.e_shifts)[-2:])]
    e = {f"{a}_vs_{b}": [] for a, b in pairs}
    for k in range(8):
        ids = full[:, k * 4096:k * 4096 + args.short_len].contiguous()
        logits = {s: first_logits(ids, s) for s in [0] + args.e_shifts}
        for a, b in pairs:
            e[f"{a}_vs_{b}"].append(compare(logits[a], logits[b]))
    report["e_first_token_logits"] = {"route": str(getattr(fe, "_long_ctx_route", "?")), "pairs": e}
    for key, rows in e.items():
        print(f"(e) shift {key:12s}: cos min {min(r['cos'] for r in rows):.5f} median "
              f"{sorted(r['cos'] for r in rows)[4]:.5f} | top1 {sum(r['top1_equal'] for r in rows)}/8 | top5 overlap "
              f"mean {sum(r['top5_overlap'] for r in rows) / 8:.2f}", flush=True)
json.dump(report, open(args.out, "w"), indent=1)
