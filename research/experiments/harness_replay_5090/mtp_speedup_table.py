"""Table of mtp_speedup.py's result. Per prompt group (synthetic length, or recorded call kind) and draft length K:
decoded tokens (the frontend decodes max_new_tokens: 64 for a synthetic prompt, the recorded output_tokens + 1 for
a recorded call), spec cycles, accepted drafts, tokens per main-model forward, decode ms, ms per decoded token, and the
single-token baseline: (decoded tokens - 1) x the replay time of the Kv = 1 main-model forward at the nearest measured
prompt length (no per-step host work counted, so the baseline is a lower bound of a single-token decode loop).

usage: python mtp_speedup_table.py MTP_SPEEDUP_JSON
"""
import collections
import json
import os
import sys

SYNTH_NEW_TOKENS = 64


def main(src: str) -> None:
    d = json.load(open(src))
    v1 = sorted((g["prompt_tokens"], g["verify_ms"]["1"]) for g in d["graphs"])

    def single_step_ms(prompt_tokens: int) -> float:
        return min(v1, key=lambda x: abs(x[0] - prompt_tokens))[1]

    recorded = {}
    for name in {r["prompt"].split(":")[0] for r in d["decode"] if not r["prompt"].startswith("synth:")}:
        for line in open(os.path.join(os.path.dirname(os.path.abspath(src)), "..", "calls", name)):
            row = json.loads(line)
            recorded[f"{name}:{row['kind']}@{row['step']}"] = min(row["output_tokens"] + 1, row["max_tokens"])
    groups = collections.defaultdict(list)
    for r in d["decode"]:
        r["decoded_tokens"] = SYNTH_NEW_TOKENS if r["prompt"].startswith("synth:") else recorded[r["prompt"]]
        name = r["prompt"]
        group = name if name.startswith("synth:") else name.split(":")[1].split("@")[0]
        groups[(group, r["k"])].append(r)
    print("| prompts | K | n | prompt tok | decoded tok | cycles | accepted | tok/forward | decode ms | ms/tok | "
          "single-token ms | speedup | same output |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (group, k), rs in groups.items():
        n_dec = sum(r["decoded_tokens"] for r in rs)
        cyc = sum(r["spec_cycles"] for r in rs)
        acc = sum(r["spec_accepted"] for r in rs)
        dec = sum(r["decode_ms"] for r in rs)
        base = sum((r["decoded_tokens"] - 1) * single_step_ms(r["prompt_tokens"]) for r in rs)
        ptok = sorted(r["prompt_tokens"] for r in rs)[len(rs) // 2]
        same = sum(1 for r in rs if r["same_as_recorded"]) if rs[0]["same_as_recorded"] is not None else "-"
        ks = sorted({r["effective_k"] for r in rs})
        print(f"| {group} | {k} ({'/'.join(map(str, ks))}) | {len(rs)} | {ptok} | {n_dec} | {cyc} | {acc} | "
              f"{(n_dec - len(rs)) / cyc:.2f} | {dec:.0f} | {dec / (n_dec - len(rs)):.1f} | {base:.0f} | "
              f"{base / dec:.2f} | {same} |")
    print()
    print("| prompt tok | " + " | ".join(f"verify Kv={kv}" for kv in d["graphs"][0]["verify_ms"]) + " | "
          + " | ".join(f"chain K={k}" for k in d["graphs"][0]["chain_ms"]) + " | MTP tail rows | MTP tail ms |")
    print("|---" * (3 + len(d["graphs"][0]["verify_ms"]) + len(d["graphs"][0]["chain_ms"])) + "|")
    for g in d["graphs"]:
        print(f"| {g['prompt_tokens']} | " + " | ".join(f"{v:.2f}" for v in g["verify_ms"].values()) + " | "
              + " | ".join(f"{v:.2f}" for v in g["chain_ms"].values())
              + f" | {g['mtp_tail_rows']} | {g['mtp_tail_prefill_ms']:.1f} |")


if __name__ == "__main__":
    main(sys.argv[1])
