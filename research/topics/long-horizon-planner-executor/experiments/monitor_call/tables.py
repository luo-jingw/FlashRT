"""Markdown tables of a monitor_call_bench.py result (no plotting dependency; runs where the bench runs).

  structure : per wrist size, prompt tokens split into the episode-constant prefix (system prompt + BEFORE images) and
              the per-call part (NOW images + text), reply tokens split into the decision (through the action value)
              and the rest; decision counts.
  capsules  : per episode and prefix, the one-time capsule build (prefix ViT + prefill + snapshot) and its size.
  cells     : per wrist size x prefix x budget, medians over calls of each stage and of the call (ms), reply / action
              equality counts against the reference reply.

usage: python tables.py RESULT_JSON
"""
import json
import statistics
import sys
from collections import Counter, defaultdict

PREFIXES = (("none", "无复用"), ("aligned", "对齐复用"), ("exact", "完整复用"))
BUDGETS = (("eos", "解码整段回复"), ("stop", "action 闭合即停"), ("decision", "预知长度解码到 action"))


def wrist_of(episode: str) -> str:
    return episode.rsplit("_", 1)[1]


def med(xs: list[float]) -> float:
    return statistics.median(xs)


def structure_table(structure: list[dict]) -> str:
    by = defaultdict(list)
    for s in structure:
        by[wrist_of(s["episode"])].append(s)
    lines = ["| wrist | calls | prompt tokens | constant prefix B | B / prompt | aligned prefix | image tokens "
             "(constant / all) | reply tokens median (range) | decision tokens median (range) | decisions |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for w, ss in sorted(by.items()):
        S, B = ss[0]["prompt_tokens"], ss[0]["boundary"]
        rep = [s["reply_tokens"] for s in ss]
        dec = [s["decision_tokens"] for s in ss]
        cnt = Counter(s["decision"] for s in ss)
        lines.append(f"| {w} | {len(ss)} | {S} | {B} | {B / S:.0%} | {ss[0]['aligned']} | "
                     f"{ss[0]['image_tokens_before_boundary']} / {ss[0]['image_tokens']} | {med(rep):.0f} "
                     f"({min(rep)}-{max(rep)}) | {med(dec):.0f} ({min(dec)}-{max(dec)}) | "
                     + ", ".join(f"{k} {v}" for k, v in cnt.most_common()) + " |")
    return "\n".join(lines)


def cell_table(cells: list[dict]) -> str:
    by = defaultdict(list)
    for c in cells:
        by[(wrist_of(c["episode"]), c["prefix"], c["budget"])].append(c)
    lines = ["| wrist | prefix | budget | calls | new tokens | preprocess | ViT | inject | restore | prefill | decode | "
             "call | same reply | same action |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = [("generate", "eos")] + [(p, b) for p, _ in PREFIXES for b, _ in BUDGETS]
    for w in sorted({k[0] for k in by}):
        for p, b in order:
            cs = by.get((w, p, b), [])
            if not cs:
                continue
            m = {k: med([c[k] for c in cs]) for k in ("new_tokens", "preprocess_ms", "vit_ms", "inject_ms",
                                                      "restore_ms", "prefill_ms", "decode_ms", "call_ms")}
            same = sum(1 for c in cs if c["same_reply"]) if b == "eos" else None
            lines.append(f"| {w} | {p} | {b} | {len(cs)} | {m['new_tokens']:.0f} | {m['preprocess_ms']:.0f} | "
                         f"{m['vit_ms']:.0f} | {m['inject_ms']:.1f} | {m['restore_ms']:.2f} | {m['prefill_ms']:.0f} | "
                         f"{m['decode_ms']:.0f} | **{m['call_ms']:.0f}** | "
                         f"{'-' if same is None else f'{same}/{len(cs)}'} | "
                         f"{sum(1 for c in cs if c['same_decision'])}/{len(cs)} |")
    return "\n".join(lines)


def capsule_table(capsules: list[dict]) -> str:
    lines = ["| episode | prefix | prefix tokens | build ms | capsule MB |", "|---|---|---|---|---|"]
    for c in capsules:
        lines.append(f"| {c['episode']} | {c['prefix']} | {c['prefix_tokens']} | {c['build_ms']:.0f} | "
                     f"{c['nbytes'] / 1e6:.0f} |")
    return "\n".join(lines)


def main() -> None:
    d = json.load(open(sys.argv[1]))
    print(structure_table(d["structure"]))
    print()
    print(capsule_table(d["capsules"]))
    print()
    print(cell_table(d["cells"]))


if __name__ == "__main__":
    main()
