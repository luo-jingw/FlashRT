"""Figures of a monitor_call_bench.py result.

  fig_structure_<tag>.png   : token composition of a monitor call (episode-constant prefix vs per-call part; decision
                              vs explanation in the reply);
  fig_latency_<tag>.png     : per wrist size, stacked stage medians (preprocess + ViT + inject, restore + prefill,
                              decode) per prefix setting, for the "eos" and "stop" budgets.

usage: python figures.py TITLE_LABEL FILE_TAG RESULT_JSON OUT_DIR
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from tables import BUDGETS, PREFIXES, med, wrist_of  # noqa: E402

INK, INK2, MUTED, LINE = "#1d1a17", "#4a443e", "#857c72", "#e6e0d8"
STAGE_COLORS = ["#9c3d17", "#eb6834", "#f6b896"]   # same phase colors as colocation_qwen27b_5090/plot_ctx_sweep.py
PART_COLORS = ["#c9bfb3", "#9c3d17"]               # constant / per-call (structure figure)
FIG_BUDGETS = BUDGETS[:2]


def cjk_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next(n for n in ("Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP") if n in names)


def style() -> None:
    plt.rcParams.update({"font.family": cjk_font(), "font.size": 11, "axes.edgecolor": MUTED,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED})


def fig_structure(structure: list[dict], out: str) -> None:
    by = defaultdict(list)
    for s in structure:
        by[wrist_of(s["episode"])].append(s)
    rows = []
    for w, ss in sorted(by.items()):
        S, B = ss[0]["prompt_tokens"], ss[0]["boundary"]
        rows.append((f"输入 · 腕部图 {w}", B, S - B, "与本回合上一次调用相同", "每次不同"))
    allrep = [s["reply_tokens"] for s in structure]
    alldec = [s["decision_tokens"] for s in structure]
    rows.append(("输出 · 回复", med(alldec), med(allrep) - med(alldec), "到 action 为止", "reason 解释"))
    style()
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=160)
    for i, (label, a, b, la, lb) in enumerate(rows):
        tot = a + b
        ax.barh(i, a / tot, color=PART_COLORS[0], edgecolor="white", linewidth=1.5)
        ax.barh(i, b / tot, left=a / tot, color=PART_COLORS[1], edgecolor="white", linewidth=1.5)
        ax.text(a / tot / 2, i, f"{la}\n{a:.0f} token（{a / tot:.0%}）", ha="center", va="center", fontsize=9.5,
                color=INK)
        ax.text(a / tot + b / tot / 2, i, f"{lb}\n{b:.0f} token", ha="center", va="center", fontsize=9.5,
                color="white")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("一次 monitor 调用的组成：大半输入与上一次相同，决策只占回复的前一小段", loc="left", fontsize=12.5,
                 color=INK, pad=10)
    n_ep = len({s["episode"] for s in structure})
    fig.text(0.01, 0.01, f"VoLoAgent Full-loop 的 monitor 调用，LIBERO-10 回合 0 / 43 / 188 的记录（{len(structure)} 次，"
             f"{n_ep} 份记录）；输出 token 为中位数。", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out)


MONITOR_INTERVAL_S = 5.33   # 80 env steps at 15 Hz (the harness's monitor interval)


def fig_latency(cells: list[dict], label: str, out: str) -> None:
    by = defaultdict(list)
    for c in cells:
        by[(wrist_of(c["episode"]), c["prefix"], c["budget"])].append(c)
    wrists = sorted({k[0] for k in by})
    style()
    fig, axes = plt.subplots(1, len(wrists), figsize=(max(10.0, 5.4 * len(wrists)), 5.4), dpi=160, sharey=True,
                             squeeze=False)
    for ax, w in zip(axes[0], wrists):
        xs, labels, bottoms = [], [], []
        x = 0.0
        for b, bl in FIG_BUDGETS:
            for p, pl in PREFIXES:
                cs = by.get((w, p, b), [])
                if not cs:
                    continue
                parts = [med([c["preprocess_ms"] + c["vit_ms"] + c["inject_ms"] for c in cs]) / 1e3,
                         med([c["restore_ms"] + c["prefill_ms"] for c in cs]) / 1e3,
                         med([c["decode_ms"] for c in cs]) / 1e3]
                bottom = 0.0
                for v, color, nm in zip(parts, STAGE_COLORS, ("预处理 + ViT + 注入", "恢复 + prefill", "解码")):
                    ax.bar(x, v, 0.7, bottom=bottom, color=color, edgecolor="white", linewidth=1.5,
                           label=nm if (x == 0) else None)
                    bottom += v
                ax.text(x, bottom + 0.01, f"{bottom * 1e3:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK2)
                xs.append(x)
                labels.append(pl)
                bottoms.append(bottom)
                x += 1
            x += 0.6
        if max(bottoms) > MONITOR_INTERVAL_S:
            ax.axhline(MONITOR_INTERVAL_S, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(xs[-1] + 0.4, MONITOR_INTERVAL_S - 0.08, f"monitor 间隔 {MONITOR_INTERVAL_S} s", ha="right",
                    va="top", fontsize=9, color=INK2)
        ax.set_xticks(xs, labels, fontsize=9)
        for g, (_, bl) in enumerate(FIG_BUDGETS):
            ax.text(1 + 3.6 * g, -0.13, bl, transform=ax.get_xaxis_transform(), ha="center", fontsize=10, color=INK2)
        ax.set_title(f"腕部图 {w}", loc="left", fontsize=11, color=INK2)
        ax.grid(axis="y", color=LINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("调用开始到决策可用（s）")
    axes[0][0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(f"monitor 调用从开始到决策可用：复用固定前缀 × action 闭合即停（{label}）", x=0.01, ha="left",
                 fontsize=12.5, color=INK)
    fig.text(0.01, 0.045, "对齐复用：前缀截到 prefill 分块（2 048）的整数倍；"
             "完整复用：前缀到第一张 NOW 图像为止。", fontsize=9, color=MUTED)
    fig.text(0.01, 0.01, "每根柱为该腕部图尺寸全部 monitor 调用的中位数（ms 标在柱顶）；解码整段回复时，决策在回复结束后才可用。",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out)


def main() -> None:
    label, tag, src, out_dir = sys.argv[1:5]
    d = json.load(open(src))
    fig_structure(d["structure"], os.path.join(out_dir, f"fig_structure_{tag}.png"))
    fig_latency(d["cells"], label, os.path.join(out_dir, f"fig_latency_{tag}.png"))


if __name__ == "__main__":
    main()
