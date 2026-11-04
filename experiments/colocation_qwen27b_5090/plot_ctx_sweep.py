"""Stacked bars of planner_ctx_sweep.py's result: one bar per prompt length, segments preprocess + ViT + inject,
prefill and decode (medians), total call time above each bar.

usage: python plot_ctx_sweep.py SWEEP_JSON OUT_PNG
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

INK, INK2, MUTED, LINE = "#1d1a17", "#4a443e", "#857c72", "#e6e0d8"
COLORS = ["#9c3d17", "#eb6834", "#f6b896"]           # the report page's phase colors (light theme)


def cjk_font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next(n for n in ("Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP") if n in names)


def main(src: str, out: str) -> None:
    rows = json.load(open(src))["rows"]
    plt.rcParams.update({"font.family": cjk_font(), "font.size": 11, "axes.edgecolor": MUTED,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED})
    n = [r["prompt_tokens"] for r in rows]
    m = [r["median"] for r in rows]
    parts = [[(x["preprocess_ms"] + x["vit_ms"] + x["inject_ms"]) / 1e3 for x in m],
             [x["prefill_ms"] / 1e3 for x in m], [x["decode_ms"] / 1e3 for x in m]]
    labels = ["预处理 + ViT + 注入", "prefill", f"解码 {rows[0]['new_tokens']} 个 token"]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    xs = range(len(rows))
    bottom = [0.0] * len(rows)
    for vals, color, label in zip(parts, COLORS, labels):
        ax.bar(xs, vals, 0.62, bottom=bottom, color=color, label=label, edgecolor="white", linewidth=1.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, total in zip(xs, bottom):
        ax.text(x, total + 0.03, f"{total:.2f}", ha="center", va="bottom", fontsize=10, color=INK2)
    ax.set_xticks(list(xs), [f"{v:,}".replace(",", " ") for v in n])
    ax.set_xlabel("输入 token 数")
    ax.set_ylabel("一次调用（s）")
    ax.set_ylim(0, max(bottom) * 1.12)
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=3)
    ax.set_title("Qwen3.8-27B 规划器单独运行（RTX 5090 整卡，FlashRT NVFP4）：上下文长度 × 一次调用时间",
                 loc="left", fontsize=12.5, color=INK, pad=12)
    fig.text(0.01, 0.01, "输入：两帧图像（512 个图像 token）+ 规划指令，前面加文本执行历史凑到各长度；每次输出 64 个 token；"
             "每个长度 3 次调用的中位数；无执行器。", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
