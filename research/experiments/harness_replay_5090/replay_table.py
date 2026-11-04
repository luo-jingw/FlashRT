"""One table over the replay cells (cells/<mode>_<share>_wrist<W>.json, three episodes each): wall time per episode
against 53.3 s of sim time, master time and calls, monitor call p50 (all / warm) and max, decision latency (max over
monitor calls, in env steps), executor latency p50 / p99 / max, share of executor queries over one env step
(66.7 ms) and over one chunk (533 ms), same-reply count, free memory.

usage: python replay_table.py CELL_JSON [CELL_JSON ...]
"""
import json
import sys


def main(files: list[str]) -> None:
    print("| mode | share | wrist | wall s / ep | master s / ep | calls | monitor p50 (warm) ms | monitor max ms | "
          "decompose p50 ms | decision lat max steps | exec p50 / p99 / max ms | > 66.7 ms | > 533 ms | same reply | "
          "free GB |")
    print("|---" * 15 + "|")
    for f in files:
        d = json.load(open(f))
        s = d["summary"]
        n_ep = len(d["episodes"])
        wall = [ep["summary"]["wall_s"] for ep in d["episodes"]]
        m, dc, e = s["calls"]["monitor"], s["calls"]["decompose"], s["exec"]
        dl = max(c["decision_latency_max_steps"] for c in s["calls"].values())
        wrist = f.rsplit("wrist", 1)[1].split(".")[0]
        print(f"| {d['mode']} | {d['share']} | {wrist} | {sum(wall) / n_ep:.1f} | {s['master_sum_s'] / n_ep:.1f} | "
              f"{s['n_calls']} | {m['p50_ms']:.0f} ({m['warm_p50_ms'] or 0:.0f}) | {m['max_ms']:.0f} | "
              f"{dc['p50_ms']:.0f} | {dl:.1f} | {e['latency_p50_ms']:.1f} / {e['latency_p99_ms']:.1f} / "
              f"{e['latency_max_ms']:.1f} | {e['over_step_pct']:.1f}% | {e['over_chunk_pct']:.1f}% | "
              f"{s['same_reply']}/{s['n_calls']} | {d['free_GB']:.1f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
