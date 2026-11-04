"""Tables of mtp_tail_model.py's results.

Per model JSON, prompt group (synthetic length, or recorded call kind), (tail, fc) and K: calls, tail rows (median),
decoded tokens, spec cycles, accepted drafts, tokens per main-model forward ((decoded - calls) / cycles), accepted
drafts per cycle, decode ms per call, tail prefill ms per call, the single-token decode estimate ((decoded - 1) x the
replay time of the Kv = 1 main-model forward at the nearest prompt length, from GRAPHS_JSON = mtp_speedup.json) and two
speedups over it: decode only, and decode + tail prefill. Then each model's references (decode budget, EOS, replies
equal to the recorded ones).

usage: python mtp_tail_model_table.py GRAPHS_JSON MODEL_JSON [MODEL_JSON ...]
"""
import collections
import json
import statistics
import sys


def main(graphs_json: str, files: list[str]) -> None:
    v1 = sorted((g["prompt_tokens"], g["verify_ms"]["1"]) for g in json.load(open(graphs_json))["graphs"])

    def single_step_ms(prompt_tokens: int) -> float:
        return min(v1, key=lambda x: abs(x[0] - prompt_tokens))[1]

    print("| model | prompts | tail / fc | K | n | tail rows | decoded | cycles | accepted | tok/forward | acc/cycle | "
          "decode ms/call | tail ms/call | single-token ms/call | speedup decode | speedup decode+tail | same |")
    print("|---" * 17 + "|")
    refs = {}
    for f in files:
        d = json.load(open(f))
        refs[d["model"]] = d["references"]
        groups = collections.defaultdict(list)
        for c in d["cells"]:
            name = c["prompt"]
            group = name if name.startswith("synth:") else name.split(":")[1].split("@")[0]
            groups[(group, c["tail"], c["fc"], c["k"])].append(c)
        for (group, tail, fc, k), cs in groups.items():
            n = len(cs)
            dec = sum(c["decoded_tokens"] for c in cs)
            cyc = sum(c["spec_cycles"] for c in cs)
            acc = sum(c["spec_accepted"] for c in cs)
            dms = sum(c["decode_ms"] for c in cs)
            tms = sum(c["tail_ms"] for c in cs)
            base = sum((c["decoded_tokens"] - 1) * single_step_ms(c["prompt_tokens"]) for c in cs)
            rows = statistics.median(c["tail_rows"] for c in cs)
            ks = "/".join(map(str, sorted({c["effective_k"] for c in cs})))
            same = sum(c["same_as_reference"] for c in cs)
            print(f"| {d['model']} | {group} | {tail} / {fc} | {k} ({ks}) | {n} | {rows:.0f} | {dec} | {cyc} | {acc} | "
                  f"{(dec - n) / cyc:.2f} | {acc / cyc:.2f} | {dms / n:.0f} | {tms / n:.1f} | {base / n:.0f} | "
                  f"{base / dms:.2f} | {base / (dms + tms):.2f} | {same}/{n} |")
    print()
    print("| model | prompts | n | prompt tokens (median) | decode budget | decoded (median) | EOS | same as recorded |")
    print("|---" * 8 + "|")
    for model, rs in refs.items():
        groups = collections.defaultdict(list)
        for r in rs:
            name = r["prompt"]
            groups[name if name.startswith("synth:") else name.split(":")[1].split("@")[0]].append(r)
        for group, g in groups.items():
            same = [r["same_as_recorded"] for r in g if r["same_as_recorded"] is not None]
            print(f"| {model} | {group} | {len(g)} | {statistics.median(r['prompt_tokens'] for r in g):.0f} | "
                  f"{g[0]['budget']} | {statistics.median(r['decoded_tokens'] for r in g):.0f} | "
                  f"{sum(r['eos'] for r in g)}/{len(g)} | {f'{sum(same)}/{len(same)}' if same else '-'} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
