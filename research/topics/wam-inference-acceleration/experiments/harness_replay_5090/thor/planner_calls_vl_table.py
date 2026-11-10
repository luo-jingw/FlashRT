"""Tables of one planner_calls_vl_<frontend>.json (written by planner_calls_vl.py): medians per call kind and draft
length, one line per timed call, and each call's decision against the reference reply of cells/mtp_tail_base.json.

A decision is the parsed JSON of a reply (a ```json fence is stripped): (status, action) for a monitor call, the
lower-cased subgoal list for a decompose call; a reply that does not parse is `unparsable`. The decision of the run is
taken from its untimed long-budget decision calls ("decisions"), which are not truncated at the reference length.
Decode ms per cycle is decode ms / spec cycles of each timed call (JSON files written before the script stored reply
texts have no "decisions"; their decision table is skipped).

usage: python research/topics/wam-inference-acceleration/experiments/harness_replay_5090/thor/planner_calls_vl_table.py <planner_calls_vl_*.json>
"""
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = "cells/mtp_tail_base.json"
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass
class Decision:
    parsed: bool
    value: tuple

    def label(self) -> str:
        return str(self.value) if self.parsed else "unparsable"


def decision(kind: str, text: str) -> Decision:
    try:
        obj = json.loads(FENCE.sub("", text.strip()))
    except json.JSONDecodeError:
        return Decision(False, ())
    if kind == "monitor":
        return Decision(True, (obj.get("status"), obj.get("action")))
    return Decision(True, tuple(s.lower() for s in obj.get("subgoals", [])))


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def ms_per_cycle(r: dict) -> float:
    return r["decode_ms"] / r["spec_cycles"] if r["spec_cycles"] else 0.0


def print_medians(records: list[dict]) -> None:
    print("kind spec_k n prompt_tokens vit_ms prefill_ms decode_ms wall_ms warmup_wall_ms cycles ms_per_cycle "
          "tok_per_forward same_text")
    for kind in ("decompose", "monitor"):
        for k in sorted({r["spec_k"] for r in records}, reverse=True):
            rs = [r for r in records if r["kind"] == kind and r["spec_k"] == k]
            if not rs:
                continue
            cols = [median([r[f] for r in rs]) for f in ("prompt_tokens", "vit_ms", "prefill_ms", "decode_ms",
                                                          "wall_ms", "warmup_wall_ms", "spec_cycles")]
            cols += [median([ms_per_cycle(r) for r in rs]), median([r["tok_per_forward"] for r in rs])]
            print(kind, k, len(rs), " ".join(f"{c:.2f}" for c in cols),
                  f"{sum(r['same_as_reference'] for r in rs)}/{len(rs)}")


def print_rows(records: list[dict]) -> None:
    print("name spec_k max_new_tokens cycles decode_ms ms_per_cycle prefill_ms wall_ms same_text")
    for r in records:
        print(r["name"], r["spec_k"], r["max_new_tokens"], r["spec_cycles"], f"{r['decode_ms']:.1f}",
              f"{ms_per_cycle(r):.2f}", f"{r['prefill_ms']:.1f}", f"{r['wall_ms']:.1f}", r["same_as_reference"])


def print_decisions(decisions: list[dict], refs: dict) -> None:
    same = unparsable = 0
    print("name reference_decision run_decision equal")
    for d in decisions:
        ref = decision(d["kind"], refs[d["name"]]["text"])
        run = decision(d["kind"], d["text"])
        equal = ref.parsed and run.parsed and ref.value == run.value
        same += equal
        unparsable += not run.parsed
        print(d["name"], ref.label(), run.label(), equal)
    print(f"decisions equal {same}/{len(decisions)}, unparsable {unparsable}")


def main(path: str) -> None:
    data = json.load(open(path))
    refs = {r["prompt"]: r for r in json.load(open(os.path.join(HERE, REFS)))["references"]}
    print("frontend", data["frontend"], data["frontend_class"], data["llm_class"], "gpu", data["gpu"],
          f"peak_GiB {data['peak_GiB']:.2f}")
    print_medians(data["records"])
    print_rows(data["records"])
    if "decisions" in data:
        print_decisions(data["decisions"], refs)


if __name__ == "__main__":
    main(sys.argv[1])
