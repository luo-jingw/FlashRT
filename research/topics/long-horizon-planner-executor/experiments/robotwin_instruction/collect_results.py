"""Collect runs/<tag>/<task>/<arm>/{episode_end_states,arm_episodes}.jsonl into one results JSON and print the
task x arm table (successes / episodes) and, per arm, how many seeds flip relative to `original`.

usage: python collect_results.py RUN_DIR OUT_JSON
"""
import json
import os
import sys
from dataclasses import asdict, dataclass

from instruction_arms import ARM_NAMES
from task_specs import TASK_SPECS


@dataclass(frozen=True)
class Episode:
    task: str
    arm: str
    episode_idx: int
    seed: int
    success: bool
    steps: int
    step_lim: int
    instruction_given: str
    instruction_original: str
    video: str


def _read_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_episodes(run_dir: str) -> list[Episode]:
    out: list[Episode] = []
    for task in TASK_SPECS:
        for arm in ARM_NAMES:
            d = os.path.join(run_dir, task, arm)
            if not os.path.isdir(d):
                continue
            given = {int(r["seed"]): r for r in _read_jsonl(os.path.join(d, "arm_episodes.jsonl"))}
            for r in _read_jsonl(os.path.join(d, "episode_end_states.jsonl")):
                g = given[int(r["seed"])]
                out.append(Episode(task=task, arm=arm, episode_idx=int(r["episode_idx"]), seed=int(r["seed"]),
                                   success=bool(r["success"]), steps=int(r["take_action_cnt"]),
                                   step_lim=int(r["step_lim"]), instruction_given=str(g["given"]),
                                   instruction_original=str(g["original"]),
                                   video="" if r["video_path"] is None else str(r["video_path"])))
    return out


def main() -> None:
    run_dir, out_json = sys.argv[1], sys.argv[2]
    eps = load_episodes(run_dir)
    table: dict[str, dict[str, dict[str, object]]] = {}
    for task in TASK_SPECS:
        row: dict[str, dict[str, object]] = {}
        base = {e.seed: e.success for e in eps if e.task == task and e.arm == "original"}
        for arm in ARM_NAMES:
            sel = [e for e in eps if e.task == task and e.arm == arm]
            if not sel:
                continue
            common = [e for e in sel if e.seed in base]
            succ_steps = [e.steps for e in sel if e.success]
            row[arm] = {
                "successes": sum(e.success for e in sel),
                "episodes": len(sel),
                "seeds_common_with_original": len(common),
                "only_original_succeeds": sum(base[e.seed] and not e.success for e in common),
                "only_this_arm_succeeds": sum(e.success and not base[e.seed] for e in common),
                "mean_steps_successful": round(sum(succ_steps) / len(succ_steps), 1) if succ_steps else None,
            }
        table[task] = row

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"table": table, "episodes": [asdict(e) for e in eps]}, f, indent=1, ensure_ascii=False)

    print("| task | " + " | ".join(ARM_NAMES) + " |")
    print("|---|" + "---|" * len(ARM_NAMES))
    for task, row in table.items():
        cells = [f"{row[a]['successes']}/{row[a]['episodes']}" if a in row else "-" for a in ARM_NAMES]
        print(f"| {task} | " + " | ".join(cells) + " |")
    for task, row in table.items():
        for arm, c in row.items():
            print(f"{task:20s} {arm:9s} {c}")


if __name__ == "__main__":
    main()
