"""Monitor calls of the closed-loop episodes with a non-`continue` monitor decision or a failure, rebuilt from a run
record (`runs/<mode>.json`) and the frames the run saved (`runs/<mode>_t<task>_e<episode>/s<step>_*.jpg`), in the
format of `harness_replay_5090/calls/*.jsonl` (one file per episode), for `../monitor_call/monitor_call_bench.py` and
`../monitor_call/reply_repeat.py` (Thor rows H11, H12).

Each call is rebuilt with `volo_calls.monitor_call` from the run's task instruction, the subgoal list in force and the
event's subgoal index, the step-0 frames (BEFORE) and the step's frames (NOW). The frames are copied to
`calls/frames/t<task>e<episode>/`; image paths in the rows are relative to `harness_replay_5090/` (the bench's base
directory). `output_text` / `output_tokens` are the run's reply (LoRA-merged master) and its token count.

usage (cwd = this directory): python build_monitor_calls.py runs/planner_eos.json calls
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_DIR = os.path.join(HERE, "..", "..", "..", "wam-inference-acceleration", "experiments", "harness_replay_5090")
sys.path.insert(0, REPLAY_DIR)

import volo_calls as vc  # noqa: E402

WRIST = 224  # closed_loop_libero.WRIST


def frames(src_dir: str, dst_dir: str, step: int) -> vc.HarnessFrames:
    paths = []
    for name in ("front", "exterior", f"wrist{WRIST}"):
        f = f"s{step:04d}_{name}.jpg"
        shutil.copyfile(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        paths.append(os.path.relpath(os.path.join(dst_dir, f), REPLAY_DIR))
    return vc.HarnessFrames(front=paths[0], exterior=paths[1], wrist=paths[2])


def selected(rec: dict) -> bool:
    mons = [e for e in rec["events"] if e["kind"] == "monitor"]
    return (not rec["success"]) or any(e["detail"]["decision"] != "continue" for e in mons)


def episode_rows(rec: dict, run_dir: str, out_dir: str) -> list[dict]:
    tag = f"t{rec['task']}e{rec['episode']}"
    src = os.path.join(run_dir, f"{rec['mode']}_t{rec['task']}_e{rec['episode']}")
    dst = os.path.join(out_dir, "frames", tag)
    os.makedirs(dst, exist_ok=True)
    before = frames(src, dst, 0)
    rows, subgoals = [], None
    for e in rec["events"]:
        if e["kind"] in ("decompose", "replan") and e["detail"].get("subgoals"):
            if e["kind"] == "decompose" or e["detail"].get("accepted"):
                subgoals = e["detail"]["subgoals"]
        if e["kind"] != "monitor":
            continue
        call = vc.monitor_call(rec["instruction"], subgoals, e["detail"]["subgoal_index"], before,
                               frames(src, dst, e["step"]), e["step"])
        row = vc.call_to_row(call)
        row.update({"output_text": e["detail"]["text"], "output_tokens": e["detail"]["new_tokens"],
                    "decision": e["detail"]["decision"], "source": f"{rec['mode']} task {rec['task']} episode "
                                                                    f"{rec['episode']}"})
        rows.append(row)
    return rows


def main() -> None:
    run_json, out_dir = sys.argv[1], sys.argv[2]
    run_dir = os.path.dirname(os.path.abspath(run_json))
    os.makedirs(out_dir, exist_ok=True)
    for rec in json.load(open(run_json)):
        if not selected(rec):
            continue
        rows = episode_rows(rec, run_dir, out_dir)
        path = os.path.join(out_dir, f"t{rec['task']}e{rec['episode']}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} monitor calls, success {rec['success']}, decisions "
              f"{[r['decision'] for r in rows]}", flush=True)


if __name__ == "__main__":
    main()
