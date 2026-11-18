"""One RoboTwin 2.0 task, several instruction arms, one ImageWAM model load.

For each arm, runs RoboTwin's own `eval_policy` loop (script/eval_policy.py) with the official ImageWAM torch
policy, starting from the same seed, so every arm sees the same expert-checked scenes. Two hooks on the task
environment instance, both explicit here:
  - `setup_demo` is wrapped to record the episode seed and to keep the eval video only for the first
    `--video-episodes` episodes of the arm;
  - `set_instruction` is wrapped so the arm's instruction (InstructionSelector) replaces the evaluator's.
The evaluator's `generate_episode_descriptions` is wrapped to seed Python's `random` with the episode seed
(object-description choice, unseeded in RoboTwin) and to keep the episode info for the "other" arm.

Outputs under OUT_DIR/<task>/<arm>/: the evaluator's episode_end_states.jsonl, and arm_episodes.jsonl
(seed, original instruction, given instruction). Run with the RoboTwin root as working directory.

usage: run_instruction_arms.py TASK EPISODES SEED OUT_DIR VIDEO_DIR VIDEO_EPISODES ARM[,ARM...]
"""
import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any

EXP_DIR: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from script import eval_policy as official  # noqa: E402
import imagewam_policy  # noqa: E402

from instruction_arms import InstructionSelector  # noqa: E402
from official_usr_args import RunSettings, build_usr_args  # noqa: E402
from robotwin_task_args import build_task_args  # noqa: E402
from task_specs import TASK_SPECS  # noqa: E402


class EpisodeHooks:
    """Per-arm state shared by the three wrappers: current seed, episode info, selector, log path."""

    def __init__(self, selector: InstructionSelector, log_path: str, video_episodes: int) -> None:
        self.selector = selector
        self.log_path = log_path
        self.video_episodes = video_episodes
        self.seed = -1
        self.episode_idx = -1
        self.episode_info: dict[str, str] = {}

    def setup_demo(self, env: Any, orig_setup: Any, **kwargs: Any) -> None:
        self.seed = int(kwargs["seed"])
        self.episode_idx = int(kwargs.get("now_ep_num", 0))
        if self.episode_idx >= self.video_episodes:
            kwargs = dict(kwargs)
            kwargs["eval_video_save_dir"] = None
        orig_setup(**kwargs)

    def describe(self, orig_describe: Any, task_name: str, episodes: list[dict[str, str]],
                 max_descriptions: int) -> list[dict[str, list[str]]]:
        self.episode_info = dict(episodes[0])
        random.seed(self.seed)
        return orig_describe(task_name, episodes, max_descriptions)

    def set_instruction(self, orig_set: Any, instruction: str) -> None:
        chosen = self.selector.select(original=str(instruction), episode_info=self.episode_info, seed=self.seed)
        record = {"episode_idx": self.episode_idx, "seed": self.seed, **asdict(chosen)}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[instruction] arm={chosen.arm} seed={self.seed} given={chosen.given!r} "
              f"original={chosen.original!r}", flush=True)
        orig_set(instruction=chosen.given)


def run_arm(arm: str, usr_args: dict[str, Any], model: Any, task_env: Any, out_dir: str, video_dir: str,
            video_episodes: int) -> tuple[int, float]:
    task = str(usr_args["task_name"])
    arm_dir = os.path.join(out_dir, task, arm)
    arm_video_dir = os.path.join(video_dir, task, arm) if video_episodes > 0 else ""
    args = build_task_args(usr_args, arm_dir, arm_video_dir)
    selector = InstructionSelector(arm, TASK_SPECS[task], official.generate_episode_descriptions,
                                   str(usr_args["instruction_type"]))
    hooks = EpisodeHooks(selector, os.path.join(arm_dir, "arm_episodes.jsonl"), video_episodes)

    cls = type(task_env)
    orig_describe = official.generate_episode_descriptions
    task_env.setup_demo = lambda **kw: hooks.setup_demo(task_env, lambda **k: cls.setup_demo(task_env, **k), **kw)
    task_env.set_instruction = lambda instruction=None: hooks.set_instruction(
        lambda instruction: cls.set_instruction(task_env, instruction=instruction), instruction)
    official.generate_episode_descriptions = lambda t, e, m=1000000: hooks.describe(orig_describe, t, e, m)

    st_seed = 100000 * (1 + int(usr_args["seed"]))
    video_size = official.get_eval_video_size(args) if arm_video_dir != "" else None
    t0 = time.time()
    try:
        _, suc = official.eval_policy(task, task_env, args, model, st_seed,
                                      test_num=int(usr_args["eval_num_episodes"]), video_size=video_size,
                                      instruction_type=usr_args["instruction_type"],
                                      skip_get_obs_within_replan=bool(usr_args["skip_get_obs_within_replan"]))
    finally:
        official.generate_episode_descriptions = orig_describe
        del task_env.setup_demo
        del task_env.set_instruction
    return int(suc), time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("episodes", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("out_dir")
    p.add_argument("video_dir")
    p.add_argument("video_episodes", type=int)
    p.add_argument("arms")
    a = p.parse_args()

    settings = RunSettings(task_name=a.task, episodes=a.episodes, seed=a.seed,
                           eval_output_dir=os.path.join(a.out_dir, a.task))
    usr_args = build_usr_args(settings, "policy/imagewam_policy/deploy_policy.yml")
    task_env = official.class_decorator(a.task)
    t0 = time.time()
    model = imagewam_policy.get_model(usr_args)
    print(f"[timing] model load {time.time() - t0:.1f}s", flush=True)
    for arm in a.arms.split(","):
        suc, dt = run_arm(arm, usr_args, model, task_env, a.out_dir, a.video_dir, a.video_episodes)
        print(f"ARM_DONE task={a.task} arm={arm} success={suc}/{a.episodes} wall={dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
