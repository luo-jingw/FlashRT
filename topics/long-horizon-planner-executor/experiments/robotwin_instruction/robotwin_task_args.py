"""Build the per-task `args` dict that RoboTwin's script/eval_policy.main builds before calling eval_policy:
task config yaml, embodiment files, camera sizes, per-run output paths. Same steps as eval_policy.main; the
only additions are an explicit output directory per arm and an optional eval video directory.

Must run with the RoboTwin root as the working directory (relative config paths, as in eval_policy).
"""
import os
from typing import Any

import yaml
from envs import CONFIGS_PATH


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_task_args(usr_args: dict[str, Any], out_dir: str, video_dir: str) -> dict[str, Any]:
    task_config = str(usr_args["task_config"])
    args = _load_yaml(f"./task_config/{task_config}.yml")
    args["task_name"] = usr_args["task_name"]
    args["task_config"] = task_config
    args["ckpt_setting"] = usr_args["ckpt_setting"]

    embodiment_types = _load_yaml(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"))
    camera_config = _load_yaml(CONFIGS_PATH + "_camera_config.yml")
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError(f"expected a single dual-arm embodiment, got {embodiment}")
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = _load_yaml(os.path.join(robot_file, "config.yml"))
    args["right_embodiment_config"] = _load_yaml(os.path.join(robot_file, "config.yml"))

    os.makedirs(out_dir, exist_ok=True)
    args["eval_episode_state_path"] = os.path.join(out_dir, "episode_end_states.jsonl")
    if video_dir != "":
        os.makedirs(video_dir, exist_ok=True)
        args["eval_video_log"] = True
        args["eval_video_save_dir"] = video_dir
    else:
        args["eval_video_log"] = False
    args["policy_name"] = usr_args["policy_name"]
    return args
