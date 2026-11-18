"""Build the `usr_args` dict that RoboTwin's script/eval_policy.py receives from ImageWAM's
experiments/robotwin/eval_robotwin_single.py, for the settings of
wam-inference-acceleration/experiments/robotwin_aog_subspace/run_task.sh (serial 10 action steps).

Same Hydra composition, same override list (built with eval_robotwin_single's own `_append_override`), parsed
the way eval_policy.parse_args_and_config parses it (yaml config, then `eval` of every override value).
"""
import os
import sys
from dataclasses import dataclass
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

IMAGEWAM_ROOT: str = "/home/jingwu/workspace/FlashRT/tmp/ImageWAM"
sys.path.insert(0, os.path.join(IMAGEWAM_ROOT, "experiments", "robotwin"))
from eval_robotwin_single import _append_override  # noqa: E402

SIM_TASK: str = "robotwin_flux2_klein_4b_base_clean_imagewam"


@dataclass(frozen=True)
class RunSettings:
    task_name: str
    episodes: int
    seed: int
    eval_output_dir: str


def _compose(settings: RunSettings) -> DictConfig:
    env = os.environ
    overrides = [
        f"task={SIM_TASK}",
        f"ckpt={env['RT_CKPT_PATH']}",
        f"EVALUATION.dataset_stats_path={env['RT_DATASET_STATS_PATH']}",
        f"model.flux2_src_path={env['FLUX2_SRC']}",
        f"model.flux2_model_path={env['FLUX2_MODEL_PATH']}",
        f"model.ae_model_path={env['FLUX2_AE_MODEL_PATH']}",
        "model.variant=klein-base-4b",
        f"model.qwen3_model_spec={env['QWEN3_MODEL_SPEC']}",
        "model.load_text_encoder=true",
        "model.pack_proprio_after_text=true",
        "model.proprio_dim=14",
        "EVALUATION.action_horizon=16",
        "EVALUATION.replan_steps=16",
        f"EVALUATION.eval_num_episodes={settings.episodes}",
        "EVALUATION.skip_get_obs_within_replan=true",
        "EVALUATION.robotwin_camera_layout=compact_288x256",
        f"EVALUATION.task_name={settings.task_name}",
        "EVALUATION.task_config=demo_clean",
        f"seed={settings.seed}",
        f"EVALUATION.output_dir={settings.eval_output_dir}",
    ]
    with initialize_config_dir(config_dir=os.path.join(IMAGEWAM_ROOT, "configs"), version_base="1.3"):
        return compose("sim_robotwin.yaml", overrides=overrides)


def build_usr_args(settings: RunSettings, policy_yaml: str) -> dict[str, Any]:
    cfg = _compose(settings)
    ev = cfg.EVALUATION
    ov: list[str] = []
    _append_override(ov, "task_name", ev.task_name)
    _append_override(ov, "task_config", ev.task_config)
    _append_override(ov, "ckpt_setting", os.path.realpath(str(cfg.ckpt)))
    _append_override(ov, "seed", cfg.seed)
    _append_override(ov, "policy_name", ev.policy_name)
    _append_override(ov, "instruction_type", ev.instruction_type)
    _append_override(ov, "eval_num_episodes", ev.eval_num_episodes)
    _append_override(ov, "sim_cfg_path", os.path.join(IMAGEWAM_ROOT, "configs", "sim_robotwin.yaml"))
    _append_override(ov, "sim_task", SIM_TASK)
    _append_override(ov, "eval_output_dir", settings.eval_output_dir)
    _append_override(ov, "mixed_precision", cfg.mixed_precision)
    _append_override(ov, "device", ev.device)
    _append_override(ov, "dataset_stats_path", os.path.realpath(str(ev.dataset_stats_path)))
    _append_override(ov, "action_horizon", ev.action_horizon)
    _append_override(ov, "replan_steps", ev.replan_steps)
    _append_override(ov, "num_inference_steps", ev.num_inference_steps)
    _append_override(ov, "sigma_shift", ev.sigma_shift)
    _append_override(ov, "text_cfg_scale", ev.text_cfg_scale)
    _append_override(ov, "negative_prompt", ev.negative_prompt)
    _append_override(ov, "rand_device", ev.rand_device)
    _append_override(ov, "tiled", ev.tiled)
    _append_override(ov, "timing_enabled", ev.timing_enabled)
    _append_override(ov, "robotwin_camera_layout", ev.robotwin_camera_layout)
    _append_override(ov, "apply_action_transform_backward", ev.apply_action_transform_backward)
    _append_override(ov, "model_overrides", OmegaConf.to_container(cfg.model, resolve=True))
    _append_override(ov, "skip_get_obs_within_replan", ev.skip_get_obs_within_replan)

    with open(policy_yaml, "r", encoding="utf-8") as f:
        usr_args: dict[str, Any] = yaml.safe_load(f)
    for i in range(0, len(ov), 2):
        key = ov[i].lstrip("-")
        value: Any = ov[i + 1]
        try:
            value = eval(value)
        except Exception:
            pass
        usr_args[key] = value
    return usr_args
