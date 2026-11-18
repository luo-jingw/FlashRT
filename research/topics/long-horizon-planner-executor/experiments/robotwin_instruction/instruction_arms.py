"""Instruction arms: which language instruction the executor receives in an episode.

`original` keeps the evaluator's own instruction (unseen templates). The other arms replace it after the
evaluator has drawn it, so the scene, the seed and the evaluator's RNG consumption are identical across arms.
"""
import random
from dataclasses import dataclass
from typing import Callable

from task_specs import TaskSpec

ARM_NAMES: tuple[str, ...] = ("original", "still", "other", "empty")
STILL_INSTRUCTION: str = "stay still and do not move"

# generate_episode_descriptions(task_name, episodes, max_descriptions) from RoboTwin's description/utils
DescribeFn = Callable[[str, list[dict[str, str]], int], list[dict[str, list[str]]]]


@dataclass(frozen=True)
class ArmInstruction:
    arm: str
    original: str
    given: str


class InstructionSelector:
    def __init__(self, arm: str, spec: TaskSpec, describe: DescribeFn, instruction_type: str) -> None:
        if arm not in ARM_NAMES:
            raise ValueError(f"unknown arm {arm}; expected one of {ARM_NAMES}")
        self.arm = arm
        self.spec = spec
        self.describe = describe
        self.instruction_type = instruction_type

    def select(self, original: str, episode_info: dict[str, str], seed: int) -> ArmInstruction:
        if self.arm == "original":
            given = original
        elif self.arm == "still":
            given = STILL_INSTRUCTION
        elif self.arm == "empty":
            given = ""
        else:
            given = self._other(episode_info, seed)
        return ArmInstruction(arm=self.arm, original=original, given=given)

    def _other(self, episode_info: dict[str, str], seed: int) -> str:
        if not self.spec.uses_template():
            return self.spec.other_fixed
        params = {dst: episode_info[src] for dst, src in self.spec.param_map.items()}
        random.seed(seed + 1)  # object-description choice inside the templates
        texts = self.describe(self.spec.other_task, [params], 1000000)[0][self.instruction_type]
        return random.Random(seed).choice(texts)
