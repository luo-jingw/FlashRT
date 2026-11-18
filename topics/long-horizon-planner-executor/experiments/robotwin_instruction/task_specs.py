"""RoboTwin 2.0 tasks of the instruction experiment and the source of each task's "other task" instruction.

The "other" instruction comes either from another task's unseen templates, filled with this episode's objects
(`other_task` + `param_map`, used when this scene contains the objects the other task names), or from a fixed
sentence of another task (`other_fixed`, used when the other task's objects are absent from this scene).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    name: str
    other_task: str
    # other task's placeholder -> this episode's info key; empty when `other_fixed` is used
    param_map: dict[str, str] = field(default_factory=dict)
    other_fixed: str = ""

    def uses_template(self) -> bool:
        return self.other_fixed == ""


TASK_SPECS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in [
        # Same scene generator; only the side (left / right of B) differs.
        TaskSpec("place_a2b_left", "place_a2b_right", {"{A}": "{A}", "{B}": "{B}", "{a}": "{a}"}),
        TaskSpec("place_a2b_right", "place_a2b_left", {"{A}": "{A}", "{B}": "{B}", "{a}": "{a}"}),
        # The can-and-pot scene also affords lifting the pot.
        TaskSpec("move_can_pot", "lift_pot", {"{A}": "{A}"}),
        TaskSpec("lift_pot", "move_can_pot", other_fixed="Pick up the sauce can and set it beside the kitchen pot."),
        TaskSpec("click_bell", "beat_block_hammer", other_fixed="Grab the hammer and beat the block."),
        TaskSpec("beat_block_hammer", "click_bell", other_fixed="Click the top center of the bell on the table."),
    ]
}
