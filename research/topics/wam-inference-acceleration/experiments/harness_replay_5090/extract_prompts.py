"""System prompts of VoLoAgent's Full loop, read from its source, for the configuration of the agent harness's
`eval_agent.sh` (`--mode subgoal --failure-monitor vlm --recovery-mode replan_tools --use-front-camera`, stack mode on,
no point tools, prompt style default).

String constants are evaluated from the AST (literals, names of other literals, `+`); nothing of VoLoAgent is imported.
The monitor prompt is the output of `failure_handlers/vlm.py:_build_system_prompt`, a pure string builder: only that
function definition is compiled and called, with the action constants of `failure_handlers/base.py`.

usage: python -I extract_prompts.py VOLOAGENT_PACKAGE_DIR OUT_JSON
       (VOLOAGENT_PACKAGE_DIR = .../third_party/VoLoAgent/vlm_orchestrator)
"""
import ast
import json
import subprocess
import sys
from pathlib import Path


def module_strings(path: Path, known: dict[str, str]) -> dict[str, str]:
    """Module-level `NAME = <expr>` whose expr is built from string literals, known names and `+`."""
    def ev(node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in out:
            return out[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return ev(node.left) + ev(node.right)
        raise ValueError(ast.dump(node)[:80])

    out = dict(known)
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ev(node.value)
            except ValueError:
                pass
    return out


def function_source(path: Path, name: str) -> ast.Module:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.Module(body=[node], type_ignores=[])
    raise KeyError(name)


def main(pkg: Path, out: Path) -> None:
    base = module_strings(pkg / "failure_handlers" / "base.py", {})
    sb = module_strings(pkg / "strategies" / "subgoal_base.py", {})
    sg = module_strings(pkg / "strategies" / "subgoal.py", sb)
    front_note = sb["_FRONT_CAM_NOTE"]

    ns: dict = {k: base[k] for k in ("ACTION_NEXT", "ACTION_CONTINUE", "ACTION_REPLAN", "ACTION_GRASP", "ACTION_PLACE")}
    ns["POINT_CONVENTION"] = "<unused: no point tools>"
    exec(compile(function_source(pkg / "failure_handlers" / "vlm.py", "_build_system_prompt"), "vlm.py", "exec"), ns)
    replan_tools = {ns[k] for k in ("ACTION_NEXT", "ACTION_CONTINUE", "ACTION_REPLAN", "ACTION_GRASP", "ACTION_PLACE")}
    monitor = ns["_build_system_prompt"](replan_tools, use_front_camera=True, stack_mode_enabled=True,
                                         point_tools=frozenset(), primary_label="Front camera")

    commit = subprocess.run(["git", "-C", str(pkg.parent), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    json.dump({
        "source": "VoLoAgent " + commit,
        "decompose": {"text": sg["DECOMPOSE_SYSTEM_PROMPT"] + front_note,
                      "from": "strategies/subgoal.py DECOMPOSE_SYSTEM_PROMPT + strategies/subgoal_base.py _FRONT_CAM_NOTE"},
        "monitor": {"text": monitor,
                    "from": "failure_handlers/vlm.py _build_system_prompt(replan_tools, use_front_camera=True, "
                            "stack_mode_enabled=True, point_tools=(), primary_label='Front camera')"},
        "replan": {"text": sb["RECYCLE_SYSTEM_PROMPT"] + front_note,
                   "from": "strategies/subgoal_base.py RECYCLE_SYSTEM_PROMPT + _FRONT_CAM_NOTE"},
    }, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
