"""Data types exchanged between the runtime, the environment, the executor and the planner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    """What the environment returns at one control step."""
    t: float                                   # simulated time, seconds
    images: dict[str, Any]                     # camera name -> image (numpy array or PIL image)
    state: list[float]                         # robot proprioception
    events: tuple[str, ...] = ()               # environment events at this step, e.g. "subtask_done"
    info: dict[str, Any] = field(default_factory=dict)   # environment-specific ground truth, never shown to planners


@dataclass(frozen=True)
class ActionChunk:
    """Actions produced by one executor call."""
    actions: list[list[float]]
    instruction: str | None                    # the instruction the executor was conditioned on
    executor_ms: float                         # executor wall time for this call


@dataclass(frozen=True)
class PlannerSnapshot:
    """Planner input: everything observed up to the snapshot time."""
    t: float
    task: str                                  # the episode-level task instruction
    history: tuple[Observation, ...]           # observations from the start of the episode up to t, in time order


@dataclass(frozen=True)
class PlannerResult:
    """Planner output for one call."""
    instruction: str | None                    # command for the executor; None keeps the current one
    status: str                                # planner-specific label, e.g. "continue", "next", "replan", "subtask"
    raw_text: str
    parse_ok: bool
    prefill_ms: float = 0.0                    # measured planner times (0 for planners that do not run a model)
    decode_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class PlannerCallRecord:
    t_snapshot: float
    latency_s: float                           # L applied by the runtime
    latency_source: str                        # "injected" or "measured"
    t_effective: float                         # when the result became the current decision
    t_first_use: float | None                  # when the executor first used it; None if never used
    result: PlannerResult


@dataclass
class ChunkRecord:
    t: float
    instruction: str | None
    executor_ms: float


@dataclass
class EpisodeResult:
    seed: int
    protocol: str                              # "sync" or "async"
    success: bool
    subtasks_done: int
    subtasks_total: int
    steps: int                                 # control steps the robot executed
    sim_time_s: float                          # including time frozen in sync mode
    calls: list[PlannerCallRecord]
    chunks: list[ChunkRecord]
