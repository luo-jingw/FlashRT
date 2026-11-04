"""ImageWAM LIBERO fp8 executor for the replay, set up as in `colocation_qwen27b_5090/colocation_bench.py`: load, one
instruction, graph recaptured on the executor's stream, text encoder and FP16 source weights freed (`slim_executor`).
One query = stage inputs + graph replay on that stream, timed with CUDA events and the wall clock.

The instruction is fixed for the process (the text encoder is freed after `set_prompt`); `instruction_switch_ms`
measures the cost of a new instruction before the encoder is freed.
"""
import gc
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
sys.path.insert(0, os.path.join(HERE, "..", "colocation_qwen27b_5090"))
from _imagewam_libero_frames import center_crop_resize  # noqa: E402
from exec_slim import record_quantized_sources, slim_executor  # noqa: E402

import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

PROMPT_PREFIX = "A video recorded from a robot's point of view executing the following instruction: "


@dataclass
class QueryRecord:
    release: float       # wall-clock time the query was due
    start: float
    end: float
    gpu_ms: float        # CUDA-event span: staging + graph replay

    @property
    def latency_ms(self) -> float:
        return (self.end - self.release) * 1e3


def observation() -> dict:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)  # timing only; content does not change the graph
    return {"view1": torch.from_numpy(center_crop_resize(frame, 224, 224)),
            "view2": torch.from_numpy(center_crop_resize(frame, 224, 224)),
            "proprio": np.zeros(8, dtype=np.float32)}


def load_frontend():
    return F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision="fp8",
                           ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                           qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                           dataset_stats_path=os.environ["DATASET_STATS_PATH"])


class ImageWamExecutor:
    def __init__(self, stream: torch.cuda.Stream, instruction: str):
        sources = record_quantized_sources(F)
        self.fe = load_frontend()
        self.fe.set_prompt(PROMPT_PREFIX + instruction)
        torch.cuda.synchronize()
        self.obs = observation()
        self.stream = stream
        # recapture the served graph on the executor stream (green partition / priority), fresh pool
        self.fe._captures.clear()
        self.fe._graph = None
        gc.collect()
        torch.cuda.empty_cache()
        self.fe._graph_pool = torch.cuda.graph_pool_handle()
        self.fe._graph_stream = stream
        self.fe._capture_graph()
        self.slim = slim_executor(self.fe, sources)
        for _ in range(10):
            self.query(time.time())
        self.allocated_GB = torch.cuda.memory_allocated() / 1e9

    def query(self, release: float) -> QueryRecord:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.time()
        with torch.cuda.stream(self.stream):
            e0.record(self.stream)
            self.fe.stage_inputs(self.obs)
            self.fe._graph.replay()
            e1.record(self.stream)
        self.stream.synchronize()
        return QueryRecord(release, start, time.time(), e0.elapsed_time(e1))


def instruction_switch_ms(instructions: list[str]) -> dict:
    """Executor alone, text encoder resident: wall ms of `set_prompt` for each instruction (first and repeated), and the
    allocated memory with the encoder resident and after slimming."""
    sources = record_quantized_sources(F)
    fe = load_frontend()
    torch.cuda.synchronize()
    out = {"allocated_with_encoder_GB": torch.cuda.memory_allocated() / 1e9, "calls": []}
    for rep in range(2):
        for text in instructions:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fe.set_prompt(PROMPT_PREFIX + text)
            torch.cuda.synchronize()
            out["calls"].append({"instruction": text, "repeat": rep, "ms": (time.perf_counter() - t0) * 1e3})
    slim_executor(fe, sources)
    gc.collect()
    torch.cuda.empty_cache()
    out["allocated_slim_GB"] = torch.cuda.memory_allocated() / 1e9
    return out
