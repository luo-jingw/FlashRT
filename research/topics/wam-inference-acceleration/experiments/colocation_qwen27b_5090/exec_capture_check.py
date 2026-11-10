"""Executor tick time by where its graph was captured (one process, the GPU otherwise idle):
(a) the graph captured by set_prompt, replayed on the default stream (`infer()` as served);
(b) the same graph replayed on a side stream;
(c) the graph recaptured on that side stream (as colocation_bench.py does), replayed there.
Per variant: 100 ticks after 20 warmup ticks, CUDA-event time of stage_inputs + replay, p50 / p99, and the maximum
absolute action difference against (a) with a fixed action noise.

usage: python exec_capture_check.py OUT_JSON
"""
import gc
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
from _imagewam_libero_frames import center_crop_resize  # noqa: E402

import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=os.environ.get("EXEC_PRECISION", "fp8"),
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"], dataset_stats_path=os.environ["DATASET_STATS_PATH"])
fe.set_prompt("A video recorded from a robot's point of view executing the following instruction: "
              "put both the alphabet soup and the tomato sauce in the basket")
frame = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
obs = {"view1": torch.from_numpy(center_crop_resize(frame, 224, 224)),
       "view2": torch.from_numpy(center_crop_resize(frame, 224, 224)), "proprio": np.zeros(8, dtype=np.float32)}
lat = fe._action_latent
noise = (0.01 * torch.randn(tuple(lat.shape), generator=torch.Generator().manual_seed(0))).to(lat.device, lat.dtype)


def run(stream: torch.cuda.Stream) -> tuple[list[float], np.ndarray]:
    times = []
    for i in range(120):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            e0.record(stream)
            fe.stage_inputs(obs, noise=noise)
            fe._graph.replay()
            e1.record(stream)
        stream.synchronize()
        if i >= 20:
            times.append(e0.elapsed_time(e1))
    return times, np.asarray(fe.read_actions(), dtype=np.float64)


res = {}
ref = None
side = torch.cuda.Stream()
for name, stream in (("a_setprompt_graph_default_stream", torch.cuda.current_stream()),
                     ("b_setprompt_graph_side_stream", side)):
    t, a = run(stream)
    ref = a if ref is None else ref
    res[name] = {"p50": float(np.percentile(t, 50)), "p99": float(np.percentile(t, 99)),
                 "max_abs_diff": float(np.abs(a - ref).max())}
fe._captures.clear()
fe._graph = None
gc.collect()
torch.cuda.empty_cache()
fe._graph_pool = torch.cuda.graph_pool_handle()
fe._graph_stream = side
fe._capture_graph()
t, a = run(side)
res["c_recaptured_side_stream"] = {"p50": float(np.percentile(t, 50)), "p99": float(np.percentile(t, 99)),
                                   "max_abs_diff": float(np.abs(a - ref).max())}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
