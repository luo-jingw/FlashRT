"""R1: ImageWAM executor + Qwen3-VL-2B planner co-located on one RTX 5090 (the 5090 counterpart of Thor row B18).

The executor runs periodically: a tick is released every PERIOD_MS; each tick stages one real LIBERO observation
(VAE outside the graph) and replays the captured ImageWAM graph on the executor stream. Response latency of a tick =
finish - release, so it includes any wait behind the planner. A tick that finishes after the next release is a miss
at deadline = PERIOD_MS; latencies are also reported against the isolated p99 x 1.5.

Planner modes (planner = planner_qwen3vl.py, one image + short prompt -> 32 tokens, at HZ calls/s, 0 = back-to-back):
  isolated  no planner
  serial    same process, one thread: a due planner call runs between two executor ticks
  process   separate planner process (default time-slicing between two CUDA contexts)
  prio      same process, planner in a second thread on a low-priority stream, executor on the highest-priority stream
  mps<P>    MPS daemon running (started by run_r1.sh); planner process limited to P % of SMs, executor to 100 - P %
  green<P>  same process, two green contexts (green_partition.py): planner thread on P % of SMs, executor on the rest
  greenalone<P>  as green<P> without the planner (executor on 100 - P % of SMs)
DRAM / SM activity: nvidia-smi utilization.{gpu,memory} sampled every 100 ms during the timed window.

EXEC_PRECISION overrides the executor precision (default fp8). NO_SMI=1 skips the nvidia-smi sampler (devices whose nvidia-smi lacks these fields). DROP_TEXT_ENCODER=1 frees the text encoder after set_prompt; START_AT=<epoch s> delays the timed window to that wall-clock time (used by run_r3.sh to align several executors).

usage: python colocation_bench.py MODE HZ DURATION_S OUT_JSON
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _imagewam_libero_frames import center_crop_resize  # noqa: E402
from planner_qwen3vl import Planner  # noqa: E402

import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

MODE, HZ, DURATION, OUT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
PRECISION = os.environ.get("EXEC_PRECISION", "fp8")
PERIOD_MS = float(os.environ.get("PERIOD_MS", "50"))
HERE = os.path.dirname(os.path.abspath(__file__))


def observation() -> dict:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)  # timing only; content does not change the graph
    return {"view1": torch.from_numpy(center_crop_resize(frame, 224, 224)),
            "view2": torch.from_numpy(center_crop_resize(frame, 224, 224)),
            "proprio": np.zeros(8, dtype=np.float32)}


SMI_FIELDS = ["util_gpu", "util_mem", "clk_sm", "clk_mem", "power_W"]


exec_stream = torch.cuda.Stream(priority=-1) if MODE == "prio" else torch.cuda.Stream()
with torch.cuda.stream(exec_stream):
    fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision=PRECISION,
                         ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                         qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                         dataset_stats_path=os.environ["DATASET_STATS_PATH"])
    fe.set_prompt("A video recorded from a robot's point of view executing the following instruction: "
                  "put both the alphabet soup and the tomato sauce in the basket")
obs = observation()
if os.environ.get("DROP_TEXT_ENCODER"):  # the prompt is fixed for the run; free the Qwen3-4B text encoder (R3, N processes)
    fe._qwen3 = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def tick() -> None:
    with torch.cuda.stream(exec_stream):
        fe.stage_inputs(obs)
        fe._graph.replay()
    exec_stream.synchronize()


green = None
if MODE.startswith("green"):
    from green_partition import GreenPartition
    pct = int(MODE.removeprefix("greenalone").removeprefix("green"))
    green = GreenPartition(max(1, round(torch.cuda.get_device_properties(0).multi_processor_count * pct / 100)))
    exec_stream = green.rest.stream
    fe._graph_stream = exec_stream  # a graph captured on another stream replays on all SMs; recapture on the partition
    fe._capture_graph()

for _ in range(10):
    tick()

planner, worker, planner_recs = None, None, []
if MODE in ("serial", "prio") or (green is not None and not MODE.startswith("greenalone")):
    planner = Planner(green.part.stream if green is not None else torch.cuda.Stream(priority=0))
    px = planner.inputs()
    for _ in range(3):
        planner.call(px)
elif MODE == "process" or MODE.startswith("mps"):
    env = dict(os.environ)
    if MODE.startswith("mps"):
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = MODE[3:]
    worker = subprocess.Popen([sys.executable, os.path.join(HERE, "planner_qwen3vl.py"), "worker", str(HZ),
                               str(DURATION), OUT + ".planner.json"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, env=env)
    while (line := worker.stdout.readline()).strip() != "READY":
        if line == "":
            raise RuntimeError(f"planner worker exited before READY (code {worker.wait()})")

smi = None if os.environ.get("NO_SMI") else subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw",
                        "--format=csv,noheader,nounits", "--loop-ms=100"], stdout=subprocess.PIPE, text=True)


def planner_thread(t_end: float) -> None:
    nxt = time.time()
    while time.time() < t_end:
        if HZ > 0:
            if time.time() < nxt:
                time.sleep(nxt - time.time())
            nxt += 1.0 / HZ
        planner_recs.append(planner.call(px).__dict__)


if os.environ.get("START_AT"):  # several executor processes (R3) start their timed windows together
    time.sleep(max(0.0, float(os.environ["START_AT"]) - time.time()))
t0 = time.time()
t_end = t0 + DURATION
if worker is not None:
    worker.stdin.write("go\n")
    worker.stdin.flush()
if planner is not None and MODE != "serial":
    threading.Thread(target=planner_thread, args=(t_end,), daemon=True).start()

ticks, release, next_plan = [], t0, t0
period = PERIOD_MS / 1e3
while release < t_end:
    now = time.time()
    if now < release:
        time.sleep(release - now)
    if MODE == "serial" and (HZ == 0 or time.time() >= next_plan):
        planner_recs.append(planner.call(px).__dict__)
        next_plan += 1.0 / HZ if HZ > 0 else 0
    start = time.time()
    tick()
    fin = time.time()
    ticks.append([release, start, fin])
    release += period
    if fin > release:  # overran: next tick released immediately, the schedule slips
        release = fin
smi_rows = []
if smi is not None:
    smi.terminate()
    smi_rows = [[float(v) for v in line.split(", ")] for line in smi.stdout.read().strip().splitlines() if line]
if worker is not None:
    worker.wait()
    planner_recs = json.load(open(OUT + ".planner.json"))["records"]
time.sleep(0.2)

lat = np.array([(f - r) * 1e3 for r, s, f in ticks])
svc = np.array([(f - s) * 1e3 for r, s, f in ticks])
smi = np.array(smi_rows)
pc = planner_recs
res = {"mode": MODE, "planner_hz": HZ, "period_ms": PERIOD_MS, "precision": PRECISION, "duration_s": DURATION,
       "gpu": torch.cuda.get_device_name(), "ticks": len(ticks),
       "torch_reserved_GB": round(torch.cuda.memory_reserved() / 1e9, 2),
       "green_sm": None if green is None else {"device": green.device_sm_count, "planner": green.part.sm_count, "executor": green.rest.sm_count},
       "latency_ms": {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99)), "max": float(lat.max())},
       "service_ms": {"p50": float(np.percentile(svc, 50)), "p99": float(np.percentile(svc, 99))},
       "miss_rate_at_period": float((lat > PERIOD_MS).mean()),
       "planner_calls": len(pc),
       "planner_call_ms_p50": float(np.median([(c["t_end"] - c["t_start"]) * 1e3 for c in pc])) if pc else None,
       "planner_prefill_ms_p50": float(np.median([c["prefill_ms"] for c in pc])) if pc else None,
       "planner_decode_ms_p50": float(np.median([c["decode_ms"] for c in pc])) if pc else None,
       "smi_samples": len(smi),
       "smi_mean": dict(zip(SMI_FIELDS, [round(float(v), 1) for v in smi.mean(0)])) if len(smi) else None,
       "latency_series_ms": [round(float(x), 3) for x in lat]}
json.dump(res, open(OUT, "w"))
print(f"{MODE} hz={HZ}: ticks {len(ticks)} lat p50 {res['latency_ms']['p50']:.2f} p99 {res['latency_ms']['p99']:.2f} "
      f"max {res['latency_ms']['max']:.1f} ms, miss@{PERIOD_MS:.0f}ms {res['miss_rate_at_period']*100:.1f}%, "
      f"planner calls {len(pc)} (call p50 {res['planner_call_ms_p50']}), smi {res['smi_mean']}", flush=True)
