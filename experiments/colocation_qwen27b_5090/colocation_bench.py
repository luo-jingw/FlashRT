"""ImageWAM executor + Qwen3.8-27B planner (both FlashRT) co-located on one RTX 5090: one cell of the grid.

Executor: ImageWAM LIBERO, FlashRT EXEC_PRECISION (default fp8), 10 denoise steps. A tick is released every PERIOD_MS
(default 50); each tick stages one observation and replays the captured graph on the executor stream. Response latency
= finish - release; a tick finishing after the next release is a miss and delays that release. After `set_prompt` the
executor is slimmed (exec_slim.py: text encoder and FP16 sources of quantized linears freed) in every mode, so that
both models fit on the card.

Planner: planner_qwen27b.py (two LIBERO frames + planning prompt, PLANNER_CTX short | long, PLANNER_NEW_TOKENS
tokens), at HZ calls/s (0 = back-to-back).

Modes:
  isolated        no planner
  serial          one thread: a due planner call runs to completion between two executor ticks
  process         planner in a separate process (driver time-slicing between two contexts)
  prio            planner thread in the executor's process; executor stream priority -1, planner streams 0
  mps<P>          planner process under the MPS daemon (started by the runner) with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
                  P; the executor's own percentage comes from the runner's environment (mps100: no limit on either)
  green<P>        one process, two green contexts: planner thread on P % of the SMs, executor on the rest (executor
                  graph recaptured on its partition)
  greenalone<P>   green<P> without the planner
  greenhog<P>     green<P> with bandwidth_hog.py (back-to-back DRAM copies) in place of the planner
  coop            as prio, plus coop_gate.py: the planner starts each GPU segment (ViT, prefill chunk of
                  FLASHRT_QWEN36_TQ_PREFILL_CHUNK tokens, decode cycle) only while the executor is idle and the
                  segment's measured length fits before the executor's next release
PLANNER_READBACK_NOGIL=1 wraps `flash_rt_kernels.cuda_read_i32_sync` (the per-cycle readback of the accepted draft
count, which waits for the verify forward while holding the GIL) so that it first waits on the stream through
torch (GIL released), then reads (research-only instrumentation; the frontend code is unchanged).
CAPTURE_THREAD_LOCAL=1 makes `torch.cuda.graph` default to capture_error_mode="thread_local", so that a CUDA call from
the other thread does not invalidate a graph the planner captures lazily during the window (default "global").
GIL_SWITCH_MS sets the interpreter's thread switch interval (sys.setswitchinterval; default 5 ms) for the in-process
modes. GPU telemetry: nvidia-smi every 100 ms (timestamped: utilization, DRAM utilization, SM / memory clock, power).

usage: python colocation_bench.py MODE HZ DURATION_S OUT_JSON
"""
import dataclasses
import datetime
import gc
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
sys.path.insert(0, os.path.join(HERE, "..", "colocation_5090"))
sys.path.insert(0, HERE)
from _imagewam_libero_frames import center_crop_resize  # noqa: E402
from exec_slim import record_quantized_sources, slim_executor  # noqa: E402

import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

MODE, HZ, DURATION, OUT = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
PRECISION = os.environ.get("EXEC_PRECISION", "fp8")
WORKLOAD = ImageWAMWorkload.libero()
if os.environ.get("EXEC_STEPS"):
    WORKLOAD = dataclasses.replace(WORKLOAD, num_steps=int(os.environ["EXEC_STEPS"]))
PERIOD_MS = float(os.environ.get("PERIOD_MS", "50"))
if os.environ.get("GIL_SWITCH_MS"):
    sys.setswitchinterval(float(os.environ["GIL_SWITCH_MS"]) / 1e3)
CAPTURE_THREAD_LOCAL = os.environ.get("CAPTURE_THREAD_LOCAL") == "1"
if CAPTURE_THREAD_LOCAL:
    class _ThreadLocalGraph(torch.cuda.graph):
        def __init__(self, cuda_graph, pool=None, stream=None, capture_error_mode: str = "thread_local", **kwargs):
            super().__init__(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode, **kwargs)

    torch.cuda.graph = _ThreadLocalGraph
READBACK_NOGIL = os.environ.get("PLANNER_READBACK_NOGIL") == "1"
if READBACK_NOGIL:
    import flash_rt.flash_rt_kernels as _fvk
    _read_i32_sync = _fvk.cuda_read_i32_sync

    def _read_i32_sync_nogil(ptr: int, stream: int) -> int:
        torch.cuda.ExternalStream(stream).synchronize()  # torch releases the GIL while it waits
        return _read_i32_sync(ptr, stream)

    _fvk.cuda_read_i32_sync = _read_i32_sync_nogil
IN_PROCESS_PLANNER = MODE in ("serial", "prio", "coop") or (MODE.startswith("green") and not MODE.startswith("greenalone")
                                                    and not MODE.startswith("greenhog"))
SMI_FIELDS = ["util_gpu", "util_mem", "clk_sm", "clk_mem", "power_W"]


def observation() -> dict:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)  # timing only; content does not change the graph
    return {"view1": torch.from_numpy(center_crop_resize(frame, 224, 224)),
            "view2": torch.from_numpy(center_crop_resize(frame, 224, 224)),
            "proprio": np.zeros(8, dtype=np.float32)}


green = None
if MODE.startswith("green"):
    from green_partition import GreenPartition
    pct = int(MODE.removeprefix("greenalone").removeprefix("greenhog").removeprefix("green"))
    green = GreenPartition(max(1, round(torch.cuda.get_device_properties(0).multi_processor_count * pct / 100)))
    exec_stream = green.rest.stream
else:
    exec_stream = torch.cuda.Stream(priority=-1) if MODE in ("prio", "coop") else torch.cuda.Stream()
quantized_sources = record_quantized_sources(F)
# Loaded on the default stream (a side current stream during loading can race with the loaders' stream-0 kernels);
# the served graph is then recaptured on the executor stream: a graph runs on the SMs of the stream it was captured
# on (green partitions), and launches with that stream's priority.
fe = F.load_imagewam(os.environ["CKPT_PATH"], WORKLOAD, precision=PRECISION,
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                     dataset_stats_path=os.environ["DATASET_STATS_PATH"])
fe.set_prompt("A video recorded from a robot's point of view executing the following instruction: "
              "put both the alphabet soup and the tomato sauce in the basket")
torch.cuda.synchronize()
obs = observation()
_lat = fe._action_latent
_noise = (0.01 * torch.randn(tuple(_lat.shape), generator=torch.Generator().manual_seed(0))).to(_lat.device, _lat.dtype)
actions_ref = np.asarray(fe.infer(obs, action_noise=_noise)["actions"], dtype=np.float64)
# Drop the set_prompt capture (cached per text length) and give the recapture a fresh pool: a pool's blocks are tied to
# the capture stream, so recapturing on another stream into the old pool would keep both graphs' temporaries.
fe._captures.clear()
fe._graph = None
gc.collect()
torch.cuda.empty_cache()
fe._graph_pool = torch.cuda.graph_pool_handle()
fe._graph_stream = exec_stream
fe._capture_graph()
slim = slim_executor(fe, quantized_sources)
with torch.cuda.stream(exec_stream):
    fe.stage_inputs(obs, noise=_noise)
    fe._graph.replay()
exec_stream.synchronize()
actions_check = np.asarray(fe.read_actions(), dtype=np.float64)
actions_max_abs_diff = float(np.abs(actions_check - actions_ref).max())


tick_gpu_ms: list[float] = []
tick_host_ms: list[list[float]] = []  # per tick: [stage_inputs, graph launch, wait for completion] host wall ms


def tick() -> None:
    """One executor tick. Appends the event span (before staging -> after the graph; includes any host-side delay
    between the two records) to tick_gpu_ms, the replay-only event span to graph_gpu_ms, and the host split to
    tick_host_ms."""
    e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    h0 = time.perf_counter()
    with torch.cuda.stream(exec_stream):
        e0.record(exec_stream)
        fe.stage_inputs(obs)
        h1 = time.perf_counter()
        e1.record(exec_stream)
        fe._graph.replay()
        e2.record(exec_stream)
        h2 = time.perf_counter()
    exec_stream.synchronize()
    h3 = time.perf_counter()
    tick_gpu_ms.append(e0.elapsed_time(e2))
    graph_gpu_ms.append(e1.elapsed_time(e2))
    tick_host_ms.append([round((h1 - h0) * 1e3, 3), round((h2 - h1) * 1e3, 3), round((h3 - h2) * 1e3, 3)])


graph_gpu_ms: list[float] = []


for _ in range(10):
    tick()
tick_gpu_ms.clear()
graph_gpu_ms.clear()
tick_host_ms.clear()
exec_alloc_GB = torch.cuda.memory_allocated() / 1e9

planner, worker, planner_recs, hog, hog_rec, msgs, gate = None, None, [], None, {}, None, None
if MODE.startswith("greenhog"):
    from bandwidth_hog import BandwidthHog
    hog = BandwidthHog(green.part.stream)
elif IN_PROCESS_PLANNER:
    from planner_qwen27b import Planner, PlannerStreams
    if green is not None:
        streams = PlannerStreams(green.part.stream, green.new_stream(0), green.new_stream(0))
    else:
        streams = PlannerStreams(torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=0),
                                 torch.cuda.Stream(priority=0))
    planner = Planner(streams)
    msgs = planner.messages()
    if MODE == "coop":
        from coop_gate import CoopGate, ExecutorClock, install
        clock = ExecutorClock(PERIOD_MS / 1e3)
        gate = CoopGate(clock, [streams.main, streams.graph, streams.snap])
        install(planner.fe, gate)
    for _ in range(3):
        planner.call(msgs)
    if gate is not None:
        gate.enabled = True
elif MODE == "process" or MODE.startswith("mps"):
    env = dict(os.environ)
    if MODE.startswith("mps"):
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = MODE[3:]
    worker = subprocess.Popen([sys.executable, os.path.join(HERE, "planner_qwen27b.py"), "worker", str(HZ),
                               str(DURATION), OUT + ".planner.json"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, env=env)
    while (line := worker.stdout.readline()).strip() != "READY":
        if line == "":
            raise RuntimeError(f"planner worker exited before READY (code {worker.wait()})")
total_alloc_GB = torch.cuda.memory_allocated() / 1e9
free_GB, total_GB = (x / 1e9 for x in torch.cuda.mem_get_info())

smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,"
                        "power.draw", "--format=csv,noheader,nounits", "--loop-ms=100"], stdout=subprocess.PIPE,
                       text=True)


def foreign_gpu_pids() -> set[int]:
    """GPU compute processes of other users (another user's process on the GPU voids the cell)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.split()
    me, pids = os.getuid(), set()
    for p in out:
        try:
            if os.stat(f"/proc/{p}").st_uid != me:
                pids.add(int(p))
        except (FileNotFoundError, ValueError):
            pids.add(int(p) if p.isdigit() else -1)
    return pids


foreign_seen: set[int] = set()


def foreign_watch(t_end: float) -> None:
    while time.time() < t_end:
        foreign_seen.update(foreign_gpu_pids())
        time.sleep(2.0)


def planner_thread(t_end: float) -> None:
    nxt = time.time()
    while time.time() < t_end:
        if HZ > 0:
            if time.time() < nxt:
                time.sleep(nxt - time.time())
            nxt += 1.0 / HZ
        planner_recs.append(dataclasses.asdict(planner.call(msgs)))
        if gate is not None:
            gate.close()


def planner_graph_count() -> int | None:
    if planner is None:
        return None
    return sum(len(v) for k, v in vars(planner.fe.llm).items() if k.startswith("_captured_") and isinstance(v, dict))


graphs_at_start = planner_graph_count()
t0 = time.time()
t_end = t0 + DURATION
if worker is not None:
    worker.stdin.write("go\n")
    worker.stdin.flush()
if planner is not None and MODE != "serial":
    threading.Thread(target=planner_thread, args=(t_end,), daemon=True).start()
threading.Thread(target=foreign_watch, args=(t_end,), daemon=True).start()
hog_thread = None
if hog is not None:
    hog_thread = threading.Thread(target=lambda: hog_rec.update(hog.run(t_end).__dict__), daemon=True)
    hog_thread.start()

ticks, release, next_plan = [], t0, t0
period = PERIOD_MS / 1e3
while release < t_end:
    now = time.time()
    if now < release:
        time.sleep(release - now)
    if MODE == "serial" and (HZ == 0 or time.time() >= next_plan):
        planner_recs.append(dataclasses.asdict(planner.call(msgs)))
        next_plan += 1.0 / HZ if HZ > 0 else 0
    start = time.time()
    if gate is not None:
        clock.tick_start()
    tick()
    fin = time.time()
    ticks.append([release, start, fin])
    release += period
    if fin > release:  # overran: next tick released immediately, the schedule slips
        release = fin
    if gate is not None:
        clock.tick_end(release, fin - start)
if hog_thread is not None:
    hog_thread.join()
smi.terminate()
smi_rows = []
for line in smi.stdout.read().strip().splitlines():
    f = [v.strip() for v in line.split(",")]
    try:
        ts = datetime.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        smi_rows.append([ts] + [float(v) for v in f[1:]])
    except ValueError:
        continue
if worker is not None:
    worker.wait()
    planner_recs = json.load(open(OUT + ".planner.json"))["records"]
time.sleep(0.2)

lat = np.array([(f - r) * 1e3 for r, s, f in ticks])
svc = np.array([(f - s) * 1e3 for r, s, f in ticks])
smi_a = np.array(smi_rows) if smi_rows else np.zeros((0, 6))
smi_a = smi_a[(smi_a[:, 0] >= t0) & (smi_a[:, 0] <= t_end)] if len(smi_a) else smi_a
pc = planner_recs


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


res = {"mode": MODE, "gil_switch_ms": sys.getswitchinterval() * 1e3, "planner_readback_nogil": READBACK_NOGIL,
       "capture_thread_local": CAPTURE_THREAD_LOCAL,
       "planner_graphs_start_end": [graphs_at_start, planner_graph_count()], "planner_hz": HZ, "planner_ctx": os.environ.get("PLANNER_CTX", "short"),
       "planner_new_tokens": int(os.environ.get("PLANNER_NEW_TOKENS", "64")), "period_ms": PERIOD_MS,
       "precision": PRECISION, "steps": WORKLOAD.num_steps, "duration_s": DURATION, "t0": t0,
       "gpu": torch.cuda.get_device_name(), "ticks": len(ticks),
       "exec_slim": dataclasses.asdict(slim), "exec_actions_max_abs_diff_vs_default_capture": actions_max_abs_diff, "exec_allocated_GB": round(exec_alloc_GB, 3),
       "process_allocated_GB": round(total_alloc_GB, 3), "device_free_GB_at_start": round(free_GB, 3),
       "device_total_GB": round(total_GB, 3),
       "green_sm": None if green is None else {"device": green.device_sm_count, "planner": green.part.sm_count,
                                               "executor": green.rest.sm_count},
       "latency_ms": {"p50": pct(lat, 50), "p99": pct(lat, 99), "max": float(lat.max())},
       "service_ms": {"p50": pct(svc, 50), "p99": pct(svc, 99)},
       "tick_gpu_ms": {"p50": pct(tick_gpu_ms, 50), "p99": pct(tick_gpu_ms, 99)},
       "tick_gpu_series_ms": [round(float(x), 3) for x in tick_gpu_ms],
       "graph_gpu_ms": {"p50": pct(graph_gpu_ms, 50), "p99": pct(graph_gpu_ms, 99)},
       "graph_gpu_series_ms": [round(float(x), 3) for x in graph_gpu_ms],
       "tick_host_ms": {k: {"p50": pct([h[i] for h in tick_host_ms], 50), "p99": pct([h[i] for h in tick_host_ms], 99)}
                        for i, k in enumerate(("stage_inputs", "graph_launch", "wait"))},
       "tick_host_series_ms": tick_host_ms,
       "miss_rate_at_period": float((lat > PERIOD_MS).mean()),
       "planner_calls": len(pc),
       "planner_last_text": (planner.last_text if planner is not None else
                             json.load(open(OUT + ".planner.json")).get("last_text") if worker is not None else None),
       "planner_call_ms_p50": float(np.median([(c["t_end"] - c["t_start"]) * 1e3 for c in pc])) if pc else None,
       "planner_prefill_ms_p50": float(np.median([c["prefill_ms"] for c in pc])) if pc else None,
       "planner_decode_ms_p50": float(np.median([c["decode_ms"] for c in pc])) if pc else None,
       "planner_vit_ms_p50": float(np.median([c["vit_ms"] for c in pc])) if pc else None,
       "planner_prompt_tokens": pc[0]["prompt_tokens"] if pc else None,
       "smi_samples": len(smi_a),
       "smi_mean": dict(zip(SMI_FIELDS, [round(float(v), 1) for v in smi_a[:, 1:].mean(0)])) if len(smi_a) else None,
       "smi_series": [[round(r[0] - t0, 3)] + [float(v) for v in r[1:]] for r in smi_a],
       "foreign_gpu_pids": sorted(foreign_seen),
       "hog": hog_rec or None,
       "coop": None if gate is None else {"prefill_chunk": os.environ.get("FLASHRT_QWEN36_TQ_PREFILL_CHUNK"),
                                          "waits": dict(gate.stats.waits),
                                          "wait_s": {k: round(v, 3) for k, v in gate.stats.wait_s.items()},
                                          "unfit": dict(gate.stats.unfit),
                                          "budget_ms": {k: round(gate.budget_s(k) * 1e3, 2) for k in gate.hist}},
       "latency_series_ms": [round(float(x), 3) for x in lat],
       "ticks_wall_s": [[round(x - t0, 4) for x in t] for t in ticks],
       "planner_records": [dict(c, t_start=round(c["t_start"] - t0, 4), t_end=round(c["t_end"] - t0, 4)) for c in pc]}
json.dump(res, open(OUT, "w"))
print(f"{MODE} ctx={res['planner_ctx']} hz={HZ}: ticks {len(ticks)} lat p50 {res['latency_ms']['p50']:.2f} p99 "
      f"{res['latency_ms']['p99']:.2f} max {res['latency_ms']['max']:.1f} ms (service p50 {res['service_ms']['p50']:.1f}, "
      f"tick GPU p50 {res['tick_gpu_ms']['p50']:.1f} p99 {res['tick_gpu_ms']['p99']:.1f}, graph GPU p50 "
      f"{res['graph_gpu_ms']['p50']:.1f} p99 {res['graph_gpu_ms']['p99']:.1f}, host stage/launch/wait p50 "
      f"{res['tick_host_ms']['stage_inputs']['p50']:.1f}/{res['tick_host_ms']['graph_launch']['p50']:.1f}/"
      f"{res['tick_host_ms']['wait']['p50']:.1f}), miss@{PERIOD_MS:.0f}ms "
      f"{res['miss_rate_at_period']*100:.1f}%, planner calls {len(pc)} (call p50 {res['planner_call_ms_p50']}, "
      f"prefill {res['planner_prefill_ms_p50']}, decode {res['planner_decode_ms_p50']}), hog {res['hog']}, "
      f"green {res['green_sm']}, graphs {res['planner_graphs_start_end']}, act diff {actions_max_abs_diff:.2e}, foreign {res['foreign_gpu_pids']}, mem exec {res['exec_allocated_GB']} proc {res['process_allocated_GB']} free "
      f"{res['device_free_GB_at_start']}, smi {res['smi_mean']}", flush=True)
sys.stdout.flush()
os._exit(0)  # skip interpreter teardown: destroying green contexts and graphs at exit aborts the process
