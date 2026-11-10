"""Cooperative slack scheduling of the planner around a periodic executor (same process, no SM partition).

The executor loop owns an `ExecutorClock` (busy flag, next release time, end time of the last tick, executor service
time) and calls `stop()` when it ends. Before each planner GPU segment the planner thread calls `CoopGate.wait(kind)`:
it drains the planner's own streams, then waits until the executor is idle and the segment's budget fits before the
executor's next release. A segment longer than the largest slack (period - executor service time) never fits; it starts
only at the beginning of a slack window (right after a tick ends) and overlaps the next tick by the difference.
Segments (kinds):
  vit      the image encoder of one call;
  chunk    one chunk of the chunked prefill (chunk length FLASHRT_QWEN36_TQ_PREFILL_CHUNK);
  cycle    one speculative-decode cycle (MTP draft chain + verify), gated at the draft chain.
Budget of a kind = 1.1 x the longest of its last 8 measured segments (gate release to the next drain), seeded from
COOP_SEED_MS. No gating while a CUDA graph is being captured or before `enabled` is set.

`install(vl_frontend, gate)` wraps three bound methods of the planner instance (research-only instrumentation; the
frontend code is unchanged): `_encode_images`, `llm.forward_own_decode_K_nvfp4_fp8kv` (eager prefill chunks, S > 1)
and `llm._ensure_mtp_chain_graph_nvfp4` (start of each decode cycle).
"""
import collections
import os
import threading
import time
from dataclasses import dataclass, field

import torch

SEED_MS = {"vit": 15.0, "chunk": 25.0, "cycle": 25.0}
SEED_MS.update({k: float(v) for k, v in (kv.split("=") for kv in os.environ.get("COOP_SEED_MS", "").split(",") if kv)})


@dataclass
class ExecutorClock:
    period_s: float
    busy: bool = False
    next_release: float = 0.0
    last_end: float = 0.0
    service_s: float = 0.0
    generation: int = 0
    cond: threading.Condition = field(default_factory=threading.Condition)

    def tick_start(self) -> None:
        with self.cond:
            self.busy = True

    def tick_end(self, next_release: float, service_s: float) -> None:
        with self.cond:
            self.next_release = next_release
            self.last_end = time.time()
            self.service_s = service_s if self.service_s == 0.0 else 0.8 * self.service_s + 0.2 * service_s
            self.busy = False
            self.generation += 1
            self.cond.notify_all()

    def stop(self) -> None:
        """The executor loop has ended: no further release, so every waiting or later planner segment fits."""
        with self.cond:
            self.next_release = float("inf")
            self.busy = False
            self.generation += 1
            self.cond.notify_all()


@dataclass
class GateStats:
    waits: dict = field(default_factory=lambda: collections.Counter())
    wait_s: dict = field(default_factory=lambda: collections.Counter())
    unfit: dict = field(default_factory=lambda: collections.Counter())


class CoopGate:
    def __init__(self, clock: ExecutorClock, streams: list):
        self.clock = clock
        self.streams = streams
        self.enabled = False
        self.hist = {k: collections.deque([v / 1e3], maxlen=8) for k, v in SEED_MS.items()}
        self.open_kind, self.open_t = None, 0.0
        self.stats = GateStats()

    def budget_s(self, kind: str) -> float:
        return 1.1 * max(self.hist[kind])

    def wait(self, kind: str) -> None:
        if not self.enabled or torch.cuda.is_current_stream_capturing():
            return
        for s in self.streams:
            s.synchronize()
        now = time.time()
        if self.open_kind is not None:
            self.hist[self.open_kind].append(now - self.open_t)
        need = self.budget_s(kind)
        c = self.clock
        fits = need <= c.period_s - c.service_s
        t0 = now
        with c.cond:
            while True:
                now = time.time()
                if not c.busy and c.generation > 0:
                    if c.next_release - now >= need:
                        break
                    if not fits and now - c.last_end < 0.003:
                        break
                gen = c.generation
                c.cond.wait_for(lambda: c.generation != gen, timeout=c.period_s)
        self.stats.waits[kind] += 1
        self.stats.wait_s[kind] += time.time() - t0
        self.stats.unfit[kind] += 0 if fits else 1
        self.open_kind, self.open_t = kind, time.time()

    def close(self) -> None:
        """End of a planner call: the open segment ends when the call returns."""
        if self.open_kind is not None:
            for s in self.streams:
                s.synchronize()
            self.hist[self.open_kind].append(time.time() - self.open_t)
            self.open_kind = None


def install(vl, gate: CoopGate) -> None:
    llm = vl.llm
    enc, fwd, chain = vl._encode_images, llm.forward_own_decode_K_nvfp4_fp8kv, llm._ensure_mtp_chain_graph_nvfp4

    def encode_images(*a, **k):
        gate.wait("vit")
        return enc(*a, **k)

    def forward_chunk(input_ids, *a, **k):
        if int(input_ids.shape[1]) > 1:
            gate.wait("chunk")
        return fwd(input_ids, *a, **k)

    def mtp_chain(*a, **k):
        gate.wait("cycle")
        return chain(*a, **k)

    vl._encode_images = encode_images
    llm.forward_own_decode_K_nvfp4_fp8kv = forward_chunk
    llm._ensure_mtp_chain_graph_nvfp4 = mtp_chain
