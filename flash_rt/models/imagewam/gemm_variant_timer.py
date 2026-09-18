"""Device timer for GEMM variant tuning (`gemm_variant_tuner.VariantTimer`).

At ImageWAM's ActionDiT shapes one GEMM takes on the order of 10 us, the
same order as a Python launch through pybind. Timing eager launches
would measure the host, not the kernel, and the served pipeline replays
a CUDA graph with no host in the loop. So each batch is captured once
into its own CUDA graph (`reps` copies back to back), and the graphs are
replayed round robin, `samples` times each, bracketed by CUDA events.
Interleaving the candidates keeps a clock or co-tenant load change from
landing entirely on one candidate. The reported value is the median
replay time divided by the launches it contains.
"""
from __future__ import annotations

import statistics
from typing import Callable, Sequence

import torch


def median_us_per_launch(samples_ms: Sequence[float], launches: int) -> float:
    """Median of CUDA-event samples (ms per replay) as microseconds per
    launch."""
    return statistics.median(samples_ms) * 1000.0 / float(launches)


class CudaGraphVariantTimer:
    """`VariantTimer` over CUDA graphs and CUDA events (any CUDA device)."""

    def __init__(self, *, reps: int = 4, samples: int = 15, warmup: int = 3):
        if reps < 1 or samples < 1 or warmup < 0:
            raise ValueError(f"reps={reps}, samples={samples}, warmup={warmup}")
        self._reps = int(reps)
        self._samples = int(samples)
        self._warmup = int(warmup)

    def us_per_launch(self, batches: Sequence[Callable[[int], None]],
                      launches_per_batch: int) -> tuple[float, ...]:
        if len(batches) == 0:
            return ()
        stream = torch.cuda.Stream()
        graphs: list[torch.cuda.CUDAGraph] = []
        for batch in batches:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                batch(stream.cuda_stream)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(self._reps):
                    batch(stream.cuda_stream)
            graphs.append(graph)
        torch.cuda.synchronize()

        for _ in range(self._warmup):
            for graph in graphs:
                graph.replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples: list[list[float]] = [[] for _ in graphs]
        for _ in range(self._samples):
            for idx, graph in enumerate(graphs):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples[idx].append(start.elapsed_time(end))
        launches = self._reps * int(launches_per_batch)
        result = tuple(median_us_per_launch(s, launches) for s in samples)
        del graphs
        torch.cuda.synchronize()
        return result
