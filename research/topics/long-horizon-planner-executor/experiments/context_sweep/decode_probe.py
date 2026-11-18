"""Decode time against decode position: one prompt, a fixed number of decoded tokens (no stop token), timed per
committed chunk. Run twice in one process: the first run records the CUDA graphs, the second is reported. Shows whether
the per-token decode time grows with the number of tokens already generated.

usage: PLANNER_CKPT=DIR PLANNER_MTP_DIR=DIR python decode_probe.py {rtx|thor} OUT_JSON CONTEXT_TOKENS DECODE_TOKENS
"""
import json
import sys
import time

from context_sweep import Sweep, SPEC_K  # noqa: E402  (sets the FlashRT environment variables)
import torch  # noqa: E402

WINDOW = 256


@torch.inference_mode()
def timed_decode(sw: Sweep, ids: torch.Tensor, budget: int) -> list[tuple[int, float]]:
    sw.llm.prefill_long_ctx_nvfp4_agent(ids, max_new_tokens=budget, K=SPEC_K)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    marks: list[tuple[int, float]] = []
    n = 0
    for chunk in sw.llm.decode_long_ctx_nvfp4_committed_stream(max_new_tokens=budget, K=SPEC_K):
        n += len(chunk)
        torch.cuda.synchronize()
        marks.append((n, (time.perf_counter() - t0) * 1e3))
    return marks


def main() -> None:
    frontend, out, ctx, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    sw = Sweep(frontend, ctx + budget)
    ids = sw.prompt_ids(ctx)
    runs = []
    for r in range(2):
        marks = timed_decode(sw, ids, budget)
        windows = []
        for w0 in range(0, budget, WINDOW):
            inside = [(n, t) for n, t in marks if w0 < n <= w0 + WINDOW]
            before = [(n, t) for n, t in marks if n <= w0]
            if not inside:
                continue
            n0, t0 = before[-1] if before else (0, 0.0)
            n1, t1 = inside[-1]
            windows.append({"from": n0, "to": n1, "ms_per_token": (t1 - t0) / max(1, n1 - n0)})
        runs.append({"run": r, "total_ms": marks[-1][1], "tokens": marks[-1][0], "windows": windows})
        print(f"run {r}: {marks[-1][0]} tokens in {marks[-1][1]:.0f} ms; ms/token by window: "
              + " ".join(f"{w['ms_per_token']:.1f}" for w in windows), flush=True)
    with open(out, "w") as f:
        json.dump({"frontend": frontend, "context_tokens": int(ids.shape[1]), "decode_tokens": budget, "runs": runs},
                  f, indent=1)


if __name__ == "__main__":
    main()
