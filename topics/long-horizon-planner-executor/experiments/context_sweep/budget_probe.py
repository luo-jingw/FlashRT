"""Decode time of the first tokens against the declared token budget (max_new_tokens), at a fixed prompt.

For each budget the prompt is prefilled with that budget and the first FIRST tokens are decoded and timed (twice; the
second run is reported). If the per-token time changes with the budget, the budget, not the reply length, sets the
decode cost.

usage: PLANNER_CKPT=DIR PLANNER_MTP_DIR=DIR python budget_probe.py {rtx|thor} OUT_JSON CONTEXT_TOKENS BUDGET [BUDGET ...]
"""
import json
import sys
import time

from context_sweep import Sweep, SPEC_K  # noqa: E402  (sets the FlashRT environment variables)
import torch  # noqa: E402

FIRST = 256


@torch.inference_mode()
def first_tokens_ms(sw: Sweep, ids: torch.Tensor, budget: int) -> tuple[float, int, int]:
    sw.llm._spec_attempts = 0
    sw.llm.prefill_long_ctx_nvfp4_agent(ids, max_new_tokens=budget, K=SPEC_K)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for chunk in sw.llm.decode_long_ctx_nvfp4_committed_stream(max_new_tokens=budget, K=SPEC_K):
        n += len(chunk)
        if n >= FIRST:
            break
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, n, int(getattr(sw.llm, "_spec_attempts", 0))


def main() -> None:
    frontend, out, ctx = sys.argv[1], sys.argv[2], int(sys.argv[3])
    budgets = [int(x) for x in sys.argv[4:]]
    sw = Sweep(frontend, ctx + max(budgets))
    ids = sw.prompt_ids(ctx)
    rows = []
    for b in budgets:
        first_tokens_ms(sw, ids, b)
        ms, n, att = first_tokens_ms(sw, ids, b)
        k = sw.llm._long_tq_effective_k(int(ids.shape[1]), SPEC_K, b)
        rows.append({"budget": b, "tokens": n, "ms": ms, "ms_per_token": ms / n, "attempts": att, "effective_k": k})
        print(f"budget {b:>5}: first {n} tokens {ms / n:6.1f} ms/token, {n / max(1, att):.2f} tokens/attempt, K {k}",
              flush=True)
    with open(out, "w") as f:
        json.dump({"frontend": frontend, "context_tokens": int(ids.shape[1]), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
