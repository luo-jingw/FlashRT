"""Linear-attention prefill variants of the Qwen3.8-27B NVFP4 planner, layer by layer on one monitor prompt (Thor row
H9).

The Thor frontend routes every prefill chunk of more than 7 rows to `Qwen36TorchFrontendThor._thor_lin_K_forward`
(split kernels and the `native` per-step GDN chunk scan); the RTX frontend uses its parent K-row path with the WY
chunk backend (`wy_lt`). This probe separates the two differences:

1. Capture: one eager long-context prefill of the first monitor call of ep0 (wrist 224) on the frontend's own path;
   for every linear-attention layer and every prefill chunk (2 048 + 2 031 rows), the chunk's input hidden rows and
   the layer's recurrent and conv states before that chunk are saved.
2. Per layer and chunk, every variant runs on the same saved input and states; its output rows and the recurrent
   state after the chunk are compared with the reference variant `per_token` (cos, max abs difference, relative L2),
   and its time is taken (CUDA-synchronized, after one warm-up call per variant).
3. Whole prompt, per variant: the full eager prefill with every linear-attention chunk of more than 7 rows routed to
   the variant; last-position logits against `per_token` (cos, top-1 equal) and prefill ms.

Variants (the RTX parent path reads its switches from the environment at call time):
  per_token      : the S = 1 decode layer (`_layer_forward_lin_nvfp4`) row by row: the per-token recurrent path the
                   Thor frontend's docstring measures drift against (reference)
  own            : the frontend's own dispatch with the environment it was loaded with (Thor: thor_native; RTX:
                   wy_lt)
  thor_native    : Qwen36TorchFrontendThor._thor_lin_K_forward
  rtx_native     : parent K-row path, FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=native (the parent's fused kernels with
                   the native GDN chunk scan: isolates the fused-kernel rounding from the WY decomposition)
  wy             : parent, backend wy
  wy_lt          : parent, backend wy_lt (the RTX default)
  wy_lt_f32gemm  : parent, wy_lt with FLASHRT_QWEN36_TQ_PREFILL_GDN_FAST_CHUNK=f32 (FP32 GEMM chunk_h)
  wy_lt_f32state : parent, wy_lt with FLASHRT_QWEN36_TQ_PREFILL_GDN_FAST_CHUNK=direct (FP32 state chunk_h)

usage (cwd anywhere; PLANNER_CKPT, PLANNER_MTP_DIR as for ../monitor_call/monitor_call_bench.py):
  PLANNER_CKPT=DIR PLANNER_MTP_DIR=DIR python gdn_variant_drift.py {rtx|thor} OUT_JSON
"""
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "monitor_call"))

from monitor_call_bench import LOOKAHEAD_ROWS, REPLAY_DIR, SPEC_K, MonitorBench  # noqa: E402

os.environ["FLASHRT_QWEN36_TQ_PREFILL_GRAPH"] = "0"  # eager prefill: the layer calls are visible to the capture

import torch  # noqa: E402

from flash_rt import flash_rt_kernels as fvk  # noqa: E402
from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_thor import Qwen36TorchFrontendThor  # noqa: E402

CALLS = os.path.join(REPLAY_DIR, "calls", "ep0_wrist224.jsonl")
BACKEND_ENV = "FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND"
FAST_CHUNK_ENV = "FLASHRT_QWEN36_TQ_PREFILL_GDN_FAST_CHUNK"
PARENT_VARIANTS = {
    "rtx_native": {BACKEND_ENV: "native"},
    "wy": {BACKEND_ENV: "wy"},
    "wy_lt": {BACKEND_ENV: "wy_lt"},
    "wy_lt_f32gemm": {BACKEND_ENV: "wy_lt", FAST_CHUNK_ENV: "f32"},
    "wy_lt_f32state": {BACKEND_ENV: "wy_lt", FAST_CHUNK_ENV: "direct"},
}
WY_SYMBOLS = ("qwen36_gdn_wy_kkt_b64_bf16", "qwen36_gdn_wy_solve_tril_b64_f32", "qwen36_gdn_wy_recompute_wu_b64_bf16",
              "qwen36_gdn_wy_chunk_h_b64_bf16", "qwen36_gdn_wy_output_o_b64_bf16",
              "linear_attn_gdn_wy_kkt_b64_bf16_cublaslt", "linear_attn_gdn_wy_chunk_h_b64_bf16_mma_fla",
              "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32gemm",
              "linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_f32state")


@dataclass
class Captured:
    layer: int
    chunk: int
    rank: int
    rows: int
    h_in: torch.Tensor
    lin_state: torch.Tensor
    conv_state: torch.Tensor


@dataclass
class LayerRow:
    variant: str
    layer: int
    chunk: int
    out_cos: float
    out_max_abs: float
    out_rel_l2: float
    state_cos: float
    ms: float


@dataclass
class PromptRow:
    variant: str
    prefill_ms: float
    logits_cos: float
    top1: int
    top1_equal: bool


def compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    x, y = a.float().flatten(), b.float().flatten()
    cos = float(torch.nn.functional.cosine_similarity(x, y, dim=0))
    return cos, float((x - y).abs().max()), float((x - y).norm() / y.norm().clamp_min(1e-30))


class VariantProbe:
    def __init__(self, bench: MonitorBench) -> None:
        self.b = bench
        self.llm = bench.llm
        self.thor_ok = True
        if not isinstance(self.llm, Qwen36TorchFrontendThor):
            try:
                Qwen36TorchFrontendThor._thor_alloc_K_row_scratch(self.llm)
            except Exception as e:  # recorded, the variant is skipped
                print(f"thor_native unavailable on this frontend: {type(e).__name__}: {e}", flush=True)
                self.thor_ok = False
        self.env0 = {k: os.environ.get(k) for k in (BACKEND_ENV, FAST_CHUNK_ENV)}
        own = self.llm._layer_forward_lin_K_nvfp4
        self.variants: dict[str, Callable] = {"per_token": self.per_token, "own": lambda L, h, K: own(L, h, K)}
        if self.thor_ok:
            self.variants["thor_native"] = lambda L, h, K: Qwen36TorchFrontendThor._thor_lin_K_forward(self.llm, L, h, K)
        for name in PARENT_VARIANTS:
            self.variants[name] = lambda L, h, K: Qwen36TorchFrontendRtx._layer_forward_lin_K_nvfp4(self.llm, L, h, K)
        self.reference = "per_token"

    def per_token(self, L: int, h_in_K: torch.Tensor, K: int) -> torch.Tensor:
        hidden = int(h_in_K.shape[-1])
        out = torch.empty_like(h_in_K)
        for r in range(K):
            h_r = h_in_K[:, r:r + 1, :].reshape(1, hidden).contiguous()
            out[:, r:r + 1, :].copy_(Qwen36TorchFrontendRtx._layer_forward_lin_nvfp4(self.llm, L, h_r).view(1, 1, hidden))
        return out

    def set_env(self, name: str) -> None:
        for k in (BACKEND_ENV, FAST_CHUNK_ENV):
            os.environ.pop(k, None)
        if name in PARENT_VARIANTS:
            os.environ.update(PARENT_VARIANTS[name])
        else:
            os.environ.update({k: v for k, v in self.env0.items() if v is not None})

    def lm_ids(self) -> torch.Tensor:
        row = next(json.loads(line) for line in open(CALLS) if json.loads(line)["kind"] == "monitor")
        rows = [json.loads(line) for line in open(CALLS)]
        img_sets = [{p["image"] for p in r["user_parts"] if "image" in p} for r in rows if r["kind"] == "monitor"]
        pc = self.b.prepare(row, REPLAY_DIR, sorted(set.intersection(*img_sets)))
        feats = self.b.encode(pc, 0)
        return self.b.inject(pc, feats, 0, int(pc.input_ids.shape[0]) + 1 + LOOKAHEAD_ROWS)

    @torch.inference_mode()
    def capture(self, ids: torch.Tensor) -> list[Captured]:
        orig = self.llm._layer_forward_lin_K_nvfp4
        seen: list[Captured] = []
        chunks: dict[int, int] = {}

        def hook(L: int, h_in_K: torch.Tensor, K: int):
            if K > 7:
                r = self.llm._linear_layer_rank(L)
                seen.append(Captured(L, chunks.get(L, 0), r, K, h_in_K.clone(), self.llm._lin_state[r].clone(),
                                     self.llm._lin_conv_state[r].clone()))
                chunks[L] = chunks.get(L, 0) + 1
            return orig(L, h_in_K, K)

        self.llm._layer_forward_lin_K_nvfp4 = hook
        try:
            self.llm.prefill_long_ctx_nvfp4_agent(ids, max_new_tokens=1, K=SPEC_K)
        finally:
            del self.llm._layer_forward_lin_K_nvfp4
        torch.cuda.synchronize()
        return sorted(seen, key=lambda c: (c.chunk, c.layer))

    def run_layer(self, name: str, c: Captured) -> tuple[torch.Tensor, torch.Tensor, float]:
        self.set_env(name)
        self.llm._lin_state[c.rank].copy_(c.lin_state)
        self.llm._lin_conv_state[c.rank].copy_(c.conv_state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.variants[name](c.layer, c.h_in.clone(), c.rows)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        return out.clone(), self.llm._lin_state[c.rank].clone(), ms

    @torch.inference_mode()
    def layers(self, caps: list[Captured]) -> list[LayerRow]:
        for name in self.variants:          # warm-up: scratch allocation, cuBLASLt algorithm selection
            self.run_layer(name, caps[0])
        rows: list[LayerRow] = []
        for c in caps:
            ref_out, ref_state, ref_ms = self.run_layer(self.reference, c)
            rows.append(LayerRow(self.reference, c.layer, c.chunk, 1.0, 0.0, 0.0, 1.0, ref_ms))
            for name in self.variants:
                if name == self.reference:
                    continue
                out, state, ms = self.run_layer(name, c)
                cos, mx, rel = compare(out, ref_out)
                scos, _, _ = compare(state, ref_state)
                rows.append(LayerRow(name, c.layer, c.chunk, cos, mx, rel, scos, ms))
        return rows

    @torch.inference_mode()
    def prompt(self, ids: torch.Tensor) -> list[PromptRow]:
        orig = self.llm._layer_forward_lin_K_nvfp4
        results: dict[str, tuple[torch.Tensor, float]] = {}
        for name in self.variants:
            self.set_env(name)
            fn = self.variants[name]
            self.llm._layer_forward_lin_K_nvfp4 = (lambda L, h, K, fn=fn: fn(L, h, K) if K > 7 else orig(L, h, K))
            try:
                times = []
                for _ in range(1 if name == "per_token" else 2):          # first: warm-up
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    self.llm.prefill_long_ctx_nvfp4_agent(ids, max_new_tokens=1, K=SPEC_K)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1e3)
                results[name] = (self.llm._logits_buf[0].float().clone(), times[-1])
            finally:
                del self.llm._layer_forward_lin_K_nvfp4
        ref = results[self.reference][0]
        out = []
        for name, (lg, ms) in results.items():
            cos, _, _ = compare(lg, ref)
            out.append(PromptRow(name, ms, cos, int(lg.argmax()), int(lg.argmax()) == int(ref.argmax())))
        return out


def main() -> None:
    frontend, out_path = sys.argv[1], sys.argv[2]
    bench = MonitorBench(frontend)
    print("loaded:", type(bench.fe).__name__, type(bench.llm).__name__, "WY symbols:",
          {s: hasattr(fvk, s) for s in WY_SYMBOLS}, flush=True)
    probe = VariantProbe(bench)
    ids = probe.lm_ids()
    probe.set_env(probe.reference)
    caps = probe.capture(ids)
    print(f"captured {len(caps)} linear-attention layers, first chunk {caps[0].rows} rows, reference "
          f"{probe.reference}", flush=True)
    layer_rows = probe.layers(caps)
    prompt_rows = probe.prompt(ids)
    bench.llm.reset_rope_rows()
    for name in probe.variants:
        for ch in sorted({r.chunk for r in layer_rows}):
            rs = [r for r in layer_rows if r.variant == name and r.chunk == ch]
            worst = min(rs, key=lambda r: r.out_cos)
            print(f"{name:15s} chunk {ch}: layer out cos median {statistics.median(r.out_cos for r in rs):.6f} min "
                  f"{worst.out_cos:.6f} (layer {worst.layer}), state cos min {min(r.state_cos for r in rs):.6f}, "
                  f"ms/layer median {statistics.median(r.ms for r in rs):.2f}", flush=True)
    for p in prompt_rows:
        print(f"{p.variant:15s} prompt: prefill {p.prefill_ms:.1f} ms, logits cos {p.logits_cos:.6f}, top-1 {p.top1} "
              f"equal {p.top1_equal}", flush=True)
    with open(out_path, "w") as f:
        json.dump({"frontend": frontend, "frontend_class": type(bench.llm).__name__, "reference": probe.reference,
                   "wy_symbols": {s: hasattr(fvk, s) for s in WY_SYMBOLS},
                   "layers": [asdict(r) for r in layer_rows], "prompt": [asdict(p) for p in prompt_rows]}, f, indent=1)


if __name__ == "__main__":
    main()
