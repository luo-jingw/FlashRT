# OPT-001

Status: not promoted

Area: ImageWAM on Thor — precision and real-weight path

## Observation

`plan.md`'s current plan uses randomly initialized weights and a
BF16-only forward, deferring FP8 quantization, calibration, real
checkpoint loading, and accuracy validation. ImageWAM's FLUX.2-4B
variant needs roughly 18GB of weights in bf16 (backbone + Qwen3-4B
text encoder + VAE) before activation memory.

## Opportunity

Once the plan's phases produce a working structural forward pass,
extend it to: load the real released checkpoint
(`yuyangalin/ImageWAM-FLUX.2-4B-LIBERO`), add FP8 quantization and
calibration following FlashRT's existing house calibration mechanism
(`docs/calibration.md`), and validate cosine similarity against the
PyTorch reference implementation.

## Expected Mechanism

Same mechanism already used by every other FlashRT Thor model: static
per-tensor/rowwise scales from a calibration forward pass, FP8 GEMMs
in the production forward, a BF16 calibration twin retained for
recalibration.

## Required Evidence

The structural plan (`plan.md`) must complete first: a real weight
loader is meaningless without the pipeline forward and CUDA Graph
capture already working, and calibration is meaningless without real
weights to calibrate.

## Promotion Condition

Promote to a plan once real ImageWAM weights are available on the
target machine and the current structural plan's phases are complete.

# OPT-002

Status: not promoted

Area: ImageWAM backbone/action-expert attention — real per-head K/V

## Observation

The Thor kernels this plan's backbone forward calls through
`ImageWAMAttnBackend` (`attention_qkv_fp16`, `attention_qkv_fp16_mot_joint`)
both take K/V as a single `(seq, HD)` buffer broadcast across all `NH`
query heads — confirmed by reading `csrc/kernels/attention_cublas.cu`
directly (plan.md Phase 2). Real FLUX.2/DiT attention uses full
per-head K/V (`num_kv_heads == num_q_heads`, each head with its own
K/V, not shared). This plan's own K/V projection weights are declared
at `HD` width (128) rather than `hidden` width (3072) as a direct
consequence — a genuine architectural simplification of ImageWAM's
real attention mechanism, separate from (and in addition to) the
random-vs-real-weight difference tracked in OPT-001.

## Opportunity

Write a real per-head-K/V masked attention kernel (K/V shaped
`(seq, NH, HD)` like Q, not `(seq, HD)`) for both the plain self-attention
site ("backbone") and the three-region masked joint site ("mot"), and
switch `_imagewam_thor_spec.py`'s K/V projection shapes back to full
`hidden` width to match the real checkpoint's fused QKV tensor exactly.

## Expected Mechanism

Same cuBLAS-composed pattern already used (QK^T GEMM -> fused masked
softmax -> PV GEMM), extended to batch over `NH` independent K/V sets
instead of broadcasting one shared set — likely a batched/strided
cuBLAS GEMM (`cublasGemmStridedBatchedEx`) rather than a single big GEMM,
since each head now has its own K/V.

## Required Evidence

Only matters once real checkpoint weights are being loaded (OPT-001)
— a random-weight structural dry run does not need real per-head
fidelity to test wiring, pointer contracts, or shapes.

## Promotion Condition

Promote alongside OPT-001, when real-weight accuracy validation
begins and a broadcast-K/V approximation is shown to diverge from the
reference implementation.

# OPT-003

Status: not promoted

Area: ImageWAM denoise step — mot_joint attention computes ~15x more than needed

## Observation

Real Thor (SM110) measurement (plan.md "Real Thor Results"): FP4/FP8
GEMM quantization speeds up backbone prefill by 1.3-1.5x but leaves the
ActionDiT denoise step essentially unchanged across all four precisions
tested (~35ms regardless of FP16/BF16/FP8/FP4). Root cause is
`attn.run("mot", ...)`'s own documented design (Phase 4's Structures
section): every denoise step computes joint attention over the WHOLE
`total` sequence (`total*NH` query rows), even though only the
`num_action` action rows' output is ever read afterward — at this
plan's own benchmark dims (`total=960, num_action=64`), that is
roughly 15x more query rows computed than needed. GEMM quantization
cannot help a step that is attention-bound by this overcompute.

## Opportunity

Either (a) slice `attn.run`'s Q input to only the action rows before
calling `attention_qkv_fp16_mot_joint` (requires confirming the kernel
accepts a Q row count different from the K/V row count — currently it
assumes `S == S_kv == total`, so this needs a kernel signature change,
not just a call-site change), or (b) accept the overcompute but batch
multiple denoise steps' worth of action-row attention into fewer,
larger kernel launches if that changes the profile favorably. Needs
real profiling on Thor to know which (or something else) actually
moves the number — this is a hypothesis grounded in a real
measurement, not yet a verified fix.

## Expected Mechanism

Reducing the mot_joint kernel's own Q row count from `total*NH` to
`num_action*NH` should reduce both the QK^T and PV cuBLAS GEMM cost
roughly proportionally (~15x fewer rows in this plan's own benchmark
dims), and shrink the softmax kernel's own row count analogously.

## Required Evidence

Profile the denoise step on Thor with `nsight`/`nvprof`-equivalent
tooling to confirm `mot_joint`'s own kernels (not something else) are
actually the dominant cost before changing the kernel signature —
Phase 4's plan.md documentation predicted this cost but this is the
first real measurement showing it does not respond to GEMM
quantization, which is consistent with (but does not by itself prove)
the attention-bound hypothesis.

## Promotion Condition

Promote once Thor profiling confirms mot_joint is the dominant
denoise-step cost and a specific kernel-signature change is proposed
and reviewed.

# OPT-004

Status: not promoted

Area: ImageWAM pipeline has none of FlashRT's real kernel-fusion or GEMM-autotuning machinery

## Observation

Real Thor (SM110) measurement (plan.md "Real Thor Results"): full
prefill+10-step-denoise steady-state is 407-453ms across all four
precisions tested — this is the first real-hardware confirmation that
this plan's own pipeline (Phase 3/4) is a direct, unfused 1:1
translation of the math (one kernel launch per op: norm, then each
GEMM separately, then attention, then residual_add, then norm, then
GEMM, then gelu, then GEMM, then residual_add), unlike real FlashRT
models such as `cosmos3_edge` which fuse aggressively
(`residual_add_rms_norm_fp8`, fused QKV projections, `bias_gate_mul_residual_bf16`)
and autotune `GemmRunner` shapes via `autotune_cached` rather than
accepting cuBLASLt's default top-1 heuristic. The benchmark that
produced this number is also graph-free (no `torch.cuda.graph(...)`
capture), unlike Phase 5's own `ImageWAMTorchFrontendThor`, which does
capture a graph — the graph-captured, whole-pipeline number on Thor is
not yet measured.

## Opportunity

In priority order (highest expected win first, per this project's own
established fusion precedent in `cosmos3_edge`/`pi05`):
1. Measure the graph-captured (not graph-free) whole-pipeline number on
   Thor via `ImageWAMTorchFrontendThor` itself, to isolate how much of
   the current number is Python/launch overhead vs. real GPU work.
2. Fuse QKV into one wide GEMM per stream (matches the real checkpoint's
   own fused tensor shape from Phase 1 — currently split into 3 GEMMs
   only because of this pipeline's own reduced-KV-width convention,
   OPT-002).
3. Fuse residual+norm (`residual_add_rms_norm_fp8`-style) at every
   block boundary.
4. Call `GemmRunner.autotune_cached` for ImageWAM's own real shapes
   instead of relying on cuBLASLt's default heuristic.

## Expected Mechanism

Same mechanism already proven in `cosmos3_edge`/`pi05`: fewer kernel
launches per layer (less Python/pybind11 dispatch + host-device
round-trip overhead) and CUDA Graph capture (near-zero per-call CPU
overhead on replay).

## Required Evidence

Step 1 above (measure the graph-captured number) should happen first —
it is cheap (no new kernel work) and tells us how much of the 407-453ms
is even reachable by kernel-level optimization versus launch overhead
already eliminated by graph capture.

## Promotion Condition

Promote once real checkpoint accuracy work (OPT-001) is underway and
Thor performance is an active concern, not before — this plan's own
stated goal (structural wiring, deferring precision) did not commit to
performance work, and premature fusion work risks needing to be redone
once real per-head K/V attention (OPT-002) changes the kernel shapes
it would fuse around.
