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
