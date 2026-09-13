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

Status: RESOLVED (fixed and verified on Ada; Thor re-measurement still pending)

Area: ImageWAM denoise step — mot_joint attention computed ~15x more than needed

## Observation (original)

Real Thor (SM110) measurement (plan.md "Real Thor Results"): FP4/FP8
GEMM quantization speeds up backbone prefill by 1.3-1.5x but leaves the
ActionDiT denoise step essentially unchanged across all four precisions
tested (~35ms regardless of FP16/BF16/FP8/FP4). Root cause was
`attn.run("mot", ...)`'s own design (Phase 4's Structures section):
every denoise step computed joint attention over the WHOLE `total`
sequence (`total*NH` query rows), even though only the `num_action`
action rows' output is ever read afterward — at this plan's own
benchmark dims (`total=960, num_action=64`), that is roughly 15x more
query rows computed than needed.

## Fix (implemented and verified)

New kernel `attention_qkv_fp16_mot_joint_action`
(`csrc/kernels/attention_cublas.cu`/`.cuh`) + softmax variant
`softmax_mot_joint_action_fp16` (`csrc/kernels/softmax.cu`/`.cuh`):
same cuBLAS-composed QK^T → masked-softmax → PV structure as the
original `mot_joint` kernel, but Q covers only `num_action*NH` rows
(not `total*NH`) — K/V still cover the whole `total` sequence, since
action rows attend into the frozen prefix K/V. Collapsing Q to action
rows only also collapses the softmax mask from three row-groups down
to ONE uniform rule for every row (`[0,x0) U [a0,total)`), since every
remaining row is an action row.

`ImageWAMAttnBackend.run()`'s `"mot_joint"` branch now requires
`q_seq=num_action` and an explicit `kv_seq=total` (previously
`kv_seq` defaulted from `q_seq=total`, i.e. they were the same value);
it computes the Q/output pointer offset (`Q_O + a0*row_width`)
internally from `a0`, which every caller already supplies for the
mask, so the pipeline call sites only needed their `q_seq`/`kv_seq`
arguments updated, not their own pointer arithmetic (they already
compute and read from `action_Q_ptr` at that same offset both before
and after the call). Updated: `pipeline_thor.py`'s
`_action_double_layer`/`_action_single_layer`, and all four
`benchmarks/imagewam_thor_*_bench.py` scripts.

**Correctness verified two ways** (`tests/test_imagewam_mot_joint_action_kernel.py`):
against a PyTorch reference (`cosine=1.000000`), and — the contract
that actually matters — bit-for-bit equivalence with what the ORIGINAL
`mot_joint` kernel produces for the same action rows when run over the
whole sequence (`cosine=1.000000` there too): this is a pure speed
optimization with zero behavior change for the rows that matter.
Existing tests (`test_imagewam_denoise.py`, `test_imagewam_frontend.py`,
`test_imagewam_attn_backend.py` — the last one updated for the new
`q_seq`/`kv_seq`/output-offset contract) all still pass.

## Real Measured Speedup (Ada, this machine)

| | one denoise step (25L) | prefill + 10-step |
|---|---|---|
| FP16, before fix | 29.5 ms | 447.8 ms |
| FP16, after fix | **5.68 ms (5.2x)** | **203.2 ms (2.2x)** |
| INT4 (GEMM-only), before fix | 24.2 ms | 283.3 ms |
| INT4 (GEMM-only), after fix | **4.37 ms (5.5x)** | **91.1 ms (3.1x)** |

Matches the ~15x theoretical query-row reduction reasonably well once
GEMM cost (which does not shrink) is accounted for — the denoise step
was almost entirely attention time before the fix, so a ~15x cheaper
attention call yields roughly the 5x step-level speedup measured.

## Required Evidence — still open

This is Ada, not Thor. The real Thor re-measurement (repeating the
"Real Thor Results" table with this fix in place) has not been done —
the FP4/FP8/INT4 benchmark scripts are updated and ready for that, but
require the user's own Thor access to run.

## Promotion Condition

Promote to `docs/` (verified fact) once re-measured on real Thor
hardware and the number holds up there too.

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

# OPT-005

Status: not promoted

Area: Real per-head MHA via FA2/FA4, not a new custom masked cuBLAS kernel

## Observation

OPT-002 proposed writing a new batched/strided cuBLAS masked-attention
kernel to fix the broadcast-K/V simplification. Found while surveying
FlashRT's own existing mechanisms: the vendored FA2 (`flash_rt_fa2.so`,
RTX) and FA4 (`flash_rt/hardware/thor/fa4_backend.py`, Thor sm_110)
flash-attention backends already support real per-head Q/K/V (confirmed
via `ThorFlashAttnBackend`'s own siglip usage:
`q_tensor/k_tensor/v_tensor` all shaped `(nv, q_seq, NH, HD)`, no
broadcast), GQA (`pack_gqa`), and are CUDA-graph-capture-safe.
`fa4_backend.py`'s own docstring records a real measured win at a
comparable VLA denoise shape ("Sq=51, Skv~891, GQA 16/2, HD=128... ~17%
faster than the vendored fmha kernel, cos=1.0").

## Opportunity

For the "backbone" site's plain self-attention (no custom mask needed),
switch from the hand-written `attention_qkv_fp16` cuBLAS-composed kernel
to FA4 (Thor) / FA2 (RTX) directly — likely both faster AND gets real
per-head K/V for free, fixing part of OPT-002 with zero new kernel code.
For the "mot" site's three-region masked joint attention, check whether
FA2/FA4's forward accepts an arbitrary additive attention bias/mask
tensor (not yet confirmed in this survey — would need reading the
vendored FA2/FA4 source directly, out of scope for this pass) before
assuming the custom masked kernel is still required there.

## Expected Mechanism

Same mechanism already measured real: FA2/FA4's own fused, highly
tuned flash-attention kernels replace the cuBLAS QK^T -> softmax -> PV
three-launch composition for sites where the masking need fits their
API.

## Required Evidence

Confirm whether FA2/FA4 support an arbitrary/block mask (not just
causal) before committing scope here for the "mot" site specifically;
the "backbone" site (plain self-attention, no mask) is a much lower-risk
first target regardless.

## Promotion Condition

Promote alongside OPT-002/OPT-004 once Thor performance work begins.

# OPT-006

Status: not promoted

Area: TeaCache-style step-skipping for the flow-matching denoise loop

## Observation

`flash_rt/models/cosmos3_edge/pipeline_thor.py`'s `CosmosEdgeThor` has a
real, already-implemented `set_teacache(compute_steps)` mechanism: a
fixed subset of denoise steps actually compute a fresh velocity, the
rest reuse the last computed velocity while the scheduler still
advances every step. This exploits the same redundancy diffusion/flow-
matching literature calls TeaCache — consecutive denoising steps often
produce very similar velocity predictions.

## Opportunity

ImageWAM's own flow-matching denoise loop (`imagewam_denoise_loop`,
Phase 4) is structurally the same shape (N fixed steps, each a full
ActionDiT forward) — a `set_teacache`-equivalent compute-step schedule
could skip a fraction of ActionDiT forwards entirely, directly reducing
the ~35ms/step cost this project has now measured twice (Ada dry-run
and real Thor hardware).

## Expected Mechanism

Same mechanism already implemented and presumably validated for
cosmos3_edge: skip N-k of N steps' full forward, reuse the last
velocity, accept whatever accuracy cost that implies (needs real-weight
validation, not assessable with random weights).

## Required Evidence

Needs real weights and an accuracy budget to determine a safe
compute-step schedule — meaningless to tune against random weights.
Should follow, not precede, OPT-001.

## Promotion Condition

Promote alongside OPT-001, once real-weight accuracy validation exists
to determine which steps are safe to skip.

# OPT-007

Status: not promoted

Area: INT4 (QuaRot W4A4, SM80 CUTLASS) as an additional precision option — confirmed buildable and runnable on Ada, not just Thor

## Observation

`csrc/gemm/cutlass_sm80_int4_rowwise.cu` — a real INT4 W4A4 rowwise
GEMM family, built for Jetson Orin SM87's QuaRot path but templated on
`cutlass::arch::Sm80` — is gated only by `ENABLE_SM80_INT8_CUTLASS`
(default ON only for `GPU_ARCH=87`, but overridable) and
`FLASHRT_ENABLE_CHAMELEON` (default OFF, opt-in), NOT by any
Blackwell-only check like NVFP4. Confirmed by actually reconfiguring
and rebuilding on this dev machine (`-DENABLE_SM80_INT8_CUTLASS=ON
-DFLASHRT_ENABLE_CHAMELEON=ON`, `GPU_ARCH=89`) and running
`cutlass_int4_rowwise_fp16out` successfully at real ImageWAM projection
shapes (`benchmarks/imagewam_gemm_precision_compare.py`) — see
plan.md's own recorded GEMM-only comparison table for the real numbers.
INT8 (SM80, same family) also works. Both are meaningfully faster than
FP16 at the shapes that work: INT4 ~9x, INT8 ~4x on the `q/proj`
(3072x3072) shape.

**Correction, found while building the full-pipeline benchmark**: the
`mlp2` (`K=9216`) failure reported above for `cutlass_int4_rowwise_fp16out`
does NOT reproduce in isolation or inside a real full-pipeline run
(`benchmarks/imagewam_thor_int4_bench.py`, all 25+25 layers including
20 calls at this exact shape, ran clean end to end). It only reproduced
inside `imagewam_gemm_precision_compare.py`'s specific call sequence
(fp16 → fp8[fails] → int8[fails] → int4, repeated per shape across 4
shapes) — isolating the exact trigger was not pursued further (several
targeted repros ruled out simple explanations: shape order, the
preceding fp8/int8 failures, and the warmup/iteration loop pattern
itself all failed to reproduce it alone). Treat this as a real but
poorly-understood flakiness in `imagewam_gemm_precision_compare.py`'s
own specific mixed-precision-in-one-process pattern, not a hard `K`
limit on the kernel — **`cutlass_int8_rowwise_fp16out` (INT8, not
INT4) does reliably fail at `K=9216`** in every reproduction attempted,
confirmed independent of the flaky int4 behavior; that one appears to
be a real, consistent INT8-specific limitation.

Also real and not yet addressed: this GEMM's own correctness contract
(per its file header) requires a QuaRot Hadamard rotation on both
activation (online FHT) and weight (offline) before quantizing to
int4 — plain per-row symmetric quantization without it is documented
in that same file as insufficient for real model activations.
**Confirmed blocking, not theoretical**: the real activation quantizer,
`fht_int4_quant_fp16`, CRASHES with an illegal memory access at
ImageWAM's real hidden dims (3072, 9216, 7680 — none are powers of 2),
while working cleanly at 128/1024/4096 (all powers of 2). This FHT
kernel needs a power-of-2 transform size; ImageWAM's real dims are not
powers of 2. `benchmarks/imagewam_thor_int4_bench.py`'s own full-
pipeline number (below) is GEMM-only for exactly this reason — it
could not include a real per-call activation quantization step even if
it wanted to.

## Full-Pipeline Result (Ada, GEMM-only, no activation quantization)

`benchmarks/imagewam_thor_int4_bench.py`: same 25+25-layer structure as
the FP16/FP8/FP4 scripts, ran clean end to end on this machine:

| | backbone prefill (25L) | one denoise step (25L) | prefill + 10-step |
|---|---|---|---|
| INT4 (GEMM-only) | 42.4 ms | 24.2 ms | 283.3 ms |
| FP16 (this machine) | 152.1 ms | 29.5 ms | 447.8 ms |

3.6x faster prefill, only ~1.2x faster denoise step (consistent with
OPT-003's diagnosis: denoise is attention-bound, not GEMM-bound, so
GEMM quantization alone caps out around the same ceiling FP8/FP4 also
hit). This is an optimistic upper bound, not a real deployment number —
see the activation-quantization caveat above.

## Opportunity

A fourth precision tier alongside FP16/FP8/FP4, IF the FHT power-of-2
requirement is resolved (either generalize the kernel, or pad
ImageWAM's activations to the next power of 2 before quantizing — 4096
for hidden=3072, 16384 for mlp_hidden=9216, 8192 for
joint_attention_dim=7680, changing GEMM shapes throughout) AND the
QuaRot rotation is validated for ImageWAM's own activation
distributions (a real correctness project, not yet started — this
would need real weights to even evaluate, same dependency as OPT-001).

## Expected Mechanism

Same mechanism the Chameleon-7B path already uses in production
(assumed, not independently re-verified here): FHT-rotated activations
+ offline-rotated weights survive int4's dynamic range at measured
cosine 0.9914 (per the kernel file's own header comment, for
Chameleon's own model and data — not yet re-measured for ImageWAM).

## Required Evidence

Resolve the FHT power-of-2 blocker above before treating INT4 as
viable for ImageWAM at all (not just the down-projection — every real
hidden dimension in this model is a non-power-of-2 multiple of 1024).
Real-weight accuracy work (OPT-001) is a hard prerequisite for any
correctness claim regardless.

## Promotion Condition

Promote only after both the FHT power-of-2 blocker and the QuaRot
rotation's real-weight accuracy validation have real answers — this is
presently a confirmed-fast, confirmed-buildable option with a real
correctness blocker, not a working precision tier.
