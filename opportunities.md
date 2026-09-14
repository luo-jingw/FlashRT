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

Status: kernel-level RESOLVED (real per-head K/V, RoPE, QK-Norm, real
mask all implemented and verified, including combined); NOT YET wired
into `pipeline_thor.py`/`_imagewam_thor_spec.py`, and NOT YET validated
against a real checkpoint (no checkpoint file available locally)

Area: ImageWAM backbone/action-expert attention — real per-head K/V,
and (found while starting this work) real RoPE, real QK-Norm, and a
real attention mask, none of which this project had ever modeled

## Observation (original)

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

## Scope correction: three MORE real gaps found while starting this work

Fetched and read the real `black-forest-labs/flux2` source at the
exact commit ImageWAM's own `docs/dependencies.md` pins
(`50fe5162777813d869182b139e83b10743caef15`) to understand the real
checkpoint's tensor layout for accuracy validation. This confirmed the
project's real dims exactly (`Klein4BParams`: hidden_size=3072,
num_heads=24, depth=5, depth_single_blocks=20, context_in_dim=7680,
in_channels=128 — the last one also fixed a wrong VAE token-width
guess, see the VAE section above) but also surfaced three more real
math gaps this project had never modeled, beyond just per-head K/V:

1. **RoPE**: real FLUX.2 applies a 4-axis rotary position embedding to
   Q/K before attention (`axes_dim=[32,32,32,32]` summing to
   `head_dim=128` — axis 0 = constant "time" value, axis 1 = image row,
   axis 2 = image column, axis 3 = a running index used by text only).
   No kernel in this project applied any RoPE at all before this.
2. **QK-Norm**: real FLUX.2 applies an independent, learned per-head
   RMSNorm to Q and K before RoPE (`QKNorm`/`RMSNorm` in the real
   source). Not modeled anywhere in this project before this.
3. **Real attention mask**: real `causal_attn_fn` is NOT plain unmasked
   self-attention (this project's own "backbone" naming assumption) —
   text+image attend to everything, but reference-image tokens attend
   ONLY to themselves. For ImageWAM's real `infer_action_flux2` action-
   inference path specifically, the target-image group is always empty
   (0 tokens), reducing this to two groups: txt (sees everything) and
   ref-image / the camera observation (sees only itself, never text).

## Implemented and verified (kernel level + combined)

- `attention_qkv_fp16_perhead` / `attention_qkv_fp16_mot_joint_action_perhead`
  (`csrc/kernels/attention_cublas.cu/.cuh`): real per-head K/V via
  `cublasGemmStridedBatchedEx`, for "backbone" (plain) and "mot" (the
  existing 3-region mask) respectively. Wired into `ImageWAMAttnBackend`
  via an opt-in `use_perhead_kv` flag (default off, mirrors `use_fa4`'s
  pattern). Verified at the kernel level (small shape + real dims,
  cosine=1.0 vs a PyTorch reference, and vs the existing broadcast
  kernel with shared-across-heads K/V) and at the backend/dispatch
  level (`tests/test_imagewam_perhead_attention_kernel.py`,
  `tests/test_imagewam_attn_backend.py`).
- `rope_apply_fp16_perhead` (`csrc/kernels/rope.cu/.cuh`) +
  `flash_rt/models/imagewam/rope.py` (Python precompute of the real
  4-axis position table). Verified bit-exact against the real,
  unmodified upstream file at the pinned commit (not committed to this
  repo), and against an independent from-scratch transcription of the
  same real formula in `tests/test_imagewam_rope_kernel.py` (small
  shape + real dims, cosine=1.0).
- QK-Norm needs **no new kernel** — `rms_norm_fp16` already computes
  the exact real formula per (token, head) row, confirmed in
  `tests/test_imagewam_qknorm_reuse.py` (small shape, real dims, and an
  in-place-safety check, all cosine=1.0).
- `attention_qkv_fp16_backbone_ref_masked_perhead`
  (`csrc/kernels/attention_cublas.cu/.cuh` +
  `softmax_backbone_ref_masked_fp16` in `softmax.cu/.cuh`): the real
  2-group mask, built real-per-head from the start. Verified against an
  independent PyTorch reference (small shape + real dims, cosine=1.0)
  AND a direct behavioral perturbation check (a ref row's output is
  provably unchanged when only txt-side K/V move, a txt row's output
  does change when ref-side K/V move) in
  `tests/test_imagewam_backbone_ref_masked_kernel.py`.
- `flash_rt/models/imagewam/real_backbone_attn.py`: chains all three
  (QK-Norm -> RoPE -> masked attention, the REAL order, confirmed from
  `DoubleStreamBlock`/`SingleStreamBlock` directly) into one real
  backbone-attention call. Verified against an independent PyTorch
  reference composing the same three real formulas in the same real
  order, at both a small shape and real dims (cosine=1.0 both) in
  `tests/test_imagewam_real_backbone_attention.py` — this catches
  ordering/interface bugs a per-piece test can't see.

## Still open

- **NOT wired into `pipeline_thor.py`/`_imagewam_thor_spec.py`** — all
  of the above lives in new, additive modules and tests; the actual
  pipeline still uses the old broadcast-K/V, no-RoPE, no-QK-Norm,
  no-mask "standard" kernel path by default. Wiring this in means
  changing `_imagewam_thor_spec.py`'s K/V projection width back to full
  `hidden` (matching the real checkpoint's fused QKV tensor) and adding
  QKNorm/RoPE weight+buffer plumbing to `pipeline_thor.py` itself.
- **Still missing, not investigated**: AdaLN modulation (the real
  time-embedding-driven shift/scale/gate mechanism), LayerNorm (the
  real blocks use `elementwise_affine=False` LayerNorm, not RMSNorm,
  for the main residual-stream normalization — separate from QKNorm),
  and the exact MLP/residual structure. `real_backbone_attn.py`'s own
  docstring flags these explicitly as out of scope for the attention-
  only combination done so far.
- **No real checkpoint file available locally** — everything above is
  verified against PyTorch references of the real published formulas,
  not against real trained weights. Real end-to-end accuracy validation
  (the original goal that surfaced all of this) still needs a real
  checkpoint, which only exists on Thor per the user's own statement.
- "mot" site's own real mask/RoPE/QKNorm requirements not yet
  investigated — this work so far only covered "backbone"; the
  ActionDiT expert and the joint "mot" attention likely have their own
  analogous (possibly different) real requirements, unchecked.

## Promotion Condition

Kernel-level math promoted (verified real, not a hypothesis). Full
promotion (default-on in the real pipeline) blocked on the "still
open" items above, especially AdaLN modulation/LayerNorm (needed for
ANY real block-level forward, not just attention) and real checkpoint
access.

# OPT-003

Status: RESOLVED (fixed and verified on both Ada and real Thor hardware)

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

## Real Measured Speedup (Ada AND real Thor hardware)

| | one denoise step (25L) | prefill + 10-step |
|---|---|---|
| Ada FP16, before fix | 29.5 ms | 447.8 ms |
| Ada FP16, after fix | **5.68 ms (5.2x)** | **203.2 ms (2.2x)** |
| Ada INT4 (GEMM-only), before fix | 24.2 ms | 283.3 ms |
| Ada INT4 (GEMM-only), after fix | **4.37 ms (5.5x)** | **91.1 ms (3.1x)** |
| Thor FP16, before fix | 35.3 ms | ~434 ms |
| Thor FP16, after fix | **5.89 ms (6.0x)** | **140.4 ms (3.1x)** |

Matches the ~15x theoretical query-row reduction reasonably well once
GEMM cost (which does not shrink) is accounted for — the denoise step
was almost entirely attention time before the fix, so a ~15x cheaper
attention call yields roughly a 5-6x step-level speedup on both GPUs.
Real, practically important consequence confirmed on Thor: prefill is
now 58% of the full path (was the minority share before this fix) —
the optimization center of gravity has moved from denoise attention to
backbone GEMM. See plan.md's "Real Thor Results — Post-OPT-003 Run"
for the full breakdown (FP8/FP4/autotune/graph numbers, all re-measured
on Thor with this fix in place).

## Promotion Condition — met

Verified on real Thor hardware with the exact fix in place (not just
re-derived): promoted, this is now a verified fact, not a hypothesis.

# OPT-004

Status: not promoted

Area: ImageWAM pipeline has none of FlashRT's real kernel-fusion or GEMM-autotuning machinery

## Observation

Real Thor (SM110) measurement (plan.md "Real Thor Results"), BEFORE
the OPT-003 fix: full prefill+10-step-denoise steady-state is 407-453ms
across all four precisions tested — the first real-hardware
confirmation that this plan's own pipeline (Phase 3/4) is a direct,
unfused 1:1 translation of the math (one kernel launch per op: norm,
then each GEMM separately, then attention, then residual_add, then
norm, then GEMM, then gelu, then GEMM, then residual_add), unlike real
FlashRT models such as `cosmos3_edge` which fuse aggressively
(`residual_add_rms_norm_fp8`, fused QKV projections, `bias_gate_mul_residual_bf16`)
and autotune `GemmRunner` shapes via `autotune_cached` rather than
accepting cuBLASLt's default top-1 heuristic. (OPT-003's fix since
brought the graph-free Ada number down to 203.2ms — the remaining gap
this entry is about is now smaller than these original Thor numbers
suggest; re-measure on Thor before treating 407-453ms as current.)

## Step 1 done: graph capture measured — launch overhead is NOT the bottleneck here

`benchmarks/imagewam_thor_graph_bench.py` (new): built the real
`ImageWAMTorchFrontendThor` at real dims (post-OPT-003 fix), captured
its CUDA Graph via `set_prompt()`, measured steady-state `infer()`
(prefill + 10-step denoise, replay only) on this machine (Ada):
**198.5ms P50** — versus the graph-FREE number at the identical dims
and post-OPT-003 fix, **203.2ms P50**
(`benchmarks/imagewam_thor_fp16_bench.py`). Only a ~2% difference.

This is a real, somewhat unexpected finding, not the large win
originally hoped for: at these shapes, individual GEMMs are large
enough (hundreds of µs to a few ms each, confirmed in the earlier
GEMM-only comparison table) that per-launch dispatch overhead (typically
single-digit µs) is a small fraction of the total — this pipeline is
solidly **compute-bound, not launch-bound**, at least on Ada. CUDA
Graph capture is still worth keeping (it is real, already built, and
free), but it is not where the remaining ~200ms is going to be found.
This redirects priority toward steps 2-4 below (real compute
reduction: fewer/larger GEMMs, better algorithms), not further
launch-overhead elimination.

**Re-confirmed on real Thor hardware, post-OPT-003**: graph-captured
129.9ms vs graph-free 140.4ms — a **7.5%** gain, bigger than Ada's ~2%
as predicted (faster GEMMs on Thor do make launch overhead a somewhat
larger relative factor), but still the SMALLER of the two available
levers — **autotune alone (126.1ms, no graph capture) already beats
graph-capture-with-default-heuristic (129.9ms)** on Thor. Direction
("compute-bound, algorithm choice matters more than launch count")
holds on both GPUs; the magnitude difference is real and worth keeping
in mind, but doesn't change the priority conclusion.

## Opportunity (steps 2-4, in priority order)

2. Fuse QKV into one wide GEMM per stream (matches the real checkpoint's
   own fused tensor shape from Phase 1 — currently split into 3 GEMMs
   only because of this pipeline's own reduced-KV-width convention,
   OPT-002).
3. Fuse residual+norm (`residual_add_rms_norm_fp8`-style) at every
   block boundary.
4. Call `GemmRunner.autotune_cached` for ImageWAM's own real shapes
   instead of relying on cuBLASLt's default heuristic.

## Step 4 done: autotuning tried — real but modest, not the big lever

`benchmarks/imagewam_thor_fp16_autotuned_bench.py` (new): each
`_Fp16Linear` calls `GemmRunner.autotune_fp16_nn` once (lazily, on
first real call, using the real weight + a representative activation)
before falling back to plain `fp16_nn` for the timed loop —
`autotune_fp16_nn` mutates the same cached cuBLASLt entry `fp16_nn`
itself reads (confirmed by reading `gemm_runner.cu` directly:
`entry.algo = heuristics[best_idx].algo` writes into the identical
`CachedGemm&` both functions share), so this can only match or beat
the default heuristic, never regress correctness.

Result at ImageWAM's real dims, this machine (Ada): backbone prefill
143.5ms → **138.8ms (~3% faster)**, full prefill+10-step 203.2ms →
**195.1ms (~4% faster)**. The autotune log itself explains why the win
is small, not a guess: most shapes only had 1 candidate algorithm
available from `cublasLtMatmulAlgoGetHeuristic` in the first place
(nothing to pick between), and where multiple candidates existed
(4-6), the "best" one was frequently the SAME as the default
heuristic's own top-1 pick, or only a few percent faster.

**Re-confirmed on real Thor hardware, post-OPT-003, and the gain is
bigger there**: full pipeline 140.4ms → **126.1ms (+10%)** — versus
Ada's own +4%. Thor's cuBLASLt evidently has a wider gap between its
default heuristic and the best available algorithm for these shapes
than Ada's does. This is now the single best-verified lever tried so
far: autotune alone beats graph-capture (126.1ms vs 129.9ms) on Thor.

## Expected Mechanism

Steps 2-4 reduce REAL compute/memory-bandwidth cost (fewer, larger,
better-tuned GEMMs), not launch overhead — the right target now that
step 1 showed this pipeline is compute-bound on both Ada and Thor.

## Suggested Next Step — attempted, hung, shelved

Tried combining graph-capture WITH pre-autotuned GEMMs
(`benchmarks/imagewam_thor_fp16_autotuned_graph_bench.py`, on Ada):
warm up on a side stream (to trigger every `_Fp16Linear`'s one-time
`autotune_fp16_nn` call), then capture one more full run into a
`CUDAGraph`. **Result: hung — 66+ minutes of CPU/GPU time with zero
new output, killed rather than let it run further.** Suspected but not
confirmed cause: `autotune_cached`'s own C++ implementation hardcodes
stream 0 for its internal benchmark loop
(`cublasLtMatmul(..., workspace_, workspace_size_, 0)` — the stream
argument passed to `autotune_fp16_nn` itself is not even accepted, let
alone threaded through), while the warmup in this script ran on an
explicit non-default side stream (needed so the same stream could be
used for graph capture) — combining the two may have deadlocked in
`cudaEventSynchronize`/`cudaDeviceSynchronize`. Not root-caused further
User's own call: not worth continuing to debug given autotune-alone
(+4%/+10%) and graph-alone (+2%/+7.5%) are both already confirmed,
independently useful, real wins — the combined win, if the hang were
fixed, would likely be smaller than the sum of both anyway. Shelved,
not attempted again without a specific reason to revisit (e.g., autotuning
on the default stream first, THEN switching to a side stream purely for
capture, never running autotune itself on a non-default stream).

## Promotion Condition

Step 1 (graph-vs-graph-free) and step 4 (autotune) are both now
verified on real Thor hardware — promoted for those two findings.
Steps 2-3 (QKV fusion, residual+norm fusion) remain unpromoted: still
require real checkpoint accuracy work (OPT-001) to be underway and
Thor performance to be an active concern, and their expected value is
now judged lower given the "combine autotune+graph" next step above is
cheaper and already has stronger individual evidence.

# OPT-005

Status: RESOLVED — verified on real Thor hardware (cosine=1.000000, rel_l2=0.000412), real 4.05x standalone speedup on the "backbone" site; NOT YET wired into the real 25-layer prefill benchmarks (still opt-in, off by default everywhere)

Area: FA4 for the "backbone" site's plain self-attention (faster kernel; does NOT independently fix OPT-002's broadcast-K/V, see correction below)

## Observation

OPT-002 proposed writing a new batched/strided cuBLAS masked-attention
kernel to fix the broadcast-K/V simplification. Found while surveying
FlashRT's own existing mechanisms: the vendored FA2 (`flash_rt_fa2.so`,
RTX) and FA4 (`flash_rt/hardware/thor/fa4_backend.py`, Thor sm_110)
flash-attention backends are CUDA-graph-capture-safe and
`fa4_backend.py`'s own docstring records a real measured win at a
comparable VLA denoise shape ("Sq=51, Skv~891, GQA 16/2, HD=128... ~17%
faster than the vendored fmha kernel, cos=1.0").

**Correction, found while reading `ThorFlashAttnBackend.run()`'s own
FA4 dispatch branch directly (not re-derived from the SigLIP usage this
entry originally cited)**: Pi0.5's "encoder" site — the actually
relevant precedent, since it's a GQA/single-KV-head site like
ImageWAM's own sites, not SigLIP's real-per-head vision attention —
calls FA4 with `k_tensor`/`v_tensor` shaped `(1, kv_seq, 1, head_dim)`
(ONE shared head) and `pack_gqa=True`, i.e. the SAME broadcast-K/V
convention OPT-002 is about, not real per-head K/V. **FA4 does not fix
OPT-002 "for free" as originally hypothesized — it is a faster
kernel for the SAME math this pipeline's own cuBLAS-composed kernel
already computes**, still useful, but OPT-002 (real per-head K/V)
remains a fully separate, unaddressed item.

## Implemented

`ImageWAMAttnBackend.__init__` gained an opt-in `use_fa4: bool = False`
parameter (default False — every existing test and caller is
unaffected, confirmed by the full regression suite passing unchanged
after this change). When `True`, the "backbone" site's `"standard"`
kernel branch dispatches through FA4 instead of `attention_qkv_fp16`,
using the EXACT tensor-view/call pattern (`k_tensor`/`v_tensor` at
`(1, kv_seq, 1, head_dim)`, `pack_gqa=True`, output via a scratch
buffer then copied into `Q_O`) that `ThorFlashAttnBackend` already uses
and has verified for Pi0.5's own "encoder" site — chosen specifically
because ImageWAM's own K/V storage is already in this exact single-
shared-head layout, so no buffer-format change was needed. Confirmed
`use_fa4=True` raises a clean, actionable `RuntimeError` on this
non-Thor machine (`ModuleNotFoundError: No module named 'cutlass'`),
matching the same import-guard discipline used for the FP4 benchmark
script. The "mot" site (three-region masked joint attention) is
untouched — FA4's plain causal/non-causal API has no evaluated
equivalent for that mask, not attempted.

`tests/test_imagewam_fa4_backbone.py` (new): compares FA4's output
against `attention_qkv_fp16`'s own already-reference-verified output on
identical random Q/K/V — skips cleanly (not a failure) when FA4 isn't
available, confirmed on this machine. The non-FA4 half of the test
harness was run directly here and produces finite, correctly-shaped
output; the FA4-specific code path itself has never executed anywhere.

`benchmarks/imagewam_fa4_vs_cublas_bench.py` (new): speed comparison at
real ImageWAM backbone dims (`NH=24, HD=128, a0=896`). The cuBLAS-only
half was run here: 0.853ms P50, matching this project's earlier
per-shape kernel-only measurements. The FA4 half is Thor-only.

## Expected Mechanism

Same mechanism already measured real for Pi0.5: FA4's own fused,
highly tuned flash-attention kernel replaces the cuBLAS QK^T -> softmax
-> PV three-launch composition, for the SAME broadcast-K/V math this
pipeline already computes (not a numerically different result, a
faster implementation of the identical math).

## Required Evidence — satisfied

Run `tests/test_imagewam_fa4_backbone.py` on Thor FIRST and confirm it
passes before trusting `benchmarks/imagewam_fa4_vs_cublas_bench.py`'s
speed number — a fast-but-wrong kernel is not a win.

## Real Thor Hardware Result

`tests/test_imagewam_fa4_backbone.py`: PASS, `cosine=1.000000,
rel_l2=0.000412` — FA4 output matches `attention_qkv_fp16`'s own
reference-verified output at real dims, confirming this is a pure
speed swap, not a numerically different computation.

`benchmarks/imagewam_fa4_vs_cublas_bench.py` (real ImageWAM backbone
dims, `NH=24, HD=128, a0=896`):

| path | P50 |
|---|---|
| cuBLAS `attention_qkv_fp16` | 0.821 ms |
| FA4 | 0.203 ms |
| speedup | **4.05x** |

Environment note for reproducing this: this venv's FlashRT reuses the
sibling openpi project's jax 0.5.3 via a `.pth` file; `nvidia-cutlass-dsl`
4.5.1's `cutlass.jax` submodule imports `jnp.float8_e8m0fnu`, which that
jax version doesn't have, crashing `import cutlass` entirely even
though FA4 only needs `cutlass.cute`. Fixed locally by wrapping that
jax submodule import in a try/except inside the venv's own
`cutlass/__init__.py` (a venv-local fix, not a change to any vendored
or repository source) — confirmed `fa4_backend.status() == "active"`
afterward.

Not yet done: this is a standalone, isolated single-call measurement
(one attention call at one shape), not wired into any of the full
25-layer prefill benchmarks (`imagewam_thor_*_bench.py` construct
`ImageWAMAttnBackend` without `use_fa4=True` anywhere) — the real
full-pipeline win from turning this on for all 25 backbone layers'
"backbone" self-attention calls has not been measured yet. Rough
estimate from the per-call number: 25 layers x roughly (0.821-0.203)ms
saved per self-attention call ≈ 15ms off the ~80-125ms prefill number,
depending on precision tier — a real but secondary win compared to
OPT-008's VAE finding below.

## Promotion Condition — met for the isolated kernel; open for pipeline integration

Kernel itself promoted: verified correct and fast on real Thor
hardware, safe to turn on. Remaining work is threading `use_fa4=True`
through the real benchmark scripts' backend construction and measuring
the actual full-prefill delta, not just the isolated per-call number.

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

Status: CLOSED for Thor (real, measured ~8.6x slower than FP16 there); remains open/unpromoted for a hypothetical true Orin/Ampere deployment

Area: INT4 (QuaRot W4A4, SM80 CUTLASS) as an additional precision option — confirmed buildable on Ada and Thor, but a real speed dead end on Thor specifically

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
the FP16/FP8/FP4 scripts, ran clean end to end on this machine. Numbers
below are AFTER the OPT-003 fix (mot_joint restricted to action
queries) — see OPT-003 for the before/after breakdown:

| | backbone prefill (25L) | one denoise step (25L) | prefill + 10-step |
|---|---|---|---|
| INT4 (GEMM-only) | 43.7 ms | 4.37 ms | 91.1 ms |
| FP16 (this machine) | 143.5 ms | 5.68 ms | 203.2 ms |

~3.3x faster prefill, ~2.2x faster overall now that OPT-003 removed
the attention-bound ceiling that used to cap the denoise step's own
speedup regardless of GEMM precision. This is still an optimistic
upper bound, not a real deployment number — see the activation-
quantization caveat above.

## INT8 Full-Pipeline Attempt (Ada) — Confirms the K=9216 Failure Is Not Just an Isolated-Shape Artifact

`benchmarks/imagewam_thor_int8_bench.py` (new, mirrors the INT4 script's
exact structure): built the full 25+25-layer INT8 pipeline successfully
(weight allocation, backend construction all fine), but `run_prefill()`
throws on the very first prefill call, inside the first double layer,
at `img_mlp2` (`cutlass_int8_rowwise_fp16out`, shape `M=768,N=3072,K=9216`,
`rc=131079`). Notably, `txt_mlp2` — the *same* K=9216 shape, called
immediately before it in the same layer, only differing in M (128 vs
768) — succeeded. This is consistent with the flakiness already
documented above for this kernel family (not purely a hard `K` gate,
since it doesn't fail on every K=9216 call), but it does settle the
question this benchmark was built to answer: the K=9216 instability is
not an artifact specific to `imagewam_gemm_precision_compare.py`'s own
mixed-precision-per-shape sequence — it reproduces immediately in a
realistic full-pipeline call pattern too, on the very first layer.
Given every one of the 25 backbone layers' `mlp2`/`mlp_down` calls uses
this exact shape, a full 25-layer prefill has effectively no chance of
completing cleanly. No further attempt made to get a full-pipeline INT8
timing number — the kernel is not currently usable end-to-end at
ImageWAM's real dims, independent of the Thor speed question (which is
moot here anyway, since OPT-007's Thor result already closed this
kernel family for Thor on speed grounds alone).

## Follow-up: does zero-padding K to a power of 2 unblock the FHT crash?

`benchmarks/imagewam_int4_hadamard_padding_probe.py` (new): real
math, not a guess — padding both activation and weight with zeros
before an orthogonal (Hadamard) rotation exactly preserves their inner
product (`<H@x_pad, H@w_pad> = <x,w>`, verified numerically, max abs
error ~1e-4). Tested whether this lets `fht_int4_quant_fp16` +
`cutlass_int4_rowwise_fp16out` run correctly at ImageWAM's real K
values once padded to the next power of 2 (weight-side rotation
implemented in plain torch for this probe only, since FlashRT itself
has no kernel for it — see the crash note above).

**Result is genuinely mixed, and inconsistent between two otherwise-
identical runs of the same logic** — worth stating plainly rather than
picking the more flattering number: K=3072→4096 works reliably
(cosine=0.983, reproduced identically across the exploratory run and 3
repeats of the formalized script). K=7680→8192 and K=9216→16384 both
FAIL in the formalized script (reproducibly, 3/3 runs) — but the
FIRST, less careful exploratory run of the identical logic had
K=7680→8192 WORKING (cosine=0.977). This exact SM80 INT4/INT8 CUTLASS
family has now shown unexplained run-to-run instability three separate
times in this project (here, and twice already in this entry's own
mlp2 findings) — treat any single "it works" result from it as
provisional until independently reproduced.

Net effect: only K=3072 (q/k/v/proj, mlp0's own input width) is
currently a reliable target for real QuaRot-rotated INT4. txt_in
(K=7680) and mlp2/mlp_down (K=9216) remain blocked — not only by the
already-known FHT crash, but now also by this padding workaround's own
unreliability at those larger sizes.

## Real Thor Hardware Result — Dead End on Thor Specifically

The user rebuilt with the same flags on real Thor (SM110) hardware:
`ENABLE_SM80_INT8_CUTLASS`/`FLASHRT_ENABLE_CHAMELEON` compile and link
fine there too, and every GEMM shape returns success (rc=0) — but
"succeeds" only means no error thrown, not numerically verified. The
real, decisive finding is speed: **this SM80-templated kernel is
dramatically SLOWER than FP16 on Thor**, not faster:

| shape | FP16 | INT8 (SM80) | INT4 (SM80) |
|---|---|---|---|
| q/proj (896×3072×3072) | 0.127 ms | 0.172 ms | **3.30 ms** |
| k/v (896×128×3072) | 0.019 ms | 0.027 ms | 0.46 ms |
| mlp0 (896×9216×3072) | 0.835 ms | 0.500 ms | **9.39 ms** |
| mlp2 (896×3072×9216) | 0.526 ms | 0.527 ms | **9.86 ms** |

Full pipeline (GEMM-only, same convention as the Ada number):
**1214ms** — versus FP16's 140.4ms on the same hardware. **~8.6x
SLOWER**, not faster. This SM80-templated CUTLASS kernel almost
certainly falls back to a compatibility instruction path that does not
use Thor's native (Blackwell) tensor cores at all — real tensor-core
INT4 throughput on Blackwell should be much faster than FP16, not
6-19x slower per-shape. Also re-ran the Hadamard-padding probe on Thor:
`K=3072→4096` matches Ada exactly (cosine=0.983); `K=7680→8192` this
time WORKS on Thor (cosine=0.977, matching Ada's very first
exploratory run rather than Ada's later reproducible failures — the
instability itself is now confirmed cross-hardware, not an
Ada-specific quirk); `K=9216→16384` fails again, with yet another
different symptom ("CUDA invalid argument" on Thor vs. Ada's
all-zero-scale-with-no-error).

**Conclusion: this SM80 INT4/INT8 path is a dead end on Thor
specifically, confirmed by real measurement, not a slower-but-usable
fallback.** Thor's own native NVFP4 (SM100) path
(`benchmarks/imagewam_thor_fp4_bench.py`) is the correct low-precision
target there. The Thor build was reset back to the default slim config
(Chameleon/INT4 off) afterward specifically to avoid this slow path
being used by accident later. This kernel family remains a legitimate
(if still K-limited and flaky above K=4096) option on the true
Ampere/Orin-class hardware it was actually built for — the negative
result here is Thor-specific, not universal.

## Opportunity

Given the Thor result above, this is now scoped OUT for Thor
deployment entirely. Remaining scope, if anyone pursues it: a fourth
precision tier on true Ampere/Orin hardware (not Thor) for
q/k/v/proj/mlp0 specifically (K=3072 only, per the padding probe
above), IF someone implements the weight-side offline rotation (this
project has none) AND the QuaRot rotation is validated for ImageWAM's
own activation distributions (a real correctness project, not yet
started — this would need real weights to even evaluate, same
dependency as OPT-001).

## Expected Mechanism

Same mechanism the Chameleon-7B path already uses in production on its
own (Orin-class) hardware (assumed, not independently re-verified
here): FHT-rotated activations + offline-rotated weights survive
int4's dynamic range at measured cosine 0.9914 (per the kernel file's
own header comment, for Chameleon's own model and data — not
re-measured for ImageWAM, and now confirmed NOT to translate to a
speed win on Thor even where it runs).

## Required Evidence

For Thor: none needed further — closed, real negative result in hand.
For a hypothetical true-Orin deployment: resolve the FHT power-of-2
blocker (every real ImageWAM hidden dimension is a non-power-of-2
multiple of 1024) and its own reliability above K=4096, plus
real-weight accuracy work (OPT-001).

## Promotion Condition

Closed for Thor (real, measured, ~8.6x slower — promoted as a verified
negative result, not pursued further there). Would need a real
Orin/Ampere deployment target and the required evidence above to be
worth reopening elsewhere.

# OPT-008

Status: real VAE-encode cost added to all local/Thor full-pipeline benchmarks; exposed a real gap in `pipeline_thor.py` itself (no `img_in` weight modeled)

Area: VAE encoder (input-image tokenization) — previously excluded from every full-pipeline speed number with no clear justification; now included

## Observation

Earlier framing treated the VAE the same as the text encoder ("both
can be precomputed, so excluding both from per-step speed numbers is
fine"). User correction: this is wrong for the VAE specifically — a
fixed task instruction can be encoded once per episode, but the camera
observation changes every control-loop iteration, so VAE encode must
run once per real inference call (once per new observation), not once
per episode. It is a real, recurring, resident cost that belongs in a
"real machine full inference steady-state speed" number.

Read ImageWAM's actual upstream source (`imagewam.py`) to confirm
scope precisely before implementing anything: `infer_action_flux2`
(the real action-only inference entry point this project's pipeline
mirrors) calls `_encode_flux2_image_tokens` (VAE **encode** only) once
at the start, denoises only the action latent, and returns
`{"action": ...}` — no `vae.decode` call anywhere in that path. The
video-editing branch that would need decode (`infer_video_flux2`) is a
separate, unused method. So only VAE **encode** is in scope here,
never decode — confirmed from source, not assumed.

## Implementation

`benchmarks/_imagewam_vae_stub.py` (new, shared by all `imagewam_thor_*_bench.py`
scripts): a standard SD/FLUX-family latent-diffusion VAE encoder
(`ch=128`, `ch_mult=(1,2,4,4)`, 2 ResnetBlocks/stage, one mid-block
self-attention, 8x spatial downsample, 16 latent channels) implemented
in plain PyTorch/cuDNN (Conv2d/GroupNorm/SiLU/`scaled_dot_product_attention`)
— **not the real FLUX.2 AE**, which lives in `black-forest-labs/flux2`
on GitHub and is not vendored anywhere on this machine (confirmed via
ImageWAM's own `docs/dependencies.md`: "user clones upstream, not
vendored") and not fetchable here. This is a representative, real
(non-trivial, real cuDNN conv) workload of the right computational
order, consistent with this project's whole "random weights, no real
checkpoint" scope — not a claim of bit-exact match to the real
architecture. Input resolution `384x512` chosen to produce exactly 768
tokens, matching `A0-X0` already used by every bench script's image
token span.

**Real gap found and exposed, not previously visible**: `pipeline_thor.py`'s
own docstring already states explicitly that only `txt_in` (text
projection) is modeled, with "the img_* analogs of all but txt_in" —
i.e. there has never been an `img_in` weight in this project's modeled
math anywhere; image tokens were always assumed to already arrive at
HIDDEN width. That was a reasonable simplification while image tokens
were pure random placeholders, but a real VAE's raw patch output is
64-dim (`16 latent channels * 2x2 patch merge`), not 3072-dim — an
`img_in: Linear(64, HIDDEN)` projection is structurally required to
connect the two, matching every real DiT-style architecture's own
input embedder. Added ONLY inside the benchmark scripts (`_Int4Linear`/
`_Fp16Linear`/`_Fp8Linear`/`_Fp4Linear`(HIDDEN, 64), matching each
script's own precision) — **not yet added to `pipeline_thor.py`
itself**, which remains a real, open gap for whenever this project
does real VAE integration in the actual pipeline, not just in speed
benchmarks.

Wired into `run_prefill()` (called once per call, before the double/
single layer loop) in `imagewam_thor_int4_bench.py`, `_fp16_bench.py`,
`_fp8_bench.py`, `_fp4_bench.py` — not `_int8_bench.py`, since that one
already fails to complete a full pipeline run for an unrelated reason
(OPT-007's own K=9216 finding) and the user's own stated policy is
"INT8 only if it fits, otherwise INT4 only" for this machine.

## Real Local Result (Ada, INT4 and FP16 — both fully run here)

Standalone VAE encode (unfused, naive plain-PyTorch implementation,
not optimized): ~55-56ms on this machine, regardless of downstream
GEMM precision (same VAE, same input, every script).

| | vae_encode (standalone) | backbone_prefill (25L + VAE) | one denoise step | full (prefill + 10-step denoise) |
|---|---|---|---|---|
| INT4 | 55.4 ms | 96.9 ms | 4.78-4.96 ms | 136.6 ms (single run) / 144.7 ms (cross-check sum) |
| FP16 | 56.4 ms | 200.4-200.5 ms | 6.15-6.22 ms | 260.2 ms |

VAE cost is a large, fixed addition independent of backbone precision
(~55ms regardless of INT4 vs FP16) — it roughly DOUBLES the INT4
prefill number (42.6ms -> 96.9ms) since INT4's own GEMM cost is small,
while it's a much smaller relative addition to FP16's own already-large
prefill (152ms range without it). This means **VAE cost matters more,
relatively, the faster the backbone gets** — a real consideration for
whether backbone-only precision work (INT4/FP8/FP4) alone is enough to
reach a target full-inference latency, or whether the VAE itself will
need optimization (fusing GroupNorm+SiLU, cuDNN autotuning, or a faster
architecture) to actually see the backbone's speedup reflected in the
full number.

FP8/FP4 scripts confirmed to run the new VAE step cleanly on this
machine before hitting their own already-known, unrelated failure
points (FP8: cuBLASLt env gap; FP4: Blackwell-only `SystemExit`) — not
verified end-to-end anywhere, same status as before this change.

## Real Thor Hardware Result — VAE Is the Single Biggest Remaining Cost

User ran all three full-scale scripts (`fp16`/`fp8`/`fp4`) on real Thor
with this change in place:

| | vae_encode | prefill (25L + VAE) | one denoise step | full (prefill + 10-step) | full, no VAE (prior measurement) |
|---|---|---|---|---|---|
| FP16 | 43.9 ms | 124.5 ms | 5.87 ms | 183.3 ms | 140.4 ms |
| FP8 (dynamic scale) | 43.7 ms | 107.8 ms | 6.17 ms | 169.5 ms | 106.6 ms (old fixed-scale, no VAE) |
| FP4 (NVFP4) | 43.8 ms | 98.7 ms | 5.51 ms | 153.7 ms | 111.1 ms |

Backbone-only numbers (VAE subtracted back out) still match the
earlier no-VAE measurements closely — FP16 ≈80.6ms vs the earlier
81.6ms, FP4 ≈54.9ms vs 55.8ms — confirming this change added a real
new cost without disturbing anything already measured. FP8's own
backbone-only prefill grew by ~4ms and its denoise step by
4.65ms->6.17ms specifically because of switching to genuine per-call
dynamic scale measurement (`quantize_fp8_device_fp16`'s real amax
kernel, see the FP8 bench script's own commit) — a real, expected cost
of not having real calibration, not a regression.

**The VAE is ~24-28% of full pipeline latency on Thor** — bigger than
the entire 10-step denoise loop post-OPT-003, and bigger than every
win OPT-004 (graph capture + autotune combined, ~10ms) found. Re-run
with this in mind: graph capture still doesn't include the VAE step
(the frontend's own `_capture_graph` only captures `imagewam_prefill`/
`imagewam_denoise_loop`, and the graph/autotune benchmark scripts still
fill image tokens with random data, matching their own pre-existing,
unaffected scope) — captured graph (129.3ms) + VAE outside the graph
(43.9ms) ≈ 173ms, vs. graph-free FP16 with VAE (183ms): graph capture
now saves only ~10ms out of a ~44ms-larger VAE cost sitting right next
to it. **This sharpens OPT-004's own "compute-bound, not launch-bound"
conclusion**: reducing real compute (the VAE, or the backbone's own
GEMMs) is worth more than shaving launch overhead, and now there's a
concrete, large, first-priority target (VAE) that dwarfs the graph/
autotune win entirely.

## Correction: Real Token Width Is 128, Not 64 — Found While Starting OPT-002's Real-Checkpoint Work

While reading ImageWAM's real upstream source (`flux2_video_expert.py`,
already cloned locally) to understand the real checkpoint's tensor
naming for OPT-002's accuracy-validation harness, found this module's
own VAE token-width assumption was wrong. Two real, verbatim facts from
that source: `Flux2VideoExpert.pre_dit`'s own docstring states packed
image tokens must be `[B,N,128]`, and `Flux2VideoExpert.pack_latents`
is `rearrange(latents, "b c h w -> b (h w) c")` -- a PURE reshape, no
2x2 patch-merge at all. Together these mean the real FLUX.2 VAE
downsamples 16x spatially AND emits 128 channels directly, not the
classic SD/FLUX.1 pattern (8x downsample + 16 latent channels + a
separate 2x2-merge to 64-dim tokens) this module wrongly assumed by
analogy when first written. Fixed: the stub encoder now downsamples
16x in one conv stack (5 stages, 4 downsamples) and emits 128 channels
directly; `pack_latents` is now a pure reshape too, matching the real
one exactly. `img_in`'s K dimension changed from 64 to 128 accordingly
in every bench script (they import the constant, not a hardcoded
value, so this needed zero changes to the scripts themselves).
Re-verified INT4/FP16 end to end on this machine after the fix (both
still run cleanly, VAE cost ~49ms, close to the pre-fix number); FP8/
FP4 still fail at the same known, unrelated points, now correctly
showing K=128 in the error message instead of K=64.

## VAE Stub Optimization Attempt (Ada, this dev machine) — Real Gain, Reverted to Off by Default for Reliability

Profiled with `torch.profiler` first rather than guessing: convolution
itself dominates (~66% of GPU time), with the rest split across
GroupNorm/SiLU and cuDNN's own NCHW<->NHWC layout-conversion kernels
(inserted automatically when its fastest conv algorithm for a given
shape prefers the other layout). Tried, in order:

1. `channels_last` memory format + `cudnn.benchmark=True`: made things
   WORSE (55ms baseline -> 63ms), most likely because the mid-block
   attention's own `.reshape()`/`.permute()` calls silently force a
   layout conversion back to contiguous NCHW, adding overhead without
   removing the conversions this was meant to avoid.
2. `torch.compile(mode="max-autotune")`: real gain in isolation
   (~55ms -> ~44ms) and confirmed end-to-end in the INT4/FP16 full-
   pipeline scripts. But running the SAME change inside the FP8 script
   made the GPU sit at 100% util / ~7.9-7.92GB of this machine's 8GB
   total (near-OOM) for 45+ seconds with zero forward progress logged
   — required a hard kill.
3. `torch.compile(mode="default")`: still a real gain (~55-61ms ->
   ~48ms), and fixed INT4/FP16 (both confirmed clean). But the SAME
   FP8 script then stalled AGAIN — 65+ seconds stuck inside
   `FullImageWAMFP8.__init__` itself (before "Built." even prints,
   i.e. before any VAE forward call happens at all), GPU pinned at
   100% util, memory again near ~7.9GB. No lingering compile-worker
   process found; INT4/FP16 use the identical `build_vae_encoder` call
   and never reproduced it. Not root-caused.

**Decision: kept `torch.compile` as an explicit opt-in
(`compile=True`), reverted the default back to `compile=False`.** The
real, measured gain (~13-20% off the VAE's own cost) is not worth an
unpredictable, unexplained path to a near-OOM multi-minute stall on a
machine whose 8GB budget is already tight — this project's own
standing memory-safety discipline says no. Worth revisiting on Thor,
which has far more memory headroom and may simply not reproduce this;
not investigated further here per the effort/risk tradeoff.

## Promotion Condition

VAE STUB's own architecture not further promotable without either (a)
fetching FLUX.2's real AE source/config to replace this representative
architecture with the real one, or (b) real weights/calibration
(OPT-001) making VAE accuracy relevant, not just its speed. **VAE
OPTIMIZATION remains the top-priority performance item** given the real
Thor latency-share result above — ahead of OPT-005's own pipeline-
integration follow-up and any further backbone GEMM work — but
`torch.compile` specifically is not itself promoted to default given
the reliability finding just above; the next attempt should either
root-cause the FP8-specific stall (ideally on Thor, where memory
pressure may not be a confound) or pursue the custom-fused-kernel path
(mirroring `cosmos3_edge/vae_native.py`'s own GroupNorm+SiLU fusion
precedent) instead of `torch.compile`. The `img_in` gap in
`pipeline_thor.py` itself is a separate, smaller follow-up worth its
own promotion once real VAE integration (not just a speed stub) is
in scope.
