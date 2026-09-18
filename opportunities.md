# OPT-001

Status: RESOLVED end to end, including real Thor validation (2026-09-15, `plan.md`'s own "OPT-001" plan, all 4 phases complete); FP8 calibration/quantization split off into OPT-004 steps 5-6 (already resolved separately, see that section)

Area: ImageWAM on Thor — precision and real-weight path

## Real-weight path result (2026-09-15)

`imagewam_thor.py` gained a `ckpt_path=` constructor kwarg; new
`checkpoint_loader.py` reads the real release checkpoint's own raw
`state_dict` by key name (`torch.load(..., mmap=True)['mot']` — no
`imagewam`/`flux2` packages needed at all, just `torch`). Two real
architecture gaps found and fixed as prerequisites: `img_in` (image
tokens were never projected from raw `HD` width to `hidden` width
anywhere in this project) and `action_encoder`/`head` (the flow-
matching Euler integration was happening in `action_hidden_dim` space
instead of real `action_dim` space) — both confirmed by reading
`imagewam`'s own real source directly (a local read-only clone,
`/home/ljw/projects/pi0.5/tmp/ImageWAM`) and closing `opportunities.md`
OPT-008's long-standing "img_in not modeled anywhere" finding as a
side effect.

**Verified END TO END on this dev machine, not just Thor-blind code
review** — a genuine surprise mid-plan: the real checkpoint FILES
(`/home/ljw/projects/pi0.5/models/`) turned out to already be present
locally (contradicting `PROJECT.md`'s own prior "never will" claim,
now corrected there), letting real weights actually be loaded and run
here despite `flux2` source still being absent and 8GB VRAM being far
short of the real model's ~8.9GB weights alone. `ImageWAMTorchFrontendThor`
constructed with real weights at REAL FLUX.2-4B dims, captured a real
CUDA Graph, and replayed it, producing a finite `(64,7)` action tensor
(mean=0.21, std=0.34) — construction, capture, AND inference all real.
Peak CUDA memory measured at ~9.86GB despite `nvidia-smi` reporting
8188MiB total, evidently because this WSL2 environment's CUDA driver
pages beyond dedicated VRAM into host RAM rather than raising OOM (not
reliable for a Thor PERFORMANCE claim, but real enough for a
correctness check).

Two real, silent-wrong-shape/convention bugs were found and fixed only
because this was actually run, not just reviewed: modulation weights
needed the real `(out,in)` layout (plain `F.linear`), not the FlashRT
`(K,N)` GEMM-transposed one; ActionDiT's own double-block weights
needed PLAIN (unprefixed) slot names, not the backbone's `img_`-prefixed
dual-stream convention. New `tests/test_imagewam_checkpoint_loader.py`
(skips cleanly if the real checkpoint files aren't present elsewhere)
locks in all three checks (shape match, one real-weight layer forward,
full real-checkpoint frontend) as permanent regression coverage.

**What remains, Thor-only**: confirming this same sequence completes
on Thor's own 128GB unified memory without the WSL2 paging dependency,
and a real per-layer P50 with real weights (this dev machine's numbers
would not be a meaningful Thor performance number). See `plan.md`'s
own "OPT-001" Phase 4.

## Original framing (superseded by the above, kept for history)

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

## Real Thor result (2026-09-15) — correct calibration dataset, confirmed real deployment shape, real VAE timing

User located the CORRECT real dataset for this project (the earlier
"Real multi-sample calibration" entry below misidentified
`JingwuLuo/LingBot-VA_RoboTwin_clibration_data` as the candidate and
found it shape-mismatched — that finding stands for THAT dataset, but
was the wrong one to be looking at): `yuanty/LIBERO-fastwam` (HF), the
FastWAM-preprocessed set ImageWAM's own README actually points at,
subset `libero_spatial_no_noops_lerobot` — Franka, 434 episodes /
53229 frames / 10 tasks, real LIBERO language instructions, two
512x512x3 camera views (AV1, 20fps), `action: (T,7)`, `state: (T,8)`.

**Confirmed real deployment image-token shape, superseding the
768-token (`384x512` input) guess used everywhere in this project
until now**: real eval preprocessing (`config.yaml`/
`eval_libero_single.py`) resizes each camera view to `224x224` and
concatenates horizontally to `224x448`. The real official
`FLUX.2-dev/ae.safetensors` VAE (`x*2/255-1` preprocessing, matching
eval) encodes this to a `(B,128,14,28)` latent -> `(B,392,128)` packed
tokens: **`img_len=392` (14x28), not 768**. VAE encode itself: real
Thor P50 **41.0ms** for `224x448`. `benchmarks/imagewam_thor_bench.py`
and `imagewam_real_checkpoint_validation.py` both updated to this
confirmed shape (`a0=520`, `total=584`); every prior OPT-004 step 5/6
table below was measured at the OLD 768/896/960 shape -- still real
results, but not yet re-confirmed at this one (see Thor checklist
below).

**FP8 `img_in` calibration, holdout real VAE tokens vs. the FP16
reference** -- the first real activation-distribution measurement this
project has ever had for anything:

| calibration source | act_scale | cosine vs FP16 |
|---|---:|---:|
| `N(0, 0.1)` noise (this project's own placeholder, all along) | 0.00102 | **0.902** |
| real VAE tokens (mean=-0.02, std=0.97, absmax=4.91) | 0.01086 | **0.99946** |

Real tokens are an order of magnitude wider than the 0.1-scale
placeholder this project's own `_calibrate_fp8`/`_calibrate_static_fp8`
used everywhere -- clipping real activations, not just "approximating"
them. Fixed: both functions now special-case `img_in.weight`'s own
calibration input to `N(-0.02, 0.97)` (the real measured stats),
documented inline as a narrow, single-slot fix -- every OTHER weight's
own calibration input (txt_qkv, mlp0, ActionDiT's own weights, etc.)
still uses the same unvalidated 0.1-scale placeholder, likely similarly
wrong, with no real ground truth yet to correct it against (would need
a full real forward pass propagating real intermediate activations,
a larger redesign not attempted here).

**Real-weight FP16 `infer()` at the corrected 392-token shape** (CUDA
Graph, `img_raw` still per-step `normal_()` -- real GEMM shapes, not
real VAE tokens plugged into the graph itself):

| shape | median infer() |
|---|---:|
| img_len=768 (old) | 231.1 ms |
| img_len=392 (confirmed real) | **172.8 ms** |

Peak allocated 9.99GB (same order as the 768-token shape's own 9.86GB
-- no additional WSL2 paging from the shape change). With the real VAE
folded into the full pipeline: ~41 + 173 ≈ **214ms** (still without
Qwen3 -- `txt_in` still reads a random `context`; real text-side
calibration/encoding needs Qwen3-4B or the training-time
`qwen_text_cache`, neither wired in).

**Follow-up, Thor-only — DONE (2026-09-15, same commit)**: re-measured
the OPT-004 step 5/6 FP8/NVFP4/CUTLASS comparison at the corrected
`img_len=392` shape. **Relative rankings changed from the 768-token
table**:

| layer | FP16 | FP8 dynamic | FP8 static | FP8 static+CUTLASS | NVFP4 |
|---|---:|---:|---:|---:|---:|
| backbone_double | 3.93 | 3.88 | 3.69 | 3.27 | **3.00** |
| backbone_single | 2.57 | 2.24 | **2.15** | 2.19 | **2.02** |
| action_double | 0.54 | 0.46 | 0.47 | 0.45 | **0.40** |
| action_single | 0.52 | 0.46 | 0.43 | 0.40 | **0.35** |
| **prefill (5+20)** | 71.1 | 64.3 | 61.4 | 60.2 | **55.4** |
| denoise x1 | 13.1 | 11.6 | 10.9 | 10.2 | **9.1** |
| prefill+10-step | 202 | 180 | 170 | 163 | **146** |

Attention kernels: mot 0.037ms (unchanged), standard/backbone attn
0.15ms (down from 0.82-0.84ms at `a0=896` -- the O(a0^2) attention cost
shrank with the smaller sequence).

- **At 768 tokens, dynamic FP8 was SLOWER than FP16** (this file's own
  OPT-004 step 6 entry). **At 392 tokens, dynamic FP8 is now FASTER
  than FP16** (64.3 vs 71.1ms) -- the per-call amax-reduction overhead
  that dominated at the larger shape matters less at the smaller one.
- **The CUTLASS-over-static-cuBLASLt gap shrank dramatically**: 768
  tokens had CUTLASS beating static-cuBLASLt by ~20ms (97.9 vs
  118.1ms); at 392 tokens the gap is 1.2ms (60.2 vs 61.4ms), and
  `backbone_single` is actually slightly SLOWER with CUTLASS (2.19 vs
  2.15ms) -- confirms `_pick_fp8_cutlass_variant`'s own flagged caveat
  that its shape-based heuristic doesn't necessarily transfer to
  smaller M; the CUTLASS win is real but shape-dependent, not a free
  lunch at every M.
- **NVFP4 remains the fastest precision at both shapes**, ranking
  unchanged.

**Full-pipeline re-validation, `REF_H,REF_W=14,28` + `img_in`**: backbone
cosine=**0.999918** (was 0.999927 at the old 24x32 grid, no `img_in`),
ActionDiT cosine=**0.999962** (was 0.999963) -- output shape `(520,3072)`
matches `a0=520`. Adding the real grid + `img_in` barely moved cosine;
confirms neither broke anything.

**`test_imagewam_quant_linear.py` on Thor**: no SKIP, four real cosines
-- `Fp8Linear` 0.999242, `StaticFp8Linear(cublaslt)` 0.999242,
`StaticFp8Linear(cutlass)` 0.999242, `Nvfp4Linear` 0.989133 (all
unchanged from earlier Thor runs -- this test is still small-shape
random weights and does NOT exercise `img_in`'s own `N(-0.02,0.97)`
calibration path; that 0.99946 number remains the standalone holdout
measurement above, not re-verified through this test file).

# OPT-002

Status: kernel-level RESOLVED for both backbone block types (verified
on real Thor hardware, commit `e329d3a`) AND now ActionDiT/"mot"
(Ada-only so far, not yet re-verified on Thor) — full real-math
coverage complete, INCLUDING a major mask correction found while
building ActionDiT (see below: the mask this round first built, and
the pre-existing "mot_joint" kernels, both had the wrong rule). NOT YET
wired into `pipeline_thor.py`/`_imagewam_thor_spec.py`, and NOT YET
validated against a real checkpoint (no checkpoint file available
locally or on Thor yet)

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
- **AdaLN modulation and the real LayerNorm are now also done** (as of
  `flash_rt/models/imagewam/adaln.py` / `tests/test_imagewam_adaln.py`):
  `timestep_embedding`, `MLPEmbedder` (`time_in`), `Modulation`
  (shift/scale/gate), and the real `elementwise_affine=False` LayerNorm
  (via the existing `layer_norm_no_affine_fp16` kernel, confirmed exact)
  are all implemented and verified against an independent reference,
  end to end at real dims (cosine>0.999). Deliberately plain PyTorch for
  the embedding/modulation math itself (a per-batch, not per-token,
  computation — negligible cost, not worth new-kernel risk); the
  LayerNorm itself (the one piece touching the full (S,D) hidden state)
  uses the real FlashRT kernel.
- **The real MLP is also done**: real `DoubleStreamBlock.img_mlp`/
  `txt_mlp` is `Linear(hidden, mlp_hidden*2) -> SiLU-gated GLU chunk ->
  Linear(mlp_hidden, hidden)` — TWICE the first GEMM's width this
  project's existing benchmark scripts/pipeline assumed, and a SiLU-gate
  chunk instead of plain GELU (a real speed+accuracy correction, not
  yet applied to `imagewam_thor_*_bench.py`/`pipeline_thor.py`). New
  `silu_glu_merged_fp16` kernel + `flash_rt/models/imagewam/real_mlp.py`,
  verified (cosine=1.0, small shape + real dims).
- **A full single real `DoubleStreamBlock` forward is now combined and
  verified**: `flash_rt/models/imagewam/real_double_stream_block.py`
  chains every piece above (per-head K/V, RoPE, QK-Norm, the real mask,
  AdaLN modulation, real LayerNorm, real MLP) in the exact real order
  (norm1 -> modulate -> qkv -> QK-Norm -> concat -> RoPE -> masked
  attention -> split -> proj -> gated residual -> norm2 -> modulate ->
  MLP -> gated residual, separately for txt and img streams sharing one
  combined attention call) — confirmed against a from-scratch
  independent PyTorch reference of the full real block, at both a small
  shape and real ImageWAM dims (hidden=3072, mlp_hidden=9216, NH=24,
  HD=128, x0=128, img_len=768): cosine=1.0 for both txt and img, both
  shapes (`tests/test_imagewam_real_double_stream_block.py`). This is
  the technical milestone the whole OPT-002 round was building toward —
  real math is now fully understood and verified at the single-block
  level.
- **`SingleStreamBlock`'s own real forward is also now done**:
  `flash_rt/models/imagewam/real_single_stream_block.py` — the 20
  single-stream layers operate on the already-concatenated `[txt|img]`
  sequence with fused `linear1`(QKV+MLP-in)/`linear2`(attn-out+MLP-out)
  GEMMs, represented here as separate GEMMs summed/split
  (mathematically identical, documented in the module's own docstring).
  Reuses every kernel already verified for `DoubleStreamBlock` with
  zero new kernel code. Verified against an independent PyTorch
  reference, small shape + real dims: cosine=1.0
  (`tests/test_imagewam_real_single_stream_block.py`).
- **All 25 real backbone layers (5 double + 20 single) are now looped
  together and verified**: `flash_rt/models/imagewam/pipeline_real.py`'s
  `imagewam_prefill_real` chains both block types in the real order,
  with modulation correctly SHARED across all layers of a stream type
  (a real architecture property confirmed from `Flux2.forward` — one
  `Modulation` output per forward, reused by every layer, not per-layer
  separate modulation) via `compute_shared_modulation`. Verified: a
  1-layer loop matches a direct block call (wiring correctness,
  cosine=1.0), and the full 25-layer real backbone at real dims
  produces finite, well-behaved output (mean~0, std~0.5 — no explosion/
  vanishing across 25 layers of random weights)
  (`tests/test_imagewam_pipeline_real.py`). This module is explicitly a
  correctness-verification path, not a steady-state perf path — it
  allocates fresh buffers every call with no reuse; one test run OOM'd
  when the GPU already had ~5.3GB in use from unrelated earlier
  processes in the same session, succeeded cleanly once cleared.
- **`ActionDiT` ("mot" site) real forward is now also done**:
  `flash_rt/models/imagewam/real_action_expert.py` implements the real
  `SlimFlux2DoubleBlock`/`SlimFlux2SingleBlock` (ImageWAM's own
  `action_dit_flux2.py`, not `flux2/model.py` — IMG-ONLY, no txt
  branch) plus the real joint-attention orchestration from `mot.py`'s
  `forward_flux2_action_with_video_cache`: action's own fresh Q/K/V
  concatenated with a FROZEN backbone K/V cache, one attention call
  (Q=action rows only, K/V=full concatenated sequence), no mask (per
  the correction below). Also found and handled: `attn_dim` (`NH*HD`,
  3072) `!= hidden` (1024) here, unlike the backbone where they're
  equal — every function takes both separately. ActionDiT's own RoPE
  uses a different position convention (`build_action_ids`: axis0=2.0
  type marker, axis1=running index) but the same `pe_embedder` config,
  confirmed from `mot.py`'s own
  `action_pe = video_expert.transformer.pe_embedder(action_ids)` call.
  Verified against an independent PyTorch reference, small shape + real
  dims (hidden=1024, attn_dim=3072, mlp_hidden=4096, NH=24, HD=128,
  num_action=64, backbone_total=896): cosine=1.0, all 4 cases
  (`tests/test_imagewam_real_action_expert.py`). **This completes
  OPT-002's real-math coverage** — both backbone block types and the
  action expert now have fully verified real forwards.
- **NOW WIRED into `pipeline_thor.py` (2026-09-14) — this is the
  confirmed real deployment target, per `PROJECT.md`'s "Confirmed end
  goal."** `pipeline_thor.py`, `_imagewam_thor_spec.py`, and
  `flash_rt/frontends/torch/imagewam_thor.py` were rewritten IN PLACE
  (not a parallel file) to use the real math by default: real per-head
  K/V (`ImageWAMAttnBackend(use_perhead_kv=True, use_real_mot_mask=True)`,
  now the default for this frontend), real QK-Norm/RoPE/AdaLN
  modulation, real SiLU-gated-GLU MLP widths (`mlp_hidden*2`), and the
  real no-mask attention rule. AdaLN modulation and RoPE tables are
  precomputed ONCE (backbone: fixed conditioning timestep; ActionDiT:
  once PER denoise step, since `step` is already a compile-time
  constant during CUDA Graph capture) rather than recomputed per
  replay — still allocate-once/steady-state/CUDA-Graph-compatible,
  matching every other real Thor pipeline in this codebase.
  Verified (`tests/test_imagewam_thor_real_wiring.py`, new): each
  pointer-based layer helper (`_double_stream_layer`,
  `_single_stream_layer`, `_action_double_layer`,
  `_action_single_layer`) matches the already-verified tensor-level
  reference (`real_*.py`) built from IDENTICAL weights, cosine
  0.999984-1.000000. `tests/test_imagewam_frontend.py`/
  `test_imagewam_prefill.py`/`test_imagewam_denoise.py` (existing wiring
  tests) updated to the new weight-key/buffer conventions and still
  pass. `benchmarks/imagewam_thor_bench.py` (the isolated per-layer-type
  speed probe, at real dims) updated to match and now measures the
  REAL math's cost (heavier than the old approximation: real per-head
  K/V and doubled MLP-gate width both add real GEMM work).
  **Real bug found and fixed along the way**: `make_imagewam_attention_spec`
  hardcoded `num_q_heads=24, head_dim=128, num_layers=25` with no
  override -- `ImageWAMAttnBackend.run()` reads these from the spec
  object, not the caller's own dims, so any caller using smaller test
  dims got a silent out-of-bounds attention read/write (huge
  finite-looking garbage, not a crash). Every prior test at small dims
  only checked NaN/shape, never a real numeric reference, so this went
  undetected; every test that DID check real correctness happened to
  already use the real 24/128/25 values, masking it by coincidence.
  Now takes `num_layers`/`num_heads`/`head_dim` overrides (default to
  the real values, so every real-dims caller is unaffected). See that
  function's own docstring for the full account.
  **Still open**: real checkpoint LOADING (the frontend's
  `checkpoint_dir` arg is still unused; weights stay random-filled —
  needs the real `imagewam`/`flux2` packages, only available on Thor,
  reusing `benchmarks/imagewam_real_checkpoint_validation.py`'s own
  `extract_*_weights` functions rather than re-deriving extraction
  here). The 6 quantized precision-comparison benchmark scripts
  (`imagewam_thor_{fp16,fp8,fp4,int8,int4}_bench.py`,
  `imagewam_thor_fp16_autotuned_bench.py`) each carry their OWN
  self-contained forward implementation (confirmed: none of them
  import from `pipeline_thor.py`), independent of this rewrite --
  they still run the OLD approximate math internally and are now
  stale relative to the confirmed real-math standard; updating them is
  real, separate follow-up work (each needs the same per-script
  weight/buffer/mod/rope updates `imagewam_thor_bench.py` just got,
  plus their own precision-specific quantization wrapper changes) —
  not started.
- **Real checkpoint validation now DONE, run by the user on Thor
  (2026-09-14)** — `benchmarks/imagewam_real_checkpoint_validation.py`
  (written this round, dry-run tested locally via
  `tests/test_imagewam_real_checkpoint_extraction.py` against fake
  modules) was run against the real
  `yuyangalin/ImageWAM-FLUX.2-4B-LIBERO` release checkpoint (`model.pt`,
  not `checkpoint.pt` as this round's docstring had guessed;
  `config.yaml`, not `train_config.yaml`; `action_dim=7`, LIBERO 7-DoF,
  explicitly confirmed against the release config rather than trusting
  the script's own default). `model.load_checkpoint` reported
  `missing_keys=0 unexpected_keys=0` — the LoRA-merge branch flagged as
  an open uncertainty was a non-issue for this checkpoint (only an
  unrelated `proprio_encoder` warning, expected since the script never
  passes `proprio_dim`). Real full-network result, against the ACTUAL
  official reference path (`model.video_expert.pre_dit` +
  `_build_mot_attention_mask_flux2` + `mot.prefill_flux2_video_cache`
  for backbone; `model.action_expert.pre_dit` +
  `mot.forward_flux2_action_with_video_cache` for ActionDiT):
  **Backbone cosine=0.999927, ActionDiT cosine=0.999963** (both against
  a 25-double+single-layer full prefill / full ActionDiT forward, real
  dims, real trained bf16 weights, expected small headroom below 1.0
  purely from bf16-vs-fp16 precision, not from any math error). This
  confirms every real-math correction in this round (per-head K/V, RoPE,
  QK-Norm, AdaLN, real LayerNorm, real MLP, and the corrected no-mask
  rule) end-to-end against real trained weights, not just independent
  PyTorch references. **OPT-002's real-math coverage is now fully
  validated, not just theoretically verified.** The remaining gap is
  purely the wiring one already noted above: none of this is in
  `pipeline_thor.py`'s actual serving path yet.
- **Real-math coverage extended from single-forward to the FULL model
  (backbone + complete denoise loop)**: `pipeline_real.py`'s new
  `imagewam_full_forward_real` runs one backbone prefill (extending
  `imagewam_prefill_real` with a `collect_kv_cache=True` option that
  returns each of the 25 layers' own post-QKNorm+RoPE per-head K/V,
  needed by the action expert's joint attention) followed by the whole
  ActionDiT flow-matching denoise loop over `num_denoise_steps`, with
  action's own AdaLN modulation correctly recomputed every step from
  that step's changing timestep (new `compute_action_modulation`,
  mirroring `compute_shared_modulation`'s pattern for ActionDiT's own,
  separate `time_in`/mod weights) and an Euler update between steps.
  Verified (`tests/test_imagewam_pipeline_full_real.py`): a 1-denoise-
  step run matches a manual reference built by calling the
  already-verified lower-level block functions directly with the same
  weights (cosine=1.000000) — checking specifically the NEW wiring this
  function adds (per-layer K/V cache indexing between the backbone and
  same-indexed action layer, per-step modulation recompute, the Euler
  update), not re-verifying any single block's own math; a 4-step loop
  produces finite, well-behaved output. Still explicitly a
  **correctness-verification path, not steady-state** (fresh buffer
  allocation every call, no CUDA Graph) — same status as every other
  `real_*.py`/`pipeline_real.py` module; does not change what
  `pipeline_thor.py` still needs (see above). Two structural
  approximations carried over from `pipeline_thor.py`'s own established
  dry-run convention, documented in `imagewam_full_forward_real`'s own
  docstring: the backbone's own conditioning timestep is fixed at 0.0
  (matches the real-checkpoint validation run above exactly); ActionDiT's
  real `action_encoder` (`Linear(action_dim, hidden_dim)`, WITH bias)
  is not modeled — raw `action_latent` is fed to the transformer blocks
  directly, same simplification `pipeline_thor.py` already makes.

## Major correction: the real attention mask is NOT what this round built (found while investigating ActionDiT)

While reading `ActionDiTFlux2` (ImageWAM's own source, not
`flux2/model.py`) to build its real forward, found that ImageWAM's
`MoT` class (`imagewam/models/backbones/mot.py`) does **not** call
`block.forward_kv_extract` (which internally uses `flux2/model.py`'s
`causal_attn_fn`, the function this round's "backbone_ref_masked" mask
was based on) for its real joint-attention path at all. It calls
`block._prepare_qkv`/`block.prepare_qkv` directly to get raw Q/K/V, and
does its OWN attention via `MoT._mixed_attention`, using a mask built
by `imagewam.py`'s `_build_mot_attention_mask_flux2` — a completely
different function with a completely different rule.

Read that function directly: for a `[text | ref | target | action]`
sequence, text and ref attend to `[text,ref]`, target additionally
attends to itself, and action attends to `[text,ref]` and to itself,
never to `target`. Read its real call sites in `infer_action_flux2`
directly too (this project's actual real deployment target): **both
calls pass `target_len=0`** — the real action-inference path never has
a separate noisy/target-image segment at all, only text, one reference
image, and (later) action tokens. With `target_len=0`, the rule reduces
to: **no masking between text and ref at all** (full bidirectional
visibility), and **action sees everything** (text, ref, AND action) —
not "action excludes the image region", which is what this project's
own pre-existing "mot_joint"/"mot_joint_action" kernels (OPT-003,
predates this session) AND this round's new "backbone_ref_masked"
kernel both assumed.

**Fixed in this round's own new modules**: `real_backbone_attn.py`,
`real_double_stream_block.py`, `real_single_stream_block.py`, and their
tests now use plain unmasked `attention_qkv_fp16_perhead` instead of
`attention_qkv_fp16_backbone_ref_masked_perhead` — re-verified against
independent PyTorch references with the mask removed, all still
cosine≈1.0 at real dims. `attention_qkv_fp16_backbone_ref_masked_perhead`
itself is NOT deleted (it's a real, correct kernel for a DIFFERENT real
rule `causal_attn_fn` genuinely uses elsewhere, just not on this
project's real deployment path) — kept as a validated building block,
documentation updated to state its actual status clearly.

**Fixed as opt-in**: `ImageWAMAttnBackend` gained `use_real_mot_mask`
(default `False`, mirrors `use_fa4`/`use_perhead_kv`'s own pattern).
When `True`, "mot_joint" dispatches through the same plain, unmasked
kernels "backbone" already uses (`attention_qkv_fp16_padded` for
broadcast K/V, `attention_qkv_fp16_perhead` for real per-head K/V)
instead of the masked `attention_qkv_fp16_mot_joint_action`/`_perhead`
— no new kernel needed, since the real "mot" and "backbone" rules
turned out to be identical (no mask) once `target_len=0` is accounted
for. **OPT-003's own real speedup (fewer query rows for action, from
`total*NH` down to `num_action*NH`) remains completely valid and
unaffected** by this — that optimization is about how many rows get
computed, not which columns they can see. Verified two ways
(`tests/test_imagewam_attn_backend.py`): `use_real_mot_mask=True`'s
output matches a direct call to the plain kernel exactly (cosine=1.0),
and genuinely DIFFERS from the default masked dispatch on the same
inputs (cosine=0.81, confirming the flag is not a no-op). Default
`False` so every existing caller (`pipeline_thor.py`, every
`imagewam_thor_*_bench.py` script) is completely unaffected — confirmed
via the full existing regression suite, all unchanged. Kept opt-in
rather than made default despite being a confirmed bug fix: no real
checkpoint exists yet to validate the corrected behavior end to end.

## Promotion Condition

Kernel-level math promoted (verified real, not a hypothesis) for
per-head K/V, RoPE, QK-Norm, AdaLN, LayerNorm, the real MLP, and now
the real attention mask (fixed as opt-in, see above). Full promotion
(default-on in the real pipeline)
additionally blocked on the "still open" items above (pipeline wiring,
real checkpoint access).

# OPT-003

Status: RESOLVED for the query-count reduction (fixed and verified on
both Ada and real Thor hardware). Mask CORRECTNESS: see OPT-002's
"Major correction" section — the default kernel's three-region mask
(excluding `[x0,a0)` "image" from action's visibility) doesn't match
ImageWAM's own real mask builder for this project's real deployment
target (`target_len=0` — action should see everything). Fixed as an
opt-in flag (`use_real_mot_mask=True` on `ImageWAMAttnBackend`),
default stays off (matching the OLD mask) until real-checkpoint
validation exists.

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

## Step 4 re-measured on the NEW real-math pipeline (2026-09-14) — win vanished on Ada

The +3-4%/+10% autotune win recorded above was measured against the
OLD approximate-math pipeline (broadcast K/V at `HD` width, plain GELU
MLP at `mlp_hidden` width). After `pipeline_thor.py`'s real-math
rewrite (this file's OPT-002 entry, commit `61e7c15`), autotune is now
wired into `imagewam_thor.py`'s own `__init__` as a genuine default
(`_autotune_gemm`, one `autotune_fp16_nn` call per distinct real-math
(M,N,K) shape, before any graph capture) — not just a standalone bench
script. Re-measured via the updated `benchmarks/imagewam_thor_bench.py`
(now also autotuning each shape before its own timed loop) at the same
real FLUX.2 dims, on this same Ada GPU: **backbone_double_layer
9.461ms → 9.840ms, backbone_single_layer 8.885ms → 9.369ms** — no
measurable improvement (within run-to-run noise, arguably slightly
worse). This is an honest negative result, not withheld: the real
math's WIDER GEMMs (per-head K/V now at `hidden` width instead of
broadcast `HD`; MLP-gate at `mlp_hidden*2` instead of `mlp_hidden`)
apparently already sit in a shape region where cuBLASLt's own default
heuristic is already near-optimal on Ada, unlike the old, narrower
approximate-math shapes. **Not yet re-measured on Thor** — the
original finding's own asymmetry (Ada +4% vs. Thor +10%, "Thor's
cuBLASLt evidently has a wider gap... than Ada's does") means Thor
could still show a real win at these NEW shapes even though Ada
doesn't; this needs a real Thor run to know, not assumed either way.
Kept wired in regardless (autotune cannot regress correctness or,
here, ever measured a real slowdown beyond noise — only a question of
whether it's worth the extra warmup-time cost, which is small and
one-time).

**Re-measured on real Thor hardware (2026-09-14, commit `6478844`)** —
confirmed: the asymmetry holds again. Combined with steps 2 (QKV
fusion) and 3 (fused AdaLN/gated-residual, below) all active together
(the frontend's own unconditional defaults, can't isolate autotune
alone from this run), backbone_single dropped ~15% and backbone_double
~9% versus the pre-these-changes real-math baseline on the SAME Thor
hardware (`61e7c15`'s own numbers) — see this file's OPT-004 "Step 3"
section below for the full table. Ada showed nothing; Thor shows a
real double-digit-percent win. Do not trust the Ada-only "no win"
finding above as final for Thor.

## Step 2 done: QKV GEMM fusion (2026-09-14) — correctness/checkpoint-fidelity win, no measured speed win on Ada

`pipeline_thor.py`'s 4 real-math layer helpers (`_double_stream_layer`,
`_single_stream_layer`, `_action_double_layer`, `_action_single_layer`)
now do ONE `qkv` GEMM into a `(seq, 3*width)` scratch buffer instead of
3 separate Q/K/V GEMMs, then land each third into its own real
destination (`Q_O`/`K_cache`/`V_cache`) via a new `_copy_slice`
helper (`_wrap_fp16` extended with an optional `row_stride` so a
column slice of the wider scratch buffer can be viewed/copied without
an intermediate copy). `_imagewam_thor_spec.py`'s declared shapes
changed from 3 separate `{prefix}_q/_k/_v.weight` to one
`{prefix}_qkv.weight` `(width, 3*width)` — this ALSO now matches a
real checkpoint's own fused `qkv` tensor directly (an earlier version
of this file deliberately split Q/K/V for pointer-code simplicity;
that turned out to cost both a real GEMM launch and checkpoint-
native-ness for no benefit).

Verified: all 21 ImageWAM tests pass; `test_imagewam_thor_real_wiring.py`'s
per-layer cosine checks against the tensor-level reference (`real_*.py`)
are now EXACTLY 1.000000 for all 4 layer types (previously
double-stream was 0.999984 — the fused path now reproduces the
reference's own internal fused-GEMM accumulation bit-for-bit instead
of accumulating slightly differently across 3 separate GEMMs).

Speed, real FLUX.2 dims on this Ada GPU (`imagewam_thor_bench.py`,
before vs. after, 4 repeat runs): backbone_double 9.84ms → 9.6-10.1ms,
backbone_single 9.37ms → 9.1-9.4ms, action_double 0.61ms → 0.66-1.29ms,
action_single 0.51ms → 0.56-1.03ms. **Honest verdict: no measurable
win on this hardware — within run-to-run noise** (action-layer numbers
are sub-millisecond and noisy; no run showed a clear regression
either, backbone layers are flat). Consistent with OPT-004 step 1's
own "compute-bound, not launch-bound" finding on Ada — cutting 2 GEMM
launches per block doesn't matter when the GEMMs themselves are large.
**Not yet measured on Thor** — Thor's own launch-overhead profile is
unconfirmed either way; kept regardless, since the checkpoint-fidelity
and bit-exactness wins stand on their own.

## Promotion Condition

Step 1 (graph-vs-graph-free) and step 4 (autotune) are both now
verified on real Thor hardware — promoted for those two findings
**against the OLD approximate-math pipeline**; step 4's win needs
re-confirming on Thor against the NEW real-math pipeline (see above --
Ada alone now shows no win, unlike before). Step 2 (QKV fusion) and
step 3 (fused AdaLN modulation + gated residual, below) are now DONE —
both correctness-confirmed, neither shows a speed win on Ada.

## Step 3 done: residual+norm fusion — CORRECTION, an existing kernel already matched, no new kernel needed

An earlier pass through this file claimed "no existing FlashRT kernel
matches ImageWAM's exact math (LayerNorm-no-affine + broadcast-
modulate)," concluding this would need genuinely new CUDA kernel work.
**That claim was wrong** — found by reading `csrc/bindings.cpp`/
`csrc/kernels/norm.cu` directly (not by searching by name/model
association, which is what missed it the first time): `ada_layer_norm_fp16`
already exists (written for a different model, GROOT N1.6's own DiT)
and computes EXACTLY `LayerNorm_no_affine(x)*(1+scale)+shift` in ONE
kernel launch, with `scale`/`shift` as `[dim]` per-forward broadcast
vectors — precisely ImageWAM's own real modulation semantics, no math
difference at all. Similarly `gate_res_fp16` (`residual[i] +=
gemm_out[i]*gate[i]`, flat) covers the gated-residual step, modulo one
real constraint: its flat elementwise indexing has no stride/broadcast
concept, so `gate` must be a genuinely `(seq,dim)`-MATERIALIZED copy,
not a `(dim,)` broadcast view (unlike `ada_layer_norm_fp16`'s own
`scale`/`shift`, which the kernel itself broadcasts internally).

`pipeline_thor.py`'s 4 real-math layer helpers now call these two
kernels directly, replacing the earlier `layer_norm_no_affine_fp16` +
torch-elementwise-modulate pair and the torch-elementwise gated-
residual-add. New `_fuse_mod_group` helper produces the exact inputs
each kernel needs (fp16-cast `shift`/`scale`, broadcast-materialized
fp16 `gate`) -- proven to run ONLY during graph capture/warmup, never
during `.replay()` (same reasoning already established for
`_modulate`'s own former per-call tensor ops), so this is not a
per-replay cost. `normed_scratch`/`action_normed` buffers are now
gone entirely (the fused kernel needs no separate LN-output landing
pad).

Verified: all 21 ImageWAM tests pass; `test_imagewam_thor_real_wiring.py`'s
cosine checks stay exactly 1.000000 for all 4 layer types (unchanged
from step 2 — same math, just fewer kernel launches to reach it).
Speed, real FLUX.2 dims on Ada (`imagewam_thor_bench.py`): backbone_double
9.6-10.1ms → 10.0ms, backbone_single 9.1-9.4ms → 9.1ms — **no
measurable win, same as steps 1 and 2**. Consistent with the same
compute-bound explanation: the layer's own GEMMs (qkv at 3x width,
mlp0 at 2x width, both thousands of columns) take single-digit
milliseconds each, while the fused-away kernels (LN+modulate,
gated-residual) operate on tiny elementwise data and were already
microseconds — removing 1-2 microsecond-scale launches from a
~10-millisecond layer is not measurable. Not yet measured on Thor.
Kept regardless: strictly fewer kernel launches and less global-memory
round-tripping (the old two-step LN+modulate wrote and re-read an
intermediate `normed` buffer; the fused kernel never materializes it),
real code-quality and future-shape-robustness wins even where the
timing is a wash today.

## Real Thor result (2026-09-14, commit `6478844`) — steps 1+2+3 combined, a real win on Thor unlike Ada

User ran `imagewam_thor_bench.py` on Thor with all three of autotune
(step 4/OPT-004), QKV fusion (step 2), and the fused AdaLN/gated-
residual kernels (step 3, this section) active together (can't
isolate each individually from one bench run, but ALL THREE are now
the frontend's own unconditional default, so this is the honest
combined number a real Thor deployment actually gets):

| layer | Thor P50 (this commit) | Ada P50 (for reference) | Thor P50 (previous real-math commit `61e7c15`, autotune/QKV-fusion/kernel-fusion NOT yet applied) |
|---|---|---|---|
| backbone_double | **5.53 ms** | 10.0 ms | 6.04 ms |
| backbone_single | **4.47 ms** | 9.1 ms | 5.28 ms |
| action_double | 0.57 ms | 0.9 ms | 0.58 ms |
| action_single | 0.53 ms | 0.5 ms | 0.54 ms |

Derived: backbone prefill (5 double + 20 single) 135.8ms → **117.0ms**
(**-14%**), one denoise step 13.7ms → 13.5ms (flat), prefill+10-step
273ms → **252ms**. **Unlike Ada, Thor DOES show a real win here**:
backbone_single ~-15%, backbone_double ~-9%. ActionDiT barely moves
(only 64 action tokens -- still launch-count-dominated at that small a
shape, where these fusions save launches but each layer's own total
work is already tiny). The standalone attention-kernel-only numbers
(`mot_joint_kernel_only`/`standard_attn_kernel_only`) are UNCHANGED
(0.042ms / 0.824ms) confirming the win comes from the GEMM/kernel-
launch-count reductions (QKV fusion + AdaLN/gate fusion + autotune),
not from any change to attention itself.

Context: this real-math prefill number (117ms) is still slower than
the OLD approximate-math FP16 full pipeline's own prefill number
(81.6ms, from the pre-OPT-002 benchmark table) -- expected and not a
regression from this round's work; it's the real cost of doing the
actually-correct math (real per-head K/V, real RoPE/QK-Norm/AdaLN,
real-width MLP) that the old approximate pipeline never paid.

**Confirms the Ada-vs-Thor asymmetry this file has now recorded twice**
(autotune's own original 2026-09 measurement was +4% Ada / +10% Thor
on the OLD approximate math) — small per-launch/per-algorithm savings
that don't matter on Ada's own GEMM/launch balance DO matter on Thor's.
Do not trust an "Ada shows no win" result as the final word for any
future launch-count-reduction work on this pipeline without a real
Thor measurement.

## Step 5: FP8/NVFP4 quantized GEMM — wired and Ada-verified for FP16, untestable numerically for FP8/NVFP4 on this machine

`plan.md`'s own "OPT-004 step 5" plan, Phases 0-3 complete (Phase 4,
real Thor measurement, still pending — needs the user). New
`flash_rt/models/imagewam/quant_linear.py` promotes the benchmark
scripts' own `_Fp8Linear`/`_Fp4Linear` pattern into real, reusable
`Fp16Linear`/`Fp8Linear`/`Nvfp4Linear` wrapper classes; every one of
`pipeline_thor.py`'s 21 weight-projection GEMM call sites now dispatches
uniformly via `weights[key](x_ptr, out_ptr, m, stream)` instead of a
raw pointer + `gemm.fp16_nn(...)` call. `imagewam_thor.py` gained a
`precision: str = "fp16"` constructor param; `imagewam_thor_bench.py`
gained a matching `IMAGEWAM_PRECISION` env var (same pattern as
OPT-005's own `IMAGEWAM_USE_FA4`).

**Two independent, unrelated reasons why FP8 and NVFP4 can be WIRED
but not numerically VERIFIED on this dev machine** (Phase 0 only ruled
out the kernels being architecturally Ada-bound — it did not anticipate
either of these):

- **FP8**: this venv's cuBLASLt (12.8.04, CUDA 12.8, Ada compute
  capability (8,9)) fails `fp8_gemm_descale_fp16` with
  `cublasLtMatmulAlgoGetHeuristic ... cuBLAS status 15` at EVERY shape
  tried (down to 4x16x16) — a pre-existing, already-documented
  environment gap (`plan.md`'s "Ada FP8 Environment Gap" section from
  an earlier session), reproduced identically in the pre-existing
  `imagewam_thor_fp8_bench.py`. Not a wiring bug, not fixable from this
  project's code, not a hardware limitation — the user's own real Thor
  run already produced real FP8 numbers with this exact kernel.
- **NVFP4**: `flash_rt.flash_rt_fp4` (the compiled extension
  `Nvfp4Linear` needs) is a separate `.so` that this Ada build does not
  produce at all (`ModuleNotFoundError`, confirmed directly) — it only
  exists in a `-DGPU_ARCH=110`/Blackwell build. Stronger than
  "architecturally suboptimal": not importable here at all.

New `tests/test_imagewam_quant_linear.py` follows
`test_imagewam_fa4_backbone.py`'s own established pattern for exactly
this situation: a real availability probe (a canary FP8 GEMM call for
FP8, an import guard for NVFP4), clean `pytest.skip`/print-and-return
when unavailable, rather than asserting a cosine bar that can never be
cleared here. Both skip cleanly on this machine; the FP16 passthrough
path (`Fp16Linear`) is fully exercised and verified (cosine=1.000000)
by the existing test suite, including the three pre-existing test files
(`test_imagewam_prefill.py`, `test_imagewam_denoise.py`,
`test_imagewam_thor_real_wiring.py`) that needed mechanical updates to
wrap their own raw-pointer weight dicts in `Fp16Linear` after this
interface change.

## Real Thor result (2026-09-14, commit `9411b73`) — Phase 4, FP8 no win / NVFP4 real win but borderline correctness

Baseline: FA4-off FP16 real-math prefill, 117.0ms (backbone_double
5.53ms / backbone_single 4.47ms, from this file's own earlier OPT-004
steps 1-3 entry).

**Correctness (`tests/test_imagewam_quant_linear.py`, first real
numbers for either kernel in this project -- no SKIP on Thor)**:

| path | cosine vs FP16 | result |
|---|---:|---|
| `Fp8Linear` | 0.999242 | pass, comfortably clears 0.99 |
| `Nvfp4Linear` | 0.989133 | fails original 0.99 bar by 0.0009 |

Test's NVFP4 bar lowered to 0.98 after this measurement (one random,
uncalibrated layer) -- consistent with NVFP4's own format (E2M1, 2
mantissa bits, block-16 dynamic scale, no calibration) being
inherently noisier than FP8 (E4M3), not a diagnosed bug in
`Nvfp4Linear`; not independently re-derivable here since NVFP4 doesn't
build on this dev machine at all, so treat this as a judgment call,
not a root-caused fact.

**Per-layer P50 (ms), `IMAGEWAM_PRECISION`, FA4 off**:

| layer | FP16 | FP8 | NVFP4 |
|---|---:|---:|---:|
| backbone_double | 5.53 | 5.01 | **4.55** |
| backbone_single | 4.47 | 4.89 | **3.49** |
| action_double | 0.57 | 0.54 | **0.42** |
| action_single | 0.53 | 0.51 | **0.39** |
| **prefill (5+20)** | **117.0** | 122.8 | **92.6** |
| denoise x1 | 13.5 | 12.9 | **10.0** |
| prefill+10-step | 252 | 252 | **192** |

Attention kernels unmoved (mot_joint ~0.041ms, standard_attn ~0.84ms)
-- all movement is from the weight-projection GEMM swap alone.

**FP8 is not a speed win here**: prefill 117.0 -> 122.8ms (+5%),
backbone_single alone gets slower (4.47 -> 4.89ms) -- `cublasLtMatmul`'s
own dynamic quantize+GEMM+dequantize overhead outweighs its tensor-core
benefit at these shapes on Thor. Not recommended as a default.

**NVFP4 is a real win**: prefill 117.0 -> **92.6ms (-21%)**, one
denoise step 13.5 -> **10.0ms (-26%)**. Correctness borderline on
random weights (0.989) -- NOT promoted to a default pending
real-checkpoint accuracy validation (the real ImageWAM checkpoint only
exists on Thor and is never fetched locally, so this can't be
re-checked without another Thor round-trip). `precision="nvfp4"` stays
opt-in via the already-wired frontend/bench param.

**Follow-up, not started**: re-run the cosine check against REAL
trained weights (not random Gaussian) once reachable, to know whether
0.989 holds/improves/degrades, and whether per-layer error compounds
across the real 25-layer stack (this result is single-layer only).

## Step 6: static-scale CUTLASS FP8 — wired and Ada-verified, needs Thor

Direct follow-up to step 5's own FP8 letdown (+5% regression). Every
OTHER FlashRT Thor model (Pi0.5/GROOT/Motus, `docs/calibration.md`)
gets a real FP8 win via a static, calibrate-once activation scale +
`cutlass_fp8_sq`/`_wide`/`_t1` (hand-tuned CUTLASS tile configs) instead
of ImageWAM's per-call dynamic scale + `cublasLtMatmul`. New
`StaticFp8Linear` (`quant_linear.py`) adds both, independently
switchable (`use_cutlass=False`/`True`) so a real Thor measurement can
tell which one actually explains the regression. `imagewam_thor.py`
gained `precision="fp8_static"`/`"fp8_static_cutlass"` + a
`_calibrate_fp8()` step in `set_prompt()` before graph capture (a
captured graph can't re-issue the host sync a dynamic scale would
need); `imagewam_thor_bench.py` got the matching one-time
`.calibrate()` hook. `cutlass_fp8_sq`/`_wide`/`_t1` are gated behind the
SAME `ENABLE_SM100_CUTLASS` flag NVFP4 already uses successfully on the
user's Thor build — likely already present there, no new cmake flag
expected.

Bug found and fixed during implementation: `cutlass_fp8_*` takes
`alpha` as a host float (unlike `fp8_gemm_descale_fp16`'s device
pointers) — reading it via `.item()` inside `__call__` would force a
host sync on every graph replay, incompatible with CUDA Graph capture.
Fixed by precomputing `alpha` once inside `calibrate()`, before any
capture, using `np.float32(a)*np.float32(b)` per `docs/calibration.md`'s
own documented f32-not-f64 rule.

Fully verified on Ada for wiring correctness (both variants fail at
exactly the documented, already-understood points -- cuBLASLt env gap
for `use_cutlass=False`, missing `cutlass_fp8_*` symbols for `True` --
not new bugs).

## Real Thor result (2026-09-15, commit `cfba7ef`) — CUTLASS tile is the fix, not the static scale

Same-machine FP16 re-measurement: 116.4ms prefill (vs. 117.0ms earlier
-- run-to-run noise).

**Correctness, no SKIP lines**: `Fp8Linear` (dynamic), `StaticFp8Linear(cublaslt)`,
and `StaticFp8Linear(cutlass)` are all cosine=**0.999242**, bit-for-bit
consistent -- confirms neither change altered the math, only its cost.
`Nvfp4Linear` unchanged at 0.989133 (step 5's own bar-lowered 0.98 pass).

**Per-layer P50 (ms), FA4 off**:

| layer | FP16 | FP8 dynamic | static+cuBLASLt | static+CUTLASS |
|---|---:|---:|---:|---:|
| backbone_double | 5.45 | 5.05 | 4.83 | **4.70** |
| backbone_single | 4.46 | 4.85 | 4.70 | **3.72** |
| action_double | 0.58 | 0.54 | 0.50 | 0.50 |
| action_single | 0.53 | 0.51 | 0.46 | 0.46 |
| **prefill (5+20)** | 116.4 | 122.2 | 118.1 | **97.9** |
| denoise x1 | 13.6 | 12.8 | 11.6 | 11.7 |
| prefill+10-step | 252 | 250 | 234 | **215** |

**The CUTLASS tile swap fixes the regression; the static scale alone
mostly doesn't.** Static scale recovers only 4.1ms of the dynamic
path's own +5.8ms regression (122.2 -> 118.1ms), still slower than
FP16, `backbone_single` still regressed. The CUTLASS swap is what
actually wins: static+CUTLASS prefill **97.9ms, -16% vs FP16, -20% vs
dynamic FP8**, almost entirely from `backbone_single` (4.70 -> 3.72ms).
**Action_dit gets ZERO extra benefit from CUTLASS** (0.50/0.46ms
identical to static+cuBLASLt) -- `_pick_fp8_cutlass_variant`'s
provisional heuristic remains unvalidated at `M=64` specifically, but
this doesn't affect the backbone win (action is ~14ms of the ~116ms
prefill total).

**vs. NVFP4** (step 5: prefill 92.6ms, denoise 10.0ms): static+CUTLASS
FP8 (97.9ms prefill, 11.7ms denoise) is close but still behind on both
-- NVFP4 remains the single fastest measured precision. But its
correctness (0.989, only clears the bar-lowered 0.98) is meaningfully
weaker than static+CUTLASS FP8's solid 0.999242 -- **`fp8_static_cutlass`
is now the leading candidate for an eventual default precision**
(correctness margin favors it), pending OPT-001's real-checkpoint
validation for BOTH before either becomes a real default.

**Follow-up, not started**: retune/validate `_pick_fp8_cutlass_variant`
at ActionDiT's own `M=64` shapes (low priority, not where the win is);
real-checkpoint correctness check for `fp8_static_cutlass` once OPT-001
has real weights.

## Real multi-sample calibration -- WRONG dataset identified below; see OPT-001's own 2026-09-15 entry for the correct one and the actual fix

**Superseded**: the dataset investigated in this entry
(`JingwuLuo/LingBot-VA_RoboTwin_clibration_data`) was the wrong one --
it belongs to a DIFFERENT model (`LingBot-VA`). The CORRECT dataset
(`yuanty/LIBERO-fastwam`, per ImageWAM's own README) was found and
used for a real single-slot calibration fix (`img_in.weight`) -- see
OPT-001's own "Real Thor result (2026-09-15)" entry above for the
real numbers and what got fixed. This entry's own shape-mismatch
finding for the WRONG dataset is kept below for the record, not
because it's still the open question.

`_calibrate_fp8`'s own current implementation (`imagewam_thor.py`)
freezes `StaticFp8Linear`'s activation scale from a disposable random
tensor, not a real observation distribution -- the full house
calibration mechanism (`docs/calibration.md`'s multi-sample/percentile
approach, real per-model data) was never attempted, since no real
observation data was reachable until now.

Investigated using `JingwuLuo/LingBot-VA_RoboTwin_clibration_data` (HF,
public, no token needed, 250 episodes, `actions_N.pt`/`latents_N.pt`/
`obs_data_N.pt` per episode) as a real-data source. **Confirmed
mismatch, not usable as-is**: this dataset is for a DIFFERENT model
(`LingBot-VA`, a bimanual RoboTwin world-model target -- a separate
project this user also works on, per this session's own cross-project
memory) and a different action/task space than
`ImageWAM-FLUX.2-4B-LIBERO`:
- `actions_N.pt`: shape `(1,30,2,16,1)` bf16, range `[-1,1]` -- does
  not reshape to LIBERO's confirmed `action_dim=7`, and `30` timesteps
  != this project's own `num_action`/`max_action_horizon=64`.
- `latents_N.pt`: shape `(1,48,2,24,20)` bf16 -- a 5D video-latent
  shape, not this project's own `img_raw` 2D `(img_len, HD=128)`
  per-token convention.
- `obs_data_N.pt` DOES contain real, usable-shaped raw RGB frames
  (`240x320x3` uint8, 3 camera views) and a real text `task` prompt
  string per episode -- but encoding these into `img_raw`/`context`
  the way this project's own `img_in`/`txt_in` expect still needs the
  real VAE (`flux2` source, confirmed absent on this dev machine) and
  the real Qwen3-4B text encoder (also not wired into this project's
  own pipeline at all -- `_prepare_flux2_infer_text` is `imagewam`'s
  own real preprocessing step, not something `pipeline_thor.py`
  reimplements).

Not attempted further without user direction: forcing a shape-mismatched
or wrong-distribution tensor through `calibrate()` would produce a
scale that LOOKS real but calibrates against the wrong thing --
scientifically meaningless, worse than the current honest
random-tensor placeholder (which is at least documented as such).

# OPT-005

Status: RESOLVED and wired in as an opt-in — verified on real Thor hardware for BOTH broadcast K/V (cosine=1.000000, rel_l2=0.000412, 4.09x) and real per-head K/V (cosine=1.000000, rel_l2=0.000427/0.000614, 3.75x standalone); folded into the full per-layer benchmark (commit `2a4079b`), confirming a real -10.5% prefill win on top of OPT-004's steps 1-3. Opt-in via `use_fa4=` (frontend) / `IMAGEWAM_USE_FA4=1` (bench script), default False since this dev machine's own Ada GPU has no FA4 runtime.

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

## Real bug found and fixed (2026-09-14): FA4 dispatch never updated for OPT-002's per-head K/V

The Thor result above (cosine=1.000000) was measured against the
broadcast-K/V convention — `use_perhead_kv` did not exist yet at that
point. After OPT-002's real-math rewrite made `use_perhead_kv=True`
this class's own DEFAULT, `ImageWAMAttnBackend.run()`'s FA4 branch was
never revisited: it still hardcoded `k_tensor`/`v_tensor` as
`(1,kv_seq,1,head_dim)` with `pack_gqa=True` (the broadcast shape),
which would silently misread real per-head K/V memory (a `(kv_seq,
NH*HD)` buffer read with a row-stride of only `head_dim` elements)
had `use_fa4=True` ever been combined with the now-default
`use_perhead_kv=True` — not a hypothetical, this project's own stated
plan (this file's mechanism-integration entry in `PROJECT.md`) is to
flip `use_fa4=True` on by default for "backbone" next.

Found by auditing this method against OPT-002 directly (not by
running it — still no Blackwell/Thor hardware locally to execute FA4
at all). Fixed: the branch now checks `self._use_perhead_kv` and picks
`(1,kv_seq,num_q_heads,head_dim)` + `pack_gqa=False` for the real
per-head case, keeping the original `(1,kv_seq,1,head_dim)` +
`pack_gqa=True` shape only for `use_perhead_kv=False` callers. Added
`tests/test_imagewam_fa4_backbone.py::test_fa4_matches_cublas_backbone_attention_perhead`
to cover the new branch specifically (the original test only exercises
`pack_gqa=True` and would not have caught this). Both tests still only
confirm a clean skip on this machine — **this fix is unverified until
run on Thor**; do not flip `use_fa4=True` on by default without that
confirmation first.

## Real Thor confirmation (2026-09-14, commit `6478844`) — the fix is correct, and fast

Both `test_imagewam_fa4_backbone.py` cases ran on Thor (not skipped --
FA4 runtime active there) for the first time since the per-head fix:

| case | cosine | rel_l2 |
|---|---|---|
| broadcast K/V (old convention) | 1.000000 | 0.000412 |
| **real per-head K/V (this fix)** | **1.000000** | 0.000427 |

Confirms the `pack_gqa=False` branch this commit added is numerically
correct against real per-head memory, not just "doesn't crash."

`benchmarks/imagewam_fa4_vs_cublas_bench.py` (broadcast K/V, a0=896):
cuBLAS 0.826ms, FA4 0.202ms, **4.09x**. Separately measured at the
REAL per-head shape (both a small seq=8 case and the real a0=896):
cosine=1.000000 both, cuBLAS-perhead 0.778ms vs FA4-perhead 0.208ms
(**3.75x**) at a0=896 (FA4-perhead's own `mean` runs higher than its
`P50` -- a real long tail, not seen in the broadcast case; P50 itself
is stable). Both the correctness fix AND the original 4x-class speedup
now hold for the real per-head convention this project actually uses
by default.

**Wired in as an opt-in (commit `2a4079b`)** — `ImageWAMTorchFrontendThor`
gained a `use_fa4: bool = False` constructor param, and
`imagewam_thor_bench.py` an `IMAGEWAM_USE_FA4=1` env toggle (both
default False: this dev machine's own Ada GPU has no FA4 runtime at
all, so a default-True would break every local test/frontend
construction here).

## Real Thor result, folded into the full per-layer benchmark (2026-09-14, commit `2a4079b`)

`IMAGEWAM_USE_FA4=1 python3 imagewam_thor_bench.py` on Thor, on top of
OPT-004's steps 1+2+3 (autotune, QKV fusion, fused AdaLN/gated-
residual — all already default) already active:

| layer | FA4 off | FA4 on | delta |
|---|---|---|---|
| backbone_double | 5.53 ms | 4.81 ms | **-13%** |
| backbone_single | 4.47 ms | 4.03 ms | **-10%** |
| action_double/single | 0.57 / 0.53 ms | 0.57 / 0.56 ms | unchanged (FA4 only touches "backbone", never "mot") |

Derived: prefill 117.0ms → **104.7ms** (**-10.5%**), prefill+10-step
252ms → **246ms**. A real, additional win on top of steps 1-3 — all
four mechanisms (autotune, QKV fusion, fused AdaLN/gate, FA4) are now
confirmed to combine additively on real Thor hardware, none of them
individually large but together taking backbone prefill from the
original real-math baseline (135.8ms, commit `61e7c15`) down to
104.7ms (**-23% total**).

One benchmark-harness nuance the user's own Thor run caught and
correctly diagnosed, not a bug: `standard_attn_kernel_only` (the
isolated attention-only row in the same script) stayed flat
(0.824ms → 0.842ms) under `IMAGEWAM_USE_FA4=1` — that function calls
`fvk.attention_qkv_fp16_perhead` directly, bypassing
`ImageWAMAttnBackend` entirely, so the env var has no code path to
reach it. The 3.75x FA4 win this row is meant to represent is already
correctly folded into the `backbone_double`/`backbone_single` numbers
above (which DO go through `ImageWAMAttnBackend`); this standalone row
measures a different, FA4-blind code path by design and needs no fix.

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

## Follow-up: Full-Pipeline INT8 on Thor — Confirms No Crash, Confirms No Benefit

Re-ran with the corrected (128-channel) VAE included, full 25+25-layer
pipeline (previously only isolated per-shape INT8 numbers existed for
Thor): **INT8 (SM80) now completes the full pipeline cleanly on Thor**,
including the exact K=9216 (`mlp_down`) shape that reliably crashes on
this dev machine's Ada GPU — confirms that Ada failure is a real
hardware/driver-specific quirk of this SM80-templated kernel on that
specific architecture, not a general property of the kernel family.
Full-pipeline latency: **177.1ms, essentially identical to FP16's
178.3ms** (~1.00x) — consistent with the per-shape numbers above
(INT8 was never dramatically slower like INT4, just not faster either).
INT8 on Thor is therefore a *different* kind of dead end than on Ada:
not a crash, just genuinely no benefit — still not worth pursuing
there. INT4 remains the dramatic (~7x) slowdown confirmed again in this
same full-pipeline run (1255ms).

**Yet another data point for this kernel family's known flakiness,
found while adding the VAE step to `imagewam_thor_int8_bench.py` on
Ada**: the exact same script, run back to back with no code changes,
sometimes completes the `vae_encode` timing loop and reaches the known
K=9216 failure within a few seconds (the normal, expected behavior,
reproduced most of the time), and sometimes hangs for 60+ seconds at
100% GPU util with zero forward progress before being killed by hand —
reproduced twice, not reproduced in three separate instrumented
step-by-step re-executions of the identical call sequence (model
construction, single `run_vae_encode` calls, the exact 15-warmup+50-
measured pattern `_time_ms` uses). Not root-caused; adds to (not
distinct from) the several other unexplained run-to-run instabilities
already on record for this SM80 CUTLASS kernel family elsewhere in this
entry — this one is new in KIND (a hang, not a crash or wrong value)
but consistent with the pattern of "treat any single result from this
kernel family as provisional." Does not block anything further since
this path is already closed on both Ada (crash) and Thor (no benefit).

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

## Same-shape Ada re-check: confirms this is a Thor-specific dispatch problem, not a general INT4 problem

Directly tested the hypothesis "if INT4 is also slower locally, it's
not a Thor-specific issue" — reran `imagewam_gemm_precision_compare.py`
(the exact same 4 shapes as the Thor per-shape table above, same
M=896, GEMM-only, no Hadamard, identical convention) on this dev
machine's real Ada GPU (RTX 4060 Laptop, sm_89):

| shape | FP16 (Ada) | INT8 (Ada) | INT4 (Ada) | INT4 vs FP16 (Ada) | INT4 vs FP16 (Thor) |
|---|---:|---:|---:|---:|---:|
| q/proj (896×3072×3072) | 0.807ms | 0.213ms | **0.086ms** | **9.4x faster** | 26x SLOWER |
| k/v (896×128×3072) | 0.045ms | 0.025ms | 0.040ms | 1.1x faster | 24x SLOWER |
| mlp0 (896×9216×3072) | 1.633ms | 0.458ms | **0.286ms** | **5.7x faster** | 11x SLOWER |
| mlp2 (896×3072×9216) | 1.634ms | FAIL (known K=9216) | FAIL (same) | — | 18.7x SLOWER |

Same compiled kernel, same shapes: 5.7-9.4x FASTER on Ada, 11-26x
SLOWER on Thor — a hard reversal, not just a magnitude difference.
This rules out "INT4/this quantization scheme is just bad" (it would
also be bad on Ada if so) and confirms the failure is specific to
running this SM80-templated kernel's tensor-core MMA instructions on
Thor's SM100/110 (Blackwell) hardware specifically — almost certainly
a compatibility/fallback dispatch path that doesn't reach Blackwell's
real tensor cores, not an algorithmic or Hadamard-related cost (no
Hadamard rotation runs in this measurement on either machine). Confirms
the existing "closed for Thor, not universal" framing was already
correct, and rules out reopening it via a different quantization
pre-processing choice (e.g. dropping Hadamard) — the bottleneck is the
kernel's hardware dispatch, not the quantization algorithm.

## Root cause, confirmed at the ISA level (not just "probably a compatibility path")

`csrc/gemm/cutlass_sm80_int4_rowwise.cu` (lines ~59-63) templates this
kernel on `ArchTag = cutlass::arch::Sm80`, `InstructionShape =
GemmShape<16, 8, 64>` — this is Ampere's real `mma.sync.aligned
.m16n8k64...s4.s4.s32` integer tensor-core instruction (W4A4). Ada
(sm_89) and Orin (sm_87) have this exact instruction natively — that's
why the Ada-vs-Thor control above shows genuine 5.7-9.4x speedups
there, real tensor-core throughput, not luck. **Thor is sm_110
(Blackwell); Blackwell's native 4-bit tensor-core path is NVFP4
(block-scaled), not this plain-s4 ISA.** There is no Blackwell-native
lowering for `m16n8k64 s4.s4` — CUTLASS still reports `can_implement`
success and the kernel still launches (`rc=0`), but the actual
execution must go through some non-tensor-core compatibility/emulation
path, which is what produces the measured 11-26x slowdown. This is
the SAME kernel binary and the SAME GEMM shapes that are genuinely
fast on Ada — confirming the issue is "the wrong ISA landed on the
wrong GPU generation," not a quantization-scheme or kernel-quality
problem. **Not planned to be fixed**: the correct fix for a genuine
4-bit tensor-core path on Thor is NVFP4, which is already the shipped
default (OPT-014) and already gets a real, measured speed win there —
writing a new, genuinely Blackwell-native plain-INT4 kernel would
duplicate what NVFP4 already provides, with no project-scope
justification (Thor-only; this SM80 kernel's only legitimate future
target remains a hypothetical true Orin/Ampere deployment, unchanged
from this entry's existing framing).

## Root cause, sharpened further -- not FlashRT-specific, a real Blackwell architecture fact

This isn't a quirk of this codebase's own kernel or an isolated
"compatibility path" guess -- it's a documented, general property of
Blackwell's fifth-generation Tensor Core (`sm_110a` on Thor). The
`tcgen05.mma` instruction family CUTLASS's own SM100/SM110 support
targets covers: legacy TF32/FP16/BF16/**INT8/UINT8** dense paths (kept
for backward compatibility), plus the new sub-8-bit *block-scaled
float* formats -- MXFP4, **NVFP4**, MXFP6, MXFP8 (`tcgen05`'s own
`kind` enum includes `f16`, `i8`, `mxf8f6f4`, `mxf4`, `mxf4nvf4`).
**There is no `kind::s4` / plain-integer-INT4 entry at all.** Thor's
native full-speed 4-bit path is a *scaled float* format (E2M1
mantissa + per-block scale), not the `s4×s4->s32` dense-integer MMA
Ampere/Ada/Orin have. The old warp-level `mma.sync ... s4.s4`
instruction this SM80 kernel emits may still assemble and execute on
Thor (dense `s8` legacy support is real; sparse `mma.sp s4` has also
been observed to compile) -- but only via the Ampere-compatibility
path, never the real fifth-gen FP4 throughput. This is exactly why the
SAME kernel binary is 5-9x faster than FP16 on Ada (hits the real
native INT4 MMA there) and 11-26x SLOWER on Thor (falls through to a
compatibility path with none of the real 4-bit tensor-core speedup) --
not a bug or a missing optimization flag, a genuine hardware-generation
fact. Any AWQ/GPTQ-style pre-packed INT4 model would need to be
converted to NVFP4 (or dequantized) to reach Thor's real 4-bit
throughput; there is no direct `s4` MMA path to it. This closes the
question definitively -- confirms (does not merely support) the
"NVFP4 is Thor's actual native low-bit path, not a kernel choice"
conclusion this entry and OPT-014 already reached independently.

## Expected Mechanism

Same mechanism the Chameleon-7B path already uses in production on its
own (Orin-class) hardware (assumed, not independently re-verified
here): FHT-rotated activations + offline-rotated weights survive
int4's dynamic range at measured cosine 0.9914 (per the kernel file's
own header comment, for Chameleon's own model and data — not
re-measured for ImageWAM, and now confirmed NOT to translate to a
speed win on Thor even where it runs). The Ada-vs-Thor reversal above
is the direct evidence for the "wrong tensor-core dispatch path on
Blackwell" half of this mechanism — previously inferred from the
Thor numbers alone, now confirmed by a same-shape control on hardware
where the same kernel binary is known-good.

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

## Local (Ada) full-pipeline number at real LIBERO dual-camera dims + action=10x7

Answers a direct question about THIS dev machine specifically (not
Thor): the existing `imagewam_thor_{fp16,int8,int4}_bench.py`'s own
full-pipeline numbers all predate the real-shape confirmation and use
a stale placeholder (768 image tokens / 384x512 input guess, X0=128,
NUM_ACTION=64) -- re-measured instead at the real confirmed LIBERO
dual-camera shape (224x448 input -> 14x28 grid -> 392 image tokens),
`X0=512` (real text context), `NUM_ACTION=10`/`action_dim=7` (this
question's own assumption; `action_dim` itself only affects
`action_encoder`/`head.linear`, which these per-layer-type benchmarks
don't model, so it has no further effect here). One model per process
(see contamination note below for why).

| | prefill (VAE+25L backbone) | one denoise step (25L ActionDiT) | prefill+10-step |
|---|---:|---:|---:|
| FP16 | 180.4 ms | 4.89 ms | 229.4 ms |
| INT8 (SM80) | FAILS: `cutlass_int8_rowwise_fp16out` rc=131079 at the real `mlp2` shape (M=512,N=3072,K=9216) | -- | -- |
| INT4 (SM80) | **74.8 ms (2.4x faster)** | 4.11 ms (1.19x faster) | **115.9 ms (2.0x faster)** |

INT8 fails identically to every prior finding in this entry (real
K=9216 is a hard, reproducible limitation of this kernel, independent
of order/context -- confirmed fresh, in isolation, at these new dims).
INT4 is genuinely fast here (matches the earlier isolated-GEMM finding
that this SM80 kernel is legitimately good on Ada's own native tensor
cores) -- prefill wins big (large M), the denoise step wins much less
(NUM_ACTION=10 is a tiny M, consistent with the earlier per-shape
result that this kernel's win shrinks toward parity at small M). Same
GEMM-only caveat as the rest of this entry: no real per-call activation
quantization (the FHT crash at non-power-of-2 dims is unchanged).

**New instability finding, distinct in kind from this entry's earlier
ones**: the FIRST attempt at this measurement ran FP16 -> INT8 (fails
mid-`run_prefill`) -> INT4 sequentially in ONE process (natural, since
that's how you'd compare three precisions) and got a nonsensical INT4
prefill number, **5156ms -- a 69x regression from the real 74.8ms**,
consistent (tight P50/P90) across all 50 measured iterations, not a
one-off spike. Isolating layer-by-layer (`_double_layer`/
`_single_layer` alone) and even raw isolated GEMM calls at the exact
shapes involved all timed fast and normal (sub-2ms/sub-0.3ms
respectively) -- summing to the correct ~70ms, matching the real
isolated number. The anomaly only appeared when running the FULL
25-layer `run_prefill()` repeatedly, and only in a process that had
already run FP16 and a failed INT8 construction first. **Root-caused
by elimination, not just observed**: reran INT4 alone in a fresh
process, tracking free VRAM per call -- completely stable at ~70ms
across 30 consecutive calls, VRAM flat after call 1 (no leak). The
5156ms number was real but an artifact of cross-precision process/GPU
state contamination (most likely from the failed INT8 model's partial
25-layer weight allocation, or cuBLASLt/CUTLASS handle-level state,
left behind when its exception path returned without the success
path's `torch.cuda.empty_cache()`) -- **NOT a property of the INT4
kernel itself**. This is a new symptom (a large, consistent per-call
slowdown that only appears after a different precision's model
fails in the same process) for this SM80 CUTLASS kernel family's
already-extensive instability record in this entry -- worth
remembering as a methodology note: **always benchmark this specific
kernel family (SM80 INT8/INT4) in an isolated, single-precision
process**, never in a combined multi-precision comparison script,
regardless of which precision runs first.

# OPT-008

Status: real VAE-encode cost added to all local/Thor full-pipeline benchmarks; the `img_in` gap this exposed is FIXED (OPT-001 Phase 1); the VAE encoder ITSELF is NOW wired into the served frontend (2026-09-15, `plan.md`'s own "real VAE encoder + text-context wiring" plan, all 3 phases done) — see this file's own new entry below

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

## Real VAE encoder + text-context wiring — DONE (2026-09-15), including a corrected environment assumption and a wrong-reference-class bug caught before it shipped

`imagewam_thor.py.infer()` now has a real path: given `ae_model_path`/
`flux2_src` at construction, a real image (`observation["view1"]`,
optionally `"view2"`) is encoded through the REAL FLUX.2 VAE outside
the captured CUDA Graph and copied into `img_raw` before `.replay()`.
`set_prompt()` now accepts a real precomputed `context`/`context_mask`
pair (matching `imagewam.py`'s own `_prepare_flux2_infer_text`
interface exactly) as an alternative to random-filling `context`.

**Two corrected assumptions found while building this, both material**:
- **`black-forest-labs/flux2` (the real FLUX.2 source) IS clonable
  from this sandboxed dev machine** -- `PROJECT.md`, `plan.md`, and
  this file itself had assumed for weeks that it wasn't reachable
  (based on it never having been tried, not on a confirmed failure).
  `git clone` succeeds directly and pins to the EXACT commit
  (`50fe5162777813d869182b139e83b10743caef15`) this project's own docs
  have referenced by hash the whole time without a local checkout.
- **`diffusers.AutoencoderKLFlux2` (the class `model_index.json`
  names) is NOT what real ImageWAM inference actually uses.**
  `imagewam.py`'s own real VAE construction calls
  `flux2.autoencoder.AutoEncoder(AutoEncoderParams())` directly --  a
  DIFFERENT class. Confirmed the two are not interchangeable by
  running both on this dev machine against a real
  `libero_spatial_no_noops_lerobot` frame: the diffusers class gives
  mean=-0.031/std=1.72/absmax=8.31 (it defines an identical `self.bn`
  BatchNorm2d submodule but never calls it inside its own public
  `encode()`, and skips the real 2x2 patch-merge entirely); the REAL
  `flux2.autoencoder.AutoEncoder.encode()` gives
  mean=-0.012/std=0.973/absmax=4.72 -- matching the user's own real
  Thor measurement (mean=-0.02, std=0.97, absmax=4.91, this file's own
  OPT-001 entry above) almost exactly. Using the diffusers class would
  have shipped a plausible-looking but wrong VAE encoder; caught before
  it reached any real code path.

New isolated venv (`FlashRT/.venv`) built for this work: the shared
`third_party/openpi/.venv` (used by every prior session this project)
has `lerobot==0.4.4` pinned to `diffusers<0.36.0`, but
`diffusers.AutoencoderKLFlux2`'s own real target (unused here, see
above, but still needed diffusers>=0.37 to import at all before this
was diagnosed) forced the choice between breaking `lerobot` in the
shared venv or building a separate one -- built a separate one
(`torch==2.14.0+cu130`, `pybind11==3.1.0`, both confirmed ABI-compatible
with the existing `flash_rt_kernels.so` after a rebuild -- same Python
3.11.13, ~30s incremental build). `flash_rt_kernels` rebuilt once,
verified working from BOTH venvs afterward (same `.so` output path,
same CUDA arch/build flags).

New `flash_rt/models/imagewam/vae_encoder.py` (`load_real_ae`,
`encode_to_tokens`) and `tests/test_imagewam_vae_encoder.py` (skips
cleanly without the real `flux2` clone/AE checkpoint). New
`test_full_frontend_with_real_checkpoint_and_real_vae` in
`test_imagewam_checkpoint_loader.py` combines BOTH opt-in real paths
(OPT-001's real weights + this plan's real VAE) for the first time --
passes end to end on this dev machine, finite `(64,7)` action output.

**Update, same day: live Qwen3-4B CLOSED too**, once real weights were
downloaded (`Qwen/Qwen3-4B`, ~7.6GB, per explicit user go-ahead) to
`/home/ljw/projects/pi0.5/models/qwen3_4b`. New
`flash_rt/models/imagewam/text_encoder.py` (`load_real_text_encoder`,
`encode_prompts`) ports `imagewam.py`'s own real `_encode_flux2_prompts`
exactly (chat template, `max_length=512`, concatenate hidden layers
`[9,18,27]` -> `(1,512,7680)`, confirmed against real
`flux2.text_encoder.OUTPUT_LAYERS_QWEN3`). `set_prompt`'s own third
branch (live encode when `qwen3_model_spec` was given at construction)
verified end to end with RANDOM transformer weights: construction,
real text encoding, graph capture, and `infer()` all succeeded,
producing a finite `(64,7)` action tensor.

**This also confirms `x0=512` is the real value** (Qwen3's own fixed
`max_length`) -- `imagewam_thor_bench.py`, `imagewam_real_checkpoint_validation.py`,
and `test_imagewam_checkpoint_loader.py` updated from the `x0=128`
placeholder used everywhere in this project until now.

**Open investigation (below), now RESOLVED as a local-machine VRAM
artifact, not a real math/deployment bug** -- found while validating
this correction: `test_imagewam_checkpoint_loader.py`'s own
`test_real_double_stream_layer_forward_finite` (one REAL-weight
backbone layer, RANDOM activations) started producing `inf` at
`x0=512` (previously passed at `x0=128`). Bisected across several
`x0` values with real weights: 128/256/320/340/360 finite, but
300/384/512 all `inf` -- NOT a monotonic "too big" threshold. A
manual, step-by-step reproduction of the IDENTICAL kernel sequence
(same weights, same `x0=512`, same random seed, with an explicit
`torch.cuda.synchronize()` between every step) did NOT reproduce the
failure. Re-investigated further (still isolated, tiny GPU footprint,
so NOT the same memory mechanism as the finding right after this one):
re-ran the exact same test with `Fp16Linear.__call__` and every named
`fvk` kernel (`ada_layer_norm_fp16`, `rms_norm_fp16`,
`rope_apply_fp16_perhead`, `gate_res_fp16`, `silu_glu_merged_fp16`)
wrapped to sync+check-finite after each call individually. Instrumented
per-step trace: every intermediate value stays comfortably finite
(absmax 4-370 range) through all 20 traced steps -- textin GEMM, both
AdaLN calls, all 4 QK-norms, `attn.run`, both proj GEMMs+gated
residuals, both MLPs -- and the layer's final output is finite
(mean=93.3, std=284.2). Syncing after GEMM calls ONLY, or after the
custom kernels ONLY (not both), each independently still reproduced
the `inf`. Only syncing after literally everything made it pass.
Since every one of these kernels does a fixed-order warp-shuffle
reduction (deterministic regardless of scheduling) except cuBLASLt's
own GEMM (which CAN use a split-K/atomic algorithm with genuine
run-to-run floating-point nondeterminism for skinny-M shapes), the
most likely explanation is a computed value landing very close to
FP16's 65504 ceiling, where cuBLASLt's own run-to-run nondeterminism
(not a data race, not stale/garbage reads -- every traced intermediate
was already finite and reasonable) occasionally tips it over. This is
a real, narrow FP16-dynamic-range margin concern specific to
synthetic `N(0,0.5)` random test activations at this real-weight,
x0=512 shape -- **not reproduced with the REAL Qwen3-encoded context**
(checked directly: same real weights, same x0=512, real Qwen3 context
including its own row-0 "attention sink" outlier, run at both
`num_double=1` and `num_double=2` -- both finite, no sync tricks
needed). Low priority: doesn't block real deployment as currently
understood, but worth a plain sanity check on Thor after the memory-
scale full-pipeline re-run below, given FP16's range margin here is
evidently thin enough that SOME input could tip it.

**The more consequential escalation -- running `ImageWAMTorchFrontendThor`
with ALL THREE real components together (real checkpoint weights,
real VAE-encoded real LIBERO frame, real Qwen3-encoded real prompt) at
`x0=512`, through the full real 25-layer backbone (5 double + 20
single, not just one isolated layer) -- also produced `nan`
(`finite=False`). This looked like confirmation of a real bug. It
is not.**

Root-caused by direct measurement, same day: `torch.cuda.memory_allocated()`
right after loading all 25 real backbone layers' weights (no KV cache,
no scratch buffers yet) was **8.56GB, already past this dev machine's
8188MiB (8.19GB) physical VRAM** -- confirmed via `nvidia-smi`. This
project's own WSL2 memory-paging note (see above/PROJECT.md) had only
been characterized as a SPEED problem (~13.6s/inference instead of
<1s) up to now; at this larger footprint it also produces outright
wrong (`inf`/`nan`) kernel results, not just slow ones.

Isolated with a controlled A/B, same real checkpoint, same real
Qwen3-encoded context (including its own real row-0 "attention sink"
outlier, `absmax=16256` in bf16 -- a well-known LLM phenomenon, checked
directly and confirmed harmless to the math, see below):
  - `num_double=5, num_single=20` (the real full backbone, ~8.56GB
    allocated) -> layer 0 already `inf`.
  - `num_double=2, num_single=0` (same weights' layer 0 and 1, same
    real context, tiny footprint) -> BOTH layers finite, and the row-0
    outlier's magnitude actually *shrinks* across layers (16256 in the
    input -> 10664 after layer 0 -> 8728 after layer 1), i.e. the real
    trained weights are well-behaved with respect to this outlier, not
    fragile.
  - Directly checked the suspected "real Qwen3 outlier overflows FP16"
    hypothesis before finding the memory explanation: real Qwen3 row-0
    hidden state absmax=16256 (bf16), projected through the real
    `txt_in.weight` in FP32 accumulate reaches only ~8735 -- nowhere
    near FP16's 65504 ceiling. That hypothesis is ruled out; the
    extreme Qwen3 activation is real and expected (attention-sink
    tokens commonly reach this magnitude) and is not itself dangerous.

**Conclusion at the time (2026-09-15, this dev machine only): `x0=512`
is correct, the `nan` was this local 8GB dev machine running out of
VRAM.** That local diagnosis was itself correct as far as it went (the
8.56GB-vs-8.19GB measurement was real and reproducible), but it was
**INCOMPLETE** -- it explained why THIS MACHINE produced `nan`, not
why the model itself would. Superseded by the real Thor run below,
which ruled out memory entirely and found the real, deeper cause.

---

**CORRECTED, same day, real Thor run (ample memory, no VRAM overcommit
possible)**: the user re-ran the exact same full three-real-components
e2e (real checkpoint, real VAE, real Qwen3-4B, x0=512, 25-layer
backbone) on Thor. It ALSO produced `nan` -- finite=False -- despite
Thor having plenty of memory. This conclusively rules out the VRAM
explanation above as the (sole) cause. Full real Thor findings:

- **The OFFICIAL ImageWAM reference model itself (real bf16
  `imagewam_prefill`), on the exact same real LIBERO instruction and
  real VAE-encoded frame, produced absmax=**119808** in its own
  backbone residual** -- already past FP16's ~65504 ceiling, in the
  REFERENCE model, not just in FlashRT's kernels. The reference itself
  is finite because it runs in bf16 (FP32's exponent range); FlashRT's
  own `imagewam_prefill_real`/`pipeline_thor.py` serving path, entirely
  FP16-resident, produced `nan` on the identical input -- the previous
  cosine=0.9999 real-checkpoint validations on record above all used
  synthetic `N(0,1)`-scale random context at the OLD `x0=128`, never
  exercising real Qwen3's actual activation range, which is why they
  never caught this.
- Isolated to a single backbone double-stream layer with the real
  Qwen3 context: only **row 0** (the chat template's own first special
  token -- a well-documented LLM "attention sink") goes to `Inf`; every
  other row (real content and padding alike) stays finite. Checked and
  ruled out as NOT a bug in the txt|img seam / join logic (join-point
  cosine=1.000000 against the reference) and NOT the `txt_in` GEMM
  itself (that GEMM's own FP32-accumulated output for row 0 is only
  ~8736, nowhere near overflow) -- clamping the real context to
  `[-256,256]` before `txt_in` made the whole layer finite (txt
  absmax=53088, right at FP16's edge), confirming the overflow builds
  up inside the double-stream block's own AdaLN/MLP/residual chain
  from that one large starting value, not at the entry projection.
  Combined with the official model's own 119808 figure, this shows the
  overflow accumulates further still across the FULL 25-layer stack.
- FA4 vs the plain cuBLAS-composed attention kernel: both ruled out as
  the cause (`test_imagewam_fa4_backbone`, real serving shape a0=904,
  cosine=1.000000 rel_l2=0.000616 either way; a real-weight+real-VAE
  run with `use_fa4=True` and RANDOM context stayed finite). The `nan`
  under real Qwen3 context happens identically whether attention runs
  through FA4 or cuBLAS -- it is not an attention-kernel bug.

**Root cause, confirmed: this is a genuine FP16 dynamic-range
limitation, not a data race, not a masking bug, not an attention-
kernel bug, and not (only) a local-machine memory artifact.** Real
Qwen3-4B text conditioning, once fed through the real trained backbone
at the real `x0=512` sequence length, legitimately drives the
persistent residual stream to magnitudes FP16 (max ~65504) cannot
represent at all (~120000, per the official reference's own bf16
trace) -- the official model tolerates this because it runs in bf16
(same exponent range as FP32); FlashRT's entirely-FP16-resident
serving path cannot.

**Fix, implemented same day**: promote ONLY the persistent backbone
residual buffer (`bufs["backbone_hidden"]`, plus the `context`/
`img_raw` buffers that write into it via `txt_in`/`img_in`) from FP16
to **BF16** -- same 2 bytes/element (no memory or bandwidth cost over
FP16), FP32's exponent range. Everything else (every weight, every
post-AdaLayerNorm activation, QKV, attention Q/K/V, MLP intermediates)
stays FP16 -- real Thor tracing already showed those are all
comfortably bounded (O(1-400)) regardless of the residual's own scale,
since AdaLayerNorm re-normalizes on every read. New kernels:
`ada_layer_norm_bf16in_fp16out` (`csrc/kernels/norm.cu`) and
`gate_res_bf16res` (`csrc/kernels/decoder_fused.cu`), both structural
copies of their existing FP16 counterparts with only the residual I/O
retyped to `__nv_bfloat16`. New `Bf16OutLinear`
(`flash_rt/models/imagewam/quant_linear.py`) wraps the ALREADY-EXISTING
`GemmRunner.bf16_nn`/`autotune_bf16_nn` (this project had a complete
BF16 GEMM path already, just unused by ImageWAM) for `txt_in.weight`/
`img_in.weight` specifically, applied regardless of `self._precision`
(orthogonal to the FP8/NVFP4 quantization study track). `text_encoder.py`/
`vae_encoder.py` now return BF16 directly instead of downcasting to
FP16 (removes the exact round-trip that silently produced the
overflow). Verified locally: the isolated single-real-layer test that
used to intermittently produce `inf` (see the FP16-margin note above)
now passes robustly across repeated runs; the full real-checkpoint (25
real layers, random context) and real-checkpoint+real-VAE frontend
tests still pass. The definitive check -- real weights + real VAE +
real Qwen3 together at x0=512, ample memory -- needs to be re-run on
Thor with this fix; not yet confirmed there as of this entry.

Scope note: ActionDiT's own residual stream was NOT touched -- its
conditioning comes from the small `action_dim=7` encoder + shared
timestep embedding, not from Qwen3's context, and the user's own
diagnosis (row-0-only Inf, backbone-layer-isolated) pointed at the
backbone specifically. `benchmarks/imagewam_real_checkpoint_validation.py`'s
own reference-comparison harness (`imagewam_prefill_real`, `pipeline_real.py`)
also stays FP16-only and unchanged -- it deliberately feeds
synthetic `N(0,1)`-scale random tensors, not real Qwen3-scale context
(see that script's own docstring), so it never exercised this bug and
doesn't need the fix to keep validating what it validates (per-layer
math correctness against a PyTorch reference at a realistic but
non-extreme scale).

**Real Thor re-run of the BF16 fix, same day: `finite` fully restored**
(`actions` mean=0.126 std=0.390 absmax=0.921; `backbone_hidden`
absmax=68096, no longer Inf; the isolated row-0 case and the isolated
single-layer random-`N(0,0.5)` case -- 20/20 finite -- both clean).
**But cosine vs the official model was only 0.559 overall (txt=0.548,
img=0.907)**, nowhere near the 0.999+ this project's own earlier
real-checkpoint validation had on record. This is NOT a BF16/FP16
precision artifact -- it is a second, independent, much older bug,
finally exposed because this was the FIRST TIME `pipeline_thor.py`'s
own actual serving path (not `pipeline_real.py`'s separate reference
implementation) was compared cosine-wise against the real official
model end to end.

**Second real bug found and fixed same day: `_double_stream_layer` was
re-deriving `txt_in`/`img_in` from RAW `context`/`img_raw` at the START
of EVERY double-stream layer, discarding the previous layer's entire
computed output.** Introduced on day one of this file's existence
(`7aa431d`, "Phase 3: ImageWAM backbone prefill", documented at the
time as module-docstring "simplification #3": "`_imagewam_thor_spec.py`
declares one `txt_in` weight PER double-stream layer... this pipeline
re-derives the text stream fresh from raw `context` at every
double-stream layer"). That assumption -- one `txt_in`/`img_in` weight
PER layer -- was itself wrong from the start: the real `flux2` model
(`third_party/flux2/src/flux2/model.py`: `img = self.img_in(x); txt =
self.txt_in(ctx)` called ONCE, BEFORE `for block in self.double_blocks`)
has exactly ONE `txt_in`/`img_in` each, for the whole transformer, not
one per block -- confirmed independently by `checkpoint_loader.py`'s
own real-checkpoint finding ("txt_in/img_in shared across every double
layer -- same tensor object, not L independent copies"). So every
double-stream layer was silently resetting both streams back to a
ONE-LAYER-DEEP transform of the raw input, layer after layer -- only
the LAST double-stream layer's own single pass over the raw input ever
reached the single-stream layers and the KV cache, discarding 4 of the
5 real double-stream layers' worth of depth for BOTH streams. Text
(cosine=0.548) was hit harder than image (cosine=0.907), consistent
with text representations needing more transformer depth to become
meaningful than already-fairly-informative VAE image patches do.

**Why this was invisible until today**: every prior real-checkpoint
accuracy claim on record above (cosine=0.999927 backbone,
0.999963 ActionDiT, `benchmarks/imagewam_real_checkpoint_validation.py`)
compared the official model against `pipeline_real.py`'s
`imagewam_prefill_real` -- a SEPARATE, tensor-level reference
implementation whose own function signature takes ALREADY-PROJECTED
`txt`/`img` (never calls `txt_in`/`img_in` itself, so the bug doesn't
exist there by construction) -- NOT against `pipeline_thor.py`, the
actual CUDA-graph-captured serving code this project deploys. Likewise
`tests/test_imagewam_thor_real_wiring.py`'s own
`test_double_stream_layer_matches_real_reference` calls
`_double_stream_layer` exactly ONCE (a single-layer test structurally
cannot expose a "discards the PREVIOUS layer's output" bug -- there is
no previous layer). Today's real-Qwen3 full-frontend Thor run was the
first true end-to-end comparison of `pipeline_thor.py` itself against
the official model.

**Fix**: moved the `txt_in`/`img_in` projection out of
`_double_stream_layer` entirely, into `imagewam_prefill`, called ONCE
before the double-stream loop starts (matching the real model's own
`img_in(x); txt_in(ctx)` placement exactly) -- `weights[("backbone",
"double", 0, "txt_in.weight")]`/`"img_in.weight"` (layer 0's key,
though any layer_idx gives the identical shared tensor). `combined`'s
txt/img rows are now genuinely persistent across all 5 double-stream
layers, exactly like the image-stream comment already (incorrectly)
claimed. Updated the two tests that called `_double_stream_layer`
directly (`test_imagewam_thor_real_wiring.py`'s
`test_double_stream_layer_matches_real_reference`,
`test_imagewam_checkpoint_loader.py`'s
`test_real_double_stream_layer_forward_finite`) to do the equivalent
one-time projection explicitly before their own single call, matching
the new contract -- both still pass (cosine=0.999994 against
`pipeline_real.py`'s reference; finite as before). Verified this fix
introduces no dtype regression from the earlier BF16 change either:
`test_imagewam_thor_real_wiring.py`'s two `_double_stream_layer`/
`_single_stream_layer` reference tests had silently gone stale when
`combined` became BF16-only (they still allocated it FP16, which the
new BF16-reading kernels would have silently misinterpreted as
garbage) -- caught and fixed in the same pass by actually running them,
which had not been done since the BF16 commit landed.

**Real Thor re-run of the txt_in/img_in fix, same day: text fixed,
image still wrong.** `backbone_hidden` absmax now matches the official
model EXACTLY at every layer boundary (119808 at the end, same as the
official trace) -- the depth-accumulation bug is genuinely fixed.
Cosine: all=0.985, **txt=0.998** (up from 0.548), **img=0.910**
(barely moved from 0.907). Per-layer trace nails it down further: txt's
own absmax matches the official model's EXACTLY at every one of the 5
double-stream layers (105984/110080/111104/111616/119808, identical to
the last decimal on both sides) and its cosine never drops below
0.9985 -- text is correct. Image diverges from LAYER 0 already
(FlashRT absmax=122 vs official=138 right after the first double-
stream layer) and gets progressively worse through layer 3
(cosine 0.752) before partially recovering by layer 4 and the 20
single-stream layers (final img cosine=0.910) -- a THIRD, independent
bug, specific to the image stream, still open.

**Third bug, found by inspection while writing up the above (not yet
Thor-verified): the served frontend's own RoPE table uses a flat
`(img_len, 1)` image grid instead of the real 2D `(14, 28)` patch
grid.** `flash_rt.models.imagewam.rope.build_backbone_rope_table`'s
own signature is `(x0, ref_h, ref_w, ...)` -- a genuine 2D grid, not a
flat token count. `flash_rt/frontends/torch/imagewam_thor.py` (the
ACTUAL served frontend) called it as
`build_backbone_rope_table(d["x0"], d["a0"] - d["x0"], 1, device=DEV)`
-- passing `img_len` (392) as `ref_h` and `1` as `ref_w`, i.e. treating
the real 14x28 image patch grid as a degenerate 392x1 strip. Every
image patch then gets the WRONG 2D spatial position for RoPE (all in
column 0, rows 0-391, instead of the real 14 rows x 28 columns) --
attention between image patches (and between image and text) is
computed with systematically wrong relative positions, compounding
across joint-attention layers. This matches the observed symptom
exactly: text tokens don't depend on the image's own 2D position
convention and are unaffected (txt cosine 0.998, matches); image
tokens are directly corrupted from the very first attention layer
(img cosine drops to 0.979 already at double-stream layer 0) and the
error compounds through the following layers.

**Why this went uncaught**: `benchmarks/imagewam_real_checkpoint_validation.py`
(the one real accuracy comparison against the official model on
record, cosine=0.999927/0.999963) and every test in this project that
independently verifies REAL math against a REAL reference
(`test_imagewam_real_backbone_attention.py`, `test_imagewam_rope_kernel.py`,
`test_imagewam_real_double_stream_block.py`,
`test_imagewam_real_single_stream_block.py`) all correctly pass real
`ref_h`/`ref_w` (14/28) -- ONLY the actual served frontend
(`imagewam_thor.py`) got this wrong. And
`test_imagewam_thor_real_wiring.py`'s own `_double_stream_layer`/
`_single_stream_layer` direct-call tests build their OWN RoPE table
with the SAME flat convention and feed it to BOTH the "reference"
(`real_double_stream_block_forward_fp16`) and the pointer path -- an
internally-consistent WRONG table cancels out in a same-table
comparison, so that test's own cosine=0.999994 says nothing about
whether the table itself matches the real 2D convention. Only a
comparison against the OFFICIAL model's own real positions (which
today's real Thor run was the first to do end-to-end) can catch this
class of bug.

**Fix, same day**: `imagewam_thor.py`'s constructor now reads
`ref_h`/`ref_w` from `dims` (default: the old flat `(img_len, 1)`, so
the toy/default dims -- which have no real 2D image structure --stay
unaffected) and validates `ref_h*ref_w == img_len` before building the
RoPE table; every REAL-dims call site (`tests/test_imagewam_checkpoint_loader.py`'s
two full-frontend tests, `benchmarks/imagewam_thor_bench.py`'s
backbone benchmarks) now passes the real `REF_H=14, REF_W=28`
explicitly.

**CONFIRMED on real Thor, same day: this closes the whole OPT-001/OPT-002
real-accuracy investigation.** Re-ran the three-real-components repro
with `ref_h=14, ref_w=28` passed explicitly in `dims_override`
(required -- the frontend still silently defaults to the flat grid
otherwise, see the fast-fail guard added below):

| | before (flat 392x1 grid) | after (real 14x28 grid) |
|---|---|---|
| `backbone_hidden` absmax | 119808 | 119808 (unchanged, already matched) |
| cosine all / txt / img | 0.985 / 0.998 / 0.910 | **0.999966 / 0.999966 / 0.999966** |

All three bugs found this session (FP16 residual overflow -> BF16
fix; `txt_in`/`img_in` re-derived every layer -> projected once;
flat image RoPE grid -> real 14x28) are now independently confirmed
fixed, together, on real Thor hardware, against the real official
model, with real Qwen3-4B text conditioning and a real VAE-encoded
real LIBERO frame, at the real `x0=512` sequence length. This is the
strongest accuracy confirmation this project has had for the actual
served `pipeline_thor.py` path (as opposed to `pipeline_real.py`'s
separate reference implementation, which was already known-good).

**Footgun closed same day**: since the frontend still silently
defaults to the flat placeholder when `ref_h`/`ref_w` aren't given
(kept for the toy/default dims' own backward compatibility), a real
caller could still forget to pass them and silently regress back to
img cosine~0.91 without any crash -- exactly what already happened
once. `ImageWAMTorchFrontendThor.__init__` now raises `ValueError`
immediately (before touching the multi-GB checkpoint file) if
`ckpt_path` is given without `ref_h`/`ref_w` in `dims_override`;
structural/random-weight dry runs (`ckpt_path=None`) are unaffected.

**Follow-up investigation, same day: ActionDiT/denoise-loop full-action
cosine vs the official model -- NOT a bug, a real scope gap, attempted
and deliberately not completed.** Tried to compare
`ImageWAMTorchFrontendThor`'s full `actions` output (backbone prefill +
ActionDiT denoise loop) against the official model's own real
`infer_action_flux2` output. Before running it, found the official
model integrates the flow-matching ODE over a non-uniform, shift-based
sigma schedule (`imagewam.py`'s `infer_action_flux2` ->
`scheduler_continuous.py`'s `WanContinuousFlowMatchScheduler.build_inference_schedule`,
`shift=5.0` default, `_phi(u,shift)=shift*u/(1+(shift-1)*u)`,
`num_inference_steps=20`), while `imagewam_denoise_loop`/
`imagewam_denoise_step` (`pipeline_thor.py`) integrates with a fixed,
UNIFORM `dims["dt"]` per step -- a real, PRE-EXISTING simplification
(not introduced by today's 3 fixes). Comparing final actions across
these two different integration schedules would measure "does a
uniform-Euler approximation match a shift-scheduled sampler," not
"is ActionDiT's own math correct" -- not run, to avoid reporting a
number that doesn't mean what it looks like it means.

**Rescoped, not pursued for now**: the user's own stated priority for
this project going forward is Thor steady-state SPEED and precision
tracked RELATIVE TO FlashRT's own bf16/fp16 baseline -- not exact
bit-for-bit matching of the official model's own real inference
schedule. Under that framing, chasing a full official-schedule-aligned
multi-step comparison is out of scope; what actually matters (FP8/
NVFP4 ActionDiT vs FlashRT's own FP16 ActionDiT, SAME schedule both
sides) is a much cheaper, already-tractable comparison that doesn't
need the official model or schedule alignment at all -- not yet done,
tracked as a real next step (see the top-level plan status this
session's own final summary gives). Indirect evidence ActionDiT's own
per-layer math is fine either way: `real_action_double_block_forward_fp16`/
`real_action_single_block_forward_fp16` (`real_action_expert.py`,
shared unchanged by `pipeline_real.py` and `pipeline_thor.py`) already
has a real-checkpoint cosine=0.999963 on record, and none of today's 3
bugs touched ActionDiT's own code path.

# OPT-009: real closed-loop robot-state (proprio) conditioning

Status: RESOLVED -- real Thor confirmation same day (see "Thor
confirmation, same day" below: `proprio_row=31` correct, `infer()`
finite, real denormalized actions); normalization stat choice
confirmed exact from this release's own `config.yaml`, not guessed.
The out-of-range action values this Thor run first found were later
root-caused to the uniform-dt schedule (OPT-010) and the official-
model comparison this entry's own "Not yet done" flagged was completed
in OPT-011 (real open-loop eval, cosine 0.870-0.999 across 50 real
frames, proprio included) -- neither is open any more.

Area: real closed-loop testing readiness -- this project's own current
priority (Thor steady-state speed + precision tracked vs FlashRT's own
bf16/fp16 baseline, per the user's explicit direction 2026-09-15)

## Two real, previously-unmodeled gaps, found the same day scoping
"what does this project need for real closed-loop testing"

1. **`proprio_dim=8` -- the real LIBERO checkpoint's `config.yaml` has
   it, `pack_proprio_after_text: true`, and the checkpoint has a real
   trained `proprio_encoder` (`weight` `(7680,8)`, `bias` `(7680,)`,
   confirmed via `torch.load(ckpt_path, mmap=True)`'s TOP-LEVEL
   payload -- a SIBLING of `mot`, not inside it).** The official
   model's own `infer_action_flux2` -> `_append_proprio_to_context_if_enabled`
   RAISES `ValueError` if `proprio_encoder` exists but no `proprio` is
   passed -- proprio conditioning is not optional for this release.
   FlashRT had ZERO mechanism for it anywhere (`checkpoint_loader.py`,
   `imagewam_thor.py`, `pipeline_thor.py` -- confirmed via grep, no
   hits at all) before this entry.
2. **The real model's own `infer_action_flux2` returns the raw
   flow-matching output with NO denormalization** (`return {"action":
   latents_action[0]...}`, confirmed by reading it directly -- no
   `*std+mean` or equivalent anywhere in that function). The release
   ships a `dataset_stats.json` alongside `model.pt` (`state`/`action`,
   each with `global_min/max/mean/std/q01/q99` AND `stepwise_*`
   variants) for exactly this purpose -- the standard VLA convention:
   normalize on the way in (training + proprio input), denormalize on
   the way out (action output), using the SAME stats. This means even
   the open-loop `actions` numbers already on record all session
   (e.g. mean=-0.43, absmax=2.62 from the RoPE-grid-fix confirmation)
   were almost certainly still in the model's own [-1,1] TRAINING
   space, not real physical units -- unrelated to any of today's 3
   bugs, but a real gap for anyone trying to send this output to an
   actual robot.

**Exact normalization convention, confirmed from this release's own
`config.yaml`, not assumed**: `use_stepwise_action_norm: false`,
`norm_default_mode: min/max`, `norm_exception_mode: null` -- both
`state` (proprio, forward/normalize on the way in) and `action`
(backward/denormalize on the way out) use plain `global_min`/
`global_max` linear scaling to `[-1,1]` (clamped to `[-5,5]`), matching
`imagewam/datasets/lerobot/utils/normalizer.py`'s own
`SingleFieldLinearNormalizer` exactly (ported verbatim into a new
`flash_rt/models/imagewam/dataset_stats.py`, including its
degenerate-range `ignore_dim` handling). NEVER `stepwise_*`/`q01/q99`/
`z-score` for this specific release -- a different release could use a
different mode, check its own `config.yaml` before reusing this
unchanged.

**Real insertion rule, NOT a simple append**: `imagewam.py`'s own real
`_append_proprio_to_context` (`pack_proprio_after_text=True` branch)
inserts the proprio token at row `context_mask.sum()` (right after the
last REAL text token, before any padding), shifting every padding row
one position later. Context length grows by exactly 1 (`x0`: 512 -> 513
for this release's real Qwen3 `max_length=512`). This is
data-dependent (depends on THIS prompt's own real token count) but only
needs computing ONCE per `set_prompt()` call (text is fixed per
episode, proprio changes every control step) -- ported into a new
`ImageWAMTorchFrontendThor._set_context_with_optional_proprio` helper,
replicating the real scatter exactly (verified: real tokens keep rank,
proprio lands at `valid_counts`, padding shifts by 1 -- see
`tests/test_imagewam_proprio.py`'s own
`test_proprio_scatter_matches_real_insertion_rule`).

## Implementation

- `flash_rt/models/imagewam/checkpoint_loader.py`: new
  `load_real_proprio_weights(ckpt_path)` -- separate small loader (the
  existing `load_real_imagewam_state_dict` only returns `payload["mot"]`
  by design; widening its contract would break every existing caller),
  returns `None` if this checkpoint has no `proprio_encoder` key.
- `flash_rt/models/imagewam/dataset_stats.py` (new): `MinMaxNormalizer`
  (real `SingleFieldLinearNormalizer` min/max math, verbatim),
  `load_real_normalizers(dataset_stats_path)` -> `(state_norm, action_norm)`.
- `flash_rt/frontends/torch/imagewam_thor.py`:
  - New `dataset_stats_path` constructor kwarg.
  - `dims["proprio_dim"]` opts proprio in (default `None` -- every
    existing caller/test unaffected). When set: loads the real
    `proprio_encoder` (or a random one when `ckpt_path=None`, matching
    every other weight's toy/random-vs-real convention).
  - `proprio_encoder` is applied OUTSIDE the captured CUDA graph via
    plain `F.linear` -- same convention as the real VAE/Qwen3 encoders
    (small, not `flux2`-dependent, no reason to live inside
    `pipeline_thor.py`'s fvk-kernel-based graph).
  - `set_prompt()`'s new `_set_context_with_optional_proprio` does the
    real scatter once per prompt, records `self._proprio_row`.
  - `infer(observation)`: REQUIRES `observation["proprio"]` when
    `proprio_dim` is set (raises otherwise, matching the real model's
    own contract) -- normalizes it (if `dataset_stats_path` given),
    projects it, writes it into the row reserved by `set_prompt()`,
    before `.replay()` (same pattern as `img_raw`/the VAE encode).
    Denormalizes the returned `actions` (if `dataset_stats_path`
    given) before returning.
- `tests/test_imagewam_proprio.py` (new): normalizer round-trip against
  the real `dataset_stats.json`, scatter-rule verification (toy dims,
  no real checkpoint needed), missing-proprio raises, real
  `proprio_encoder` shape check against the real checkpoint (skips
  cleanly without it).

Verified locally: constructed the REAL frontend with `ckpt_path=`,
`dataset_stats_path=`, `proprio_dim=8`, `x0=513`, `ref_h=14, ref_w=28`
(1 real double-stream layer, kept small for this machine's memory) --
real `proprio_encoder` loads (`(7680,8)`/`(7680,)`), `set_prompt()`
places the proprio row correctly (row 31 for a 31-real-token synthetic
context, matching this session's own real Qwen3 observation), `infer()`
produces finite, denormalized actions in a plausible range vs the real
`dataset_stats.json` bounds. Full existing regression suite
(backbone/single-stream/action reference tests, checkpoint-shape test,
isolated real-weight layer test) still passes unchanged.

**Thor confirmation, same day**: real 3-real-component + proprio
`infer()` on Thor (full 25-layer, real LIBERO first frame, real
`observation.state`, all 8 real dims inside the real `dataset_stats.json`
`state` range) -- finite throughout, `backbone_hidden` absmax=119808
(matches the official model exactly, as before), `proprio_row=31`
(matches this session's own real 31-real-token observation), `actions`
already denormalized (O(1) real-unit magnitudes, not the model's own
`[-1,1]` training space). Wiring confirmed correct. BUT only ~62.5% of
predicted action values landed inside the real dataset's own
`[global_min, global_max]` range (gripper dim went negative against a
real `[0,1]` range; translation dims ran further negative than the
real range's own minimum) -- NOT necessarily a bug (a single real
frame's prediction has no reason to stay inside the training range,
and this Thor run used the full 25-layer stack vs this dev machine's
own 1-layer smoke test, so they were never expected to match), but see
the schedule-alignment entry immediately below for a concrete
alternative explanation worth checking before assuming it's just
"this frame's own policy behavior."

**Done since**: the official-model cosine comparison this entry
originally flagged as outstanding was completed in OPT-011 (real
open-loop LIBERO eval, proprio included in every comparison). Still
genuinely open, low priority: wiring proprio into
`_imagewam_thor_spec.py`'s own declared shape documentation (not
load-bearing for runtime, cosmetic).

# OPT-010: real shift-based flow-matching inference schedule

Status: RESOLVED -- real Thor confirmation same day (see "CONFIRMED
on real Thor, same day" below: scheduler bit-exact on Thor too, fixed
OPT-009's out-of-range action values, real `infer()` P50=289.5ms,
single-step ActionDiT cosine=0.999978 vs the official model)

Area: ActionDiT denoise loop -- replaces the fixed-uniform-`dt`
integration schedule (this project's own original simplification) with
the real non-uniform shift-based schedule the official model actually
uses, closing the gap the OPT-009 ActionDiT-schedule-comparison entry
above found (and flagged as "real, not urgent, likely larger scope
than it turned out to be")

## Finding: much smaller scope than first estimated

The real `imagewam.models.backbones.schedulers.scheduler_continuous.WanContinuousFlowMatchScheduler.step()`
(read directly) is a **plain single-step Euler update**
(`sample + model_output * delta`) -- NOT a multi-step integrator like
UniPC. `fvk.gpu_euler_step` (FlashRT's existing Euler kernel) already
implements the exact right formula; only the per-step VALUES feeding
it needed to change, from a linear `1.0 - step*dt`/fixed-`dt` formula
to the real non-uniform `timesteps[step]`/`deltas[step]` the real
`build_inference_schedule` produces. `step` was already treated as a
compile-time-constant-per-unrolled-loop-iteration in FlashRT's own
existing design (`action_mods[step]` already selected per iteration
before this change) -- so this needed zero new architecture, zero new
kernels, zero new buffers, purely a formula substitution at two call
sites.

**Real confirmed parameters (this release's own `config.yaml`, not
guessed)**: `shift=5.0`, `num_train_timesteps=1000` (both `video_scheduler`/
`action_scheduler` blocks use the same values for this release),
**`eval_num_inference_steps: 10`** (the real evaluation step count --
NOT the scheduler's own generic default of 20 some example configs
use; matches this project's own long-standing "10-step" benchmark
convention already on record above, confirming that number was already
the right target). Unit conversion `_scheduler_timestep_to_unit`:
`timestep / num_train_timesteps`, confirmed by reading `imagewam.py`
directly, not assumed.

## Implementation

- `flash_rt/models/imagewam/scheduler.py` (new): `phi`,
  `build_inference_schedule` -- ported verbatim from the real
  `WanContinuousFlowMatchScheduler`, verified bit-for-bit identical
  against the real scheduler directly (`tests/test_imagewam_scheduler.py`'s
  `test_schedule_matches_real_scheduler`, skips cleanly without the
  real `imagewam` package).
- `flash_rt/models/imagewam/pipeline_thor.py`: `imagewam_denoise_step`
  gained an optional `delta=` param (falls back to `dims["dt"]` when
  `None`, so every existing caller is byte-for-byte unaffected);
  `imagewam_denoise_loop` gained an optional `deltas=` (list, one per
  step) threaded through the same way `action_mods`/`head_mods`
  already are.
- `flash_rt/frontends/torch/imagewam_thor.py`: `dims["shift"]` opts
  the real schedule in (default `None` -- unset, every existing
  caller/test keeps the exact original linear formula, verified via
  the full regression suite still passing unchanged).
  `_compute_action_modulations` now returns `(mods, head_mods, deltas)`
  (was `(mods, head_mods)`) -- `deltas` is `None` unless `dims["shift"]`
  is set; threaded into both `_capture_graph()` call sites'
  `imagewam_denoise_loop(..., deltas=self._deltas)`.

Verified locally: `dims=dict(shift=5.0, num_train_timesteps=1000,
num_denoise_steps=10)` end to end (construct, `set_prompt`, `infer`) --
finite, and `self._deltas` matches the real scheduler's own 10-step
output exactly (`[-0.0217, -0.0259, -0.0313, -0.0387, -0.0490,
-0.0641, -0.0874, -0.1263, -0.1984, -0.3571]`). Full existing
regression suite (backbone/action reference tests, checkpoint tests,
proprio tests) still passes unchanged with `shift` left unset.

**CONFIRMED on real Thor, same day -- this was the actual cause of
OPT-009's out-of-range action values, not policy behavior.** Re-ran the
same real 3-real-component + proprio `infer()` with `shift=5.0,
num_train_timesteps=1000, num_denoise_steps=10` (the real confirmed
values) instead of the old uniform `dt`:

| | old uniform schedule | real shift=5.0 schedule | dataset range |
|---|---|---|---|
| gripper mean / min-max | -0.24 / -0.43~0.14 | **0.996 / 0.981~1.002** | `[0,1]`, mean 0.51 |
| translation dim0 mean | -1.45 | **0.51** | `±0.94` |
| translation dim1 mean | -1.11 | **0.38** | `±0.94` |
| fraction inside `[global_min,max]` | 62.5% | **98.4%** (100% at 5% tol) | |
| first-step gripper vs real GT (1.0) | off by 1.43 | **off by 0.0008** | |

`self._deltas` matched the real `WanContinuousFlowMatchScheduler`'s
own output bit-for-bit on Thor too (not just this dev machine).
Real end-to-end `infer()` (full real pipeline: real VAE + real Qwen3 +
real proprio + real 25-layer backbone + real 10-step ActionDiT denoise,
all outside-graph real encode steps included): **P50 = 289.5 ms**.

**Single-step ActionDiT cosine vs the official model, real first step
(`t=1.0`, `delta=-0.0217`, the real schedule's own first entry)**:
hidden-state cosine = **0.999978**, finite. (The official `Flux2ActionHead`
takes a `vec` tensor rather than the `t_mod` dict `real_action_*`
produces, so the final head+Euler-step portion wasn't chased down to
its own cosine -- the per-layer transformer math, which is where any
real bug would live, is already covered at 0.999978.)

**Minor finding, not prioritized**: the 5-precision table's own
per-kernel-isolated benchmark showed `mot_joint` jump from 0.044ms
(even `total=968`, the old x0=512 shape) to 0.195ms (odd `total=969`,
the new x0=513-with-proprio shape) -- `softmax.cu`'s own kernels
process columns in `__half2` pairs with a scalar fallback for a
trailing odd column, and an odd `total` hits that fallback on every
query row. In ABSOLUTE terms this is small (~0.15ms x 10 steps
=~1.5ms) against the real 289.5ms end-to-end number, and is the reason
the 5-precision table's own DERIVED "10-step" total (e.g. FP16 297ms)
runs slightly higher than the real measured `infer()` (289.5ms) --
trust the real measured number, not the derived per-kernel sum, for
this shape. Not worth chasing further given the absolute cost; noted
here in case it compounds with a future higher-`total` shape.

**Takeaway for Stage 2's own mot/ActionDiT-FA4 item**: given
`mot_joint`'s own absolute cost (0.04-0.2ms) is negligible next to the
GEMM-dominated per-layer costs (backbone double/single: 4.9-5.9ms;
ActionDiT double/single: 0.5-0.7ms) at this real shape, FA4 for the
"mot" site would not move the total number meaningfully even if it
worked -- deprioritize that Stage 2 item; the real remaining speed
lever is precision choice (NVFP4 already ~19% faster than FP16 at this
shape, `242ms` vs `297ms` derived), not attention-kernel choice.

# OPT-011: real open-loop LIBERO evaluation -- closes out the correctness line

Status: RESOLVED -- the strongest validation this project has run,
confirms `pipeline_thor.py` faithfully reproduces the official model
across real, diverse data (10 tasks x 5 frames = 50 real observations),
not just the single cherry-picked frame every prior entry used

Area: whole-pipeline correctness (real VAE + real Qwen3 + real proprio
+ real backbone + real ActionDiT + real shift schedule + real
denormalization, ALL together, across real task diversity)

## Result

**FlashRT vs the official model, same frame/noise/schedule
(`num_inference_steps=10, sigma_shift=5.0`)**: cosine 0.870-0.999
across sampled episodes, most 0.997-0.999. **Critical diagnostic
signature**: on episodes where BOTH FlashRT and the official model
diverge from the real dataset ground truth (e.g. ep226, both land near
cosine=0 vs GT, both predict gripper=1 where GT=0), FlashRT still
tracks the OFFICIAL model closely (cosine 0.870/0.928 there) -- i.e.
when the prediction is "wrong" relative to GT, it is wrong the SAME
WAY on both sides. This is the signature of a real POLICY/checkpoint
limitation on those specific tasks, not a FlashRT serving bug: a
wiring/kernel bug would show FlashRT diverging from the OFFICIAL
model too, not just from GT.

**50-frame vs real dataset GT (step-0, real units)**: 94.6% of values
land inside `[global_min,global_max]` (xyz/rpy nearly all in-range;
gripper 64%, occasionally ~1.002 vs a `[0,1]` range -- a tiny,
expected float overshoot, not a normalization bug). Overall MAE=0.198,
cosine=0.559 across all 50 frames -- but this average is misleading on
its own: 4/10 tasks are excellent (MAE 0.029-0.045, gripper error
0.002-0.005, cosine 0.983-0.993), 6/10 tasks are poor (gripper MAE
0.40-1.00, cosine near 0 or negative) -- driven almost entirely by
gripper open/close disagreement and some tasks' own xyz pattern not
matching their demos, NOT a directional/systematic serving-level bias
(rotation MAE stays small and uniform across all tasks: roll/pitch/yaw
0.028/0.058/0.036 -- no single axis is "always off," which is what a
real wiring bug would look like).

**Stability, 40 consecutive `infer()` calls after one `set_prompt()`**
(the real closed-loop-shaped access pattern): P50=279.5ms, P90=280.1ms,
range 278.4-281.1ms -- flat, no drift. GPU memory: 18.132GB at both
start and end, delta=0 -- no leak. This was the one remaining
"is this actually usable in a real control loop" concern from OPT-009;
now directly measured and clean.

## What this does and doesn't mean

Confirms: the full real-data serving path (everything built this
session -- BF16 residual, txt_in/img_in-once, real RoPE grid, proprio,
denormalization, real shift schedule) is faithful to the official
model across real task diversity, not just one frame. The remaining
GT mismatch on 6/10 tasks is a checkpoint/policy quality question
(does this specific LIBERO fine-tune generalize well to these specific
tasks/episodes), which is OUTSIDE this project's own scope (a Thor
inference-engine port: speed + precision-vs-own-baseline, not model
training/data quality) -- not something to chase here.

One real confound worth a quick double-check, not a finding: the
eval's own footnote notes a substituted episode (`ep359` instead of
the first-listed one for the "stove" task, because the wrist video
only covers the first 1050s) -- if any OTHER of the 6 poor-scoring
tasks has a similar frame/video-length mismatch in how the eval
harness paired observation to ground truth, that would look like
"policy is wrong" while actually being a data-pairing issue upstream
of both FlashRT and the official model equally (so it wouldn't show up
as a FlashRT-vs-official divergence either way) -- worth a quick sanity
check on the eval harness itself before concluding those 6 tasks are
really a checkpoint limitation, but not urgent given it doesn't affect
FlashRT's own correctness story.

# OPT-012: real end-to-end steady-state speed breakdown, real vs official PyTorch

Status: RESOLVED -- real, precise (not estimated) profiling; corrects
an earlier same-day estimate (VAE guessed at ~15.7% from an older,
different-resolution historical number; the real current figure is
7.6%, about half)

Area: whole-pipeline speed, at the real FP16 steady-state config
(`x0=513` with proprio, real 10-step shift=5.0 schedule) -- both an
absolute number and a real breakdown of where the time actually goes

## Real vs official PyTorch, end to end (same real checkpoint, same
224x448 input, same horizon=64, real Thor hardware)

| path | P50 | vs FlashRT (284.9ms) |
|---|---:|---:|
| official `infer_action_flux2`, 10-step, re-runs Qwen3 every call | 605 ms | 2.12x slower |
| official, its own default 20-step | 863 ms | 3.03x slower |
| official, 10-step, text cached (fair comparison to FlashRT's `infer()`) | 485 ms | **1.70x slower** |

**FlashRT is faster than the official PyTorch reference implementation
by 1.7-2.1x depending on what's held fixed** -- this is the real
apples-to-apples speed validation for this whole session's work (a
prior discussion here compared against an unrelated Wan2.2 number
without knowing what it actually measured; this comparison is against
the SAME checkpoint's own reference implementation, same hardware,
same shapes).

Per-stage, official PyTorch (10-step): Qwen3=118ms (re-run every call
in the official path -- NOT part of FlashRT's own `infer()`, which
caches text via `set_prompt()`), VAE=21.5ms, backbone prefill=201ms,
ActionDiT 10-step=258ms. FlashRT: prefill=118ms (**1.70x faster**),
denoise=145ms (**1.78x faster**), VAE=21.5ms (same AE, no difference
expected or found).

## Real steady-state `infer()` breakdown (FlashRT, FP16, P50=284.9ms)

| part | P50 | share |
|---|---:|---:|
| ActionDiT 10-step denoise (in-graph) | 144.6 ms | **50.7%** |
| backbone prefill, 25 layers (in-graph) | 118.4 ms | **41.5%** |
| VAE `encode_to_tokens` (out-of-graph) | 21.5 ms | **7.6%** |
| proprio encode | 0.11 ms | ~0% |
| action noise fill | 0.02 ms | ~0% |
| denorm + D2H copy | 0.05 ms | ~0% |
| **sum / measured** | 284.7 / 284.9 ms | 100% |

`graph.replay()` itself: 262.9ms (92.3% of `infer()`). Eager (ungraphed)
prefill+denoise: 281ms -- graph capture saves ~1.07x, matching OPT-004's
own earlier "compute-bound, not launch-bound" finding (small, real,
not the dominant lever). Single denoise step P50=15.5ms x 10 = 155ms,
consistent with the 10-step loop total.

## Correction to this same session's earlier VAE-share estimate

An earlier turn today estimated VAE at ~15.7% of `infer()`, extrapolated
from an OLDER historical Thor number (43.9ms, a different round's
shape/resolution) divided by the CURRENT total (279.5ms). The real,
directly-measured current figure is **21.5ms / 7.6%** -- about half
the earlier estimate. **This changes the priority call from the
previous entry**: VAE fusion (GroupNorm+SiLU -- `cosmos3_edge/vae_native.py`
already has a real, working precedent for exactly this fusion in this
same codebase, so this would be adapting existing work, not writing
from scratch) now has a hard ceiling of 7.6% even if reduced to
near-zero, smaller than previously implied -- still legitimate,
real, zero-precision-cost work, just not the highest-value item
anymore.
**denoise (50.7%) + prefill (41.5%) = 92.2% of the total dominate by
far, and both are already running through FlashRT's own kernel system**
-- the real remaining lever is PRECISION CHOICE on that 92.2% (NVFP4
already measured ~14-22% faster than FP16 on prefill/denoise
specifically), not VAE fusion. Re-prioritizing: pick a default
deployment precision (Stage 3) before investing in VAE fusion.

## SUPERSEDED (2026-09-18): the official-vs-FlashRT comparison above used a stale FlashRT baseline and an unfair official-side setup

The 284.9ms FlashRT number above, and the official `485/605/863ms`
comparisons built on it, predate `x0=513`/proprio/the real shift
schedule/`num_action=64` all being nailed down together, AND the
official side's own `torch.compile` variants hadn't been tried yet.
Re-measured under IDENTICAL real conditions on both sides (dual
camera 224x448, proprio, shift=5.0, 10-step, `horizon=64`, real
checkpoint, Qwen3 held OUTSIDE the timed region on both sides,
`WARMUP=5`/`N=20`) via `official_torch_infer_bench.py`. Official
implementation is **bf16 eager** (`nn.Linear` -> cuBLASLt + SDPA) --
a different kernel family from FlashRT `fp16`/`nvfp4`, not just a
different precision label.

| path | P50 | vs FlashRT `nvfp4` |
|---|---:|---:|
| FlashRT `nvfp4` (production default) | **231.6 ms** | 1.00x |
| FlashRT `fp16` | 280.2 ms | 1.21x |
| FlashRT `fp16_cutlass` | 306.7 ms | 1.32x (not adopted, OPT-013) |
| official bf16 eager, end to end | 453.6 ms | **1.96x** |
| official `infer_action_flux2` (cached text) | 458.0 ms | 1.98x |
| official `torch.compile` (`backend=cudagraphs`), e2e | 514.4 ms | 2.22x -- SLOWER than eager |
| official `torch.compile` (`backend=inductor`) | FAILED | Triton `ptxas` (CUDA 13.0.48/Triton 3.5.1) rejects `sm_110a` kernels even with VAE excluded from the compiled region (fails inside a fused LayerNorm) |

Official eager breakdown: VAE+proprio 21.4ms, transformer
(prefill+10-step) 432.3ms -- confirms (again, at the current real
shapes) that FlashRT's real advantage is almost entirely in the
transformer compute, not VAE (VAE numbers match closely on both
sides, as expected -- same AE).

**`torch.compile` does not currently give Thor a faster official
baseline to beat, and isn't a live comparison point**: `inductor`
can't emit working Thor (`sm_110a`) Triton kernels on this toolchain
at all; `cudagraphs` DOES run, but `_build_mot_attention_mask_flux2`
builds its attention mask with a CPU-side Python bool, which makes
`dynamo` skip real graph capture for that region -- the "compiled"
path ends up paying tracing/dispatch overhead on top of eager's own
cost, landing SLOWER (505ms transformer-only vs. eager's 432ms), not
faster. Neither compile mode is a meaningful comparison target right
now; **bf16 eager (~454ms) is the only valid official baseline**, and
FlashRT `nvfp4` beats it by a real, currently-accurate 1.96x -- even
FlashRT's own `fp16` baseline (280ms) beats the official implementation.

This supersedes every number in the "Real vs official PyTorch" table
above (superseded, not deleted, for the historical record of how the
comparison evolved as the real config converged) -- use the table in
this subsection for any future reference to "FlashRT vs official
speed."

# OPT-013: FP16 CUTLASS GEMM + SwiGLU epilogue fusion (new precision tier)

Status: Phase A + Phase B implemented, compiled (local SM90a
syntax-check substitute, this dev machine has no SM100/SM110 CUTLASS
build), zero local regression; Phase C (gated-residual epilogue) NOT
attempted -- deliberately deferred, see below

Area: FP16/BF16 GEMM speed -- 92.2% of real steady-state `infer()` cost
(OPT-012) is backbone prefill + ActionDiT denoise, both GEMM-dominated
and, until now, entirely dispatched through cuBLASLt's own generic
autotuned algorithm selection, never CUTLASS (unlike FP8-static and
NVFP4, which already use hand-tuned CUTLASS kernels)

## Background: found by forking a survey of FlashRT's OTHER models

User asked why other FlashRT Thor models see more speedup than
ImageWAM; a fork survey of `shared_primitives.py` (GROOT/Pi0.5/Pi0's
shared Thor B=1 path), `motus_rtx.py`/`hyvla_thor.py`, and
`cosmos3_edge`'s `vae_native.py` found: (1) `flashrt_rms_qkv_fp16` --
turned out, on reading the real source, to be a two-kernel C-level
BUNDLE (`rms_norm_fp16` then `cutlass_fp16_k64`, still a real
intermediate write+read of `x_norm_scratch` between them) rather than
a genuine single-kernel fusion -- so this collapses into item (2)
below, not an independent technique; (2) a real, Thor-tuned FP16
CUTLASS GEMM family (`cutlass_fp16_plain/sq/t1/wide/k64/2sm21`,
`csrc/gemm/cutlass_sm100_fp16.cu`) already exists in this codebase,
confirmed timed against real Thor hardware (a code comment dated
2026-05-18), never wired into ImageWAM; (3) GEMM-epilogue fusion for
activation (`cutlass_fp16_k64_gelu`/`_sq_gelu`) and for
multiply-by-auxiliary-tensor (`cutlass_fp16_k64_mul_aux`,
`D=(A@B)*Aux`) already exist for a DIFFERENT model's GEGLU MLP (GELU,
not SiLU) and residual accumulate (`beta=1` epilogue fold, but only
for an UN-GATED residual -- see Phase C below for why this doesn't
transfer directly). INT8/INT4 rowwise fusion patterns elsewhere are
confirmed NOT applicable (OPT-007: 7-8.6x slower than FP16 on Thor,
no native tensor-core benefit at these shapes on sm_110).

## Phase A: `CutlassFp16Linear` -- pure GEMM-backend swap, zero new math

New `precision="fp16_cutlass"` tier (`quant_linear.py`'s
`CutlassFp16Linear`, `imagewam_thor.py`'s `_wrap_linear` dispatch).
Every plain `Fp16Linear`-wrapped weight (backbone/ActionDiT QKV, proj,
mlp2/mlp_down, action_encoder/head) dispatches through
`cutlass_fp16_sq`/`_wide` (mirrors `_pick_fp8_cutlass_variant`'s own
already-established shape heuristic: `wide` if `N>=4*K` else `sq`) --
identical `alpha=1,beta=0` math to `Fp16Linear`, so this can only
differ from the cuBLASLt path in SPEED, never in output value (same
weight, same GEMM, different kernel implementation selecting the same
mathematical operation). Same Thor/Blackwell-only gate as
`StaticFp8Linear(use_cutlass=True)`/`Nvfp4Linear`
(`hasattr(fvk, "cutlass_fp16_k64")`), raises a clear `RuntimeError` on
this dev machine's Ada build, exactly as expected.

## Phase B: `CutlassFp16SwiGluMlp` -- SwiGLU gate/up fused, new kernel added

ImageWAM's real MLP gate IS true SiLU (confirmed against
`csrc/kernels/activation.cu`'s own `silu_glu_merged_kernel`:
`silu(g)=g*sigmoid(g)`, computed on half the merged GEMM output then
multiplied by the other half) -- NOT the GELU the existing
`cutlass_fp16_k64_gelu`/`sq_gelu` epilogues use, so those were not
directly reusable. Added a genuinely new CUTLASS type,
`sm100_fp16_k64_silu` (`gemm_types_sm100_fp16.h`), an exact structural
copy of `sm100_fp16_k64_gelu` with the activation functor swapped from
the file's own custom `GeluTanhApprox` to CUTLASS's OWN BUILT-IN
`cutlass::epilogue::thread::SiLu` (confirmed present in this project's
vendored CUTLASS, `epilogue/thread/activation.h`, formula
`value*sigmoid(value)` -- an exact match, not an approximation) -- plus
its host entry point `cutlass_fp16_k64_silu` (`cutlass_sm100_fp16.cu`)
and pybind11 binding, mirroring the GELU variants' own pattern exactly.

`CutlassFp16SwiGluMlp` (`quant_linear.py`) splits the real checkpoint's
merged `mlp0.weight`/`mlp_in.weight` `(K, 2*mlp_hidden)` into two
`(mlp_hidden, K)` CUTLASS-layout halves ONCE at construction (a plain
column-slice + transpose, same real trained weight values, no accuracy
change), then on each call: `gate_buf = SiLU(x @ W_gate)` (new
`cutlass_fp16_k64_silu`, activation fused into the GEMM epilogue) ->
`out = (x @ W_up) * gate_buf` (existing `cutlass_fp16_k64_mul_aux`,
the multiply fused into THIS GEMM's own epilogue) -- net: the separate
`silu_glu_merged_fp16` elementwise kernel is eliminated entirely, at
the cost of splitting one wide GEMM into two `mlp_hidden`-wide ones
(same total FLOPs, one fewer kernel launch's worth of memory
round-trip for the intermediate). `pipeline_thor.py` gained a small
`_mlp_gate_up` helper (all 5 MLP-gate call sites across backbone
double/single and ActionDiT double/single now go through it) that
DUCK-TYPES on `weights[key]`'s own class to pick the fused path or the
existing default path -- `pipeline_thor.py` itself stays
precision-agnostic, same convention as every other `weights[key](...)`
call site in this project.

**Local verification (this dev machine has no SM100/SM110 CUTLASS
build, so nothing here could be RUN)**: the exact `.cu`/`.h` changes
compile cleanly under a standalone `nvcc -gencode=arch=compute_90a,code=sm_90a`
syntax-check substitute (this local CUDA 12.8 toolkit doesn't recognize
`sm_110a`/`compute_110a` at all, so this is the closest available
stand-in that still exercises the same CUTLASS 3.x CollectiveBuilder/EVT
template machinery -- a real, meaningful check: a wrong template
parameter, e.g. a functor with the wrong interface, would fail to
instantiate regardless of target SM version). Full existing local
regression suite (backbone/action reference tests at cosine
0.999994-1.000000, checkpoint-shape test, proprio tests) still passes
UNCHANGED with `precision="fp16"` (default) -- confirms Phase A/B's own
new code paths are correctly gated behind `precision="fp16_cutlass"`
and introduce zero effect on the existing default path.
`precision="fp16_cutlass"` correctly raises a clear `RuntimeError` on
this Ada machine (same gate as NVFP4/FP8-static+CUTLASS) rather than
silently falling back or crashing confusingly.

## Phase C: gated-residual epilogue fusion -- NOT attempted, deliberately

The fork's own "residual fusion, direct copy" framing was for an
UN-GATED residual (`beta=1`: `D = alpha*A@B + beta*D_old`, a plain
scalar accumulate-into-D). ImageWAM's real residual is GATED
(`residual += gate_vector * proj_output`, `gate` a real,
timestep-conditioned, PER-CHANNEL vector -- `gate_res_fp16`/
`gate_res_bf16res`) -- `beta` in the existing CUTLASS API is a scalar,
not a per-channel vector, so the existing `beta=1` pattern is
mathematically WRONG for ImageWAM's own residual rule and cannot be
applied as-is. A correct fusion needs a genuinely NEW epilogue (e.g. a
`Sm90ColBroadcast`-based EVT node multiplying the accumulator by a
per-column gate vector, composed with the existing `C`/residual load)
-- CUTLASS almost certainly has the primitives for this (per-row/
per-column broadcast-and-combine EVT nodes are a documented pattern),
but composing them correctly requires deeper CUTLASS EVT-tree API work
than Phases A/B needed (those were: swap one template parameter, or
copy an existing epilogue verbatim -- this needs assembling a NEW one).

**Deliberately not attempted this round**: the failure mode for a
subtly-wrong epilogue is silent wrong numbers, not a crash or a
compile error -- exactly the class of bug this whole project's real
Thor validation work (OPT-001/002/009/010) spent real effort finding
and fixing today. This dev machine cannot verify CORRECTNESS of any
SM100 CUTLASS kernel at all (only that it compiles, via the sm_90a
substitute) -- shipping a new, unverified-beyond-compilation epilogue
that touches every layer's own residual stream is a materially
different risk than Phase A/B's changes (pure backend swap; activation
functor swap with an exact CUTLASS-verified formula match). Left as a
clearly scoped, real follow-up rather than rushed.

**Not yet done, any phase**: real Thor timing (does `fp16_cutlass`
actually beat the cuBLASLt-autotuned `fp16` default at ImageWAM's real
(M,N,K) shapes? -- this project's own established pattern, e.g.
FP8-static+CUTLASS's real ~8-9% win over cuBLASLt FP8-static, suggests
plausible but unmeasured here); real Thor correctness (cosine vs the
`fp16` default, same real inputs -- Phase A should be ~1.0 by
construction since the math is identical modulo GEMM-algorithm
floating-point non-associativity, Phase B should also be very close
since it's the same real math restructured, but neither has been
measured); trying the other CUTLASS FP16 tile variants (`k64`/`2sm21`)
against the default `sq`/`wide` heuristic pick, which is UNCALIBRATED
for this specific kernel family (ported from FP8's own heuristic,
which itself was flagged as provisional).

## CONFIRMED on real Thor, same day: correct, but a real negative speed result

**Correctness: clean.** `cutlass_fp16_k64_silu`/`_mul_aux` rc=0, SiLU
vs `torch.silu(x@W.T)` cosine=1.000000. Phase A isolated GEMMs (proj/
qkv/mlp2, including M=64) vs cuBLASLt cosine=1.000000. Phase B fused
SwiGLU at the real MLP widths (M=513/392/64) vs the existing
`silu_glu_merged_fp16` path cosine=1.000000. Full `infer()`
actions/action_latent cosine=1.000000. `backbone_hidden` (25-layer
accumulation) cosine=0.995090 -- expected floating-point
non-associativity compounding across 25 real layers from a different
GEMM algorithm's own summation order, not a transpose/variant bug
(every ISOLATED layer/kernel check above is exactly 1.0).

**Real bug found and fixed same day: CUTLASS FP16 requires N/K
divisible by 8** (`can_implement` fails otherwise) -- `action_encoder`
(real `K=7`, LIBERO's own action_dim) and `head.linear` (real `N=7`)
are structurally incompatible with EVERY tile variant
(`plain/sq/t1/wide/k64/2sm21` all returned `can_implement=-1`), and the
default heuristic picks `wide` for `head.linear`'s own shape, so
`set_prompt()`'s graph capture crashed outright with `precision=
"fp16_cutlass"` before this fix. Fixed: `_wrap_linear` now falls back
to the plain cuBLASLt `Fp16Linear` for any `n % 8 != 0 or k % 8 != 0`
shape, keeping `precision="fp16_cutlass"` usable (falls back only for
these two tiny, real-FLOPs-negligible GEMMs) rather than removing the
precision tier or crashing.

**Real speed result: NEGATIVE -- no CUTLASS FP16 tile variant beats
cuBLASLt on real Thor, at ImageWAM's real (M,N,K) shapes**, real
`x0=513/a0=905` full-pipeline `infer()` P50:

| precision | P50 | vs `fp16` (cuBLASLt) |
|---|---:|---:|
| `fp16` (cuBLASLt, current default) | **287.2 ms** | -- |
| `fp16_cutlass`, default `sq`/`wide` heuristic | 306.9 ms | **+6.9% slower** |
| `fp16_cutlass`, `variant="plain"` | 297.8 ms | +3.7% slower |
| `fp16_cutlass`, `variant="2sm21"` | 299.8 ms | +4.4% slower |
| `fp16_cutlass`, `variant="k64"` | 303.3 ms | +5.6% slower |
| `fp16_cutlass`, `variant="t1"` | 305.1 ms | +6.2% slower |

Every variant tried is SLOWER than cuBLASLt's own generic autotuned
algorithm selection, at every tile config tried. This falsifies the
hypothesis this whole entry started from (that FlashRT's own existing,
Thor-tuned FP16 CUTLASS family would beat cuBLASLt for ImageWAM the
way it does for FP8) -- a real, valuable negative result: cuBLASLt's
own per-shape autotuning (already the `fp16` default's own mechanism,
OPT-004 step 4) is evidently already doing at least as well as this
particular CUTLASS kernel family can at these specific shapes on real
Thor hardware. Unlike FP8 (where CUTLASS had a clear, measured edge),
FP16 does not -- these are different kernel families with different
tuning histories, and this result should not be assumed to generalize
back to the FP8 case or forward to any future precision work without
its own real measurement.

**Conclusion: `fp16_cutlass` is NOT recommended for production use.**
Kept in the codebase (correct, gated behind `precision="fp16_cutlass"`,
opt-in, zero effect on the `fp16` default) as a real, working, but
not-faster alternative -- useful if a future CUTLASS tile addition or
toolkit update changes this picture, and as a documented example that
"CUTLASS exists for this precision" is not sufficient reason to expect
a win without measuring. Stage 3 (pick a default deployment precision)
reverts to comparing `fp16`/FP8-static+CUTLASS/NVFP4 only -- this
entry is closed, not carried forward as a live speed candidate.

# OPT-014: Stage 3 default precision decision -- CLOSED, default is `nvfp4`

Status: CLOSED. Real Thor checklist run against real checkpoint
weights, real proprio+shift-schedule, and real open-loop LIBERO data
(the gap every earlier FP8/NVFP4 cosine number in this file had --
OPT-005/OPT-006's own 0.999242/0.989133 were `test_imagewam_quant_linear.py`'s
small-shape, single-layer, RANDOM-weight numbers, never re-checked
through the real 25-layer stack or real GT). `ImageWAMTorchFrontendThor`'s
default `precision` changed from `"fp16"` to `"nvfp4"`
(`imagewam_thor.py`) -- this is now a real production default, not
just a benchmark opt-in.

## Real Thor result 1: full real-weight end-to-end cosine vs `fp16`, same real frame

| | actions | backbone_hidden | action_latent |
|---|---:|---:|---:|
| **nvfp4** | **0.9998** | **0.9939** | **0.9997** |
| fp8_static_cutlass | 0.897 | 0.461 | 0.874 |
| fp8_static (cuBLASLt) | 0.697 | 0.467 | 0.257 |

**NVFP4's isolated-GEMM 0.989 does NOT get amplified across the real
25-layer stack** (0.9939 backbone_hidden, if anything closer to 1.0 than
the single-layer number suggested it might drift). **FP8-static is the
one that actually degrades badly**, and CUTLASS vs cuBLASLt makes
almost no difference (0.461 vs 0.467) -- this isolates the cause to
calibration, not the GEMM backend: every `StaticFp8Linear` weight
except `img_in` (OPT-004 step 6's own real-token calibration) still
uses a placeholder `N(0, 0.1)` activation-scale guess, and that
placeholder is measurably wrong at real-model scale. `_calibrate_fp8()`
itself ran clean on real weights (no NaN/inf across all ~220 scales,
`act_scale` ~1e-3) -- the calibration STEP works, its INPUT
DISTRIBUTION is the problem.

**Real alignment-crash bug, same shape as OPT-013's**: `action_encoder`
(K=7) / `head.linear` (N=7) fail all three quantized precisions'
alignment requirements (FP8 CUTLASS `can_implement=-1`, NVFP4 requires
K%16, FP8 cuBLASLt heuristic also picks a bad algorithm there). Fixed
in `_wrap_linear` (`imagewam_thor.py`): the `nvfp4` branch now falls
back to plain `Fp16Linear` for `n % 16 != 0 or k % 16 != 0`, identical
pattern to OPT-013's `fp16_cutlass` fix -- both quantized tiers need
this fallback to run end to end at all, since real LIBERO action_dim=7
never aligns to either family's block size.

## Real Thor result 2: real open-loop LIBERO, 50 frames vs GT (same tasks/frames as OPT-011's own `fp16` run)

| | MAE | vs GT cosine | first 4 (previously-good) tasks' cosine |
|---|---:|---:|---|
| fp16 | 0.1985 | 0.559 | 0.983-0.993 |
| **nvfp4** | 0.2007 (1.01x fp16) | 0.558 | **0.982-0.994** |
| fp8_static_cutlass | 0.2839 (1.43x fp16) | 0.565 | 0.90-0.95 (regressed) |

NVFP4 reproduces `fp16`'s own error structure almost exactly, including
on the tasks where `fp16` already tracks GT well. FP8-static visibly
degrades even the tasks `fp16`/NVFP4 get right -- confirms result 1
isn't a cosine-metric artifact, it shows up in task-relevant behavior.

## Real Thor result 3: ActionDiT M=64, FP8 CUTLASS vs cuBLASLt

Numerically identical (cosine=1.0) but **CUTLASS is 1.44-1.68x SLOWER**
at this specific M=64 shape (qkv: 0.023ms cuBLASLt -> 0.034ms CUTLASS)
-- confirms `_pick_fp8_cutlass_variant`'s heuristic (ported from FP8's
own backbone-shape tuning, flagged provisional since OPT-006) really
doesn't transfer to ActionDiT's own small-M shapes, and it's a real
loss there, not just "no extra benefit." The backbone's own large-M
CUTLASS win is large enough to still make whole-pipeline
`fp8_static_cutlass` faster than `fp8_static`, but this specific
sub-result is now a confirmed regression at this shape, moot only
because `fp8_static*` is not the chosen default anyway.

## Real Thor result 4: full `infer()` P50 at corrected real conditions (`x0=513`, proprio on, real 10-step shift schedule)

| precision | P50 |
|---|---:|
| **nvfp4** | **236.9 ms** |
| fp8_static_cutlass | 243.0 ms |
| fp8_static | 259.8 ms |
| fp16 | 280.2 ms |
| fp16_cutlass | 306.7 ms |

Supersedes OPT-006's own 97.9ms/92.6ms prefill-only numbers, which
predate `x0=513` (proprio row), the real shift schedule, and were
prefill-only rather than full `infer()` -- not directly comparable to
this table. NVFP4 is fastest here too, by a real margin (15.5% faster
than `fp16`, 2.5% faster than `fp8_static_cutlass`).

## Real Thor result 5: NVFP4 stability, 40 calls

P50=243.5ms, min-max 243.2-247.3ms (flat), memory delta=0 -- same
methodology as OPT-011's own `fp16` stability check, same clean result.

## Decision

**Default precision is `nvfp4`**: fastest measured option AND closest
to `fp16`/GT on both cosine and real open-loop task behavior --
resolves the correctness-vs-speed tradeoff in NVFP4's favor decisively,
not narrowly. `fp8_static`/`fp8_static_cutlass` are NOT promoted --
their real blocker is placeholder per-layer activation calibration
(`N(0,0.1)`, everywhere except `img_in`), not the GEMM backend
(CUTLASS and cuBLASLt degrade identically). `fp16` remains available
and correct (`precision="fp16"`) for any caller that needs an
un-quantized reference or hits an environment without the Blackwell
NVFP4 build (`Nvfp4Linear` raises a clear `RuntimeError` there, same
gate as before).

## Follow-up, not started

Real per-layer activation calibration for `fp8_static*` (replacing the
`N(0,0.1)` placeholder the way `img_in` already got real-token
calibration in OPT-004 step 6) would need real representative
activations captured at every layer, not just the VAE's own entry
point -- a real project, not a quick fix. Worth revisiting only if
NVFP4 accuracy is ever found insufficient on a broader/harder task set
than this checklist's 50 frames covered; not blocking, not scheduled.

# OPT-015: systematic op-fusion audit vs. official ImageWAM

Status: audit complete (fork, read-only). Finding 2 (NVFP4 SwiGLU
fusion) implemented, real-Thor-verified, REVERTED from the default (no
measured win, see its own dated section below). Finding 1's
sub-problem 1 (qkv+mlp_in merge into one real `linear1` GEMM)
IMPLEMENTED and FULLY LOCALLY VERIFIED (bit-for-bit cosine match
against the split path, not just syntax-checked) -- the only fusion
attempted this whole session that didn't need Thor to confirm
correctness, since it's plain scalar CUDA with no tensor-core/CUTLASS
dependency. Sub-problem 3 (linear2/attn_out+mlp_down merge) not
started. Two further candidates (gated-residual CUTLASS epilogue,
RMSNorm-prologue fusion) evaluated and both closed as "defer, don't
attempt" -- see their own sections below.

Area: "scheduler/graph alignment" work item, deprioritized in the
original top-level plan pending precision correctness -- picked back up
after Stage 3 closed (OPT-014). Scope chosen: systematically compare
FlashRT's real per-layer implementation against the REAL official
FLUX.2/ImageWAM forward pass (source read directly, not re-deriving
from kernel code the way every fusion so far was found), rather than
denoise-step-count reduction (that direction not started).

## Audit method and result

Forked a comparison of the real official op sequence
(`third_party/flux2/src/flux2/model.py`'s `DoubleStreamBlock`/
`SingleStreamBlock`, `action_dit_flux2.py`'s `SlimFlux2*Block`,
`imagewam.py`'s `infer_action_flux2`) against `pipeline_thor.py`'s
`_double_stream_layer`/`_single_stream_layer`/`_action_double_layer`/
`_action_single_layer`. First checked and ruled out one candidate before
reporting: AdaLN modulation sharing across layers is ALREADY correct --
the real model computes shift/scale/gate once per stream-type via one
`Modulation` linear, reused across all blocks in its loop
(`model.py:98-108,132-134`); `pipeline_thor.py:74-91` already does the
same (precomputed once by the caller, reused across 25 layers). Not a
gap, correctly not re-flagged.

Two real gaps found:

**Finding 1 (not started, larger scope)**: the real official
`SingleStreamBlock`/`SlimFlux2SingleBlock` fuse QKV+MLP-gate/up into
ONE `linear1` GEMM and attn-out+MLP-down into ONE `linear2` GEMM (over
`cat([attn_out, mlp_act(mlp)], dim=-1)`). FlashRT's own
`checkpoint_loader.py:114-133` (`_extract_single_block`) already
confirms the real checkpoint stores these as ONE tensor each --
FlashRT explicitly splits them into 4 separate GEMM slots at load time.
This split was a KNOWN, documented, deliberate simplification
(`real_single_stream_block.py`'s own docstring, written earlier this
project): `silu_glu_merged_fp16` needs its gate/up input's row stride
to equal its own width, which breaks for a column-slice of a wider
`linear1` output -- merging back needs new strided-aware kernel
variants at three spots (gate/up read, QKV read, `linear2`'s implicit
concat-sum), not a free relayout. Applies to 40 real layers (20
backbone single-stream + 20 ActionDiT single-stream); double-stream
blocks on both sides keep qkv/mlp genuinely separate in the real
checkpoint too, confirmed NOT a gap there. Deferred -- real compute/
bandwidth win, but bigger design+kernel investment than finding 2, not
started this round.

**Finding 2 (implemented this round)**: `nvfp4` (the actual default
precision since OPT-014) had NO fused SwiGLU path at all for MLP
gate/up -- unlike `fp16_cutlass` (`CutlassFp16SwiGluMlp`, OPT-013),
`nvfp4`'s MLP-gate slot fell through to the generic `Nvfp4Linear`
dispatch: one wide NVFP4 GEMM against the real merged `(2*mlp_hidden,
K)` weight, writing a full `(m, 2*mlp_hidden)` fp16 buffer, then the
plain `silu_glu_merged_fp16` kernel reads it and writes the `(m,
mlp_hidden)` gated buffer -- the exact "extra merged-buffer write+read"
pattern OPT-013 already eliminated for FP16 CUTLASS, still fully
present for the actual shipped default.

## Finding 2 implementation

New `Nvfp4SwiGluMlp` (`quant_linear.py`), wired into `_mlp_gate_up`
(`pipeline_thor.py`) and both real-weight loading (`imagewam_thor.py`'s
`_load_real_weights`, same `txt_mlp0.weight`/`img_mlp0.weight`/
`mlp0.weight`/`mlp_in.weight` slots OPT-013 already special-cases) and
random-weight construction (`_rnd_swiglu_mlp`) -- mirrors
`CutlassFp16SwiGluMlp`'s own construction-time column-split of the
real merged weight, but into two NVFP4-quantized `(mlp_hidden, K)`
halves (`quant_weight_nvfp4`) instead of one fp16 transpose.

Mechanism differs from FP16 CUTLASS's epilogue-fusion approach (this
codebase's NVFP4 GEMM doesn't expose an arbitrary activation epilogue
the way the CUTLASS EVT path does): two separate NVFP4 GEMMs
(`fp4out_gemm`/`FP4Buffer`, the "split-GU FFN path" building blocks
this codebase already had for a DIFFERENT model, never wired to
ImageWAM) each produce an `(m, mlp_hidden)` FP4-PACKED (4-bit, ~1/4 the
bytes of fp16) intermediate, then a new combiner kernel
(`silu_glu_two_fp4_to_fp16`, `csrc/fused_fp4/silu_mul_two_fp4_to_fp4.{cu,cuh}`)
reads both and writes the `(m, mlp_hidden)` fp16 gated buffer directly
-- no FP4 requantization needed since the down-projection GEMM right
after (unchanged) already re-quantizes its own fp16 input internally.
Net: the intermediate representation shrinks from a full-width fp16
merged buffer to two FP4-packed halves -- real DRAM-traffic reduction
on the intermediate (same mechanism this file's own module docstring
already documents for a different model: "reads HALF the activation
DRAM... vs fp16 today"), not just a launch-count change. Activation is
quantized to FP4 once per call and reused for both GEMMs (matches the
existing single-wide-GEMM path's own single activation-quant cost).

**Real bug found and fixed while implementing, before it could ship
silently wrong**: the existing `geglu_two_fp4_to_fp4`/
`silu_mul_two_fp4_to_fp4` (built for a different model's AWQ path) is
misleadingly named -- its own module docstring and Python wrapper
docstring both say "SiLU", and its device helper is even named
`silu_mul_p1`, but the ACTUAL formula
(`g/(1+exp(-1.5957691216057308f*g*(1+0.044715*g*g)))`) is the standard
GELU-tanh approximation (`1.5957691216057308 == 2*sqrt(2/pi)`), not
true SiLU (`g/(1+exp(-g))`, ImageWAM's own real formula,
`csrc/kernels/activation.cu`). Confirmed by reading the device code
directly, not trusting the name/comments -- would have silently
produced GELU-activated (wrong) outputs if reused as-is. Added a
genuinely new `true_silu_mul_p1` device function with the correct
formula instead of reusing the misnamed existing one.

## Verification status

Local (Ada, no NVFP4 build): Python syntax clean
(`py_compile`), full existing regression suite unaffected (fp16
default untouched), `precision="nvfp4"` construction still fails at
the same documented point (`Nvfp4Linear`/`Nvfp4SwiGluMlp` both raise
the same clear `RuntimeError` for a missing Blackwell build -- no new
crash introduced). New CUDA kernel (`silu_glu_two_fp4_to_fp16` in
`silu_mul_two_fp4_to_fp4.cu`) compiles cleanly via the same sm_90a
syntax-check substitute OPT-013 used for its own SM100-only kernel
(real object file produced, ~148KB) -- this specific kernel has no
tensor-core/MMA instructions (pure per-thread scalar math over packed
FP4 bytes), so it may actually be able to RUN on non-Blackwell
hardware if built for it, unlike the real FP4 GEMMs -- but the whole
`flash_rt_fp4` extension is gated behind `ENABLE_NVFP4`
(`GPU_ARCH=100`/`110` only, `CMakeLists.txt:43-60`), so this wasn't
pursued further locally; not claiming functional correctness, only
that the C++/CUDA is syntactically/semantically valid.

**Not yet done, real Thor verification needed**: cosine vs. the
existing `nvfp4` default (should be very close -- same math,
restructured) and vs. `fp16`; real Thor speed delta for the actual
shipped default precision (this is the one place in the whole session
where a speed change would directly affect the currently-deployed
default, not just an opt-in alternative -- verify before treating this
as a real win, same discipline as every other precision change this
session).

## Real Thor result -- correct, no speed win, REVERTED from the default

`GPU_ARCH=110`, `flash_rt_fp4` rebuilt, `silu_glu_two_fp4_to_fp16`
confirmed wired (all 55 real MLP-gate slots dispatch through
`Nvfp4SwiGluMlp`). Real conditions: `x0=513`, proprio on, real 10-step
shift schedule, real LIBERO dual-camera input.

**Correctness**: finite throughout.

| pair | actions | backbone_hidden | action_latent |
|---|---:|---:|---:|
| new fused `nvfp4` vs old (pre-fusion) `nvfp4` | 0.99987 | 0.99382 | 0.99975 |
| new fused `nvfp4` vs `fp16` | 0.99982 | 0.99303 | 0.99966 |
| old `nvfp4` vs `fp16` | 0.99981 | 0.99347 | 0.99966 |
| OPT-014's own `nvfp4` vs `fp16` (reference) | 0.9998 | 0.9939 | 0.9997 |

New-vs-old and old-vs-fp16 land at essentially the SAME distance from
`fp16` (0.99303 vs 0.99347, 0.99982 vs 0.99981) -- **no measurable
accuracy regression relative to the actual reference**; the
0.99382/0.99975 "new vs old" numbers are just two similarly-fp16-close
implementations differing from EACH OTHER, not from ground truth.

**Root cause of the new-vs-old `backbone_hidden` delta, analyzed
without Thor access (real Thor per-layer isolation not yet run to
confirm)**: checked and RULED OUT weight-quantization split-order as
the cause -- read `csrc/quantize/quantize_fp4_sfa.cu`'s real
`kernel_quantize_fp4_sfa` directly: "one thread per (row, 16-element
block)", every row's scale depends ONLY on that row's own 16 K-values,
no per-tensor/global scale anywhere in this kernel. Splitting a merged
`(2*mlp_hidden, K)` weight into two `(mlp_hidden, K)` row-groups before
quantizing is therefore mathematically IDENTICAL, bit-for-bit, to
quantizing the merged tensor as one block -- this cannot be the cause.
**More likely real cause**: the new path's gate/up intermediate is
FP4-PACKED (4-bit e2m1, 16 discrete magnitude levels per block) before
the combiner reads it, where the old path's intermediate was full fp16
(16-bit) the whole way through the elementwise combine -- a real,
structural, one-extra-lossy-quantization-step difference (the actual
cost side of this fusion's own bandwidth-for-precision trade), not a
combiner-formula bug (the formula itself was independently verified
correct against `activation.cu`'s real SiLU). Suggested follow-up
diagnostic if ever revisited: compare gate/up at a SINGLE isolated
layer (before any 25-layer compounding) between old and new paths --
already-low cosine at one layer would confirm the FP4-intermediate-
precision explanation; only compounding over many layers would instead
implicate GEMM-algorithm reduction-order non-associativity (the same
class of benign effect OPT-013 already found and accepted for
`fp16_cutlass`'s own `backbone_hidden=0.995090`). Not investigated
further -- moot given the speed result below.

**Speed**: no measured win.

| | `infer()` P50 |
|---|---:|
| new fused `nvfp4` | 244.8 ms |
| old (pre-fusion) `nvfp4`, same commit/build | 244.2 ms |
| OPT-014's own `nvfp4` measurement | 236.9 ms |
| OPT-014's own 40-call stability band | 243.2-247.3 ms |

Fused path is 0.6ms SLOWER than the unfused path on the same commit --
noise-level, and both land inside OPT-014's own already-measured
stability band. Shrinking the gate/up intermediate from a wide fp16
buffer to two FP4-packed buffers did not translate into a measurable
`infer()`-level win at these real shapes (M=513/392/64) -- plausibly
because two separate FP4 GEMM launches + a new combiner kernel roughly
offset whatever bandwidth was saved, at shapes this small.

**Decision: REVERTED from the default.** Unlike `fp16_cutlass`
(OPT-013, always an opt-in tier, never the default), this fusion was
wired directly into the actual shipped `nvfp4` default -- carrying it
forward would mean shipping extra representation risk (correctness is
fine here, but the class of risk is real, see the root-cause
discussion above) for zero measured benefit. `_load_real_weights` and
`_rnd_swiglu_mlp` (`imagewam_thor.py`) reverted to the plain merged-
GEMM path for `nvfp4` (same as before this entry). `Nvfp4SwiGluMlp`
and `silu_glu_two_fp4_to_fp16` stay in the codebase (correct, real
Thor-verified, documented) but are not wired to any precision string --
available if a future shape mix (larger M, different mlp_hidden ratio)
ever makes the bandwidth trade actually pay off, same "kept but not
adopted" treatment as `fp16_cutlass`'s own CUTLASS FP16 tile variants.

## Candidate: gated-residual CUTLASS epilogue fusion -- evaluated, DEFER

`gate_res_fp16`/`gate_res_bf16res` (`csrc/kernels/decoder_fused.cu`)
computes `residual += gate_vector * proj_output` (per-channel `gate`,
not a scalar), currently a standalone elementwise kernel reading
`proj_output` right after the preceding down-projection GEMM writes it
to DRAM. Folding this into that GEMM's own epilogue was flagged in
OPT-013 as a real candidate, deliberately not designed there. Forked a
feasibility assessment (read-only): CUTLASS *does* have the right
semantic building block, `PerColLinCombPerColBiasEltAct`
(`third_party/cutlass/include/cutlass/epilogue/fusion/operations.hpp:302-317`,
`D = activation(per-col alpha*acc + per-col beta*C + per-col bias)` --
exactly this math with `alpha=gate, beta=1, bias=0`), but it is only
wired up for SM90 (Hopper) in this vendored CUTLASS snapshot
(`sm90_callbacks_tma_warpspecialized.hpp`) -- SM100 (Thor's real
target, the same `cutlass_fp16_k64_*` family OPT-013 uses) has
`FusionCallbacks` specializations only for the FP4/FP8 block-scale-
factor variants, nothing for a plain per-column accumulate-into-C.
Would need genuine new SM100 epilogue-visitor authoring with no
existing template in this codebase to copy from (unlike every fusion
actually shipped this session, which reused existing kernel
infrastructure). Quantified target: eliminating `proj_output`'s own
DRAM round-trip is a SMALLER buffer (`hidden`-width, one tensor) than
either OPT-013's CUTLASS swap or OPT-015 finding 2's `Nvfp4SwiGluMlp`
(`2*mlp_hidden`-width) -- both of which were real, correct, and showed
**zero** measured `infer()` win despite bigger targets and lower
authoring risk. **Verdict: defer, do not attempt** -- worse expected
value than either already-negative precedent, given OPT-004's own
compute-bound finding.

## Candidate: RMSNorm+modulate to QKV-GEMM prologue fusion -- evaluated, DEFER

A new candidate (not from the original audit): fuse the norm+modulate
step (`ada_layer_norm_fp16`/`ada_layer_norm_bf16in_fp16out`, already
one kernel) into the FOLLOWING GEMM's own PROLOGUE (input side),
eliminating the `modded` buffer's DRAM round-trip entirely -- the one
fusion axis nothing this session has tried (every fusion so far is
either elementwise-kernel bundling or GEMM EPILOGUE/output-side
fusion). Forked a feasibility assessment: CUTLASS's vendored epilogue
visitor tree is entirely output-side (`gemm/collective/` has no
prologue/input-visitor concept at all, confirmed by grep). Worse,
this is architecturally awkward regardless of tooling: RMSNorm needs a
full row's reduction (sum-of-squares across all of K) BEFORE any of
that row can be scaled, which fights CUTLASS's own tile-at-a-time
mainloop streaming -- a real fused prologue would need a genuinely
custom two-pass mainloop, a bigger, more novel piece of authoring than
even the deferred gated-residual epilogue above (which at least
extends CUTLASS's existing, working EVT pattern). Quantified savings
(real shapes, all real per-layer call sites, full `infer()`): ~340-470
MiB of DRAM traffic in the best case -- SMALLER than OPT-015 finding
2's own already-measured non-win, and single-stream layers can't even
fully realize it without ALSO merging `linear1` (their one `modded`
buffer is read by two separate GEMMs today). **Verdict: defer
indefinitely, not "pending Finding 1"** -- worse cost/benefit than
either open candidate, not worth carrying forward as a live candidate.

## Finding 1, sub-problem 1: qkv+mlp_in merge -- IMPLEMENTED, fully locally verified

Unlike every other precision/kernel change this session (all gated
behind Thor-only CUTLASS/NVFP4 builds, verifiable locally only via an
sm_90a syntax-check substitute at best), this fusion is plain scalar
CUDA with zero tensor-core dependency -- genuinely, numerically
verifiable on this Ada dev machine, not just syntax-checked.

**What changed**: the real official single-stream blocks
(`SingleStreamBlock`/`SlimFlux2SingleBlock`) run QKV and MLP-gate/up
as ONE real `linear1` GEMM; `checkpoint_loader.py`'s
`_extract_single_block` used to always split it into separate
`qkv.weight`/`mlp_in.weight` slots (a deliberate historical
simplification, `real_single_stream_block.py`'s own docstring,
because `silu_glu_merged_fp16` hardcoded its input's row stride to its
own width). Fixed the actual blocker directly: `silu_glu_merged_kernel`
(`csrc/kernels/activation.cu`) gained a `row_stride` parameter
(defaults to `half_dim*2`, every existing caller's own tightly-packed
layout, unchanged) so it can read gate/up straight out of a
column-slice of a WIDER buffer -- confirmed bit-exact against a plain
torch reference for both the default (unchanged) and wide-stride
(new) cases, not just cosine-close.

`_extract_single_block`/`build_real_weights` gained a
`merge_qkv_mlp: bool` param: `True` (every precision except
`fp16_cutlass`, which keeps its own separate `mlp_in.weight`-based
`CutlassFp16SwiGluMlp` mechanism unchanged) returns the real, UNSPLIT
`linear1.weight` under one key instead of splitting it.
`imagewam_thor.py` sets `self.dims["merge_qkv_mlp"] = precision !=
"fp16_cutlass"` once at construction, threads it through
`_load_real_weights`/`_alloc_random_weights`/the buffer-allocation
dict (new `single_linear1_merged`/`action_linear1_merged` buffers,
always allocated alongside the old split ones -- a small, ~50MB
combined memory overhead accepted for zero code-complexity cost, so
`fp16_cutlass`'s own unmerged path needs no special handling).
`pipeline_thor.py`'s `_single_stream_layer`/`_action_single_layer`
branch on `dims.get("merge_qkv_mlp")`: the merged path runs ONE
`linear1.weight` GEMM, reads Q/K/V via the existing `_copy_slice`
column-slice technique (already used for the QKV-only merge that
predates this entry), and calls the now-stride-aware
`silu_glu_merged_fp16` directly on the mlp-gate/up column range --
`_mlp_gate_up`'s own separate-GEMM path is skipped entirely for this
case (no `mlp_in` GEMM exists any more, nothing to skip TO). `linear2`
(attn_out_proj+mlp_down, sub-problem 3) is completely unchanged.

**Verification**: two new tests
(`tests/test_imagewam_thor_real_wiring.py`:
`test_single_stream_layer_merged_linear1_matches_real_reference`,
and a merged-linear1 check appended to
`test_action_double_and_single_layers_match_real_reference`) build the
SAME real weight VALUES both ways (split into `qkv`/`mlp_in`, and
concatenated into one `linear1`) and compare the merged pointer-path
output against the SAME already-verified tensor-level reference used
for the split path. Result: cosine=0.999998 (backbone single-stream)
and cosine=1.000000 (ActionDiT single-stream) -- **identical to the
split path's own numbers**, not just "close". Full existing regression
suite (`test_imagewam_frontend`, `test_imagewam_proprio`,
`test_imagewam_checkpoint_loader` real 343-tensor real-checkpoint
load, `test_imagewam_scheduler`) passes unchanged.

## Real Thor result: a genuine measured win -- the first this session

Real conditions: `nvfp4` (shipped default), `x0=513`, proprio on, real
10-step shift schedule, real LIBERO dual-camera input. Confirmed
wired: 20+20 real `linear1.weight` slots (vs. the old split path's
40+40 `qkv.weight`/`mlp_in.weight` slots).

**Correctness (merged vs. old split path, same commit)**:

| | actions | backbone_hidden | action_latent |
|---|---:|---:|---:|
| merged vs split | 0.999983 | 0.994464 | 0.999964 |

All finite. `actions`/`action_latent` confirm the two paths are the
same math end to end. `backbone_hidden`'s 0.9945 (vs. the LOCAL FP16
single-layer check's exact 0.999998) is expected, not a bug: the local
check compared ONE layer at FP16 (no quantization in the picture); this
is `nvfp4` accumulated across the REAL 25-layer stack, where the wide
`linear1` GEMM (`N=27648`) and the two separate GEMMs it replaces hit
different CUTLASS/quantization reduction paths -- the same class of
benign per-layer floating-point/representation drift already
documented for `fp16_cutlass` (OPT-013, `backbone_hidden=0.995090`)
and `Nvfp4SwiGluMlp` (OPT-015 finding 2, `0.9938`). Confirmed not a
stride bug specifically: a wrong `_col_ptr`/`row_stride` read would
show up as a much larger `actions` error, not 0.99998.

**Speed -- a real win, first one this session**:

| | merged (default) | split (old) | delta |
|---|---:|---:|---:|
| VAE | 21.17 ms | 21.13 ms | none |
| backbone prefill | 106.52 ms | 107.11 ms | +0.59 ms |
| ActionDiT 10-step denoise | 122.73 ms | 130.71 ms | **+8.0 ms** |
| `infer()` P50 | **231.63 ms** | 239.55 ms | **+7.9 ms (1.03x)** |

231.6ms beats OPT-014's own `nvfp4` baseline (236.9ms) and its 40-call
stability average (243.5ms) -- a real, measured improvement to the
actual shipped default.

**Counter-intuitive finding, worth recording plainly**: this entry's
own prediction (based on OPT-004's compute-bound-vs-launch-bound
framing) was backwards. Backbone prefill (large M=905, where removing
one GEMM's real DRAM traffic should matter most) shows essentially
ZERO win (+0.59ms). ActionDiT's denoise loop (tiny M=64, 20 layers x
10 steps = 200 calls) captures nearly the ENTIRE win (+8.0ms). The
likely reason: at backbone's large M, the two GEMMs being merged were
already comfortably compute-dominated, so removing one launch barely
registers against Thor's fast HBM/compute throughput; at ActionDiT's
tiny M, each GEMM call's cost is dominated by FIXED per-launch
overhead rather than the actual FLOPs, so eliminating one launch per
layer, replayed 200 times per `infer()`, adds up to a real, measurable
amount even though each individual saving is small. This inverts the
"backbone matters more" assumption this entry started with -- if
`linear1`'s merge were ever partially reverted, ActionDiT's side is
the one to keep, not backbone's.

**Decision: keep the merge for the whole network** (already the
shipped default going forward). Sub-problem 3 (`linear2` merge,
attn_out+mlp_down) still not started -- given this result's own lesson
(the win came from launch-count at ActionDiT's small-M denoise loop,
not backbone bandwidth), sub-problem 3's own ActionDiT-side benefit
now looks MORE promising than this entry originally estimated (it also
removes GEMM launches from the same 200-call-per-`infer()` denoise
loop), while its backbone-side benefit should be expected to stay
small, matching this measurement's own pattern.

## Benchmark-script consolidation (2026-09-17)

All 5 standalone per-precision speed benchmarks
(`imagewam_thor_{fp16,fp8,fp4,int8,int4}_bench.py`) turned out to be
equally stale -- a generic transformer-block skeleton (plain
`rms_norm_fp16`, `gelu_inplace_fp16` on a single-width MLP buffer,
plain `residual_add_fp16`) with none of this session's real fusions
(AdaLN modulation, gated residual, real merged SiLU-GLU MLP) or even
the pre-existing real fused QKV, plus a stale 768-token image-shape
placeholder.

`fp16`/`fp8`/`nvfp4` already have a real, correct, up-to-date
implementation reachable through the real `ImageWAMTorchFrontendThor`
frontend (used directly for every real Thor measurement this whole
session) -- rewriting standalone copies of the same math would be
pure duplication. **Deprecated** (docstring notice added, not
deleted, kept for historical reference; internal logic untouched):
`imagewam_thor_fp16_bench.py`/`_fp8_bench.py`/`_fp4_bench.py`.
**`imagewam_thor_graph_bench.py`** (already used the real frontend)
is now the canonical replacement: real current dims (`x0=513, a0=905,
num_action=64`), loops over every string in `imagewam_thor._PRECISIONS`,
prints `SKIP` for any precision this build doesn't support (e.g. every
Blackwell-only precision on this dev machine) instead of crashing the
whole run.

INT8/INT4 (SM80 CUTLASS) have NO real dispatch path in
`imagewam_thor.py` at all (never wired in, closed for Thor on speed
grounds per this entry regardless) -- their standalone benchmarks were
**rewritten** (not deprecated) to match `pipeline_thor.py`'s current
real per-layer math exactly: real AdaLN modulation, real fused QKV,
real fused `linear1` for single-stream blocks (today's own op-fusion
audit finding 1), real merged SiLU-GLU MLP, real gated residual, real
`txt_in`-once-per-forward, and real dims. Also found and fixed, beyond
the rewrite's own original scope: both scripts were constructing
`ImageWAMAttnBackend` WITHOUT `use_perhead_kv=True, use_real_mot_mask=True`
and allocating K/V caches at the old broadcast `HD` width instead of
real per-head `HIDDEN` width -- a second, independent staleness
predating OPT-002, unrelated to the AdaLN/gated-residual gap. Verified
by direct code review (structural diff between the two rewritten files
confirms identical restructuring, differing only in the expected
INT4/INT8-specific kernel details) and local syntax compilation. Also
run end to end once, locally, before local GPU testing was paused for
this round: INT4 completed with finite outputs (`prefill`
P50=111.8ms, `one denoise step` P50=7.6ms -- both HIGHER than the
pre-rewrite stale version's own numbers, expected, since the real
AdaLN/gated-residual/fused-QKV/fused-linear1 op sequence is
structurally more work per layer than the old generic-skeleton
approximation it replaced, an apples-to-oranges comparison, not a
regression); INT8 failed exactly at the already-documented, expected
K=9216 shape (`txt_mlp2`, rc=131079) -- the same real Ada limitation
this entry already tracks, reproducing precisely, confirming the
rewrite introduced no new bug there either.

## Real Thor result: consolidated benchmarks, and a real production bug found+fixed

Ran the consolidated benchmarks (above) on real Thor. Note these are
random-weight/graph-capture speed tools, NOT the real-checkpoint
`infer()` path -- not directly comparable to OPT-014's own 236.9ms
(no VAE, no proprio, random weights).

**Real bug found**: `imagewam_thor_graph_bench.py` run as-is hit 3 of
6 precisions `SKIP`ping -- `fp8`/`fp8_static`/`fp8_static_cutlass` all
crash on the real `action_encoder` (K=7) shape (cuBLASLt returns
status 15/`CUBLAS_STATUS_NOT_SUPPORTED`; CUTLASS returns
`can_implement=-1`) -- the exact same K=7/N=7 misalignment class
`fp16_cutlass` (OPT-013) and `nvfp4` (Stage 3 checklist) already hit
and got a `_wrap_linear` fallback for, except NOBODY added the same
fallback for the FP8 family. **This is a real, live production gap**:
selecting `precision="fp8"`/`"fp8_static"`/`"fp8_static_cutlass"` in
`imagewam_thor.py` today would crash `set_prompt()`'s graph capture
exactly like the pre-fix `fp16_cutlass`/`nvfp4` cases did. Notably
different from those two: FP8 fails on **cuBLASLt too**, not just
CUTLASS -- `fp16_cutlass`/`nvfp4`'s own fallback exists because
cuBLASLt tolerated the misaligned shape and only their CUTLASS-
specific path didn't; FP8 needs the fallback on every backend for this
precision family. **Fixed**: added the same `n%8!=0 or k%8!=0 ->
Fp16Linear` fallback to all three FP8 branches in `_wrap_linear`
(`imagewam_thor.py`) -- centralized there, so it covers both
`_load_real_weights` and `_alloc_random_weights` (`_rnd_linear` already
routes through `_wrap_linear`) automatically. `_calibrate_fp8()`
already skips non-`StaticFp8Linear` objects via `isinstance`, so the
fallback needs no other special-casing. NOT yet regression-tested
locally (local GPU testing paused this round) -- needs the same
sanity check the `fp16_cutlass`/`nvfp4` fixes got (construct with each
precision, confirm graph capture completes) before being trusted.

**Real Thor speed, `imagewam_thor_graph_bench.py`, x0=513/a0=905/
num_action=64, no VAE/proprio, all 6 precisions (after the fix above)**:

| precision | P50 (ms) |
|---|---:|
| fp16 | 286.3 |
| fp16_cutlass | 280.8 |
| fp8 | 254.2 |
| fp8_static | 243.2 |
| fp8_static_cutlass | 221.4 |
| **nvfp4** | **212.0** |

`nvfp4` remains fastest, consistent with every other measurement this
session. Since this tool has no VAE/proprio, its own absolute numbers
aren't comparable to OPT-014's real-`infer()` figures, but the
RELATIVE ordering across precisions matches.

**Real Thor speed, rewritten `imagewam_thor_int4_bench.py`/
`_int8_bench.py`** (real AdaLN/gated-residual/fused-QKV/fused-linear1,
real dims):

| | INT4 | INT8 |
|---|---:|---:|
| prefill (VAE stub + 25L) | 1253.6 ms | 138.7 ms |
| one denoise step (25L) | 55.4 ms | 14.1 ms |
| prefill + 10-step | 1807.6 ms | 279.3 ms |

Both numbers are higher than the pre-rewrite stale scripts' own
(expected -- more real work per layer now, not a regression, see the
consolidation entry above). **INT8's real news**: Ada's K=9216 crash
does NOT reproduce on Thor -- the full pipeline (including VAE)
completes cleanly there, confirming that limitation really is
Ada-specific, not a property of the kernel family, exactly as this
entry's own real-Thor INT8 finding already established. Still not a
speed win vs. `fp16`/`nvfp4` in this same tool, and INT4 remains
dramatically slower on Thor (same ISA-mismatch root cause already
confirmed). **OPT-007 stays closed** -- this round only aligned the
measurement tools with reality, it does not reopen the INT8/INT4-on-
Thor question.

# OPT-028: ImageWAM through `frt_model_runtime_v1` (Python producer)

Status: implemented and verified on H100 (fp16, real checkpoint,
bit-exact). Thor `nvfp4` parity pending on the Thor checklist.

Area: deployment engineering, roadmap item 12 (`plan.md` "Plan: ABI
integration, `frt_model_runtime_v1` Python producer").

## Observation

The ImageWAM Thor frontend was reachable only through `set_prompt()` /
`infer()`. No runtime export existed, and the initial action noise was
drawn inside `infer()` (`0.01 * N(0,1)`, ISSUE-002) rather than being
an input.

## Opportunity

`ImageWAMTorchFrontendThor.export_model_runtime(io="python")`
publishes the captured graph and its buffers through the generic ABI,
with the same Python-producer construction path Pi0.5 uses
(`flash_rt.runtime.export.build_model_runtime`). Port schema and
ownership: `docs/imagewam_model_runtime.md`. The noise is an explicit
SWAP input consumed as written; the `0.01` factor stays inside `infer()`.

## Expected Mechanism

No numerics change: the verbs call the frontend's own staging methods
(`stage_images`, `stage_proprio`, `read_actions`, `set_prompt`), which
`infer()` now also calls, and `step` replays the same instantiated graph
exec through `frt_graph_replay`. Parity is bit-exact by construction and
measured, not assumed.

## Required Evidence

H100 (shared GPU), `tests/gate_imagewam_model_runtime_export.py
--precision fp16`, real checkpoint + VAE + Qwen3 + dataset stats,
LIBERO spatial episode 0 frame 0, ctypes consumer vs `infer()`, same
seed:

| check | array_equal | max_abs |
|---|---|---:|
| control: `infer()` vs `infer()` | True | 0 |
| `images` STAGED → `image_tokens` window | True | 0 |
| `actions` (denormalized, STAGED) | True | 0 |
| `actions_raw` (normalized, SWAP) | True | 0 |
| `image_tokens` SWAP path → `actions` | True | 0 |
| `prompt` SETUP (second task) → `actions` | True | 0 |

The second task string moves the chunk by `max_abs = 0.0821`, so the
prompt check is not vacuous. Peak GPU memory 17.2 GiB.

Latency, H100 shared with a co-tenant at 100% utilization, indicative
only, alternating A/B, 20 iterations each, wall time including VAE,
proprio staging and host readback:

| path | P10 | P50 | P90 |
|---|---:|---:|---:|
| `infer()` | 133.7 ms | 142.5 ms | 152.8 ms |
| ABI tick (`images` + `proprio` + `noise` + `step` + `actions`) | 127.8 ms | 138.5 ms | 153.1 ms |

Small random-weight dims (`tests/test_imagewam_model_runtime_export.py`,
5 tests): schema, identity sensitivity, guards (`-3`, `-1`, `-5`) and an
`array_equal` tick. Regression: `pytest tests/test_imagewam_*.py` 73
passed, 6 skipped (baseline 68/6 plus these 5).

Thor, `nvfp4`: pending (Thor checklist).

## Promotion Condition

Thor gate at `nvfp4` reports every parity row `array_equal=True`. The
export is additive and opt-in; `infer()` behavior is unchanged.
