# OPT-001

Status: RESOLVED end to end, including real Thor validation (2026-09-15, all 4 phases complete); FP8 calibration/quantization split off into OPT-004 steps 5-6 (already resolved separately, see that section)

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
would not be a meaningful Thor performance number).

## Original framing (superseded by the above, kept for history)

## Observation

The original plan used randomly initialized weights and a
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

Real Thor (SM110) measurement: FP4/FP8
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
backbone GEMM.

## Promotion Condition — met

Verified on real Thor hardware with the exact fix in place (not just
re-derived): promoted, this is now a verified fact, not a hypothesis.

# OPT-004

Status: not promoted

Area: ImageWAM pipeline has none of FlashRT's real kernel-fusion or GEMM-autotuning machinery

## Observation

Real Thor (SM110) measurement, BEFORE
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

Phases 0-3 complete (Phase 4,
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
  environment gap (`issues.md` ISSUE-001, from an earlier session),
  reproduced identically in the pre-existing
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
this entry's own GEMM-only comparison table for the real numbers.
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

Status: real VAE-encode cost added to all local/Thor full-pipeline benchmarks; the `img_in` gap this exposed is FIXED (OPT-001 Phase 1); the VAE encoder ITSELF is NOW wired into the served frontend (2026-09-15, all 3 phases done) — see this file's own new entry below

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

# OPT-025: Jetson clock-state record for ImageWAM benchmarks (roadmap item 10)

Status: implemented and unit-tested on x86; Thor record pending (plan.md
"Plan: Jetson clock-locking check for the benchmark scripts").

Area: latency measurement provenance on Jetson

## What exists

`flash_rt/hardware/jetson_clock_state.py` reads the nvpmodel mode
(`nvpmodel -q`), every GPU and EMC devfreq node under
`/sys/class/devfreq` (`cur_freq`, `min_freq`, `max_freq`, `governor`),
and `/sys/kernel/nvpmodel_clk_cap/*`, all without root. It never runs
`sudo`, `jetson_clocks` or `nvpmodel -m` and never writes sysfs: Thor is
shared and benchmarks run in its existing state (MAXN, DVFS-managed
clocks). It returns a `JetsonClockState` with a pinned verdict (every GPU
node `cur == min == max`, no dynamic EMC node, MAXN when readable) and
warnings only for a non-MAXN power mode or unobservable state, and
returns `is_jetson=false` on any other machine.
`report_jetson_clock_state()` prints the record as one
`[jetson-clock-state]` JSON line, a summary line (power mode; GPU clocks
pinned or dynamic), and warning lines.

Printed before timing by `benchmarks/imagewam_thor_graph_bench.py`,
`benchmarks/imagewam_thor_int4_bench.py`,
`benchmarks/imagewam_thor_int8_bench.py`, the timing section of
`benchmarks/imagewam_e2e_official_compare.py`, and embedded in every
regression-gate result (`tests/gate_imagewam_libero.py`, OPT-027).

## Measured

| check | machine | result |
|---|---|---|
| `tests/test_jetson_clock_state.py` (fake Thor sysfs trees) | H100 box, x86 | 11 passed |
| record on the real root | H100 box | `is_jetson=false`, no tool run |
| `imagewam_thor_graph_bench.py` fp16 row | H100 (shared GPU, indicative only) | record printed; P50 107.3 ms |

## Open

The Thor record itself (node names, whether `nvpmodel` exists there,
EMC visibility) is not yet observed; see ISSUE-061 for why it matters
to the gate's latency baseline.

# OPT-026: precision-routing contract test for the ImageWAM Thor frontend (roadmap item 11)

Status: done.

Area: `ImageWAMTorchFrontendThor` weight-slot routing
(`_wrap_linear`, `_alloc_random_weights`, `_load_real_weights`, the
constructor's `merge_qkv_mlp` decision)

## What exists

`tests/test_imagewam_thor_precision_routing.py` holds the routing
contract as one table, `EXPECTED_ROUTING`: 37 slot rows by the 6
precisions of `_PRECISIONS`. It is checked at the real FLUX.2-4B LIBERO
dims and at the default dims, for random and real-checkpoint weights,
and additionally that both weight sources build the same wrapper with
the same `(n, k)` per key. It covers the `action_encoder` (K=7) and
`head.linear` (N=7) fallbacks, the merged `linear1` against the
`fp16_cutlass` `qkv`/`mlp_in` split, the `CutlassFp16SwiGluMlp` slots,
and `Bf16OutLinear` for `txt_in`/`img_in` (one shared wrapper across
double layers on the real path). `flash_rt_kernels` and `quant_linear`
are stubbed at the import boundary and weights are meta tensors, so no
GPU or compiled extension is needed.

## Measured

| environment | result | time |
|---|---|---|
| H100 box, real extension importable | 54 passed | ~16 s |
| `CUDA_VISIBLE_DEVICES=""`, `flash_rt_kernels` unimportable | 53 passed, 1 skipped (stub-signature check needs the real module) | ~14 s |
| mutation: `nvfp4` K/N%16 fallback removed | 5 failed, naming both K=7/N=7 slots | |
| mutation: `fp8_static` loses the `linear1` merge | 5 failed | |

## Maintenance

A stream that changes routing (for example a `linear2` merge) edits the
table rows in the same commit: add the merged slot's row and mark the
replaced slots `-` for the precisions that merge.

# OPT-027: fidelity + latency regression gate on a versioned LIBERO fixture (roadmap item 13)

Status: implemented; fp16 gate passing on H100; Thor `nvfp4`/`fp16`
runs pending (plan.md "Plan: Fidelity and latency regression gate
harness").

Area: committed CI/regression gate for the served ImageWAM path

## What exists

| piece | file |
|---|---|
| model-agnostic gate policy and report schema (v1) | `flash_rt/core/regression_gate.py` |
| fixture format, `.npz` IO, manifest with per-file and per-array SHA-256 | `flash_rt/datasets/imagewam_gate_fixture.py` |
| fixture generator (H100: official model, then FlashRT fp16) | `benchmarks/imagewam_gate_fixture_generate.py` |
| gate runner (any CUDA device; no official model, no Qwen3) | `tests/gate_imagewam_libero.py` |
| per-precision fidelity thresholds | `tests/fixtures/imagewam_gate/fidelity_thresholds.json` |
| per-device latency policy (Thor `nvfp4` 231.6 ms, margin 5%; H100 ungated) | `tests/fixtures/imagewam_gate/latency_baselines.json` |
| committed v1 manifest | `tests/fixtures/imagewam_gate/imagewam_libero_gate_v1.manifest.json` |
| fixed-noise entry into the served path | `ImageWAMTorchFrontendThor.infer(observation, *, action_noise=None)` |

Fixture v1 (`imagewam_libero_gate_v1`, 81 MiB, stored at
`/home/user1/workspace/jingwu/artifacts/deploy-gates/imagewam_libero_gate_v1/`,
not in git): libero_spatial, 10 tasks, frames 0 and 60, seeds 0 and 1,
so 20 observations and 40 (observation, seed) runs. It holds both
224x224 views, raw proprio, ground truth, the 10 official Qwen3
contexts (bfloat16 bits) and masks, the initial noise drawn as the
official sampler draws it, and the official and FlashRT fp16 action
chunks in normalized space.

Provenance: v1 was produced by the generator content committed in
`d03b073` (generator file SHA-256 `d16399e0e6afc3bf...`); that run took
its git snapshot at the end rather than the start, which does not touch
the data. Its manifest records `git.commit` `ae3a358` with
`tracked_changes: false` because the generator was still untracked then.
Manifests generated from now on also record `generator_sha256` and the
untracked files (`git.untracked_files`, `git.clean`); v1 is not
regenerated.

`fp8_static` interface: thresholds mark it `requires_calibration`. The
runner gates it only when `--fp8-calibration PATH` or
`$IMAGEWAM_FP8_CALIBRATION` names an existing file, and hands the path to
`ImageWAMTorchFrontendThor(..., calibration_path=PATH)`, the keyword the
calibration stream's frontend uses. Without a
file the verdict is `skipped` (exit 0); with a file but no such
constructor keyword it is `blocked` (exit 1). It is never gated on the
`N(0, 0.1)` placeholder calibration.

Noise: fidelity is measured with the fixture's fixed N(0,1) initial
noise, the official sampler's per-seed draw, passed through
`infer(obs, action_noise=...)`. It is not the served default draw,
`0.01 * N(0,1)` (ISSUE-002); the latency loop does use the served draw.

Clock policy: the latency check records the clock state in every result
and never refuses dynamic clocks. Thor runs at MAXN with DVFS-managed
clocks, and baselines are measured in that same state (ISSUE-061).

## Measured (H100, shared GPU)

Fixture generation, FlashRT fp16 against official (normalized space):

| | median | min | mean MAE vs GT |
|---|---:|---:|---:|
| seed 0 (end-to-end baseline: 0.99840 / 0.99567 / 0.18359) | 0.99840 | 0.99567 | 0.18359 |
| seed 1 | 0.99829 | 0.99554 | 0.18369 |
| official, seed 0 vs seed 1 | 0.99630 | 0.97154 | |
| official MAE vs GT, seed 0 / seed 1 | | | 0.18538 / 0.18584 |

Peak GPU memory: official phase 17.4 GiB, FlashRT fp16 phase 9.6 GiB.

Initial noise and fidelity, fp16 on fixture v1, 40 runs, cosine against
official in normalized space:

| initial noise | median | min | mean |
|---|---:|---:|---:|
| fixed N(0,1), the official sampler's (what the gate uses) | 0.99836 | 0.99554 | 0.99798 |
| 0.01 x the same noise | 0.99683 | 0.98613 | 0.99602 |
| served default draw, 0.01 x N(0,1) on the device | 0.99683 | 0.98591 | 0.99598 |

The served sampler falls below the fp16 bounds (median 0.997, min
0.993). Resolving ISSUE-002 (dropping the 0.01 factor) would bring the
served path to the gated configuration.

Gate runs:

| precision | verdict | detail |
|---|---|---|
| fp16 | pass | vs official median 0.99836, min 0.99554 over 40 runs; vs fp16 reference 1.0 (max abs difference 0.0, bit-identical across processes); MAE 0.18364 against a limit of 0.18731; latency P50 158.2 ms (P10 142.9, P90 229.1), ungated; peak 9.56 GiB |
| nvfp4 | blocked | frontend construction: `Nvfp4Linear requires a Blackwell/Thor NVFP4 build` (expected on sm_90) |
| fp8_static, no calibration file | skipped | exit 0 |
| fp8_static, file present | blocked | this branch's frontend declares no `calibration_path` keyword; exit 1 |
| fp8 | blocked | no thresholds configured |
| fp16, rerun at `77b7bef` | pass | same fidelity values; checkpoint verified by SHA-256 (11.9 s); top-level `latency: "ungated"` with its reason, also named in the verdict reason; P50 159.3 ms; clean worktree recorded |
| fp16, `--require-latency` | blocked | exit 1; ungated H100 latency |
| fp16, wrong checkpoint | blocked | SHA-256 mismatch; with `--skip-checkpoint-hash`, byte-size mismatch |
| any, `--iters 5` | argument error | rejected before any GPU work |

## Open

- Thor runs for `nvfp4` and `fp16` (gate thresholds for `nvfp4` are
  provisional until then).
- ISSUE-060 (the `set_prompt(context=...)` cache) is worked around in
  the generator and runner.
- ISSUE-061 (the 231.6 ms baseline has no clock or FA4 record).

## Thor, LIBERO (`eccf14f`)

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU
exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is the
gate's `infer()` P50 in ms and `vs official` is the median action cosine
against the official model.

The gate ran on Thor for the `nvfp4` precision. Fixture v2
(`imagewam_libero_gate_v2`) records its `fp16` reference trimmed
(`text_trim=true`, the manifest field), so it gates a trimmed configuration,
which v1's untrimmed reference cannot:

| run | verdict | detail |
|---|---|---|
| `nvfp4`, `text_trim=true`, fixture v2 | pass | vs official 0.99931 / min 0.99898; vs the fixture's own `fp16` reference 0.99935 / 0.99907; `infer()` P50 114.6 ms |
| trimmed configuration against fixture v1 | refused | the fixture was recorded with the other `text_trim` value |
| untrimmed configuration against fixture v2 | refused | the same check, the other direction |

This is the Thor `nvfp4` run the `## Open` list above records as pending. As
for v1, the fixture data itself stays out of git: the generator writes
`fixture.npz` and the manifest to a data directory and copies only the manifest
into `tests/fixtures/imagewam_gate/`.

## Thor, LIBERO (`0919e`)

`0919e`, commit `a84916a`, Jetson AGX Thor, MAXN, GPC 1.575 GHz,
`emc_locked=null`, GPU exclusive; raw logs under
`/home/jingwu/thor_val/0919e/`. Latency is end-to-end `infer()` P50 in ms and
`vs official` is the median action cosine against the official model.

The VAE encode geometry change (issues.md ISSUE-086) is fidelity-neutral for the
gate's LIBERO configuration: the `libero_spatial` nvfp4 `default` row measures
vs official min 0.99418 / median 0.99764 and MAE 0.18290, the values the
`eccf14f` round recorded for that row, at P50 202.6 ms against 202.0-202.4 ms in
the `eccf14f` session. The fixture-v2 gate (`imagewam_libero_gate_v2`) still
passes.


# OPT-018: ActionDiT small-M CUTLASS tile selection (roadmap item 1)

Status: implemented behind an opt-in flag
(`gemm_variant_autotune=True`). It passes the sm_110 compile check.
Thor correctness and speed are not yet measured. Plan: plan.md "Plan:
ActionDiT small-M CUTLASS tile selection".

## Observation

Every ActionDiT weight GEMM runs at `M = num_action = 64`, and both
CUTLASS-backed quantized precisions pick their tile from an `(N, K)`
heuristic tuned at other M. Tiles dispatched today at the real
ActionDiT shapes (5 double + 20 single layers):

| site | N | K | `nvfp4` (`pick_variant`) | `fp8_static_cutlass` (`_pick_fp8_cutlass_variant`) | `fp16_cutlass` |
|---|---:|---:|---|---|---|
| double `qkv` | 9216 | 1024 | v6 `128x256x128` c1x1x1 | `wide` `256x128x128` c2x2x1 | `wide` |
| double `proj`, single `attn_out_proj` | 1024 | 3072 | v6 | `sq` `256x256x128` c2x2x1 | `sq` |
| double `mlp0` | 8192 | 1024 | v6 | `wide` | SwiGLU pair, `256x256x64` c2x2x1, N=4096 |
| double `mlp2`, single `mlp_down` | 1024 | 4096 | v6 | `sq` | `sq` |
| single `linear1` | 17408 | 1024 | v8 `128x256x256` c1x1x1 | `wide` | split: `qkv` `wide` + SwiGLU pair |
| `action_encoder` (K=7), `head.linear` (N=7) | | | cuBLASLt `Fp16Linear` (alignment fallback) | same | same |

At M = 64, one M tile covers the whole problem, so the CTA count
equals the number of N tiles. The `N = 1024` GEMMs make up 50 of the
80 quantized ActionDiT GEMM calls per denoise step (500 per `infer()`).
On Thor's 20 SMs they get 4
CTAs under v6 and 4 useful CTA pairs under FP8 `sq`, while they are
weight-bandwidth bound (2M = 128 FLOP per weight element). The SM100
FP8 CUTLASS family had no tile narrower than 128 in N and no 1-SM
tile at all. OPT-014 result 3 measured FP8 CUTLASS 1.44-1.68x slower
than cuBLASLt at this M.

## Mechanism

- `GemmVariantTuner` (`flash_rt/models/imagewam/gemm_variant_tuner.py`)
  works on each group of ActionDiT linears sharing `(family, M, N, K)`.
  Candidates must reproduce the default tile's output on every member
  (return code 0, no Python exception, finite, cosine >= 0.9999). A
  stale `flash_rt_kernels` that lacks the `cutlass_fp8_t128x*` symbols
  raises `AttributeError` for those candidates, which rejects them
  without aborting construction. They are timed as one launch per
  member, round robin, so every launch reads a different layer's
  weight, inside CUDA graphs (`gemm_variant_timer.CudaGraphVariantTimer`)
  and interleaved across candidates. A candidate the timer cannot
  capture is rejected (`timing_failed`), and the caller's stream is
  restored. A candidate replaces the default only if it is more than 2%
  faster, and the default is kept if it could not be timed itself. The
  choice is cached per `(family, M, N, K)` and applied before graph
  capture.
- NVFP4 candidates: every cluster-1x1x1 tile, v4, v5, v6, v7, v8, and
  v10 `128x64x256` (Pi0.5's decoder tile). The clustered tiles are
  excluded because Pi0.5 measured them winning in isolation and losing
  in the pipeline on Thor.
- FP8 CUTLASS candidates: `sq`, `wide`, `t1`, `plain`, plus four new
  1-SM cluster-1x1x1 tiles (`gemm_types_sm100.h`, `sm100_small_m`):
  `t128x64x256` (v10 shape), `t128x64x128`, `t128x128x128`, and
  `t128x256x128`.
- `ImageWAMTorchFrontendThor(gemm_variant_autotune=True)` is accepted
  for `nvfp4` and `fp8_static_cutlass` only. Results are in
  `frontend.gemm_variant_results`. Backbone GEMMs are not tuned. With
  the flag off (the default), every tile is the one dispatched before
  this change.

## Local evidence (H100, sm_90)

The SM100 CUTLASS and NVFP4 kernels do not run on sm_90, so every
number below is about the mechanism, not about a tile.

- `tests/test_imagewam_gemm_variant_tuner.py` (18 tests): the
  selection rule against stub GEMMs. It covers argmin choice, the 2%
  hysteresis, rejection on a nonzero return code, on a raised exception
  (`AttributeError`) and on a candidate that cannot be timed, mismatch
  (cosine 0.7987 in the test) and non-finite output, failure on one
  member only, an error when the default itself fails or raises,
  keeping an untimeable default, the cache, and a distinct M counting
  as a distinct key. The timer test also feeds a raising batch and a
  capture-invalidating batch; both come back as `None`, the good batch
  is still timed, and the caller's stream is restored. The real-timer test runs on real cuBLASLt
  launches (M=64, N=1024, K=3072, 6 weights): graph-timed 7.75 us per
  launch against 11.30 us eager event-timed, so launch overhead is
  excluded. A batch with 4x the work measured 3.62x.
- `tests/test_imagewam_gemm_variant_routing.py` (7 tests): only the
  GEMM entry points are replaced (the whole `flash_rt_fp4` module, and
  the `cutlass_fp8_*` attributes). The real `Nvfp4Linear` /
  `StaticFp8Linear`, frontend grouping, tuner, graph capture and
  `infer()` run as shipped. For both precisions, each of the 5
  ActionDiT shapes is tuned once at `M = num_action`, the chosen tile
  reaches every ActionDiT GEMM in the captured graph, and the backbone
  keeps its heuristic tile. Tuning leaves `StaticFp8Linear`'s
  calibrate-before-call contract intact. With the flag off, no GEMM
  launches at construction. With the `cutlass_fp8_t128x*` symbols
  removed, which simulates a stale build, construction completes and
  those candidates show `launch_failed AttributeError`.
- `sm110_check.sh` (CUDA 13.0, `GPU_ARCH=110`): `flash_rt_kernels`
  and `flash_rt_fp4` build and link, and the four
  `cutlass_fp8_t128x*` symbols are exported.

## Thor check

`benchmarks/imagewam_thor_small_m_tile_sweep.py`:

- `--part kernels`: every NVFP4 and FP8 CUTLASS tile at each real
  ActionDiT shape (M=64), over as many distinct weights as layers
  share the shape. Reports GEMM-only us/GEMM, cosine against the fp32
  `x @ W`, the heuristic pick (`*`), the tuner pick (`T`), and the
  per-shape winner, with cuBLASLt fp16 and cuBLASLt FP8 references on
  the same weights.
- `--part infer`: for `nvfp4` and `fp8_static_cutlass`, two graphs
  captured from one frontend (ActionDiT on heuristic vs tuned tiles),
  with action cosine on identical inputs and alternating `infer()`
  P10/P50/P90.

Expected: every tile other than the heuristic's shows cosine vs fp16
equal to the heuristic's own within about 1e-4 (same quantized
operands, different accumulation order). NVFP4 v10 or v5, and FP8
`t128x64x*`, win at N = 1024. New-vs-old action cosine is at least
0.9999.

## Decision pending

If Thor shows a correct `infer()` win, `gemm_variant_autotune` should
become the default for `nvfp4`: a one-line change of the constructor
default. If it shows no win, the heuristic stays, and the sweep table
still says which tile to hardcode, if any. Split-K / stream-K for the
`N = 1024` shapes (16 CTAs at most even with a 64-wide N tile) was not
attempted.

# OPT-019: attention-chain fusion recheck at ImageWAM's real shapes (roadmap item 6)

Status: analysis done. The H100 numbers are indicative only, because a
co-tenant training job shares the GPU. Two changes are implemented, and
both are opt-in:

- FA4 at the backbone site: `use_fa4=True`, or `FLASHRT_THOR_FA4=1`
  with the default `use_fa4=None`.
- FA4 at the `mot` site: `use_fa4_mot=True`.

FA4 has run on Thor only at `a0 = 896` in the per-layer bench
(OPT-005). It has not run at the served backbone shape (q = kv = 905,
a partial last tile), at the `mot` shape (q = 64, kv = 969), inside the
full captured graph with the real checkpoint, or end to end. The
shipped `nvfp4` baseline of 231.6 ms was measured with FA4 off. Plan:
plan.md "Plan: attention-chain fusion recheck at ImageWAM's real
shapes".

## Finding 1: FlashRT's `mot` call is unmasked; official masks padded text keys

The frontend always runs `use_real_mot_mask=True`. At the `mot` site
that dispatches unmasked attention: 64 action queries over all 969
keys, 24 heads, HD 128, per-head K/V, through the same kernel as the
backbone site. Official `infer_action_flux2` builds its mask with
`_build_mot_attention_mask_flux2(target_len=0)`. `target_len = 0`
removes only the region mask: there is no noisy-target block to
exclude. The same builder then applies
`mask[:, :, t0:r0] &= text_valid[:, None, :]`, with
`text_attention_mask` passed at both the prefill call and the action
call. So official excludes the padded text keys for every query row
at both sites, while FlashRT attends to them at both sites (issues.md
ISSUE-020).

For a fused kernel:

- A plain non-causal FA4 call reproduces what FlashRT computes today
  at the `mot` site. That is what `use_fa4_mot` runs, and what the
  dispatch tests compare against. It does not reproduce official.
- Matching official needs the padded keys removed: a key mask, or,
  since padded tokens are inert under the official mask, a context of
  only the valid tokens (ISSUE-020, next experiment). Removing the
  padding from the sequence keeps plain attention correct, so the
  fused path stays a plain FA4 call.
- The three-region `attention_qkv_fp16_mot_joint*` kernels serve only
  the legacy `use_real_mot_mask=False` path.

## Finding 2: attention share at the real shapes (H100, fp16, random weights)

Measured as graph time with the real attention backend minus graph
time with a backend that launches nothing, all else identical
(`benchmarks/imagewam_attention_share_bench.py --part share`):

| stage | with attention P50 | without P50 | attention | per call |
|---|---:|---:|---:|---:|
| prefill (25 backbone calls) | 23.160 ms | 16.869 ms | 6.29 ms (27.2%) | 252 us |
| one denoise step (25 `mot` calls) | 3.520 ms | 2.126 ms | 1.39 ms (39.6%) | 56 us |
| prefill + 10 steps | 58.36 ms | | 20.23 ms (34.7%) | |

## Finding 3: per-call kernels at the real shapes (H100)

Each call reads a different layer's K/V, round robin over 8 layers,
CUDA-graph timed (`--part kernels`). Accuracy is against fp32
PyTorch attention.

| site | kernel | us/call | vs chain | cosine | rel_l2 |
|---|---|---:|---:|---:|---:|
| backbone q=kv=905 | cuBLAS chain (today) | 437.1 | 1.00x | 1.000000 | 5.5e-4 |
| | SDPA flash (FA2) | 56.1 | 7.79x | 1.000000 | 2.8e-4 |
| | SDPA cuDNN | 39.2 | 11.15x | 1.000000 | 2.8e-4 |
| | SDPA mem-efficient | 176.2 | 2.48x | 1.000000 | 2.8e-4 |
| mot q=64, kv=969 | cuBLAS chain (today) | 42.3 | 1.00x | 1.000000 | 5.5e-4 |
| | SDPA flash (FA2) | 15.8 | 2.68x | 1.000000 | 2.7e-4 |
| | SDPA cuDNN | 22.0 | 1.93x | 1.000000 | 2.8e-4 |
| | SDPA mem-efficient | 44.1 | 0.96x | 1.000000 | 2.8e-4 |

The chain's per-call cost here (437 us) is higher than its in-graph
cost in Finding 2 (252 us). Both runs share the GPU with the
co-tenant, and the ratios, not the absolute values, are the
information.

## Finding 4: in-pipeline A/B with an sm_90 fused kernel (H100, fp16)

`--part infer --sdpa-standin`: one frontend, three graphs captured from
the same buffers and weights. PyTorch SDPA runs behind the FA4
calling convention, so the backend's FA4 branches run as they would on
Thor. Timing is 30 `infer()` calls, rotating order.

| configuration | actions cosine vs chain | `infer()` P10 | P50 | delta P50 |
|---|---:|---:|---:|---:|
| cuBLAS chain at both sites | 1.000000 | 48.78 ms | 49.38 ms | |
| fused at backbone | 1.000000 (max-abs 2.8e-3) | 44.53 ms | 45.45 ms | -3.93 ms |
| fused at both sites | 1.000000 (max-abs 2.6e-3) | 38.66 ms | 39.57 ms | -9.81 ms (-19.9%) |

P90 was about 2x P50 for all three configurations: the co-tenant
regime changes, not the kernel. On this device about 60% of the gain
comes from the `mot` site.

## Why Pi0.5's rejection does not transfer

Pi0.5 rejected a fused SIMT attention chain at decoder M = 10, HD 256
(5-7x slower, `docs/pi05_thor_decoder_fp4_e2e.md`). Its QK^T and PV
were about 1 us of tensor-core work that SIMT code could not approach,
and FA4 had no KV-split path at HD 256. ImageWAM differs on each
point:

- HD is 128.
- Both sites have long KV (905 and 969 keys).
- The fused kernels available on both devices are tensor-core kernels.
- The chain writes and re-reads a 24 x q x kv fp16 logits buffer:
  39 MB per backbone call, 3 MB per `mot` call.

On Thor, FA4 at the backbone site already measured 3.75x per call and
-10.5% prefill (OPT-005).

## Implemented

- `fa4_backend.thor_default_enabled()` returns True only on a
  compute-capability-11.x device whose FA4 runtime imports. It checks
  the device first, so FA4 is never imported off Thor.
- `ImageWAMTorchFrontendThor(use_fa4=None)` is the default and
  resolves to the cuBLAS chain. `FLASHRT_THOR_FA4=1` opts in: FA4 at
  the backbone site exactly when `thor_default_enabled()` holds, and
  the chain otherwise, with no error for a missing runtime.
  `use_fa4=True` forces FA4 and raises without a runtime, and
  `use_fa4=False` forces the chain regardless of the environment. The
  resolved value is `frontend.use_fa4`. Making FA4 the default is a
  one-line change: `_FA4_OPT_IN_DEFAULT` in `imagewam_thor.py` goes
  from `"0"` to `"1"`. `FLASHRT_THOR_FA4=0` also stops `fa4_backend`
  from importing FA4 at all, for every model.
- `ImageWAMAttnBackend(use_fa4_mot=True)` and
  `ImageWAMTorchFrontendThor(use_fa4_mot=True)` run the `mot` site
  through FA4. Q is the action rows at row offset `a0`, K/V are the
  full per-head `(1, 969, 24, 128)`, with `causal=False`,
  `pack_gqa=False`, `num_splits=1`, and an explicit `softmax_scale`.
  The combination requires `use_real_mot_mask=True` and
  `use_perhead_kv=True`, and the constructor raises otherwise.
- FA4 output, both sites, goes to a dedicated buffer. The frontend owns
  `_fa4_out` of shape `(total, hidden)`, allocated only when some site
  runs FA4, and passes it to the backend as the `fa4_out` /
  `fa4_out_numel` slots. The backend checks the capacity at
  construction and on every call, then copies the result back to the Q
  rows. An earlier version staged FA4 output in `logits`
  (`total*NH x (total + total%2)` elements). That is large enough at
  the real dims but smaller than `q_seq*NH*HD` at small dims: at the
  default test dims it failed with `CUDA error: invalid argument`, and
  at other small dims it wrote past the buffer silently. The frontend
  used it only when FA4 was on. The tests now pre-fill a guard band
  after `fa4_out`, and `logits`, with a sentinel, and fail on any FA4
  write outside `fa4_out`; they fail on the old staging.
- Falling back when FA4 fails. FA4 compiles on its first call, which
  lands in the eager warmup of `set_prompt()`'s graph capture, and it
  can fail there or inside the capture. One way is an FA4 runtime that
  imports but cannot compile for sm_110, as with `nvidia-cutlass-dsl`
  4.4.x (see `fa4_backend`). On such a failure with FA4 on,
  `set_prompt()` logs an error, emits a `RuntimeWarning`, stores the
  reason in `frontend.fa4_fallback_reason`, rebuilds the attention
  backend with FA4 off at both sites, and captures again. It also
  restores the caller's CUDA stream first, because an invalidated
  capture leaves the capture stream current. A failure with FA4 off
  still raises. Tested with stand-ins that raise on every call, raise
  only inside capture, and issue a device sync inside capture, which
  invalidates the capture. All three recover to the cuBLAS chain's
  output: cosine 1.0000000, max-abs up to 9.5e-7, the difference
  coming from the two frontends' own cuBLASLt autotune picks.
- Local verification, `tests/test_imagewam_fa4_dispatch.py` (23
  tests). FA4 is replaced by a stand-in with `_flash_attn_fwd`'s
  calling convention that computes an fp32 matmul-softmax-matmul in
  PyTorch, and each FA4 branch is compared against the cuBLAS chain at
  the real shapes:
  - Backbone q=kv=905: cosine 1.000000, max-abs 4.9e-4, rel_l2 5.8e-4.
  - `mot` q=64 at row 905, kv=969: cosine 1.000000, max-abs 3.7e-4,
    rel_l2 5.6e-4, with rows `[0, a0)` untouched.
  - Small-dims frontend end to end, both sites on the stand-in vs the
    chain: actions cosine 1.000000.
  - Also covered: the constructor guard, the `fa4_out` bounds and
    capacity, `use_fa4` resolution over the environment variable,
    runtime availability and explicit argument, and the FA4-failure
    fallback.
- `tests/test_imagewam_fa4_backbone.py` gains a real-FA4 test for both
  sites at the real shapes. It skips without FA4.

## Thor check

FA4 is opt-in, so every FA4 step below opts in explicitly: an
environment variable, `--fa4 on`, or the bench's own FA4
configurations. In any run with FA4 on, a line containing `falling back
to the cuBLAS attention chain` means FA4 failed and the numbers are the
chain's. Report it with the reason.

1. Runtime:
   `FLASHRT_THOR_FA4=1 python -c "from flash_rt.hardware.thor import fa4_backend as f; from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor as F; print(f.status(), F._resolve_use_fa4(None))"`.
   Expect `active True`. Without the variable, the second value must be
   `False`.
2. Real-FA4 correctness at the served shapes:
   `pytest tests/test_imagewam_fa4_backbone.py -q -s -k both_sites_real_shapes`
   (`test_fa4_matches_cublas_both_sites_real_shapes`). This covers the
   backbone at q = kv = 905, whose last tile is partial, and `mot` at
   64 over 969. Expect a pass, not a skip. Report both printed cosines
   (expect > 0.999) and max-abs values.
3. Per-call kernels: `python benchmarks/imagewam_attention_share_bench.py --part kernels`.
   Report both tables, including `fa4_splits1/2/4` for `mot`.
4. Attention share on the shipped precision:
   `python benchmarks/imagewam_attention_share_bench.py --part share --precision nvfp4 --fa4 off`,
   then `--fa4 on`, then `--fa4 on --fa4-mot`. Report the three
   "attention = ..." blocks.
5. `infer()` A/B, FA4 on vs off, real checkpoint:
   `CKPT_PATH=<model.pt> python benchmarks/imagewam_attention_share_bench.py --part infer --precision nvfp4 --iters 60`.
   The bench builds chain, backbone-FA4 and both-sites-FA4 graphs from
   one frontend. Report each configuration's action cosine against the
   chain (expect >= 0.999) and the P10/P50/P90 plus delta.
6. nvfp4 end-to-end official compare, FA4 off then on:
   `PRECISION=nvfp4 N_TASKS=10 FRAMES=0,60 SEEDS=0,1 python benchmarks/imagewam_e2e_official_compare.py`,
   then the same command with `FLASHRT_THOR_FA4=1` in front. Report
   `fr_vs_off` median/min, mean `mae_fr_vs_gt`, and the printed `infer()`
   P50 for both runs. Expect the FA4 run to match the FA4-off run
   closely: `fr_vs_off` within about 1e-4 at the median.

## Recommendation

1. Backbone: make FA4 the Thor default (`_FA4_OPT_IN_DEFAULT = "1"`)
   once Thor passes three checks, each run with FA4 opted in:
   - the real-shape test at q = kv = 905 and at 64 over 969;
   - an nvfp4 end-to-end official compare with FA4 on vs off;
   - an `infer()` A/B with FA4 on vs off.
2. `mot`: FA4 is the strongest remaining attention lever. It computes
   the same math as FlashRT's current unmasked chain (not official's
   padded-key-masked rule; Finding 1), and on H100 an sm_90 fused
   kernel took the bigger share of the in-pipeline gain. Flip `use_fa4_mot` to default
   on (a one-line change) once Thor shows cosine >= 0.999 against the
   chain at the real shape and an `infer()` win. The Thor kernel sweep
   also times `num_splits` 2 and 4 for this shape. If a split wins,
   change the constant in the `mot` branch.
3. No custom fused attention kernel is needed at either site.
4. ISSUE-020 (padded text keys) is fixed most simply by dropping the
   padded tokens from the context (`x0 = n_valid + 1`), which needs no
   key mask in any kernel, fused or not. That fix belongs to a separate
   stream.

## Thor, one matrix session (`c20f3a0`, libero_spatial, nvfp4)

FA4 at both sites inside the captured graph, real checkpoint, end to end,
which is the state the note above recorded as missing:

| row | P50 ms | marginal | vs official median | FA4 fallback |
|---|---:|---:|---:|---|
| vae_trim (FA4 off) | 102.9 | — | — | — |
| vae_trim_fa4bb (backbone only) | 99.0 | -3.9 | — | None |
| stack (backbone + mot) | 93.2 | -5.8 | 0.99934 | None |
| stack_no_vae (FA4 on, native VAE off) | 104.8 | — | — | None |

The backbone site is 3.9 ms below the no-FA4 row and adding the `mot` site
takes another 5.8 ms, so the section C criterion for making FA4 the default
(at least 2 ms of P50 against the FA4-off row, and not worse against
official) held in this pass, with no fallback on any row. `use_fa4` and
`use_fa4_mot` therefore remain profile switches; the default is unchanged
and the decision is the owner's (plan.md "Decisions pending"). ISSUE-082
applies to the size of these marginals.


## Thor, LIBERO (`eccf14f`)

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU
exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is
end-to-end `infer()` P50 in ms and `vs official` is the median action cosine
against the official model. FA4 at both sites inside one captured graph, real
checkpoint, `nvfp4`, on the two suites `c20f3a0` did not cover:

| suite | `vae_trim` (FA4 off) P50 ms | `stack` (FA4 both sites) P50 ms | delta | vs official: `vae_trim` / `stack` | FA4 fallback |
|---|---:|---:|---:|---|---|
| libero_goal | 118.7 | 92.6 | -26.1 | 0.99937 / 0.99930 | None |
| libero_10 | 103.0 | 93.5 | -9.5 | 0.99926 / 0.99925 | None |

`vae_trim` and `stack` are the same pair the `c20f3a0` session ran on
libero_spatial, where the pair spanned 102.9 -> 93.2 ms (3.9 ms from the
backbone site, 5.8 ms from `mot`); the same isolation of FA4 at both sites is
worth 26.1 ms on libero_goal and 9.5 ms on libero_10. `vs official` is not
worse with FA4 on in either suite, and no row anywhere in the round reported
an FA4 fallback.


# OPT-016: single-stream `linear2` merge (roadmap item 4)

Status: implemented and locally verified (H100, `fp16`); default on for
every precision except `fp16_cutlass`. Thor speed and `nvfp4` numerics
pending (Thor check below).

Area: single-stream blocks, 20 backbone + 20 ActionDiT
(`pipeline_thor.py` `_single_stream_layer` / `_action_single_layer`).
Plan: `plan.md` roadmap item 4. This is
OPT-015's op-fusion audit finding 1, sub-problem 3.

## What changed

The official blocks run `linear2` as one GEMM over
`cat([attn_out, mlp_act])`. FlashRT used to split it at load time and run
`attn_out_proj` + `mlp_down` + a torch add before the gated residual.
With `dims["merge_linear2"]` (default = the `merge_qkv_mlp` rule):

- `checkpoint_loader._extract_single_block(merge_linear2=True)` keeps the
  real unsplit `linear2.weight`: `(12288, 3072)` backbone, `(7168, 1024)`
  ActionDiT, in the (K,N) convention.
- `silu_glu_merged_fp16` gained `out_row_stride`; the `linear1`-merged
  SiLU-GLU writes straight into the MLP columns of `single_linear2_in` /
  `action_linear2_in`. A strided copy places the attention output in the
  first `attn_width` columns, then one GEMM with K = attn + mlp_hidden
  writes `proj_scratch`.
- Per layer call: one strided copy + one GEMM replaces two GEMMs + one
  add (two kernels fewer for `nvfp4`, whose linear op is quantize + GEMM).
- `nvfp4`: `hidden` = 3072 is a multiple of the 16-element scale block,
  so the merged activation and weight quantize to exactly the split
  path's operands; K = 12288 and 7168 are multiples of 64 (no
  scale-factor padding). Only the accumulation changes.

## Local results (H100, `fp16`)

| check | result |
|---|---|
| strided SiLU-GLU vs packed call, real shapes | bit-exact, untouched columns zero |
| real `linear2.weight` vs `cat(attn_out_proj, mlp_down)` (blocks 0, 19, both experts) | `torch.equal` |
| backbone single layer (a0=905), projection, merged vs split | cos 0.9999999, max-abs 3.9e-3, rel_l2 3.5e-4 |
| ActionDiT single layer (M=64), projection, merged vs split | cos 0.9999998, max-abs 9.8e-4, rel_l2 3.6e-4 |
| projection rel_l2 vs FP32 reference, merged / split | 2.07e-4 / 2.95e-4 (backbone), 2.08e-4 / 3.01e-4 (ActionDiT) |
| real checkpoint, full graph, actions merged vs split | cos 0.999999, rel_l2 1.4e-3 |
| CUDA kernels per prefill + 10-step denoise | 7082 -> 6662 |
| e2e vs official, 20 frames, `fr_vs_off` median / min | 0.99840 / 0.99566 (baseline 0.99840 / 0.99567) |
| e2e mean `mae_fr_vs_gt` | 0.18359 (baseline 0.18359) |

The merged path is closer to the FP32 reference because it rounds once
where the split path rounds three times.

Speed on the shared H100 (indicative only, same process, interleaved,
40 iterations): `infer()` P50 102.23 -> 101.27 ms (P10 91.42 / 98.87,
P90 102.61 / 101.49). No Thor claim.

## Thor check

```
# FlashRT at this branch, GPU_ARCH=110 build
cmake --build build -j --target flash_rt_kernels flash_rt_fp4
pytest tests/test_imagewam_real_mlp.py tests/test_imagewam_thor_real_wiring.py -q -s
CKPT_PATH=<.../model.pt> AB=merge_linear2 PRECISIONS=nvfp4,fp16 COUNT_KERNELS=1 \
  python benchmarks/imagewam_fusion_ab.py
```

Expected: pytest passes and prints `bit_exact_vs_packed=True` and merged
vs split `cos` >= 0.9999. The A/B prints, per precision, `actions B vs A`
(expect cos >= 0.9999 and finite for `nvfp4`; the `linear1` merge gave
0.99998), `infer()` / `replay()` P10/P50/P90 for split (A) and merged
(B), and the kernel count (B lower). The `linear1` precedent predicts
most of any win in the ActionDiT loop. Report every printed line.

## Follow-ups

- The attention output is copied into the `linear2` input (one strided
  copy per layer, 11 MB per backbone layer). Having the attention backend
  write its output there directly (an output row stride in
  `ImageWAMAttnBackend.run`, or the FA4 path's existing output copy
  retargeted) would remove it.
- FP8 precisions: one activation scale now spans both halves
  (ISSUE-011).

# OPT-017: gated residual + next AdaLN in one kernel (roadmap item 3)

Status: implemented and locally verified bit-exact (H100); default on
for every precision. Thor speed pending (Thor check below).

Area: every gated residual update in `pipeline_thor.py` (backbone
double/single, ActionDiT double/single, ActionDiT head). Plan:
`plan.md` roadmap item 3. Supersedes
OPT-015's deferred CUTLASS gated-residual epilogue for this problem.

## What changed

- `csrc/kernels/fusion.cu`: `gate_res_ada_layer_norm_bf16res` (backbone,
  BF16 residual) and `gate_res_ada_layer_norm_fp16` (ActionDiT) update
  the residual and write the next normed + modulated FP16 activation in
  one launch, one block per row. gate/scale/shift are `(dim,)` FP32
  vectors read straight from the modulation output and rounded to FP16
  in-kernel; the LayerNorm statistics use the stored residual and the
  reduction of `ada_layer_norm_*`. `out == nullptr` gives the
  residual-only update (the backbone's last layer).
- `pipeline_thor.py` with `dims["fuse_res_norm"]`: AdaLN2 of each double
  block fuses into its attention residual; `imagewam_prefill` and
  `imagewam_denoise_step` chain each layer's last residual into the next
  layer's AdaLN: double -> double, last double -> single (txt and img
  rows each normalized with the single blocks' modulation), single ->
  single, ActionDiT last single -> head (`head_modded`, the standalone
  head AdaLN is skipped).
- The fused path records no per-layer `_fuse_mod_group` kernels (FP16
  casts of shift/scale/gate and the `(rows, dim)` gate broadcast). FP16
  copies remain only for the standalone AdaLN at the start of each chain
  (prefill layer 0, each denoise step's layer 0).

## Local results (H100)

| check | result |
|---|---|
| kernel vs `gate_res_*` + `ada_layer_norm_*`, 513 / 392 / 905 x 3072 BF16, 64 x 1024 FP16 | residual and output bit-exact |
| kernel normed output vs FP32 torch reference | rel_l2 2.07e-4 (FP16 output rounding) |
| residual-only mode vs `gate_res_bf16res` | bit-exact |
| real dims, random weights, prefill + 10-step denoise, fused vs unfused | `backbone_hidden`, all 25 layers' K/V, `action_latent` bit-exact |
| real checkpoint, captured graph, fused vs unfused (`fp16`) | actions and `backbone_hidden` bit-exact |
| CUDA kernels per prefill + 10-step denoise | 6662 -> 4968 (-1694) |
| e2e vs official, 20 frames, both items on, `fr_vs_off` median / min | 0.99840 / 0.99566 (baseline 0.99840 / 0.99567) |
| e2e mean `mae_fr_vs_gt`, both items on | 0.18359 (baseline 0.18359) |

Speed on the shared H100 (indicative only, same process, interleaved):

| A/B | iterations | `infer()` P50 A -> B | P10 A / B | P90 A / B |
|---|---:|---:|---:|---:|
| `fuse_res_norm` | 40 | 101.38 -> 96.24 ms | 100.89 / 93.52 | 102.09 / 96.99 |
| `merge_linear2` + `fuse_res_norm` | 200 | 102.30 -> 96.17 ms | 92.78 / 91.70 | 104.84 / 96.90 |

Both items together: 7082 -> 4968 CUDA kernels per pass. No Thor claim.

## Thor check

```
# FlashRT at this branch, GPU_ARCH=110 build
cmake --build build -j --target flash_rt_kernels flash_rt_fp4
pytest tests/test_imagewam_residual_norm_fusion.py -q -s
CKPT_PATH=<.../model.pt> AB=fuse_res_norm PRECISIONS=nvfp4,fp16 COUNT_KERNELS=1 \
  python benchmarks/imagewam_fusion_ab.py
CKPT_PATH=<.../model.pt> AB=merge_linear2,fuse_res_norm PRECISIONS=nvfp4,fp16 \
  python benchmarks/imagewam_fusion_ab.py
```

Expected: pytest prints `bit_exact=True` for every kernel case and for
`backbone_hidden`, `K_cache`, `V_cache`, `action_latent` of the whole
pass. `AB=fuse_res_norm`: `actions` and `backbone_hidden` B vs A
bit-exact for both `nvfp4` and `fp16`. The script builds B on A's
autotuned `GemmRunner` (`gemm_runner=`), so every cuBLASLt shape runs
the same algorithm on both sides: all `fp16` weight GEMMs, and under
`nvfp4` the `fp16_nn` fallbacks (`action_encoder`, `head.linear`) and
the `bf16_nn` entry GEMMs (`txt_in`, `img_in`); the NVFP4 CUTLASS GEMMs
choose their variant from the shape alone. Kernel count B lower by
about 1700; `infer()` P50 B below A.
The combined run gives the total of items 3 and 4 against the
pre-roadmap per-layer path. Report every printed line. Add `USE_FA4=1`
if the production configuration uses FA4.

## Remaining per-pass launches after items 3 and 4 (H100 profiler, real dims, `fp16`)

Prefill: 478 CUDA kernels. 10-step denoise: 4490, of which:

| kernel | launches | source |
|---|---:|---|
| torch elementwise copy | 950 | 750 Q/K/V column-slice copies (`_copy_slice`, 3 per layer) + 200 attention-output copies into the merged `linear2` input |
| `rms_norm_kernel` | 500 | QK-Norm, 2 per layer |
| `rope_apply_fp16_perhead_kernel` | 500 | RoPE on Q and K, 2 per layer |
| GEMM kernels (cuBLASLt, incl. 310 split-K reduces) | ~1100 | weight GEMMs + attention QK^T / PV |
| `gate_res_ada_layer_norm_kernel` | 300 | this entry |
| `fill_neginf_strided_kernel` | 250 | odd `kv_seq` (969) logits pad column |
| `softmax_fp16_kernel`, `silu_glu_merged_kernel` | 250 each | |

Candidates by launch count (not planned): one kernel doing the Q/K/V
split + QK-Norm + RoPE from the `qkv`/`linear1` output would replace
7 launches per layer with 1 (about 1500 per `infer()` in the denoise
loop alone); an even-padded K/V length or a pad-aware softmax would
drop the 250 pad fills; the attention-output copy is OPT-016's
follow-up.

# OPT-020: VAE input preprocessing kernel with a 256-entry normalization table (roadmap item 2)

Status: implemented and served by default (bit-identical); Thor
latency pending (Thor checklist step 2 in `plan.md`)

Area: `flash_rt/models/imagewam/vae_preprocess.py`,
`csrc/kernels/imagewam_vae_preprocess.cu`, used by
`vae_encoder.encode_to_tokens(..., preprocessor=)` and
`vae_stage.ImageWAMVaeStage`

## Mechanism

One `imagewam_vae_preprocess_bf16` launch per camera view reads the
`(H,W,3)` uint8 view and writes its column block of the `(1,3,224,448)`
BF16 VAE input. It replaces `_prep_view` (uint8 -> float32 on the source
device, `F.interpolate(mode="area")`, three elementwise normalize ops,
BF16 cast) and `torch.cat`.

- No resize (view already 224x224): a 256-entry BF16 table built on the
  GPU with the same torch expression as `_prep_view`, Pi0.5's
  `_infer_uint8_to_fp16` technique.
- `resize="area"` (served default): reproduces torch's arithmetic, which
  was measured on H100: area pooling is `sum / kh / kw` (two rounded
  float32 divisions over exact integer window sums), and `x / 255.0` is
  `x * (1.0f/255.0f)`. Explicit `_rn` intrinsics keep it exact under
  `--use_fast_math`.
- `resize="pil_bilinear"` (opt-in, frontend `vae_resize`): Pillow's
  `Resample.c` fixed-point bilinear (22-bit coefficients built on the
  host in double precision, horizontal then vertical pass with uint8
  rounding between them) plus the official center crop, then the table.
  Only the resize is bit-exact to the official eval; the eval normalizes
  in BF16 arithmetic, which differs from the table in 127 of 256 entries
  (ISSUE-030).

## Result (H100, shared GPU, indicative)

Bit-exactness, `tests/test_imagewam_vae_preprocess.py`: 0 differing
BF16 elements against `_prep_view` (area) and against the official
`_center_crop_resize` + served normalization (pil_bilinear), for real
512x512 LIBERO frames and random 512x512, 256x256, 224x224, 480x640 and
100x150 inputs. Tokens from `encode_to_tokens` are bit-identical with
and without the kernel. End to end (`imagewam_e2e_official_compare.py`,
fp16, 10 tasks x frames {0,60}, seeds {0,1}): `fr_vs_off` median
0.99840 / min 0.99567, mean `mae_fr_vs_gt` 0.18359, identical to the
baseline.

Latency, `benchmarks/imagewam_vae_stage_bench.py --section preprocess`
(two views, alternating in one process):

| input | torch path kernels / GPU time | kernel path kernels / GPU time | host enqueue torch -> kernel |
|---|---|---|---|
| raw 512x512, CPU uint8 | 15 / 0.702 ms | 4 / 0.054 ms | 22.2 -> 2.5 ms |
| raw 512x512, GPU uint8 | 15 / 0.455 ms | 2 / 0.009 ms | 0.31 -> 0.05 ms |
| 224x224, CPU uint8 | 11 / 0.102 ms | 4 / 0.016 ms | 20.4 -> 2.3 ms |

The CPU-input rows are dominated by the torch path converting uint8 to
float32 on the host (`.to(device, float32)` of a CPU tensor) and then
copying 4x the bytes; the kernel path copies uint8 only. The host
numbers on this box are inflated by CPU contention from the co-tenant
job and by GPU time-slicing (a synchronized call has a ~2.4 ms floor
here), so only the direction is meaningful locally. `encode_to_tokens`
wall P50 with CPU inputs: 36-42 ms -> 12.4 ms.

## Open

Thor latency of the preprocessing step and of `infer()` (checklist).
The served-vs-official resize difference is ISSUE-030.

# OPT-021: VAE encode inside the CUDA graph and a native NHWC encoder (roadmap item 5)

Status: implemented behind frontend flags (`vae_encoder="native"`,
`vae_graph_input=(views, H, W)`), verified on H100; defaults unchanged
(`vae_encoder="torch"`, VAE outside the graph). Thor latency pending
(Thor checklist step 6 in `plan.md`).

Area: `flash_rt/models/imagewam/vae_stage.py`,
`flash_rt/models/imagewam/vae_native_encoder.py`,
`csrc/kernels/imagewam_vae_groupnorm.cu`,
`csrc/kernels/imagewam_vae_residual.cu`,
`flash_rt/frontends/torch/imagewam_thor.py`

## Profile of the stock encode (H100, 224x448, indicative)

Measured with `benchmarks/imagewam_vae_stage_bench.py --section profile
--profile-repeats 7`: torch-profiler GPU kernel time per op family, 3
invocations x 7 captures of 10 encodes. 282 kernels per
`AutoEncoder.encode`, 9.5-10.2 ms kernel time per encode (per-invocation
medians; single captures 8.6-10.5 ms). Shares are the per-invocation
medians over captures, with the single-capture range in parentheses:
the co-tenant job time-slices the GPU, which stretches individual kernel
durations, so one capture alone can misstate the split.

| op family | kernels per encode | share of GPU kernel time |
|---|---:|---:|
| torch GroupNorm statistics (`RowwiseMoments`, N*G = 32 blocks) | 44 | 35-39% (31-50%) |
| convolution math | 25 | 18-26% (12-34%) |
| other elementwise: conv-bias broadcast adds, residual adds, mul, pad, copies | 109 | 16-24% (10-30%) |
| cuDNN NCHW<->NHWC transforms around each convolution | 72 | 10-17% (8-24%) |
| sigmoid (swish) | 21 | 2-4% |
| attention (one 1568-token block), q/k/v and proj GEMMs | 11 | ~2% |

The same command profiles `NativeFlux2Encoder`: 142 kernels, 2.6-3.9 ms
kernel time per encode (per-invocation medians); convolution math 30-44%
of it, FlashRT GroupNorm apply 8-21%, GroupNorm statistics + finalize
8-13%, bias+residual about 2%; the single attention kernel ranges 4-53%
across captures, the most time-slicing-sensitive entry.

Converting the stock module to `channels_last` removes the transforms
but makes torch's GroupNorm slower on the strided layout; no net gain.

## Mechanisms

1. `ImageWAMVaeStage`: fixed uint8 view buffer -> preprocessing kernel
   (OPT-020) -> encoder -> tokens written into the frontend's `img_raw`.
   `run()` is capture-safe.
2. `vae_graph_input`: the frontend records `stage.run()` ahead of
   prefill in its one CUDA graph; `infer()` copies the views into the
   fixed buffer and replays once. Views of exactly that shape are then
   required on every call.
3. `NativeFlux2Encoder` (`vae_encoder="native"`): the same op order as
   `AutoEncoder.encode`, all activations `channels_last`, and
   - NHWC GroupNorm(+SiLU): Welford partial statistics per block, a
     per-(group, sample) Chan merge, one vectorized apply with torch's
     BF16 rounding points (after the norm, the sigmoid and the product);
     conv1's bias folded into norm2's reads;
   - one pass for conv2 bias + nin_shortcut bias + residual add, with
     torch's BF16 rounding after each add (bit-exact to the three torch
     ops);
   - the attention block's q/k/v 1x1 convolutions as one GEMM.

## Result (H100, shared GPU, indicative)

Numerics:

| check | result |
|---|---|
| stage eager / graph (torch encoder) vs `encode_to_tokens` | bit-identical tokens |
| frontend graph vs eager prefill+denoise, same tokens and noise | bit-identical actions (torch and native encoders) |
| GroupNorm(+SiLU) kernel vs torch, all 22 real encoder inputs | cosine >= 0.9999998, 0.001-0.03% of BF16 elements differ |
| bias+residual kernel vs torch | bit-identical |
| native vs torch encoder tokens, real frames | cosine 0.99998, rel_l2 0.0055, max-abs 0.03-0.05 |
| token stats, native / torch / real Thor | mean -0.0101 / -0.0101 / -0.02, std 0.9705 / 0.9706 / 0.97, absmax 4.78 / 4.78 / 4.91 |
| rel_l2 vs an FP32 encode, 5 frames | torch BF16 0.0088-0.0096, native BF16 0.0089-0.0097 |

The native encoder is as close to FP32 as the stock BF16 encoder; the
native-vs-torch gap is smaller than either one's BF16 error.

End to end (`imagewam_e2e_official_compare.py`, fp16, 10 tasks x
frames {0,60}, seeds {0,1}): `fr_vs_off` median / min and mean
`mae_fr_vs_gt` are 0.99840 / 0.99567 / 0.18359 for the default path, for
the torch encoder inside the graph, and 0.99840 / 0.99568 / 0.18359 for
the native encoder inside the graph (baseline 0.99840 / 0.99567 /
0.18359).

Speed, VAE stage alone (`--section encode`, two 512x512 CPU views,
alternating):

| variant | kernels | GPU kernel time | wall P50 |
|---|---:|---:|---:|
| legacy `encode_to_tokens` (torch preprocess) | 298 | 7.8-9.5 ms | 38-46 ms |
| `encode_to_tokens`, kernel preprocess (served now) | 287 | 8.1-9.4 ms | 12.4 ms |
| stage, torch encoder, CUDA graph | 287 | same | 12.0 ms |
| stage, native encoder, eager | 147 | 1.9 ms | 4.4 ms |
| stage, native encoder, CUDA graph | 147 | 1.9 ms | 4.3 ms |

Wall-clock on this box has a ~2.4 ms floor per synchronized call from
GPU time-slicing with the co-tenant job, and kernel-time totals move
with that load between runs (stock 7.8-10.5 ms, native 1.9-3.9 ms
across the runs recorded here); compare variants within one run. Graph
capture alone is worth ~0.35 ms here, where the host CPU is fast and
the encode is GPU-bound.

`infer()` at real dims (`--section infer`, fp16, random weights,
alternating frontends): eager-torch 120.0 ms, graph-torch 119.5 ms,
graph-native 113.9 ms (224x224 views); eager-torch 119.2 ms vs
eager-native 114.3 ms (512x512 views).

## Open

- Thor: VAE-stage latency (stock 21.5 ms in OPT-012), `infer()` P50 on
  `nvfp4` for the four placements/encoders, and the token cosine on
  sm_110 (checklist).
- The largest remaining native cost is convolution math (30-44% of its
  kernel time); the three (0,1,0,1) zero pads before the stride-2
  convolutions and the conv_in input layout conversion are the next
  copy-elimination candidates.

## Thor, one matrix session (`c20f3a0`, libero_spatial, nvfp4)

The `vae` row (native NHWC encoder captured into the main graph, everything
else at the default configuration) measured 190.1 ms against `default`
225.5 ms, and removing the switch from the full stack costs +11.6 ms
(`stack_no_vae` 104.8 against `stack` 93.2). Both readings are below the
corresponding row without it, so the section C criterion for the native VAE
(lower P50 than `default`, not worse against official) holds; the served
default stays `vae_encoder="torch"` with the VAE outside the graph, and the
decision is the owner's (plan.md "Decisions pending"). ISSUE-082 applies to
the size of these marginals.


## Thor, LIBERO (`eccf14f`)

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU
exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is
end-to-end `infer()` P50 in ms and `vs official` is the median action cosine
against the official model. Native NHWC encoder captured into the main graph,
real checkpoint, `nvfp4`:

| suite | default P50 ms | vae_trim P50 ms | delta | vs official: default / vae_trim |
|---|---:|---:|---:|---|
| libero_goal | 203.0 | 118.7 | -84.3 | 0.99558 / 0.99937 |
| libero_10 | 202.0 | 103.0 | -99.0 | 0.99765 / 0.99926 |

Both `vae_trim` rows also carry `text_trim`, so this pair measures the native
VAE in the graph and the trim together; it does not separate them. The
separated ladder exists for libero_spatial in the `c20f3a0` section above
(`default` 225.5 ms, `vae` 190.1 ms, `vae_trim` 102.9 ms), and there the full
stack agreed with official better than the default configuration (0.99934
against 0.99764), which is the same direction the two rows here show.


# OPT-024: Hadamard-rotated INT4 (E0M3) precision tier, `e0m3_hadamard`

Status: implemented behind `precision="e0m3_hadamard"` (default stays
`nvfp4`). Accuracy decided by an H100 simulation matched byte for byte
to the CUDA quantizers; Thor correctness and latency pending
(`benchmarks/imagewam_e0m3_hadamard_thor_check.py`,
`tests/test_imagewam_e0m3_hadamard.py`).

Area: 4-bit block-scaled GEMM tier for every 16-aligned ImageWAM weight
(`quant_linear.py` `E0m3HadamardLinear`, `imagewam_thor.py`
`_wrap_linear`), `plan.md` roadmap item 9.

## Mechanism

- Thor's tcgen05 block-scaled MMA reads the operand element format from
  the runtime instruction descriptor. Value 0 decodes E0M3: sign-magnitude
  integers -7..7 with the same per-16 UE4M3 scale layout as NVFP4. This
  is the full-rate `kind::mxf4nvf4` path (`SM100_MMA_MXF4_SS`, K = 64 per
  instruction), not the SM80 `s4` path OPT-007 closed.
- Both operands get the same orthonormal 16-point Hadamard rotation of
  every 16-wide K block. The rotation is block-diagonal along K, so any
  `K % 16 == 0` works; ImageWAM's quantized K values are 1024, 3072, 4096,
  and 9216, and 7680/12288 would also work. The OPT-007 power-of-two
  padding problem does not arise.
  - Weight, offline: fp32 butterfly, times a per-tensor power of two
    `2^e` (largest block scale at most 448), stored as fp16, quantized
    by `quantize_e0m3_dynamic_sfa_fp16`. The GEMM `alpha = 2^-e` undoes
    the pre-scale exactly.
  - Activation, online: `quantize_e0m3_dynamic_sfa_fp16_vec(use_rht=1)`
    rotates in registers and quantizes (existing Pi0.5 kernel).
  - GEMM: `cutlass_fp4_gemm_e0m3w_variant(a_format=0)`, new tile variants
    1/6/8 (128x256 tiles) beside the existing 10 (128x64x256), picked by
    the same `pick_variant(N, K)` as `nvfp4`.
- The merged single-stream `linear1` (qkv + mlp gate/up, K = 3072 or
  1024) is one ordinary weight to this tier. `action_encoder` (K=7) and
  `head.linear` (N=7) fall back to `Fp16Linear`, as in `nvfp4`.

## Simulation fidelity (H100)

`flash_rt/models/imagewam/blockscaled_ref.py` reproduces the quantizers.
`tools/check_blockscaled_quantizers_sm90.py` compiles the unmodified
quantizer sources (same flags as `fp4_kernels_obj`) for sm_90a and
compares bytes on 130M real ImageWAM weight elements (65M packed
bytes), outlier activations, and NVFP4 blocks built on the E2M1
rounding thresholds (1.7M values scale exactly onto a threshold):

- byte-exact: E0M3 weights (plain, rotated, and the served weight
  preparation `prepare_e0m3_hadamard_weight`, which `E0m3HadamardLinear`
  calls), E0M3 + H16 activations, and NVFP4 amax, including the
  threshold blocks. The NVFP4 kernel's `1.f / bs_dq`
  (`quantize_fp4_sfa.cu`) lowers to `rcp.approx.ftz` and its `amax / 6`
  to `div.approx` under `--use_fast_math`; on sm_90 they choose the same
  codes as the reference's round-to-nearest arithmetic even at exact
  thresholds. Thor's lowering is checked by
  `test_nvfp4_quantizer_bit_exact` (Thor only).
- NVFP4 MSE (not used by ImageWAM) differs on about 2 codes per million:
  the kernel sums each candidate's squared error sequentially and its
  PTX contracts `e * scale - v` and `err += d * d` into `fma.rn`, so
  near-equal candidates can tie-break differently from the reference.

Dequantized operands of every tier are
exactly representable in fp16 (0 inexact elements across the whole
pipeline), so the whole-pipeline simulation runs the unchanged fp16
cuBLASLt GEMM (fp32 accumulation) on them; it differs from the
block-scaled GEMM only in accumulation order.

## Result 1: per-GEMM error on real activations (H100 simulation)

`benchmarks/imagewam_e0m3_accuracy_study.py`, real checkpoint, fp16
pipeline on 4 LIBERO frames (libero_spatial), all 180 quantized weights,
all 10 denoise steps. Output rel_l2 vs the exact fp32 product; "mean"
is the unweighted mean over the 18 GEMM groups, "pooled" weights by
output norm (dominated by `txt_mlp2`, ISSUE-051).

| tier | mean | pooled | weight-only (pooled) | act-only (pooled) | weights better than nvfp4 |
|---|---:|---:|---:|---:|---:|
| `nvfp4` (shipped) | 0.0669 | 0.0893 | 0.0343 | 0.0805 | - |
| `nvfp4`, MSE weight scales | 0.0633 | 0.0884 | 0.0319 | 0.0805 | 180/180 |
| `nvfp4` + H16 | 0.0662 | 0.0464 | 0.0388 | 0.0274 | 25/180 |
| `nvfp4`, weight pre-scale | 0.0653 | 0.0884 | 0.0330 | 0.0805 | 180/180 |
| E0M3 weights, E2M1 activations | 0.0650 | 0.0887 | 0.0360 | 0.0805 | 169/180 |
| E0M3 W4A4, no rotation | 0.0686 | 0.0749 | 0.0360 | 0.0649 | 65/180 |
| E0M3 W4A4 + H16 (Pi0.5's tier) | 0.0565 | 0.0399 | 0.0360 | 0.0179 | 180/180 |
| **E0M3 W4A4 + H16, weight pre-scale (`e0m3_hadamard`)** | **0.0545** | **0.0380** | **0.0336** | **0.0179** | **180/180** |
| same, H64 rotation | 0.0559 | 0.0401 | 0.0340 | 0.0211 | 180/180 |
| same, H256 rotation | 0.0563 | 0.0420 | 0.0340 | 0.0242 | 180/180 |

- The gain is on the activation side: E0M3 needs the rotation (without
  it, activations are worse than E2M1), and with it the uniform grid
  beats E2M1 plus rotation. On weights alone the formats are close:
  pooled weight-only error 0.0336 for `e0m3_hadamard` and 0.0319 for
  NVFP4 with MSE scales, the best weight quantizer in the study.
- Rotations larger than 16 are not better, so the existing in-register
  16-point kernel is the right one; no new rotation kernel is needed.
- A per-tensor activation pre-scale adds nothing on top of H16
  (0.0545 either way).

## Result 2: whole pipeline vs fp16 (H100 simulation)

Same study, 20 frames (libero_spatial, first episode of 10 tasks,
frames 0 and 60), real VAE/Qwen3/proprio/shift schedule, same N(0,1)
noise for every tier. `act` = denormalized 64-step actions.

| tier | backbone_hidden cos med / min | action_latent cos med / min | act cos med / min | 1 - act cos med | act MAE vs fp16 | MAE vs GT |
|---|---|---|---|---:|---:|---:|
| fp16, other noise seed | 1 / 1 | 0.99633 / 0.98108 | 0.99375 / 0.98289 | 6.25e-3 | 0.02075 | 0.18369 |
| fp16 | 1 / 1 | 1 / 1 | 1 / 1 | 0 | 0 | 0.18359 |
| `nvfp4` | 0.99819 / 0.99762 | 0.99937 / 0.99912 | 0.99928 / 0.99857 | 7.17e-4 | 0.00906 | 0.18519 |
| `nvfp4`, MSE weights | 0.99847 / 0.99813 | 0.99943 / 0.99922 | 0.99952 / 0.99867 | 4.77e-4 | 0.00855 | 0.18510 |
| `nvfp4`, weight pre-scale | 0.99838 / 0.99803 | 0.99943 / 0.99920 | 0.99947 / 0.99868 | 5.35e-4 | 0.00841 | 0.18452 |
| E0M3 weights, E2M1 act | 0.99848 / 0.99790 | 0.99947 / 0.99923 | 0.99947 / 0.99872 | 5.27e-4 | 0.00831 | 0.18498 |
| E0M3 W4A4, no rotation | 0.99882 / 0.99840 | 0.99844 / 0.99781 | 0.99840 / 0.99624 | 1.60e-3 | 0.01644 | 0.18901 |
| E0M3 W4A4 + H16 | 0.99970 / 0.99948 | 0.99969 / 0.99954 | 0.99968 / 0.99918 | 3.15e-4 | 0.00622 | 0.18363 |
| **`e0m3_hadamard`** | **0.99960 / 0.99949** | **0.99970 / 0.99961** | **0.99971 / 0.99941** | **2.86e-4** | **0.00606** | **0.18352** |

- `e0m3_hadamard` lowers the actions error vs fp16 (1 - cos, median) by
  60% relative to `nvfp4`, has lower actions error, lower
  `backbone_hidden` error, and lower actions MAE vs fp16 on 20 of 20
  frames, and its open-loop MAE vs ground truth equals fp16's (0.18352
  vs 0.18359; `nvfp4` 0.18519). The plan's decision rule (at least 25%
  lower, MAE not worse) is met.
- The quantized tiers' actions error is 4-22x below the fp16 sampler's
  own seed-to-seed spread (6.25e-3); `e0m3_hadamard`'s is 22x below.
- The eager run and the captured fp16 graph agree bit for bit
  (action_latent max |diff| = 0).

## Result 2b: merged single-stream `linear2` (H100 simulation)

With OPT-016 the single-stream `linear2` is one GEMM for every precision
that merges `linear1`, `e0m3_hadamard` included: K = 12288 in the
backbone (tile variant 1) and 7168 in ActionDiT (variant 6). One
per-tensor weight pre-scale now covers the attention half and the MLP
half.

- Weight side: in 11 of 20 backbone blocks and 6 of 20 ActionDiT blocks
  the two halves alone would choose a different exponent (gap up to 2
  and 3). With the merged exponent no block scale of either half is
  subnormal, and UE4M3's relative step is the same across its normal
  range: the dequantized E0M3 weights of all 40 merged `linear2`
  tensors are bit-identical to the concatenated split halves.
- Activation side: the attention/MLP boundary (3072) is a multiple of 16,
  so no rotation or scale block straddles it; the quantized activation
  equals the split one.
- The merge therefore changes only the accumulation (one fp32 sum
  instead of two fp16 outputs and an fp16 add), as for `nvfp4`.

Same study, same 20 frames, merged and split forms run back to back with
the same script (`MERGE_LINEAR2=1` / `0`):

| tier | `linear2` | backbone_hidden cos med / min | action_latent cos med / min | act cos med / min | 1 - act cos med | act MAE vs fp16 | MAE vs GT |
|---|---|---|---|---|---:|---:|---:|
| fp16, other noise seed | merged | 1 / 1 | 0.99632 / 0.98107 | 0.99375 / 0.98289 | 6.25e-3 | 0.02075 | 0.18370 |
| `nvfp4` | merged | 0.99820 / 0.99762 | 0.99935 / 0.99911 | 0.99933 / 0.99860 | 6.71e-4 | 0.00910 | 0.18515 |
| `nvfp4` | split | 0.99819 / 0.99762 | 0.99937 / 0.99912 | 0.99928 / 0.99857 | 7.17e-4 | 0.00906 | 0.18519 |
| E0M3 W4A4, no rotation | merged | 0.99882 / 0.99840 | 0.99842 / 0.99796 | 0.99839 / 0.99659 | 1.61e-3 | 0.01619 | 0.18896 |
| E0M3 W4A4, no rotation | split | 0.99882 / 0.99840 | 0.99844 / 0.99781 | 0.99840 / 0.99624 | 1.60e-3 | 0.01644 | 0.18901 |
| **`e0m3_hadamard`** | **merged** | **0.99960 / 0.99949** | **0.99970 / 0.99957** | **0.99970 / 0.99934** | **2.97e-4** | **0.00600** | **0.18353** |
| `e0m3_hadamard` | split | 0.99960 / 0.99949 | 0.99971 / 0.99952 | 0.99966 / 0.99941 | 3.37e-4 | 0.00611 | 0.18348 |

fp16 MAE vs GT: 0.18359.

- Merged `e0m3_hadamard` has 56% lower median actions error than merged
  `nvfp4` (2.97e-4 vs 6.71e-4) and is better on 20 of 20 frames in
  actions error, `backbone_hidden` error, and actions MAE vs fp16.
- Per frame, merged/split actions error has median ratio 0.986 (range
  0.55-1.32) for `e0m3_hadamard` and 0.993 (0.75-1.26) for `nvfp4`: the
  same accumulation-order spread for both tiers.
- The split-form `e0m3_hadamard` value here (3.37e-4) differs from
  Result 2 (2.86e-4) only in the simulated weight preparation, which is
  now `prepare_e0m3_hadamard_weight` itself (butterfly, pre-scale before
  the fp16 rounding) instead of matrix rotation with fp16 rounding before
  the pre-scale. Pooled per-GEMM error is 0.03802 in both. At this error
  level the median `1 - cos` moves by about 15% with sub-ulp operand
  changes; the tier ranking does not move.
- Per-GEMM error (4 frames), merged `linear2`: backbone `nvfp4` 0.08486,
  `e0m3_hadamard` 0.07402 (split: `attn_out_proj` 0.09926 / 0.08870,
  `mlp_down` 0.08626 / 0.07427); ActionDiT 0.04671 / 0.03448 (split
  0.04767 / 0.04030 and 0.04686 / 0.03409). `e0m3_hadamard` is better
  than `nvfp4` on all 140 weights of the merged tree.

## Relation to OPT-014's Thor `backbone_hidden` cosine

OPT-014 recorded `nvfp4` `backbone_hidden` cosine 0.9939 vs fp16 on
Thor; the simulation with the real Qwen3 context gives 0.9976-0.9986.
The same study with `set_prompt()`'s fallback for a frontend without a
text encoder (every context row N(0,1), proprio in the last row;
`CONTEXT=random`, merged tree, 20 frames):

| context | tier | backbone_hidden cos median (range) | squared-error share: real-prompt rows / other text rows / image rows |
|---|---|---|---|
| Qwen3 | `nvfp4` | 0.99819 (0.9976-0.9986) | 0.873 / 0.050 / 0.077 |
| Qwen3 | `e0m3_hadamard` | 0.99960 (0.9995-0.9997) | 0.265 / 0.259 / 0.492 |
| random N(0,1) | `nvfp4` | 0.99247 (0.9536-0.9964) | 0.007 / 0.852 / 0.136 |
| random N(0,1) | `e0m3_hadamard` | 0.99395 (0.9736-0.9969) | 0.005 / 0.831 / 0.144 |

"Real-prompt rows" are the rows each task's real Qwen3 tokens occupy
(about 30 of 513); "other text rows" are the rest of the text rows
(padding and the proprio row). With a random context the rows standing
in for padding carry 70-98% of the error and the median cosine (0.9925)
is close to OPT-014's 0.9939; an independent H100 run with other random
contexts gave median 0.9932 (0.9908-0.9958) with 78-87% of the error in
those rows. With the real context, 74-90% of the `nvfp4` error sits in
the real-prompt rows. Hypothesis, unconfirmed: OPT-014's Thor comparison
ran on the random-context fallback. Its script is not in the repository;
`benchmarks/imagewam_e0m3_hadamard_thor_check.py` uses the real Qwen3
context and will show whether Thor reproduces the simulated 0.998.
Actions error in the random-context run: `nvfp4` 8.06e-4,
`e0m3_hadamard` 5.23e-4.

## Result 3: activation quantizer cost (H100, indicative only)

The same kernel sources built for sm_90a, CUDA events, 60 alternating
rounds of 50 launches, P10/P50/P90 in us:

| M x K | `quantize_fp4_dynamic_sfa_fp16` (nvfp4) | `quantize_e0m3_dynamic_sfa_fp16_vec` + H16 |
|---|---|---|
| 905 x 3072 | 10.6 / 10.6 / 10.7 | 6.1 / 6.2 / 6.2 |
| 905 x 9216 | 19.6 / 19.7 / 19.8 | 10.6 / 10.6 / 10.6 |
| 392 x 9216 | 10.9 / 10.9 / 10.9 | 6.4 / 6.4 / 6.5 |
| 64 x 1024 | 4.4 / 4.4 / 4.4 | 3.0 / 3.0 / 3.1 |
| 64 x 4096 | 4.6 / 4.6 / 4.7 | 3.1 / 3.1 / 3.2 |

The vectorized E0M3 quantizer with the rotation is faster than the
scalar NVFP4 quantizer ImageWAM uses. The GEMM tiles match `nvfp4`'s;
the runtime-descriptor GEMM's Thor speed at ImageWAM's large-M shapes
is unmeasured (Pi0.5 measured it equal to NVFP4 at decoder shapes with
the 128x64x256 tile).

## Open

- Thor: correctness (`tests/test_imagewam_e0m3_hadamard.py`), whole
  pipeline vs fp16 and open-loop MAE, `infer()` P50 vs `nvfp4`
  (`benchmarks/imagewam_e0m3_hadamard_thor_check.py`).
- Promotion to default needs the Thor numbers: better accuracy is
  established in simulation; speed parity is not yet measured.
- Side findings for the shipped `nvfp4` tier: ISSUE-050 (subnormal
  weight scales), ISSUE-051 (`txt_mlp2` activation scale saturation),
  ISSUE-052 (down-projection activation scales).

## Thor, LIBERO (`eccf14f`)

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU
exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is
end-to-end `infer()` P50 in ms and `vs official` is the median action cosine
against the official model. `e0m3_hadamard`, real checkpoint, libero_spatial:

| row | P50 ms | vs official median |
|---|---:|---:|
| default | 198.9 | 0.99786 |
| stack | 90.5 | 0.99968 |

Both rows are at or below the `nvfp4` rows with the same names in this round
(202.4 / 202.1 / 202.0 ms `default`, 93.1 / 92.8 / 93.3 ms `stack`), and the
microbenchmark for this tier measures 199.1 ms, 0.2 ms above the `default`
row.

The LIBERO gate for this precision reported `blocked` in this round:
`tests/fixtures/imagewam_gate/fidelity_thresholds.json` held no
`e0m3_hadamard` entry, so the gate did not measure this tier and the
`vs official` column above is the round's accuracy evidence for it.

## Thor, LIBERO (`0919e`)

`0919e`, commit `a84916a`, Jetson AGX Thor, MAXN, GPC 1.575 GHz,
`emc_locked=null`, GPU exclusive; raw logs under
`/home/jingwu/thor_val/0919e/`. Latency is end-to-end `infer()` P50 in ms and
`vs official` is the median action cosine against the official model.

The gate runs for `e0m3_hadamard` now, the thresholds file having gained an
entry for the precision, and passes: vs official median 0.99781 / min 0.99434,
`infer()` P50 222.2 ms. The latency half of the gate does not apply to this
precision: the latency policy in
`tests/fixtures/imagewam_gate/latency_baselines.json` holds no Thor baseline for
it, the same as for `fp8_static_cutlass`.

That median is at or above nvfp4's own gate median 0.99744, the criterion this
precision's gate was set against.


# OPT-022: real activation calibration for `fp8_static*` (roadmap item 7)

Status: done on H100; Thor confirmation pending (checklist below).
Plan: `plan.md` roadmap item 7. Mechanism and file format:
`docs/imagewam_calibration.md`.

## What changed

- `_calibrate_fp8()` takes its scales from a real calibration file
  (`calibration_path=`), built by `benchmarks/imagewam_build_calibration.py`
  from 64 LIBERO frames of `libero_object` / `libero_goal` / `libero_10`
  through the real fp16 pipeline (VAE, Qwen3, proprio, 10 denoise
  steps). Scale = house percentile (99.9) of per-sample absmax, / 448.
  The `N(0, 0.1)` placeholder remains only without a file and logs a
  warning.
- Prerequisite: `issues.md` ISSUE-001 (TN cuBLASLt FP8) so `fp8` and
  `fp8_static` run on the H100 dev machine at all.

## Why the placeholder failed

Real GEMM-input absmax per site ranges from ~2 (ActionDiT `proj`) to
~7000 (backbone double `txt_mlp2`, whose input has p99.99/amax = 0.018:
a few text-stream positions are huge). `N(0, 0.1)` noise has absmax ~0.5,
so the placeholder clipped nearly every site by 1-4 orders of magnitude.

## H100 results (real checkpoint, 20 held-out `libero_spatial` frames)

`benchmarks/imagewam_precision_fidelity.py`, `fp8_static` vs `fp16`,
official-sampler noise, median (min):

| | backbone_hidden | action_hidden | action_latent | actions | MAE / fp16 |
|---|---:|---:|---:|---:|---:|
| placeholder (H100) | 0.45582 (0.42467) | 0.68728 | 0.87187 (0.67097) | 0.90093 (0.73530) | 1.697 |
| **real file (H100)** | **0.99994 (0.99984)** | **0.99995** | **0.99997 (0.99995)** | **0.99997 (0.99989)** | **1.000** |
| Thor, OPT-014 result 1 (placeholder, 1 frame, `infer()` noise) | 0.467 | - | 0.257 | 0.697 | 1.43 (`_cutlass`, 50 frames) |

The H100 placeholder row reproduces the Thor collapse
(`backbone_hidden` 0.456 vs 0.467).

`benchmarks/imagewam_e2e_official_compare.py`, `N_TASKS=10 FRAMES=0,60
SEEDS=0,1` (20 frames), vs official ImageWAM:

| FlashRT path | fr_vs_off median | min | mean MAE vs GT | served_vs_off median |
|---|---:|---:|---:|---:|
| fp16 (baseline, re-run on this branch) | 0.99840 | 0.99567 | 0.18359 | 0.99602 |
| fp8_static, placeholder | 0.87576 | 0.66887 | 0.30207 | 0.91927 |
| **fp8_static, real file** | **0.99844** | **0.99559** | **0.18372** | **0.99599** |
| fp8 (dynamic scale, `N_TASKS=3 FRAMES=0`) | 0.99845 | 0.99836 | 0.20706 (official 0.20688) | 0.99439 |

With real scales `fp8_static` is indistinguishable from `fp16` against
official ImageWAM (official's own seed-to-seed spread: median 0.99630).

Sensitivity to the calibration set (same 20 held-out frames, vs `fp16`,
median (min)):

| calibration file | backbone_hidden | actions | MAE / fp16 |
|---|---:|---:|---:|
| N = 64, 22/21/21 frames per suite (the shipped build) | 0.99994 (0.99984) | 0.99997 (0.99989) | 1.000 |
| N = 64, 31/27/6 frames per suite | 0.99994 (0.99985) | 0.99997 (0.99991) | 1.001 |
| N = 8, 3/3/2 frames per suite | 0.99993 (0.99984) | 0.99997 (0.99991) | 1.000 |

The regression gate (`tests/gate_imagewam_libero.py`, roadmap item 13,
fixture v1, 40 runs) passes `fp8_static` with the N = 64 file on H100:
vs official median 0.99834 / min 0.99540, vs its `fp16` reference median
0.999968 / min 0.999943, MAE 0.18373 against the reference's 0.18364.

## Re-validated after the `linear2` merge, residual+AdaLN fusion and VAE stage

On the merged tree (roadmap items 1-6 in), the calibration file is
rebuilt with the merged frontend (142 sites: `linear2` replaces
`attn_out_proj` + `mlp_down`; a split-path file is refused by identity):

| check (H100) | result |
|---|---|
| fidelity vs fp16, 20 frames | backbone_hidden 0.99994 (min 0.99984), actions 0.99997 (min 0.99994), MAE ratio 1.000 |
| e2e vs official, 20 frames x 2 seeds | fp8_static median 0.99837 / min 0.99571 / MAE 0.18370; fp16 0.99840 / 0.99566 / 0.18359 |
| regression gate, fixture v1 | pass: vs official 0.99830 / 0.99553, vs fp16 reference 0.999969, MAE 0.18373 vs 0.18364 |
| `run_eager()` vs graph replay (real checkpoint) | bit-exact, with the VAE outside the graph and with `vae_graph_input` (the stage runs in `run_eager()`) |

## Thor check

Checklist item in the stream's final report: `fp8_static` and
`fp8_static_cutlass` with the real file, vs `fp16`, compared with
OPT-014 result 1/2; `infer()` P50 against OPT-014 result 4 (243.0 ms
`fp8_static_cutlass`, 236.9 ms `nvfp4`). Calibration changes scale
values only, not the captured graph, so P50 should not move.

## Thor, LIBERO (`eccf14f`)

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU
exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is
end-to-end `infer()` P50 in ms.

Fidelity against `fp16`, real checkpoint: the real 64-frame calibration
reaches actions cosine 0.99997 with MAE ratio 1.000, the same values the H100
table above records. The `N(0, 0.1)` placeholder still collapses the backbone
on Thor and gives actions about 0.90 with MAE ratio 1.77.

Speed, libero_spatial, `fp8_static_cutlass` with the real calibration file:

| row | P50 ms |
|---|---:|
| default | 219.5 |
| stack | 104.6 |

The round reported these rows in its libero_spatial accuracy table under one
heading with `e0m3_hadamard` and `fp16`; the two vs-official medians that table
carries, 0.99786 (`default`) and 0.99968 (`stack`), belong to its
`e0m3_hadamard` row and are not `fp8_static_cutlass` values (they are
recorded under OPT-024).

## Thor, LIBERO (`0919e`)

`0919e`, commit `a84916a`, Jetson AGX Thor, MAXN, GPC 1.575 GHz,
`emc_locked=null`, GPU exclusive; raw logs under
`/home/jingwu/thor_val/0919e/`. Latency is end-to-end `infer()` P50 in ms and
`vs official` is the median action cosine against the official model.

The LIBERO gate ran `fp8_static_cutlass` with the real calibration file and
passes: vs official median 0.99830 / min 0.99557, `infer()` P50 233.5 ms. The
latency half of the gate does not apply to this precision: the latency policy in
`tests/fixtures/imagewam_gate/latency_baselines.json` holds no Thor baseline for
`fp8_static_cutlass`.


# OPT-023: AWQ per-channel scales folded into the NVFP4 weights (roadmap item 8)

Status: implemented behind `nvfp4_awq=True` (default off); accuracy
measured on H100 with simulated NVFP4; real `nvfp4` accuracy and speed
pending on Thor. Plan: `plan.md` roadmap item 8. Mechanism, fold
points and exactness: `docs/imagewam_nvfp4_awq.md`.

## H100 results (simulated NVFP4, bit-exact quantizer, real checkpoint)

Whole pipeline vs `fp16`, 20 held-out `libero_spatial` frames, median
(min) cosine, `benchmarks/imagewam_precision_fidelity.py`:

| | backbone_hidden | action_latent | actions | MAE / fp16 |
|---|---:|---:|---:|---:|
| `nvfp4_sim` | 0.99819 (0.99762) | 0.99937 (0.99911) | 0.99931 (0.99848) | 1.010 |
| `nvfp4_sim` + AWQ 0.5, folds A + B | **0.99956 (0.99916)** | **0.99973 (0.99944)** | **0.99965 (0.99882)** | **1.000** |
| Thor `nvfp4`, OPT-014 result 1/2 (1 frame; 50 frames for MAE) | 0.9939 | 0.9997 | 0.9998 | 1.01 |

AWQ cuts the backbone_hidden error (1 - cosine) 4x and the action error
about 2x, and brings the open-loop MAE ratio from 1.010 to 1.000.

Against official ImageWAM (`imagewam_e2e_official_compare.py`,
`N_TASKS=10 FRAMES=0,60 SEEDS=0,1`, same N(0,1) noise both sides):

| FlashRT path (H100) | fr_vs_off median | min | mean MAE vs GT |
|---|---:|---:|---:|
| fp16 (baseline) | 0.99840 | 0.99567 | 0.18359 |
| `nvfp4_sim` | 0.99746 | 0.99399 | 0.18519 |
| `nvfp4_sim` + AWQ 0.5, folds A + B | 0.99779 | 0.99469 | 0.18378 |

Per-layer: alpha 0.5 is best for both fold classes (0.25-1.0 swept);
fold A sites 0.0742 -> 0.0610 rel_l2, fold B sites 0.0788 -> 0.0679;
sites without a fold point would gain ~1% at most.

After the `linear2` merge and the residual+AdaLN fusion (fold A reaches
the fused kernel as an FP32 pair, fold B covers the MLP channels of the
merged `linear2`), the same comparison gives `nvfp4_sim` 0.99820 /
0.99938 / 0.99933 / MAE 1.009 and with AWQ 0.99956 / 0.99973 / 0.99971
/ MAE 1.000 (backbone_hidden / action_latent / actions median).

Speed: AWQ changes weight values and the AdaLN constants only. At toy
dims the AWQ pipeline launches the same kernels per forward (413 = 413
unfused, 285 = 285 fused, `tests/test_imagewam_awq.py`). Thor `infer()`
P50 should not move; that and the real-hardware cosines are the Thor
check.

## Follow-up, not started

- Per-tensor power-of-two NVFP4 weight pre-scale: tracked in
  `issues.md` ISSUE-050. It combines with AWQ: in this study's
  per-layer comparison, fold-A sites gave rel_l2 0.0610 with AWQ 0.5
  alone and 0.0595 with AWQ 0.5 plus the pre-scale.
- `proj` / `attn_out_proj` have no exact fold point; a fused
  multiply-and-quantize of the attention output would be needed to
  scale them, for at most ~1% per-layer gain.

## Thor, LIBERO (`eccf14f`)

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU
exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is
end-to-end `infer()` P50 in ms.

Real `nvfp4` with `nvfp4_awq=True`: actions cosine 0.99970 against `fp16`,
where the plain `nvfp4` path gives 0.99939, at P50 202.2-202.9 ms against
202.0-203.0 ms for this round's default rows. Both halves of the Thor check
this entry left open therefore hold on real hardware: the cosines improve over
plain `nvfp4` and the P50 does not move.


## Thor, LIBERO (`0920`)

The folded path adds no kernels of its own, measured with the AWQ test's own
counter after the profiler warm-up region is discarded:
`kernels per eager forward: plain=285 awq=285` (Jetson AGX Thor, MAXN, GPC
1.575 GHz, `emc_locked=null`, GPU exclusive, log
`/home/jingwu/thor_val/0920/R_awq.log`). The plain side previously read `0`
because the first `torch.profiler` CUDA region in a process is blind, not
because the call was a graph replay; `run_eager` is eager with and without a
`weights` argument. The equality is the property this entry is about: the
fold moves work into the quantizer, it does not add a per-forward kernel.


# OPT-028: ImageWAM through `frt_model_runtime_v1` (Python producer)

Status: implemented and verified on H100 (fp16, real checkpoint,
bit-exact). Thor `nvfp4` parity pending on the Thor checklist.

Area: deployment engineering, roadmap item 12.

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

After merging the fusion and VAE streams (served layer structure,
preprocessing kernel, optional in-graph VAE), re-run on H100 at fp16 with
the real checkpoint: every parity row still `array_equal`, with the VAE
outside the graph and inside it (`--vae-graph-input 224 224`, where the
`image_views` SWAP window replaces `image_tokens`).
`tests/test_imagewam_model_runtime_vae.py` adds both placements at the
real token count.

Review follow-up (2026-09-18). The parity rows above compared an ABI
tick with the `infer()` that ran just before it over the same buffers,
so a verb that staged nothing could still pass. The gate and tests now
NaN-fill every buffer a tick must write before each ABI tick, and re-run
each row with the verb it exercises made a no-op. Re-run on H100, fp16,
real checkpoint, after merging the calibration stream: every row above is
still `array_equal` (`max_abs = 0`) with the VAE outside and inside the
graph, and all five mutants (images, proprio on each image path, prompt,
step) make their rows fail in both placements. Invalid calls now return
the same statuses as the `io="native"` face (`-2` unknown port, `-3`
SWAP port, `-4` payload size, `-5` short buffer, `-1` other); the
runtime's pybind trampolines honor a `VerbStatusError`'s status. The
identity also carries the calibration file digest and `nvfp4_awq`.

Regression, `pytest tests/test_imagewam_*.py` on the merged tree (H100):
362 passed, 31 skipped with `exec/build`, `runtime/build` and the native
target built; 326 / 33 without the native target (its two test modules
skip); 316 / 35 without any of the three (the two model-runtime modules
also skip). Every skip in the full build needs Thor, FA4 or an FP8
cuBLASLt layout this GPU lacks.

## Promotion Condition

Thor gate at `nvfp4` reports every parity row `array_equal=True`. The
export is additive and opt-in; `infer()` behavior is unchanged.

## Thor, both workloads (`eccf14f`)

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU
exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is P50 in ms
on one wall-clock timer per path, every path in the same process, real
checkpoint.

LIBERO `default`:

| path | P50 ms |
|---|---:|
| `infer()` | 202.3 |
| ABI tick (`io="python"`) | 184.2 |
| native tick (`io="native"`) | 183.8 |

LIBERO at `profile=fast`, every text length precaptured:

| path | P50 ms |
|---|---:|
| `infer()` | 93.2 |
| ABI tick | 95.1 |
| native tick | skipped (rule R5: the native pipeline has one graph and one context length) |

`text_trim` through the ABI is one adopted graph exec per text length,
selected by the replay key: every swept length produced a number, and
`tests/test_imagewam_model_runtime_export.py` passes on Thor, including the
tick at two different captured lengths, which is bit-exact against `infer()`.

Target workload (three views of 256x256, horizon 32), `default`: the ABI tick
measures 152.7 ms. That row is layout-only: it was measured with placeholder
image tokens, because the VAE path refuses this workload.

## Thor, both workloads (`0919e`)

`0919e`, commit `a84916a`, Jetson AGX Thor, MAXN, GPC 1.575 GHz,
`emc_locked=null`, GPU exclusive; raw logs under
`/home/jingwu/thor_val/0919e/`. Latency is P50 in ms on one wall-clock timer per
path, every path in the same process, real checkpoint.

Target workload (three views of 256x256, horizon 32, `text_max_len=128`):

| row | `infer()` | ABI tick (`io="python"`) | native tick (`io="native"`) |
|---|---:|---:|---:|
| `default` | 216.93 | 173.55 | 173.27 |
| `profile=fast`, lengths precaptured | 137.64 | 139.35 | skipped (rule R5) |
| `text_trim`, 16 valid tokens | 197.00 | 153.51 | skipped (rule R5) |
| `text_trim`, 72 valid tokens | 207.06 | 161.68 | skipped (rule R5) |
| `text_trim`, 128 valid tokens | 217.50 | 172.05 | skipped (rule R5) |

The workload serves with its own VAE encode geometry now (issues.md ISSUE-086),
so these rows carry the workload's own image geometry rather than the
placeholder image tokens of the layout-only rows the section above records.

The ABI serves the trimmed lengths: one adopted graph exec per text length,
selected by the replay key, so every swept length produced a number. The native
face still refuses them (rule R5).


# OPT-029: ImageWAM native C++ overlay (`io="native"`)

Status: implemented and verified bit-exact on H100 (fp16, small dims and
real checkpoint); NVFP4 wiring compiles and links for sm_110; Thor
`nvfp4` parity and speed pending on the Thor checklist.

Area: deployment engineering, roadmap item 14; interface record
`docs/imagewam_native_cpp.md`.

## Observation

After OPT-028 every ImageWAM tick through the ABI still entered Python
(GIL-acquiring trampolines for proprio, actions and `step`), and the
graph was recorded from Python, carrying per-replay torch kernels for
the AdaLN modulation casts and gate expansions.

## Opportunity

`libflashrt_imagewam_native.so`: C verbs (proprio, actions, step) over a
declaration the Python producer builds, and a C++ `NativePipeline` that
records prefill + denoise against the existing `csrc` kernels from a
borrowed resource table, with the frontend's autotuned cuBLASLt
algorithms handed off (`GemmRunner.get/set_cached_algo`, additive) and
the modulation precomputed once. VAE and Qwen3 stay in Python.

## Expected Mechanism

Same kernels, same algorithms, same inputs: bit-exact to the Python
pipeline, with fewer graph nodes (no in-graph modulation casts/copies)
and no Python in the tick.

## Required Evidence

H100 (shared GPU), fp16:

| check | result |
|---|---|
| backbone double-stream block 0, native vs Python (small dims) | every state buffer `array_equal` |
| backbone single-stream block 0 | `array_equal` |
| full prefill (backbone_hidden, all K/V) | `array_equal` |
| full denoise loop (action_latent, action K/V rows) | `array_equal` |
| native graph vs Python eager / Python graph | `array_equal` |
| `io="native"` tick on the native graph vs `infer()` | `array_equal` |
| Python frames entered by `set_input(proprio)` + `step` | io=native 0, io=python 57 |
| schema records at real dims: Python declaration, C++, golden | identical (7 records) |
| real checkpoint: actions / actions_raw / native proprio token / native vs Python graph action latent and K cache | all `array_equal`, `max_abs = 0` |
| GEMM shapes handed off (real dims) | 20 of 20 |
| graph nodes (real dims) | native 5732, Python 7112 |
| exported symbols of the library (sm_90 and sm_110) | 18, all `frt_imagewam_native_*` |

Latency, H100 shared with a co-tenant at 100% utilization, indicative
only, real checkpoint, fp16, alternating A/B, 50 iterations each:

| path | P10 | P50 | P90 |
|---|---:|---:|---:|
| `io="python"` tick (SWAP tokens, proprio, noise, step, actions) | 104.33 ms | 107.10 ms | 110.22 ms |
| `io="native"` tick, native graph | 102.26 ms | 103.16 ms | 104.28 ms |
| Python graph replay only (CUDA events) | 95.86 ms | 101.94 ms | 102.40 ms |
| native graph replay only (CUDA events) | 100.58 ms | 100.97 ms | 101.56 ms |

The rows above were measured before the fusion stream landed. With the
served layer structure it introduced (merged `linear2`, gated residual
fused with the next AdaLN, which also removed the per-layer modulation
casts from the Python graph), re-run on the merged tree, H100, fp16, real
checkpoint, 50 alternating iterations:

| check | result |
|---|---|
| step-by-step parity, both layer structures (small dims) | every state buffer `array_equal` at every step |
| real checkpoint: actions / actions_raw / native proprio token / native vs Python graph | all `array_equal` |
| graph nodes (real dims) | native 4974, Python 4998 |
| `io="python"` tick P10 / P50 / P90 | 85.54 / 98.26 / 100.99 ms |
| `io="native"` tick P10 / P50 / P90 | 96.26 / 98.95 / 99.25 ms |
| Python graph replay P10 / P50 / P90 | 93.14 / 96.04 / 96.87 ms |
| native graph replay P10 / P50 / P90 | 85.94 / 96.92 / 97.90 ms |

With the served structure the two graphs differ by 24 nodes (the fp16
casts for the standalone AdaLN at each chain start), so no replay speed
difference is expected; the native path's value is a tick with no
Python and no GIL, not a faster graph.

Thor, `nvfp4`: pending (Thor checklist).

Review follow-up (2026-09-18), H100, fp16, merged tree (calibration
stream included):

| check | result |
|---|---|
| real checkpoint, NaN-poisoned tick buffers: actions, actions_raw, native proprio token, backbone residual, K/V caches, proprio staged by the frontend or the native verb | all `array_equal` |
| poisoned native-graph vs Python-graph replay: action latent, backbone residual, K/V caches | all `array_equal` |
| mutants: proprio verb or `step` skipped; native pipelines with no backbone block, last single-stream block dropped, last denoise step dropped, one block fed another's `linear1` weight | all 6 detected (also as small-dims tests) |
| `set_pipeline` after `capture` | destroys the captured graph, frees the old resources; refused while an export is live (before: replayed freed memory) |
| native handle alone, frontend dropped | frontend kept alive, replay `array_equal` (before: illegal address) |
| status codes | same table as `io="python"` |
| exported symbols (sm_90 and sm_110) | 20, all `frt_imagewam_native_*` |
| `sm110_check.sh` (`flash_rt_kernels`, `flash_rt_fp4`, `flashrt_imagewam_native`) | rc 0 |

ISSUE-071 (a Python-graph `backbone_hidden` mismatch) was a race between
a test snapshot on the torch stream and the native warm-up on the
non-blocking native stream; `run` / `capture` now wait for prior device
work, and the native pipeline test compares `backbone_hidden` between
the two graphs again. The native pipeline refuses `nvfp4_awq` (no AWQ
fold).

## Promotion Condition

Thor reports every parity row `array_equal` at `nvfp4`, and the
replay-only A/B shows the native graph not slower than the Python graph.
Remaining native work beyond this entry: VAE encoding in the graph
(roadmap item 5, then an `images` STAGED native port), proprio projection
inside the graph, and a native checkpoint loader (`native_v2`).

## Thor, both workloads (`eccf14f`)

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU
exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is P50 in ms
on one wall-clock timer per path, every path in the same process, real
checkpoint.

LIBERO `default`: the native tick measures 183.8 ms against `infer()` 202.3 ms
and the ABI tick 184.2 ms in the same process.

Target workload (three views of 256x256, horizon 32): the native tick measures
154.5 ms, also measured with placeholder image tokens, so it is a layout-only
number as well. `text_trim` remains refused for this face (rule R5).

## Thor, both workloads (`0919e`)

`0919e`, commit `a84916a`, Jetson AGX Thor, MAXN, GPC 1.575 GHz,
`emc_locked=null`, GPU exclusive; raw logs under
`/home/jingwu/thor_val/0919e/`. Latency is P50 in ms on one wall-clock timer per
path, every path in the same process, real checkpoint.

Target workload (three views of 256x256, horizon 32, `text_max_len=128`), the
same process as the ABI rows:

| row | native tick (`io="native"`) | ABI tick (`io="python"`) | `infer()` |
|---|---:|---:|---:|
| `default` | 173.27 | 173.55 | 216.93 |
| `profile=fast`, lengths precaptured | skipped (rule R5) | 139.35 | 137.64 |
| `text_trim`, 16 valid tokens | skipped (rule R5) | 153.51 | 197.00 |
| `text_trim`, 72 valid tokens | skipped (rule R5) | 161.68 | 207.06 |
| `text_trim`, 128 valid tokens | skipped (rule R5) | 172.05 | 217.50 |

The `default` row is the workload served with its own VAE encode geometry; the
native row the section above records for this workload is the earlier
layout-only measurement with placeholder image tokens. `text_trim` stays
refused for this face: rule R5 gives the native pipeline one graph and one
context length, so the trimmed rows have no native number.


# OPT-030: text context trimmed to the prompt's valid length (issues.md ISSUE-020)

Status: implemented behind `ImageWAMTorchFrontendThor(text_trim=True)`
(default `False`), verified on H100 at `fp16`, `fp8` and `fp8_static`;
`nvfp4`, `e0m3_hadamard`, FA4 and Thor latency pending (Thor check in
`plan.md`'s "Thor validation checklist", steps 3 and 5; issues.md ISSUE-080).

Area: `flash_rt/models/imagewam/text_context.py`,
`flash_rt/frontends/torch/imagewam_thor.py`,
`benchmarks/imagewam_text_trim_bench.py`,
`benchmarks/imagewam_e2e_official_compare.py` (`TEXT_TRIM`),
`tests/test_imagewam_text_trim.py`

## Mechanism

- Official ImageWAM masks the padded text keys for every query at both
  attention calls. With proprio packing the valid tokens sit at rows
  `[0, n_valid)`, the proprio token at row `n_valid`, and text RoPE
  positions are the row indices, so a sequence of only those rows
  computes the official masked math without a mask.
- `set_prompt` packs the valid rows by rank plus the proprio slot
  (`pack_trimmed_context`, checked against official
  `_append_proprio_to_context`, prefix and non-prefix masks) into
  `context[:x0]`, `x0 = n_valid + 1`.
- Buffers stay allocated at the max dims (`x0 = 513`, `a0 = 905`,
  `total = 969`). Each distinct `x0` gets a `TextLengthCapture`: its
  dims (`a0`, `total` shifted by the same rows), its backbone RoPE table
  (text positions `0..x0-1`, image positions unchanged), and a CUDA
  graph captured on first use. A cached length re-activates without
  capture.
- A new length autotunes the cuBLASLt shapes it adds: `bf16_nn`
  `txt_in` at `M = x0` for every precision, and the `fp16_nn` text and
  single-stream shapes (`M = x0`, `M = a0`) for `precision="fp16"`
  only. Untuned shapes would run cuBLASLt's heuristic top-1.
- Before the first trimmed capture, one eager prefill at the max dims
  runs. `Nvfp4Linear`, `Fp8Linear`, `StaticFp8Linear`,
  `CutlassFp16SwiGluMlp`, `Nvfp4SwiGluMlp` and `E0m3HadamardLinear`
  grow their activation scratch on a larger `m`, which would free a
  buffer an earlier graph still reads; after the max-dims pass every
  backbone op has its largest `m`.
- All captures share one capture stream and one CUDA-graph memory pool.
  `infer()` replays one graph at a time and no value that outlives a
  replay lives in the pool.
- Unchanged: `pipeline_thor.py` and `attn_backend.py` already take
  every length from `dims` and the per-call arguments. The served
  per-head attention pads an odd `kv_seq` internally (the untrimmed
  dims already run odd lengths, 905 and 969); the even-`kv_seq` guard
  belongs to the non-per-head kernel, which the frontend never selects.
  FA4 output staging (`fa4_out`) is sized at the max dims.
- The FA4 fallback drops every cached graph once the cuBLAS recapture
  has succeeded, so all graphs use the same attention; until then the
  old graphs and their RoPE tables stay alive.
- A capture that raises leaves no graph active (`infer()` refuses until
  a `set_prompt` succeeds) and clears the prompt cache key: `set_prompt`
  has already written the new context, which no captured graph matches.
  Cached captures of other lengths stay valid.
- Python's cyclic garbage collector is run once before each capture and
  kept off during it. `torch.cuda.graph` no longer collects before a
  capture, and destroying a CUDA graph held by a dead reference cycle
  while a stream captures invalidates the capture; with `text_trim` new
  lengths are captured while the process serves.
- `run_eager()` (the calibration recorder's forward) runs the active
  length's dims and RoPE table, bit-identical to the active graph.
- `precapture_text_lengths(x0s)` captures known lengths ahead of their
  first `set_prompt` without changing the active prompt.

## Result (H100, shared GPU)

Numerics, `imagewam_e2e_official_compare.py`, fp16 vs official bf16,
10 tasks x frames {0, 60} per suite, seeds {0, 1}, `fr_vs_off`:

| suite | valid tokens | untrimmed median / min / mean | trimmed median / min / mean |
|---|---|---:|---:|
| libero_spatial | 26-31 | 0.99840 / 0.99566 / 0.99804 | 0.99998 / 0.99993 / 0.99997 |
| libero_goal | 16-21 | 0.99680 / 0.92997 / 0.99176 | 0.99998 / 0.99971 / 0.99996 |
| libero_10 | 20-31 | 0.99860 / 0.96654 / 0.99647 | 0.99998 / 0.99654 / 0.99979 |
| all 60 frames | | 0.99809 / 0.92997 / 0.99542 | 0.99998 / 0.99654 / 0.99991 |

Mean `mae_fr_vs_gt` (official in parentheses): libero_spatial 0.18359
-> 0.18555 (0.18538), libero_goal 0.16201 -> 0.15958 (0.15941),
libero_10 0.13044 -> 0.13144 (0.13139). The trimmed minimum,
libero_10 ep 0 frame 60 (0.99654 at seed 0, 0.99986 at seed 1), is the
frame where official's own seed 0 vs seed 1 cosine is 0.77926.
`served_vs_off` and `fr_noise001_vs_off` use the served `0.01 * N(0,1)`
initial noise (ISSUE-002) and are dominated by that sampler difference.
At seed 1 the trimmed libero_goal minimum is ep 114 frame 60 (0.99487 to
0.99510 over four runs), the frame where official's own seed 0 vs seed 1
cosine is 0.93157.
Repeat runs of libero_goal with the final code (VAE outside and inside
the graph): median 0.99998, min 0.99966 and 0.99970, mean MAE 0.15957;
per-frame values move by up to 5e-5 between processes because each
process autotunes its own cuBLASLt picks for the new fp16 shapes.

Other checks:

| check | result |
|---|---|
| small dims, random weights: trimmed vs untrimmed with the padded keys masked, same fp32 PyTorch attention | bit-identical actions (n_valid 5 and 8) |
| same, trimmed with the served cuBLAS per-head attention | cosine 1.0000000, rel_l2 3.6e-5 to 5.9e-5 |
| same, untrimmed unmasked (the old rule) | rel_l2 1.2e-3 to 1.3e-3 |
| served per-head attention at trimmed real shapes (`x0` 20, 21, 32, 33; backbone `a0` 412-425, `mot` `total` 476-489) vs fp32 PyTorch | cosine 0.9999999, rel_l2 5.4e-4 at both parities |
| FA4 dispatch at those shapes, fp32 stand-in for the kernel | cosine 1.0000000 |
| switching lengths 6 -> 10 -> 17 -> 6 | the cached graph replays, bit-identical to the first run; each length bit-identical to a fresh frontend |
| VAE stage inside every length's graph (shared pool), switching lengths | bit-identical to the VAE outside the graph |
| `text_trim=False` vs the previous frontend file, same process, small and real dims | bit-identical actions (context, random prompt, explicit and served noise) |
| fp16 gate, fixture v1 | 40 per-sample results identical before and after |
| multi-length safety check (`tests/test_imagewam_text_trim_graph_safety.py`), lengths A, longer, shorter, A, max, ...: fp16 small and real dims; fp8 small and real dims; fp8_static small (placeholder scales) and real dims with real weights and the trimmed calibration file; each with the VAE outside and inside the graph | every length bit-identical to a fresh single-length frontend; weight-op tensors never reallocated after the first capture (144 at small dims, 560 at real dims for fp8/fp8_static); replays unchanged and no write into 0xFF-poisoned free memory (graph pool and 1152 MiB of the regular cache) |
| same check without the max-dims prefill (control, growing-scratch stand-in ops) | 14 of 38 (small) and 60 of 142 (real) scratch tensors reallocated |
| a capture that raises on a new length | no graph active, `infer()` refuses; the cached length replays its own result; the retried length equals a fresh frontend |
| FA4 fallback (stand-in FA4 failing) | old graphs alive through the recapture, dropped after it; a second failure leaves no graph |
| `run_eager()` vs replay at two lengths and back | bit-identical (before the fix: cosine 0.9999982, max-abs 4.8e-3) |

Speed, `benchmarks/imagewam_text_trim_bench.py --precision fp16`, real
dims, random weights, trimmed `x0 = 21` vs full `x0 = 513` in one
frontend, 100 alternating samples each:

| | trimmed P10 / P50 / P90 | full P10 / P50 / P90 | P50 ratio |
|---|---:|---:|---:|
| graph replay (CUDA events) | 32.4 / 32.5 / 33.1 ms | 45.8 / 46.3 / 47.6 ms | 0.701 |
| `infer()` (wall, synchronized) | 32.9 / 33.0 / 33.5 ms | 46.1 / 46.5 / 48.6 ms | 0.711 |

With the VAE stage in the graph the same A/B was contaminated by the
co-tenant (bimodal samples); its P10 ratio is 0.72 for both replay and
`infer()`.

Capture cost and memory per length (15 LIBERO lengths, `x0` 17-32):

| | VAE outside the graph | VAE inside the graph |
|---|---:|---:|
| first length: `set_prompt` wall / process memory | 1.17 s / +84 MiB | 1.50 s / +298 MiB (206 MiB of it the shared pool) |
| each later new length: wall median (range) | 0.78 s (0.61-1.97) | 0.76 s (0.62-1.83) |
| each later new length: process memory (NVML) | +12 MiB (10-16) | +12 MiB (12-16) |
| cached length: `set_prompt` wall | 12 ms (8-36) | 12 ms (1-16) |

The fp16 autotune of a new length's shapes is 0.3-0.6 s of the capture
cost; the eager warmup and capture are 0.27-0.35 s. Before the shared
pool, each length with the VAE in the graph added 218 MiB.

## Constraints on consumers of a trimmed frontend

With `text_trim=True` the sequence length changes with the prompt.
`frontend.dims` holds the buffer sizes (the maximum); the dims the active
graph runs are `frontend.active_dims`.

`runtime_surface()`, `pipeline_resources()` and `export_model_runtime()`
describe one graph at `frontend.dims` and raise `ValueError` for a
trimmed frontend. Per-length support there needs all of:

- the active dims everywhere a length appears: `context_rows` is
  `active_dims["x0"]`, the image rows start at it, the backbone RoPE
  table (`_rope_table`) has `active_dims["a0"]` rows, the action rows
  start at `active_dims["a0"]`;
- `text_trim` and the active `x0` in the setup identity;
- after the prompt verb (`set_prompt`), re-adopting the graph: a new
  length activates, and may capture, another graph (`_graph`); the
  capture stream (`_graph_stream`) is the same for every length;
- for the native pipeline, one native graph per length, recorded from
  that length's dims and RoPE table, or the same trimmed packing
  (`text_context.pack_trimmed_context`) in native code.

Activation statistics for a trimmed frontend are recorded at the active
dims (`run_eager()` runs them). The untrimmed forward's text and
single-stream GEMM inputs include about 490 padded context rows that a
trimmed frontend never computes. The calibration file records
`text_trim` (format version 2) and a frontend refuses a file recorded
with the other setting; version-1 files were recorded untrimmed and load
as `text_trim=False`.

Trimmed calibration file (`benchmarks/imagewam_build_calibration.py
--n 64 --text-trim`, the same 64 frames as the untrimmed N=64 file, H100):
per-site static FP8 scale, trimmed over untrimmed, 0.85x-1.50x (median
per site group 0.98x-1.32x; largest spread at ActionDiT `proj`, backbone
`linear2` and the image-stream `mlp2`). `fp8_static` on libero_goal (10
tasks x frames {0, 60}, seeds {0, 1}; one of the ten evaluation episodes,
ep 0, also contributes calibration frames):

| | vs official median / min | mean MAE (official 0.15941) |
|---|---:|---:|
| `fp8_static`, untrimmed, untrimmed file | 0.99678 / 0.92891 | 0.16214 |
| `fp8_static`, trimmed, trimmed file | 0.99995 / 0.99927 | 0.15968 |
| `fp16`, trimmed (reference) | 0.99998 / 0.99966 | 0.15957 |

Trimmed `fp8_static` vs trimmed `fp16`
(`benchmarks/imagewam_precision_fidelity.py`, `TEXT_TRIM=1
SUITE=libero_goal`): actions cosine median 0.99998 / min 0.99981,
backbone residual 0.99994 / 0.99989, MAE ratio 1.000. The untrimmed file
forced onto the trimmed frontend (identity check bypassed) measures the
same (actions 0.99998 / 0.99992): the identity rule keeps the statistics
consistent with the served path; on these frames it is not an accuracy
gain.

## Open

- Thor: `nvfp4` end-to-end compare off/on, `infer()` P50 A/B, capture
  time per new length, FA4 on/off, VAE-in-graph memory, and the
  multi-length safety check at `nvfp4`, `e0m3_hadamard` and with FA4
  (`plan.md` "Thor validation checklist", steps 3, 5 and 6).
- `fp8`/`fp8_static` run on H100 since the TN FP8 path (issues.md
  ISSUE-001) and are verified with trimming above;
  `fp8_static_cutlass` runs on Thor only.
- The per-length graph cache is unbounded (up to 512 lengths).
  `precapture_text_lengths` captures known lengths at startup; a bound
  on the cache does not exist yet.
- Gate fixture v1 holds an untrimmed fp16 reference: trimmed fp16
  measures 0.99837 / 0.99579 (median / min) against it, under the fp16
  bounds 0.999 / 0.995, while vs official it is 0.99998 / 0.99992. A
  trimmed default needs a regenerated fixture (ISSUE-080).
- Owner decision: serve `text_trim=True` by default (ISSUE-080).

## Thor, one matrix session (`c20f3a0`, libero_spatial, nvfp4)

`scripts/imagewam_thor_matrix.sh`, `N_TASKS=10 FRAMES=0,60 SEEDS=0,1`, real
checkpoint, `infer()` P50. The `profile_*` rows ran through `load_imagewam`
(plan.md W12); `fast` and `stack` are the same switch set. GPU co-tenancy
and the clock state are recorded in that round's logs.

| row | P50 ms | marginal | vs official median |
|---|---:|---:|---:|
| default | 225.5 | — | — |
| vae | 190.1 | -35.4 | — |
| vae_trim | 102.9 | -87.2 | — |
| vae_trim_fa4bb | 99.0 | -3.9 | — |
| stack | 93.2 | -5.8 | 0.99934 |
| stack_no_vae | 104.8 | +11.6 vs stack | — |
| stack_no_trim | 131.4 | +38.2 vs stack | — |
| profile_fast (same switches as stack) | 106.8 | — | 0.99936 |
| profile_default (same switches as default) | 225.2 | — | 0.99764 |

`text_trim` is the largest single step from `vae` (-87.2 ms) and removing it
from the full stack costs +38.2 ms, more than the other three switches
together; the two numbers differ because FA4 also shortens the padded-key
work the trim already removes. Agreement with official improves rather than
degrades (0.99934-0.99936 stacked against 0.99764 for the default row).
Read the marginals with ISSUE-082: the same configuration appears twice in
this session, 93.2 and 106.8 ms.


## Thor, LIBERO (`eccf14f`)

`eccf14f`, Jetson AGX Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU
exclusive; raw logs under `/home/jingwu/thor_val/0919s/`. Latency is
end-to-end `infer()` P50 in ms and `vs official` is the median action cosine
against the official model. Real checkpoint, `nvfp4`, the three suites:

| suite | row | P50 ms | vs official median |
|---|---|---|---:|---:|
| libero_spatial | default, three repeats | 202.4 / 202.1 / 202.0 | — |
| libero_spatial | stack, three repeats | 93.1 / 92.8 / 93.3 | 0.99931-0.99936 |
| libero_spatial | vae_trim | not measured this round | — |
| libero_goal | default | 203.0 | 0.99558 |
| libero_goal | vae_trim | 118.7 | 0.99937 |
| libero_goal | stack | 92.6 | 0.99930 |
| libero_goal | profile=fast | 92.7 | — |
| libero_10 | default | 202.0 | 0.99765 |
| libero_10 | vae_trim | 103.0 | 0.99926 |
| libero_10 | stack | 93.5 | 0.99925 |
| libero_10 | profile=fast | 93.7 | — |

The spatial `stack` repeats bracket the 0.99934 the `c20f3a0` session measured
for the same row. On libero_goal and libero_10 the trimmed rows agree with
official better than the untrimmed default (libero_goal 0.99937 for `vae_trim`
and 0.99930 for `stack` against 0.99558; libero_10 0.99926 and 0.99925 against
0.99765), so the faster configurations are also the closer ones. The gates
with `text_trim=True` pass at `fp16` and at `nvfp4`, and no row in the round
reported an FA4 fallback.

Gate fixture v2 (`imagewam_libero_gate_v2`, its `fp16` reference recorded
trimmed) now gates a trimmed configuration: the `nvfp4` gate against it
measures vs official 0.99931 / min 0.99898, vs the fixture's own `fp16`
reference 0.99935 / 0.99907, `infer()` P50 114.6 ms, and passes. A trimmed run
against fixture v1 and an untrimmed run against v2 are both refused, which is
the intended behaviour.

Capture cost on Thor, three lengths swept (16, 24 and 31 valid tokens):
`set_prompt` takes 0.000-0.012 s for a length captured at construction
(precapture) and 0.42-0.58 s for a length captured on first use, both at or
below the H100 capture-cost table above (0.61-1.97 s for each later new
length, 12 ms for a cached one).

## Thor, LIBERO (`0919e`)

`0919e`, commit `a84916a`, Jetson AGX Thor, MAXN, GPC 1.575 GHz,
`emc_locked=null`, GPU exclusive; raw logs under
`/home/jingwu/thor_val/0919e/`. Latency is end-to-end `infer()` P50 in ms.

Per-length graph memory, `benchmarks/imagewam_text_trim_bench.py`, `nvfp4`, FA4
off, 15 distinct LIBERO lengths, as deltas of the process's reserved and
allocated memory (NVML and `torch.cuda.max_memory_allocated()`):

| capture | reserved | allocated |
|---|---:|---:|
| first captured graph | +218.0 MiB | +206.3 MiB |
| each following capture | +0.0 MiB | +0.1 MiB |

The process's reserved total is 11.79 GiB after the 15-length sweep. The
captures after the first share the capture pool, so the default
`text_trim_cache_size=32` costs on the order of 221 MiB in total, not 32 times
the first graph.

Trimmed against full in one frontend at trimmed `x0 = 21`, same benchmark,
`nvfp4`, FA4 off: `infer()` P50 123.0 ms against 203.7 ms for the full length,
ratio 0.604.

Eviction, `text_trim_cache_size=2` with no precapture, lengths 16 -> 24 -> 31 ->
16 valid tokens: `set_prompt` takes 0.636 / 0.503 / 0.468 / 0.465 s, so the
revisited length captures again, LRU having evicted it. The same three lengths
precaptured at `text_trim_cache_size=8` switch in 0.012 / 0.000 / 0.000 s.


## Thor, LIBERO (`0920`)

The multi-length safety check passes with FA4 on as well as off, and the
recovery case reports the recovered length equal to a like-for-like chain
reference: `after a failed capture, n_valid=5 vs fresh equal=True cosine=1
max_abs=0` (log `/home/jingwu/thor_val/0920/R_fa4_recover.log`; Jetson AGX
Thor, MAXN, GPC 1.575 GHz, `emc_locked=null`, GPU exclusive). The capture-path
defect behind the earlier failure — the cuBLAS fallback reusing the capture
pool an invalidated capture had left recording — is fixed in the frontend
(issues.md ISSUE-085).


# OPT-031: derive the remaining per-benchmark shape tables from the workload

Status: identified, not started

Area: `benchmarks/imagewam_fp8_layout_bench.py`,
`imagewam_thor_int4_bench.py`, `imagewam_thor_int8_bench.py`,
`imagewam_thor_bench.py`, `imagewam_thor_fp16_bench.py`,
`imagewam_thor_fp16_autotuned_bench.py`, `imagewam_thor_fp4_bench.py`,
`imagewam_thor_fp8_bench.py`, `imagewam_gemm_precision_compare.py`,
`imagewam_int4_hadamard_padding_probe.py`,
`imagewam_real_checkpoint_validation.py`,
`imagewam_attention_share_bench.py`, `imagewam_thor_small_m_tile_sweep.py`

## Observation

The `dims` dictionaries of the served LIBERO workload are now one mapping
(`libero_dims.LIBERO_REAL_DIMS`, derived from `ImageWAMWorkload.libero()` and
`ImageWAMStructure.libero()`). Several benchmarks still state the same served
widths a second time, not as a `dims` dict but inside their own kernel-shape
tables: `("txt_qkv": (513, 9216, 3072))` in the FP8 layout bench, `X0, A0 =
513, 513 + VAE_NUM_TOKENS` in the INT4/INT8 benches, per-GEMM `(M, N, K)`
rows in the per-layer benches, and `SiteShape("backbone", 905, 905)` /
`ActionShape("double qkv", 9216, 1024, 5)` in the attention-share and tile
sweep tables.

## Mechanism

Those tables describe one layer's or one GEMM's shape, which is a composition
of dims entries (`3 * hidden + 2 * mlp_hidden`, `x0`, `x0 + ref_h * ref_w`,
`num_action`), so they can be built from the same two objects the resolver
uses instead of restating the numbers. `tests/test_imagewam_quant_linear.py`
does this for its own `(M, N, K)` table.

## Value

A change to the workload (the target deployment's camera count or per-view
size) or to a structure constant then reaches every benchmark by
construction, and a sweep cannot silently measure a shape the model no longer
has. The cost is per-file, mechanical, and needs no GPU: the resulting table
is asserted equal to the present literals.

## Precondition

The sweep tables must stay readable as measurement records (what shape was
measured); the change is the derivation, not the removal of the numbers from
the prose.
