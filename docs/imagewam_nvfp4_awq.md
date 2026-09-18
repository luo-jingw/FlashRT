# ImageWAM NVFP4: simulator and AWQ

## NVFP4 numerics (`flash_rt/models/imagewam/nvfp4_sim.py`)

`Nvfp4Linear` quantizes weights (offline) and activations (per call)
with `quantize_fp4_dynamic_sfa_fp16` (`csrc/quantize/quantize_fp4_sfa.cu`):

- blocks of 16 elements along K;
- block scale `__nv_fp8_e4m3(max(amax / 6, 1e-12))`: E4M3,
  round-to-nearest-even, saturating at 448, subnormal below 2^-6; no
  per-tensor global scale (`fp4_gemm` runs with `alpha = 1`);
- elements E2M1 of `x / scale` with thresholds 0.25, 0.75, 1.25, 1.75,
  2.5, 3.5, 5.0 (`<=`, ties to the smaller magnitude).

`nvfp4_sim.quantize_nvfp4` reproduces this bit for bit: 0 packed-byte and
0 scale-byte mismatches against the real device code
(`quantize_fp4_dynamic.cu`, same helpers, JIT-compiled for sm_90) on
random, weight-like and adversarial inputs (subnormal and zero scales,
saturation, threshold ties) — `tests/test_imagewam_nvfp4_sim.py`, which
uses `flash_rt_fp4`'s own quantizer on a Blackwell build. Dequantized
values are exact in fp16, so `quant_linear.SimNvfp4Linear` (fake-quantized operands,
fp16 GEMM with fp32 accumulation) differs from the hardware GEMM only in
accumulation order: on `test_imagewam_quant_linear.py`'s NVFP4 case it
gives cosine 0.989133 against `Fp16Linear`, the value the real
`Nvfp4Linear` gave on Thor.

`precision="nvfp4_sim"` serves the simulator through the normal frontend
(same K=7 / N=7 fp16 fallback as `nvfp4`). It is an accuracy tool:
`infer()` takes ~440 ms on the H100 against ~150 ms for `fp16`.

On the real checkpoint 73-99% (median ~96%) of the weight blocks per
site group have a subnormal or zero E4M3 scale: ImageWAM weights are
~N(0, 0.02), so `amax / 6` of a block sits below 2^-6.

## AWQ (`flash_rt/models/imagewam/awq.py`)

Per NVFP4 GEMM input site, from the calibration file's per-channel
absmax `a` ([`imagewam_calibration.md`](imagewam_calibration.md)):
`s = clamp((a / mean(a)) ** alpha, 0.25, 4)` (Pi0.5's convention). The
weight rows are multiplied by `s` before quantization and the GEMM input
must be `x / s`. Two exact folds produce `x / s` without a kernel:

- Fold A, AdaLN-fed GEMMs (backbone double `{txt,img}_qkv`,
  `{txt,img}_mlp0`; backbone single `linear1`; ActionDiT double `qkv`,
  `mlp0`; ActionDiT single `linear1`): the modulation pair of the
  `ada_layer_norm_*` kernel that produces the input becomes
  `1 + scale' = (1 + scale) / s`, `shift' = shift / s`, computed in fp32
  and stored in fp16. Modulations are shared by all layers of a stream,
  so each layer caches its own folded pair per modulation
  (`AwqScaledLinear.folded_modulation`); the cache fills during graph
  warmup and replay only reads it.
- Fold B, down projections (backbone double `{txt,img}_mlp2`, single
  `mlp_down`; ActionDiT double `mlp2`, single `mlp_down`): the input is
  `silu(gate) * up`, so the preceding weight's `up` columns are
  multiplied by `1/s` before that weight is quantized.
- `proj` / `attn_out_proj` have no exact fold point (V is shared by both
  backbone streams and the ActionDiT's joint attention) and stay
  unscaled.

Fold precision (`tests/test_imagewam_awq.py`, `seq = 905`, `dim = 3072`,
BF16 residual, near-cancelling `1 + scale` included):

| output vs ideal fp32 | rel_l2 | max-abs |
|---|---:|---:|
| unfolded AdaLN (the kernel's own fp16 rounding) vs `x` | 2.40e-4 | 4.3e-3 |
| fold A vs `x / s` | 2.72e-4 | 1.8e-2 |
| separate per-channel multiply vs `x / s` | 2.59e-4 | 1.7e-2 |
| fold B vs `(silu(g) * u) / s` | 2.24e-4 | 5.2e-3 |

At toy dims the full pipeline with every AWQ weight transformed and the
fold hook active, but no quantization, reproduces the untransformed
pipeline (cosine 1.0000000, `action_latent` rel_l2 4.8e-5), and launches
the same number of kernels per forward (413 = 413).

Frontend: `ImageWAMTorchFrontendThor(precision="nvfp4" | "nvfp4_sim",
ckpt_path=..., calibration_path=..., nvfp4_awq=True, awq_alpha=0.5,
awq_scope="adaln+down")`. Default `nvfp4_awq=False`.

## Measured accuracy (H100, simulated NVFP4, real checkpoint)

Per-layer (`benchmarks/imagewam_nvfp4_awq_study.py`, 10 held-out
`libero_spatial` frames): rel_l2 of each simulated NVFP4 GEMM against
the fp32 reference on its true fp16 input, averaged over the sites of a
fold class; AWQ applied at every site to show each class's potential.

| sites | nvfp4 | AWQ 0.25 | AWQ 0.5 | AWQ 0.75 | AWQ 1.0 |
|---|---:|---:|---:|---:|---:|
| fold A (70) | 0.0742 | 0.0650 | **0.0610** | 0.0624 | 0.0657 |
| fold B (55) | 0.0788 | 0.0694 | **0.0679** | 0.0716 | 0.0784 |
| no fold point (55) | 0.0790 | 0.0781 | 0.0782 | 0.0787 | 0.0801 |

Largest per-group gains at alpha 0.5 (median rel_l2): backbone
`txt_mlp2` 0.0610 -> 0.0222, ActionDiT single `linear1` 0.0923 ->
0.0642, ActionDiT single `mlp_down` 0.0611 -> 0.0453; backbone single
`mlp_down` 0.1118 -> 0.1124 is the one group that does not improve.

Whole pipeline (`benchmarks/imagewam_precision_fidelity.py`,
`nvfp4_sim` vs `fp16`, 20 held-out frames, median (min) cosine):

| | backbone_hidden | action_hidden | action_latent | actions | MAE / fp16 |
|---|---:|---:|---:|---:|---:|
| no AWQ | 0.99819 (0.99762) | 0.99964 | 0.99937 (0.99911) | 0.99931 (0.99848) | 1.010 |
| AWQ 0.5, fold A only | 0.99807 (0.99768) | 0.99976 | 0.99964 (0.99958) | 0.99966 (0.99925) | 1.004 |
| AWQ 0.5, folds A + B | 0.99956 (0.99916) | 0.99979 | 0.99973 (0.99944) | 0.99965 (0.99882) | 1.000 |

Fold B carries the backbone gain (`txt_mlp2`'s outlier channels); fold A
carries most of the ActionDiT gain.

Against official ImageWAM (`imagewam_e2e_official_compare.py`,
`N_TASKS=10 FRAMES=0,60 SEEDS=0,1`):

| FlashRT path | fr_vs_off median | min | mean MAE vs GT |
|---|---:|---:|---:|
| `fp16` | 0.99840 | 0.99567 | 0.18359 |
| `nvfp4_sim` | 0.99746 | 0.99399 | 0.18519 |
| `nvfp4_sim` + AWQ 0.5, folds A + B | 0.99779 | 0.99469 | 0.18378 |
