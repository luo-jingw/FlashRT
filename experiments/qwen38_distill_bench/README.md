# Qwen3.8-27B LoRA-merged FP8 checkpoint on the FlashRT Qwen3.6 frontend (RTX 5090)

Checkpoint: `qwen38-27b-master-full-gpt55-first-recovery-fp8` (Hugging Face dataset `LGG100/qwen27-tmp`, one 24.8 GB
`tar.zst`). Local copies: `tmp/checkpoints/qwen27-tmp/` (FP8 as published), `tmp/checkpoints/qwen27-tmp-nvfp4/`
(converted), `tmp/checkpoints/qwen38-27b-fp8-official-mtp/` (MTP head from `Qwen/Qwen3.8-27B-FP8`).

## Structure

- Origin (`merged_from.json`, `quantized_from.json`): `Qwen/Qwen3.8-27B` plus a LoRA adapter, merged, then quantized
  to FP8 in the layout of `Qwen/Qwen3.8-27B-FP8`.
- Architecture: `Qwen3_5ForConditionalGeneration`, the same config as Qwen3.6-27B and Qwen3.8-27B (64 layers, 48
  Gated DeltaNet linear-attention + 16 full-attention, hidden 5120, vocab 248,320). The config differs from the official
  FP8 config only in `dtype` fields (`text_config.dtype` float32) and the transformers version.
- Tensors: 1584, all identical in name, dtype and shape to `Qwen/Qwen3.8-27B-FP8`; linear layers FP8 e4m3 block-128 with
  BF16 `weight_scale_inv`; norms, embeddings, `lm_head`, linear-attention `A_log` / `dt_bias` / `conv1d` /
  `in_proj_a/b` BF16; vision tower (333 tensors) BF16.
- Missing relative to the official FP8 checkpoint: the MTP head (22 `mtp.*` tensors).
- Weight change against `Qwen/Qwen3.8-27B-FP8` (sampled layers, relative L2 of dequantized weights): 1.5–3.0 % on
  linear-attention, full-attention and MLP projections, 1.3 % on a vision block; norms identical.

## Running on FlashRT

- `quant='fp8'` (legacy path): loads through HF `from_pretrained` with `attn_implementation='flash_attention_2'`,
  which needs the `flash_attn` package (not in the `flashrt` environment), and runs prefill through the HF model; not
  used here.
- `quant='nvfp4'` (main path): requires a compressed-tensors NVFP4 checkpoint. `fp8_to_nvfp4_ct.py` converts the FP8
  checkpoint: MLP and full-attention projections to NVFP4 (256 linears, gate/up sharing one global scale),
  linear-attention projections dequantized to BF16 (144), vision tower kept in BF16 (333 tensors, read by the image
  frontend `qwen36_vl_rtx.py`), MTP dropped. 27.70 GB, 90 s (26.78 GB without the vision tower). NVFP4 vs
  dequantized FP8: relative error 0.095 per weight. The encoding was checked against `unsloth/Qwen3.8-27B-NVFP4`
  (`ct_convention_check.py`: published tensor decodes to rel. err 0.129 against the FP8 reference, 1.417 with the
  nibble order swapped).
- Kernel build: the Qwen3.6 linear-attention kernels are excluded by `FLASHRT_SLIM_BUILD=ON`; `build_full/` builds
  `flash_rt_kernels` with it OFF (515 bindings, a superset of the 436 in the slim build).
- Chat check (greedy, 128 tokens): coherent English with a leading `<think>` block.

## Speed (`context_bench.py`, `results_5090_nvfp4.json`)

RTX 5090, NVFP4 path, MTP speculative decode with the official Qwen3.8 MTP head, FP8 KV long-context route
(`max_seq` 8320, chunked prefill), 64 new tokens, one warmup then 3 timed runs (median; runs agree within 0.2 %).
Decode tok/s = 63 × 1000 / decode ms. `spec` = draft cycles / drafted tokens accepted.

| prompt | ctx | prefill ms | prefill tok/s | decode ms | decode tok/s | spec | peak alloc GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeated paragraph | 1024 | 76.5 | 13,377 | 735.8 | 85.6 | 31 / 34 | 20.45 |
| repeated paragraph | 4096 | 308.6 | 13,274 | 944.0 | 66.7 | 39 / 24 | 20.60 |
| repeated paragraph | 8192 | 654.4 | 12,519 | 937.7 | 67.2 | 38 / 27 | 20.60 |
| documentation text | 1024 | 76.7 | 13,342 | 831.4 | 75.8 | 35 / 29 | 20.60 |
| documentation text | 4096 | 309.6 | 13,230 | 871.9 | 72.3 | 36 / 27 | 20.60 |
| documentation text | 8192 | 656.6 | 12,476 | 891.3 | 70.7 | 36 / 29 | 20.60 |

- Prefill scales linearly with context (about 13 k tok/s) and does not depend on the prompt kind.
- Decode is set by MTP acceptance: 0.6–1.1 accepted drafts per cycle at 23–25 ms per cycle. The Qwen3.6-27B table in
  `docs/qwen36_nvfp4.md` (paired MTP head) has 2.2–3.0 accepted per cycle and 106–133 tok/s at 1K–8K. The MTP head
  here was trained for the base Qwen3.8-27B, not for the LoRA-merged weights.
