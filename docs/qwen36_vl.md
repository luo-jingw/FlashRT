# Image input for Qwen3.6 / Qwen3.8-27B NVFP4 on RTX 5090 and Thor

`Qwen36VlTorchFrontendRtx` (`flash_rt/frontends/torch/qwen36_vl_rtx.py`) adds image input to the Qwen3.5-architecture
27B models (`Qwen3_5ForConditionalGeneration`: Qwen3.6-27B, Qwen3.8-27B and fine-tunes of them) on the NVFP4
language-model path of `Qwen36TorchFrontendRtx`. Video input is not supported.

## How it works

- Vision tower: `Qwen3VlVisionRtx`. The Qwen3.5 vision tower is the Qwen3-VL ViT without DeepStack (transformers
  `Qwen3_5VisionModel` subclasses `Qwen3VLVisionModel` and deletes the DeepStack mergers); depth 27, hidden 1152,
  16 heads, patch 16, 2x2 merge, output 5120. One ViT call per image.
- Image features: `Qwen36TorchFrontendRtx(..., extra_embedding_rows=N)` appends N rows to the embedding table. Each
  request writes the image features into those rows (`write_embedding_rows`) and replaces the prompt's image-token
  ids with `embedding_slot_base + j`; every embedding lookup (prefill, decode, verify, MTP) reads them.
- Positions: interleaved MRoPE (`mrope_section` [11, 11, 10] over the 64 rotary dimensions). Each request rewrites the
  language model's RoPE tables in place (`set_rope_rows`): MRoPE rows (t, h, w) for the prompt, 1-D rows continued
  from max(prompt position) + 1 after it; `reset_rope_rows` restores the tables. Every RoPE application, captured CUDA
  graphs included, reads these rows by cache position. Rows come from `_qwen36_mrope.text_rope_rows`, which equals
  transformers' `Qwen3_5TextRotaryEmbedding` exactly and the 1-D table bit for bit when the three axes are equal.
- The language model's forward code is unchanged.
- Draft length: `generate(..., spec_k=K)` passes K to the language model's speculative generation; K below 6 caps its
  prompt-length table (`_long_tq_effective_k`), 6 (the default) or more takes the table. On 22 recorded VoLoAgent
  harness calls (official Qwen3.8-27B with its MTP head; `research/experiments/harness_replay_5090/thor/
  planner_calls_vl_rtx.json`), the table gives K = 6 for decompose (2 130 tokens) and K = 3 for monitor (4 076):
  decompose decode 320 ms at K = 6 and 312 ms at K = 3, monitor 652 ms at both; replies identical.

## Thor (SM110)

`Qwen36VlTorchFrontendThor` (`flash_rt/frontends/torch/qwen36_vl_thor.py`) runs the same request flow with two
different components, chosen in the construction methods `_build_language_model` and `_build_vision`:

- language model `Qwen36TorchFrontendThor`, which subclasses the RTX frontend and keeps its embedding table, RoPE
  tables and their write / reset methods;
- ViT `Qwen3VlVisionRtx(fp8=False, attention_backend='sdpa')`: BF16 linears and SDPA attention (the Thor Qwen3-VL
  module has no FP8 block-128 ViT GEMM; FA2 is not built on Thor).

On the RTX 5090, this ViT configuration against the FP8 / FA2 default on the three images of a recorded decompose call:
merged-feature cosine mean 0.9986-0.9993 (min 0.962-0.996), 1.2-1.7x the time per image
(`research/experiments/harness_replay_5090/smoke/vit_thor_config_check.log`; on SM120 its linears run
`w16a16_gemm_sm120_bf16`, on Thor the Qwen3-VL module's cuBLASLt GEMM). With the RTX language model, this ViT
configuration changes 42 of 44 greedy replies on the 22 recorded harness calls and 4 of 22 parsed decisions
(`research/experiments/harness_replay_5090/README.md`).

Measured on a Jetson AGX Thor (THOR_STATUS_SUMMARY.md round `1104_h3`; official Qwen3.8-27B NVFP4 with its MTP head,
the 22 recorded calls, medians per call): decompose (2 131 tokens) ViT 140, prefill 3 652, decode 1 365-1 435, call
5 173-5 242 ms; monitor (4 079 tokens) ViT 269, prefill 4 356, decode 2 306-2 352, call 6 968-7 020 ms; 2.76 tokens
per main-model forward on monitor. Replies equal to the RTX 5090 references: 0 of 44. Parsed decisions (untimed
160-token calls, round `1104_h3b`): 18 of 22 equal to the references, the same count as the RTX language model with
this ViT configuration on the 5090, with the same two changes in meaning. With the MTP tail fc on cuBLASLt (now the
Thor frontend's default, inherited from the RTX frontend; round `1104_h3c`, commit `9609be92`): decompose ViT 136,
prefill 2 103, decode 1 122-1 136, call 3 377-3 390 ms; monitor ViT 269-270, prefill 3 986, decode 2 228-2 311, call
6 506-6 590 ms (5.9-7.6 s per call, all above the harness's 5.33 s monitor interval); 2.81-2.82 tokens per forward on
monitor; decisions 18 of 22. Decode ms per speculative cycle on the Thor is 136-137 ms in the first 2.5 minutes of a
round's GPU load and 87-88 ms after it, with the readable CPU and GPU clocks unchanged (issues.md ISSUE-112).

## Requirements

- Build: `flash_rt_kernels` without `FLASHRT_SLIM_BUILD` (Qwen3.6 linear-attention kernels) and with
  `-DFLASHRT_BUILD_QWEN3_VL=ON` (module `flash_rt_qwen3_vl_kernels`, and `w16a16_gemm_sm120_bf16` in
  `flash_rt_kernels`).
- Checkpoint: one directory with the compressed-tensors NVFP4 language model and the BF16 `model.visual.*` tensors in
  `model.safetensors`, plus `config.json`, tokenizer and processor files. A block-128 FP8 release converts with
  `research/experiments/qwen38_distill_bench/fp8_to_nvfp4_ct.py` (MLP and full-attention projections to NVFP4,
  linear-attention projections to BF16).
- MTP head for speculative decode: `FLASHRT_QWEN36_MTP_CKPT_DIR`, as for the text frontend.
- Prefill route: below `FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ` (default 512) a prompt is prefilled one token at a
  time: a single-image prompt of 277 tokens takes 19.7 s. With `max_seq` above `FLASHRT_QWEN36_LONG_CTX_THRESHOLD` and
  `FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ=1`, every prompt takes the chunked FP8-KV route (32.7 ms for the same
  prompt).

```python
import os
os.environ.update({"FLASHRT_QWEN36_MTP_CKPT_DIR": "<dir with mtp.safetensors>",
                   "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096", "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
                   "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "1"})
from PIL import Image
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx

fe = Qwen36VlTorchFrontendRtx("<nvfp4 dir with model.visual>", max_seq=8192, max_image_tokens=4096)
msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open("frame.png")},
                                     {"type": "text", "text": "Which objects are on the table?"}]}]
r = fe.generate(msgs, max_new_tokens=128)
print(r.text, r.timings_ms)
```

## Measured (RTX 5090, Qwen3.8-27B LoRA-merged checkpoint, 2026-11-03)

Component parity against transformers 5.17 (`research/experiments/qwen38_vision/`):

| check | result |
|---|---|
| ViT merged features vs `Qwen3_5VisionModel`, 6 inputs (grids 32x32 to 46x136, two images) | FP8 GEMMs cosine 0.9959-0.9984, BF16 GEMMs 0.9990-0.9998 |
| MRoPE position ids vs `get_rope_index`, 5 prompts | equal; deltas equal |
| RoPE rows vs `Qwen3_5TextRotaryEmbedding` | max abs difference 0 |
| text-only generation with 4096 reserved rows | tokens identical; +0.039 GiB; prefill / decode unchanged |
| prompt tokens moved into reserved rows (their own embeddings) | tokens identical |

Latency per stage, chunked route, 128 new tokens (`vl_bench_route1_5090.json`):

| input | image tokens | preprocess | ViT | prefill | decode (128 tok) |
|---|---:|---:|---:|---:|---:|
| one LIBERO frame | 256 | 3.4 ms | 7.0 ms | 32.7 ms | 1.14 s |
| two LIBERO frames | 512 | 4.3 ms | 13.5 ms | 51.7 ms | 1.19 s |
| FlashRT.png, full resolution | 1564 | 9.2 ms | 56.2 ms | 123.4 ms | 1.46 s |
| chart, full resolution | 6596 | 36.4 ms | 622 ms | 545 ms | 1.16 s |

GPU allocated after construction 20.83 GiB (4096 reserved rows); peak 22.74 GiB at 6619 prompt tokens.

End to end against transformers on the FP8 checkpoint dequantized to BF16 (the FP8 checkpoint does not fit the GPU
in transformers; 24 GiB on the GPU, the rest in CPU memory), first-token logits (`e2e_compare*_5090.json`):

| prompts | prompt tokens | cosine | top-1 | top-5 overlap |
|---|---|---|---|---|
| text, short (3) | 19-39 | 0.978-0.986 | 2/3 | 1.0 |
| text, length-matched (3) | 279-1579 | 0.946-0.968 | 3/3 | 0.6-0.8 |
| image (6) | 245-1582 | 0.892-0.951 | 6/6 | 0.6-1.0 |

On text prompts, shifting every RoPE position by one (which only re-rounds the BF16 table rows) moves this model's
first-token logits to cosine 0.924-0.99 against the unshifted run; the image rows lie within that spread.
