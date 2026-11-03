# Image input for the Qwen3.8-27B NVFP4 frontend: validation on RTX 5090

Checkpoint: `tmp/checkpoints/qwen27-tmp-nvfp4/` (converted from the LoRA-merged FP8 release, vision tower kept);
reference for the end-to-end rows: the FP8 release in transformers 5.17, dequantized to BF16. Results are summarized
in `docs/qwen36_vl.md` and in `plan.md` ("image input for the Qwen3.6 / Qwen3.8-27B NVFP4 frontend").

| script | measures | result files |
|---|---|---|
| `vit_parity.py` | `Qwen3VlVisionRtx` (FP8 / BF16 GEMMs) vs HF `Qwen3_5VisionModel` | `vit_parity_5090.json` |
| `mrope_parity.py` | MRoPE position ids vs HF `get_rope_index`; `text_rope_rows` vs HF rotary | `mrope_parity_5090.json` |
| `hook_checks.py` | reserved embedding rows and RoPE-row hooks on text prompts; RoPE re-rounding sensitivity | `hook_{baseline,hooks}_5090.json` |
| `vl_bench.py` | answers, per-stage latency, image tokens, memory, at long-route thresholds 512 and 1 | `vl_bench_route512.log`, `vl_bench_route1{.log,_5090.json}` |
| `hf_reference.py` | HF first-token logits and greedy tokens on `e2e_prompts.py` (local `.pt`, not tracked) | `hf_reference*.log` |
| `e2e_compare.py` | FlashRT vs the HF reference (FP8 ViT; `--vit-bf16`; length-matched text) | `e2e_compare*_5090.json`, `e2e_compare*.log` |
