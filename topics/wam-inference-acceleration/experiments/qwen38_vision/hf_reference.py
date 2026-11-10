"""HF reference for the e2e prompt set: transformers Qwen3_5ForConditionalGeneration on the FP8 checkpoint.

Per prompt (e2e_prompts.build_prompts, same processor settings as the FlashRT side): the prompt's token ids, the
first-token logits of one forward over the prompt, and NEW_TOKENS greedy tokens (``generate``, do_sample=False).
Weights load in the checkpoint's FP8 format, BF16 elsewhere, on one GPU; with --dequantize the FP8 weights are
dequantized to BF16 at load (FineGrainedFP8Config(dequantize=True); no FP8 matmul kernels needed, no activation
quantization). With --offload-gib N, layers beyond N GiB of GPU memory go to CPU memory. Saves a torch file with one
entry per prompt plus peak GPU memory and timings.

usage: python hf_reference.py FP8_DIR PROCESSOR_DIR DATA_ROOT OUT_PT [--new-tokens 64] [--offload-gib N] [--dequantize]
                              [--only NAME ...]   (--new-tokens 0: first-token logits only)
"""
import argparse
import time

import torch
from transformers import AutoProcessor, FineGrainedFP8Config, Qwen3_5ForConditionalGeneration

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo

import e2e_prompts

ap = argparse.ArgumentParser()
ap.add_argument("fp8_dir")
ap.add_argument("processor_dir")
ap.add_argument("data_root")
ap.add_argument("out")
ap.add_argument("--new-tokens", type=int, default=64)
ap.add_argument("--only", nargs="+", default=None)
ap.add_argument("--offload-gib", type=float, default=None)
ap.add_argument("--dequantize", action="store_true")
args = ap.parse_args()

if args.offload_gib is None:
    device_map, max_memory = {"": "cuda:0"}, None
else:
    device_map, max_memory = "auto", {0: f"{args.offload_gib}GiB", "cpu": "56GiB"}
quant = {"quantization_config": FineGrainedFP8Config(dequantize=True)} if args.dequantize else {}
t0 = time.time()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    args.fp8_dir, dtype=torch.bfloat16, device_map=device_map, max_memory=max_memory,
    attn_implementation="sdpa", **quant).eval()
load_s = time.time() - t0
print(f"HF loaded in {load_s:.0f} s, GPU allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
      f"offload {args.offload_gib}", flush=True)
proc = AutoProcessor.from_pretrained(args.processor_dir)
out = {"meta": {"fp8_dir": args.fp8_dir, "offload_gib": args.offload_gib, "dequantize": args.dequantize,
                "load_s": load_s,
                "new_tokens": args.new_tokens}, "prompts": {}}
torch.cuda.reset_peak_memory_stats()
for p in [q for q in e2e_prompts.build_prompts(args.data_root, proc.tokenizer) if not args.only or q.name in args.only]:
    geo.set_processor_max_pixels(proc, p.max_pixels)
    inp = proc.apply_chat_template(p.messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt", **e2e_prompts.CHAT_TEMPLATE_KWARGS).to("cuda:0")
    t1 = time.time()
    with torch.no_grad():
        logits = model(**inp).logits[0, -1].float().cpu()
        t2 = time.time()
        gen = model.generate(**inp, do_sample=False, max_new_tokens=args.new_tokens) if args.new_tokens else None
    t3 = time.time()
    tokens = gen[0, inp["input_ids"].shape[1]:].tolist() if gen is not None else []
    out["prompts"][p.name] = {"input_ids": inp["input_ids"][0].cpu(), "first_logits": logits.to(torch.bfloat16),
                              "tokens": tokens, "forward_s": t2 - t1, "generate_s": t3 - t2}
    print(f"{p.name:22s} S {inp['input_ids'].shape[1]:5d} forward {t2 - t1:.1f} s generate {t3 - t2:.1f} s | "
          f"{proc.tokenizer.decode(tokens, skip_special_tokens=True)[:160]!r}", flush=True)
out["meta"]["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
torch.save(out, args.out)
print(f"saved {args.out}; peak GPU allocated {out['meta']['peak_allocated_gib']:.2f} GiB", flush=True)
