"""FlashRT Qwen36VlTorchFrontendRtx (NVFP4) vs the HF reference (hf_reference.py, FP8 checkpoint) on the e2e prompts.

Per prompt: the prompt's token ids must equal the reference's; first-token logits (the language model's
``_logits_buf`` after a one-token generation; its argmax is checked against the generated token) vs the reference:
cosine, top-1 and top-5 agreement; NEW_TOKENS greedy tokens: identical prefix length and matching-token fraction,
both sides cut at the first end-of-sequence token. Text-only rows give the NVFP4-vs-FP8 floor that image rows are read
against. The language model runs with max_seq MAX_SEQ above the long-context threshold, FP8 KV and the chunked
prefill route for every prompt (ROUTE_MIN_SEQ 1). With --vit-bf16 the frontend's ViT is rebuilt with BF16 GEMMs
(``Qwen3VlVisionRtx(fp8=False)``) to separate the ViT's FP8 error from the language model's.

usage: python e2e_compare.py NVFP4_DIR MTP_DIR DATA_ROOT REF_PT OUT_JSON [--vit-bf16] [--only NAME ...]
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("mtp_dir")
ap.add_argument("data_root")
ap.add_argument("ref")
ap.add_argument("out")
ap.add_argument("--max-seq", type=int, default=8192)
ap.add_argument("--max-image-tokens", type=int, default=4096)
ap.add_argument("--vit-bf16", action="store_true")
ap.add_argument("--only", nargs="+", default=None)
args = ap.parse_args()
os.environ.update({
    "FLASHRT_QWEN36_MTP_CKPT_DIR": args.mtp_dir,
    "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096",
    "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
    "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

import torch  # noqa: E402

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo  # noqa: E402
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402

import e2e_prompts  # noqa: E402

ref = torch.load(args.ref)
fe = Qwen36VlTorchFrontendRtx(args.ckpt, max_seq=args.max_seq, max_image_tokens=args.max_image_tokens)
if args.vit_bf16:
    from flash_rt.frontends.torch._qwen3_vl_vision_rtx import Qwen3VlVisionRtx
    vcfg = json.load(open(os.path.join(args.ckpt, "config.json")))["vision_config"]
    fe.vision = None
    fe.vision = Qwen3VlVisionRtx(args.ckpt, device=fe.device, config=vcfg, fp8=False)
eos = fe._eos_token_ids
rows = []
for p in [q for q in e2e_prompts.build_prompts(args.data_root, fe.processor.tokenizer) if not args.only or q.name in args.only]:
    r = ref["prompts"][p.name]
    geo.set_processor_max_pixels(fe.processor, p.max_pixels)
    ids = fe.processor.apply_chat_template(p.messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", **e2e_prompts.CHAT_TEMPLATE_KWARGS)["input_ids"][0]
    ids_equal = bool(torch.equal(ids, r["input_ids"]))
    one = fe.generate(p.messages, max_new_tokens=1, chat_template_kwargs=e2e_prompts.CHAT_TEMPLATE_KWARGS)
    logits = fe.llm._logits_buf.float().view(-1).cpu()
    argmax_matches_token = (not one.token_ids) or int(logits.argmax()) == one.token_ids[0]
    full = fe.generate(p.messages, max_new_tokens=max(ref["meta"]["new_tokens"], 1),
                       chat_template_kwargs=e2e_prompts.CHAT_TEMPLATE_KWARGS)
    hf_logits = r["first_logits"].float()
    hf_tokens = r["tokens"]
    hf_tokens = hf_tokens[:next((i for i, t in enumerate(hf_tokens) if t in eos), len(hf_tokens))]
    ours = full.token_ids
    n = max(len(hf_tokens), len(ours), 1)
    prefix = next((i for i, (a, b) in enumerate(zip(ours, hf_tokens)) if a != b), min(len(ours), len(hf_tokens)))
    row = {"name": p.name, "kind": p.kind, "prompt_len": full.prompt_len, "image_tokens": full.num_image_tokens,
           "ids_equal": ids_equal, "argmax_matches_token": bool(argmax_matches_token),
           "first_logit_cos": float(torch.nn.functional.cosine_similarity(logits, hf_logits, dim=0)),
           "top1_equal": bool(logits.argmax() == hf_logits.argmax()),
           "top5_overlap": len(set(logits.topk(5).indices.tolist()) & set(hf_logits.topk(5).indices.tolist())) / 5,
           "identical_prefix": prefix, "ours_len": len(ours), "hf_len": len(hf_tokens),
           "match_fraction": sum(a == b for a, b in zip(ours, hf_tokens)) / n,
           "ours_text": full.text, "hf_text": fe.processor.tokenizer.decode(hf_tokens, skip_special_tokens=True),
           "timings_ms": full.timings_ms}
    rows.append(row)
    print(f"{p.name:22s} {p.kind:5s} S {row['prompt_len']:5d} ids_equal {ids_equal} | logit cos "
          f"{row['first_logit_cos']:.4f} top1 {row['top1_equal']} top5 {row['top5_overlap']:.1f} | prefix "
          f"{prefix}/{row['hf_len']} match {row['match_fraction']:.2f}", flush=True)
    print(f"   ours: {full.text[:200]!r}\n   hf:   {row['hf_text'][:200]!r}", flush=True)
json.dump({"ckpt": args.ckpt, "ref": args.ref, "vit_bf16": args.vit_bf16, "ref_meta": ref["meta"], "rows": rows}, open(args.out, "w"), indent=1)
