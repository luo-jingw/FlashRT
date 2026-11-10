"""Image prompts on Qwen36VlTorchFrontendRtx: answers verbatim, per-stage timings, image tokens, GPU memory.

Per prompt: one warmup generation (captures graphs), then one timed generation; greedy, NEW_TOKENS tokens, thinking
disabled in the chat template. Stages: preprocess (processor), vit, inject (embedding rows + RoPE rows), prefill and
decode (the language model's CUDA-event windows). Prompts run at each MAX_PIXELS setting (``none`` = checkpoint
default). The language model is built with max_seq MAX_SEQ above the long-context threshold, FP8 KV, and the long
(chunked) prefill route for prompts of ROUTE_MIN_SEQ tokens or more.

usage: python vl_bench.py NVFP4_DIR MTP_DIR DATA_ROOT OUT_JSON [--route-min-seq 512] [--max-pixels none 262144]
"""
import argparse
import json
import os
import sys
from dataclasses import asdict

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("mtp_dir")
ap.add_argument("data_root")
ap.add_argument("out")
ap.add_argument("--max-seq", type=int, default=8192)
ap.add_argument("--max-image-tokens", type=int, default=4096)
ap.add_argument("--route-min-seq", type=int, default=512)
ap.add_argument("--max-pixels", nargs="+", default=["none", "262144"])
ap.add_argument("--new-tokens", type=int, default=128)
args = ap.parse_args()
os.environ.update({
    "FLASHRT_QWEN36_MTP_CKPT_DIR": args.mtp_dir,
    "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096",
    "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
    "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": str(args.route_min_seq),
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo  # noqa: E402
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from _imagewam_libero_frames import _episode_video, _suite_root, read_video_frame  # noqa: E402

root = _suite_root(args.data_root, "libero_10")
agent = Image.fromarray(read_video_frame(_episode_video(root, "observation.images.image", 0), 30))
wrist = Image.fromarray(read_video_frame(_episode_video(root, "observation.images.wrist_image", 0), 30))
logo = Image.open(os.path.join(REPO, "FlashRT.png")).convert("RGB")
chart = Image.open(os.path.join(REPO, "research/topics/wam-inference-acceleration/experiments/actiondit_representation/preview.png")).convert("RGB")


def user(*content) -> list:
    return [{"role": "user", "content": [{"type": "image", "image": c} if isinstance(c, Image.Image)
                                         else {"type": "text", "text": c} for c in content]}]


prompts = [
    ("text_only", user("Explain quantum entanglement in one short paragraph.")),
    ("libero_agent", user(agent, "Describe the scene. Which objects are on the table?")),
    ("libero_wrist", user(wrist, "This is a robot's wrist camera. What is in front of the gripper?")),
    ("libero_two_views", user(agent, wrist, "These are a robot's front camera and wrist camera. "
                                            "What task is the robot most likely doing?")),
    ("logo_text", user(logo, "What text is written in this image?")),
    ("chart", user(chart, "What kind of figure is this, and what does it show?")),
]
fe = Qwen36VlTorchFrontendRtx(args.ckpt, max_seq=args.max_seq, max_image_tokens=args.max_image_tokens)
torch.cuda.synchronize()
print(f"loaded: GPU allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
kw = {"enable_thinking": False}
rows = []
for mp in args.max_pixels:
    geo.set_processor_max_pixels(fe.processor, None if mp == "none" else int(mp))
    for name, msgs in prompts:
        if mp != args.max_pixels[0] and name in ("text_only",):
            continue
        fe.generate(msgs, max_new_tokens=args.new_tokens, chat_template_kwargs=kw)
        torch.cuda.reset_peak_memory_stats()
        r = fe.generate(msgs, max_new_tokens=args.new_tokens, chat_template_kwargs=kw)
        row = {"max_pixels": mp, "prompt": name, **asdict(r), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        rows.append(row)
        t = r.timings_ms
        print(f"[{mp} {name}] S {r.prompt_len} img_tok {r.num_image_tokens} route {r.route} | preprocess "
              f"{t['preprocess']:.1f} vit {t['vit']:.1f} inject {t['inject']:.1f} prefill {t['prefill']:.1f} decode "
              f"{t['decode']:.1f} ms ({len(r.token_ids)} tok) | peak {row['peak_allocated_gib']:.2f} GiB", flush=True)
        print("   answer:", repr(r.text[:500]), flush=True)
json.dump({"ckpt": args.ckpt, "route_min_seq": args.route_min_seq, "max_seq": args.max_seq,
           "gpu": torch.cuda.get_device_name(), "rows": rows}, open(args.out, "w"), indent=1)
