"""The ViT configuration of Qwen36VlTorchFrontendThor (BF16 linears, SDPA attention) against the RTX default (FP8
block-128 linears, FA2), both on this GPU, on the three images of the recorded decompose call of episode 0: per
image, cosine similarity and max abs difference of the merged features, and ms per image (CUDA-synchronized wall).
On SM120 the BF16 linears run w16a16_gemm_sm120; on Thor they run the Qwen3-VL module's cuBLASLt GEMM, so this
checks the configuration, not Thor's kernels. Output: smoke/vit_thor_config_check.log."""
import json
import os
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo
from flash_rt.frontends.torch._qwen3_vl_vision_rtx import Qwen3VlVisionRtx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(ckpt: str) -> None:
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    vc = cfg["vision_config"]
    merge = int(vc["spatial_merge_size"])
    proc = AutoProcessor.from_pretrained(ckpt)
    row = json.loads(open(os.path.join(HERE, "calls/ep0_wrist224.jsonl")).readline())
    images = [Image.open(os.path.join(HERE, p["image"])).convert("RGB") for p in row["user_parts"] if "image" in p]
    towers = {"rtx_fp8_fa2": Qwen3VlVisionRtx(ckpt, device="cuda:0", config=vc),
              "thor_bf16_sdpa": Qwen3VlVisionRtx(ckpt, device="cuda:0", config=vc, fp8=False,
                                                 attention_backend="sdpa")}
    vcos, vsin = geo.build_vision_rope_cache(max_hw=4608 * merge, head_dim=vc["hidden_size"] // vc["num_heads"],
                                             device="cuda:0")
    for i, img in enumerate(images):
        inp = proc.image_processor(images=[img], return_tensors="pt")
        pv = inp["pixel_values"].to("cuda:0", torch.bfloat16)
        grid = inp["image_grid_thw"]
        cos, sin = geo.vision_rope_cos_sin_cached(grid, vcos, vsin, spatial_merge_size=merge)
        out = {}
        for name, t in towers.items():
            pos = geo.vision_pos_embeds(grid, t.pos_embed, num_grid_per_side=int(vc["num_position_embeddings"] ** 0.5),
                                        spatial_merge_size=merge, device="cuda:0")
            t.forward(pv, pos, cos, sin)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            emb, _ = t.forward(pv, pos, cos, sin)
            torch.cuda.synchronize()
            out[name] = (emb.float(), (time.perf_counter() - t0) * 1e3)
        a, b = out["rtx_fp8_fa2"][0], out["thor_bf16_sdpa"][0]
        c = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        print(f"image {i} grid {grid.tolist()[0]} tokens {a.shape[0]}: cos mean {c.mean().item():.5f} "
              f"min {c.min().item():.5f}, max |diff| {(a - b).abs().max().item():.4g} (max |x| {a.abs().max().item():.3g}); "
              f"ms rtx_fp8_fa2 {out['rtx_fp8_fa2'][1]:.1f}, thor_bf16_sdpa {out['thor_bf16_sdpa'][1]:.1f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
