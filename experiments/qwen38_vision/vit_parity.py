"""Qwen3VlVisionRtx on a Qwen3.5-architecture vision tower vs HF Qwen3_5VisionModel.

Both read the BF16 ``model.visual.*`` weights of CKPT_DIR. Inputs come from the checkpoint's image processor. Per
case: cosine, relative L2 and max abs error of the merged features (HF ``pooler_output``, FlashRT ``image_embeds``)
against HF, for Qwen3VlVisionRtx with FP8 GEMMs and with BF16 GEMMs, the minimum per-token cosine, and eager
forward latency (median of 10, CUDA events) of each.

Cases: LIBERO agent and wrist views at native size, the agent view resized to 448x336 and 640x480, FlashRT.png, and
the two LIBERO views as one two-image input (HF runs both in one call with per-image attention windows; FlashRT runs
the tower once per image).

usage: python vit_parity.py CKPT_DIR DATA_ROOT OUT_JSON
"""
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo
from flash_rt.frontends.torch._qwen3_vl_vision_rtx import Qwen3VlVisionRtx

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from _imagewam_libero_frames import _episode_video, _suite_root, read_video_frame  # noqa: E402

CKPT, DATA_ROOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEV = "cuda"


@dataclass
class ErrorStats:
    cos: float
    rel_l2: float
    max_abs: float
    min_token_cos: float


@dataclass
class CaseResult:
    name: str
    image_sizes: list
    grids: list
    tokens: int
    fp8: ErrorStats
    bf16: ErrorStats
    hf_ms: float
    fp8_ms: float
    bf16_ms: float


def errors(x: torch.Tensor, ref: torch.Tensor) -> ErrorStats:
    x, ref = x.float(), ref.float()
    tok_cos = torch.nn.functional.cosine_similarity(x, ref, dim=-1)
    return ErrorStats(cos=float(torch.nn.functional.cosine_similarity(x.flatten(), ref.flatten(), dim=0)),
                      rel_l2=float((x - ref).norm() / ref.norm()), max_abs=float((x - ref).abs().max()),
                      min_token_cos=float(tok_cos.min()))


def median_ms(fn, n: int = 10) -> float:
    fn()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


cfg = json.load(open(os.path.join(CKPT, "config.json")))
vcfg = cfg["vision_config"]
hf_cfg = Qwen3_5VisionConfig(**{k: v for k, v in vcfg.items() if k != "dtype"})
hf_cfg._attn_implementation = "sdpa"
hf = Qwen3_5VisionModel(hf_cfg).to(DEV, torch.bfloat16).eval()
with safe_open(os.path.join(CKPT, "model.safetensors"), framework="pt") as f:
    sd = {k[len("model.visual."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.visual.")}
missing, unexpected = hf.load_state_dict(sd, strict=False)
print(f"HF vision: {len(sd)} tensors loaded, missing {missing}, unexpected {unexpected}", flush=True)

rtx = {"fp8": Qwen3VlVisionRtx(CKPT, device=DEV, config=vcfg, fp8=True),
       "bf16": Qwen3VlVisionRtx(CKPT, device=DEV, config=vcfg, fp8=False)}
merge = int(vcfg["spatial_merge_size"])
grid_side = int(vcfg["num_position_embeddings"] ** 0.5)
vcos_cache, vsin_cache = geo.build_vision_rope_cache(max_hw=2048, head_dim=vcfg["hidden_size"] // vcfg["num_heads"],
                                                     device=DEV)
proc = AutoProcessor.from_pretrained(CKPT)


def flashrt_forward(model: Qwen3VlVisionRtx, pix: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    outs, off = [], 0
    for g in grid:
        g1 = g.view(1, 3)
        n = int(g.prod())
        vcos, vsin = geo.vision_rope_cos_sin_cached(g1, vcos_cache, vsin_cache, spatial_merge_size=merge)
        pos = geo.vision_pos_embeds(g1, model.pos_embed, num_grid_per_side=grid_side, spatial_merge_size=merge,
                                    device=DEV)
        emb, ds = model.forward(pix[off:off + n], pos, vcos, vsin)
        assert ds == [], "deepstack features returned for a tower without deepstack layers"
        outs.append(emb)
        off += n
    return torch.cat(outs, dim=0)


root = _suite_root(DATA_ROOT, "libero_10")
agent = read_video_frame(_episode_video(root, "observation.images.image", 0), 30)
wrist = read_video_frame(_episode_video(root, "observation.images.wrist_image", 0), 30)
agent_img, wrist_img = Image.fromarray(agent), Image.fromarray(wrist)
cases = [
    ("libero_agent_native", [agent_img]),
    ("libero_wrist_native", [wrist_img]),
    ("libero_agent_448x336", [agent_img.resize((448, 336), Image.BICUBIC)]),
    ("libero_agent_640x480", [agent_img.resize((640, 480), Image.BICUBIC)]),
    ("flashrt_logo", [Image.open(os.path.join(REPO, "FlashRT.png")).convert("RGB")]),
    ("two_images_agent_wrist", [agent_img, wrist_img]),
]
results = []
with torch.no_grad():
    for name, imgs in cases:
        inp = proc.image_processor(images=imgs, return_tensors="pt")
        pix = inp["pixel_values"].to(DEV, torch.bfloat16)
        grid = inp["image_grid_thw"].to(DEV)
        ref = hf(pix, grid_thw=grid).pooler_output
        got = {k: flashrt_forward(m, pix, grid) for k, m in rtx.items()}
        assert all(g.shape == ref.shape for g in got.values()), (ref.shape, [g.shape for g in got.values()])
        r = CaseResult(name=name, image_sizes=[list(i.size) for i in imgs], grids=grid.tolist(), tokens=ref.shape[0],
                       fp8=errors(got["fp8"], ref), bf16=errors(got["bf16"], ref),
                       hf_ms=median_ms(lambda: hf(pix, grid_thw=grid)),
                       fp8_ms=median_ms(lambda: flashrt_forward(rtx["fp8"], pix, grid)),
                       bf16_ms=median_ms(lambda: flashrt_forward(rtx["bf16"], pix, grid)))
        results.append(r)
        print(f"{name:24s} grids {r.grids} tokens {r.tokens:5d} | fp8 cos {r.fp8.cos:.6f} rel {r.fp8.rel_l2:.4f} "
              f"min_tok {r.fp8.min_token_cos:.4f} | bf16 cos {r.bf16.cos:.6f} rel {r.bf16.rel_l2:.4f} "
              f"min_tok {r.bf16.min_token_cos:.4f} | ms hf {r.hf_ms:.2f} fp8 {r.fp8_ms:.2f} bf16 {r.bf16_ms:.2f}",
              flush=True)
json.dump({"ckpt": CKPT, "gpu": torch.cuda.get_device_name(), "hf_vision_config": hf_cfg.to_dict(),
           "cases": [asdict(r) for r in results]}, open(OUT, "w"), indent=1, default=str)
