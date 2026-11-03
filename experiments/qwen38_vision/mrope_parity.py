"""MRoPE position ids and RoPE rows for Qwen3.5-architecture image prompts vs HF.

Position ids: ``_qwen3_vl_geometry.mrope_position_ids`` vs HF ``Qwen3_5Model.get_rope_index`` (exact equality and
the rope delta max(position) + 1 - S). Rows: ``_qwen36_mrope.text_rope_rows`` vs HF ``Qwen3_5TextRotaryEmbedding`` on
the same ids (max abs difference of the BF16 cos and sin), and, for a text-only prompt, bit equality with the 1-D table
``Qwen36TorchFrontendRtx._build_rope_table`` builds on the NVFP4 path.

Prompts (checkpoint chat template, LIBERO frames): text only; one image then text; two images then text; text, image,
text, image, text; one image resized to 640x480.

usage: python mrope_parity.py CKPT_DIR DATA_ROOT OUT_JSON
"""
import json
import os
import sys
import types
from dataclasses import asdict, dataclass

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextRotaryEmbedding

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo
from flash_rt.frontends.torch._qwen36_mrope import text_rope_rows

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from _imagewam_libero_frames import _episode_video, _suite_root, read_video_frame  # noqa: E402

CKPT, DATA_ROOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


@dataclass
class PromptResult:
    name: str
    seq_len: int
    image_tokens: int
    grids: list
    ids_equal: bool
    delta_ours: int
    delta_hf: int
    cos_max_abs: float
    sin_max_abs: float
    equal_to_1d_table: bool | None


cfg_json = json.load(open(os.path.join(CKPT, "config.json")))
hf_cfg = Qwen3_5Config(**{k: v for k, v in cfg_json.items() if k not in ("architectures", "transformers_version")})
tc = cfg_json["text_config"]
rp = tc["rope_parameters"]
rotary_dim = int(tc["head_dim"] * rp["partial_rotary_factor"])
mrope_section = tuple(rp["mrope_section"])
theta = float(rp["rope_theta"])
img_id, vid_id, vs_id = cfg_json["image_token_id"], cfg_json["video_token_id"], cfg_json["vision_start_token_id"]
merge = cfg_json["vision_config"]["spatial_merge_size"]

hf_self = types.SimpleNamespace(config=hf_cfg)
hf_self.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, hf_self)
rotary = Qwen3_5TextRotaryEmbedding(hf_cfg.text_config)


def table_1d(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``Qwen36TorchFrontendRtx._build_rope_table`` on the NVFP4 path, for positions [0, n)."""
    from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx
    ns = types.SimpleNamespace(_quant_format="nvfp4", device="cpu", max_seq=n,
                               _cfg={"head_dim": tc["head_dim"], "partial_rotary_factor": rp["partial_rotary_factor"],
                                     "rope_theta": theta})
    Qwen36TorchFrontendRtx._build_rope_table(ns)
    return ns._rope_cos_table, ns._rope_sin_table


root = _suite_root(DATA_ROOT, "libero_10")
agent = Image.fromarray(read_video_frame(_episode_video(root, "observation.images.image", 0), 30))
wrist = Image.fromarray(read_video_frame(_episode_video(root, "observation.images.wrist_image", 0), 30))
img = lambda im: {"type": "image", "image": im}  # noqa: E731
txt = lambda s: {"type": "text", "text": s}  # noqa: E731
prompts = [
    ("text_only", [txt("Describe the next action of the robot arm in one sentence.")]),
    ("one_image", [img(agent), txt("What is on the table?")]),
    ("two_images", [img(agent), img(wrist), txt("Compare the two camera views.")]),
    ("interleaved", [txt("Agent view:"), img(agent), txt("Wrist view:"), img(wrist), txt("Which object is closest?")]),
    ("one_image_640x480", [img(agent.resize((640, 480), Image.BICUBIC)), txt("What is on the table?")]),
]
proc = AutoProcessor.from_pretrained(CKPT)
results = []
for name, content in prompts:
    inp = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt")
    ids = inp["input_ids"]
    grid = inp.get("image_grid_thw")
    ours = geo.mrope_position_ids(ids[0], grid, None, image_token_id=img_id, video_token_id=vid_id,
                                  vision_start_token_id=vs_id, spatial_merge_size=merge)
    mm = (ids == img_id).int() + 2 * (ids == vid_id).int()
    hf_pos, hf_delta = Qwen3_5Model.get_rope_index(hf_self, ids, mm, image_grid_thw=grid)
    hf_pos = hf_pos[:, 0]
    S = ids.shape[1]
    cos, sin = text_rope_rows(hf_pos, rotary_dim=rotary_dim, rope_theta=theta, mrope_section=mrope_section)
    hcos, hsin = rotary(torch.zeros(1, dtype=torch.bfloat16), hf_pos[:, None, :])
    eq1d = None
    if grid is None:
        c1, s1 = table_1d(S)
        eq1d = bool(torch.equal(cos, c1) and torch.equal(sin, s1))
    r = PromptResult(name=name, seq_len=S, image_tokens=int((ids == img_id).sum()),
                     grids=[] if grid is None else grid.tolist(), ids_equal=bool(torch.equal(ours, hf_pos)),
                     delta_ours=int(ours.max()) + 1 - S, delta_hf=int(hf_delta), cos_max_abs=float(
                         (cos.float() - hcos[0].float()).abs().max()),
                     sin_max_abs=float((sin.float() - hsin[0].float()).abs().max()), equal_to_1d_table=eq1d)
    results.append(r)
    print(f"{name:18s} S {S:5d} img_tok {r.image_tokens:4d} grids {r.grids} | ids equal {r.ids_equal} | delta ours "
          f"{r.delta_ours} hf {r.delta_hf} | cos max|d| {r.cos_max_abs:.3g} sin max|d| {r.sin_max_abs:.3g} | "
          f"= 1-D table {r.equal_to_1d_table}", flush=True)
json.dump({"ckpt": CKPT, "rotary_dim": rotary_dim, "mrope_section": mrope_section, "rope_theta": theta,
           "prompts": [asdict(r) for r in results]}, open(OUT, "w"), indent=1)
