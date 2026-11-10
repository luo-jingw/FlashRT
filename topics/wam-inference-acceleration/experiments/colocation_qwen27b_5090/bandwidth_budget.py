"""Weight bytes the 27B planner reads per decode step on the FlashRT NVFP4 path, from the checkpoint headers (CPU only).

Runtime formats (flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py): MLP and full-attention projections NVFP4 as
shipped; linear-attention in_proj_qkv / in_proj_z / out_proj and lm_head quantized BF16 -> NVFP4 at load; in_proj_a/b,
conv1d, norms BF16; the embedding is a row gather (not counted); the MTP head converted to NVFP4 at load. NVFP4 bytes
per weight = 0.5 (e2m1) + 1/16 (one FP8 scale per 16).

Per speculative cycle the planner runs K MTP drafts (MTP layer + lm_head each) and one verify forward of the main
model over K+1 rows (all weights once). Output: bytes per verify forward, per draft, per cycle for K = 3..6.

usage: python -I bandwidth_budget.py PLANNER_DIR MTP_DIR OUT_JSON
"""
import json
import struct
import sys

NVFP4_B = 0.5 + 1 / 16
DT_B = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}


def header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def numel(v: dict) -> int:
    n = 1
    for s in v["shape"]:
        n *= s
    return n


def main(planner_dir: str, mtp_dir: str, out: str) -> None:
    h = header(f"{planner_dir}/model.safetensors")
    groups: dict[str, float] = {}

    def add(g: str, b: float) -> None:
        groups[g] = groups.get(g, 0.0) + b

    for k, v in h.items():
        if k.startswith("model.visual") or "embed_tokens" in k:
            continue
        b = (v["data_offsets"][1] - v["data_offsets"][0])
        if k.startswith("lm_head"):
            add("lm_head (NVFP4 at load)", numel(v) * NVFP4_B)
        elif ".linear_attn." in k and any(p in k for p in ("in_proj_qkv.", "in_proj_z.", "out_proj.")):
            add("linear-attn projections (NVFP4 at load)", numel(v) * NVFP4_B if v["dtype"] == "BF16" else b)
        elif ".linear_attn." in k:
            add("linear-attn small tensors (BF16)", b)
        elif ".mlp." in k:
            add("MLP (NVFP4)", b)
        elif ".self_attn." in k:
            add("full-attn projections (NVFP4)", b)
        else:
            add("norms and other", b)
    verify = sum(groups.values())
    mh = header(f"{mtp_dir}/mtp.safetensors")
    mtp_layer = sum(numel(v) * (NVFP4_B if v["dtype"] == "F8_E4M3" else 0) +
                    (v["data_offsets"][1] - v["data_offsets"][0] if v["dtype"] in ("BF16", "F32") and
                     "scale" not in k else 0) for k, v in mh.items())
    draft = mtp_layer + groups["lm_head (NVFP4 at load)"]
    res = {"verify_forward_GB": verify / 1e9, "groups_GB": {k: v / 1e9 for k, v in groups.items()},
           "mtp_layer_GB": mtp_layer / 1e9, "draft_GB": draft / 1e9,
           "cycle_GB": {K: (verify + K * draft) / 1e9 for K in (3, 4, 5, 6)}}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
