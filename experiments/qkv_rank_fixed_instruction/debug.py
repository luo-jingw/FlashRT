import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
import flash_rt.frontends.torch.imagewam_thor as F
import flash_rt.models.imagewam.pipeline_thor as P
from flash_rt.models.imagewam.pipeline_thor import _ptr_offset, _wrap_fp16
from flash_rt.models.imagewam.workload import ImageWAMWorkload
fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision="fp16",
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"], dataset_stats_path=os.environ["DATASET_STATS_PATH"])
d = fe.dims; W, A0, na = d["action_attn_width"], d["a0"], d["num_action"]
print("dims a0", A0, "x0", d["x0"], "total", d["total"], "W", W, type(fe._attn).__name__)
orig = fe._attn.run
seen = []
def run(site, li, *a, **k):
    if site == "mot" and len(seen) < 2:
        torch.cuda.synchronize()
        p = fe._attn.get_slot_ptrs("mot", li)
        row = {}
        for kind in ("Q", "K", "V"):
            for off in (0, d["x0"], A0):
                t = _wrap_fp16(_ptr_offset(p[kind], off, W), na, W).float()
                row[(kind, off)] = (round(float(t.norm()), 3), round(float(t[:, :8].std()), 4))
        seen.append((li, p, a, k, row))
    return orig(site, li, *a, **k)
fe.set_prompt("A video recorded from a robot's point of view executing the following instruction: put both the alphabet soup and the tomato sauce in the basket")
for f in range(2):
    obs = {"view1": torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8), "view2": torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8),
           "proprio": np.random.randn(8).astype(np.float32)}
    fe.stage_inputs(obs, noise=0.01 * torch.randn(na, d["action_dim"], device="cuda"))
    seen.clear(); fe._attn.run = run
    try: fe.run_eager()
    finally: fe._attn.run = orig
    for li, p, a, k, row in seen[:1]:
        print("frame", f, "layer", li, "args", a, k); print({kk: v for kk, v in p.items()}); print(row)
