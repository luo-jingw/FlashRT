"""Shrink a served ImageWAM frontend so it fits on one 32 GB GPU next to the 27B planner.

After `set_prompt` the captured graph reads only the quantized copies of the FP8 / NVFP4 linears. The frontend still
keeps two things the served graph does not read:
  - the Qwen3-4B text encoder (needed again only when the instruction changes);
  - the FP16 source weight of every quantized linear (`Fp8Linear` / `Nvfp4LinearSm120` quantize it once at
    construction and store their own copy).

`record_quantized_sources(frontend_module)` (called before `load_imagewam`) wraps the frontend module's `Fp8Linear` and
`Nvfp4LinearSm120` so that each records the FP16 source pointer it is constructed from; `slim_executor(fe, sources)`
then frees exactly those `_keepalive` tensors and the text encoder, and returns what it freed.

Validation (`python exec_slim.py OUT_JSON`): actions of one `infer()` with a fixed observation and fixed action noise
before slimming, after slimming, and after the freed memory has been reallocated and filled with NaN; plus memory
after each stage.
"""
import gc
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class SlimReport:
    text_encoder_freed: bool
    sources_freed: int
    sources_freed_GB: float
    allocated_before_GB: float
    allocated_after_GB: float
    reserved_after_GB: float


def _recording(base: type, sources: set[int]) -> type:
    class Recording(base):
        def __init__(self, weight_ptr: int, *args, **kwargs):
            sources.add(int(weight_ptr))
            super().__init__(weight_ptr, *args, **kwargs)
    Recording.__name__ = base.__name__
    return Recording


def record_quantized_sources(frontend_module) -> set[int]:
    sources: set[int] = set()
    for name in ("Fp8Linear", "Nvfp4LinearSm120"):
        setattr(frontend_module, name, _recording(getattr(frontend_module, name), sources))
    return sources


def slim_executor(fe, sources: set[int]) -> SlimReport:
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    text_freed = fe._qwen3 is not None
    fe._qwen3 = None
    keep, freed, freed_bytes = [], 0, 0
    for t in fe._keepalive:
        if t.data_ptr() in sources:
            freed += 1
            freed_bytes += t.numel() * t.element_size()
        else:
            keep.append(t)
    fe._keepalive = keep
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return SlimReport(text_freed, freed, round(freed_bytes / 1e9, 3), round(before / 1e9, 3),
                      round(torch.cuda.memory_allocated() / 1e9, 3), round(torch.cuda.memory_reserved() / 1e9, 3))


def _validate(out_path: str) -> None:
    sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
    from _imagewam_libero_frames import center_crop_resize

    import flash_rt.frontends.torch.imagewam_thor as F
    from flash_rt.models.imagewam.workload import ImageWAMWorkload

    precision = os.environ.get("EXEC_PRECISION", "fp8")
    workload = ImageWAMWorkload.libero()
    mem = {}
    sources = record_quantized_sources(F)
    fe = F.load_imagewam(os.environ["CKPT_PATH"], workload, precision=precision,
                         ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                         qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                         dataset_stats_path=os.environ["DATASET_STATS_PATH"])
    mem["after_load_GB"] = round(torch.cuda.memory_allocated() / 1e9, 3)
    fe.set_prompt("A video recorded from a robot's point of view executing the following instruction: "
                  "put both the alphabet soup and the tomato sauce in the basket")
    mem["after_set_prompt_GB"] = round(torch.cuda.memory_allocated() / 1e9, 3)
    mem["peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    obs = {"view1": torch.from_numpy(center_crop_resize(frame, 224, 224)),
           "view2": torch.from_numpy(center_crop_resize(frame, 224, 224)),
           "proprio": np.zeros(8, dtype=np.float32)}
    lat = fe._action_latent
    noise = (0.01 * torch.randn(tuple(lat.shape), generator=torch.Generator().manual_seed(0))).to(lat.device, lat.dtype)

    def actions() -> np.ndarray:
        return np.asarray(fe.infer(obs, action_noise=noise)["actions"], dtype=np.float64)

    a0 = actions()
    rep = slim_executor(fe, sources)
    a1 = actions()
    free_b = torch.cuda.mem_get_info()[0]
    filler = torch.full((int(free_b * 0.9) // 2,), float("nan"), dtype=torch.float16, device="cuda")
    a2 = actions()
    del filler
    torch.cuda.empty_cache()
    res = {"precision": precision, "memory": mem, "slim": asdict(rep),
           "actions_equal_after_slim": bool(np.array_equal(a0, a1)),
           "actions_equal_after_nan_fill": bool(np.array_equal(a0, a2)),
           "actions_finite": bool(np.isfinite(a2).all()), "actions_shape": list(a0.shape),
           "filler_GB": round(free_b * 0.9 / 1e9, 2), "gpu": torch.cuda.get_device_name()}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    _validate(sys.argv[1])
