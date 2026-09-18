"""Gate: the ImageWAM `io="native"` schema, Python vs C++ vs golden.

Builds the real-dims frontend (LIBERO release: img_len 392, HD 128,
64 x 7 actions, proprio 8), the native handle, and the `io="native"`
declaration, then compares three record sets line for line:

  1. the region/port/stage records in the Python-built declaration's
     canonical identity (flash_rt/models/imagewam/runtime_export.py),
  2. the records the C++ verbs implement
     (frt_imagewam_native_schema_records, cpp/models/imagewam/src/native_schema.cpp),
  3. tests/data/imagewam_native_schema.records.

Weights do not enter the schema: without CKPT_PATH the frontend uses
random weights at the real dims. Needs exec/build, runtime/build and
`cmake --build build --target flashrt_imagewam_native`.

    python tests/gate_imagewam_native_schema_parity.py [--precision nvfp4]
"""
from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

GOLDEN = Path(__file__).with_name("data") / "imagewam_native_schema.records"
REAL_DIMS = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=64, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=0.1, num_denoise_steps=10,
    ref_h=14, ref_w=28, proprio_dim=8, shift=5.0, num_train_timesteps=1000,
)


def _diff(label: str, actual: list[str], expected: list[str]) -> bool:
    if actual == expected:
        print(f"  {label:<26} {len(actual)} records, identical")
        return True
    print(f"  {label:<26} MISMATCH")
    print("\n".join(difflib.unified_diff(expected, actual, "golden", label, lineterm="")))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16")
    args = ap.parse_args()

    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
    from flash_rt.models.imagewam.native_runtime import ImageWAMNativeRuntime

    ckpt = os.environ.get("CKPT_PATH")
    fe = ImageWAMTorchFrontendThor(
        precision=args.precision, dims_override=dict(REAL_DIMS), ckpt_path=ckpt,
        dataset_stats_path=os.path.join(os.path.dirname(ckpt), "dataset_stats.json") if ckpt else None)
    fe.set_prompt("pick up the black bowl")
    surface = fe.runtime_surface()
    native = ImageWAMNativeRuntime.create(surface)
    native.use_graph(surface.graph_exec)
    mr = fe.export_model_runtime(io="native", native=native, identity={"gate": "native_schema_parity"})
    try:
        python_records = [line for line in mr.identity.splitlines()
                          if line.startswith(("region:", "port:", "stage:"))]
        native_records = native.schema_records()
        golden = GOLDEN.read_text().splitlines()
        print(f"ImageWAM io=native schema parity ({args.precision}, "
              f"{'real checkpoint' if ckpt else 'random weights'}, real dims):")
        ok = _diff("python declaration", python_records, golden)
        ok = _diff("c++ native verbs", native_records, golden) and ok
        print("PASS" if ok else "FAIL", "- Python and C++ io=native schemas match the golden records")
        return 0 if ok else 1
    finally:
        mr.release()
        native.close()


if __name__ == "__main__":
    sys.exit(main())
