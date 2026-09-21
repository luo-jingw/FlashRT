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

The frontend is built with `use_fa4=False`, like every native-path
consumer: the native face serves the cuBLAS attention chain (rule R6), so
none of this gate's records depend on `FLASHRT_THOR_FA4`.

    python tests/gate_imagewam_native_schema_parity.py [--precision nvfp4]
"""
from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

from flash_rt.models.imagewam.libero_dims import LIBERO_REAL_DIMS as REAL_DIMS

GOLDEN = Path(__file__).with_name("data") / "imagewam_native_schema.records"


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
        precision=args.precision, use_fa4=False, dims_override=dict(REAL_DIMS), ckpt_path=ckpt,
        dataset_stats_path=os.path.join(os.path.dirname(ckpt), "dataset_stats.json") if ckpt else None)
    fe.set_prompt("pick up the black bowl")
    surface = fe.runtime_surface()
    native = ImageWAMNativeRuntime.create(surface)
    native.use_graph(surface.graph_variants.active_key, surface.graph_exec)
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
