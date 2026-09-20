"""Gate: the real ImageWAM checkpoint through the `io="native"` model runtime.

The native face's verbs are the C functions of libflashrt_imagewam_native
(no Python, no GIL in a tick). A ctypes consumer drives it and is compared
with `ImageWAMTorchFrontendThor.infer()` on the same observation, prompt
and initial noise (`infer(..., action_noise=)` and the `noise` window get the
same latent):

  1. image_tokens SWAP (the tokens infer() staged) + noise SWAP, proprio
     staged by the frontend -> actions, actions_raw, and the backbone
     residual and K/V caches the graph leaves behind, vs infer()
  2. the same with the native proprio verb (C++ normalization + cuBLASLt
     projection): also the context-row token vs torch F.linear
  3. `--graph native`: the graph the C++ pipeline recorded and captured,
     plus native-graph vs Python-graph replay from identical inputs

Before every tick or replay, every buffer it must write is NaN-filled
(`img_raw`, the proprio row, K/V caches, `Q_O`, backbone residual, action
latent), so nothing can pass on what the reference `infer()` left behind.
Unless `--no-mutants`, the rows are re-run against mutants that must make
them fail: the proprio verb or `step` not called, and (`--graph native`)
native pipelines whose resource table runs no backbone block, skips the
last single-stream block, skips the last denoise step, or feeds one block
another block's weight.

Also an indicative, alternating latency A/B of one `io="python"` tick vs
one `io="native"` tick (both SWAP image tokens) and of the two graphs.

    python tests/gate_imagewam_native_parity.py --precision fp16 [--graph native]   # H100
    python tests/gate_imagewam_native_parity.py --precision nvfp4 --graph native    # Thor

Env: CKPT_PATH (dataset_stats.json beside it), FLUX2_AE_MODEL_PATH (or
AE_MODEL_PATH), FLUX2_SRC, QWEN3_MODEL_SPEC. Needs exec/build,
runtime/build and the `flashrt_imagewam_native` target.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch

from _helpers.imagewam_abi_checks import PIPELINE_MUTATIONS, MutatedPipelineSource, bits, poison_tick_state
from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path

from flash_rt.models.imagewam.libero_dims import (
    LIBERO_HORIZON as HORIZON,
    LIBERO_REAL_DIMS as REAL_DIMS,
)

PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
CHUNK = (HORIZON, 7)


def _compare(label: str, a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    cos = float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    row = {"check": label, "array_equal": bool(np.array_equal(a, b)),
           "max_abs": float(np.max(np.abs(a64 - b64))), "cos": cos}
    print(f"  {label:<56} array_equal={row['array_equal']!s:<5} max_abs={row['max_abs']:.3g} cos={cos:.8f}")
    return row


def _compare_bits(label: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Exact comparison of raw bit patterns (large buffers, NaN-safe)."""
    n_diff = int(np.count_nonzero(a != b))
    row = {"check": label, "array_equal": n_diff == 0, "elements_differing": n_diff}
    print(f"  {label:<56} array_equal={row['array_equal']!s:<5} elements_differing={n_diff}")
    return row


def _percentiles(ms: list[float]) -> str:
    q = np.percentile(ms, [10, 50, 90])
    return f"P10={q[0]:.2f} P50={q[1]:.2f} P90={q[2]:.2f} ms (n={len(ms)})"


def _cudart() -> ctypes.CDLL:
    lib = ctypes.CDLL("libcudart.so")
    lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaGraphGetNodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    return lib


def _python_graph_nodes(fe) -> int:
    """Node count of the Python pipeline recorded the way set_prompt() does
    (a fresh torch graph kept un-instantiated for inspection)."""
    import flash_rt.flash_rt_kernels as fvk
    from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_loop, imagewam_prefill
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=s):
        imagewam_prefill(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s.cuda_stream,
                         attn=fe._attn, mod_txt=fe._mod_txt, mod_img=fe._mod_img, mod_single=fe._mod_single,
                         rope_table=fe._rope_table.data_ptr())
        imagewam_denoise_loop(fe._ctx, fvk, fe._gemm, fe._bufs, fe._weights, fe.dims, stream=s.cuda_stream,
                              attn=fe._attn, action_mods=fe._action_mods, head_mods=fe._head_mods,
                              action_rope_table=fe._action_rope_table.data_ptr(), deltas=fe._deltas)
    count = ctypes.c_size_t(0)
    _cudart().cudaGraphGetNodes(graph.raw_cuda_graph(), None, ctypes.byref(count))
    return int(count.value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--graph", choices=("python", "native"), default="python")
    ap.add_argument("--no-mutants", action="store_true", help="skip the mutants")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ckpt = os.environ["CKPT_PATH"]
    ae_path = os.environ.get("FLUX2_AE_MODEL_PATH") or os.environ["AE_MODEL_PATH"]
    flux2_src = os.environ["FLUX2_SRC"]
    sys.path.insert(0, flux2_src + "/src")
    from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
    from flash_rt.models.imagewam.native_runtime import ImageWAMNativeRuntime

    t0 = time.time()
    fe = ImageWAMTorchFrontendThor(
        precision=args.precision, dims_override=dict(REAL_DIMS), ckpt_path=ckpt,
        ae_model_path=ae_path, flux2_src=flux2_src, qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
        dataset_stats_path=os.path.join(os.path.dirname(ckpt), "dataset_stats.json"))
    fe.set_prompt(PROMPT)
    rng = np.random.default_rng(args.seed)
    frames = [np.ascontiguousarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    state = rng.uniform(-0.5, 0.5, REAL_DIMS["proprio_dim"]).astype(np.float32)
    obs = {"view1": torch.from_numpy(frames[0]), "view2": torch.from_numpy(frames[1]), "proprio": state}
    print(f"frontend ({args.precision}) ready in {time.time() - t0:.1f}s")

    surface = fe.runtime_surface()
    row = surface.proprio_row

    def build_native(source) -> ImageWAMNativeRuntime:
        native = ImageWAMNativeRuntime.create(surface)
        if args.graph == "python":
            native.use_graph(surface.graph_variants.active_key, surface.graph_exec)
        else:
            native.set_pipeline(source)
            native.capture()
        return native

    t0 = time.time()
    native = build_native(fe)
    if args.graph == "native":
        print(f"native pipeline: {len(native.gemm_shapes)} GEMM shapes, {native.gemm_algos_installed} "
              f"algorithms handed off, graph captured in {time.time() - t0:.1f}s, "
              f"{native.graph_nodes} nodes (Python graph: {_python_graph_nodes(fe)} nodes)")
    mr = fe.export_model_runtime(io="native", native=native, identity={"gate": "imagewam_native_parity"})
    consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
    python_face = fe.export_model_runtime(io="python")
    python_consumer = ModelRuntimeConsumer(python_face.ptr, exec_library_path())
    rows, mutants = [], []
    try:
        print(f"io=native runtime: ports={[p.name for p in consumer.ports]} graph={native.graph_producer} "
              f"fingerprint=0x{consumer.fingerprint:016x}")

        torch.manual_seed(args.seed)
        noise_t = torch.empty_like(fe._action_latent).normal_().mul_(0.01)
        noise = noise_t.cpu().numpy()
        ref = fe.infer(obs, action_noise=noise_t)["actions"]
        torch.cuda.synchronize()
        ref_raw = fe._action_latent.detach().cpu().numpy().copy()
        tokens = bits(fe._img_raw)
        ref_token = bits(fe._context[row])
        ref_state = {"backbone_hidden": bits(fe._backbone_hidden), "K_cache": bits(fe._K_cache),
                     "V_cache": bits(fe._V_cache)}

        def tick(c: ModelRuntimeConsumer, *, proprio: str, step: bool = True) -> dict:
            """One NaN-poisoned native tick; proprio staged by the frontend
            ("python"), by the native verb ("native"), or not at all ("skip")."""
            poison_tick_state(fe)
            if proprio == "python":
                fe.stage_proprio(state)
                torch.cuda.synchronize()
            c.write_swap("image_tokens", tokens)
            c.write_swap("noise", noise)
            if proprio == "native":
                c.set_input("proprio", state.tobytes())
            if step:
                c.step()
            out = {"actions": c.get_output("actions", np.float32, CHUNK),
                   "actions_raw": c.read_swap("actions_raw", np.float32, CHUNK)}
            torch.cuda.synchronize()
            out["token"] = bits(fe._context[row])
            out.update({k: bits(getattr(fe, "_" + k)) for k in ref_state})
            return out

        def compare_tick(prefix: str, out: dict, *, token: bool) -> list:
            result = [_compare(f"{prefix}: actions", out["actions"], ref),
                      _compare(f"{prefix}: actions_raw", out["actions_raw"], ref_raw)]
            if token:
                result.append(_compare_bits(f"{prefix}: proprio token (bf16 bits)", out["token"], ref_token))
            result += [_compare_bits(f"{prefix}: {k} (bits)", out[k], v) for k, v in ref_state.items()]
            return result

        def detected(out: dict) -> bool:
            return not (np.array_equal(out["actions"], ref) and np.array_equal(out["token"], ref_token)
                        and all(np.array_equal(out[k], v) for k, v in ref_state.items()))

        print("parity (io=native consumer vs frontend.infer(), buffers NaN-poisoned before every tick):")
        rows += compare_tick("proprio staged by the frontend", tick(consumer, proprio="python"), token=False)
        rows += compare_tick("native proprio verb", tick(consumer, proprio="native"), token=True)

        cudart = _cudart()
        stream = native.stream
        graphs = {"python graph": surface.graph_exec, "native graph": native.graph_exec}
        if args.graph == "native":
            finals = {}
            for label, exec_ in graphs.items():
                poison_tick_state(fe)
                fe._context[row].copy_(torch.from_numpy(ref_token).cuda().view(torch.bfloat16))
                fe._img_raw.copy_(torch.from_numpy(tokens).cuda().view(torch.bfloat16))
                fe._action_latent.copy_(noise_t)
                torch.cuda.synchronize()
                cudart.cudaGraphLaunch(exec_, stream)
                cudart.cudaStreamSynchronize(stream)
                finals[label] = {"action_latent": bits(fe._action_latent), "backbone_hidden": bits(fe._backbone_hidden),
                                 "K_cache": bits(fe._K_cache), "V_cache": bits(fe._V_cache)}
            for k in finals["native graph"]:
                rows.append(_compare_bits(f"native graph vs python graph: {k}", finals["native graph"][k],
                                          finals["python graph"][k]))
        ok = all(r["array_equal"] for r in rows)

        if not args.no_mutants:
            print("mutants (each must make the native-proprio row fail):")
            for label, kwargs in (("proprio verb not called", {"proprio": "skip"}),
                                  ("step not called", {"proprio": "native", "step": False})):
                hit = detected(tick(consumer, **kwargs))
                mutants.append({"mutant": label, "detected": hit})
                print(f"  {label:<56} detected={hit}")
            if args.graph == "native":
                for mutation in PIPELINE_MUTATIONS:
                    native_m = build_native(MutatedPipelineSource(fe, mutation))
                    mr_m = fe.export_model_runtime(io="native", native=native_m)
                    c_m = ModelRuntimeConsumer(mr_m.ptr, exec_library_path())
                    try:
                        hit = detected(tick(c_m, proprio="native"))
                    finally:
                        c_m.close()
                        mr_m.release()
                        native_m.close()
                    mutants.append({"mutant": f"native pipeline: {mutation}", "detected": hit})
                    print(f"  {'native pipeline: ' + mutation:<56} detected={hit}")
            ok = ok and all(m["detected"] for m in mutants)

        if args.bench_iters > 0:
            def bench_tick(c: ModelRuntimeConsumer) -> None:
                c.write_swap("image_tokens", tokens)
                c.set_input("proprio", state.tobytes())
                c.write_swap("noise", noise)
                c.step()
                c.get_output("actions", np.float32, CHUNK)

            for _ in range(3):
                bench_tick(python_consumer)
                bench_tick(consumer)
            ms = {"io=python tick": [], "io=native tick": []}
            for _ in range(args.bench_iters):
                for label, c in (("io=python tick", python_consumer), ("io=native tick", consumer)):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    bench_tick(c)
                    ms[label].append((time.perf_counter() - t) * 1e3)
            print(f"latency, alternating A/B, SWAP image tokens, {native.graph_producer} graph in the native "
                  "face (indicative only):")
            for label, v in ms.items():
                print(f"  {label:<15} {_percentiles(v)}")
            if args.graph == "native":
                ext = torch.cuda.ExternalStream(stream)
                for label, exec_ in graphs.items():
                    cudart.cudaGraphLaunch(exec_, stream)
                replay_ms = {label: [] for label in graphs}
                for _ in range(args.bench_iters):
                    for label, exec_ in graphs.items():
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(ext)
                        cudart.cudaGraphLaunch(exec_, stream)
                        end.record(ext)
                        end.synchronize()
                        replay_ms[label].append(start.elapsed_time(end))
                print("graph replay only, CUDA events, alternating on one stream (indicative only):")
                for label, v in replay_ms.items():
                    print(f"  {label:<15} {_percentiles(v)}")
                ms.update({f"replay {k}": v for k, v in replay_ms.items()})
            rows.append({"latency_ms": ms})

        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"precision": args.precision, "graph": args.graph, "rows": rows, "mutants": mutants},
                          f, indent=1)
        print("PASS" if ok else "FAIL", f"- ImageWAM io=native ({native.graph_producer} graph) vs infer()")
        return 0 if ok else 1
    finally:
        python_consumer.close()
        python_face.release()
        consumer.close()
        mr.release()
        native.close()


if __name__ == "__main__":
    raise SystemExit(main())
