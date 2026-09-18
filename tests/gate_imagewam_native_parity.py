"""Gate: the real ImageWAM checkpoint through the `io="native"` model runtime.

The native face's verbs are the C functions of libflashrt_imagewam_native
(no Python, no GIL in a tick). A ctypes consumer drives it and is compared
with `ImageWAMTorchFrontendThor.infer()` on the same observation, prompt
and initial noise (`infer(..., action_noise=)` and the `noise` window get the
same latent):

  1. image_tokens SWAP (the tokens infer() staged) + noise SWAP with the
     proprio token infer() staged -> actions / actions_raw vs infer()
  2. the native proprio verb (C++ normalization + cuBLASLt projection):
     context-row token vs torch F.linear, and actions vs infer()
  3. `--graph native`: the same checks on the graph the C++ pipeline
     recorded and captured, plus Python-graph vs native-graph replay

and an indicative, alternating latency A/B of one `io="python"` tick vs
one `io="native"` tick (both SWAP image tokens, same graph).

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

from _helpers.model_runtime_consumer import ModelRuntimeConsumer, exec_library_path

HORIZON, STEPS, SHIFT = 64, 10, 5.0
REAL_DIMS = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    num_action=HORIZON, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    dt=1.0 / STEPS, num_denoise_steps=STEPS,
    ref_h=14, ref_w=28, proprio_dim=8, shift=SHIFT, num_train_timesteps=1000,
)
PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
CHUNK = (HORIZON, 7)


def _compare(label: str, a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    cos = float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-30))
    row = {"check": label, "array_equal": bool(np.array_equal(a, b)),
           "max_abs": float(np.max(np.abs(a64 - b64))), "cos": cos}
    print(f"  {label:<52} array_equal={row['array_equal']!s:<5} max_abs={row['max_abs']:.3g} cos={cos:.8f}")
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
    native = ImageWAMNativeRuntime.create(surface)
    if args.graph == "python":
        native.use_graph(surface.graph_exec)
    else:
        t0 = time.time()
        native.set_pipeline(fe)
        native.capture()
        print(f"native pipeline: {len(native.gemm_shapes)} GEMM shapes, {native.gemm_algos_installed} "
              f"algorithms handed off, graph captured in {time.time() - t0:.1f}s, "
              f"{native.graph_nodes} nodes (Python graph: {_python_graph_nodes(fe)} nodes)")
    mr = fe.export_model_runtime(io="native", native=native, identity={"gate": "imagewam_native_parity"})
    consumer = ModelRuntimeConsumer(mr.ptr, exec_library_path())
    python_face = fe.export_model_runtime(io="python")
    python_consumer = ModelRuntimeConsumer(python_face.ptr, exec_library_path())
    rows = []
    try:
        print(f"io=native runtime: ports={[p.name for p in consumer.ports]} graph={native.graph_producer} "
              f"fingerprint=0x{consumer.fingerprint:016x}")

        def draw_noise(seed: int) -> torch.Tensor:
            torch.manual_seed(seed)
            return torch.empty_like(fe._action_latent).normal_().mul_(0.01)

        noise_t = draw_noise(args.seed)
        ref = fe.infer(obs, action_noise=noise_t)["actions"]
        ref_raw = fe._action_latent.detach().cpu().numpy().copy()
        tokens = fe._img_raw.detach().view(torch.int16).cpu().numpy().copy()
        row = surface.proprio_row
        ref_token = fe._context[row].detach().clone()
        noise = noise_t.cpu().numpy()

        print("parity (io=native consumer vs frontend.infer()):")
        consumer.write_swap("image_tokens", tokens)
        consumer.write_swap("noise", noise)
        consumer.step()
        rows.append(_compare("proprio staged by infer(): actions", consumer.get_output("actions", np.float32, CHUNK), ref))
        rows.append(_compare("proprio staged by infer(): actions_raw",
                             consumer.read_swap("actions_raw", np.float32, CHUNK), ref_raw))

        fe._context[row].zero_()
        torch.cuda.synchronize()
        consumer.set_input("proprio", state.tobytes())
        consumer.sync()
        rows.append(_compare("native proprio verb: context-row token (bf16 bits)",
                             fe._context[row].detach().view(torch.int16).cpu().numpy(),
                             ref_token.view(torch.int16).cpu().numpy()))
        consumer.write_swap("noise", noise)
        consumer.step()
        rows.append(_compare("native proprio verb: actions", consumer.get_output("actions", np.float32, CHUNK), ref))

        cudart = _cudart()
        stream = native.stream
        graphs = {"python graph": surface.graph_exec, "native graph": native.graph_exec}
        if args.graph == "native":
            # the two captured graphs replayed from identical inputs on one stream
            finals = {}
            for label, exec_ in graphs.items():
                fe._context[row].copy_(ref_token)
                fe._img_raw.copy_(torch.from_numpy(tokens).cuda().view(torch.bfloat16))
                fe._action_latent.copy_(torch.from_numpy(noise).cuda())
                torch.cuda.synchronize()
                cudart.cudaGraphLaunch(exec_, stream)
                cudart.cudaStreamSynchronize(stream)
                finals[label] = (fe._action_latent.detach().cpu().numpy().copy(),
                                 fe._K_cache.detach().view(torch.int16).cpu().numpy())
            rows.append(_compare("native graph vs python graph: action latent",
                                 finals["native graph"][0], finals["python graph"][0]))
            rows.append(_compare("native graph vs python graph: K cache (fp16 bits)",
                                 finals["native graph"][1], finals["python graph"][1]))
        ok = all(r["array_equal"] for r in rows)

        if args.bench_iters > 0:
            def tick(c: ModelRuntimeConsumer) -> None:
                c.write_swap("image_tokens", tokens)
                c.set_input("proprio", state.tobytes())
                c.write_swap("noise", noise)
                c.step()
                c.get_output("actions", np.float32, CHUNK)

            for _ in range(3):
                tick(python_consumer)
                tick(consumer)
            ms = {"io=python tick": [], "io=native tick": []}
            for _ in range(args.bench_iters):
                for label, c in (("io=python tick", python_consumer), ("io=native tick", consumer)):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    tick(c)
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
                json.dump({"precision": args.precision, "graph": args.graph, "rows": rows}, f, indent=1)
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
