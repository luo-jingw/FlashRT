#!/usr/bin/env python
"""ImageWAM FP4 (NVFP4) full-scale Thor steady-state speed benchmark.

UNTESTED ON REAL HARDWARE. Written by reading flash_rt.executors.fp4_utils
(the documented "Pi0.5 FP4 frontend, Phase 4.3" NVFP4 wrapper) and the real
Thor SM100 NVFP4 W4A16 GEMM build block in CMakeLists.txt, but never run on
a Blackwell part -- this project's own dev machine is Ada (sm_89), which
CMakeLists.txt disables ENABLE_NVFP4/ENABLE_CUTLASS_SM100_NVFP4_W4A16 for
unconditionally (see the "DISABLED (requires Blackwell sm_120a/sm_121a...)"
message it prints). Expect to debug real issues on first run: this script
has never executed successfully anywhere.

Requirements to actually run this on Thor:
  - Built with GPU_ARCH=110 (enables ENABLE_CUTLASS_SM100_NVFP4_W4A16
    automatically per CMakeLists.txt's own GPU_ARCH STREQUAL "110" gate).
  - flash_rt_fp4.so present and flash_rt.flash_rt_fp4.has_nvfp4() == True.

Scope: unlike benchmarks/imagewam_thor_bench.py (isolated 1-layer timing,
written for this dev machine's own 8GB memory ceiling), this script builds
and runs the FULL 25-layer backbone + 25-layer ActionDiT pipeline in one
graph-free timed loop -- Thor's unified memory should make the full
random-weight FP4 footprint (~2GB, see plan.md's "Ada Steady-State Speed"
section for the BF16/FP8/FP4 estimate derivation) a non-issue, unlike on
this dev machine where even full BF16 was judged too risky to attempt.

Precision is explicitly NOT validated here (random weights, no reference
comparison) -- this measures GEMM/kernel latency only, matching the
"先不看精度" scope of every benchmark in this project so far. `alpha=1.0`
and `variant_idx=-1` (fp4_utils.pick_variant's shape-based auto-choice,
not tuned for ImageWAM's own shapes) are both placeholders that affect
correctness/optimality, not whether the GEMM executes -- fine for a speed
number, wrong for anything else.

Attention itself is UNCHANGED from the rest of this plan: the existing
FP16 attention_qkv_fp16 / attention_qkv_fp16_mot_joint kernels (see
plan.md's Phase 2/3/4 -- OPT-002's broadcast-K/V simplification still
applies) run exactly as before. FP4 only replaces the big GEMMs
(q/k/v/proj/mlp projections); fp4_gemm's own output dtype is FP16
already (flash_rt.executors.fp4_utils.fp4_gemm's docstring: "out[M,N]
(fp16)"), so it drops directly into the same Q_O/K_cache/V_cache buffers
the attention kernels already expect -- no BF16<->FP16 cast layer needed
anywhere in this script.

Already includes real quantize + dequantize cost in the timed loop, no
change needed for that (checked when the FP8 sibling script was updated
to do the same): `quant_act_nvfp4`/`quant_weight_nvfp4` call
`quantize_fp4_dynamic_sfa_fp16`, which computes genuine per-16-block
scale factors from the actual (random) tensor data every time it's
called -- there is no fixed/placeholder-scale code path in this kernel
to begin with, unlike FP8's `quantize_fp8_static_fp16` (which the FP8
script used to use before switching to the dynamic
`quantize_fp8_device_fp16`). `fp4_gemm` decodes those block scales
inside the CUTLASS kernel itself and writes fp16 output directly, so
dequantization is likewise already fused into the timed GEMM call, not
a separate step that could be missing.
"""
from __future__ import annotations

import statistics

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec

try:
    # fp4_utils.py itself does `import flash_rt.flash_rt_fp4` at module
    # level (unconditionally) -- on a non-Blackwell build that raises
    # ModuleNotFoundError from INSIDE fp4_utils.py, not from an import of
    # flash_rt_fp4 directly. Both imports must share this one try/except,
    # confirmed by actually triggering this exact failure on this dev
    # machine (Ada sm_89, no flash_rt_fp4.so) while writing this script.
    import flash_rt.flash_rt_fp4 as fvk_fp4
    from flash_rt.executors.fp4_utils import FP4ActScratch, fp4_gemm, quant_act_nvfp4, quant_weight_nvfp4
except ImportError as e:
    raise SystemExit(
        "flash_rt.flash_rt_fp4 (or flash_rt.executors.fp4_utils, which "
        "imports it) is not importable -- this build was not compiled "
        "with NVFP4 support. On Thor, configure cmake with -DGPU_ARCH=110 "
        "(this enables ENABLE_CUTLASS_SM100_NVFP4_W4A16 automatically per "
        "CMakeLists.txt) and rebuild."
    ) from e

if not fvk_fp4.has_nvfp4():
    raise SystemExit(
        "flash_rt_fp4.has_nvfp4() returned False -- NVFP4 kernels were not "
        "built into this .so even though the module imported. Check the "
        "cmake configure log for 'SM100 CUTLASS NVFP4 W4A16 GEMM (Thor): "
        "ENABLED' -- if it says DISABLED, GPU_ARCH was not 110."
    )

DEV = "cuda"
FP16 = torch.float16

# Real confirmed dims (_imagewam_thor_spec.py) -- identical to
# imagewam_thor_bench.py for direct comparison against the FP16 numbers
# already measured on this dev machine.
HIDDEN, HD, NH, MLP_HIDDEN, JOINT_ATTN_DIM = 3072, 128, 24, 9216, 7680
ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH, ACTION_MLP_HIDDEN = 1024, 3072, 4096
NUM_DOUBLE, NUM_SINGLE = 5, 20
MAX_ACTION_HORIZON = 64

X0, A0 = 128, 896
NUM_ACTION = MAX_ACTION_HORIZON
TOTAL = A0 + NUM_ACTION  # 960, kept under softmax_mot_joint_fp16's 1024-column
                          # ceiling -- see imagewam_thor_bench.py's own docstring;
                          # that FP16 attention kernel limitation is unchanged
                          # by weight precision and applies on Thor too.

WARMUP, ITERS = 15, 50

_keepalive = []


def _rand_fp16(*shape):
    t = torch.randn(*shape, dtype=FP16, device=DEV)
    _keepalive.append(t)
    return t


def _zeros_fp16(*shape):
    t = torch.zeros(*shape, dtype=FP16, device=DEV)
    _keepalive.append(t)
    return t


def _qw(n: int, k: int) -> dict:
    """Quantize a random [N, K] fp16 weight to NVFP4, once."""
    w = torch.randn(n, k, dtype=FP16, device=DEV)
    wq = quant_weight_nvfp4(w)
    _keepalive.append(w)  # not read again after quantization, but keep the
                           # temporary from being reused by another alloc
                           # mid-quantize (see Phase 3/4's own lifetime bug)
    return wq


class _Fp4Linear:
    """out[M,N] (fp16) = x[M,K] (fp16, quantized on the fly) @ W[N,K]^T (fp4).

    One instance per (weight shape), reused every call -- the activation
    scratch is allocated once (at whatever M is passed on the FIRST call)
    and reused thereafter, matching this benchmark's own steady-state
    convention (allocate during warmup, measure only steady-state calls).
    """

    def __init__(self, n: int, k: int):
        self.n, self.k = n, k
        self.wq = _qw(n, k)
        self.scratch = None

    def __call__(self, x: torch.Tensor, out: torch.Tensor, m: int, stream: int = 0):
        if self.scratch is None:
            self.scratch = FP4ActScratch(m, self.k, device=DEV)
        quant_act_nvfp4(x, self.scratch, m, stream)
        fp4_gemm(self.scratch, self.wq, out, m, self.n, self.k, stream=stream)


def _time_ms(fn, warmup=WARMUP, iters=ITERS) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.9)], statistics.mean(times)


def _make_backend(num_layers: int):
    spec = make_imagewam_attention_spec(max_prefix_seq=A0, max_total_seq=TOTAL)
    spec.sites["backbone"].num_layers = num_layers
    spec.sites["mot"].num_layers = num_layers
    ctx = fvk.FvkContext()
    K_cache = _zeros_fp16(num_layers, TOTAL, HD)
    V_cache = _zeros_fp16(num_layers, TOTAL, HD)
    Q_O = _zeros_fp16(TOTAL, HIDDEN)
    logits = _zeros_fp16(TOTAL * NH, TOTAL + (TOTAL % 2))
    backend = ImageWAMAttnBackend(
        spec, ctx,
        backbone_slots={
            "Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
        },
        mot_slots={
            "Q_O": Q_O.data_ptr(), "K": K_cache.data_ptr(), "V": V_cache.data_ptr(),
            "logits": logits.data_ptr(), "scale": 1.0 / (HD ** 0.5),
            "layer_stride": K_cache[0].numel() * 2,
        },
    )
    return ctx, backend, Q_O, K_cache, V_cache


def _ptr_offset(base_ptr: int, row_offset: int, row_width: int) -> int:
    return int(base_ptr) + int(row_offset) * int(row_width) * 2


class FullImageWAMFP4:
    """All 25 backbone layers + all 25 ActionDiT layers, FP4-quantized GEMMs.

    Backbone and ActionDiT share ONE per-layer K/V cache spanning the
    whole [prefix|image|action] TOTAL rows -- backbone writes rows
    [0,a0) once (run_prefill), ActionDiT overwrites rows [a0,total)
    every denoise step (run_denoise_step), matching
    flash_rt/models/imagewam/pipeline_thor.py's own real (FP16) design
    (plan.md Phase 3/4)."""

    def __init__(self):
        self.ctx, self.attn, self.Q_O, self.K_cache, self.V_cache = _make_backend(NUM_DOUBLE + NUM_SINGLE)
        self.hidden_buf = _rand_fp16(A0, HIDDEN)
        self.normed = _zeros_fp16(A0, HIDDEN)
        self.txt_mlp = _zeros_fp16(X0, MLP_HIDDEN)
        self.img_mlp = _zeros_fp16(A0 - X0, MLP_HIDDEN)
        self.single_mlp = _zeros_fp16(A0, MLP_HIDDEN)
        self.proj_scratch = _zeros_fp16(A0, HIDDEN)
        self.ones = torch.ones(HIDDEN, dtype=FP16, device=DEV)
        _keepalive.append(self.ones)
        self.context = _rand_fp16(X0, JOINT_ATTN_DIM)

        self.double_layers = []
        for _ in range(NUM_DOUBLE):
            self.double_layers.append(dict(
                txt_in=_Fp4Linear(HIDDEN, JOINT_ATTN_DIM),
                txt_q=_Fp4Linear(HIDDEN, HIDDEN), txt_k=_Fp4Linear(HD, HIDDEN),
                txt_v=_Fp4Linear(HD, HIDDEN), txt_proj=_Fp4Linear(HIDDEN, HIDDEN),
                txt_mlp0=_Fp4Linear(MLP_HIDDEN, HIDDEN), txt_mlp2=_Fp4Linear(HIDDEN, MLP_HIDDEN),
                img_q=_Fp4Linear(HIDDEN, HIDDEN), img_k=_Fp4Linear(HD, HIDDEN),
                img_v=_Fp4Linear(HD, HIDDEN), img_proj=_Fp4Linear(HIDDEN, HIDDEN),
                img_mlp0=_Fp4Linear(MLP_HIDDEN, HIDDEN), img_mlp2=_Fp4Linear(HIDDEN, MLP_HIDDEN),
            ))
        self.single_layers = []
        for _ in range(NUM_SINGLE):
            self.single_layers.append(dict(
                q=_Fp4Linear(HIDDEN, HIDDEN), k=_Fp4Linear(HD, HIDDEN), v=_Fp4Linear(HD, HIDDEN),
                mlp_in=_Fp4Linear(MLP_HIDDEN, HIDDEN),
                attn_out_proj=_Fp4Linear(HIDDEN, HIDDEN), mlp_down=_Fp4Linear(HIDDEN, MLP_HIDDEN),
            ))

        self._init_action_layers()

    def _double_layer(self, li: int, stream: int):
        w = self.double_layers[li]
        x0, a0, img_len = X0, A0, A0 - X0
        combined = self.hidden_buf
        ptrs = self.attn.get_slot_ptrs("backbone", li)
        Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]

        w["txt_in"](self.context, combined[:x0], x0, stream)
        txt_x = combined[:x0]
        fvk.rms_norm_fp16(txt_x.data_ptr(), self.ones.data_ptr(), self.normed[:x0].data_ptr(),
                           x0, HIDDEN, 1e-6, stream)
        txt_normed = self.normed[:x0]
        w["txt_q"](txt_normed, _view_fp16(Q_O, x0, HIDDEN), x0, stream)
        w["txt_k"](txt_normed, _view_fp16(K_cache, x0, HD), x0, stream)
        w["txt_v"](txt_normed, _view_fp16(V_cache, x0, HD), x0, stream)

        img_x = combined[x0:a0]
        img_normed = self.normed[x0:a0]
        fvk.rms_norm_fp16(img_x.data_ptr(), self.ones.data_ptr(), img_normed.data_ptr(),
                           img_len, HIDDEN, 1e-6, stream)
        img_Q_ptr = _ptr_offset(Q_O, x0, HIDDEN)
        img_K_ptr = _ptr_offset(K_cache, x0, HD)
        img_V_ptr = _ptr_offset(V_cache, x0, HD)
        w["img_q"](img_normed, _view_fp16(img_Q_ptr, img_len, HIDDEN), img_len, stream)
        w["img_k"](img_normed, _view_fp16(img_K_ptr, img_len, HD), img_len, stream)
        w["img_v"](img_normed, _view_fp16(img_V_ptr, img_len, HD), img_len, stream)

        self.attn.run("backbone", li, q_seq=a0, stream=stream)

        proj = self.proj_scratch
        w["txt_proj"](_view_fp16(Q_O, x0, HIDDEN), proj[:x0], x0, stream)
        fvk.residual_add_fp16(txt_x.data_ptr(), proj[:x0].data_ptr(), x0 * HIDDEN, stream)
        w["img_proj"](_view_fp16(img_Q_ptr, img_len, HIDDEN), proj[x0:a0], img_len, stream)
        fvk.residual_add_fp16(img_x.data_ptr(), proj[x0:a0].data_ptr(), img_len * HIDDEN, stream)

        fvk.rms_norm_fp16(txt_x.data_ptr(), self.ones.data_ptr(), txt_normed.data_ptr(), x0, HIDDEN, 1e-6, stream)
        w["txt_mlp0"](txt_normed, self.txt_mlp, x0, stream)
        fvk.gelu_inplace_fp16(self.txt_mlp.data_ptr(), x0 * MLP_HIDDEN, stream)
        w["txt_mlp2"](self.txt_mlp, proj[:x0], x0, stream)
        fvk.residual_add_fp16(txt_x.data_ptr(), proj[:x0].data_ptr(), x0 * HIDDEN, stream)

        fvk.rms_norm_fp16(img_x.data_ptr(), self.ones.data_ptr(), img_normed.data_ptr(), img_len, HIDDEN, 1e-6, stream)
        w["img_mlp0"](img_normed, self.img_mlp, img_len, stream)
        fvk.gelu_inplace_fp16(self.img_mlp.data_ptr(), img_len * MLP_HIDDEN, stream)
        w["img_mlp2"](self.img_mlp, proj[x0:a0], img_len, stream)
        fvk.residual_add_fp16(img_x.data_ptr(), proj[x0:a0].data_ptr(), img_len * HIDDEN, stream)

    def _single_layer(self, li: int, site_li: int, stream: int):
        w = self.single_layers[li]
        a0 = A0
        combined = self.hidden_buf
        ptrs = self.attn.get_slot_ptrs("backbone", site_li)
        Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]

        fvk.rms_norm_fp16(combined.data_ptr(), self.ones.data_ptr(), self.normed.data_ptr(),
                           a0, HIDDEN, 1e-6, stream)
        normed = self.normed
        w["q"](normed, _view_fp16(Q_O, a0, HIDDEN), a0, stream)
        w["k"](normed, _view_fp16(K_cache, a0, HD), a0, stream)
        w["v"](normed, _view_fp16(V_cache, a0, HD), a0, stream)
        w["mlp_in"](normed, self.single_mlp, a0, stream)
        fvk.gelu_inplace_fp16(self.single_mlp.data_ptr(), a0 * MLP_HIDDEN, stream)

        self.attn.run("backbone", site_li, q_seq=a0, stream=stream)

        proj = self.proj_scratch
        w["attn_out_proj"](_view_fp16(Q_O, a0, HIDDEN), proj, a0, stream)
        fvk.residual_add_fp16(combined.data_ptr(), proj.data_ptr(), a0 * HIDDEN, stream)
        w["mlp_down"](self.single_mlp, proj, a0, stream)
        fvk.residual_add_fp16(combined.data_ptr(), proj.data_ptr(), a0 * HIDDEN, stream)

    def run_prefill(self, stream: int = 0):
        for li in range(NUM_DOUBLE):
            self._double_layer(li, stream)
        for i in range(NUM_SINGLE):
            self._single_layer(i, NUM_DOUBLE + i, stream)

    # ---- ActionDiT (denoise loop), mirrors pipeline_thor.py's
    # _action_double_layer/_action_single_layer exactly, FP4 GEMMs ----

    def _init_action_layers(self):
        self.action_hidden = _rand_fp16(NUM_ACTION, ACTION_HIDDEN_DIM)
        self.action_normed = _zeros_fp16(NUM_ACTION, ACTION_HIDDEN_DIM)
        self.action_proj_scratch = _zeros_fp16(NUM_ACTION, ACTION_HIDDEN_DIM)
        self.action_mlp_scratch = _zeros_fp16(NUM_ACTION, ACTION_MLP_HIDDEN)
        self.action_ones = torch.ones(ACTION_HIDDEN_DIM, dtype=FP16, device=DEV)
        _keepalive.append(self.action_ones)
        self.action_latent = torch.zeros(NUM_ACTION, ACTION_HIDDEN_DIM, dtype=torch.float32, device=DEV)
        _keepalive.append(self.action_latent)

        self.action_double_layers = []
        for _ in range(NUM_DOUBLE):
            self.action_double_layers.append(dict(
                q=_Fp4Linear(ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
                k=_Fp4Linear(HD, ACTION_HIDDEN_DIM), v=_Fp4Linear(HD, ACTION_HIDDEN_DIM),
                proj=_Fp4Linear(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
                mlp0=_Fp4Linear(ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM),
                mlp2=_Fp4Linear(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
            ))
        self.action_single_layers = []
        for _ in range(NUM_SINGLE):
            self.action_single_layers.append(dict(
                q=_Fp4Linear(ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
                k=_Fp4Linear(HD, ACTION_HIDDEN_DIM), v=_Fp4Linear(HD, ACTION_HIDDEN_DIM),
                mlp_in=_Fp4Linear(ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM),
                attn_out_proj=_Fp4Linear(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
                mlp_down=_Fp4Linear(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
            ))

    def _action_double_layer(self, li: int, site_li: int, stream: int):
        w = self.action_double_layers[li]
        a0, num_action = A0, NUM_ACTION
        action_x = self.action_hidden
        normed = self.action_normed
        ptrs = self.attn.get_slot_ptrs("mot", site_li)
        Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]
        action_Q_ptr = _ptr_offset(Q_O, a0, ACTION_ATTN_WIDTH)
        action_K_ptr = _ptr_offset(K_cache, a0, HD)
        action_V_ptr = _ptr_offset(V_cache, a0, HD)

        fvk.rms_norm_fp16(action_x.data_ptr(), self.action_ones.data_ptr(), normed.data_ptr(),
                           num_action, ACTION_HIDDEN_DIM, 1e-6, stream)
        w["q"](normed, _view_fp16(action_Q_ptr, num_action, ACTION_ATTN_WIDTH), num_action, stream)
        w["k"](normed, _view_fp16(action_K_ptr, num_action, HD), num_action, stream)
        w["v"](normed, _view_fp16(action_V_ptr, num_action, HD), num_action, stream)

        self.attn.run("mot", site_li, q_seq=NUM_ACTION, kv_seq=TOTAL, stream=stream, x0=X0, a0=a0)

        proj = self.action_proj_scratch
        w["proj"](_view_fp16(action_Q_ptr, num_action, ACTION_ATTN_WIDTH), proj, num_action, stream)
        fvk.residual_add_fp16(action_x.data_ptr(), proj.data_ptr(), num_action * ACTION_HIDDEN_DIM, stream)

        fvk.rms_norm_fp16(action_x.data_ptr(), self.action_ones.data_ptr(), normed.data_ptr(),
                           num_action, ACTION_HIDDEN_DIM, 1e-6, stream)
        mlp = self.action_mlp_scratch
        w["mlp0"](normed, mlp, num_action, stream)
        fvk.gelu_inplace_fp16(mlp.data_ptr(), num_action * ACTION_MLP_HIDDEN, stream)
        w["mlp2"](mlp, proj, num_action, stream)
        fvk.residual_add_fp16(action_x.data_ptr(), proj.data_ptr(), num_action * ACTION_HIDDEN_DIM, stream)

    def _action_single_layer(self, li: int, site_li: int, stream: int):
        w = self.action_single_layers[li]
        a0, num_action = A0, NUM_ACTION
        action_x = self.action_hidden
        normed = self.action_normed
        ptrs = self.attn.get_slot_ptrs("mot", site_li)
        Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]
        action_Q_ptr = _ptr_offset(Q_O, a0, ACTION_ATTN_WIDTH)
        action_K_ptr = _ptr_offset(K_cache, a0, HD)
        action_V_ptr = _ptr_offset(V_cache, a0, HD)

        fvk.rms_norm_fp16(action_x.data_ptr(), self.action_ones.data_ptr(), normed.data_ptr(),
                           num_action, ACTION_HIDDEN_DIM, 1e-6, stream)
        w["q"](normed, _view_fp16(action_Q_ptr, num_action, ACTION_ATTN_WIDTH), num_action, stream)
        w["k"](normed, _view_fp16(action_K_ptr, num_action, HD), num_action, stream)
        w["v"](normed, _view_fp16(action_V_ptr, num_action, HD), num_action, stream)
        mlp = self.action_mlp_scratch
        w["mlp_in"](normed, mlp, num_action, stream)
        fvk.gelu_inplace_fp16(mlp.data_ptr(), num_action * ACTION_MLP_HIDDEN, stream)

        self.attn.run("mot", site_li, q_seq=NUM_ACTION, kv_seq=TOTAL, stream=stream, x0=X0, a0=a0)

        proj = self.action_proj_scratch
        w["attn_out_proj"](_view_fp16(action_Q_ptr, num_action, ACTION_ATTN_WIDTH), proj, num_action, stream)
        fvk.residual_add_fp16(action_x.data_ptr(), proj.data_ptr(), num_action * ACTION_HIDDEN_DIM, stream)
        w["mlp_down"](mlp, proj, num_action, stream)
        fvk.residual_add_fp16(action_x.data_ptr(), proj.data_ptr(), num_action * ACTION_HIDDEN_DIM, stream)

    def run_denoise_step(self, dt: float, stream: int = 0):
        n = NUM_ACTION * ACTION_HIDDEN_DIM
        fvk.gpu_cast_fp32_to_fp16(self.action_latent.data_ptr(), self.action_hidden.data_ptr(), n, stream)
        for li in range(NUM_DOUBLE):
            self._action_double_layer(li, li, stream)
        for i in range(NUM_SINGLE):
            self._action_single_layer(i, NUM_DOUBLE + i, stream)
        fvk.gpu_euler_step(self.action_latent.data_ptr(), self.action_hidden.data_ptr(),
                            NUM_ACTION, ACTION_HIDDEN_DIM, dt, 0, stream)

    def run_full(self, num_denoise_steps: int, stream: int = 0):
        self.run_prefill(stream)
        dt = 1.0 / num_denoise_steps
        for _ in range(num_denoise_steps):
            self.run_denoise_step(dt, stream)


def _view_fp16(ptr: int, rows: int, cols: int) -> torch.Tensor:
    """Zero-copy fp16 tensor view over a raw device pointer."""
    interface = {"data": (int(ptr), False), "shape": (rows, cols), "typestr": "<f2", "version": 3}
    owner = type("_Fp16View", (), {"__cuda_array_interface__": interface})()
    return torch.as_tensor(owner, device=DEV)


def main():
    print("NVFP4 available:", fvk_fp4.has_nvfp4())
    print(f"Dims: hidden={HIDDEN} HD={HD} NH={NH} mlp_hidden={MLP_HIDDEN} "
          f"| action_hidden_dim={ACTION_HIDDEN_DIM} action_attn_width={ACTION_ATTN_WIDTH} "
          f"action_mlp_hidden={ACTION_MLP_HIDDEN} "
          f"| x0={X0} a0={A0} num_action={NUM_ACTION} total={TOTAL}")
    print(f"Building full {NUM_DOUBLE + NUM_SINGLE}-layer FP4 backbone + "
          f"{NUM_DOUBLE + NUM_SINGLE}-layer FP4 ActionDiT "
          f"(quantizes every weight once -- may take a while)...")

    model = FullImageWAMFP4()
    torch.cuda.synchronize()
    print("Built. Running steady-state timing "
          f"({WARMUP} warmup + {ITERS} measured iterations per row)...\n")

    num_denoise_steps = 10

    p50, p90, mean = _time_ms(lambda: model.run_prefill(0))
    print(f"backbone_prefill_fp4 (25 layers)    P50={p50:8.3f} ms  P90={p90:8.3f} ms  mean={mean:8.3f} ms")

    p50d, p90d, meand = _time_ms(lambda: model.run_denoise_step(1.0 / num_denoise_steps, 0))
    print(f"one_denoise_step_fp4 (25 layers)    P50={p50d:8.3f} ms  P90={p90d:8.3f} ms  mean={meand:8.3f} ms")

    p50f, p90f, meanf = _time_ms(lambda: model.run_full(num_denoise_steps, 0), warmup=3, iters=10)
    print(f"full (prefill + {num_denoise_steps}-step denoise), single measured run:")
    print(f"                                    P50={p50f:8.3f} ms  P90={p90f:8.3f} ms  mean={meanf:8.3f} ms")
    print(f"  (cross-check: prefill + {num_denoise_steps}xstep from the rows above = "
          f"{p50 + num_denoise_steps * p50d:.3f} ms)")


if __name__ == "__main__":
    main()
