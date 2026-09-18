#!/usr/bin/env python
"""DEPRECATED (2026-09-17): see imagewam_thor_fp16_bench.py's own
deprecation note -- same stale per-layer approximation (no AdaLN
modulation, no gated residual, no real merged SiLU-GLU MLP, stale
768-token image shape), kept only for historical reference. Use
imagewam_thor_graph_bench.py instead.

`pipeline_thor.py` now runs single-stream `linear2` as one GEMM and fuses each gated
residual with the next AdaLN (roadmap items 4 and 3); this script keeps the old split
per-layer path and is not updated for them.

ImageWAM FP8 full-scale steady-state speed benchmark.

Companion to imagewam_thor_fp16_bench.py / imagewam_thor_fp4_bench.py --
same dims, same structure, same methodology, directly comparable.

NOT VERIFIED END-TO-END ON THIS DEV MACHINE, for a specific, confirmed
reason (not "Blackwell-only" like FP4): this venv's cuBLASLt (12.8.04,
CUDA 12.8, Ada sm_89) returns CUBLAS_STATUS_NOT_SUPPORTED from
cublasLtMatmulAlgoGetHeuristic for FP8 (E4M3) matmul at EVERY shape
tried, including a trivial 64x64x64 -- confirmed via two independent
code paths (GemmRunner.fp8_nn_dev_fp16 and the standalone
fp8_gemm_descale_fp16), see benchmarks/imagewam_gemm_precision_compare.py
and plan.md's own recorded finding. Ada Lovelace has real FP8 tensor
core hardware in general; this is a version/environment-specific gap in
THIS venv, not a hardware limitation -- the user's own Thor run already
got real FP8 numbers (61.0ms/34.7ms/407.5ms), so the same
quantize_fp8_device_fp16 + fp8_gemm_descale_fp16 API pair used here
is real and known to work there.

**Scale strategy (per user instruction, no real checkpoint/calibration
available)**: both weight and activation scales are computed live via
`quantize_fp8_device_fp16` -- the same GPU-only "absmax -> compute_scale
-> quantize" kernel `shared_primitives.py`'s own `_measure_scale_gpu`
names as the real (non-calibration-path) dynamic-FP8 primitive used
elsewhere in this codebase (cudaMemsetAsync + 2 kernel launches, no host
sync, CUDA-Graph-safe). This replaces an earlier version of this script
that used a hardcoded placeholder scale (1/448) via
`quantize_fp8_static_fp16` -- that skipped the scale-measurement kernels
entirely, understating real inference cost. Using the dynamic kernel on
random weights/activations naturally produces a scale value that varies
run to run (there being no real calibrated constant to reproduce), and,
more importantly, makes the activation-side amax measurement a real
per-call cost included in the timed loop, matching what an actual
uncalibrated deployment would have to pay every forward pass.

The per-layer orchestration (loop counts, weight-dict-equivalent
structure, pointer offsets, attn.run() calls) is IDENTICAL to
imagewam_thor_fp16_bench.py, which IS fully verified on this machine --
only the `_Fp8Linear` class differs from `_Fp16Linear`. That gives high
confidence in the orchestration; the actual FP8 GEMM call itself is
unverified here for the environment reason above, not because it was
untested by the same methodology used for FP4.

**Now also includes a real VAE encode step**, same addition made to
the INT4 sibling script -- see that script's own docstring for the
full rationale (the input image changes every control-loop iteration
and cannot be precomputed once per episode like a fixed text prompt)
and for the `img_in` caveat (not modeled in `pipeline_thor.py` today).
The VAE itself is plain PyTorch/cuDNN, unaffected by the cuBLASLt
FP8 gap above -- confirmed it still runs to completion here before
the script hits the known GEMM failure downstream.
"""
from __future__ import annotations

import statistics

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from _imagewam_vae_stub import build_vae_encoder, pack_latents, VAE_IMG_H, VAE_IMG_W, VAE_PATCH_TOKEN_DIM, VAE_NUM_TOKENS

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
assert VAE_NUM_TOKENS == A0 - X0, (
    f"VAE stub produces {VAE_NUM_TOKENS} image tokens but this script's own "
    f"A0-X0={A0-X0} image-token span expects that many -- keep them in sync.")

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


class _Fp8Linear:
    """out[M,N] (fp16) = x[M,K] (fp16, quantized to fp8 on the fly) @ W[K,N] (fp8).

    Weight quantized once at construction; activation quantized fresh
    every call -- both via `quantize_fp8_device_fp16` (GPU-only
    absmax -> compute_scale -> quantize, no calibration, no host sync),
    so both scales are real values derived from the actual (random)
    data rather than a fixed placeholder, and the activation-side
    measurement cost is paid every call just like real uncalibrated
    inference would. See module docstring for why this replaced the
    earlier `quantize_fp8_static_fp16` + hardcoded-scale version.
    """
    F8 = torch.float8_e4m3fn

    def __init__(self, n: int, k: int):
        self.n, self.k = n, k
        w = torch.randn(k, n, dtype=FP16, device=DEV)
        self.w_f8 = torch.empty(k, n, dtype=self.F8, device=DEV)
        self.w_scale = torch.zeros(1, dtype=torch.float32, device=DEV)
        fvk.quantize_fp8_device_fp16(w.data_ptr(), self.w_f8.data_ptr(), self.w_scale.data_ptr(), k * n, 0)
        self.act_scale = torch.zeros(1, dtype=torch.float32, device=DEV)
        self.act_f8 = None
        _keepalive.append(w)
        _keepalive.append(self.w_f8)
        _keepalive.append(self.w_scale)
        _keepalive.append(self.act_scale)

    def __call__(self, x: torch.Tensor, out: torch.Tensor, m: int, stream: int = 0):
        if self.act_f8 is None:
            self.act_f8 = torch.empty(m, self.k, dtype=self.F8, device=DEV)
            _keepalive.append(self.act_f8)
        fvk.quantize_fp8_device_fp16(x.data_ptr(), self.act_f8.data_ptr(), self.act_scale.data_ptr(), m * self.k, stream)
        fvk.fp8_gemm_descale_fp16(self.act_f8.data_ptr(), self.w_f8.data_ptr(), out.data_ptr(),
                                   m, self.n, self.k, self.act_scale.data_ptr(), self.w_scale.data_ptr(), stream)


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


class FullImageWAMFP8:
    """All 25 backbone layers + all 25 ActionDiT layers, FP8-quantized GEMMs.

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

        self.vae = build_vae_encoder(DEV, FP16)
        self.vae_input = _rand_fp16(1, 3, VAE_IMG_H, VAE_IMG_W)
        self.img_in = _Fp8Linear(HIDDEN, VAE_PATCH_TOKEN_DIM)

        self.double_layers = []
        for _ in range(NUM_DOUBLE):
            self.double_layers.append(dict(
                txt_in=_Fp8Linear(HIDDEN, JOINT_ATTN_DIM),
                txt_q=_Fp8Linear(HIDDEN, HIDDEN), txt_k=_Fp8Linear(HD, HIDDEN),
                txt_v=_Fp8Linear(HD, HIDDEN), txt_proj=_Fp8Linear(HIDDEN, HIDDEN),
                txt_mlp0=_Fp8Linear(MLP_HIDDEN, HIDDEN), txt_mlp2=_Fp8Linear(HIDDEN, MLP_HIDDEN),
                img_q=_Fp8Linear(HIDDEN, HIDDEN), img_k=_Fp8Linear(HD, HIDDEN),
                img_v=_Fp8Linear(HD, HIDDEN), img_proj=_Fp8Linear(HIDDEN, HIDDEN),
                img_mlp0=_Fp8Linear(MLP_HIDDEN, HIDDEN), img_mlp2=_Fp8Linear(HIDDEN, MLP_HIDDEN),
            ))
        self.single_layers = []
        for _ in range(NUM_SINGLE):
            self.single_layers.append(dict(
                q=_Fp8Linear(HIDDEN, HIDDEN), k=_Fp8Linear(HD, HIDDEN), v=_Fp8Linear(HD, HIDDEN),
                mlp_in=_Fp8Linear(MLP_HIDDEN, HIDDEN),
                attn_out_proj=_Fp8Linear(HIDDEN, HIDDEN), mlp_down=_Fp8Linear(HIDDEN, MLP_HIDDEN),
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

    def run_vae_encode(self, stream: int = 0):
        """Real VAE encode + patchify + img_in projection, once per call --
        matches the real cadence (once per new observation), not once per
        denoise step. See imagewam_thor_int4_bench.py's own docstring for
        the img_in caveat."""
        with torch.no_grad():
            latents = self.vae(self.vae_input)
        tokens = pack_latents(latents).view(VAE_NUM_TOKENS, VAE_PATCH_TOKEN_DIM)
        self.img_in(tokens, self.hidden_buf[X0:A0], VAE_NUM_TOKENS, stream)

    def run_prefill(self, stream: int = 0):
        self.run_vae_encode(stream)
        for li in range(NUM_DOUBLE):
            self._double_layer(li, stream)
        for i in range(NUM_SINGLE):
            self._single_layer(i, NUM_DOUBLE + i, stream)

    # ---- ActionDiT (denoise loop), mirrors pipeline_thor.py's
    # _action_double_layer/_action_single_layer exactly, FP8-quantized GEMMs ----

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
                q=_Fp8Linear(ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
                k=_Fp8Linear(HD, ACTION_HIDDEN_DIM), v=_Fp8Linear(HD, ACTION_HIDDEN_DIM),
                proj=_Fp8Linear(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
                mlp0=_Fp8Linear(ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM),
                mlp2=_Fp8Linear(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
            ))
        self.action_single_layers = []
        for _ in range(NUM_SINGLE):
            self.action_single_layers.append(dict(
                q=_Fp8Linear(ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
                k=_Fp8Linear(HD, ACTION_HIDDEN_DIM), v=_Fp8Linear(HD, ACTION_HIDDEN_DIM),
                mlp_in=_Fp8Linear(ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM),
                attn_out_proj=_Fp8Linear(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
                mlp_down=_Fp8Linear(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
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
    print(f"Dims: hidden={HIDDEN} HD={HD} NH={NH} mlp_hidden={MLP_HIDDEN} "
          f"| action_hidden_dim={ACTION_HIDDEN_DIM} action_attn_width={ACTION_ATTN_WIDTH} "
          f"action_mlp_hidden={ACTION_MLP_HIDDEN} "
          f"| x0={X0} a0={A0} num_action={NUM_ACTION} total={TOTAL}")
    print(f"Building full {NUM_DOUBLE + NUM_SINGLE}-layer FP8 backbone + "
          f"{NUM_DOUBLE + NUM_SINGLE}-layer FP8 ActionDiT "
          f"(quantizes every weight once -- may take a while)...")

    model = FullImageWAMFP8()
    torch.cuda.synchronize()
    print("Built. Running steady-state timing "
          f"({WARMUP} warmup + {ITERS} measured iterations per row)...\n")

    num_denoise_steps = 10

    p50v, p90v, meanv = _time_ms(lambda: model.run_vae_encode(0))
    print(f"vae_encode (standalone, incl. img_in) P50={p50v:8.3f} ms  P90={p90v:8.3f} ms  mean={meanv:8.3f} ms")

    p50, p90, mean = _time_ms(lambda: model.run_prefill(0))
    print(f"backbone_prefill_fp8 (25L + VAE)    P50={p50:8.3f} ms  P90={p90:8.3f} ms  mean={mean:8.3f} ms")

    p50d, p90d, meand = _time_ms(lambda: model.run_denoise_step(1.0 / num_denoise_steps, 0))
    print(f"one_denoise_step_fp8 (25 layers)    P50={p50d:8.3f} ms  P90={p90d:8.3f} ms  mean={meand:8.3f} ms")

    p50f, p90f, meanf = _time_ms(lambda: model.run_full(num_denoise_steps, 0), warmup=3, iters=10)
    print(f"full (prefill + {num_denoise_steps}-step denoise), single measured run:")
    print(f"                                    P50={p50f:8.3f} ms  P90={p90f:8.3f} ms  mean={meanf:8.3f} ms")
    print(f"  (cross-check: prefill + {num_denoise_steps}xstep from the rows above = "
          f"{p50 + num_denoise_steps * p50d:.3f} ms)")


if __name__ == "__main__":
    main()
