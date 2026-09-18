#!/usr/bin/env python
"""ImageWAM INT8 (SM80 CUTLASS) full-scale speed benchmark.

**Rewritten 2026-09-17** to match the CURRENT real per-layer math in
`flash_rt/models/imagewam/pipeline_thor.py` -- the previous version of
this file predated real AdaLN modulation, real gated residual, and the
real merged SiLU-GLU MLP structure entirely (plain unweighted
`rms_norm_fp16` + `gelu_inplace_fp16` on a single-width MLP buffer +
plain `residual_add_fp16` -- a generic transformer-block skeleton, not
FLUX.2/ImageWAM's real DiT block), used a stale placeholder image-token
shape (768 via a 384x512 guess), and configured `ImageWAMAttnBackend`
without `use_perhead_kv=True, use_real_mot_mask=True` (real per-head
K/V cache width `HIDDEN`, not the old broadcast `HD` width -- a second,
separate staleness from before opportunities.md OPT-002). Also missing:
`txt_in`/`img_in` were re-run every double-stream layer instead of
once before the loop (the same bug OPT-001 found and fixed in
`pipeline_thor.py` itself). All fixed here to match current reality,
including today's own real fused-`linear1` merge for single-stream
blocks (op-fusion audit finding 1, `dims.get("merge_qkv_mlp")` in
`pipeline_thor.py`). Companion to `imagewam_thor_int4_bench.py` -- same
rewrite, same real dims, same GEMM-only caveat below.

Known going in, not a surprise if it reproduces on Ada:
`imagewam_gemm_precision_compare.py` found `cutlass_int8_rowwise_fp16out`
reliably fails at K=9216 (the mlp2/mlp_down shape) in every isolated-
shape reproduction attempted on this dev machine (Ada sm_89) --
confirmed real Thor (SM110) runs this full pipeline (including VAE
encode) to completion cleanly, so this is an Ada-specific limitation,
not a general property of this kernel family (opportunities.md OPT-007).
This rewrite does not change or fix that -- expect `run_prefill()` to
raise at the first `mlp2`/`mlp_down`/`linear1` call that hits K=9216
on THIS machine (`mlp2`/`mlp_down`, K=`MLP_HIDDEN`=9216, and the new
merged single-stream `linear1`, which also has K=`HIDDEN`=3072 -- only
the double-stream `mlp2` calls, K=9216, are expected to hit this).

GEMM-ONLY, NO PER-CALL ACTIVATION QUANTIZATION -- this is the important
caveat that makes this an optimistic/upper-bound number, not a real
deployment estimate (unchanged from before this rewrite): weights are
random int8 bytes (fine -- quantized once offline in any real
deployment too), activations are ALSO random int8 bytes, reused across
every replay -- NOT re-quantized from a real fp16 activation each call.
Correctness is out of scope here, same as everywhere else in this
project's speed work.

**Real VAE encode step**, unchanged from before this rewrite: runs once
per `run_prefill()` call (matching the real cadence of "once per new
observation"). `img_in` is a REAL gap in `pipeline_thor.py` itself
(image tokens enter the backbone already at hidden width there, no
`img_in` weight modeled) -- this benchmark still adds one here for the
same reason as before (a real VAE's raw patch output cannot otherwise
reach `HIDDEN` width), still ignoring the real VAE output values for
its own INT8 GEMM (see the caveat above).
"""
from __future__ import annotations

import statistics

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from _imagewam_vae_stub import build_vae_encoder, pack_latents, VAE_PATCH_TOKEN_DIM

DEV = "cuda"
FP16 = torch.float16

# Real confirmed dims (_imagewam_thor_spec.py).
HIDDEN, HD, NH, MLP_HIDDEN, JOINT_ATTN_DIM = 3072, 128, 24, 9216, 7680
ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH, ACTION_MLP_HIDDEN = 1024, 3072, 4096
NUM_DOUBLE, NUM_SINGLE = 5, 20
MAX_ACTION_HORIZON = 64

# Real confirmed LIBERO dual-camera input (opportunities.md, 2026-09-15/17):
# 224x448 (two 224x224 views concatenated horizontally) -> 8x VAE
# downsample + 2x2 patch merge -> 14x28 grid -- NOT the stale 768-token/
# 384x512 placeholder this file used before this rewrite. Overridden
# HERE, not in the shared `_imagewam_vae_stub.py` (its own historical
# default may still be referenced elsewhere).
VAE_IMG_H, VAE_IMG_W = 224, 448
VAE_NUM_TOKENS = (VAE_IMG_H // 16) * (VAE_IMG_W // 16)  # 14*28 = 392

X0, A0 = 513, 513 + VAE_NUM_TOKENS  # 513 = real text context (512 real tokens + 1
                                     # reserved proprio row); A0 = 905
NUM_ACTION = MAX_ACTION_HORIZON
TOTAL = A0 + NUM_ACTION  # 969, under softmax_mot_joint_fp16's 1024-column ceiling
                          # (unchanged limitation, see the old version of this
                          # docstring / imagewam_thor_bench.py's own).

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


def _wrap_fp16(ptr: int, seq: int, dim: int, row_stride: int | None = None) -> torch.Tensor:
    """Zero-copy fp16 tensor view over a raw pointer -- same technique
    (and same `row_stride` extension) as `pipeline_thor.py`'s own
    `_wrap_fp16`, needed here now that this file does real fused-QKV
    and fused-`linear1` column-slicing instead of separate GEMMs per
    Q/K/V/mlp-gate."""
    stride = int(dim) if row_stride is None else int(row_stride)
    interface = {
        "data": (int(ptr), False), "shape": (int(seq), int(dim)),
        "strides": (stride * 2, 2), "typestr": "<f2", "version": 3,
    }
    owner = type("_Fp16View", (), {"__cuda_array_interface__": interface})()
    return torch.as_tensor(owner, device=DEV)


def _ptr_offset(base_ptr: int, row_offset: int, row_width: int) -> int:
    return int(base_ptr) + int(row_offset) * int(row_width) * 2


def _col_ptr(base_ptr: int, col_offset: int) -> int:
    return int(base_ptr) + int(col_offset) * 2


def _copy_slice(dst_ptr: int, src_ptr: int, seq: int, dim: int, *,
                 dst_row_stride: int | None = None, src_row_stride: int | None = None) -> None:
    """Plain `dst[:] = src[:]` (no elementwise math) -- lands one Q/K/V
    (or mlp gate/up) third/slice of a fused GEMM's output into its own
    real destination buffer. Same convention as `pipeline_thor.py`'s
    own `_copy_slice`."""
    dst = _wrap_fp16(dst_ptr, seq, dim, dst_row_stride)
    src = _wrap_fp16(src_ptr, seq, dim, src_row_stride)
    dst.copy_(src)


class _Int8Linear:
    """out[M,N] (fp16) = x[M,K] (int8, NOT quantized per-call -- see module
    docstring) @ W[N,K] (int8, allocated once at construction). No K-
    alignment requirement (unlike INT4's k%32).
    """
    def __init__(self, n: int, k: int):
        self.n, self.k = n, k
        self.w = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=DEV)
        self.w_scale = torch.ones(n, dtype=torch.float32, device=DEV)
        self.act_cache = {}
        self.act_scale_cache = {}
        _keepalive.append(self.w)
        _keepalive.append(self.w_scale)

    def __call__(self, x_ptr, out_ptr, m: int, stream: int = 0):
        a = self.act_cache.get(m)
        if a is None:
            a = torch.randint(-127, 127, (m, self.k), dtype=torch.int8, device=DEV)
            asc = torch.ones(m, dtype=torch.float32, device=DEV)
            self.act_cache[m] = a
            self.act_scale_cache[m] = asc
            _keepalive.append(a)
            _keepalive.append(asc)
        asc = self.act_scale_cache[m]
        out_p = out_ptr.data_ptr() if isinstance(out_ptr, torch.Tensor) else int(out_ptr)
        rc = fvk.cutlass_int8_rowwise_fp16out(
            a.data_ptr(), self.w.data_ptr(), asc.data_ptr(), self.w_scale.data_ptr(),
            out_p, m, self.n, self.k, stream)
        if rc != 0:
            raise RuntimeError(f"cutlass_int8_rowwise_fp16out failed rc={rc} shape=({m},{self.n},{self.k})")


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
    # Real per-head K/V cache width is HIDDEN (opportunities.md OPT-002),
    # NOT the old broadcast HD width this file used before this rewrite.
    K_cache = _zeros_fp16(num_layers, TOTAL, HIDDEN)
    V_cache = _zeros_fp16(num_layers, TOTAL, HIDDEN)
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
        # Real per-head K/V + real (no-mask) mot attention, matching
        # pipeline_thor.py's own real ImageWAMAttnBackend construction
        # exactly (opportunities.md OPT-002/OPT-003) -- this file used
        # neither flag before this rewrite.
        use_perhead_kv=True, use_real_mot_mask=True,
    )
    return ctx, backend, Q_O, K_cache, V_cache


def _mod(seq: int, dim: int):
    """Random (shift, scale, gate) AdaLN triple in the exact contiguous
    fp16 form `ada_layer_norm_fp16`/`gate_res_fp16` consume directly
    (`pipeline_thor.py`'s own `_fuse_mod_group` real convention) --
    VALUES are random (this is a GEMM-only throughput bench, real
    modulation values don't affect timing), but the shapes/kernel call
    sequence match the real one exactly."""
    shift = _rand_fp16(dim) * 0.02
    scale = _rand_fp16(dim) * 0.02
    gate = _rand_fp16(seq, dim) * 0.02
    return shift, scale, gate


class FullImageWAMInt8:
    """All 25 backbone layers + all 25 ActionDiT layers, INT8-quantized
    GEMMs, real per-layer math (AdaLN modulation, fused QKV, real
    merged SiLU-GLU MLP, gated residual, real fused `linear1` for
    single-stream blocks -- matches `pipeline_thor.py`'s CURRENT real
    structure, see this file's own module docstring for what changed).
    Backbone and ActionDiT share ONE per-layer K/V cache spanning the
    whole [prefix|image|action] TOTAL rows -- backbone writes rows
    [0,a0) once (run_prefill), ActionDiT overwrites rows [a0,total)
    every denoise step (run_denoise_step)."""

    def __init__(self):
        self.ctx, self.attn, self.Q_O, self.K_cache, self.V_cache = _make_backend(NUM_DOUBLE + NUM_SINGLE)
        self.hidden_buf = _rand_fp16(A0, HIDDEN)
        self.modded = _zeros_fp16(A0, HIDDEN)
        self.txt_qkv_merged = _zeros_fp16(X0, 3 * HIDDEN)
        self.img_qkv_merged = _zeros_fp16(A0 - X0, 3 * HIDDEN)
        self.txt_mlp_merged = _zeros_fp16(X0, MLP_HIDDEN * 2)
        self.txt_mlp_gated = _zeros_fp16(X0, MLP_HIDDEN)
        self.img_mlp_merged = _zeros_fp16(A0 - X0, MLP_HIDDEN * 2)
        self.img_mlp_gated = _zeros_fp16(A0 - X0, MLP_HIDDEN)
        self.single_linear1 = _zeros_fp16(A0, 3 * HIDDEN + 2 * MLP_HIDDEN)
        self.single_mlp_gated = _zeros_fp16(A0, MLP_HIDDEN)
        self.proj_scratch = _zeros_fp16(A0, HIDDEN)
        self.proj_scratch2 = _zeros_fp16(A0, HIDDEN)
        self.context = _rand_fp16(X0, JOINT_ATTN_DIM)

        self.vae = build_vae_encoder(DEV, FP16)
        self.vae_input = _rand_fp16(1, 3, VAE_IMG_H, VAE_IMG_W)
        self.img_in = _Int8Linear(HIDDEN, VAE_PATCH_TOKEN_DIM)
        self.txt_in = _Int8Linear(HIDDEN, JOINT_ATTN_DIM)

        self.double_layers = []
        for _ in range(NUM_DOUBLE):
            self.double_layers.append(dict(
                txt_qkv=_Int8Linear(3 * HIDDEN, HIDDEN), txt_proj=_Int8Linear(HIDDEN, HIDDEN),
                txt_mlp0=_Int8Linear(MLP_HIDDEN * 2, HIDDEN), txt_mlp2=_Int8Linear(HIDDEN, MLP_HIDDEN),
                img_qkv=_Int8Linear(3 * HIDDEN, HIDDEN), img_proj=_Int8Linear(HIDDEN, HIDDEN),
                img_mlp0=_Int8Linear(MLP_HIDDEN * 2, HIDDEN), img_mlp2=_Int8Linear(HIDDEN, MLP_HIDDEN),
                mod_txt1=_mod(X0, HIDDEN), mod_txt2=_mod(X0, HIDDEN),
                mod_img1=_mod(A0 - X0, HIDDEN), mod_img2=_mod(A0 - X0, HIDDEN),
            ))
        self.single_layers = []
        for _ in range(NUM_SINGLE):
            self.single_layers.append(dict(
                linear1=_Int8Linear(3 * HIDDEN + 2 * MLP_HIDDEN, HIDDEN),
                attn_out_proj=_Int8Linear(HIDDEN, HIDDEN), mlp_down=_Int8Linear(HIDDEN, MLP_HIDDEN),
                mod=_mod(A0, HIDDEN),
            ))

        self._init_action_layers()

    def _double_layer(self, li: int, stream: int):
        w = self.double_layers[li]
        x0, a0, img_len = X0, A0, A0 - X0
        combined = self.hidden_buf
        modded = self.modded
        ptrs = self.attn.get_slot_ptrs("backbone", li)
        Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]

        txt_shift1, txt_scale1, txt_gate1 = w["mod_txt1"]
        txt_shift2, txt_scale2, txt_gate2 = w["mod_txt2"]
        img_shift1, img_scale1, img_gate1 = w["mod_img1"]
        img_shift2, img_scale2, img_gate2 = w["mod_img2"]

        txt_x_ptr = combined.data_ptr()
        fvk.ada_layer_norm_fp16(txt_x_ptr, txt_scale1.data_ptr(), txt_shift1.data_ptr(),
                                 modded.data_ptr(), x0, HIDDEN, 1e-6, stream)
        qkv = self.txt_qkv_merged
        w["txt_qkv"](modded.data_ptr(), qkv, x0, stream)
        _copy_slice(Q_O, qkv.data_ptr(), x0, HIDDEN, src_row_stride=3 * HIDDEN)
        _copy_slice(K_cache, _col_ptr(qkv.data_ptr(), HIDDEN), x0, HIDDEN, src_row_stride=3 * HIDDEN)
        _copy_slice(V_cache, _col_ptr(qkv.data_ptr(), 2 * HIDDEN), x0, HIDDEN, src_row_stride=3 * HIDDEN)

        img_x_ptr = _ptr_offset(combined.data_ptr(), x0, HIDDEN)
        img_modded_ptr = _ptr_offset(modded.data_ptr(), x0, HIDDEN)
        img_Q_ptr = _ptr_offset(Q_O, x0, HIDDEN)
        img_K_ptr = _ptr_offset(K_cache, x0, HIDDEN)
        img_V_ptr = _ptr_offset(V_cache, x0, HIDDEN)
        fvk.ada_layer_norm_fp16(img_x_ptr, img_scale1.data_ptr(), img_shift1.data_ptr(),
                                 img_modded_ptr, img_len, HIDDEN, 1e-6, stream)
        img_qkv = self.img_qkv_merged
        w["img_qkv"](img_modded_ptr, img_qkv, img_len, stream)
        _copy_slice(img_Q_ptr, img_qkv.data_ptr(), img_len, HIDDEN, src_row_stride=3 * HIDDEN)
        _copy_slice(img_K_ptr, _col_ptr(img_qkv.data_ptr(), HIDDEN), img_len, HIDDEN, src_row_stride=3 * HIDDEN)
        _copy_slice(img_V_ptr, _col_ptr(img_qkv.data_ptr(), 2 * HIDDEN), img_len, HIDDEN, src_row_stride=3 * HIDDEN)

        self.attn.run("backbone", li, q_seq=a0, stream=stream)

        proj = self.proj_scratch
        w["txt_proj"](Q_O, proj.data_ptr(), x0, stream)
        fvk.gate_res_fp16(proj.data_ptr(), txt_gate1.data_ptr(), txt_x_ptr, x0 * HIDDEN, stream)
        img_proj_ptr = _ptr_offset(proj.data_ptr(), x0, HIDDEN)
        w["img_proj"](img_Q_ptr, img_proj_ptr, img_len, stream)
        fvk.gate_res_fp16(img_proj_ptr, img_gate1.data_ptr(), img_x_ptr, img_len * HIDDEN, stream)

        fvk.ada_layer_norm_fp16(txt_x_ptr, txt_scale2.data_ptr(), txt_shift2.data_ptr(),
                                 modded.data_ptr(), x0, HIDDEN, 1e-6, stream)
        w["txt_mlp0"](modded.data_ptr(), self.txt_mlp_merged, x0, stream)
        fvk.silu_glu_merged_fp16(self.txt_mlp_merged.data_ptr(), self.txt_mlp_gated.data_ptr(), x0, MLP_HIDDEN, stream)
        w["txt_mlp2"](self.txt_mlp_gated, proj.data_ptr(), x0, stream)
        fvk.gate_res_fp16(proj.data_ptr(), txt_gate2.data_ptr(), txt_x_ptr, x0 * HIDDEN, stream)

        fvk.ada_layer_norm_fp16(img_x_ptr, img_scale2.data_ptr(), img_shift2.data_ptr(),
                                 img_modded_ptr, img_len, HIDDEN, 1e-6, stream)
        w["img_mlp0"](img_modded_ptr, self.img_mlp_merged, img_len, stream)
        fvk.silu_glu_merged_fp16(self.img_mlp_merged.data_ptr(), self.img_mlp_gated.data_ptr(), img_len, MLP_HIDDEN, stream)
        w["img_mlp2"](self.img_mlp_gated, img_proj_ptr, img_len, stream)
        fvk.gate_res_fp16(img_proj_ptr, img_gate2.data_ptr(), img_x_ptr, img_len * HIDDEN, stream)

    def _single_layer(self, li: int, site_li: int, stream: int):
        w = self.single_layers[li]
        a0 = A0
        combined = self.hidden_buf
        modded = self.modded
        ptrs = self.attn.get_slot_ptrs("backbone", site_li)
        Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]
        shift, scale, gate = w["mod"]

        fvk.ada_layer_norm_fp16(combined.data_ptr(), scale.data_ptr(), shift.data_ptr(),
                                 modded.data_ptr(), a0, HIDDEN, 1e-6, stream)

        # Real fused linear1 (op-fusion audit finding 1): ONE GEMM for
        # qkv+mlp-gate/up, sliced via column views -- mirrors
        # pipeline_thor.py's own `dims.get("merge_qkv_mlp")` branch.
        linear1_width = 3 * HIDDEN + 2 * MLP_HIDDEN
        linear1_out = self.single_linear1
        w["linear1"](modded.data_ptr(), linear1_out, a0, stream)
        _copy_slice(Q_O, linear1_out.data_ptr(), a0, HIDDEN, src_row_stride=linear1_width)
        _copy_slice(K_cache, _col_ptr(linear1_out.data_ptr(), HIDDEN), a0, HIDDEN, src_row_stride=linear1_width)
        _copy_slice(V_cache, _col_ptr(linear1_out.data_ptr(), 2 * HIDDEN), a0, HIDDEN, src_row_stride=linear1_width)
        mlp_gated = self.single_mlp_gated
        fvk.silu_glu_merged_fp16(_col_ptr(linear1_out.data_ptr(), 3 * HIDDEN), mlp_gated.data_ptr(),
                                  a0, MLP_HIDDEN, stream, linear1_width)

        self.attn.run("backbone", site_li, q_seq=a0, stream=stream)

        from_attn = self.proj_scratch
        from_mlp = self.proj_scratch2
        w["attn_out_proj"](Q_O, from_attn.data_ptr(), a0, stream)
        w["mlp_down"](mlp_gated.data_ptr(), from_mlp.data_ptr(), a0, stream)
        from_attn.add_(from_mlp)
        fvk.gate_res_fp16(from_attn.data_ptr(), gate.data_ptr(), combined.data_ptr(), a0 * HIDDEN, stream)

    def run_vae_encode(self, stream: int = 0):
        """Real VAE encode + patchify + img_in projection, once per call --
        matches the real cadence (once per new observation), not once per
        denoise step. See module docstring for the img_in caveat."""
        with torch.no_grad():
            latents = self.vae(self.vae_input)
        tokens = pack_latents(latents).view(VAE_NUM_TOKENS, VAE_PATCH_TOKEN_DIM)
        self.img_in(tokens.data_ptr(), self.hidden_buf[X0:A0], VAE_NUM_TOKENS, stream)

    def run_prefill(self, stream: int = 0):
        # txt_in projected ONCE, before the double-stream loop -- matches
        # the real FLUX.2 model (`img_in`/`run_vae_encode` above already
        # does the image side once). The stale version of this file used
        # to run neither once (img_in didn't exist) and had no txt_in
        # equivalent at all; both are real, once-per-forward projections
        # in the real model.
        self.txt_in(self.context.data_ptr(), self.hidden_buf[:X0], X0, stream)
        self.run_vae_encode(stream)
        for li in range(NUM_DOUBLE):
            self._double_layer(li, stream)
        for i in range(NUM_SINGLE):
            self._single_layer(i, NUM_DOUBLE + i, stream)

    # ---- ActionDiT (denoise loop), mirrors pipeline_thor.py's
    # _action_double_layer/_action_single_layer exactly, INT8-quantized GEMMs ----

    def _init_action_layers(self):
        self.action_hidden = _rand_fp16(NUM_ACTION, ACTION_HIDDEN_DIM)
        self.action_modded = _zeros_fp16(NUM_ACTION, ACTION_HIDDEN_DIM)
        self.action_qkv_merged = _zeros_fp16(NUM_ACTION, 3 * ACTION_ATTN_WIDTH)
        self.action_proj_scratch = _zeros_fp16(NUM_ACTION, ACTION_HIDDEN_DIM)
        self.action_proj_scratch2 = _zeros_fp16(NUM_ACTION, ACTION_HIDDEN_DIM)
        self.action_mlp_merged = _zeros_fp16(NUM_ACTION, ACTION_MLP_HIDDEN * 2)
        self.action_mlp_gated = _zeros_fp16(NUM_ACTION, ACTION_MLP_HIDDEN)
        self.action_linear1 = _zeros_fp16(NUM_ACTION, 3 * ACTION_ATTN_WIDTH + 2 * ACTION_MLP_HIDDEN)
        self.action_latent = torch.zeros(NUM_ACTION, ACTION_HIDDEN_DIM, dtype=torch.float32, device=DEV)
        _keepalive.append(self.action_latent)

        self.action_double_layers = []
        for _ in range(NUM_DOUBLE):
            self.action_double_layers.append(dict(
                qkv=_Int8Linear(3 * ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
                proj=_Int8Linear(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
                mlp0=_Int8Linear(ACTION_MLP_HIDDEN * 2, ACTION_HIDDEN_DIM),
                mlp2=_Int8Linear(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
                mod1=_mod(NUM_ACTION, ACTION_HIDDEN_DIM), mod2=_mod(NUM_ACTION, ACTION_HIDDEN_DIM),
            ))
        self.action_single_layers = []
        for _ in range(NUM_SINGLE):
            self.action_single_layers.append(dict(
                linear1=_Int8Linear(3 * ACTION_ATTN_WIDTH + 2 * ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM),
                attn_out_proj=_Int8Linear(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
                mlp_down=_Int8Linear(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
                mod=_mod(NUM_ACTION, ACTION_HIDDEN_DIM),
            ))

    def _action_double_layer(self, li: int, site_li: int, stream: int):
        w = self.action_double_layers[li]
        a0, num_action = A0, NUM_ACTION
        action_x = self.action_hidden
        modded = self.action_modded
        shift1, scale1, gate1 = w["mod1"]
        shift2, scale2, gate2 = w["mod2"]
        ptrs = self.attn.get_slot_ptrs("mot", site_li)
        Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]
        action_Q_ptr = _ptr_offset(Q_O, a0, ACTION_ATTN_WIDTH)
        action_K_ptr = _ptr_offset(K_cache, a0, ACTION_ATTN_WIDTH)
        action_V_ptr = _ptr_offset(V_cache, a0, ACTION_ATTN_WIDTH)

        fvk.ada_layer_norm_fp16(action_x.data_ptr(), scale1.data_ptr(), shift1.data_ptr(),
                                 modded.data_ptr(), num_action, ACTION_HIDDEN_DIM, 1e-6, stream)
        qkv = self.action_qkv_merged
        w["qkv"](modded.data_ptr(), qkv, num_action, stream)
        _copy_slice(action_Q_ptr, qkv.data_ptr(), num_action, ACTION_ATTN_WIDTH, src_row_stride=3 * ACTION_ATTN_WIDTH)
        _copy_slice(action_K_ptr, _col_ptr(qkv.data_ptr(), ACTION_ATTN_WIDTH), num_action, ACTION_ATTN_WIDTH,
                    src_row_stride=3 * ACTION_ATTN_WIDTH)
        _copy_slice(action_V_ptr, _col_ptr(qkv.data_ptr(), 2 * ACTION_ATTN_WIDTH), num_action, ACTION_ATTN_WIDTH,
                    src_row_stride=3 * ACTION_ATTN_WIDTH)

        self.attn.run("mot", site_li, q_seq=num_action, kv_seq=TOTAL, stream=stream, x0=X0, a0=a0)

        proj = self.action_proj_scratch
        w["proj"](action_Q_ptr, proj.data_ptr(), num_action, stream)
        fvk.gate_res_fp16(proj.data_ptr(), gate1.data_ptr(), action_x.data_ptr(), num_action * ACTION_HIDDEN_DIM, stream)

        fvk.ada_layer_norm_fp16(action_x.data_ptr(), scale2.data_ptr(), shift2.data_ptr(),
                                 modded.data_ptr(), num_action, ACTION_HIDDEN_DIM, 1e-6, stream)
        w["mlp0"](modded.data_ptr(), self.action_mlp_merged, num_action, stream)
        fvk.silu_glu_merged_fp16(self.action_mlp_merged.data_ptr(), self.action_mlp_gated.data_ptr(),
                                  num_action, ACTION_MLP_HIDDEN, stream)
        w["mlp2"](self.action_mlp_gated, proj.data_ptr(), num_action, stream)
        fvk.gate_res_fp16(proj.data_ptr(), gate2.data_ptr(), action_x.data_ptr(), num_action * ACTION_HIDDEN_DIM, stream)

    def _action_single_layer(self, li: int, site_li: int, stream: int):
        w = self.action_single_layers[li]
        a0, num_action = A0, NUM_ACTION
        action_x = self.action_hidden
        modded = self.action_modded
        shift, scale, gate = w["mod"]
        ptrs = self.attn.get_slot_ptrs("mot", site_li)
        Q_O, K_cache, V_cache = ptrs["Q"], ptrs["K"], ptrs["V"]
        action_Q_ptr = _ptr_offset(Q_O, a0, ACTION_ATTN_WIDTH)
        action_K_ptr = _ptr_offset(K_cache, a0, ACTION_ATTN_WIDTH)
        action_V_ptr = _ptr_offset(V_cache, a0, ACTION_ATTN_WIDTH)

        fvk.ada_layer_norm_fp16(action_x.data_ptr(), scale.data_ptr(), shift.data_ptr(),
                                 modded.data_ptr(), num_action, ACTION_HIDDEN_DIM, 1e-6, stream)
        linear1_width = 3 * ACTION_ATTN_WIDTH + 2 * ACTION_MLP_HIDDEN
        linear1_out = self.action_linear1
        w["linear1"](modded.data_ptr(), linear1_out, num_action, stream)
        _copy_slice(action_Q_ptr, linear1_out.data_ptr(), num_action, ACTION_ATTN_WIDTH, src_row_stride=linear1_width)
        _copy_slice(action_K_ptr, _col_ptr(linear1_out.data_ptr(), ACTION_ATTN_WIDTH), num_action, ACTION_ATTN_WIDTH,
                    src_row_stride=linear1_width)
        _copy_slice(action_V_ptr, _col_ptr(linear1_out.data_ptr(), 2 * ACTION_ATTN_WIDTH), num_action, ACTION_ATTN_WIDTH,
                    src_row_stride=linear1_width)
        mlp_gated = self.action_mlp_gated
        fvk.silu_glu_merged_fp16(_col_ptr(linear1_out.data_ptr(), 3 * ACTION_ATTN_WIDTH), mlp_gated.data_ptr(),
                                  num_action, ACTION_MLP_HIDDEN, stream, linear1_width)

        self.attn.run("mot", site_li, q_seq=num_action, kv_seq=TOTAL, stream=stream, x0=X0, a0=a0)

        from_attn = self.action_proj_scratch
        from_mlp = self.action_proj_scratch2
        w["attn_out_proj"](action_Q_ptr, from_attn.data_ptr(), num_action, stream)
        w["mlp_down"](mlp_gated.data_ptr(), from_mlp.data_ptr(), num_action, stream)
        from_attn.add_(from_mlp)
        fvk.gate_res_fp16(from_attn.data_ptr(), gate.data_ptr(), action_x.data_ptr(), num_action * ACTION_HIDDEN_DIM, stream)

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


def main():
    report_jetson_clock_state()
    print(f"Dims: hidden={HIDDEN} HD={HD} NH={NH} mlp_hidden={MLP_HIDDEN} "
          f"| action_hidden_dim={ACTION_HIDDEN_DIM} action_attn_width={ACTION_ATTN_WIDTH} "
          f"action_mlp_hidden={ACTION_MLP_HIDDEN} "
          f"| x0={X0} a0={A0} num_action={NUM_ACTION} total={TOTAL}")
    print(f"Building full {NUM_DOUBLE + NUM_SINGLE}-layer INT8 backbone + "
          f"{NUM_DOUBLE + NUM_SINGLE}-layer INT8 ActionDiT "
          f"(quantizes every weight once -- may take a while)...")
    model = FullImageWAMInt8()
    torch.cuda.synchronize()
    print("Built. Timing...\n")

    p50, p90, mean = _time_ms(lambda: model.run_prefill())
    print(f"prefill (VAE+txt_in+25L backbone): P50={p50:9.3f}  P90={p90:9.3f}  mean={mean:9.3f}")

    dt = 1.0 / 10
    p50s, p90s, means = _time_ms(lambda: model.run_denoise_step(dt))
    print(f"one denoise step (25L ActionDiT):  P50={p50s:9.3f}  P90={p90s:9.3f}  mean={means:9.3f}")

    for n in (1, 10):
        print(f"prefill + {n}-step denoise loop: {p50 + n * p50s:.2f} ms")


if __name__ == "__main__":
    main()
