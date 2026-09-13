#!/usr/bin/env python
"""ImageWAM INT4 (SM80 CUTLASS, QuaRot-family) full-scale speed benchmark.

Companion to imagewam_thor_fp16_bench.py / fp8_bench.py / fp4_bench.py --
same dims, same 25+25-layer structure, same methodology. Uses the
SEPARATE, non-Blackwell SM80-family CUTLASS INT4 rowwise GEMM
(csrc/gemm/cutlass_sm80_int4_rowwise.cu, see OPT-007) confirmed to run
on this dev machine's Ada sm_89 after reconfiguring with
-DENABLE_SM80_INT8_CUTLASS=ON -DFLASHRT_ENABLE_CHAMELEON=ON.

GEMM-ONLY, NO PER-CALL ACTIVATION QUANTIZATION -- this is the important
caveat that makes this an optimistic/upper-bound number, not a real
deployment estimate:

- Weights are random already-packed int4 bytes (fine -- weights are
  quantized once offline in any real deployment too, and correctness
  isn't in scope here regardless).
- Activations are ALSO random already-packed int4 bytes, reused across
  every replay -- NOT re-quantized from a real fp16 activation each
  call. A real deployment must quantize the activation fresh every
  layer; that cost is NOT included in this number.
- Found while trying to include it honestly: `fht_int4_quant_fp16`,
  the ONLY real activation quantizer for this specific INT4 scheme
  (QuaRot Hadamard rotation + int4 pack), CRASHES with an illegal
  memory access at ImageWAM's real hidden dims (3072, 9216, 7680 --
  confirmed directly, isolated one shape per process to avoid CUDA
  context corruption from the crash: works cleanly at dim=128/1024/4096
  (all powers of 2), crashes at dim=3072 (not a power of 2). This
  kernel needs a power-of-2 transform size and ImageWAM's real
  dimensions are not powers of 2 -- a genuine blocker for a *correct*
  INT4 pipeline here, not just an inconvenience, tracked in OPT-007.

So: this number answers "how fast are the GEMMs if activation
quantization were free/already done" -- a real useful signal for
whether INT4 is worth pursuing further, but NOT the number a real
INT4 deployment would see, and there is presently no known way to
produce that real number with this specific kernel family at
ImageWAM's actual dimensions.
"""
from __future__ import annotations

import statistics

import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec

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


class _Int4Linear:
    """out[M,N] (fp16) = x[M,K] (int4, NOT quantized per-call -- see module
    docstring) @ W[N,K] (int4, packed once at construction).
    """
    def __init__(self, n: int, k: int):
        assert k % 32 == 0, f"K={k} must be 32-aligned for this SM80 INT4 kernel"
        self.n, self.k = n, k
        self.w_packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=DEV)
        self.w_scale = torch.ones(n, dtype=torch.float32, device=DEV)
        self.act_packed = {}
        self.act_scale = {}
        _keepalive.append(self.w_packed)
        _keepalive.append(self.w_scale)

    def __call__(self, x: torch.Tensor, out: torch.Tensor, m: int, stream: int = 0):
        ap = self.act_packed.get(m)
        if ap is None:
            ap = torch.randint(0, 256, (m, self.k // 2), dtype=torch.uint8, device=DEV)
            asc = torch.ones(m, dtype=torch.float32, device=DEV)
            self.act_packed[m] = ap
            self.act_scale[m] = asc
            _keepalive.append(ap)
            _keepalive.append(asc)
        asc = self.act_scale[m]
        rc = fvk.cutlass_int4_rowwise_fp16out(
            ap.data_ptr(), self.w_packed.data_ptr(), asc.data_ptr(), self.w_scale.data_ptr(),
            out.data_ptr(), m, self.n, self.k, stream)
        if rc != 0:
            raise RuntimeError(f"cutlass_int4_rowwise_fp16out failed rc={rc} shape=({m},{self.n},{self.k})")


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


class FullImageWAMInt4:
    """All 25 backbone layers + all 25 ActionDiT layers, INT4-quantized GEMMs (no per-call activation quantization, see module docstring).

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
                txt_in=_Int4Linear(HIDDEN, JOINT_ATTN_DIM),
                txt_q=_Int4Linear(HIDDEN, HIDDEN), txt_k=_Int4Linear(HD, HIDDEN),
                txt_v=_Int4Linear(HD, HIDDEN), txt_proj=_Int4Linear(HIDDEN, HIDDEN),
                txt_mlp0=_Int4Linear(MLP_HIDDEN, HIDDEN), txt_mlp2=_Int4Linear(HIDDEN, MLP_HIDDEN),
                img_q=_Int4Linear(HIDDEN, HIDDEN), img_k=_Int4Linear(HD, HIDDEN),
                img_v=_Int4Linear(HD, HIDDEN), img_proj=_Int4Linear(HIDDEN, HIDDEN),
                img_mlp0=_Int4Linear(MLP_HIDDEN, HIDDEN), img_mlp2=_Int4Linear(HIDDEN, MLP_HIDDEN),
            ))
        self.single_layers = []
        for _ in range(NUM_SINGLE):
            self.single_layers.append(dict(
                q=_Int4Linear(HIDDEN, HIDDEN), k=_Int4Linear(HD, HIDDEN), v=_Int4Linear(HD, HIDDEN),
                mlp_in=_Int4Linear(MLP_HIDDEN, HIDDEN),
                attn_out_proj=_Int4Linear(HIDDEN, HIDDEN), mlp_down=_Int4Linear(HIDDEN, MLP_HIDDEN),
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
    # _action_double_layer/_action_single_layer exactly, INT4-quantized GEMMs ----

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
                q=_Int4Linear(ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
                k=_Int4Linear(HD, ACTION_HIDDEN_DIM), v=_Int4Linear(HD, ACTION_HIDDEN_DIM),
                proj=_Int4Linear(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
                mlp0=_Int4Linear(ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM),
                mlp2=_Int4Linear(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
            ))
        self.action_single_layers = []
        for _ in range(NUM_SINGLE):
            self.action_single_layers.append(dict(
                q=_Int4Linear(ACTION_ATTN_WIDTH, ACTION_HIDDEN_DIM),
                k=_Int4Linear(HD, ACTION_HIDDEN_DIM), v=_Int4Linear(HD, ACTION_HIDDEN_DIM),
                mlp_in=_Int4Linear(ACTION_MLP_HIDDEN, ACTION_HIDDEN_DIM),
                attn_out_proj=_Int4Linear(ACTION_HIDDEN_DIM, ACTION_ATTN_WIDTH),
                mlp_down=_Int4Linear(ACTION_HIDDEN_DIM, ACTION_MLP_HIDDEN),
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

        self.attn.run("mot", site_li, q_seq=TOTAL, stream=stream, x0=X0, a0=a0)

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

        self.attn.run("mot", site_li, q_seq=TOTAL, stream=stream, x0=X0, a0=a0)

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
    print(f"Building full {NUM_DOUBLE + NUM_SINGLE}-layer INT4 backbone + "
          f"{NUM_DOUBLE + NUM_SINGLE}-layer INT4 ActionDiT "
          f"(quantizes every weight once -- may take a while)...")

    model = FullImageWAMInt4()
    torch.cuda.synchronize()
    print("Built. Running steady-state timing "
          f"({WARMUP} warmup + {ITERS} measured iterations per row)...\n")

    num_denoise_steps = 10

    p50, p90, mean = _time_ms(lambda: model.run_prefill(0))
    print(f"backbone_prefill_int4 (25 layers)    P50={p50:8.3f} ms  P90={p90:8.3f} ms  mean={mean:8.3f} ms")

    p50d, p90d, meand = _time_ms(lambda: model.run_denoise_step(1.0 / num_denoise_steps, 0))
    print(f"one_denoise_step_int4 (25 layers)    P50={p50d:8.3f} ms  P90={p90d:8.3f} ms  mean={meand:8.3f} ms")

    p50f, p90f, meanf = _time_ms(lambda: model.run_full(num_denoise_steps, 0), warmup=3, iters=10)
    print(f"full (prefill + {num_denoise_steps}-step denoise), single measured run:")
    print(f"                                    P50={p50f:8.3f} ms  P90={p90f:8.3f} ms  mean={meanf:8.3f} ms")
    print(f"  (cross-check: prefill + {num_denoise_steps}xstep from the rows above = "
          f"{p50 + num_denoise_steps * p50d:.3f} ms)")


if __name__ == "__main__":
    main()
