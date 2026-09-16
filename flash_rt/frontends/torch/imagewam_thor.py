"""ImageWAM (FLUX.2-4B variant) Thor torch frontend.

**Rewritten 2026-09-14** alongside `pipeline_thor.py`'s own real-math
rewrite (opportunities.md OPT-002) -- weights here are still
random-initialized (see `checkpoint_dir` below), but the MATH they now
feed is real: per-head K/V, real 4-axis RoPE, QK-Norm, real AdaLN
modulation, real SiLU-gated-GLU MLP widths, and the real (no-mask)
attention rule, confirmed against the real trained checkpoint on Thor
(see `benchmarks/imagewam_real_checkpoint_validation.py`). This is now
the confirmed target for real Thor deployment (`PROJECT.md`'s
"Confirmed end goal"), following `_template/frontend.py`'s STEP 1-6
shape -- adapted where the real, working `CosmosEdgeThor`
(`flash_rt/models/cosmos3_edge/pipeline_thor.py`) precedent differs
from the generic template (plain `torch.cuda.Tensor` + `.data_ptr()`
throughout, not the template's `CudaBuffer` ctypes wrapper).

`checkpoint_dir` is accepted for interface parity with every other
FlashRT frontend but still unused: every shape here is random-filled
from `dims`, never loaded from a real checkpoint. Real checkpoint
loading needs the actual `imagewam`/`flux2` Python packages (only
available on Thor -- see `PROJECT.md`'s "Real checkpoint testing
happens ONLY on Thor" note) and would reuse
`benchmarks/imagewam_real_checkpoint_validation.py`'s own
`extract_*_weights` functions (already verified against the real
checkpoint, cosine=0.9999+) rather than re-deriving weight extraction
here -- tracked as still-open in `opportunities.md` OPT-002.

`context_mask` (declared as an input shape in `_imagewam_thor_spec.py`)
is accepted by `set_prompt` but never read by anything --
`pipeline_thor.py`'s double-stream blocks treat every context row as
valid, no padding mask. Not modeled, consistent with everything else
already deferred to real-checkpoint work.

AdaLN modulation and RoPE tables are precomputed ONCE here (backbone's
own conditioning timestep is fixed, ActionDiT's varies per denoise
step but `step` is itself a compile-time constant during graph
capture -- see `pipeline_thor.py`'s own module docstring) and passed
into the captured graph as small, fixed-address read-only buffers --
never recomputed per replay.

Dims default to a small, deliberately-not-real-FLUX.2-4B-size
structural test scale (this machine's own 8GB GPU headroom, per
PROJECT.md) -- pass `dims_override` for Thor-scale testing. `HD=128`
is NOT a free "keep it small" parameter here (unlike the other dims):
real 4-axis RoPE (`axes_dim=(32,32,32,32)`, opportunities.md OPT-002)
sums to a fixed 128, so every default/override dims dict below keeps
`HD=128` and only shrinks `NH`/`hidden`/`mlp_hidden`/sequence lengths.
"""
from __future__ import annotations

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_loop, imagewam_prefill
from flash_rt.models.imagewam.pipeline_real import (
    compute_action_head_modulation,
    compute_action_modulation,
    compute_shared_modulation,
)
from flash_rt.models.imagewam.quant_linear import (
    Bf16OutLinear,
    CutlassFp16Linear,
    CutlassFp16SwiGluMlp,
    Fp8Linear,
    Fp16Linear,
    Nvfp4Linear,
    StaticFp8Linear,
)
from flash_rt.models.imagewam.rope import build_action_rope_table, build_backbone_rope_table

_PRECISIONS = ("fp16", "fp16_cutlass", "fp8", "nvfp4", "fp8_static", "fp8_static_cutlass")
# OPT-004 step 6 (plan.md): the two `StaticFp8Linear` variants need a
# one-time calibration call in set_prompt() before graph capture (see
# _calibrate_fp8 below) -- everything else needs no such step.
_STATIC_FP8_PRECISIONS = ("fp8_static", "fp8_static_cutlass")

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16
F32 = torch.float32

_DEFAULT_DIMS = dict(
    hidden=256, HD=128, NH=2, mlp_hidden=384, joint_attention_dim=64,
    x0=3, a0=8, num_layers_double=2, num_layers_single=3,
    action_hidden_dim=128, action_attn_width=256, action_mlp_hidden=192,
    action_dim=7,  # OPT-001: real LIBERO 7-DoF action width
    num_action=4, total=12,
    action_num_layers_double=2, action_num_layers_single=3,
    dt=0.5, num_denoise_steps=2,
)


class ImageWAMTorchFrontendThor:
    """Thor frontend for ImageWAM's real-math (random-weight) dry run.

    Naming rule: `<Model><Framework>Frontend<Hardware>` per
    `docs/adding_new_model.md` §0 rule 2.
    """

    def __init__(self, checkpoint_dir=None, *, dims_override: dict | None = None,
                 use_fa4: bool = False, precision: str = "fp16",
                 ckpt_path: str | None = None,
                 ae_model_path: str | None = None, flux2_src: str | None = None,
                 qwen3_model_spec: str | None = None,
                 dataset_stats_path: str | None = None,
                 **kwargs):
        del checkpoint_dir, kwargs
        if precision not in _PRECISIONS:
            raise ValueError(f"precision={precision!r} -- must be one of {_PRECISIONS}")
        # Real VAE + text-context wiring plan: independent of ckpt_path
        # (OPT-001) -- one loads real transformer weights, this loads a
        # real image encoder. Loaded here (once), used inside infer().
        if (ae_model_path is None) != (flux2_src is None):
            raise ValueError("ae_model_path and flux2_src must be given together, or not at all")
        self._ae = None
        if ae_model_path is not None:
            from flash_rt.models.imagewam.vae_encoder import load_real_ae
            self._ae = load_real_ae(ae_model_path, flux2_src)
        # Live Qwen3 text encoding (real VAE + text-context wiring
        # plan's own deferred item, closed once real Qwen3-4B weights
        # were downloaded -- see opportunities.md). Independent of
        # ae_model_path/ckpt_path; loaded here once, used in set_prompt().
        self._qwen3 = None
        if qwen3_model_spec is not None:
            from flash_rt.models.imagewam.text_encoder import load_real_text_encoder
            self._qwen3 = load_real_text_encoder(qwen3_model_spec)
        # OPT-004 step 5 (plan.md): every weight-projection GEMM in
        # pipeline_thor.py dispatches through weights[key](...), a
        # callable built here by _rnd_linear -- "nvfp4" requires a
        # Blackwell/Thor build (flash_rt.flash_rt_fp4) and will raise a
        # clear RuntimeError from Nvfp4Linear's own constructor on any
        # other machine (see quant_linear.py's own module docstring),
        # not here.
        self._precision = precision
        self._keepalive = []
        self.dims = dict(_DEFAULT_DIMS)
        if dims_override:
            self.dims.update(dims_override)
        d = self.dims
        if d["action_attn_width"] != d["hidden"]:
            raise ValueError(
                f"action_attn_width ({d['action_attn_width']}) must equal hidden "
                f"({d['hidden']}) -- required for mot_joint attention (both experts' "
                f"Q/K/V must land in the same per-head geometry)")
        if d["HD"] != 128:
            raise ValueError(
                f"HD={d['HD']} -- real 4-axis RoPE (axes_dim=(32,32,32,32)) sums to a "
                f"fixed 128; HD is not a free structural-test parameter")
        if ckpt_path is not None and ("ref_h" not in d or "ref_w" not in d):
            # `ref_h`/`ref_w`: the REAL image RoPE needs the actual 2D
            # patch grid (14x28 for the real confirmed 224x448 input,
            # NOT a flat (img_len, 1) placeholder) -- bug found
            # 2026-09-15 via a real Thor cosine comparison against the
            # official model (opportunities.md OPT-002's correction):
            # this silent fallback previously cost 0.09 of cosine
            # similarity (img=0.91 vs 0.999966 with the real grid)
            # without ever crashing. Checked here, BEFORE loading the
            # (multi-GB) checkpoint below, so a real-checkpoint caller
            # that forgot this fails fast, not after a slow load.
            # Structural/random-weight dry runs (ckpt_path=None) keep
            # the silent flat-grid default below -- no accuracy claim
            # there to silently break.
            raise ValueError(
                "ckpt_path given without ref_h/ref_w in dims_override -- the real "
                "image RoPE grid (14x28 for the real confirmed 224x448 input) must "
                "be passed explicitly for real-checkpoint accuracy")

        self._ctx = fvk.FvkContext()
        self._gemm = fvk.GemmRunner()
        self._autotune_gemm(d)

        # OPT-001 (plan.md): ckpt_path switches every weight/modulation
        # source from random to the real checkpoint's own tensors --
        # precision selection (self._precision) is UNCHANGED and
        # ORTHOGONAL either way, same as OPT-004 step 5's own design.
        real_mod = None
        if ckpt_path is not None:
            from flash_rt.models.imagewam.checkpoint_loader import (
                build_real_modulation_weights, load_real_imagewam_state_dict,
            )
            sd = load_real_imagewam_state_dict(ckpt_path)
            self._weights = self._load_real_weights(d, sd)
            real_mod = build_real_modulation_weights(sd)
            del sd
        else:
            self._weights = self._alloc_random_weights(d)

        # Real closed-loop robot-state conditioning (opportunities.md,
        # found 2026-09-15 scoping real closed-loop testing): a plain
        # biased Linear(proprio_dim -> joint_attention_dim), applied
        # OUTSIDE the captured graph in infer() (same convention as the
        # real VAE/Qwen3 encoders -- see that method's own docstring),
        # NOT through Fp16Linear/quant_linear.py. `dims["proprio_dim"]`
        # opts this in; omitted (None) by default so every existing
        # caller/test (no proprio conditioning) is unaffected.
        self._proprio_dim = d.get("proprio_dim")
        self._proprio_w = None
        self._proprio_b = None
        if self._proprio_dim is not None:
            if ckpt_path is not None:
                from flash_rt.models.imagewam.checkpoint_loader import load_real_proprio_weights
                pe = load_real_proprio_weights(ckpt_path)
                if pe is None:
                    raise ValueError(
                        f"dims['proprio_dim']={self._proprio_dim} given but {ckpt_path} has no "
                        f"top-level 'proprio_encoder' key -- this checkpoint was not trained "
                        f"with proprio conditioning")
                w, b = pe
                if tuple(w.shape) != (d["joint_attention_dim"], self._proprio_dim):
                    raise ValueError(
                        f"real proprio_encoder.weight shape {tuple(w.shape)} != "
                        f"(joint_attention_dim={d['joint_attention_dim']}, proprio_dim={self._proprio_dim})")
                self._proprio_w = self._own(w.to(DEV, dtype=BF16))
                self._proprio_b = self._own(b.to(DEV, dtype=BF16))
            else:
                self._proprio_w = self._own(
                    torch.randn(d["joint_attention_dim"], self._proprio_dim, dtype=BF16, device=DEV) * 0.02)
                self._proprio_b = self._own(torch.zeros(d["joint_attention_dim"], dtype=BF16, device=DEV))

        # Real min/max normalization (dataset_stats.json), same
        # closed-loop scope: `state` normalizes real proprio INTO the
        # model's [-1,1] space (infer()'s own input side); `action`
        # denormalizes the model's flow-matching output back OUT to
        # real units (infer()'s own return value). Omitted by default
        # -- every existing caller keeps getting the model's own raw
        # (still-normalized-space) action_latent, unchanged.
        self._state_norm = None
        self._action_norm = None
        if dataset_stats_path is not None:
            from flash_rt.models.imagewam.dataset_stats import load_real_normalizers
            self._state_norm, self._action_norm = load_real_normalizers(dataset_stats_path, device=DEV)

        self._bufs = self._alloc_buffers(d)
        # `ref_h`/`ref_w`: the REAL image RoPE needs the actual 2D patch
        # grid (14x28 for the real confirmed 224x448 input, NOT a flat
        # (img_len, 1) "392x1" placeholder) -- see the ckpt_path check
        # above for the full account (opportunities.md OPT-002's
        # correction). Defaults to the OLD flat placeholder so the toy/
        # default dims (no real 2D image structure) are unaffected.
        img_len = d["a0"] - d["x0"]
        ref_h = d.get("ref_h", img_len)
        ref_w = d.get("ref_w", 1)
        if ref_h * ref_w != img_len:
            raise ValueError(
                f"ref_h*ref_w ({ref_h}*{ref_w}={ref_h * ref_w}) must equal img_len "
                f"(a0-x0={img_len}) -- every image patch needs exactly one RoPE position")
        self._rope_table = self._own(build_backbone_rope_table(
            d["x0"], ref_h, ref_w, device=DEV))
        self._action_rope_table = self._own(build_action_rope_table(d["num_action"], device=DEV))
        self._mod_txt, self._mod_img, self._mod_single = self._compute_backbone_modulation(
            d, real_mod=real_mod["backbone"] if real_mod else None)
        self._action_mods, self._head_mods, self._deltas = self._compute_action_modulations(
            d, real_mod=real_mod["action"] if real_mod else None)

        num_layers = d["num_layers_double"] + d["num_layers_single"]
        HD, hidden = d["HD"], d["hidden"]
        # num_layers/num_heads/head_dim must match THIS frontend's own
        # dims, not make_imagewam_attention_spec's real-FLUX.2-4B
        # defaults (24/128/25) -- see that function's own docstring
        # for the real bug this fixes (an unparameterized spec here
        # silently caused an out-of-bounds attention read/write at any
        # non-real dims, found via tests/test_imagewam_thor_real_wiring.py).
        spec = make_imagewam_attention_spec(
            max_prefix_seq=d["a0"], max_total_seq=d["total"],
            num_layers=num_layers, num_heads=d["NH"], head_dim=HD)
        # Real per-head K/V: (num_layers, total, hidden) -- NOT the old
        # broadcast-K/V (num_layers, total, HD) shape (opportunities.md
        # OPT-002).
        self._K_cache = self._own(torch.zeros(num_layers, d["total"], hidden, dtype=FP16, device=DEV))
        self._V_cache = self._own(torch.zeros(num_layers, d["total"], hidden, dtype=FP16, device=DEV))
        self._Q_O = self._own(torch.zeros(d["total"], hidden, dtype=FP16, device=DEV))
        self._logits = self._own(
            torch.zeros(d["total"] * d["NH"], d["total"] + (d["total"] % 2), dtype=FP16, device=DEV))
        layer_stride = self._K_cache[0].numel() * 2
        self._attn = ImageWAMAttnBackend(
            spec, self._ctx,
            backbone_slots={
                "Q_O": self._Q_O.data_ptr(), "K": self._K_cache.data_ptr(),
                "V": self._V_cache.data_ptr(), "logits": self._logits.data_ptr(),
                "scale": 1.0 / (HD ** 0.5),
            },
            mot_slots={
                "Q_O": self._Q_O.data_ptr(), "K": self._K_cache.data_ptr(),
                "V": self._V_cache.data_ptr(), "logits": self._logits.data_ptr(),
                "scale": 1.0 / (HD ** 0.5), "layer_stride": layer_stride,
            },
            # OPT-002: real per-head K/V + the real (no-mask) attention
            # rule, both confirmed against the real trained checkpoint
            # (benchmarks/imagewam_real_checkpoint_validation.py) --
            # this is now the default for this frontend, not opt-in.
            use_perhead_kv=True, use_real_mot_mask=True,
            # OPT-005: FA4 for the "backbone" site only ("mot" has no
            # FA4-equivalent mask support, unaffected either way).
            # Default False, NOT True: this frontend also runs on this
            # dev machine's own Ada GPU, which has no FA4 runtime at
            # all -- `ImageWAMAttnBackend`'s own constructor raises if
            # `use_fa4=True` without one, so a default-True here would
            # break every local test/construction. Verified correct
            # AND fast on real Thor hardware for the real per-head
            # convention this class now always uses (cosine=1.000000,
            # 3.75x standalone -- opportunities.md OPT-005's own
            # 2026-09-14 entry); pass `use_fa4=True` explicitly when
            # constructing this frontend on Thor to get the win.
            use_fa4=use_fa4,
        )

        self._graph = None
        self._current_prompt = None
        # Row index inside self._context where the proprio token lives
        # for the CURRENTLY captured prompt -- computed once in
        # set_prompt() (depends on that prompt's own real token count),
        # reused by every infer() call until the next set_prompt().
        self._proprio_row = None

    def _own(self, t: torch.Tensor) -> torch.Tensor:
        """Keep a buffer tensor alive for the frontend's own lifetime.

        A bare `torch.zeros(...).data_ptr()` expression drops the only
        Python reference the instant that expression finishes --
        PyTorch's caching allocator is then free to hand the same
        memory to the next allocation, silently corrupting an
        already-stored pointer. Every buffer here goes through this
        helper for that reason.
        """
        self._keepalive.append(t)
        return t

    def _autotune_gemm(self, d: dict) -> None:
        """Autotune `GemmRunner.fp16_nn` once per distinct (M,N,K) shape
        this frontend's own real math uses (OPT-004 step 4,
        opportunities.md), at construction time -- before any weight/
        buffer allocation even, so it never touches this frontend's
        own real pointers (uses disposable zero-filled scratch of the
        right shape/dtype instead; autotune only times candidate
        cuBLASLt algorithms, it does not need meaningful values).

        `autotune_fp16_nn(x_ptr, w_ptr, out_ptr, m, n, k, num_algos)`
        real-benchmarks up to `num_algos` candidates and writes the
        winner into the SAME per-(op,M,N,K) cache `fp16_nn` reads from
        afterward (confirmed by reading `csrc/gemm/gemm_runner.cu`
        directly -- see `benchmarks/imagewam_thor_fp16_autotuned_bench.py`'s
        own docstring for the full account) -- this can only match or
        beat the default heuristic's own top-1 pick, never regress
        (identical math, only the algorithm choice differs). Every
        layer of the same type shares an identical shape (confirmed
        elsewhere in this codebase, e.g. `imagewam_thor_bench.py`'s own
        docstring: "every layer of the same type has identical shapes
        and therefore identical steady-state cost"), so exactly one
        autotune call per distinct shape below covers every layer of
        that type -- not one call per layer.
        """
        hidden, mlp_hidden, HD = d["hidden"], d["mlp_hidden"], d["HD"]
        joint_attention_dim = d["joint_attention_dim"]
        x0, a0 = d["x0"], d["a0"]
        img_len = a0 - x0
        ahd, aaw, amh, num_action = (
            d["action_hidden_dim"], d["action_attn_width"],
            d["action_mlp_hidden"], d["num_action"])
        action_dim = d["action_dim"]

        # txt_in/img_in are BF16-in/out (`Bf16OutLinear`, OPT-001 "FP16
        # residual overflow") -- autotuned separately via autotune_bf16_nn
        # below, a distinct GemmRunner cache key from fp16_nn's.
        bf16_shapes = {
            (x0, hidden, joint_attention_dim),      # txt_in
            (img_len, hidden, HD),                    # img_in (OPT-001/OPT-008)
        }
        for m, n, k in bf16_shapes:
            x = torch.zeros(m, k, dtype=BF16, device=DEV)
            w = torch.zeros(k, n, dtype=BF16, device=DEV)
            out = torch.zeros(m, n, dtype=BF16, device=DEV)
            self._gemm.autotune_bf16_nn(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, 16)

        shapes = {
            (x0, 3 * hidden, hidden),                 # txt_qkv (OPT-004 step 2, fused)
            (x0, hidden, hidden),                      # txt_proj
            (x0, mlp_hidden * 2, hidden),               # txt_mlp0
            (x0, hidden, mlp_hidden),                   # txt_mlp2
            (img_len, 3 * hidden, hidden),              # img_qkv (fused)
            (img_len, hidden, hidden),                  # img_proj
            (img_len, mlp_hidden * 2, hidden),          # img_mlp0
            (img_len, hidden, mlp_hidden),               # img_mlp2
            (a0, 3 * hidden, hidden),                     # single qkv (fused)
            (a0, hidden, hidden),                          # single attn_out_proj
            (a0, mlp_hidden * 2, hidden),                   # single mlp_in
            (a0, hidden, mlp_hidden),                        # single mlp_down
            (num_action, 3 * aaw, ahd),                       # action qkv (fused)
            (num_action, ahd, aaw),                            # action proj/attn_out_proj
            (num_action, amh * 2, ahd),                         # action mlp0/mlp_in
            (num_action, ahd, amh),                              # action mlp2/mlp_down
            (num_action, ahd, action_dim),                        # action_encoder (OPT-001)
            (num_action, action_dim, ahd),                         # head.linear (OPT-001)
        }
        for m, n, k in shapes:
            x = torch.zeros(m, k, dtype=FP16, device=DEV)
            w = torch.zeros(k, n, dtype=FP16, device=DEV)
            out = torch.zeros(m, n, dtype=FP16, device=DEV)
            self._gemm.autotune_fp16_nn(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, 16)
        torch.cuda.synchronize()

    def _rnd_linear(self, n: int, k: int):
        """Real GEMM (K,N) convention: `n` = output width, `k` = input
        width, stored as (k, n) (a real checkpoint's own (out,in)
        `nn.Linear` weight would need `.t().contiguous()` at load time
        -- see `CosmosEdgeThor`'s own precedent). Returns a linear-op
        OBJECT (`Fp16Linear`/`Fp8Linear`/`Nvfp4Linear`, selected by
        `self._precision`), NOT a raw pointer -- OPT-004 step 5
        (`plan.md`): `pipeline_thor.py`'s own weight-projection call
        sites are `weights[key](x_ptr, out_ptr, m, stream)` uniformly.
        The real FP16 weight is always materialized first (quantized
        classes read it once, at construction, to build their own
        quantized copy) and kept alive via `self._own` regardless of
        which precision ultimately uses it.
        """
        w = self._own(torch.randn(k, n, dtype=FP16, device=DEV) * 0.02)
        return self._wrap_linear(w, n, k)

    def _rnd_bf16out_linear(self, n: int, k: int) -> Bf16OutLinear:
        """`_rnd_linear`'s counterpart for `txt_in.weight`/`img_in.weight`
        specifically -- see `Bf16OutLinear`'s own docstring (opportunities.md
        OPT-001 "FP16 residual overflow"). Applied regardless of
        `self._precision`, unlike `_rnd_linear` -> `_wrap_linear`."""
        w = self._own(torch.randn(k, n, dtype=BF16, device=DEV) * 0.02)
        return Bf16OutLinear(self._gemm, w.data_ptr(), n, k)

    def _rnd_swiglu_mlp(self, n: int, k: int):
        """`_rnd_linear`'s counterpart for the merged MLP gate/up
        projection (`{txt,img}_mlp0.weight`/`mlp0.weight`/`mlp_in.weight`,
        `n=2*mlp_hidden`) -- opportunities.md OPT-013. Only
        `precision=="fp16_cutlass"` gets the fused `CutlassFp16SwiGluMlp`;
        every other precision falls back to the plain `_rnd_linear`
        (one wide GEMM, `pipeline_thor.py`'s own `_mlp_gate_up` helper
        then does the separate `silu_glu_merged_fp16` step as before)."""
        if self._precision != "fp16_cutlass":
            return self._rnd_linear(n, k)
        mlp_hidden = n // 2
        w = self._own(torch.randn(k, n, dtype=FP16, device=DEV) * 0.02)
        return CutlassFp16SwiGluMlp(w.data_ptr(), mlp_hidden, k)

    def _wrap_linear(self, w: torch.Tensor, n: int, k: int):
        """Wrap an already-materialized `(k,n)` fp16 CUDA weight tensor
        in the `self._precision`-selected linear-op object -- the part
        of `_rnd_linear` that's shared with `_load_real_weights` (OPT-001),
        which sources `w` from the real checkpoint instead of
        `torch.randn`. `w` must already be `self._own`'d by the caller."""
        if self._precision == "fp16":
            return Fp16Linear(self._gemm, w.data_ptr(), n, k)
        if self._precision == "fp16_cutlass":
            return CutlassFp16Linear(w.data_ptr(), n, k)
        if self._precision == "fp8":
            return Fp8Linear(w.data_ptr(), n, k)
        if self._precision == "nvfp4":
            return Nvfp4Linear(w.data_ptr(), n, k)
        if self._precision == "fp8_static":
            return StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=False)
        if self._precision == "fp8_static_cutlass":
            return StaticFp8Linear(w.data_ptr(), n, k, use_cutlass=True)
        raise ValueError(f"unknown precision {self._precision!r}")  # pragma: no cover -- validated in __init__

    # OPT-004 step 6 follow-up (2026-09-15, real Thor measurement against
    # the real FLUX.2-dev VAE on real LIBERO-fastwam frames): calibrating
    # img_in's own activation scale against N(0, 0.1) noise (the
    # blanket default below) is not just "a rough approximation" -- it
    # is actively WRONG and measurably harmful. Real VAE-encoded image
    # tokens have mean=-0.02, std=0.97 (holdout, `libero_spatial_no_noops_lerobot`,
    # 224x448 input) -- an order of magnitude wider than this noise
    # placeholder. cosine vs. the FP16 reference: 0.902 (N(0,0.1)
    # calibration) vs. 0.99946 (real-token calibration). This is
    # img_in's OWN entry-point distribution specifically, not a general
    # fix for every downstream weight's own calibration input (see
    # `_calibrate_fp8`'s own docstring for why those remain unvalidated
    # placeholders) -- narrowly scoped to the one slot real ground
    # truth now exists for.
    _REAL_CALIB_STATS = {
        "img_in.weight": (-0.02, 0.97),  # (mean, std), real VAE tokens, see above
    }

    def _calibrate_fp8(self, d: dict) -> None:
        """OPT-004 step 6 (plan.md): freeze every `StaticFp8Linear`
        weight's activation scale ONCE, here, before `_capture_graph()`
        -- a captured CUDA Graph replays identical kernel launches
        forever, so the scale must already be fixed by the time capture
        starts (see `StaticFp8Linear`'s own docstring for the ordering
        contract this calls into). No-op for every other precision.

        Deliberately NOT a full forward pass through
        `imagewam_prefill`/`imagewam_denoise_loop` threaded with a
        "calibration mode" flag (`pipeline_real.py`'s own real
        activations don't exist yet at this random-weight dry-run
        stage anyway, see `plan.md`'s own "deliberately not in scope"
        note) -- each `StaticFp8Linear` is calibrated in isolation
        against a disposable random activation of the SAME shape
        (m, k) its real call site actually uses (`m` from the site
        family -- backbone rows use `d["a0"]`, action_dit rows use
        `d["num_action"]`, matching `_autotune_gemm`'s own per-shape
        convention in this same file). Default scale=0.1 to match this
        project's own established "realistic activation magnitude"
        convention (`test_imagewam_prefill.py`'s own `backbone_hidden`
        scale, not the 0.02 weight-init scale) -- **KNOWN WRONG for
        `img_in.weight` specifically, see `_REAL_CALIB_STATS` above**,
        overridden there with the real measured (mean,std); every OTHER
        slot still uses the unvalidated 0.1-scale placeholder, likely
        similarly wrong but with no real ground truth yet to correct it
        against (would need a real forward pass propagating actual
        intermediate activations, not attempted here -- see
        `opportunities.md`'s own "Real multi-sample calibration" entry).
        """
        if self._precision not in _STATIC_FP8_PRECISIONS:
            return
        a0, num_action = d["a0"], d["num_action"]
        scratch_by_shape: dict[tuple, torch.Tensor] = {}
        for key, lin in self._weights.items():
            if not isinstance(lin, StaticFp8Linear):
                continue
            m = a0 if key[0] == "backbone" else num_action
            slot = key[-1]
            mean, std = self._REAL_CALIB_STATS.get(slot, (0.0, 0.1))
            shape = (m, lin.k, mean, std)
            x = scratch_by_shape.get(shape)
            if x is None:
                x = torch.randn(m, lin.k, dtype=FP16, device=DEV) * std + mean
                scratch_by_shape[shape] = x
            lin.calibrate(x.data_ptr(), m, 0)
        torch.cuda.synchronize()

    def _rnd_norm_scale(self, HD: int) -> int:
        """QK-Norm scale, real-checkpoint-typical positive bias (avoids
        near-zero/huge random values that would make a random-weight
        dry run's own NaN/Inf check meaningless for unrelated reasons)."""
        return self._own((torch.randn(HD, dtype=torch.float32, device=DEV).abs() + 0.5).to(FP16)).data_ptr()

    def _alloc_random_weights(self, d: dict) -> dict:
        hidden, HD, mlp_hidden = d["hidden"], d["HD"], d["mlp_hidden"]
        joint_attention_dim = d["joint_attention_dim"]
        weights = {}
        for L in range(d["num_layers_double"]):
            weights[("backbone", "double", L, "txt_in.weight")] = self._rnd_bf16out_linear(hidden, joint_attention_dim)
            weights[("backbone", "double", L, "img_in.weight")] = self._rnd_bf16out_linear(hidden, HD)
            for prefix in ("txt", "img"):
                weights[("backbone", "double", L, f"{prefix}_qkv.weight")] = self._rnd_linear(3 * hidden, hidden)
                weights[("backbone", "double", L, f"{prefix}_proj.weight")] = self._rnd_linear(hidden, hidden)
                weights[("backbone", "double", L, f"{prefix}_mlp0.weight")] = self._rnd_swiglu_mlp(mlp_hidden * 2, hidden)
                weights[("backbone", "double", L, f"{prefix}_mlp2.weight")] = self._rnd_linear(hidden, mlp_hidden)
                weights[("backbone", "double", L, f"{prefix}_query_norm")] = self._rnd_norm_scale(HD)
                weights[("backbone", "double", L, f"{prefix}_key_norm")] = self._rnd_norm_scale(HD)
        for L in range(d["num_layers_single"]):
            weights[("backbone", "single", L, "qkv.weight")] = self._rnd_linear(3 * hidden, hidden)
            weights[("backbone", "single", L, "attn_out_proj.weight")] = self._rnd_linear(hidden, hidden)
            weights[("backbone", "single", L, "mlp_in.weight")] = self._rnd_swiglu_mlp(mlp_hidden * 2, hidden)
            weights[("backbone", "single", L, "mlp_down.weight")] = self._rnd_linear(hidden, mlp_hidden)
            weights[("backbone", "single", L, "query_norm")] = self._rnd_norm_scale(HD)
            weights[("backbone", "single", L, "key_norm")] = self._rnd_norm_scale(HD)

        ahd, aaw, amh = d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"]
        action_dim = d["action_dim"]
        # OPT-001: real action_encoder (WITH bias, the one biased weight
        # in this project) / head, once per denoise step, not per layer
        # -- see imagewam_denoise_step's own docstring.
        weights[("action_dit", "shared", 0, "action_encoder.weight")] = self._rnd_linear(ahd, action_dim)
        weights[("action_dit", "shared", 0, "action_encoder.bias")] = self._own(
            (torch.randn(ahd, dtype=torch.float32, device=DEV) * 0.02).to(FP16)).data_ptr()
        weights[("action_dit", "shared", 0, "head.linear.weight")] = self._rnd_linear(action_dim, ahd)
        for L in range(d["action_num_layers_double"]):
            weights[("action_dit", "double", L, "qkv.weight")] = self._rnd_linear(3 * aaw, ahd)
            weights[("action_dit", "double", L, "proj.weight")] = self._rnd_linear(ahd, aaw)
            weights[("action_dit", "double", L, "mlp0.weight")] = self._rnd_swiglu_mlp(amh * 2, ahd)
            weights[("action_dit", "double", L, "mlp2.weight")] = self._rnd_linear(ahd, amh)
            weights[("action_dit", "double", L, "query_norm")] = self._rnd_norm_scale(HD)
            weights[("action_dit", "double", L, "key_norm")] = self._rnd_norm_scale(HD)
        for L in range(d["action_num_layers_single"]):
            weights[("action_dit", "single", L, "qkv.weight")] = self._rnd_linear(3 * aaw, ahd)
            weights[("action_dit", "single", L, "attn_out_proj.weight")] = self._rnd_linear(ahd, aaw)
            weights[("action_dit", "single", L, "mlp_in.weight")] = self._rnd_swiglu_mlp(amh * 2, ahd)
            weights[("action_dit", "single", L, "mlp_down.weight")] = self._rnd_linear(ahd, amh)
            weights[("action_dit", "single", L, "query_norm")] = self._rnd_norm_scale(HD)
            weights[("action_dit", "single", L, "key_norm")] = self._rnd_norm_scale(HD)
        return weights

    def _load_real_weights(self, d: dict, sd: dict) -> dict:
        """OPT-001 (plan.md): real checkpoint counterpart to
        `_alloc_random_weights` -- same structural shape (same 4-tuple
        keys), sourced from `checkpoint_loader.build_real_weights`
        instead of `torch.randn`. `sd` is the checkpoint's own flat
        state_dict (`checkpoint_loader.load_real_imagewam_state_dict`'s
        own return value), already loaded once by the caller.

        Norm scales and `action_encoder.bias` stay raw CUDA pointers
        (same convention as `_rnd_norm_scale`); every other value goes
        through `_wrap_linear` so precision selection stays uniform and
        orthogonal to weight source, exactly like `_rnd_linear`.
        """
        from flash_rt.models.imagewam.checkpoint_loader import build_real_weights

        raw = build_real_weights(
            sd, num_double=d["num_layers_double"], num_single=d["num_layers_single"],
            action_num_double=d["action_num_layers_double"], action_num_single=d["action_num_layers_single"],
            action_attn_width=d["action_attn_width"])

        weights = {}
        seen_shared: dict[int, object] = {}  # id(cpu tensor) -> wrapped/ptr, for shared txt_in/img_in
        for key, t in raw.items():
            slot = key[-1]
            cpu_id = id(t)
            if cpu_id in seen_shared:
                weights[key] = seen_shared[cpu_id]
                continue
            if slot.endswith("_norm") or slot == "action_encoder.bias":
                tg = self._own(t.to(DEV))
                value = tg.data_ptr()
            elif key[0] == "backbone" and slot in ("txt_in.weight", "img_in.weight"):
                # OPT-001 "FP16 residual overflow" fix -- see Bf16OutLinear's
                # own docstring. `t` is already the real checkpoint weight,
                # FP16 (checkpoint_loader.py's own uniform convention) --
                # upcasting FP16->BF16 here loses nothing meaningful (this
                # weight's own absmax is ~0.26, comfortably exact in FP16
                # already; the only reason for BF16 is the OUTPUT range,
                # not this weight's own precision).
                n, k = t.shape[1], t.shape[0]
                tg = self._own(t.to(DEV, dtype=BF16).contiguous())
                value = Bf16OutLinear(self._gemm, tg.data_ptr(), n, k)
            elif self._precision == "fp16_cutlass" and slot in (
                    "txt_mlp0.weight", "img_mlp0.weight", "mlp0.weight", "mlp_in.weight"):
                # opportunities.md OPT-013: fused SwiGLU gate/up (see
                # CutlassFp16SwiGluMlp's own docstring) -- `t` is the
                # real checkpoint's own merged (K, 2*mlp_hidden) weight,
                # split internally, same real trained values.
                n, k = t.shape[1], t.shape[0]
                tg = self._own(t.to(DEV).contiguous())
                value = CutlassFp16SwiGluMlp(tg.data_ptr(), n // 2, k)
            else:
                n, k = t.shape[1], t.shape[0]  # already (K,N) convention, see checkpoint_loader._w
                tg = self._own(t.to(DEV).contiguous())
                value = self._wrap_linear(tg, n, k)
            weights[key] = value
            seen_shared[cpu_id] = value
        return weights

    def _alloc_buffers(self, d: dict) -> dict:
        hidden, mlp_hidden, x0, a0, HD = d["hidden"], d["mlp_hidden"], d["x0"], d["a0"], d["HD"]
        img_len = a0 - x0
        joint_attention_dim = d["joint_attention_dim"]
        ahd, aaw, amh, num_action = (
            d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"], d["num_action"])
        action_dim = d["action_dim"]
        z = lambda *shape: self._own(torch.zeros(*shape, dtype=FP16, device=DEV))
        # BF16, not FP16 -- OPT-001 "FP16 residual overflow" (opportunities.md):
        # real Qwen3-4B text conditioning, once projected through the real
        # trained backbone at real x0=512, legitimately drives this
        # persistent residual buffer to ~120000 in magnitude, which FP16
        # (max ~65504) cannot represent. `context`/`img_raw` need the same
        # dtype since they feed `combined` directly via `txt_in`/`img_in`
        # (`Bf16OutLinear`, see that class's own docstring).
        self._context = self._own(torch.zeros(x0, joint_attention_dim, dtype=BF16, device=DEV))
        self._backbone_hidden = self._own(torch.zeros(a0, hidden, dtype=BF16, device=DEV))
        self._img_raw = self._own(torch.zeros(img_len, HD, dtype=BF16, device=DEV))
        # OPT-001: real action_dim width (e.g. 7), not action_hidden_dim
        # -- see imagewam_denoise_step's own docstring.
        self._action_latent = self._own(torch.zeros(num_action, action_dim, dtype=F32, device=DEV))
        return {
            "context": self._context.data_ptr(),
            "backbone_hidden": self._backbone_hidden.data_ptr(),
            "img_raw": self._img_raw.data_ptr(),
            "modded_scratch": z(a0, hidden).data_ptr(),
            "txt_qkv_merged": z(x0, 3 * hidden).data_ptr(),
            "img_qkv_merged": z(img_len, 3 * hidden).data_ptr(),
            "single_qkv_merged": z(a0, 3 * hidden).data_ptr(),
            "action_qkv_merged": z(num_action, 3 * aaw).data_ptr(),
            "action_latent_fp16": z(num_action, action_dim).data_ptr(),
            "velocity": z(num_action, action_dim).data_ptr(),
            "head_modded": z(num_action, ahd).data_ptr(),
            "txt_mlp_merged": z(x0, mlp_hidden * 2).data_ptr(),
            "txt_mlp_gated": z(x0, mlp_hidden).data_ptr(),
            "img_mlp_merged": z(img_len, mlp_hidden * 2).data_ptr(),
            "img_mlp_gated": z(img_len, mlp_hidden).data_ptr(),
            "single_mlp_merged": z(a0, mlp_hidden * 2).data_ptr(),
            "single_mlp_gated": z(a0, mlp_hidden).data_ptr(),
            "proj_scratch": z(a0, hidden).data_ptr(),
            "proj_scratch2": z(a0, hidden).data_ptr(),
            "action_latent": self._action_latent.data_ptr(),
            "action_hidden": z(num_action, ahd).data_ptr(),
            "action_modded": z(num_action, ahd).data_ptr(),
            "action_proj_scratch": z(num_action, ahd).data_ptr(),
            "action_proj_scratch2": z(num_action, ahd).data_ptr(),
            "action_mlp_merged": z(num_action, amh * 2).data_ptr(),
            "action_mlp_gated": z(num_action, amh).data_ptr(),
        }

    def _compute_backbone_modulation(self, d: dict, *, real_mod: dict | None = None):
        """Backbone's own AdaLN modulation, computed ONCE: real
        inference always conditions the reference/context encode on a
        FIXED timestep=0 (confirmed against the real checkpoint run,
        `benchmarks/imagewam_real_checkpoint_validation.py`'s own
        `video_timestep = torch.zeros(1)`), so this never needs
        recomputing per replay -- see pipeline_thor.py's own docstring.

        `real_mod`: OPT-001, the real `mod_w` dict from
        `checkpoint_loader.build_real_modulation_weights()["backbone"]`
        -- when given, used INSTEAD of random weights (moved to CUDA
        here, same as every other real-weight tensor in this file).
        """
        hidden = d["hidden"]
        if real_mod is not None:
            mod_w = {k: self._own(v.to(DEV)) for k, v in real_mod.items()}
        else:
            mod_w = {
                "time_in_w1": self._own(torch.randn(hidden, 256, dtype=torch.float32, device=DEV) * 0.02),
                "time_in_w2": self._own(torch.randn(hidden, hidden, dtype=torch.float32, device=DEV) * 0.02),
                "mod_double_txt": self._own(torch.randn(6 * hidden, hidden, dtype=torch.float32, device=DEV) * 0.02),
                "mod_double_img": self._own(torch.randn(6 * hidden, hidden, dtype=torch.float32, device=DEV) * 0.02),
                "mod_single": self._own(torch.randn(3 * hidden, hidden, dtype=torch.float32, device=DEV) * 0.02),
            }
        timestep = self._own(torch.zeros(1, dtype=torch.float32, device=DEV))
        mod_txt, mod_img, mod_single = compute_shared_modulation(timestep, mod_w, hidden)
        for group in (mod_txt[0], mod_txt[1], mod_img[0], mod_img[1], mod_single):
            for t in group:
                self._own(t)
        return mod_txt, mod_img, mod_single

    def _compute_action_modulations(self, d: dict, *, real_mod: dict | None = None):
        """ActionDiT's own AdaLN modulation, ONE tuple PER DENOISE STEP:
        its conditioning timestep changes every step, but `step` is
        itself a compile-time Python constant during CUDA Graph
        capture, so every step's own modulation is ALSO a compile-time
        constant -- precomputed here, once, never recomputed per
        replay.

        Timestep schedule: `d["shift"]` set -> the REAL non-uniform
        shift-based schedule (opportunities.md OPT-009's follow-up,
        `scheduler.build_inference_schedule`, ported verbatim from the
        real `WanContinuousFlowMatchScheduler` -- confirmed real LIBERO
        release values `shift=5.0`, `num_train_timesteps=1000`,
        `eval_num_inference_steps=10` in that release's own
        `config.yaml`). `d["shift"]` unset (default) -> this project's
        ORIGINAL fixed-uniform `action_timestep = 1.0 - step*dt`
        simplification, unchanged -- every existing caller/test is
        unaffected. Also returns `deltas` (`None` in the unset case) --
        the caller threads it into `imagewam_denoise_loop`'s own
        `deltas=` for the matching per-step Euler step size.

        `real_mod`: OPT-001, the real `mod_w` dict from
        `checkpoint_loader.build_real_modulation_weights()["action"]`
        (includes `head_adaln`) -- when given, used instead of random.
        """
        ahd = d["action_hidden_dim"]
        if real_mod is not None:
            mod_w = {k: self._own(v.to(DEV)) for k, v in real_mod.items()}
        else:
            mod_w = {
                "time_in_w1": self._own(torch.randn(ahd, 256, dtype=torch.float32, device=DEV) * 0.02),
                "time_in_w2": self._own(torch.randn(ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
                "mod_double": self._own(torch.randn(6 * ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
                "mod_single": self._own(torch.randn(3 * ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
                # OPT-001: head's own AdaLN modulation (shift/scale only, no
                # gate -- see adaln.head_modulation's own docstring).
                "head_adaln": self._own(torch.randn(2 * ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
            }
        shift = d.get("shift")
        deltas_out = None
        if shift is not None:
            from flash_rt.models.imagewam.scheduler import build_inference_schedule
            num_train_timesteps = d.get("num_train_timesteps", 1000)
            timesteps, deltas = build_inference_schedule(
                d["num_denoise_steps"], shift=shift, num_train_timesteps=num_train_timesteps, device=DEV)
            action_timesteps = (timesteps / num_train_timesteps).tolist()
            deltas_out = deltas.tolist()
        else:
            dt = d["dt"]
            action_timesteps = [1.0 - step * dt for step in range(d["num_denoise_steps"])]
        mods, head_mods = [], []
        for step, action_timestep in enumerate(action_timesteps):
            timestep = self._own(torch.full((1,), action_timestep, dtype=torch.float32, device=DEV))
            mod_double, mod_single = compute_action_modulation(timestep, mod_w, ahd)
            for t in mod_double[0]:
                self._own(t)
            for t in mod_double[1]:
                self._own(t)
            for t in mod_single:
                self._own(t)
            mods.append((mod_double, mod_single))
            head_shift, head_scale = compute_action_head_modulation(timestep, mod_w, ahd)
            self._own(head_shift)
            self._own(head_scale)
            head_mods.append((head_shift, head_scale))
        return mods, head_mods, deltas_out

    def _capture_graph(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                imagewam_prefill(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                  self.dims, stream=s.cuda_stream, attn=self._attn,
                                  mod_txt=self._mod_txt, mod_img=self._mod_img,
                                  mod_single=self._mod_single, rope_table=self._rope_table.data_ptr())
                imagewam_denoise_loop(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                       self.dims, stream=s.cuda_stream, attn=self._attn,
                                       action_mods=self._action_mods, head_mods=self._head_mods,
                                       action_rope_table=self._action_rope_table.data_ptr(),
                                       deltas=self._deltas)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            imagewam_prefill(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                              self.dims, stream=s.cuda_stream, attn=self._attn,
                              mod_txt=self._mod_txt, mod_img=self._mod_img,
                              mod_single=self._mod_single, rope_table=self._rope_table.data_ptr())
            imagewam_denoise_loop(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                   self.dims, stream=s.cuda_stream, attn=self._attn,
                                   action_mods=self._action_mods, head_mods=self._head_mods,
                                   action_rope_table=self._action_rope_table.data_ptr(),
                                   deltas=self._deltas)
        self._graph = graph

    def _set_context_with_optional_proprio(self, text_ctx: torch.Tensor, text_mask: torch.Tensor) -> None:
        """`text_ctx`: `(text_len, joint_attention_dim)` BF16 -- the
        real (or precomputed) Qwen3 text context, BEFORE proprio.
        `text_mask`: `(text_len,)` bool, `1` for real tokens.

        `self._proprio_dim is None`: `text_ctx` must already be exactly
        `dims["x0"]` rows -- copied in directly, unchanged behavior.

        `self._proprio_dim` set: replicates `imagewam.py`'s own real
        `_append_proprio_to_context` (`pack_proprio_after_text=True`
        branch) EXACTLY -- real tokens keep their rank, the proprio
        slot lands at row `valid_counts` (right after the last real
        token), padding shifts one row later to make room. Real tokens'
        own RoPE positions are unaffected (same indices either way);
        the proprio slot's position must match the real model's
        placement for its own RoPE position to be correct -- found
        while scoping real closed-loop testing, opportunities.md.
        `dims["x0"]` must equal `text_ctx`'s own length + 1 (validated
        below). The proprio ROW ITSELF is left zero here -- `infer()`
        overwrites it with the real `proprio_encoder(proprio)` output
        every call, since proprio (unlike the text prompt) changes
        every control step.
        """
        x0 = self.dims["x0"]
        if self._proprio_dim is None:
            if text_ctx.shape[0] != x0:
                raise ValueError(f"context length {text_ctx.shape[0]} != dims['x0']={x0}")
            self._context.copy_(text_ctx)
            return
        text_len = text_ctx.shape[0]
        if x0 != text_len + 1:
            raise ValueError(
                f"dims['x0']={x0} must equal the text context length ({text_len}) + 1 "
                f"(the proprio slot) when dims['proprio_dim'] is set")
        valid_counts = int(text_mask.sum().item())
        self._proprio_row = valid_counts
        self._context.zero_()
        self._context[:valid_counts].copy_(text_ctx[:valid_counts])
        self._context[valid_counts + 1:x0].copy_(text_ctx[valid_counts:text_len])

    def set_prompt(self, prompt_text: str | None = None, *,
                    context: torch.Tensor | None = None,
                    context_mask: torch.Tensor | None = None) -> None:
        """Random-fills the text-context input (default), OR loads a
        real precomputed `context`/`context_mask` pair, OR (if this
        frontend was constructed with `qwen3_model_spec=`) live-encodes
        `prompt_text` through the real Qwen3 text encoder -- then
        captures the graph. Matches `imagewam.py`'s own real
        `_prepare_flux2_infer_text`: a raw prompt XOR a precomputed
        `context`/`context_mask` pair, never both.

        `context_mask` is accepted (matching the real interface and
        `_imagewam_thor_spec.py`'s own declared input shape). Used for
        two things when `dims["proprio_dim"]` is set (see
        `_set_context_with_optional_proprio`): finding the real proprio
        insertion row, and nothing else -- `pipeline_thor.py`'s own
        attention math still treats every context row as valid
        regardless of padding, a pre-existing, separately-documented
        gap (see that module's own docstring), not something this
        change fixes.
        """
        if prompt_text is not None and context is not None:
            raise ValueError("set_prompt: prompt_text and context are mutually exclusive "
                              "(matches imagewam.py's own _prepare_flux2_infer_text)")
        cache_key = (prompt_text, context is not None)
        if cache_key == self._current_prompt:
            return
        if context is not None:
            if context_mask is None:
                raise ValueError("set_prompt(context=...) requires context_mask too "
                                  "(matches imagewam.py's own _prepare_flux2_infer_text)")
            self._set_context_with_optional_proprio(
                context.to(device=DEV, dtype=BF16), context_mask.to(device=DEV, dtype=torch.bool))
        elif self._qwen3 is not None and prompt_text is not None:
            from flash_rt.models.imagewam.text_encoder import encode_prompts
            model, tokenizer = self._qwen3
            real_context, real_mask = encode_prompts(model, tokenizer, [prompt_text])
            self._set_context_with_optional_proprio(
                real_context[0].to(device=DEV, dtype=BF16), real_mask[0].to(device=DEV))
        else:
            self._context.normal_()
            if self._proprio_dim is not None:
                # No real context/mask on this structural path -- no
                # "real token count" to place the proprio slot after,
                # so it goes at the very last row. Arbitrary, no
                # accuracy claim on this path either way (matches
                # every other random-fill branch in this class).
                self._proprio_row = self.dims["x0"] - 1
        if self._graph is None:
            self._calibrate_fp8(self.dims)
            self._capture_graph()
        self._current_prompt = cache_key

    def infer(self, observation: dict) -> dict:
        """Replay the captured graph with a new observation.

        `observation` random-fills `img_raw` by default (unchanged
        placeholder, standing in for whatever a real VAE would have
        produced) UNLESS this frontend was constructed with
        `ae_model_path=`/`flux2_src=` AND `observation` contains a real
        `"view1"` (optionally `"view2"`) camera frame -- then the real
        VAE (`vae_encoder.encode_to_tokens`) runs OUTSIDE the captured
        graph (plain PyTorch/`flux2`-dependent code has no business
        being captured) and its result is copied into `img_raw` before
        `.replay()`. `backbone_hidden`'s own image rows are WRITTEN by
        `img_in.weight` inside the graph itself either way, not filled
        directly here.

        `observation["proprio"]` -- real closed-loop robot-state
        conditioning (opportunities.md, found 2026-09-15), REQUIRED
        (raises `ValueError`, matching the real model's own
        `_append_proprio_to_context_if_enabled`) when this frontend was
        constructed with `dims["proprio_dim"]` set. Unlike the text
        prompt (fixed per `set_prompt()` episode), proprio genuinely
        changes every control step -- normalized via the real
        `dataset_stats.json` `state` min/max (if `dataset_stats_path`
        was given at construction; raw otherwise, no accuracy claim),
        projected through the real `proprio_encoder` OUTSIDE the graph
        (plain `F.linear`, same convention as the VAE/Qwen3 encoders),
        and copied into the row `set_prompt()` already reserved for it
        (`self._proprio_row`) -- BEFORE `.replay()`, same pattern as
        `img_raw`.
        """
        if self._graph is None:
            raise RuntimeError("call set_prompt() before infer()")
        if self._ae is not None and "view1" in observation:
            from flash_rt.models.imagewam.vae_encoder import encode_to_tokens
            tokens = encode_to_tokens(self._ae, observation["view1"], observation.get("view2"))
            self._img_raw.copy_(tokens[0].to(dtype=BF16))
        else:
            self._img_raw.normal_()
        if self._proprio_dim is not None:
            proprio = observation.get("proprio")
            if proprio is None:
                raise ValueError(
                    "infer(observation=...) requires observation['proprio'] when "
                    "dims['proprio_dim'] is set (matches imagewam.py's own "
                    "_append_proprio_to_context_if_enabled)")
            proprio_t = torch.as_tensor(proprio, dtype=torch.float32, device=DEV).reshape(1, self._proprio_dim)
            if self._state_norm is not None:
                proprio_t = self._state_norm.forward(proprio_t)
            proprio_tok = torch.nn.functional.linear(
                proprio_t.to(dtype=BF16), self._proprio_w, self._proprio_b)
            self._context[self._proprio_row].copy_(proprio_tok[0])
        self._action_latent.normal_()
        self._action_latent.mul_(0.01)
        self._graph.replay()
        torch.cuda.synchronize()
        actions = self._action_latent.detach()
        if self._action_norm is not None:
            actions = self._action_norm.backward(actions)
        return {"actions": actions.cpu().numpy()}
