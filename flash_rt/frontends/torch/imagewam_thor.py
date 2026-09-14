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
from flash_rt.models.imagewam.pipeline_real import compute_action_modulation, compute_shared_modulation
from flash_rt.models.imagewam.rope import build_action_rope_table, build_backbone_rope_table

DEV = "cuda"
FP16 = torch.float16
F32 = torch.float32

_DEFAULT_DIMS = dict(
    hidden=256, HD=128, NH=2, mlp_hidden=384, joint_attention_dim=64,
    x0=3, a0=8, num_layers_double=2, num_layers_single=3,
    action_hidden_dim=128, action_attn_width=256, action_mlp_hidden=192,
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
                 use_fa4: bool = False, **kwargs):
        del checkpoint_dir, kwargs
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

        self._ctx = fvk.FvkContext()
        self._gemm = fvk.GemmRunner()
        self._autotune_gemm(d)

        self._weights = self._alloc_random_weights(d)
        self._bufs = self._alloc_buffers(d)
        self._rope_table = self._own(build_backbone_rope_table(
            d["x0"], d["a0"] - d["x0"], 1, device=DEV))
        self._action_rope_table = self._own(build_action_rope_table(d["num_action"], device=DEV))
        self._mod_txt, self._mod_img, self._mod_single = self._compute_backbone_modulation(d)
        self._action_mods = self._compute_action_modulations(d)

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
        hidden, mlp_hidden = d["hidden"], d["mlp_hidden"]
        joint_attention_dim = d["joint_attention_dim"]
        x0, a0 = d["x0"], d["a0"]
        img_len = a0 - x0
        ahd, aaw, amh, num_action = (
            d["action_hidden_dim"], d["action_attn_width"],
            d["action_mlp_hidden"], d["num_action"])

        shapes = {
            (x0, hidden, joint_attention_dim),      # txt_in
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
        }
        for m, n, k in shapes:
            x = torch.zeros(m, k, dtype=FP16, device=DEV)
            w = torch.zeros(k, n, dtype=FP16, device=DEV)
            out = torch.zeros(m, n, dtype=FP16, device=DEV)
            self._gemm.autotune_fp16_nn(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, k, 16)
        torch.cuda.synchronize()

    def _rnd_linear(self, n: int, k: int) -> int:
        """Real GEMM (K,N) convention: `n` = output width, `k` = input
        width, stored as (k, n) so `gemm.fp16_nn` reads it directly (a
        real checkpoint's own (out,in) `nn.Linear` weight would need
        `.t().contiguous()` at load time -- see `CosmosEdgeThor`'s own
        precedent)."""
        return self._own(torch.randn(k, n, dtype=FP16, device=DEV) * 0.02).data_ptr()

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
            weights[("backbone", "double", L, "txt_in.weight")] = self._rnd_linear(hidden, joint_attention_dim)
            for prefix in ("txt", "img"):
                weights[("backbone", "double", L, f"{prefix}_qkv.weight")] = self._rnd_linear(3 * hidden, hidden)
                weights[("backbone", "double", L, f"{prefix}_proj.weight")] = self._rnd_linear(hidden, hidden)
                weights[("backbone", "double", L, f"{prefix}_mlp0.weight")] = self._rnd_linear(mlp_hidden * 2, hidden)
                weights[("backbone", "double", L, f"{prefix}_mlp2.weight")] = self._rnd_linear(hidden, mlp_hidden)
                weights[("backbone", "double", L, f"{prefix}_query_norm")] = self._rnd_norm_scale(HD)
                weights[("backbone", "double", L, f"{prefix}_key_norm")] = self._rnd_norm_scale(HD)
        for L in range(d["num_layers_single"]):
            weights[("backbone", "single", L, "qkv.weight")] = self._rnd_linear(3 * hidden, hidden)
            weights[("backbone", "single", L, "attn_out_proj.weight")] = self._rnd_linear(hidden, hidden)
            weights[("backbone", "single", L, "mlp_in.weight")] = self._rnd_linear(mlp_hidden * 2, hidden)
            weights[("backbone", "single", L, "mlp_down.weight")] = self._rnd_linear(hidden, mlp_hidden)
            weights[("backbone", "single", L, "query_norm")] = self._rnd_norm_scale(HD)
            weights[("backbone", "single", L, "key_norm")] = self._rnd_norm_scale(HD)

        ahd, aaw, amh = d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"]
        for L in range(d["action_num_layers_double"]):
            weights[("action_dit", "double", L, "qkv.weight")] = self._rnd_linear(3 * aaw, ahd)
            weights[("action_dit", "double", L, "proj.weight")] = self._rnd_linear(ahd, aaw)
            weights[("action_dit", "double", L, "mlp0.weight")] = self._rnd_linear(amh * 2, ahd)
            weights[("action_dit", "double", L, "mlp2.weight")] = self._rnd_linear(ahd, amh)
            weights[("action_dit", "double", L, "query_norm")] = self._rnd_norm_scale(HD)
            weights[("action_dit", "double", L, "key_norm")] = self._rnd_norm_scale(HD)
        for L in range(d["action_num_layers_single"]):
            weights[("action_dit", "single", L, "qkv.weight")] = self._rnd_linear(3 * aaw, ahd)
            weights[("action_dit", "single", L, "attn_out_proj.weight")] = self._rnd_linear(ahd, aaw)
            weights[("action_dit", "single", L, "mlp_in.weight")] = self._rnd_linear(amh * 2, ahd)
            weights[("action_dit", "single", L, "mlp_down.weight")] = self._rnd_linear(ahd, amh)
            weights[("action_dit", "single", L, "query_norm")] = self._rnd_norm_scale(HD)
            weights[("action_dit", "single", L, "key_norm")] = self._rnd_norm_scale(HD)
        return weights

    def _alloc_buffers(self, d: dict) -> dict:
        hidden, mlp_hidden, x0, a0 = d["hidden"], d["mlp_hidden"], d["x0"], d["a0"]
        img_len = a0 - x0
        joint_attention_dim = d["joint_attention_dim"]
        ahd, aaw, amh, num_action = (
            d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"], d["num_action"])
        z = lambda *shape: self._own(torch.zeros(*shape, dtype=FP16, device=DEV))
        self._context = self._own(torch.zeros(x0, joint_attention_dim, dtype=FP16, device=DEV))
        self._backbone_hidden = self._own(torch.zeros(a0, hidden, dtype=FP16, device=DEV))
        self._action_latent = self._own(torch.zeros(num_action, ahd, dtype=F32, device=DEV))
        return {
            "context": self._context.data_ptr(),
            "backbone_hidden": self._backbone_hidden.data_ptr(),
            "modded_scratch": z(a0, hidden).data_ptr(),
            "txt_qkv_merged": z(x0, 3 * hidden).data_ptr(),
            "img_qkv_merged": z(img_len, 3 * hidden).data_ptr(),
            "single_qkv_merged": z(a0, 3 * hidden).data_ptr(),
            "action_qkv_merged": z(num_action, 3 * aaw).data_ptr(),
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

    def _compute_backbone_modulation(self, d: dict):
        """Backbone's own AdaLN modulation, computed ONCE: real
        inference always conditions the reference/context encode on a
        FIXED timestep=0 (confirmed against the real checkpoint run,
        `benchmarks/imagewam_real_checkpoint_validation.py`'s own
        `video_timestep = torch.zeros(1)`), so this never needs
        recomputing per replay -- see pipeline_thor.py's own docstring.
        """
        hidden = d["hidden"]
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

    def _compute_action_modulations(self, d: dict):
        """ActionDiT's own AdaLN modulation, ONE tuple PER DENOISE STEP:
        its conditioning timestep changes every step (flow-matching
        schedule, `1.0 -> 0.0` uniform), but `step` is itself a
        compile-time Python constant during CUDA Graph capture, so
        every step's own modulation is ALSO a compile-time constant --
        precomputed here, once, never recomputed per replay.
        """
        ahd = d["action_hidden_dim"]
        mod_w = {
            "time_in_w1": self._own(torch.randn(ahd, 256, dtype=torch.float32, device=DEV) * 0.02),
            "time_in_w2": self._own(torch.randn(ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
            "mod_double": self._own(torch.randn(6 * ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
            "mod_single": self._own(torch.randn(3 * ahd, ahd, dtype=torch.float32, device=DEV) * 0.02),
        }
        dt = d["dt"]
        mods = []
        for step in range(d["num_denoise_steps"]):
            action_timestep = 1.0 - step * dt
            timestep = self._own(torch.full((1,), action_timestep, dtype=torch.float32, device=DEV))
            mod_double, mod_single = compute_action_modulation(timestep, mod_w, ahd)
            for t in mod_double[0]:
                self._own(t)
            for t in mod_double[1]:
                self._own(t)
            for t in mod_single:
                self._own(t)
            mods.append((mod_double, mod_single))
        return mods

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
                                       action_mods=self._action_mods,
                                       action_rope_table=self._action_rope_table.data_ptr())
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            imagewam_prefill(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                              self.dims, stream=s.cuda_stream, attn=self._attn,
                              mod_txt=self._mod_txt, mod_img=self._mod_img,
                              mod_single=self._mod_single, rope_table=self._rope_table.data_ptr())
            imagewam_denoise_loop(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                   self.dims, stream=s.cuda_stream, attn=self._attn,
                                   action_mods=self._action_mods,
                                   action_rope_table=self._action_rope_table.data_ptr())
        self._graph = graph

    def set_prompt(self, prompt_text: str) -> None:
        """Random-fills the text-context input, then captures the graph.

        No real Qwen3-4B forward, no calibration-cache lookup: ImageWAM's
        own config always sets `load_text_encoder: false` (see
        `_imagewam_thor_spec.py`), so `context`/`context_mask` are inputs
        this integration never computes from a real prompt string --
        `prompt_text` only distinguishes cache hits from misses, same as
        the generic template's `set_prompt`, minus the actual embedding
        step it would otherwise do.
        """
        if prompt_text == self._current_prompt:
            return
        self._context.normal_()
        if self._graph is None:
            self._capture_graph()
        self._current_prompt = prompt_text

    def infer(self, observation: dict) -> dict:
        """Replay the captured graph with a new (random) observation.

        `observation` is accepted for interface parity but its contents
        are not used: the real image-encode step is a VAE forward, out
        of scope (see `imagewam_encode_once`'s own docstring) -- the
        image rows of `backbone_hidden` are random-filled here in place
        of a real encoded observation, standing in for whatever a real
        VAE would have produced.
        """
        del observation
        if self._graph is None:
            raise RuntimeError("call set_prompt() before infer()")
        x0, a0 = self.dims["x0"], self.dims["a0"]
        img_rows = self._backbone_hidden[x0:a0]
        img_rows.normal_()
        self._action_latent.normal_()
        self._action_latent.mul_(0.01)
        self._graph.replay()
        torch.cuda.synchronize()
        return {"actions": self._action_latent.detach().cpu().numpy()}
