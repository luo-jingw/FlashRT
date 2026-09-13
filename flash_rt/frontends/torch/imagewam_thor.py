"""ImageWAM (FLUX.2-4B variant) Thor torch frontend.

plan.md Phase 5. Structural dry-run scope only (see PROJECT.md,
pipeline_thor.py's own docstring): random-initialized weights, no
real checkpoint, no calibration. Wires Phases 1-4 into `set_prompt()`
and `infer()`, following `_template/frontend.py`'s STEP 1-6 shape --
adapted where the real, working `CosmosEdgeThor`
(`flash_rt/models/cosmos3_edge/pipeline_thor.py`) precedent differs
from the generic template (plain `torch.cuda.Tensor` + `.data_ptr()`
throughout, not the template's `CudaBuffer` ctypes wrapper -- this
project has used torch tensors for every buffer since Phase 2, and
switching to `CudaBuffer` here for no reason would just be a second,
inconsistent buffer-ownership convention in the same codebase).

`checkpoint_dir` is accepted for interface parity with every other
FlashRT frontend but unused: every shape here is random-filled from
`dims`, never loaded from a real checkpoint (see `opportunities.md`
OPT-001).

`context_mask` (declared as an input shape in `_imagewam_thor_spec.py`)
is accepted by `set_prompt` but never read by anything --
`pipeline_thor.py`'s double-stream blocks treat every context row as
valid, no padding mask. Not modeled, consistent with everything else
already deferred to real-checkpoint work.

Dims default to a small, deliberately-not-real-FLUX.2-4B-size
structural test scale (this machine's own 8GB GPU headroom, per
PROJECT.md) -- pass `dims_override` for Thor-scale testing.
"""
from __future__ import annotations

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.hardware.thor.attn_backend import ImageWAMAttnBackend, make_imagewam_attention_spec
from flash_rt.models.imagewam.pipeline_thor import imagewam_denoise_loop, imagewam_prefill

DEV = "cuda"
FP16 = torch.float16
F32 = torch.float32

_DEFAULT_DIMS = dict(
    hidden=96, HD=16, NH=6, mlp_hidden=192, joint_attention_dim=64,
    x0=4, a0=8, num_layers_double=2, num_layers_single=3,
    action_hidden_dim=32, action_attn_width=96, action_mlp_hidden=64,
    num_action=3, total=11,
    action_num_layers_double=2, action_num_layers_single=3,
    dt=0.5, num_denoise_steps=2,
)


class ImageWAMTorchFrontendThor:
    """Thor frontend for ImageWAM's structural (random-weight) dry run.

    Naming rule: `<Model><Framework>Frontend<Hardware>` per
    `docs/adding_new_model.md` §0 rule 2.
    """

    def __init__(self, checkpoint_dir=None, *, dims_override: dict | None = None, **kwargs):
        del checkpoint_dir, kwargs
        self._keepalive = []
        self.dims = dict(_DEFAULT_DIMS)
        if dims_override:
            self.dims.update(dims_override)
        d = self.dims

        self._ctx = fvk.FvkContext()
        self._gemm = fvk.GemmRunner()

        self._weights = self._alloc_random_weights(d)
        self._bufs = self._alloc_buffers(d)

        spec = make_imagewam_attention_spec(max_prefix_seq=d["a0"], max_total_seq=d["total"])
        num_layers = d["num_layers_double"] + d["num_layers_single"]
        HD, hidden = d["HD"], d["hidden"]
        self._K_cache = self._own(torch.zeros(num_layers, d["total"], HD, dtype=FP16, device=DEV))
        self._V_cache = self._own(torch.zeros(num_layers, d["total"], HD, dtype=FP16, device=DEV))
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
        )

        self._graph = None
        self._current_prompt = None

    def _own(self, t: torch.Tensor) -> torch.Tensor:
        """Keep a buffer tensor alive for the frontend's own lifetime.

        A bare `torch.zeros(...).data_ptr()` drops the only Python
        reference the instant that expression finishes -- PyTorch's
        caching allocator is then free to hand the same memory to the
        next allocation, silently corrupting an already-stored pointer.
        Found the hard way in Phase 3/4's own tests; every buffer here
        goes through this helper for the same reason.
        """
        self._keepalive.append(t)
        return t

    def _alloc_random_weights(self, d: dict) -> dict:
        rnd = lambda *shape: self._own(torch.randn(*shape, dtype=FP16, device=DEV)).data_ptr()
        weights = {}
        hidden, HD, mlp_hidden = d["hidden"], d["HD"], d["mlp_hidden"]
        joint_attention_dim = d["joint_attention_dim"]
        for L in range(d["num_layers_double"]):
            weights[("backbone", "double", L, "txt_in")] = rnd(joint_attention_dim, hidden)
            for prefix in ("txt", "img"):
                weights[("backbone", "double", L, f"{prefix}_q")] = rnd(hidden, hidden)
                weights[("backbone", "double", L, f"{prefix}_k")] = rnd(hidden, HD)
                weights[("backbone", "double", L, f"{prefix}_v")] = rnd(hidden, HD)
                weights[("backbone", "double", L, f"{prefix}_proj")] = rnd(hidden, hidden)
                weights[("backbone", "double", L, f"{prefix}_mlp0")] = rnd(hidden, mlp_hidden)
                weights[("backbone", "double", L, f"{prefix}_mlp2")] = rnd(mlp_hidden, hidden)
        for L in range(d["num_layers_single"]):
            weights[("backbone", "single", L, "q")] = rnd(hidden, hidden)
            weights[("backbone", "single", L, "k")] = rnd(hidden, HD)
            weights[("backbone", "single", L, "v")] = rnd(hidden, HD)
            weights[("backbone", "single", L, "mlp_in")] = rnd(hidden, mlp_hidden)
            weights[("backbone", "single", L, "attn_out_proj")] = rnd(hidden, hidden)
            weights[("backbone", "single", L, "mlp_down")] = rnd(mlp_hidden, hidden)

        ahd, aaw, amh = d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"]
        for L in range(d["action_num_layers_double"]):
            weights[("action_dit", "double", L, "q")] = rnd(ahd, aaw)
            weights[("action_dit", "double", L, "k")] = rnd(ahd, HD)
            weights[("action_dit", "double", L, "v")] = rnd(ahd, HD)
            weights[("action_dit", "double", L, "proj")] = rnd(aaw, ahd)
            weights[("action_dit", "double", L, "mlp0")] = rnd(ahd, amh)
            weights[("action_dit", "double", L, "mlp2")] = rnd(amh, ahd)
        for L in range(d["action_num_layers_single"]):
            weights[("action_dit", "single", L, "q")] = rnd(ahd, aaw)
            weights[("action_dit", "single", L, "k")] = rnd(ahd, HD)
            weights[("action_dit", "single", L, "v")] = rnd(ahd, HD)
            weights[("action_dit", "single", L, "mlp_in")] = rnd(ahd, amh)
            weights[("action_dit", "single", L, "attn_out_proj")] = rnd(aaw, ahd)
            weights[("action_dit", "single", L, "mlp_down")] = rnd(amh, ahd)
        return weights

    def _alloc_buffers(self, d: dict) -> dict:
        hidden, mlp_hidden, x0, a0 = d["hidden"], d["mlp_hidden"], d["x0"], d["a0"]
        joint_attention_dim = d["joint_attention_dim"]
        ahd, amh, num_action = d["action_hidden_dim"], d["action_mlp_hidden"], d["num_action"]
        z = lambda *shape: self._own(torch.zeros(*shape, dtype=FP16, device=DEV))
        ones_hidden = self._own(torch.ones(hidden, dtype=FP16, device=DEV))
        ones_action = self._own(torch.ones(ahd, dtype=FP16, device=DEV))
        self._context = self._own(torch.zeros(x0, joint_attention_dim, dtype=FP16, device=DEV))
        self._backbone_hidden = self._own(torch.zeros(a0, hidden, dtype=FP16, device=DEV))
        self._action_latent = self._own(torch.zeros(num_action, ahd, dtype=F32, device=DEV))
        return {
            "context": self._context.data_ptr(),
            "backbone_hidden": self._backbone_hidden.data_ptr(),
            "txt_mlp_hidden": z(x0, mlp_hidden).data_ptr(),
            "img_mlp_hidden": z(a0 - x0, mlp_hidden).data_ptr(),
            "single_mlp_hidden": z(a0, mlp_hidden).data_ptr(),
            "proj_scratch": z(a0, hidden).data_ptr(),
            "normed_scratch": z(a0, hidden).data_ptr(),
            "norm_ones": ones_hidden.data_ptr(),
            "action_latent": self._action_latent.data_ptr(),
            "action_hidden": z(num_action, ahd).data_ptr(),
            "action_normed": z(num_action, ahd).data_ptr(),
            "action_norm_ones": ones_action.data_ptr(),
            "action_proj_scratch": z(num_action, ahd).data_ptr(),
            "action_mlp_hidden": z(num_action, amh).data_ptr(),
        }

    def _capture_graph(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                imagewam_prefill(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                  self.dims, stream=s.cuda_stream, attn=self._attn)
                imagewam_denoise_loop(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                       self.dims, stream=s.cuda_stream, attn=self._attn)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            imagewam_prefill(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                              self.dims, stream=s.cuda_stream, attn=self._attn)
            imagewam_denoise_loop(self._ctx, fvk, self._gemm, self._bufs, self._weights,
                                   self.dims, stream=s.cuda_stream, attn=self._attn)
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
