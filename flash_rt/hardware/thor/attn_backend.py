"""FlashRT — Thor attention backend.

Implements the AttentionBackend protocol (``flash_rt/hardware/backend.py``)
on Thor SM110 using the fvk attention primitives. Pipeline-owned memory
model: the pipeline allocates all Q/K/V/O/logits buffers (and the layered
KV cache in weights) and passes pointers at construction. The backend is a
thin dispatch layer — it looks up (site, layer_idx) → fvk call + pointer
arguments and returns the output pointer.

Currently supports Pi0.5's three sites:

  * ``siglip``   — multi-view vision FMHA via ``fmha_strided_full``.
                   QKV is interleaved in a single buffer (stride=3*D);
                   backend derives Q/K/V ptrs via pointer arithmetic.
  * ``encoder``  — PaliGemma 18-layer self-attention via
                   ``attention_qkv_fp16``. Q/O alias the same buffer
                   (``attn_out`` from the pipeline).
  * ``decoder``  — cross-attention into the encoder KV cache via the
                   same ``attention_qkv_fp16`` primitive. Reuses the
                   encoder's Kc/Vc pointers (shared cache).

Pi0 / Pi0-FAST / GROOT are out of scope for Stage 1 — Pi0-FAST is
explicitly excluded by ``docs/stable_api.md``; Pi0 and GROOT will be
addressed in later stages.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec


def _fp16_tensor_from_ptr(ptr: int, shape, strides=None, offset: int = 0):
    """Create a zero-copy CUDA torch tensor view over a raw fp16 pointer."""
    import torch

    interface = {
        "data": (int(ptr) + int(offset) * 2, False),
        "shape": tuple(int(dim) for dim in shape),
        "typestr": "<f2",
        "version": 3,
    }
    if strides is not None:
        interface["strides"] = tuple(int(stride) * 2 for stride in strides)
    owner = type(
        "_Fp16CudaArrayInterface",
        (),
        {"__cuda_array_interface__": interface},
    )()
    return torch.as_tensor(owner, device="cuda")


class ThorFlashAttnBackend(AttentionBackendBase):
    """Pi0.5 attention backend on Thor (SM110).

    Constructed by the frontend after it has allocated all pipeline
    buffers and loaded weights; injected into ``siglip_forward`` /
    ``encoder_forward`` / ``decoder_forward`` via an optional ``attn=``
    kwarg. Legacy fallback path (attn=None) remains available during the
    staged rollout so frontends can opt in one at a time.
    """

    # ────────────────────────────────────────────────────────────────
    # Construction
    # ────────────────────────────────────────────────────────────────
    def __init__(self, spec: AttentionSpec, ctx, *,
                 siglip_slots: dict, encoder_slots: dict,
                 decoder_slots: dict,
                 fixed_control: Optional[dict] = None,
                 use_fa4: bool = False) -> None:
        """
        Args:
            spec: AttentionSpec with exactly the sites
                {"siglip", "encoder", "decoder"}.
            ctx:  the object passed as the first argument to
                ``fvk.attention_qkv_fp16``. Accepted forms:
                  * raw ``fvk.FvkContext`` (Pi0.5 frontends)
                  * ``flash_rt.core.context.FvkContext`` Python wrapper
                    (newer call sites)
                Backend passes it through unchanged; it auto-detects the
                wrapper form via a ``.cpp`` attribute at call time.
            siglip_slots: dict with keys:
                qkv (int)  — interleaved QKV buffer ptr, stride = 3*D
                O   (int)  — attn_out buffer ptr, [num_views*spv, D] fp16
                D   (int)  — hidden dim; used for pointer arithmetic
            encoder_slots / decoder_slots: dict with keys:
                Q_O          (int) — attn_out; Q input AND O output
                Kc, Vc       (int) — layered KV cache base pointers
                logits       (int) — scratch for P = QK^T
                layer_stride (int) — bytes between successive layers in Kc/Vc
                scale        (float) — 1/sqrt(head_dim)
            fixed_control: optional dict for fixed-shape state-prompt mode.
                When omitted and torch CUDA is available, the backend
                allocates the three int32 device scalars itself. JAX callers
                pass CudaBuffer-backed pointers plus a setter callback to keep
                this module independent of PyTorch.
            use_fa4: explicitly require FA4 for SigLIP and encoder attention.
                Construction fails if FA4 is unavailable. Decoder attention
                remains on the lower-overhead fvk kernel.

        Invariants enforced:
            * spec has the three expected sites with correct layer counts.
            * All pointer slots are non-zero.
            * ``siglip_slots['D']`` matches spec's siglip num_q_heads *
              head_dim.
        """
        super().__init__(spec)

        expected_sites = {"siglip", "encoder", "decoder"}
        got = set(spec.sites.keys())
        if got != expected_sites:
            raise ValueError(
                f"ThorFlashAttnBackend expects sites {expected_sites}, "
                f"got {got}")

        # Unwrap Python FvkContext wrapper if present; otherwise pass raw
        # fvk.FvkContext through. Both Pi0.5 (raw) and Pi0/Groot (wrapper)
        # call sites work without further conditioning in run().
        self._ctx_cpp = ctx.cpp if hasattr(ctx, "cpp") else ctx
        self._use_fa4 = bool(use_fa4)
        self._fa4_fwd = None
        if self._use_fa4:
            from flash_rt.hardware.thor import fa4_backend

            self._fa4_fwd = fa4_backend.fa4_fwd()
            if self._fa4_fwd is None:
                raise RuntimeError(
                    "Pi0.5 use_fa4=True requires an active Thor FA4 runtime: "
                    f"{fa4_backend.status()}")
        self._slots = {
            "siglip":  dict(siglip_slots),
            "encoder": dict(encoder_slots),
            "decoder": dict(decoder_slots),
        }

        # Validate required keys + non-zero ptrs.
        self._require_keys("siglip",  ("qkv", "O", "D"))
        self._require_keys("encoder",
                           ("Q_O", "Kc", "Vc", "logits",
                            "layer_stride", "scale"))
        self._require_keys("decoder",
                           ("Q_O", "Kc", "Vc", "logits",
                            "layer_stride", "scale"))

        sig = spec.site("siglip")
        exp_D = sig.num_q_heads * sig.head_dim
        if int(self._slots["siglip"]["D"]) != exp_D:
            raise ValueError(
                f"siglip_slots['D']={self._slots['siglip']['D']} does not "
                f"match spec (num_q_heads*head_dim = {exp_D})")

        # Precompute per-layer KV pointers for encoder/decoder.
        # Both sites share the same Kc/Vc base and layer_stride
        # (decoder reuses encoder's cache by design).
        self._per_layer_kv: dict[str, list[tuple[int, int]]] = {}
        for site_name in ("encoder", "decoder"):
            s = self._slots[site_name]
            nL = spec.site(site_name).num_layers
            stride = int(s["layer_stride"])
            Kc = int(s["Kc"])
            Vc = int(s["Vc"])
            self._per_layer_kv[site_name] = [
                (Kc + l * stride, Vc + l * stride) for l in range(nL)
            ]

        # Lazy-import fvk — importing at class definition time would
        # couple this module to module-load order and break tests that
        # construct the backend without a fully-initialised fvk env.
        self._fvk = None

        self._fixed_shape = False
        self._fixed_set_valid_len = None
        self.enc_seqused = None
        self.dec_seqused = None
        self.dec_devpos = None
        if fixed_control is not None:
            self.enc_seqused = fixed_control.get("enc_seqused")
            self.dec_seqused = fixed_control.get("dec_seqused")
            self.dec_devpos = fixed_control.get("dec_devpos")
            self._fixed_set_valid_len = fixed_control.get("set_valid_len")
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    self.enc_seqused = torch.zeros(
                        1, dtype=torch.int32, device="cuda")
                    self.dec_seqused = torch.zeros(
                        1, dtype=torch.int32, device="cuda")
                    self.dec_devpos = torch.zeros(
                        1, dtype=torch.int32, device="cuda")
            except Exception:
                pass
        self._enc_seqused_ptr = self._device_ptr(self.enc_seqused)
        self._dec_seqused_ptr = self._device_ptr(self.dec_seqused)
        self._dec_devpos_ptr = self._device_ptr(self.dec_devpos)

    @staticmethod
    def _device_ptr(obj) -> int:
        if obj is None:
            return 0
        if isinstance(obj, int):
            return int(obj)
        if hasattr(obj, "data_ptr"):
            return int(obj.data_ptr())
        if hasattr(obj, "ptr"):
            return int(obj.ptr.value)
        return int(obj)

    def _require_keys(self, site: str, keys: tuple[str, ...]) -> None:
        slot = self._slots[site]
        for k in keys:
            if k not in slot:
                raise ValueError(f"{site}_slots missing required key {k!r}")
            # Pointer-typed keys must be non-zero; numeric-typed keys
            # (D, layer_stride, scale) can be non-zero but may be floats.
            if k in ("qkv", "O", "Q_O", "Kc", "Vc", "logits"):
                if int(slot[k]) == 0:
                    raise ValueError(
                        f"{site}_slots[{k!r}] is a null device pointer")

    def _fvk_mod(self):
        if self._fvk is None:
            import flash_rt.flash_rt_kernels as fvk
            self._fvk = fvk
        return self._fvk

    def _fa4_stream_context(self, stream: int):
        """Run FA4 tensor operations on the caller-provided CUDA stream.

        The fvk API carries a raw ``cudaStream_t`` while FA4 follows PyTorch's
        current stream.  Bind both stream zero (the CUDA default stream) and
        explicit non-zero streams to the same stream, including any output
        copy enclosed by the caller.
        """
        stream = int(stream)
        import torch

        current = int(torch.cuda.current_stream().cuda_stream)
        if stream == 0:
            target = int(torch.cuda.default_stream().cuda_stream)
        else:
            target = stream
        if current == target:
            return nullcontext()
        if stream == 0:
            return torch.cuda.stream(torch.cuda.default_stream())
        return torch.cuda.stream(torch.cuda.ExternalStream(target))

    def set_fixed_shape(self, enabled: bool) -> None:
        """Enable/disable fixed-shape state-prompt masking.

        When enabled, encoder and decoder standard attention use
        ``attention_qkv_fp16_seqused``. The captured K/V max shape stays fixed;
        :meth:`set_fixed_valid_len` updates the device-side valid lengths.
        """
        if enabled and self._use_fa4:
            raise RuntimeError(
                "Pi0.5 use_fa4=True does not support fixed-shape state-prompt "
                "attention")
        if enabled and not (
            self._enc_seqused_ptr and self._dec_seqused_ptr
            and self._dec_devpos_ptr
        ):
            raise RuntimeError(
                "Thor fixed-shape attention requires enc_seqused, "
                "dec_seqused, and dec_devpos device scalars")
        self._fixed_shape = bool(enabled)

    def set_fixed_valid_len(self, valid_prefix_len: int) -> None:
        """Update device-side valid prefix length for fixed-shape replay."""
        v = int(valid_prefix_len)
        enc_max = self._spec.site("encoder").max_kv_seq
        dec_max = self._spec.site("decoder").max_kv_seq
        chunk = self._spec.site("decoder").max_q_seq
        if not (0 < v <= enc_max):
            raise ValueError(
                f"valid_prefix_len={v} out of range for encoder max {enc_max}")
        if v + chunk > dec_max:
            raise ValueError(
                f"decoder valid length {v + chunk} exceeds max {dec_max}")
        if self._fixed_set_valid_len is not None:
            self._fixed_set_valid_len(v, v + chunk, v)
            return
        if not (hasattr(self.enc_seqused, "fill_")
                and hasattr(self.dec_seqused, "fill_")
                and hasattr(self.dec_devpos, "fill_")):
            raise RuntimeError(
                "Thor fixed-shape scalars are pointer-only; caller must "
                "provide fixed_control['set_valid_len']")
        self.enc_seqused.fill_(v)
        self.dec_seqused.fill_(v + chunk)
        self.dec_devpos.fill_(v)
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────
    # Protocol: get_slot_ptrs
    # ────────────────────────────────────────────────────────────────
    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        """Return {Q, K, V, O} device pointer ints for (site, layer_idx).

        * siglip: Q/K/V derived from the single interleaved ``qkv``
          buffer by pointer arithmetic (byte offsets: 0, D*2, 2*D*2).
          ``layer_idx`` is ignored (all 27 layers share one scratch
          buffer — pipeline reuses it across layers).
        * encoder / decoder: Q_O is the shared attn_out buffer (Q
          input + O output alias); K/V come from the pre-computed
          per-layer KV cache offsets.
        """
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")

        if site == "siglip":
            s = self._slots[site]
            D2 = int(s["D"]) * 2  # bytes (fp16 = 2 bytes per elem)
            base = int(s["qkv"])
            return {
                "Q": base,
                "K": base + D2,
                "V": base + 2 * D2,
                "O": int(s["O"]),
            }

        # encoder / decoder
        nL = self._spec.site(site).num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} "
                f"(num_layers={nL})")
        K_ptr, V_ptr = self._per_layer_kv[site][layer_idx]
        q_o = int(self._slots[site]["Q_O"])
        return {"Q": q_o, "K": K_ptr, "V": V_ptr, "O": q_o}

    # ────────────────────────────────────────────────────────────────
    # Protocol: run
    # ────────────────────────────────────────────────────────────────
    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq: Optional[int] = None, stream: int = 0,
            state_nk: Optional[int] = None) -> int:
        """Dispatch the fvk attention kernel for (site, layer_idx).

        Returns the output device pointer int. For Thor the output
        always aliases a pipeline-owned buffer (siglip.O for vision,
        encoder/decoder Q_O for the language/decoder paths).

        Kernel selection (encoder/decoder sites) is driven by
        ``SiteSpec.extra["kernel"]``:
          * absent / ``"standard"`` → ``fvk.attention_qkv_fp16``
          * ``"state_masked"``      → ``fvk.attention_qkv_fp16_state_masked``
                                      (Pi0 decoder; requires ``state_nk``).

        This class supports only Pi0.5's fixed site set (see the module
        docstring); ``ImageWAMAttnBackend`` below implements the same
        protocol for ImageWAM's own sites.
        """
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")

        fvk = self._fvk_mod()
        site_spec = self._spec.site(site)

        if site == "siglip":
            s = self._slots[site]
            D = int(s["D"])
            nv = site_spec.batch_axis
            NH = site_spec.num_q_heads
            HD = site_spec.head_dim
            if kv_seq is None:
                kv_seq = q_seq  # self-attention
            stride = 3 * D
            Q = int(s["qkv"])
            K = Q + D * 2
            V = Q + 2 * D * 2
            if self._use_fa4:
                tensor_strides = (q_seq * stride, stride, HD, 1)
                q_tensor = _fp16_tensor_from_ptr(
                    Q, (nv, q_seq, NH, HD), tensor_strides)
                k_tensor = _fp16_tensor_from_ptr(
                    Q, (nv, q_seq, NH, HD), tensor_strides,
                    offset=NH * HD)
                v_tensor = _fp16_tensor_from_ptr(
                    Q, (nv, q_seq, NH, HD), tensor_strides,
                    offset=2 * NH * HD)
                output = _fp16_tensor_from_ptr(
                    int(s["O"]), (nv, q_seq, NH, HD))
                with self._fa4_stream_context(stream):
                    self._fa4_fwd(
                        q_tensor, k_tensor, v_tensor, causal=False,
                        num_splits=1, pack_gqa=False, out=output)
            else:
                fvk.fmha_strided_full(Q, K, V, int(s["O"]),
                                       nv, q_seq, kv_seq, NH, NH, HD,
                                       stride, stride, stream)
            return int(s["O"])

        # encoder / decoder — kernel chosen by site.extra
        nL = site_spec.num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} "
                f"(num_layers={nL})")
        s = self._slots[site]
        K_ptr, V_ptr = self._per_layer_kv[site][layer_idx]
        if kv_seq is None:
            kv_seq = q_seq  # default self-attention; decoder caller
                             # always supplies kv_seq explicitly.

        kernel = site_spec.extra.get("kernel", "standard")
        if kernel == "state_masked":
            if state_nk is None:
                raise ValueError(
                    f"site {site!r} uses state_masked kernel but no "
                    f"state_nk was provided to run()")
            if not (0 < int(state_nk) <= kv_seq):
                raise ValueError(
                    f"state_nk={state_nk} out of range (kv_seq={kv_seq})")
            fvk.attention_qkv_fp16_state_masked(
                self._ctx_cpp,
                int(s["Q_O"]), K_ptr, V_ptr,
                int(s["logits"]), int(s["Q_O"]),
                q_seq, kv_seq,
                site_spec.num_q_heads, site_spec.head_dim,
                int(state_nk),
                float(s["scale"]), stream,
            )
        elif kernel == "standard":
            if self._use_fa4 and site == "encoder":
                q_tensor = _fp16_tensor_from_ptr(
                    int(s["Q_O"]),
                    (1, q_seq, site_spec.num_q_heads, site_spec.head_dim))
                k_tensor = _fp16_tensor_from_ptr(
                    K_ptr, (1, kv_seq, 1, site_spec.head_dim))
                v_tensor = _fp16_tensor_from_ptr(
                    V_ptr, (1, kv_seq, 1, site_spec.head_dim))
                output = _fp16_tensor_from_ptr(
                    int(s["logits"]),
                    (1, q_seq, site_spec.num_q_heads, site_spec.head_dim))
                with self._fa4_stream_context(stream):
                    self._fa4_fwd(
                        q_tensor, k_tensor, v_tensor, causal=False,
                        num_splits=1, pack_gqa=True, out=output)
                    q_tensor.copy_(output)
                return int(s["Q_O"])
            if self._fixed_shape and site in ("encoder", "decoder"):
                seqused = (self._enc_seqused_ptr if site == "encoder"
                           else self._dec_seqused_ptr)
                # v2 folds the seqused mask into the softmax kernel
                # (one fewer launch per call); numerics identical.
                attn_seqused = (
                    fvk.attention_qkv_fp16_seqused_v2
                    if getattr(self, "use_fused_softmax", False)
                    else fvk.attention_qkv_fp16_seqused)
                attn_seqused(
                    self._ctx_cpp,
                    int(s["Q_O"]), K_ptr, V_ptr,
                    int(s["logits"]), int(s["Q_O"]),
                    q_seq, kv_seq,
                    site_spec.num_q_heads, site_spec.head_dim,
                    seqused,
                    float(s["scale"]), stream,
                )
            else:
                fvk.attention_qkv_fp16(
                    self._ctx_cpp,
                    int(s["Q_O"]), K_ptr, V_ptr,
                    int(s["logits"]), int(s["Q_O"]),
                    q_seq, kv_seq,
                    site_spec.num_q_heads, site_spec.head_dim,
                    float(s["scale"]), stream,
                )
        else:
            raise ValueError(
                f"unknown kernel {kernel!r} for site {site!r} "
                f"(supported: 'standard', 'state_masked')")
        return int(s["Q_O"])


# ════════════════════════════════════════════════════════════════════
# Spec builder for Pi0.5
# ════════════════════════════════════════════════════════════════════

def make_pi05_attention_spec(*, num_views: int, enc_seq_max: int,
                              chunk_size: int = 10) -> AttentionSpec:
    """Build the Pi0.5 AttentionSpec (3 sites: siglip/encoder/decoder).

    Args:
        num_views: number of camera views used by SigLIP (1, 2, or 3).
                   Stored as ``batch_axis`` on the siglip site.
        enc_seq_max: maximum encoder sequence length (prompt_len + vision
                     tokens). Depends on tokenizer max_len and view count.
        chunk_size: action-chunk length used by the decoder
                    (== decoder Q length == number of action tokens).

    The per-site dimensions are Pi0.5-specific (PaliGemma 2B + SigLIP-L):

        siglip  : 27 layers, 16 heads × 72 head_dim,  256 tokens/view
        encoder : 18 layers, 8 Q heads, 1 KV head, 256 head_dim (GQA 8)
        decoder : 18 layers, same GQA config, cross-attends encoder KV
                  cache extended by ``chunk_size`` action tokens.
    """
    spec = AttentionSpec()
    spec.add_site(
        "siglip",
        num_layers=27, num_q_heads=16, num_kv_heads=16, head_dim=72,
        max_q_seq=256, max_kv_seq=256, batch_axis=int(num_views),
    )
    spec.add_site(
        "encoder",
        num_layers=18, num_q_heads=8, num_kv_heads=1, head_dim=256,
        max_q_seq=int(enc_seq_max), max_kv_seq=int(enc_seq_max),
    )
    spec.add_site(
        "decoder",
        num_layers=18, num_q_heads=8, num_kv_heads=1, head_dim=256,
        max_q_seq=int(chunk_size),
        max_kv_seq=int(enc_seq_max) + int(chunk_size),
    )
    return spec


def make_pi0_attention_spec(*, num_views: int, enc_seq_max: int,
                             S_dec: int) -> AttentionSpec:
    """Build the Pi0 AttentionSpec (3 sites: siglip/encoder/decoder).

    Shares siglip + encoder site shapes with Pi0.5. The decoder site
    differs:
        * ``max_q_seq = S_dec = chunk_size + 1`` — Pi0 prepends a
          state token to the action chunk before running the decoder.
        * ``extra = {"kernel": "state_masked"}`` — Pi0 decoder uses
          ``fvk.attention_qkv_fp16_state_masked``; the state token
          (row 0) is forbidden from attending to action K/V
          (columns ``[enc_seq_max + 1:]``).

    Args:
        num_views: number of camera views used by SigLIP (1, 2, or 3).
        enc_seq_max: maximum encoder sequence length.
        S_dec: decoder Q length (``chunk_size + 1``; includes state token).
    """
    spec = AttentionSpec()
    spec.add_site(
        "siglip",
        num_layers=27, num_q_heads=16, num_kv_heads=16, head_dim=72,
        max_q_seq=256, max_kv_seq=256, batch_axis=int(num_views),
    )
    spec.add_site(
        "encoder",
        num_layers=18, num_q_heads=8, num_kv_heads=1, head_dim=256,
        max_q_seq=int(enc_seq_max), max_kv_seq=int(enc_seq_max),
    )
    spec.add_site(
        "decoder",
        num_layers=18, num_q_heads=8, num_kv_heads=1, head_dim=256,
        max_q_seq=int(S_dec),
        max_kv_seq=int(enc_seq_max) + int(S_dec),
        extra={"kernel": "state_masked"},
    )
    return spec


# ════════════════════════════════════════════════════════════════════
# Spec builder for ImageWAM (FLUX.2-4B variant)
# ════════════════════════════════════════════════════════════════════

def make_imagewam_attention_spec(*, max_prefix_seq: int,
                                  max_total_seq: int) -> AttentionSpec:
    """Build the ImageWAM AttentionSpec (two sites: "backbone", "mot").

    Two sites, not one -- corrected after tracing ``infer_action_flux2``
    (``imagewam.py``) closely: `self.video_expert.pre_dit(...)` runs
    ONCE, before the denoise loop, and IS the backbone's own 25-layer
    double/single-stream forward -- self-attention over just its own
    [prefix | target-image] tokens (action tokens do not exist yet at
    this point, so there is nothing to jointly attend to). Only
    `mot.prefill_flux2_video_cache`'s OUTPUT (this forward's own K/V)
    feeds the LATER `mot.forward_action_with_video_cache` calls inside
    the denoise loop, where each step's action-expert Q joins that
    cached K/V through the joint (`mot_joint`) attention. An earlier
    version of this function assumed one shared site was correct for
    both phases; it was not.

        "backbone": plain self-attention (``kernel="standard"``), used
            by ``imagewam_prefill`` (Phase 3) over the
            [prefix | target-image] sequence only.
        "mot": joint attention (``kernel="mot_joint"``), used by
            ``imagewam_denoise_step`` (Phase 4) once per action step,
            reading the "backbone" site's own K/V cache (written during
            prefill) plus this step's action-expert Q/K/V, over the
            FULL [prefix | target-image | action] sequence.

    ``num_q_heads``/``head_dim`` (24/128) are shared by the FLUX.2-4B
    backbone and ActionDiT by construction (see
    ``flash_rt/frontends/torch/_imagewam_thor_spec.py`` -- this is
    what makes ``mot_joint`` attention valid: both experts' Q/K/V land
    in the same per-head geometry after their own separate projections).

    Args:
        max_prefix_seq: upper bound on the backbone's own
            [prefix | target-image] sequence length (no action tokens).
        max_total_seq: upper bound on the full combined
            [prefix | target-image | action] sequence length the "mot"
            site will ever run with. Not yet confirmed against a real
            deployment image resolution -- see the >=1024-column
            caveat on ``softmax_mot_joint_fp16``
            (``csrc/kernels/softmax.cuh``); this must also respect
            that ceiling until a block-level (not warp-level) softmax
            variant exists.
    """
    spec = AttentionSpec()
    spec.add_site(
        "backbone",
        num_layers=25,  # backbone_num_layers_double + backbone_num_layers_single (5+20)
        num_q_heads=24, num_kv_heads=24, head_dim=128,
        max_q_seq=int(max_prefix_seq), max_kv_seq=int(max_prefix_seq),
    )
    spec.add_site(
        "mot",
        num_layers=25,
        num_q_heads=24, num_kv_heads=24, head_dim=128,
        max_q_seq=int(max_total_seq), max_kv_seq=int(max_total_seq),
        extra={"kernel": "mot_joint"},
    )
    return spec


# ════════════════════════════════════════════════════════════════════
# Backend for ImageWAM (FLUX.2-4B variant)
# ════════════════════════════════════════════════════════════════════

class ImageWAMAttnBackend(AttentionBackendBase):
    """Implements the AttentionBackend protocol for ImageWAM's two sites.

    Correction (found while starting Phase 3): an earlier draft of this
    plan added a ``mot_joint`` dispatch branch directly to
    ``ThorFlashAttnBackend.run()`` above, following
    ``docs/adding_new_model.md``'s generic instruction to "extend the
    dispatch branches" for a new kernel value. That class's own
    constructor unconditionally rejects any site set other than
    Pi0.5's fixed ``{"siglip", "encoder", "decoder"}`` (see its module
    docstring: "Currently supports Pi0.5's three sites... Pi0/GROOT out
    of scope for Stage 1") -- so a ``ThorFlashAttnBackend`` can never
    actually be constructed for ImageWAM's ``{"backbone", "mot"}``
    sites, and the added branch was unreachable dead code. Removed and
    replaced with this separate class instead of loosening Pi0.5's own
    constructor validation, to keep zero risk of a regression on the
    model this fork does not otherwise touch.

    Pipeline-owned memory model, same convention as
    ``ThorFlashAttnBackend``: the pipeline (``imagewam/pipeline_thor.py``)
    allocates every Q/K/V/O/logits buffer and passes pointers in at
    construction time; this class only dispatches kernels.

    Both sites share ONE physical per-layer K/V buffer pair (see
    ``imagewam_prefill`` / ``imagewam_denoise_step`` in
    ``flash_rt/models/imagewam/pipeline_thor.py``): rows
    ``[0, a0)`` (prefix + target-image) are written once by
    ``imagewam_prefill``'s "backbone" self-attention and never
    rewritten again; rows ``[a0, total)`` (action tokens) are
    overwritten every denoise step by that step's fresh ActionDiT K/V
    before the "mot" site's joint attention call. K/V here are a single
    shared set per position, not one set per head (see
    ``attention_qkv_fp16_mot_joint``'s own Q/K/V layout: Q is
    ``(seq*NH, HD)``, K/V are ``(seq, HD)``, broadcast across heads --
    confirmed by reading ``csrc/kernels/attention_cublas.cu`` directly,
    not assumed).
    """

    def __init__(self, spec: AttentionSpec, ctx, *,
                 backbone_slots: dict, mot_slots: dict, use_fa4: bool = False,
                 use_perhead_kv: bool = False, use_real_mot_mask: bool = False):
        """
        Args:
            spec: built by ``make_imagewam_attention_spec``.
            ctx: FvkContext (raw or ``.cpp``-wrapped, same convention
                as ``ThorFlashAttnBackend``).
            backbone_slots: {"Q_O": ptr, "K": ptr, "V": ptr,
                "logits": ptr, "scale": float}. K/V point at the same
                per-layer buffers as ``mot_slots`` -- the pipeline owns
                one KV cache, not two.
            mot_slots: same shape as ``backbone_slots``, plus
                ``"layer_stride"`` (bytes between consecutive layers'
                K/V, used to derive per-layer pointers exactly like
                ``ThorFlashAttnBackend`` does for encoder/decoder).
            use_fa4: OPT-005 (opportunities.md). Dispatch the "backbone"
                site's plain self-attention through FA4 instead of the
                cuBLAS-composed ``attention_qkv_fp16`` kernel, following
                the EXACT tensor-view/call pattern
                ``ThorFlashAttnBackend`` already uses (and has verified)
                for Pi0.5's own "encoder" site -- same
                single-shared-KV-head convention (``pack_gqa=True``), so
                no buffer-format change is needed. Does NOT change
                "mot" (the three-region masked joint attention still
                uses the custom cuBLAS kernel; FA4's plain
                causal/non-causal API has no equivalent for that mask,
                not evaluated here). Default False so every existing
                caller is unaffected; UNTESTED beyond this class's own
                construction-time checks -- this project has no
                Blackwell/Thor hardware to run FA4 on directly (see
                PROJECT.md), so this was written by mirroring the
                already-proven Pi0.5 pattern as closely as possible,
                not independently verified end-to-end.
            use_perhead_kv: OPT-002 (opportunities.md). Dispatch BOTH
                sites through the real per-head K/V kernels
                (``attention_qkv_fp16_perhead`` /
                ``attention_qkv_fp16_mot_joint_action_perhead``) instead
                of the broadcast-K/V kernels this class uses by default.
                **Requires a K/V buffer FORMAT change the caller is
                responsible for**: broadcast K/V is ``(seq, HD)``
                per layer; per-head K/V is ``(seq, NH*HD)`` = ``(seq,
                HIDDEN)`` per layer (real per-head, one independent K/V
                vector per query head, matching a real checkpoint's
                fused QKV projection width) -- passing broadcast-sized
                K/V buffers here with this flag on reads/writes out of
                bounds. Mutually compatible with ``use_fa4`` (FA4 stays
                opt-in for "backbone" only; when both are set, "backbone"
                goes through FA4 -- which already uses real per-head Q
                against a still-broadcast K/V internally via
                ``pack_gqa=True``, see the ``use_fa4`` docstring above --
                and "mot" goes through the new per-head cuBLAS kernel).
                Default False so every existing caller and buffer layout
                is unaffected; verified at the kernel level (real Thor
                dims, cosine=1.0 against a PyTorch reference and against
                this class's own existing broadcast kernel) in
                ``tests/test_imagewam_perhead_attention_kernel.py`` --
                NOT yet verified at this class's own dispatch level with
                real per-head-shaped buffers (see
                ``tests/test_imagewam_attn_backend.py``'s own
                ``use_perhead_kv`` case for that).
            use_real_mot_mask: **Bug fix, opportunities.md OPT-002/
                OPT-003 "Major correction"**. The "mot_joint" kernel this
                class dispatches to by default
                (``attention_qkv_fp16_mot_joint_action`` /
                ``_perhead``) excludes the ``[x0,a0)`` "image" region
                from action's visibility -- found, by reading ImageWAM's
                own real mask builder (``imagewam.py``'s
                ``_build_mot_attention_mask_flux2``) and its real call
                sites in ``infer_action_flux2`` directly, that this is
                WRONG for this project's real deployment target: those
                real call sites always pass ``target_len=0``, which
                means action should see the ENTIRE combined sequence
                (prefix, image, and action), not exclude the image
                region. When ``True``, "mot_joint" dispatches through
                plain, unmasked ``attention_qkv_fp16_padded`` (broadcast
                K/V, handles odd ``kv_seq`` safely) or
                ``attention_qkv_fp16_perhead`` (real per-head K/V, when
                combined with ``use_perhead_kv=True``) instead -- the
                SAME kernels the "backbone" site's "standard" dispatch
                already uses, since the real "mot" and "backbone" rules
                turned out to be the same (no mask) once
                ``target_len=0`` is accounted for. Default ``False`` so
                every existing caller, including every
                ``imagewam_thor_*_bench.py`` script and
                ``pipeline_thor.py`` itself, is completely unaffected --
                this flips a real, previously-unverified correctness
                assumption, not a drop-in-safe optimization like
                ``use_fa4``/``use_perhead_kv``, so it stays opt-in until
                validated against a real checkpoint.
        """
        super().__init__(spec)
        expected_sites = {"backbone", "mot"}
        got = set(spec.sites.keys())
        if got != expected_sites:
            raise ValueError(
                f"ImageWAMAttnBackend expects sites {expected_sites}, "
                f"got {got}")

        self._ctx_cpp = ctx.cpp if hasattr(ctx, "cpp") else ctx
        self._use_fa4 = bool(use_fa4)
        self._use_perhead_kv = bool(use_perhead_kv)
        self._use_real_mot_mask = bool(use_real_mot_mask)
        self._fa4_fwd = None
        if self._use_fa4:
            from flash_rt.hardware.thor import fa4_backend

            self._fa4_fwd = fa4_backend.fa4_fwd()
            if self._fa4_fwd is None:
                raise RuntimeError(
                    "ImageWAMAttnBackend use_fa4=True requires an active "
                    f"Thor FA4 runtime: {fa4_backend.status()}")
        self._slots = {"backbone": dict(backbone_slots), "mot": dict(mot_slots)}
        for site_name in ("backbone", "mot"):
            slot = self._slots[site_name]
            for key in ("Q_O", "K", "V", "logits"):
                if key not in slot:
                    raise ValueError(f"{site_name}_slots missing required key {key!r}")
                if int(slot[key]) == 0:
                    raise ValueError(f"{site_name}_slots[{key!r}] is a null device pointer")

        # Both sites index into the same per-layer K/V buffers -- one
        # cache, not two. layer_stride comes from "mot" (the site with
        # the full 25-layer forward that owns the buffer's lifetime).
        stride = int(mot_slots["layer_stride"])
        K_base = int(mot_slots["K"])
        V_base = int(mot_slots["V"])
        self._per_layer_kv = [
            (K_base + l * stride, V_base + l * stride)
            for l in range(spec.site("mot").num_layers)
        ]

        self._fvk = None

    def _fvk_mod(self):
        if self._fvk is None:
            import flash_rt.flash_rt_kernels as fvk
            self._fvk = fvk
        return self._fvk

    def _fa4_stream_context(self, stream: int):
        """Identical to ThorFlashAttnBackend's own helper (not shared via
        a common base -- these two classes are independent). See there
        for the docstring: FA4 follows PyTorch's current-stream
        convention while the fvk API carries a raw ``cudaStream_t``;
        this binds both to the same stream for the duration of the FA4
        call and its output copy."""
        stream = int(stream)
        import torch

        current = int(torch.cuda.current_stream().cuda_stream)
        if stream == 0:
            target = int(torch.cuda.default_stream().cuda_stream)
        else:
            target = stream
        if current == target:
            return nullcontext()
        if stream == 0:
            return torch.cuda.stream(torch.cuda.default_stream())
        return torch.cuda.stream(torch.cuda.ExternalStream(target))

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")
        nL = self._spec.site(site).num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} "
                f"(num_layers={nL})")
        K_ptr, V_ptr = self._per_layer_kv[layer_idx]
        q_o = int(self._slots[site]["Q_O"])
        return {"Q": q_o, "K": K_ptr, "V": V_ptr, "O": q_o}

    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq: Optional[int] = None, stream: int = 0,
            state_nk: Optional[int] = None,
            x0: Optional[int] = None, a0: Optional[int] = None) -> int:
        """Dispatch attention for ImageWAM's "backbone" or "mot" site.

        "backbone" (``imagewam_prefill``): plain self-attention,
        ``kernel="standard"`` -> ``fvk.attention_qkv_fp16``, ``x0``/``a0``
        ignored.

        "mot" (``imagewam_denoise_step``): joint attention,
        ``kernel="mot_joint"`` -> ``fvk.attention_qkv_fp16_mot_joint``,
        requires ``x0``/``a0`` (the block boundaries of the combined
        [prefix | target-image | action] sequence -- see
        ``csrc/kernels/attention_cublas.cuh`` for the exact visibility
        rule). ``state_nk`` is accepted for protocol-signature parity
        with ``ThorFlashAttnBackend.run`` but unused by either site.
        """
        if site not in self._slots:
            raise KeyError(f"unknown site {site!r}")

        fvk = self._fvk_mod()
        site_spec = self._spec.site(site)
        nL = site_spec.num_layers
        if not (0 <= layer_idx < nL):
            raise IndexError(
                f"layer_idx {layer_idx} out of range for site {site!r} "
                f"(num_layers={nL})")
        s = self._slots[site]
        K_ptr, V_ptr = self._per_layer_kv[layer_idx]
        if kv_seq is None:
            kv_seq = q_seq

        kernel = site_spec.extra.get("kernel", "standard")
        if kernel == "standard":
            if self._use_fa4:
                # OPT-005: same tensor-view/call pattern
                # ThorFlashAttnBackend already uses (and has verified)
                # for Pi0.5's own "encoder" site -- K/V as a single
                # shared head (pack_gqa=True lets FA4 broadcast it
                # across all NH query heads internally), matching this
                # class's own existing K/V storage convention exactly
                # (OPT-002's broadcast-K/V simplification), so no
                # buffer-format change is needed here. `logits` is
                # reused as FA4's own output scratch (FA4 cannot alias
                # its Q input as output); the result is then copied
                # into Q_O, matching Pi0.5's own pattern and this
                # class's own pointer-stability contract (callers read
                # the attention result from Q_O, not from `logits`).
                q_tensor = _fp16_tensor_from_ptr(
                    int(s["Q_O"]), (1, q_seq, site_spec.num_q_heads, site_spec.head_dim))
                k_tensor = _fp16_tensor_from_ptr(
                    K_ptr, (1, kv_seq, 1, site_spec.head_dim))
                v_tensor = _fp16_tensor_from_ptr(
                    V_ptr, (1, kv_seq, 1, site_spec.head_dim))
                output = _fp16_tensor_from_ptr(
                    int(s["logits"]), (1, q_seq, site_spec.num_q_heads, site_spec.head_dim))
                with self._fa4_stream_context(stream):
                    self._fa4_fwd(
                        q_tensor, k_tensor, v_tensor, causal=False,
                        num_splits=1, pack_gqa=True, out=output)
                    q_tensor.copy_(output)
                return int(s["Q_O"])
            # attention_qkv_fp16's own softmax reinterprets each logits
            # row as __half2 (csrc/kernels/softmax.cu::softmax_fp16_kernel)
            # with no internal even-padding (unlike mot_joint/state_masked,
            # which both compute their own *_pad = n + (n & 1)) -- an odd
            # kv_seq makes every odd-indexed row start at a 2-byte-aligned,
            # not 4-byte-aligned, address, which crashes with a CUDA
            # "misaligned address" error at kernel launch. Found by running
            # the Phase 3 wiring test with an odd a0, not by inspection.
            # Pi0.5's own "standard" sites apparently never hit this
            # because their enc_seq_max/kv_seq happen to always be even in
            # practice; ImageWAM's do not have that guarantee, so this
            # class checks explicitly instead of inheriting the same luck.
            # (FA4, above, has no such restriction -- this guard only
            # applies to the cuBLAS-composed fallback path.)
            if self._use_perhead_kv:
                # OPT-002: real per-head K/V, no even-kv_seq restriction
                # (attention_qkv_fp16_perhead's own softmax call already
                # handles odd S_kv via the same *_pad convention every
                # masked kernel in this file uses).
                fvk.attention_qkv_fp16_perhead(
                    self._ctx_cpp,
                    int(s["Q_O"]), K_ptr, V_ptr,
                    int(s["logits"]), int(s["Q_O"]),
                    q_seq, kv_seq,
                    site_spec.num_q_heads, site_spec.head_dim,
                    float(s["scale"]), stream,
                )
                return int(s["Q_O"])
            if kv_seq % 2 != 0:
                raise ValueError(
                    f"site {site!r} kernel='standard' requires an even "
                    f"kv_seq (attention_qkv_fp16's softmax is __half2-"
                    f"vectorized per row with no internal padding); got "
                    f"kv_seq={kv_seq}")
            fvk.attention_qkv_fp16(
                self._ctx_cpp,
                int(s["Q_O"]), K_ptr, V_ptr,
                int(s["logits"]), int(s["Q_O"]),
                q_seq, kv_seq,
                site_spec.num_q_heads, site_spec.head_dim,
                float(s["scale"]), stream,
            )
        elif kernel == "mot_joint":
            # OPT-003 fix (opportunities.md): Q covers only the action
            # rows (q_seq = num_action), not the whole combined sequence
            # -- during the real denoise loop the prefix/image rows never
            # have a live query (their own Q was consumed once during
            # prefill and never read again), so computing attention for
            # them was pure waste, confirmed on real Thor hardware to
            # leave the denoise step's cost flat across every precision
            # tested. K/V still cover the whole combined sequence
            # (kv_seq = total), since action rows attend into the frozen
            # prefix K/V. Q/out live at row-offset `a0` within the shared
            # Q_O buffer (computed here from `a0`, which every mot_joint
            # caller already supplies for the mask) -- not at row 0 --
            # since the pipeline writes the action rows' fresh Q there,
            # not at the buffer's base.
            if x0 is None or a0 is None:
                raise ValueError(
                    f"site {site!r} uses mot_joint kernel but x0/a0 "
                    f"block boundaries were not provided to run()")
            if not (0 < int(x0) <= int(a0) <= kv_seq):
                raise ValueError(
                    f"invalid block boundaries x0={x0}, a0={a0} for "
                    f"combined sequence length kv_seq={kv_seq} (require "
                    f"0 < x0 <= a0 <= kv_seq)")
            num_action = q_seq
            row_width = site_spec.num_q_heads * site_spec.head_dim
            q_out_ptr = int(s["Q_O"]) + int(a0) * row_width * 2  # fp16 = 2 bytes/elem
            if self._use_real_mot_mask:
                # Bug fix (opportunities.md OPT-002/OPT-003): the real
                # mask needs NO exclusion at all -- see this flag's own
                # docstring. Dispatch through the same plain kernels
                # "backbone" already uses.
                if self._use_perhead_kv:
                    fvk.attention_qkv_fp16_perhead(
                        self._ctx_cpp,
                        q_out_ptr, K_ptr, V_ptr,
                        int(s["logits"]), q_out_ptr,
                        num_action, kv_seq, site_spec.num_q_heads, site_spec.head_dim,
                        float(s["scale"]), stream,
                    )
                else:
                    fvk.attention_qkv_fp16_padded(
                        self._ctx_cpp,
                        q_out_ptr, K_ptr, V_ptr,
                        int(s["logits"]), q_out_ptr,
                        num_action, kv_seq, site_spec.num_q_heads, site_spec.head_dim,
                        float(s["scale"]), stream,
                    )
                return q_out_ptr
            mot_kernel = (fvk.attention_qkv_fp16_mot_joint_action_perhead
                          if self._use_perhead_kv else fvk.attention_qkv_fp16_mot_joint_action)
            mot_kernel(
                self._ctx_cpp,
                q_out_ptr, K_ptr, V_ptr,
                int(s["logits"]), q_out_ptr,
                num_action, kv_seq, site_spec.num_q_heads, site_spec.head_dim,
                int(x0), int(a0),
                float(s["scale"]), stream,
            )
            return q_out_ptr
        else:
            raise ValueError(
                f"unknown kernel {kernel!r} for site {site!r} "
                f"(supported: 'standard', 'mot_joint')")
        return int(s["Q_O"])
