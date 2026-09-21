"""`text_trim=True` and the consumers of a captured frontend's graphs.

A trimmed frontend runs one graph per prompt length, so the runtime surface
carries the variant table keyed by that length (`GraphVariants`) and each
consumer carries one graph per key, `step` replaying the length the prompt
set (plan.md phase S2 for the ABI face, S4 for the native one):

- `runtime_surface()` serves a trimmed frontend: every field describes the
  active length's graph (`context_rows = active_dims["x0"]`) and
  `graph_variants` names the exec of every captured length;
- `export_model_runtime(io="python")` adopts every captured length and
  declares the table (the manifest's `text_lengths`);
- `ImageWAMNativeRuntime.create` hands the same table to the native handle
  (`frt_imagewam_io_config.text_lengths`); `use_graph(key, exec)` adopts one
  exec per captured length, `set_text_length(key)` — legal while a model
  runtime over the handle is live, unlike adoption — selects the length the
  C `step` replays, and `export_model_runtime(io="native")` adopts the
  handle's table and refuses a captured length the handle does not hold;
- `pipeline_resources()` describes the ACTIVE length and the handle installs
  one pipeline (and captures one graph) per length
  (`ImageWAMNativeRuntime.capture_pipeline_text_lengths`), so the native
  C++ pipeline `step` replays serves the trim like the other two consumers.

`graph_variant_plan` and `uncaptured_text_length_message` are the pure part
of that contract ("which keys and which default key does this table
declare", "what does `step` say about a length with no variant"); the io
config's length table, the handle's key plumbing, the per-length resource
table and the install loop are checked here against stubs of the native
library and of the frontend. Those are covered without a GPU; everything
that captures a real frontend needs one.
"""
from __future__ import annotations

import ctypes
import dataclasses
import json

import numpy as np
import pytest
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.native_library import (
    ImageWAMGemmShape,
    ImageWAMIoConfig,
    ImageWAMPipelineConfig,
)
from flash_rt.models.imagewam.native_resources import build_io_config
from flash_rt.models.imagewam.native_runtime import ImageWAMNativeError, ImageWAMNativeRuntime
from flash_rt.models.imagewam.pipeline_resources import (
    ImageWAMPipelineResources,
    PipelineBuffers,
)
from flash_rt.models.imagewam.quant_linear import Bf16OutLinear, Fp16Linear
from flash_rt.models.imagewam.runtime_surface import (
    GraphVariants,
    ImageWAMRuntimeSurface,
    TextLengthGraph,
    graph_variant_plan,
    uncaptured_text_length_message,
)
from flash_rt.models.imagewam.text_context import trimmed_sequence_dims

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TEXT_LEN = 16
DIMS = dict(x0=TEXT_LEN + 1, a0=TEXT_LEN + 1 + 10, total=TEXT_LEN + 1 + 10 + 4,
            joint_attention_dim=16, proprio_dim=3)
SHORT_VALID = 5
LONG_VALID = 9
SHORT_KEY, LONG_KEY = SHORT_VALID + 1, LONG_VALID + 1        # x0 = 6 and 10
SHORT_EXEC, LONG_EXEC = 0xA11CE, 0xB0B


# -- the pure seam: the table -> the declaration --------------------------

def test_graph_variant_plan_keys_every_captured_length_and_defaults_to_the_active_one():
    plan = graph_variant_plan(GraphVariants(
        active_key=45,
        entries=(TextLengthGraph(key=45, graph_exec=0xbeef),
                 TextLengthGraph(key=30, graph_exec=0xaaaa)),
        per_prompt_length=True))
    assert plan.keys == (30, 45), "keys ascending, one per captured length"
    assert plan.default_key == 45, "the length active at export"
    assert plan.max_variants == 2, "the table's own size: the export never captures"


def test_graph_variant_plan_refuses_a_table_it_cannot_declare():
    with pytest.raises(ValueError, match="at least one graph variant"):
        graph_variant_plan(GraphVariants(active_key=0, entries=(), per_prompt_length=False))
    with pytest.raises(ValueError, match="keys repeat"):
        graph_variant_plan(GraphVariants(
            active_key=30,
            entries=(TextLengthGraph(key=30, graph_exec=1), TextLengthGraph(key=30, graph_exec=2)),
            per_prompt_length=True))
    with pytest.raises(ValueError, match="active key 9 is not in the graph variant table"):
        graph_variant_plan(GraphVariants(
            active_key=9, entries=(TextLengthGraph(key=30, graph_exec=1),),
            per_prompt_length=True))
    # text_trim=False is the one-entry case
    plan = graph_variant_plan(GraphVariants(
        active_key=513, entries=(TextLengthGraph(key=513, graph_exec=7),),
        per_prompt_length=False))
    assert plan.keys == (513,) and plan.default_key == 513 and plan.max_variants == 1


def test_uncaptured_text_length_message_names_the_length_and_the_adopted_table():
    message = uncaptured_text_length_message(77, (30, 45))
    print(message)
    assert "x0=77" in message and "[30, 45]" in message
    assert "precapture_text_lengths" in message


# -- the native handle: the key table, without a GPU -----------------------
#
# `build_io_config` and `ImageWAMNativeRuntime` are checked against a stub of
# the native library: the config struct is the real ctypes mirror, so the
# key table it hands over is what the C++ side reads, and the stub's returns
# mirror what the C handle answers for a handle that holds exactly what
# Python adopted.

def _cpu_surface(keys: tuple[int, ...]) -> ImageWAMRuntimeSurface:
    """A `text_trim=True` surface over CPU tensors, with one graph variant
    per `keys` entry and the last length active.

    Only the fields the io config and the surface's own contract name are
    filled; `stream` is None because nothing here replays (`build_io_config`
    and the handle's key calls never touch it)."""
    x0_max = keys[-1]
    entries = tuple(TextLengthGraph(key=key, graph_exec=0x100 + key) for key in keys)
    variants = GraphVariants(active_key=x0_max, entries=entries, per_prompt_length=True)
    return ImageWAMRuntimeSurface(
        graph_exec=0x100 + x0_max, graph_variants=variants, stream=None,
        img_raw=torch.zeros((4, 8), dtype=torch.bfloat16),
        context=torch.zeros((x0_max, 16), dtype=torch.bfloat16),
        action_latent=torch.zeros((4, 7), dtype=torch.float32),
        img_len=4, token_dim=8, num_action=4, action_dim=7, proprio_dim=3,
        has_vae=False, has_text_encoder=False, action_denormalized=False,
        setup_identity=(("text_trim", "True"),), context_rows=x0_max, context_width=16,
        proprio_row=x0_max - 1,
        proprio_weight=torch.zeros((16, 3), dtype=torch.bfloat16),
        proprio_bias=torch.zeros(16, dtype=torch.bfloat16),
        state_scale=None, state_offset=None, action_scale=None, action_offset=None,
        view_shape=(2, 224, 224), views_u8=None, owner=object())


class _StubNativeLib:
    """`ImageWAMNativeLibrary` stand-in: the calls `ImageWAMNativeRuntime`
    makes, recorded, with the answers a handle that adopted `use_graph`'s
    execs gives.

    The stub is its own `.lib` because the runtime only reaches the C
    functions through `library.lib`. `created_config` keeps the struct that
    was handed to `frt_imagewam_native_create`, so a test can read the key
    table back out of the real ctypes layout; the pipeline calls mirror the C
    handle's own rules (a pipeline may only be installed for a declared
    length, `set_pipeline` installs for the key it carries and selects it,
    `capture` records a graph for the active key, `gemm_shapes` describes the
    installed pipeline, whose context GEMM rows are that key)."""

    def __init__(self) -> None:
        self.lib = self
        self.calls: list[tuple] = []
        self.created_config: ImageWAMIoConfig | None = None
        self.handle = 0x1000
        self.variants: dict[int, int] = {}
        self.declared: tuple[int, ...] = ()
        self.active = 0
        self.pipelines: dict[int, int] = {}

    def frt_imagewam_native_create(self, config, out):
        # The runtime passes `byref` objects: cast them to the ctypes
        # mirrors, so the stub reads the real layout.
        c = ctypes.cast(config, ctypes.POINTER(ImageWAMIoConfig)).contents
        self.created_config = c
        self.declared = tuple(int(c.text_lengths[i]) for i in range(c.num_text_lengths))
        self.active = int(c.context_rows)
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p)).contents.value = self.handle
        self.calls.append(("create", self.declared, self.active))
        return 0

    def frt_imagewam_native_last_error(self, handle):
        return b""

    def frt_imagewam_native_release(self, handle):
        self.calls.append(("release", handle))

    def frt_imagewam_native_set_proprio_row(self, handle, row):
        self.calls.append(("set_proprio_row", handle, int(row)))
        return 0

    def frt_imagewam_native_use_graph(self, handle, key, graph_exec):
        if int(key) not in self.declared:
            return -2
        self.variants[int(key)] = int(graph_exec)
        self.calls.append(("use_graph", handle, int(key), int(graph_exec)))
        return 0

    def frt_imagewam_native_has_variant(self, handle, key):
        return 1 if int(key) in self.variants else 0

    def frt_imagewam_native_variant_exec(self, handle, key):
        return self.variants.get(int(key), 0)

    def frt_imagewam_native_set_text_length(self, handle, key):
        self.calls.append(("set_text_length", handle, int(key)))
        if int(key) not in self.variants:
            return -2
        self.active = int(key)
        return 0

    def frt_imagewam_native_text_length(self, handle):
        return self.active

    def frt_imagewam_native_graph_exec(self, handle):
        return self.variants.get(self.active, 0)

    # -- the native pipeline's own capture path ---------------------------

    def frt_imagewam_native_set_pipeline(self, handle, config):
        x0 = int(ctypes.cast(config, ctypes.POINTER(ImageWAMPipelineConfig)).contents.x0)
        if x0 not in self.declared:
            return -2
        # One pipeline per key: installing a key again replaces its own
        # pipeline and drops the graph THIS handle captured for it.
        if self.variants.get(x0, 0) == self._captured_exec(x0):
            del self.variants[x0]
        self.pipelines[x0] = x0
        self.active = x0
        self.calls.append(("set_pipeline", handle, x0))
        return 0

    @staticmethod
    def _captured_exec(key: int) -> int:
        return 0xC0FFEE + key

    def frt_imagewam_native_gemm_shapes(self, handle, out, capacity, count):
        # One context GEMM shape per pipeline, with the key's own row count:
        # the shapes a length adds are what the hand-off is installed for.
        shapes = [(0, self.active, 8, 8)]
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint64)).contents.value = len(shapes)
        if out:
            for i, (kind, m, n, k) in enumerate(shapes):
                out[i] = ImageWAMGemmShape(kind=kind, m=m, n=n, k=k)
        self.calls.append(("gemm_shapes", handle, None if not out else tuple(shapes)))
        return 0

    def frt_imagewam_native_set_gemm_algo(self, handle, shape, algo, bytes_):
        s = ctypes.cast(shape, ctypes.POINTER(ImageWAMGemmShape)).contents
        self.calls.append(("set_gemm_algo", handle, int(s.m), len(algo)))
        return 0

    def frt_imagewam_native_capture(self, handle):
        self.variants[self.active] = self._captured_exec(self.active)
        self.calls.append(("capture", handle, self.active))
        return 0

    def frt_imagewam_native_graph_nodes(self, handle, count):
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint64)).contents.value = 0
        return 0


def test_io_config_declares_the_surface_length_table():
    """`frt_imagewam_io_config` carries one entry per captured text length
    (the keys of the same `GraphVariantPlan` the declaration adopts), with
    the active length as `context_rows`."""
    surface = _cpu_surface((SHORT_KEY, LONG_KEY))
    handoff = build_io_config(surface)
    c = handoff.config
    table = np.ctypeslib.as_array(c.text_lengths, shape=(c.num_text_lengths,)).tolist()
    print(f"io config: context_rows={c.context_rows} text_lengths={table} "
          f"({len(handoff.keepalive)} keepalive entries)")
    assert c.num_text_lengths == 2 and table == [SHORT_KEY, LONG_KEY]
    assert c.context_rows == LONG_KEY == surface.context_rows
    assert any(getattr(entry, "dtype", None) == np.uint32 for entry in handoff.keepalive), \
        "the host array behind text_lengths must be kept alive by the handoff"

    # text_trim=False is the one-entry table, and the key is dims["x0"].
    plain = dataclasses.replace(
        surface, graph_variants=GraphVariants(
            active_key=513, per_prompt_length=False,
            entries=(TextLengthGraph(key=513, graph_exec=7),)),
        context_rows=513, proprio_row=512, graph_exec=7)
    one = build_io_config(plain).config
    assert one.num_text_lengths == 1 and one.context_rows == 513
    assert np.ctypeslib.as_array(one.text_lengths, shape=(1,)).tolist() == [513]


def test_native_handle_adopts_and_selects_one_graph_per_captured_length():
    """`create` accepts a trimmed surface, `use_graph(key, exec)` adopts each
    captured length under its own key, and `set_text_length` selects what
    `graph_exec` reports: the handle's key space is the surface's table."""
    surface = _cpu_surface((SHORT_KEY, LONG_KEY))
    library = _StubNativeLib()
    native = ImageWAMNativeRuntime.create(surface, library)
    print(f"stub calls: {library.calls}")
    assert library.declared == (SHORT_KEY, LONG_KEY), "the config declares the surface's table"
    assert native.text_length == LONG_KEY, "the active length is the surface's active length"
    assert native.graph_exec == 0, "no graph adopted yet"

    native.use_graph(SHORT_KEY, SHORT_EXEC)
    native.use_graph(LONG_KEY, LONG_EXEC)
    assert native.has_variant(SHORT_KEY) and native.has_variant(LONG_KEY)
    assert not native.has_variant(SHORT_KEY + 100)
    assert native.variant_exec(SHORT_KEY) == SHORT_EXEC
    assert native.variant_exec(LONG_KEY) == LONG_EXEC == native.graph_exec

    # A shorter length: the exec, and with it what `step` replays, follows
    # the key the prompt set.
    native.set_text_length(SHORT_KEY)
    assert native.text_length == SHORT_KEY
    assert native.graph_exec == SHORT_EXEC
    assert ("set_text_length", library.handle, SHORT_KEY) in library.calls

    # A length the handle holds no graph for is refused by the C side (-2).
    with pytest.raises(ImageWAMNativeError) as exc:
        native.set_text_length(LONG_KEY + 1000)
    print(f"set_text_length(undeclared): {exc.value}")
    assert exc.value.status == -2
    native.close()


# -- the native pipeline's per-length path, without a GPU ------------------
#
# `pipeline_resources()` is the frontend's own method, bound to a stub
# instance: the tensors, the weight ops and `_bufs` are CPU stand-ins, the
# method and the resource dataclasses are the real ones. What that reaches
# without a GPU is the per-length resource table and the install loop: the
# table describes the ACTIVE length (sequence dims, AdaLN row counts, the
# backbone RoPE table) while the pipeline buffers stay the ones allocated for
# the longest length, and `capture_pipeline_text_lengths` walks the lengths
# the source has captured, installing and capturing one pipeline each.

MAX_DIMS = dict(x0=TEXT_LEN + 1, a0=TEXT_LEN + 1 + 10, total=TEXT_LEN + 1 + 10 + 4,
                hidden=32, HD=128, NH=2, mlp_hidden=16, joint_attention_dim=16, num_action=4,
                action_dim=7, action_hidden_dim=8, action_attn_width=32, action_mlp_hidden=16,
                num_layers_double=1, num_layers_single=1, action_num_layers_double=1,
                action_num_layers_single=1, num_denoise_steps=1, dt=0.1, ref_h=2, ref_w=5,
                merge_qkv_mlp=True, merge_linear2=True, fuse_res_norm=True, proprio_dim=3)


class _CpuPipelineFrontend:
    """`ImageWAMTextLengthPipelineSource` over CPU stand-ins: one active
    length at a time, `_activate_text_length` selecting it (as the frontend
    does), and the real `pipeline_resources()` bound to this instance.

    The stand-in carries what that method reads: `_active_dims` (the max dims
    with `x0`/`a0`/`total` of the active length), one backbone RoPE table per
    length (the longest keeps the max-size one, as `_capture_text_length`
    does), the device windows (`data_ptr()` of CPU tensors), one `_bufs`
    entry per pipeline buffer (to be read as the max-size allocation), the
    modulation chunks and the weight ops the table describes
    (`Fp16Linear` / `Bf16OutLinear`, the same classes the real weights are)."""

    def __init__(self, max_dims: dict, lengths: tuple[int, ...]) -> None:
        self.dims = dict(max_dims)
        self._lengths = tuple(lengths)
        # `_graph` is None until a prompt is set, as in the frontend.
        self._graph = None
        # The native path's own contract (rule R6): no FA4 attention, so
        # `pipeline_resources()` is legal on this stub.
        self.use_fa4 = False
        self.use_fa4_mot = False
        self._vae_stage = None
        self._nvfp4_awq = False
        hidden, ahd = self.dims["hidden"], self.dims["action_hidden_dim"]
        self._bufs = {name: 0xB0000000 + 0x100 * i
                      for i, name in enumerate(PipelineBuffers.__dataclass_fields__)}
        # One distinguishable table per length: the longest is the max one.
        self._rope_tables = {x0: torch.full((8,), float(x0), dtype=torch.float16) for x0 in lengths}
        self._active_dims = self.dims
        self._rope_table = torch.zeros(8, dtype=torch.float16)
        self._action_rope_table = torch.zeros(8, dtype=torch.float16)
        self._Q_O = torch.zeros(8, dtype=torch.float16)
        self._K_cache = torch.zeros(4, 8, dtype=torch.float16)
        self._V_cache = torch.zeros(4, 8, dtype=torch.float16)
        self._logits = torch.zeros(4, 8, dtype=torch.float32)
        mod = lambda d: torch.zeros((1, 1, d), dtype=torch.float32)
        backbone_group = (mod(hidden), mod(hidden), mod(hidden))
        # `_mod_txt` / `_mod_img`: one (shift, scale, gate) group per site,
        # `_mod_single` one group; `_head_mods[step]` is a gate-less pair.
        self._mod_txt = (backbone_group, backbone_group)
        self._mod_img = (backbone_group, backbone_group)
        self._mod_single = backbone_group
        # one denoise step: `_action_mods[step]` is (double1, double2), single
        action_group = (mod(ahd), mod(ahd), mod(ahd))
        self._action_mods = [((action_group, action_group), action_group)]
        self._head_mods = [(mod(ahd), mod(ahd))]
        self._deltas = None
        self._weights = self._weight_ops()
        self.activated: list[int] = []
        if lengths:
            self._graph = object()                # a prompt was captured
            self._activate_text_length(max(lengths))

    def _weight_ops(self) -> dict:
        """One op per `_weights` key the resource table reads, in the key
        scheme the frontend uses (`(stack, block, layer, slot)`)."""
        d = self.dims
        hidden, ahd = d["hidden"], d["action_hidden_dim"]
        w: dict = {}

        def ptr() -> int:
            return 0x100000 + 0x10 * len(w)

        for stack, blocks, slots, norms in (
                ("backbone", "double", ("txt_qkv", "img_qkv", "txt_proj", "img_proj",
                                        "txt_mlp0", "img_mlp0", "txt_mlp2", "img_mlp2"),
                 ("txt_query_norm", "txt_key_norm", "img_query_norm", "img_key_norm")),
                ("action_dit", "double", ("qkv", "proj", "mlp0", "mlp2"),
                 ("query_norm", "key_norm"))):
            for layer in range(2):
                for slot in slots:
                    w[(stack, blocks, layer, f"{slot}.weight")] = Fp16Linear(None, ptr(), hidden, hidden)
                for slot in norms:
                    w[(stack, blocks, layer, slot)] = ptr()
        for stack in ("backbone", "action_dit"):
            for layer in range(2):
                w[(stack, "single", layer, "linear1.weight")] = Fp16Linear(None, ptr(), hidden, hidden)
                w[(stack, "single", layer, "linear2.weight")] = Fp16Linear(None, ptr(), hidden, hidden)
                for slot in ("query_norm", "key_norm"):
                    w[(stack, "single", layer, slot)] = ptr()
        w[("action_dit", "shared", 0, "action_encoder.weight")] = Fp16Linear(None, ptr(), ahd, d["action_dim"])
        w[("action_dit", "shared", 0, "head.linear.weight")] = Fp16Linear(None, ptr(), ahd, ahd)
        w[("action_dit", "shared", 0, "action_encoder.bias")] = ptr()
        w[("backbone", "double", 0, "txt_in.weight")] = Bf16OutLinear(
            None, ptr(), hidden, d["joint_attention_dim"])
        w[("backbone", "double", 0, "img_in.weight")] = Bf16OutLinear(None, ptr(), hidden, 128)
        return w

    # -- `ImageWAMTextLengthPipelineSource` ------------------------------

    def _activate_text_length(self, x0: int) -> None:
        """The frontend's own rule: the sequence dims shrink by the dropped
        rows and the longest length keeps the max-size RoPE table."""
        self._active_dims = trimmed_sequence_dims(self.dims, int(x0))
        self._rope_table = self._rope_tables[int(x0)]
        self.activated.append(int(x0))

    @property
    def active_dims(self) -> dict:
        return dict(self._active_dims)

    @property
    def captured_text_lengths(self) -> tuple[int, ...]:
        """The frontend's own form: a property, not a method
        (`ImageWAMTextLengthPipelineSource`)."""
        return tuple(sorted(self._lengths))

    def pipeline_resources(self) -> ImageWAMPipelineResources:
        """The frontend's own method, bound to this stub instance."""
        return ImageWAMTorchFrontendThor.pipeline_resources(self)

    def gemm_algo(self, kind: int, m: int, n: int, k: int) -> bytes:
        # One planned algorithm per shape: what the hand-off installs.
        return bytes([kind, m, n, k]) + bytes(60)


def test_pipeline_resources_describes_the_active_text_length():
    """One resource table describes the active length: the sequence dims, the
    AdaLN row counts and the backbone RoPE table follow it, while the pipeline
    buffers are the same maximal ones. That is what makes a trimmed frontend
    installable (`capture_pipeline_text_lengths` takes one table per length)."""
    source = _CpuPipelineFrontend(MAX_DIMS, (SHORT_KEY, LONG_KEY))
    source._activate_text_length(SHORT_KEY)
    short = source.pipeline_resources()
    source._activate_text_length(LONG_KEY)
    long = source.pipeline_resources()
    drop = MAX_DIMS["x0"] - SHORT_KEY
    print(f"x0={SHORT_KEY}: dims={(short.dims.x0, short.dims.a0, short.dims.total)} "
          f"rope={short.attention.rope_table:#x} txt_mod1.gate={tuple(short.txt_mod1.gate.shape)}")
    print(f"x0={LONG_KEY}: dims={(long.dims.x0, long.dims.a0, long.dims.total)} "
          f"rope={long.attention.rope_table:#x} txt_mod1.gate={tuple(long.txt_mod1.gate.shape)} "
          f"buffers identical={long.buffers == short.buffers}")

    assert (short.dims.x0, short.dims.a0, short.dims.total) == (SHORT_KEY, MAX_DIMS["a0"] - drop,
                                                                MAX_DIMS["total"] - drop)
    assert (long.dims.x0, long.dims.a0, long.dims.total) == (LONG_KEY, MAX_DIMS["a0"] - MAX_DIMS["x0"] + LONG_KEY,
                                                             MAX_DIMS["total"] - MAX_DIMS["x0"] + LONG_KEY)
    # the dims the buffers were allocated for are not the ones the table describes
    assert short.dims.x0 < MAX_DIMS["x0"] and long.dims.x0 < MAX_DIMS["x0"]
    assert (short.dims.hidden, short.dims.num_action, short.dims.num_steps) == \
           (MAX_DIMS["hidden"], MAX_DIMS["num_action"], MAX_DIMS["num_denoise_steps"])

    # the backbone RoPE table is the active length's
    assert short.attention.rope_table == source._rope_tables[SHORT_KEY].data_ptr()
    assert long.attention.rope_table == source._rope_tables[LONG_KEY].data_ptr() \
        != short.attention.rope_table
    # the AdaLN row counts follow the active length (`txt_mod1` over x0,
    # `single_mod` over a0): the kernels re-normalize the rows the graph reads
    assert short.txt_mod1.gate.shape == (SHORT_KEY, MAX_DIMS["hidden"])
    assert long.txt_mod1.gate.shape == (LONG_KEY, MAX_DIMS["hidden"])
    assert short.single_mod.gate.shape[0] == short.dims.a0
    assert long.single_mod.gate.shape[0] == long.dims.a0

    # the buffers are one set, sized for the longest length: identical for
    # both tables, and the same pointers the frontend allocated
    assert long.buffers == short.buffers == PipelineBuffers(**source._bufs)
    assert short.attention.q_o == source._Q_O.data_ptr() == long.attention.q_o
    assert short.attention.action_rope_table == source._action_rope_table.data_ptr() \
        == long.attention.action_rope_table
    assert short.attention.kv_layer_stride_bytes == source._K_cache[0].numel() * 2

    # text_trim=False: the one length is dims, so the table is the max one
    plain = _CpuPipelineFrontend(MAX_DIMS, (MAX_DIMS["x0"],))
    untrimmed = plain.pipeline_resources()
    assert (untrimmed.dims.x0, untrimmed.dims.a0, untrimmed.dims.total) == \
           (MAX_DIMS["x0"], MAX_DIMS["a0"], MAX_DIMS["total"])
    assert untrimmed.attention.rope_table == plain._rope_tables[MAX_DIMS["x0"]].data_ptr()
    assert untrimmed.buffers == short.buffers


def test_capture_pipeline_text_lengths_installs_and_captures_every_length():
    """`capture_pipeline_text_lengths` walks the source's captured lengths
    ascending and installs + captures the native pipeline for each
    (`set_pipeline`, whose key is that length, then `capture`), handing off
    the GEMM algorithm of each pipeline's own shapes; the source's active
    length is what it was before the call and the handle serves that key
    afterwards."""
    dims = dict(MAX_DIMS, x0=LONG_KEY, a0=LONG_KEY + 10, total=LONG_KEY + 14)
    source = _CpuPipelineFrontend(dims, (SHORT_KEY, LONG_KEY))
    source._activate_text_length(SHORT_KEY)        # the prompt the deployment serves
    library = _StubNativeLib()
    native = ImageWAMNativeRuntime.create(_cpu_surface((SHORT_KEY, LONG_KEY)), library)
    try:
        keys = native.capture_pipeline_text_lengths(source)
        print(f"source activations: {source.activated}")
        print(f"handle calls: {library.calls}")
        assert keys == (SHORT_KEY, LONG_KEY)
        assert native.has_variant(SHORT_KEY) and native.has_variant(LONG_KEY)
        assert native.text_length == SHORT_KEY, "the handle serves the length the source is on"
        assert source.active_dims["x0"] == SHORT_KEY
        assert native.graph_producer == "native"
        # one install + one capture per length, in ascending order, and the
        # GEMM hand-off of the shapes that length's pipeline launches
        assert library.calls == [
            ("create", (SHORT_KEY, LONG_KEY), LONG_KEY),
            ("set_proprio_row", library.handle, LONG_KEY - 1),
            ("set_pipeline", library.handle, SHORT_KEY),
            ("gemm_shapes", library.handle, None),
            ("gemm_shapes", library.handle, ((0, SHORT_KEY, 8, 8),)),
            ("set_gemm_algo", library.handle, SHORT_KEY, 64),
            ("capture", library.handle, SHORT_KEY),
            ("set_pipeline", library.handle, LONG_KEY),
            ("gemm_shapes", library.handle, None),
            ("gemm_shapes", library.handle, ((0, LONG_KEY, 8, 8),)),
            ("set_gemm_algo", library.handle, LONG_KEY, 64),
            ("capture", library.handle, LONG_KEY),
            ("set_text_length", library.handle, SHORT_KEY),
        ]
        # the reported shapes are the pipeline installed last (the longest)
        assert native.gemm_shapes == [(0, LONG_KEY, 8, 8)] and native.gemm_algos_installed == 1
        # the restored length is the one the surface declares as active
        assert native.graph_exec == library.variants[SHORT_KEY]
    finally:
        native.close()

    # A source with no capture is refused before any install.
    empty = _CpuPipelineFrontend(dims, ())
    unused = _StubNativeLib()
    native = ImageWAMNativeRuntime.create(_cpu_surface((SHORT_KEY, LONG_KEY)), unused)
    try:
        with pytest.raises(ValueError, match="captured no text length"):
            native.capture_pipeline_text_lengths(empty)
        print(f"calls after the refused call: {unused.calls}")
        assert not any(call[0] == "set_pipeline" for call in unused.calls)
    finally:
        native.close()


# -- the surface and the export -------------------------------------------

def _frontend(text_trim: bool, valid: int = SHORT_VALID) -> ImageWAMTorchFrontendThor:
    """`use_fa4=False`: the rows this file checks over a real frontend are
    the native consumer's own table (`pipeline_resources()`, the surface the
    `io="native"` face adopts), and the native pipeline has no FA4 attention
    (rule R6), so the attention path is not part of them and
    `FLASHRT_THOR_FA4` must not decide whether they run."""
    torch.manual_seed(0)
    fe = ImageWAMTorchFrontendThor(precision="fp16", use_fa4=False, dims_override=dict(DIMS),
                                   text_trim=text_trim)
    mask = torch.zeros(TEXT_LEN, dtype=torch.bool)
    mask[:valid] = True
    fe.set_prompt(context=torch.randn(TEXT_LEN, DIMS["joint_attention_dim"]).to(torch.bfloat16),
                  context_mask=mask)
    return fe


@needs_cuda
def test_runtime_surface_carries_one_graph_per_captured_length():
    fe = _frontend(True, SHORT_VALID)
    first = fe.runtime_surface()
    short_x0 = SHORT_VALID + 1  # valid text rows + the proprio row
    print(f"trimmed surface: context_rows={first.context_rows} active_key="
          f"{first.graph_variants.active_key} keys={[e.key for e in first.graph_variants.entries]}")
    assert first.context_rows == short_x0, "the surface describes the active length, not the buffer size"
    assert first.graph_variants.active_key == short_x0
    assert first.graph_variants.per_prompt_length is True
    assert dict(first.setup_identity)["text_trim"] == "True"
    assert dict(first.setup_identity)["dims.x0"] == str(short_x0)

    # A second length: the table grows, the surface turns to it, and the
    # first length's exec stays declared.
    fe.set_prompt(context=torch.randn(TEXT_LEN, DIMS["joint_attention_dim"]).to(torch.bfloat16),
                  context_mask=torch.cat([torch.ones(LONG_VALID, dtype=torch.bool),
                                          torch.zeros(TEXT_LEN - LONG_VALID, dtype=torch.bool)]))
    long_x0 = LONG_VALID + 1
    second = fe.runtime_surface()
    assert (second.context_rows, second.graph_variants.active_key) == (long_x0, long_x0)
    assert tuple(e.key for e in second.graph_variants.entries) == (short_x0, long_x0)
    execs = {e.key: e.graph_exec for e in second.graph_variants.entries}
    print(f"variants: {execs}")
    assert execs[long_x0] == second.graph_exec, "graph_exec is the active key's entry"
    assert execs[short_x0] != execs[long_x0], "one exec per length"

    # text_trim=False: one graph, every prompt, keyed by dims["x0"].
    plain = _frontend(False, LONG_VALID).runtime_surface()
    assert plain.context_rows == DIMS["x0"] and plain.graph_variants.active_key == DIMS["x0"]
    assert plain.graph_variants.per_prompt_length is False
    assert tuple(e.key for e in plain.graph_variants.entries) == (DIMS["x0"],)
    assert plain.graph_exec == plain.graph_variants.entries[0].graph_exec
    assert dict(plain.setup_identity)["text_trim"] == "False"


@needs_cuda
def test_pipeline_resources_describes_the_active_text_length_with_captures():
    """The same table on a real trimmed frontend: `pipeline_resources()`
    describes the ACTIVE captured length (sequence dims, AdaLN row counts and
    the backbone RoPE table with it), the pipeline buffers stay the one
    max-size set, and the table turns to the next length when the prompt
    does — one table per length is what the native pipeline is installed with.
    The CPU-only pin of the same contract is
    `test_pipeline_resources_describes_the_active_text_length`."""
    fe = _frontend(True, SHORT_VALID)
    short_x0 = SHORT_VALID + 1
    short = fe.pipeline_resources()
    print(f"x0={short_x0}: dims={(short.dims.x0, short.dims.a0, short.dims.total)} "
          f"txt_mod1.gate={tuple(short.txt_mod1.gate.shape)} rope={short.attention.rope_table:#x}")
    assert short.dims.x0 == fe.active_dims["x0"] == short_x0
    assert short.attention.rope_table == fe._rope_table.data_ptr()
    assert short.txt_mod1.gate.shape == (short_x0, fe.dims["hidden"])
    assert short.dims.a0 == DIMS["a0"] - (DIMS["x0"] - short_x0) < DIMS["a0"]

    fe.set_prompt(context=torch.randn(TEXT_LEN, DIMS["joint_attention_dim"]).to(torch.bfloat16),
                  context_mask=torch.cat([torch.ones(LONG_VALID, dtype=torch.bool),
                                          torch.zeros(TEXT_LEN - LONG_VALID, dtype=torch.bool)]))
    long_x0 = LONG_VALID + 1
    long = fe.pipeline_resources()
    print(f"x0={long_x0}: dims={(long.dims.x0, long.dims.a0, long.dims.total)} "
          f"txt_mod1.gate={tuple(long.txt_mod1.gate.shape)} rope={long.attention.rope_table:#x} "
          f"same buffers={long.buffers == short.buffers}")
    assert long.dims.x0 == fe.active_dims["x0"] == long_x0
    assert long.attention.rope_table == fe._rope_table.data_ptr() != short.attention.rope_table
    assert long.buffers == short.buffers, "the buffers are the max-size ones for every length"
    assert long.dims.a0 - short.dims.a0 == long_x0 - short_x0

    # text_trim=False: one length, the max dims, and the max-size RoPE table
    plain_fe = _frontend(False, LONG_VALID)
    plain = plain_fe.pipeline_resources()
    assert (plain.dims.x0, plain.dims.a0, plain.dims.total) == \
           (DIMS["x0"], DIMS["a0"], DIMS["total"])
    assert plain.attention.rope_table == plain_fe._rope_table.data_ptr() == \
        plain_fe._max_rope_table.data_ptr()


@needs_cuda
def test_export_model_runtime_serves_the_trimmed_abi_face():
    pytest.importorskip("flash_rt.runtime.exec", exc_type=ImportError)
    pytest.importorskip("flash_rt.runtime.export", exc_type=ImportError)
    fe = _frontend(True, SHORT_VALID)
    fe.set_prompt(context=torch.randn(TEXT_LEN, DIMS["joint_attention_dim"]).to(torch.bfloat16),
                  context_mask=torch.cat([torch.ones(LONG_VALID, dtype=torch.bool),
                                          torch.zeros(TEXT_LEN - LONG_VALID, dtype=torch.bool)]))
    short_x0, long_x0 = SHORT_VALID + 1, LONG_VALID + 1
    runtime = fe.export_model_runtime()
    manifest = json.loads(runtime.manifest)
    print(f"manifest graphs={manifest['graphs']} text_lengths={manifest['text_lengths']}")
    assert manifest["text_lengths"] == {"default_key": long_x0, "keys": [short_x0, long_x0],
                                        "per_prompt_length": True}
    graphs = {g["name"]: g for g in manifest["graphs"]}
    assert graphs["infer"]["default_key"] == long_x0
    assert graphs["infer"]["keys"] == [short_x0, long_x0]
    runtime.release()

    # The native face serves the same table, so a trimmed frontend is no
    # longer refused for its trim: it is refused for having no native handle.
    with pytest.raises(ValueError, match="requires native="):
        fe.export_model_runtime(io="native")
    plain = _frontend(False, LONG_VALID)
    with pytest.raises(ValueError, match="requires native="):
        plain.export_model_runtime(io="native")
    runtime = plain.export_model_runtime()
    assert json.loads(runtime.manifest)["text_lengths"] == {
        "default_key": DIMS["x0"], "keys": [DIMS["x0"]], "per_prompt_length": False}
    runtime.release()


@needs_cuda
def test_native_handle_declares_the_trimmed_surface_length_table():
    """`ImageWAMNativeRuntime.create` accepts a trimmed surface and hands its
    length table to the handle, which the C `create` copies and validates
    (`context_rows` among the declared lengths)."""
    fe = _frontend(True, SHORT_VALID)
    fe.set_prompt(context=torch.randn(TEXT_LEN, DIMS["joint_attention_dim"]).to(torch.bfloat16),
                  context_mask=torch.cat([torch.ones(LONG_VALID, dtype=torch.bool),
                                          torch.zeros(TEXT_LEN - LONG_VALID, dtype=torch.bool)]))
    surface = fe.runtime_surface()
    keys = tuple(entry.key for entry in surface.graph_variants.entries)
    handoff = build_io_config(surface)
    c = handoff.config
    table = np.ctypeslib.as_array(c.text_lengths, shape=(c.num_text_lengths,)).tolist()
    print(f"trimmed surface keys={keys} io config: context_rows={c.context_rows} "
          f"text_lengths={table}")
    assert table == list(keys) == [SHORT_KEY, LONG_KEY]
    assert c.context_rows == surface.graph_variants.active_key == LONG_KEY
    assert c.img_len == surface.img_len and c.context_width == surface.context_width
    assert c.num_text_lengths == len(surface.graph_variants.entries)
