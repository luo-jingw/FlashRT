"""`text_trim=True` and the consumers of a captured frontend's graphs.

A trimmed frontend runs one graph per prompt length, so the runtime surface
carries the variant table keyed by that length (`GraphVariants`) and each
consumer adopts one exec per key, `step` replaying the length the prompt set
(plan.md phase S2 for the ABI face, S4 for the native one). What each
consumer does now:

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
- `pipeline_resources()` still refuses: the native C++ pipeline records one
  graph at one context length from one resource table, and that is what rule
  R5 refuses `text_trim` for (`consumer="native"`), not a silent gap.

`graph_variant_plan` and `uncaptured_text_length_message` are the pure part
of that contract ("which keys and which default key does this table
declare", "what does `step` say about a length with no variant"); the io
config's length table and the handle's key plumbing are checked here against
a stub of the native library. Both are covered without a GPU; everything
that touches a captured frontend needs one.
"""
from __future__ import annotations

import ctypes
import dataclasses
import json

import numpy as np
import pytest
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.native_library import ImageWAMIoConfig
from flash_rt.models.imagewam.native_resources import build_io_config
from flash_rt.models.imagewam.native_runtime import ImageWAMNativeError, ImageWAMNativeRuntime
from flash_rt.models.imagewam.runtime_surface import (
    GraphVariants,
    ImageWAMRuntimeSurface,
    TextLengthGraph,
    graph_variant_plan,
    uncaptured_text_length_message,
)

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
    table back out of the real ctypes layout."""

    def __init__(self) -> None:
        self.lib = self
        self.calls: list[tuple] = []
        self.created_config: ImageWAMIoConfig | None = None
        self.handle = 0x1000
        self.variants: dict[int, int] = {}
        self.declared: tuple[int, ...] = ()
        self.active = 0

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


# -- the surface and the export -------------------------------------------

def _frontend(text_trim: bool, valid: int = SHORT_VALID) -> ImageWAMTorchFrontendThor:
    torch.manual_seed(0)
    fe = ImageWAMTorchFrontendThor(precision="fp16", dims_override=dict(DIMS), text_trim=text_trim)
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
def test_pipeline_resources_refuses_text_trim():
    """The native pipeline records one graph at one context length from one
    resource table; a per-length resource table is a later phase. This is
    what rule R5 refuses `text_trim` for on the native consumer, while the
    native model runtime serves a trimmed frontend."""
    with pytest.raises(ValueError, match="text_trim=True") as e:
        _frontend(True).pipeline_resources()
    print(f"pipeline_resources: {e.value}")
    assert "native pipeline" in str(e.value)
    assert "set_text_length" in str(e.value), "the refusal names what does serve a trimmed frontend"
    resources = _frontend(False).pipeline_resources()
    assert resources.dims.x0 == DIMS["x0"]


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
