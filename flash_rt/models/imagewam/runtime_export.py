"""Runtime-export producer for the ImageWAM Thor frontend.

Lowers one captured `ImageWAMTorchFrontendThor` into an
`frt_model_runtime_v1` (docs/model_runtime_api.md). The frontend keeps
ownership of the graphs, the weights and every device buffer; this module
wraps the device windows (`frt_buffer_wrap`), adopts the torch graph execs
(`frt_graph_adopt`, not owned), declares ports, stage and region, and
supplies the Python verbs. The verbs dispatch to the frontend's own
staging operations (`ImageWAMRuntimeSource`), the same operations
`infer()` uses, and run them on the capture stream so staged writes are
ordered before the replay. Interface record: docs/imagewam_model_runtime.md.

One captured text length is one graph variant, keyed by that length
(`ShapeKey = x0`, `runtime_surface.GraphVariants`): the declaration adopts
one exec per key, and `step` replays the key of the length the prompt set.
`text_trim=False` is the one-entry case (`dims["x0"]`).

`io="python"` port schema, in port-index order (ports absent from a
deployment are skipped, the rest keep this relative order):

  images        IN   STAGED  IMAGE   u8    (views, H, W, 3)   [VAE loaded]
  image_tokens  IN   SWAP    TENSOR  bf16  (img_len, HD)       img_raw   [VAE outside the graph]
  image_views   IN   SWAP    IMAGE   u8    (views, H, W, 3)   views_u8  [VAE inside the graph]
  proprio       IN   STAGED  STATE   f32   (proprio_dim,)      [proprio_dim set]
  noise         IN   SWAP    TENSOR  f32   (num_action, action_dim)  action_latent
  actions       OUT  STAGED  ACTION  f32   (num_action, action_dim)
  actions_raw   OUT  SWAP    TENSOR  f32   (num_action, action_dim)  action_latent
  prompt        IN   SETUP   TEXT    u8    (-1,)               [Qwen3 loaded]

`(views, H, W)` is the frontend's resolved view shape
(`ImageWAMTorchFrontendThor._input_view_shape`): the workload's
`vae_graph_input()` when the frontend has a workload, else the in-graph VAE
stage's spec, else `(2, 224, 224)` for a caller that passed dims by hand.
With the VAE in the graph the graph writes `img_raw` itself, so
`image_tokens` is replaced by the graph's uint8 view buffer.

`noise` is the initial action latent consumed exactly as written; the
graph integrates it in place, so the same window holds the normalized
chunk (`actions_raw`) after `step`. It must be rewritten before every
`step`. It is the ABI form of `infer(..., action_noise=...)`; with no
`action_noise`, `infer()` writes `0.01 * N(0,1)` there (issues.md
ISSUE-002). The ABI applies no scale.

`io="native"` keeps `image_tokens`, `proprio`, `noise`, `actions` and
`actions_raw` (same windows, `image_tokens` always required) and drops
`images` and `prompt`, whose transforms (VAE, Qwen3) stay in Python.
Its verbs are the C functions of `libflashrt_imagewam_native.so`
(docs/imagewam_native_cpp.md), installed over the declaration with
`frt_model_runtime_override_verbs`; the declaration's stream and graph are
the native handle's. The native handle holds one graph at one context
geometry, so this face refuses a `text_trim=True` frontend (rule R5).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from flash_rt.models.imagewam.native_runtime import ImageWAMNativeRuntime
from flash_rt.models.imagewam.runtime_surface import (
    GraphVariantPlan,
    ImageWAMRuntimeSource,
    ImageWAMRuntimeSurface,
    graph_variant_plan,
    uncaptured_text_length_message,
)
from flash_rt.runtime import exec as frt_exec
from flash_rt.runtime import export as frt_export
from flash_rt.runtime.export import VerbStatusError

PIXEL_RGB8 = 0

STATUS_OK = 0
STATUS_INVALID = -1
STATUS_NOT_FOUND = -2
STATUS_UNSUPPORTED = -3
STATUS_SHAPE = -4


class FrtImageView(ctypes.Structure):
    """ctypes mirror of `frt_image_view` (runtime/include/flashrt/model_runtime.h)."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("bytes", ctypes.c_uint64),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("stride_bytes", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("timestamp_ns", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class ImageWAMPortLayout:
    """Port names in declaration (index) order, for verb dispatch."""

    names: tuple[str, ...]

    def index(self, name: str) -> int:
        return self.names.index(name)


def view_names(num_views: int) -> list[str]:
    """Camera view order of the `images` port: view1 (agent), view2 (wrist), ..."""
    return [f"view{i + 1}" for i in range(num_views)]


def decode_image_views(payload: bytes, view_shape: tuple[int, int, int]) -> list[torch.Tensor]:
    """Decode an `frt_image_view[views]` payload into `views` `(H, W, 3)`
    uint8 CPU tensors (host pixels are copied), `view_shape = (views, H, W)`.
    Raises `VerbStatusError`: -4 when the view count or a view's geometry
    differs from the declared `images` port, -1 for a malformed view
    (struct size, pixel format, stride, missing or short pixel data)."""
    num_views, height, width = view_shape
    view_size = ctypes.sizeof(FrtImageView)
    if len(payload) != num_views * view_size:
        raise VerbStatusError(STATUS_SHAPE,
                              f"images payload must be {num_views} frt_image_view "
                              f"({num_views * view_size} bytes), got {len(payload)} bytes")
    views = (FrtImageView * num_views).from_buffer_copy(payload)
    row_bytes = width * 3
    frames = []
    for i, view in enumerate(views):
        if view.struct_size != view_size:
            raise VerbStatusError(STATUS_INVALID, f"images[{i}].struct_size={view.struct_size}, "
                                                  f"expected {view_size}")
        if view.pixel_format != PIXEL_RGB8:
            raise VerbStatusError(STATUS_INVALID, f"images[{i}].pixel_format={view.pixel_format}, "
                                                  f"expected RGB8 ({PIXEL_RGB8})")
        if (view.width, view.height) != (width, height):
            raise VerbStatusError(STATUS_SHAPE, f"images[{i}] is {view.width}x{view.height}, "
                                                f"expected {width}x{height}")
        if view.stride_bytes < row_bytes:
            raise VerbStatusError(STATUS_INVALID, f"images[{i}].stride_bytes={view.stride_bytes} < {row_bytes}")
        needed = view.stride_bytes * (height - 1) + row_bytes
        if view.bytes < needed or not view.data:
            raise VerbStatusError(STATUS_INVALID, f"images[{i}] holds {view.bytes} bytes, needs {needed}")
        raw = np.frombuffer(ctypes.string_at(view.data, needed), dtype=np.uint8)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(height, row_bytes), strides=(view.stride_bytes, 1))
        frames.append(torch.from_numpy(rows.copy().reshape(height, width, 3)))
    return frames


class ImageWAMPythonVerbs:
    """The `io="python"` verbs. Each STAGED verb calls one frontend staging
    operation on the capture stream; `step` replays the graph variant of the
    length the prompt actually set. Failures raise `VerbStatusError` with the
    status the `io="native"` verbs return for the same case: -2 unknown port
    (or a text length this runtime adopted no graph variant for), -3 SWAP
    port, -4 payload size or geometry, -1 anything else invalid (including a
    stream that is not the exported one)."""

    def __init__(self, source: ImageWAMRuntimeSource, surface: ImageWAMRuntimeSurface,
                 layout: ImageWAMPortLayout, graph: frt_exec.Graph, stream_id: int,
                 plan: GraphVariantPlan):
        self._source = source
        self._surface = surface
        self._layout = layout
        self._graph = graph
        self._stream_id = stream_id
        self._plan = plan

    def _port_name(self, verb: str, port: int, stream: int) -> str:
        if not 0 <= port < len(self._layout.names):
            raise VerbStatusError(STATUS_NOT_FOUND, f"{verb}: unknown port index {port}")
        if stream not in (-1, self._stream_id):
            raise VerbStatusError(STATUS_INVALID, f"{verb}: stream {stream} is not an exported stream "
                                                  f"(use -1 or {self._stream_id})")
        return self._layout.names[port]

    def set_input(self, port: int, payload: bytes, stream: int) -> int:
        name = self._port_name("set_input", port, stream)
        with torch.cuda.stream(self._surface.stream):
            if name == "images":
                frames = decode_image_views(payload, self._surface.view_shape)
                self._source.stage_images(*frames)
                return STATUS_OK
            if name == "proprio":
                expected = self._surface.proprio_dim * 4
                if len(payload) != expected:
                    raise VerbStatusError(STATUS_SHAPE, f"set_input(proprio): payload must be {expected} "
                                                        f"bytes (f32), got {len(payload)}")
                self._source.stage_proprio(np.frombuffer(payload, dtype=np.float32).copy())
                return STATUS_OK
            if name == "prompt":
                self._source.set_prompt(payload.decode("utf-8"))
                return STATUS_OK
        raise VerbStatusError(STATUS_UNSUPPORTED, f"set_input: {name!r} is a SWAP port, write its buffer window")

    def get_output(self, port: int, stream: int) -> bytes:
        name = self._port_name("get_output", port, stream)
        if name != "actions":
            raise VerbStatusError(STATUS_UNSUPPORTED, f"get_output: {name!r} is a SWAP port, read its "
                                                      "buffer window")
        with torch.cuda.stream(self._surface.stream):
            actions = self._source.read_actions()
        return np.ascontiguousarray(actions, dtype=np.float32).tobytes()

    def step(self) -> int:
        """Replays the graph variant of the length the prompt set
        (`source.active_dims["x0"]`, the same source `infer()` replays), on
        the exported stream. A trimmed frontend has one graph per prompt
        length, and the length is the variant key.

        A length this runtime adopted no variant for is refused with -2 and
        a message naming the length and the adopted ones: the deployment
        declares its lengths at export time (`precapture_text_lengths`), and
        `frt_graph_has_variant` decides, since the exec layer's live table
        is the truth."""
        key = int(self._source.active_dims["x0"])
        if not self._graph.has_variant(key):
            raise VerbStatusError(STATUS_NOT_FOUND,
                                  uncaptured_text_length_message(key, self._plan.keys))
        return int(self._graph.replay(key, self._stream_id))


def _ports(surface: ImageWAMRuntimeSurface, windows: Mapping[str, frt_exec.Buffer],
           io: str) -> list[frt_export.PortSpec]:
    chunk = (surface.num_action, surface.action_dim)
    staged_images = io == "python" and surface.has_vae
    frame_shape = (*surface.view_shape, 3)
    ports = []
    if staged_images:
        ports.append(frt_export.PortSpec(
            "images", "image", "u8", "nhwc", "in", "staged", required=True, shape=frame_shape))
    if surface.views_u8 is None:
        ports.append(frt_export.PortSpec(
            "image_tokens", "tensor", "bf16", "flat", "in", "swap", required=not staged_images,
            shape=(surface.img_len, surface.token_dim), buffer=windows["img_raw"]))
    else:
        ports.append(frt_export.PortSpec(
            "image_views", "image", "u8", "nhwc", "in", "swap", shape=frame_shape,
            buffer=windows["views_u8"]))
    if surface.proprio_dim is not None:
        ports.append(frt_export.PortSpec(
            "proprio", "state", "f32", "flat", "in", "staged", required=True,
            shape=(surface.proprio_dim,)))
    ports.append(frt_export.PortSpec(
        "noise", "tensor", "f32", "flat", "in", "swap", required=True,
        shape=chunk, buffer=windows["action_latent"]))
    ports.append(frt_export.PortSpec(
        "actions", "action", "f32", "flat", "out", "staged",
        shape=chunk, nbytes=surface.num_action * surface.action_dim * 4))
    ports.append(frt_export.PortSpec(
        "actions_raw", "tensor", "f32", "flat", "out", "swap",
        shape=chunk, buffer=windows["action_latent"]))
    if io == "python" and surface.has_text_encoder:
        ports.append(frt_export.PortSpec(
            "prompt", "text", "u8", "flat", "in", "setup", shape=(-1,)))
    return ports


def _identity(surface: ImageWAMRuntimeSurface, io: str, graph_producer: str,
              extra: Mapping[str, str] | None) -> dict[str, str]:
    ident = {"model": "imagewam"}
    ident.update(surface.setup_identity)
    ident.update({
        "io": io,
        "graph_producer": graph_producer,
        "views": ",".join(view_names(surface.view_shape[0])) if surface.has_vae else "",
        "vae_in_graph": str(surface.views_u8 is not None),
        "has_vae": str(surface.has_vae),
        "has_text_encoder": str(surface.has_text_encoder),
        "proprio_dim": str(surface.proprio_dim),
        "action_denormalized": str(surface.action_denormalized),
    })
    ident.update({str(k): str(v) for k, v in (extra or {}).items()})
    return ident


def export_model_runtime(source: ImageWAMRuntimeSource, *, identity: Mapping[str, str] | None = None,
                         io: str = "python",
                         native: ImageWAMNativeRuntime | None = None) -> frt_export.ModelRuntime:
    """Package a captured ImageWAM frontend as an `frt_model_runtime_v1`.

    Requires `set_prompt()` first (the graph must be captured). `identity`
    adds canonical identity pairs; production deployments pass a weights
    digest. Returns a `ModelRuntime` whose `ptr` a native consumer adopts;
    the runtime anchors the frontend for its lifetime.

    `io="python"`: Python verbs over the frontend's own graphs and stream.
    The declaration adopts one exec per captured text length
    (`surface.graph_variants`), `GraphSpec.default_key` is the length active
    at export, and `step` replays the key of the length the prompt set. The
    manifest's `text_lengths` states the adopted table.

    `io="native"`: `native` (an `ImageWAMNativeRuntime` with a graph, from
    `use_graph` or `capture`) supplies the stream, the graph and the C
    verbs; the declaration is checked by `native.bind_declaration` and
    never published with placeholder verbs. The native handle holds one
    graph at one context geometry, so a `text_trim=True` frontend is
    refused (rule R5).
    """
    if io not in ("python", "native"):
        raise ValueError(f"unknown ImageWAM model-runtime io face {io!r} (supported: 'python', 'native')")
    surface = source.runtime_surface()
    plan = graph_variant_plan(surface.graph_variants)
    if io == "native" and surface.graph_variants.per_prompt_length:
        raise ValueError("io='native' does not support text_trim=True (rule R5): the native handle "
                         "replays one graph at one context geometry, while a trimmed frontend runs one "
                         "graph per prompt length (ISSUE-080 condition 5); the ABI face "
                         "(io='python') serves a trimmed frontend")
    if io == "native" and (native is None or not native.graph_exec):
        raise ValueError("io='native' requires native=ImageWAMNativeRuntime with a graph "
                         "(use_graph or capture first)")
    if io == "native" and surface.views_u8 is not None:
        raise ValueError("io='native' needs the VAE outside the graph (vae_graph_input=None): the "
                         "native face takes VAE tokens through image_tokens")
    ctx = frt_exec.Ctx()
    stream_handle = int(surface.stream.cuda_stream) if io == "python" else native.stream
    graph_producer = "python" if io == "python" else native.graph_producer
    stream_id = ctx.wrap_stream(stream_handle)
    if io == "python":
        # One adopted exec per captured text length; `step` replays the
        # length the prompt set (`ImageWAMPythonVerbs.step`).
        default_key, keys = plan.default_key, plan.keys
        graph = ctx.graph("imagewam_infer", plan.max_variants)
        for variant in surface.graph_variants.entries:
            graph.adopt(variant.key, variant.graph_exec)
    else:
        # The native handle owns its one graph, at the context geometry the
        # surface had when the handle was created.
        default_key = plan.default_key
        keys = (default_key,)
        graph = ctx.graph("imagewam_infer", 1)
        graph.adopt(default_key, native.graph_exec)

    def wrap(name: str, tensor: torch.Tensor) -> frt_exec.Buffer:
        return ctx.wrap(name, tensor.data_ptr(), tensor.numel() * tensor.element_size())

    windows = {
        "img_raw": wrap("img_raw", surface.img_raw),
        "context": wrap("context", surface.context),
        "action_latent": wrap("action_latent", surface.action_latent),
    }
    if surface.views_u8 is not None:
        windows["views_u8"] = wrap("views_u8", surface.views_u8)
    ports = _ports(surface, windows, io)
    manifest = {
        "io": io,
        "graph_producer": graph_producer,
        "stage_plan": {"name": "full", "stages": [{"name": "infer", "graph": "infer", "after": []}]},
        "image_views": view_names(surface.view_shape[0]) if (io == "python" and surface.has_vae) else [],
        "vae_in_graph": surface.views_u8 is not None,
        "noise": {
            "window": "action_latent",
            "consumed_as_written": True,
            "overwritten_by_output": True,
            "rewrite_before_every_step": True,
            "frontend_infer_fill": "0.01 * N(0, 1)",
        },
        "actions": {"denormalized": surface.action_denormalized},
        # The deployment's record of which text lengths the runtime can
        # serve: one adopted graph per `keys` entry, `default_key` the
        # length active at export (`step` replays the length the prompt
        # set, so a host must not assume the key is fixed while
        # `per_prompt_length` is true).
        "text_lengths": {
            "default_key": default_key,
            "keys": list(keys),
            "per_prompt_length": surface.graph_variants.per_prompt_length,
        },
    }
    if io == "native":
        manifest["prompt"] = "set through the setup producer; set_proprio_row after a prompt change"
    common = dict(
        streams=[frt_export.StreamSpec("main", stream_id, native_handle=stream_handle)],
        graphs=[frt_export.GraphSpec("infer", graph, default_key, keys)],
        buffers=[
            frt_export.BufferSpec("img_raw", windows["img_raw"], "input"),
            frt_export.BufferSpec("context", windows["context"], ("input", "state")),
            frt_export.BufferSpec("action_latent", windows["action_latent"], ("input", "output")),
        ] + ([frt_export.BufferSpec("views_u8", windows["views_u8"], "input")]
             if "views_u8" in windows else []),
        regions=[frt_export.RegionSpec("rollout_boundary", windows["action_latent"])],
        ports=ports,
        stages=[frt_export.StageSpec("infer")],
        identity=_identity(surface, io, graph_producer, identity),
        manifest_extra=manifest,
    )
    if io == "python":
        layout = ImageWAMPortLayout(tuple(p.name for p in ports))
        verbs = ImageWAMPythonVerbs(source, surface, layout, graph, stream_id, plan)
        return frt_export.build_model_runtime(
            ctx, owner=(source, surface, windows, verbs),
            set_input=verbs.set_input, get_output=verbs.get_output, step=verbs.step, **common)

    # The declaration exists only inside this call: placeholder callables
    # satisfy the builder's STAGED check until the native verbs replace them.
    declaration = frt_export.build_model_runtime(
        ctx, owner=(source, surface, windows, native),
        set_input=lambda _port, _payload, _stream: STATUS_UNSUPPORTED,
        get_output=lambda _port, _stream: b"",
        step=lambda: STATUS_UNSUPPORTED, **common)
    try:
        native.bind_declaration(declaration.ptr)
        return frt_export.override_model_runtime_verbs(
            declaration, verbs=native.verbs, verbs_self=native.handle, owner=native.handle,
            retain_owner=native.retain_fn, release_owner=native.release_fn, anchor=native)
    finally:
        declaration.release()
