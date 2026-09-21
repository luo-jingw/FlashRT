"""Python-side owner of one ImageWAM native handle (setup operations only).

`ImageWAMNativeRuntime` creates an `frt_imagewam_native` from a captured
frontend's runtime surface and exposes the setup calls (`use_graph`,
`set_text_length`, `set_pipeline`, `capture`, `set_proprio_row`,
`schema_records`, `bind_declaration`). It holds one reference to the
handle, the surface (whose `owner`, the frontend, owns the graph execs,
weights and buffers the handle borrows), one source and handoff per
installed pipeline, and the tensors and host arrays its handoff structs
point to, so the handle stays valid without other references to the
frontend. The hot-path verbs are the library's C functions;
`runtime_export.py` installs them on the `io="native"` declaration.
Interface record: docs/imagewam_native_cpp.md.

One graph (and one native pipeline) per text length: the handle holds a
graph variant table keyed by the context length `x0` (the same key space
the ABI face's `GraphVariants` uses), filled by `use_graph(key, exec)` for
every captured length of the surface — or by `capture_pipeline_text_lengths`
/ `capture` for lengths whose native pipeline this handle recorded itself.
`set_text_length(key)` selects the length the next ticks serve; `step` (a C
function) replays that key's exec, so the active length is carried by the
handle and not by the declaration. The same active length selects the
pipeline every pipeline call resolves against.
"""
from __future__ import annotations

import ctypes

from flash_rt.models.imagewam.native_library import ImageWAMGemmShape, ImageWAMNativeLibrary
from flash_rt.models.imagewam.native_resources import NativeHandoff, build_io_config, build_pipeline_config
from flash_rt.models.imagewam.pipeline_resources import (
    ImageWAMPipelineSource,
    ImageWAMTextLengthPipelineSource,
)
from flash_rt.models.imagewam.runtime_surface import ImageWAMRuntimeSurface


class ImageWAMNativeError(RuntimeError):
    def __init__(self, what: str, status: int, message: str):
        super().__init__(f"{what} failed rc={status}: {message}")
        self.status = status


class ImageWAMNativeRuntime:
    """One `frt_imagewam_native` handle and the buffers it borrows."""

    def __init__(self, library: ImageWAMNativeLibrary, handle: int, handoff: NativeHandoff,
                 surface: ImageWAMRuntimeSurface):
        self.library = library
        self.handle = handle
        self._handoff = handoff
        self._surface = surface
        # One entry per text length a pipeline is installed for: the handoff
        # (whose pointers the handle borrows) and the source it came from.
        self._pipeline_handoffs: dict[int, NativeHandoff] = {}
        self._pipeline_sources: dict[int, ImageWAMPipelineSource] = {}
        # Who recorded the graph the handle holds for a text length: "python"
        # after use_graph, "native" after capture.
        self._graph_producers: dict[int, str] = {}
        self.gemm_algos_installed = 0

    @classmethod
    def create(cls, surface: ImageWAMRuntimeSurface,
               library: ImageWAMNativeLibrary | None = None) -> "ImageWAMNativeRuntime":
        """One handle over `surface`'s buffers, weights and text length
        table.

        The io config carries the lengths `surface.graph_variants` names
        (the same key table the ABI declaration adopts), with
        `surface.context_rows` — the active length — as the length the
        handle starts on. A `text_trim=True` surface is served either by
        adopting one exec per captured length (`use_graph(key, exec)`) and
        selecting the length the prompt set (`set_text_length(key)`, which is
        legal while a model runtime over this handle is live; adoption is
        not), or by recording this handle's own graph per length
        (`capture_pipeline_text_lengths`, which installs one native pipeline
        per length). Either way the handle only accepts the lengths it was
        created with, so a deployment captures or declares them before
        `create`."""
        library = library or ImageWAMNativeLibrary()
        handoff = build_io_config(surface)
        out = ctypes.c_void_p()
        rc = library.lib.frt_imagewam_native_create(ctypes.byref(handoff.config), ctypes.byref(out))
        if rc != 0:
            raise ImageWAMNativeError("frt_imagewam_native_create", rc,
                                      library.lib.frt_imagewam_native_last_error(None).decode())
        runtime = cls(library, int(out.value), handoff, surface)
        if surface.proprio_row is not None:
            runtime.set_proprio_row(surface.proprio_row)
        return runtime

    def _check(self, what: str, rc: int) -> None:
        if rc != 0:
            raise ImageWAMNativeError(what, rc, self.last_error())

    def last_error(self) -> str:
        return (self.library.lib.frt_imagewam_native_last_error(self.handle) or b"").decode()

    @property
    def stream(self) -> int:
        return int(self.library.lib.frt_imagewam_native_stream(self.handle) or 0)

    @property
    def graph_exec(self) -> int:
        """The exec `step` replays: the active text length's graph variant
        (0 while the handle holds none)."""
        return int(self.library.lib.frt_imagewam_native_graph_exec(self.handle) or 0)

    @property
    def text_length(self) -> int:
        """The context length (`x0`) the next ticks serve."""
        return int(self.library.lib.frt_imagewam_native_text_length(self.handle))

    @property
    def graph_producer(self) -> str:
        """Who recorded the graph `step` replays — the active text length's
        variant: "python" after `use_graph` for it, "native" after `capture`
        for it, "" while the handle holds none for it (before either, or
        after `set_pipeline` dropped the graph this handle had captured for
        it). The identity of an `io="native"` export states this producer for
        the length the export declares as its default key, which is the
        handle's active length."""
        return self._graph_producers.get(self.text_length, "")

    def use_graph(self, key: int, graph_exec: int) -> None:
        """Replay a graph the Python frontend captured, for the text length
        `key` (the context length `x0` of the capture); the exec is
        borrowed. Setup only: refused while a model runtime over this handle
        is live, so every length a deployment serves is adopted before its
        export."""
        self._check("use_graph", self.library.lib.frt_imagewam_native_use_graph(
            self.handle, int(key), graph_exec))
        self._graph_producers[int(key)] = "python"

    def has_variant(self, key: int) -> bool:
        """Whether the handle holds a graph for the text length `key`."""
        return bool(self.library.lib.frt_imagewam_native_has_variant(self.handle, int(key)))

    def variant_exec(self, key: int) -> int:
        """The exec the handle holds for the text length `key` (0 when it
        holds none): what `step` replays once `key` is the active length."""
        return int(self.library.lib.frt_imagewam_native_variant_exec(self.handle, int(key)) or 0)

    def set_text_length(self, key: int) -> None:
        """Select the text length (`x0`) the next ticks serve: the graph
        `step` replays, the row bound of `set_proprio_row` and the length
        every pipeline call resolves against. Legal while a model runtime
        over this handle is live, like `set_proprio_row`: the setup producer
        calls both after every prompt change, since C++ cannot see the Python
        prompt. Refused (-2) for a length the handle holds no graph for."""
        self._check("set_text_length",
                    self.library.lib.frt_imagewam_native_set_text_length(self.handle, int(key)))

    def set_pipeline(self, source: ImageWAMPipelineSource) -> None:
        """Install the native pipeline for the text length `source`'s
        resource table describes (`source.pipeline_resources().dims.x0`, one
        of the lengths this handle was created with) and hand off every GEMM
        algorithm `source` has planned for the pipeline's shapes.

        The library installs it for that key and makes it the active text
        length, so the call is one of a set: a key installed again replaces
        its own pipeline and drops the graph this handle captured for it,
        while the other keys keep their pipeline and their graph. The GEMM
        shapes and algorithms below describe the pipeline just installed."""
        handoff = build_pipeline_config(source.pipeline_resources())
        key = int(handoff.config.x0)
        self._check("set_pipeline", self.library.lib.frt_imagewam_native_set_pipeline(
            self.handle, ctypes.byref(handoff.config)))
        # The library destroyed the graph THIS handle captured for `key`, if
        # it had one; an exec the producer adopted under that key stays, and
        # so does every other key's graph.
        self._pipeline_handoffs[key] = handoff
        self._pipeline_sources[key] = source
        if not self.has_variant(key):
            self._graph_producers.pop(key, None)
        count = ctypes.c_uint64(0)
        self.library.lib.frt_imagewam_native_gemm_shapes(self.handle, None, 0, ctypes.byref(count))
        shapes = (ImageWAMGemmShape * int(count.value))()
        self._check("gemm_shapes", self.library.lib.frt_imagewam_native_gemm_shapes(
            self.handle, shapes, count.value, ctypes.byref(count)))
        installed = 0
        for shape in shapes:
            algo = source.gemm_algo(shape.kind, shape.m, shape.n, shape.k)
            if algo is None:
                continue
            self._check("set_gemm_algo", self.library.lib.frt_imagewam_native_set_gemm_algo(
                self.handle, ctypes.byref(shape), algo, len(algo)))
            installed += 1
        self.gemm_algos_installed = installed
        self.gemm_shapes = [(s.kind, s.m, s.n, s.k) for s in shapes]

    def capture_pipeline_text_lengths(self, source: ImageWAMTextLengthPipelineSource) -> tuple[int, ...]:
        """Record this handle's own native graph for every text length
        `source` has captured: for each key, `source._activate_text_length(key)`
        (its resource table then describes that length), `set_pipeline(source)`
        (which installs for the key the table carries and selects it), then
        `capture`.

        The lengths are `source.captured_text_lengths()`, ascending, and each
        one must be a length this handle was created with (the surface's
        `graph_variants` keys at `create`). The source's active length is what
        it was before the call, and the handle serves that key again
        afterwards, so the one-key path and this per-length one leave the same
        state. Returns the keys, ascending; `step` serves each of them once
        `set_text_length` selects it."""
        keys = source.captured_text_lengths()
        if not keys:
            raise ValueError("capture_pipeline_text_lengths: the source has captured no text length; "
                             "call set_prompt() or precapture_text_lengths() first")
        active = int(source.active_dims["x0"])
        if active not in keys:
            raise ValueError(f"capture_pipeline_text_lengths: the source is active at x0={active}, which "
                             f"is not one of its captured lengths {list(keys)}")
        for key in keys:
            source._activate_text_length(key)
            self.set_pipeline(source)
            self.capture()
        source._activate_text_length(active)
        self.set_text_length(active)
        return keys

    def run(self, segment: int, index: int = 0) -> None:
        """One eager segment on the native stream (synchronized) of the
        pipeline installed for the active text length."""
        self._check(f"run({segment}, {index})",
                    self.library.lib.frt_imagewam_native_run(self.handle, segment, index))

    def capture(self) -> None:
        """Warm up and capture prefill + denoise for the active text length
        into a graph this handle owns; `step` replays it. The active length's
        pipeline must be installed (`set_pipeline`, whose key then is the
        active length)."""
        self._check("capture", self.library.lib.frt_imagewam_native_capture(self.handle))
        self._graph_producers[self.text_length] = "native"

    @property
    def graph_nodes(self) -> int:
        """Node count of the graph `step` replays, when this handle captured
        it (0 for an adopted exec)."""
        count = ctypes.c_uint64(0)
        self.library.lib.frt_imagewam_native_graph_nodes(self.handle, ctypes.byref(count))
        return int(count.value)

    def set_proprio_row(self, row: int) -> None:
        self._check("set_proprio_row",
                    self.library.lib.frt_imagewam_native_set_proprio_row(self.handle, int(row)))

    def schema_records(self) -> list[str]:
        needed = ctypes.c_uint64(0)
        self.library.lib.frt_imagewam_native_schema_records(self.handle, None, 0, ctypes.byref(needed))
        buf = ctypes.create_string_buffer(int(needed.value) + 1)
        self._check("schema_records", self.library.lib.frt_imagewam_native_schema_records(
            self.handle, buf, needed.value, ctypes.byref(needed)))
        return buf.raw[:needed.value].decode().splitlines()

    def bind_declaration(self, model_runtime_ptr: int) -> None:
        self._check("bind_declaration",
                    self.library.lib.frt_imagewam_native_bind_declaration(self.handle, model_runtime_ptr))

    @property
    def verbs(self) -> int:
        return int(self.library.lib.frt_imagewam_native_verbs())

    @property
    def retain_fn(self) -> int:
        """Owner retain for a model runtime over this handle's verbs: while
        one is live, use_graph / set_pipeline / capture are refused."""
        return self.library.function_address("frt_imagewam_native_declaration_retain")

    @property
    def release_fn(self) -> int:
        return self.library.function_address("frt_imagewam_native_declaration_release")

    def close(self) -> None:
        """Drop this owner's reference (the model runtime keeps its own)."""
        if self.handle:
            self.library.lib.frt_imagewam_native_release(self.handle)
            self.handle = 0

    def __del__(self) -> None:
        self.close()
