"""Python-side owner of one ImageWAM native handle (setup operations only).

`ImageWAMNativeRuntime` creates an `frt_imagewam_native` from a captured
frontend's runtime surface and exposes the setup calls (`use_graph`,
`set_pipeline`, `capture`, `set_proprio_row`, `schema_records`,
`bind_declaration`). It holds one reference to the handle, the surface
(whose `owner`, the frontend, owns the graph exec, weights and buffers the
handle borrows), the source of the installed pipeline, and the tensors
and host arrays its handoff structs point to, so the handle stays valid
without other references to the frontend. The hot-path
verbs are the library's C functions; `runtime_export.py` installs them
on the `io="native"` declaration. Interface record:
docs/imagewam_native_cpp.md.
"""
from __future__ import annotations

import ctypes

from flash_rt.models.imagewam.native_library import ImageWAMGemmShape, ImageWAMNativeLibrary
from flash_rt.models.imagewam.native_resources import NativeHandoff, build_io_config, build_pipeline_config
from flash_rt.models.imagewam.pipeline_resources import ImageWAMPipelineSource
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
        self._pipeline_handoff: NativeHandoff | None = None
        self._pipeline_source: ImageWAMPipelineSource | None = None
        self._graph_producer = ""
        self.gemm_algos_installed = 0

    @classmethod
    def create(cls, surface: ImageWAMRuntimeSurface,
               library: ImageWAMNativeLibrary | None = None) -> "ImageWAMNativeRuntime":
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
        return int(self.library.lib.frt_imagewam_native_graph_exec(self.handle) or 0)

    @property
    def graph_producer(self) -> str:
        """Who recorded the graph `step` replays: "python" after `use_graph`,
        "native" after `capture`, "" while there is none (before either, or
        after `set_pipeline` replaced the pipeline a captured graph came from)."""
        return self._graph_producer

    def use_graph(self, graph_exec: int) -> None:
        """Replay a graph the Python frontend captured (borrowed exec)."""
        self._check("use_graph", self.library.lib.frt_imagewam_native_use_graph(self.handle, graph_exec))
        self._graph_producer = "python"

    def set_pipeline(self, source: ImageWAMPipelineSource) -> None:
        """Install the native pipeline over `source`'s resources and hand off
        every GEMM algorithm `source` has planned for the pipeline's shapes."""
        handoff = build_pipeline_config(source.pipeline_resources())
        self._check("set_pipeline", self.library.lib.frt_imagewam_native_set_pipeline(
            self.handle, ctypes.byref(handoff.config)))
        # The library destroyed any graph captured from the previous
        # pipeline, so the previous handoff's resources can go.
        self._pipeline_handoff = handoff
        self._pipeline_source = source
        if self._graph_producer == "native":
            self._graph_producer = ""
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

    def run(self, segment: int, index: int = 0) -> None:
        """One eager segment on the native stream (synchronized)."""
        self._check(f"run({segment}, {index})",
                    self.library.lib.frt_imagewam_native_run(self.handle, segment, index))

    def capture(self) -> None:
        """Warm up and capture prefill + denoise; `step` replays it."""
        self._check("capture", self.library.lib.frt_imagewam_native_capture(self.handle))
        self._graph_producer = "native"

    @property
    def graph_nodes(self) -> int:
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
