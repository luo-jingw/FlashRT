"""Python-side owner of one ImageWAM native handle (setup operations only).

`ImageWAMNativeRuntime` creates an `frt_imagewam_native` from a captured
frontend, holds one reference plus the keepalive of every borrowed
handoff buffer, and exposes the setup calls (`use_graph`,
`set_proprio_row`, `schema_records`, `bind_declaration`). The hot-path
verbs are the library's C functions; `runtime_export.py` installs them
on the `io="native"` declaration. Interface record:
docs/imagewam_native_cpp.md.
"""
from __future__ import annotations

import ctypes

from flash_rt.models.imagewam.native_library import ImageWAMNativeLibrary
from flash_rt.models.imagewam.native_resources import NativeHandoff, build_io_config
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
        self._graph_producer = ""

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
        """Who recorded the graph `step` replays: "python" after `use_graph`."""
        return self._graph_producer

    def use_graph(self, graph_exec: int) -> None:
        """Replay a graph the Python frontend captured (borrowed exec)."""
        self._check("use_graph", self.library.lib.frt_imagewam_native_use_graph(self.handle, graph_exec))
        self._graph_producer = "python"

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
        return self.library.function_address("frt_imagewam_native_retain")

    @property
    def release_fn(self) -> int:
        return self.library.function_address("frt_imagewam_native_release")

    def close(self) -> None:
        """Drop this owner's reference (the model runtime keeps its own)."""
        if self.handle:
            self.library.lib.frt_imagewam_native_release(self.handle)
            self.handle = 0

    def __del__(self):
        self.close()
