"""ctypes consumer of `frt_model_runtime_v1`.

Drives a model runtime only through the surface a native host has: the
struct's C function pointers, its port and stream descriptors,
`frt_buffer_dptr` from `libflashrt_exec`, and the CUDA runtime for SWAP
window copies. No producer object is touched.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
PORT_SWAP = 0
PORT_STAGED = 1
PORT_SETUP = 2


class PortDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p),
                ("modality", ctypes.c_uint32), ("dtype", ctypes.c_uint32),
                ("layout", ctypes.c_uint32), ("direction", ctypes.c_uint32),
                ("update", ctypes.c_uint32), ("required", ctypes.c_uint32),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("rank", ctypes.c_uint32),
                ("cadence_hint_hz", ctypes.c_uint32),
                ("buffer", ctypes.c_void_p),
                ("offset", ctypes.c_uint64), ("bytes", ctypes.c_uint64)]


class StreamDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("stream_id", ctypes.c_int),
                ("priority", ctypes.c_int), ("native_handle", ctypes.c_void_p)]


RetainReleaseFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class ExportV1(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
                ("ctx", ctypes.c_void_p),
                ("streams", ctypes.POINTER(StreamDesc)), ("n_streams", ctypes.c_uint64),
                ("graphs", ctypes.c_void_p), ("n_graphs", ctypes.c_uint64),
                ("buffers", ctypes.c_void_p), ("n_buffers", ctypes.c_uint64),
                ("capsule_regions", ctypes.c_void_p), ("n_capsule_regions", ctypes.c_uint64),
                ("fingerprint", ctypes.c_uint64),
                ("identity", ctypes.c_char_p), ("manifest_json", ctypes.c_char_p),
                ("owner", ctypes.c_void_p),
                ("retain", RetainReleaseFn), ("release", RetainReleaseFn)]


SetInputFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
                              ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int)
GetOutputFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
                               ctypes.c_void_p, ctypes.c_uint64,
                               ctypes.POINTER(ctypes.c_uint64), ctypes.c_int)
PrepareFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64)
StepFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
LastErrorFn = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)


class Verbs(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("set_input", SetInputFn), ("get_output", GetOutputFn),
                ("prepare", PrepareFn), ("step", StepFn), ("last_error", LastErrorFn)]


class ModelV1(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
                ("exp", ctypes.POINTER(ExportV1)),
                ("ports", ctypes.POINTER(PortDesc)), ("n_ports", ctypes.c_uint64),
                ("stages", ctypes.c_void_p), ("n_stages", ctypes.c_uint64),
                ("self_", ctypes.c_void_p), ("verbs", Verbs),
                ("owner", ctypes.c_void_p),
                ("retain", RetainReleaseFn), ("release", RetainReleaseFn)]


@dataclass(frozen=True)
class ConsumerPort:
    index: int
    name: str
    modality: int
    dtype: int
    layout: int
    direction: int
    update: int
    required: bool
    shape: tuple[int, ...]
    buffer: int
    offset: int
    nbytes: int


class ModelRuntimeError(RuntimeError):
    def __init__(self, what: str, status: int, message: str):
        super().__init__(f"{what} failed rc={status}: {message}")
        self.status = status


def exec_library_path() -> str:
    """Path of the `libflashrt_exec` the loaded `_flashrt_exec` module uses."""
    import _flashrt_exec
    return os.path.join(os.path.dirname(os.path.abspath(_flashrt_exec.__file__)), "libflashrt_exec.so")


class ModelRuntimeConsumer:
    """Adopts one `frt_model_runtime_v1*` (retain on construction, release on
    `close`) and drives it by port name."""

    def __init__(self, runtime_ptr: int, exec_library: str):
        self._m = ctypes.cast(runtime_ptr, ctypes.POINTER(ModelV1)).contents
        if self._m.abi_version != 1:
            raise ValueError(f"unsupported model-runtime ABI version {self._m.abi_version}")
        self._m.retain(self._m.owner)
        self._open = True
        self._exec = ctypes.CDLL(exec_library)
        self._exec.frt_buffer_dptr.restype = ctypes.c_void_p
        self._exec.frt_buffer_dptr.argtypes = [ctypes.c_void_p]
        self._cudart = ctypes.CDLL("libcudart.so")
        self._cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                                 ctypes.c_int, ctypes.c_void_p]
        self._cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        ports = []
        for i in range(self._m.n_ports):
            d = self._m.ports[i]
            ports.append(ConsumerPort(
                index=i, name=d.name.decode(), modality=d.modality, dtype=d.dtype,
                layout=d.layout, direction=d.direction, update=d.update,
                required=bool(d.required), shape=tuple(d.shape[k] for k in range(d.rank)),
                buffer=int(d.buffer or 0), offset=int(d.offset), nbytes=int(d.bytes)))
        self.ports = tuple(ports)
        exp = self._m.exp.contents
        self.n_stages = int(self._m.n_stages)
        self.fingerprint = int(exp.fingerprint)
        self.identity = exp.identity.decode()
        self.stream_id = int(exp.streams[0].stream_id)
        self.stream_handle = int(exp.streams[0].native_handle or 0)

    def close(self) -> None:
        if self._open:
            self._m.release(self._m.owner)
            self._open = False

    def port(self, name: str) -> ConsumerPort:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(name)

    def last_error(self) -> str:
        return (self._m.verbs.last_error(self._m.self_) or b"").decode(errors="replace")

    def _index(self, port: str | int) -> int:
        return port if isinstance(port, int) else self.port(port).index

    def set_input_status(self, port: str | int, payload: bytes | ctypes.Array, stream: int = -1) -> int:
        """Raw `set_input` status (for negative tests); `port` is a name or an index."""
        index = self._index(port)
        if isinstance(payload, bytes):
            buf = ctypes.create_string_buffer(payload, len(payload))
            return self._m.verbs.set_input(self._m.self_, index, ctypes.cast(buf, ctypes.c_void_p),
                                           len(payload), stream)
        return self._m.verbs.set_input(self._m.self_, index, ctypes.cast(payload, ctypes.c_void_p),
                                       ctypes.sizeof(payload), stream)

    def set_input(self, name: str, payload: bytes | ctypes.Array) -> None:
        rc = self.set_input_status(name, payload)
        if rc != 0:
            raise ModelRuntimeError(f"set_input({name})", rc, self.last_error())

    def _window_ptr(self, p: ConsumerPort) -> int:
        if p.update != PORT_SWAP or not p.buffer:
            raise ValueError(f"port {p.name!r} has no SWAP window")
        return int(self._exec.frt_buffer_dptr(p.buffer)) + p.offset

    def write_swap(self, name: str, data: np.ndarray) -> None:
        """Host-to-device copy into a SWAP input window, enqueued on the
        exported stream so it is ordered before the next `step`. The host
        array may be reused on return (pageable source is staged)."""
        p = self.port(name)
        data = np.ascontiguousarray(data)
        if data.nbytes != p.nbytes:
            raise ValueError(f"{name}: {data.nbytes} bytes for a {p.nbytes}-byte window")
        rc = self._cudart.cudaMemcpyAsync(self._window_ptr(p), data.ctypes.data, p.nbytes,
                                          CUDA_MEMCPY_HOST_TO_DEVICE, self.stream_handle)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync H2D into {name} failed: {rc}")

    def read_swap(self, name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        """Copy a SWAP window to host on the exported stream, then synchronize it."""
        p = self.port(name)
        out = np.empty(shape, dtype=dtype)
        if out.nbytes != p.nbytes:
            raise ValueError(f"{name}: {out.nbytes} bytes requested from a {p.nbytes}-byte window")
        rc = self._cudart.cudaMemcpyAsync(out.ctypes.data, self._window_ptr(p), p.nbytes,
                                          CUDA_MEMCPY_DEVICE_TO_HOST, self.stream_handle)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync D2H from {name} failed: {rc}")
        self.sync()
        return out

    def sync(self) -> None:
        rc = self._cudart.cudaStreamSynchronize(self.stream_handle)
        if rc != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed: {rc}")

    def step(self) -> None:
        rc = self._m.verbs.step(self._m.self_)
        if rc != 0:
            raise ModelRuntimeError("step", rc, self.last_error())

    def get_output_status(self, port: str | int, capacity: int, stream: int = -1) -> tuple[int, bytes, int]:
        buf = ctypes.create_string_buffer(max(capacity, 1))
        written = ctypes.c_uint64(0)
        rc = self._m.verbs.get_output(self._m.self_, self._index(port), ctypes.cast(buf, ctypes.c_void_p),
                                      capacity, ctypes.byref(written), stream)
        return rc, buf.raw[:int(written.value)] if rc == 0 else b"", int(written.value)

    def get_output(self, name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        rc, data, written = self.get_output_status(name, nbytes)
        if rc != 0:
            raise ModelRuntimeError(f"get_output({name})", rc, self.last_error())
        if written != nbytes:
            raise ValueError(f"get_output({name}) wrote {written} bytes, expected {nbytes}")
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def make_image_views(frames: list[np.ndarray], view_type: type[ctypes.Structure]) -> ctypes.Array:
    """`frt_image_view[n]` over contiguous `(H, W, 3)` uint8 host frames (RGB8).
    The frames must stay alive while the views are used."""
    views = (view_type * len(frames))()
    for i, im in enumerate(frames):
        views[i].struct_size = ctypes.sizeof(view_type)
        views[i].pixel_format = 0
        views[i].data = ctypes.c_void_p(im.ctypes.data)
        views[i].bytes = im.nbytes
        views[i].width = int(im.shape[1])
        views[i].height = int(im.shape[0])
        views[i].stride_bytes = int(im.strides[0])
        views[i].reserved = 0
        views[i].timestamp_ns = 0
    return views
