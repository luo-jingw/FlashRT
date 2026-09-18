"""ctypes binding of `libflashrt_imagewam_native.so`.

Mirrors cpp/models/imagewam/include/flashrt/cpp/models/imagewam/c_api.h:
the config structs and the function prototypes. Build the library with
`cmake --build build --target flashrt_imagewam_native` (it is written to
`flash_rt/`). Interface record: docs/imagewam_native_cpp.md.
"""
from __future__ import annotations

import ctypes
import os

LIBRARY_NAME = "libflashrt_imagewam_native.so"


class ImageWAMIoConfig(ctypes.Structure):
    """`frt_imagewam_io_config`."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("img_len", ctypes.c_uint32),
        ("token_dim", ctypes.c_uint32),
        ("num_action", ctypes.c_uint32),
        ("action_dim", ctypes.c_uint32),
        ("proprio_dim", ctypes.c_uint32),
        ("context_rows", ctypes.c_uint32),
        ("context_width", ctypes.c_uint32),
        ("img_raw", ctypes.c_void_p),
        ("context", ctypes.c_void_p),
        ("action_latent", ctypes.c_void_p),
        ("proprio_weight_t", ctypes.c_void_p),
        ("proprio_bias", ctypes.c_void_p),
        ("state_scale", ctypes.POINTER(ctypes.c_float)),
        ("state_offset", ctypes.POINTER(ctypes.c_float)),
        ("action_scale", ctypes.POINTER(ctypes.c_float)),
        ("action_offset", ctypes.POINTER(ctypes.c_float)),
    ]


def default_library_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), LIBRARY_NAME)


class ImageWAMNativeLibrary:
    """The loaded shared library with typed prototypes."""

    def __init__(self, path: str | None = None):
        self.path = path or default_library_path()
        if not os.path.isfile(self.path):
            raise ImportError(
                f"{self.path} not found; build it with "
                "`cmake --build build --target flashrt_imagewam_native`")
        lib = ctypes.CDLL(self.path)
        handle_p = ctypes.c_void_p
        lib.frt_imagewam_native_create.argtypes = [ctypes.POINTER(ImageWAMIoConfig),
                                                    ctypes.POINTER(ctypes.c_void_p)]
        lib.frt_imagewam_native_create.restype = ctypes.c_int
        lib.frt_imagewam_native_retain.argtypes = [handle_p]
        lib.frt_imagewam_native_retain.restype = None
        lib.frt_imagewam_native_release.argtypes = [handle_p]
        lib.frt_imagewam_native_release.restype = None
        lib.frt_imagewam_native_last_error.argtypes = [handle_p]
        lib.frt_imagewam_native_last_error.restype = ctypes.c_char_p
        lib.frt_imagewam_native_stream.argtypes = [handle_p]
        lib.frt_imagewam_native_stream.restype = ctypes.c_void_p
        lib.frt_imagewam_native_use_graph.argtypes = [handle_p, ctypes.c_void_p]
        lib.frt_imagewam_native_use_graph.restype = ctypes.c_int
        lib.frt_imagewam_native_graph_exec.argtypes = [handle_p]
        lib.frt_imagewam_native_graph_exec.restype = ctypes.c_void_p
        lib.frt_imagewam_native_set_proprio_row.argtypes = [handle_p, ctypes.c_int32]
        lib.frt_imagewam_native_set_proprio_row.restype = ctypes.c_int
        lib.frt_imagewam_native_schema_records.argtypes = [
            handle_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        lib.frt_imagewam_native_schema_records.restype = ctypes.c_int
        lib.frt_imagewam_native_bind_declaration.argtypes = [handle_p, ctypes.c_void_p]
        lib.frt_imagewam_native_bind_declaration.restype = ctypes.c_int
        lib.frt_imagewam_native_verbs.argtypes = []
        lib.frt_imagewam_native_verbs.restype = ctypes.c_void_p
        self.lib = lib

    def function_address(self, name: str) -> int:
        """Address of an exported C function (for owner callbacks)."""
        return ctypes.cast(getattr(self.lib, name), ctypes.c_void_p).value
