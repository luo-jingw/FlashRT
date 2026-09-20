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
        ("num_text_lengths", ctypes.c_uint32),
        ("text_lengths", ctypes.POINTER(ctypes.c_uint32)),
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


class ImageWAMLinear(ctypes.Structure):
    """`frt_imagewam_linear`."""

    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("n", ctypes.c_int32),
        ("k", ctypes.c_int32),
        ("fp4_variant", ctypes.c_int32),
        ("weight", ctypes.c_void_p),
        ("weight_scales", ctypes.c_void_p),
        ("act_packed", ctypes.c_void_p),
        ("act_scales", ctypes.c_void_p),
    ]


class ImageWAMAdaLN(ctypes.Structure):
    """`frt_imagewam_adaln`."""

    _fields_ = [("shift", ctypes.c_void_p), ("scale", ctypes.c_void_p), ("gate", ctypes.c_void_p),
                ("shift_f32", ctypes.c_void_p), ("scale_f32", ctypes.c_void_p),
                ("gate_f32", ctypes.c_void_p)]


class ImageWAMDoubleLayer(ctypes.Structure):
    """`frt_imagewam_double_layer`."""

    _fields_ = [(name, ImageWAMLinear) for name in (
        "txt_qkv", "img_qkv", "txt_proj", "img_proj", "txt_mlp0", "img_mlp0", "txt_mlp2", "img_mlp2")] + [
        (name, ctypes.c_void_p) for name in (
            "txt_query_norm", "txt_key_norm", "img_query_norm", "img_key_norm")]


class ImageWAMSingleLayer(ctypes.Structure):
    """`frt_imagewam_single_layer`."""

    _fields_ = [("linear1", ImageWAMLinear), ("attn_out_proj", ImageWAMLinear),
                ("mlp_down", ImageWAMLinear), ("linear2", ImageWAMLinear),
                ("query_norm", ctypes.c_void_p), ("key_norm", ctypes.c_void_p)]


class ImageWAMActionDoubleLayer(ctypes.Structure):
    """`frt_imagewam_action_double_layer`."""

    _fields_ = [("qkv", ImageWAMLinear), ("proj", ImageWAMLinear), ("mlp0", ImageWAMLinear),
                ("mlp2", ImageWAMLinear), ("query_norm", ctypes.c_void_p),
                ("key_norm", ctypes.c_void_p)]


class ImageWAMActionStep(ctypes.Structure):
    """`frt_imagewam_action_step`."""

    _fields_ = [("double1", ImageWAMAdaLN), ("double2", ImageWAMAdaLN), ("single", ImageWAMAdaLN),
                ("head", ImageWAMAdaLN), ("delta", ctypes.c_float), ("reserved", ctypes.c_uint32)]


PIPELINE_DIM_FIELDS = (
    "hidden", "head_dim", "num_heads", "mlp_hidden", "joint_attention_dim",
    "x0", "a0", "total", "num_action", "action_dim",
    "action_hidden_dim", "action_attn_width", "action_mlp_hidden",
    "num_double", "num_single", "action_num_double", "action_num_single", "num_steps",
    "merge_linear2", "fuse_res_norm",
)
PIPELINE_BUFFER_FIELDS = (
    "context", "backbone_hidden", "img_raw", "modded_scratch",
    "txt_qkv_merged", "img_qkv_merged", "single_linear1_merged", "action_linear1_merged",
    "action_qkv_merged", "action_latent_fp16", "velocity", "head_modded",
    "txt_mlp_merged", "txt_mlp_gated", "img_mlp_merged", "img_mlp_gated", "single_mlp_gated",
    "proj_scratch", "proj_scratch2", "action_latent", "action_hidden", "action_modded",
    "action_proj_scratch", "action_proj_scratch2", "action_mlp_merged", "action_mlp_gated",
    "single_linear2_in", "action_linear2_in",
)


class ImageWAMPipelineConfig(ctypes.Structure):
    """`frt_imagewam_pipeline_config`."""

    _fields_ = (
        [("struct_size", ctypes.c_uint32)]
        + [(name, ctypes.c_int32) for name in PIPELINE_DIM_FIELDS]
        + [("eps", ctypes.c_float)]
        + [(name, ctypes.c_void_p) for name in PIPELINE_BUFFER_FIELDS]
        + [("q_o", ctypes.c_void_p), ("k_cache", ctypes.c_void_p), ("v_cache", ctypes.c_void_p),
           ("logits", ctypes.c_void_p), ("kv_layer_stride_bytes", ctypes.c_uint64),
           ("attn_scale", ctypes.c_float), ("reserved", ctypes.c_uint32),
           ("rope_table", ctypes.c_void_p), ("action_rope_table", ctypes.c_void_p)]
        + [(name, ImageWAMLinear) for name in ("txt_in", "img_in", "action_encoder", "head_linear")]
        + [("action_encoder_bias", ctypes.c_void_p)]
        + [(name, ImageWAMAdaLN) for name in ("txt_mod1", "txt_mod2", "img_mod1", "img_mod2", "single_mod")]
        + [("double_layers", ctypes.POINTER(ImageWAMDoubleLayer)),
           ("single_layers", ctypes.POINTER(ImageWAMSingleLayer)),
           ("action_double_layers", ctypes.POINTER(ImageWAMActionDoubleLayer)),
           ("action_single_layers", ctypes.POINTER(ImageWAMSingleLayer)),
           ("steps", ctypes.POINTER(ImageWAMActionStep))]
    )


class ImageWAMGemmShape(ctypes.Structure):
    """`frt_imagewam_gemm_shape`."""

    _fields_ = [("kind", ctypes.c_int32), ("m", ctypes.c_int32), ("n", ctypes.c_int32),
                ("k", ctypes.c_int32)]


SEGMENT_DOUBLE_LAYER = 0
SEGMENT_SINGLE_LAYER = 1
SEGMENT_PREFILL = 2
SEGMENT_DENOISE_STEP = 3
SEGMENT_DENOISE = 4
SEGMENT_FULL = 5


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
        lib.frt_imagewam_native_use_graph.argtypes = [handle_p, ctypes.c_uint64, ctypes.c_void_p]
        lib.frt_imagewam_native_use_graph.restype = ctypes.c_int
        lib.frt_imagewam_native_has_variant.argtypes = [handle_p, ctypes.c_uint64]
        lib.frt_imagewam_native_has_variant.restype = ctypes.c_int
        lib.frt_imagewam_native_variant_exec.argtypes = [handle_p, ctypes.c_uint64]
        lib.frt_imagewam_native_variant_exec.restype = ctypes.c_void_p
        lib.frt_imagewam_native_set_text_length.argtypes = [handle_p, ctypes.c_uint64]
        lib.frt_imagewam_native_set_text_length.restype = ctypes.c_int
        lib.frt_imagewam_native_text_length.argtypes = [handle_p]
        lib.frt_imagewam_native_text_length.restype = ctypes.c_uint64
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
        lib.frt_imagewam_native_abi_sizes.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        lib.frt_imagewam_native_abi_sizes.restype = None
        lib.frt_imagewam_native_set_pipeline.argtypes = [handle_p, ctypes.POINTER(ImageWAMPipelineConfig)]
        lib.frt_imagewam_native_set_pipeline.restype = ctypes.c_int
        lib.frt_imagewam_native_gemm_shapes.argtypes = [
            handle_p, ctypes.POINTER(ImageWAMGemmShape), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        lib.frt_imagewam_native_gemm_shapes.restype = ctypes.c_int
        lib.frt_imagewam_native_set_gemm_algo.argtypes = [
            handle_p, ctypes.POINTER(ImageWAMGemmShape), ctypes.c_char_p, ctypes.c_uint64]
        lib.frt_imagewam_native_set_gemm_algo.restype = ctypes.c_int
        lib.frt_imagewam_native_run.argtypes = [handle_p, ctypes.c_uint32, ctypes.c_int32]
        lib.frt_imagewam_native_run.restype = ctypes.c_int
        lib.frt_imagewam_native_capture.argtypes = [handle_p]
        lib.frt_imagewam_native_capture.restype = ctypes.c_int
        lib.frt_imagewam_native_graph_nodes.argtypes = [handle_p, ctypes.POINTER(ctypes.c_uint64)]
        lib.frt_imagewam_native_graph_nodes.restype = ctypes.c_int
        self.lib = lib
        self._check_layout()

    def _check_layout(self) -> None:
        sizes = (ctypes.c_uint64 * 4)()
        self.lib.frt_imagewam_native_abi_sizes(sizes)
        expected = (ctypes.sizeof(ImageWAMIoConfig), ctypes.sizeof(ImageWAMPipelineConfig),
                    ctypes.sizeof(ImageWAMLinear), ctypes.sizeof(ImageWAMActionStep))
        if tuple(sizes) != expected:
            raise ImportError(f"{self.path}: config struct sizes {tuple(sizes)} differ from the "
                              f"ctypes mirror {expected}; rebuild the library")

    def function_address(self, name: str) -> int:
        """Address of an exported C function (for owner callbacks)."""
        return ctypes.cast(getattr(self.lib, name), ctypes.c_void_p).value
