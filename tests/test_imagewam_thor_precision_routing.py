"""Precision-routing contract for ``ImageWAMTorchFrontendThor``.

For every precision in ``imagewam_thor._PRECISIONS`` this asserts which
GEMM wrapper every weight slot routes to, through the frontend's own
code: the constructor's routing decisions (precision validation,
``dims["merge_qkv_mlp"]``), ``_wrap_linear``, ``_alloc_random_weights``
and ``_load_real_weights``. It checks the real FLUX.2-4B LIBERO dims and
the frontend's default dims, and both weight sources.

``EXPECTED_ROUTING`` below is the contract. A change that re-routes a
slot (a new precision, a new merged GEMM such as a ``linear2`` merge, a
new alignment fallback) updates that table in the same commit.

No GPU and no compiled extension are needed. As in
``tests/test_pi05_thor_fp4_routing.py``, the stubs sit at the import
boundary:

* ``flash_rt.flash_rt_kernels`` is a stub whose ``FvkContext()`` raises
  ``NoGpuContext``. The constructor makes every routing decision before
  it creates the kernel context, so catching ``NoGpuContext`` leaves an
  instance that holds the real decisions and no GPU state.
* ``flash_rt.models.imagewam.quant_linear`` is replaced by recording
  classes with the real constructor signatures
  (``test_stub_signatures_match_quant_linear`` keeps them honest wherever
  the real module imports).
* The frontend's ``DEV`` is ``"meta"``, so real-dim weights allocate no
  memory.

The frontend is imported fresh under the stubs and every module the
import added or replaced is restored afterwards.
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import sys
import types
from collections.abc import Iterator

import pytest
import torch

import flash_rt

FRONTEND_MODULE = "flash_rt.frontends.torch.imagewam_thor"
KERNELS_MODULE = "flash_rt.flash_rt_kernels"
QUANT_LINEAR_MODULE = "flash_rt.models.imagewam.quant_linear"

# ── the contract ────────────────────────────────────────────────────────
#
# Route labels:
#   f16        Fp16Linear (cuBLASLt fp16)
#   bf16out    Bf16OutLinear (BF16 in/out; the residual-writing entry GEMMs)
#   cutlass16  CutlassFp16Linear
#   swiglu16   CutlassFp16SwiGluMlp (fused SwiGLU gate/up)
#   fp8        Fp8Linear (dynamic scale)
#   nvfp4      Nvfp4Linear
#   nvfp4sim   SimNvfp4Linear (NVFP4 numerics emulated with fp16 GEMMs)
#   e0m3h      E0m3HadamardLinear (Hadamard-rotated E0M3 weights and activations)
#   sfp8       StaticFp8Linear(use_cutlass=False)
#   sfp8c      StaticFp8Linear(use_cutlass=True)
#   swiglu4    Nvfp4SwiGluMlp (exists, currently routed to by no precision)
#   ptr        raw device pointer (QK-norm scales, action_encoder bias)
#   -          the slot must not exist for this precision
#
# action_encoder (K=action_dim=7) and head.linear (N=7) are the alignment
# fallbacks: every precision routes them to f16. Single-stream blocks use
# one merged linear1 GEMM (dims["merge_qkv_mlp"]) and one merged linear2
# GEMM (dims["merge_linear2"]) except under fp16_cutlass, which keeps
# qkv + mlp_in (so the SwiGLU slot can use CutlassFp16SwiGluMlp) and
# attn_out_proj + mlp_down.

PRECISION_COLUMNS = ("fp16", "fp16_cutlass", "fp8", "nvfp4", "fp8_static", "fp8_static_cutlass", "e0m3_hadamard",
                     "nvfp4_sim")
ABSENT = "-"

EXPECTED_ROUTING: dict[tuple[str, str, str], tuple[str, ...]] = {
    # (site, block, slot)                                fp16       fp16_cutlass  fp8       nvfp4     fp8_static  fp8_static_cutlass  e0m3_hadamard  nvfp4_sim
    ("backbone", "double", "txt_in.weight"):            ("bf16out", "bf16out",   "bf16out", "bf16out", "bf16out", "bf16out", "bf16out", "bf16out"),
    ("backbone", "double", "img_in.weight"):            ("bf16out", "bf16out",   "bf16out", "bf16out", "bf16out", "bf16out", "bf16out", "bf16out"),
    ("backbone", "double", "txt_qkv.weight"):           ("f16",     "cutlass16", "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "double", "img_qkv.weight"):           ("f16",     "cutlass16", "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "double", "txt_proj.weight"):          ("f16",     "cutlass16", "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "double", "img_proj.weight"):          ("f16",     "cutlass16", "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "double", "txt_mlp0.weight"):          ("f16",     "swiglu16",  "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "double", "img_mlp0.weight"):          ("f16",     "swiglu16",  "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "double", "txt_mlp2.weight"):          ("f16",     "cutlass16", "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "double", "img_mlp2.weight"):          ("f16",     "cutlass16", "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "double", "txt_query_norm"):           ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("backbone", "double", "txt_key_norm"):             ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("backbone", "double", "img_query_norm"):           ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("backbone", "double", "img_key_norm"):             ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("backbone", "single", "linear1.weight"):           ("f16",     ABSENT,      "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "single", "qkv.weight"):               (ABSENT,    "cutlass16", ABSENT,    ABSENT,    ABSENT,    ABSENT, ABSENT, ABSENT),
    ("backbone", "single", "mlp_in.weight"):            (ABSENT,    "swiglu16",  ABSENT,    ABSENT,    ABSENT,    ABSENT, ABSENT, ABSENT),
    ("backbone", "single", "linear2.weight"):           ("f16",     ABSENT,      "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("backbone", "single", "attn_out_proj.weight"):     (ABSENT,    "cutlass16", ABSENT,    ABSENT,    ABSENT,    ABSENT, ABSENT, ABSENT),
    ("backbone", "single", "mlp_down.weight"):          (ABSENT,    "cutlass16", ABSENT,    ABSENT,    ABSENT,    ABSENT, ABSENT, ABSENT),
    ("backbone", "single", "query_norm"):               ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("backbone", "single", "key_norm"):                 ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("action_dit", "shared", "action_encoder.weight"):  ("f16",     "f16",       "f16",     "f16",     "f16",     "f16", "f16", "f16"),
    ("action_dit", "shared", "action_encoder.bias"):    ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("action_dit", "shared", "head.linear.weight"):     ("f16",     "f16",       "f16",     "f16",     "f16",     "f16", "f16", "f16"),
    ("action_dit", "double", "qkv.weight"):             ("f16",     "cutlass16", "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("action_dit", "double", "proj.weight"):            ("f16",     "cutlass16", "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("action_dit", "double", "mlp0.weight"):            ("f16",     "swiglu16",  "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("action_dit", "double", "mlp2.weight"):            ("f16",     "cutlass16", "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("action_dit", "double", "query_norm"):             ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("action_dit", "double", "key_norm"):               ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("action_dit", "single", "linear1.weight"):         ("f16",     ABSENT,      "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("action_dit", "single", "qkv.weight"):             (ABSENT,    "cutlass16", ABSENT,    ABSENT,    ABSENT,    ABSENT, ABSENT, ABSENT),
    ("action_dit", "single", "mlp_in.weight"):          (ABSENT,    "swiglu16",  ABSENT,    ABSENT,    ABSENT,    ABSENT, ABSENT, ABSENT),
    ("action_dit", "single", "linear2.weight"):         ("f16",     ABSENT,      "fp8",     "nvfp4",   "sfp8",    "sfp8c", "e0m3h", "nvfp4sim"),
    ("action_dit", "single", "attn_out_proj.weight"):   (ABSENT,    "cutlass16", ABSENT,    ABSENT,    ABSENT,    ABSENT, ABSENT, ABSENT),
    ("action_dit", "single", "mlp_down.weight"):        (ABSENT,    "cutlass16", ABSENT,    ABSENT,    ABSENT,    ABSENT, ABSENT, ABSENT),
    ("action_dit", "single", "query_norm"):             ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
    ("action_dit", "single", "key_norm"):               ("ptr",     "ptr",       "ptr",     "ptr",     "ptr",     "ptr", "ptr", "ptr"),
}

# Real FLUX.2-4B LIBERO dims (benchmarks/imagewam_e2e_official_compare.py's
# REAL_DIMS, structural subset). Every weight except action_encoder (K=7)
# and head.linear (N=7) is 16-aligned here, so the fallbacks above are the
# only alignment-driven routes.
REAL_DIMS = dict(
    hidden=3072, HD=128, NH=24, mlp_hidden=9216, joint_attention_dim=7680,
    x0=513, a0=905, num_layers_double=5, num_layers_single=20,
    action_hidden_dim=1024, action_attn_width=3072, action_mlp_hidden=4096,
    action_dim=7, num_action=64, total=969,
    action_num_layers_double=5, action_num_layers_single=20,
    ref_h=14, ref_w=28,
)
DIMS_CASES = {"real": REAL_DIMS, "default": None}


# ── stubs ───────────────────────────────────────────────────────────────


class NoGpuContext(Exception):
    """Raised by the stub ``FvkContext``: the frontend stops at its GPU boundary."""


class _StubGemmRunner:
    """Stands in for ``fvk.GemmRunner`` as the ``gemm`` argument of wrappers."""


def _kernels_stub() -> types.ModuleType:
    module = types.ModuleType(KERNELS_MODULE)

    class FvkContext:
        def __init__(self) -> None:
            raise NoGpuContext("stub flash_rt_kernels: no GPU context in the routing test")

    module.FvkContext = FvkContext
    module.GemmRunner = _StubGemmRunner
    return module


_POS = inspect.Parameter.POSITIONAL_OR_KEYWORD
_KW = inspect.Parameter.KEYWORD_ONLY

# Real constructor signatures of flash_rt/models/imagewam/quant_linear.py:
# class -> (route label, ((param, kind, default), ...)).
QUANT_LINEAR_STUBS: dict[str, tuple[str, tuple[tuple[str, inspect._ParameterKind, object], ...]]] = {
    "Fp16Linear": ("f16", (("gemm", _POS, inspect.Parameter.empty),
                           ("weight_ptr", _POS, inspect.Parameter.empty),
                           ("n", _POS, inspect.Parameter.empty), ("k", _POS, inspect.Parameter.empty))),
    "Bf16OutLinear": ("bf16out", (("gemm", _POS, inspect.Parameter.empty),
                                  ("weight_ptr", _POS, inspect.Parameter.empty),
                                  ("n", _POS, inspect.Parameter.empty), ("k", _POS, inspect.Parameter.empty))),
    "CutlassFp16Linear": ("cutlass16", (("weight_fp16_ptr", _POS, inspect.Parameter.empty),
                                        ("n", _POS, inspect.Parameter.empty), ("k", _POS, inspect.Parameter.empty),
                                        ("variant", _KW, None))),
    "CutlassFp16SwiGluMlp": ("swiglu16", (("merged_weight_fp16_ptr", _POS, inspect.Parameter.empty),
                                          ("mlp_hidden", _POS, inspect.Parameter.empty),
                                          ("k", _POS, inspect.Parameter.empty))),
    "Fp8Linear": ("fp8", (("weight_fp16_ptr", _POS, inspect.Parameter.empty),
                          ("n", _POS, inspect.Parameter.empty), ("k", _POS, inspect.Parameter.empty),
                          ("layout", _KW, None))),
    "StaticFp8Linear": ("sfp8", (("weight_fp16_ptr", _POS, inspect.Parameter.empty),
                                 ("n", _POS, inspect.Parameter.empty), ("k", _POS, inspect.Parameter.empty),
                                 ("use_cutlass", _KW, False), ("layout", _KW, None))),
    "Nvfp4Linear": ("nvfp4", (("weight_fp16_ptr", _POS, inspect.Parameter.empty),
                              ("n", _POS, inspect.Parameter.empty), ("k", _POS, inspect.Parameter.empty),
                              ("awq_inv_s", _KW, None))),
    "E0m3HadamardLinear": ("e0m3h", (("weight_fp16_ptr", _POS, inspect.Parameter.empty),
                                     ("n", _POS, inspect.Parameter.empty), ("k", _POS, inspect.Parameter.empty))),
    "SimNvfp4Linear": ("nvfp4sim", (("gemm", _POS, inspect.Parameter.empty),
                                    ("weight_fp16_ptr", _POS, inspect.Parameter.empty),
                                    ("n", _POS, inspect.Parameter.empty), ("k", _POS, inspect.Parameter.empty),
                                    ("awq_inv_s", _KW, None))),
    "Nvfp4SwiGluMlp": ("swiglu4", (("merged_weight_fp16_ptr", _POS, inspect.Parameter.empty),
                                   ("mlp_hidden", _POS, inspect.Parameter.empty),
                                   ("k", _POS, inspect.Parameter.empty))),
}
_SHAPE_PARAMS = ("n", "k", "mlp_hidden")


def _recording_class(class_name: str, label: str,
                     params: tuple[tuple[str, inspect._ParameterKind, object], ...]) -> type:
    """A stub wrapper that binds its arguments against the real signature."""
    signature = inspect.Signature(
        [inspect.Parameter(name, kind, default=default) for name, kind, default in params])

    def __init__(self, *args: object, **kwargs: object) -> None:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        self.arguments = dict(bound.arguments)
        self.shape = tuple(int(v) for name, v in self.arguments.items() if name in _SHAPE_PARAMS)
        self.route = label
        if class_name == "StaticFp8Linear" and self.arguments["use_cutlass"]:
            self.route = "sfp8c"

    __init__.__signature__ = inspect.Signature(
        [inspect.Parameter("self", _POS)] + list(signature.parameters.values()))
    return type(class_name, (), {"__init__": __init__, "__module__": QUANT_LINEAR_MODULE})


def _quant_linear_stub() -> types.ModuleType:
    module = types.ModuleType(QUANT_LINEAR_MODULE)
    for class_name, (label, params) in QUANT_LINEAR_STUBS.items():
        setattr(module, class_name, _recording_class(class_name, label, params))
    return module


def route_of(value: object) -> str:
    """Route label of one ``weights`` value."""
    if isinstance(value, int):
        return "ptr"
    return getattr(value, "route", f"unexpected:{type(value).__name__}")


def _restore_modules(saved: dict[str, types.ModuleType]) -> None:
    """Put ``sys.modules`` and parent-package attributes back as saved."""
    touched = [name for name, module in list(sys.modules.items()) if saved.get(name) is not module]
    touched += [name for name in saved if name not in sys.modules]
    for name in touched:
        if name in saved:
            sys.modules[name] = saved[name]
        else:
            sys.modules.pop(name, None)
    for name in touched:
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        if name in saved:
            setattr(parent, child, saved[name])
        elif child in vars(parent):
            delattr(parent, child)


@contextlib.contextmanager
def stubbed_frontend_module() -> Iterator[types.ModuleType]:
    """Import the frontend fresh against the stubs; restore afterwards."""
    saved = dict(sys.modules)
    saved_kernels_attr = vars(flash_rt).get("flash_rt_kernels")
    kernels = _kernels_stub()
    try:
        sys.modules[KERNELS_MODULE] = kernels
        # `import flash_rt.flash_rt_kernels as fvk` resolves through the
        # package attribute first, so the attribute is patched as well.
        flash_rt.flash_rt_kernels = kernels
        sys.modules[QUANT_LINEAR_MODULE] = _quant_linear_stub()
        sys.modules.pop(FRONTEND_MODULE, None)
        module = importlib.import_module(FRONTEND_MODULE)
        module.DEV = "meta"
        yield module
    finally:
        _restore_modules(saved)
        if saved_kernels_attr is not None:
            flash_rt.flash_rt_kernels = saved_kernels_attr
        elif "flash_rt_kernels" in vars(flash_rt):
            del flash_rt.flash_rt_kernels


@pytest.fixture
def frontend_module() -> Iterator[types.ModuleType]:
    with stubbed_frontend_module() as module:
        yield module


def construct_until_gpu(module: types.ModuleType, precision: str, dims: dict | None) -> object:
    """Run the real constructor up to its first kernel-context call."""
    cls = module.ImageWAMTorchFrontendThor
    frontend = cls.__new__(cls)
    with pytest.raises(NoGpuContext):
        frontend.__init__(dims_override=dict(dims) if dims else None, precision=precision)
    frontend._gemm = _StubGemmRunner()
    return frontend


def fake_real_state_dict(d: dict) -> dict[str, torch.Tensor]:
    """Meta tensors under the real checkpoint key names, real (out, in) shapes."""
    hidden, hd, mlp = d["hidden"], d["HD"], d["mlp_hidden"]
    ahd, aaw, amh, adim = d["action_hidden_dim"], d["action_attn_width"], d["action_mlp_hidden"], d["action_dim"]

    def t(*shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=torch.bfloat16, device="meta")

    video = "mixtures.video.transformer"
    sd = {f"{video}.txt_in.weight": t(hidden, d["joint_attention_dim"]),
          f"{video}.img_in.weight": t(hidden, hd)}
    for i in range(d["num_layers_double"]):
        for side in ("txt", "img"):
            p = f"{video}.double_blocks.{i}"
            sd[f"{p}.{side}_attn.qkv.weight"] = t(3 * hidden, hidden)
            sd[f"{p}.{side}_attn.proj.weight"] = t(hidden, hidden)
            sd[f"{p}.{side}_attn.norm.query_norm.scale"] = t(hd)
            sd[f"{p}.{side}_attn.norm.key_norm.scale"] = t(hd)
            sd[f"{p}.{side}_mlp.0.weight"] = t(2 * mlp, hidden)
            sd[f"{p}.{side}_mlp.2.weight"] = t(hidden, mlp)
    for i in range(d["num_layers_single"]):
        p = f"{video}.single_blocks.{i}"
        sd[f"{p}.linear1.weight"] = t(3 * hidden + 2 * mlp, hidden)
        sd[f"{p}.linear2.weight"] = t(hidden, hidden + mlp)
        sd[f"{p}.norm.query_norm.scale"] = t(hd)
        sd[f"{p}.norm.key_norm.scale"] = t(hd)
    action = "mixtures.action"
    sd[f"{action}.action_encoder.weight"] = t(ahd, adim)
    sd[f"{action}.action_encoder.bias"] = t(ahd)
    sd[f"{action}.head.linear.weight"] = t(adim, ahd)
    for i in range(d["action_num_layers_double"]):
        p = f"{action}.double_blocks.{i}"
        sd[f"{p}.img_attn.qkv.weight"] = t(3 * aaw, ahd)
        sd[f"{p}.img_attn.proj.weight"] = t(ahd, aaw)
        sd[f"{p}.img_attn.norm.query_norm.scale"] = t(hd)
        sd[f"{p}.img_attn.norm.key_norm.scale"] = t(hd)
        sd[f"{p}.img_mlp.0.weight"] = t(2 * amh, ahd)
        sd[f"{p}.img_mlp.2.weight"] = t(ahd, amh)
    for i in range(d["action_num_layers_single"]):
        p = f"{action}.single_blocks.{i}"
        sd[f"{p}.linear1.weight"] = t(3 * aaw + 2 * amh, ahd)
        sd[f"{p}.linear2.weight"] = t(ahd, aaw + amh)
        sd[f"{p}.norm.query_norm.scale"] = t(hd)
        sd[f"{p}.norm.key_norm.scale"] = t(hd)
    return sd


def routing_mismatches(weights: dict, dims: dict, precision: str) -> list[str]:
    """Every difference between ``weights`` and ``EXPECTED_ROUTING``."""
    column = PRECISION_COLUMNS.index(precision)
    layer_counts = {
        ("backbone", "double"): dims["num_layers_double"],
        ("backbone", "single"): dims["num_layers_single"],
        ("action_dit", "shared"): 1,
        ("action_dit", "double"): dims["action_num_layers_double"],
        ("action_dit", "single"): dims["action_num_layers_single"],
    }
    problems = []
    for (site, block, layer, slot), value in sorted(weights.items(), key=lambda kv: str(kv[0])):
        row = EXPECTED_ROUTING.get((site, block, slot))
        got = route_of(value)
        if row is None:
            problems.append(f"slot not in EXPECTED_ROUTING: {(site, block, layer, slot)} -> {got}")
        elif got != row[column]:
            problems.append(f"{(site, block, layer, slot)}: expected {row[column]}, got {got}")
    for (site, block, slot), row in EXPECTED_ROUTING.items():
        if row[column] == ABSENT:
            continue
        for layer in range(layer_counts[(site, block)]):
            if (site, block, layer, slot) not in weights:
                problems.append(f"missing slot {(site, block, layer, slot)} (expected {row[column]})")
    return problems


# ── tests ───────────────────────────────────────────────────────────────


def test_contract_covers_every_frontend_precision(frontend_module):
    assert PRECISION_COLUMNS == tuple(frontend_module._PRECISIONS)
    assert all(len(row) == len(PRECISION_COLUMNS) for row in EXPECTED_ROUTING.values())
    labels = {label for label, _ in QUANT_LINEAR_STUBS.values()} | {"sfp8c", "ptr", ABSENT}
    assert {label for row in EXPECTED_ROUTING.values() for label in row} <= labels


def test_unknown_precision_is_rejected_before_any_gpu_work(frontend_module):
    cls = frontend_module.ImageWAMTorchFrontendThor
    with pytest.raises(ValueError, match="precision="):
        cls(precision="int4")


@pytest.mark.parametrize("precision", PRECISION_COLUMNS)
def test_constructor_merge_decision_matches_contract(frontend_module, precision):
    frontend = construct_until_gpu(frontend_module, precision, REAL_DIMS)
    column = PRECISION_COLUMNS.index(precision)
    expect_merged = EXPECTED_ROUTING[("backbone", "single", "linear1.weight")][column] != ABSENT
    assert frontend.dims["merge_qkv_mlp"] is expect_merged
    expect_merged2 = EXPECTED_ROUTING[("backbone", "single", "linear2.weight")][column] != ABSENT
    assert frontend.dims["merge_linear2"] is expect_merged2
    assert frontend._precision == precision


@pytest.mark.parametrize("dims_case", sorted(DIMS_CASES))
@pytest.mark.parametrize("precision", PRECISION_COLUMNS)
def test_random_weight_routing(frontend_module, precision, dims_case):
    frontend = construct_until_gpu(frontend_module, precision, DIMS_CASES[dims_case])
    weights = frontend._alloc_random_weights(frontend.dims)
    problems = routing_mismatches(weights, frontend.dims, precision)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("dims_case", sorted(DIMS_CASES))
@pytest.mark.parametrize("precision", PRECISION_COLUMNS)
def test_real_checkpoint_routing(frontend_module, precision, dims_case):
    frontend = construct_until_gpu(frontend_module, precision, DIMS_CASES[dims_case])
    weights = frontend._load_real_weights(frontend.dims, fake_real_state_dict(frontend.dims))
    problems = routing_mismatches(weights, frontend.dims, precision)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("precision", PRECISION_COLUMNS)
def test_random_and_real_paths_build_the_same_gemms(frontend_module, precision):
    """Same keys, same wrapper, same (n, k) from both weight sources."""
    frontend = construct_until_gpu(frontend_module, precision, REAL_DIMS)
    random_w = frontend._alloc_random_weights(frontend.dims)
    real_w = frontend._load_real_weights(frontend.dims, fake_real_state_dict(frontend.dims))
    assert set(random_w) == set(real_w)
    differ = {key: ((route_of(random_w[key]), getattr(random_w[key], "shape", None)),
                    (route_of(real_w[key]), getattr(real_w[key], "shape", None)))
              for key in random_w
              if (route_of(random_w[key]), getattr(random_w[key], "shape", None))
              != (route_of(real_w[key]), getattr(real_w[key], "shape", None))}
    assert not differ, differ


@pytest.mark.parametrize("precision", PRECISION_COLUMNS)
def test_alignment_fallback_slots_have_the_misaligned_shapes(frontend_module, precision):
    """The K=7 / N=7 fallbacks are exercised at the real action width."""
    frontend = construct_until_gpu(frontend_module, precision, REAL_DIMS)
    weights = frontend._alloc_random_weights(frontend.dims)
    encoder = weights[("action_dit", "shared", 0, "action_encoder.weight")]
    head = weights[("action_dit", "shared", 0, "head.linear.weight")]
    assert (route_of(encoder), encoder.shape) == ("f16", (1024, 7))
    assert (route_of(head), head.shape) == ("f16", (7, 1024))


def test_merged_linear1_width_is_qkv_plus_gate_up(frontend_module):
    frontend = construct_until_gpu(frontend_module, "nvfp4", REAL_DIMS)
    weights = frontend._load_real_weights(frontend.dims, fake_real_state_dict(frontend.dims))
    assert weights[("backbone", "single", 0, "linear1.weight")].shape == (3 * 3072 + 2 * 9216, 3072)
    assert weights[("action_dit", "single", 0, "linear1.weight")].shape == (3 * 3072 + 2 * 4096, 1024)


def test_merged_linear2_width_is_attn_plus_mlp(frontend_module):
    """(n, k) of the merged linear2: N = residual width, K = attn width + mlp_hidden."""
    frontend = construct_until_gpu(frontend_module, "nvfp4", REAL_DIMS)
    weights = frontend._load_real_weights(frontend.dims, fake_real_state_dict(frontend.dims))
    assert weights[("backbone", "single", 0, "linear2.weight")].shape == (3072, 3072 + 9216)
    assert weights[("action_dit", "single", 0, "linear2.weight")].shape == (1024, 3072 + 4096)


def test_linear2_merge_can_be_turned_off_for_ab(frontend_module):
    """dims_override={"merge_linear2": False} restores the split slots (A/B path);
    requesting the merge under fp16_cutlass (split linear1) is rejected."""
    frontend = construct_until_gpu(frontend_module, "nvfp4", dict(REAL_DIMS, merge_linear2=False))
    weights = frontend._alloc_random_weights(frontend.dims)
    keys = {slot for (site, block, _, slot) in weights if block == "single"}
    assert {"attn_out_proj.weight", "mlp_down.weight", "linear1.weight"} <= keys
    assert "linear2.weight" not in keys
    cls = frontend_module.ImageWAMTorchFrontendThor
    with pytest.raises(ValueError, match="merge_linear2"):
        cls(dims_override=dict(REAL_DIMS, merge_linear2=True), precision="fp16_cutlass")


def test_cutlass_swiglu_split_slots_carry_mlp_hidden(frontend_module):
    frontend = construct_until_gpu(frontend_module, "fp16_cutlass", REAL_DIMS)
    weights = frontend._load_real_weights(frontend.dims, fake_real_state_dict(frontend.dims))
    assert weights[("backbone", "single", 0, "mlp_in.weight")].shape == (9216, 3072)
    assert weights[("backbone", "double", 0, "img_mlp0.weight")].shape == (9216, 3072)
    assert weights[("action_dit", "double", 0, "mlp0.weight")].shape == (4096, 1024)
    assert weights[("backbone", "single", 0, "qkv.weight")].shape == (3 * 3072, 3072)


# AWQ fold A slots (flash_rt/models/imagewam/awq.py): the GEMMs whose input
# is an AdaLN output carry awq_inv_s; every other NVFP4 GEMM does not (the
# down projections get their 1/s through the preceding up columns).
AWQ_FOLD_A_SLOTS = {
    ("backbone", "double"): {"txt_qkv.weight", "img_qkv.weight", "txt_mlp0.weight", "img_mlp0.weight"},
    ("backbone", "single"): {"linear1.weight"},
    ("action_dit", "double"): {"qkv.weight", "mlp0.weight"},
    ("action_dit", "single"): {"linear1.weight"},
}


@pytest.mark.parametrize("precision", ("nvfp4", "nvfp4_sim"))
def test_nvfp4_awq_routing(frontend_module, precision):
    """`nvfp4_awq=True` keeps every route of the precision column and adds
    `awq_inv_s` exactly on the fold-A slots (synthetic per-channel
    statistics; the constructor's calibration-file check is bypassed by
    setting the parsed file directly)."""
    import types as _types

    import numpy as np

    from flash_rt.models.imagewam.checkpoint_loader import build_real_weights

    frontend = construct_until_gpu(frontend_module, precision, REAL_DIMS)
    d = frontend.dims
    sd = fake_real_state_dict(d)
    raw = build_real_weights(
        sd, num_double=d["num_layers_double"], num_single=d["num_layers_single"],
        action_num_double=d["action_num_layers_double"], action_num_single=d["action_num_layers_single"],
        action_attn_width=d["action_attn_width"], merge_qkv_mlp=d["merge_qkv_mlp"],
        merge_linear2=d["merge_linear2"])
    rng = np.random.default_rng(0)
    sites = {".".join(str(p) for p in key): _types.SimpleNamespace(
                 channel_amax=rng.random(t.shape[0]).astype(np.float32) + 0.1)
             for key, t in raw.items() if t.ndim == 2}
    frontend._calibration = _types.SimpleNamespace(sites=sites)
    frontend._nvfp4_awq, frontend._awq_alpha, frontend._awq_scope = True, 0.5, "adaln+down"
    weights = frontend._load_real_weights(d, sd)
    assert not routing_mismatches(weights, d, precision)
    with_inv_s = {key for key, v in weights.items()
                  if getattr(v, "arguments", {}).get("awq_inv_s") is not None}
    expected = {key for key in weights if key[3] in AWQ_FOLD_A_SLOTS.get(key[:2], set())}
    assert with_inv_s == expected, sorted(with_inv_s ^ expected)[:5]


@pytest.mark.parametrize("precision", PRECISION_COLUMNS)
def test_real_entry_projections_are_shared_across_double_layers(frontend_module, precision):
    """One txt_in / img_in wrapper serves every double layer (FLUX.2 has one of each)."""
    frontend = construct_until_gpu(frontend_module, precision, REAL_DIMS)
    weights = frontend._load_real_weights(frontend.dims, fake_real_state_dict(frontend.dims))
    for slot in ("txt_in.weight", "img_in.weight"):
        wrappers = {id(weights[("backbone", "double", layer, slot)]) for layer in range(5)}
        assert len(wrappers) == 1, slot


def test_stub_signatures_match_quant_linear():
    """The stubs prove routing only if they mirror the real constructors."""
    # exc_type=ImportError: a flash_rt_kernels .so that is present but does
    # not load here (wrong ABI) raises ImportError, not ModuleNotFoundError,
    # and must skip too.
    real = pytest.importorskip(
        QUANT_LINEAR_MODULE, reason="real quant_linear needs a loadable flash_rt_kernels",
        exc_type=ImportError)
    for class_name, (_, params) in QUANT_LINEAR_STUBS.items():
        real_params = [(p.name, p.kind, p.default)
                       for p in inspect.signature(getattr(real, class_name)).parameters.values()]
        assert real_params == list(params), class_name


def test_stubbed_import_leaves_modules_untouched():
    before = {name: sys.modules.get(name) for name in (FRONTEND_MODULE, KERNELS_MODULE, QUANT_LINEAR_MODULE)}
    kernels_attr = vars(flash_rt).get("flash_rt_kernels")
    with stubbed_frontend_module() as module:
        assert module.ImageWAMTorchFrontendThor.__module__ == FRONTEND_MODULE
        assert sys.modules[QUANT_LINEAR_MODULE].Nvfp4Linear.__module__ == QUANT_LINEAR_MODULE
    after = {name: sys.modules.get(name) for name in before}
    assert after == before
    assert vars(flash_rt).get("flash_rt_kernels") is kernels_attr
