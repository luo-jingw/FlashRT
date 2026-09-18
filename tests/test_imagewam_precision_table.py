"""The `Precision` property table: its values, and the frontend's use of it.

`flash_rt/models/imagewam/precision.py` holds, as data, what the frontend
(`flash_rt/frontends/torch/imagewam_thor.py`) decides: which precision needs
a calibration file, takes AWQ input scales, has a switchable CUTLASS tile,
falls back to the plain fp16 linear at which alignment, runs the merged
single-stream `linear1`, keeps the fused SwiGLU MLP, and autotunes the
`fp16_nn` backbone GEMMs.

Since plan phase W7 the frontend has no second copy of that knowledge: its
four module-level tuples are comprehensions over `Precision`/`PROPERTIES`,
`_wrap_linear` takes its fallback from `Precision.alignment_fallback`, and
the constructor, `_rnd_swiglu_mlp`, `_load_real_weights` and the
`fp16_nn_shapes=` call site read the matching properties. So this test pins
two things:

* the VALUES, as literal expectation tables in this file (the table is the
  source of truth, and a silent edit to it fails here);
* the DELEGATION, by parsing the frontend's SOURCE TEXT with `ast` (no import
  of the frontend, torch or the compiled kernels): every tuple is the
  comprehension over the property it must be, `_wrap_linear` keeps exactly one
  branch per enum member, takes its fallback decision from
  `alignment_fallback`, and carries no precision string constant of its own.

The libcudart-free, torch-free load of `precision.py` is checked at the end.
"""
from __future__ import annotations

import ast
import importlib.util
import itertools
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "flash_rt/frontends/torch/imagewam_thor.py"
RESOURCES_SRC = ROOT / "flash_rt/models/imagewam/pipeline_resources.py"
PRECISION_SRC = ROOT / "flash_rt/models/imagewam/precision.py"


def _load_precision_module():
    """Load precision.py by path: the package `__init__` imports torch."""
    spec = importlib.util.spec_from_file_location("_imagewam_precision_under_test", PRECISION_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses/enum need the module registered
    spec.loader.exec_module(mod)
    return mod


precision_mod = _load_precision_module()
Precision = precision_mod.Precision
ALL = tuple(Precision)

_FRONTEND_TREE = ast.parse(FRONTEND_SRC.read_text())


# -- the pin: literal expectation tables ---------------------------------------

# plan.md W3: the native GEMM of a tier needs both `n` and `k` of a (k, n)
# linear to be a multiple of this; 1 means no constraint (plain fp16).
EXPECTED_ALIGNMENT = {
    "fp16": 1, "fp16_cutlass": 8, "fp8": 8, "nvfp4": 16,
    "fp8_static": 8, "fp8_static_cutlass": 8, "e0m3_hadamard": 16, "nvfp4_sim": 16,
}
EXPECTED_NEEDS_CALIBRATION = frozenset({"fp8_static", "fp8_static_cutlass"})
EXPECTED_SUPPORTS_AWQ = frozenset({"nvfp4", "nvfp4_sim"})
EXPECTED_SUPPORTS_TILE_AUTOTUNE = frozenset({"nvfp4", "fp8_static_cutlass"})
# Every precision except `fp16_cutlass` merges qkv+mlp gate/up: that tier has
# its own fused SwiGLU MLP, which needs `mlp_in.weight` as a standalone tensor.
EXPECTED_MERGE_QKV_MLP = frozenset(
    p.value for p in ALL if p.value != "fp16_cutlass")
EXPECTED_FUSED_SWIGLU_MLP = frozenset({"fp16_cutlass"})
EXPECTED_FP16_NN_BACKBONE_GEMM = frozenset({"fp16"})
# `pipeline_resources.linear_resource` accepts Fp16Linear, Bf16OutLinear and
# Nvfp4Linear only, and the native pipeline records the merged linear1.
EXPECTED_SUPPORTS_NATIVE_RUNTIME = frozenset({"fp16", "nvfp4"})
# The classes `_wrap_linear` returns per branch (source-checked below).
EXPECTED_WRAP_LINEAR_CLASSES = {
    "fp16": frozenset({"Fp16Linear"}),
    "fp16_cutlass": frozenset({"Fp16Linear", "CutlassFp16Linear"}),
    "fp8": frozenset({"Fp16Linear", "Fp8Linear"}),
    "nvfp4": frozenset({"Fp16Linear", "Nvfp4Linear"}),
    "fp8_static": frozenset({"Fp16Linear", "StaticFp8Linear"}),
    "fp8_static_cutlass": frozenset({"Fp16Linear", "StaticFp8Linear"}),
    "e0m3_hadamard": frozenset({"Fp16Linear", "E0m3HadamardLinear"}),
    "nvfp4_sim": frozenset({"Fp16Linear", "SimNvfp4Linear"}),
}

_BY_NAME = {p.value: p for p in ALL}


# -- (a) the values -------------------------------------------------------------

def test_alignment_values():
    for name, alignment in EXPECTED_ALIGNMENT.items():
        assert _BY_NAME[name].alignment == alignment, name


def test_needs_calibration_values():
    for p in ALL:
        assert p.needs_calibration is (p.value in EXPECTED_NEEDS_CALIBRATION), p


def test_supports_awq_values():
    for p in ALL:
        assert p.supports_awq is (p.value in EXPECTED_SUPPORTS_AWQ), p


def test_supports_tile_autotune_values():
    for p in ALL:
        assert p.supports_tile_autotune is (p.value in EXPECTED_SUPPORTS_TILE_AUTOTUNE), p


def test_merge_and_fused_swiglu_values():
    for p in ALL:
        assert p.merge_qkv_mlp is (p.value in EXPECTED_MERGE_QKV_MLP), p
        assert p.merge_linear2_allowed is (p.value in EXPECTED_MERGE_QKV_MLP), p
        assert p.fused_swiglu_mlp is (p.value in EXPECTED_FUSED_SWIGLU_MLP), p
        # the fused MLP is a standalone weight, which is what excludes the merge
        assert not (p.fused_swiglu_mlp and p.merge_qkv_mlp), p


def test_fp16_nn_backbone_gemm_and_native_runtime_values():
    for p in ALL:
        assert p.fp16_nn_backbone_gemm is (p.value in EXPECTED_FP16_NN_BACKBONE_GEMM), p
        assert p.supports_native_runtime is (p.value in EXPECTED_SUPPORTS_NATIVE_RUNTIME), p


def test_members_and_plain_strings():
    assert tuple(p.value for p in ALL) == ("fp16", "fp16_cutlass", "fp8", "nvfp4", "fp8_static",
                                          "fp8_static_cutlass", "e0m3_hadamard", "nvfp4_sim")
    for p in ALL:
        assert p == p.value and isinstance(p, str)
        assert str(p) == p.value
        assert f"{p}" == p.value
        assert Precision(p.value) is p
    assert Precision("nvfp4") == "nvfp4"
    assert "nvfp4" in tuple(ALL)


def test_table_has_a_row_per_member():
    assert set(precision_mod.PROPERTIES) == set(ALL)


def test_accepts_calibration_path_matches_the_two_consumers():
    # `calibration_path` is legal iff the precision is static-FP8 (activation
    # scales) or, with nvfp4_awq, in the AWQ family (per-channel statistics).
    for p in ALL:
        assert p.accepts_calibration_path() is (p.value in EXPECTED_NEEDS_CALIBRATION), p
        assert p.accepts_calibration_path(nvfp4_awq=True) is (
            p.value in EXPECTED_NEEDS_CALIBRATION or p.value in EXPECTED_SUPPORTS_AWQ), p


# -- (b) fallback behaviour -----------------------------------------------------

def _grid():
    dims = [1, 7, 8, 9, 15, 16, 17, 24, 32, 48, 64, 100, 128, 256, 384, 1024, 3072, 4096, 12288]
    return list(itertools.product(dims, dims))


@pytest.mark.parametrize("p", ALL, ids=lambda p: p.value)
def test_alignment_fallback_matches_the_alignment(p):
    for n, k in _grid():
        assert p.alignment_fallback(n, k) is (n % p.alignment != 0 or k % p.alignment != 0), (p, n, k)


def test_alignment_fallback_named_cases():
    # action_encoder: K=7 (real LIBERO 7-DoF), head.linear: N=7
    for p in ALL:
        if p is Precision.FP16:
            assert not p.alignment_fallback(128, 7) and not p.alignment_fallback(7, 128)
        else:
            assert p.alignment_fallback(128, 7), p
            assert p.alignment_fallback(7, 128), p
    # 8-multiple that is not a 16-multiple: only the block-scaled tiers fall back
    for p in ALL:
        expect = p.value in ("nvfp4", "e0m3_hadamard", "nvfp4_sim")
        assert p.alignment_fallback(24, 8) is expect, p
        assert p.alignment_fallback(8, 24) is expect, p
    # both multiples of 16: nobody falls back
    for p in ALL:
        assert not p.alignment_fallback(3072, 4096), p
    # one misaligned dim is enough
    assert Precision.FP8.alignment_fallback(16, 9)
    assert Precision.NVFP4.alignment_fallback(15, 16)


@pytest.mark.parametrize("bad", ["bogus", "", "FP16", "nvfp4 ", "fp8_dynamic", None, 4])
def test_unknown_precision_raises_value_error(bad):
    with pytest.raises(ValueError):
        Precision(bad)


# -- frontend source: delegation, no second copy --------------------------------

def _assign_value(name: str) -> ast.AST:
    """Module-level `name = <expr>` in the frontend."""
    for node in _FRONTEND_TREE.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise AssertionError(f"{name} not found in {FRONTEND_SRC.name}")


def _method(name: str) -> ast.FunctionDef:
    for cls in (n for n in _FRONTEND_TREE.body if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == name:
                return fn
    raise AssertionError(f"method {name} not found")


def _tuple_comprehension(name: str) -> str | None:
    """`name = tuple(<gen>)` over `Precision`; returns the `if` attribute read
    on the loop variable (`p.supports_awq` -> "supports_awq"), or None for a
    plain `tuple(p.value for p in Precision)`."""
    value = _assign_value(name)
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "tuple", \
        f"{name} is no longer tuple(...)"
    gen = value.args[0]
    assert isinstance(gen, ast.GeneratorExp), f"{name} is not a generator comprehension"
    assert isinstance(gen.elt, ast.Attribute) and gen.elt.attr == "value", ast.dump(gen.elt)
    assert isinstance(gen.elt.value, ast.Name) and gen.elt.value.id == "p", ast.dump(gen.elt)
    assert len(gen.generators) == 1, name
    comp = gen.generators[0]
    assert isinstance(comp.iter, ast.Name) and comp.iter.id == "Precision", ast.dump(comp.iter)
    assert isinstance(comp.target, ast.Name) and comp.target.id == "p", ast.dump(comp.target)
    if not comp.ifs:
        return None
    assert len(comp.ifs) == 1, name
    cond = comp.ifs[0]
    assert isinstance(cond, ast.Attribute) and isinstance(cond.value, ast.Name) and cond.value.id == "p", \
        ast.dump(cond)
    return cond.attr


def _attribute_names(node: ast.AST, attr: str) -> bool:
    return any(isinstance(n, ast.Attribute) and n.attr == attr for n in ast.walk(node))


def test_module_tuples_are_comprehensions_over_the_table():
    assert _tuple_comprehension("_PRECISIONS") is None
    assert _tuple_comprehension("_NVFP4_PRECISIONS") == "supports_awq"
    assert _tuple_comprehension("_STATIC_FP8_PRECISIONS") == "needs_calibration"
    assert _tuple_comprehension("_VARIANT_TUNED_PRECISIONS") == "supports_tile_autotune"


def test_constructor_reads_merge_qkv_mlp_from_the_table():
    found = None
    for node in ast.walk(_method("__init__")):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].slice, ast.Constant)
                and node.targets[0].slice.value == "merge_qkv_mlp"):
            found = node.value
    assert found is not None, "merge_qkv_mlp assignment not found"
    assert isinstance(found, ast.Attribute) and found.attr == "merge_qkv_mlp", ast.dump(found)
    call = found.value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Precision", \
        ast.dump(call)
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "precision", ast.dump(call)


def test_swiglu_and_fp16_nn_sites_read_the_table():
    # `_rnd_swiglu_mlp`: `if not Precision(self._precision).fused_swiglu_mlp: return self._rnd_linear(...)`
    swiglu = _method("_rnd_swiglu_mlp")
    first_if = next(stmt for stmt in swiglu.body if isinstance(stmt, ast.If))
    assert _attribute_names(first_if.test, "fused_swiglu_mlp"), ast.dump(first_if.test)
    assert "_rnd_linear" in ast.dump(ast.Module(body=first_if.body, type_ignores=[]))
    # the real-weights path branches on the same property and builds the fused MLP
    lrw = _method("_load_real_weights")
    matches = [n for n in ast.walk(lrw)
               if isinstance(n, ast.If) and _attribute_names(n.test, "fused_swiglu_mlp")
               and "CutlassFp16SwiGluMlp" in ast.dump(ast.Module(body=n.body, type_ignores=[]))]
    assert len(matches) == 1, "the fused SwiGLU branch in _load_real_weights is gone or duplicated"
    # the backbone-GEMM autotune flag
    sites = [node.value for node in ast.walk(_FRONTEND_TREE)
             if isinstance(node, ast.keyword) and node.arg == "fp16_nn_shapes"]
    assert len(sites) == 1, f"expected one fp16_nn_shapes call site, found {len(sites)}"
    assert _attribute_names(sites[0], "fp16_nn_backbone_gemm"), ast.dump(sites[0])


def test_wrap_linear_takes_the_fallback_from_the_table():
    """One branch per enum member, one shared `fallback` decision from
    `alignment_fallback(n, k)`, no precision string constant of its own."""
    fn = _method("_wrap_linear")
    fallbacks = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "fallback" for t in n.targets)]
    assert len(fallbacks) == 1, f"expected one fallback assignment, found {len(fallbacks)}"
    call = fallbacks[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute), ast.dump(call)
    assert call.func.attr == "alignment_fallback", ast.dump(call)
    assert [a.id for a in call.args] == ["n", "k"], ast.dump(call)
    # the AWQ guard reads supports_awq
    assert _attribute_names(fn, "supports_awq"), "the awq_inv_s guard no longer reads supports_awq"

    branches: dict[str, set[str]] = {}
    for stmt in fn.body:
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Is)
                and isinstance(test.comparators[0], ast.Attribute)
                and isinstance(test.comparators[0].value, ast.Name)
                and test.comparators[0].value.id == "Precision"):
            continue
        member = test.comparators[0].attr
        classes = {n.value.func.id for n in ast.walk(stmt)
                   if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Name)}
        # every branch except fp16 returns the plain linear when the table says so
        guards = [n for n in ast.walk(stmt) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == "fallback"]
        assert len(guards) == (0 if member == "FP16" else 1), (member, len(guards))
        for g in guards:
            ret = g.body[0]
            assert isinstance(ret, ast.Return) and ret.value.func.id == "Fp16Linear", ast.dump(g)
        assert member not in branches, f"duplicate branch {member}"
        branches[member] = classes
    assert set(branches) == {p.name for p in ALL}, sorted(branches)
    for member, classes in branches.items():
        value = Precision[member].value
        assert classes == set(EXPECTED_WRAP_LINEAR_CLASSES[value]), (member, sorted(classes))
    # no precision string constant survives in the rewritten method
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in {p.value for p in ALL}, node.value


def test_native_runtime_support_matches_linear_resource_dispatch():
    accepted = _linear_resource_accepted_classes()
    assert {"Fp16Linear", "Bf16OutLinear", "Nvfp4Linear"} <= accepted
    merge = {p.value: p.merge_qkv_mlp for p in ALL}
    for p in ALL:
        classes = EXPECTED_WRAP_LINEAR_CLASSES[p.value]
        expected = classes <= accepted and merge[p.value]
        assert p.supports_native_runtime is expected, (p, sorted(classes))
    # the two precisions the native pipeline serves today
    assert {p.value for p in ALL if p.supports_native_runtime} == {"fp16", "nvfp4"}


def _linear_resource_accepted_classes() -> set[str]:
    tree = ast.parse(RESOURCES_SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "linear_resource")
    accepted = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance"
                and isinstance(node.args[1], ast.Name)):
            accepted.add(node.args[1].id)
    assert accepted, "no isinstance dispatch found in linear_resource"
    return accepted


def test_alignment_fallback_covers_awq_eligibility():
    # `_plan_awq` eligible: t.shape[0] % 16 == 0 and t.shape[1] % 16 == 0
    src = ast.dump(_method("_plan_awq"))
    assert src.count("Mod()") == 2 and "value=16" in src
    for p in ALL:
        if p.supports_awq:
            assert p.alignment == 16
            for n, k in _grid():
                assert p.alignment_fallback(n, k) is not (n % 16 == 0 and k % 16 == 0)


# -- leaf module ----------------------------------------------------------------

def test_module_imports_without_torch_or_frontend():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('p', {str(PRECISION_SRC)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['p'] = m; spec.loader.exec_module(m)\n"
        "bad = [n for n in ('torch', 'numpy', 'flash_rt', 'flash_rt.flash_rt_kernels') if n in sys.modules]\n"
        "assert not bad, bad\n"
        "assert m.Precision('nvfp4').supports_awq\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
