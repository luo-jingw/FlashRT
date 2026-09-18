"""The `Precision` property table against the frontend's own source.

`flash_rt/models/imagewam/precision.py` holds, as data, what
`flash_rt/frontends/torch/imagewam_thor.py` decides with string tuples and
an `if self._precision == ...` chain. This test parses the frontend's SOURCE
TEXT with `ast` (no import of the frontend, torch or the compiled kernels)
and checks that the table says the same thing:

* the enum members equal `_PRECISIONS` (order included);
* `needs_calibration`, `supports_awq`, `supports_tile_autotune` equal
  membership in `_STATIC_FP8_PRECISIONS`, `_NVFP4_PRECISIONS`,
  `_VARIANT_TUNED_PRECISIONS`;
* `merge_qkv_mlp`, `fused_swiglu_mlp`, `fp16_nn_backbone_gemm` equal the
  single-precision comparisons in the constructor / `_rnd_swiglu_mlp` /
  the `fp16_nn_shapes=` call site;
* `alignment_fallback(n, k)` equals the fallback rule extracted from each
  branch of `_wrap_linear`, on a grid of (n, k);
* `supports_native_runtime` equals "every linear class `_wrap_linear`
  returns for that precision is one `pipeline_resources.linear_resource`
  accepts", and the merged path holds.

When the frontend stops carrying these tuples (plan W7) the extraction
helpers here move to the new source of truth; the expectations do not.
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


def _module_tuple(name: str) -> tuple[str, ...]:
    for node in _FRONTEND_TREE.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple) and all(isinstance(v, str) for v in value), name
            return value
    raise AssertionError(f"{name} not found in {FRONTEND_SRC.name}")


def _method(name: str) -> ast.FunctionDef:
    for cls in (n for n in _FRONTEND_TREE.body if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == name:
                return fn
    raise AssertionError(f"method {name} not found")


def _is_self_precision(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "_precision" and \
        isinstance(node.value, ast.Name) and node.value.id == "self"


def _is_precision_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "precision"


def _single_compare(node: ast.AST, subject) -> tuple[type, str] | None:
    """`<subject> ==|!= "literal"` -> (op type, literal)."""
    if (isinstance(node, ast.Compare) and len(node.ops) == 1 and subject(node.left)
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
            and isinstance(node.comparators[0], ast.Constant)):
        return type(node.ops[0]), node.comparators[0].value
    return None


def _predicate_from_compare(node: ast.AST, subject) -> dict[str, bool]:
    """Evaluate a `subject ==|!= "lit"` expression for every precision."""
    found = _single_compare(node, subject)
    assert found is not None, ast.dump(node)
    op, lit = found
    return {p.value: ((p.value == lit) if op is ast.Eq else (p.value != lit)) for p in ALL}


# -- extraction from the source ------------------------------------------------

def _wrap_linear_branches() -> dict[str, dict]:
    """precision -> {"alignment": int | None, "classes": set[str]} from the
    top-level `if self._precision == "<p>":` blocks of `_wrap_linear`."""
    branches: dict[str, dict] = {}
    for stmt in _method("_wrap_linear").body:
        if not isinstance(stmt, ast.If):
            continue
        found = _single_compare(stmt.test, _is_self_precision)
        if found is None or found[0] is not ast.Eq:
            continue  # the awq_inv_s guard is not a precision branch
        name = found[1]
        alignment = None
        classes: set[str] = set()
        for node in ast.walk(stmt):
            if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.Or):
                mults = set()
                for cmp_ in node.test.values:
                    assert (isinstance(cmp_, ast.Compare) and isinstance(cmp_.ops[0], ast.NotEq)
                            and isinstance(cmp_.left, ast.BinOp) and isinstance(cmp_.left.op, ast.Mod)
                            and isinstance(cmp_.left.left, ast.Name) and cmp_.left.left.id in ("n", "k")
                            and isinstance(cmp_.left.right, ast.Constant)
                            and cmp_.comparators[0].value == 0), ast.dump(cmp_)
                    mults.add((cmp_.left.left.id, cmp_.left.right.value))
                assert {v for v, _ in mults} == {"n", "k"} and len({m for _, m in mults}) == 1, mults
                # the fallback must be the fp16 linear
                ret = node.body[0]
                assert isinstance(ret, ast.Return) and ret.value.func.id == "Fp16Linear", ast.dump(node)
                alignment = next(iter(mults))[1]
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call) and \
                    isinstance(node.value.func, ast.Name):
                classes.add(node.value.func.id)
        assert name not in branches, f"duplicate branch {name}"
        branches[name] = {"alignment": alignment, "classes": classes}
    return branches


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


def _merge_qkv_mlp_rule() -> dict[str, bool]:
    """`self.dims["merge_qkv_mlp"] = precision != "fp16_cutlass"` in __init__."""
    for node in ast.walk(_method("__init__")):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].slice, ast.Constant)
                and node.targets[0].slice.value == "merge_qkv_mlp"):
            return _predicate_from_compare(node.value, _is_precision_name)
    raise AssertionError("merge_qkv_mlp assignment not found")


def _fused_swiglu_rule() -> dict[str, bool]:
    """`_rnd_swiglu_mlp`: `if self._precision != "fp16_cutlass": return self._rnd_linear(...)`;
    the rest builds `CutlassFp16SwiGluMlp`. `_load_real_weights` uses the
    `self._precision == "fp16_cutlass"` form for the same class."""
    fn = _method("_rnd_swiglu_mlp")
    plain = None
    for stmt in fn.body:
        if isinstance(stmt, ast.If) and _single_compare(stmt.test, _is_self_precision):
            plain = _predicate_from_compare(stmt.test, _is_self_precision)
            assert isinstance(stmt.body[0], ast.Return)
            assert "_rnd_linear" in ast.dump(stmt.body[0])
    assert plain is not None
    fused = {p: (not v if list(_single_compare_ops(fn))[0] is ast.NotEq else v) for p, v in plain.items()}
    # the real-weights path must agree
    lrw = None
    for node in ast.walk(_method("_load_real_weights")):
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And):
            first = node.test.values[0]
            if _single_compare(first, _is_self_precision):
                body_dump = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                if "CutlassFp16SwiGluMlp" in body_dump:
                    lrw = _predicate_from_compare(first, _is_self_precision)
    assert lrw == fused, (lrw, fused)
    return fused


def _single_compare_ops(fn: ast.FunctionDef):
    for stmt in fn.body:
        if isinstance(stmt, ast.If):
            found = _single_compare(stmt.test, _is_self_precision)
            if found:
                yield found[0]


def _fp16_nn_rule() -> dict[str, bool]:
    """`self._autotune_gemm(dims, fp16_nn_shapes=self._precision == "fp16")`."""
    rules = [_predicate_from_compare(node.value, _is_self_precision)
             for node in ast.walk(_FRONTEND_TREE)
             if isinstance(node, ast.keyword) and node.arg == "fp16_nn_shapes"]
    assert len(rules) == 1, f"expected one fp16_nn_shapes call site, found {len(rules)}"
    return rules[0]


BRANCHES = _wrap_linear_branches()

# The plan's W3 statement of the rules, kept as a second, hand-written check.
PLAN_ALIGNMENT = {
    "fp16": None, "fp16_cutlass": 8, "fp8": 8, "fp8_static": 8, "fp8_static_cutlass": 8,
    "nvfp4": 16, "e0m3_hadamard": 16, "nvfp4_sim": 16,
}


# -- (a) members ---------------------------------------------------------------

def test_members_equal_frontend_precisions_tuple():
    frontend = _module_tuple("_PRECISIONS")
    assert tuple(p.value for p in Precision) == frontend


def test_values_are_plain_strings():
    for p in Precision:
        assert p == p.value and isinstance(p, str)
        assert str(p) == p.value
        assert f"{p}" == p.value
        assert f"precision={p!r}".startswith("precision=<Precision.")  # repr stays the enum repr
        assert Precision(p.value) is p
    assert Precision("nvfp4") == "nvfp4"
    assert "nvfp4" in tuple(Precision)


def test_table_has_a_row_per_member():
    assert set(precision_mod.PROPERTIES) == set(Precision)


# -- (b) properties equal tuple membership -------------------------------------

@pytest.mark.parametrize("prop,tuple_name", [
    ("needs_calibration", "_STATIC_FP8_PRECISIONS"),
    ("supports_awq", "_NVFP4_PRECISIONS"),
    ("supports_tile_autotune", "_VARIANT_TUNED_PRECISIONS"),
])
def test_property_equals_frontend_tuple_membership(prop, tuple_name):
    members = _module_tuple(tuple_name)
    assert set(members) <= {p.value for p in Precision}
    for p in Precision:
        assert getattr(p, prop) is (p.value in members), (prop, p)


def test_merge_qkv_mlp_matches_constructor_rule():
    rule = _merge_qkv_mlp_rule()
    for p in Precision:
        assert p.merge_qkv_mlp is rule[p.value], p
        assert p.merge_linear2_allowed is rule[p.value], p


def test_fused_swiglu_matches_frontend_rule():
    rule = _fused_swiglu_rule()
    for p in Precision:
        assert p.fused_swiglu_mlp is rule[p.value], p
    # the fused MLP is a standalone weight, which is what excludes the merge
    for p in Precision:
        assert not (p.fused_swiglu_mlp and p.merge_qkv_mlp), p


def test_fp16_nn_backbone_gemm_matches_frontend_rule():
    rule = _fp16_nn_rule()
    for p in Precision:
        assert p.fp16_nn_backbone_gemm is rule[p.value], p


def test_accepts_calibration_path_matches_frontend_check():
    # `calibration_path is not None` is legal iff precision in the static FP8
    # tuple or nvfp4_awq (itself legal only for the NVFP4 tuple).
    static = set(_module_tuple("_STATIC_FP8_PRECISIONS"))
    nvfp4 = set(_module_tuple("_NVFP4_PRECISIONS"))
    for p in Precision:
        assert p.accepts_calibration_path() is (p.value in static)
        assert p.accepts_calibration_path(nvfp4_awq=True) is (p.value in static or p.value in nvfp4)


def test_native_runtime_support_matches_linear_resource_dispatch():
    accepted = _linear_resource_accepted_classes()
    assert {"Fp16Linear", "Bf16OutLinear", "Nvfp4Linear"} <= accepted
    merge = _merge_qkv_mlp_rule()
    for p in Precision:
        classes = BRANCHES[p.value]["classes"]
        expected = classes <= accepted and merge[p.value]
        assert p.supports_native_runtime is expected, (p, classes)
    # the two precisions the native pipeline serves today
    assert {p.value for p in Precision if p.supports_native_runtime} == {"fp16", "nvfp4"}


# -- (c) alignment_fallback vs _wrap_linear ------------------------------------

def test_every_precision_has_a_wrap_linear_branch():
    assert set(BRANCHES) == {p.value for p in Precision}


def test_alignment_matches_wrap_linear_source_and_plan():
    for p in Precision:
        src = BRANCHES[p.value]["alignment"]
        assert src == PLAN_ALIGNMENT[p.value], (p, src)
        assert p.alignment == (src or 1), p


def _grid():
    dims = [1, 7, 8, 9, 15, 16, 17, 24, 32, 48, 64, 100, 128, 256, 384, 1024, 3072, 4096, 12288]
    return list(itertools.product(dims, dims))


def _source_fallback(p: Precision, n: int, k: int) -> bool:
    a = BRANCHES[p.value]["alignment"]
    return a is not None and (n % a != 0 or k % a != 0)


@pytest.mark.parametrize("p", ALL, ids=lambda p: p.value)
def test_alignment_fallback_matches_wrap_linear(p):
    for n, k in _grid():
        assert p.alignment_fallback(n, k) is _source_fallback(p, n, k), (p, n, k)


def test_alignment_fallback_named_cases():
    # action_encoder: K=7 (real LIBERO 7-DoF), head.linear: N=7
    for p in Precision:
        if p is Precision.FP16:
            assert not p.alignment_fallback(128, 7) and not p.alignment_fallback(7, 128)
        else:
            assert p.alignment_fallback(128, 7), p
            assert p.alignment_fallback(7, 128), p
    # 8-multiple that is not a 16-multiple: only the block-scaled tiers fall back
    for p in Precision:
        expect = p.value in ("nvfp4", "e0m3_hadamard", "nvfp4_sim")
        assert p.alignment_fallback(24, 8) is expect, p
        assert p.alignment_fallback(8, 24) is expect, p
    # both multiples of 16: nobody falls back
    for p in Precision:
        assert not p.alignment_fallback(3072, 4096), p
    # one misaligned dim is enough
    assert Precision.FP8.alignment_fallback(16, 9)
    assert Precision.NVFP4.alignment_fallback(15, 16)


def test_alignment_fallback_covers_awq_eligibility():
    # `_plan_awq` eligible: t.shape[0] % 16 == 0 and t.shape[1] % 16 == 0
    src = ast.dump(_method("_plan_awq"))
    assert src.count("Mod()") == 2 and "value=16" in src
    for p in Precision:
        if p.supports_awq:
            assert p.alignment == 16
            for n, k in _grid():
                assert p.alignment_fallback(n, k) is not (n % 16 == 0 and k % 16 == 0)


# -- (d) invalid names ----------------------------------------------------------

@pytest.mark.parametrize("bad", ["bogus", "", "FP16", "nvfp4 ", "fp8_dynamic", None, 4])
def test_unknown_precision_raises_value_error(bad):
    with pytest.raises(ValueError):
        Precision(bad)


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
