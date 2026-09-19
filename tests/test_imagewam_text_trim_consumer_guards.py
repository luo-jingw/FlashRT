"""`text_trim=True` and the consumers of a captured frontend's graphs.

A trimmed frontend runs one graph per prompt length, so the runtime surface
carries the variant table keyed by that length (`GraphVariants`) and the
export adopts one exec per key, `step` replaying the length the prompt set
(plan.md phase S2). What each consumer does now:

- `runtime_surface()` serves a trimmed frontend: every field describes the
  active length's graph (`context_rows = active_dims["x0"]`) and
  `graph_variants` names the exec of every captured length;
- `export_model_runtime(io="python")` adopts every captured length and
  declares the table (the manifest's `text_lengths`);
- `pipeline_resources()` still refuses: the native C++ pipeline replays one
  graph at one context length (`frt_imagewam_native` holds one graph exec
  and one `context_rows`), which is a later phase, not a silent gap;
- `export_model_runtime(io="native")` and `ImageWAMNativeRuntime.create`
  refuse the same way, and `resolve_config` refuses it for
  `consumer="native"` only (rule R5).

`graph_variant_plan` and `uncaptured_text_length_message` are the pure part
of that contract ("which keys and which default key does this table
declare", "what does `step` say about a length with no variant"), so they
are checked here without a GPU. Everything that touches a captured frontend
needs one.
"""
from __future__ import annotations

import json

import pytest
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor
from flash_rt.models.imagewam.runtime_surface import (
    GraphVariants,
    TextLengthGraph,
    graph_variant_plan,
    uncaptured_text_length_message,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TEXT_LEN = 16
DIMS = dict(x0=TEXT_LEN + 1, a0=TEXT_LEN + 1 + 10, total=TEXT_LEN + 1 + 10 + 4,
            joint_attention_dim=16, proprio_dim=3)
SHORT_VALID = 5
LONG_VALID = 9


# -- the pure seam: the table -> the declaration --------------------------

def test_graph_variant_plan_keys_every_captured_length_and_defaults_to_the_active_one():
    plan = graph_variant_plan(GraphVariants(
        active_key=45,
        entries=(TextLengthGraph(key=45, graph_exec=0xbeef),
                 TextLengthGraph(key=30, graph_exec=0xaaaa)),
        per_prompt_length=True))
    assert plan.keys == (30, 45), "keys ascending, one per captured length"
    assert plan.default_key == 45, "the length active at export"
    assert plan.max_variants == 2, "the table's own size: the export never captures"


def test_graph_variant_plan_refuses_a_table_it_cannot_declare():
    with pytest.raises(ValueError, match="at least one graph variant"):
        graph_variant_plan(GraphVariants(active_key=0, entries=(), per_prompt_length=False))
    with pytest.raises(ValueError, match="keys repeat"):
        graph_variant_plan(GraphVariants(
            active_key=30,
            entries=(TextLengthGraph(key=30, graph_exec=1), TextLengthGraph(key=30, graph_exec=2)),
            per_prompt_length=True))
    with pytest.raises(ValueError, match="active key 9 is not in the graph variant table"):
        graph_variant_plan(GraphVariants(
            active_key=9, entries=(TextLengthGraph(key=30, graph_exec=1),),
            per_prompt_length=True))
    # text_trim=False is the one-entry case
    plan = graph_variant_plan(GraphVariants(
        active_key=513, entries=(TextLengthGraph(key=513, graph_exec=7),),
        per_prompt_length=False))
    assert plan.keys == (513,) and plan.default_key == 513 and plan.max_variants == 1


def test_uncaptured_text_length_message_names_the_length_and_the_adopted_table():
    message = uncaptured_text_length_message(77, (30, 45))
    print(message)
    assert "x0=77" in message and "[30, 45]" in message
    assert "precapture_text_lengths" in message


# -- the surface and the export -------------------------------------------

def _frontend(text_trim: bool, valid: int = SHORT_VALID) -> ImageWAMTorchFrontendThor:
    torch.manual_seed(0)
    fe = ImageWAMTorchFrontendThor(precision="fp16", dims_override=dict(DIMS), text_trim=text_trim)
    mask = torch.zeros(TEXT_LEN, dtype=torch.bool)
    mask[:valid] = True
    fe.set_prompt(context=torch.randn(TEXT_LEN, DIMS["joint_attention_dim"]).to(torch.bfloat16),
                  context_mask=mask)
    return fe


@needs_cuda
def test_runtime_surface_carries_one_graph_per_captured_length():
    fe = _frontend(True, SHORT_VALID)
    first = fe.runtime_surface()
    short_x0 = SHORT_VALID + 1  # valid text rows + the proprio row
    print(f"trimmed surface: context_rows={first.context_rows} active_key="
          f"{first.graph_variants.active_key} keys={[e.key for e in first.graph_variants.entries]}")
    assert first.context_rows == short_x0, "the surface describes the active length, not the buffer size"
    assert first.graph_variants.active_key == short_x0
    assert first.graph_variants.per_prompt_length is True
    assert dict(first.setup_identity)["text_trim"] == "True"
    assert dict(first.setup_identity)["dims.x0"] == str(short_x0)

    # A second length: the table grows, the surface turns to it, and the
    # first length's exec stays declared.
    fe.set_prompt(context=torch.randn(TEXT_LEN, DIMS["joint_attention_dim"]).to(torch.bfloat16),
                  context_mask=torch.cat([torch.ones(LONG_VALID, dtype=torch.bool),
                                          torch.zeros(TEXT_LEN - LONG_VALID, dtype=torch.bool)]))
    long_x0 = LONG_VALID + 1
    second = fe.runtime_surface()
    assert (second.context_rows, second.graph_variants.active_key) == (long_x0, long_x0)
    assert tuple(e.key for e in second.graph_variants.entries) == (short_x0, long_x0)
    execs = {e.key: e.graph_exec for e in second.graph_variants.entries}
    print(f"variants: {execs}")
    assert execs[long_x0] == second.graph_exec, "graph_exec is the active key's entry"
    assert execs[short_x0] != execs[long_x0], "one exec per length"

    # text_trim=False: one graph, every prompt, keyed by dims["x0"].
    plain = _frontend(False, LONG_VALID).runtime_surface()
    assert plain.context_rows == DIMS["x0"] and plain.graph_variants.active_key == DIMS["x0"]
    assert plain.graph_variants.per_prompt_length is False
    assert tuple(e.key for e in plain.graph_variants.entries) == (DIMS["x0"],)
    assert plain.graph_exec == plain.graph_variants.entries[0].graph_exec
    assert dict(plain.setup_identity)["text_trim"] == "False"


@needs_cuda
def test_pipeline_resources_refuses_text_trim():
    """The native pipeline replays one graph at one context length; its
    per-length table is a later phase (plan.md S3)."""
    with pytest.raises(ValueError, match="text_trim=True") as e:
        _frontend(True).pipeline_resources()
    print(f"pipeline_resources: {e.value}")
    assert "native pipeline" in str(e.value)
    resources = _frontend(False).pipeline_resources()
    assert resources.dims.x0 == DIMS["x0"]


@needs_cuda
def test_export_model_runtime_serves_the_trimmed_abi_face():
    pytest.importorskip("flash_rt.runtime.exec", exc_type=ImportError)
    pytest.importorskip("flash_rt.runtime.export", exc_type=ImportError)
    fe = _frontend(True, SHORT_VALID)
    fe.set_prompt(context=torch.randn(TEXT_LEN, DIMS["joint_attention_dim"]).to(torch.bfloat16),
                  context_mask=torch.cat([torch.ones(LONG_VALID, dtype=torch.bool),
                                          torch.zeros(TEXT_LEN - LONG_VALID, dtype=torch.bool)]))
    short_x0, long_x0 = SHORT_VALID + 1, LONG_VALID + 1
    runtime = fe.export_model_runtime()
    manifest = json.loads(runtime.manifest)
    print(f"manifest graphs={manifest['graphs']} text_lengths={manifest['text_lengths']}")
    assert manifest["text_lengths"] == {"default_key": long_x0, "keys": [short_x0, long_x0],
                                        "per_prompt_length": True}
    graphs = {g["name"]: g for g in manifest["graphs"]}
    assert graphs["infer"]["default_key"] == long_x0
    assert graphs["infer"]["keys"] == [short_x0, long_x0]
    runtime.release()

    # The native face has one graph at one context geometry (rule R5).
    with pytest.raises(ValueError, match="text_trim=True"):
        fe.export_model_runtime(io="native")
    plain = _frontend(False, LONG_VALID)
    with pytest.raises(ValueError, match="requires native="):
        plain.export_model_runtime(io="native")
    runtime = plain.export_model_runtime()
    assert json.loads(runtime.manifest)["text_lengths"] == {
        "default_key": DIMS["x0"], "keys": [DIMS["x0"]], "per_prompt_length": False}
    runtime.release()


@needs_cuda
def test_native_handle_refuses_a_trimmed_surface():
    """`ImageWAMNativeRuntime` borrows one `context_rows` and replays one
    graph exec, so it refuses a trimmed surface before it loads the native
    library."""
    from flash_rt.models.imagewam.native_runtime import ImageWAMNativeRuntime

    with pytest.raises(ValueError, match="text_trim=True"):
        ImageWAMNativeRuntime.create(_frontend(True).runtime_surface())
