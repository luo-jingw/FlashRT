"""The stage x operator grid's trace parsing, classification and rendering
(benchmarks/imagewam_stage_operator_grid.py). CPU only, on a synthetic Chrome trace."""
from __future__ import annotations

from benchmarks import imagewam_stage_operator_grid as g

DIMS = dict(hidden=3072, mlp_hidden=9216, action_hidden_dim=1024, action_attn_width=3072,
            action_mlp_hidden=4096, num_layers_double=5, num_layers_single=20,
            action_num_layers_double=5, action_num_layers_single=20, x0=25, a0=417,
            num_action=64, num_denoise_steps=10)


def trace():
    ev = []
    # a range for a backbone single layer, and one outer prefill range around it
    ev.append({"cat": "user_annotation", "name": "stage:bb.prefill_rest", "ts": 0.0, "dur": 1000.0})
    ev.append({"cat": "user_annotation", "name": "stage:bb.single", "ts": 100.0, "dur": 300.0})
    for i, (ts, name, dur) in enumerate([(110.0, "cutlass_nvfp4_gemm", 200.0), (350.0, "silu_glu_kernel", 5.0),
                                         (500.0, "quantize_fp4_kernel", 8.0), (600.0, "mystery_kernel", 3.0)]):
        ev.append({"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": ts, "dur": 1.0, "args": {"correlation": i}})
        ev.append({"cat": "kernel", "name": name, "ts": ts + 50, "dur": dur, "args": {"correlation": i}})
    ev.append({"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 2000.0, "dur": 1.0, "args": {"correlation": 9}})
    ev.append({"cat": "kernel", "name": "elementwise", "ts": 2050.0, "dur": 4.0, "args": {"correlation": 9}})
    return {"traceEvents": ev}


def test_classify_by_name():
    # 0921x: kernel_quantize_fp4_sfa_vec's full symbol embeds the `flash_rt::fp4` namespace, so
    # a bare "flash" needle misclassified it as attention -- quantize is checked first now.
    assert g.classify("void flash_rt::fp4::kernel_quantize_fp4_sfa_vec<...>(...)") == "quantize"
    assert g.classify("cutlass_nvfp4_gemm") == "gemm"
    assert g.classify("nvjet_hsh_448x64") == "gemm"
    assert g.classify("quantize_fp4_kernel") == "quantize"
    assert g.classify("fmha_fwd") == "attention"
    assert g.classify("adaln_modulate") == "norm"
    assert g.classify("rope_apply") == "rope"
    assert g.classify("silu_glu") == "glu"
    assert g.classify("gated_residual_add") == "residual"
    assert g.classify("Memcpy DtoD") == "copy"
    assert g.classify("something_else") == "other"


def test_kernels_go_to_the_innermost_range_that_launched_them():
    rows = g.attribute(trace())
    by = {name: stage for stage, _, name, _ in rows}
    assert by["cutlass_nvfp4_gemm"] == "bb.single"      # inside both ranges: the innermost wins
    assert by["silu_glu_kernel"] == "bb.single"
    assert by["quantize_fp4_kernel"] == "bb.prefill_rest"  # only the outer range contains its launch
    assert by["elementwise"] == "(none)"                 # outside every range


def test_grid_totals_short_kernel_share_and_unclassified():
    grid = g.build_grid(g.attribute(trace()))
    s = grid["bb.single"]
    assert s["us"]["gemm"] == 200.0 and s["us"]["glu"] == 5.0 and s["n"]["gemm"] == 1
    assert s["short"][10] == 5.0
    assert g.unclassified(grid, 5) == [(4.0, "elementwise"), (3.0, "mystery_kernel")]


def test_analytic_flops_and_bytes_match_the_model_dimensions():
    ref = g.analytic(DIMS, 2.0)
    p = g._params(DIMS)
    assert ref["bb.single"][0] == 2 * 20 * 417 * p["sgl"]
    assert ref["bb.single"][1] == 20 * p["sgl"] * 2.0
    assert ref["act.single"][0] == 2 * 10 * 20 * 64 * p["a_sgl"]
    assert abs(p["sgl"] * 20 / 1e9 - 2.45) < 0.02        # the backbone's single-stream weights, billions


def test_render_has_the_grid_the_rates_and_the_short_share():
    text = g.render(g.build_grid(g.attribute(trace())), DIMS, 0.5625)
    assert "stage" in text and "gemm" in text and "bb.single" in text and "ALL" in text
    assert "shorter than 10 / 25 us" in text
    assert "TFLOPs" in text
