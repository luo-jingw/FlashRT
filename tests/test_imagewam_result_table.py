"""The final result tables' schema (benchmarks/imagewam_result_table.py).

CPU only. Pins the record set's rules: the six rows in order, the standard
step count per table, a complete workload once anything is measured,
GEMM-only rows carrying no fidelity and no ratio, and a ratio against the
official row only inside one session.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks import imagewam_result_table as rt

COMMITTED = Path(__file__).resolve().parents[1] / "docs" / "imagewam_results.json"
ROBOTWIN = dict(num_views=3, image_h=256, image_w=256, text_max_len=128, valid_tokens_min=16,
                valid_tokens_max=128, action_horizon=32, action_dim=14, proprio_dim=14, num_steps=30,
                shift=5.0, num_train_timesteps=1000)


def measured(row, p50, *, fid=True, session="s1", scope=None):
    row.update(status="measured", session=session, reason="",
               latency={"p10_ms": p50 - 1, "p50_ms": p50, "p90_ms": p50 + 1, "n": 100})
    if scope:
        row["scope"] = scope
    if row["id"] != "official_torch":
        row["config"] = {"effective_config": "effective_config precision=x", "calibration": None}
    if fid and row["id"] != "official_torch" and row["scope"] == "full_infer":
        row["fidelity"] = {"source": "libero", "vs_official_cos_median": 0.9993,
                           "vs_official_cos_min": 0.9988, "mae_vs_gt_median": 0.186, "n": 40}


def filled(table="libero"):
    doc = rt.skeleton()
    t = doc["tables"][table]
    if table == "robotwin":
        t["workload"] = dict(ROBOTWIN)
    t["checkpoint"] = {"name": "ckpt", "sha256_16": "0123456789abcdef"}
    t["measurement"].update(device="Thor", commit="abc", date="2026-09-22", gpu_exclusive=True,
                            warmup=20, iters=100)
    t["session"] = "s1"
    return doc, t


def test_skeleton_is_valid_and_renders():
    doc = rt.skeleton()
    assert rt.validate(doc) == []
    text = rt.render(doc)
    assert "## libero" in text and "## robotwin" in text
    assert text.count("not_measured") == 12


def test_the_two_tables_carry_their_standard_step_counts():
    doc = rt.skeleton()
    assert doc["tables"]["libero"]["workload"]["num_steps"] == 10
    assert doc["tables"]["robotwin"]["workload"]["num_steps"] == 30
    doc["tables"]["robotwin"]["workload"]["num_steps"] = 10
    assert any("num_steps" in e and "30" in e for e in rt.validate(doc))


def test_the_committed_record_set_is_valid():
    assert rt.validate(json.loads(COMMITTED.read_text())) == []


def test_rows_are_the_six_in_order():
    assert rt.ROW_IDS == ("official_torch", "flashrt_fp16", "flashrt_fp8", "flashrt_fp4",
                          "flashrt_int8", "flashrt_int4")
    doc = rt.skeleton()
    doc["tables"]["libero"]["rows"].reverse()
    assert any("rows" in e for e in rt.validate(doc))


def test_precisions_map_to_frontend_tiers():
    tiers = {r[0]: r[3] for r in rt.ROWS}
    assert tiers == {"official_torch": None, "flashrt_fp16": "fp16",
                     "flashrt_fp8": "fp8_static_cutlass", "flashrt_fp4": "nvfp4",
                     "flashrt_int8": None, "flashrt_int4": None}
    # int8/int4 have no frontend tier: their default scope is the GEMM-only bench
    assert {r[0]: r[4] for r in rt.ROWS if r[4] == "gemm_only"} == {
        "flashrt_int8": "gemm_only", "flashrt_int4": "gemm_only"}


def test_measured_row_needs_a_latency_a_complete_workload_and_the_measurement_block():
    doc = rt.skeleton()
    measured(doc["tables"]["robotwin"]["rows"][0], 400.0)
    errs = rt.validate(doc)
    assert any("workload lacks" in e for e in errs)
    assert any("measurement.device" in e for e in errs)
    assert any("checkpoint is not identified" in e for e in errs)
    doc, t = filled("robotwin")
    measured(t["rows"][0], 400.0)
    t["rows"][0]["latency"]["p10_ms"] = 500.0
    assert any("p10 <= p50 <= p90" in e for e in rt.validate(doc))


def test_not_measured_and_unsupported_rows_need_a_reason():
    doc = rt.skeleton()
    doc["tables"]["libero"]["rows"][5].update(status="not_supported", reason="")
    assert any("needs a reason" in e for e in rt.validate(doc))


def test_flashrt_row_records_its_effective_config():
    doc, t = filled()
    measured(t["rows"][1], 200.0)
    del t["rows"][1]["config"]
    assert any("effective_config" in e for e in rt.validate(doc))


def test_gemm_only_rows_have_no_fidelity_and_no_ratio():
    doc, t = filled()
    measured(t["rows"][0], 450.0)
    measured(t["rows"][4], 130.0, fid=False)
    assert rt.validate(doc) == []
    row = next(l for l in rt.render(doc).splitlines() if "int8" in l)
    assert "†" in row and "| — |" in row
    t["rows"][4]["fidelity"] = {"source": "libero", "vs_official_cos_median": 1.0,
                                "vs_official_cos_min": 1.0, "mae_vs_gt_median": 0.1, "n": 1}
    assert any("gemm_only row has no fidelity" in e for e in rt.validate(doc))


def test_ratio_against_official_only_inside_one_session():
    doc, t = filled()
    measured(t["rows"][0], 450.0, session="s1")
    measured(t["rows"][3], 100.0, session="s1")
    line = next(l for l in rt.render(doc).splitlines() if "fp4" in l)
    assert "4.50x" in line
    t["rows"][3]["session"] = "s2"
    line = next(l for l in rt.render(doc).splitlines() if "fp4" in l)
    assert "‡" in line and "4.50x" not in line


def test_robotwin_table_with_its_workload_renders():
    doc, t = filled("robotwin")
    measured(t["rows"][0], 1200.0)
    measured(t["rows"][3], 300.0)
    assert rt.validate(doc) == []
    text = rt.render(doc)
    assert "**30 denoise steps**" in text and "3 views 256x256" in text


def test_render_refuses_an_invalid_document():
    doc = rt.skeleton()
    doc["tables"]["libero"]["workload"]["num_steps"] = 7
    with pytest.raises(ValueError):
        rt.render(doc)


def test_the_cli_round_trips(tmp_path, capsys):
    path = tmp_path / "r.json"
    assert rt.main(["x", "skeleton"]) == 0
    path.write_text(capsys.readouterr().out)
    assert rt.main(["x", "check", str(path)]) == 0
    doc = json.loads(path.read_text())
    bad = copy.deepcopy(doc)
    bad["schema_version"] = 2
    path.write_text(json.dumps(bad))
    assert rt.main(["x", "check", str(path)]) == 1
