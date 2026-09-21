#!/usr/bin/env python
"""The final ImageWAM result tables: schema, validation, rendering.

Two tables, one per standard workload (`libero`, 10 denoise steps; `robotwin`,
30 denoise steps), each with the same six rows: the official torch
implementation as the baseline and FlashRT at fp16, fp8, fp4, int8 and int4.
Every number is a steady-state latency of one call from camera frames and
proprio (text context already encoded) to the de-normalised action chunk.

The record set is one JSON document (`docs/imagewam_results.json`); the
Markdown tables (`docs/imagewam_results.md`) are generated from it and are
never edited by hand:

    python benchmarks/imagewam_result_table.py skeleton   > docs/imagewam_results.json
    python benchmarks/imagewam_result_table.py check  docs/imagewam_results.json
    python benchmarks/imagewam_result_table.py render docs/imagewam_results.json > docs/imagewam_results.md

Schema (`schema_version` 1). Top level: `schema_version`, `tables`. Each
`tables[<name>]` has:

  workload     the served workload, every field of `ImageWAMWorkload` plus the
               valid instruction token range (`valid_tokens_min/max`). A table
               with a measured row has no null field, and `num_steps` equals
               the table's standard step count (`STANDARD_STEPS`).
  checkpoint   {name, sha256_16}: the checkpoint every row ran.
  boundary     what one timed call covers (the same for every row).
  session      default session id of the rows (see below).
  measurement  {device, commit, date, gpu_exclusive, clock: {nvpmodel,
               gpu_locked, emc_locked}, warmup, iters}.
  rows         exactly the six `ROWS` ids, in that order. A row has:
               id, status (`measured` | `not_measured` | `not_supported`),
               scope (`full_infer` | `gemm_only`), reason (required unless
               measured), session (optional override), config
               ({effective_config, calibration}), latency ({p50_ms, p10_ms,
               p90_ms, n}), fidelity ({source, vs_official_cos_median,
               vs_official_cos_min, mae_vs_gt_median, n} or null), note.

Rules the checker enforces (one line each in its output):

  * every table has the six rows once, in order, and the workload's
    `num_steps` is the table's standard (libero 10, robotwin 30);
  * a `measured` row has a latency with p10 <= p50 <= p90 and n >= 1, and
    the table's workload is complete (a `gemm_only` row has `latency.p50_ms`
    and `components` {prefill_p50_ms, denoise_step_p50_ms, num_steps}, and
    p50 equals prefill + num_steps x step: the benches print no percentiles
    of the composed call);
  * a `gemm_only` row (the int8 / int4 rows: no frontend precision tier
    runs them, the benches time the GEMMs with random packed operands and no
    activation quantization) carries no fidelity, and its speedup is not
    computed: it is an upper bound, not a deployment number;
  * a speedup against the official row is only computed when both rows ran
    in the same `session`; otherwise the cell says so instead of dividing
    numbers taken on different machine states (issues.md ISSUE-082).
"""
from __future__ import annotations

import json
import sys
from typing import Any

SCHEMA_VERSION = 1
TABLES = ("libero", "robotwin")
STANDARD_STEPS = {"libero": 10, "robotwin": 30}

# id, engine, label, the frontend precision the row runs (None: no tier),
# default scope.
ROWS: tuple[tuple[str, str, str, str | None, str], ...] = (
    ("official_torch", "official", "official torch (bf16 eager)", None, "full_infer"),
    ("flashrt_fp16", "flashrt", "FlashRT fp16", "fp16", "full_infer"),
    ("flashrt_fp8", "flashrt", "FlashRT fp8 (static, CUTLASS)", "fp8_static_cutlass", "full_infer"),
    ("flashrt_fp4", "flashrt", "FlashRT fp4 (nvfp4)", "nvfp4", "full_infer"),
    ("flashrt_int8", "flashrt", "FlashRT int8 (SM80 CUTLASS)", None, "gemm_only"),
    ("flashrt_int4", "flashrt", "FlashRT int4 (SM80 CUTLASS)", None, "gemm_only"),
)
ROW_IDS = tuple(r[0] for r in ROWS)
STATUSES = ("measured", "not_measured", "not_supported")
SCOPES = ("full_infer", "gemm_only")

WORKLOAD_FIELDS = (
    "num_views", "image_h", "image_w", "text_max_len", "valid_tokens_min", "valid_tokens_max",
    "action_horizon", "action_dim", "proprio_dim", "num_steps", "shift", "num_train_timesteps")
BOUNDARY = ("one call from camera frames and proprio (text context already encoded, so the text "
            "encoder is outside) to the de-normalised action chunk on the host; steady state "
            "(graphs captured, caches warm)")


def _workload(table: str) -> dict[str, Any]:
    """The table's workload declaration. LIBERO is `ImageWAMWorkload.libero()`
    and the served instruction range; RoboTwin declares only what is fixed
    (the step count) until the deployment's own values are entered."""
    if table == "libero":
        return dict(num_views=2, image_h=224, image_w=224, text_max_len=512, valid_tokens_min=16,
                    valid_tokens_max=31, action_horizon=64, action_dim=7, proprio_dim=8,
                    num_steps=STANDARD_STEPS["libero"], shift=5.0, num_train_timesteps=1000)
    fields: dict[str, Any] = {f: None for f in WORKLOAD_FIELDS}
    fields["num_steps"] = STANDARD_STEPS["robotwin"]
    return fields


def skeleton() -> dict[str, Any]:
    """An empty record set: every row `not_measured`."""
    tables: dict[str, Any] = {}
    for name in TABLES:
        rows = []
        for rid, _engine, _label, _prec, scope in ROWS:
            rows.append({"id": rid, "status": "not_measured", "scope": scope,
                         "reason": "not run yet", "config": None, "latency": None,
                         "fidelity": None, "note": ""})
        tables[name] = {
            "workload": _workload(name),
            "checkpoint": {"name": None, "sha256_16": None},
            "boundary": BOUNDARY, "session": None,
            "measurement": {"device": None, "commit": None, "date": None, "gpu_exclusive": None,
                            "clock": {"nvpmodel": None, "gpu_locked": None, "emc_locked": None},
                            "warmup": None, "iters": None},
            "rows": rows}
    return {"schema_version": SCHEMA_VERSION, "tables": tables}


def _row_session(table: dict, row: dict) -> Any:
    return row.get("session", table.get("session"))


def validate(doc: dict) -> list[str]:
    """Every violated rule as one line each; empty when the document is sound."""
    errors: list[str] = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        return [f"schema_version {doc.get('schema_version')!r} != {SCHEMA_VERSION}"]
    tables = doc.get("tables", {})
    if tuple(tables) != TABLES:
        errors.append(f"tables {tuple(tables)} != {TABLES}")
    for name in TABLES:
        t = tables.get(name)
        if t is None:
            continue
        where = f"[{name}]"
        rows = t.get("rows", [])
        if tuple(r.get("id") for r in rows) != ROW_IDS:
            errors.append(f"{where} rows {tuple(r.get('id') for r in rows)} != {ROW_IDS}")
            continue
        wl = t.get("workload", {})
        if set(wl) != set(WORKLOAD_FIELDS):
            errors.append(f"{where} workload fields {sorted(set(wl) ^ set(WORKLOAD_FIELDS))} differ")
        if wl.get("num_steps") != STANDARD_STEPS[name]:
            errors.append(f"{where} workload num_steps {wl.get('num_steps')!r} != the standard "
                          f"{STANDARD_STEPS[name]}")
        any_measured = False
        for row in rows:
            rid = row["id"]
            here = f"{where} {rid}"
            if row.get("status") not in STATUSES:
                errors.append(f"{here}: status {row.get('status')!r} not in {STATUSES}")
                continue
            if row.get("scope") not in SCOPES:
                errors.append(f"{here}: scope {row.get('scope')!r} not in {SCOPES}")
            if row["status"] != "measured":
                if not row.get("reason"):
                    errors.append(f"{here}: {row['status']} needs a reason")
                continue
            any_measured = True
            lat = row.get("latency") or {}
            if row.get("scope") == "gemm_only":
                # The int benches time the prefill and one denoise step separately and
                # print no percentiles of the composed call: the row is their sum.
                comp = row.get("components") or {}
                keys = ("prefill_p50_ms", "denoise_step_p50_ms", "num_steps")
                if lat.get("p50_ms") is None or any(comp.get(k) is None for k in keys):
                    errors.append(f"{here}: a gemm_only row needs latency.p50_ms and components {keys}")
                elif abs(lat["p50_ms"] - (comp[keys[0]] + comp[keys[2]] * comp[keys[1]])) > 0.05:
                    errors.append(f"{here}: p50_ms {lat['p50_ms']} != prefill + num_steps x step "
                                  f"({comp[keys[0]]} + {comp[keys[2]]} x {comp[keys[1]]})")
                elif comp[keys[2]] != wl.get("num_steps"):
                    errors.append(f"{here}: components.num_steps {comp[keys[2]]} != the workload's "
                                  f"{wl.get('num_steps')}")
            else:
                p10, p50, p90, n = (lat.get(k) for k in ("p10_ms", "p50_ms", "p90_ms", "n"))
                if None in (p10, p50, p90, n) or not (p10 <= p50 <= p90) or n < 1:
                    errors.append(f"{here}: latency needs p10 <= p50 <= p90 and n >= 1, got {lat}")
            if not (row.get("config") or {}).get("effective_config") and rid != "official_torch":
                errors.append(f"{here}: a FlashRT row records its effective_config")
            if row.get("scope") == "gemm_only" and row.get("fidelity"):
                errors.append(f"{here}: a gemm_only row has no fidelity (random packed operands)")
            fid = row.get("fidelity")
            if fid and not fid.get("source"):
                errors.append(f"{here}: fidelity needs its data source")
        if any_measured:
            missing = [f for f in WORKLOAD_FIELDS if wl.get(f) is None]
            if missing:
                errors.append(f"{where} a row is measured but the workload lacks {missing}")
            for k in ("device", "commit", "date", "gpu_exclusive", "warmup", "iters"):
                if (t.get("measurement") or {}).get(k) is None:
                    errors.append(f"{where} a row is measured but measurement.{k} is empty")
            if not (t.get("checkpoint") or {}).get("sha256_16"):
                errors.append(f"{where} a row is measured but the checkpoint is not identified")
    return errors


def _s(v: Any) -> str:
    return "—" if v is None else str(v)


def _fmt(v: float | None, spec: str = ".1f") -> str:
    return "—" if v is None else format(v, spec)


def render(doc: dict) -> str:
    """The Markdown tables for a valid document."""
    errors = validate(doc)
    if errors:
        raise ValueError("invalid result document:\n  " + "\n  ".join(errors))
    labels = {r[0]: r for r in ROWS}
    out = ["# ImageWAM steady-state results (generated by benchmarks/imagewam_result_table.py; "
           "do not edit)", ""]
    for name in TABLES:
        t = doc["tables"][name]
        wl = t["workload"]
        m = t["measurement"]
        out.append(f"## {name}")
        out.append("")
        wl_text = ("not declared yet" if any(wl[f] is None for f in WORKLOAD_FIELDS) else
                   f"{wl['num_views']} views {wl['image_h']}x{wl['image_w']}, text {wl['text_max_len']} "
                   f"tokens ({wl['valid_tokens_min']}-{wl['valid_tokens_max']} valid), horizon "
                   f"{wl['action_horizon']}, action_dim {wl['action_dim']}, proprio {wl['proprio_dim']}, "
                   f"shift {wl['shift']}")
        out.append(f"Workload: {wl_text}; **{wl['num_steps']} denoise steps**.  ")
        clk = m.get("clock") or {}
        out.append(f"Device {_s(m.get('device'))}, commit {_s(m.get('commit'))}, "
                   f"{_s(m.get('date'))}; GPU exclusive: {m.get('gpu_exclusive')}, nvpmodel "
                   f"{clk.get('nvpmodel')}, gpu_locked {clk.get('gpu_locked')}, emc_locked "
                   f"{clk.get('emc_locked')}; warmup {m.get('warmup')}, iters {m.get('iters')}.  ")
        out.append(f"Checkpoint: {t['checkpoint'].get('name')} ({t['checkpoint'].get('sha256_16')}). "
                   f"Timed call: {t['boundary']}.")
        out.append("")
        out.append("| Row | Scope | P50 ms | P10–P90 ms | vs official | cos vs official (median / min) "
                   "| MAE vs GT | Note |")
        out.append("|---|---|---:|---:|---:|---:|---:|---|")
        official = next(r for r in t["rows"] if r["id"] == "official_torch")
        for row in t["rows"]:
            _, _engine, label, _prec, _scope = labels[row["id"]]
            if row["status"] != "measured":
                out.append(f"| {label} | {row['scope']} | — | — | — | — | — | {row['status']}: "
                           f"{row.get('reason')} |")
                continue
            lat = row["latency"]
            fid = row.get("fidelity") or {}
            note = row.get("note") or ""
            if row["id"] == "official_torch":
                speed = "1.00x"
            elif row["scope"] == "gemm_only":
                speed = "—"
                note = ("† GEMM-only upper bound: random packed operands, no activation "
                        "quantization; not comparable to a full call. " + note).strip()
            elif official["status"] != "measured":
                speed = "—"
            elif _row_session(t, row) != _row_session(t, official) or _row_session(t, row) is None:
                speed = "‡"
                note = ("‡ not the official row's session: no ratio computed. " + note).strip()
            else:
                speed = f"{official['latency']['p50_ms'] / lat['p50_ms']:.2f}x"
            cos = "—" if not fid else (f"{_fmt(fid.get('vs_official_cos_median'), '.5f')} / "
                                       f"{_fmt(fid.get('vs_official_cos_min'), '.5f')}")
            spread = (f"{lat['p10_ms']:.1f}–{lat['p90_ms']:.1f}"
                      if lat.get("p10_ms") is not None and lat.get("p90_ms") is not None else "—")
            out.append(f"| {label} | {row['scope']} | {lat['p50_ms']:.1f} | "
                       f"{spread} | {speed} | {cos} | "
                       f"{_fmt(fid.get('mae_vs_gt_median'), '.4f') if fid else '—'} | {note} |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "skeleton":
        print(json.dumps(skeleton(), indent=2))
        return 0
    if len(argv) == 3 and argv[1] in ("check", "render"):
        with open(argv[2]) as f:
            doc = json.load(f)
        if argv[1] == "check":
            errors = validate(doc)
            print("\n".join(errors) if errors else "ok")
            return 1 if errors else 0
        try:
            print(render(doc))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
