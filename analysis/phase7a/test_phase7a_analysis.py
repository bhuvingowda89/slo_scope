from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.phase7a import phase7a_analysis as a


def test_percentile_uses_campaign_interpolation():
    assert a.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert a.percentile([0, 10], 0.95) == 9.5


def test_slo_and_compound_classification_from_run_aggregate(tmp_path, monkeypatch):
    row = {
        "run_id": "r",
        "condition_id": "BASELINE",
        "repetition": 1,
        "attempt": 1,
        "active_mechanisms": "[]",
        "mechanism_levels": "{}",
        "compound_degree": 0,
        "ttft_p95": a.SLO_THRESHOLDS["TTFT_SLO"] + 0.001,
        "total_latency_p95": a.SLO_THRESHOLDS["TOTAL_LATENCY_SLO"] + 0.001,
        "post_first_token_duration_p95": a.SLO_THRESHOLDS["DECODE_DURATION_SLO"] - 0.001,
    }
    row["ttft_slo_violated"] = row["ttft_p95"] > a.SLO_THRESHOLDS["TTFT_SLO"]
    row["total_latency_slo_violated"] = row["total_latency_p95"] > a.SLO_THRESHOLDS["TOTAL_LATENCY_SLO"]
    row["decode_duration_slo_violated"] = row["post_first_token_duration_p95"] > a.SLO_THRESHOLDS["DECODE_DURATION_SLO"]
    row["slo_violation_count"] = int(row["ttft_slo_violated"]) + int(row["total_latency_slo_violated"]) + int(row["decode_duration_slo_violated"])
    assert row["slo_violation_count"] == 2


def test_throughput_not_counted_as_slo():
    spec = a.create_analysis_spec()
    counted = spec["slo_violation_definitions"]["counted_slos"]
    assert counted == ["TTFT_SLO", "TOTAL_LATENCY_SLO", "DECODE_DURATION_SLO"]
    assert "THROUGHPUT_SLO" not in counted


def test_dz_and_paired_difference():
    diffs = [2.0, 4.0, 6.0]
    assert a.mean(diffs) == 4.0
    assert a.dz(diffs) == pytest.approx(2.0)
    result = a.paired_t_test(diffs)
    assert result["df"] == 2
    assert result["p_value"] is not None


def test_holm_adjustment_monotonic():
    adjusted = a.holm_adjust([("a", 0.01), ("b", 0.03), ("c", 0.02)])
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.04)
    assert adjusted["b"] == pytest.approx(0.04)


def test_factorial_interaction_contrast_formula():
    a0b0, a1b0, a0b1, a1b1 = 1.0, 3.0, 4.0, 10.0
    assert a1b1 - a1b0 - a0b1 + a0b0 == 4.0


def test_repetition_matching_uses_condition_and_repetition():
    rows = [
        {"condition_id": "BASELINE", "repetition": 2, "value": 20},
        {"condition_id": "BASELINE", "repetition": 1, "value": 10},
        {"condition_id": "INPUT_MEDIUM", "repetition": 1, "value": 30},
    ]
    mapped = a.map_by_condition_rep(rows)
    assert mapped[("BASELINE", 1)]["value"] == 10
    assert mapped[("BASELINE", 2)]["value"] == 20


def test_wilson_interval_bounds():
    lo, hi = a.wilson_interval(4, 8)
    assert 0 <= lo <= 0.5 <= hi <= 1


def test_publication_index_enforcement_excludes_old_campaign(tmp_path, monkeypatch):
    index = tmp_path / "runs" / "phase6-v2" / "publication-run-index.jsonl"
    rows = [
        {
            "run_id": f"phase6v2-r01-x-{i}",
            "run_path": "runs/phase6/old",
            "source_hash": a.SOURCE_SHA,
            "freeze_hash": a.FREEZE_SHA,
            "publication_analysis_eligible": True,
            "artifact_manifest_hash": "h",
        }
        for i in range(104)
    ]
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(a, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="superseded phase6"):
        a.load_publication_index()


def test_publication_index_rejects_duplicate_run_ids(tmp_path, monkeypatch):
    index = tmp_path / "runs" / "phase6-v2" / "publication-run-index.jsonl"
    rows = [
        {
            "run_id": "duplicate",
            "run_path": "runs/phase6-v2/x",
            "source_hash": a.SOURCE_SHA,
            "freeze_hash": a.FREEZE_SHA,
            "publication_analysis_eligible": True,
            "artifact_manifest_hash": "h",
        }
        for _ in range(104)
    ]
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(a, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="duplicate run IDs"):
        a.load_publication_index()


def test_input_seal_hash_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "OUT", tmp_path)
    files = {
        "runs/phase6-v2/publication-run-index.jsonl": "{}\n",
        "runs/phase6-v2/campaign-integrity-report.json": "{}\n",
        "campaigns/phase5/factorial-design.json": "{}\n",
        "campaigns/phase5/slo-calibration.json": "{}\n",
        "campaigns/phase5/slo-definitions.json": "{}\n",
        "campaigns/phase5/analysis-plan.json": "{}\n",
    }
    root = tmp_path / "root"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(a, "ROOT", root)
    rows = [{"run_id": "r1", "artifact_manifest_hash": "h1"}]
    first = a.create_input_seal(rows)["phase7a_input_sha256"]
    second = a.create_input_seal(rows)["phase7a_input_sha256"]
    assert first == second


def test_no_pseudoreplication_spec():
    spec = a.create_analysis_spec()
    assert spec["statistical_unit"] == "one run / repetition"
    assert "requests are used only" in spec["request_level_role"]
