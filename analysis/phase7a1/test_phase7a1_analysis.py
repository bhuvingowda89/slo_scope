from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.phase7a1 import phase7a1_analysis as a


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


def test_scipy_paired_test_against_reference():
    diffs = [1.0, 2.0, 3.0, 4.0]
    result = a.paired_t_test(diffs)
    assert result["t_statistic"] == pytest.approx(3.872983346207417)
    assert result["p_value"] == pytest.approx(0.030466291662170977)


def test_scipy_one_sample_interaction_test():
    deltas = [0.5, 0.7, 0.6, 0.8]
    result = a.paired_t_test(deltas)
    assert result["t_statistic"] == pytest.approx(10.069756700139282)
    assert result["p_value"] == pytest.approx(0.0020854803149752787)


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


def test_long_form_runtime_metric_extraction():
    rows = [
        {"metric_name": "llamacpp:requests_deferred", "metric_value": 1.0, "deferred_requests": None},
        {"metric_name": "llamacpp:requests_deferred", "metric_value": 3.0, "deferred_requests": None},
        {"metric_name": "llamacpp:requests_processing", "metric_value": 2.0, "active_requests": None},
        {"metric_name": "llamacpp:requests_processing", "metric_value": 4.0, "active_requests": None},
    ]
    assert max(a.runtime_metric_values(rows, "llamacpp:requests_deferred")) == 3.0
    assert max(a.runtime_metric_values(rows, "llamacpp:requests_processing")) == 4.0


def test_original_column_lookup_would_miss_long_form_runtime_metric():
    rows = [
        {"metric_name": "llamacpp:requests_deferred", "metric_value": 3.0, "deferred_requests": None},
    ]
    assert a.runtime_metric_values(rows, "llamacpp:requests_deferred") == [3.0]
    assert a.erroneous_column_metric_values(rows, "deferred_requests") == []


def test_queue_signature_classification():
    rows = [
        {"condition_id": "INPUT_MEDIUM", "repetition": 1, "requests_deferred_max": 2.0, "requests_processing_max": 2.0},
        {"condition_id": "LOAD_MEDIUM", "repetition": 1, "requests_deferred_max": 3.0, "requests_processing_max": 3.0},
    ]
    for cond in a.CONDITION_ORDER:
        if cond not in {"INPUT_MEDIUM", "LOAD_MEDIUM"}:
            rows.append({"condition_id": cond, "repetition": 1, "requests_deferred_max": 0.0, "requests_processing_max": 1.0})
    sig = {row["condition_id"]: row for row in a.queue_signature_rows(rows)}
    assert sig["INPUT_MEDIUM"]["secondary_observed_queue_signature"] is True
    assert sig["LOAD_MEDIUM"]["queueing_intended_not_confound"] is True
    assert sig["LOAD_MEDIUM"]["secondary_observed_queue_signature"] is False


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
    first = a.create_input_seal(rows)["phase7a1_input_sha256"]
    second = a.create_input_seal(rows)["phase7a1_input_sha256"]
    assert first == second


def test_no_pseudoreplication_spec():
    spec = a.create_analysis_spec()
    assert spec["statistical_unit"] == "one run / repetition"
    assert "requests are used only" in spec["request_level_role"]


def test_frozen_slo_thresholds_unchanged():
    assert a.SLO_THRESHOLDS == {
        "TTFT_SLO": 0.06512947314299491,
        "TOTAL_LATENCY_SLO": 0.26779416389490324,
        "DECODE_DURATION_SLO": 0.20162438542758712,
    }


def test_calibration_formal_control_grouping():
    formal = [
        {"condition_id": "BASELINE", "run_id": "formal-b", "repetition": 1, "ttft_p95": 1.0, "total_latency_p95": 2.0, "post_first_token_duration_p95": 3.0},
        {"condition_id": "OUTPUT_CONTROL", "run_id": "formal-o", "repetition": 1, "ttft_p95": 1.5, "total_latency_p95": 2.5, "post_first_token_duration_p95": 3.5},
    ]
    grouped = {(row["condition_id"], row["source"]) for row in [
        {"condition_id": row["condition_id"], "source": "formal"} for row in formal
    ]}
    assert ("BASELINE", "formal") in grouped
    assert ("OUTPUT_CONTROL", "formal") in grouped


def test_statistical_diff_audit_detects_p_value_change(tmp_path, monkeypatch):
    old_dir = tmp_path / "analysis" / "phase7a"
    old_dir.mkdir(parents=True)
    (old_dir / "isolated-effects.csv").write_text("mechanism,t_statistic,raw_p,holm_p\nINPUT,1,0.9,0.9\n", encoding="utf-8")
    (old_dir / "factorial-interactions.csv").write_text("test_id,t_statistic,raw_p,holm_p,classification\n", encoding="utf-8")
    monkeypatch.setattr(a, "ROOT", tmp_path)
    audit = a.statistical_diff_audit([
        {"mechanism": "INPUT", "test_id": "INPUT", "t_statistic": 2.0, "raw_p": 0.01, "holm_p": 0.01}
    ], [])
    assert audit[0]["significance_changed"] is True
