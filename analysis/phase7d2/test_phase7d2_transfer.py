from __future__ import annotations

import json
from pathlib import Path

import phase7d2_transfer as t


def test_publication_index_enforcement_and_pilot_exclusion():
    source = t.load_publication_index(t.SOURCE_INDEX, 104, "source")
    target = t.load_publication_index(t.TARGET_INDEX, 66, "target")
    assert len({row["run_id"] for row in source}) == 104
    assert len({row["run_id"] for row in target}) == 66
    assert all("diagnostics/phase7d0" not in row["run_path"] for row in target)
    assert all(not row["run_path"].startswith("runs/phase6/") for row in source)


def test_target_label_extraction_and_distribution():
    target = t.load_publication_index(t.TARGET_INDEX, 66, "target")
    rows = [t.run_features(row, "target") for row in target[:11]]
    assert {row["condition_id"] for row in rows} <= set(t.TARGET_CONDITIONS)
    for row in rows:
        assert row["truth_set"] == t.set_key([c for c in t.CAUSES if row[f"label_{c}"]])


def test_source_only_preprocessing_shapes_and_no_target_fit():
    source_rows = t.load_source_dataset()
    train = [row for row in source_rows if int(row["compound_degree"]) <= 1]
    target = [t.run_features(row, "target") for row in t.load_publication_index(t.TARGET_INDEX, 66, "target")[:3]]
    _xtr, _xte, prep = t.median_impute_and_standardize(train, target, t.CORE_F2)
    assert set(prep["medians"]) == set(t.CORE_F2)
    # Source medians are computed from source training rows only.
    feat = t.CORE_F2[0]
    source_values = [t.as_float(row.get(feat)) for row in train]
    assert prep["medians"][feat] == t.median([v for v in source_values if v is not None])


def test_f2_f2t_f3_feature_contracts_and_separation():
    assert set(t.CORE_F2).issubset(set(t.F2T))
    assert "server_prompt_tokens_median" not in t.CORE_F2
    assert "server_output_tokens_median" not in t.F2T
    assert set(t.F3_DIRECT).issubset(set(t.F3))


def test_generalization_drop_and_retention():
    source = 0.8
    target = 0.6
    assert source - target == 0.20000000000000007
    assert target / source == 0.7499999999999999


def test_exact_64_signflip_permutations_and_h4_rule():
    diffs = [0, 0, 0, 0, 0, 0]
    assert t.exact_signflip(diffs) == 1.0
    positive = [1, 1, 1, 1, 1, 1]
    assert t.exact_signflip(positive) == 0.03125


def test_pair_transfer_and_control_false_alarm_metrics():
    rows = [
        {"true_INPUT": 0, "true_OUTPUT": 0, "true_LOAD": 0, "true_DOWNSTREAM": 0, "pred_INPUT": 1, "pred_OUTPUT": 0, "pred_LOAD": 0, "pred_DOWNSTREAM": 0, "predicted_set": "INPUT"},
        {"true_INPUT": 0, "true_OUTPUT": 0, "true_LOAD": 0, "true_DOWNSTREAM": 0, "pred_INPUT": 0, "pred_OUTPUT": 0, "pred_LOAD": 0, "pred_DOWNSTREAM": 0, "predicted_set": "NONE"},
    ]
    assert t.metric_summary(rows)["control_false_alarm_rate"] == 0.5


def test_feature_shift_sms_and_range_overlap():
    src = [{"x": 0.0}, {"x": 2.0}]
    tgt = [{"x": 1.0}, {"x": 3.0}]
    row = t.feature_shift(src, tgt, ["x"])[0]
    assert row["standardized_mean_shift"] > 0
    assert 0 <= row["range_overlap"] <= 1


def test_control_normalization_uses_controls_only():
    source = [
        {"condition_id": "BASELINE", "x": 1.0},
        {"condition_id": "OUTPUT_CONTROL", "x": 3.0},
        {"condition_id": "INPUT_MEDIUM", "x": 1000.0},
    ]
    target = [
        {"condition_id": "BASELINE", "x": 10.0},
        {"condition_id": "OUTPUT_CONTROL", "x": 14.0},
        {"condition_id": "INPUT_MEDIUM", "x": 12.0},
    ]
    aligned = t.control_align_target(source, target, ["x"])
    assert aligned[2]["x"] != target[2]["x"]
    assert aligned[2]["x"] < 1000.0


def test_target_csd_prohibited_in_spec_and_slo_excluded():
    spec = json.loads(Path("analysis/phase7d2/generalization-spec.json").read_text())
    assert "prohibited" in spec["target_csd_policy"]
    assert "excluded" in spec["slo_policy"]


def test_source_references_reproduced():
    refs = t.source_reference_check()
    assert refs == t.SOURCE_REFERENCES


def test_determinism_audit_hashes_identical():
    audit = json.loads(Path("analysis/phase7d2/determinism-audit.json").read_text())
    assert audit["prediction_hashes_identical"] is True
    assert audit["table_hashes_identical"] is True


def test_no_target_rows_in_training_index():
    rows = Path("analysis/phase7d2/source-training-index.csv").read_text()
    assert "phase7d-qwen15b" not in rows


def test_no_target_rows_in_logistic_fitting_by_contract():
    spec = json.loads(Path("analysis/phase7d2/generalization-spec.json").read_text())
    assert "source-only" in spec["preprocessing"]
    assert "source Phase6-V2" in spec["source_training"]


def test_no_target_rows_in_d4_evidence_fitting_by_contract():
    spec = json.loads(Path("analysis/phase7d2/generalization-spec.json").read_text())
    assert "source-only distributions" in spec["d4_contract"]


def test_target_evaluation_subsets_defined():
    spec = json.loads(Path("analysis/phase7d2/generalization-spec.json").read_text())
    assert set(spec["target_evaluation_subsets"]) == {"TARGET-ALL", "TARGET-CONTROL", "TARGET-SINGLE", "TARGET-COMPOUND"}


def test_target_condition_counts_are_six_each():
    rows = list(__import__("csv").DictReader(Path("analysis/phase7d2/target-evaluation-index.csv").open()))
    counts = {}
    for row in rows:
        counts[row["condition_id"]] = counts.get(row["condition_id"], 0) + 1
    assert counts == {condition: 6 for condition in t.TARGET_CONDITIONS}


def test_target_all_compound_control_counts():
    rows = list(__import__("csv").DictReader(Path("analysis/phase7d2/target-evaluation-index.csv").open()))
    assert len(rows) == 66
    assert sum(row["compound_degree"] == "0" for row in rows) == 12
    assert sum(row["compound_degree"] == "1" for row in rows) == 24
    assert sum(row["compound_degree"] == "2" for row in rows) == 30


def test_repetition_level_target_metrics_have_six_blocks():
    rows = list(__import__("csv").DictReader(Path("analysis/phase7d2/h4-fold-comparison.csv").open()))
    reps = [row for row in rows if row["repetition"] != "SUMMARY"]
    assert len(reps) == 6
    assert {row["repetition"] for row in reps} == {"1", "2", "3", "4", "5", "6"}


def test_h4_decision_rule_applied():
    h4 = json.loads(Path("analysis/phase7d2/h4-test.json").read_text())
    expected = "SUPPORTED" if h4["mean_difference"] > 0 and h4["exact_signflip_p"] < 0.05 else "NOT_SUPPORTED"
    assert h4["h4_status"] == expected


def test_pair_transfer_has_all_method_pair_rows():
    rows = list(__import__("csv").DictReader(Path("analysis/phase7d2/table-13-pair-transfer.csv").open()))
    methods = {"M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"}
    assert len(rows) == len(methods) * len(t.COMPOUND_CONDITIONS)
    assert {(row["method"], row["compound_pair"]) for row in rows} == {(m, c) for m in methods for c in t.COMPOUND_CONDITIONS}


def test_control_false_alarm_rates_present():
    rows = list(__import__("csv").DictReader(Path("analysis/phase7d2/table-12-zero-shot-generalization.csv").open()))
    assert all(row["target_control_false_alarm_rate"] != "" for row in rows)


def test_source_slo_features_not_in_primary_methods():
    registry = json.loads(Path("analysis/phase7d2/feature-registry.json").read_text())
    forbidden = {"ttft_slo_violated", "total_latency_slo_violated", "decode_duration_slo_violated"}
    assert forbidden.isdisjoint(registry["F2"])
    assert forbidden.isdisjoint(registry["F2T"])


def test_queue_metrics_excluded_from_primary_transfer():
    registry = json.loads(Path("analysis/phase7d2/feature-registry.json").read_text())
    assert registry["queue_metrics"]["llamacpp:requests_deferred"].startswith("excluded")


def test_control_normalized_predictions_are_secondary():
    spec = json.loads(Path("analysis/phase7d2/generalization-spec.json").read_text())
    assert "control-normalized M2/F2T" in spec["secondary_comparisons"]


def test_control_normalized_fault_labels_not_in_spec_formula():
    spec = json.loads(Path("analysis/phase7d2/generalization-spec.json").read_text())
    assert "BASELINE and OUTPUT_CONTROL only" in spec["control_normalization"]


def test_figures_exist():
    for name in [
        "figure-18-source-vs-target-rca.svg",
        "figure-19-pair-transfer.svg",
        "figure-20-feature-shift.svg",
        "figure-21-cause-recall-transfer.svg",
    ]:
        assert (Path("analysis/phase7d2/figures") / name).exists()


def test_phase7d2_hashes_present():
    seal = json.loads(Path("analysis/phase7d2/input-seal.json").read_text())
    spec = json.loads(Path("analysis/phase7d2/generalization-spec.json").read_text())
    assert len(seal["phase7d2_input_sha256"]) == 64
    assert len(spec["phase7d2_analysis_spec_sha256"]) == 64
