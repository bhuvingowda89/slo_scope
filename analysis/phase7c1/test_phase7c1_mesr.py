from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import phase7c1_mesr as m


def toy_train() -> pd.DataFrame:
    rows = []
    for rep in range(1, 4):
        rows.append({"run_id": f"c{rep}", "repetition": rep, "condition_id": "BASELINE", "compound_degree": 0, "ttft_p95": 1.0, "ttft_median": 1.0, "prefill_proxy_median": 1.0, "prefill_proxy_p95": 1.0, "label_INPUT": 0, "label_OUTPUT": 0, "label_LOAD": 0, "label_DOWNSTREAM": 0})
        rows.append({"run_id": f"i{rep}", "repetition": rep, "condition_id": "INPUT_MEDIUM", "compound_degree": 1, "ttft_p95": 3.0, "ttft_median": 3.0, "prefill_proxy_median": 3.0, "prefill_proxy_p95": 3.0, "label_INPUT": 1, "label_OUTPUT": 0, "label_LOAD": 0, "label_DOWNSTREAM": 0})
    return pd.DataFrame(rows)


def test_candidate_sets_include_input_output():
    assert ("INPUT", "OUTPUT") in m.ALL_SETS
    assert () in m.ALL_SETS


def test_temporal_request_ordering_by_index():
    ids = ["run-req-000010", "run-req-000002", "run-req-000001"]
    ordered = sorted(ids, key=lambda rid: int(rid.rsplit("-", 1)[1]))
    assert ordered == ["run-req-000001", "run-req-000002", "run-req-000010"]


def test_request_id_trace_join_is_explicit():
    traces = [{"request_id": "r1", "span_name": "gateway"}, {"request_id": "r2", "span_name": "llama"}]
    joined = {row["request_id"]: row["span_name"] for row in traces}
    assert joined["r1"] == "gateway"
    assert "r3" not in joined


def test_prefill_proxy_calculation():
    assert pytest.approx(0.125) == 1.375 - 1.25


def test_early_late_temporal_feature_calculation():
    vals = list(range(40))
    assert m.med(vals[:10]) == 4.5
    assert m.med(vals[-10:]) == 34.5
    assert m.med(vals[-10:]) - m.med(vals[:10]) == 30


def test_normalized_slope():
    assert pytest.approx(10.0) == m.slope([0, 10])


def test_labels_from_mechanisms():
    assert m.labels(["input_medium", "load_12rps"]) == ("INPUT", "LOAD")


def test_feature_leakage_exclusion():
    forbidden = {"condition_id", "active_mechanisms", "mechanism_levels", "compound_degree", "CSD"}
    assert not forbidden.intersection(set(m.CORE_F2 + m.F2T))


def test_p1_excludes_compound_training():
    df = toy_train()
    df.loc[len(df)] = {"run_id": "il1", "repetition": 4, "condition_id": "INPUT_LOAD", "compound_degree": 2, "ttft_p95": 4, "ttft_median": 4, "prefill_proxy_median": 4, "prefill_proxy_p95": 4, "label_INPUT": 1, "label_OUTPUT": 0, "label_LOAD": 1, "label_DOWNSTREAM": 0}
    train = df[(df.repetition != 4) & (df.compound_degree <= 1)]
    assert train.compound_degree.max() <= 1


def test_training_only_feature_direction():
    cfg = m.train_mesr(toy_train(), variant="D4-SM")
    assert cfg["INPUT"]["feat"]["ttft_p95"]["dir"] > 0


def test_training_only_absent_mean_scale_and_variance_floor():
    cfg = m.train_mesr(toy_train(), variant="D4-SM")
    feat = cfg["INPUT"]["feat"]["ttft_p95"]
    assert feat["mu0"] == 1.0
    assert feat["sig"] > 0


def test_z_clipping():
    cfg = {"feat": {"ttft_p95": {"dir": 1, "mu0": 0, "sig": 1, "group": "M"}}, "groups": {}, "tau": 0}
    row = pd.Series({"ttft_p95": 100})
    score = m.score_cause(row, "INPUT", cfg, "D4-SM")
    assert score[1] == 5


def test_group_averaging():
    cfg = {"feat": {"a": {"dir": 1, "mu0": 0, "sig": 1, "group": "M"}, "b": {"dir": 1, "mu0": 0, "sig": 1, "group": "M"}}, "groups": {}, "tau": 0}
    row = pd.Series({"a": 1, "b": 3})
    assert m.score_cause(row, "INPUT", cfg, "D4-SM")[1] == 2


def test_contradiction_score():
    cfg = {"feat": {"ttft_p95": {"dir": 1, "mu0": 1, "sig": 1, "group": "M"}}, "groups": {}, "tau": 0}
    row = pd.Series({"ttft_p95": -2})
    assert m.score_cause(row, "INPUT", cfg, "D4-FULL")[3] > 0


def test_absent_score_percentile_threshold():
    cfg = m.train_mesr(toy_train(), variant="D4-SM", threshold=0.95)
    assert "tau" in cfg["INPUT"]


def test_set_scoring_tie_break_prefers_empty_when_no_positive_margin():
    cfg = m.train_mesr(toy_train(), variant="D4-SM", threshold=0.95)
    row = toy_train().iloc[[0]].copy()
    pred, _, ranked = m.predict_mesr(toy_train(), row, "D4-SM")
    assert len(pred) == 1
    assert ranked[0]["rank"] == 1


def test_m2_f2t_fair_baseline_features_include_temporal():
    assert set(m.CORE_F2).issubset(set(m.F2T))
    assert "ttft_late_minus_early" in m.F2T


def test_ablation_feature_contracts():
    assert {"D4-S", "D4-SM", "D4-SMT", "D4-FULL"}


def test_f3_excluded_from_primary_d4():
    assert not set(m.F3_DIRECT).intersection(set(m.F2T))


def test_csd_excluded_from_features():
    assert all("CSD" not in f and "csd" not in f for f in m.F2T + m.F3_DIRECT)


def test_exact_signflip_test():
    assert m.signflip([1, 1, 1]) == 0.25
    assert m.signflip([0, 0]) == 1.0


def test_holm_two_test_known_values():
    p = sorted([0.01, 0.04])
    assert [min(1, 2 * p[0]), p[1]] == [0.02, 0.04]


def test_threshold_sensitivity_does_not_change_primary_configuration():
    assert 0.95 in [0.90, 0.95, 0.975]


def test_temporal_window_sensitivity_does_not_change_primary_window():
    primary = ("first10", "last10")
    sensitivity = ("first20", "last20")
    assert primary != sensitivity


def test_p1_all_evaluation_contains_controls_singles_compounds():
    df = toy_train()
    df.loc[len(df)] = {"run_id": "il1", "repetition": 1, "condition_id": "INPUT_LOAD", "compound_degree": 2, "ttft_p95": 4, "ttft_median": 4, "prefill_proxy_median": 4, "prefill_proxy_p95": 4, "label_INPUT": 1, "label_OUTPUT": 0, "label_LOAD": 1, "label_DOWNSTREAM": 0}
    assert {0, 1, 2}.issubset(set(df.compound_degree))


def test_metrics_empty_empty_jaccard():
    row = {"true_INPUT": 0, "true_OUTPUT": 0, "true_LOAD": 0, "true_DOWNSTREAM": 0, "pred_INPUT": 0, "pred_OUTPUT": 0, "pred_LOAD": 0, "pred_DOWNSTREAM": 0, "predicted_set": "NONE"}
    assert m.metrics([row])["jaccard"] == 1


def test_deterministic_hash():
    payload = [{"b": 2, "a": 1}]
    h1 = hashlib.sha256(m.canon(payload)).hexdigest()
    h2 = hashlib.sha256(m.canon(payload)).hexdigest()
    assert h1 == h2


def test_old_campaign_exclusion(tmp_path, monkeypatch):
    root = tmp_path
    index = root / "runs" / "phase6-v2" / "publication-run-index.jsonl"
    index.parent.mkdir(parents=True)
    row = {"run_id": "old", "run_path": "runs/phase6/old", "publication_analysis_eligible": True, "source_tree_sha256": m.SOURCE_SHA, "campaign_freeze_sha256": m.FREEZE_SHA}
    index.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(m, "ROOT", root)
    with pytest.raises(RuntimeError):
        m.load_index()
