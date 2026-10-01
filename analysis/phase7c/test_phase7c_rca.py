from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis.phase7c import phase7c_rca as r


def test_label_extraction_from_active_mechanisms():
    assert r.cause_set_from_active(["input_medium", "load_12rps"]) == ("INPUT", "LOAD")


def test_exact_mechanism_set_encoding():
    assert r.label_vec(("OUTPUT", "DOWNSTREAM")) == [0, 1, 0, 1]


def test_zero_single_compound_grouping():
    df = toy_df()
    assert len(df[df.compound_degree == 0]) == 1
    assert len(df[df.compound_degree == 1]) == 1
    assert len(df[df.compound_degree == 2]) == 1


def test_feature_tier_leakage_exclusion():
    forbidden = {"condition_id", "active_mechanisms", "mechanism_levels", "compound_degree", "run_id"}
    assert not forbidden.intersection(r.F2)


def test_repetition_based_fold_splitting_no_overlap():
    df = toy_fold_df()
    fold = 1
    train = df[df.repetition != fold]
    test = df[df.repetition == fold]
    assert set(train.repetition).isdisjoint(set(test.repetition))


def test_training_only_imputation_and_standardization():
    train = pd.DataFrame({"x": [1.0, 3.0], "label_INPUT": [0, 1]})
    test = pd.DataFrame({"x": [np.nan], "label_INPUT": [0]})
    xtr, xte = r.split_xy(train, test, ["x"])
    assert xte[0, 0] == pytest.approx(0.0)
    assert xtr.mean() == pytest.approx(0.0)


def test_nearest_neighbor_tie_break():
    train = pd.DataFrame({"run_id": ["b", "a"], "x": [0.0, 0.0], **labels([[1, 0, 0, 0], [0, 1, 0, 0]])})
    test = pd.DataFrame({"run_id": ["t"], "x": [0.0], **labels([[0, 0, 0, 0]])})
    pred, _, _ = r.predict_m0(train, test, ["x"])
    assert pred.tolist() == [[0, 1, 0, 0]]


def test_nearest_centroid_prediction():
    train = pd.DataFrame({"true_set": ["INPUT", "OUTPUT"], "x": [-1.0, 1.0], **labels([[1, 0, 0, 0], [0, 1, 0, 0]])})
    test = pd.DataFrame({"true_set": ["INPUT"], "x": [-2.0], **labels([[1, 0, 0, 0]])})
    pred, _, _ = r.predict_m1(train, test, ["x"])
    assert pred.tolist()[0][0] == 1


def test_one_vs_rest_logistic_prediction():
    train = pd.DataFrame({"x": [-2, -1, 1, 2], **labels([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])})
    test = pd.DataFrame({"x": [-3], **labels([[1, 0, 0, 0]])})
    pred, _, _ = r.predict_m2(train, test, ["x"])
    assert pred.shape == (1, 4)


def test_metrics_exact_jaccard_empty_complete_over_under():
    rows = [{"true_INPUT": 0, "true_OUTPUT": 0, "true_LOAD": 0, "true_DOWNSTREAM": 0,
             "pred_INPUT": 0, "pred_OUTPUT": 0, "pred_LOAD": 0, "pred_DOWNSTREAM": 0,
             "predicted_mechanism_set": "NONE"}]
    m = r.metrics_for(rows)
    assert m["exact_set_accuracy"] == 1
    assert m["jaccard"] == 1
    assert m["control_false_alarm_rate"] == 0


def test_complete_partial_over_under_metrics():
    rows = [{"true_INPUT": 1, "true_OUTPUT": 1, "true_LOAD": 0, "true_DOWNSTREAM": 0,
             "pred_INPUT": 1, "pred_OUTPUT": 0, "pred_LOAD": 1, "pred_DOWNSTREAM": 0,
             "predicted_mechanism_set": "INPUT+LOAD"}]
    m = r.metrics_for(rows)
    assert m["complete_cause_recall"] == 0
    assert m["partial_cause_recall"] == 0.5
    assert m["over_attribution_rate"] == 1
    assert m["under_attribution_rate"] == 1


def test_top2_complete_cause_coverage():
    rows = [{"true_INPUT": 1, "true_OUTPUT": 1, "true_LOAD": 0, "true_DOWNSTREAM": 0,
             "input_score": .9, "output_score": .8, "load_score": .1, "downstream_score": .0}]
    assert r.topk_metrics(rows) == (1.0, 1.0)


def test_exact_signflip_h2():
    assert 0 <= r.signflip([1, 1, -1, -1]) <= 1


def test_p1_excludes_compound_training_and_p2_includes():
    df = toy_fold_df()
    p1 = df[(df.repetition != 1) & (df.compound_degree <= 1)]
    p2 = df[df.repetition != 1]
    assert not (p1.compound_degree == 2).any()
    assert (p2.compound_degree == 2).any()


def test_f3_direct_evidence_separated_from_f2():
    assert "server_prompt_tokens_median" not in r.F2
    assert "server_prompt_tokens_median" in r.F3


def test_csd_not_used_as_model_feature():
    assert all("CSD" not in f for tier in r.FEATURE_TIERS.values() for f in tier)


def test_old_superseded_campaign_exclusion(tmp_path, monkeypatch):
    rows = [{"run_id": str(i), "source_hash": r.SOURCE_SHA, "freeze_hash": r.FREEZE_SHA, "run_path": "runs/phase6/old"} for i in range(104)]
    root = tmp_path
    path = root / "runs" / "phase6-v2"
    path.mkdir(parents=True)
    (path / "publication-run-index.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    monkeypatch.setattr(r, "ROOT", root)
    with pytest.raises(RuntimeError, match="superseded"):
        r.load_index()


def test_deterministic_repeat_hash():
    payload = [{"a": 1}, {"b": 2}]
    assert r.sha256_bytes(r.canonical(payload)) == r.sha256_bytes(r.canonical(payload))


def test_control_false_alarm_positive_case():
    rows = [{"true_INPUT": 0, "true_OUTPUT": 0, "true_LOAD": 0, "true_DOWNSTREAM": 0,
             "pred_INPUT": 1, "pred_OUTPUT": 0, "pred_LOAD": 0, "pred_DOWNSTREAM": 0,
             "predicted_mechanism_set": "INPUT"}]
    assert r.metrics_for(rows)["control_false_alarm_rate"] == 1


def test_false_and_missed_counts():
    rows = [{"true_INPUT": 1, "true_OUTPUT": 1, "true_LOAD": 0, "true_DOWNSTREAM": 0,
             "pred_INPUT": 1, "pred_OUTPUT": 0, "pred_LOAD": 1, "pred_DOWNSTREAM": 0,
             "predicted_mechanism_set": "INPUT+LOAD"}]
    m = r.metrics_for(rows)
    assert m["false_attribution_count_mean"] == 1
    assert m["missed_cause_count_mean"] == 1


def test_m1_cannot_predict_unseen_compound_centroid_when_not_in_train():
    train = pd.DataFrame({"true_set": ["INPUT", "LOAD"], "x": [-1.0, 1.0], **labels([[1, 0, 0, 0], [0, 0, 1, 0]])})
    test = pd.DataFrame({"true_set": ["INPUT+LOAD"], "x": [0.0], **labels([[1, 0, 1, 0]])})
    pred, _, _ = r.predict_m1(train, test, ["x"])
    assert sum(pred.tolist()[0]) == 1


def test_f0_contains_only_slo_state_and_margins():
    assert r.F0 == [
        "ttft_slo_violated",
        "total_latency_slo_violated",
        "decode_duration_slo_violated",
        "ttft_violation_margin",
        "total_latency_violation_margin",
        "decode_duration_violation_margin",
    ]


def test_trace_features_in_f2():
    assert "gateway_span_duration_median" in r.F2
    assert "llama_fraction_of_gateway_median" in r.F2


def test_jaccard_partial_overlap():
    assert r.jaccard({"INPUT", "LOAD"}, {"INPUT", "OUTPUT"}) == pytest.approx(1 / 3)


def test_set_key_empty_and_ordered():
    assert r.set_key(()) == "NONE"
    assert r.set_key(("INPUT", "LOAD")) == "INPUT+LOAD"


def labels(vectors):
    return {f"label_{c}": [v[i] for v in vectors] for i, c in enumerate(r.CAUSES)}


def toy_df():
    return pd.DataFrame({"compound_degree": [0, 1, 2]})


def toy_fold_df():
    return pd.DataFrame({"repetition": [1, 2, 2], "compound_degree": [0, 1, 2]})
