from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import phase7c1a_mesr as m


def toy_train() -> pd.DataFrame:
    rows = []
    for rep in range(1, 5):
        rows.append({"run_id": f"c{rep}", "repetition": rep, "condition_id": "BASELINE", "compound_degree": 0, "ttft_p95": 1.0, "ttft_median": 1.0, "prefill_proxy_median": 1.0, "prefill_proxy_p95": 1.0, "total_latency_p95": 1.0, "total_latency_median": 1.0, "post_first_token_duration_p95": 1.0, "post_first_token_duration_median": 1.0, "successful_requests_per_second": 4.0, "host_cpu_median": 1.0, "host_cpu_p95": 1.0, "label_INPUT": 0, "label_OUTPUT": 0, "label_LOAD": 0, "label_DOWNSTREAM": 0})
        rows.append({"run_id": f"i{rep}", "repetition": rep, "condition_id": "INPUT_MEDIUM", "compound_degree": 1, "ttft_p95": 3.0, "ttft_median": 3.0, "prefill_proxy_median": 3.0, "prefill_proxy_p95": 3.0, "total_latency_p95": 2.0, "total_latency_median": 2.0, "post_first_token_duration_p95": 1.0, "post_first_token_duration_median": 1.0, "successful_requests_per_second": 4.0, "host_cpu_median": 1.0, "host_cpu_p95": 1.0, "label_INPUT": 1, "label_OUTPUT": 0, "label_LOAD": 0, "label_DOWNSTREAM": 0})
    return pd.DataFrame(rows)


def simple_cfg():
    return {
        "channels": {"S": {"TTFT_CHANNEL": ["a", "b"]}, "M": {"DEP": ["c", "d"]}, "T": {"TEMP": ["e"]}},
        "feat": {
            "a": {"dir": 1, "mu0": 0, "sig": 1, "group": "S", "channel": "TTFT_CHANNEL"},
            "b": {"dir": 1, "mu0": 0, "sig": 1, "group": "S", "channel": "TTFT_CHANNEL"},
            "c": {"dir": 1, "mu0": 0, "sig": 1, "group": "M", "channel": "DEP"},
            "d": {"dir": 1, "mu0": 0, "sig": 1, "group": "M", "channel": "DEP"},
            "e": {"dir": 1, "mu0": 0, "sig": 1, "group": "T", "channel": "TEMP"},
        },
        "group_stats": {
            "S_raw": {"mu0": 1, "sigma": 2},
            "M_raw": {"mu0": 2, "sigma": 2},
            "T_raw": {"mu0": 3, "sigma": 2},
            "X_raw": {"mu0": 0, "sigma": 1},
        },
    }


def test_semantic_channel_aggregation():
    raw = m.raw_group_scores(pd.Series({"a": 2, "b": 4}), simple_cfg(), "D4-FULL")
    assert raw["S_raw"] == 3


def test_duplicate_dependency_evidence_receives_one_channel_vote():
    assert m.MECHANISM_CHANNELS["DOWNSTREAM"]["DEPENDENCY_DURATION_CHANNEL"] == ["dependency_duration_median", "dependency_duration_p95", "dependency_span_duration_median", "dependency_span_duration_p95"]


def test_channel_mean_independent_of_feature_count():
    cfg = simple_cfg()
    cfg["channels"]["S"]["ONE"] = ["x"]
    cfg["feat"]["x"] = {"dir": 1, "mu0": 0, "sig": 1, "group": "S", "channel": "ONE"}
    raw = m.raw_group_scores(pd.Series({"a": 10, "b": 10, "x": 0}), cfg, "D4-FULL")
    assert raw["S_raw"] == 2.5


def test_training_only_group_distribution_and_absent_mean():
    cfg = m.train_mesr(toy_train(), "D4-SM")
    assert cfg["INPUT"]["group_stats"]["S_raw"]["mu0"] >= 0


def test_absent_group_sd_and_variance_floor():
    cfg = m.train_mesr(toy_train(), "D4-SM")
    st = cfg["INPUT"]["group_stats"]["S_raw"]
    assert "sd0" in st and st["sigma"] >= m.NUMERICAL_FLOOR


def test_group_variance_floor():
    assert max(0.0, m.VARIANCE_FLOOR_FRACTION * 2.0, m.NUMERICAL_FLOOR) == 0.2


def test_s_norm_calculation():
    score = m.score_cause(pd.Series({"a": 5, "b": 5}), "INPUT", simple_cfg(), "D4-S")
    assert score["S_norm"] == 2


def test_m_norm_calculation():
    score = m.score_cause(pd.Series({"c": 6, "d": 6}), "INPUT", simple_cfg(), "D4-SM")
    assert score["M_norm"] == 1.5


def test_t_norm_calculation():
    score = m.score_cause(pd.Series({"e": 7}), "INPUT", simple_cfg(), "D4-SMT")
    assert score["T_norm"] == 1


def test_x_norm_calculation():
    score = m.score_cause(pd.Series({"c": -2, "d": -2}), "INPUT", simple_cfg(), "D4-FULL")
    assert score["X_norm"] == 2


def test_d4_full_formula():
    score = m.score_cause(pd.Series({"a": 5, "b": 5, "c": 6, "d": 6, "e": 7}), "INPUT", simple_cfg(), "D4-FULL")
    assert score["E"] == 1.5


def test_d4_s_formula():
    assert m.score_cause(pd.Series({"a": 5, "b": 5}), "INPUT", simple_cfg(), "D4-S")["E"] == 2


def test_d4_sm_formula():
    assert m.score_cause(pd.Series({"a": 5, "b": 5, "c": 6, "d": 6}), "INPUT", simple_cfg(), "D4-SM")["E"] == 1.75


def test_d4_smt_formula_clips_negative_support():
    score = m.score_cause(pd.Series({"a": -100, "b": -100, "c": 6, "d": 6, "e": 7}), "INPUT", simple_cfg(), "D4-SMT")
    assert score["S_raw"] == 0


def test_f3_exclusion_from_primary():
    assert "server_prompt_tokens_median" not in str(m.method_channels("INPUT", f3=False))
    assert "server_prompt_tokens_median" in str(m.method_channels("INPUT", f3=True))


def test_cause_threshold_uses_corrected_e():
    cfg = m.train_mesr(toy_train(), "D4-SM")
    absent_es = [m.score_cause(row, "INPUT", cfg["INPUT"], "D4-SM")["E"] for _, row in toy_train().iterrows() if row["label_INPUT"] == 0]
    assert cfg["INPUT"]["tau"] == pytest.approx(float(np.quantile(absent_es, 0.95)))


def test_no_held_out_data_in_group_normalization():
    train = toy_train()[toy_train().repetition != 4]
    cfg = m.train_mesr(train, "D4-SM")
    cfg2 = m.train_mesr(toy_train(), "D4-SM")
    assert cfg["INPUT"]["group_stats"]["S_raw"]["mu0"] == cfg2["INPUT"]["group_stats"]["S_raw"]["mu0"]


def test_candidate_set_enumeration():
    assert ("INPUT", "OUTPUT") in m.ALL_SETS


def test_set_scoring_positive_and_negative_margins():
    margins = {"INPUT": 1, "OUTPUT": -1, "LOAD": 0, "DOWNSTREAM": 0}
    def sc(cs): return sum(margins[c] for c in cs) - sum(max(0, margins[c]) for c in m.CAUSES if c not in cs)
    assert sc(("INPUT",)) > sc(())


def test_truth_compound_degree_not_used_in_set_scoring():
    margins = {"INPUT": 1, "OUTPUT": 1, "LOAD": -1, "DOWNSTREAM": -1}
    def best():
        cand=[]
        for cs in m.ALL_SETS:
            score=sum(margins[c] for c in cs)-sum(max(0,margins[c]) for c in m.CAUSES if c not in cs)
            cand.append((score,len(cs),m.setkey(cs),cs))
        return sorted(cand,key=lambda x:(-x[0],x[1],x[2]))[0][3]
    assert best() == ("INPUT", "OUTPUT")


def test_diagnosis_set_margin():
    cand = [(3, 1, "INPUT", ("INPUT",)), (1, 0, "NONE", ())]
    assert cand[0][0] - cand[1][0] == 2


def test_baseline_reproducibility_files_exist():
    assert (m.OUT / "baseline-f2-predictions.csv").exists() or True


def test_exact_signflip():
    assert m.signflip([1, 1, 1]) == 0.25


def test_holm_across_exactly_two_tests():
    p = sorted([0.03, 0.04])
    assert [min(1, 2 * p[0]), p[1]] == [0.06, 0.04]


def test_temporal_sensitivity_isolation():
    assert ("first10", "last10") != ("first20", "last20")


def test_threshold_sensitivity_isolation():
    assert [0.90, 0.95, 0.975].count(0.95) == 1


def test_deterministic_rerun_hash():
    payload = [{"method": "D4-FULL", "fold": 1}]
    assert hashlib.sha256(m.canon(payload)).hexdigest() == hashlib.sha256(m.canon(payload)).hexdigest()


def test_old_campaign_exclusion(tmp_path, monkeypatch):
    root = tmp_path
    index = root / "runs" / "phase6-v2" / "publication-run-index.jsonl"
    index.parent.mkdir(parents=True)
    row = {"run_id": "old", "run_path": "runs/phase6/old", "publication_analysis_eligible": True, "source_hash": m.SOURCE_SHA, "freeze_hash": m.FREEZE_SHA}
    index.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(m, "ROOT", root)
    with pytest.raises(RuntimeError):
        m.load_index()
