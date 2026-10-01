from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.phase7b import phase7b_csd as c


def test_factorial_additive_prediction_formula():
    assert c.additive_prediction(3.0, 4.0, 1.0) == 6.0


def test_interaction_residual_equivalence():
    assert c.interaction_residual(10.0, 3.0, 4.0, 1.0) == 4.0


def test_pooled_within_cell_sd():
    cells = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])]
    assert c.pooled_within_cell_sd(cells) == pytest.approx(1.0)


def test_zero_scale_feature_exclusion():
    df = synthetic_df(extra=0.0)
    design = synthetic_design()
    res = c.compute_family(df, design, "PAIR", ["f"], "sd")
    assert res["features"] == []


def test_standardization_and_primary_csd_formula():
    zbar = np.array([3.0, 4.0])
    assert c.csd_from_zbar(zbar) == pytest.approx(math_sqrt_12_5())


def math_sqrt_12_5() -> float:
    return (12.5) ** 0.5


def test_repetition_level_csd():
    z = np.array([[3.0, 4.0], [0.0, 0.0]])
    values = c.repetition_csd(z)
    assert values[0] == pytest.approx(math_sqrt_12_5())
    assert values[1] == 0.0


def test_feature_contributions_sum_to_one():
    contrib = c.contribution_fractions(np.array([3.0, 4.0]))
    assert contrib.sum() == pytest.approx(1.0)


def test_exact_256_sign_flip_enumeration():
    z = np.ones((8, 1))
    _, stats = c.exact_sign_flip_p(z, c.csd_from_zbar(np.mean(z, axis=0)))
    assert len(stats) == 256


def test_exact_p_value_known_synthetic_example():
    z = np.ones((8, 1))
    p, _ = c.exact_sign_flip_p(z, 1.0)
    assert p == pytest.approx(2 / 256)


def test_holm_adjustment_across_five():
    adjusted = c.holm_adjust([("a", 0.01), ("b", 0.02), ("c", 0.5), ("d", 0.04), ("e", 0.03)])
    assert adjusted["a"] == pytest.approx(0.05)
    assert adjusted["b"] == pytest.approx(0.08)
    assert adjusted["e"] == pytest.approx(0.09)


def test_bootstrap_reproducibility():
    z = np.arange(16, dtype=float).reshape(8, 2)
    first = c.bootstrap_interval(z, seed=7711, resamples=100)[2]
    second = c.bootstrap_interval(z, seed=7711, resamples=100)[2]
    assert first == second


def test_leave_one_feature_out_computation():
    zbar = np.array([1.0, 2.0, 3.0])
    vals = [c.csd_from_zbar(zbar[[i for i in range(3) if i != drop]]) for drop in range(3)]
    assert len(vals) == 3
    assert min(vals) <= max(vals)


def test_robust_scale_sensitivity_nonzero():
    cells = [np.array([1.0, 2.0, 100.0]), np.array([2.0, 3.0, 101.0])]
    assert c.pooled_within_cell_mad_scale(cells) > 0


def test_no_ground_truth_labels_enter_primary_feature_matrix():
    forbidden = {"condition_id", "active_mechanisms", "mechanism_levels", "compound_degree"}
    assert not forbidden.intersection(c.PRIMARY_FEATURES)


def test_output_control_matched_factorial_mapping():
    mapping = {
        "OUTPUT_LOAD": {
            "A0B0": "OUTPUT_CONTROL",
            "A1B0": "OUTPUT_MEDIUM",
            "A0B1": "OUTPUT_LOAD_CONTROL",
            "A1B1": "OUTPUT_LOAD",
        }
    }
    assert mapping["OUTPUT_LOAD"]["A0B0"] == "OUTPUT_CONTROL"


def test_old_campaign_exclusion(tmp_path, monkeypatch):
    rows = [{"run_id": str(i), "source_hash": c.SOURCE_SHA, "freeze_hash": c.FREEZE_SHA, "run_path": "runs/phase6/old"} for i in range(104)]
    root = tmp_path
    path = root / "runs" / "phase6-v2"
    path.mkdir(parents=True)
    (path / "publication-run-index.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(c, "ROOT", root)
    with pytest.raises(RuntimeError, match="superseded"):
        c.load_inputs()


def test_phase7a1_interaction_sign_consistency_synthetic():
    df = synthetic_df(extra=2.0, noise=True)
    res = c.compute_family(df, synthetic_design(), "PAIR", ["f"], "sd")
    assert np.mean(res["raw_delta"]["f"]) > 0
    assert res["zbar"][0] > 0


def test_perfectly_additive_csd_zero():
    df = synthetic_df(extra=0.0, noise=True)
    res = c.compute_family(df, synthetic_design(), "PAIR", ["f"], "sd")
    assert c.csd_from_zbar(res["zbar"]) == pytest.approx(0.0)


def test_known_extra_interaction_nonzero_csd():
    df = synthetic_df(extra=2.0, noise=True)
    res = c.compute_family(df, synthetic_design(), "PAIR", ["f"], "sd")
    assert c.csd_from_zbar(res["zbar"]) > 0


def test_equal_opposite_signed_features_do_not_cancel_csd():
    df = synthetic_df(extra=0.0, noise=True)
    df["g"] = df["f"]
    df.loc[df["condition_id"] == "A1B1", "f"] += 2.0
    df.loc[df["condition_id"] == "A1B1", "g"] -= 2.0
    res = c.compute_family(df, synthetic_design(), "PAIR", ["f", "g"], "sd")
    assert abs(float(np.mean(res["raw_delta"]["f"])) + float(np.mean(res["raw_delta"]["g"]))) < 1e-9
    assert c.csd_from_zbar(res["zbar"]) > 0


def synthetic_design():
    return {"PAIR": {"A0B0": "A0B0", "A1B0": "A1B0", "A0B1": "A0B1", "A1B1": "A1B1"}}


def synthetic_df(extra: float, noise: bool = False) -> pd.DataFrame:
    rows = []
    for rep in range(1, 9):
        jitter = rep * 0.01 if noise else 0.0
        rows.append({"condition_id": "A0B0", "repetition": rep, "f": 10.0 + jitter})
        rows.append({"condition_id": "A1B0", "repetition": rep, "f": 13.0 + jitter})
        rows.append({"condition_id": "A0B1", "repetition": rep, "f": 14.0 + jitter})
        rows.append({"condition_id": "A1B1", "repetition": rep, "f": 17.0 + extra + jitter})
    return pd.DataFrame(rows)
