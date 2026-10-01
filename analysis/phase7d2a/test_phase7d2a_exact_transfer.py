from __future__ import annotations

import csv
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7d2a_exact_transfer as p7d2a


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "phase7d2a"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


class Phase7D2AComplianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((OUT / "phase7d2a-report.json").read_text())
        cls.source_audit = json.loads((OUT / "source-baseline-reproduction-audit.json").read_text())
        cls.feature_audit = json.loads((OUT / "target-feature-reproduction-audit.json").read_text())
        cls.h4 = json.loads((OUT / "h4-test.json").read_text())
        cls.det = json.loads((OUT / "determinism-audit.json").read_text())

    def test_exact_sklearn_dependency_recorded(self) -> None:
        provenance = json.loads((OUT / "analysis-provenance.json").read_text())
        self.assertEqual(provenance["sklearn"], sklearn.__version__)
        self.assertEqual(provenance["sklearn"], "1.9.1")

    def test_source_f2_prediction_reproduction(self) -> None:
        self.assertEqual(self.source_audit["f2_prediction_mismatches"], 0)
        self.assertEqual(self.source_audit["row_count_f2"], 104)

    def test_source_f2t_prediction_reproduction(self) -> None:
        self.assertEqual(self.source_audit["f2t_prediction_mismatches"], 0)
        self.assertEqual(self.source_audit["row_count_f2t"], 104)

    def test_source_reference_metric_reproduction(self) -> None:
        self.assertAlmostEqual(self.source_audit["source_metrics"]["M2/F2"]["exact_set_accuracy"], 0.575)
        self.assertAlmostEqual(self.source_audit["source_metrics"]["M2/F2T"]["exact_set_accuracy"], 0.825)

    def test_target_feature_hash_stability(self) -> None:
        self.assertEqual(
            self.feature_audit["target_features_csv"],
            p7d2a.sha_file(ROOT / "analysis/phase7d2/target-features.csv"),
        )

    def test_source_only_scaler_and_imputation_fit(self) -> None:
        source = pd.read_csv(ROOT / "analysis/phase7c1a/mesr-dataset.csv")
        target = pd.read_csv(ROOT / "analysis/phase7d2/target-features.csv")
        _, _, prep = p7d2a.split_xy(source[source.compound_degree <= 1], target, p7d2a.CORE_F2)
        self.assertEqual(set(prep["medians"]), set(p7d2a.CORE_F2))
        self.assertNotIn("run_id", prep["medians"])

    def test_exact_frozen_logistic_parameters(self) -> None:
        spec = json.loads((OUT / "analysis-spec.json").read_text())
        self.assertEqual(spec["logistic_parameters"]["solver"], "liblinear")
        self.assertEqual(spec["logistic_parameters"]["C"], 1.0)
        self.assertEqual(spec["logistic_parameters"]["class_weight"], "balanced")
        self.assertEqual(spec["logistic_parameters"]["threshold"], 0.5)

    def test_no_target_training_contract(self) -> None:
        spec = json.loads((OUT / "analysis-spec.json").read_text())
        self.assertTrue(spec["source_only_training"])

    def test_d4_historical_hash_stability(self) -> None:
        self.assertTrue(self.report["d4_hash_verified"])
        self.assertAlmostEqual(self.report["d4_target_compound_exact"], 0.6)

    def test_optimizer_difference_audit_shape(self) -> None:
        rows = csv_rows(OUT / "optimizer-difference-audit.csv")
        self.assertEqual(len(rows), 66 * 2 * 4)
        self.assertEqual(sum(r["prediction_changed"] == "True" for r in rows), self.report["optimizer_changed_cause_decisions"])

    def test_generalization_drop_calculation(self) -> None:
        rows = csv_rows(OUT / "generalization-drop.csv")
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertAlmostEqual(float(row["absolute_drop"]), float(row["source_exact"]) - float(row["target_exact"]))

    def test_six_fold_h4_comparison(self) -> None:
        rows = [r for r in csv_rows(OUT / "h4-fold-comparison.csv") if r["repetition"] != "SUMMARY"]
        self.assertEqual(len(rows), 6)
        self.assertEqual([float(r["D4_minus_M2F2"]) for r in rows], self.h4["differences"])

    def test_exact_64_sign_flips(self) -> None:
        self.assertEqual(2**6, 64)
        self.assertEqual(p7d2a.signflip([0.0] * 6), 1.0)
        self.assertEqual(p7d2a.signflip([0.4] * 6), 1 / 32)

    def test_h4_decision_rule(self) -> None:
        supported = self.h4["mean_difference"] > 0 and self.h4["exact_signflip_p"] < 0.05
        self.assertEqual(self.h4["h4_status"], "SUPPORTED" if supported else "NOT_SUPPORTED")

    def test_pair_specific_target_metrics(self) -> None:
        rows = csv_rows(OUT / "table-13a-pair-transfer.csv")
        self.assertEqual(len(rows), 4 * 5)

    def test_control_false_alarm_rates(self) -> None:
        rows = csv_rows(OUT / "table-12a-zero-shot-generalization.csv")
        self.assertTrue(all("target_control_false_alarm_rate" in r for r in rows))

    def test_control_normalization_frozen(self) -> None:
        rows = csv_rows(OUT / "control-normalized-summary.csv")
        self.assertEqual({r["method"] for r in rows}, {"M2/F2-control-normalized", "M2/F2T-control-normalized"})
        self.assertEqual({r["subset"] for r in rows}, {"TARGET-ALL", "TARGET-COMPOUND"})

    def test_deterministic_rerun_hash(self) -> None:
        self.assertTrue(self.det["rerun_hashes_identical"])
        for key in ["M2_prediction_hash", "M2F2T_prediction_hash", "table12a_hash", "table13a_hash", "table14a_hash"]:
            self.assertRegex(self.det[key], r"^[0-9a-f]{64}$")

    def test_target_csd_prohibited(self) -> None:
        self.assertFalse((OUT / "target-csd.csv").exists())
        self.assertFalse((OUT / "csd-summary.csv").exists())

    def test_historical_vs_corrected_status_revised_when_material_changes_exist(self) -> None:
        audit = csv_rows(OUT / "phase7d2-vs-phase7d2a-audit.csv")
        changed = any(r["target_exact_changed"] == "True" for r in audit)
        self.assertEqual(self.report["result_status"], "REVISED" if changed else "CONFIRMED")


if __name__ == "__main__":
    unittest.main()
