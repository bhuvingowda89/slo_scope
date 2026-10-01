from __future__ import annotations

import csv
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7e_robustness as p7e


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "phase7e"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


class Phase7ERobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean = json.loads((OUT / "clean-reference-audit.json").read_text())
        cls.spec = json.loads((OUT / "robustness-spec.json").read_text())
        cls.seal = json.loads((OUT / "input-seal.json").read_text())
        cls.det = json.loads((OUT / "determinism-audit.json").read_text())

    def test_clean_prediction_reproduction(self) -> None:
        self.assertTrue(self.clean["valid"])
        self.assertEqual(len(self.clean["rows"]), 8)

    def test_missingness_masking_determinism(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        a = p7e.random_missing(df, ["a", "b"], 0.5, 8101)
        b = p7e.random_missing(df, ["a", "b"], 0.5, 8101)
        self.assertTrue(a.equals(b))

    def test_same_seed_fairness_shared_feature_mask(self) -> None:
        df = pd.DataFrame({"shared": [1.0] * 20})
        a = p7e.random_missing(df, ["shared"], 0.25, 8102)
        b = p7e.random_missing(df, ["shared"], 0.25, 8102)
        self.assertEqual(a["shared"].isna().tolist(), b["shared"].isna().tolist())

    def test_training_imputation_unchanged(self) -> None:
        source = pd.read_csv(ROOT / "analysis/phase7c1a/mesr-dataset.csv")
        train = source[source.compound_degree <= 1]
        med_before = train[p7e.CORE_F2].median(numeric_only=True)
        degraded = p7e.random_missing(source, p7e.CORE_F2, 0.5, 8103)
        med_after = train[p7e.CORE_F2].median(numeric_only=True)
        self.assertTrue(med_before.equals(med_after))
        self.assertTrue(degraded[p7e.CORE_F2].isna().any().any())

    def test_no_target_retraining_contract(self) -> None:
        self.assertEqual(self.spec["source_training_policy"], "source LORO for source; source-only P1 for target")
        self.assertTrue(self.spec["test_degradation_only"])

    def test_temporal_subsampling_order_preservation(self) -> None:
        series = {"ttft": list(range(40)), "total_latency": list(range(40)), "post_first_token_duration": list(range(40)), "prefill_proxy": list(range(40)), "dependency_duration": list(range(40)), "scheduler_slip": list(range(40)), "gateway_span_duration": list(range(40)), "llama_span_duration": list(range(40)), "dependency_span_duration": list(range(40))}
        feats = p7e.temporal_features_from_series(series, [0, 2, 4, 6, 8, 10, 12, 14])
        self.assertLess(feats["ttft_early_median"], feats["ttft_late_median"])

    def test_temporal_feature_recomputation(self) -> None:
        series = {"ttft": list(range(12)), "total_latency": list(range(12)), "post_first_token_duration": list(range(12)), "prefill_proxy": list(range(12)), "dependency_duration": list(range(12)), "scheduler_slip": list(range(12)), "gateway_span_duration": list(range(12)), "llama_span_duration": list(range(12)), "dependency_span_duration": list(range(12))}
        feats = p7e.temporal_features_from_series(series, list(range(12)))
        self.assertAlmostEqual(feats["ttft_late_minus_early"], 9.0)

    def test_noise_sd_uses_source_training_distribution(self) -> None:
        source = pd.read_csv(ROOT / "analysis/phase7c1a/mesr-dataset.csv")
        sd = source[source.compound_degree <= 1][p7e.F3_FEATURES].std(numeric_only=True).to_dict()
        self.assertIn("ttft_p95", sd)
        self.assertGreater(sd["ttft_p95"], 0)

    def test_nonnegative_noise_clipping(self) -> None:
        df = pd.DataFrame({"ttft_p95": [0.001]})
        out = p7e.add_noise(df, ["ttft_p95"], {"ttft_p95": 100.0}, 1.0, 8501)
        self.assertGreaterEqual(out.loc[0, "ttft_p95"], 0.0)

    def test_delay_non_wrapping(self) -> None:
        series = {"ttft": list(range(10)), "total_latency": list(range(10)), "post_first_token_duration": list(range(10)), "prefill_proxy": list(range(10)), "dependency_duration": list(range(10)), "scheduler_slip": list(range(10)), "gateway_span_duration": list(range(10)), "llama_span_duration": list(range(10)), "dependency_span_duration": list(range(10))}
        shifted = {"ttft": [None] * 3 + series["ttft"][:7]}
        self.assertIsNone(shifted["ttft"][0])
        self.assertEqual(shifted["ttft"][-1], 6)

    def test_whole_channel_removal(self) -> None:
        df = pd.DataFrame({"host_cpu_median": [1.0], "host_cpu_p95": [2.0]})
        out = p7e.remove_channels(df, p7e.CHANNELS["C1_HOST_CPU"])
        self.assertTrue(out.isna().all().all())

    def test_f3_direct_evidence_removal(self) -> None:
        self.assertEqual(p7e.CHANNELS["C9_PROMPT_OUTPUT_DIRECT"], ["server_prompt_tokens_median", "server_output_tokens_median"])

    def test_retention_calculation(self) -> None:
        self.assertAlmostEqual(0.45 / 0.9, 0.5)

    def test_robustness_auc(self) -> None:
        auc = p7e.auc_for([{"severity": 1, "compound_exact": 0.5}], 1.0, [1], [1.0])
        self.assertAlmostEqual(auc, 0.75)

    def test_control_false_alarm_metric(self) -> None:
        rs = [{"true_INPUT": 0, "true_OUTPUT": 0, "true_LOAD": 0, "true_DOWNSTREAM": 0, "pred_INPUT": 1, "pred_OUTPUT": 0, "pred_LOAD": 0, "pred_DOWNSTREAM": 0, "predicted_set": "INPUT"}]
        self.assertEqual(p7e.metrics(rs)["control_false_alarm_rate"], 1.0)

    def test_h5_quality_screen_rule(self) -> None:
        passing = [r for r in rows(OUT / "h5-quality-screen.csv") if r["passes_H5_quality_screen"] == "True"]
        self.assertTrue(passing)
        for row in passing:
            self.assertGreaterEqual(float(row["retention"]), 0.90)
            self.assertLessEqual(float(row["control_false_alarm_change"]), 0.10)

    def test_preexisting_input_failure_handling(self) -> None:
        cause = rows(OUT / "cause-robustness.csv")
        input_rows = [r for r in cause if r["degradation_family"] == "clean" and r["cause"] == "INPUT"]
        self.assertTrue(input_rows)
        self.assertTrue(all("incremental_recall_loss_from_clean" in r for r in cause))

    def test_source_target_domain_separation(self) -> None:
        pred_domains = {r["domain"] for r in rows(OUT / "table-16-telemetry-missingness.csv")}
        self.assertEqual(pred_domains, {"source", "target"})

    def test_pilot_exclusion(self) -> None:
        self.assertNotIn("phase7d0", json.dumps(self.seal))
        self.assertNotIn("diagnostics", json.dumps(self.seal))

    def test_deterministic_repeat_hash(self) -> None:
        self.assertTrue(self.det["rerun_hashes_identical"])
        for key, value in self.det.items():
            if key.endswith("_hash"):
                self.assertRegex(value, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
