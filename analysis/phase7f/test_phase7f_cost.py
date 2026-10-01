from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase7f_cost as p7f


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "phase7f"
RUN_ROOT = ROOT / "runs" / "phase7f-cost"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


class Phase7FCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((ROOT / "campaigns" / "phase7f-cost" / "campaign-manifest.json").read_text())
        cls.h5 = json.loads((OUT / "h5-final-evaluation.json").read_text())
        cls.registry = json.loads((OUT / "telemetry-cost-registry.json").read_text())

    def test_telemetry_configuration_contracts(self) -> None:
        self.assertEqual(set(p7f.TELEMETRY_CONFIGS), {"T0_FULL", "T1_NO_HOST", "T2_NO_RUNTIME", "T3_LIGHT"})

    def test_request_telemetry_always_enabled(self) -> None:
        self.assertTrue(all(c["request_telemetry"] for c in p7f.TELEMETRY_CONFIGS.values()))

    def test_traces_always_enabled_primary_campaign(self) -> None:
        self.assertTrue(all(c["traces"] for c in p7f.TELEMETRY_CONFIGS.values()))

    def test_system_telemetry_absence_for_t1_t3(self) -> None:
        metrics = pd.read_csv(OUT / "cost-run-metrics.csv")
        self.assertEqual(metrics[metrics.telemetry_config.isin(["T1_NO_HOST", "T3_LIGHT"])].system_metric_rows.max(), 0)

    def test_runtime_telemetry_absence_for_t2_t3(self) -> None:
        metrics = pd.read_csv(OUT / "cost-run-metrics.csv")
        self.assertEqual(metrics[metrics.telemetry_config.isin(["T2_NO_RUNTIME", "T3_LIGHT"])].runtime_metric_rows.max(), 0)

    def test_paired_block_construction(self) -> None:
        counts = {}
        for row in self.manifest["rows"]:
            counts[(row["repetition"], row["workload"])] = counts.get((row["repetition"], row["workload"]), 0) + 1
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(len(counts), 12)

    def test_48_unique_run_ids(self) -> None:
        self.assertEqual(len(self.manifest["rows"]), 48)
        self.assertEqual(len({r["run_id"] for r in self.manifest["rows"]}), 48)

    def test_artifact_byte_measurement(self) -> None:
        metrics = pd.read_csv(OUT / "cost-run-metrics.csv")
        self.assertTrue((metrics.request_bytes > 0).all())
        self.assertTrue((metrics.trace_bytes > 0).all())

    def test_optional_bytes_calculation(self) -> None:
        metrics = pd.read_csv(OUT / "cost-run-metrics.csv")
        first = metrics.iloc[0]
        self.assertAlmostEqual(first.optional_telemetry_bytes, first.system_metric_bytes + first.runtime_metric_bytes + first.trace_bytes)

    def test_bytes_per_request_normalization(self) -> None:
        metrics = pd.read_csv(OUT / "cost-run-metrics.csv")
        first = metrics.iloc[0]
        self.assertAlmostEqual(first.optional_telemetry_bytes_per_measured_request, first.optional_telemetry_bytes / first.request_rows)

    def test_quality_mapping_from_phase7e(self) -> None:
        frontier = pd.read_csv(OUT / "quality-cost-frontier.csv")
        no_host = frontier[frontier.telemetry_config == "T1_NO_HOST"].iloc[0]
        self.assertAlmostEqual(no_host.source_M2F2_compound_exact, 0.6)
        self.assertAlmostEqual(no_host.target_M2F2_compound_exact, 0.6)

    def test_t2_feature_equivalence_justification(self) -> None:
        self.assertIn("runtime_metrics_cost_only", self.registry)
        frontier = pd.read_csv(OUT / "quality-cost-frontier.csv")
        full = frontier[frontier.telemetry_config == "T0_FULL"].iloc[0]
        t2 = frontier[frontier.telemetry_config == "T2_NO_RUNTIME"].iloc[0]
        self.assertAlmostEqual(full.source_M2F2_compound_exact, t2.source_M2F2_compound_exact)

    def test_t3_quality_mapping(self) -> None:
        frontier = pd.read_csv(OUT / "quality-cost-frontier.csv")
        t1 = frontier[frontier.telemetry_config == "T1_NO_HOST"].iloc[0]
        t3 = frontier[frontier.telemetry_config == "T3_LIGHT"].iloc[0]
        self.assertAlmostEqual(t1.source_M2F2_compound_exact, t3.source_M2F2_compound_exact)

    def test_paired_storage_reduction(self) -> None:
        paired = pd.read_csv(OUT / "paired-cost-comparisons.csv")
        t1 = paired[paired.comparison == "T1_NO_HOST_vs_T0_FULL"]
        self.assertEqual(len(t1), 12)
        self.assertTrue((t1.storage_reduction > 0).all())

    def test_service_performance_guard(self) -> None:
        guard = pd.read_csv(OUT / "service-performance-guard.csv")
        self.assertEqual(len(guard), 2)
        self.assertEqual(set(guard.passes_plus_5pct_guard.astype(bool)), {False})

    def test_pareto_dominance(self) -> None:
        frontier = pd.read_csv(OUT / "pareto-frontier.csv")
        self.assertIn("T3_LIGHT", set(frontier.telemetry_config))

    def test_target_storage_projection(self) -> None:
        proj = pd.read_csv(OUT / "target-storage-projection.csv").iloc[0]
        self.assertEqual(proj.campaign, "target")
        self.assertGreater(proj.campaign_total_system_bytes, 0)

    def test_h5_decision_rule(self) -> None:
        expected = "SUPPORTED" if all([
            self.h5["criterion_A_source_quality"],
            self.h5["criterion_B_target_quality"],
            self.h5["criterion_C_measured_cost_reduction_all_12_pairs"],
            self.h5["criterion_D_service_performance_guard"],
        ]) else "NOT_SUPPORTED"
        self.assertEqual(self.h5["final_H5_status"], expected)

    def test_source_target_formal_campaign_exclusion(self) -> None:
        self.assertTrue(all("phase7f-cost" in r["run_directory"] for r in self.manifest["rows"]))
        self.assertFalse(any("phase6-v2" in r["run_directory"] or "phase7d-qwen15b" in r["run_directory"] for r in self.manifest["rows"]))

    def test_deterministic_analysis_rerun(self) -> None:
        det = json.loads((OUT / "determinism-audit.json").read_text())
        self.assertTrue(det["rerun_hashes_identical"])


if __name__ == "__main__":
    unittest.main()
