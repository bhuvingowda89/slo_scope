from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "phase7b"
FIGURES = OUT / "figures"
CAMPAIGN_ID = "sloscope-phase6-v2"
SOURCE_SHA = "8e442c813619586072c27a4ec30d7e65d367c3fc1375922935f67c72e2122582"
FREEZE_SHA = "65d9798ea033bc910a832c624324fe068e48552b2a16e6ed639f33763ca028e0"

PRIMARY_FEATURES = [
    "ttft_p95",
    "total_latency_p95",
    "post_first_token_duration_p95",
    "ttft_median",
    "total_latency_median",
    "post_first_token_duration_median",
    "scheduler_slip_p95",
    "successful_requests_per_second",
]
EXTENDED_FEATURES = [
    "host_cpu_median",
    "host_cpu_p95",
    "dependency_duration_median",
    "dependency_duration_p95",
    "server_prompt_tokens_median",
    "server_output_tokens_median",
]
UNAVAILABLE_FEATURES = ["llamacpp:requests_processing", "llamacpp:requests_deferred"]
COMPOUNDS = ["INPUT_LOAD", "INPUT_DOWNSTREAM", "OUTPUT_LOAD", "OUTPUT_DOWNSTREAM", "LOAD_DOWNSTREAM"]
BOOTSTRAP_SEED = 7711
BOOTSTRAP_RESAMPLES = 10_000
TOL = 1e-12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, p_value) in enumerate(ordered, start=1):
        running = max(running, min(1.0, (m - rank + 1) * p_value))
        adjusted[name] = running
    return adjusted


def pooled_within_cell_sd(cells: list[np.ndarray]) -> float:
    numerator = 0.0
    denominator = 0
    for values in cells:
        if len(values) < 2:
            continue
        numerator += (len(values) - 1) * float(np.var(values, ddof=1))
        denominator += len(values) - 1
    if denominator == 0:
        return 0.0
    return math.sqrt(numerator / denominator)


def mad(values: np.ndarray) -> float:
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def pooled_within_cell_mad_scale(cells: list[np.ndarray]) -> float:
    scales = []
    weights = []
    for values in cells:
        if len(values) == 0:
            continue
        scales.append((1.4826 * mad(values)) ** 2)
        weights.append(len(values))
    if not scales:
        return 0.0
    return math.sqrt(float(np.average(scales, weights=weights)))


def additive_prediction(a1b0: float, a0b1: float, a0b0: float) -> float:
    return a1b0 + a0b1 - a0b0


def interaction_residual(a1b1: float, a1b0: float, a0b1: float, a0b0: float) -> float:
    return a1b1 - a1b0 - a0b1 + a0b0


def csd_from_zbar(zbar: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(np.square(zbar))))) if len(zbar) else float("nan")


def repetition_csd(z: np.ndarray) -> list[float]:
    return [float(math.sqrt(float(np.mean(np.square(row))))) for row in z]


def contribution_fractions(zbar: np.ndarray) -> np.ndarray:
    denom = float(np.sum(np.square(zbar)))
    if denom <= 0:
        return np.zeros_like(zbar)
    return np.square(zbar) / denom


def exact_sign_flip_p(z: np.ndarray, observed: float, tolerance: float = 1e-12) -> tuple[float, list[float]]:
    stats_out = []
    for signs in itertools.product([-1.0, 1.0], repeat=z.shape[0]):
        signed = z * np.array(signs)[:, None]
        stats_out.append(csd_from_zbar(np.mean(signed, axis=0)))
    count = sum(1 for value in stats_out if value + tolerance >= observed)
    return count / len(stats_out), stats_out


def bootstrap_interval(z: np.ndarray, seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, list[float]]:
    rng = np.random.default_rng(seed)
    values = []
    n = z.shape[0]
    for _ in range(resamples):
        sample = z[rng.integers(0, n, size=n)]
        values.append(csd_from_zbar(np.mean(sample, axis=0)))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high), [float(v) for v in values]


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    index_rows = [json.loads(line) for line in (ROOT / "runs" / "phase6-v2" / "publication-run-index.jsonl").read_text().splitlines() if line.strip()]
    if len(index_rows) != 104 or len({row["run_id"] for row in index_rows}) != 104:
        raise RuntimeError("publication index must contain 104 unique V2 runs")
    for row in index_rows:
        if row.get("source_hash") != SOURCE_SHA or row.get("freeze_hash") != FREEZE_SHA:
            raise RuntimeError(f"hash mismatch in publication index for {row.get('run_id')}")
        if str(row.get("run_path", "")).startswith("runs/phase6/"):
            raise RuntimeError("superseded phase6 run found in publication index")
    df = pd.read_csv(ROOT / "analysis" / "phase7a1" / "run-level-metrics.csv")
    if len(df) != 104:
        raise RuntimeError("Phase 7A.1 run-level metrics must contain 104 rows")
    return df, {"rows": index_rows}


def create_input_seal(index: dict[str, Any]) -> dict[str, Any]:
    phase7a1_input = read_json(ROOT / "analysis" / "phase7a1" / "input-seal.json")
    phase7a1_spec = read_json(ROOT / "analysis" / "phase7a1" / "analysis-spec.json")
    seal = {
        "schema_version": "sloscope.phase7b.input_seal.v1",
        "campaign_id": CAMPAIGN_ID,
        "phase7a1_input_sha256": phase7a1_input["phase7a1_input_sha256"],
        "phase7a1_analysis_spec_sha256": phase7a1_spec["phase7a1_analysis_spec_sha256"],
        "phase7a1_run_level_metrics_sha256": sha256_file(ROOT / "analysis" / "phase7a1" / "run-level-metrics.csv"),
        "factorial_design_sha256": sha256_file(ROOT / "campaigns" / "phase5" / "factorial-design.json"),
        "publication_run_index_sha256": sha256_file(ROOT / "runs" / "phase6-v2" / "publication-run-index.jsonl"),
        "source_tree_sha256": SOURCE_SHA,
        "campaign_freeze_sha256": FREEZE_SHA,
        "run_ids": [row["run_id"] for row in index["rows"]],
    }
    seal["phase7b_input_sha256"] = sha256_bytes(canonical_json(seal))
    write_json(OUT / "input-seal.json", seal)
    return seal


def create_csd_spec() -> dict[str, Any]:
    spec = {
        "schema_version": "sloscope.phase7b.csd_spec.v1",
        "feature_registry": {
            "primary_features": PRIMARY_FEATURES,
            "extended_features": EXTENDED_FEATURES,
            "unavailable_features": UNAVAILABLE_FEATURES,
            "excluded_from_csd": ["condition_id", "active_mechanisms", "mechanism_levels", "compound_degree", "SLO labels"],
        },
        "scaling": "pooled within-cell SD per factorial family and feature",
        "interaction_vector": "Delta_rj = X_A1B1,rj - X_A1B0,rj - X_A0B1,rj + X_A0B0,rj",
        "csd": "sqrt(mean_j(mean_r(Delta_rj / pooled_sd_j)^2))",
        "repetition_level_csd": "sqrt(mean_j((Delta_rj / pooled_sd_j)^2))",
        "permutation_test": {"type": "exact repetition-level sign flip", "permutations": 256, "tolerance": TOL},
        "multiple_comparison_family": COMPOUNDS,
        "multiple_comparison_correction": "Holm across five primary CSD exact p-values",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "interval": "percentile 2.5/97.5"},
        "sensitivity": ["robust pooled MAD scale", "leave-one-feature-out"],
        "figures": {
            "figure_6": "CSD summary with bootstrap interval and Holm support",
            "figure_7": "signed residual heatmap",
            "figure_8": "standardized additive prediction versus observed compound signature",
            "figure_9": "feature contribution fractions",
        },
        "notes": [
            "Absolute SLO crossings are not input features.",
            "Output-control drift is handled through matched factorial cells.",
            "Queue metrics are unavailable and must not be inferred from latency.",
        ],
    }
    spec["phase7b_csd_spec_sha256"] = sha256_bytes(canonical_json(spec))
    write_json(OUT / "csd-spec.json", spec)
    return spec


def build_feature_registry(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    definitions = {
        "ttft_p95": ("requests.parquet", "95th percentile first_token_time - actual_arrival", "seconds"),
        "total_latency_p95": ("requests.parquet", "95th percentile completion_time - actual_arrival", "seconds"),
        "post_first_token_duration_p95": ("requests.parquet", "95th percentile completion_time - first_token_time", "seconds"),
        "ttft_median": ("requests.parquet", "median first_token_time - actual_arrival", "seconds"),
        "total_latency_median": ("requests.parquet", "median completion_time - actual_arrival", "seconds"),
        "post_first_token_duration_median": ("requests.parquet", "median completion_time - first_token_time", "seconds"),
        "scheduler_slip_p95": ("requests.parquet", "95th percentile scheduler_slip", "seconds"),
        "successful_requests_per_second": ("requests.parquet", "successful measured requests per elapsed run window", "requests/second"),
        "host_cpu_median": ("system_metrics.parquet", "median host CPU utilization", "percent"),
        "host_cpu_p95": ("system_metrics.parquet", "95th percentile host CPU utilization", "percent"),
        "dependency_duration_median": ("requests.parquet", "median gateway dependency duration", "seconds"),
        "dependency_duration_p95": ("requests.parquet", "95th percentile gateway dependency duration", "seconds"),
        "server_prompt_tokens_median": ("requests.parquet", "median server_prompt_tokens", "tokens"),
        "server_output_tokens_median": ("requests.parquet", "median server_output_tokens", "tokens"),
    }
    for feature in PRIMARY_FEATURES + EXTENDED_FEATURES:
        missing = int(df[feature].isna().sum()) if feature in df.columns else len(df)
        available = feature in df.columns and missing == 0
        leakage = False
        rows.append(
            {
                "feature": feature,
                "source_artifact": definitions[feature][0],
                "definition": definitions[feature][1],
                "unit": definitions[feature][2],
                "feature_view": "primary" if feature in PRIMARY_FEATURES else "extended",
                "availability": "available" if available else "unavailable",
                "missingness": missing,
                "included": available,
                "exclusion_reason": None if available else "missing values in Phase 7A.1 run-level metrics",
                "scale": "family-specific pooled within-cell SD; see csd-signatures.csv" if available else None,
                "ground_truth_leakage_risk": leakage,
            }
        )
    for feature in UNAVAILABLE_FEATURES:
        rows.append(
            {
                "feature": feature,
                "source_artifact": "runtime_metrics.parquet",
                "definition": "llama.cpp runtime queue metric",
                "unit": "count",
                "feature_view": "excluded_unavailable",
                "availability": "unavailable",
                "missingness": 104,
                "included": False,
                "exclusion_reason": "absent from formal runtime_metrics artifacts",
                "scale": None,
                "ground_truth_leakage_risk": False,
            }
        )
    write_json(OUT / "feature-registry.json", {"schema_version": "sloscope.phase7b.feature_registry.v1", "features": rows})
    return rows


def cell_values(df: pd.DataFrame, condition: str, feature: str) -> np.ndarray:
    sub = df[df["condition_id"] == condition].sort_values("repetition")
    return sub[feature].to_numpy(dtype=float)


def compute_family(df: pd.DataFrame, design: dict[str, Any], compound: str, features: list[str], scale_kind: str = "sd") -> dict[str, Any]:
    mapping = design[compound]
    cells = {name: mapping[name] for name in ["A0B0", "A1B0", "A0B1", "A1B1"]}
    included = []
    scales = {}
    raw_delta = {}
    additive = {}
    observed = {}
    cell_means = {}
    for feature in features:
        arrays = [cell_values(df, cells[name], feature) for name in ["A0B0", "A1B0", "A0B1", "A1B1"]]
        scale = pooled_within_cell_sd(arrays) if scale_kind == "sd" else pooled_within_cell_mad_scale(arrays)
        if scale <= TOL or not np.isfinite(scale):
            continue
        included.append(feature)
        scales[feature] = scale
        a0b0, a1b0, a0b1, a1b1 = arrays
        delta = a1b1 - a1b0 - a0b1 + a0b0
        raw_delta[feature] = delta
        additive[feature] = a1b0 + a0b1 - a0b0
        observed[feature] = a1b1
        cell_means[feature] = {
            "A0B0": float(np.mean(a0b0)),
            "A1B0": float(np.mean(a1b0)),
            "A0B1": float(np.mean(a0b1)),
            "A1B1": float(np.mean(a1b1)),
        }
    z = np.column_stack([raw_delta[f] / scales[f] for f in included]) if included else np.empty((8, 0))
    zbar = np.mean(z, axis=0) if included else np.array([])
    csd = csd_from_zbar(zbar)
    rep = repetition_csd(z) if included else []
    contrib = contribution_fractions(zbar) if included else np.array([])
    return {
        "compound": compound,
        "mapping": mapping,
        "features": included,
        "scales": scales,
        "raw_delta": raw_delta,
        "additive": additive,
        "observed": observed,
        "cell_means": cell_means,
        "z": z,
        "zbar": zbar,
        "csd": csd,
        "rep_csd": rep,
        "contrib": contrib,
    }


def analyze_csd(df: pd.DataFrame, registry: list[dict[str, Any]]) -> dict[str, Any]:
    design = {k: v for k, v in read_json(ROOT / "campaigns" / "phase5" / "factorial-design.json").items() if k in COMPOUNDS}
    primary_features = [r["feature"] for r in registry if r["feature_view"] == "primary" and r["included"]]
    extended_features = primary_features + [r["feature"] for r in registry if r["feature_view"] == "extended" and r["included"]]
    results = {}
    p_values = []
    for compound in COMPOUNDS:
        res = compute_family(df, design, compound, primary_features, "sd")
        p_value, perm_stats = exact_sign_flip_p(res["z"], res["csd"])
        low, high, boot = bootstrap_interval(res["z"])
        robust = compute_family(df, design, compound, primary_features, "mad")
        extended = compute_family(df, design, compound, extended_features, "sd")
        p_values.append((compound, p_value))
        results[compound] = {
            **res,
            "exact_p": p_value,
            "permutation_stats": perm_stats,
            "bootstrap_low": low,
            "bootstrap_high": high,
            "bootstrap_values": boot,
            "robust_csd": robust["csd"],
            "extended_csd": extended["csd"],
        }
    holm = holm_adjust(p_values)
    for compound, value in holm.items():
        results[compound]["holm_p"] = value
    return results


def materialize_outputs(df: pd.DataFrame, results: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    phase7a1 = pd.read_csv(ROOT / "analysis" / "phase7a1" / "factorial-interactions.csv")
    phase7a1_sign = {(row.compound, row.metric): math.copysign(1, row.mean_interaction_delta) if row.mean_interaction_delta != 0 else 0 for row in phase7a1.itertuples()}
    run_residuals = []
    signatures = []
    contributions = []
    summary = []
    permutations = []
    bootstrap_rows = []
    loo_rows = []
    scaling = []
    consistency = []
    slo = pd.read_csv(ROOT / "analysis" / "phase7a1" / "condition-slo-summary.csv")
    slo_by_condition = {row.condition_id: row for row in slo.itertuples()}
    for compound, res in results.items():
        features = res["features"]
        for rep_idx in range(8):
            row = {"compound": compound, "repetition": rep_idx + 1, "CSD_r": res["rep_csd"][rep_idx]}
            for fi, feature in enumerate(features):
                row[f"{feature}_raw_delta"] = float(res["raw_delta"][feature][rep_idx])
                row[f"{feature}_standardized_delta"] = float(res["z"][rep_idx, fi])
            run_residuals.append(row)
        for fi, feature in enumerate(features):
            raw_mean = float(np.mean(res["raw_delta"][feature]))
            zbar = float(res["zbar"][fi])
            pred_mean = float(np.mean(res["additive"][feature]))
            observed_mean = float(np.mean(res["observed"][feature]))
            signatures.append(
                {
                    "compound": compound,
                    "feature": feature,
                    "control_mean": res["cell_means"][feature]["A0B0"],
                    "A_only_mean": res["cell_means"][feature]["A1B0"],
                    "B_only_mean": res["cell_means"][feature]["A0B1"],
                    "observed_compound_mean": observed_mean,
                    "additive_predicted_compound_mean": pred_mean,
                    "raw_residual": observed_mean - pred_mean,
                    "standardized_residual": zbar,
                    "scale": res["scales"][feature],
                    "feature_contribution": float(res["contrib"][fi]),
                }
            )
            contributions.append(
                {
                    "compound": compound,
                    "feature": feature,
                    "raw_interaction_mean": raw_mean,
                    "standardized_interaction_mean": zbar,
                    "interaction_direction": "positive" if zbar > 0 else "negative" if zbar < 0 else "zero",
                    "contribution_fraction": float(res["contrib"][fi]),
                }
            )
            if (compound, feature) in phase7a1_sign:
                sign = math.copysign(1, raw_mean) if raw_mean != 0 else 0
                consistency.append({"compound": compound, "feature": feature, "sign_matches_phase7a1": sign == phase7a1_sign[(compound, feature)]})
        for idx, value in enumerate(res["permutation_stats"]):
            permutations.append({"compound": compound, "permutation_index": idx, "T_b": value, "observed_CSD": res["csd"], "raw_exact_p": res["exact_p"]})
        for idx, value in enumerate(res["bootstrap_values"]):
            bootstrap_rows.append({"compound": compound, "bootstrap_index": idx, "CSD": value})
        loo_values = []
        for drop in features:
            keep = [i for i, f in enumerate(features) if f != drop]
            value = csd_from_zbar(res["zbar"][keep]) if keep else float("nan")
            loo_values.append((drop, value, abs(value - res["csd"])))
            loo_rows.append({"compound": compound, "dropped_feature": drop, "CSD_without_feature": value, "absolute_change": abs(value - res["csd"])})
        scaling.append({"compound": compound, "primary_CSD": res["csd"], "robust_scale_CSD": res["robust_csd"], "extended_CSD": res["extended_csd"]})
        largest = max([(features[i], float(res["contrib"][i])) for i in range(len(features))], key=lambda x: x[1])
        compound_condition = res["mapping"]["A1B1"]
        summary.append(
            {
                "compound": compound,
                "feature_count": len(features),
                "CSD": res["csd"],
                "bootstrap_95_low": res["bootstrap_low"],
                "bootstrap_95_high": res["bootstrap_high"],
                "exact_p": res["exact_p"],
                "Holm_p": res["holm_p"],
                "statistically_supported": res["holm_p"] < 0.05,
                "largest_contribution_feature": largest[0],
                "largest_contribution_fraction": largest[1],
                "robust_scale_CSD": res["robust_csd"],
                "extended_CSD": res["extended_csd"],
                "LOO_CSD_min": min(v for _, v, _ in loo_values),
                "LOO_CSD_max": max(v for _, v, _ in loo_values),
                "LOO_feature_largest_change": max(loo_values, key=lambda x: x[2])[0],
                "compound_slo_violations": f"{int(slo_by_condition[compound_condition].compound_slo_runs)}/8",
            }
        )
    if not all(row["sign_matches_phase7a1"] for row in consistency):
        raise RuntimeError("Phase 7B signed residuals disagree with Phase 7A.1 interaction signs")
    return {
        "run_residuals": run_residuals,
        "signatures": signatures,
        "contributions": contributions,
        "summary": summary,
        "permutations": permutations,
        "bootstrap": bootstrap_rows,
        "loo": loo_rows,
        "scaling": scaling,
        "consistency": consistency,
    }


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:10px}.title{font-size:15px;font-weight:bold}.axis{stroke:#333}.ci{stroke:#111;stroke-width:2}.supported{fill:#238b45}.plain{fill:#6baed6}.zero{stroke:#333;stroke-dasharray:4 3}.pos{fill:#de2d26}.neg{fill:#3182bd}</style>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + ["</svg>\n"]), encoding="utf-8")


def figures(outputs: dict[str, list[dict[str, Any]]]) -> None:
    summary = outputs["summary"]
    width, height = 900, 360
    vals = [r["bootstrap_95_high"] for r in summary]
    vmax = max(vals) * 1.12
    lines = svg_header(width, height)
    lines.append('<text class="title" x="20" y="24">Figure 6. Primary CSD summary</text>')
    left, right, top, bottom = 180, 850, 55, 290
    def x(v: float) -> float: return left + v / vmax * (right - left)
    lines.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    for i, r in enumerate(summary):
        y = top + i * 42
        lines.append(f'<text x="20" y="{y+4}">{r["compound"]}</text>')
        lines.append(f'<line class="ci" x1="{x(r["bootstrap_95_low"]):.2f}" y1="{y}" x2="{x(r["bootstrap_95_high"]):.2f}" y2="{y}"/>')
        cls = "supported" if r["statistically_supported"] else "plain"
        lines.append(f'<circle class="{cls}" cx="{x(r["CSD"]):.2f}" cy="{y}" r="5"/>')
    write_svg(FIGURES / "figure-6-csd-summary.svg", lines)

    contrib = outputs["contributions"]
    features = PRIMARY_FEATURES
    z = {(r["compound"], r["feature"]): r["standardized_interaction_mean"] for r in contrib}
    max_abs = max(abs(v) for v in z.values()) or 1
    lines = svg_header(1120, 360)
    lines.append('<text class="title" x="20" y="24">Figure 7. Signed standardized CSD residuals</text>')
    x0, y0, cw, ch = 210, 60, 105, 42
    for j, f in enumerate(features):
        lines.append(f'<text transform="translate({x0+j*cw+20},{y0-8}) rotate(-35)">{f}</text>')
    for i, comp in enumerate(COMPOUNDS):
        lines.append(f'<text x="20" y="{y0+i*ch+25}">{comp}</text>')
        for j, f in enumerate(features):
            val = z.get((comp, f), 0)
            intensity = int(245 - min(180, abs(val) / max_abs * 180))
            color = f"rgb(255,{intensity},{intensity})" if val >= 0 else f"rgb({intensity},{intensity},255)"
            lines.append(f'<rect x="{x0+j*cw}" y="{y0+i*ch}" width="{cw-3}" height="{ch-3}" fill="{color}" stroke="#fff"/>')
            lines.append(f'<text x="{x0+j*cw+8}" y="{y0+i*ch+24}">{val:.2f}</text>')
    write_svg(FIGURES / "figure-7-csd-residual-heatmap.svg", lines)

    sig = outputs["signatures"]
    lines = svg_header(1120, 540)
    lines.append('<text class="title" x="20" y="24">Figure 8. Standardized additive prediction vs observed compound</text>')
    for ci, comp in enumerate(COMPOUNDS):
        sx, sy = 70 + (ci % 2) * 520, 60 + (ci // 2) * 150
        lines.append(f'<text x="{sx}" y="{sy-12}">{comp}</text>')
        rows = [r for r in sig if r["compound"] == comp]
        for j, r in enumerate(rows):
            pred = r["additive_predicted_compound_mean"] / r["scale"]
            obs = r["observed_compound_mean"] / r["scale"]
            base = sx + j * 58
            lines.append(f'<line class="axis" x1="{base}" y1="{sy+80}" x2="{base+36}" y2="{sy+80}"/>')
            lines.append(f'<circle fill="#999" cx="{base+12}" cy="{sy+80-min(70,abs(pred)*3):.2f}" r="3"/>')
            lines.append(f'<circle fill="#de2d26" cx="{base+25}" cy="{sy+80-min(70,abs(obs)*3):.2f}" r="3"/>')
        lines.append(f'<text x="{sx}" y="{sy+105}">gray=additive, red=observed</text>')
    write_svg(FIGURES / "figure-8-additive-vs-observed.svg", lines)

    lines = svg_header(1120, 430)
    lines.append('<text class="title" x="20" y="24">Figure 9. CSD feature contribution fractions</text>')
    for i, comp in enumerate(COMPOUNDS):
        x, y = 190, 60 + i * 64
        lines.append(f'<text x="20" y="{y+18}">{comp}</text>')
        start = x
        rows = [r for r in contrib if r["compound"] == comp]
        for j, r in enumerate(rows):
            w = r["contribution_fraction"] * 820
            lines.append(f'<rect x="{start:.2f}" y="{y}" width="{w:.2f}" height="24" fill="hsl({(j*43)%360},55%,65%)" stroke="#fff"/>')
            if w > 30:
                lines.append(f'<text x="{start+3:.2f}" y="{y+16}">{r["feature"]}</text>')
            start += w
    write_svg(FIGURES / "figure-9-csd-feature-contributions.svg", lines)


def rca_handoff(registry: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "sloscope.phase7b.rca_handoff.v1",
        "performance_features": PRIMARY_FEATURES,
        "telemetry_features": [f for f in EXTENDED_FEATURES if f.startswith("host_cpu") or f.startswith("dependency")],
        "direct_mechanism_evidence": ["server_prompt_tokens_median", "server_output_tokens_median", "dependency_duration_median"],
        "unavailable_features": UNAVAILABLE_FEATURES,
        "labels_separate_not_features": ["active_mechanisms", "mechanism_levels"],
        "feature_registry_path": "analysis/phase7b/feature-registry.json",
    }


def write_report(seal: dict[str, Any], spec: dict[str, Any], outputs: dict[str, list[dict[str, Any]]], registry: list[dict[str, Any]]) -> None:
    report = {
        "schema_version": "sloscope.phase7b.report.v1",
        "campaign_id": CAMPAIGN_ID,
        "phase7b_input_sha256": seal["phase7b_input_sha256"],
        "phase7b_csd_spec_sha256": spec["phase7b_csd_spec_sha256"],
        "source_tree_sha256": SOURCE_SHA,
        "campaign_freeze_sha256": FREEZE_SHA,
        "summary": outputs["summary"],
        "feature_registry": registry,
        "queue_metric_limitation": "llamacpp:requests_processing and requests_deferred are unavailable in formal artifacts and are not inferred from latency.",
        "output_control_drift_handling": "CSD uses contemporaneous matched factorial cells; absolute SLO crossings are not features.",
        "h1_evaluation": "Some retained pairs show Holm-supported multivariate non-additivity; this is not universal across all compounds.",
    }
    write_json(OUT / "phase7b-report.json", report)
    lines = [
        "# Phase 7B Compound Signature Divergence",
        "",
        f"phase7b_input_sha256: `{seal['phase7b_input_sha256']}`",
        f"phase7b_csd_spec_sha256: `{spec['phase7b_csd_spec_sha256']}`",
        "",
        "## Summary",
        "",
    ]
    for row in outputs["summary"]:
        support = "Holm-supported" if row["statistically_supported"] else "not Holm-supported"
        lines.append(f"- {row['compound']}: CSD={row['CSD']:.4g}, 95% bootstrap [{row['bootstrap_95_low']:.4g}, {row['bootstrap_95_high']:.4g}], exact p={row['exact_p']:.4g}, Holm p={row['Holm_p']:.4g}, {support}.")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "CSD quantifies multivariate departure from additive matched-factorial predictions. It is not RCA and does not use ground-truth mechanism labels as features.",
        "Queue metrics are unavailable in the formal artifacts, so queue-depth divergence is not directly measured.",
        "Output-control drift is handled by matched contemporaneous factorial cells; absolute SLO crossings are secondary context only.",
    ])
    (OUT / "phase7b-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance(seal: dict[str, Any], spec: dict[str, Any]) -> None:
    script = Path(__file__)
    provenance = {
        "schema_version": "sloscope.phase7b.provenance.v1",
        "timestamp": utc_now(),
        "phase7b_input_sha256": seal["phase7b_input_sha256"],
        "phase7b_csd_spec_sha256": spec["phase7b_csd_spec_sha256"],
        "phase7a1_input_sha256": seal["phase7a1_input_sha256"],
        "phase7a1_analysis_spec_sha256": seal["phase7a1_analysis_spec_sha256"],
        "publication_run_index_sha256": seal["publication_run_index_sha256"],
        "source_tree_sha256": SOURCE_SHA,
        "campaign_freeze_sha256": FREEZE_SHA,
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": {
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "analysis_script_hashes": {
            str(script.relative_to(ROOT)): sha256_file(script),
            "analysis/phase7b/test_phase7b_csd.py": sha256_file(ROOT / "analysis" / "phase7b" / "test_phase7b_csd.py") if (ROOT / "analysis" / "phase7b" / "test_phase7b_csd.py").exists() else None,
        },
    }
    write_json(OUT / "analysis-provenance.json", provenance)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df, index = load_inputs()
    seal = create_input_seal(index)
    spec = create_csd_spec()
    registry = build_feature_registry(df)
    outputs = materialize_outputs(df, analyze_csd(df, registry))
    write_csv(OUT / "csd-run-residuals.csv", outputs["run_residuals"])
    write_csv(OUT / "csd-signatures.csv", outputs["signatures"])
    write_csv(OUT / "csd-feature-contributions.csv", outputs["contributions"])
    write_csv(OUT / "csd-summary.csv", outputs["summary"])
    write_csv(OUT / "csd-permutation-tests.csv", outputs["permutations"])
    write_csv(OUT / "csd-bootstrap.csv", outputs["bootstrap"])
    write_csv(OUT / "csd-scaling-sensitivity.csv", outputs["scaling"])
    write_csv(OUT / "csd-leave-one-feature-out.csv", outputs["loo"])
    write_csv(OUT / "table-4-csd-summary.csv", outputs["summary"])
    write_csv(OUT / "table-5-csd-feature-contributions.csv", outputs["contributions"])
    figures(outputs)
    write_json(OUT / "rca-feature-handoff.json", rca_handoff(registry))
    write_report(seal, spec, outputs, registry)
    write_provenance(seal, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
