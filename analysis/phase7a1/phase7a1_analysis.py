from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sloscope.artifacts.writer import read_table as read_parquet_table

OUT = ROOT / "analysis" / "phase7a1"
FIGURES = OUT / "figures"

CAMPAIGN_ID = "sloscope-phase6-v2"
SOURCE_SHA = "8e442c813619586072c27a4ec30d7e65d367c3fc1375922935f67c72e2122582"
FREEZE_SHA = "65d9798ea033bc910a832c624324fe068e48552b2a16e6ed639f33763ca028e0"
SLO_THRESHOLDS = {
    "TTFT_SLO": 0.06512947314299491,
    "TOTAL_LATENCY_SLO": 0.26779416389490324,
    "DECODE_DURATION_SLO": 0.20162438542758712,
}
CONDITION_ORDER = [
    "BASELINE",
    "OUTPUT_CONTROL",
    "OUTPUT_LOAD_CONTROL",
    "OUTPUT_DOWNSTREAM_CONTROL",
    "INPUT_MEDIUM",
    "OUTPUT_MEDIUM",
    "LOAD_MEDIUM",
    "DOWNSTREAM_MEDIUM",
    "INPUT_LOAD",
    "INPUT_DOWNSTREAM",
    "OUTPUT_LOAD",
    "OUTPUT_DOWNSTREAM",
    "LOAD_DOWNSTREAM",
]
COUNTED_METRICS = ["ttft_p95", "total_latency_p95", "post_first_token_duration_p95"]
ISOLATED_COMPARISONS = [
    {
        "mechanism": "INPUT",
        "condition": "INPUT_MEDIUM",
        "control": "BASELINE",
        "primary_metric": "ttft_p95",
        "direct_metric": "server_prompt_tokens_median",
    },
    {
        "mechanism": "OUTPUT",
        "condition": "OUTPUT_MEDIUM",
        "control": "OUTPUT_CONTROL",
        "primary_metric": "post_first_token_duration_p95",
        "direct_metric": "server_output_tokens_median",
    },
    {
        "mechanism": "LOAD",
        "condition": "LOAD_MEDIUM",
        "control": "BASELINE",
        "primary_metric": "ttft_p95",
        "direct_metric": "requests_deferred_max",
    },
    {
        "mechanism": "DOWNSTREAM",
        "condition": "DOWNSTREAM_MEDIUM",
        "control": "BASELINE",
        "primary_metric": "total_latency_p95",
        "direct_metric": "dependency_duration_median",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_value(value: str, typ: str) -> Any:
    if value == "":
        return None
    if typ in {"int64", "int"}:
        return int(value)
    if typ in {"double", "float"}:
        return float(value)
    if typ == "bool":
        return value.lower() == "true"
    return value


def read_table(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    try:
        return read_parquet_table(path)
    except Exception:
        pass
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        types = next(reader)
        schema = dict(zip(header, types))
        rows = []
        for raw in reader:
            rows.append({key: parse_value(value, schema[key]) for key, value in zip(header, raw)})
    return schema, rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def percentile(values: Iterable[float | None], p: float) -> float | None:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def median(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


T_CRIT_975 = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    11: 2.200985,
    12: 2.178813,
    13: 2.160369,
    14: 2.144787,
    15: 2.131450,
    16: 2.119905,
    17: 2.109816,
    18: 2.100922,
    19: 2.093024,
    20: 2.085963,
}


def t_critical_975(df: int) -> float:
    if df in T_CRIT_975:
        return T_CRIT_975[df]
    return 1.959964


def t_pdf(x: float, df: int) -> float:
    return (
        math.gamma((df + 1.0) / 2.0)
        / (math.sqrt(df * math.pi) * math.gamma(df / 2.0))
        * (1.0 + (x * x) / df) ** (-(df + 1.0) / 2.0)
    )


def t_cdf(t_value: float, df: int) -> float:
    if t_value == 0:
        return 0.5
    sign = 1.0 if t_value > 0 else -1.0
    upper = abs(t_value)
    # Composite Simpson integration is sufficient for deterministic analysis;
    # p-values are not used to tune the experiment.
    n = max(2000, int(upper * 2000))
    if n % 2:
        n += 1
    h = upper / n
    total = t_pdf(0.0, df) + t_pdf(upper, df)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * t_pdf(i * h, df)
    area = total * h / 3.0
    return 0.5 + sign * area


def paired_t_test(differences: list[float]) -> dict[str, float | None]:
    n = len(differences)
    m = mean(differences)
    sd = sample_sd(differences)
    if n < 2 or m is None or sd in (None, 0):
        return {"t_statistic": None, "df": n - 1, "p_value": None}
    result = scipy_stats.ttest_1samp(differences, popmean=0.0)
    return {"t_statistic": float(result.statistic), "df": n - 1, "p_value": float(result.pvalue)}


def mean_ci(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    m = mean(values)
    sd = sample_sd(values)
    if n < 2 or m is None or sd is None:
        return {"n": n, "mean": m, "sd": sd, "ci_low": None, "ci_high": None}
    margin = float(scipy_stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "ci_low": m - margin, "ci_high": m + margin}


def dz(values: list[float]) -> float | None:
    m = mean(values)
    sd = sample_sd(values)
    if m is None or sd in (None, 0):
        return None
    return m / sd


def holm_adjust(p_values: list[tuple[str, float | None]]) -> dict[str, float | None]:
    present = [(name, p) for name, p in p_values if p is not None]
    m = len(present)
    ordered = sorted(present, key=lambda item: item[1])
    adjusted: dict[str, float | None] = {name: None for name, _ in p_values}
    running = 0.0
    for rank, (name, p) in enumerate(ordered, start=1):
        value = min(1.0, (m - rank + 1) * p)
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def format_ci(mean_value: float | None, low: float | None, high: float | None) -> str:
    if mean_value is None or low is None or high is None:
        return ""
    return f"{mean_value:.6g} [{low:.6g}, {high:.6g}]"


def runtime_metric_values(rows: list[dict[str, Any]], metric_name: str) -> list[float]:
    return [
        float(row["metric_value"])
        for row in rows
        if row.get("metric_name") == metric_name and row.get("metric_value") is not None
    ]


def erroneous_column_metric_values(rows: list[dict[str, Any]], column_name: str) -> list[float]:
    return [float(row[column_name]) for row in rows if row.get(column_name) is not None]


def load_publication_index() -> list[dict[str, Any]]:
    rows = read_jsonl(ROOT / "runs" / "phase6-v2" / "publication-run-index.jsonl")
    if len(rows) != 104:
        raise RuntimeError(f"expected 104 publication rows, got {len(rows)}")
    if len({row["run_id"] for row in rows}) != 104:
        raise RuntimeError("publication index has duplicate run IDs")
    if any(row.get("publication_analysis_eligible") is not True for row in rows):
        raise RuntimeError("publication index contains ineligible rows")
    for row in rows:
        if row.get("source_hash") != SOURCE_SHA:
            raise RuntimeError(f"source hash mismatch for {row['run_id']}")
        if row.get("freeze_hash") != FREEZE_SHA:
            raise RuntimeError(f"freeze hash mismatch for {row['run_id']}")
        if str(row.get("run_path", "")).startswith("runs/phase6/"):
            raise RuntimeError("superseded phase6 run found in publication index")
    return rows


def create_input_seal(index_rows: list[dict[str, Any]]) -> dict[str, Any]:
    seal = {
        "schema_version": "sloscope.phase7a1.input_seal.v1",
        "campaign_id": CAMPAIGN_ID,
        "publication_run_index_sha256": sha256_file(ROOT / "runs" / "phase6-v2" / "publication-run-index.jsonl"),
        "campaign_integrity_report_sha256": sha256_file(ROOT / "runs" / "phase6-v2" / "campaign-integrity-report.json"),
        "factorial_design_sha256": sha256_file(ROOT / "campaigns" / "phase5" / "factorial-design.json"),
        "slo_calibration_sha256": sha256_file(ROOT / "campaigns" / "phase5" / "slo-calibration.json"),
        "slo_definitions_sha256": sha256_file(ROOT / "campaigns" / "phase5" / "slo-definitions.json"),
        "analysis_plan_sha256": sha256_file(ROOT / "campaigns" / "phase5" / "analysis-plan.json"),
        "run_ids": [row["run_id"] for row in index_rows],
        "artifact_manifest_hashes": {row["run_id"]: row["artifact_manifest_hash"] for row in index_rows},
        "source_tree_sha256": SOURCE_SHA,
        "campaign_freeze_sha256": FREEZE_SHA,
    }
    seal["phase7a1_input_sha256"] = sha256_bytes(canonical_json(seal))
    write_json(OUT / "input-seal.json", seal)
    return seal


def create_analysis_spec() -> dict[str, Any]:
    spec = {
        "schema_version": "sloscope.phase7a1.analysis_spec.v1",
        "statistical_unit": "one run / repetition",
        "request_level_role": "requests are used only to compute run-level aggregates",
        "run_level_aggregates": [
            "ttft_p95",
            "total_latency_p95",
            "post_first_token_duration_p95",
            "ttft_median",
            "total_latency_median",
            "post_first_token_duration_median",
            "scheduler_slip_p95",
            "successful_requests_per_second",
            "completion_fraction",
            "server_prompt_tokens_median",
            "server_output_tokens_median",
            "dependency_duration_median",
            "requests_deferred_max",
            "requests_deferred_median",
            "requests_processing_max",
            "requests_processing_median",
        ],
        "matched_controls": {
            "INPUT_MEDIUM": "BASELINE",
            "OUTPUT_MEDIUM": "OUTPUT_CONTROL",
            "LOAD_MEDIUM": "BASELINE",
            "DOWNSTREAM_MEDIUM": "BASELINE",
        },
        "isolated_primary_comparisons": ISOLATED_COMPARISONS,
        "factorial_mappings_source": "campaigns/phase5/factorial-design.json",
        "effect_definitions": {
            "paired_delta": "condition_metric - matched_control_metric by repetition",
            "matched_ratio": "condition_metric / matched_control_metric by repetition",
            "factorial_interaction_delta": "Y_A1B1 - Y_A1B0 - Y_A0B1 + Y_A0B0",
        },
        "ci_procedure": "95% two-sided Student-t CI across eight run-level repetitions using scipy.stats.t",
        "paired_test_procedure": "two-sided paired Student-t test on eight repetition-level deltas using scipy.stats.ttest_1samp",
        "holm_families": {
            "isolated": ["INPUT", "OUTPUT", "LOAD", "DOWNSTREAM"],
            "factorial_interactions": "5 retained factorial families x 3 counted SLO metrics",
        },
        "slo_violation_definitions": {
            "NO_COUNTED_SLO_VIOLATION": 0,
            "SINGLE_SLO_VIOLATION": 1,
            "COMPOUND_SLO_VIOLATION": ">=2",
            "counted_slos": ["TTFT_SLO", "TOTAL_LATENCY_SLO", "DECODE_DURATION_SLO"],
        },
        "figures": {
            "figure_1": "run-level SLO metric distributions with threshold lines",
            "figure_2": "paired isolated-mechanism deltas with 95% CI",
            "figure_3": "condition x counted-SLO violation frequency matrix",
            "figure_4": "factorial interaction forest plot for 15 contrasts",
            "figure_5": "dedicated calibration controls versus formal controls",
        },
        "control_transportability_audit": {
            "families": ["BASELINE", "OUTPUT_CONTROL"],
            "metrics": COUNTED_METRICS,
            "calibration_source": "calibrations/phase5.2-slo",
        },
        "no_outlier_removal": True,
    }
    spec["phase7a1_analysis_spec_sha256"] = sha256_bytes(canonical_json(spec))
    write_json(OUT / "analysis-spec.json", spec)
    return spec


def aggregate_run(index_row: dict[str, Any]) -> dict[str, Any]:
    run_dir = ROOT / index_row["run_path"]
    condition = read_json(run_dir / "experimental_condition.json")
    _, requests = read_table(run_dir / "requests.parquet")
    _, runtime = read_table(run_dir / "runtime_metrics.parquet")
    _, system = read_table(run_dir / "system_metrics.parquet")
    _, traces = read_table(run_dir / "traces.parquet")
    success = [row for row in requests if row.get("status") == "success"]
    ttft = [row["first_token_time"] - row["actual_arrival"] for row in success if row.get("first_token_time") is not None]
    total = [row["completion_time"] - row["actual_arrival"] for row in success if row.get("completion_time") is not None]
    decode = [
        row["completion_time"] - row["first_token_time"]
        for row in success
        if row.get("completion_time") is not None and row.get("first_token_time") is not None
    ]
    arrivals = [row["actual_arrival"] for row in requests if row.get("actual_arrival") is not None]
    completions = [row["completion_time"] for row in success if row.get("completion_time") is not None]
    throughput = None
    if arrivals and completions and max(completions) > min(arrivals):
        throughput = len(success) / (max(completions) - min(arrivals))
    def values(name: str, rows: list[dict[str, Any]]) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]
    deferred = runtime_metric_values(runtime, "llamacpp:requests_deferred")
    active = runtime_metric_values(runtime, "llamacpp:requests_processing")
    host_cpu = values("host_cpu_percent", system)
    server_cpu = values("server_process_cpu_percent", system)
    server_rss = values("server_process_rss_bytes", system)
    row = {
        "run_id": index_row["run_id"],
        "condition_id": condition["condition_id"],
        "repetition": int(condition["repetition"]),
        "attempt": int(condition["attempt"]),
        "active_mechanisms": json.dumps(condition.get("active_mechanisms", []), sort_keys=True),
        "mechanism_levels": json.dumps(condition.get("mechanism_levels", {}), sort_keys=True),
        "compound_degree": int(condition["compound_degree"]),
        "ttft_p95": percentile(ttft, 0.95),
        "total_latency_p95": percentile(total, 0.95),
        "post_first_token_duration_p95": percentile(decode, 0.95),
        "ttft_median": median(ttft),
        "total_latency_median": median(total),
        "post_first_token_duration_median": median(decode),
        "scheduler_slip_p95": percentile(values("scheduler_slip", requests), 0.95),
        "successful_requests_per_second": throughput,
        "completion_fraction": len(success) / len(requests) if requests else None,
        "server_prompt_tokens_median": median(values("server_prompt_tokens", success)),
        "server_prompt_tokens_p95": percentile(values("server_prompt_tokens", success), 0.95),
        "server_output_tokens_median": median(values("server_output_tokens", success)),
        "server_output_tokens_p95": percentile(values("server_output_tokens", success), 0.95),
        "dependency_duration_median": median(values("dependency_duration", success)),
        "dependency_duration_p95": percentile(values("dependency_duration", success), 0.95),
        "requests_processing_max": max(active) if active else None,
        "requests_processing_median": median(active),
        "requests_deferred_max": max(deferred) if deferred else None,
        "requests_deferred_median": median(deferred),
        "runtime_queue_metrics_unavailable_reason": None
        if active or deferred
        else "llama.cpp long-form requests_processing/requests_deferred metrics absent from formal runtime_metrics artifacts",
        "host_cpu_median": median(host_cpu),
        "host_cpu_p95": percentile(host_cpu, 0.95),
        "server_cpu_median": median(server_cpu),
        "server_cpu_p95": percentile(server_cpu, 0.95),
        "server_rss_median": median(server_rss),
        "server_rss_max": max(server_rss) if server_rss else None,
        "server_process_metrics_unavailable_reason": None if server_cpu or server_rss else "server PID telemetry not supplied in formal configs",
        "successful_requests": len(success),
        "failed_requests": len([row for row in requests if row.get("status") == "failed"]),
        "admission_failed": len([row for row in requests if row.get("status") == "admission_failed"]),
        "measured_request_count": len(requests),
        "trace_row_count": len(traces),
    }
    row["ttft_slo_violated"] = bool(row["ttft_p95"] is not None and row["ttft_p95"] > SLO_THRESHOLDS["TTFT_SLO"])
    row["total_latency_slo_violated"] = bool(
        row["total_latency_p95"] is not None and row["total_latency_p95"] > SLO_THRESHOLDS["TOTAL_LATENCY_SLO"]
    )
    row["decode_duration_slo_violated"] = bool(
        row["post_first_token_duration_p95"] is not None and row["post_first_token_duration_p95"] > SLO_THRESHOLDS["DECODE_DURATION_SLO"]
    )
    row["slo_violation_count"] = int(row["ttft_slo_violated"]) + int(row["total_latency_slo_violated"]) + int(row["decode_duration_slo_violated"])
    row["slo_violation_class"] = (
        "NO_COUNTED_SLO_VIOLATION"
        if row["slo_violation_count"] == 0
        else "SINGLE_SLO_VIOLATION"
        if row["slo_violation_count"] == 1
        else "COMPOUND_SLO_VIOLATION"
    )
    return row


def condition_summaries(run_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_cond = defaultdict(list)
    for row in run_rows:
        by_cond[row["condition_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    metrics = [
        "ttft_p95",
        "total_latency_p95",
        "post_first_token_duration_p95",
        "ttft_median",
        "total_latency_median",
        "post_first_token_duration_median",
        "scheduler_slip_p95",
        "successful_requests_per_second",
        "completion_fraction",
        "server_prompt_tokens_median",
        "server_prompt_tokens_p95",
        "server_output_tokens_median",
        "server_output_tokens_p95",
        "dependency_duration_median",
        "dependency_duration_p95",
        "requests_processing_max",
        "requests_processing_median",
        "requests_deferred_max",
        "requests_deferred_median",
        "host_cpu_median",
        "host_cpu_p95",
        "server_cpu_median",
        "server_cpu_p95",
        "server_rss_median",
        "server_rss_max",
    ]
    for cond in CONDITION_ORDER:
        rows = sorted(by_cond[cond], key=lambda item: item["repetition"])
        base = {"condition_id": cond, "n": len(rows)}
        for metric in metrics:
            vals = [float(row[metric]) for row in rows if row.get(metric) is not None]
            ci = mean_ci(vals)
            base[f"{metric}_mean"] = ci["mean"]
            base[f"{metric}_sd"] = ci["sd"]
            base[f"{metric}_ci_low"] = ci["ci_low"]
            base[f"{metric}_ci_high"] = ci["ci_high"]
            base[f"{metric}_median"] = median(vals)
            base[f"{metric}_min"] = min(vals) if vals else None
            base[f"{metric}_max"] = max(vals) if vals else None
        summaries.append(base)
        counts = Counter(row["slo_violation_class"] for row in rows)
        vrow = {
            "condition_id": cond,
            "n": len(rows),
            "ttft_violations": sum(1 for row in rows if row["ttft_slo_violated"]),
            "total_latency_violations": sum(1 for row in rows if row["total_latency_slo_violated"]),
            "decode_duration_violations": sum(1 for row in rows if row["decode_duration_slo_violated"]),
            "no_slo_runs": counts["NO_COUNTED_SLO_VIOLATION"],
            "single_slo_runs": counts["SINGLE_SLO_VIOLATION"],
            "compound_slo_runs": counts["COMPOUND_SLO_VIOLATION"],
        }
        for key in ["ttft_violations", "total_latency_violations", "decode_duration_violations", "no_slo_runs", "single_slo_runs", "compound_slo_runs"]:
            lo, hi = wilson_interval(int(vrow[key]), len(rows))
            vrow[f"{key}_proportion"] = int(vrow[key]) / len(rows) if rows else None
            vrow[f"{key}_wilson_low"] = lo
            vrow[f"{key}_wilson_high"] = hi
        violations.append(vrow)
    return summaries, violations


def map_by_condition_rep(run_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["condition_id"], row["repetition"]): row for row in run_rows}


def isolated_effects(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = map_by_condition_rep(run_rows)
    rows = []
    raw_p: list[tuple[str, float | None]] = []
    for comp in ISOLATED_COMPARISONS:
        diffs = []
        ratios = []
        direct_values = []
        control_direct_values = []
        for rep in range(1, 9):
            cond = by[(comp["condition"], rep)]
            ctrl = by[(comp["control"], rep)]
            y = cond[comp["primary_metric"]]
            x = ctrl[comp["primary_metric"]]
            diffs.append(y - x)
            ratios.append(y / x if x else None)
            if cond.get(comp["direct_metric"]) is not None:
                direct_values.append(cond.get(comp["direct_metric"]))
            if ctrl.get(comp["direct_metric"]) is not None:
                control_direct_values.append(ctrl.get(comp["direct_metric"]))
        ci = mean_ci(diffs)
        test = paired_t_test(diffs)
        row = {
            "mechanism": comp["mechanism"],
            "condition": comp["condition"],
            "control": comp["control"],
            "primary_metric": comp["primary_metric"],
            "paired_differences": json.dumps(diffs),
            "mean_paired_delta": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "matched_ratio_median": median(ratios),
            "matched_ratio_min": min(v for v in ratios if v is not None),
            "matched_ratio_max": max(v for v in ratios if v is not None),
            "dz": dz(diffs),
            "t_statistic": test["t_statistic"],
            "df": test["df"],
            "raw_p": test["p_value"],
            "direct_metric": comp["direct_metric"],
            "direct_evidence_values": json.dumps(direct_values),
            "direct_evidence_median": median(direct_values),
            "control_direct_evidence_median": median(control_direct_values),
            "direct_evidence_unavailable_reason": cond.get("runtime_queue_metrics_unavailable_reason")
            if comp["direct_metric"] in {"requests_deferred_max", "requests_processing_max"} and not direct_values
            else None,
        }
        rows.append(row)
        raw_p.append((comp["mechanism"], test["p_value"]))
    adjusted = holm_adjust(raw_p)
    for row in rows:
        row["holm_p"] = adjusted[row["mechanism"]]
    return rows


def direct_evidence_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cond = defaultdict(list)
    for row in run_rows:
        by_cond[row["condition_id"]].append(row)
    specs = {
        "INPUT_MEDIUM": ["server_prompt_tokens_median", "server_prompt_tokens_p95", "requests_deferred_max", "requests_deferred_median", "requests_processing_max", "requests_processing_median"],
        "OUTPUT_MEDIUM": ["server_output_tokens_median", "server_output_tokens_p95", "requests_deferred_max", "requests_deferred_median", "requests_processing_max", "requests_processing_median"],
        "LOAD_MEDIUM": ["requests_deferred_max", "requests_deferred_median", "requests_processing_max", "requests_processing_median"],
        "DOWNSTREAM_MEDIUM": ["dependency_duration_median", "dependency_duration_p95", "requests_deferred_max", "requests_deferred_median", "requests_processing_max", "requests_processing_median"],
        "INPUT_LOAD": ["server_prompt_tokens_median", "requests_deferred_max", "requests_deferred_median", "requests_processing_max", "requests_processing_median"],
        "INPUT_DOWNSTREAM": ["server_prompt_tokens_median", "dependency_duration_median", "requests_deferred_max", "requests_deferred_median", "requests_processing_max", "requests_processing_median"],
        "OUTPUT_LOAD": ["server_output_tokens_median", "requests_deferred_max", "requests_deferred_median", "requests_processing_max", "requests_processing_median"],
        "OUTPUT_DOWNSTREAM": ["server_output_tokens_median", "dependency_duration_median", "requests_deferred_max", "requests_deferred_median", "requests_processing_max", "requests_processing_median"],
        "LOAD_DOWNSTREAM": ["requests_deferred_max", "requests_deferred_median", "requests_processing_max", "requests_processing_median", "dependency_duration_median"],
    }
    rows = []
    for cond, metrics in specs.items():
        for metric in metrics:
            values = [row.get(metric) for row in by_cond[cond] if row.get(metric) is not None]
            rows.append(
                {
                    "condition_id": cond,
                    "metric": metric,
                    "values": json.dumps(values),
                    "median": median(values),
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "queue_secondary_signature": bool(metric == "requests_deferred_max" and cond not in {"LOAD_MEDIUM", "INPUT_LOAD", "OUTPUT_LOAD", "LOAD_DOWNSTREAM"} and values and max(values) > 0),
                    "unavailable_reason": by_cond[cond][0].get("runtime_queue_metrics_unavailable_reason")
                    if metric in {"requests_deferred_max", "requests_processing_max", "requests_deferred_median", "requests_processing_median"} and not values and by_cond[cond]
                    else None,
                }
            )
    return rows


def queue_signature_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cond = defaultdict(list)
    for row in run_rows:
        by_cond[row["condition_id"]].append(row)
    rows = []
    load_conditions = {"LOAD_MEDIUM", "INPUT_LOAD", "OUTPUT_LOAD", "LOAD_DOWNSTREAM"}
    for cond in CONDITION_ORDER:
        cond_rows = sorted(by_cond[cond], key=lambda row: row["repetition"])
        deferred_max_values = [row.get("requests_deferred_max") for row in cond_rows if row.get("requests_deferred_max") is not None]
        processing_max_values = [row.get("requests_processing_max") for row in cond_rows if row.get("requests_processing_max") is not None]
        queue_observed = bool(deferred_max_values and max(deferred_max_values) > 0)
        intended = cond in load_conditions
        rows.append(
            {
                "condition_id": cond,
                "ground_truth_load_mechanism": intended,
                "queueing_intended_not_confound": intended,
                "secondary_observed_queue_signature": bool(queue_observed and not intended),
                "requests_deferred_max_values": json.dumps(deferred_max_values),
                "requests_deferred_max_median": median(deferred_max_values),
                "requests_deferred_max_max": max(deferred_max_values) if deferred_max_values else None,
                "requests_processing_max_values": json.dumps(processing_max_values),
                "requests_processing_max_median": median(processing_max_values),
                "requests_processing_max_max": max(processing_max_values) if processing_max_values else None,
                "unavailable_reason": cond_rows[0].get("runtime_queue_metrics_unavailable_reason") if cond_rows and not deferred_max_values and not processing_max_values else None,
            }
        )
    return rows


def factorial_effects(run_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    design = read_json(ROOT / "campaigns" / "phase5" / "factorial-design.json")
    by = map_by_condition_rep(run_rows)
    interactions: list[dict[str, Any]] = []
    marginals: list[dict[str, Any]] = []
    raw_tests: list[tuple[str, float | None]] = []
    for compound in ["INPUT_LOAD", "INPUT_DOWNSTREAM", "OUTPUT_LOAD", "OUTPUT_DOWNSTREAM", "LOAD_DOWNSTREAM"]:
        mapping = design[compound]
        for metric in COUNTED_METRICS:
            deltas = []
            for rep in range(1, 9):
                a0b0 = by[(mapping["A0B0"], rep)][metric]
                a1b0 = by[(mapping["A1B0"], rep)][metric]
                a0b1 = by[(mapping["A0B1"], rep)][metric]
                a1b1 = by[(mapping["A1B1"], rep)][metric]
                delta = a1b1 - a1b0 - a0b1 + a0b0
                deltas.append(delta)
                marginals.append(
                    {
                        "compound": compound,
                        "metric": metric,
                        "repetition": rep,
                        "A0B0": mapping["A0B0"],
                        "A1B0": mapping["A1B0"],
                        "A0B1": mapping["A0B1"],
                        "A1B1": mapping["A1B1"],
                        "effect_A_when_B0": a1b0 - a0b0,
                        "effect_A_when_B1": a1b1 - a0b1,
                        "effect_B_when_A0": a0b1 - a0b0,
                        "effect_B_when_A1": a1b1 - a1b0,
                        "interaction_delta": delta,
                    }
                )
            ci = mean_ci(deltas)
            test = paired_t_test(deltas)
            if ci["ci_low"] is not None and ci["ci_low"] > 0:
                classification = "SUPER_ADDITIVE_EVIDENCE"
            elif ci["ci_high"] is not None and ci["ci_high"] < 0:
                classification = "SUB_ADDITIVE_EVIDENCE"
            else:
                classification = "INTERACTION_UNCERTAIN"
            name = f"{compound}:{metric}"
            interactions.append(
                {
                    "compound": compound,
                    "metric": metric,
                    "factor_a": mapping["factor_a"],
                    "factor_b": mapping["factor_b"],
                    "A0B0": mapping["A0B0"],
                    "A1B0": mapping["A1B0"],
                    "A0B1": mapping["A0B1"],
                    "A1B1": mapping["A1B1"],
                    "interaction_deltas": json.dumps(deltas),
                    "mean_interaction_delta": ci["mean"],
                    "sd": ci["sd"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "t_statistic": test["t_statistic"],
                    "df": test["df"],
                    "raw_p": test["p_value"],
                    "dz": dz(deltas),
                    "classification": classification,
                    "test_id": name,
                }
            )
            raw_tests.append((name, test["p_value"]))
    adjusted = holm_adjust(raw_tests)
    for row in interactions:
        row["holm_p"] = adjusted[row["test_id"]]
    return marginals, interactions


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def control_rows_from_calibration() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [
        ROOT / "calibrations" / "phase5.2-slo" / "baseline-runs.csv",
        ROOT / "calibrations" / "phase5.2-slo" / "output-control-runs.csv",
    ]
    for path in sources:
        for raw in read_csv_dicts(path):
            rows.append(
                {
                    "source": "calibration",
                    "condition_id": raw["condition_id"],
                    "run_id": raw["run_id"],
                    "repetition": int(raw["repetition"]),
                    "ttft_p95": float(raw["ttft_p95"]),
                    "total_latency_p95": float(raw["total_latency_p95"]),
                    "post_first_token_duration_p95": float(raw["post_first_token_duration_p95"]),
                }
            )
    return rows


def control_transportability_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calibration = control_rows_from_calibration()
    formal = [
        {
            "source": "formal",
            "condition_id": row["condition_id"],
            "run_id": row["run_id"],
            "repetition": row["repetition"],
            "ttft_p95": row["ttft_p95"],
            "total_latency_p95": row["total_latency_p95"],
            "post_first_token_duration_p95": row["post_first_token_duration_p95"],
            "host_cpu_median": row.get("host_cpu_median"),
            "host_cpu_p95": row.get("host_cpu_p95"),
        }
        for row in run_rows
        if row["condition_id"] in {"BASELINE", "OUTPUT_CONTROL"}
    ]
    rows = []
    for condition_id in ["BASELINE", "OUTPUT_CONTROL"]:
        for metric in COUNTED_METRICS:
            cvals = [row[metric] for row in calibration if row["condition_id"] == condition_id]
            fvals = [row[metric] for row in formal if row["condition_id"] == condition_id]
            cmean = mean(cvals)
            fmean = mean(fvals)
            shift = fmean - cmean if fmean is not None and cmean is not None else None
            rows.append(
                {
                    "condition_id": condition_id,
                    "metric": metric,
                    "calibration_n": len(cvals),
                    "calibration_mean": cmean,
                    "calibration_sd": sample_sd(cvals),
                    "calibration_min": min(cvals) if cvals else None,
                    "calibration_max": max(cvals) if cvals else None,
                    "formal_n": len(fvals),
                    "formal_mean": fmean,
                    "formal_sd": sample_sd(fvals),
                    "formal_min": min(fvals) if fvals else None,
                    "formal_max": max(fvals) if fvals else None,
                    "absolute_shift": shift,
                    "percent_shift": (shift / cmean * 100.0) if shift is not None and cmean else None,
                    "welch_t": float(scipy_stats.ttest_ind(fvals, cvals, equal_var=False).statistic) if len(cvals) > 1 and len(fvals) > 1 else None,
                    "welch_p_diagnostic": float(scipy_stats.ttest_ind(fvals, cvals, equal_var=False).pvalue) if len(cvals) > 1 and len(fvals) > 1 else None,
                }
            )
    return rows


def control_slo_crossing_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calibration = control_rows_from_calibration()
    formal = [
        {
            "source": "formal",
            "condition_id": row["condition_id"],
            "run_id": row["run_id"],
            "repetition": row["repetition"],
            "ttft_p95": row["ttft_p95"],
            "total_latency_p95": row["total_latency_p95"],
            "post_first_token_duration_p95": row["post_first_token_duration_p95"],
        }
        for row in run_rows
        if row["condition_id"] in {"BASELINE", "OUTPUT_CONTROL"}
    ]
    rows = []
    metric_to_slo = {
        "ttft_p95": "TTFT_SLO",
        "total_latency_p95": "TOTAL_LATENCY_SLO",
        "post_first_token_duration_p95": "DECODE_DURATION_SLO",
    }
    for source, source_rows in [("calibration", calibration), ("formal", formal)]:
        for condition_id in ["BASELINE", "OUTPUT_CONTROL"]:
            subset = [row for row in source_rows if row["condition_id"] == condition_id]
            out = {"source": source, "condition_id": condition_id, "n": len(subset)}
            total_any = 0
            for metric, slo in metric_to_slo.items():
                count = sum(1 for row in subset if row[metric] > SLO_THRESHOLDS[slo])
                out[f"{metric}_crossings"] = count
                out[f"{metric}_threshold"] = SLO_THRESHOLDS[slo]
            total_any = sum(
                1
                for row in subset
                if any(row[metric] > SLO_THRESHOLDS[slo] for metric, slo in metric_to_slo.items())
            )
            out["any_counted_slo_crossings"] = total_any
            rows.append(out)
    return rows


def statistical_diff_audit(isolated: list[dict[str, Any]], interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old_rows = []
    for path in [ROOT / "analysis" / "phase7a" / "isolated-effects.csv", ROOT / "analysis" / "phase7a" / "factorial-interactions.csv"]:
        if path.exists():
            old_rows.extend(read_csv_dicts(path))
    old_by_id: dict[str, dict[str, str]] = {}
    for row in old_rows:
        test_id = row.get("test_id") or row.get("mechanism")
        if test_id:
            old_by_id[test_id] = row
    new_rows = []
    for row in isolated:
        new_rows.append({"test_id": row["mechanism"], **row})
    for row in interactions:
        new_rows.append(row)
    audit = []
    for row in new_rows:
        test_id = row["test_id"]
        old = old_by_id.get(test_id, {})
        old_raw = as_float(old.get("raw_p"))
        new_raw = as_float(row.get("raw_p"))
        old_holm = as_float(old.get("holm_p"))
        new_holm = as_float(row.get("holm_p"))
        old_class = old.get("classification")
        new_class = row.get("classification")
        audit.append(
            {
                "test_id": test_id,
                "old_t": as_float(old.get("t_statistic")),
                "new_t": as_float(row.get("t_statistic")),
                "old_raw_p": old_raw,
                "new_raw_p": new_raw,
                "old_holm_p": old_holm,
                "new_holm_p": new_holm,
                "significance_changed": (old_holm is not None and new_holm is not None and (old_holm < 0.05) != (new_holm < 0.05)),
                "classification_changed": bool(old_class and new_class and old_class != new_class),
            }
        )
    return audit


def table_outputs(condition_summary: list[dict[str, Any]], slo_violations: list[dict[str, Any]], isolated: list[dict[str, Any]], factorial: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_by_cond = {row["condition_id"]: row for row in condition_summary}
    vio_by_cond = {row["condition_id"]: row for row in slo_violations}
    table1 = []
    for cond in CONDITION_ORDER:
        s = summary_by_cond[cond]
        v = vio_by_cond[cond]
        table1.append(
            {
                "condition": cond,
                "n": s["n"],
                "TTFT p95 mean [95% CI]": format_ci(s["ttft_p95_mean"], s["ttft_p95_ci_low"], s["ttft_p95_ci_high"]),
                "total latency p95 mean [95% CI]": format_ci(s["total_latency_p95_mean"], s["total_latency_p95_ci_low"], s["total_latency_p95_ci_high"]),
                "decode p95 mean [95% CI]": format_ci(s["post_first_token_duration_p95_mean"], s["post_first_token_duration_p95_ci_low"], s["post_first_token_duration_p95_ci_high"]),
                "TTFT violations": f"{v['ttft_violations']}/8",
                "total violations": f"{v['total_latency_violations']}/8",
                "decode violations": f"{v['decode_duration_violations']}/8",
                "compound-SLO violations": f"{v['compound_slo_runs']}/8",
            }
        )
    table2 = [
        {
            "mechanism": row["mechanism"],
            "control": row["control"],
            "control compound-SLO violations": f"{vio_by_cond[row['control']]['compound_slo_runs']}/8",
            "primary metric": row["primary_metric"],
            "mean paired delta": row["mean_paired_delta"],
            "95% CI": format_ci(row["mean_paired_delta"], row["ci_low"], row["ci_high"]),
            "matched ratio": f"{row['matched_ratio_median']:.4g} [{row['matched_ratio_min']:.4g}, {row['matched_ratio_max']:.4g}]",
            "dz": row["dz"],
            "raw p": row["raw_p"],
            "Holm p": row["holm_p"],
            "direct-evidence summary": f"{row['direct_metric']} median={row['direct_evidence_median']}",
            "direct-evidence note": row.get("direct_evidence_unavailable_reason"),
        }
        for row in isolated
    ]
    table3 = [
        {
            "compound": row["compound"],
            "metric": row["metric"],
            "A0B0": row["A0B0"],
            "A1B0": row["A1B0"],
            "A0B1": row["A0B1"],
            "A1B1": row["A1B1"],
            "A0B0 compound-SLO violations": f"{vio_by_cond[row['A0B0']]['compound_slo_runs']}/8",
            "mean interaction delta": row["mean_interaction_delta"],
            "95% CI": format_ci(row["mean_interaction_delta"], row["ci_low"], row["ci_high"]),
            "dz": row["dz"],
            "raw p": row["raw_p"],
            "Holm p": row["holm_p"],
            "classification": row["classification"],
        }
        for row in factorial
    ]
    return table1, table2, table3


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:10px}.title{font-size:14px;font-weight:bold}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.threshold{stroke:#b00020;stroke-width:1.5;stroke-dasharray:4 3}.point{fill:#1f77b4;opacity:.75}.ci{stroke:#111;stroke-width:1.5}.bar{fill:#6baed6}.heat0{fill:#f7fbff}.heat1{fill:#deebf7}.heat2{fill:#9ecae1}.heat3{fill:#3182bd}</style>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + ["</svg>\n"]), encoding="utf-8")


def figure_1(run_rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("ttft_p95", SLO_THRESHOLDS["TTFT_SLO"], "TTFT p95"),
        ("total_latency_p95", SLO_THRESHOLDS["TOTAL_LATENCY_SLO"], "Total latency p95"),
        ("post_first_token_duration_p95", SLO_THRESHOLDS["DECODE_DURATION_SLO"], "Decode duration p95"),
    ]
    width, height = 1200, 720
    lines = svg_header(width, height)
    lines.append('<text class="title" x="20" y="24">Figure 1. Run-level counted SLO metric distributions by condition</text>')
    panel_h = 205
    left, right = 160, 1180
    for mi, (metric, threshold, label) in enumerate(metrics):
        y0 = 50 + mi * 220
        vals = [row[metric] for row in run_rows if row[metric] is not None]
        vmin, vmax = min(vals + [threshold]), max(vals + [threshold])
        pad = (vmax - vmin) * 0.08 or 0.01
        vmin -= pad
        vmax += pad
        def y(value: float) -> float:
            return y0 + panel_h - (value - vmin) / (vmax - vmin) * panel_h
        lines.append(f'<text x="20" y="{y0+12}">{label}</text>')
        lines.append(f'<line class="axis" x1="{left}" y1="{y0}" x2="{left}" y2="{y0+panel_h}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{y0+panel_h}" x2="{right}" y2="{y0+panel_h}"/>')
        th_y = y(threshold)
        lines.append(f'<line class="threshold" x1="{left}" y1="{th_y}" x2="{right}" y2="{th_y}"/>')
        lines.append(f'<text x="{right-120}" y="{th_y-4}">SLO threshold</text>')
        step = (right - left) / len(CONDITION_ORDER)
        for ci, cond in enumerate(CONDITION_ORDER):
            cx = left + ci * step + step / 2
            rows = [row for row in run_rows if row["condition_id"] == cond]
            for ri, row in enumerate(sorted(rows, key=lambda item: item["repetition"])):
                jitter = (ri - 3.5) * min(4, step / 18)
                lines.append(f'<circle class="point" cx="{cx+jitter:.2f}" cy="{y(row[metric]):.2f}" r="3"/>')
            lines.append(f'<text transform="translate({cx-5:.2f},{y0+panel_h+74}) rotate(-55)">{cond}</text>')
    write_svg(FIGURES / "figure-1-slo-distributions.svg", lines)


def figure_2(isolated: list[dict[str, Any]]) -> None:
    width, height = 900, 320
    lines = svg_header(width, height)
    lines.append('<text class="title" x="20" y="24">Figure 2. Paired isolated-mechanism primary effects</text>')
    vals = []
    for row in isolated:
        vals.extend(json.loads(row["paired_differences"]))
        vals.extend([row["ci_low"], row["ci_high"]])
    vmin, vmax = min(vals), max(vals)
    pad = (vmax - vmin) * 0.1 or 0.01
    vmin -= pad
    vmax += pad
    left, right, top, bottom = 220, 860, 55, 280
    def x(value: float) -> float:
        return left + (value - vmin) / (vmax - vmin) * (right - left)
    lines.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    zx = x(0.0)
    lines.append(f'<line class="threshold" x1="{zx}" y1="{top}" x2="{zx}" y2="{bottom}"/>')
    row_gap = 48
    for i, row in enumerate(isolated):
        y = top + 25 + i * row_gap
        diffs = json.loads(row["paired_differences"])
        lines.append(f'<text x="20" y="{y+4}">{row["mechanism"]} ({row["primary_metric"]})</text>')
        lines.append(f'<line class="ci" x1="{x(row["ci_low"]):.2f}" y1="{y}" x2="{x(row["ci_high"]):.2f}" y2="{y}"/>')
        lines.append(f'<circle fill="#111" cx="{x(row["mean_paired_delta"]):.2f}" cy="{y}" r="4"/>')
        for j, d in enumerate(diffs):
            lines.append(f'<circle class="point" cx="{x(d):.2f}" cy="{y + (j-3.5)*2.2:.2f}" r="2.5"/>')
    write_svg(FIGURES / "figure-2-isolated-effects.svg", lines)


def figure_3(slo_violations: list[dict[str, Any]]) -> None:
    width, height = 900, 500
    metrics = ["ttft_violations", "total_latency_violations", "decode_duration_violations", "compound_slo_runs"]
    labels = ["TTFT", "Total", "Decode", "Compound SLO"]
    lines = svg_header(width, height)
    lines.append('<text class="title" x="20" y="24">Figure 3. Counted-SLO violation frequencies by condition</text>')
    cell_w, cell_h = 120, 28
    x0, y0 = 240, 60
    by_cond = {row["condition_id"]: row for row in slo_violations}
    for j, label in enumerate(labels):
        lines.append(f'<text x="{x0+j*cell_w+8}" y="{y0-10}">{label}</text>')
    for i, cond in enumerate(CONDITION_ORDER):
        y = y0 + i * cell_h
        lines.append(f'<text x="20" y="{y+18}">{cond}</text>')
        row = by_cond[cond]
        for j, metric in enumerate(metrics):
            k = int(row[metric])
            cls = f"heat{min(3, k // 3 + (1 if k else 0))}"
            x = x0 + j * cell_w
            lines.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{cell_w-4}" height="{cell_h-4}" stroke="#fff"/>')
            lines.append(f'<text x="{x+45}" y="{y+17}">{k}/8</text>')
    write_svg(FIGURES / "figure-3-slo-violation-heatmap.svg", lines)


def figure_4(interactions: list[dict[str, Any]]) -> None:
    width, height = 1050, 720
    vals = []
    for row in interactions:
        vals.extend([row["ci_low"], row["ci_high"], row["mean_interaction_delta"]])
    vmin, vmax = min(vals), max(vals)
    pad = (vmax - vmin) * 0.1 or 0.01
    vmin -= pad
    vmax += pad
    left, right, top = 320, 1010, 50
    def x(value: float) -> float:
        return left + (value - vmin) / (vmax - vmin) * (right - left)
    lines = svg_header(width, height)
    lines.append('<text class="title" x="20" y="24">Figure 4. Factorial interaction contrasts</text>')
    zx = x(0.0)
    lines.append(f'<line class="threshold" x1="{zx}" y1="{top}" x2="{zx}" y2="{height-30}"/>')
    for i, row in enumerate(interactions):
        y = top + 22 + i * 42
        label = f"{row['compound']} / {row['metric']}"
        lines.append(f'<text x="20" y="{y+4}">{label}</text>')
        lines.append(f'<line class="ci" x1="{x(row["ci_low"]):.2f}" y1="{y}" x2="{x(row["ci_high"]):.2f}" y2="{y}"/>')
        color = "#238b45" if row["classification"] == "SUPER_ADDITIVE_EVIDENCE" else "#cb181d" if row["classification"] == "SUB_ADDITIVE_EVIDENCE" else "#555"
        lines.append(f'<circle fill="{color}" cx="{x(row["mean_interaction_delta"]):.2f}" cy="{y}" r="4"/>')
    write_svg(FIGURES / "figure-4-factorial-interactions.svg", lines)


def figure_5_control_transportability(control_transportability: list[dict[str, Any]]) -> None:
    width, height = 980, 520
    metrics = [
        ("ttft_p95", SLO_THRESHOLDS["TTFT_SLO"], "TTFT p95"),
        ("total_latency_p95", SLO_THRESHOLDS["TOTAL_LATENCY_SLO"], "Total latency p95"),
        ("post_first_token_duration_p95", SLO_THRESHOLDS["DECODE_DURATION_SLO"], "Decode p95"),
    ]
    by = {(row["condition_id"], row["metric"]): row for row in control_transportability}
    lines = svg_header(width, height)
    lines.append('<text class="title" x="20" y="24">Figure 5. Dedicated calibration controls vs formal controls</text>')
    panel_w, panel_h = 280, 170
    for mi, (metric, threshold, label) in enumerate(metrics):
        x0 = 80 + mi * 300
        for ci, condition in enumerate(["BASELINE", "OUTPUT_CONTROL"]):
            y0 = 55 + ci * 220
            row = by[(condition, metric)]
            vals = [row["calibration_min"], row["calibration_max"], row["formal_min"], row["formal_max"], threshold]
            vmin, vmax = min(vals), max(vals)
            pad = (vmax - vmin) * 0.12 or 0.01
            vmin -= pad
            vmax += pad
            def y(value: float) -> float:
                return y0 + panel_h - (value - vmin) / (vmax - vmin) * panel_h
            lines.append(f'<text x="{x0}" y="{y0-12}">{condition} / {label}</text>')
            lines.append(f'<line class="axis" x1="{x0}" y1="{y0+panel_h}" x2="{x0+panel_w}" y2="{y0+panel_h}"/>')
            lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+panel_h}"/>')
            th_y = y(threshold)
            lines.append(f'<line class="threshold" x1="{x0}" y1="{th_y}" x2="{x0+panel_w}" y2="{th_y}"/>')
            for group, xpos, color in [("calibration", x0 + 85, "#3182bd"), ("formal", x0 + 195, "#de2d26")]:
                mean_val = row[f"{group}_mean"]
                low_val = row[f"{group}_min"]
                high_val = row[f"{group}_max"]
                lines.append(f'<line stroke="{color}" stroke-width="2" x1="{xpos}" y1="{y(low_val):.2f}" x2="{xpos}" y2="{y(high_val):.2f}"/>')
                lines.append(f'<circle fill="{color}" cx="{xpos}" cy="{y(mean_val):.2f}" r="4"/>')
                lines.append(f'<text x="{xpos-28}" y="{y0+panel_h+16}">{group}</text>')
    write_svg(FIGURES / "figure-5-control-transportability.svg", lines)


def generate_png_placeholders() -> None:
    # The publication-quality figures are SVG. Placeholder text files make the
    # dependency choice explicit instead of pretending PNG previews exist.
    (FIGURES / "README.txt").write_text("Phase 7A.1 generated SVG figures. PNG previews were not generated; SciPy/NumPy/Pandas were used for analysis statistics only.\n", encoding="utf-8")


def write_reports(
    seal: dict[str, Any],
    spec: dict[str, Any],
    run_rows: list[dict[str, Any]],
    condition_summary: list[dict[str, Any]],
    slo_violations: list[dict[str, Any]],
    isolated: list[dict[str, Any]],
    direct: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    queue_signatures: list[dict[str, Any]],
    control_transportability: list[dict[str, Any]],
    control_crossings: list[dict[str, Any]],
    statistical_diff: list[dict[str, Any]],
) -> None:
    baseline_crossings = [
        {
            "run_id": row["run_id"],
            "condition_id": row["condition_id"],
            "ttft": row["ttft_slo_violated"],
            "total": row["total_latency_slo_violated"],
            "decode": row["decode_duration_slo_violated"],
        }
        for row in run_rows
        if row["condition_id"] in {"BASELINE", "OUTPUT_CONTROL"}
        and (row["ttft_slo_violated"] or row["total_latency_slo_violated"] or row["decode_duration_slo_violated"])
    ]
    report = {
        "schema_version": "sloscope.phase7a1.report.v1",
        "campaign_id": CAMPAIGN_ID,
        "source_tree_sha256": SOURCE_SHA,
        "campaign_freeze_sha256": FREEZE_SHA,
        "original_phase7a_input_sha256": read_json(ROOT / "analysis" / "phase7a" / "input-seal.json").get("phase7a_input_sha256"),
        "original_phase7a_analysis_spec_sha256": read_json(ROOT / "analysis" / "phase7a" / "analysis-spec.json").get("phase7a_analysis_spec_sha256"),
        "phase7a1_input_sha256": seal["phase7a1_input_sha256"],
        "phase7a1_analysis_spec_sha256": spec["phase7a1_analysis_spec_sha256"],
        "analyzed_runs": len(run_rows),
        "run_level_request_count": sum(row["measured_request_count"] for row in run_rows),
        "successful_requests": sum(row["successful_requests"] for row in run_rows),
        "trace_rows": sum(row["trace_row_count"] for row in run_rows),
        "slo_thresholds": SLO_THRESHOLDS,
        "slo_violation_summary_by_condition": slo_violations,
        "isolated_effects": isolated,
        "direct_evidence": direct,
        "queue_signatures": queue_signatures,
        "factorial_interactions": interactions,
        "control_transportability": control_transportability,
        "control_slo_crossings": control_crossings,
        "statistical_diff_audit": statistical_diff,
        "formal_baseline_control_slo_crossings": baseline_crossings,
        "rq1_summary": {
            "input": "Direct prompt-token increase, matched-control TTFT effect, and absolute frozen-SLO crossings are reported separately.",
            "output": "The output mechanism is interpreted primarily by OUTPUT_MEDIUM vs OUTPUT_CONTROL matched effects because formal OUTPUT_CONTROL itself crosses frozen thresholds.",
            "load": "Corrected long-form runtime queue extraction is used where available; queue metrics are not inferred from latency.",
            "downstream": "Dependency-duration evidence is reported separately from total-latency SLO effects.",
        },
        "rq2_summary": {
            "scope": "Five frozen pairwise factorial families; INPUT_OUTPUT deferred.",
            "primary_interpretation": "Matched factorial contrasts are primary interaction evidence; absolute compound-SLO frequencies are secondary descriptive evidence.",
            "interaction_classifications": Counter(row["classification"] for row in interactions),
        },
    }
    write_json(OUT / "phase7a1-report.json", report)
    lines = [
        "# Phase 7A.1 Corrected Primary Analysis Report",
        "",
        f"campaign_id: `{CAMPAIGN_ID}`",
        f"phase7a1_input_sha256: `{seal['phase7a1_input_sha256']}`",
        f"phase7a1_analysis_spec_sha256: `{spec['phase7a1_analysis_spec_sha256']}`",
        f"analyzed_runs: {len(run_rows)}",
        "",
        "## SLO Thresholds",
        "",
        f"- TTFT_SLO: {SLO_THRESHOLDS['TTFT_SLO']} s",
        f"- TOTAL_LATENCY_SLO: {SLO_THRESHOLDS['TOTAL_LATENCY_SLO']} s",
        f"- DECODE_DURATION_SLO: {SLO_THRESHOLDS['DECODE_DURATION_SLO']} s",
        "- Throughput: diagnostic only",
        "",
        "## RQ1",
        "",
        "Direct mechanism evidence, matched-control metric effects, and absolute frozen-SLO crossings are reported separately. For OUTPUT, matched-control evidence is primary because formal OUTPUT_CONTROL crossed frozen SLO thresholds.",
        "",
        "## RQ2",
        "",
        "Matched factorial interaction contrasts are the primary RQ2 interaction evidence. Frozen compound-SLO frequencies are secondary descriptive evidence. No CSD/RCA analysis is performed here.",
        "",
        "## Baseline/Control SLO Crossings",
        "",
        f"{len(baseline_crossings)} baseline/control runs crossed at least one counted SLO.",
        "",
        "## Queue Metrics",
        "",
        "Runtime queue metrics are extracted from long-form `metric_name`/`metric_value` rows for `llamacpp:requests_processing` and `llamacpp:requests_deferred`. Missing long-form metrics remain missing and are not inferred from latency.",
        "",
        "## Limitations",
        "",
        "- Analysis uses run/repetition as the inferential unit (n=8 per condition).",
        "- SciPy is used for Student-t confidence intervals and tests in this corrected analysis.",
        "- Control transportability changed for OUTPUT_CONTROL; frozen SLO thresholds were not recalibrated post hoc.",
        "- Server process CPU/RSS fields are null in the formal artifacts because server PID telemetry was not supplied in the formal configs; host CPU telemetry remains available.",
        "",
    ]
    (OUT / "phase7a1-report.md").write_text("\n".join(lines), encoding="utf-8")


def write_analysis_provenance(seal: dict[str, Any], spec: dict[str, Any], analyzed_rows: list[dict[str, Any]]) -> None:
    script = Path(__file__)
    provenance = {
        "schema_version": "sloscope.phase7a1.provenance.v1",
        "timestamp": utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "source_tree_sha256": SOURCE_SHA,
        "campaign_freeze_sha256": FREEZE_SHA,
        "original_phase7a_input_sha256": read_json(ROOT / "analysis" / "phase7a" / "input-seal.json").get("phase7a_input_sha256"),
        "original_phase7a_analysis_spec_sha256": read_json(ROOT / "analysis" / "phase7a" / "analysis-spec.json").get("phase7a_analysis_spec_sha256"),
        "phase7a1_input_sha256": seal["phase7a1_input_sha256"],
        "phase7a1_analysis_spec_sha256": spec["phase7a1_analysis_spec_sha256"],
        "publication_run_index_sha256": seal["publication_run_index_sha256"],
        "python_version": sys.version,
        "platform": platform.platform(),
        "analysis_dependency_versions": {
            "python_stdlib": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy_stats.__version__ if hasattr(scipy_stats, "__version__") else __import__("scipy").__version__,
            "matplotlib": None,
            "pyarrow": __import__("pyarrow").__version__,
        },
        "analysis_script_hashes": {
            str(script.relative_to(ROOT)): sha256_file(script),
            "analysis/phase7a1/test_phase7a1_analysis.py": sha256_file(ROOT / "analysis" / "phase7a1" / "test_phase7a1_analysis.py")
            if (ROOT / "analysis" / "phase7a1" / "test_phase7a1_analysis.py").exists()
            else None,
        },
        "analyzed_run_ids": [row["run_id"] for row in analyzed_rows],
    }
    write_json(OUT / "analysis-provenance.json", provenance)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    index_rows = load_publication_index()
    seal = create_input_seal(index_rows)
    spec = create_analysis_spec()
    run_rows = [aggregate_run(row) for row in index_rows]
    write_csv(OUT / "run-level-metrics.csv", run_rows)
    slo_rows = [
        {
            "run_id": row["run_id"],
            "condition_id": row["condition_id"],
            "repetition": row["repetition"],
            "ttft_slo_violated": row["ttft_slo_violated"],
            "total_latency_slo_violated": row["total_latency_slo_violated"],
            "decode_duration_slo_violated": row["decode_duration_slo_violated"],
            "slo_violation_count": row["slo_violation_count"],
            "slo_violation_class": row["slo_violation_class"],
        }
        for row in run_rows
    ]
    write_csv(OUT / "slo-violations.csv", slo_rows)
    condition_summary, condition_violations = condition_summaries(run_rows)
    write_csv(OUT / "condition-summary.csv", condition_summary)
    write_csv(OUT / "condition-slo-summary.csv", condition_violations)
    isolated = isolated_effects(run_rows)
    write_csv(OUT / "isolated-effects.csv", isolated)
    direct = direct_evidence_rows(run_rows)
    write_csv(OUT / "direct-evidence.csv", direct)
    queue_signatures = queue_signature_rows(run_rows)
    write_csv(OUT / "queue-signatures.csv", queue_signatures)
    marginals, interactions = factorial_effects(run_rows)
    write_csv(OUT / "factorial-marginal-effects.csv", marginals)
    write_csv(OUT / "factorial-interactions.csv", interactions)
    statistical_diff = statistical_diff_audit(isolated, interactions)
    write_csv(OUT / "statistical-diff-audit.csv", statistical_diff)
    control_transportability = control_transportability_rows(run_rows)
    write_csv(OUT / "control-transportability.csv", control_transportability)
    control_crossings = control_slo_crossing_rows(run_rows)
    write_csv(OUT / "control-slo-crossings.csv", control_crossings)
    holm_rows = []
    for row in isolated:
        holm_rows.append({"family": "isolated", "test_id": row["mechanism"], "raw_p": row["raw_p"], "holm_p": row["holm_p"]})
    for row in interactions:
        holm_rows.append({"family": "factorial_interactions", "test_id": row["test_id"], "raw_p": row["raw_p"], "holm_p": row["holm_p"]})
    write_csv(OUT / "holm-adjustments.csv", holm_rows)
    table1, table2, table3 = table_outputs(condition_summary, condition_violations, isolated, interactions)
    write_csv(OUT / "table-1-condition-summary.csv", table1)
    write_csv(OUT / "table-2-isolated-effects.csv", table2)
    write_csv(OUT / "table-3-factorial-interactions.csv", table3)
    figure_1(run_rows)
    figure_2(isolated)
    figure_3(condition_violations)
    figure_4(interactions)
    figure_5_control_transportability(control_transportability)
    generate_png_placeholders()
    write_reports(
        seal,
        spec,
        run_rows,
        condition_summary,
        condition_violations,
        isolated,
        direct,
        interactions,
        queue_signatures,
        control_transportability,
        control_crossings,
        statistical_diff,
    )
    write_analysis_provenance(seal, spec, run_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
