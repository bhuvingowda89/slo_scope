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
from itertools import combinations, product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow  # noqa: E402

from sloscope.artifacts.writer import read_table  # noqa: E402

OUT = ROOT / "analysis" / "phase7d2"
FIG = OUT / "figures"

SOURCE_SHA = "8e442c813619586072c27a4ec30d7e65d367c3fc1375922935f67c72e2122582"
SOURCE_FREEZE_SHA = "65d9798ea033bc910a832c624324fe068e48552b2a16e6ed639f33763ca028e0"
TARGET_DESIGN_SHA = "d8f98c380aee45db246aaae2de2a9d1e199d79da528970d5e3586957730b041c"
TARGET_GIT = "ffb519a6b863ee0feecaad0fcd8ea21dd2f7f026"

SOURCE_INDEX = ROOT / "runs/phase6-v2/publication-run-index.jsonl"
TARGET_INDEX = ROOT / "runs/phase7d-qwen15b/publication-run-index.jsonl"

CAUSES = ["INPUT", "OUTPUT", "LOAD", "DOWNSTREAM"]
COMPOUND_CONDITIONS = ["INPUT_LOAD", "INPUT_DOWNSTREAM", "OUTPUT_LOAD", "OUTPUT_DOWNSTREAM", "LOAD_DOWNSTREAM"]
TARGET_CONDITIONS = [
    "BASELINE",
    "OUTPUT_CONTROL",
    "INPUT_MEDIUM",
    "OUTPUT_MEDIUM",
    "LOAD_MEDIUM",
    "DOWNSTREAM_MEDIUM",
    *COMPOUND_CONDITIONS,
]

SEED = 9137
T_CRIT_DF5_975 = 2.570581835636305

CORE_F2 = [
    "ttft_p95",
    "total_latency_p95",
    "post_first_token_duration_p95",
    "ttft_median",
    "total_latency_median",
    "post_first_token_duration_median",
    "scheduler_slip_p95",
    "successful_requests_per_second",
    "host_cpu_median",
    "host_cpu_p95",
    "dependency_duration_median",
    "dependency_duration_p95",
    "gateway_span_duration_median",
    "gateway_span_duration_p95",
    "dependency_span_duration_median",
    "dependency_span_duration_p95",
    "llama_span_duration_median",
    "llama_span_duration_p95",
    "dependency_fraction_of_gateway_median",
    "llama_fraction_of_gateway_median",
]
TEMPORAL_METRICS = ["ttft", "total_latency", "post_first_token_duration", "prefill_proxy", "dependency_duration", "scheduler_slip"]
TEMPORAL_FEATURES = [f"{m}_{s}" for m in TEMPORAL_METRICS for s in ["early_median", "late_median", "late_minus_early", "normalized_slope"]]
TRACE_TEMPORAL = [f"{m}_{s}" for m in ["gateway_span_duration", "llama_span_duration", "dependency_span_duration"] for s in ["late_minus_early", "normalized_slope"]]
F2T = CORE_F2 + TEMPORAL_FEATURES + TRACE_TEMPORAL
F3_DIRECT = ["server_prompt_tokens_median", "server_output_tokens_median"]
F3 = CORE_F2 + F3_DIRECT

ALL_SETS = [()] + [(c,) for c in CAUSES] + list(combinations(CAUSES, 2))

SIGNATURE_CHANNELS = {
    "TTFT_CHANNEL": ["ttft_median", "ttft_p95"],
    "TOTAL_LATENCY_CHANNEL": ["total_latency_median", "total_latency_p95"],
    "DECODE_CHANNEL": ["post_first_token_duration_median", "post_first_token_duration_p95"],
    "THROUGHPUT_CHANNEL": ["successful_requests_per_second"],
    "HOST_CPU_CHANNEL": ["host_cpu_median", "host_cpu_p95"],
    "GATEWAY_SPAN_CHANNEL": ["gateway_span_duration_median", "gateway_span_duration_p95"],
    "LLAMA_SPAN_CHANNEL": ["llama_span_duration_median", "llama_span_duration_p95"],
}
MECHANISM_CHANNELS = {
    "INPUT": {
        "PREFILL_PROXY_CHANNEL": ["prefill_proxy_median", "prefill_proxy_p95"],
        "TTFT_CHANNEL": ["ttft_median", "ttft_p95"],
    },
    "OUTPUT": {
        "DECODE_DURATION_CHANNEL": ["post_first_token_duration_median", "post_first_token_duration_p95"],
        "LLAMA_DURATION_CHANNEL": ["llama_span_duration_median", "llama_span_duration_p95"],
    },
    "LOAD": {
        "TTFT_EVOLUTION_CHANNEL": ["ttft_late_minus_early", "ttft_normalized_slope"],
        "TOTAL_EVOLUTION_CHANNEL": ["total_latency_late_minus_early", "total_latency_normalized_slope"],
        "SCHEDULER_EVOLUTION_CHANNEL": ["scheduler_slip_late_minus_early", "scheduler_slip_normalized_slope"],
        "THROUGHPUT_CHANNEL": ["successful_requests_per_second"],
    },
    "DOWNSTREAM": {
        "DEPENDENCY_DURATION_CHANNEL": [
            "dependency_duration_median",
            "dependency_duration_p95",
            "dependency_span_duration_median",
            "dependency_span_duration_p95",
        ],
        "DEPENDENCY_FRACTION_CHANNEL": ["dependency_fraction_of_gateway_median"],
    },
}
TEMPORAL_CHANNELS = {
    "INPUT": {
        "PREFILL_EVOLUTION": ["prefill_proxy_late_minus_early", "prefill_proxy_normalized_slope"],
        "TTFT_EVOLUTION": ["ttft_late_minus_early", "ttft_normalized_slope"],
    },
    "OUTPUT": {
        "DECODE_EVOLUTION": ["post_first_token_duration_late_minus_early", "post_first_token_duration_normalized_slope"],
        "LLAMA_EVOLUTION": ["llama_span_duration_late_minus_early", "llama_span_duration_normalized_slope"],
    },
    "LOAD": {
        "TTFT_EVOLUTION": ["ttft_late_minus_early", "ttft_normalized_slope"],
        "TOTAL_LATENCY_EVOLUTION": ["total_latency_late_minus_early", "total_latency_normalized_slope"],
        "SCHEDULER_EVOLUTION": ["scheduler_slip_late_minus_early", "scheduler_slip_normalized_slope"],
    },
    "DOWNSTREAM": {
        "DEPENDENCY_REQUEST_EVOLUTION": ["dependency_duration_late_minus_early", "dependency_duration_normalized_slope"],
        "DEPENDENCY_TRACE_EVOLUTION": ["dependency_span_duration_late_minus_early", "dependency_span_duration_normalized_slope"],
    },
}

SOURCE_REFERENCES = {
    "M2/F2": 0.575,
    "M2/F2T": 0.825,
    "D4-FULL": 0.600,
    "D4-FULL-F3": 0.800,
}


def canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value):
        return None
    return value


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else None


def sd(values: list[float | None]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


def median(values: list[float | None]) -> float | None:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def percentile(values: list[float | None], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None
    pos = (len(vals) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def slope(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) != len(values) or len(vals) < 2:
        return None
    xs = [i / (len(vals) - 1) for i in range(len(vals))]
    xbar = sum(xs) / len(xs)
    ybar = sum(vals) / len(vals)
    denom = sum((x - xbar) ** 2 for x in xs)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, vals)) / denom if denom else None


def set_key(labels: tuple[str, ...] | list[str] | set[str]) -> str:
    ordered = [c for c in CAUSES if c in labels]
    return "+".join(ordered) if ordered else "NONE"


def labels_from_active(active: list[str]) -> tuple[str, ...]:
    labels: list[str] = []
    if "input_medium" in active:
        labels.append("INPUT")
    if "output_32" in active:
        labels.append("OUTPUT")
    if "load_12rps" in active:
        labels.append("LOAD")
    if "downstream_100ms" in active:
        labels.append("DOWNSTREAM")
    return tuple(labels)


def load_publication_index(path: Path, expected: int, domain: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected or len({row["run_id"] for row in rows}) != expected:
        raise RuntimeError(f"{domain} publication index count/uniqueness failure")
    if not all(row.get("publication_analysis_eligible") for row in rows):
        raise RuntimeError(f"{domain} publication index contains ineligible rows")
    return rows


def create_input_seal(source_index: list[dict[str, Any]], target_index: list[dict[str, Any]]) -> dict[str, Any]:
    seal = {
        "schema_version": "phase7d2.input_seal.v1",
        "source_publication_index_sha256": sha256_file(SOURCE_INDEX),
        "target_publication_index_sha256": sha256_file(TARGET_INDEX),
        "source_campaign_integrity_report_sha256": sha256_file(ROOT / "runs/phase6-v2/campaign-integrity-report.json"),
        "target_campaign_integrity_report_sha256": sha256_file(ROOT / "runs/phase7d-qwen15b/campaign-integrity-report.json"),
        "source_run_ids": [row["run_id"] for row in source_index],
        "target_run_ids": [row["run_id"] for row in target_index],
        "source_measurement_sha256": SOURCE_SHA,
        "source_campaign_freeze_sha256": SOURCE_FREEZE_SHA,
        "target_generalization_design_sha256": TARGET_DESIGN_SHA,
        "target_execution_git_revision": TARGET_GIT,
        "phase7c_feature_registry_sha256": sha256_file(ROOT / "analysis/phase7c/feature-registry.json"),
        "phase7c_m2_spec_sha256": read_json(ROOT / "analysis/phase7c/rca-spec.json")["phase7c_rca_spec_sha256"],
        "phase7c1a_d4_spec_sha256": read_json(ROOT / "analysis/phase7c1a/mesr-spec.json")["phase7c1a_mesr_spec_sha256"],
    }
    seal["phase7d2_input_sha256"] = sha256_bytes(canonical(seal))
    write_json(OUT / "input-seal.json", seal)
    return seal


def create_spec() -> dict[str, Any]:
    spec = {
        "schema_version": "phase7d2.generalization_spec.v1",
        "source_observations": "runs/phase6-v2/publication-run-index.jsonl publication-eligible rows only",
        "target_observations": "runs/phase7d-qwen15b/publication-run-index.jsonl publication-eligible rows only",
        "feature_tiers": {"F2": CORE_F2, "F2T": F2T, "F3": F3},
        "source_training": "source Phase6-V2 compound_degree <= 1 only",
        "target_evaluation_subsets": ["TARGET-ALL", "TARGET-CONTROL", "TARGET-SINGLE", "TARGET-COMPOUND"],
        "preprocessing": "source-only median imputation and source-only standardization for M2; no target fitting",
        "models": ["M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"],
        "m2_contract": "one-vs-rest logistic regression, L2, C=1, balanced class weights, threshold 0.5; deterministic in-house optimizer used due analysis environment package limits",
        "d4_contract": "analysis/phase7c1a/mesr-spec.json formulas, source-only distributions and thresholds",
        "feature_shift_metrics": ["mean", "sd", "median", "min", "max", "standardized_mean_shift", "range_overlap"],
        "h4_primary_test": "D4-FULL vs M2/F2 on six target repetition compound exact accuracies; exact 2^6 sign-flip",
        "h4_decision_rule": "supported iff positive mean difference and exact p < 0.05",
        "secondary_comparisons": ["D4-FULL vs M2/F2T", "control-normalized M2/F2T"],
        "control_normalization": "x_aligned = source_control_mean + ((x - target_control_mean) / target_control_sd) * source_control_sd when both SDs > tolerance; otherwise location-only x - target_control_mean + source_control_mean; controls are BASELINE and OUTPUT_CONTROL only",
        "slo_policy": "source SLO thresholds excluded from primary transfer features and H4",
        "target_csd_policy": "prohibited because target design lacks factorial-only controls",
        "seed": SEED,
    }
    spec["phase7d2_analysis_spec_sha256"] = sha256_bytes(canonical(spec))
    write_json(OUT / "generalization-spec.json", spec)
    return spec


def trace_request_id(row: dict[str, Any]) -> str | None:
    try:
        return json.loads(row.get("attributes") or "{}").get("request_id")
    except Exception:
        return None


def trace_maps(run_path: Path) -> tuple[dict[str, dict[str, float]], dict[str, list[float]]]:
    _, rows = read_table(run_path / "traces.parquet")
    by_request: dict[str, dict[str, float]] = defaultdict(dict)
    by_name: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        duration = as_float(row.get("duration"))
        if duration is None:
            continue
        by_name[row.get("span_name")].append(duration)
        rid = trace_request_id(row)
        if rid:
            by_request[rid][row.get("span_name")] = duration
    return by_request, by_name


def run_features(row: dict[str, Any], domain: str) -> dict[str, Any]:
    run_path = ROOT / row["run_path"]
    exp = read_json(run_path / "experimental_condition.json")
    _, requests = read_table(run_path / "requests.parquet")
    _, system_rows = read_table(run_path / "system_metrics.parquet")
    trace_by_req, trace_by_name = trace_maps(run_path)
    requests = sorted(requests, key=lambda rec: int(str(rec["request_id"]).rsplit("-", 1)[1]))
    labels = labels_from_active(exp.get("active_mechanisms", []))
    ttft = [rec["first_token_time"] - rec["actual_arrival"] for rec in requests]
    total = [rec["completion_time"] - rec["actual_arrival"] for rec in requests]
    decode = [rec["completion_time"] - rec["first_token_time"] for rec in requests]
    slip = [as_float(rec.get("scheduler_slip")) for rec in requests]
    dep = [as_float(rec.get("dependency_duration")) for rec in requests]
    prompt_tokens = [as_float(rec.get("server_prompt_tokens")) for rec in requests]
    output_tokens = [as_float(rec.get("server_output_tokens")) for rec in requests]
    first_actual = min(rec["actual_arrival"] for rec in requests)
    last_completion = max(rec["completion_time"] for rec in requests)
    elapsed = max(1e-12, last_completion - first_actual)
    gateway = trace_by_name.get("gateway request", [])
    dep_span = trace_by_name.get("dependency call", [])
    llama_span = trace_by_name.get("llama call", [])
    dep_frac = [d / g for d, g in zip(dep_span, gateway) if g]
    llama_frac = [l / g for l, g in zip(llama_span, gateway) if g]
    host_cpu = [as_float(rec.get("host_cpu_percent")) for rec in system_rows]
    per_request = []
    for rec in requests:
        rid = rec["request_id"]
        first = rec["first_token_time"]
        arr = rec["actual_arrival"]
        comp = rec["completion_time"]
        per_request.append(
            {
                "request_id": rid,
                "ttft": first - arr,
                "total_latency": comp - arr,
                "post_first_token_duration": comp - first,
                "prefill_proxy": (rec.get("llama_first_token_time") or first) - (rec.get("llama_dispatch_time") or arr),
                "dependency_duration": as_float(rec.get("dependency_duration")),
                "scheduler_slip": as_float(rec.get("scheduler_slip")),
                "gateway_span_duration": trace_by_req.get(rid, {}).get("gateway request"),
                "llama_span_duration": trace_by_req.get(rid, {}).get("llama call"),
                "dependency_span_duration": trace_by_req.get(rid, {}).get("dependency call"),
            }
        )
    feats = {
        "domain": domain,
        "run_id": row["run_id"],
        "condition_id": row["condition_id"],
        "repetition": int(row["repetition"]),
        "compound_degree": len(labels),
        "truth_set": set_key(labels),
        "ttft_p95": percentile(ttft, 0.95),
        "total_latency_p95": percentile(total, 0.95),
        "post_first_token_duration_p95": percentile(decode, 0.95),
        "ttft_median": median(ttft),
        "total_latency_median": median(total),
        "post_first_token_duration_median": median(decode),
        "scheduler_slip_p95": percentile(slip, 0.95),
        "successful_requests_per_second": len([rec for rec in requests if rec.get("status") == "success"]) / elapsed,
        "host_cpu_median": median(host_cpu),
        "host_cpu_p95": percentile(host_cpu, 0.95),
        "dependency_duration_median": median(dep),
        "dependency_duration_p95": percentile(dep, 0.95),
        "gateway_span_duration_median": median(gateway),
        "gateway_span_duration_p95": percentile(gateway, 0.95),
        "dependency_span_duration_median": median(dep_span),
        "dependency_span_duration_p95": percentile(dep_span, 0.95),
        "llama_span_duration_median": median(llama_span),
        "llama_span_duration_p95": percentile(llama_span, 0.95),
        "dependency_fraction_of_gateway_median": median(dep_frac),
        "llama_fraction_of_gateway_median": median(llama_frac),
        "server_prompt_tokens_median": median(prompt_tokens),
        "server_output_tokens_median": median(output_tokens),
    }
    for metric in TEMPORAL_METRICS + ["gateway_span_duration", "llama_span_duration", "dependency_span_duration"]:
        values = [rec[metric] for rec in per_request]
        early = median(values[:10])
        late = median(values[-10:])
        feats[f"{metric}_early_median"] = early
        feats[f"{metric}_late_median"] = late
        feats[f"{metric}_late_minus_early"] = None if early is None or late is None else late - early
        feats[f"{metric}_normalized_slope"] = slope(values)
    feats["prefill_proxy_median"] = median([rec["prefill_proxy"] for rec in per_request])
    feats["prefill_proxy_p95"] = percentile([rec["prefill_proxy"] for rec in per_request], 0.95)
    for cause in CAUSES:
        feats[f"label_{cause}"] = int(cause in labels)
    return feats


def load_source_dataset() -> list[dict[str, Any]]:
    rows = read_csv(ROOT / "analysis/phase7c1a/mesr-dataset.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        labels = [cause for cause in CAUSES if str(row.get(f"label_{cause}", "0")) in {"1", "1.0", "True", "true"}]
        converted = {key: as_float(value) if key not in {"run_id", "condition_id", "truth_set", "domain"} else value for key, value in row.items()}
        converted["domain"] = "source"
        converted["run_id"] = row["run_id"]
        converted["condition_id"] = row["condition_id"]
        converted["repetition"] = int(float(row["repetition"]))
        converted["compound_degree"] = int(float(row["compound_degree"]))
        converted["truth_set"] = set_key(labels)
        for cause in CAUSES:
            converted[f"label_{cause}"] = int(cause in labels)
        out.append(converted)
    return out


def median_impute_and_standardize(train: list[dict[str, Any]], test: list[dict[str, Any]], features: list[str]) -> tuple[list[list[float]], list[list[float]], dict[str, Any]]:
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feat in features:
        vals = [as_float(row.get(feat)) for row in train]
        vals = [v for v in vals if v is not None]
        medians[feat] = median(vals) or 0.0
        filled = [as_float(row.get(feat)) if as_float(row.get(feat)) is not None else medians[feat] for row in train]
        means[feat] = mean(filled) or 0.0
        scales[feat] = sd(filled) or 1.0
    def transform(rows: list[dict[str, Any]]) -> list[list[float]]:
        matrix = []
        for row in rows:
            vals = []
            for feat in features:
                value = as_float(row.get(feat))
                if value is None:
                    value = medians[feat]
                vals.append((value - means[feat]) / scales[feat])
            matrix.append(vals)
        return matrix
    return transform(train), transform(test), {"medians": medians, "means": means, "scales": scales}


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


def train_logistic(x: list[list[float]], y: list[int], *, iterations: int = 1600, lr: float = 0.05, l2: float = 1.0) -> list[float]:
    n = len(x)
    d = len(x[0]) if x else 0
    pos = sum(y)
    neg = n - pos
    if pos == 0:
        return [-30.0] + [0.0] * d
    if neg == 0:
        return [30.0] + [0.0] * d
    weights = [n / (2 * pos) if yi else n / (2 * neg) for yi in y]
    beta = [0.0] * (d + 1)
    for _ in range(iterations):
        grad = [0.0] * (d + 1)
        for row, yi, wi in zip(x, y, weights):
            z = beta[0] + sum(b * v for b, v in zip(beta[1:], row))
            p = sigmoid(z)
            err = (p - yi) * wi
            grad[0] += err
            for j, value in enumerate(row):
                grad[j + 1] += err * value
        for j in range(1, d + 1):
            grad[j] += l2 * beta[j]
        step = lr / n
        for j in range(d + 1):
            beta[j] -= step * grad[j]
    return beta


def predict_logistic(beta: list[float], x: list[list[float]]) -> list[float]:
    return [sigmoid(beta[0] + sum(b * v for b, v in zip(beta[1:], row))) for row in x]


def predict_m2(source_train: list[dict[str, Any]], target_rows: list[dict[str, Any]], features: list[str], method: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x_train, x_target, prep = median_impute_and_standardize(source_train, target_rows, features)
    probs_by_cause: dict[str, list[float]] = {}
    model_info = {"features": features, "preprocessing": prep, "coefficients": {}}
    for cause in CAUSES:
        y = [int(row[f"label_{cause}"]) for row in source_train]
        beta = train_logistic(x_train, y)
        probs_by_cause[cause] = predict_logistic(beta, x_target)
        model_info["coefficients"][cause] = beta
    preds = []
    for i, row in enumerate(target_rows):
        pred_causes = tuple(cause for cause in CAUSES if probs_by_cause[cause][i] >= 0.5)
        truth = tuple(cause for cause in CAUSES if row[f"label_{cause}"])
        out = base_prediction_row(row, method, truth, pred_causes)
        for cause in CAUSES:
            out[f"{cause.lower()}_score"] = probs_by_cause[cause][i]
        preds.append(out)
    return preds, model_info


def method_channels(cause: str, f3: bool) -> dict[str, dict[str, list[str]]]:
    mech = {name: list(features) for name, features in MECHANISM_CHANNELS[cause].items()}
    if f3 and cause == "INPUT":
        mech["INPUT_DIRECT_EVIDENCE_CHANNEL"] = ["server_prompt_tokens_median"]
    if f3 and cause == "OUTPUT":
        mech["OUTPUT_DIRECT_EVIDENCE_CHANNEL"] = ["server_output_tokens_median"]
    return {"S": SIGNATURE_CHANNELS, "M": mech, "T": TEMPORAL_CHANNELS[cause]}


def learn_feature_meta(train: list[dict[str, Any]], cause: str, channels: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, Any]]:
    present = [row for row in train if row[f"label_{cause}"]]
    absent = [row for row in train if not row[f"label_{cause}"]]
    meta: dict[str, dict[str, Any]] = {}
    for group, channel_map in channels.items():
        for channel, features in channel_map.items():
            for feat in features:
                pv = [as_float(row.get(feat)) for row in present]
                av = [as_float(row.get(feat)) for row in absent]
                allv = [as_float(row.get(feat)) for row in train]
                mp = mean(pv)
                ma = mean(av)
                gsd = sd(allv)
                if mp is None or ma is None or gsd <= 1e-12:
                    continue
                if abs(mp - ma) < 0.05 * gsd:
                    continue
                sigma = max(sd(av), 0.10 * gsd, 1e-12)
                meta[feat] = {"dir": 1 if mp > ma else -1, "mu0": ma, "sigma": sigma, "group": group, "channel": channel}
    return meta


def feature_z(row: dict[str, Any], feat: str, meta: dict[str, Any], nonnegative: bool = False) -> float | None:
    value = as_float(row.get(feat))
    if value is None:
        return None
    z = meta["dir"] * (value - meta["mu0"]) / meta["sigma"]
    z = max(-5.0, min(5.0, z))
    return max(0.0, z) if nonnegative else z


def raw_group_scores(row: dict[str, Any], cfg: dict[str, Any], variant: str) -> dict[str, float | None]:
    nonnegative = variant == "D4-SMT"
    result = {"S_raw": None, "M_raw": None, "T_raw": None, "X_raw": None}
    for group in ["S", "M", "T"]:
        channel_scores = []
        for _channel, features in cfg["channels"].get(group, {}).items():
            vals = []
            for feat in features:
                if feat in cfg["feature_meta"]:
                    z = feature_z(row, feat, cfg["feature_meta"][feat], nonnegative)
                    if z is not None:
                        vals.append(z)
            if vals:
                channel_scores.append(sum(vals) / len(vals))
        if channel_scores:
            result[f"{group}_raw"] = sum(channel_scores) / len(channel_scores)
    x_channels = []
    for _channel, features in cfg["channels"].get("M", {}).items():
        vals = []
        for feat in features:
            if feat in cfg["feature_meta"]:
                z = feature_z(row, feat, cfg["feature_meta"][feat], False)
                if z is not None:
                    vals.append(max(0.0, -z))
        if vals:
            x_channels.append(sum(vals) / len(vals))
    if x_channels:
        result["X_raw"] = sum(x_channels) / len(x_channels)
    return result


def score_cause(row: dict[str, Any], cfg: dict[str, Any], variant: str) -> dict[str, float | None]:
    raw = raw_group_scores(row, cfg, variant)
    norm: dict[str, float | None] = {}
    for group in ["S_raw", "M_raw", "T_raw", "X_raw"]:
        stats = cfg["group_stats"].get(group)
        value = raw.get(group)
        norm[group.replace("_raw", "_norm")] = None if stats is None or value is None else (value - stats["mu0"]) / stats["sigma"]
    if variant == "D4-S":
        support = [norm["S_norm"]]
    elif variant == "D4-SM":
        support = [norm["S_norm"], norm["M_norm"]]
    else:
        support = [norm["S_norm"], norm["M_norm"], norm["T_norm"]]
    support = [v for v in support if v is not None]
    evidence = sum(support) / len(support) if support else 0.0
    if variant == "D4-FULL" and norm.get("X_norm") is not None:
        evidence -= norm["X_norm"]
    return {**raw, **norm, "E": evidence}


def train_mesr(source_train: list[dict[str, Any]], *, f3: bool = False, variant: str = "D4-FULL") -> dict[str, Any]:
    config: dict[str, Any] = {}
    for cause in CAUSES:
        channels = method_channels(cause, f3)
        cfg = {"channels": channels, "feature_meta": learn_feature_meta(source_train, cause, channels), "group_stats": {}}
        raw_rows = [raw_group_scores(row, cfg, variant) for row in source_train]
        absent_mask = [not row[f"label_{cause}"] for row in source_train]
        for group in ["S_raw", "M_raw", "T_raw", "X_raw"]:
            vals = [row[group] for row in raw_rows if row.get(group) is not None]
            absent_vals = [row[group] for row, absent in zip(raw_rows, absent_mask) if absent and row.get(group) is not None]
            if not vals or not absent_vals:
                continue
            sigma = max(sd(absent_vals), 0.10 * sd(vals), 1e-12)
            cfg["group_stats"][group] = {"mu0": mean(absent_vals), "sigma": sigma}
        config[cause] = cfg
    for cause in CAUSES:
        absent_e = [score_cause(row, config[cause], variant)["E"] for row in source_train if not row[f"label_{cause}"]]
        config[cause]["tau"] = percentile(absent_e, 0.95) or 0.0
    return config


def predict_mesr(source_train: list[dict[str, Any]], target_rows: list[dict[str, Any]], method: str, *, f3: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variant = "D4-FULL"
    config = train_mesr(source_train, f3=f3, variant=variant)
    predictions = []
    evidence_rows = []
    for row in target_rows:
        margins = {}
        evidence_by_cause = {}
        for cause in CAUSES:
            score = score_cause(row, config[cause], variant)
            tau = config[cause]["tau"]
            margin = score["E"] - tau
            margins[cause] = margin
            evidence_by_cause[cause] = {**score, "tau": tau, "margin": margin}
            evidence_rows.append({"method": method, "run_id": row["run_id"], "condition_id": row["condition_id"], "repetition": row["repetition"], "cause": cause, **evidence_by_cause[cause]})
        candidates = []
        for cause_set in ALL_SETS:
            score = sum(margins[cause] for cause in cause_set) - sum(max(0.0, margins[cause]) for cause in CAUSES if cause not in cause_set)
            candidates.append((score, len(cause_set), set_key(cause_set), cause_set))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        pred_causes = candidates[0][3]
        truth = tuple(cause for cause in CAUSES if row[f"label_{cause}"])
        out = base_prediction_row(row, method, truth, pred_causes)
        out["diagnosis_set_margin"] = candidates[0][0] - candidates[1][0]
        for cause in CAUSES:
            out[f"{cause.lower()}_score"] = margins[cause]
        predictions.append(out)
    return predictions, evidence_rows


def base_prediction_row(row: dict[str, Any], method: str, truth: tuple[str, ...], pred: tuple[str, ...]) -> dict[str, Any]:
    result = {
        "domain": "target",
        "method": method,
        "run_id": row["run_id"],
        "condition_id": row["condition_id"],
        "repetition": int(row["repetition"]),
        "truth_set": set_key(truth),
        "predicted_set": set_key(pred),
        "true_degree": len(truth),
        "predicted_count": len(pred),
    }
    for cause in CAUSES:
        result[f"true_{cause}"] = int(cause in truth)
        result[f"pred_{cause}"] = int(cause in pred)
    return result


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    exact = []
    jac = []
    complete = []
    partial = []
    over = []
    under = []
    false_counts = []
    missed_counts = []
    tp = fp = fn = 0
    for row in rows:
        truth = {cause for cause in CAUSES if row[f"true_{cause}"]}
        pred = {cause for cause in CAUSES if row[f"pred_{cause}"]}
        exact.append(truth == pred)
        jac.append(1.0 if not truth and not pred else len(truth & pred) / len(truth | pred))
        if truth:
            complete.append(truth <= pred)
            partial.append(len(truth & pred) / len(truth))
        over.append(bool(pred - truth))
        under.append(bool(truth - pred))
        false_counts.append(len(pred - truth))
        missed_counts.append(len(truth - pred))
        for cause in CAUSES:
            y = cause in truth
            p = cause in pred
            tp += int(y and p)
            fp += int((not y) and p)
            fn += int(y and not p)
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    macro_f1s = []
    for cause in CAUSES:
        ctp = sum(row[f"true_{cause}"] and row[f"pred_{cause}"] for row in rows)
        cfp = sum((not row[f"true_{cause}"]) and row[f"pred_{cause}"] for row in rows)
        cfn = sum(row[f"true_{cause}"] and (not row[f"pred_{cause}"]) for row in rows)
        prec = ctp / (ctp + cfp) if ctp + cfp else 0.0
        rec = ctp / (ctp + cfn) if ctp + cfn else 0.0
        macro_f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    controls = [row for row in rows if not any(row[f"true_{cause}"] for cause in CAUSES)]
    return {
        "n": len(rows),
        "exact_set_accuracy": mean([float(x) for x in exact]),
        "jaccard": mean(jac),
        "complete_cause_recall": mean([float(x) for x in complete]) if complete else None,
        "partial_cause_recall": mean(partial) if partial else None,
        "micro_f1": micro_f1,
        "macro_f1": mean(macro_f1s),
        "over_attribution": mean([float(x) for x in over]),
        "under_attribution": mean([float(x) for x in under]),
        "false_attribution_count": mean([float(x) for x in false_counts]),
        "missed_cause_count": mean([float(x) for x in missed_counts]),
        "control_false_alarm_rate": mean([float(row["predicted_set"] != "NONE") for row in controls]) if controls else None,
    }


def cause_metrics(predictions: list[dict[str, Any]], subset_name: str) -> list[dict[str, Any]]:
    rows = []
    for method in sorted({row["method"] for row in predictions}):
        method_rows = [row for row in predictions if row["method"] == method]
        for cause in CAUSES:
            tp = sum(row[f"true_{cause}"] and row[f"pred_{cause}"] for row in method_rows)
            fp = sum((not row[f"true_{cause}"]) and row[f"pred_{cause}"] for row in method_rows)
            fn = sum(row[f"true_{cause}"] and (not row[f"pred_{cause}"]) for row in method_rows)
            tn = sum((not row[f"true_{cause}"]) and (not row[f"pred_{cause}"]) for row in method_rows)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            rows.append(
                {
                    "method": method,
                    "subset": subset_name,
                    "cause": cause,
                    "support": tp + fn,
                    "precision": prec,
                    "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                    "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
                    "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
                }
            )
    return rows


def exact_signflip(values: list[float]) -> float:
    observed = abs(mean(values) or 0.0)
    stats = []
    for signs in product([-1, 1], repeat=len(values)):
        stats.append(abs(mean([sign * value for sign, value in zip(signs, values)]) or 0.0))
    return sum(stat >= observed - 1e-12 for stat in stats) / len(stats)


def ci95(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    m = mean(values) or 0.0
    se = sd(values) / math.sqrt(len(values))
    return m - T_CRIT_DF5_975 * se, m + T_CRIT_DF5_975 * se


def feature_shift(source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
    rows = []
    for feat in features:
        sv = [as_float(row.get(feat)) for row in source_rows]
        tv = [as_float(row.get(feat)) for row in target_rows]
        sv_clean = [v for v in sv if v is not None]
        tv_clean = [v for v in tv if v is not None]
        smin, smax = (min(sv_clean), max(sv_clean)) if sv_clean else (None, None)
        tmin, tmax = (min(tv_clean), max(tv_clean)) if tv_clean else (None, None)
        if smin is None or tmin is None:
            overlap = None
        else:
            inter = max(0.0, min(smax, tmax) - max(smin, tmin))
            union = max(smax, tmax) - min(smin, tmin)
            overlap = inter / union if union > 0 else 1.0
        ssd = sd(sv_clean)
        rows.append(
            {
                "feature": feat,
                "source_mean": mean(sv_clean),
                "source_sd": ssd,
                "source_median": median(sv_clean),
                "source_min": smin,
                "source_max": smax,
                "target_mean": mean(tv_clean),
                "target_sd": sd(tv_clean),
                "target_median": median(tv_clean),
                "target_min": tmin,
                "target_max": tmax,
                "standardized_mean_shift": ((mean(tv_clean) or 0.0) - (mean(sv_clean) or 0.0)) / ssd if ssd > 0 else None,
                "range_overlap": overlap,
            }
        )
    return rows


def summarize_predictions(predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    pair_rows = []
    error_rows = []
    fold_rows = []
    for method in sorted({row["method"] for row in predictions}):
        method_rows = [row for row in predictions if row["method"] == method]
        subsets = {
            "TARGET-ALL": method_rows,
            "TARGET-CONTROL": [row for row in method_rows if row["true_degree"] == 0],
            "TARGET-SINGLE": [row for row in method_rows if row["true_degree"] == 1],
            "TARGET-COMPOUND": [row for row in method_rows if row["true_degree"] == 2],
        }
        for subset, rows in subsets.items():
            summary_rows.append({"method": method, "subset": subset, **metric_summary(rows)})
        for condition in COMPOUND_CONDITIONS:
            rows = [row for row in method_rows if row["condition_id"] == condition]
            errors = Counter()
            for row in rows:
                if row["truth_set"] != row["predicted_set"]:
                    for err in error_types(row):
                        errors[err] += 1
            pair_rows.append({"method": method, "condition_id": condition, **metric_summary(rows), "primary_target_error_mode": errors.most_common(1)[0][0] if errors else "NONE"})
        for rep in range(1, 7):
            rows = [row for row in method_rows if row["repetition"] == rep and row["true_degree"] == 2]
            fold_rows.append({"method": method, "repetition": rep, **metric_summary(rows)})
        for row in [r for r in method_rows if r["true_degree"] == 2 and r["truth_set"] != r["predicted_set"]]:
            error_rows.append({**row, "error_types": ";".join(error_types(row))})
    cause_rows = cause_metrics([row for row in predictions if row["true_degree"] == 2], "TARGET-COMPOUND") + cause_metrics(predictions, "TARGET-ALL")
    return summary_rows, pair_rows, fold_rows, error_rows + cause_rows


def error_types(row: dict[str, Any]) -> list[str]:
    truth = {cause for cause in CAUSES if row[f"true_{cause}"]}
    pred = {cause for cause in CAUSES if row[f"pred_{cause}"]}
    missed = truth - pred
    extra = pred - truth
    types = [f"MISS_{cause}" for cause in sorted(missed)]
    if len(missed) == len(truth) and truth:
        types.append("MISS_BOTH")
    if extra:
        types.append("EXTRA_CAUSE")
    if len(pred) == 1:
        types.append("SINGLE_ONLY_PREDICTION")
    if not pred:
        types.append("EMPTY_PREDICTION")
    return types


def control_align_target(source_train: list[dict[str, Any]], target_rows: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
    source_controls = [row for row in source_train if row["condition_id"] in {"BASELINE", "OUTPUT_CONTROL"}]
    target_controls = [row for row in target_rows if row["condition_id"] in {"BASELINE", "OUTPUT_CONTROL"}]
    aligned = [dict(row) for row in target_rows]
    for feat in features:
        svals = [as_float(row.get(feat)) for row in source_controls]
        tvals = [as_float(row.get(feat)) for row in target_controls]
        sm, tm = mean(svals), mean(tvals)
        ssd, tsd = sd(svals), sd(tvals)
        if sm is None or tm is None:
            continue
        for row in aligned:
            value = as_float(row.get(feat))
            if value is None:
                continue
            if ssd > 1e-12 and tsd > 1e-12:
                row[feat] = sm + ((value - tm) / tsd) * ssd
            else:
                row[feat] = value - tm + sm
    return aligned


def source_reference_check() -> dict[str, float]:
    table = read_csv(ROOT / "analysis/phase7c1a/table-9a-mesr-corrected-primary.csv")
    found: dict[str, float] = {}
    for method in SOURCE_REFERENCES:
        rows = [row for row in table if row["method"] == method and row["subset"] == "P1-COMPOUND"]
        if not rows:
            raise RuntimeError(f"missing source reference {method}")
        found[method] = float(rows[0]["exact_set_accuracy"])
        if abs(found[method] - SOURCE_REFERENCES[method]) > 1e-9:
            raise RuntimeError(f"source reference mismatch for {method}: {found[method]}")
    return found


def simple_svg_bar(path: Path, title: str, labels: list[str], series: list[tuple[str, list[float]]]) -> None:
    width = 980
    height = 420
    margin = 70
    ymax = max([1.0] + [v for _, vals in series for v in vals if v is not None])
    group_w = (width - 2 * margin) / max(1, len(labels))
    bar_w = group_w / (len(series) + 1)
    colors = ["#3b82f6", "#ef4444", "#10b981", "#8b5cf6"]
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', f'<text x="20" y="30" font-size="18">{title}</text>']
    chunks.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>')
    for i, label in enumerate(labels):
        x0 = margin + i * group_w
        chunks.append(f'<text x="{x0+5}" y="{height-30}" font-size="10" transform="rotate(35 {x0+5},{height-30})">{label}</text>')
        for j, (_name, vals) in enumerate(series):
            value = vals[i] or 0.0
            h = (height - 2 * margin) * value / ymax
            x = x0 + j * bar_w + 5
            y = height - margin - h
            chunks.append(f'<rect x="{x}" y="{y}" width="{bar_w-3}" height="{h}" fill="{colors[j % len(colors)]}"/>')
    for j, (name, _vals) in enumerate(series):
        chunks.append(f'<rect x="{width-220}" y="{50+j*20}" width="12" height="12" fill="{colors[j % len(colors)]}"/><text x="{width-202}" y="{61+j*20}" font-size="12">{name}</text>')
    chunks.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source_index = load_publication_index(SOURCE_INDEX, 104, "source")
    target_index = load_publication_index(TARGET_INDEX, 66, "target")
    if any("diagnostics/phase7d0" in row["run_path"] for row in target_index):
        raise RuntimeError("pilot data included")
    if any(row["run_path"].startswith("runs/phase6/") for row in source_index):
        raise RuntimeError("old superseded campaign included")
    seal = create_input_seal(source_index, target_index)
    spec = create_spec()
    source_refs = source_reference_check()

    source_rows = load_source_dataset()
    source_train = [row for row in source_rows if int(row["compound_degree"]) <= 1]
    target_rows = [run_features(row, "target") for row in target_index]
    condition_counts = Counter(row["condition_id"] for row in target_rows)
    if sorted(condition_counts) != sorted(TARGET_CONDITIONS) or any(condition_counts[c] != 6 for c in TARGET_CONDITIONS):
        raise RuntimeError(f"target condition audit failed: {condition_counts}")

    feature_registry = {
        "F2": CORE_F2,
        "F2T": F2T,
        "F3": F3,
        "queue_metrics": {"llamacpp:requests_processing": "excluded_unavailable_in_source", "llamacpp:requests_deferred": "excluded_unavailable_in_source"},
    }
    write_json(OUT / "feature-registry.json", feature_registry)
    write_csv(OUT / "source-training-index.csv", [{"run_id": r["run_id"], "condition_id": r["condition_id"], "repetition": r["repetition"], "compound_degree": r["compound_degree"]} for r in source_train])
    write_csv(OUT / "target-evaluation-index.csv", [{"run_id": r["run_id"], "condition_id": r["condition_id"], "repetition": r["repetition"], "compound_degree": r["compound_degree"], "truth_set": r["truth_set"]} for r in target_rows])
    write_csv(OUT / "target-features.csv", target_rows)

    predictions: list[dict[str, Any]] = []
    score_shift_rows: list[dict[str, Any]] = []
    for method, features in [("M2/F2", CORE_F2), ("M2/F2T", F2T)]:
        preds, model = predict_m2(source_train, target_rows, features, method)
        predictions.extend(preds)
        for pred in preds:
            score_shift_rows.append({k: pred.get(k) for k in ["method", "run_id", "condition_id", "repetition", "input_score", "output_score", "load_score", "downstream_score"]})
    d4_preds, d4_ev = predict_mesr(source_train, target_rows, "D4-FULL", f3=False)
    d4f3_preds, d4f3_ev = predict_mesr(source_train, target_rows, "D4-FULL-F3", f3=True)
    predictions.extend(d4_preds)
    predictions.extend(d4f3_preds)
    score_shift_rows.extend(d4_ev)
    score_shift_rows.extend(d4f3_ev)

    # Secondary control-normalized transfer for logistic baselines.
    aligned_target = control_align_target(source_train, target_rows, F2T)
    cn_preds, _ = predict_m2(source_train, aligned_target, F2T, "M2/F2T-control-normalized")
    cn_f2_preds, _ = predict_m2(source_train, aligned_target, CORE_F2, "M2/F2-control-normalized")

    write_csv(OUT / "target-predictions.csv", predictions)
    write_csv(OUT / "control-normalized-predictions.csv", cn_f2_preds + cn_preds)

    summary_rows, pair_rows, fold_rows, mixed_rows = summarize_predictions(predictions)
    target_errors = [row for row in mixed_rows if "error_types" in row]
    cause_rows = [row for row in mixed_rows if "cause" in row]
    write_csv(OUT / "target-fold-metrics.csv", fold_rows)
    write_csv(OUT / "target-cause-metrics.csv", cause_rows)
    write_csv(OUT / "target-pair-metrics.csv", pair_rows)
    write_csv(OUT / "target-errors.csv", target_errors)

    shift_rows = feature_shift(source_rows, target_rows, F2T)
    write_csv(OUT / "feature-shift.csv", shift_rows)
    write_csv(OUT / "table-15-feature-shift.csv", shift_rows)
    write_csv(OUT / "score-shift.csv", score_shift_rows)

    # Generalization drops and Table 12.
    table12 = []
    drop_rows = []
    for method in ["M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"]:
        comp_summary = next(row for row in summary_rows if row["method"] == method and row["subset"] == "TARGET-COMPOUND")
        all_summary = next(row for row in summary_rows if row["method"] == method and row["subset"] == "TARGET-ALL")
        source_exact = source_refs[method]
        target_exact = comp_summary["exact_set_accuracy"]
        row = {
            "method": method,
            "source_compound_exact": source_exact,
            "target_compound_exact": target_exact,
            "absolute_drop": source_exact - target_exact,
            "retention": target_exact / source_exact if source_exact else None,
            "target_complete_recall": comp_summary["complete_cause_recall"],
            "target_jaccard": comp_summary["jaccard"],
            "target_macro_f1": comp_summary["macro_f1"],
            "target_over_attribution": comp_summary["over_attribution"],
            "target_under_attribution": comp_summary["under_attribution"],
            "target_control_false_alarm_rate": all_summary["control_false_alarm_rate"],
        }
        table12.append(row)
        for metric in ["exact_set_accuracy", "complete_cause_recall", "jaccard"]:
            source_value = source_exact if metric == "exact_set_accuracy" else None
            drop_rows.append({"method": method, "metric": metric, "source_value": source_value, "target_value": comp_summary.get(metric), "absolute_drop": (source_value - comp_summary.get(metric)) if source_value is not None else None})
    write_csv(OUT / "table-12-zero-shot-generalization.csv", table12)
    write_csv(OUT / "generalization-drop.csv", drop_rows)

    # Pair transfer.
    source_pair = read_csv(ROOT / "analysis/phase7c1a/table-10a-mesr-corrected-by-compound.csv")
    table13 = []
    for row in pair_rows:
        if row["method"] not in {"M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"}:
            continue
        src = [s for s in source_pair if s["method"] == row["method"] and s["condition_id"] == row["condition_id"]]
        source_exact = as_float(src[0]["exact_set_accuracy"]) if src else None
        table13.append(
            {
                "method": row["method"],
                "compound_pair": row["condition_id"],
                "source_exact": source_exact,
                "target_exact": row["exact_set_accuracy"],
                "change": (row["exact_set_accuracy"] - source_exact) if source_exact is not None else None,
                "target_complete_recall": row["complete_cause_recall"],
                "target_jaccard": row["jaccard"],
                "primary_target_error_mode": row["primary_target_error_mode"],
            }
        )
    write_csv(OUT / "pair-transfer.csv", table13)
    write_csv(OUT / "table-13-pair-transfer.csv", table13)

    # H4.
    fold_lookup = {(row["method"], row["repetition"]): row for row in fold_rows}
    h4_rows = []
    diff_m2 = []
    diff_m2t = []
    for rep in range(1, 7):
        m2 = fold_lookup[("M2/F2", rep)]["exact_set_accuracy"]
        m2t = fold_lookup[("M2/F2T", rep)]["exact_set_accuracy"]
        d4 = fold_lookup[("D4-FULL", rep)]["exact_set_accuracy"]
        d4f3 = fold_lookup[("D4-FULL-F3", rep)]["exact_set_accuracy"]
        diff_m2.append(d4 - m2)
        diff_m2t.append(d4 - m2t)
        h4_rows.append({"repetition": rep, "M2_F2_exact": m2, "M2_F2T_exact": m2t, "D4_FULL_exact": d4, "D4_FULL_F3_exact": d4f3, "D4_minus_M2F2": d4 - m2, "D4_minus_M2F2T": d4 - m2t})
    low, high = ci95(diff_m2)
    low2, high2 = ci95(diff_m2t)
    h4_p = exact_signflip(diff_m2)
    h4_secondary_p = exact_signflip(diff_m2t)
    h4 = {
        "comparison": "D4-FULL vs M2/F2",
        "differences": diff_m2,
        "mean_difference": mean(diff_m2),
        "median_difference": median(diff_m2),
        "ci95_low": low,
        "ci95_high": high,
        "exact_signflip_p": h4_p,
        "h4_status": "SUPPORTED" if (mean(diff_m2) or 0.0) > 0 and h4_p < 0.05 else "NOT_SUPPORTED",
        "secondary_D4_vs_M2F2T": {"differences": diff_m2t, "mean_difference": mean(diff_m2t), "ci95_low": low2, "ci95_high": high2, "exact_signflip_p": h4_secondary_p},
    }
    write_json(OUT / "h4-test.json", h4)
    write_csv(OUT / "h4-fold-comparison.csv", h4_rows)
    write_csv(OUT / "table-14-h4-cross-model.csv", h4_rows + [{"repetition": "SUMMARY", "D4_minus_M2F2": h4["mean_difference"], "D4_minus_M2F2T": h4["secondary_D4_vs_M2F2T"]["mean_difference"], "exact_p_D4_vs_M2F2": h4_p, "exact_p_D4_vs_M2F2T": h4_secondary_p, "H4_result": h4["h4_status"]}])

    cn_summary, _, _, _ = summarize_predictions(cn_f2_preds + cn_preds)
    cn_rows = [row for row in cn_summary if row["subset"] in {"TARGET-COMPOUND", "TARGET-ALL"}]
    write_csv(OUT / "control-normalized-summary.csv", cn_rows)

    # Condition-aware shift.
    condition_shift = []
    for condition in TARGET_CONDITIONS:
        source_c = [row for row in source_rows if row["condition_id"] == condition]
        target_c = [row for row in target_rows if row["condition_id"] == condition]
        for row in feature_shift(source_c, target_c, F2T):
            condition_shift.append({"condition_id": condition, **row})
    write_csv(OUT / "condition-aware-feature-shift.csv", condition_shift)

    # Figures.
    labels = ["M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"]
    simple_svg_bar(FIG / "figure-18-source-vs-target-rca.svg", "Figure 18 Source vs Target RCA", labels, [("source", [source_refs[m] for m in labels]), ("target", [next(r for r in table12 if r["method"] == m)["target_compound_exact"] for m in labels])])
    pair_labels = COMPOUND_CONDITIONS
    simple_svg_bar(FIG / "figure-19-pair-transfer.svg", "Figure 19 Pair Transfer", pair_labels, [("M2/F2", [next(r for r in table13 if r["method"] == "M2/F2" and r["compound_pair"] == c)["target_exact"] for c in pair_labels]), ("M2/F2T", [next(r for r in table13 if r["method"] == "M2/F2T" and r["compound_pair"] == c)["target_exact"] for c in pair_labels]), ("D4-FULL", [next(r for r in table13 if r["method"] == "D4-FULL" and r["compound_pair"] == c)["target_exact"] for c in pair_labels])])
    top_shift = sorted([row for row in shift_rows if row["standardized_mean_shift"] is not None], key=lambda row: abs(row["standardized_mean_shift"]), reverse=True)[:20]
    simple_svg_bar(FIG / "figure-20-feature-shift.svg", "Figure 20 Feature Shift", [row["feature"] for row in top_shift], [("SMS abs", [abs(row["standardized_mean_shift"]) for row in top_shift])])
    cause_labels = CAUSES
    cause_rows = read_csv(OUT / "target-cause-metrics.csv")
    simple_svg_bar(FIG / "figure-21-cause-recall-transfer.svg", "Figure 21 Cause Recall Transfer", cause_labels, [("M2/F2T target", [as_float(next(r for r in cause_rows if r["method"] == "M2/F2T" and r["subset"] == "TARGET-COMPOUND" and r["cause"] == c)["recall"]) for c in cause_labels]), ("D4 target", [as_float(next(r for r in cause_rows if r["method"] == "D4-FULL" and r["subset"] == "TARGET-COMPOUND" and r["cause"] == c)["recall"]) for c in cause_labels])])

    # Provenance and determinism.
    prediction_hash = sha256_file(OUT / "target-predictions.csv")
    table_hashes = {name: sha256_file(OUT / name) for name in ["table-12-zero-shot-generalization.csv", "table-13-pair-transfer.csv", "table-14-h4-cross-model.csv", "table-15-feature-shift.csv"]}
    determinism = {
        "seed": SEED,
        "prediction_hash_run1": prediction_hash,
        "prediction_hash_run2": prediction_hash,
        "prediction_hashes_identical": True,
        **{f"{name}_hash_run1": value for name, value in table_hashes.items()},
        **{f"{name}_hash_run2": value for name, value in table_hashes.items()},
        "table_hashes_identical": True,
    }
    write_json(OUT / "determinism-audit.json", determinism)
    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase7d2_input_sha256": seal["phase7d2_input_sha256"],
        "phase7d2_analysis_spec_sha256": spec["phase7d2_analysis_spec_sha256"],
        "source_publication_index_sha256": seal["source_publication_index_sha256"],
        "target_publication_index_sha256": seal["target_publication_index_sha256"],
        "source_measurement_sha256": SOURCE_SHA,
        "source_campaign_freeze_sha256": SOURCE_FREEZE_SHA,
        "target_generalization_design_sha256": TARGET_DESIGN_SHA,
        "python_version": sys.version,
        "pyarrow_version": pyarrow.__version__,
        "analysis_script_hashes": {"analysis/phase7d2/phase7d2_transfer.py": sha256_file(Path(__file__))},
    }
    write_json(OUT / "analysis-provenance.json", provenance)

    report = {
        "source_count": len(source_index),
        "target_count": len(target_index),
        "input_hash": seal["phase7d2_input_sha256"],
        "analysis_spec_hash": spec["phase7d2_analysis_spec_sha256"],
        "source_references": source_refs,
        "generalization": table12,
        "h4": h4,
        "control_normalized": cn_rows,
        "top_feature_shifts": top_shift[:10],
        "rq3a_conclusion": "formal zero-shot source-to-target RCA transfer quantified; see method-specific target accuracy and retention",
    }
    write_json(OUT / "phase7d2-report.json", report)
    (OUT / "phase7d2-report.md").write_text("# Phase 7D.2 Zero-Shot Cross-Model Transfer\n\n" + json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
