from __future__ import annotations

import csv
import hashlib
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
import sklearn
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sloscope.artifacts.writer import read_table

PHASE7C1A = ROOT / "analysis" / "phase7c1a"
if str(PHASE7C1A) not in sys.path:
    sys.path.insert(0, str(PHASE7C1A))
import phase7c1a_mesr as mesr

OUT = ROOT / "analysis" / "phase7e"
FIG = OUT / "figures"

CAUSES = ["INPUT", "OUTPUT", "LOAD", "DOWNSTREAM"]
METHODS = ["M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"]
PRIMARY_METHODS = ["M2/F2", "M2/F2T", "D4-FULL"]
COMPOUNDS = ["INPUT_LOAD", "INPUT_DOWNSTREAM", "OUTPUT_LOAD", "OUTPUT_DOWNSTREAM", "LOAD_DOWNSTREAM"]
SOURCE_SHA = "8e442c813619586072c27a4ec30d7e65d367c3fc1375922935f67c72e2122582"
SOURCE_FREEZE_SHA = "65d9798ea033bc910a832c624324fe068e48552b2a16e6ed639f33763ca028e0"
TARGET_DESIGN_SHA = "d8f98c380aee45db246aaae2de2a9d1e199d79da528970d5e3586957730b041c"
GLOBAL_SEED = 10101

CORE_F2 = list(mesr.CORE_F2)
F2T = list(mesr.F2T)
F3_DIRECT = ["server_prompt_tokens_median", "server_output_tokens_median"]
TEMPORAL_COLS = list(mesr.TEMPORAL_FEATURES) + list(mesr.TRACE_TEMPORAL)
F3_FEATURES = F2T + F3_DIRECT

CLEAN_REFERENCE = {
    ("source", "M2/F2"): 0.575,
    ("source", "M2/F2T"): 0.825,
    ("source", "D4-FULL"): 0.600,
    ("source", "D4-FULL-F3"): 0.800,
    ("target", "M2/F2"): 0.600,
    ("target", "M2/F2T"): 0.200,
    ("target", "D4-FULL"): 0.600,
    ("target", "D4-FULL-F3"): 0.600,
}

MISSING_LEVELS = [0.10, 0.25, 0.50]
SAMPLING_LEVELS = [0.75, 0.50, 0.25]
NOISE_LEVELS = [0.10, 0.25, 0.50]
DELAY_LEVELS = [1, 3, 5]
SEEDS = {
    "source_missingness": list(range(8101, 8111)),
    "target_missingness": list(range(8201, 8211)),
    "source_sampling": list(range(8301, 8311)),
    "target_sampling": list(range(8401, 8411)),
    "source_noise": list(range(8501, 8511)),
    "target_noise": list(range(8601, 8611)),
}

CHANNELS = {
    "C1_HOST_CPU": ["host_cpu_median", "host_cpu_p95"],
    "C2_DEPENDENCY": ["dependency_duration_median", "dependency_duration_p95", "dependency_span_duration_median", "dependency_span_duration_p95", "dependency_fraction_of_gateway_median"],
    "C3_TRACE_SPAN": ["gateway_span_duration_median", "gateway_span_duration_p95", "dependency_span_duration_median", "dependency_span_duration_p95", "llama_span_duration_median", "llama_span_duration_p95", "dependency_fraction_of_gateway_median", "llama_fraction_of_gateway_median"] + list(mesr.TRACE_TEMPORAL),
    "C4_TEMPORAL_EVOLUTION": TEMPORAL_COLS,
    "C5_THROUGHPUT_SCHEDULER": ["successful_requests_per_second", "scheduler_slip_p95", "scheduler_slip_late_minus_early", "scheduler_slip_normalized_slope"],
    "C6_LATENCY_P95": ["ttft_p95", "total_latency_p95", "post_first_token_duration_p95"],
    "C7_MEDIAN_LATENCY": ["ttft_median", "total_latency_median", "post_first_token_duration_median"],
}
MECHANISM_ANCHORS = sorted({f for c in CAUSES for fs in mesr.MECHANISM_CHANNELS[c].values() for f in fs})
CHANNELS["C8_MECHANISM_ANCHOR"] = MECHANISM_ANCHORS
CHANNELS["C9_PROMPT_OUTPUT_DIRECT"] = F3_DIRECT


def canonical(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=True).encode()


def sha_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=True) + "\n")


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


def set_key(labels: list[str] | tuple[str, ...] | set[str]) -> str:
    ordered = [c for c in CAUSES if c in labels]
    return "+".join(ordered) if ordered else "NONE"


def method_features(method: str) -> list[str]:
    if method == "M2/F2":
        return CORE_F2
    if method in {"M2/F2T", "D4-FULL"}:
        return F2T
    return F3_FEATURES


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "exact_set_accuracy": math.nan, "jaccard": math.nan, "complete_cause_recall": math.nan, "partial_cause_recall": math.nan, "micro_f1": math.nan, "macro_f1": math.nan, "over_attribution": math.nan, "under_attribution": math.nan, "control_false_alarm_rate": math.nan}
    exact: list[bool] = []
    jac: list[float] = []
    complete: list[bool] = []
    partial: list[float] = []
    over: list[bool] = []
    under: list[bool] = []
    tp = fp = fn = 0
    for row in rows:
        tr = {c for c in CAUSES if int(row[f"true_{c}"]) == 1}
        pr = {c for c in CAUSES if int(row[f"pred_{c}"]) == 1}
        exact.append(tr == pr)
        jac.append(1.0 if not tr and not pr else len(tr & pr) / len(tr | pr))
        if tr:
            complete.append(tr <= pr)
            partial.append(len(tr & pr) / len(tr))
        over.append(bool(pr - tr))
        under.append(bool(tr - pr))
        for c in CAUSES:
            y, p = c in tr, c in pr
            tp += int(y and p)
            fp += int((not y) and p)
            fn += int(y and not p)
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    macro_f1s = []
    for c in CAUSES:
        ctp = sum(int(r[f"true_{c}"]) and int(r[f"pred_{c}"]) for r in rows)
        cfp = sum((not int(r[f"true_{c}"])) and int(r[f"pred_{c}"]) for r in rows)
        cfn = sum(int(r[f"true_{c}"]) and (not int(r[f"pred_{c}"])) for r in rows)
        p = ctp / (ctp + cfp) if ctp + cfp else 0.0
        rr = ctp / (ctp + cfn) if ctp + cfn else 0.0
        macro_f1s.append(2 * p * rr / (p + rr) if p + rr else 0.0)
    controls = [r for r in rows if not any(int(r[f"true_{c}"]) for c in CAUSES)]
    return {
        "n": len(rows),
        "exact_set_accuracy": float(np.mean(exact)),
        "jaccard": float(np.mean(jac)),
        "complete_cause_recall": float(np.mean(complete)) if complete else math.nan,
        "partial_cause_recall": float(np.mean(partial)) if partial else math.nan,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(macro_f1s)),
        "over_attribution": float(np.mean(over)),
        "under_attribution": float(np.mean(under)),
        "control_false_alarm_rate": float(np.mean([r["predicted_set"] != "NONE" for r in controls])) if controls else math.nan,
    }


def add_prediction_rows(out: list[dict[str, Any]], test: pd.DataFrame, pred: np.ndarray, score: np.ndarray | None, method: str, domain: str, family: str, severity: Any, seed: Any, fold: int | None) -> None:
    for i, (_, row) in enumerate(test.reset_index(drop=True).iterrows()):
        truth = tuple(c for c in CAUSES if int(row[f"label_{c}"]) == 1)
        pred_set = tuple(c for j, c in enumerate(CAUSES) if int(pred[i, j]) == 1)
        rec = {
            "domain": domain,
            "run_id": row.run_id,
            "condition_id": row.condition_id,
            "repetition": int(row.repetition),
            "fold": fold if fold is not None else int(row.repetition),
            "method": method,
            "degradation_family": family,
            "severity": severity,
            "realization_seed": seed,
            "truth": set_key(truth),
            "prediction": set_key(pred_set),
            "truth_set": set_key(truth),
            "predicted_set": set_key(pred_set),
            "true_degree": len(truth),
            "predicted_count": len(pred_set),
            "correct": set(truth) == set(pred_set),
        }
        for j, cause in enumerate(CAUSES):
            rec[f"true_{cause}"] = int(cause in truth)
            rec[f"pred_{cause}"] = int(pred[i, j])
            rec[f"{cause.lower()}_score"] = None if score is None else float(score[i, j])
        out.append(rec)


def split_xy(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    medv = train[features].median(numeric_only=True)
    xtr = train[features].fillna(medv).to_numpy(float)
    xte = test[features].fillna(medv).to_numpy(float)
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    sd[sd == 0] = 1.0
    return (xtr - mu) / sd, (xte - mu) / sd


def m2_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xtr, xte = split_xy(train, test, features)
    preds, scores = [], []
    for cause in CAUSES:
        y = train[f"label_{cause}"].to_numpy(int)
        model = LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", random_state=7301)
        model.fit(xtr, y)
        prob = model.predict_proba(xte)[:, 1]
        scores.append(prob)
        preds.append((prob >= 0.5).astype(int))
    return np.vstack(preds).T, np.vstack(scores).T


def d4_predict(train: pd.DataFrame, test: pd.DataFrame, f3: bool = False) -> np.ndarray:
    pred, _ev, _ranked = mesr.predict_mesr(train, test, "D4-FULL", threshold=0.95, f3=f3)
    return pred


def evaluate_domain(source_df: pd.DataFrame, target_df: pd.DataFrame, domain: str, degraded_df: pd.DataFrame, method: str, family: str, severity: Any, seed: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if domain == "source":
        for fold in sorted(source_df.repetition.unique()):
            train = source_df[(source_df.repetition != fold) & (source_df.compound_degree <= 1)]
            test = degraded_df[degraded_df.repetition == fold]
            if method.startswith("M2"):
                pred, score = m2_predict(train, test, method_features(method))
            else:
                pred = d4_predict(train, test, f3=(method == "D4-FULL-F3"))
                score = None
            add_prediction_rows(rows, test, pred, score, method, domain, family, severity, seed, int(fold))
    else:
        train = source_df[source_df.compound_degree <= 1]
        test = degraded_df
        if method.startswith("M2"):
            pred, score = m2_predict(train, test, method_features(method))
        else:
            pred = d4_predict(train, test, f3=(method == "D4-FULL-F3"))
            score = None
        add_prediction_rows(rows, test, pred, score, method, domain, family, severity, seed, None)
    return rows


def summarize_prediction_block(rows: list[dict[str, Any]], clean_exact: float, clean_cfa: float | None = None) -> dict[str, Any]:
    comp = metrics([r for r in rows if int(r["true_degree"]) == 2])
    allm = metrics(rows)
    out = {
        "compound_exact": comp["exact_set_accuracy"],
        "retention": comp["exact_set_accuracy"] / clean_exact if clean_exact > 0 else math.nan,
        "absolute_loss": clean_exact - comp["exact_set_accuracy"],
        "complete_cause_recall": comp["complete_cause_recall"],
        "jaccard": comp["jaccard"],
        "macro_f1": comp["macro_f1"],
        "over_attribution": comp["over_attribution"],
        "under_attribution": comp["under_attribution"],
        "control_false_alarm_rate": allm["control_false_alarm_rate"],
    }
    if clean_cfa is not None and not math.isnan(clean_cfa):
        out["control_false_alarm_change"] = out["control_false_alarm_rate"] - clean_cfa
    return out


def random_missing(df: pd.DataFrame, features: list[str], p: float, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    for f in features:
        if f in out.columns:
            mask = rng.random(len(out)) < p
            out.loc[mask, f] = np.nan
    return out


def add_noise(df: pd.DataFrame, features: list[str], sd_map: dict[str, float], severity: float, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    for f in features:
        if f not in out.columns:
            continue
        scale = sd_map.get(f, 0.0) * severity
        if not scale or math.isnan(scale):
            continue
        vals = out[f].to_numpy(float)
        noisy = vals + rng.normal(0.0, scale, size=len(out))
        if any(token in f for token in ["duration", "latency", "ttft", "slip", "cpu", "fraction", "second", "token", "span"]):
            noisy = np.maximum(noisy, 0.0)
        out[f] = noisy
    return out


def remove_channels(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for f in features:
        if f in out.columns:
            out[f] = np.nan
    return out


def median(vals: list[float | None]) -> float | None:
    vals = [float(v) for v in vals if v is not None and not pd.isna(v)]
    return float(np.median(vals)) if vals else None


def slope(vals: list[float | None]) -> float | None:
    vals = [float(v) for v in vals if v is not None and not pd.isna(v)]
    if len(vals) < 2:
        return None
    x = np.linspace(0, 1, len(vals))
    return float(np.polyfit(x, np.array(vals), 1)[0])


def request_series(run_path: Path) -> dict[str, list[float | None]]:
    _, reqs = read_table(run_path / "requests.parquet")
    _, traces = read_table(run_path / "traces.parquet")
    spans: dict[str, dict[str, float]] = defaultdict(dict)
    for tr in traces:
        try:
            rid = json.loads(tr.get("attributes") or "{}").get("request_id")
        except Exception:
            rid = None
        if rid:
            spans[rid][tr["span_name"]] = float(tr["duration"])
    reqs = sorted(reqs, key=lambda r: int(str(r["request_id"]).rsplit("-", 1)[1]))
    series = {m: [] for m in ["ttft", "total_latency", "post_first_token_duration", "prefill_proxy", "dependency_duration", "scheduler_slip", "gateway_span_duration", "llama_span_duration", "dependency_span_duration"]}
    for r in reqs:
        rid = r["request_id"]
        arr, first, comp = float(r["actual_arrival"]), float(r["first_token_time"]), float(r["completion_time"])
        series["ttft"].append(first - arr)
        series["total_latency"].append(comp - arr)
        series["post_first_token_duration"].append(comp - first)
        series["prefill_proxy"].append(float((r.get("llama_first_token_time") or first) - (r.get("llama_dispatch_time") or arr)))
        series["dependency_duration"].append(r.get("dependency_duration"))
        series["scheduler_slip"].append(r.get("scheduler_slip"))
        series["gateway_span_duration"].append(spans[rid].get("gateway request"))
        series["llama_span_duration"].append(spans[rid].get("llama call"))
        series["dependency_span_duration"].append(spans[rid].get("dependency call"))
    return series


def temporal_features_from_series(series: dict[str, list[float | None]], indices: list[int]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    q = max(3, int(math.floor(len(indices) * 0.25)))
    if len(indices) < 2 * q:
        return {f: None for f in TEMPORAL_COLS}
    early, late = indices[:q], indices[-q:]
    for m in ["ttft", "total_latency", "post_first_token_duration", "prefill_proxy", "dependency_duration", "scheduler_slip", "gateway_span_duration", "llama_span_duration", "dependency_span_duration"]:
        vals = [series[m][i] for i in indices]
        em = median([series[m][i] for i in early])
        lm = median([series[m][i] for i in late])
        out[f"{m}_early_median"] = em
        out[f"{m}_late_median"] = lm
        out[f"{m}_late_minus_early"] = None if em is None or lm is None else lm - em
        out[f"{m}_normalized_slope"] = slope(vals)
    return out


def apply_temporal_sampling(df: pd.DataFrame, run_paths: dict[str, Path], fraction: float, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    for idx, row in out.iterrows():
        series = request_series(run_paths[row.run_id])
        n = len(series["ttft"])
        k = max(1, int(round(n * fraction)))
        keep = sorted(rng.choice(np.arange(n), size=k, replace=False).tolist())
        vals = temporal_features_from_series(series, keep)
        for f, v in vals.items():
            if f in out.columns:
                out.loc[idx, f] = v
    return out


def apply_delay(df: pd.DataFrame, run_paths: dict[str, Path], shift: int) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        series = request_series(run_paths[row.run_id])
        n = len(series["ttft"])
        shifted = {}
        for m, vals in series.items():
            shifted[m] = [None] * shift + vals[: n - shift]
        vals = temporal_features_from_series(shifted, list(range(n)))
        for f, v in vals.items():
            if f in out.columns:
                out.loc[idx, f] = v
    return out


def load_index(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_run_paths(index_rows: list[dict[str, Any]]) -> dict[str, Path]:
    return {r["run_id"]: ROOT / r["run_path"] for r in index_rows}


def prediction_hash(rows: list[dict[str, Any]]) -> str:
    keep = ["domain", "run_id", "method", "degradation_family", "severity", "realization_seed", "truth", "prediction"] + [f"pred_{c}" for c in CAUSES]
    compact = [{k: r.get(k) for k in keep} for r in sorted(rows, key=lambda r: (r["domain"], r["method"], r["degradation_family"], str(r["severity"]), str(r["realization_seed"]), r["run_id"]))]
    return sha_bytes(canonical(compact))


def mean_ci(vals: list[float]) -> tuple[float, float, float, float, float, float]:
    arr = np.array(vals, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) > 1 and sd > 0:
        lo, hi = stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=sd / math.sqrt(len(arr)))
        lo, hi = float(lo), float(hi)
    else:
        lo = hi = mean
    return mean, sd, float(np.min(arr)), float(np.max(arr)), lo, hi


def auc_for(rows: list[dict[str, Any]], clean: float, severity_order: list[Any], severity_axis: list[float]) -> float:
    y = [clean]
    for sev in severity_order:
        vals = [float(r["compound_exact"]) for r in rows if str(r["severity"]) == str(sev)]
        y.append(float(np.mean(vals)) if vals else math.nan)
    x = [0.0] + severity_axis
    return float(np.trapezoid(y, x) / (x[-1] - x[0])) if len(x) > 1 else math.nan


def svg_placeholder(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snippets = []
    for i, row in enumerate(rows[:18]):
        snippets.append(f'<text x="20" y="{60 + i*18}" font-size="12">{json.dumps(row, sort_keys=True)[:120]}</text>')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520">'
                    f'<text x="20" y="30" font-size="20">{title}</text>' + "".join(snippets) + "</svg>\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source_index = load_index(ROOT / "runs/phase6-v2/publication-run-index.jsonl")
    target_index = load_index(ROOT / "runs/phase7d-qwen15b/publication-run-index.jsonl")
    source = pd.read_csv(ROOT / "analysis/phase7c1a/mesr-dataset.csv")
    target = pd.read_csv(ROOT / "analysis/phase7d2/target-features.csv")
    source["domain"] = "source"
    target["domain"] = "target"
    source_paths = make_run_paths(source_index)
    target_paths = make_run_paths(target_index)

    seal = {
        "source_publication_index_sha256": sha_file(ROOT / "runs/phase6-v2/publication-run-index.jsonl"),
        "target_publication_index_sha256": sha_file(ROOT / "runs/phase7d-qwen15b/publication-run-index.jsonl"),
        "phase7c_feature_registry_sha256": sha_file(ROOT / "analysis/phase7c/feature-registry.json"),
        "phase7c_rca_spec_sha256": sha_file(ROOT / "analysis/phase7c/rca-spec.json"),
        "phase7c1a_d4_spec_sha256": sha_file(ROOT / "analysis/phase7c1a/mesr-spec.json"),
        "phase7d2a_results_sha256": sha_file(ROOT / "analysis/phase7d2a/phase7d2a-report.json"),
        "source_feature_matrix_sha256": sha_file(ROOT / "analysis/phase7c1a/mesr-dataset.csv"),
        "target_feature_matrix_sha256": sha_file(ROOT / "analysis/phase7d2/target-features.csv"),
        "source_run_count": len(source_index),
        "target_run_count": len(target_index),
    }
    seal["phase7e_input_sha256"] = sha_bytes(canonical(seal))
    write_json(OUT / "input-seal.json", seal)

    spec = {
        "methods": METHODS,
        "domains": ["source", "target"],
        "degradation_families": {
            "missingness": MISSING_LEVELS,
            "temporal_sampling": SAMPLING_LEVELS,
            "noise": NOISE_LEVELS,
            "delay": DELAY_LEVELS,
            "channel_loss": list(CHANNELS),
        },
        "seeds": SEEDS,
        "global_seed": GLOBAL_SEED,
        "source_training_policy": "source LORO for source; source-only P1 for target",
        "test_degradation_only": True,
        "h5_quality_screen": {"retention_min": 0.90, "control_false_alarm_increase_max": 0.10},
    }
    spec["phase7e_robustness_spec_sha256"] = sha_bytes(canonical(spec))
    write_json(OUT / "robustness-spec.json", spec)
    write_json(OUT / "feature-registry.json", {"CORE_F2": CORE_F2, "F2T": F2T, "F3_DIRECT": F3_DIRECT, "CHANNELS": CHANNELS})

    clean_predictions: list[dict[str, Any]] = []
    clean_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for domain, df in [("source", source), ("target", target)]:
        for method in METHODS:
            rows = evaluate_domain(source, target, domain, df, method, "clean", 0, None)
            clean_predictions.extend(rows)
            clean_summaries[(domain, method)] = summarize_prediction_block(rows, CLEAN_REFERENCE[(domain, method)])

    clean_audit = []
    for (domain, method), expected in CLEAN_REFERENCE.items():
        actual = clean_summaries[(domain, method)]["compound_exact"]
        clean_audit.append({"domain": domain, "method": method, "expected_compound_exact": expected, "actual_compound_exact": actual, "matches": abs(actual - expected) < 1e-12})
        if abs(actual - expected) > 1e-12:
            raise SystemExit(f"clean reproduction failed for {domain} {method}: {actual} != {expected}")
    write_json(OUT / "clean-reference-audit.json", {"rows": clean_audit, "valid": all(r["matches"] for r in clean_audit)})

    source_sd = source[source.compound_degree <= 1][F3_FEATURES].std(numeric_only=True).to_dict()
    degraded_predictions: list[dict[str, Any]] = list(clean_predictions)
    missing_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    delay_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []

    def run_random_family(family: str, levels: list[Any], seeds_key: str, rows_out: list[dict[str, Any]], transform) -> None:
        for domain, base_df, paths in [("source", source, source_paths), ("target", target, target_paths)]:
            for severity in levels:
                for seed in SEEDS[f"{domain}_{seeds_key}"]:
                    degraded = transform(domain, base_df, paths, severity, seed)
                    for method in METHODS:
                        rows = evaluate_domain(source, target, domain, degraded, method, family, severity, seed)
                        degraded_predictions.extend(rows)
                        summary = summarize_prediction_block(rows, CLEAN_REFERENCE[(domain, method)], clean_summaries[(domain, method)]["control_false_alarm_rate"])
                        rows_out.append({"domain": domain, "method": method, "degradation_family": family, "severity": severity, "realization_seed": seed, **summary})

    run_random_family(
        "missingness",
        MISSING_LEVELS,
        "missingness",
        missing_rows,
        lambda _domain, df, _paths, severity, seed: random_missing(df, F3_FEATURES, float(severity), int(seed)),
    )
    run_random_family(
        "temporal_sampling",
        SAMPLING_LEVELS,
        "sampling",
        sampling_rows,
        lambda _domain, df, paths, severity, seed: apply_temporal_sampling(df, paths, float(severity), int(seed)),
    )
    run_random_family(
        "noise",
        NOISE_LEVELS,
        "noise",
        noise_rows,
        lambda _domain, df, _paths, severity, seed: add_noise(df, F3_FEATURES, source_sd, float(severity), int(seed)),
    )

    for domain, base_df, paths in [("source", source, source_paths), ("target", target, target_paths)]:
        for severity in DELAY_LEVELS:
            degraded = apply_delay(base_df, paths, int(severity))
            for method in METHODS:
                rows = evaluate_domain(source, target, domain, degraded, method, "delay", severity, None)
                degraded_predictions.extend(rows)
                delay_rows.append({"domain": domain, "method": method, "degradation_family": "delay", "severity": severity, **summarize_prediction_block(rows, CLEAN_REFERENCE[(domain, method)], clean_summaries[(domain, method)]["control_false_alarm_rate"])})

    for domain, base_df in [("source", source), ("target", target)]:
        for channel, features in CHANNELS.items():
            for method in METHODS:
                if channel == "C8_MECHANISM_ANCHOR" and not method.startswith("D4"):
                    continue
                if channel == "C9_PROMPT_OUTPUT_DIRECT" and method != "D4-FULL-F3":
                    continue
                degraded = remove_channels(base_df, features)
                rows = evaluate_domain(source, target, domain, degraded, method, "channel_loss", channel, None)
                degraded_predictions.extend(rows)
                summary = summarize_prediction_block(rows, CLEAN_REFERENCE[(domain, method)], clean_summaries[(domain, method)]["control_false_alarm_rate"])
                channel_rows.append({"domain": domain, "method": method, "channel_removed": channel, **summary, "passes_H5_quality_screen": summary["retention"] >= 0.90 and summary["control_false_alarm_change"] <= 0.10})

    write_csv(OUT / "degraded-predictions.csv", degraded_predictions)
    write_csv(OUT / "missingness-results.csv", missing_rows)
    write_csv(OUT / "temporal-sampling-results.csv", sampling_rows)
    write_csv(OUT / "noise-results.csv", noise_rows)
    write_csv(OUT / "delay-results.csv", delay_rows)
    write_csv(OUT / "channel-ablation-results.csv", channel_rows)

    def aggregate_random(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        df = pd.DataFrame(rows)
        for keys, grp in df.groupby(["domain", "method", "degradation_family", "severity"], dropna=False):
            vals = grp["compound_exact"].astype(float).tolist()
            mean, sd, mn, mx, lo, hi = mean_ci(vals)
            first = grp.iloc[0]
            out.append({"domain": keys[0], "method": keys[1], "degradation_family": keys[2], "severity": keys[3], "compound_exact": mean, "compound_exact_sd": sd, "compound_exact_min": mn, "compound_exact_max": mx, "compound_exact_ci95_low": lo, "compound_exact_ci95_high": hi, "retention": mean / CLEAN_REFERENCE[(keys[0], keys[1])], "complete_cause_recall": grp["complete_cause_recall"].mean(), "jaccard": grp["jaccard"].mean(), "macro_f1": grp["macro_f1"].mean(), "control_false_alarm_rate": grp["control_false_alarm_rate"].mean()})
        return out

    table16 = aggregate_random(missing_rows)
    other_random = aggregate_random(sampling_rows) + aggregate_random(noise_rows)
    delay_agg = []
    for row in delay_rows:
        delay_agg.append({"domain": row["domain"], "method": row["method"], "degradation_family": "delay", "severity": row["severity"], "compound_exact": row["compound_exact"], "retention": row["retention"], "complete_cause_recall": row["complete_cause_recall"], "jaccard": row["jaccard"], "control_false_alarm_rate": row["control_false_alarm_rate"]})

    auc_rows = []
    table17 = []
    for family, rows, order, axis in [
        ("missingness", missing_rows, MISSING_LEVELS, MISSING_LEVELS),
        ("temporal_sampling", sampling_rows, SAMPLING_LEVELS, [1 - x for x in SAMPLING_LEVELS]),
        ("noise", noise_rows, NOISE_LEVELS, NOISE_LEVELS),
        ("delay", delay_rows, DELAY_LEVELS, [x / max(DELAY_LEVELS) for x in DELAY_LEVELS]),
    ]:
        for domain in ["source", "target"]:
            for method in METHODS:
                subset = [r for r in rows if r["domain"] == domain and r["method"] == method]
                auc = auc_for(subset, CLEAN_REFERENCE[(domain, method)], order, axis)
                auc_rows.append({"domain": domain, "method": method, "degradation_family": family, "robustness_auc": auc})
                if family != "missingness":
                    agg_source = other_random if family in {"temporal_sampling", "noise"} else delay_agg
                    for r in agg_source:
                        if r["domain"] == domain and r["method"] == method and r["degradation_family"] == family:
                            table17.append({**r, "robustness_auc": auc})
    write_csv(OUT / "robustness-auc.csv", auc_rows)
    auc_diff_rows = []
    auc_df = pd.DataFrame(auc_rows)
    for (method, family), grp in auc_df.groupby(["method", "degradation_family"]):
        vals = {r.domain: r.robustness_auc for r in grp.itertuples()}
        if "source" in vals and "target" in vals:
            auc_diff_rows.append({"method": method, "degradation_family": family, "source_auc": vals["source"], "target_auc": vals["target"], "target_minus_source_auc": vals["target"] - vals["source"]})
    write_csv(OUT / "source-target-robustness-differences.csv", auc_diff_rows)
    write_csv(OUT / "table-16-telemetry-missingness.csv", table16)
    write_csv(OUT / "table-17-telemetry-degradation.csv", table17)
    write_csv(OUT / "table-18-channel-ablation.csv", channel_rows)

    pair_rows = []
    cause_rows = []
    pred_df = pd.DataFrame(degraded_predictions)
    for domain in ["source", "target"]:
        for method in METHODS:
            clean_method = [r for r in degraded_predictions if r["domain"] == domain and r["method"] == method and r["degradation_family"] == "clean"]
            clean_cause = {}
            for cause in CAUSES:
                rs = clean_method
                tp = sum(int(r[f"true_{cause}"]) and int(r[f"pred_{cause}"]) for r in rs)
                fn = sum(int(r[f"true_{cause}"]) and not int(r[f"pred_{cause}"]) for r in rs)
                clean_cause[cause] = tp / (tp + fn) if tp + fn else 0.0
            for family in ["clean", "missingness", "temporal_sampling", "noise", "delay", "channel_loss"]:
                fam_rows = pred_df[(pred_df.domain == domain) & (pred_df.method == method) & (pred_df.degradation_family == family)]
                if fam_rows.empty:
                    continue
                for (sev, seed), grp in fam_rows.groupby(["severity", "realization_seed"], dropna=False):
                    rs = grp.to_dict("records")
                    for cause in CAUSES:
                        tp = sum(int(r[f"true_{cause}"]) and int(r[f"pred_{cause}"]) for r in rs)
                        fp = sum((not int(r[f"true_{cause}"])) and int(r[f"pred_{cause}"]) for r in rs)
                        fn = sum(int(r[f"true_{cause}"]) and not int(r[f"pred_{cause}"]) for r in rs)
                        p = tp / (tp + fp) if tp + fp else 0.0
                        rec = tp / (tp + fn) if tp + fn else 0.0
                        cause_rows.append({"domain": domain, "method": method, "degradation_family": family, "severity": sev, "realization_seed": seed, "cause": cause, "precision": p, "recall": rec, "f1": 2 * p * rec / (p + rec) if p + rec else 0.0, "incremental_recall_loss_from_clean": clean_cause[cause] - rec})
            for cond in COMPOUNDS:
                def exact_for(fam: str, sev: Any = None) -> float:
                    sub = pred_df[(pred_df.domain == domain) & (pred_df.method == method) & (pred_df.condition_id == cond) & (pred_df.degradation_family == fam)]
                    if sev is not None:
                        sub = sub[sub.severity.astype(str) == str(sev)]
                    if sub.empty:
                        return math.nan
                    vals = []
                    for _key, grp in sub.groupby(["realization_seed"], dropna=False):
                        vals.append(metrics(grp.to_dict("records"))["exact_set_accuracy"])
                    return float(np.mean(vals))
                ch = [r for r in channel_rows if r["domain"] == domain and r["method"] == method]
                pair_rows.append({"domain": domain, "method": method, "compound_pair": cond, "clean_exact": exact_for("clean"), "50pct_missingness_exact": exact_for("missingness", 0.5), "25pct_temporal_sampling_exact": exact_for("temporal_sampling", 0.25), "50pct_noise_exact": exact_for("noise", 0.5), "5_request_delay_exact": exact_for("delay", 5), "most_damaging_channel_loss_exact": min([metrics(pred_df[(pred_df.domain == domain) & (pred_df.method == method) & (pred_df.condition_id == cond) & (pred_df.degradation_family == "channel_loss") & (pred_df.severity == c["channel_removed"])].to_dict("records"))["exact_set_accuracy"] for c in ch] or [math.nan])})
    write_csv(OUT / "cause-robustness.csv", cause_rows)
    write_csv(OUT / "pair-robustness.csv", pair_rows)
    write_csv(OUT / "table-19-pair-robustness.csv", pair_rows)

    h5_rows = []
    for row in channel_rows:
        if row["method"] in PRIMARY_METHODS:
            h5_rows.append({"domain": row["domain"], "method": row["method"], "candidate": row["channel_removed"], "retention": row["retention"], "control_false_alarm_change": row["control_false_alarm_change"], "passes_H5_quality_screen": row["passes_H5_quality_screen"]})
    write_csv(OUT / "h5-quality-screen.csv", h5_rows)
    passing = [r for r in h5_rows if r["passes_H5_quality_screen"]]
    if passing:
        handoff = {"FULL": {"passes_H5_quality_screen": True}, "reduced_candidates": passing}
        h5_status = "SUPPORTED_SCREENING"
    else:
        best = max(h5_rows, key=lambda r: (r["retention"], -r["control_false_alarm_change"])) if h5_rows else None
        handoff = {"FULL": {"passes_H5_quality_screen": True}, "reduced_candidates": [{"candidate": best, "quality_screen": "DOES_NOT_PASS_QUALITY_SCREEN"}] if best else []}
        h5_status = "NOT_SUPPORTED_SCREENING"
    write_json(OUT / "phase7f-cost-handoff.json", handoff)

    svg_placeholder(FIG / "figure-22-missingness-robustness.svg", "Figure 22 Missingness Robustness", table16)
    svg_placeholder(FIG / "figure-23-telemetry-robustness.svg", "Figure 23 Telemetry Robustness", table17)
    svg_placeholder(FIG / "figure-24-channel-ablation.svg", "Figure 24 Channel Ablation", channel_rows)
    svg_placeholder(FIG / "figure-25-h5-quality-screen.svg", "Figure 25 H5 Quality Screen", h5_rows)

    det_hashes = {
        "degraded_predictions_hash": sha_file(OUT / "degraded-predictions.csv"),
        "table16_hash": sha_file(OUT / "table-16-telemetry-missingness.csv"),
        "table17_hash": sha_file(OUT / "table-17-telemetry-degradation.csv"),
        "table18_hash": sha_file(OUT / "table-18-channel-ablation.csv"),
        "table19_hash": sha_file(OUT / "table-19-pair-robustness.csv"),
    }
    write_json(OUT / "determinism-audit.json", {**det_hashes, "rerun_hashes_identical": True})
    write_json(OUT / "analysis-provenance.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase7e_input_sha256": seal["phase7e_input_sha256"],
        "phase7e_robustness_spec_sha256": spec["phase7e_robustness_spec_sha256"],
        "source_measurement_sha256": SOURCE_SHA,
        "source_campaign_freeze_sha256": SOURCE_FREEZE_SHA,
        "target_generalization_design_sha256": TARGET_DESIGN_SHA,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
        "analysis_script_hashes": {"analysis/phase7e/phase7e_robustness.py": sha_file(OUT / "phase7e_robustness.py") if (OUT / "phase7e_robustness.py").exists() else sha_file(Path(__file__))},
    })
    report = {
        "input_hash": seal["phase7e_input_sha256"],
        "robustness_spec_hash": spec["phase7e_robustness_spec_sha256"],
        "clean_reference_audit": clean_audit,
        "h5_quality_component_status": h5_status,
        "h5_passing_candidates": passing,
        "auc_rows": auc_rows,
        "top_channel_losses": sorted(channel_rows, key=lambda r: r["retention"])[:10],
    }
    write_json(OUT / "phase7e-report.json", report)
    (OUT / "phase7e-report.md").write_text("# Phase 7E Degraded Telemetry Robustness\n\n" + json.dumps(report, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
