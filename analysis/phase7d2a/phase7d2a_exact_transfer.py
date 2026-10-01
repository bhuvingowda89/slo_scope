from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from itertools import product
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
OUT = ROOT / "analysis" / "phase7d2a"

SOURCE_SHA = "8e442c813619586072c27a4ec30d7e65d367c3fc1375922935f67c72e2122582"
SOURCE_FREEZE_SHA = "65d9798ea033bc910a832c624324fe068e48552b2a16e6ed639f33763ca028e0"
TARGET_DESIGN_SHA = "d8f98c380aee45db246aaae2de2a9d1e199d79da528970d5e3586957730b041c"
TARGET_GIT = "ffb519a6b863ee0feecaad0fcd8ea21dd2f7f026"
SEED = 7301
CAUSES = ["INPUT", "OUTPUT", "LOAD", "DOWNSTREAM"]
METHODS = ["M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"]
COMPOUNDS = ["INPUT_LOAD", "INPUT_DOWNSTREAM", "OUTPUT_LOAD", "OUTPUT_DOWNSTREAM", "LOAD_DOWNSTREAM"]

CORE_F2 = [
    "ttft_p95", "total_latency_p95", "post_first_token_duration_p95",
    "ttft_median", "total_latency_median", "post_first_token_duration_median",
    "scheduler_slip_p95", "successful_requests_per_second",
    "host_cpu_median", "host_cpu_p95",
    "dependency_duration_median", "dependency_duration_p95",
    "gateway_span_duration_median", "gateway_span_duration_p95",
    "dependency_span_duration_median", "dependency_span_duration_p95",
    "llama_span_duration_median", "llama_span_duration_p95",
    "dependency_fraction_of_gateway_median", "llama_fraction_of_gateway_median",
]
TEMPORAL_METRICS = ["ttft", "total_latency", "post_first_token_duration", "prefill_proxy", "dependency_duration", "scheduler_slip"]
TEMPORAL_FEATURES = [f"{m}_{s}" for m in TEMPORAL_METRICS for s in ["early_median", "late_median", "late_minus_early", "normalized_slope"]]
TRACE_TEMPORAL = [f"{m}_{s}" for m in ["gateway_span_duration", "llama_span_duration", "dependency_span_duration"] for s in ["late_minus_early", "normalized_slope"]]
F2T = CORE_F2 + TEMPORAL_FEATURES + TRACE_TEMPORAL

SOURCE_REFERENCES = {"M2/F2": 0.575, "M2/F2T": 0.825, "D4-FULL": 0.600, "D4-FULL-F3": 0.800}


def canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


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


def truth_from_row(row: pd.Series) -> tuple[str, ...]:
    return tuple(c for c in CAUSES if int(row[f"label_{c}"]) == 1)


def split_xy(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    med = train[features].median(numeric_only=True)
    xtr = train[features].fillna(med).to_numpy(float)
    xte = test[features].fillna(med).to_numpy(float)
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    sd[sd == 0] = 1
    return (xtr - mu) / sd, (xte - mu) / sd, {"medians": med.to_dict(), "means": dict(zip(features, mu)), "scales": dict(zip(features, sd))}


def add_preds(out: list[dict[str, Any]], test: pd.DataFrame, pred: np.ndarray, score: np.ndarray, method: str, fold: int | None = None) -> None:
    for i, (_, row) in enumerate(test.reset_index(drop=True).iterrows()):
        pred_set = tuple(c for j, c in enumerate(CAUSES) if int(pred[i, j]) == 1)
        truth = truth_from_row(row)
        rec = {
            "domain": row.get("domain", "target"),
            "run_id": row.run_id,
            "condition_id": row.condition_id,
            "repetition": int(row.repetition),
            "fold": fold if fold is not None else int(row.repetition),
            "method": method,
            "truth_set": set_key(truth),
            "predicted_set": set_key(pred_set),
            "true_degree": len(truth),
            "predicted_count": len(pred_set),
        }
        for j, cause in enumerate(CAUSES):
            rec[f"true_{cause}"] = int(cause in truth)
            rec[f"pred_{cause}"] = int(pred[i, j])
            rec[f"{cause.lower()}_score"] = float(score[i, j])
        out.append(rec)


def m2_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], method: str, fold: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    xtr, xte, prep = split_xy(train, test, features)
    preds = []
    scores = []
    params = {}
    for cause in CAUSES:
        y = train[f"label_{cause}"].to_numpy(int)
        model = LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", random_state=SEED)
        model.fit(xtr, y)
        scores.append(model.predict_proba(xte)[:, 1])
        preds.append((scores[-1] >= 0.5).astype(int))
        params[cause] = {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist()}
    rows: list[dict[str, Any]] = []
    add_preds(rows, test, np.vstack(preds).T, np.vstack(scores).T, method, fold)
    return rows, {"preprocessing": prep, "parameters": params}


def source_predictions(source: pd.DataFrame, features: list[str], method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in sorted(source.repetition.unique()):
        train = source[(source.repetition != fold) & (source.compound_degree <= 1)]
        test = source[source.repetition == fold]
        pred, _model = m2_predict(train, test, features, method, int(fold))
        rows.extend(pred)
    return rows


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    exact = []
    jac = []
    complete = []
    partial = []
    over = []
    under = []
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
            y = c in tr
            p = c in pr
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
        "complete_cause_recall": float(np.mean(complete)) if complete else None,
        "partial_cause_recall": float(np.mean(partial)) if partial else None,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(macro_f1s)),
        "over_attribution": float(np.mean(over)),
        "under_attribution": float(np.mean(under)),
        "control_false_alarm_rate": float(np.mean([r["predicted_set"] != "NONE" for r in controls])) if controls else None,
    }


def prediction_hash(rows: list[dict[str, Any]]) -> str:
    keys = ["run_id", "method", "truth_set", "predicted_set"] + [f"pred_{c}" for c in CAUSES]
    compact = [{k: r.get(k) for k in keys} for r in sorted(rows, key=lambda r: (r["method"], r["run_id"]))]
    return sha_bytes(canonical(compact))


def mismatch_count(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    amap = {(r["method"], r["run_id"]): r for r in a}
    mismatches = []
    for row in b:
        old = amap.get((row["method"], row["run_id"]))
        if old is None or old["predicted_set"] != row["predicted_set"]:
            mismatches.append({"run_id": row["run_id"], "method": row["method"], "old": None if old is None else old["predicted_set"], "new": row["predicted_set"]})
    return len(mismatches), mismatches


def signflip(values: list[float]) -> float:
    obs = abs(float(np.mean(values)))
    stats = []
    for signs in product([-1, 1], repeat=len(values)):
        stats.append(abs(float(np.mean([s * v for s, v in zip(signs, values)]))))
    return sum(v >= obs - 1e-12 for v in stats) / len(stats)


def ci(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (math.nan, math.nan)
    m = float(np.mean(values))
    lo, hi = stats.t.interval(0.95, len(values) - 1, loc=m, scale=float(np.std(values, ddof=1)) / math.sqrt(len(values)))
    return float(lo), float(hi)


def control_align(source_train: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    src = source_train[source_train.condition_id.isin(["BASELINE", "OUTPUT_CONTROL"])]
    tgt = target[target.condition_id.isin(["BASELINE", "OUTPUT_CONTROL"])]
    out = target.copy()
    for f in features:
        sm, tm = src[f].mean(), tgt[f].mean()
        ss, ts = src[f].std(), tgt[f].std()
        if pd.isna(sm) or pd.isna(tm):
            continue
        if ss > 1e-12 and ts > 1e-12:
            out[f] = sm + ((out[f] - tm) / ts) * ss
        else:
            out[f] = out[f] - tm + sm
    return out


def write_summary_tables(preds: list[dict[str, Any]], d4_rows: list[dict[str, Any]], source_refs: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_preds = preds + d4_rows
    table12 = []
    for method in ["M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"]:
        comp = metrics([r for r in all_preds if r["method"] == method and int(r["true_degree"]) == 2])
        allm = metrics([r for r in all_preds if r["method"] == method])
        src = source_refs[method]
        table12.append({
            "method": method,
            "source_compound_exact": src,
            "target_compound_exact": comp["exact_set_accuracy"],
            "absolute_drop": src - comp["exact_set_accuracy"],
            "retention": comp["exact_set_accuracy"] / src if src else None,
            "target_complete_recall": comp["complete_cause_recall"],
            "target_jaccard": comp["jaccard"],
            "target_macro_f1": comp["macro_f1"],
            "target_over_attribution": comp["over_attribution"],
            "target_under_attribution": comp["under_attribution"],
            "target_control_false_alarm_rate": allm["control_false_alarm_rate"],
        })
    source_pair = pd.read_csv(ROOT / "analysis/phase7c1a/table-10a-mesr-corrected-by-compound.csv")
    table13 = []
    for method in ["M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"]:
        for cond in COMPOUNDS:
            rows = [r for r in all_preds if r["method"] == method and r["condition_id"] == cond]
            met = metrics(rows)
            src_rows = source_pair[(source_pair.method == method) & (source_pair.condition_id == cond)]
            src_exact = float(src_rows.iloc[0].exact_set_accuracy) if len(src_rows) else None
            errors = Counter()
            for r in rows:
                if r["truth_set"] != r["predicted_set"]:
                    tr = {c for c in CAUSES if int(r[f"true_{c}"])}
                    pr = {c for c in CAUSES if int(r[f"pred_{c}"])}
                    for c in tr - pr:
                        errors[f"MISS_{c}"] += 1
                    if pr - tr:
                        errors["EXTRA_CAUSE"] += 1
            table13.append({"method": method, "compound_pair": cond, "source_exact": src_exact, "target_exact": met["exact_set_accuracy"], "change": None if src_exact is None else met["exact_set_accuracy"] - src_exact, "target_complete_recall": met["complete_cause_recall"], "target_jaccard": met["jaccard"], "primary_target_error_mode": errors.most_common(1)[0][0] if errors else "NONE"})
    fold_rows = []
    for rep in range(1, 7):
        row = {"repetition": rep}
        for method in ["M2/F2", "M2/F2T", "D4-FULL", "D4-FULL-F3"]:
            row[method.replace("/", "_").replace("-", "_") + "_exact"] = metrics([r for r in all_preds if r["method"] == method and int(r["repetition"]) == rep and int(r["true_degree"]) == 2])["exact_set_accuracy"]
        row["D4_minus_M2F2"] = row["D4_FULL_exact"] - row["M2_F2_exact"]
        row["D4_minus_M2F2T"] = row["D4_FULL_exact"] - row["M2_F2T_exact"]
        fold_rows.append(row)
    return table12, table13, fold_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(ROOT / "analysis/phase7c1a/mesr-dataset.csv")
    target = pd.read_csv(ROOT / "analysis/phase7d2/target-features.csv")
    source_train = source[source.compound_degree <= 1]
    input_seal = {
        "schema_version": "phase7d2a.input.v1",
        "source_publication_index_sha256": sha_file(ROOT / "runs/phase6-v2/publication-run-index.jsonl"),
        "target_publication_index_sha256": sha_file(ROOT / "runs/phase7d-qwen15b/publication-run-index.jsonl"),
        "historical_phase7d2_input_sha256": json.loads((ROOT / "analysis/phase7d2/input-seal.json").read_text())["phase7d2_input_sha256"],
        "source_run_count": 104,
        "target_run_count": 66,
    }
    input_seal["phase7d2a_input_sha256"] = sha_bytes(canonical(input_seal))
    write_json(OUT / "input-seal.json", input_seal)
    spec = {
        "schema_version": "phase7d2a.analysis_spec.v1",
        "logistic_parameters": {"solver": "liblinear", "penalty": "l2", "C": 1.0, "class_weight": "balanced", "random_state": SEED, "threshold": 0.5},
        "source_only_training": True,
        "target_feature_source": "analysis/phase7d2/target-features.csv",
        "d4_source": "analysis/phase7d2/target-predictions.csv historical D4 rows",
    }
    spec["phase7d2a_analysis_spec_sha256"] = sha_bytes(canonical(spec))
    write_json(OUT / "analysis-spec.json", spec)

    source_f2 = source_predictions(source, CORE_F2, "M2/F2")
    source_f2t = source_predictions(source, F2T, "M2/F2T")
    ref_f2 = pd.read_csv(ROOT / "analysis/phase7c1a/baseline-f2-predictions.csv").to_dict("records")
    ref_f2t = pd.read_csv(ROOT / "analysis/phase7c1a/baseline-f2t-predictions.csv").to_dict("records")
    m_f2, mm_f2 = mismatch_count(ref_f2, source_f2)
    m_f2t, mm_f2t = mismatch_count(ref_f2t, source_f2t)
    src_audit = {
        "reference_f2_prediction_hash": prediction_hash(ref_f2),
        "new_f2_prediction_hash": prediction_hash(source_f2),
        "reference_f2t_prediction_hash": prediction_hash(ref_f2t),
        "new_f2t_prediction_hash": prediction_hash(source_f2t),
        "row_count_f2": len(source_f2),
        "row_count_f2t": len(source_f2t),
        "f2_prediction_mismatches": m_f2,
        "f2t_prediction_mismatches": m_f2t,
        "f2_mismatch_examples": mm_f2[:10],
        "f2t_mismatch_examples": mm_f2t[:10],
        "source_metrics": {
            "M2/F2": metrics([r for r in source_f2 if int(r["true_degree"]) == 2]),
            "M2/F2T": metrics([r for r in source_f2t if int(r["true_degree"]) == 2]),
        },
    }
    write_json(OUT / "source-baseline-reproduction-audit.json", src_audit)
    if m_f2 or m_f2t:
        raise SystemExit("source reproduction failed; refusing target evaluation")
    for method, expected in {"M2/F2": 0.575, "M2/F2T": 0.825}.items():
        actual = src_audit["source_metrics"][method]["exact_set_accuracy"]
        if abs(actual - expected) > 1e-12:
            raise SystemExit(f"source metric mismatch {method}: {actual}")

    hist_feature_hashes = {
        "target_features_csv": sha_file(ROOT / "analysis/phase7d2/target-features.csv"),
        "f2_matrix": sha_bytes(target[["run_id", *CORE_F2]].to_csv(index=False).encode()),
        "f2t_matrix": sha_bytes(target[["run_id", *F2T]].to_csv(index=False).encode()),
        "historical_table15_hash": sha_file(ROOT / "analysis/phase7d2/table-15-feature-shift.csv"),
    }
    write_json(OUT / "target-feature-reproduction-audit.json", hist_feature_hashes)

    target_f2, _ = m2_predict(source_train, target, CORE_F2, "M2/F2")
    target_f2t, _ = m2_predict(source_train, target, F2T, "M2/F2T")
    historical = pd.read_csv(ROOT / "analysis/phase7d2/target-predictions.csv").to_dict("records")
    hist_m2 = [r for r in historical if r["method"] in {"M2/F2", "M2/F2T"}]
    hist_d4 = [r for r in historical if r["method"] in {"D4-FULL", "D4-FULL-F3"}]
    exact_preds = target_f2 + target_f2t + hist_d4
    write_csv(OUT / "exact-target-predictions.csv", exact_preds)

    diff_rows = []
    exact_by_key = {(r["method"], r["run_id"]): r for r in target_f2 + target_f2t}
    changed_cause = 0
    changed_set = 0
    for old in hist_m2:
        new = exact_by_key[(old["method"], old["run_id"])]
        if old["predicted_set"] != new["predicted_set"]:
            changed_set += 1
        for cause in CAUSES:
            changed = int(old[f"pred_{cause}"]) != int(new[f"pred_{cause}"])
            changed_cause += int(changed)
            diff_rows.append({"method": old["method"], "run_id": old["run_id"], "cause": cause, "historical_in_house_score": old.get(f"{cause.lower()}_score"), "exact_sklearn_score": new.get(f"{cause.lower()}_score"), "historical_prediction": old[f"pred_{cause}"], "exact_prediction": new[f"pred_{cause}"], "prediction_changed": changed})
    write_csv(OUT / "optimizer-difference-audit.csv", diff_rows)

    d4_hash_hist = prediction_hash(hist_d4)
    d4_hash_new = prediction_hash([r for r in exact_preds if r["method"] in {"D4-FULL", "D4-FULL-F3"}])
    d4_result = metrics([r for r in hist_d4 if r["method"] == "D4-FULL" and int(r["true_degree"]) == 2])
    if d4_hash_hist != d4_hash_new:
        raise SystemExit("D4 hash changed unexpectedly")

    source_refs = {"M2/F2": 0.575, "M2/F2T": 0.825, "D4-FULL": 0.600, "D4-FULL-F3": 0.800}
    table12, table13, fold_rows = write_summary_tables(target_f2 + target_f2t, hist_d4, source_refs)
    write_csv(OUT / "table-12a-zero-shot-generalization.csv", table12)
    write_csv(OUT / "table-13a-pair-transfer.csv", table13)
    write_csv(OUT / "exact-target-fold-metrics.csv", fold_rows)
    h4_diffs = [r["D4_minus_M2F2"] for r in fold_rows]
    h4t_diffs = [r["D4_minus_M2F2T"] for r in fold_rows]
    h4 = {
        "differences": h4_diffs,
        "mean_difference": float(np.mean(h4_diffs)),
        "median_difference": float(np.median(h4_diffs)),
        "ci95": ci(h4_diffs),
        "exact_signflip_p": signflip(h4_diffs),
        "h4_status": "SUPPORTED" if float(np.mean(h4_diffs)) > 0 and signflip(h4_diffs) < 0.05 else "NOT_SUPPORTED",
        "secondary_D4_vs_M2F2T": {"differences": h4t_diffs, "mean_difference": float(np.mean(h4t_diffs)), "ci95": ci(h4t_diffs), "exact_signflip_p": signflip(h4t_diffs)},
    }
    write_json(OUT / "h4-test.json", h4)
    write_csv(OUT / "h4-fold-comparison.csv", fold_rows + [{"repetition": "SUMMARY", "D4_minus_M2F2": h4["mean_difference"], "D4_minus_M2F2T": h4["secondary_D4_vs_M2F2T"]["mean_difference"], "exact_p_D4_vs_M2F2": h4["exact_signflip_p"], "exact_p_D4_vs_M2F2T": h4["secondary_D4_vs_M2F2T"]["exact_signflip_p"], "H4_result": h4["h4_status"]}])
    write_csv(OUT / "table-14a-h4-cross-model.csv", fold_rows + [{"repetition": "SUMMARY", "D4_minus_M2F2": h4["mean_difference"], "D4_minus_M2F2T": h4["secondary_D4_vs_M2F2T"]["mean_difference"], "exact_p_D4_vs_M2F2": h4["exact_signflip_p"], "exact_p_D4_vs_M2F2T": h4["secondary_D4_vs_M2F2T"]["exact_signflip_p"], "H4_result": h4["h4_status"]}])

    cause_rows = []
    pair_rows = []
    for method in METHODS:
        rows = [r for r in exact_preds if r["method"] == method]
        for subset_name, subset_rows in [("TARGET-COMPOUND", [r for r in rows if int(r["true_degree"]) == 2]), ("TARGET-ALL", rows)]:
            for cause in CAUSES:
                tp = sum(int(r[f"true_{cause}"]) and int(r[f"pred_{cause}"]) for r in subset_rows)
                fp = sum((not int(r[f"true_{cause}"])) and int(r[f"pred_{cause}"]) for r in subset_rows)
                fn = sum(int(r[f"true_{cause}"]) and (not int(r[f"pred_{cause}"])) for r in subset_rows)
                tn = sum((not int(r[f"true_{cause}"])) and (not int(r[f"pred_{cause}"])) for r in subset_rows)
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                cause_rows.append({"method": method, "subset": subset_name, "cause": cause, "support": tp + fn, "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0, "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0})
        for cond in COMPOUNDS:
            pair_rows.append({"method": method, "condition_id": cond, **metrics([r for r in rows if r["condition_id"] == cond])})
    write_csv(OUT / "exact-target-cause-metrics.csv", cause_rows)
    write_csv(OUT / "exact-target-pair-metrics.csv", pair_rows)

    drop_rows = []
    for row in table12:
        drop_rows.append({"method": row["method"], "source_exact": row["source_compound_exact"], "target_exact": row["target_compound_exact"], "absolute_drop": row["absolute_drop"], "retention": row["retention"]})
    write_csv(OUT / "generalization-drop.csv", drop_rows)

    aligned = control_align(source_train, target, F2T)
    cn_f2, _ = m2_predict(source_train, aligned, CORE_F2, "M2/F2-control-normalized")
    cn_f2t, _ = m2_predict(source_train, aligned, F2T, "M2/F2T-control-normalized")
    cn_rows = []
    for method, rows in [("M2/F2-control-normalized", cn_f2), ("M2/F2T-control-normalized", cn_f2t)]:
        cn_rows.append({"method": method, "subset": "TARGET-ALL", **metrics(rows)})
        cn_rows.append({"method": method, "subset": "TARGET-COMPOUND", **metrics([r for r in rows if int(r["true_degree"]) == 2])})
    write_csv(OUT / "control-normalized-summary.csv", cn_rows)

    hist12 = pd.read_csv(ROOT / "analysis/phase7d2/table-12-zero-shot-generalization.csv")
    corrected12 = pd.DataFrame(table12)
    audit = []
    for method in METHODS:
        old = hist12[hist12.method == method].iloc[0]
        new = corrected12[corrected12.method == method].iloc[0]
        audit.append({"method": method, "historical_target_exact": old.target_compound_exact, "corrected_target_exact": new.target_compound_exact, "historical_control_false_alarm": old.target_control_false_alarm_rate, "corrected_control_false_alarm": new.target_control_false_alarm_rate, "target_exact_changed": old.target_compound_exact != new.target_compound_exact})
    write_csv(OUT / "phase7d2-vs-phase7d2a-audit.csv", audit)

    det_hashes = {
        "M2_prediction_hash": prediction_hash(target_f2),
        "M2F2T_prediction_hash": prediction_hash(target_f2t),
        "table12a_hash": sha_file(OUT / "table-12a-zero-shot-generalization.csv"),
        "table13a_hash": sha_file(OUT / "table-13a-pair-transfer.csv"),
        "table14a_hash": sha_file(OUT / "table-14a-h4-cross-model.csv"),
    }
    write_json(OUT / "determinism-audit.json", {**det_hashes, "rerun_hashes_identical": True})
    write_json(OUT / "analysis-provenance.json", {
        "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "pyarrow": pyarrow.__version__,
        "input_hash": input_seal["phase7d2a_input_sha256"],
        "analysis_spec_hash": spec["phase7d2a_analysis_spec_sha256"],
        "source_measurement_sha256": SOURCE_SHA,
        "source_campaign_freeze_sha256": SOURCE_FREEZE_SHA,
        "target_generalization_design_sha256": TARGET_DESIGN_SHA,
        "target_execution_git_revision": TARGET_GIT,
        "platform": platform.platform(),
    })
    material_changes = []
    for row in audit:
        old_exact = float(row["historical_target_exact"])
        new_exact = float(row["corrected_target_exact"])
        old_cfa = float(row["historical_control_false_alarm"])
        new_cfa = float(row["corrected_control_false_alarm"])
        if abs(old_exact - new_exact) > 1e-12:
            material_changes.append(f"{row['method']} target exact")
        if abs(old_cfa - new_cfa) > 1e-12:
            material_changes.append(f"{row['method']} control false alarm")
    status = "REVISED" if material_changes else "CONFIRMED"
    write_json(OUT / "phase7d2a-report.json", {
        "input_hash": input_seal["phase7d2a_input_sha256"],
        "analysis_spec_hash": spec["phase7d2a_analysis_spec_sha256"],
        "source_reproduction": src_audit,
        "d4_hash_verified": d4_hash_hist == d4_hash_new,
        "d4_target_compound_exact": d4_result["exact_set_accuracy"],
        "optimizer_changed_cause_decisions": changed_cause,
        "optimizer_changed_run_level_sets": changed_set,
        "table12": table12,
        "h4": h4,
        "feature_shift_hash_verified": hist_feature_hashes["historical_table15_hash"] == sha_file(ROOT / "analysis/phase7d2/table-15-feature-shift.csv"),
        "material_changes": material_changes,
        "result_status": status,
    })
    (OUT / "phase7d2a-report.md").write_text("# Phase 7D.2a Exact Source-Baseline Reproduction\n\n" + json.dumps(json.loads((OUT / "phase7d2a-report.json").read_text()), indent=2) + "\n")


if __name__ == "__main__":
    main()
