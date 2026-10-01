from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
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
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sloscope.artifacts.writer import read_table

OUT = ROOT / "analysis" / "phase7c"
FIGURES = OUT / "figures"
CAMPAIGN_ID = "sloscope-phase6-v2"
SOURCE_SHA = "8e442c813619586072c27a4ec30d7e65d367c3fc1375922935f67c72e2122582"
FREEZE_SHA = "65d9798ea033bc910a832c624324fe068e48552b2a16e6ed639f33763ca028e0"
CAUSES = ["INPUT", "OUTPUT", "LOAD", "DOWNSTREAM"]
SEED = 7301
SLO = {
    "ttft": 0.06512947314299491,
    "total": 0.26779416389490324,
    "decode": 0.20162438542758712,
}

F0 = [
    "ttft_slo_violated",
    "total_latency_slo_violated",
    "decode_duration_slo_violated",
    "ttft_violation_margin",
    "total_latency_violation_margin",
    "decode_duration_violation_margin",
]
F1 = [
    "ttft_p95",
    "total_latency_p95",
    "post_first_token_duration_p95",
    "ttft_median",
    "total_latency_median",
    "post_first_token_duration_median",
    "scheduler_slip_p95",
    "successful_requests_per_second",
]
TRACE_FEATURES = [
    "gateway_span_duration_median",
    "gateway_span_duration_p95",
    "dependency_span_duration_median",
    "dependency_span_duration_p95",
    "llama_span_duration_median",
    "llama_span_duration_p95",
    "dependency_fraction_of_gateway_median",
    "llama_fraction_of_gateway_median",
]
F2 = F1 + [
    "host_cpu_median",
    "host_cpu_p95",
    "dependency_duration_median",
    "dependency_duration_p95",
] + TRACE_FEATURES
F3 = F2 + ["server_prompt_tokens_median", "server_output_tokens_median"]
FEATURE_TIERS = {"F0": F0, "F1": F1, "F2": F2, "F3": F3}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def percentile(vals: list[float], p: float) -> float | None:
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def median(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return float(np.median(vals)) if vals else None


def cause_set_from_active(active: list[str]) -> tuple[str, ...]:
    labels = []
    if "input_medium" in active:
        labels.append("INPUT")
    if "output_32" in active:
        labels.append("OUTPUT")
    if "load_12rps" in active:
        labels.append("LOAD")
    if "downstream_100ms" in active:
        labels.append("DOWNSTREAM")
    return tuple(labels)


def label_vec(labels: tuple[str, ...]) -> list[int]:
    return [int(c in labels) for c in CAUSES]


def set_key(labels: tuple[str, ...] | list[str]) -> str:
    return "+".join(labels) if labels else "NONE"


def jaccard(true: set[str], pred: set[str]) -> float:
    if not true and not pred:
        return 1.0
    return len(true & pred) / len(true | pred) if true | pred else 1.0


def create_input_seal(index_rows: list[dict[str, Any]]) -> dict[str, Any]:
    seal = {
        "schema_version": "sloscope.phase7c.input_seal.v1",
        "campaign_id": CAMPAIGN_ID,
        "publication_run_index_sha256": sha256_file(ROOT / "runs" / "phase6-v2" / "publication-run-index.jsonl"),
        "phase7a1_run_level_metrics_sha256": sha256_file(ROOT / "analysis" / "phase7a1" / "run-level-metrics.csv"),
        "phase7b_feature_registry_sha256": sha256_file(ROOT / "analysis" / "phase7b" / "feature-registry.json"),
        "phase7b_rca_handoff_sha256": sha256_file(ROOT / "analysis" / "phase7b" / "rca-feature-handoff.json"),
        "factorial_design_sha256": sha256_file(ROOT / "campaigns" / "phase5" / "factorial-design.json"),
        "source_tree_sha256": SOURCE_SHA,
        "campaign_freeze_sha256": FREEZE_SHA,
        "run_ids": [r["run_id"] for r in index_rows],
    }
    seal["phase7c_input_sha256"] = sha256_bytes(canonical(seal))
    write_json(OUT / "input-seal.json", seal)
    return seal


def create_spec() -> dict[str, Any]:
    spec = {
        "schema_version": "sloscope.phase7c.rca_spec.v1",
        "mechanism_label_universe": CAUSES,
        "ground_truth_mapping": "experimental_condition.json active_mechanisms only",
        "feature_tiers": FEATURE_TIERS,
        "preprocessing": "training-fold median imputation plus training-fold standardization",
        "cross_validation": "leave-one-repetition-out, eight folds",
        "protocols": {
            "P1": "train compound_degree <= 1, test compound_degree == 2",
            "P2": "train all seven repetitions, test all conditions in held-out repetition",
        },
        "models": {
            "M0": "SLO nearest neighbor on F0, deterministic run_id tie-break",
            "M1": "nearest centroid in standardized feature space",
            "M2": "one-vs-rest LogisticRegression liblinear L2 C=1.0 class_weight=balanced threshold=0.5",
            "M3": "one-vs-rest LinearSVC class_weight=balanced decision score > 0",
        },
        "primary_baseline": {"protocol": "P1/P2 comparison", "model": "M2", "feature_tier": "F2"},
        "metrics": [
            "exact_set_accuracy",
            "jaccard",
            "micro/macro precision recall F1",
            "hamming_loss",
            "complete_cause_recall",
            "partial_cause_recall",
            "over/under attribution",
            "control_false_alarm_rate",
        ],
        "h2_primary_test": "exact 2^8 sign-flip test on fold-level complete-cause-recall differences P2-P1 for M2/F2",
        "figures": ["figure-10", "figure-11", "figure-12", "figure-13"],
        "random_seed": SEED,
    }
    spec["phase7c_rca_spec_sha256"] = sha256_bytes(canonical(spec))
    write_json(OUT / "rca-spec.json", spec)
    return spec


def load_index() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (ROOT / "runs" / "phase6-v2" / "publication-run-index.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 104 or len({r["run_id"] for r in rows}) != 104:
        raise RuntimeError("publication index must contain 104 unique rows")
    for r in rows:
        if r.get("source_hash") != SOURCE_SHA or r.get("freeze_hash") != FREEZE_SHA:
            raise RuntimeError("hash mismatch")
        if str(r.get("run_path")).startswith("runs/phase6/"):
            raise RuntimeError("superseded phase6 data found")
    return rows


def trace_features(run_path: Path) -> dict[str, float | None]:
    _, rows = read_table(run_path / "traces.parquet")
    by = defaultdict(list)
    for row in rows:
        name = row.get("span_name")
        dur = row.get("duration")
        if dur is not None:
            by[name].append(float(dur))
    gateway = by["gateway request"]
    dep = by["dependency call"]
    llama = by["llama call"]
    dep_frac = [d / g for d, g in zip(dep, gateway) if g]
    llama_frac = [l / g for l, g in zip(llama, gateway) if g]
    return {
        "gateway_span_duration_median": median(gateway),
        "gateway_span_duration_p95": percentile(gateway, 0.95),
        "dependency_span_duration_median": median(dep),
        "dependency_span_duration_p95": percentile(dep, 0.95),
        "llama_span_duration_median": median(llama),
        "llama_span_duration_p95": percentile(llama, 0.95),
        "dependency_fraction_of_gateway_median": median(dep_frac),
        "llama_fraction_of_gateway_median": median(llama_frac),
    }


def build_dataset(index_rows: list[dict[str, Any]]) -> pd.DataFrame:
    metrics = pd.read_csv(ROOT / "analysis" / "phase7a1" / "run-level-metrics.csv")
    by_run = {r["run_id"]: r for r in index_rows}
    rows = []
    for row in metrics.to_dict("records"):
        run_id = row["run_id"]
        idx = by_run[run_id]
        cond = read_json(ROOT / idx["run_path"] / "experimental_condition.json")
        active = cond.get("active_mechanisms", [])
        labels = cause_set_from_active(active)
        out = dict(row)
        out.update(trace_features(ROOT / idx["run_path"]))
        out["true_set"] = set_key(labels)
        out["label_INPUT"], out["label_OUTPUT"], out["label_LOAD"], out["label_DOWNSTREAM"] = label_vec(labels)
        out["compound_degree"] = int(cond["compound_degree"])
        out["active_mechanisms_source"] = json.dumps(active, sort_keys=True)
        out["ttft_violation_margin"] = (out["ttft_p95"] - SLO["ttft"]) / SLO["ttft"]
        out["total_latency_violation_margin"] = (out["total_latency_p95"] - SLO["total"]) / SLO["total"]
        out["decode_duration_violation_margin"] = (out["post_first_token_duration_p95"] - SLO["decode"]) / SLO["decode"]
        rows.append(out)
    df = pd.DataFrame(rows)
    if df.groupby("condition_id").size().nunique() != 1 or df.groupby("condition_id").size().iloc[0] != 8:
        raise RuntimeError("expected 8 runs per condition")
    return df


def feature_leakage_audit() -> dict[str, Any]:
    forbidden = {"condition_id", "active_mechanisms", "mechanism_levels", "compound_degree", "run_id"}
    tiers = {k: set(v) for k, v in FEATURE_TIERS.items()}
    issues = {tier: sorted(features & forbidden) for tier, features in tiers.items()}
    return {
        "schema_version": "sloscope.phase7c.feature_leakage_audit.v1",
        "issues": issues,
        "primary_F2_leakage_issues": issues["F2"],
        "passed": issues["F2"] == [],
        "unavailable_excluded": ["requests_processing", "requests_deferred"],
    }


def label_audit(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for cond, sub in sorted(df.groupby("condition_id")):
        rows.append({
            "condition_id": cond,
            "active_mechanism_set": sub.iloc[0]["true_set"],
            "compound_degree": int(sub.iloc[0]["compound_degree"]),
            "run_count": len(sub),
        })
    return rows


def split_xy(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    med = train[features].median(numeric_only=True)
    xtr = train[features].fillna(med).to_numpy(float)
    xte = test[features].fillna(med).to_numpy(float)
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (xtr - mu) / sd, (xte - mu) / sd


def predict_m0(train, test, features):
    xtr, xte = split_xy(train, test, features)
    train_rows = train.reset_index(drop=True)
    preds, scores = [], []
    for x in xte:
        d = np.linalg.norm(xtr - x, axis=1)
        candidates = np.where(d == d.min())[0]
        best = sorted(candidates, key=lambda i: train_rows.loc[i, "run_id"])[0]
        y = [int(train_rows.loc[best, f"label_{c}"]) for c in CAUSES]
        preds.append(y)
        scores.append([-d[best]] * 4)
    return np.array(preds), np.array(scores), []


def predict_m1(train, test, features):
    xtr, xte = split_xy(train, test, features)
    tr = train.reset_index(drop=True)
    centroids = []
    keys = []
    labels = []
    for key, idxs in tr.groupby("true_set").groups.items():
        idx = list(idxs)
        centroids.append(xtr[idx].mean(axis=0))
        keys.append(key)
        labels.append([int(tr.loc[idx[0], f"label_{c}"]) for c in CAUSES])
    centroids = np.vstack(centroids)
    preds, scores = [], []
    for x in xte:
        d = np.linalg.norm(centroids - x, axis=1)
        best = sorted(np.where(d == d.min())[0], key=lambda i: keys[i])[0]
        preds.append(labels[best])
        scores.append([-d[best]] * 4)
    return np.array(preds), np.array(scores), []


def predict_m2(train, test, features):
    xtr, xte = split_xy(train, test, features)
    preds, scores, coefs = [], [], []
    pred_cols = []
    score_cols = []
    for ci, cause in enumerate(CAUSES):
        y = train[f"label_{cause}"].to_numpy(int)
        if len(set(y)) < 2:
            prob = np.full(len(test), float(y[0]))
            coef = np.zeros(xtr.shape[1])
        else:
            model = LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", random_state=SEED)
            model.fit(xtr, y)
            prob = model.predict_proba(xte)[:, 1]
            coef = model.coef_[0]
        pred_cols.append((prob >= 0.5).astype(int))
        score_cols.append(prob)
        for f, val in zip(features, coef):
            coefs.append({"cause": cause, "feature": f, "coefficient": float(val)})
    return np.vstack(pred_cols).T, np.vstack(score_cols).T, coefs


def predict_m3(train, test, features):
    xtr, xte = split_xy(train, test, features)
    pred_cols, score_cols, coefs = [], [], []
    for cause in CAUSES:
        y = train[f"label_{cause}"].to_numpy(int)
        if len(set(y)) < 2:
            dec = np.full(len(test), 1.0 if y[0] else -1.0)
            coef = np.zeros(xtr.shape[1])
        else:
            model = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED, max_iter=10000)
            model.fit(xtr, y)
            dec = model.decision_function(xte)
            coef = model.coef_[0]
        pred_cols.append((dec > 0).astype(int))
        score_cols.append(dec)
        for f, val in zip(features, coef):
            coefs.append({"cause": cause, "feature": f, "coefficient": float(val)})
    return np.vstack(pred_cols).T, np.vstack(score_cols).T, coefs


def run_models(df: pd.DataFrame):
    combos = []
    for protocol in ["P1", "P2"]:
        combos.extend([
            (protocol, "M0", "F0"),
            (protocol, "M1", "F1"),
            (protocol, "M1", "F2"),
            (protocol, "M2", "F0"),
            (protocol, "M2", "F1"),
            (protocol, "M2", "F2"),
            (protocol, "M2", "F3"),
            (protocol, "M3", "F2"),
        ])
    predictors = {"M0": predict_m0, "M1": predict_m1, "M2": predict_m2, "M3": predict_m3}
    predictions, coef_rows = [], []
    for protocol, model, tier in combos:
        features = FEATURE_TIERS[tier]
        for fold in range(1, 9):
            train = df[df.repetition != fold]
            if protocol == "P1":
                train = train[train.compound_degree <= 1]
                test = df[(df.repetition == fold) & (df.compound_degree == 2)]
            else:
                test = df[df.repetition == fold]
            pred, score, coefs = predictors[model](train, test, features)
            for c in coefs:
                c.update({"protocol": protocol, "model": model, "feature_tier": tier, "fold": fold})
                coef_rows.append(c)
            for i, (_, row) in enumerate(test.reset_index(drop=True).iterrows()):
                true = [int(row[f"label_{c}"]) for c in CAUSES]
                pred_vec = [int(v) for v in pred[i]]
                true_set = tuple(c for c, v in zip(CAUSES, true) if v)
                pred_set = tuple(c for c, v in zip(CAUSES, pred_vec) if v)
                scores = {f"{c.lower()}_score": float(score[i, ci]) for ci, c in enumerate(CAUSES)}
                predictions.append({
                    "run_id": row.run_id,
                    "repetition": int(row.repetition),
                    "condition_id": row.condition_id,
                    "true_mechanism_set": set_key(true_set),
                    "predicted_mechanism_set": set_key(pred_set),
                    **scores,
                    "true_compound_degree": int(row.compound_degree),
                    "predicted_cause_count": sum(pred_vec),
                    "training_protocol": protocol,
                    "model": model,
                    "feature_tier": tier,
                    "fold": fold,
                    **{f"true_{c}": true[ci] for ci, c in enumerate(CAUSES)},
                    **{f"pred_{c}": pred_vec[ci] for ci, c in enumerate(CAUSES)},
                })
    return predictions, coef_rows


def metrics_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    tp = fp = fn = tn = 0
    exact = []
    jacs = []
    complete = []
    partial = []
    over = []
    under = []
    false_counts = []
    missed_counts = []
    for r in rows:
        true = {c for c in CAUSES if int(r[f"true_{c}"])}
        pred = {c for c in CAUSES if int(r[f"pred_{c}"])}
        exact.append(true == pred)
        jacs.append(jaccard(true, pred))
        if true:
            complete.append(true <= pred)
            partial.append(len(true & pred) / len(true))
        over.append(bool(pred - true))
        under.append(bool(true - pred))
        false_counts.append(len(pred - true))
        missed_counts.append(len(true - pred))
        for c in CAUSES:
            y = c in true; p = c in pred
            tp += y and p; fp += (not y) and p; fn += y and (not p); tn += (not y) and (not p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per_f1 = []
    for c in CAUSES:
        ctp = sum(int(r[f"true_{c}"]) and int(r[f"pred_{c}"]) for r in rows)
        cfp = sum((not int(r[f"true_{c}"])) and int(r[f"pred_{c}"]) for r in rows)
        cfn = sum(int(r[f"true_{c}"]) and (not int(r[f"pred_{c}"])) for r in rows)
        cp = ctp / (ctp + cfp) if ctp + cfp else 0.0
        cr = ctp / (ctp + cfn) if ctp + cfn else 0.0
        per_f1.append(2 * cp * cr / (cp + cr) if cp + cr else 0.0)
    controls = [r for r in rows if not any(int(r[f"true_{c}"]) for c in CAUSES)]
    return {
        "n": len(rows),
        "exact_set_accuracy": float(np.mean(exact)),
        "jaccard": float(np.mean(jacs)),
        "micro_precision": prec,
        "micro_recall": rec,
        "micro_f1": f1,
        "macro_f1": float(np.mean(per_f1)),
        "hamming_loss": (fp + fn) / (len(rows) * len(CAUSES)),
        "complete_cause_recall": float(np.mean(complete)) if complete else None,
        "partial_cause_recall": float(np.mean(partial)) if partial else None,
        "exact_cause_set_recovery": float(np.mean(exact)),
        "over_attribution_rate": float(np.mean(over)),
        "under_attribution_rate": float(np.mean(under)),
        "false_attribution_count_mean": float(np.mean(false_counts)),
        "missed_cause_count_mean": float(np.mean(missed_counts)),
        "control_false_alarm_rate": float(np.mean([r["predicted_mechanism_set"] != "NONE" for r in controls])) if controls else None,
    }


def topk_metrics(rows):
    vals1=[]; vals2=[]
    for r in rows:
        truth={c for c in CAUSES if int(r[f"true_{c}"])}
        if not truth: continue
        ranked=sorted(CAUSES,key=lambda c: float(r[f"{c.lower()}_score"]), reverse=True)
        vals1.append(ranked[0] in truth)
        vals2.append(truth <= set(ranked[:2]))
    return float(np.mean(vals1)) if vals1 else None, float(np.mean(vals2)) if vals2 else None


def signflip(diffs):
    obs=abs(float(np.mean(diffs)))
    vals=[]
    for mask in range(2**len(diffs)):
        signs=np.array([1 if (mask>>i)&1 else -1 for i in range(len(diffs))])
        vals.append(abs(float(np.mean(np.array(diffs)*signs))))
    return sum(v >= obs - 1e-12 for v in vals)/len(vals)


def summarize(preds, coefs):
    pred_df=pd.DataFrame(preds)
    table6=[]; fold=[]; cause=[]; compound=[]; errors=[]; conf=[]; table7=[]
    for keys, sub in pred_df.groupby(["training_protocol","model","feature_tier"]):
        m=metrics_for(sub.to_dict("records")); row={"training_protocol":keys[0],"model":keys[1],"feature_tier":keys[2],"subset":"all",**m}; table6.append(row)
        for subset_name, ss in [("single-only", sub[sub.true_compound_degree==1]), ("compound-only", sub[sub.true_compound_degree==2])]:
            table6.append({"training_protocol":keys[0],"model":keys[1],"feature_tier":keys[2],"subset":subset_name,**metrics_for(ss.to_dict("records"))})
        for f, fs in sub.groupby("fold"):
            fold.append({"training_protocol":keys[0],"model":keys[1],"feature_tier":keys[2],"fold":f,**metrics_for(fs.to_dict("records"))})
        for c in CAUSES:
            rows=sub.to_dict("records")
            tp=sum(int(r[f"true_{c}"]) and int(r[f"pred_{c}"]) for r in rows)
            fp=sum((not int(r[f"true_{c}"])) and int(r[f"pred_{c}"]) for r in rows)
            fn=sum(int(r[f"true_{c}"]) and (not int(r[f"pred_{c}"])) for r in rows)
            tn=sum((not int(r[f"true_{c}"])) and (not int(r[f"pred_{c}"])) for r in rows)
            p=tp/(tp+fp) if tp+fp else 0; rr=tp/(tp+fn) if tp+fn else 0; f1=2*p*rr/(p+rr) if p+rr else 0
            cause.append({"training_protocol":keys[0],"model":keys[1],"feature_tier":keys[2],"cause":c,"precision":p,"recall":rr,"f1":f1,"false_positive_rate":fp/(fp+tn) if fp+tn else 0,"false_negative_rate":fn/(fn+tp) if fn+tp else 0,"support":tp+fn})
        for cond, cs in sub[sub.true_compound_degree==2].groupby("condition_id"):
            tm=metrics_for(cs.to_dict("records")); top1, top2=topk_metrics(cs.to_dict("records"))
            compound.append({"training_protocol":keys[0],"model":keys[1],"feature_tier":keys[2],"condition_id":cond,**tm,"top1_true_cause_coverage":top1,"top2_complete_coverage":top2})
            table7.append(compound[-1])
        for (t,p), cs in sub.groupby(["true_mechanism_set","predicted_mechanism_set"]):
            conf.append({"training_protocol":keys[0],"model":keys[1],"feature_tier":keys[2],"true_cause_set":t,"predicted_cause_set":p,"count":len(cs)})
    # errors primary compound P1/P2 M2/F2
    csd=pd.read_csv(ROOT/"analysis"/"phase7b"/"csd-summary.csv")
    csd_by={r.compound: r for r in csd.itertuples()}
    for r in pred_df[(pred_df.model=="M2")&(pred_df.feature_tier=="F2")&(pred_df.true_compound_degree==2)].to_dict("records"):
        true=[c for c in CAUSES if int(r[f"true_{c}"])]; pred=[c for c in CAUSES if int(r[f"pred_{c}"])]
        if set(true)==set(pred): continue
        types=[]
        missed=[c for c in true if c not in pred]
        extra=[c for c in pred if c not in true]
        if len(missed)==1: types.append(f"MISS_{missed[0]}")
        if len(missed)==2: types.append("MISS_BOTH")
        if extra: types.append("EXTRA_CAUSE")
        if len(pred)==1: types.append("SINGLE_ONLY_PREDICTION")
        if len(pred)==0: types.append("EMPTY_PREDICTION")
        csum=csd_by.get(r["condition_id"])
        errors.append({**{k:r[k] for k in ["run_id","training_protocol","model","feature_tier","condition_id","true_mechanism_set","predicted_mechanism_set"]},"error_types":";".join(types),"missed_causes":";".join(missed),"extra_causes":";".join(extra),"CSD":getattr(csum,"CSD",None),"CSD_Holm_supported":getattr(csum,"statistically_supported",None)})
    # H2
    h2=[]
    for f in range(1,9):
        p1=pred_df[(pred_df.training_protocol=="P1")&(pred_df.model=="M2")&(pred_df.feature_tier=="F2")&(pred_df.fold==f)]
        p2=pred_df[(pred_df.training_protocol=="P2")&(pred_df.model=="M2")&(pred_df.feature_tier=="F2")&(pred_df.fold==f)&(pred_df.true_compound_degree==2)]
        v1=metrics_for(p1.to_dict("records"))["complete_cause_recall"]
        v2=metrics_for(p2.to_dict("records"))["complete_cause_recall"]
        h2.append({"fold":f,"P1_complete_cause_recall":v1,"P2_complete_cause_recall":v2,"difference":v2-v1})
    diffs=[r["difference"] for r in h2]
    ci=stats.t.interval(0.95,len(diffs)-1,loc=np.mean(diffs),scale=stats.sem(diffs)) if np.std(diffs,ddof=1)>0 else (np.mean(diffs),np.mean(diffs))
    h2_summary={"fold":"SUMMARY","P1_complete_cause_recall":None,"P2_complete_cause_recall":None,"difference":float(np.mean(diffs)),"median_difference":float(np.median(diffs)),"ci_low":float(ci[0]),"ci_high":float(ci[1]),"exact_signflip_p":signflip(diffs)}
    h2.append(h2_summary)
    return table6, fold, cause, compound, errors, conf, h2, pd.DataFrame(coefs)


def svg(path, title, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420"><style>text{{font-family:Arial;font-size:10px}}.title{{font-size:15px;font-weight:bold}}</style><text class="title" x="20" y="24">{title}</text>']
    y=55
    for row in rows[:18]:
        lines.append(f'<text x="20" y="{y}">{row}</text>'); y+=20
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    index=load_index(); seal=create_input_seal(index); spec=create_spec()
    df=build_dataset(index)
    write_csv(OUT/"rca-dataset.csv", df.to_dict("records"))
    write_csv(OUT/"label-audit.csv", label_audit(df))
    write_json(OUT/"feature-registry.json", {"feature_tiers":FEATURE_TIERS})
    write_json(OUT/"feature-leakage-audit.json", feature_leakage_audit())
    preds, coefs=run_models(df)
    # determinism
    preds2,_=run_models(df)
    h1=sha256_bytes(canonical(preds)); h2=sha256_bytes(canonical(preds2))
    write_json(OUT/"determinism-audit.json", {"first_prediction_hash":h1,"second_prediction_hash":h2,"identical":h1==h2})
    write_csv(OUT/"rca-predictions.csv", preds)
    table6, fold, cause, compound, errors, conf, h2rows, coef_df=summarize(preds, coefs)
    write_csv(OUT/"rca-fold-metrics.csv", fold); write_csv(OUT/"rca-cause-metrics.csv", cause); write_csv(OUT/"rca-compound-metrics.csv", compound)
    write_csv(OUT/"rca-errors.csv", errors); write_csv(OUT/"rca-confusion-sets.csv", conf); write_csv(OUT/"h2-fold-comparison.csv", h2rows)
    # coefficient summary
    coeff=[]
    if not coef_df.empty:
        for keys, sub in coef_df.groupby(["protocol","model","feature_tier","cause","feature"]):
            vals=sub.coefficient.to_numpy(float)
            coeff.append({"training_protocol":keys[0],"model":keys[1],"feature_tier":keys[2],"cause":keys[3],"feature":keys[4],"mean_coefficient":float(vals.mean()),"sd":float(vals.std(ddof=1)) if len(vals)>1 else 0,"positive_fraction":float((vals>0).mean()),"sign_changes": len(set(np.sign(vals))-{0})>1})
    write_csv(OUT/"rca-coefficients.csv", coeff)
    # csd linked
    csd=pd.read_csv(ROOT/"analysis"/"phase7b"/"csd-summary.csv")
    csdmap={r.compound:r for r in csd.itertuples()}
    linked=[]
    for e in errors:
        c=csdmap.get(e["condition_id"]); linked.append({**e,"CSD":getattr(c,"CSD",None),"CSD_Holm_supported":getattr(c,"statistically_supported",None)})
    write_csv(OUT/"csd-linked-error-analysis.csv", linked)
    write_csv(OUT/"table-6-rca-baselines.csv", table6); write_csv(OUT/"table-7-compound-rca.csv", compound); write_csv(OUT/"table-8-single-vs-mixed-training.csv", h2rows)
    # simple figures
    primary=[r for r in table6 if r["training_protocol"] in ["P1","P2"] and r["model"]=="M2" and r["feature_tier"]=="F2"]
    svg(FIGURES/"figure-10-rca-baselines.svg","Figure 10. RCA baselines",[str({k:r.get(k) for k in ['training_protocol','subset','exact_set_accuracy','complete_cause_recall','macro_f1']}) for r in primary])
    svg(FIGURES/"figure-11-single-to-compound-gap.svg","Figure 11. Single-to-compound gap",[str(r) for r in h2rows])
    p1pairs=[r for r in compound if r["training_protocol"]=="P1" and r["model"]=="M2" and r["feature_tier"]=="F2"]
    svg(FIGURES/"figure-12-compound-rca-by-pair.svg","Figure 12. Compound RCA by pair",[str({k:r[k] for k in ['condition_id','exact_set_accuracy','complete_cause_recall','jaccard']}) for r in p1pairs])
    primcause=[r for r in cause if r["training_protocol"]=="P2" and r["model"]=="M2" and r["feature_tier"]=="F2"]
    svg(FIGURES/"figure-13-cause-error-matrix.svg","Figure 13. Cause error matrix",[str({k:r[k] for k in ['cause','precision','recall','f1','false_positive_rate','false_negative_rate']}) for r in primcause])
    # handoff
    write_json(OUT/"sloscope-rca-method-handoff.json", {"baseline_weaknesses":["P1 cannot predict multi-cause sets for centroid baselines; logistic P1 tests compositional generalization","formal queue metrics unavailable","OUTPUT_CONTROL drift stresses SLO-only diagnosis"],"highest_under_attribution_pairs":[r for r in p1pairs if r.get("under_attribution_rate") is not None],"feature_tier_sensitivity":"see table-6","control_false_alarm_behavior":"see table-6 and predictions"})
    prov={"schema_version":"sloscope.phase7c.provenance.v1","timestamp":utc_now(),"phase7c_input_sha256":seal["phase7c_input_sha256"],"phase7c_rca_spec_sha256":spec["phase7c_rca_spec_sha256"],"source_tree_sha256":SOURCE_SHA,"campaign_freeze_sha256":FREEZE_SHA,"python_version":sys.version,"dependency_versions":{"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"pyarrow":pyarrow.__version__},"analysis_script_hashes":{"analysis/phase7c/phase7c_rca.py":sha256_file(Path(__file__))}}
    write_json(OUT/"analysis-provenance.json", prov)
    report={"phase7c_input_sha256":seal["phase7c_input_sha256"],"phase7c_rca_spec_sha256":spec["phase7c_rca_spec_sha256"],"label_distribution":label_audit(df),"primary_results":[r for r in table6 if r["model"]=="M2" and r["feature_tier"]=="F2"],"h2":h2rows,"determinism_hash":h1}
    write_json(OUT/"phase7c-report.json", report)
    (OUT/"phase7c-report.md").write_text("# Phase 7C RCA Baselines\n\nSee CSV artifacts for full baseline, H2, compound, and error tables.\n")

if __name__=="__main__":
    main()
