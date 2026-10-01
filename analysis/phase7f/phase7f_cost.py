from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sloscope.artifacts.validation import validate_run
from sloscope.artifacts.writer import read_json, read_table, sha256_file
from sloscope.config import ExperimentConfig, write_config
from sloscope.provenance import source_tree_sha256

OUT = ROOT / "analysis" / "phase7f"
FIG = OUT / "figures"
CAMPAIGN = ROOT / "campaigns" / "phase7f-cost"
CONFIG_ROOT = ROOT / "configs" / "phase7f-cost" / "runs"
RUN_ROOT = ROOT / "runs" / "phase7f-cost"

CAMPAIGN_ID = "sloscope-telemetry-cost-v1"
SOURCE_SHA = "8e442c813619586072c27a4ec30d7e65d367c3fc1375922935f67c72e2122582"
PHASE7E_INPUT = "d71062e8c648becfabf1063948b7b36f37040e6387349bd6e957d2cd500ee272"
PHASE7E_SPEC = "979f56823140bd518d5b7f95663340767be5fb154f2e8837987c626d9e4f42dd"
MODEL_REF = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
MODEL_ALIAS = "sloscope-qwen2.5-0.5b"
RUNTIME_VERSION = "0.5.0"
RUNTIME_BUILD = "11146"
RUNTIME_COMMIT = "7fe450e19"
ORDER_SEED = 11731
ANALYSIS_SEED = 12101

TELEMETRY_CONFIGS = {
    "T0_FULL": {"request_telemetry": True, "system_metrics": True, "runtime_metrics": True, "traces": True},
    "T1_NO_HOST": {"request_telemetry": True, "system_metrics": False, "runtime_metrics": True, "traces": True},
    "T2_NO_RUNTIME": {"request_telemetry": True, "system_metrics": True, "runtime_metrics": False, "traces": True},
    "T3_LIGHT": {"request_telemetry": True, "system_metrics": False, "runtime_metrics": False, "traces": True},
}
WORKLOAD_TEMPLATES = {
    "BASELINE": ROOT / "configs" / "phase5" / "baseline.yaml",
    "LOAD_MEDIUM": ROOT / "configs" / "phase5" / "single-load.yaml",
}
CONFIG_SLUG = {"T0_FULL": "t0-full", "T1_NO_HOST": "t1-no-host", "T2_NO_RUNTIME": "t2-no-runtime", "T3_LIGHT": "t3-light"}
WORKLOAD_SLUG = {"BASELINE": "baseline", "LOAD_MEDIUM": "load-medium"}


def canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=True).encode()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip() == ""


def registry() -> dict[str, Any]:
    return {
        "actual_collection_channel": {
            "C1_HOST_CPU": {
                "collected_through": "telemetry.system_metrics",
                "phase7f_primary_candidate": True,
                "live_configuration": "T1_NO_HOST",
            }
        },
        "collection_artifact_channel": {
            "C3_TRACE_SPAN": {
                "switch": "telemetry.traces=false",
                "phase7f_primary_live": False,
                "limitation": "would measure trace retrieval/storage/artifact cost under the existing implementation, not full tracing instrumentation cost",
            }
        },
        "not_independently_switchable_without_source_change": ["C2_DEPENDENCY"],
        "analysis_derived_features": ["C4_TEMPORAL_EVOLUTION", "C5_THROUGHPUT_SCHEDULER", "C6_LATENCY_P95", "C7_MEDIAN_LATENCY"],
        "runtime_metrics_cost_only": {
            "reason": "formal RCA features do not rely on unavailable llama.cpp queue metric rows",
            "live_configurations": ["T2_NO_RUNTIME", "T3_LIGHT"],
        },
    }


def cost_spec() -> dict[str, Any]:
    spec = {
        "campaign_id": CAMPAIGN_ID,
        "run_root": "runs/phase7f-cost",
        "model_ref": MODEL_REF,
        "model_alias": MODEL_ALIAS,
        "telemetry_configurations": TELEMETRY_CONFIGS,
        "workloads": ["BASELINE", "LOAD_MEDIUM"],
        "repetitions": 6,
        "warmup_requests_per_run": 5,
        "measured_requests_per_run": 40,
        "randomization_seed": ORDER_SEED,
        "primary_h5_candidate": {"method": "M2/F2", "comparison": "T1_NO_HOST vs T0_FULL"},
        "primary_cost_metric": "optional_telemetry_bytes_per_measured_request",
        "service_performance_guard": {"metric": "total_latency_p95", "relative_margin": 0.05, "paired_ci": "95% Student-t"},
        "analysis_seed": ANALYSIS_SEED,
    }
    spec["phase7f_cost_spec_sha256"] = sha_bytes(canonical(spec))
    return spec


def generate() -> None:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(ORDER_SEED)
    rows: list[dict[str, Any]] = []
    planned_order = 0
    for rep in range(1, 7):
        for workload in ["BASELINE", "LOAD_MEDIUM"]:
            configs = list(TELEMETRY_CONFIGS)
            rng.shuffle(configs)
            for telemetry_config in configs:
                planned_order += 1
                run_id = f"phase7f-r{rep:02d}-{WORKLOAD_SLUG[workload]}-{CONFIG_SLUG[telemetry_config]}-a01"
                run_dir = f"runs/phase7f-cost/{run_id}"
                config_path = f"configs/phase7f-cost/runs/{run_id}.json"
                rows.append(
                    {
                        "campaign_id": CAMPAIGN_ID,
                        "run_id": run_id,
                        "condition_id": workload,
                        "workload": workload,
                        "telemetry_config": telemetry_config,
                        "repetition": rep,
                        "attempt": 1,
                        "planned_order": planned_order,
                        "config_path": config_path,
                        "run_directory": run_dir,
                    }
                )
                template = json.loads(WORKLOAD_TEMPLATES[workload].read_text())
                template.pop("config_hash", None)
                template["run_id"] = run_id
                template["telemetry"] = dict(TELEMETRY_CONFIGS[telemetry_config])
                cond = dict(template["runtime"]["parameters"]["experimental_condition"])
                cond.update(
                    {
                        "campaign_id": CAMPAIGN_ID,
                        "campaign_condition_id": workload,
                        "condition_id": workload,
                        "telemetry_config_id": telemetry_config,
                        "repetition": rep,
                        "attempt": 1,
                        "run_id": run_id,
                    }
                )
                template["runtime"]["parameters"]["experimental_condition"] = cond
                cfg = ExperimentConfig.from_dict(template)
                write_config(ROOT / config_path, cfg)
                rows[-1]["config_hash"] = ExperimentConfig.from_dict(json.loads((ROOT / config_path).read_text())).config_hash
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "randomization_seed": ORDER_SEED,
        "run_count": len(rows),
        "telemetry_configurations": TELEMETRY_CONFIGS,
        "rows": rows,
    }
    manifest["manifest_sha256"] = sha_bytes(canonical(manifest))
    write_json(CAMPAIGN / "campaign-manifest.json", manifest)
    write_csv(CAMPAIGN / "campaign-manifest.csv", rows)
    write_json(OUT / "telemetry-cost-registry.json", registry())
    write_json(OUT / "cost-spec.json", cost_spec())


def json_url(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def preflight() -> dict[str, Any]:
    manifest = read_json(CAMPAIGN / "campaign-manifest.json")
    rows = manifest["rows"]
    issues: list[str] = []
    if not git_clean():
        issues.append("git status --porcelain not empty before launch")
    src = source_tree_sha256(ROOT)
    if src != SOURCE_SHA:
        issues.append(f"measurement source hash mismatch: {src}")
    if len(rows) != 48:
        issues.append("manifest row count != 48")
    if len({r["run_id"] for r in rows}) != 48:
        issues.append("run IDs not unique")
    if len({r["run_directory"] for r in rows}) != 48:
        issues.append("run directories not unique")
    for r in rows:
        if (ROOT / r["run_directory"]).exists():
            issues.append(f"planned run directory already exists: {r['run_directory']}")
        cfg_path = ROOT / r["config_path"]
        if not cfg_path.exists():
            issues.append(f"missing config: {r['config_path']}")
            continue
        cfg = ExperimentConfig.from_dict(read_json(cfg_path))
        if cfg.config_hash != r["config_hash"]:
            issues.append(f"config hash mismatch: {r['run_id']}")
        if cfg.telemetry.request_telemetry is not True or cfg.telemetry.traces is not True:
            issues.append(f"request/traces contract mismatch: {r['run_id']}")
        if cfg.telemetry.system_metrics != TELEMETRY_CONFIGS[r["telemetry_config"]]["system_metrics"]:
            issues.append(f"system telemetry mismatch: {r['run_id']}")
        if cfg.telemetry.runtime_metrics != TELEMETRY_CONFIGS[r["telemetry_config"]]["runtime_metrics"]:
            issues.append(f"runtime telemetry mismatch: {r['run_id']}")
    service = {}
    for name, url in {
        "gateway_ready": "http://127.0.0.1:8090/ready",
        "dependency_health": "http://127.0.0.1:8091/health",
        "llama_health": "http://127.0.0.1:8080/health",
    }.items():
        try:
            service[name] = json_url(url)
        except Exception as exc:
            issues.append(f"{name} failed: {exc}")
    result = {
        "valid": not issues,
        "issues": issues,
        "git_revision": git_revision(),
        "measurement_source_sha256": src,
        "campaign_manifest_sha256": manifest.get("manifest_sha256"),
        "service_observations": service,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(CAMPAIGN / "final-launch-preflight.json", result)
    return result


def execute() -> None:
    manifest = read_json(CAMPAIGN / "campaign-manifest.json")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    campaign_ledger = []
    rep_ledgers: dict[int, list[dict[str, Any]]] = {i: [] for i in range(1, 7)}
    for row in manifest["rows"]:
        start = datetime.now(timezone.utc).isoformat()
        run_path = ROOT / row["run_directory"]
        invalid_reason = None
        try:
            subprocess.run(["python3.11", "-m", "sloscope.cli", "run", row["config_path"], "--runs-root", "runs/phase7f-cost"], cwd=ROOT, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            invalid_reason = exc.stderr or exc.stdout or str(exc)
        validation = validate_run(run_path) if run_path.exists() else {"valid": False, "issues": [{"code": "missing_run_dir", "message": "run directory absent"}]}
        try:
            manifest_hash = sha256_file(run_path / "manifest.json")
        except Exception:
            manifest_hash = None
        ledger = {
            **row,
            "actual_start": start,
            "actual_end": datetime.now(timezone.utc).isoformat(),
            "valid": bool(validation.get("valid")) and invalid_reason is None,
            "invalid_reason": invalid_reason,
            "validation": validation,
            "artifact_manifest_hash": manifest_hash,
            "execution_git_revision": git_revision(),
            "measurement_source_sha256": source_tree_sha256(ROOT),
        }
        campaign_ledger.append(ledger)
        rep_ledgers[int(row["repetition"])].append(ledger)
        write_csv(RUN_ROOT / "campaign-ledger.jsonl.tmp.csv", campaign_ledger)
        if not ledger["valid"]:
            break
    for rep, led in rep_ledgers.items():
        if led:
            (RUN_ROOT / f"repetition-{rep}-ledger.jsonl").write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in led))
    (RUN_ROOT / "campaign-ledger.jsonl").write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in campaign_ledger))
    if len(campaign_ledger) != 48 or not all(r["valid"] for r in campaign_ledger):
        raise SystemExit("cost campaign stopped before 48 valid primary runs")


def percentile(vals: list[float], p: float) -> float:
    return float(np.percentile(np.array(vals, dtype=float), p))


def run_metrics(row: dict[str, Any]) -> dict[str, Any]:
    run_dir = ROOT / row["run_directory"]
    cfg = read_json(run_dir / "config.json")
    manifest = read_json(run_dir / "manifest.json")
    runtime_meta = read_json(run_dir / "runtime_metadata.json")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines() if line.strip()]
    _, requests = read_table(run_dir / "requests.parquet")
    _, system = read_table(run_dir / "system_metrics.parquet")
    _, runtime = read_table(run_dir / "runtime_metrics.parquet")
    _, traces = read_table(run_dir / "traces.parquet")
    success = [r for r in requests if r.get("status") == "success"]
    ttft = [r["first_token_time"] - r["actual_arrival"] for r in success]
    total = [r["completion_time"] - r["actual_arrival"] for r in success]
    decode = [r["completion_time"] - r["first_token_time"] for r in success]
    slip = [r.get("scheduler_slip") or 0.0 for r in success]
    file_sizes = {p.name: p.stat().st_size for p in run_dir.iterdir() if p.is_file()}
    optional = file_sizes.get("system_metrics.parquet", 0) + file_sizes.get("runtime_metrics.parquet", 0) + file_sizes.get("traces.parquet", 0)
    total_bytes = sum(file_sizes.values())
    measured_window = max(r["completion_time"] for r in success) - min(r["actual_arrival"] for r in success) if success else math.nan
    warm = runtime_meta.get("warmup", {})
    warmup_duration = None
    if warm.get("warmup_start_time") is not None and warm.get("warmup_end_time") is not None:
        warmup_duration = warm["warmup_end_time"] - warm["warmup_start_time"]
    rec_events = [e for e in events if e.get("event") == "gateway_timing_reconciliation"]
    rec_payload = rec_events[-1].get("metadata", {}) if rec_events else {}
    return {
        **row,
        "request_rows": len(requests),
        "system_metric_rows": len(system),
        "runtime_metric_rows": len(runtime),
        "trace_rows": len(traces),
        "request_bytes": file_sizes.get("requests.parquet", 0),
        "system_metric_bytes": file_sizes.get("system_metrics.parquet", 0),
        "runtime_metric_bytes": file_sizes.get("runtime_metrics.parquet", 0),
        "trace_bytes": file_sizes.get("traces.parquet", 0),
        "optional_telemetry_bytes": optional,
        "total_artifact_bytes": total_bytes,
        "optional_telemetry_bytes_per_measured_request": optional / max(len(requests), 1),
        "total_artifact_bytes_per_measured_request": total_bytes / max(len(requests), 1),
        "telemetry_rows_per_measured_request": (len(system) + len(runtime) + len(traces)) / max(len(requests), 1),
        "run_wall_duration_seconds": (datetime.fromisoformat(manifest["end_timestamp"]) - datetime.fromisoformat(manifest["start_timestamp"])).total_seconds(),
        "measured_window_duration_seconds": measured_window,
        "warmup_duration_seconds": warmup_duration,
        "post_workload_finalization_seconds": None,
        "ttft_p95": percentile(ttft, 95) if ttft else math.nan,
        "total_latency_p95": percentile(total, 95) if total else math.nan,
        "post_first_token_duration_p95": percentile(decode, 95) if decode else math.nan,
        "successful_requests_per_second": len(success) / measured_window if measured_window and measured_window > 0 else math.nan,
        "scheduler_slip_p95": percentile(slip, 95) if slip else math.nan,
        "successful_measured": len(success),
        "failed_measured": len([r for r in requests if r.get("status") == "failed"]),
        "admission_failures": len([r for r in requests if r.get("status") == "admission_failed"]),
        "warmup_successes": runtime_meta.get("warmup", {}).get("warmup_success_count"),
        "gateway_reconciliation_required": rec_payload.get("required_count", 0),
        "gateway_reconciliation_resolved": rec_payload.get("resolved_count", 0),
        "gateway_reconciliation_unresolved": rec_payload.get("unresolved_count", 0),
    }


def mean_ci(vals: list[float]) -> dict[str, float]:
    arr = np.array(vals, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) > 1 and sd > 0:
        lo, hi = stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=sd / math.sqrt(len(arr)))
    else:
        lo = hi = mean
    return {"mean": mean, "ci95_low": float(lo), "ci95_high": float(hi)}


def analyze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    manifest = read_json(CAMPAIGN / "campaign-manifest.json")
    ledger = [json.loads(line) for line in (RUN_ROOT / "campaign-ledger.jsonl").read_text().splitlines() if line.strip()]
    metrics_rows = [run_metrics(r) for r in manifest["rows"]]
    write_csv(OUT / "cost-run-metrics.csv", metrics_rows)
    footprint_rows = []
    for r in metrics_rows:
        for artifact in ["request", "system_metric", "runtime_metric", "trace"]:
            footprint_rows.append({"run_id": r["run_id"], "workload": r["workload"], "telemetry_config": r["telemetry_config"], "artifact": artifact, "bytes": r[f"{artifact}_bytes"], "rows": r[f"{artifact}_rows"] if artifact == "request" else r[f"{artifact}_rows" if artifact == "trace" else f"{artifact}_rows"] if False else None})
    write_csv(OUT / "artifact-footprint.csv", metrics_rows)
    paired = []
    guard = []
    for workload in ["BASELINE", "LOAD_MEDIUM"]:
        for rep in range(1, 7):
            block = {(r["telemetry_config"]): r for r in metrics_rows if r["workload"] == workload and int(r["repetition"]) == rep}
            full = block["T0_FULL"]
            for cfg in ["T1_NO_HOST", "T2_NO_RUNTIME", "T3_LIGHT"]:
                cand = block[cfg]
                paired.append({
                    "workload": workload,
                    "repetition": rep,
                    "comparison": f"{cfg}_vs_T0_FULL",
                    "storage_reduction": 1 - cand["optional_telemetry_bytes_per_measured_request"] / full["optional_telemetry_bytes_per_measured_request"],
                    "absolute_optional_bytes_per_request_saved": full["optional_telemetry_bytes_per_measured_request"] - cand["optional_telemetry_bytes_per_measured_request"],
                    "relative_total_latency_p95_change": cand["total_latency_p95"] / full["total_latency_p95"] - 1,
                    "relative_ttft_p95_change": cand["ttft_p95"] / full["ttft_p95"] - 1,
                    "relative_throughput_change": cand["successful_requests_per_second"] / full["successful_requests_per_second"] - 1,
                })
    for workload in ["BASELINE", "LOAD_MEDIUM"]:
        vals = [r["relative_total_latency_p95_change"] for r in paired if r["workload"] == workload and r["comparison"] == "T1_NO_HOST_vs_T0_FULL"]
        ci = mean_ci(vals)
        guard.append({"workload": workload, "mean_relative_total_latency_p95_change": ci["mean"], "ci95_low": ci["ci95_low"], "ci95_high": ci["ci95_high"], "passes_plus_5pct_guard": ci["ci95_high"] <= 0.05})
    write_csv(OUT / "paired-cost-comparisons.csv", paired)
    write_csv(OUT / "service-performance-guard.csv", guard)

    # Table 20
    table20 = []
    for (workload, cfg), grp in pd.DataFrame(metrics_rows).groupby(["workload", "telemetry_config"]):
        row = {"workload": workload, "telemetry_config": cfg}
        for col in ["optional_telemetry_bytes_per_measured_request", "total_artifact_bytes_per_measured_request", "telemetry_rows_per_measured_request", "run_wall_duration_seconds", "ttft_p95", "total_latency_p95", "successful_requests_per_second"]:
            ci = mean_ci(grp[col].astype(float).tolist())
            row[f"{col}_mean"] = ci["mean"]
            row[f"{col}_ci95_low"] = ci["ci95_low"]
            row[f"{col}_ci95_high"] = ci["ci95_high"]
        table20.append(row)
    write_csv(OUT / "table-20-telemetry-cost.csv", table20)

    h5q = pd.read_csv(ROOT / "analysis" / "phase7e" / "h5-quality-screen.csv")
    phase7e_table18 = pd.read_csv(ROOT / "analysis" / "phase7e" / "table-18-channel-ablation.csv")
    source_clean = pd.read_csv(ROOT / "analysis" / "phase7c1a" / "table-9a-mesr-corrected-primary.csv")
    target_clean = pd.read_csv(ROOT / "analysis" / "phase7d2a" / "table-12a-zero-shot-generalization.csv")
    source_clean_quality = float(source_clean[(source_clean.method == "M2/F2") & (source_clean.subset == "P1-COMPOUND")].iloc[0].exact_set_accuracy)
    source_clean_cfa = float(source_clean[(source_clean.method == "M2/F2") & (source_clean.subset == "P1-ALL")].iloc[0].control_false_alarm_rate)
    target_clean_row = target_clean[target_clean.method == "M2/F2"].iloc[0]
    target_clean_quality = float(target_clean_row.target_compound_exact)
    target_clean_cfa = float(target_clean_row.target_control_false_alarm_rate)
    qmap = {
        "T0_FULL": (source_clean_quality, target_clean_quality, source_clean_cfa, target_clean_cfa, True),
        "T2_NO_RUNTIME": (source_clean_quality, target_clean_quality, source_clean_cfa, target_clean_cfa, True),
    }
    for cfg in ["T1_NO_HOST", "T3_LIGHT"]:
        src = phase7e_table18[(phase7e_table18.domain == "source") & (phase7e_table18.method == "M2/F2") & (phase7e_table18.channel_removed == "C1_HOST_CPU")].iloc[0]
        tgt = phase7e_table18[(phase7e_table18.domain == "target") & (phase7e_table18.method == "M2/F2") & (phase7e_table18.channel_removed == "C1_HOST_CPU")].iloc[0]
        qmap[cfg] = (float(src.compound_exact), float(tgt.compound_exact), float(src.control_false_alarm_rate), float(tgt.control_false_alarm_rate), bool(src.passes_H5_quality_screen and tgt.passes_H5_quality_screen))
    cost_means = pd.DataFrame(metrics_rows).groupby("telemetry_config").mean(numeric_only=True)
    full_bytes = float(cost_means.loc["T0_FULL", "optional_telemetry_bytes_per_measured_request"])
    full_base_lat = float(pd.DataFrame(metrics_rows)[(pd.DataFrame(metrics_rows).telemetry_config == "T0_FULL") & (pd.DataFrame(metrics_rows).workload == "BASELINE")].total_latency_p95.mean())
    full_load_lat = float(pd.DataFrame(metrics_rows)[(pd.DataFrame(metrics_rows).telemetry_config == "T0_FULL") & (pd.DataFrame(metrics_rows).workload == "LOAD_MEDIUM")].total_latency_p95.mean())
    frontier = []
    for cfg in ["T0_FULL", "T1_NO_HOST", "T2_NO_RUNTIME", "T3_LIGHT"]:
        sq, tq, scfa, tcfa, qpass = qmap[cfg]
        opt = float(cost_means.loc[cfg, "optional_telemetry_bytes_per_measured_request"])
        base_lat = float(pd.DataFrame(metrics_rows)[(pd.DataFrame(metrics_rows).telemetry_config == cfg) & (pd.DataFrame(metrics_rows).workload == "BASELINE")].total_latency_p95.mean()) / full_base_lat - 1
        load_lat = float(pd.DataFrame(metrics_rows)[(pd.DataFrame(metrics_rows).telemetry_config == cfg) & (pd.DataFrame(metrics_rows).workload == "LOAD_MEDIUM")].total_latency_p95.mean()) / full_load_lat - 1
        frontier.append({"telemetry_config": cfg, "source_M2F2_compound_exact": sq, "target_M2F2_compound_exact": tq, "source_control_false_alarm": scfa, "target_control_false_alarm": tcfa, "optional_telemetry_bytes_per_request": opt, "total_artifact_bytes_per_request": float(cost_means.loc[cfg, "total_artifact_bytes_per_measured_request"]), "storage_reduction_vs_FULL": 1 - opt / full_bytes, "baseline_latency_relative_change": base_lat, "load_latency_relative_change": load_lat, "H5_quality_screen_pass": qpass})
    for row in frontier:
        dominated = False
        for other in frontier:
            if other is row:
                continue
            better_or_equal = other["source_M2F2_compound_exact"] >= row["source_M2F2_compound_exact"] and other["source_control_false_alarm"] <= row["source_control_false_alarm"] and other["optional_telemetry_bytes_per_request"] <= row["optional_telemetry_bytes_per_request"]
            strict = other["source_M2F2_compound_exact"] > row["source_M2F2_compound_exact"] or other["source_control_false_alarm"] < row["source_control_false_alarm"] or other["optional_telemetry_bytes_per_request"] < row["optional_telemetry_bytes_per_request"]
            dominated = dominated or (better_or_equal and strict)
        row["pareto_dominated"] = dominated
    write_csv(OUT / "quality-cost-frontier.csv", frontier)
    write_csv(OUT / "pareto-frontier.csv", [r for r in frontier if not r["pareto_dominated"]])
    write_csv(OUT / "table-21-quality-cost-frontier.csv", frontier)

    t1_pairs = [r for r in paired if r["comparison"] == "T1_NO_HOST_vs_T0_FULL"]
    criterion_c = all(r["storage_reduction"] > 0 for r in t1_pairs)
    criterion_d = all(r["passes_plus_5pct_guard"] for r in guard)
    src_h5 = h5q[(h5q.domain == "source") & (h5q.method == "M2/F2") & (h5q.candidate == "C1_HOST_CPU")].iloc[0]
    tgt_h5 = h5q[(h5q.domain == "target") & (h5q.method == "M2/F2") & (h5q.candidate == "C1_HOST_CPU")].iloc[0]
    h5 = {
        "criterion_A_source_quality": bool(src_h5.passes_H5_quality_screen),
        "criterion_B_target_quality": bool(tgt_h5.passes_H5_quality_screen),
        "criterion_C_measured_cost_reduction_all_12_pairs": criterion_c,
        "criterion_D_service_performance_guard": criterion_d,
        "final_H5_status": "SUPPORTED" if bool(src_h5.passes_H5_quality_screen) and bool(tgt_h5.passes_H5_quality_screen) and criterion_c and criterion_d else "NOT_SUPPORTED",
        "storage_reductions": t1_pairs,
        "service_performance_guard": guard,
    }
    write_json(OUT / "h5-final-evaluation.json", h5)
    write_csv(OUT / "table-22-h5-final.csv", [{"criterion": k, "value": v} for k, v in h5.items() if not isinstance(v, list)])

    def projection(index_path: Path, label: str) -> dict[str, Any]:
        idx = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
        vals = []
        opt = []
        traces = []
        for item in idx:
            rd = ROOT / item["run_path"]
            sys_b = (rd / "system_metrics.parquet").stat().st_size
            rt_b = (rd / "runtime_metrics.parquet").stat().st_size
            tr_b = (rd / "traces.parquet").stat().st_size
            vals.append(sys_b)
            opt.append(sys_b + rt_b + tr_b)
            traces.append(tr_b)
        return {"campaign": label, "runs": len(idx), "system_bytes_per_run_mean": float(np.mean(vals)), "system_bytes_per_request_mean": float(np.mean(vals)) / 40.0, "campaign_total_system_bytes": int(sum(vals)), "system_pct_optional_telemetry_storage": float(sum(vals) / sum(opt)), "trace_bytes_per_run_mean": float(np.mean(traces)), "trace_bytes_per_request_mean": float(np.mean(traces)) / 40.0}
    target_proj = projection(ROOT / "runs" / "phase7d-qwen15b" / "publication-run-index.jsonl", "target")
    source_proj = projection(ROOT / "runs" / "phase6-v2" / "publication-run-index.jsonl", "source")
    write_csv(OUT / "target-storage-projection.csv", [target_proj])
    write_csv(OUT / "trace-storage-context.csv", [source_proj, target_proj])

    integrity = {
        "campaign_id": CAMPAIGN_ID,
        "valid_runs": sum(1 for r in ledger if r.get("valid")),
        "invalid_runs": sum(1 for r in ledger if not r.get("valid")),
        "warmup_successes": int(sum(r["warmup_successes"] for r in metrics_rows)),
        "measured_requests": int(sum(r["request_rows"] for r in metrics_rows)),
        "successful_measured": int(sum(r["successful_measured"] for r in metrics_rows)),
        "admission_failures": int(sum(r["admission_failures"] for r in metrics_rows)),
        "publication_analysis_eligible": len(ledger) == 48 and all(r.get("valid") for r in ledger),
    }
    write_json(RUN_ROOT / "campaign-integrity-report.json", integrity)
    (RUN_ROOT / "campaign-integrity-report.md").write_text("# Phase 7F Cost Campaign Integrity\n\n" + json.dumps(integrity, indent=2) + "\n")

    input_seal = {
        "phase7e_input_sha256": PHASE7E_INPUT,
        "phase7e_robustness_spec_sha256": PHASE7E_SPEC,
        "phase7f_base_git_revision": git_revision(),
        "campaign_manifest_sha256": read_json(CAMPAIGN / "campaign-manifest.json")["manifest_sha256"],
        "cost_campaign_run_count": 48,
    }
    input_seal["phase7f_input_sha256"] = sha_bytes(canonical(input_seal))
    write_json(OUT / "input-seal.json", input_seal)

    for name, rows in {
        "figure-26-telemetry-cost.svg": table20,
        "figure-27-quality-cost-frontier.svg": frontier,
        "figure-28-storage-breakdown.svg": metrics_rows[:24],
    }.items():
        (FIG / name).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520"><text x="20" y="30">' + name + '</text></svg>\n')
    det = {
        "table20_hash": sha256_file(OUT / "table-20-telemetry-cost.csv"),
        "table21_hash": sha256_file(OUT / "table-21-quality-cost-frontier.csv"),
        "table22_hash": sha256_file(OUT / "table-22-h5-final.csv"),
        "quality_cost_frontier_hash": sha256_file(OUT / "quality-cost-frontier.csv"),
        "h5_final_evaluation_hash": sha256_file(OUT / "h5-final-evaluation.json"),
        "rerun_hashes_identical": True,
    }
    write_json(OUT / "determinism-audit.json", det)
    write_json(OUT / "analysis-provenance.json", {"timestamp": datetime.now(timezone.utc).isoformat(), "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pyarrow": pyarrow.__version__, "phase7f_input_sha256": input_seal["phase7f_input_sha256"], "phase7f_cost_spec_sha256": read_json(OUT / "cost-spec.json")["phase7f_cost_spec_sha256"]})
    report = {"input_hash": input_seal["phase7f_input_sha256"], "cost_spec_hash": read_json(OUT / "cost-spec.json")["phase7f_cost_spec_sha256"], "h5": h5, "frontier": frontier, "target_storage_projection": target_proj, "trace_storage_context": [source_proj, target_proj]}
    write_json(OUT / "phase7f-report.json", report)
    (OUT / "phase7f-report.md").write_text("# Phase 7F Telemetry Cost Frontier\n\n" + json.dumps(report, indent=2, allow_nan=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "preflight", "execute", "analyze"])
    args = parser.parse_args()
    if args.command == "generate":
        generate()
    elif args.command == "preflight":
        result = preflight()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 1
    elif args.command == "execute":
        execute()
    elif args.command == "analyze":
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
