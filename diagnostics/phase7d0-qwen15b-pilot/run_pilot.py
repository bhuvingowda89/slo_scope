from __future__ import annotations

import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DIAG = ROOT / "diagnostics" / "phase7d0-qwen15b-pilot"
CONFIG_DIR = DIAG / "configs"
RUNS_ROOT = DIAG / "runs"
LOG_DIR = DIAG / "logs"

SOURCE_SHA = "8e442c813619586072c27a4ec30d7e65d367c3fc1375922935f67c72e2122582"
FREEZE_SHA = "65d9798ea033bc910a832c624324fe068e48552b2a16e6ed639f33763ca028e0"
MODEL_REF = "Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M"
MODEL_ALIAS = "sloscope-qwen2.5-1.5b"
PILOT_SEED = 7319
CONDITIONS = [
    ("BASELINE", "configs/phase5/baseline.yaml"),
    ("OUTPUT_CONTROL", "configs/phase5/output-control.yaml"),
    ("INPUT_MEDIUM", "configs/phase5/single-input.yaml"),
    ("OUTPUT_MEDIUM", "configs/phase5/single-output.yaml"),
    ("LOAD_MEDIUM", "configs/phase5/single-load.yaml"),
    ("DOWNSTREAM_MEDIUM", "configs/phase5/single-downstream.yaml"),
    ("INPUT_LOAD", "configs/phase5/compound-input-load.yaml"),
    ("INPUT_DOWNSTREAM", "configs/phase5/compound-input-downstream.yaml"),
    ("OUTPUT_LOAD", "configs/phase5/compound-output-load.yaml"),
    ("OUTPUT_DOWNSTREAM", "configs/phase5/compound-output-downstream.yaml"),
    ("LOAD_DOWNSTREAM", "configs/phase5/compound-load-downstream.yaml"),
]


def url_json(url: str, timeout: float = 5.0, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if payload is not None else "GET", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def wait_json(url: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            return url_json(url, timeout=2.0)
        except Exception as exc:
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"service not ready: {url}: {last}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_configs() -> list[dict]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    order = list(CONDITIONS)
    random.Random(PILOT_SEED).shuffle(order)
    rows = []
    for idx, (condition, template_path) in enumerate(order, 1):
        cfg = json.loads((ROOT / template_path).read_text(encoding="utf-8"))
        run_id = f"diagnostic-phase7d0-qwen15b-{idx:02d}-{condition.lower().replace('_','-')}"
        cfg["run_id"] = run_id
        cfg.pop("config_hash", None)
        cfg["runtime"]["model_id"] = MODEL_ALIAS
        params = cfg["runtime"]["parameters"]
        params["model_ref"] = MODEL_REF
        params["warmup_request_count"] = 2
        params["server_arguments"] = [
            "-hf", MODEL_REF,
            "--alias", MODEL_ALIAS,
            "--host", "127.0.0.1",
            "--port", "8080",
            "--metrics",
            "--parallel", "2",
            "--ctx-size", "2048",
            "--no-cache-prompt",
        ]
        params["experimental_condition"]["campaign_id"] = "diagnostic-phase7d0-qwen15b-pilot"
        params["experimental_condition"]["run_id"] = run_id
        params["experimental_condition"]["repetition"] = 0
        params["experimental_condition"]["attempt"] = 1
        cfg["workload"]["request_count"] = 12
        # Keep scientific workload/rate/output/dependency settings otherwise unchanged.
        out = CONFIG_DIR / f"{run_id}.json"
        out.write_text(json.dumps(cfg, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        rows.append({"pilot_order": idx, "condition_id": condition, "run_id": run_id, "config_path": str(out.relative_to(ROOT)), "run_path": str((RUNS_ROOT / run_id).relative_to(ROOT))})
    write_json(DIAG / "pilot-manifest.json", {
        "schema_version": "phase7d0.pilot_manifest.v1",
        "campaign_id": "diagnostic-phase7d0-qwen15b-pilot",
        "pilot_order_seed": PILOT_SEED,
        "model_ref": MODEL_REF,
        "model_alias": MODEL_ALIAS,
        "warmup_requests_per_run": 2,
        "measured_requests_per_run": 12,
        "conditions": rows,
        "source_tree_sha256": SOURCE_SHA,
        "campaign_freeze_sha256": FREEZE_SHA,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return rows


def start_services() -> list[subprocess.Popen]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    dep_log = (LOG_DIR / "dependency.log").open("wb")
    procs.append(subprocess.Popen([sys.executable, "-m", "sloscope.gateway.dependency", "--host", "127.0.0.1", "--port", "8091", "--maximum-delay-ms", "500"], cwd=ROOT, env=env, stdout=dep_log, stderr=subprocess.STDOUT))
    wait_json("http://127.0.0.1:8091/health", timeout=10)
    llama_log = (LOG_DIR / "llama-server.log").open("wb")
    llama_cmd = ["llama-server", "-hf", MODEL_REF, "--alias", MODEL_ALIAS, "--host", "127.0.0.1", "--port", "8080", "--metrics", "--parallel", "2", "--ctx-size", "2048", "--no-cache-prompt"]
    procs.append(subprocess.Popen(llama_cmd, cwd=ROOT, stdout=llama_log, stderr=subprocess.STDOUT))
    wait_json("http://127.0.0.1:8080/health", timeout=240)
    gw_log = (LOG_DIR / "gateway.log").open("wb")
    procs.append(subprocess.Popen([sys.executable, "-m", "sloscope.gateway.server", "--host", "127.0.0.1", "--port", "8090", "--llama-base-url", "http://127.0.0.1:8080", "--dependency-base-url", "http://127.0.0.1:8091", "--timeout", "120"], cwd=ROOT, env=env, stdout=gw_log, stderr=subprocess.STDOUT))
    wait_json("http://127.0.0.1:8090/health", timeout=30)
    return procs


def stop_services(procs: list[subprocess.Popen]) -> None:
    try:
        url_json("http://127.0.0.1:8091/control/delay", timeout=2.0, payload={"delay_ms": 0})
    except Exception:
        pass
    for proc in reversed(procs):
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10
    for proc in reversed(procs):
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()


def run_pilot(rows: list[dict]) -> list[dict]:
    ledger = []
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for row in rows:
        started = datetime.now(timezone.utc).isoformat()
        run_dir = RUNS_ROOT / row["run_id"]
        if run_dir.exists():
            result = subprocess.CompletedProcess([], 0, stdout=json.dumps({"run_dir": str(run_dir), "resumed_existing": True}), stderr="")
        else:
            cmd = [sys.executable, "-m", "sloscope.cli", "run", str(ROOT / row["config_path"]), "--runs-root", str(RUNS_ROOT)]
            result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=600)
        ended = datetime.now(timezone.utc).isoformat()
        validation = {}
        runtime_metadata = {}
        request_rows = []
        traces = []
        if run_dir.exists():
            validation_path = run_dir / "validation.json"
            if validation_path.exists():
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
            metadata_path = run_dir / "runtime_metadata.json"
            if metadata_path.exists():
                runtime_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            from sloscope.artifacts.writer import read_table
            if (run_dir / "requests.parquet").exists():
                _, request_rows = read_table(run_dir / "requests.parquet")
            if (run_dir / "traces.parquet").exists():
                _, traces = read_table(run_dir / "traces.parquet")
        measured_successes = sum(1 for r in request_rows if r.get("status") == "success")
        admission_failures = sum(1 for r in request_rows if r.get("status") == "admission_failed")
        recon = runtime_metadata.get("gateway_timing_reconciliation", {})
        warmup = runtime_metadata.get("warmup", {})
        out_tokens = [r.get("server_output_tokens") for r in request_rows if r.get("server_output_tokens") is not None]
        pilot_valid = bool(validation.get("valid")) and len(request_rows) == 12 and measured_successes == 12 and admission_failures == 0 and recon.get("unresolved_count", 0) == 0
        entry = {**row, "started": started, "ended": ended, "process_returncode": result.returncode, "stdout": result.stdout.strip(), "stderr_tail": result.stderr[-2000:], "validation_valid": validation.get("valid"), "validation_issues": validation.get("issues", []), "pilot_valid": pilot_valid, "measured_accounted": len(request_rows), "measured_successes": measured_successes, "admission_failures": admission_failures, "trace_rows": len(traces), "warmup_success_count": warmup.get("warmup_success_count"), "warmup_failure_count": warmup.get("warmup_failure_count"), "reconciliation_required": recon.get("required_count", 0), "reconciliation_resolved": recon.get("resolved_count", 0), "reconciliation_unresolved": recon.get("unresolved_count", 0), "min_output_tokens": min(out_tokens) if out_tokens else None, "output_control_ok": None}
        if row["condition_id"] in {"OUTPUT_MEDIUM", "OUTPUT_LOAD", "OUTPUT_DOWNSTREAM"}:
            entry["output_control_ok"] = sum(1 for v in out_tokens if v >= 29) >= 11
            pilot_valid = pilot_valid and entry["output_control_ok"]
            entry["pilot_valid"] = pilot_valid
        ledger.append(entry)
        with (DIAG / "pilot-ledger.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        if not pilot_valid:
            write_json(DIAG / "pilot-failure-report.json", {"failed_entry": entry, "failure_time": datetime.now(timezone.utc).isoformat()})
            break
    return ledger


def main() -> int:
    for path in [DIAG / "pilot-ledger.jsonl", DIAG / "pilot-failure-report.json"]:
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing diagnostic artifact: {path}")
    rows = generate_configs()
    procs: list[subprocess.Popen] = []
    try:
        procs = start_services()
        runtime = url_json("http://127.0.0.1:8080/props", timeout=5.0)
        models = url_json("http://127.0.0.1:8080/v1/models", timeout=5.0)
        write_json(DIAG / "target-runtime-metadata.json", {"props": runtime, "server_args": ["-hf", MODEL_REF, "--alias", MODEL_ALIAS, "--host", "127.0.0.1", "--port", "8080", "--metrics", "--parallel", "2", "--ctx-size", "2048", "--no-cache-prompt"]})
        write_json(DIAG / "target-model-metadata.json", {"models": models, "model_ref": MODEL_REF, "model_alias": MODEL_ALIAS})
        ledger = run_pilot(rows)
    finally:
        stop_services(procs)
    totals = {
        "intended_runs": len(rows),
        "executed_runs": len(ledger),
        "valid_runs": sum(1 for r in ledger if r.get("pilot_valid")),
        "invalid_runs": sum(1 for r in ledger if not r.get("pilot_valid")),
        "measured_successes": sum(r.get("measured_successes", 0) for r in ledger),
        "admission_failures": sum(r.get("admission_failures", 0) for r in ledger),
        "reconciliation_required": sum(r.get("reconciliation_required", 0) for r in ledger),
        "reconciliation_resolved": sum(r.get("reconciliation_resolved", 0) for r in ledger),
        "reconciliation_unresolved": sum(r.get("reconciliation_unresolved", 0) for r in ledger),
    }
    with (DIAG / "pilot-summary.csv").open("w", newline="", encoding="utf-8") as fh:
        if ledger:
            writer = csv.DictWriter(fh, fieldnames=list(ledger[0].keys()))
            writer.writeheader()
            writer.writerows(ledger)
    report = {"schema_version": "phase7d0.pilot_report.v1", "totals": totals, "exact_source_settings_operationally_viable": totals["executed_runs"] == len(rows) and totals["invalid_runs"] == 0, "generated_at": datetime.now(timezone.utc).isoformat()}
    write_json(DIAG / "pilot-report.json", report)
    (DIAG / "pilot-report.md").write_text("# Phase 7D.0 Qwen-1.5B Pilot\n\n" + json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["exact_source_settings_operationally_viable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
