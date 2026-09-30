from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sloscope.artifacts.validation import validate_run
from sloscope.artifacts.writer import git_revision, read_json, read_table, write_json, write_jsonl
from sloscope.config import ExperimentConfig, MechanismConfig, RuntimeConfig, TelemetryConfig, WorkloadConfig
from sloscope.ground_truth import active_mechanism_set
from sloscope.runner import ExperimentRunner
from sloscope.telemetry.system import SystemTelemetryCollector
from sloscope.lifecycle import SystemClock


PILOT_ID = "phase4a-calibration"
RUN_ORDER_SEED = 4242
MODEL_REF = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
MODEL_ID = "sloscope-qwen2.5-0.5b"
LLAMA_VERSION = "0.5.0"
LLAMA_BUILD = "11146"
LLAMA_REVISION = "7fe450e19"
LLAMA_ARGS = [
    "-hf",
    MODEL_REF,
    "--alias",
    MODEL_ID,
    "--host",
    "127.0.0.1",
    "--port",
    "8080",
    "--metrics",
    "--parallel",
    "2",
    "--ctx-size",
    "2048",
    "--no-cache-prompt",
]


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    condition: str
    mechanism_type: str
    magnitude_value: float | int | None
    magnitude_unit: str | None
    repetition: int
    planned_order: int


@dataclass
class ServiceHandle:
    name: str
    process: subprocess.Popen
    log_path: Path
    port: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_request(method: str, url: str, payload: dict | None = None, timeout: float = 5.0) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return int(resp.status), json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload2 = json.loads(body)
        except Exception:
            payload2 = {"error": body}
        return int(exc.code), payload2


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def choose_port(preferred: int) -> int:
    if port_is_free(preferred):
        return preferred
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_json(url: str, *, timeout_seconds: float = 180.0, expect_ok: bool = True) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last: Any = None
    while time.monotonic() < deadline:
        try:
            status, payload = json_request("GET", url, timeout=2.0)
            last = {"status": status, "payload": payload}
            if (not expect_ok) or status == 200:
                return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for {url}: {last}")


def parse_llama_version() -> dict:
    result = subprocess.run(["llama-server", "--version"], text=True, capture_output=True, check=True)
    text = result.stdout + result.stderr
    required = [f"version: {LLAMA_VERSION}", f"build {LLAMA_BUILD}", f"commit {LLAMA_REVISION}", "Darwin arm64"]
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"llama-server version changed; missing {missing}; output={text!r}")
    return {
        "version": LLAMA_VERSION,
        "build": LLAMA_BUILD,
        "commit": LLAMA_REVISION,
        "target": "Darwin arm64",
        "raw": text.strip(),
    }


def start_service(name: str, cmd: list[str], log_path: Path, port: int) -> ServiceHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("ab")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=Path.cwd())
    return ServiceHandle(name, proc, log_path, port)


def terminate_services(services: Iterable[ServiceHandle]) -> list[dict]:
    results: list[dict] = []
    for service in services:
        proc = service.process
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        results.append({"name": service.name, "pid": proc.pid, "returncode": proc.returncode, "port": service.port})
    return results


def build_run_order() -> list[PlannedRun]:
    rng = random.Random(RUN_ORDER_SEED)
    cpu = [
        ("cpu", value, rep)
        for value in (0.20, 0.35, 0.50)
        for rep in (1, 2)
    ]
    downstream = [
        ("downstream", value, rep)
        for value in (50, 100, 200)
        for rep in (1, 2)
    ]
    rng.shuffle(cpu)
    rng.shuffle(downstream)
    planned: list[PlannedRun] = []
    order = 1
    for idx in range(6):
        block: list[tuple[str, float | int | None, int]] = [("healthy", None, idx + 1), cpu[idx], downstream[idx]]
        rng.shuffle(block)
        for kind, value, rep in block:
            if kind == "healthy":
                run_id = f"phase4a-healthy-r{rep:02d}"
                planned.append(PlannedRun(run_id, "healthy", "healthy", None, None, rep, order))
            elif kind == "cpu":
                run_id = f"phase4a-cpu-{int(round(float(value) * 100)):03d}-r{rep:02d}"
                planned.append(PlannedRun(run_id, "cpu", "cpu_contention", value, "logical_cpu_worker_fraction", rep, order))
            else:
                run_id = f"phase4a-downstream-{int(value)}ms-r{rep:02d}"
                planned.append(PlannedRun(run_id, "downstream", "downstream_latency", value, "ms", rep, order))
            order += 1
    return planned


def make_config(plan: PlannedRun, *, gateway_port: int, dependency_port: int, service_pids: dict[str, int], llama_version: dict) -> ExperimentConfig:
    mechanisms: list[MechanismConfig] = []
    safety_cpu = 0.0
    safety_dep = 250
    require_window = False
    if plan.condition == "cpu":
        safety_cpu = 0.50
        require_window = True
        mechanisms.append(
            MechanismConfig(
                "cpu-contention-01",
                "cpu_contention",
                float(plan.magnitude_value),
                "host_cpu",
                0.0,
                12.0,
                {"verification_interval_seconds": 0.30},
            )
        )
    elif plan.condition == "downstream":
        require_window = True
        mechanisms.append(
            MechanismConfig(
                f"downstream-latency-{int(plan.magnitude_value)}ms",
                "downstream_latency",
                0.0,
                "synthetic_dependency",
                0.0,
                12.0,
                {
                    "control_timeout_seconds": 2.0,
                    "delay_ms": int(plan.magnitude_value),
                    "dependency_base_url": f"http://127.0.0.1:{dependency_port}",
                    "verify_min_fraction": 0.8,
                    "verify_request": True,
                },
            )
        )
    data = {
        "schema_version": "sloscope.config.v1",
        "run_id": plan.run_id,
        "seed": 4242,
        "runtime": {
            "runtime_id": "llamacpp-local",
            "runtime_type": "llamacpp",
            "model_id": MODEL_ID,
            "parameters": {
                "base_url": f"http://127.0.0.1:{gateway_port}",
                "completion_parameters": {},
                "dependency_pid": service_pids["dependency"],
                "gateway_pid": service_pids["gateway"],
                "health_timeout": 5.0,
                "llama_cpp_build_number": LLAMA_BUILD,
                "llama_cpp_revision": LLAMA_REVISION,
                "max_outstanding_requests": 4,
                "metrics_enabled": True,
                "request_endpoint": "/v1/completions",
                "request_timeout": 60.0,
                "seed": 4242,
                "server_arguments": list(LLAMA_ARGS),
                "server_command": "llama-server",
                "server_metadata": {
                    "brew_package": f"llama.cpp {LLAMA_VERSION}",
                    "build": llama_version["build"],
                    "commit": llama_version["commit"],
                    "target": llama_version["target"],
                    "version": llama_version["version"],
                },
                "server_pid": service_pids["llama"],
                "telemetry_interval_seconds": 0.10,
                "temperature": 0,
            },
        },
        "workload": {
            "arrival_pattern": "constant_open_loop",
            "request_count": 20,
            "concurrency": 2,
            "prompt_profile": "synthetic-prose",
            "output_profile": "fixed",
            "target_output_tokens": 16,
            "start_offset_seconds": 1.0,
            "inter_arrival_seconds": 0.25,
            "burst_size": 2,
            "burst_interval_seconds": 1.0,
            "randomized_output": False,
            "output_jitter_tokens": 0,
        },
        "mechanisms": [m.__dict__ for m in mechanisms],
        "telemetry": {"request_telemetry": True, "system_metrics": True, "runtime_metrics": True, "traces": True},
        "safety": {
            "maximum_cpu_stress": safety_cpu,
            "maximum_allocated_pressure_memory": 0,
            "maximum_disk_io": 0,
            "maximum_dependency_delay_ms": safety_dep,
            "require_requests_within_mechanism_window": require_window,
            "mechanism_watchdog_interval_seconds": 0.25,
            "experiment_timeout": 180.0,
        },
    }
    cfg = ExperimentConfig.from_dict(data)
    cfg.validate()
    return cfg


def percentile(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return num / den


def summarize_run(run_dir: Path, plan: PlannedRun) -> dict:
    _, requests = read_table(run_dir / "requests.parquet")
    _, system = read_table(run_dir / "system_metrics.parquet")
    _, traces = read_table(run_dir / "traces.parquet")
    validation = read_json(run_dir / "validation.json")
    ground_truth = read_json(run_dir / "ground_truth.json")
    success = [r for r in requests if r["status"] == "success"]
    failed = [r for r in requests if r["status"] == "failed"]
    admission_failed = [r for r in requests if r["status"] == "admission_failed"]
    ttft = [r["first_token_time"] - r["actual_arrival"] for r in success if r.get("first_token_time") is not None]
    total = [r["completion_time"] - r["actual_arrival"] for r in success if r.get("completion_time") is not None]
    slip = [r["scheduler_slip"] for r in requests if r.get("scheduler_slip") is not None]
    dep = [r["dependency_duration"] for r in success if r.get("dependency_duration") is not None]
    worker_fraction = None
    worker_delta = None
    verification_probe = None
    for mech in ground_truth.get("mechanisms", []):
        ev = (mech.get("verification_evidence") or {}).get("evidence_value") or {}
        if mech.get("mechanism_type") == "cpu_contention":
            worker_fraction = ev.get("actual_worker_fraction")
            worker_delta = ev.get("total_worker_cpu_time_delta")
        if mech.get("mechanism_type") == "downstream_latency":
            verification_probe = ev.get("observed_dependency_request_duration")
    row = {
        "run_id": plan.run_id,
        "planned_order": plan.planned_order,
        "condition": plan.condition,
        "mechanism_type": plan.mechanism_type,
        "magnitude_value": plan.magnitude_value,
        "magnitude_unit": plan.magnitude_unit,
        "repetition": plan.repetition,
        "valid": validation.get("valid"),
        "issues": ";".join(issue.get("code", "") for issue in validation.get("issues", [])),
        "planned_requests": validation.get("planned_requests"),
        "accounted_requests": validation.get("accounted_requests"),
        "successful": len(success),
        "failed": len(failed),
        "admission_failed": len(admission_failed),
        "trace_count": len(traces),
        "active_mechanisms": ",".join(sorted(active_mechanism_set(ground_truth))),
        "ttft_median": median(ttft),
        "ttft_p95": percentile(ttft, 0.95),
        "total_latency_median": median(total),
        "total_latency_p95": percentile(total, 0.95),
        "scheduler_slip_median": median(slip),
        "scheduler_slip_p95": percentile(slip, 0.95),
        "dependency_duration_median": median(dep),
        "dependency_duration_p95": percentile(dep, 0.95),
        "host_cpu_median": median([r["host_cpu_percent"] for r in system if r.get("host_cpu_percent") is not None]),
        "host_cpu_p95": percentile([r["host_cpu_percent"] for r in system if r.get("host_cpu_percent") is not None], 0.95),
        "server_cpu_median": median([r["server_process_cpu_percent"] for r in system if r.get("server_process_cpu_percent") is not None]),
        "server_cpu_p95": percentile([r["server_process_cpu_percent"] for r in system if r.get("server_process_cpu_percent") is not None], 0.95),
        "gateway_cpu_median": median([r["gateway_process_cpu_percent"] for r in system if r.get("gateway_process_cpu_percent") is not None]),
        "gateway_cpu_p95": percentile([r["gateway_process_cpu_percent"] for r in system if r.get("gateway_process_cpu_percent") is not None], 0.95),
        "server_rss_median": median([r["server_process_rss_bytes"] for r in system if r.get("server_process_rss_bytes") is not None]),
        "server_rss_max": max([r["server_process_rss_bytes"] for r in system if r.get("server_process_rss_bytes") is not None], default=None),
        "worker_actual_fraction": worker_fraction,
        "worker_cpu_delta": worker_delta,
        "dependency_verification_probe": verification_probe,
    }
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def observe_idle(path: Path, service_pids: dict[str, int]) -> dict:
    collector = SystemTelemetryCollector(
        SystemClock(),
        server_pid=service_pids["llama"],
        gateway_pid=service_pids["gateway"],
        dependency_pid=service_pids["dependency"],
    )
    samples = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        samples.append(collector.sample())
        time.sleep(0.10)
    summary = {
        "sample_count": len(samples),
        "host_cpu_median": median([s["host_cpu_percent"] for s in samples]),
        "host_memory_median": median([s["host_memory_percent"] for s in samples]),
        "server_cpu_median": median([s["server_process_cpu_percent"] for s in samples if s.get("server_process_cpu_percent") is not None]),
        "gateway_cpu_median": median([s["gateway_process_cpu_percent"] for s in samples if s.get("gateway_process_cpu_percent") is not None]),
        "dependency_cpu_median": median([s["dependency_process_cpu_percent"] for s in samples if s.get("dependency_process_cpu_percent") is not None]),
        "samples": samples,
    }
    write_json(path, summary)
    return summary


def health_reset(gateway_port: int, dependency_port: int) -> None:
    status, payload = json_request("POST", f"http://127.0.0.1:{dependency_port}/control/delay", {"delay_ms": 0}, timeout=3.0)
    if status != 200:
        raise RuntimeError(f"dependency delay reset failed: {status} {payload}")
    status, payload = json_request("GET", f"http://127.0.0.1:{gateway_port}/ready", timeout=5.0)
    if status != 200 or payload.get("status") != "ok":
        raise RuntimeError(f"gateway service path not ready: {status} {payload}")
    status, payload = json_request("POST", f"http://127.0.0.1:{gateway_port}/sloscope/reset", {}, timeout=3.0)
    if status != 200 or payload.get("gateway") != "sloscope":
        raise RuntimeError(f"gateway reset failed: {status} {payload}")


def add_ratios(rows: list[dict]) -> None:
    healthy = [r for r in rows if r["condition"] == "healthy" and r["valid"]]
    for row in rows:
        if row["condition"] == "healthy":
            row["paired_healthy_run_id"] = None
            row["ttft_median_ratio"] = None
            row["total_latency_median_ratio"] = None
            continue
        if not healthy:
            continue
        paired = min(healthy, key=lambda h: (abs(h["planned_order"] - row["planned_order"]), h["planned_order"] > row["planned_order"]))
        row["paired_healthy_run_id"] = paired["run_id"]
        row["ttft_median_ratio"] = ratio(row["ttft_median"], paired["ttft_median"])
        row["total_latency_median_ratio"] = ratio(row["total_latency_median"], paired["total_latency_median"])


def descriptive(values: list[float]) -> dict:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"min": None, "median": None, "max": None, "iqr": None, "cv": None}
    med = statistics.median(vals)
    q1 = percentile(vals, 0.25)
    q3 = percentile(vals, 0.75)
    mean = statistics.mean(vals)
    return {
        "min": vals[0],
        "median": med,
        "max": vals[-1],
        "iqr": None if q1 is None or q3 is None else q3 - q1,
        "cv": None if mean == 0 else statistics.pstdev(vals) / mean,
    }


def qualitative(ratios: list[float | None]) -> str:
    vals = [v for v in ratios if v is not None]
    if not vals:
        return "weak observed effect"
    high = max(vals)
    if high < 1.10:
        return "weak observed effect"
    if high < 1.50:
        return "moderate observed effect"
    return "strong observed effect"


def build_analysis(root: Path, rows: list[dict], ledger: list[dict]) -> None:
    add_ratios(rows)
    write_csv(root / "summary.csv", rows)
    healthy = [r for r in rows if r["condition"] == "healthy"]
    healthy_rows = []
    for metric in ("ttft_median", "ttft_p95", "total_latency_median", "total_latency_p95"):
        desc = descriptive([r[metric] for r in healthy])
        healthy_rows.append({"metric": metric, **desc})
    write_csv(root / "healthy-stability.csv", healthy_rows)
    cpu_rows = [r for r in rows if r["condition"] == "cpu"]
    downstream_rows = [r for r in rows if r["condition"] == "downstream"]
    write_csv(root / "cpu-calibration.csv", cpu_rows)
    write_csv(root / "downstream-calibration.csv", downstream_rows)
    selection = []
    for value in (0.20, 0.35, 0.50):
        group = [r for r in cpu_rows if abs(float(r["magnitude_value"]) - value) < 1e-9]
        selection.append(
            {
                "family": "cpu",
                "magnitude": value,
                "unit": "logical_cpu_worker_fraction",
                "actual_worker_fraction": [r["worker_actual_fraction"] for r in group],
                "run_validity": [r["valid"] for r in group],
                "request_success": [r["successful"] for r in group],
                "host_cpu_response": [r["host_cpu_median"] for r in group],
                "ttft_ratio_range": [min([r["ttft_median_ratio"] for r in group if r.get("ttft_median_ratio") is not None], default=None), max([r["ttft_median_ratio"] for r in group if r.get("ttft_median_ratio") is not None], default=None)],
                "latency_ratio_range": [min([r["total_latency_median_ratio"] for r in group if r.get("total_latency_median_ratio") is not None], default=None), max([r["total_latency_median_ratio"] for r in group if r.get("total_latency_median_ratio") is not None], default=None)],
                "scheduler_slip_behavior": [r["scheduler_slip_p95"] for r in group],
                "qualitative_note": qualitative([r.get("total_latency_median_ratio") for r in group]),
            }
        )
    for value in (50, 100, 200):
        group = [r for r in downstream_rows if int(r["magnitude_value"]) == value]
        selection.append(
            {
                "family": "downstream",
                "magnitude": value,
                "unit": "ms",
                "verified_dependency_duration": [r["dependency_duration_median"] for r in group],
                "run_validity": [r["valid"] for r in group],
                "request_success": [r["successful"] for r in group],
                "ttft_ratio_range": [min([r["ttft_median_ratio"] for r in group if r.get("ttft_median_ratio") is not None], default=None), max([r["ttft_median_ratio"] for r in group if r.get("ttft_median_ratio") is not None], default=None)],
                "latency_ratio_range": [min([r["total_latency_median_ratio"] for r in group if r.get("total_latency_median_ratio") is not None], default=None), max([r["total_latency_median_ratio"] for r in group if r.get("total_latency_median_ratio") is not None], default=None)],
                "qualitative_note": qualitative([r.get("total_latency_median_ratio") for r in group]),
            }
        )
    write_csv(root / "severity-selection.csv", selection)
    report = [
        "# Phase 4A Calibration Report",
        "",
        "This is a descriptive pilot artifact, not a publication campaign.",
        "",
        f"- intended_runs: {len(ledger)}",
        f"- completed_runs: {sum(1 for item in ledger if item.get('actual_end'))}",
        f"- valid_runs: {sum(1 for item in ledger if item.get('validation_status') == 'valid')}",
        f"- invalid_runs: {sum(1 for item in ledger if item.get('validation_status') == 'invalid')}",
        "",
        "## Healthy Stability",
    ]
    for row in healthy_rows:
        report.append(f"- {row['metric']}: min={row['min']} median={row['median']} max={row['max']} iqr={row['iqr']} cv={row['cv']}")
    report.extend(["", "## Calibration Tables", "", "See `cpu-calibration.csv`, `downstream-calibration.csv`, and `severity-selection.csv`."])
    write_jsonl(root / "pilot-ledger.jsonl", ledger)
    (root / "calibration-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="pilot_runs/phase4a-calibration")
    parser.add_argument("--logs", default="logs/phase4a")
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    root = Path(args.root)
    runs_root = root / "runs"
    logs = Path(args.logs)
    root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    llama_version = parse_llama_version()
    llama_port = choose_port(8080)
    gateway_port = choose_port(8090)
    dependency_port = choose_port(8091)
    llama_args = list(LLAMA_ARGS)
    llama_args[llama_args.index("--port") + 1] = str(llama_port)
    services: list[ServiceHandle] = []
    ledger: list[dict] = []
    rows: list[dict] = []
    service_cleanup: list[dict] = []
    try:
        services.append(start_service("dependency", [sys.executable, "-m", "sloscope.gateway.dependency", "--host", "127.0.0.1", "--port", str(dependency_port), "--maximum-delay-ms", "250"], logs / "dependency.log", dependency_port))
        wait_json(f"http://127.0.0.1:{dependency_port}/control/status", timeout_seconds=20)
        services.append(start_service("llama", ["llama-server", *llama_args], logs / "llama-server.log", llama_port))
        wait_json(f"http://127.0.0.1:{llama_port}/health", timeout_seconds=180)
        services.append(start_service("gateway", [sys.executable, "-m", "sloscope.gateway.server", "--host", "127.0.0.1", "--port", str(gateway_port), "--llama-base-url", f"http://127.0.0.1:{llama_port}", "--dependency-base-url", f"http://127.0.0.1:{dependency_port}", "--timeout", "60.0"], logs / "gateway.log", gateway_port))
        wait_json(f"http://127.0.0.1:{gateway_port}/ready", timeout_seconds=180)
        pids = {service.name: service.process.pid for service in services}
        manifest = {
            "pilot_id": PILOT_ID,
            "purpose": "phase4a_calibration",
            "created_at": utc_now(),
            "sloscope_revision": git_revision(),
            "llama_cpp": llama_version,
            "model": {"model_ref": MODEL_REF, "model_id": MODEL_ID},
            "server_arguments": llama_args,
            "gateway": {"host": "127.0.0.1", "port": gateway_port, "pid": pids["gateway"]},
            "dependency": {"host": "127.0.0.1", "port": dependency_port, "pid": pids["dependency"], "maximum_delay_ms": 250},
            "llama": {"host": "127.0.0.1", "port": llama_port, "pid": pids["llama"]},
            "prompt_cache_enabled": False,
            "candidate_severities": {"cpu": [0.20, 0.35, 0.50], "downstream_ms": [50, 100, 200]},
            "workload": {"arrival_pattern": "constant_open_loop", "request_count": 20, "start_offset_seconds": 1.0, "inter_arrival_seconds": 0.25, "concurrency": 2, "prompt_profile": "synthetic-prose", "output_profile": "fixed", "target_output_tokens": 16, "temperature": 0, "seed": 4242, "max_outstanding_requests": 4},
            "seed_policy": "same deterministic workload seed for every run",
            "run_order_seed": RUN_ORDER_SEED,
            "cooldown_seconds": args.cooldown_seconds,
            "service_logs": {service.name: str(service.log_path) for service in services},
        }
        planned = build_run_order()
        manifest["run_order"] = [plan.__dict__ for plan in planned]
        write_json(root / "pilot-manifest.json", manifest)
        for plan in planned:
            item = {
                "run_id": plan.run_id,
                "condition": plan.condition,
                "mechanism_magnitude": plan.magnitude_value,
                "magnitude_unit": plan.magnitude_unit,
                "repetition": plan.repetition,
                "planned_order": plan.planned_order,
                "run_directory": str(runs_root / plan.run_id),
                "actual_start": None,
                "actual_end": None,
                "validation_status": "not_started",
                "config_hash": None,
                "failure_reason": None,
            }
            ledger.append(item)
            write_jsonl(root / "pilot-ledger.jsonl", ledger)
            try:
                health_reset(gateway_port, dependency_port)
                run_dir = runs_root / plan.run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                observe_idle(run_dir / "pre_run_idle_observation.json", pids)
                cfg = make_config(plan, gateway_port=gateway_port, dependency_port=dependency_port, service_pids=pids, llama_version=llama_version)
                write_json(run_dir / "planned-config.json", cfg.with_hash().to_dict(include_hash=True))
                item["config_hash"] = cfg.compute_hash()
                item["actual_start"] = utc_now()
                run_path = ExperimentRunner(cfg, runs_root=runs_root).run()
                result = validate_run(run_path)
                item["actual_end"] = utc_now()
                item["validation_status"] = "valid" if result.get("valid") else "invalid"
                item["failure_reason"] = None if result.get("valid") else ";".join(issue.get("code", "") for issue in result.get("issues", []))
                rows.append(summarize_run(run_path, plan))
            except Exception as exc:
                item["actual_end"] = utc_now()
                item["validation_status"] = "invalid"
                item["failure_reason"] = repr(exc)
            finally:
                try:
                    json_request("POST", f"http://127.0.0.1:{dependency_port}/control/delay", {"delay_ms": 0}, timeout=3.0)
                except Exception:
                    pass
                write_jsonl(root / "pilot-ledger.jsonl", ledger)
                if plan != planned[-1]:
                    time.sleep(args.cooldown_seconds)
        build_analysis(root, rows, ledger)
        return 0 if all(item["validation_status"] == "valid" for item in ledger) else 1
    finally:
        try:
            json_request("POST", f"http://127.0.0.1:{dependency_port}/control/delay", {"delay_ms": 0}, timeout=3.0)
        except Exception:
            pass
        service_cleanup = terminate_services(reversed(services))
        write_json(root / "service-cleanup.json", service_cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
