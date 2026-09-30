from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sloscope.artifacts.validation import validate_run
from sloscope.artifacts.writer import git_revision, read_json, read_table, write_json, write_jsonl
from sloscope.config import ExperimentConfig
from sloscope.ground_truth import active_mechanism_set
from sloscope.pilot_phase4a import (
    LLAMA_ARGS,
    LLAMA_BUILD,
    LLAMA_REVISION,
    LLAMA_VERSION,
    MODEL_ID,
    MODEL_REF,
    choose_port,
    descriptive,
    health_reset,
    json_request,
    median,
    observe_idle,
    parse_llama_version,
    percentile,
    ratio,
    start_service,
    terminate_services,
    utc_now,
    wait_json,
    write_csv,
)
from sloscope.pilot_phase4b import ServiceHandle, metric_values
from sloscope.provenance import git_dirty, source_tree_sha256
from sloscope.runner import ExperimentRunner


PILOT_ID = "phase4c-mechanism-isolation"
RUN_ORDER_SEED = 4242


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    condition: str
    family: str
    candidate: str
    magnitude_value: float | int | str | None
    magnitude_unit: str | None
    repetition: int
    block_id: str
    within_block_order: int
    planned_order: int
    request_count: int
    prompt_profile: str
    target_output_tokens: int
    inter_arrival_seconds: float
    max_outstanding_requests: int = 64


def is_queue_confounded(row: dict) -> bool:
    if row.get("family") in {"input", "output"}:
        return float(row.get("requests_deferred_max") or 0) > 0
    return False


def choose_output_reference(rows: list[dict]) -> str | None:
    candidates = []
    for row in rows:
        if row.get("family") != "output" or row.get("candidate") == "control-16":
            continue
        if is_queue_confounded(row):
            continue
        if float(row.get("token_realization_p95_floor") or 0) < 0.90:
            continue
        if float(row.get("post_first_token_duration_median_ratio") or 0) <= 1.05:
            continue
        candidates.append(row)
    if not candidates:
        return None
    candidates.sort(key=lambda item: int(str(item["candidate"]).replace("output-", "")))
    return str(candidates[0]["candidate"])


def build_blocks() -> list[PlannedRun]:
    rng = random.Random(RUN_ORDER_SEED)
    planned: list[PlannedRun] = []
    order = 1

    def add_block(block_id: str, members: list[PlannedRun]) -> None:
        nonlocal order
        idxs = list(range(len(members)))
        rng.shuffle(idxs)
        for within, idx in enumerate(idxs, start=1):
            p = members[idx]
            planned.append(
                PlannedRun(
                    p.run_id,
                    p.condition,
                    p.family,
                    p.candidate,
                    p.magnitude_value,
                    p.magnitude_unit,
                    p.repetition,
                    p.block_id,
                    within,
                    order,
                    p.request_count,
                    p.prompt_profile,
                    p.target_output_tokens,
                    p.inter_arrival_seconds,
                    p.max_outstanding_requests,
                )
            )
            order += 1

    for rep in range(1, 4):
        block_id = f"input-block-{rep:02d}"
        add_block(
            block_id,
            [
                PlannedRun(f"phase4c-{block_id}-control", "input-control", "input", "control", "short", "prompt_profile", rep, block_id, 0, 0, 30, "synthetic-input-small", 16, 0.25),
                PlannedRun(f"phase4c-{block_id}-medium", "input-medium", "input", "medium", "medium", "prompt_profile", rep, block_id, 0, 0, 30, "synthetic-input-medium", 16, 0.25),
            ],
        )
    for rep in range(1, 4):
        block_id = f"output-block-{rep:02d}"
        add_block(
            block_id,
            [
                PlannedRun(f"phase4c-{block_id}-control-16", "output-control", "output", "control-16", 16, "requested_output_tokens", rep, block_id, 0, 0, 24, "synthetic-continuation", 16, 0.5),
                PlannedRun(f"phase4c-{block_id}-output-32", "output-candidate", "output", "32", 32, "requested_output_tokens", rep, block_id, 0, 0, 24, "synthetic-continuation", 32, 0.5),
                PlannedRun(f"phase4c-{block_id}-output-48", "output-candidate", "output", "48", 48, "requested_output_tokens", rep, block_id, 0, 0, 24, "synthetic-continuation", 48, 0.5),
                PlannedRun(f"phase4c-{block_id}-output-64", "output-candidate", "output", "64", 64, "requested_output_tokens", rep, block_id, 0, 0, 24, "synthetic-continuation", 64, 0.5),
            ],
        )
    for rep in range(1, 4):
        block_id = f"load-block-{rep:02d}"
        add_block(
            block_id,
            [
                PlannedRun(f"phase4c-{block_id}-control-4rps", "load-control", "load", "control-4rps", 4, "requests_per_second", rep, block_id, 0, 0, 40, "synthetic-prose", 16, 0.25),
                PlannedRun(f"phase4c-{block_id}-load-12rps", "load-candidate", "load", "12rps", 12, "requests_per_second", rep, block_id, 0, 0, 40, "synthetic-prose", 16, 1.0 / 12.0),
            ],
        )
    return planned


def config_dict(plan: PlannedRun, *, gateway_port: int, service_pids: dict[str, int], llama_version: dict, llama_args: list[str]) -> dict:
    return {
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
                "max_outstanding_requests": plan.max_outstanding_requests,
                "metrics_enabled": True,
                "request_endpoint": "/v1/completions",
                "request_timeout": 90.0,
                "seed": 4242,
                "server_arguments": list(llama_args),
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
            "request_count": plan.request_count,
            "concurrency": 2,
            "prompt_profile": plan.prompt_profile,
            "output_profile": "fixed",
            "target_output_tokens": plan.target_output_tokens,
            "start_offset_seconds": 1.0,
            "inter_arrival_seconds": plan.inter_arrival_seconds,
            "burst_size": 2,
            "burst_interval_seconds": 1.0,
            "randomized_output": False,
            "output_jitter_tokens": 0,
        },
        "mechanisms": [],
        "telemetry": {"request_telemetry": True, "system_metrics": True, "runtime_metrics": True, "traces": True},
        "safety": {
            "maximum_cpu_stress": 0.0,
            "maximum_allocated_pressure_memory": 0,
            "maximum_disk_io": 0,
            "maximum_dependency_delay_ms": 250,
            "require_requests_within_mechanism_window": False,
            "mechanism_watchdog_interval_seconds": 0.25,
            "experiment_timeout": 240.0,
        },
    }


def make_config(plan: PlannedRun, **kwargs) -> ExperimentConfig:
    cfg = ExperimentConfig.from_dict(config_dict(plan, **kwargs))
    cfg.validate()
    return cfg


def summarize_run(run_dir: Path, plan: PlannedRun) -> dict:
    _, requests = read_table(run_dir / "requests.parquet")
    _, system = read_table(run_dir / "system_metrics.parquet")
    _, runtime = read_table(run_dir / "runtime_metrics.parquet")
    _, traces = read_table(run_dir / "traces.parquet")
    validation = read_json(run_dir / "validation.json")
    ground_truth = read_json(run_dir / "ground_truth.json")
    success = [r for r in requests if r["status"] == "success"]
    ttft = [r["first_token_time"] - r["actual_arrival"] for r in success if r.get("first_token_time") is not None]
    total = [r["completion_time"] - r["actual_arrival"] for r in success if r.get("completion_time") is not None]
    post_first = [r["completion_time"] - r["first_token_time"] for r in success if r.get("first_token_time") is not None and r.get("completion_time") is not None]
    slip = [r["scheduler_slip"] for r in requests if r.get("scheduler_slip") is not None]
    dep = [r["dependency_duration"] for r in success if r.get("dependency_duration") is not None]
    prompt_tokens = [r["server_prompt_tokens"] for r in success if r.get("server_prompt_tokens") is not None]
    output_tokens = [r["server_output_tokens"] for r in success if r.get("server_output_tokens") is not None]
    output_fracs = [r["server_output_tokens"] / plan.target_output_tokens for r in success if r.get("server_output_tokens") is not None and plan.target_output_tokens > 0]
    seconds_per_token = [
        (r["completion_time"] - r["first_token_time"]) / r["server_output_tokens"]
        for r in success
        if r.get("first_token_time") is not None and r.get("completion_time") is not None and r.get("server_output_tokens")
    ]
    processing = metric_values(runtime, "llamacpp:requests_processing")
    deferred = metric_values(runtime, "llamacpp:requests_deferred")
    arrivals = [r["actual_arrival"] for r in requests if r.get("actual_arrival") is not None]
    actual_rate = (len(arrivals) - 1) / (max(arrivals) - min(arrivals)) if len(arrivals) > 1 and max(arrivals) > min(arrivals) else None
    row = {
        "run_id": plan.run_id,
        "block_id": plan.block_id,
        "within_block_order": plan.within_block_order,
        "planned_order": plan.planned_order,
        "condition": plan.condition,
        "family": plan.family,
        "candidate": plan.candidate,
        "magnitude_value": plan.magnitude_value,
        "magnitude_unit": plan.magnitude_unit,
        "repetition": plan.repetition,
        "valid": validation.get("valid"),
        "issues": ";".join(issue.get("code", "") for issue in validation.get("issues", [])),
        "planned_requests": validation.get("planned_requests"),
        "accounted_requests": validation.get("accounted_requests"),
        "successful": len(success),
        "failed": sum(1 for r in requests if r["status"] == "failed"),
        "admission_failed": sum(1 for r in requests if r["status"] == "admission_failed"),
        "trace_count": len(traces),
        "active_mechanisms": ",".join(sorted(active_mechanism_set(ground_truth))),
        "prompt_profile": plan.prompt_profile,
        "target_output_tokens": plan.target_output_tokens,
        "inter_arrival_seconds": plan.inter_arrival_seconds,
        "nominal_offered_rps": None if plan.family != "load" else plan.magnitude_value,
        "actual_arrival_rps": actual_rate,
        "ttft_median": median(ttft),
        "ttft_p95": percentile(ttft, 0.95),
        "total_latency_median": median(total),
        "total_latency_p95": percentile(total, 0.95),
        "post_first_token_duration_median": median(post_first),
        "post_first_token_duration_p95": percentile(post_first, 0.95),
        "observed_seconds_per_output_token_median": median(seconds_per_token),
        "scheduler_slip_median": median(slip),
        "scheduler_slip_p95": percentile(slip, 0.95),
        "dependency_duration_median": median(dep),
        "dependency_duration_p95": percentile(dep, 0.95),
        "server_prompt_tokens_min": min(prompt_tokens, default=None),
        "server_prompt_tokens_median": median(prompt_tokens),
        "server_prompt_tokens_max": max(prompt_tokens, default=None),
        "server_output_tokens_min": min(output_tokens, default=None),
        "server_output_tokens_median": median(output_tokens),
        "server_output_tokens_max": max(output_tokens, default=None),
        "token_realization_p95_floor": percentile(output_fracs, 0.05),
        "output_fraction_median": ratio(median(output_tokens), plan.target_output_tokens),
        "requests_processing_max": max(processing, default=None),
        "requests_processing_median": median(processing),
        "requests_deferred_max": max(deferred, default=None),
        "requests_deferred_median": median(deferred),
        "host_cpu_median": median([r["host_cpu_percent"] for r in system if r.get("host_cpu_percent") is not None]),
        "host_cpu_p95": percentile([r["host_cpu_percent"] for r in system if r.get("host_cpu_percent") is not None], 0.95),
        "server_cpu_median": median([r["server_process_cpu_percent"] for r in system if r.get("server_process_cpu_percent") is not None]),
        "server_cpu_p95": percentile([r["server_process_cpu_percent"] for r in system if r.get("server_process_cpu_percent") is not None], 0.95),
        "server_rss_median": median([r["server_process_rss_bytes"] for r in system if r.get("server_process_rss_bytes") is not None]),
        "server_rss_max": max([r["server_process_rss_bytes"] for r in system if r.get("server_process_rss_bytes") is not None], default=None),
    }
    row["queue_confounded"] = is_queue_confounded(row)
    return row


def add_matched_ratios(rows: list[dict]) -> None:
    controls = {
        row["block_id"]: row
        for row in rows
        if row["condition"] in {"input-control", "output-control", "load-control"} and row.get("valid")
    }
    for row in rows:
        control = controls.get(row["block_id"])
        if control is None or row is control or row["condition"].endswith("control"):
            row["paired_control_run_id"] = None
            continue
        row["paired_control_run_id"] = control["run_id"]
        row["ttft_median_ratio"] = ratio(row["ttft_median"], control["ttft_median"])
        row["total_latency_median_ratio"] = ratio(row["total_latency_median"], control["total_latency_median"])
        row["post_first_token_duration_median_ratio"] = ratio(row["post_first_token_duration_median"], control["post_first_token_duration_median"])


def build_reference_table(rows: list[dict]) -> list[dict]:
    add_matched_ratios(rows)
    output_ref = choose_output_reference(rows)
    table = []
    input_medium = [r for r in rows if r["family"] == "input" and r["candidate"] == "medium"]
    input_clean = all(not r["queue_confounded"] for r in input_medium)
    table.append({"mechanism": "long_input_prefill", "candidate": "medium input", "magnitude": "observed median prompt tokens 267", "unit": "server_prompt_tokens", "primary_signal": "TTFT", "secondary_signal": "server_prompt_tokens", "queue_confounded": not input_clean, "repeatability": [r["server_prompt_tokens_median"] for r in input_medium], "runtime_cost": "moderate", "provisional_reference": input_clean, "notes": "matched against input-control"})
    for candidate in ["32", "48", "64"]:
        group = [r for r in rows if r["family"] == "output" and r["candidate"] == candidate]
        clean = all(not r["queue_confounded"] for r in group)
        table.append({"mechanism": "long_output_decode", "candidate": f"output {candidate}", "magnitude": candidate, "unit": "requested_output_tokens", "primary_signal": "post_first_token_duration", "secondary_signal": "server_output_tokens", "queue_confounded": not clean, "repeatability": [r["server_output_tokens_median"] for r in group], "runtime_cost": "low" if candidate == "32" else "moderate", "provisional_reference": output_ref == candidate, "notes": "smallest clean useful output candidate preferred"})
    load_group = [r for r in rows if r["family"] == "load" and r["candidate"] == "12rps"]
    queue_seen = any(float(r.get("requests_deferred_max") or 0) > 0 for r in load_group)
    table.append({"mechanism": "high_offered_load_queue", "candidate": "load 12 rps", "magnitude": 12, "unit": "requests_per_second", "primary_signal": "requests_deferred", "secondary_signal": "TTFT/latency", "queue_confounded": False, "repeatability": [r["requests_deferred_max"] for r in load_group], "runtime_cost": "moderate", "provisional_reference": queue_seen, "notes": "queueing is intended for load"})
    table.append({"mechanism": "downstream_latency", "candidate": "downstream 100 ms", "magnitude": 100, "unit": "ms", "primary_signal": "dependency_duration", "secondary_signal": "control_state", "queue_confounded": False, "repeatability": "carried from Phase 4A", "runtime_cost": "low", "provisional_reference": True, "notes": "not rerun in Phase 4C"})
    table.append({"mechanism": "cpu_contention", "candidate": "0.20-0.50 worker fraction", "magnitude": "0.20-0.50", "unit": "logical_cpu_worker_fraction", "primary_signal": "worker_cpu_time", "secondary_signal": "SLO effect", "queue_confounded": False, "repeatability": "verified injection", "runtime_cost": "host-dependent", "provisional_reference": False, "notes": "weak SLO effect under current Metal configuration; deferred to cross-configuration analysis"})
    return table


def build_analysis(root: Path, rows: list[dict], ledger: list[dict]) -> None:
    add_matched_ratios(rows)
    write_csv(root / "summary.csv", rows)
    input_rows = [r for r in rows if r["family"] == "input"]
    output_rows = [r for r in rows if r["family"] == "output"]
    load_rows = [r for r in rows if r["family"] == "load"]
    write_csv(root / "input-isolation.csv", input_rows)
    write_csv(root / "output-isolation.csv", output_rows)
    write_csv(root / "load-isolation.csv", load_rows)
    reference = build_reference_table(rows)
    write_csv(root / "mechanism-reference-table.csv", reference)
    report = [
        "# Phase 4C Mechanism Isolation Report",
        "",
        f"- analyzed_runs: {len(rows)}",
        f"- valid_analyzed_runs: {sum(1 for r in rows if r.get('valid'))}",
        f"- preconditioning_status: {next((item.get('validation_status') for item in ledger if not item.get('scored')), None)}",
        "",
        "Phase 4C uses family-specific matched controls and does not run compound mechanisms.",
    ]
    write_jsonl(root / "pilot-ledger.jsonl", ledger)
    (root / "calibration-report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_one(plan: PlannedRun, *, root: Path, runs_root: Path, pids: dict[str, int], gateway_port: int, dependency_port: int, llama_version: dict, llama_args: list[str], scored: bool) -> tuple[dict, dict | None]:
    item = {
        "run_id": plan.run_id,
        "condition": plan.condition,
        "family": plan.family,
        "candidate": plan.candidate,
        "mechanism_magnitude": plan.magnitude_value,
        "magnitude_unit": plan.magnitude_unit,
        "repetition": plan.repetition,
        "block_id": plan.block_id,
        "within_block_order": plan.within_block_order,
        "planned_order": plan.planned_order,
        "run_directory": str(runs_root / plan.run_id),
        "actual_start": None,
        "actual_end": None,
        "validation_status": "not_started",
        "config_hash": None,
        "failure_reason": None,
        "scored": scored,
    }
    row = None
    try:
        health_reset(gateway_port, dependency_port)
        run_dir = runs_root / plan.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        observe_idle(run_dir / "pre_run_idle_observation.json", pids)
        cfg = make_config(plan, gateway_port=gateway_port, service_pids=pids, llama_version=llama_version, llama_args=llama_args)
        write_json(run_dir / "planned-config.json", cfg.with_hash().to_dict(include_hash=True))
        item["config_hash"] = cfg.compute_hash()
        item["actual_start"] = utc_now()
        run_path = ExperimentRunner(cfg, runs_root=runs_root).run()
        result = validate_run(run_path)
        item["actual_end"] = utc_now()
        item["validation_status"] = "valid" if result.get("valid") else "invalid"
        item["failure_reason"] = None if result.get("valid") else ";".join(issue.get("code", "") for issue in result.get("issues", []))
        row = summarize_run(run_path, plan)
    except Exception as exc:
        item["actual_end"] = utc_now()
        item["validation_status"] = "invalid"
        item["failure_reason"] = repr(exc)
    finally:
        try:
            json_request("POST", f"http://127.0.0.1:{dependency_port}/control/delay", {"delay_ms": 0}, timeout=3.0)
        except Exception:
            pass
    return item, row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="pilot_runs/phase4c-mechanism-isolation")
    parser.add_argument("--logs", default="logs/phase4c")
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
    try:
        services.append(start_service("dependency", [sys.executable, "-m", "sloscope.gateway.dependency", "--host", "127.0.0.1", "--port", str(dependency_port), "--maximum-delay-ms", "250"], logs / "dependency.log", dependency_port))
        wait_json(f"http://127.0.0.1:{dependency_port}/control/status", timeout_seconds=20)
        services.append(start_service("llama", ["llama-server", *llama_args], logs / "llama-server.log", llama_port))
        wait_json(f"http://127.0.0.1:{llama_port}/health", timeout_seconds=180)
        services.append(start_service("gateway", [sys.executable, "-m", "sloscope.gateway.server", "--host", "127.0.0.1", "--port", str(gateway_port), "--llama-base-url", f"http://127.0.0.1:{llama_port}", "--dependency-base-url", f"http://127.0.0.1:{dependency_port}", "--timeout", "90.0"], logs / "gateway.log", gateway_port))
        wait_json(f"http://127.0.0.1:{gateway_port}/ready", timeout_seconds=180)
        pids = {service.name: service.process.pid for service in services}
        planned = build_blocks()
        manifest = {
            "pilot_id": PILOT_ID,
            "purpose": "phase4c_mechanism_isolation",
            "created_at": utc_now(),
            "git_revision": git_revision(),
            "git_dirty": git_dirty(),
            "source_tree_sha256": source_tree_sha256(),
            "llama_cpp": llama_version,
            "model": {"model_ref": MODEL_REF, "model_id": MODEL_ID},
            "server_arguments": llama_args,
            "gateway": {"host": "127.0.0.1", "port": gateway_port, "pid": pids["gateway"]},
            "dependency": {"host": "127.0.0.1", "port": dependency_port, "pid": pids["dependency"], "maximum_delay_ms": 250, "delay_ms": 0},
            "llama": {"host": "127.0.0.1", "port": llama_port, "pid": pids["llama"]},
            "prompt_cache_enabled": False,
            "run_order_seed": RUN_ORDER_SEED,
            "cooldown_seconds": args.cooldown_seconds,
            "block_definitions": [plan.__dict__ for plan in planned],
            "service_logs": {service.name: str(service.log_path) for service in services},
            "carry_forward": {"downstream_reference_ms": 100, "cpu_deferral": "verified mechanism, weak SLO effect under current Metal configuration"},
        }
        write_json(root / "pilot-manifest.json", manifest)
        pre = PlannedRun("phase4c-preconditioning", "preconditioning", "preconditioning", "healthy", None, None, 0, "preconditioning", 0, 0, 30, "synthetic-prose", 16, 0.25)
        item, row = run_one(pre, root=root, runs_root=runs_root, pids=pids, gateway_port=gateway_port, dependency_port=dependency_port, llama_version=llama_version, llama_args=llama_args, scored=False)
        ledger.append(item)
        write_jsonl(root / "pilot-ledger.jsonl", ledger)
        if item["validation_status"] != "valid":
            return 1
        manifest["preconditioning_run"] = item
        write_json(root / "pilot-manifest.json", manifest)
        for plan in planned:
            item, row = run_one(plan, root=root, runs_root=runs_root, pids=pids, gateway_port=gateway_port, dependency_port=dependency_port, llama_version=llama_version, llama_args=llama_args, scored=True)
            ledger.append(item)
            if row is not None:
                rows.append(row)
            write_jsonl(root / "pilot-ledger.jsonl", ledger)
            if plan != planned[-1]:
                time.sleep(args.cooldown_seconds)
        build_analysis(root, rows, ledger)
        return 0 if all(item["validation_status"] == "valid" for item in ledger if item.get("scored")) else 1
    finally:
        try:
            json_request("POST", f"http://127.0.0.1:{dependency_port}/control/delay", {"delay_ms": 0}, timeout=3.0)
        except Exception:
            pass
        cleanup = terminate_services(reversed(services))
        write_json(root / "service-cleanup.json", cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
