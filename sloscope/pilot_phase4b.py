from __future__ import annotations

import argparse
import csv
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    ServiceHandle,
    choose_port,
    descriptive,
    health_reset,
    json_request,
    median,
    observe_idle,
    parse_llama_version,
    percentile,
    port_is_free,
    ratio,
    start_service,
    terminate_services,
    utc_now,
    wait_json,
    write_csv,
)
from sloscope.provenance import git_dirty, source_tree_sha256
from sloscope.runner import ExperimentRunner


PILOT_ID = "phase4b-workload-calibration"
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
    max_outstanding_requests: int


def build_blocks() -> list[PlannedRun]:
    rng = random.Random(RUN_ORDER_SEED)
    input_candidates = ["small", "medium", "large"] * 2
    output_candidates = [32, 64, 128] * 2
    load_candidates = [8, 12, 16] * 2
    rng.shuffle(input_candidates)
    rng.shuffle(output_candidates)
    rng.shuffle(load_candidates)
    planned: list[PlannedRun] = []
    order = 1
    for idx in range(6):
        block_id = f"block-{idx + 1:02d}"
        rep = idx + 1
        members = [
            ("healthy", "healthy", "healthy", None, None),
            ("input", "input", input_candidates[idx], input_candidates[idx], "prompt_profile"),
            ("output", "output", str(output_candidates[idx]), output_candidates[idx], "requested_output_tokens"),
            ("load", "load", f"{load_candidates[idx]}rps", load_candidates[idx], "requests_per_second"),
        ]
        rng.shuffle(members)
        for within, (condition, family, candidate, magnitude, unit) in enumerate(members, start=1):
            if condition == "healthy":
                run_id = f"phase4b-{block_id}-healthy"
                request_count = 30
                prompt_profile = "synthetic-prose"
                target_output_tokens = 16
                inter_arrival = 0.25
                max_outstanding = 64
            elif condition == "input":
                run_id = f"phase4b-{block_id}-input-{candidate}"
                request_count = 30
                prompt_profile = f"synthetic-input-{candidate}"
                target_output_tokens = 16
                inter_arrival = 0.25
                max_outstanding = 64
            elif condition == "output":
                run_id = f"phase4b-{block_id}-output-{candidate}"
                request_count = 30
                prompt_profile = "synthetic-continuation"
                target_output_tokens = int(magnitude)
                inter_arrival = 0.25
                max_outstanding = 64
            else:
                run_id = f"phase4b-{block_id}-load-{candidate}"
                request_count = 40
                prompt_profile = "synthetic-prose"
                target_output_tokens = 16
                inter_arrival = 1.0 / int(magnitude)
                max_outstanding = 64
            planned.append(
                PlannedRun(
                    run_id,
                    condition,
                    family,
                    str(candidate),
                    magnitude,
                    unit,
                    rep,
                    block_id,
                    within,
                    order,
                    request_count,
                    prompt_profile,
                    target_output_tokens,
                    inter_arrival,
                    max_outstanding,
                )
            )
            order += 1
    return planned


def config_dict(
    plan: PlannedRun,
    *,
    gateway_port: int,
    service_pids: dict[str, int],
    llama_version: dict,
    llama_args: list[str],
) -> dict:
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


def metric_values(rows: list[dict], metric_name: str) -> list[float]:
    return [
        float(row["metric_value"])
        for row in rows
        if row.get("metric_name") == metric_name and row.get("metric_value") is not None
    ]


def summarize_run(run_dir: Path, plan: PlannedRun) -> dict:
    _, requests = read_table(run_dir / "requests.parquet")
    _, system = read_table(run_dir / "system_metrics.parquet")
    _, runtime = read_table(run_dir / "runtime_metrics.parquet")
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
    prompt_tokens = [r["server_prompt_tokens"] for r in success if r.get("server_prompt_tokens") is not None]
    output_tokens = [r["server_output_tokens"] for r in success if r.get("server_output_tokens") is not None]
    processing = metric_values(runtime, "llamacpp:requests_processing")
    deferred = metric_values(runtime, "llamacpp:requests_deferred")
    arrivals = [r["actual_arrival"] for r in requests if r.get("actual_arrival") is not None]
    actual_rate = None
    if len(arrivals) > 1 and max(arrivals) > min(arrivals):
        actual_rate = (len(arrivals) - 1) / (max(arrivals) - min(arrivals))
    return {
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
        "failed": len(failed),
        "admission_failed": len(admission_failed),
        "trace_count": len(traces),
        "active_mechanisms": ",".join(sorted(active_mechanism_set(ground_truth))),
        "request_count": plan.request_count,
        "target_output_tokens": plan.target_output_tokens,
        "inter_arrival_seconds": plan.inter_arrival_seconds,
        "nominal_offered_rps": None if plan.condition != "load" else plan.magnitude_value,
        "actual_arrival_rps": actual_rate,
        "ttft_median": median(ttft),
        "ttft_p95": percentile(ttft, 0.95),
        "total_latency_median": median(total),
        "total_latency_p95": percentile(total, 0.95),
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


def add_block_ratios(rows: list[dict]) -> None:
    by_block = {row["block_id"]: row for row in rows if row["condition"] == "healthy" and row["valid"]}
    for row in rows:
        healthy = by_block.get(row["block_id"])
        if row["condition"] == "healthy" or healthy is None:
            row["paired_healthy_run_id"] = None
            row["ttft_median_ratio"] = None
            row["total_latency_median_ratio"] = None
            continue
        row["paired_healthy_run_id"] = healthy["run_id"]
        row["ttft_median_ratio"] = ratio(row["ttft_median"], healthy["ttft_median"])
        row["total_latency_median_ratio"] = ratio(row["total_latency_median"], healthy["total_latency_median"])


def build_analysis(root: Path, rows: list[dict], ledger: list[dict], context_safety: dict, output_control: dict) -> None:
    add_block_ratios(rows)
    write_csv(root / "summary.csv", rows)
    healthy_rows = []
    healthy = [r for r in rows if r["condition"] == "healthy"]
    for metric in ("ttft_median", "ttft_p95", "total_latency_median", "total_latency_p95"):
        healthy_rows.append({"metric": metric, **descriptive([r[metric] for r in healthy])})
    write_csv(root / "healthy-stability.csv", healthy_rows)
    input_rows = [r for r in rows if r["condition"] == "input"]
    output_rows = [r for r in rows if r["condition"] == "output"]
    load_rows = [r for r in rows if r["condition"] == "load"]
    write_csv(root / "input-calibration.csv", input_rows)
    write_csv(root / "output-calibration.csv", output_rows)
    write_csv(root / "load-calibration.csv", load_rows)
    selection = []
    for family, candidates, group_rows in [
        ("input", ["small", "medium", "large"], input_rows),
        ("output", ["32", "64", "128"], output_rows),
        ("load", ["8rps", "12rps", "16rps"], load_rows),
    ]:
        for candidate in candidates:
            group = [r for r in group_rows if r["candidate"] == candidate]
            selection.append(
                {
                    "family": family,
                    "candidate": candidate,
                    "run_validity": [r["valid"] for r in group],
                    "request_success": [r["successful"] for r in group],
                    "ttft_ratio_range": [
                        min([r["ttft_median_ratio"] for r in group if r.get("ttft_median_ratio") is not None], default=None),
                        max([r["ttft_median_ratio"] for r in group if r.get("ttft_median_ratio") is not None], default=None),
                    ],
                    "latency_ratio_range": [
                        min([r["total_latency_median_ratio"] for r in group if r.get("total_latency_median_ratio") is not None], default=None),
                        max([r["total_latency_median_ratio"] for r in group if r.get("total_latency_median_ratio") is not None], default=None),
                    ],
                    "prompt_token_median": [r["server_prompt_tokens_median"] for r in group],
                    "output_fraction_median": [r["output_fraction_median"] for r in group],
                    "requests_deferred_max": [r["requests_deferred_max"] for r in group],
                }
            )
    write_csv(root / "severity-selection.csv", selection)
    report = [
        "# Phase 4B Workload Calibration Report",
        "",
        "This is descriptive calibration output, not a publication campaign.",
        "",
        f"- intended_analyzed_runs: {len(ledger) - 1}",
        f"- valid_analyzed_runs: {sum(1 for item in ledger if item.get('scored') and item.get('validation_status') == 'valid')}",
        f"- invalid_analyzed_runs: {sum(1 for item in ledger if item.get('scored') and item.get('validation_status') == 'invalid')}",
        f"- preconditioning_status: {next((item.get('validation_status') for item in ledger if not item.get('scored')), None)}",
        "",
        "## Context Safety",
        f"- {context_safety}",
        "",
        "## Output Controllability",
        f"- {output_control}",
        "",
        "## Phase 4A Carry-Forward Notes",
        "- CPU contention was independently verified but did not produce an SLO degradation effect at nominal 0.20-0.50 worker fractions under the current llama.cpp + Metal configuration; it is deferred from the primary Metal mechanism set and retained for later cross-configuration studies.",
        "- Phase 4A downstream latency at 50/100/200 ms produced controlled monotonic dependency delay; 100 ms remains a provisional medium downstream reference for a later compound pilot, not a final severity freeze.",
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


def preflight_context_and_output(root: Path, runs_root: Path, pids: dict[str, int], gateway_port: int, dependency_port: int, llama_version: dict, llama_args: list[str]) -> tuple[dict, dict]:
    context: dict[str, Any] = {"usable_per_slot_context": 1024, "target_output_tokens": 16, "profiles": {}, "safe": True}
    for idx, profile in enumerate(["synthetic-input-small", "synthetic-input-medium", "synthetic-input-large"], start=1):
        plan = PlannedRun(f"phase4b-preflight-context-{profile}", "preflight", "input-context", profile, profile, "prompt_profile", idx, "preflight", idx, idx, 1, profile, 16, 0.25, 2)
        item, row = run_one(plan, root=root, runs_root=runs_root, pids=pids, gateway_port=gateway_port, dependency_port=dependency_port, llama_version=llama_version, llama_args=llama_args, scored=False)
        if item["validation_status"] != "valid" or row is None:
            raise RuntimeError(f"context preflight failed: {item}")
        tokens = row["server_prompt_tokens_median"]
        safe = tokens is not None and tokens + 16 + 64 < 1024
        context["profiles"][profile] = {"server_prompt_tokens": tokens, "safe": safe}
        context["safe"] = bool(context["safe"] and safe)
    output: dict[str, Any] = {"candidates": {}, "controllable": True}
    for target in [32, 64, 128]:
        plan = PlannedRun(f"phase4b-preflight-output-{target}", "preflight", "output-control", str(target), target, "requested_output_tokens", target, "preflight", target, target, 3, "synthetic-continuation", target, 0.25, 4)
        item, row = run_one(plan, root=root, runs_root=runs_root, pids=pids, gateway_port=gateway_port, dependency_port=dependency_port, llama_version=llama_version, llama_args=llama_args, scored=False)
        if item["validation_status"] != "valid" or row is None:
            raise RuntimeError(f"output preflight failed: {item}")
        frac = row["output_fraction_median"]
        ok = frac is not None and frac >= 0.90
        output["candidates"][str(target)] = {"server_output_tokens_median": row["server_output_tokens_median"], "fraction": frac, "controllable": ok}
        output["controllable"] = bool(output["controllable"] and ok)
    write_json(root / "preflight-calibration.json", {"context_safety": context, "output_control": output})
    if not context["safe"]:
        raise RuntimeError(f"context safety failed: {context}")
    if not output["controllable"]:
        raise RuntimeError(f"output controllability failed: {output}")
    return context, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="pilot_runs/phase4b-workload-calibration")
    parser.add_argument("--logs", default="logs/phase4b")
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
            "purpose": "phase4b_workload_calibration",
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
            "candidate_settings": {"input": ["small", "medium", "large"], "output_tokens": [32, 64, 128], "load_rps": [8, 12, 16]},
            "workload_base": {"request_count": 30, "arrival_pattern": "constant_open_loop", "start_offset_seconds": 1.0, "inter_arrival_seconds": 0.25, "target_output_tokens": 16, "prompt_profile": "synthetic-prose", "temperature": 0, "seed": 4242, "max_outstanding_requests": 64},
            "block_definitions": [plan.__dict__ for plan in planned],
            "service_logs": {service.name: str(service.log_path) for service in services},
            "phase4a_notes": {
                "cpu_deferral": "CPU contention was independently verified but did not produce an SLO degradation effect at nominal 0.20-0.50 worker fractions under the current llama.cpp + Metal configuration; deferred from the primary Metal mechanism set for this configuration.",
                "downstream_reference": "50/100/200 ms produced controlled monotonic dependency delay in Phase 4A; 100 ms is provisionally retained as a medium downstream-latency reference for the next compound pilot.",
            },
        }
        write_json(root / "pilot-manifest.json", manifest)
        preconditioning = PlannedRun("phase4b-preconditioning", "preconditioning", "healthy", "preconditioning", None, None, 0, "preconditioning", 0, 0, 30, "synthetic-prose", 16, 0.25, 64)
        item, row = run_one(preconditioning, root=root, runs_root=runs_root, pids=pids, gateway_port=gateway_port, dependency_port=dependency_port, llama_version=llama_version, llama_args=llama_args, scored=False)
        ledger.append(item)
        write_jsonl(root / "pilot-ledger.jsonl", ledger)
        if item["validation_status"] != "valid":
            write_json(root / "service-cleanup.json", terminate_services(reversed(services)))
            return 1
        manifest["preconditioning_run"] = item
        write_json(root / "pilot-manifest.json", manifest)
        context_safety, output_control = preflight_context_and_output(root, runs_root, pids, gateway_port, dependency_port, llama_version, llama_args)
        manifest["context_safety"] = context_safety
        manifest["output_control"] = output_control
        write_json(root / "pilot-manifest.json", manifest)
        for plan in planned:
            item, row = run_one(plan, root=root, runs_root=runs_root, pids=pids, gateway_port=gateway_port, dependency_port=dependency_port, llama_version=llama_version, llama_args=llama_args, scored=True)
            ledger.append(item)
            if row is not None:
                rows.append(row)
            write_jsonl(root / "pilot-ledger.jsonl", ledger)
            if plan != planned[-1]:
                time.sleep(args.cooldown_seconds)
        build_analysis(root, rows, ledger, context_safety, output_control)
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
