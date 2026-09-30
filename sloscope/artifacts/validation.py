from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from sloscope.artifacts.writer import ARTIFACT_SCHEMA_VERSION, SUPPORTED_ARTIFACT_SCHEMA_VERSIONS, read_json, read_jsonl, read_table, sha256_file
from sloscope.config import ExperimentConfig, SUPPORTED_SCHEMA_VERSIONS
from sloscope.lifecycle import validate_lifecycle_history
from sloscope.telemetry.schemas import RequestRecord

REQUIRED = [
    "manifest.json", "config.json", "ground_truth.json", "workload.json", "requests.parquet",
    "system_metrics.parquet", "runtime_metrics.parquet", "traces.parquet", "events.jsonl", "validation.json",
]


def _issue(issues: list[dict], code: str, message: str) -> None:
    issues.append({"code": code, "message": message})


def validate_run(run_dir: Union[str, Path]) -> Dict[str, Any]:
    root = Path(run_dir)
    issues: List[dict] = []
    loaded: Dict[str, Any] = {}
    for name in REQUIRED:
        path = root / name
        if not path.exists():
            _issue(issues, "missing_artifact", f"missing {name}")
            continue
        try:
            if name.endswith(".json"):
                loaded[name] = read_json(path)
            elif name.endswith(".jsonl"):
                loaded[name] = read_jsonl(path)
            elif name.endswith(".parquet"):
                loaded[name] = read_table(path)
        except Exception as exc:
            _issue(issues, "corrupted_artifact", f"{name}: {exc}")
    optional_condition = root / "experimental_condition.json"
    if optional_condition.exists():
        try:
            loaded["experimental_condition.json"] = read_json(optional_condition)
        except Exception as exc:
            _issue(issues, "corrupted_artifact", f"experimental_condition.json: {exc}")
    cfg_data = loaded.get("config.json")
    manifest = loaded.get("manifest.json", {})
    workload = loaded.get("workload.json", {}).get("requests", []) if isinstance(loaded.get("workload.json"), dict) else []
    requests = loaded.get("requests.parquet", ({}, []))[1] if "requests.parquet" in loaded else []
    system_metric_rows = loaded.get("system_metrics.parquet", ({}, []))[1] if "system_metrics.parquet" in loaded else []
    runtime_metric_rows = loaded.get("runtime_metrics.parquet", ({}, []))[1] if "runtime_metrics.parquet" in loaded else []
    trace_rows = loaded.get("traces.parquet", ({}, []))[1] if "traces.parquet" in loaded else []
    ground_truth = loaded.get("ground_truth.json", {})
    experimental_condition = loaded.get("experimental_condition.json")
    events = loaded.get("events.jsonl", [])
    run_id = cfg_data.get("run_id") if isinstance(cfg_data, dict) else root.name
    if manifest and manifest.get("artifact_schema_version") not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        _issue(issues, "artifact_version_mismatch", "unknown artifact schema version")
    if isinstance(manifest, dict):
        inventory = set(manifest.get("artifact_inventory", []))
        for item in inventory:
            if not (root / item).exists():
                _issue(issues, "manifest_inventory_missing_file", item)
        for name in REQUIRED:
            if (root / name).exists() and name not in inventory:
                _issue(issues, "manifest_inventory_omission", name)
        for name, expected in manifest.get("artifact_hashes", {}).items():
            path = root / name
            if not path.exists():
                _issue(issues, "missing_hashed_artifact", name)
            elif sha256_file(path) != expected:
                _issue(issues, "artifact_hash_mismatch", f"{name} hash mismatch")
        if "experimental_condition.json" in inventory and "experimental_condition.json" not in manifest.get("artifact_hashes", {}):
            _issue(issues, "missing_condition_hash", "experimental_condition.json is inventoried but not hashed")
    config = None
    if isinstance(cfg_data, dict):
        try:
            config = ExperimentConfig.from_dict(cfg_data)
            if config.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
                _issue(issues, "unknown_schema_version", f"unknown config schema {config.schema_version}")
            if config.config_hash != config.compute_hash():
                _issue(issues, "config_hash_mismatch", "config hash does not match canonical config")
            if manifest and manifest.get("config_hash") != config.config_hash:
                _issue(issues, "manifest_config_hash_mismatch", "manifest/config hash mismatch")
        except Exception as exc:
            _issue(issues, "malformed_config", str(exc))
    if isinstance(experimental_condition, dict) and config is not None:
        expected_run_id = experimental_condition.get("run_id")
        if expected_run_id and expected_run_id != config.run_id:
            _issue(issues, "experimental_condition_mismatch", "run_id mismatch")
        if experimental_condition.get("campaign_condition_id") and config.runtime.parameters.get("experimental_condition"):
            if experimental_condition != config.runtime.parameters.get("experimental_condition"):
                _issue(issues, "experimental_condition_mismatch", "condition artifact differs from config runtime metadata")
        labels = set(experimental_condition.get("active_mechanisms") or [])
        levels = experimental_condition.get("mechanism_levels") or {}
        wl = config.workload
        def require(label: str, ok: bool, message: str) -> None:
            if label in labels and not ok:
                _issue(issues, "experimental_condition_mismatch", message)
        require("input_medium", wl.prompt_profile == "synthetic-input-medium", "input_medium requires synthetic-input-medium")
        require("output_32", wl.prompt_profile == "synthetic-continuation" and wl.target_output_tokens == 32, "output_32 requires continuation prompt and 32 target tokens")
        require("load_12rps", abs(wl.inter_arrival_seconds - (1.0 / 12.0)) < 1e-9, "load_12rps requires 12 rps inter-arrival")
        if "downstream_100ms" in labels:
            if not any(m.mechanism_type == "downstream_latency" and int(m.parameters.get("delay_ms", m.intensity)) == 100 for m in config.mechanisms):
                _issue(issues, "experimental_condition_mismatch", "downstream_100ms requires configured downstream latency injector")
        if int(experimental_condition.get("compound_degree", len(labels))) != len(labels):
            _issue(issues, "experimental_condition_mismatch", "compound_degree mismatch")
        if set(levels) != labels:
            _issue(issues, "experimental_condition_mismatch", "mechanism_levels keys do not match active mechanisms")
    elif config is not None and config.runtime.parameters.get("experimental_condition_required"):
        _issue(issues, "missing_experimental_condition", "experimental_condition.json is required")
    planned_ids = [r.get("request_id") for r in workload if isinstance(r, dict)]
    if len(planned_ids) != len(set(planned_ids)):
        _issue(issues, "duplicate_request_id", "duplicate request IDs in workload")
    request_ids = [r.get("request_id") for r in requests if isinstance(r, dict)]
    if len(request_ids) != len(set(request_ids)):
        _issue(issues, "duplicate_request_id", "duplicate request IDs in request telemetry")
    if set(request_ids) - set(planned_ids):
        _issue(issues, "unknown_request", "request telemetry contains unplanned request IDs")
    successful_request_ids = {r.get("request_id") for r in requests if isinstance(r, dict) and r.get("status") == "success"}
    trace_by_request: dict[str, list[dict]] = {}
    span_ids: set[str] = set()
    span_keys: set[tuple[str, str]] = set()
    hex32 = re.compile(r"^[0-9a-f]{32}$")
    hex16 = re.compile(r"^[0-9a-f]{16}$")
    for span in trace_rows:
        if not isinstance(span, dict):
            _issue(issues, "malformed_trace_span", "span record is not an object")
            continue
        trace_id = span.get("trace_id")
        span_id = span.get("span_id")
        parent_span_id = span.get("parent_span_id")
        span_name = span.get("span_name")
        if not isinstance(trace_id, str) or not hex32.match(trace_id):
            _issue(issues, "malformed_trace_id", str(trace_id))
        if not isinstance(span_id, str) or not hex16.match(span_id):
            _issue(issues, "malformed_span_id", str(span_id))
        if span_id in span_ids:
            _issue(issues, "duplicate_trace_span", f"duplicate span_id {span_id}")
        span_ids.add(span_id)
        try:
            attributes = json.loads(span.get("attributes") or "{}")
        except Exception:
            attributes = {}
            _issue(issues, "malformed_trace_span", f"{span_id} attributes are not JSON")
        rid = attributes.get("request_id")
        if rid not in planned_ids:
            _issue(issues, "foreign_trace_request", str(rid))
            continue
        key = (str(rid), str(span_name))
        if key in span_keys:
            _issue(issues, "duplicate_trace_span", f"{rid}:{span_name}")
        span_keys.add(key)
        trace_by_request.setdefault(str(rid), []).append(span)
        if parent_span_id is not None and (not isinstance(parent_span_id, str) or not hex16.match(parent_span_id)):
            _issue(issues, "malformed_span_id", f"parent {parent_span_id}")
    for row in requests:
        try:
            rec = RequestRecord(**row)
            for problem in rec.validate_ordering():
                _issue(issues, "invalid_timestamp_ordering", f"{rec.request_id}: {problem}")
            if config and config.runtime.runtime_type == "llamacpp" and rec.status != "admission_failed":
                if rec.actual_arrival is not None and rec.scheduled_arrival is not None and rec.actual_arrival + 0.005 < rec.scheduled_arrival:
                    _issue(issues, "negative_scheduler_slip", rec.request_id)
                if rec.scheduler_slip is not None and rec.actual_arrival is not None:
                    expected_slip = rec.actual_arrival - rec.scheduled_arrival
                    if abs(rec.scheduler_slip - expected_slip) > 1e-6:
                        _issue(issues, "scheduler_slip_mismatch", rec.request_id)
            if rec.status == "admission_failed":
                _issue(issues, "client_max_outstanding_exceeded", rec.request_id)
            if config and (rec.runtime_id != config.runtime.runtime_id or rec.model_id != config.runtime.model_id):
                _issue(issues, "unknown_runtime_or_model", rec.request_id)
        except Exception as exc:
            _issue(issues, "malformed_request", str(exc))
    accounted = len(requests)
    planned = len(workload)
    if planned != accounted:
        _issue(issues, "incorrect_request_accounting", f"planned {planned}, accounted {accounted}")
    mech_rows = ground_truth.get("mechanisms") if isinstance(ground_truth, dict) else None
    if not isinstance(mech_rows, list):
        _issue(issues, "malformed_ground_truth", "ground_truth.mechanisms must be a list")
        mech_rows = []
    mech_ids = [m.get("mechanism_id") for m in mech_rows if isinstance(m, dict)]
    if len(mech_ids) != len(set(mech_ids)):
        _issue(issues, "duplicate_mechanism_id", "duplicate mechanism IDs")
    if isinstance(experimental_condition, dict):
        labels = set(experimental_condition.get("active_mechanisms") or [])
        if "downstream_100ms" in labels:
            verified_downstream = any(
                isinstance(m, dict)
                and m.get("mechanism_type") == "downstream_latency"
                and (m.get("verification_evidence") or {}).get("verified") is True
                for m in mech_rows
            )
            if not verified_downstream:
                _issue(issues, "experimental_condition_mismatch", "downstream_100ms requires verified downstream injector evidence")
    injecting_times = [e.get("monotonic_time") for e in events if e.get("event_type") == "state" and e.get("state") == "INJECTING"]
    inject_start = min(injecting_times) if injecting_times else None
    configured = {m.mechanism_id: m for m in config.mechanisms} if config else {}
    if config is not None:
        if set(configured) != set(mech_ids):
            _issue(issues, "ground_truth_config_mechanism_mismatch", "configured mechanism IDs do not match ground truth")
        if config.runtime.runtime_type == "llamacpp":
            expected_condition = "HEALTHY_NO_DEGRADATION" if not config.mechanisms else "DEGRADATION_INJECTED"
            if ground_truth.get("run_condition") != expected_condition:
                _issue(issues, "invalid_ground_truth_condition", f"expected {expected_condition}")
    for mech in mech_rows:
        if not isinstance(mech, dict):
            _issue(issues, "malformed_ground_truth", "mechanism record is not an object")
            continue
        mid = mech.get("mechanism_id", "unknown")
        cfg_mech = configured.get(mid)
        if cfg_mech is None and config is not None:
            _issue(issues, "unknown_ground_truth_mechanism", str(mid))
        elif cfg_mech is not None:
            if mech.get("mechanism_type") != cfg_mech.mechanism_type:
                _issue(issues, "ground_truth_config_mechanism_mismatch", f"{mid} mechanism_type mismatch")
            if mech.get("target") != cfg_mech.target:
                _issue(issues, "ground_truth_config_mechanism_mismatch", f"{mid} target mismatch")
            if mech.get("intensity") != cfg_mech.intensity:
                _issue(issues, "ground_truth_config_mechanism_mismatch", f"{mid} intensity mismatch")
            if mech.get("scheduled_onset") != cfg_mech.scheduled_onset or mech.get("scheduled_stop") != cfg_mech.scheduled_stop:
                _issue(issues, "mechanism_schedule_violation", f"{mid} scheduled window mismatch")
        evidence = mech.get("verification_evidence")
        if mech.get("verification_state") in {"ACTIVE", "STOPPED"} and not evidence:
            _issue(issues, "active_without_evidence", mech.get("mechanism_id", "unknown"))
        if mech.get("verification_state") in {"ACTIVE", "STOPPED"} and evidence and evidence.get("verified") is not True:
            _issue(issues, "active_without_evidence", f"{mid} evidence verified=false")
        if mech.get("verification_state") in {"ACTIVE", "STOPPED"} and evidence:
            ev = evidence.get("evidence_value", {})
            if mech.get("actual_onset") is not None and ev.get("activation_time") is not None and abs(mech["actual_onset"] - ev["activation_time"]) > 1e-6:
                _issue(issues, "mechanism_onset_not_activation_time", str(mid))
            if mech.get("verification_time") is None or ev.get("verification_time") is None:
                _issue(issues, "mechanism_missing_verification_time", str(mid))
        if mech.get("verification_state") == "VERIFICATION_FAILED":
            _issue(issues, "mechanism_verification_failed", str(mid))
        if mech.get("actual_onset") is not None and inject_start is not None and mech["actual_onset"] < inject_start:
            _issue(issues, "mechanism_activation_before_injecting", mech.get("mechanism_id", "unknown"))
        if (
            mech.get("actual_onset") is not None
            and inject_start is not None
            and cfg_mech is not None
            and mech["actual_onset"] + 1e-12 < inject_start + cfg_mech.scheduled_onset
        ):
            _issue(issues, "mechanism_schedule_violation", f"{mid} actual onset precedes configured schedule")
        if mech.get("actual_onset") is not None and mech.get("actual_stop") is not None and mech["actual_stop"] < mech["actual_onset"]:
            _issue(issues, "mechanism_stop_before_onset", str(mid))
        if not mech.get("cleaned_up"):
            _issue(issues, "mechanism_not_cleaned_up", mech.get("mechanism_id", "unknown"))
        cleanup = ((evidence or {}).get("evidence_value") or {}).get("cleanup") if isinstance(evidence, dict) else None
        if cleanup and cleanup.get("cleaned") is not True:
            _issue(issues, "mechanism_not_cleaned_up", mech.get("mechanism_id", "unknown"))
    for problem in validate_lifecycle_history(events):
        _issue(issues, "illegal_lifecycle_history", problem)
    state_events = [e for e in events if e.get("event_type") == "state"]
    event_times = [e.get("monotonic_time") for e in events if isinstance(e.get("monotonic_time"), (int, float))]
    min_event = min(event_times) if event_times else None
    max_event = max(event_times) if event_times else None
    if min_event is not None and max_event is not None:
        for row in runtime_metric_rows:
            ts = row.get("timestamp")
            if isinstance(ts, (int, float)) and (ts + 1e-9 < min_event or ts - 1e-9 > max_event):
                _issue(issues, "runtime_metric_timestamp_out_of_bounds", str(ts))
        for row in system_metric_rows:
            ts = row.get("timestamp")
            if isinstance(ts, (int, float)) and (ts + 1e-9 < min_event or ts - 1e-9 > max_event):
                _issue(issues, "system_metric_timestamp_out_of_bounds", str(ts))
    if config and config.runtime.runtime_type == "llamacpp":
        if config.telemetry.system_metrics and not system_metric_rows:
            _issue(issues, "system_metrics_required_missing", "telemetry.system_metrics=true but no system metrics were recorded")
        if not config.telemetry.system_metrics and system_metric_rows:
            _issue(issues, "system_metrics_disabled_but_present", "telemetry.system_metrics=false but rows were recorded")
        if not config.telemetry.runtime_metrics and runtime_metric_rows:
            _issue(issues, "runtime_metrics_disabled_but_present", "telemetry.runtime_metrics=false but rows were recorded")
        if config.telemetry.traces and successful_request_ids and not trace_rows:
            _issue(issues, "required_traces_missing", "telemetry.traces=true but no trace spans were recorded for successful requests")
        if config.telemetry.traces and trace_rows:
            expected_names = {"gateway request", "dependency call", "llama call"}
            for rid in successful_request_ids:
                spans = trace_by_request.get(str(rid), [])
                names = [s.get("span_name") for s in spans]
                if set(names) != expected_names or len(spans) != 3:
                    _issue(issues, "missing_request_span", f"{rid}: expected exactly {sorted(expected_names)}, got {names}")
                    continue
                roots = [s for s in spans if s.get("span_name") == "gateway request"]
                if len(roots) != 1 or roots[0].get("parent_span_id") is not None:
                    _issue(issues, "missing_request_span", f"{rid}: expected one root gateway span")
                    continue
                root_span = roots[0]
                root_span_id = root_span.get("span_id")
                root_trace_id = root_span.get("trace_id")
                for child_name in {"dependency call", "llama call"}:
                    child = next((s for s in spans if s.get("span_name") == child_name), None)
                    if child is None:
                        _issue(issues, "missing_request_span", f"{rid}: missing {child_name}")
                        continue
                    if child.get("trace_id") != root_trace_id or child.get("parent_span_id") != root_span_id:
                        _issue(issues, "orphan_trace_span", f"{rid}:{child_name}")
        metrics_required = bool(config.runtime.parameters.get("metrics_enabled", False))
        metric_names = [row.get("metric_name") for row in runtime_metric_rows if row.get("metric_name")]
        if metrics_required and config.telemetry.runtime_metrics and not metric_names:
            _issue(issues, "runtime_metrics_required_missing", "metrics_enabled=true but no scraped runtime metrics were recorded")
        manifest_metadata = manifest.get("runtime_metadata", {}) if isinstance(manifest, dict) else {}
        if not (root / "runtime_metadata.json").exists():
            _issue(issues, "missing_runtime_metadata", "llamacpp run requires runtime_metadata.json")
        if "runtime_metadata.json" not in (manifest.get("artifact_inventory", []) if isinstance(manifest, dict) else []):
            _issue(issues, "manifest_inventory_omission", "runtime_metadata.json")
        if "runtime_metadata.json" not in (manifest.get("artifact_hashes", {}) if isinstance(manifest, dict) else {}):
            _issue(issues, "missing_runtime_metadata_hash", "runtime_metadata.json")
        reported = (manifest_metadata.get("model_metadata") or {}).get("selected_model_id")
        if reported and reported != config.runtime.model_id and not config.runtime.parameters.get("allow_model_mismatch", False):
            _issue(issues, "runtime_model_identity_mismatch", f"expected {config.runtime.model_id}, got {reported}")
        if "--no-cache-prompt" in config.runtime.parameters.get("server_arguments", []) and manifest_metadata.get("prompt_cache_enabled") is not False:
            _issue(issues, "prompt_cache_provenance_mismatch", "server configured --no-cache-prompt but runtime metadata does not record prompt_cache_enabled=false")
        server_pid = config.runtime.parameters.get("server_pid")
        gateway_pid = config.runtime.parameters.get("gateway_pid")
        dependency_pid = config.runtime.parameters.get("dependency_pid")
        for row in system_metric_rows:
            if server_pid is None and (row.get("server_process_cpu_percent") is not None or row.get("server_process_rss_bytes") is not None or row.get("server_pid") is not None):
                _issue(issues, "unexpected_server_process_telemetry", "server telemetry present without configured server_pid")
            if server_pid is not None and row.get("server_pid") != server_pid:
                _issue(issues, "server_pid_mismatch", f"expected {server_pid}, got {row.get('server_pid')}")
            if gateway_pid is None and (row.get("gateway_process_cpu_percent") is not None or row.get("gateway_process_rss_bytes") is not None or row.get("gateway_pid") is not None):
                _issue(issues, "unexpected_gateway_process_telemetry", "gateway telemetry present without configured gateway_pid")
            if gateway_pid is not None and row.get("gateway_pid") != gateway_pid:
                _issue(issues, "gateway_pid_mismatch", f"expected {gateway_pid}, got {row.get('gateway_pid')}")
            if dependency_pid is None and (row.get("dependency_process_cpu_percent") is not None or row.get("dependency_process_rss_bytes") is not None or row.get("dependency_pid") is not None):
                _issue(issues, "unexpected_dependency_process_telemetry", "dependency telemetry present without configured dependency_pid")
            if dependency_pid is not None and row.get("dependency_pid") != dependency_pid:
                _issue(issues, "dependency_pid_mismatch", f"expected {dependency_pid}, got {row.get('dependency_pid')}")
        if config.mechanisms and any(m.mechanism_type == "downstream_latency" for m in config.mechanisms):
            for row in requests:
                if row.get("status") != "success":
                    continue
                required_gateway_fields = [
                    "gateway_receive_time",
                    "dependency_start_time",
                    "dependency_end_time",
                    "llama_dispatch_time",
                    "gateway_completion_time",
                    "dependency_duration",
                ]
                missing = [name for name in required_gateway_fields if row.get(name) is None]
                if missing:
                    _issue(issues, "missing_gateway_timing", f"{row.get('request_id')}: {','.join(missing)}")
    if state_events and isinstance(manifest, dict):
        terminal = state_events[-1].get("state")
        if terminal != manifest.get("final_lifecycle_state"):
            _issue(issues, "terminal_manifest_lifecycle_mismatch", "manifest final lifecycle state disagrees with events")
        if terminal == "FAILED":
            _issue(issues, "failed_lifecycle_state", "run ended in FAILED")
        if terminal == "COMPLETE":
            recovery_times = [e.get("monotonic_time") for e in state_events if e.get("state") == "RECOVERY"]
            complete_time = state_events[-1].get("monotonic_time")
            if not recovery_times:
                _issue(issues, "complete_before_recovery", "COMPLETE requires RECOVERY")
            request_completion = [r.get("completion_time") for r in requests if isinstance(r.get("completion_time"), (int, float))]
            if request_completion and isinstance(complete_time, (int, float)) and complete_time < max(request_completion):
                _issue(issues, "complete_before_request_completion", "COMPLETE precedes request completion")
            if any(not m.get("cleaned_up") for m in mech_rows if isinstance(m, dict)):
                _issue(issues, "complete_before_cleanup", "COMPLETE requires mechanism cleanup")
        if config and config.runtime.runtime_type == "llamacpp":
            baseline_times = [e.get("monotonic_time") for e in state_events if e.get("state") == "BASELINE"]
            recovery_times = [e.get("monotonic_time") for e in state_events if e.get("state") == "RECOVERY"]
            if baseline_times and recovery_times:
                first_recovery = min(recovery_times)
                for row in requests:
                    if row.get("actual_arrival") is not None and row["actual_arrival"] >= first_recovery:
                        _issue(issues, "workload_not_in_baseline", row.get("request_id", "unknown"))
            if any(e.get("event_type") == "experiment_timeout" for e in events):
                _issue(issues, "experiment_timeout", "experiment timed out")
            if config.safety.require_requests_within_mechanism_window and mech_rows:
                active_windows = [(m.get("actual_onset"), m.get("actual_stop")) for m in mech_rows if m.get("verification_state") == "STOPPED"]
                for row in requests:
                    if row.get("status") != "success":
                        continue
                    covered = bool(active_windows) and all(
                        start is not None
                        and stop is not None
                        and row.get("actual_arrival") is not None
                        and row.get("completion_time") is not None
                        and row["actual_arrival"] + 1e-9 >= start
                        and row["completion_time"] <= stop + 1e-9
                        for start, stop in active_windows
                    )
                    if not covered:
                        _issue(issues, "request_outside_required_mechanism_window", row.get("request_id", "unknown"))
    result = {
        "valid": not issues,
        "issues": issues,
        "run_id": run_id,
        "planned_requests": planned,
        "accounted_requests": accounted,
    }
    return result
