from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from sloscope.artifacts.writer import ARTIFACT_SCHEMA_VERSION, SUPPORTED_ARTIFACT_SCHEMA_VERSIONS, read_json, read_jsonl, read_table, sha256_file
from sloscope.config import ExperimentConfig, SUPPORTED_SCHEMA_VERSION
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
    cfg_data = loaded.get("config.json")
    manifest = loaded.get("manifest.json", {})
    workload = loaded.get("workload.json", {}).get("requests", []) if isinstance(loaded.get("workload.json"), dict) else []
    requests = loaded.get("requests.parquet", ({}, []))[1] if "requests.parquet" in loaded else []
    system_metric_rows = loaded.get("system_metrics.parquet", ({}, []))[1] if "system_metrics.parquet" in loaded else []
    runtime_metric_rows = loaded.get("runtime_metrics.parquet", ({}, []))[1] if "runtime_metrics.parquet" in loaded else []
    ground_truth = loaded.get("ground_truth.json", {})
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
    config = None
    if isinstance(cfg_data, dict):
        try:
            config = ExperimentConfig.from_dict(cfg_data)
            if config.schema_version != SUPPORTED_SCHEMA_VERSION:
                _issue(issues, "unknown_schema_version", f"unknown config schema {config.schema_version}")
            if config.config_hash != config.compute_hash():
                _issue(issues, "config_hash_mismatch", "config hash does not match canonical config")
            if manifest and manifest.get("config_hash") != config.config_hash:
                _issue(issues, "manifest_config_hash_mismatch", "manifest/config hash mismatch")
        except Exception as exc:
            _issue(issues, "malformed_config", str(exc))
    planned_ids = [r.get("request_id") for r in workload if isinstance(r, dict)]
    if len(planned_ids) != len(set(planned_ids)):
        _issue(issues, "duplicate_request_id", "duplicate request IDs in workload")
    request_ids = [r.get("request_id") for r in requests if isinstance(r, dict)]
    if len(request_ids) != len(set(request_ids)):
        _issue(issues, "duplicate_request_id", "duplicate request IDs in request telemetry")
    if set(request_ids) - set(planned_ids):
        _issue(issues, "unknown_request", "request telemetry contains unplanned request IDs")
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
        if config.telemetry.traces:
            _issue(issues, "llamacpp_traces_not_supported", "Phase 3B llama.cpp requires traces=false")
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
        for row in system_metric_rows:
            if server_pid is None and (row.get("server_process_cpu_percent") is not None or row.get("server_process_rss_bytes") is not None or row.get("server_pid") is not None):
                _issue(issues, "unexpected_server_process_telemetry", "server telemetry present without configured server_pid")
            if server_pid is not None and row.get("server_pid") != server_pid:
                _issue(issues, "server_pid_mismatch", f"expected {server_pid}, got {row.get('server_pid')}")
    if state_events and isinstance(manifest, dict):
        terminal = state_events[-1].get("state")
        if terminal != manifest.get("final_lifecycle_state"):
            _issue(issues, "terminal_manifest_lifecycle_mismatch", "manifest final lifecycle state disagrees with events")
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
                    covered = any(
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
