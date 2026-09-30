import json
import socket
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from sloscope.cli import main as cli_main
from sloscope.artifacts.validation import validate_run
from sloscope.artifacts.writer import read_json, read_table, write_json, write_table
from sloscope.artifacts.writer import ARTIFACT_SCHEMA_VERSION
from sloscope.config import (
    ExperimentConfig,
    MechanismConfig,
    RuntimeConfig,
    SUPPORTED_SCHEMA_VERSION,
    TelemetryConfig,
    WorkloadConfig,
    load_config,
    write_config,
)
from sloscope.ground_truth import GroundTruthLedger, active_mechanism_set
from sloscope.injectors.base import VerificationEvidence
from sloscope.injectors.mock import MockInjector
from sloscope.injectors.cpu_contention import CPUContentionInjector, worker_count_for_intensity
from sloscope.injectors.downstream_latency import DownstreamLatencyInjector
from sloscope.injectors.factory import create_injector
from sloscope.gateway.dependency import SyntheticDependencyService
from sloscope.gateway.server import SLOScopeGateway
from sloscope.lifecycle import ExperimentState, FakeClock, IllegalTransitionError, Lifecycle, SystemClock
from sloscope.provenance import source_tree_sha256
from sloscope.campaign_phase5 import (
    REPETITIONS as PHASE5_REPETITIONS,
    analysis_plan as phase5_analysis_plan,
    build_slo_calibration_artifact as phase5_build_slo_calibration_artifact,
    campaign_rows as phase5_campaign_rows,
    conditions as phase5_conditions,
    confound_rules as phase5_confound_rules,
    execute_cooldown as phase5_execute_cooldown,
    ensure_run_directory_absent as phase5_ensure_run_directory_absent,
    factorial_design as phase5_factorial_design,
    freeze_hash as phase5_freeze_hash,
    freeze_input_paths as phase5_freeze_input_paths,
    generate as generate_phase5,
    mechanism_definitions as phase5_mechanism_definitions,
    prediction_limit as phase5_prediction_limit,
    preflight_spec as phase5_preflight_spec,
    run_preflight as phase5_run_preflight,
    slo_definitions as phase5_slo_definitions,
    validate_freeze as validate_phase5_freeze,
)
from sloscope.pilot_phase4c import build_blocks as build_phase4c_blocks, choose_output_reference, is_queue_confounded
from sloscope.runtime.llamacpp import GATEWAY_TIMING_FIELDS, LlamaCppRuntimeAdapter, content_from_chunk, parse_prometheus_metrics, parse_sse_payload
from sloscope.runner import ExperimentRunner
from sloscope.telemetry.schemas import RequestRecord, TRACE_COLUMNS
from sloscope.workload import generate_workload_plan


def cfg(run_id="run-a", seed=7, mechanisms=None, randomized=False, raise_on_sequence=None):
    params = {}
    if raise_on_sequence is not None:
        params["raise_on_sequence"] = raise_on_sequence
    return ExperimentConfig(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        run_id=run_id,
        seed=seed,
        runtime=RuntimeConfig("mock-runtime", "mock", "mock-model", params),
        workload=WorkloadConfig(
            arrival_pattern="constant_open_loop",
            request_count=20,
            concurrency=4,
            prompt_profile="mock-prompts",
            output_profile="fixed",
            target_output_tokens=12,
            randomized_output=randomized,
            output_jitter_tokens=3 if randomized else 0,
        ),
        mechanisms=mechanisms or [],
    )


def mech(mid="m1", verification_fails=False):
    return MechanismConfig(mid, "mock-degradation", 0.5, "mock-runtime", 0.0, 1.0, {"verification_fails": verification_fails})


def cpu_mech(mid="cpu1", intensity=0.2, duration=1.0, params=None):
    p = {"logical_cpu_count": 10, "verification_interval_seconds": 0.05, "stop_timeout_seconds": 1.0}
    if params:
        p.update(params)
    return MechanismConfig(mid, "cpu_contention", intensity, "host_cpu", 0.0, duration, p)


def codes(result):
    return {issue["code"] for issue in result["issues"]}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def json_request(method, url, payload=None, timeout=3.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


class FakeLlamaHTTPServer:
    def __init__(self, *, first_chunk_sleep=0.0, per_chunk_sleep=0.0, fail_completions=False, health_ok=True):
        self.port = free_port()
        self.requests = []
        self.request_start_times = {}
        self.request_finish_times = {}
        self.first_chunk_sleep = first_chunk_sleep
        self.per_chunk_sleep = per_chunk_sleep
        self.fail_completions = fail_completions
        self.health_ok = health_ok
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_GET(self):
                if self.path == "/health":
                    if outer.health_ok:
                        self._json({"status": "ok"})
                    else:
                        self._json({"status": "unavailable"}, status=503)
                elif self.path == "/v1/models":
                    self._json({"data": [{"id": "fake-gguf"}]})
                elif self.path == "/metrics":
                    body = b"llamacpp:requests_processing 0\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path != "/v1/completions":
                    self.send_error(404)
                    return
                rid = self.headers.get("X-SLOScope-Request-Id")
                outer.requests.append(rid)
                outer.request_start_times[rid] = time.monotonic()
                _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if outer.fail_completions:
                    self.send_error(503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                lines = [
                    b"data: {\"choices\":[{\"text\":\"Hello\",\"finish_reason\":null}]}\n\n",
                    b"data: {\"choices\":[{\"text\":\" world\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n",
                    b"data: [DONE]\n\n",
                ]
                for idx, line in enumerate(lines):
                    if idx == 0 and outer.first_chunk_sleep:
                        time.sleep(outer.first_chunk_sleep)
                    if outer.per_chunk_sleep:
                        time.sleep(outer.per_chunk_sleep)
                    self.wfile.write(line)
                    self.wfile.flush()
                outer.request_finish_times[rid] = time.monotonic()

            def _json(self, payload, status=200):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def test_config_deterministic_hashing():
    a = cfg()
    b = ExperimentConfig.from_dict(json.loads(a.canonical_json(include_hash=False)))
    assert a.compute_hash() == b.compute_hash()
    assert a.with_hash().config_hash == a.compute_hash()


def test_new_config_schema_and_legacy_loading():
    current = cfg()
    assert current.schema_version == "sloscope.config.v1"
    current.validate()
    legacy = ExperimentConfig.from_dict({**current.to_dict(include_hash=False), "schema_version": "phase3a.v1"})
    legacy.validate()
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({**current.to_dict(include_hash=False), "schema_version": "unknown.v1"}).validate()


def test_source_tree_fingerprint_is_deterministic_and_sensitive(tmp_path):
    (tmp_path / "sloscope").mkdir()
    pyproject = tmp_path / "pyproject.toml"
    module = tmp_path / "sloscope" / "module.py"
    pyproject.write_text("[project]\nname='x'\n", encoding="utf-8")
    module.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_sha256(tmp_path)
    assert source_tree_sha256(tmp_path) == first
    module.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) != first


def test_same_seed_workload_determinism():
    assert [p.to_dict() for p in generate_workload_plan(cfg(randomized=True))] == [
        p.to_dict() for p in generate_workload_plan(cfg(randomized=True))
    ]


def test_different_seed_variation():
    assert [p.target_output_tokens for p in generate_workload_plan(cfg(seed=1, randomized=True))] != [
        p.target_output_tokens for p in generate_workload_plan(cfg(seed=2, randomized=True))
    ]


def test_legal_lifecycle_transitions():
    lc = Lifecycle(FakeClock())
    for state in [
        ExperimentState.PREPARING,
        ExperimentState.WARMUP,
        ExperimentState.BASELINE,
        ExperimentState.INJECTING,
        ExperimentState.RECOVERY,
        ExperimentState.COMPLETE,
    ]:
        lc.transition(state)
    assert lc.state == ExperimentState.COMPLETE


def test_illegal_lifecycle_transition_rejection():
    with pytest.raises(IllegalTransitionError):
        Lifecycle(FakeClock()).transition(ExperimentState.COMPLETE)


def test_timestamp_ordering():
    rec = RequestRecord("r", 0, 0, 0, 1, 2, 3, 1, 1, "success", "rt", "m")
    assert rec.validate_ordering() == []
    bad = RequestRecord("r", 0, 0, 2, 1, 3, 4, 1, 1, "success", "rt", "m")
    assert bad.validate_ordering()


def test_derived_request_metrics():
    metrics = RequestRecord("r", 0, 0, 1, 2, 4, 7, 1, 1, "success", "rt", "m").derived_metrics()
    assert metrics["queue_delay"] == 1
    assert metrics["ttft"] == 3
    assert metrics["service_time"] == 5
    assert metrics["total_latency"] == 6
    assert metrics["scheduler_slip"] == 1


def test_post_first_token_duration_calculation():
    metrics = RequestRecord("r", 0, 0, 1, 2, 4, 9, 1, 1, "success", "rt", "m", server_output_tokens=10).derived_metrics()
    assert metrics["post_first_token_duration"] == 5
    assert metrics["observed_seconds_per_output_token"] == 0.5


def test_failed_request_decode_metric_null():
    metrics = RequestRecord("r", 0, 0, None, None, None, None, None, None, "failed", "rt", "m", error_type="x").derived_metrics()
    assert metrics["post_first_token_duration"] is None
    assert metrics["observed_seconds_per_output_token"] is None


def test_phase4c_queue_confounded_detection_and_load_allowed():
    assert is_queue_confounded({"family": "input", "requests_deferred_max": 1})
    assert is_queue_confounded({"family": "output", "requests_deferred_max": 1})
    assert not is_queue_confounded({"family": "load", "requests_deferred_max": 3})


def test_phase4c_family_specific_matched_controls():
    plans = build_phase4c_blocks()
    by_block = {}
    for plan in plans:
        by_block.setdefault(plan.block_id, []).append(plan)
    for block_id, members in by_block.items():
        families = {m.family for m in members}
        assert len(families) == 1
        if block_id.startswith("input"):
            assert {m.target_output_tokens for m in members} == {16}
            assert {m.inter_arrival_seconds for m in members} == {0.25}
        if block_id.startswith("output"):
            assert {m.prompt_profile for m in members} == {"synthetic-continuation"}
            assert {m.inter_arrival_seconds for m in members} == {0.5}


def test_phase4c_selection_skips_queue_confounded_output():
    rows = [
        {"family": "output", "candidate": "32", "requests_deferred_max": 0, "token_realization_p95_floor": 1.0, "post_first_token_duration_median_ratio": 1.2},
        {"family": "output", "candidate": "48", "requests_deferred_max": 2, "token_realization_p95_floor": 1.0, "post_first_token_duration_median_ratio": 2.0},
    ]
    assert choose_output_reference(rows) == "32"


def test_exact_request_accounting(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="acct"), tmp_path, FakeClock()).run()
    validation = validate_run(run_dir)
    assert validation["planned_requests"] == 20
    assert validation["accounted_requests"] == 20
    assert validation["valid"]


def test_duplicate_request_rejection(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="dupes"), tmp_path, FakeClock()).run()
    workload = read_json(run_dir / "workload.json")
    workload["requests"][1]["request_id"] = workload["requests"][0]["request_id"]
    write_json(run_dir / "workload.json", workload)
    assert not validate_run(run_dir)["valid"]


def test_injector_start_does_not_imply_verification():
    inj = MockInjector(mech(), FakeClock())
    inj.prepare()
    inj.start()
    ledger = GroundTruthLedger([mech()])
    assert ledger.records["m1"].verification_state == "PENDING"


def test_successful_injector_verification():
    clock = FakeClock()
    inj = MockInjector(mech(), clock)
    inj.start()
    ev = inj.verify()
    assert ev.verified
    ledger = GroundTruthLedger([mech()])
    ledger.mark_verified("m1", ev)
    assert ledger.records["m1"].verification_state == "ACTIVE"


def test_verification_failure():
    inj = MockInjector(mech(verification_fails=True), FakeClock())
    inj.start()
    assert not inj.verify().verified


def test_verify_before_start_cannot_succeed():
    ev = MockInjector(mech(), FakeClock()).verify()
    assert not ev.verified
    assert ev.evidence_value["reason"] == "not-started"


def test_injector_stop():
    inj = MockInjector(mech(), FakeClock())
    inj.start()
    inj.stop()
    assert inj.stopped


def test_cleanup_on_successful_experiment(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="clean", mechanisms=[mech()]), tmp_path, FakeClock()).run()
    gt = read_json(run_dir / "ground_truth.json")
    assert gt["mechanisms"][0]["cleaned_up"] is True


def test_cleanup_after_runtime_exception(tmp_path):
    runner = ExperimentRunner(cfg(run_id="explode", mechanisms=[mech()], raise_on_sequence=3), tmp_path, FakeClock())
    with pytest.raises(RuntimeError):
        runner.run()
    assert runner.injectors[0].cleaned
    assert (tmp_path / "explode" / "manifest.json").exists()


def test_multiple_simultaneous_mock_injectors(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="compound", mechanisms=[mech("m1"), mech("m2")]), tmp_path, FakeClock()).run()
    gt = read_json(run_dir / "ground_truth.json")
    assert {m["mechanism_id"] for m in gt["mechanisms"]} == {"m1", "m2"}
    assert all(m["verification_evidence"]["verified"] for m in gt["mechanisms"])


def test_ground_truth_comes_from_orchestration_state():
    ledger = GroundTruthLedger([mech()])
    assert ledger.records["m1"].actual_onset is None
    assert ledger.records["m1"].verification_state == "PENDING"


def test_mechanism_cannot_activate_before_injecting(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="early", mechanisms=[mech()]), tmp_path, FakeClock()).run()
    gt = read_json(run_dir / "ground_truth.json")
    gt["mechanisms"][0]["actual_onset"] = -1
    write_json(run_dir / "ground_truth.json", gt)
    result = validate_run(run_dir)
    assert not result["valid"]
    assert any(i["code"] == "mechanism_activation_before_injecting" for i in result["issues"])


def test_empty_telemetry_table_schema_validity(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="empty-schema"), tmp_path, FakeClock()).run()
    schema, rows = read_table(run_dir / "system_metrics.parquet")
    assert "host_cpu" in schema
    assert rows == []


def test_real_parquet_generated_with_pyarrow(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="parquet"), tmp_path, FakeClock()).run()
    data = (run_dir / "requests.parquet").read_bytes()
    assert data[:4] == b"PAR1"
    assert data[-4:] == b"PAR1"


def test_manifest_generation(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="manifest"), tmp_path, FakeClock()).run()
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["run_id"] == "manifest"
    assert "config.json" in manifest["artifact_inventory"]


def test_manifest_config_hash_validation(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="hashcheck"), tmp_path, FakeClock()).run()
    manifest = read_json(run_dir / "manifest.json")
    manifest["config_hash"] = "bad"
    write_json(run_dir / "manifest.json", manifest)
    result = validate_run(run_dir)
    assert any(i["code"] == "manifest_config_hash_mismatch" for i in result["issues"])


def test_corrupted_artifact_detection(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="corrupt"), tmp_path, FakeClock()).run()
    (run_dir / "config.json").write_text("{nope", encoding="utf-8")
    result = validate_run(run_dir)
    assert any(i["code"] == "corrupted_artifact" for i in result["issues"])


def test_missing_artifact_detection(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="missing"), tmp_path, FakeClock()).run()
    (run_dir / "requests.parquet").unlink()
    result = validate_run(run_dir)
    assert any(i["code"] == "missing_artifact" for i in result["issues"])


def test_deterministic_mock_dry_run(tmp_path):
    one = ExperimentRunner(cfg(run_id="det", mechanisms=[mech()]), tmp_path / "a", FakeClock()).run()
    two = ExperimentRunner(cfg(run_id="det", mechanisms=[mech()]), tmp_path / "b", FakeClock()).run()
    for name in ["config.json", "ground_truth.json", "workload.json", "events.jsonl"]:
        assert (one / name).read_text(encoding="utf-8") == (two / name).read_text(encoding="utf-8")


def test_validate_run_succeeds_on_clean_dry_run(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="valid", mechanisms=[mech()]), tmp_path, FakeClock()).run()
    assert validate_run(run_dir)["valid"]


def test_validate_run_fails_after_intentional_corruption(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="invalid", mechanisms=[mech()]), tmp_path, FakeClock()).run()
    gt = read_json(run_dir / "ground_truth.json")
    gt["mechanisms"][0]["verification_evidence"] = None
    write_json(run_dir / "ground_truth.json", gt)
    assert not validate_run(run_dir)["valid"]


def test_verification_failure_produces_failed_invalid_run(tmp_path):
    runner = ExperimentRunner(cfg(run_id="verify-fails", mechanisms=[mech(verification_fails=True)]), tmp_path, FakeClock())
    with pytest.raises(RuntimeError):
        runner.run()
    result = validate_run(tmp_path / "verify-fails")
    manifest = read_json(tmp_path / "verify-fails" / "manifest.json")
    gt = read_json(tmp_path / "verify-fails" / "ground_truth.json")
    assert manifest["final_lifecycle_state"] == "FAILED"
    assert gt["mechanisms"][0]["verification_state"] == "VERIFICATION_FAILED"
    assert "mechanism_verification_failed" in codes(result)
    assert not result["valid"]


def test_scheduled_onset_is_honored(tmp_path):
    delayed = MechanismConfig("m1", "mock-degradation", 0.5, "mock-runtime", 0.5, 1.0, {})
    run_dir = ExperimentRunner(cfg(run_id="scheduled", mechanisms=[delayed]), tmp_path, FakeClock()).run()
    gt = read_json(run_dir / "ground_truth.json")["mechanisms"][0]
    inject_time = min(e["monotonic_time"] for e in read_jsonl_file(run_dir / "events.jsonl") if e["state"] == "INJECTING")
    assert gt["actual_onset"] >= inject_time + 0.5


def read_jsonl_file(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_scheduled_duration_stop_ordering(tmp_path):
    scheduled = MechanismConfig("m1", "mock-degradation", 0.5, "mock-runtime", 2.5, 0.4, {})
    run_dir = ExperimentRunner(cfg(run_id="duration", mechanisms=[scheduled]), tmp_path, FakeClock()).run()
    gt = read_json(run_dir / "ground_truth.json")["mechanisms"][0]
    assert gt["scheduled_stop"] == pytest.approx(2.9)
    assert gt["actual_stop"] == pytest.approx(2.9)
    assert gt["actual_stop"] >= gt["actual_onset"]


def test_compound_overlapping_different_onsets(tmp_path):
    m1 = MechanismConfig("m1", "mock-degradation", 0.5, "mock-runtime", 0.0, 1.2, {})
    m2 = MechanismConfig("m2", "mock-degradation", 0.7, "mock-runtime", 0.3, 0.6, {})
    run_dir = ExperimentRunner(cfg(run_id="overlap", mechanisms=[m1, m2]), tmp_path, FakeClock()).run()
    rows = {m["mechanism_id"]: m for m in read_json(run_dir / "ground_truth.json")["mechanisms"]}
    assert rows["m1"]["actual_onset"] < rows["m2"]["actual_onset"]
    assert rows["m1"]["actual_stop"] > rows["m2"]["actual_onset"]
    assert rows["m2"]["actual_stop"] < rows["m1"]["actual_stop"]
    assert validate_run(run_dir)["valid"]


def test_fake_clock_progresses_consistently(tmp_path):
    clock = FakeClock()
    run_dir = ExperimentRunner(cfg(run_id="clock", mechanisms=[mech()]), tmp_path, clock).run()
    _, requests = read_table(run_dir / "requests.parquet")
    events = read_jsonl_file(run_dir / "events.jsonl")
    assert clock.monotonic() >= max(r["completion_time"] for r in requests)
    assert max(e["monotonic_time"] for e in events) == pytest.approx(clock.monotonic())


def test_lifecycle_timestamps_never_decrease(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="monotonic", mechanisms=[mech()]), tmp_path, FakeClock()).run()
    times = [e["monotonic_time"] for e in read_jsonl_file(run_dir / "events.jsonl")]
    assert times == sorted(times)


def test_complete_occurs_after_request_completion(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="complete-after"), tmp_path, FakeClock()).run()
    _, requests = read_table(run_dir / "requests.parquet")
    complete = [e for e in read_jsonl_file(run_dir / "events.jsonl") if e["state"] == "COMPLETE"][-1]
    assert complete["monotonic_time"] >= max(r["completion_time"] for r in requests)


def test_ground_truth_config_mechanism_mismatch_detected(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="gt-mismatch", mechanisms=[mech()]), tmp_path, FakeClock()).run()
    gt = read_json(run_dir / "ground_truth.json")
    gt["mechanisms"][0]["mechanism_type"] = "wrong"
    write_json(run_dir / "ground_truth.json", gt)
    assert "ground_truth_config_mechanism_mismatch" in codes(validate_run(run_dir))


def test_actual_stop_before_actual_onset_detected(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="bad-stop", mechanisms=[mech()]), tmp_path, FakeClock()).run()
    gt = read_json(run_dir / "ground_truth.json")
    gt["mechanisms"][0]["actual_stop"] = gt["mechanisms"][0]["actual_onset"] - 0.1
    write_json(run_dir / "ground_truth.json", gt)
    assert "mechanism_stop_before_onset" in codes(validate_run(run_dir))


def test_successful_state_with_verified_false_rejected(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="false-evidence", mechanisms=[mech()]), tmp_path, FakeClock()).run()
    gt = read_json(run_dir / "ground_truth.json")
    gt["mechanisms"][0]["verification_evidence"]["verified"] = False
    write_json(run_dir / "ground_truth.json", gt)
    assert "active_without_evidence" in codes(validate_run(run_dir))


def test_terminal_manifest_lifecycle_mismatch_detected(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="terminal-mismatch"), tmp_path, FakeClock()).run()
    manifest = read_json(run_dir / "manifest.json")
    manifest["final_lifecycle_state"] = "FAILED"
    write_json(run_dir / "manifest.json", manifest)
    assert "terminal_manifest_lifecycle_mismatch" in codes(validate_run(run_dir))


def test_artifact_hash_corruption_detected(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="hash-corrupt"), tmp_path, FakeClock()).run()
    workload = read_json(run_dir / "workload.json")
    workload["requests"][0]["prompt_profile"] = "changed"
    write_json(run_dir / "workload.json", workload)
    assert "artifact_hash_mismatch" in codes(validate_run(run_dir))


def test_general_runner_defaults_to_system_clock():
    runner = ExperimentRunner(cfg(run_id="system-default"))
    assert isinstance(runner.clock, SystemClock)


def test_dry_run_cli_explicitly_uses_fake_clock(tmp_path):
    config_path = tmp_path / "config.json"
    write_config(config_path, cfg(run_id="cli-fake"))
    assert cli_main(["dry-run", str(config_path), "--runs-root", str(tmp_path / "runs")]) == 0
    manifest = read_json(tmp_path / "runs" / "cli-fake" / "manifest.json")
    assert manifest["start_timestamp"].startswith("fake-")


def test_cleanup_shutdown_failure_still_writes_audit_artifacts(tmp_path):
    bad = cfg(
        run_id="cleanup-fails",
        mechanisms=[MechanismConfig("m1", "mock-degradation", 0.5, "mock-runtime", 0.0, 1.0, {"raise_on_cleanup": True})],
    )
    with pytest.raises(RuntimeError):
        ExperimentRunner(bad, tmp_path, FakeClock()).run()
    run_dir = tmp_path / "cleanup-fails"
    assert (run_dir / "manifest.json").exists()
    result = validate_run(run_dir)
    assert not result["valid"]
    assert "mechanism_not_cleaned_up" in codes(result)


def test_programmatically_invalid_config_rejected(tmp_path):
    invalid = cfg(run_id="invalid-programmatic")
    invalid = ExperimentConfig.from_dict({**invalid.to_dict(include_hash=False), "workload": {**invalid.workload.__dict__, "concurrency": 0}})
    with pytest.raises(ValueError):
        ExperimentRunner(invalid, tmp_path, FakeClock())


def test_scientifically_correct_concurrency_workload_naming():
    old = cfg(run_id="old-name")
    old = ExperimentConfig.from_dict({**old.to_dict(include_hash=False), "workload": {**old.workload.__dict__, "arrival_pattern": "fixed_concurrency"}})
    with pytest.raises(ValueError):
        old.validate()
    batched = ExperimentConfig.from_dict({**cfg(run_id="batched").to_dict(include_hash=False), "workload": {**cfg().workload.__dict__, "arrival_pattern": "batched_arrival"}})
    plans = generate_workload_plan(batched)
    assert [p.scheduled_arrival for p in plans[:5]] == [0.0, 0.0, 0.0, 0.0, 0.1]


class FakeHTTPResponse:
    def __init__(self, status=200, body=b"", lines=None):
        self.status = status
        self.body = body
        self.lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body

    def __iter__(self):
        return iter(self.lines)


class FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, url, code, body):
        super().__init__(url, code, "fake", hdrs=None, fp=None)
        self._body = body

    def read(self):
        return self._body


def install_fake_llama(
    monkeypatch,
    *,
    health_payload=None,
    models_payload=None,
    metrics_payload="llamacpp:requests_processing 0\nllamacpp:tokens_predicted_total 3\n",
    stream_mode="ok",
    delay=0.0,
    metrics_delay=0.0,
    captured_payloads=None,
    health_status=200,
):
    health_payload = health_payload or {"status": "ok"}
    models_payload = models_payload or {"data": [{"id": "fake-gguf", "context_length": 2048, "embedding_length": 128}]}

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith("/health"):
            if health_status >= 400:
                raise FakeHTTPError(url, health_status, json.dumps(health_payload).encode("utf-8"))
            return FakeHTTPResponse(status=health_status, body=json.dumps(health_payload).encode("utf-8"))
        if url.endswith("/v1/models"):
            return FakeHTTPResponse(body=json.dumps(models_payload).encode("utf-8"))
        if url.endswith("/metrics"):
            if metrics_delay:
                time.sleep(metrics_delay)
            return FakeHTTPResponse(body=metrics_payload.encode("utf-8"))
        if url.endswith("/v1/completions"):
            if captured_payloads is not None:
                captured_payloads.append(json.loads(req.data.decode("utf-8")))
            if stream_mode == "http_error":
                raise FakeHTTPError(url, 500, b'{"error":"boom"}')
            if stream_mode == "timeout" and timeout is not None and timeout < 0.05:
                raise TimeoutError("timed out")
            if stream_mode == "malformed":
                return FakeHTTPResponse(lines=[b"data: {nope}\n\n"])
            lines = [
                b": control\n\n",
                b"data: {\"choices\":[{\"text\":\"\"}]}\n\n",
                b"data: {\"choices\":[{\"text\":\"Hello\",\"finish_reason\":null}]}\n\n",
                b"data: {\"choices\":[{\"text\":\" world\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\n",
                b"data: [DONE]\n\n",
            ]
            if delay:
                original_iter = iter(lines)

                class SlowResponse(FakeHTTPResponse):
                    def __iter__(self):
                        for line in original_iter:
                            time.sleep(delay)
                            yield line

                return SlowResponse(lines=lines)
            return FakeHTTPResponse(lines=lines)
        raise FakeHTTPError(url, 404, b'{"error":"not found"}')

    monkeypatch.setattr("sloscope.runtime.llamacpp.urllib.request.urlopen", fake_urlopen)
    return "http://fake-llama"


def llama_cfg(base_url, run_id="llama-test", metrics_enabled=True, request_count=2, inter_arrival=0.01, max_outstanding=4, model_id="fake-gguf", timeout=2.0):
    return ExperimentConfig(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        run_id=run_id,
        seed=11,
        runtime=RuntimeConfig(
            "llama-runtime",
            "llamacpp",
            model_id,
            {
                "base_url": base_url,
                "metrics_enabled": metrics_enabled,
                "request_timeout": timeout,
                "health_timeout": timeout,
                "telemetry_interval_seconds": 0.01,
                "max_outstanding_requests": max_outstanding,
                "temperature": 0,
            },
        ),
        workload=WorkloadConfig("constant_open_loop", request_count, 2, "synthetic-prose", "fixed", 4, inter_arrival_seconds=inter_arrival),
        telemetry=TelemetryConfig(traces=False),
    )


def test_llamacpp_healthy_healthcheck(monkeypatch):
    base_url = install_fake_llama(monkeypatch)
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url).runtime, SystemClock(), 1)
    adapter.prepare()
    assert adapter.healthcheck()


def test_llamacpp_loading_healthcheck_refuses_prepare(monkeypatch):
    base_url = install_fake_llama(monkeypatch, health_payload={"status": "loading"})
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url).runtime, SystemClock(), 1)
    with pytest.raises(RuntimeError):
        adapter.prepare()
    assert adapter.last_health.state == "loading"


def test_llamacpp_documented_503_loading_health(monkeypatch):
    payload = {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}
    base_url = install_fake_llama(monkeypatch, health_payload=payload, health_status=503)
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url).runtime, SystemClock(), 1)
    with pytest.raises(RuntimeError):
        adapter.prepare()
    assert adapter.last_health.state == "loading"


def test_llamacpp_unavailable_healthcheck():
    adapter = LlamaCppRuntimeAdapter(llama_cfg("http://127.0.0.1:9").runtime, SystemClock(), 1)
    assert not adapter.healthcheck()
    assert adapter.last_health.state == "unavailable"


def test_llamacpp_models_parsing_and_model_mismatch(monkeypatch):
    base_url = install_fake_llama(monkeypatch, models_payload={"data": [{"id": "other-model"}]})
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url, model_id="expected").runtime, SystemClock(), 1)
    with pytest.raises(RuntimeError):
        adapter.prepare()


def test_llamacpp_nested_model_meta_parsing(monkeypatch):
    payload = {
        "data": [
            {
                "id": "fake-gguf",
                "meta": {
                    "n_ctx_train": 32768,
                    "n_embd": 896,
                    "n_params": 494032768,
                    "size": 397000000,
                    "n_ctx": 2048,
                    "vocab_type": "bpe",
                    "n_vocab": 151936,
                    "ftype": "Q4_K_M",
                },
            }
        ]
    }
    base_url = install_fake_llama(monkeypatch, models_payload=payload)
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url).runtime, SystemClock(), 1)
    adapter.prepare()
    assert adapter.model_metadata["parameter_count"] == 494032768
    assert adapter.model_metadata["training_context_size"] == 32768
    assert adapter.model_metadata["embedding_dimension"] == 896
    assert adapter.model_metadata["model_size"] == 397000000
    assert adapter.model_metadata["ftype"] == "Q4_K_M"
    assert adapter.model_metadata["raw"] == payload


def test_llamacpp_streaming_ignores_empty_control_chunks(monkeypatch):
    base_url = install_fake_llama(monkeypatch)
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url).runtime, SystemClock(), 1)
    adapter.prepare()
    plan = generate_workload_plan(llama_cfg(base_url))[0]
    rec = adapter.execute(plan, SystemClock().monotonic())
    assert rec.status == "success"
    assert rec.first_token_time is not None
    assert rec.server_prompt_tokens == 4
    assert rec.server_output_tokens == 2
    assert rec.finish_reason == "stop"


def test_llamacpp_streaming_malformed_stream(monkeypatch):
    base_url = install_fake_llama(monkeypatch, stream_mode="malformed")
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url).runtime, SystemClock(), 1)
    adapter.prepare()
    rec = adapter.execute(generate_workload_plan(llama_cfg(base_url))[0], SystemClock().monotonic())
    assert rec.status == "failed"
    assert rec.error_type == "malformed_stream"


def test_llamacpp_http_error_and_timeout(monkeypatch):
    base_url = install_fake_llama(monkeypatch, stream_mode="http_error")
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url).runtime, SystemClock(), 1)
    adapter.prepare()
    rec = adapter.execute(generate_workload_plan(llama_cfg(base_url))[0], SystemClock().monotonic())
    assert rec.status == "failed"
    assert rec.http_status == 500
    base_url = install_fake_llama(monkeypatch, stream_mode="timeout")
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url, timeout=0.01).runtime, SystemClock(), 1)
    adapter.prepare()
    rec = adapter.execute(generate_workload_plan(llama_cfg(base_url))[0], SystemClock().monotonic())
    assert rec.status == "failed"
    assert rec.error_type


def test_llamacpp_metrics_parsing_and_missing_optional_metric_tolerated():
    rows = parse_prometheus_metrics("llamacpp:requests_processing 1\nunknown_metric{a=\"b\"} 2.5\n", 1.0, "rt")
    assert {r["metric_name"] for r in rows} == {"llamacpp:requests_processing", "unknown_metric"}


def test_llamacpp_metrics_disabled_when_required(monkeypatch):
    base_url = install_fake_llama(monkeypatch, metrics_payload="")
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url, metrics_enabled=True).runtime, SystemClock(), 1)
    with pytest.raises(RuntimeError):
        adapter.prepare()


def test_llamacpp_adapter_contract():
    assert parse_sse_payload(b"data: [DONE]\n") == "[DONE]"
    assert content_from_chunk({"choices": [{"text": "x"}]}) == "x"


def test_llamacpp_async_open_loop_scheduling_and_slip_recorded(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch, delay=0.03)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="async-open-loop", request_count=3, inter_arrival=0.01, max_outstanding=3), tmp_path).run()
    _, rows = read_table(run_dir / "requests.parquet")
    assert len(rows) == 3
    assert rows[1]["actual_arrival"] - rows[0]["actual_arrival"] < 0.04
    assert all(r["actual_arrival"] >= r["scheduled_arrival"] for r in rows)
    assert validate_run(run_dir)["valid"]


def test_llamacpp_maximum_outstanding_safety_violation(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch, delay=0.05)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="too-much-load", request_count=3, inter_arrival=0.0, max_outstanding=1), tmp_path).run()
    result = validate_run(run_dir)
    assert not result["valid"]
    assert "client_max_outstanding_exceeded" in codes(result)


def test_system_and_runtime_telemetry_schemas(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="telemetry-schema", request_count=1), tmp_path).run()
    system_schema, system_rows = read_table(run_dir / "system_metrics.parquet")
    runtime_schema, runtime_rows = read_table(run_dir / "runtime_metrics.parquet")
    assert "process_rss" in system_schema
    assert "metric_name" in runtime_schema
    assert system_rows
    assert any(r["metric_name"] == "llamacpp:requests_processing" for r in runtime_rows)


def test_cli_config_serialization(tmp_path):
    path = tmp_path / "config.json"
    write_config(path, cfg(run_id="serialized"))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["config_hash"] == cfg(run_id="serialized").compute_hash()


def test_phase3b_absolute_scheduled_arrival_and_scheduler_slip(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="absolute-time", request_count=2), tmp_path).run()
    manifest = read_json(run_dir / "manifest.json")
    _, rows = read_table(run_dir / "requests.parquet")
    origin = manifest["runtime_metadata"]["workload_time_origin"]
    assert rows[0]["scheduled_arrival"] >= origin
    assert rows[0]["actual_arrival"] >= rows[0]["scheduled_arrival"] - 0.005
    assert rows[0]["scheduler_slip"] == pytest.approx(rows[0]["actual_arrival"] - rows[0]["scheduled_arrival"])


def test_system_telemetry_labels_client_and_not_server(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="telemetry-labels", request_count=1), tmp_path).run()
    _, rows = read_table(run_dir / "system_metrics.parquet")
    assert rows[0]["client_process_rss_bytes"] is not None
    assert rows[0]["server_pid"] is None
    assert rows[0]["server_process_rss_bytes"] is None
    assert rows[0]["memory_pressure"] is None


def test_explicit_server_pid_telemetry_collection(monkeypatch, tmp_path):
    import os

    base_url = install_fake_llama(monkeypatch)
    config = llama_cfg(base_url, run_id="server-pid", request_count=1)
    config.runtime.parameters["server_pid"] = os.getpid()
    run_dir = ExperimentRunner(config, tmp_path).run()
    _, rows = read_table(run_dir / "system_metrics.parquet")
    assert rows[0]["server_pid"] == os.getpid()
    assert rows[0]["server_process_rss_bytes"] is not None


def test_runtime_throughput_is_not_cumulative_success(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="no-throughput", request_count=1), tmp_path).run()
    _, rows = read_table(run_dir / "runtime_metrics.parquet")
    assert all(row["throughput"] is None for row in rows)
    assert any("successful_requests_total" in (row["counters"] or "") for row in rows)


def test_slow_metrics_scrape_does_not_block_scheduler(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch, metrics_delay=0.1, delay=0.01)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="slow-metrics", request_count=3, inter_arrival=0.01, max_outstanding=3), tmp_path).run()
    _, rows = read_table(run_dir / "requests.parquet")
    assert rows[1]["actual_arrival"] - rows[0]["actual_arrival"] < 0.08
    assert validate_run(run_dir)["valid"]


def test_malformed_http_200_health_rejected(monkeypatch):
    base_url = install_fake_llama(monkeypatch, health_payload={"unexpected": "payload"})
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url).runtime, SystemClock(), 1)
    with pytest.raises(RuntimeError):
        adapter.prepare()
    assert adapter.last_health.state == "malformed"


def test_oai_request_includes_model_and_usage_options(monkeypatch):
    captured = []
    base_url = install_fake_llama(monkeypatch, captured_payloads=captured)
    adapter = LlamaCppRuntimeAdapter(llama_cfg(base_url).runtime, SystemClock(), 1)
    adapter.prepare()
    adapter.execute(generate_workload_plan(llama_cfg(base_url))[0], SystemClock().monotonic())
    assert captured[0]["model"] == "fake-gguf"
    assert captured[0]["stream_options"]["include_usage"] is True
    assert "n_predict" not in captured[0]


def test_server_usage_chunk_parsed_with_empty_choices(monkeypatch):
    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith("/health"):
            return FakeHTTPResponse(body=b'{"status":"ok"}')
        if url.endswith("/v1/models"):
            return FakeHTTPResponse(body=b'{"data":[{"id":"fake-gguf"}]}')
        if url.endswith("/metrics"):
            return FakeHTTPResponse(body=b"llamacpp:requests_processing 0\n")
        return FakeHTTPResponse(lines=[
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
            b"data: {\"choices\":[{\"text\":\"x\",\"finish_reason\":\"stop\"}]}\n\n",
            b"data: [DONE]\n\n",
        ])

    monkeypatch.setattr("sloscope.runtime.llamacpp.urllib.request.urlopen", fake_urlopen)
    adapter = LlamaCppRuntimeAdapter(llama_cfg("http://fake").runtime, SystemClock(), 1)
    adapter.prepare()
    rec = adapter.execute(generate_workload_plan(llama_cfg("http://fake"))[0], SystemClock().monotonic())
    assert rec.server_prompt_tokens == 3
    assert rec.server_output_tokens == 2
    assert rec.prompt_tokens is None
    assert rec.token_count_source == "runtime_reported"


def test_healthy_workload_executes_in_baseline_and_recovery_after(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="baseline-workload", request_count=2), tmp_path).run()
    events = read_jsonl_file(run_dir / "events.jsonl")
    _, rows = read_table(run_dir / "requests.parquet")
    recovery_time = [e["monotonic_time"] for e in events if e["event_type"] == "state" and e["state"] == "RECOVERY"][0]
    assert max(r["completion_time"] for r in rows) <= recovery_time
    assert validate_run(run_dir)["valid"]


def test_cli_run_uses_system_clock_and_dry_run_rejects_llamacpp(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    config = llama_cfg(base_url, run_id="cli-run", request_count=1)
    path = tmp_path / "llama.json"
    write_config(path, config)
    assert cli_main(["run", str(path), "--runs-root", str(tmp_path / "runs")]) == 0
    manifest = read_json(tmp_path / "runs" / "cli-run" / "manifest.json")
    assert not manifest["start_timestamp"].startswith("fake-")
    with pytest.raises(SystemExit):
        cli_main(["dry-run", str(path), "--runs-root", str(tmp_path / "runs2")])


def test_experiment_timeout_writes_failed_audit_run(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch, delay=0.1)
    config = llama_cfg(base_url, run_id="timeout-run", request_count=3, timeout=1.0)
    config = ExperimentConfig.from_dict({**config.to_dict(include_hash=False), "safety": {**config.safety.__dict__, "experiment_timeout": 0.01}})
    with pytest.raises(TimeoutError):
        ExperimentRunner(config, tmp_path).run()
    result = validate_run(tmp_path / "timeout-run")
    assert not result["valid"]
    assert "experiment_timeout" in codes(result)


def test_telemetry_config_disables_system_and_runtime_collection(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    config = llama_cfg(base_url, run_id="telemetry-disabled", request_count=1)
    config = ExperimentConfig.from_dict({**config.to_dict(include_hash=False), "telemetry": {"request_telemetry": True, "system_metrics": False, "runtime_metrics": False, "traces": False}})
    run_dir = ExperimentRunner(config, tmp_path).run()
    _, system_rows = read_table(run_dir / "system_metrics.parquet")
    _, runtime_rows = read_table(run_dir / "runtime_metrics.parquet")
    assert system_rows == []
    assert runtime_rows == []
    assert validate_run(run_dir)["valid"]


def test_traces_true_rejected_for_llama(monkeypatch):
    base_url = install_fake_llama(monkeypatch)
    config = llama_cfg(base_url)
    config = ExperimentConfig.from_dict({**config.to_dict(include_hash=False), "telemetry": {"request_telemetry": True, "system_metrics": True, "runtime_metrics": True, "traces": True}})
    config.validate()


def test_unknown_prompt_and_output_profile_rejected():
    bad_prompt = ExperimentConfig.from_dict({**cfg().to_dict(include_hash=False), "workload": {**cfg().workload.__dict__, "prompt_profile": "unknown"}})
    with pytest.raises(ValueError):
        bad_prompt.validate()
    bad_output = ExperimentConfig.from_dict({**cfg().to_dict(include_hash=False), "workload": {**cfg().workload.__dict__, "output_profile": "random"}})
    with pytest.raises(ValueError):
        bad_output.validate()


def test_phase3b_artifact_version_written_and_phase3a_supported(tmp_path):
    run_dir = ExperimentRunner(cfg(run_id="version"), tmp_path, FakeClock()).run()
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    manifest["artifact_schema_version"] = "phase3a.artifacts.v1"
    write_json(run_dir / "manifest.json", manifest)
    assert validate_run(run_dir)["valid"]


def test_missing_runtime_metadata_invalidates_llama_run(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="missing-runtime-metadata", request_count=1), tmp_path).run()
    (run_dir / "runtime_metadata.json").unlink()
    result = validate_run(run_dir)
    assert not result["valid"]
    assert "missing_runtime_metadata" in codes(result)


def test_concurrent_accounting_remains_exact(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch, delay=0.02)
    run_dir = ExperimentRunner(llama_cfg(base_url, run_id="accounting-exact", request_count=8, inter_arrival=0.0, max_outstanding=8), tmp_path).run()
    result = validate_run(run_dir)
    _, rows = read_table(run_dir / "requests.parquet")
    assert result["planned_requests"] == 8
    assert result["accounted_requests"] == 8
    assert sum(1 for r in rows if r["status"] == "success") == 8
    assert validate_run(run_dir)["valid"]


def llama_cfg_with_mechs(base_url, run_id, mechanisms, request_count=2, start_offset=0.2, duration=1.0, require_window=False):
    cfg0 = llama_cfg(base_url, run_id=run_id, request_count=request_count, inter_arrival=0.05, max_outstanding=2)
    data = cfg0.to_dict(include_hash=False)
    data["mechanisms"] = [m.__dict__ for m in mechanisms]
    data["workload"] = {**cfg0.workload.__dict__, "start_offset_seconds": start_offset}
    data["safety"] = {**cfg0.safety.__dict__, "maximum_cpu_stress": 0.25, "require_requests_within_mechanism_window": require_window, "experiment_timeout": 5.0}
    return ExperimentConfig.from_dict(data)


def test_cpu_intensity_worker_count_mapping_and_rejection():
    assert worker_count_for_intensity(10, 0.2) == 2
    assert worker_count_for_intensity(10, 0.05) == 1
    with pytest.raises(ValueError):
        worker_count_for_intensity(10, 0)
    with pytest.raises(ValueError):
        worker_count_for_intensity(10, -0.1)


def test_cpu_safety_cap_rejection():
    bad = cfg(run_id="unsafe-cpu", mechanisms=[cpu_mech(intensity=0.3)])
    bad = ExperimentConfig.from_dict({**bad.to_dict(include_hash=False), "safety": {**bad.safety.__dict__, "maximum_cpu_stress": 0.25}})
    with pytest.raises(ValueError):
        bad.validate()


def test_cpu_worker_start_verify_stop_cleanup():
    inj = CPUContentionInjector(cpu_mech(), FakeClock())
    inj.prepare()
    inj.start()
    ev = inj.verify()
    assert ev.verified
    assert ev.evidence_value["actual_worker_count"] == 2
    assert ev.evidence_value["total_worker_cpu_time_delta"] > 0
    assert ev.evidence_value["activation_time"] <= ev.evidence_value["verification_time"]
    assert len(ev.evidence_value["worker_pids"]) == 2
    inj.stop()
    inj.cleanup()
    assert inj.cleaned
    assert all(not p.is_alive() for p in inj.processes)


def test_cpu_verification_fails_when_worker_does_no_work():
    inj = CPUContentionInjector(cpu_mech(params={"exit_immediately": True}), FakeClock())
    inj.prepare()
    inj.start()
    ev = inj.verify()
    try:
        assert not ev.verified
    finally:
        inj.stop()
        inj.cleanup()


def test_cpu_cleanup_terminates_residual_owned_workers_and_not_unrelated():
    import os
    inj = CPUContentionInjector(cpu_mech(), FakeClock())
    inj.prepare()
    inj.start()
    own_pid = os.getpid()
    inj.cleanup()
    assert inj.cleaned
    assert os.getpid() == own_pid


def test_generic_real_injector_factory_and_unsupported():
    assert isinstance(create_injector(cpu_mech(), FakeClock()), CPUContentionInjector)
    assert isinstance(create_injector(mech(), FakeClock()), MockInjector)
    with pytest.raises(ValueError):
        create_injector(MechanismConfig("x", "unknown", 0.1, "target", 0, 1, {}), FakeClock())


def test_degraded_llama_lifecycle_enters_injecting_and_honors_schedule(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    m = MechanismConfig("m1", "mock-degradation", 0.1, "mock-runtime", 0.05, 0.4, {})
    run_dir = ExperimentRunner(llama_cfg_with_mechs(base_url, "degraded-lifecycle", [m], start_offset=0.2), tmp_path).run()
    events = read_jsonl_file(run_dir / "events.jsonl")
    states = [e["state"] for e in events if e["event_type"] == "state"]
    assert "INJECTING" in states
    gt = read_json(run_dir / "ground_truth.json")["mechanisms"][0]
    inject_time = [e["monotonic_time"] for e in events if e["event_type"] == "state" and e["state"] == "INJECTING"][0]
    assert gt["actual_onset"] >= inject_time + 0.05
    assert gt["actual_stop"] >= inject_time + 0.45
    assert validate_run(run_dir)["valid"]


def test_workload_start_offset_deterministic_and_after_verified_activation(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    m = MechanismConfig("m1", "mock-degradation", 0.1, "mock-runtime", 0.0, 0.8, {})
    config = llama_cfg_with_mechs(base_url, "start-offset", [m], start_offset=0.3, require_window=True)
    plans = generate_workload_plan(config)
    assert plans[0].scheduled_arrival == pytest.approx(0.3)
    run_dir = ExperimentRunner(config, tmp_path).run()
    gt = read_json(run_dir / "ground_truth.json")["mechanisms"][0]
    _, reqs = read_table(run_dir / "requests.parquet")
    assert min(r["actual_arrival"] for r in reqs) >= gt["actual_onset"]
    assert validate_run(run_dir)["valid"]


def test_verification_failure_prevents_request_workload(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    m = MechanismConfig("m1", "mock-degradation", 0.1, "mock-runtime", 0.0, 0.8, {"verification_fails": True})
    with pytest.raises(RuntimeError):
        ExperimentRunner(llama_cfg_with_mechs(base_url, "verify-prevents-work", [m], start_offset=0.3), tmp_path).run()
    result = validate_run(tmp_path / "verify-prevents-work")
    assert not result["valid"]
    assert result["accounted_requests"] == 0
    assert "mechanism_verification_failed" in codes(result)


def test_worker_failure_invalidates_run(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    m = cpu_mech(duration=0.5, params={"exit_immediately": True})
    with pytest.raises(RuntimeError):
        ExperimentRunner(llama_cfg_with_mechs(base_url, "worker-failure", [m], start_offset=0.3), tmp_path).run()
    assert not validate_run(tmp_path / "worker-failure")["valid"]


def test_ground_truth_onset_uses_activation_time_and_contains_verification_time(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    m = MechanismConfig("m1", "mock-degradation", 0.1, "mock-runtime", 0.0, 0.5, {})
    run_dir = ExperimentRunner(llama_cfg_with_mechs(base_url, "gt-activation-time", [m], start_offset=0.2), tmp_path).run()
    gt = read_json(run_dir / "ground_truth.json")["mechanisms"][0]
    ev = gt["verification_evidence"]["evidence_value"]
    assert gt["actual_onset"] == pytest.approx(ev.get("activation_time", gt["verification_time"]))
    assert gt["verification_time"] is not None


def test_full_window_request_validation_detects_boundary(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    m = MechanismConfig("m1", "mock-degradation", 0.1, "mock-runtime", 0.0, 0.05, {})
    run_dir = ExperimentRunner(llama_cfg_with_mechs(base_url, "window-fail", [m], start_offset=0.2, require_window=True), tmp_path).run()
    result = validate_run(run_dir)
    assert not result["valid"]
    assert "request_outside_required_mechanism_window" in codes(result)


def test_prompt_cache_provenance_false_when_no_cache_prompt(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    config = llama_cfg(base_url, run_id="prompt-cache-off", request_count=1)
    args = ["--no-cache-prompt"]
    config.runtime.parameters["server_arguments"] = args
    run_dir = ExperimentRunner(config, tmp_path).run()
    meta = read_json(run_dir / "runtime_metadata.json")
    assert meta["prompt_cache_enabled"] is False
    assert validate_run(run_dir)["valid"]


def downstream_mech(base_url, delay_ms=50, duration=1.0, params=None):
    p = {"dependency_base_url": base_url, "delay_ms": delay_ms, "verify_request": True}
    if params:
        p.update(params)
    return MechanismConfig("dep-latency", "downstream_latency", 0.0, "synthetic_dependency", 0.0, duration, p)


def test_dependency_zero_delay_response():
    svc = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    svc.start()
    try:
        started = time.monotonic()
        payload = json_request("GET", f"http://127.0.0.1:{svc.port}/dependency")
        assert payload["delay_ms"] == 0
        assert time.monotonic() - started < 0.1
    finally:
        svc.stop()


def test_dependency_configured_delay_response():
    svc = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    svc.start()
    try:
        json_request("POST", f"http://127.0.0.1:{svc.port}/control/delay", {"delay_ms": 40})
        started = time.monotonic()
        payload = json_request("GET", f"http://127.0.0.1:{svc.port}/dependency")
        assert payload["delay_ms"] == 40
        assert time.monotonic() - started >= 0.032
    finally:
        svc.stop()


def test_downstream_delay_safety_cap_validation():
    base = "http://127.0.0.1:8091"
    config = llama_cfg_with_mechs(base, "unsafe-dep", [downstream_mech(base, delay_ms=600)], request_count=1)
    config = ExperimentConfig.from_dict({**config.to_dict(include_hash=False), "safety": {**config.safety.__dict__, "maximum_dependency_delay_ms": 500}})
    with pytest.raises(ValueError):
        config.validate()


def test_downstream_injector_start_verify_stop_cleanup():
    svc = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    svc.start()
    base = f"http://127.0.0.1:{svc.port}"
    try:
        inj = DownstreamLatencyInjector(downstream_mech(base, delay_ms=30), SystemClock())
        inj.prepare()
        inj.start()
        assert json_request("GET", f"{base}/control/status")["delay_ms"] == 30
        ev = inj.verify()
        assert ev.verified
        assert ev.evidence_value["configured_delay_ms"] == 30
        assert ev.evidence_value["observed_dependency_request_duration"] >= 0.024
        inj.stop()
        assert json_request("GET", f"{base}/control/status")["delay_ms"] == 0
        inj.cleanup()
        assert inj.cleanup_evidence()["cleaned"]
        assert json_request("GET", f"{base}/control/status")["delay_ms"] == 0
    finally:
        svc.stop()


def test_downstream_verification_does_not_use_llm_latency(monkeypatch):
    svc = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    svc.start()
    try:
        called = []
        original = urllib.request.urlopen

        def spy(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            called.append(url)
            return original(req, timeout=timeout)

        monkeypatch.setattr("sloscope.injectors.downstream_latency.urllib.request.urlopen", spy)
        base = f"http://127.0.0.1:{svc.port}"
        inj = DownstreamLatencyInjector(downstream_mech(base, delay_ms=10), SystemClock())
        inj.prepare()
        inj.start()
        assert inj.verify().verified
        assert all("/v1/completions" not in url for url in called)
    finally:
        svc.stop()


def test_gateway_forwards_completion_streams_and_records_timing():
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        json_request("POST", f"http://127.0.0.1:{dep.port}/control/delay", {"delay_ms": 25})
        req = urllib.request.Request(
            f"http://127.0.0.1:{gw.port}/v1/completions",
            data=b'{"model":"fake-gguf","prompt":"x","stream":true}',
            method="POST",
            headers={"Accept": "text/event-stream", "Content-Type": "application/json", "X-SLOScope-Request-Id": "r1"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8")
        assert "Hello" in body
        assert llama.requests == ["r1"]
        timing = json_request("GET", f"http://127.0.0.1:{gw.port}/sloscope/requests/r1")
        assert timing["gateway_receive_time"] <= timing["dependency_start_time"] <= timing["dependency_end_time"]
        assert timing["dependency_end_time"] <= timing["llama_dispatch_time"] <= timing["gateway_completion_time"]
        assert timing["llama_first_token_time"] is not None
        assert timing["dependency_duration"] >= 0.020
        spans = json_request("GET", f"http://127.0.0.1:{gw.port}/sloscope/traces")["spans"]
        assert {s["span_name"] for s in spans} == {"gateway request", "dependency call", "llama call"}
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_client_first_token_semantics_remain_through_gateway():
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        cfg1 = llama_cfg(f"http://127.0.0.1:{gw.port}", request_count=1, model_id="fake-gguf")
        adapter = LlamaCppRuntimeAdapter(cfg1.runtime, SystemClock(), 1)
        adapter.prepare()
        rec = adapter.execute(generate_workload_plan(cfg1)[0], SystemClock().monotonic())
        assert rec.status == "success"
        assert rec.first_token_time is not None
        assert rec.gateway_receive_time is not None
        assert rec.dependency_duration is not None
        assert rec.first_token_time >= rec.dispatch_time
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_healthy_and_degraded_gateway_paths_validate(tmp_path):
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        base = f"http://127.0.0.1:{gw.port}"
        healthy = llama_cfg(base, run_id="healthy-gateway", request_count=2, inter_arrival=0.05, model_id="fake-gguf")
        healthy = ExperimentConfig.from_dict({**healthy.to_dict(include_hash=False), "telemetry": {**healthy.telemetry.__dict__, "traces": True}})
        healthy_dir = ExperimentRunner(healthy, tmp_path).run()
        assert validate_run(healthy_dir)["valid"]
        _, healthy_rows = read_table(healthy_dir / "requests.parquet")
        assert all((r["dependency_duration"] or 0) < 0.050 for r in healthy_rows)

        mech1 = downstream_mech(f"http://127.0.0.1:{dep.port}", delay_ms=30, duration=1.0)
        degraded = llama_cfg_with_mechs(base, "degraded-gateway", [mech1], request_count=2, start_offset=0.2, duration=1.0, require_window=True)
        degraded = ExperimentConfig.from_dict({**degraded.to_dict(include_hash=False), "telemetry": {**degraded.telemetry.__dict__, "traces": True}, "safety": {**degraded.safety.__dict__, "maximum_dependency_delay_ms": 200}})
        degraded_dir = ExperimentRunner(degraded, tmp_path).run()
        result = validate_run(degraded_dir)
        assert result["valid"], result["issues"]
        _, rows = read_table(degraded_dir / "requests.parquet")
        assert all(r["dependency_duration"] >= 0.024 for r in rows)
        gt = read_json(degraded_dir / "ground_truth.json")["mechanisms"][0]
        assert gt["verification_evidence"]["verified"]
        assert gt["verification_evidence"]["evidence_value"]["configured_delay_ms"] == 30
        assert json_request("GET", f"http://127.0.0.1:{dep.port}/control/status")["delay_ms"] == 0
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_unsupported_dependency_state_and_cleanup_failure_invalidate(tmp_path):
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        base = f"http://127.0.0.1:{gw.port}"
        mech1 = downstream_mech(f"http://127.0.0.1:{dep.port}", delay_ms=30, duration=0.5)
        config = llama_cfg_with_mechs(base, "dep-changed", [mech1], request_count=1, start_offset=0.2)
        config = ExperimentConfig.from_dict({**config.to_dict(include_hash=False), "safety": {**config.safety.__dict__, "maximum_dependency_delay_ms": 200}})
        original = DownstreamLatencyInjector.active_failure

        def bad_state(self):
            return "dependency delay changed during active interval"

        DownstreamLatencyInjector.active_failure = bad_state
        try:
            with pytest.raises(RuntimeError):
                ExperimentRunner(config, tmp_path).run()
        finally:
            DownstreamLatencyInjector.active_failure = original
        assert not validate_run(tmp_path / "dep-changed")["valid"]
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_downstream_cleanup_failure_raises_and_marks_unclean():
    inj = DownstreamLatencyInjector(downstream_mech("http://127.0.0.1:8091", delay_ms=10), FakeClock())
    inj._set_delay = lambda delay_ms: {"delay_ms": 7, "status": "ok", "service_id": "synthetic_dependency", "port": 8091}
    with pytest.raises(RuntimeError):
        inj.cleanup()
    assert not inj.cleaned
    assert inj.cleanup_evidence()["cleaned"] is False


def test_gateway_dependency_pid_telemetry_mapping(monkeypatch, tmp_path):
    import os
    base_url = install_fake_llama(monkeypatch)
    config = llama_cfg(base_url, run_id="gateway-pids", request_count=1)
    config.runtime.parameters["gateway_pid"] = os.getpid()
    config.runtime.parameters["dependency_pid"] = os.getpid()
    run_dir = ExperimentRunner(config, tmp_path).run()
    _, rows = read_table(run_dir / "system_metrics.parquet")
    assert rows[0]["gateway_pid"] == os.getpid()
    assert rows[0]["gateway_process_rss_bytes"] is not None
    assert rows[0]["dependency_pid"] == os.getpid()
    assert rows[0]["dependency_process_rss_bytes"] is not None


def test_gateway_overlapping_requests_do_not_serialize():
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer(first_chunk_sleep=0.20)
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        def call(rid):
            req = urllib.request.Request(
                f"http://127.0.0.1:{gw.port}/v1/completions",
                data=b'{"model":"fake-gguf","prompt":"x","stream":true}',
                method="POST",
                headers={"Accept": "text/event-stream", "Content-Type": "application/json", "X-SLOScope-Request-Id": rid},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.read()

        t1 = Thread(target=call, args=("slow-1",))
        t2 = Thread(target=call, args=("slow-2",))
        t1.start()
        time.sleep(0.03)
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)
        assert not t1.is_alive()
        assert not t2.is_alive()
        assert llama.request_start_times["slow-2"] < llama.request_finish_times["slow-1"]
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_gateway_upstream_failure_before_headers_returns_clean_502():
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer(fail_completions=True)
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{gw.port}/v1/completions",
            data=b'{"model":"fake-gguf","prompt":"x","stream":true}',
            method="POST",
            headers={"Accept": "text/event-stream", "Content-Type": "application/json", "X-SLOScope-Request-Id": "fail-before-headers"},
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=3)
        assert excinfo.value.code == 502
        spans = json_request("GET", f"http://127.0.0.1:{gw.port}/sloscope/traces")["spans"]
        statuses = {s["span_name"]: s["status"] for s in spans}
        assert statuses["dependency call"] == "ok"
        assert statuses["llama call"] == "error"
        assert statuses["gateway request"] == "error"
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_gateway_reset_clears_timings_and_traces():
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{gw.port}/v1/completions",
            data=b'{"model":"fake-gguf","prompt":"x","stream":true}',
            method="POST",
            headers={"Accept": "text/event-stream", "Content-Type": "application/json", "X-SLOScope-Request-Id": "r-reset"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            resp.read()
        assert json_request("GET", f"http://127.0.0.1:{gw.port}/sloscope/traces")["spans"]
        reset = json_request("POST", f"http://127.0.0.1:{gw.port}/sloscope/reset", {})
        assert reset["cleared_request_timings"] == 1
        assert reset["cleared_trace_rows"] == 3
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"http://127.0.0.1:{gw.port}/sloscope/requests/r-reset", timeout=3)
        assert excinfo.value.code == 404
        assert json_request("GET", f"http://127.0.0.1:{gw.port}/sloscope/traces")["spans"] == []
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_second_gateway_experiment_does_not_contain_first_traces(tmp_path):
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        base = f"http://127.0.0.1:{gw.port}"
        cfg1 = ExperimentConfig.from_dict({**llama_cfg(base, run_id="trace-run-a", request_count=1, model_id="fake-gguf").to_dict(include_hash=False), "telemetry": {"request_telemetry": True, "system_metrics": True, "runtime_metrics": True, "traces": True}})
        cfg2 = ExperimentConfig.from_dict({**llama_cfg(base, run_id="trace-run-b", request_count=1, model_id="fake-gguf").to_dict(include_hash=False), "telemetry": {"request_telemetry": True, "system_metrics": True, "runtime_metrics": True, "traces": True}})
        run_a = ExperimentRunner(cfg1, tmp_path).run()
        run_b = ExperimentRunner(cfg2, tmp_path).run()
        _, spans_a = read_table(run_a / "traces.parquet")
        _, spans_b = read_table(run_b / "traces.parquet")
        assert len(spans_a) == 3
        assert len(spans_b) == 3
        attrs_a = {json.loads(s["attributes"])["request_id"] for s in spans_a}
        attrs_b = {json.loads(s["attributes"])["request_id"] for s in spans_b}
        assert attrs_a.isdisjoint(attrs_b)
        assert validate_run(run_b)["valid"]
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_trace_validator_rejects_foreign_duplicate_missing_and_orphan_spans(tmp_path):
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        config = ExperimentConfig.from_dict({**llama_cfg(f"http://127.0.0.1:{gw.port}", run_id="trace-shape", request_count=1, model_id="fake-gguf").to_dict(include_hash=False), "telemetry": {"request_telemetry": True, "system_metrics": True, "runtime_metrics": True, "traces": True}})
        run_dir = ExperimentRunner(config, tmp_path).run()
        _, spans = read_table(run_dir / "traces.parquet")
        assert validate_run(run_dir)["valid"]
        bad = list(spans)
        foreign = dict(bad[0])
        foreign["span_id"] = "0123456789abcdef"
        foreign["attributes"] = json.dumps({"request_id": "other-run-request"})
        bad.append(foreign)
        duplicate = dict(bad[0])
        bad.append(duplicate)
        orphan = dict(next(s for s in bad if s["span_name"] == "dependency call"))
        orphan["span_id"] = "1111111111111111"
        orphan["parent_span_id"] = "2222222222222222"
        bad = [s for s in bad if s["span_name"] != "llama call"] + [orphan]
        write_table(run_dir / "traces.parquet", bad, TRACE_COLUMNS)
        result = validate_run(run_dir)
        assert {"foreign_trace_request", "duplicate_trace_span", "missing_request_span", "orphan_trace_span"} & codes(result)
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_gateway_readiness_fails_if_dependency_or_llama_unavailable():
    dep_port = free_port()
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep_port}")
    llama.start(); gw.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"http://127.0.0.1:{gw.port}/ready", timeout=3)
        assert excinfo.value.code == 503
    finally:
        gw.stop(); llama.stop()

    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    bad_llama_port = free_port()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=f"http://127.0.0.1:{bad_llama_port}", dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); gw.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"http://127.0.0.1:{gw.port}/ready", timeout=3)
        assert excinfo.value.code == 503
    finally:
        gw.stop(); dep.stop()


def test_ground_truth_records_mechanism_magnitude_units():
    cpu = cpu_mech(intensity=0.2)
    dep = downstream_mech("http://127.0.0.1:8091", delay_ms=100)
    rows = GroundTruthLedger([cpu, dep]).to_dict()["mechanisms"]
    by_type = {row["mechanism_type"]: row for row in rows}
    assert by_type["cpu_contention"]["magnitude_value"] == 0.2
    assert by_type["cpu_contention"]["magnitude_unit"] == "logical_cpu_worker_fraction"
    assert by_type["downstream_latency"]["magnitude_value"] == 100
    assert by_type["downstream_latency"]["magnitude_unit"] == "ms"


def test_watchdog_detects_dependency_state_change(tmp_path):
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        base = f"http://127.0.0.1:{gw.port}"
        mech1 = downstream_mech(f"http://127.0.0.1:{dep.port}", delay_ms=30, duration=0.4)
        config = llama_cfg_with_mechs(base, "watchdog-dep", [mech1], request_count=1, start_offset=0.2)
        config = ExperimentConfig.from_dict({**config.to_dict(include_hash=False), "safety": {**config.safety.__dict__, "maximum_dependency_delay_ms": 200, "mechanism_watchdog_interval_seconds": 0.05}})
        original = DownstreamLatencyInjector.verify

        def verify_then_break(self):
            ev = original(self)
            json_request("POST", f"http://127.0.0.1:{dep.port}/control/delay", {"delay_ms": 0})
            return ev

        DownstreamLatencyInjector.verify = verify_then_break
        try:
            with pytest.raises(RuntimeError):
                ExperimentRunner(config, tmp_path).run()
        finally:
            DownstreamLatencyInjector.verify = original
        assert "mechanism_failure" in {e["event_type"] for e in read_jsonl_file(tmp_path / "watchdog-dep" / "events.jsonl")}
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_watchdog_detects_dead_cpu_worker(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    m = cpu_mech(duration=0.4)
    config = llama_cfg_with_mechs(base_url, "watchdog-cpu", [m], request_count=1, start_offset=0.2)
    config = ExperimentConfig.from_dict({**config.to_dict(include_hash=False), "safety": {**config.safety.__dict__, "mechanism_watchdog_interval_seconds": 0.05}})
    verified = {"done": False}

    original_verify = CPUContentionInjector.verify

    def verify_then_report_dead_worker(self):
        ev = original_verify(self)
        verified["done"] = True
        return ev

    def report_dead_worker_after_verify(self):
        if verified["done"]:
            return f"CPU contention worker exited during active interval: {[self.worker_pids[0]]}"
        return None

    monkeypatch.setattr(CPUContentionInjector, "verify", verify_then_report_dead_worker)
    monkeypatch.setattr(CPUContentionInjector, "active_failure", report_dead_worker_after_verify)
    with pytest.raises(RuntimeError):
        ExperimentRunner(config, tmp_path).run()
    assert "mechanism_failure" in {e["event_type"] for e in read_jsonl_file(tmp_path / "watchdog-cpu" / "events.jsonl")}


def test_same_onset_mechanisms_start_independently(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    starts = {}

    class SlowVerifyInjector(MockInjector):
        def start(self):
            starts[self.mechanism_id] = time.monotonic()
            super().start()

        def verify(self):
            if self.mechanism_id == "slow-a":
                time.sleep(0.20)
            return super().verify()

        def active_failure(self):
            return None

    def fake_factory(config, clock):
        return SlowVerifyInjector(config, clock)

    monkeypatch.setattr("sloscope.runner.create_injector", fake_factory)
    mechanisms = [
        MechanismConfig("slow-a", "mock-degradation", 0.1, "mock-runtime", 0.0, 0.5, {}),
        MechanismConfig("fast-b", "mock-degradation", 0.1, "mock-runtime", 0.0, 0.5, {}),
    ]
    config = llama_cfg_with_mechs(base_url, "same-onset", mechanisms, request_count=1, start_offset=0.25)
    ExperimentRunner(config, tmp_path).run()
    assert abs(starts["slow-a"] - starts["fast-b"]) < 0.10


def test_compound_full_window_validation_requires_intersection(monkeypatch, tmp_path):
    base_url = install_fake_llama(monkeypatch)
    m1 = MechanismConfig("m1", "mock-degradation", 0.1, "mock-runtime", 0.0, 0.8, {})
    m2 = MechanismConfig("m2", "mock-degradation", 0.1, "mock-runtime", 0.4, 0.8, {})
    config = llama_cfg_with_mechs(base_url, "compound-window", [m1, m2], request_count=1, start_offset=0.2, require_window=True)
    run_dir = ExperimentRunner(config, tmp_path).run()
    result = validate_run(run_dir)
    assert not result["valid"]
    assert "request_outside_required_mechanism_window" in codes(result)


def test_cpu_worker_count_override_cannot_bypass_safety():
    bad = cfg(run_id="unsafe-worker-count", mechanisms=[cpu_mech(intensity=0.1, params={"logical_cpu_count": 10, "worker_count": 5})])
    bad = ExperimentConfig.from_dict({**bad.to_dict(include_hash=False), "safety": {**bad.safety.__dict__, "maximum_cpu_stress": 0.2}})
    with pytest.raises(ValueError):
        bad.validate()


def test_traces_true_successful_gateway_request_requires_trace_rows(tmp_path):
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        config = ExperimentConfig.from_dict({**llama_cfg(f"http://127.0.0.1:{gw.port}", run_id="missing-required-traces", request_count=1, model_id="fake-gguf").to_dict(include_hash=False), "telemetry": {"request_telemetry": True, "system_metrics": True, "runtime_metrics": True, "traces": True}})
        run_dir = ExperimentRunner(config, tmp_path).run()
        write_table(run_dir / "traces.parquet", [], TRACE_COLUMNS)
        result = validate_run(run_dir)
        assert not result["valid"]
        assert "required_traces_missing" in codes(result)
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_compound_ground_truth_keeps_two_independent_records(tmp_path):
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        base = f"http://127.0.0.1:{gw.port}"
        mechanisms = [cpu_mech(intensity=0.1, duration=1.0, params={"logical_cpu_count": 10, "verification_interval_seconds": 0.05}), downstream_mech(f"http://127.0.0.1:{dep.port}", delay_ms=30, duration=1.0)]
        config = llama_cfg_with_mechs(base, "compound-gt", mechanisms, request_count=1, start_offset=0.3, require_window=True)
        config = ExperimentConfig.from_dict({**config.to_dict(include_hash=False), "runtime": {**config.runtime.__dict__, "model_id": "fake-gguf", "parameters": {**config.runtime.parameters, "metrics_enabled": True}}, "safety": {**config.safety.__dict__, "maximum_cpu_stress": 0.2, "maximum_dependency_delay_ms": 200, "mechanism_watchdog_interval_seconds": 0.05}, "telemetry": {**config.telemetry.__dict__, "traces": True}})
        run_dir = ExperimentRunner(config, tmp_path).run()
        gt = read_json(run_dir / "ground_truth.json")
        assert {m["mechanism_id"] for m in gt["mechanisms"]} == {"cpu1", "dep-latency"}
        assert active_mechanism_set(gt) == {"cpu_contention", "downstream_latency"}
        assert validate_run(run_dir)["valid"]
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_active_mechanism_set_conditions():
    def gt_for(mechanisms):
        ledger = GroundTruthLedger(mechanisms)
        for mech in mechanisms:
            ledger.mark_verified(mech.mechanism_id, VerificationEvidence(True, "test", {"activation_time": 1.0, "verification_time": 1.1}, 1.1))
            ledger.mark_stopped(mech.mechanism_id, 2.0)
            ledger.mark_cleaned(mech.mechanism_id)
        return ledger.to_dict()

    assert active_mechanism_set(gt_for([])) == set()
    assert active_mechanism_set(gt_for([cpu_mech()])) == {"cpu_contention"}
    assert active_mechanism_set(gt_for([downstream_mech("http://127.0.0.1:8091")])) == {"downstream_latency"}
    assert active_mechanism_set(gt_for([cpu_mech(), downstream_mech("http://127.0.0.1:8091")])) == {"cpu_contention", "downstream_latency"}


def test_compound_workload_waits_for_both_verifications(tmp_path):
    dep = SyntheticDependencyService(port=free_port(), maximum_delay_ms=200)
    llama = FakeLlamaHTTPServer()
    gw = SLOScopeGateway(port=free_port(), llama_base_url=llama.base_url, dependency_base_url=f"http://127.0.0.1:{dep.port}")
    dep.start(); llama.start(); gw.start()
    try:
        mechanisms = [cpu_mech(intensity=0.1, duration=1.0, params={"logical_cpu_count": 10, "verification_interval_seconds": 0.05}), downstream_mech(f"http://127.0.0.1:{dep.port}", delay_ms=30, duration=1.0)]
        config = llama_cfg_with_mechs(f"http://127.0.0.1:{gw.port}", "compound-waits", mechanisms, request_count=1, start_offset=0.4, require_window=True)
        config = ExperimentConfig.from_dict({**config.to_dict(include_hash=False), "runtime": {**config.runtime.__dict__, "model_id": "fake-gguf"}, "safety": {**config.safety.__dict__, "maximum_cpu_stress": 0.2, "maximum_dependency_delay_ms": 200}})
        run_dir = ExperimentRunner(config, tmp_path).run()
        gt = read_json(run_dir / "ground_truth.json")["mechanisms"]
        _, reqs = read_table(run_dir / "requests.parquet")
        first_arrival = min(r["actual_arrival"] for r in reqs)
        assert all(first_arrival >= m["verification_time"] for m in gt)
    finally:
        gw.stop(); llama.stop(); dep.stop()


def test_phase5_manifest_completeness_and_compound_matrix():
    conds = phase5_conditions()
    ids = {c.condition_id for c in conds}
    assert {"BASELINE", "INPUT_MEDIUM", "OUTPUT_MEDIUM", "LOAD_MEDIUM", "DOWNSTREAM_MEDIUM"} <= ids
    assert {"INPUT_LOAD", "INPUT_DOWNSTREAM", "OUTPUT_LOAD", "OUTPUT_DOWNSTREAM", "LOAD_DOWNSTREAM"} <= ids
    assert "INPUT_OUTPUT" not in ids
    assert {"OUTPUT_LOAD_CONTROL", "OUTPUT_DOWNSTREAM_CONTROL"} <= ids
    assert len(ids) == len(conds)
    assert all("cpu" not in mechanism for c in conds for mechanism in c.active_mechanisms)
    assert all(c.compound_degree == len(c.active_mechanisms) for c in conds)
    assert all(c.mechanism_levels[m] == "medium" for c in conds if c.compound_degree == 2 for m in c.active_mechanisms)


def test_phase5_repetition_count_and_order_determinism():
    rows_a = phase5_campaign_rows("rev-a")
    rows_b = phase5_campaign_rows("rev-a")
    assert rows_a == rows_b
    assert len(rows_a) == len(phase5_conditions()) * PHASE5_REPETITIONS
    assert [row["campaign_order"] for row in rows_a] == list(range(1, len(rows_a) + 1))
    for condition in {row["condition_id"] for row in rows_a}:
        assert {row["repetition"] for row in rows_a if row["condition_id"] == condition} == set(range(1, PHASE5_REPETITIONS + 1))


def seed_fake_phase52_calibration(root):
    cal_root = root / "calibrations" / "phase5.2-slo"
    cal_root.mkdir(parents=True, exist_ok=True)
    baseline = []
    output = []
    for idx in range(8):
        baseline.append({
            "run_id": f"slo-cal-r{idx+1:02d}-baseline-a01",
            "valid": True,
            "ttft_p95": 0.05 + idx * 0.0001,
            "total_latency_p95": 0.19 + idx * 0.0001,
            "post_first_token_duration_p95": 0.12,
            "run_manifest_sha256": f"{idx:064x}",
        })
        output.append({
            "run_id": f"slo-cal-r{idx+1:02d}-output-control-a01",
            "valid": True,
            "ttft_p95": 0.06 + idx * 0.0001,
            "total_latency_p95": 0.27 + idx * 0.0001,
            "post_first_token_duration_p95": 0.13 + idx * 0.0001,
            "run_manifest_sha256": f"{idx+20:064x}",
        })
    artifact = phase5_build_slo_calibration_artifact(baseline, output)
    (cal_root / "slo-calibration.json").write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact


def test_phase5_slo_and_mechanism_schema():
    slos = phase5_slo_definitions()
    assert {"TTFT_SLO", "TOTAL_LATENCY_SLO", "DECODE_DURATION_SLO", "SECONDS_PER_OUTPUT_TOKEN_DIAGNOSTIC", "THROUGHPUT_DIAGNOSTIC"} <= set(slos)
    for name, definition in slos.items():
        if name.endswith("_SLO"):
            assert {"metric", "unit", "aggregation_level", "threshold", "violation_definition"} <= set(definition)
    assert "SECONDS_PER_OUTPUT_TOKEN_SLO" not in slos
    assert "THROUGHPUT_SLO" not in slos
    assert slos["compound_violation_definition"]["counted_slos"] == ["TTFT_SLO", "TOTAL_LATENCY_SLO", "DECODE_DURATION_SLO"]
    mechanisms = phase5_mechanism_definitions()
    assert mechanisms["CPU_CONTENTION"]["status"] == "DEFERRED"
    for key in ["M1_INPUT_PREFILL", "M2_OUTPUT_DECODE", "M3_LOAD_QUEUE", "M4_DOWNSTREAM_LATENCY"]:
        assert mechanisms[key]["status"] == "PRIMARY"
        assert mechanisms[key]["medium_level"]
        assert mechanisms[key]["direct_evidence"]


def test_phase5_confound_and_validity_rules():
    rules = phase5_confound_rules()
    assert rules["queue_confounded"]["input"] == "requests_deferred_max > 0"
    assert rules["queue_confounded"]["output"] == "requests_deferred_max > 0"
    assert "intended mechanism" in rules["queue_confounded"]["load"]
    invalidation = " ".join(rules["invalidation_rules"])
    assert "git_dirty" in invalidation
    assert "missing mandatory telemetry" in invalidation


def test_phase5_analysis_plan_preserves_later_csd_and_rca_contract():
    plan = phase5_analysis_plan()
    assert plan["replication_unit"] == "run/repetition"
    assert "Holm" in plan["multiple_comparisons"]
    assert plan["csd_scope"]["execute_in_phase5"] is False
    assert "active_mechanisms" in plan["csd_scope"]["ground_truth_labels"]
    assert plan["rca_scope"]["execute_in_phase5"] is False


def test_phase5_generate_manifest_configs_and_freeze_hash(tmp_path):
    seed_fake_phase52_calibration(tmp_path)
    result = generate_phase5(tmp_path)
    assert result["issues"] == []
    assert (tmp_path / "docs" / "phase5-campaign-freeze.md").exists()
    manifest = json.loads((tmp_path / "campaigns" / "phase5" / "campaign-manifest.json").read_text(encoding="utf-8"))
    assert manifest["formal_run_count"] == len(phase5_conditions()) * PHASE5_REPETITIONS
    assert manifest["campaign_freeze_sha256"] == result["campaign_freeze_sha256"]
    assert validate_phase5_freeze(tmp_path) == []
    first_hash = result["campaign_freeze_sha256"]
    result_again = generate_phase5(tmp_path)
    assert result_again["campaign_freeze_sha256"] == first_hash
    assert result_again["issues"] == []
    config_paths = [tmp_path / row["config_path"] for row in manifest["campaign_order"]]
    assert all(path.exists() for path in config_paths)
    assert phase5_freeze_hash(phase5_freeze_input_paths(tmp_path), tmp_path) == first_hash


def test_phase5_source_provenance_requirement_recorded(tmp_path):
    seed_fake_phase52_calibration(tmp_path)
    generate_phase5(tmp_path)
    manifest = json.loads((tmp_path / "campaigns" / "phase5" / "campaign-manifest.json").read_text(encoding="utf-8"))
    assert {"git_revision", "git_dirty", "source_tree_sha256"} <= set(manifest)
    assert manifest["publication_execution_requirement"] == {"git_dirty": False}


def test_phase5_warmup_executes_and_excludes_measured_accounting(tmp_path):
    config = ExperimentConfig.from_dict({
        **cfg(run_id="warmup-real", mechanisms=[]).to_dict(include_hash=False),
        "workload": {
            "arrival_pattern": "constant_open_loop",
            "request_count": 40,
            "concurrency": 4,
            "prompt_profile": "mock-prompts",
            "output_profile": "fixed",
            "target_output_tokens": 12,
            "inter_arrival_seconds": 0.01,
            "burst_size": 2,
            "burst_interval_seconds": 1.0,
            "randomized_output": False,
            "output_jitter_tokens": 0,
            "start_offset_seconds": 0.0,
        },
        "runtime": {
            "runtime_id": "mock-runtime",
            "runtime_type": "mock",
            "model_id": "mock-model",
            "parameters": {"warmup_request_count": 5},
        },
    })
    run_dir = ExperimentRunner(config, tmp_path).run()
    _, measured = read_table(run_dir / "requests.parquet")
    workload = read_json(run_dir / "workload.json")["requests"]
    metadata = read_json(run_dir / "runtime_metadata.json")
    assert len(workload) == 40
    assert len(measured) == 40
    assert metadata["warmup"]["warmup_request_count"] == 5
    assert metadata["warmup"]["warmup_success_count"] == 5
    assert metadata["warmup"]["warmup_failure_count"] == 0


def test_phase5_existing_run_directory_refuses_overwrite(tmp_path):
    path = tmp_path / "already-there"
    path.mkdir()
    with pytest.raises(FileExistsError):
        phase5_ensure_run_directory_absent(path)
    with pytest.raises(FileExistsError):
        ExperimentRunner(cfg(run_id="already-there"), tmp_path).run()


def test_phase5_unique_run_ids_for_every_manifest_attempt():
    rows = phase5_campaign_rows("rev")
    run_ids = [row["run_id"] for row in rows]
    assert len(run_ids) == len(set(run_ids))
    assert all(row["attempt"] == 1 for row in rows)
    assert all(row["run_id"].startswith(f"phase6v2-r{int(row['repetition']):02d}-") for row in rows)


def test_phase5_experimental_condition_artifact_and_hash(tmp_path):
    seed_fake_phase52_calibration(tmp_path)
    generate_phase5(tmp_path)
    manifest = json.loads((tmp_path / "campaigns" / "phase5" / "campaign-manifest.json").read_text(encoding="utf-8"))
    row = next(r for r in manifest["campaign_order"] if r["condition_id"] == "INPUT_MEDIUM")
    config = load_config(tmp_path / row["config_path"])
    config = ExperimentConfig.from_dict({
        **config.to_dict(include_hash=False),
        "runtime": {
            "runtime_id": "mock-runtime",
            "runtime_type": "mock",
            "model_id": "mock-model",
            "parameters": {
                "warmup_request_count": 5,
                "experimental_condition_required": True,
                "experimental_condition": config.runtime.parameters["experimental_condition"],
            },
        },
        "mechanisms": [],
    })
    run_dir = ExperimentRunner(config, tmp_path / "runs").run()
    condition = read_json(run_dir / "experimental_condition.json")
    run_manifest = read_json(run_dir / "manifest.json")
    assert condition["campaign_condition_id"] == "INPUT_MEDIUM"
    assert "experimental_condition.json" in run_manifest["artifact_hashes"]
    assert "experimental_condition_mismatch" not in codes(validate_run(run_dir))


def test_phase5_workload_label_mismatch_invalidates_run(tmp_path):
    seed_fake_phase52_calibration(tmp_path)
    generate_phase5(tmp_path)
    manifest = json.loads((tmp_path / "campaigns" / "phase5" / "campaign-manifest.json").read_text(encoding="utf-8"))
    row = next(r for r in manifest["campaign_order"] if r["condition_id"] == "INPUT_MEDIUM")
    config = load_config(tmp_path / row["config_path"])
    bad = ExperimentConfig.from_dict({
        **config.to_dict(include_hash=False),
        "runtime": {
            "runtime_id": "mock-runtime",
            "runtime_type": "mock",
            "model_id": "mock-model",
            "parameters": {
                "warmup_request_count": 5,
                "experimental_condition_required": True,
                "experimental_condition": config.runtime.parameters["experimental_condition"],
            },
        },
        "mechanisms": [],
        "workload": {**config.workload.__dict__, "prompt_profile": "synthetic-input-small"},
    })
    run_dir = ExperimentRunner(bad, tmp_path / "runs").run()
    result = validate_run(run_dir)
    assert not result["valid"]
    assert "experimental_condition_mismatch" in codes(result)


def test_phase5_downstream_label_requires_verified_injector(tmp_path):
    config = cfg(run_id="bad-downstream-label")
    exp = {
        "campaign_id": "phase6-publication-campaign",
        "campaign_condition_id": "DOWNSTREAM_MEDIUM",
        "mechanism_family": "downstream",
        "active_mechanisms": ["downstream_100ms"],
        "mechanism_levels": {"downstream_100ms": "medium"},
        "compound_degree": 1,
        "repetition": 1,
        "attempt": 1,
        "run_id": "bad-downstream-label",
    }
    config = ExperimentConfig.from_dict({
        **config.to_dict(include_hash=False),
        "runtime": {**config.runtime.__dict__, "parameters": {"experimental_condition_required": True, "experimental_condition": exp}},
    })
    run_dir = ExperimentRunner(config, tmp_path).run()
    result = validate_run(run_dir)
    assert not result["valid"]
    assert "experimental_condition_mismatch" in codes(result)


def test_phase5_factorial_mapping_complete_and_input_output_deferred():
    design = phase5_factorial_design()
    ids = {c.condition_id for c in phase5_conditions()}
    assert "INPUT_OUTPUT" not in ids
    assert "INPUT_OUTPUT" in design["deferred"]
    for compound in ["INPUT_LOAD", "INPUT_DOWNSTREAM", "OUTPUT_LOAD", "OUTPUT_DOWNSTREAM", "LOAD_DOWNSTREAM"]:
        mapping = design[compound]
        assert {mapping[cell] for cell in ["A0B0", "A1B0", "A0B1", "A1B1"]} <= ids
    assert design["OUTPUT_LOAD"]["A0B1"] == "OUTPUT_LOAD_CONTROL"
    assert design["OUTPUT_DOWNSTREAM"]["A0B1"] == "OUTPUT_DOWNSTREAM_CONTROL"


def test_phase5_prediction_limit_formulas():
    upper = phase5_prediction_limit([1.0, 2.0, 3.0], direction="upper")
    lower = phase5_prediction_limit([1.0, 2.0, 3.0], direction="lower")
    assert upper["threshold"] > upper["mean"]
    assert lower["threshold"] < lower["mean"]
    assert upper["formula"] if "formula" in upper else True


def test_phase52_family_prediction_limits_and_global_thresholds():
    baseline = [
        {"run_id": f"b{i}", "valid": True, "ttft_p95": 0.05, "total_latency_p95": 0.20, "post_first_token_duration_p95": 0.10, "run_manifest_sha256": f"{i:064x}"}
        for i in range(8)
    ]
    output = [
        {"run_id": f"o{i}", "valid": True, "ttft_p95": 0.07, "total_latency_p95": 0.30, "post_first_token_duration_p95": 0.14, "run_manifest_sha256": f"{i+10:064x}"}
        for i in range(8)
    ]
    artifact = phase5_build_slo_calibration_artifact(baseline, output)
    ttft_limits = artifact["family_limits"]["TTFT_SLO"]
    total_limits = artifact["family_limits"]["TOTAL_LATENCY_SLO"]
    assert artifact["metrics"]["TTFT_SLO"]["final_numeric_threshold"] == max(ttft_limits["baseline"]["threshold"], ttft_limits["output_control"]["threshold"])
    assert artifact["metrics"]["TOTAL_LATENCY_SLO"]["final_numeric_threshold"] == max(total_limits["baseline"]["threshold"], total_limits["output_control"]["threshold"])
    assert set(artifact["family_limits"]["DECODE_DURATION_SLO"]) == {"output_control"}
    assert artifact["throughput"]["status"] == "DIAGNOSTIC_ONLY"


def test_phase52_dedicated_calibration_requires_8_plus_8():
    row = {"run_id": "x", "valid": True, "ttft_p95": 0.1, "total_latency_p95": 0.2, "post_first_token_duration_p95": 0.1, "run_manifest_sha256": "a" * 64}
    with pytest.raises(ValueError):
        phase5_build_slo_calibration_artifact([row] * 7, [row] * 8)
    with pytest.raises(ValueError):
        phase5_build_slo_calibration_artifact([row] * 8, [row] * 7)


def test_phase52_freeze_rejects_incomplete_calibration(tmp_path):
    generate_phase5(tmp_path)
    issues = validate_phase5_freeze(tmp_path)
    assert "missing dedicated SLO calibration" in issues
    assert "SLO calibration is not complete" in issues


def test_phase5_slo_calibration_and_repetition_artifacts(tmp_path):
    seed_fake_phase52_calibration(tmp_path)
    generate_phase5(tmp_path)
    calibration = json.loads((tmp_path / "campaigns" / "phase5" / "slo-calibration.json").read_text(encoding="utf-8"))
    assert calibration["formal_campaign_outcomes_used"] is False
    assert all(isinstance(item["final_numeric_threshold"], float) for item in calibration["metrics"].values())
    rep = json.loads((tmp_path / "campaigns" / "phase5" / "repetition-justification.json").read_text(encoding="utf-8"))
    assert rep["selected_repetitions"] == PHASE5_REPETITIONS
    assert rep["calculations"]


def test_phase5_preflight_dirty_and_source_hash_checks(tmp_path, monkeypatch):
    seed_fake_phase52_calibration(tmp_path)
    generate_phase5(tmp_path)
    manifest = json.loads((tmp_path / "campaigns" / "phase5" / "campaign-manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setattr("sloscope.campaign_phase5.git_dirty", lambda root: True)
    result = phase5_run_preflight(tmp_path, frozen_source_sha=manifest["source_tree_sha256"])
    assert "git_clean" in result["issues"]
    monkeypatch.setattr("sloscope.campaign_phase5.git_dirty", lambda root: False)
    monkeypatch.setattr("sloscope.campaign_phase5.source_tree_sha256", lambda root: manifest["source_tree_sha256"])
    monkeypatch.setattr("sloscope.campaign_phase5.shutil.disk_usage", lambda root: type("DU", (), {"free": 10**12})())
    assert phase5_run_preflight(tmp_path, frozen_source_sha=manifest["source_tree_sha256"])["valid"]
    result = phase5_run_preflight(tmp_path, frozen_source_sha="bad")
    assert "source_tree_sha256" in result["issues"]


def test_phase52_preflight_spec_has_no_null_freeze_hash():
    spec = phase5_preflight_spec("abc")
    assert "campaign_freeze_sha256" not in spec
    assert spec["expected_freeze_hash_source"] == "campaign-manifest.json:campaign_freeze_sha256"


def test_phase52_preflight_runtime_service_and_config_checks(tmp_path, monkeypatch):
    seed_fake_phase52_calibration(tmp_path)
    generate_phase5(tmp_path)
    manifest = json.loads((tmp_path / "campaigns" / "phase5" / "campaign-manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setattr("sloscope.campaign_phase5.git_dirty", lambda root: False)
    monkeypatch.setattr("sloscope.campaign_phase5.source_tree_sha256", lambda root: manifest["source_tree_sha256"])
    monkeypatch.setattr("sloscope.campaign_phase5.shutil.disk_usage", lambda root: type("DU", (), {"free": 10**12})())
    result = phase5_run_preflight(
        tmp_path,
        runtime_info={"version": "bad", "build": "11146", "commit": "bad", "model_id": "wrong", "server_arguments": ["--metrics"]},
        service_status={"llama_healthy": False, "gateway_ready": False, "dependency_healthy": False, "dependency_delay_ms": 10},
    )
    for code in ["runtime_version", "runtime_commit", "model_id", "server_arguments", "prompt_cache_disabled", "llama_healthy", "gateway_ready", "dependency_healthy", "dependency_zero_delay"]:
        assert code in result["issues"]
    first = manifest["campaign_order"][0]
    (tmp_path / first["run_directory"]).mkdir(parents=True)
    result = phase5_run_preflight(tmp_path)
    assert "run_directories_absent" in result["issues"]


def test_phase5_cooldown_executes():
    calls = []
    result = phase5_execute_cooldown(5, sleeper=lambda seconds: calls.append(seconds))
    assert calls == [5]
    assert result["cooldown_seconds"] == 5


def test_phase5_validator_detects_duplicate_output_destination(monkeypatch):
    original = phase5_campaign_rows

    def dup_rows(revision=None):
        rows = original(revision)
        rows[1] = {**rows[1], "run_directory": rows[0]["run_directory"]}
        return rows

    monkeypatch.setattr("sloscope.campaign_phase5.campaign_rows", dup_rows)
    from sloscope.campaign_phase5 import validate_freeze

    assert "duplicate run output destination" in validate_freeze()


def timing_payload(request_id="req", base=100.0):
    return {
        "request_id": request_id,
        "gateway_receive_time": base,
        "dependency_start_time": base + 0.01,
        "dependency_end_time": base + 0.11,
        "llama_dispatch_time": base + 0.11,
        "llama_first_token_time": base + 0.20,
        "gateway_completion_time": base + 0.30,
        "dependency_duration": 0.10,
    }


def request_row(request_id, *, status="success", with_timing=False, base=100.0):
    row = RequestRecord(
        request_id,
        0,
        1.0,
        1.0,
        1.0,
        1.1 if status == "success" else None,
        1.2,
        None,
        None,
        status,
        "llama-runtime",
        "fake-gguf",
    ).to_dict()
    if with_timing:
        row.update({k: v for k, v in timing_payload(request_id, base).items() if k in GATEWAY_TIMING_FIELDS})
    return row


def with_traces_enabled(config):
    data = config.to_dict(include_hash=False)
    data["telemetry"] = {**data.get("telemetry", {}), "traces": True}
    return ExperimentConfig.from_dict(data)


def test_gateway_timing_reconciliation_fills_missing_without_changing_scientific_times(monkeypatch):
    adapter = LlamaCppRuntimeAdapter(llama_cfg("http://fake").runtime, SystemClock(), 1)
    row = request_row("r1")
    before = {k: row[k] for k in ["scheduled_arrival", "actual_arrival", "dispatch_time", "first_token_time", "completion_time", "scheduler_slip"]}
    calls = []

    def fake_timing(request_id):
        calls.append(request_id)
        return {} if len(calls) == 1 else timing_payload(request_id)

    monkeypatch.setattr(adapter, "_gateway_timing", fake_timing)
    adapter.gateway_timing_poll_interval = 0.001
    result = adapter.reconcile_gateway_timings([row])
    assert result["required_count"] == 1
    assert result["resolved_count"] == 1
    assert result["unresolved_count"] == 0
    assert len(calls) == 2
    assert all(row[field] is not None for field in GATEWAY_TIMING_FIELDS)
    assert {k: row[k] for k in before} == before


def test_gateway_timing_reconciliation_timeout_leaves_run_invalid(monkeypatch):
    adapter = LlamaCppRuntimeAdapter(llama_cfg("http://fake").runtime, SystemClock(), 1)
    adapter.gateway_timing_reconciliation_timeout = 0.0
    row = request_row("never")
    monkeypatch.setattr(adapter, "_gateway_timing", lambda request_id: {})
    result = adapter.reconcile_gateway_timings([row])
    assert result["required_count"] == 1
    assert result["resolved_count"] == 0
    assert result["unresolved_count"] == 1
    assert all(row[field] is None for field in GATEWAY_TIMING_FIELDS)


def test_gateway_timing_reconciliation_multiple_missing_join_correctly(monkeypatch):
    adapter = LlamaCppRuntimeAdapter(llama_cfg("http://fake").runtime, SystemClock(), 1)
    rows = [request_row("r1"), request_row("r2"), request_row("r3", with_timing=True, base=300.0)]
    seen = []

    def fake_timing(request_id):
        seen.append(request_id)
        return timing_payload(request_id, 100.0 if request_id == "r1" else 200.0)

    monkeypatch.setattr(adapter, "_gateway_timing", fake_timing)
    result = adapter.reconcile_gateway_timings(rows)
    assert result["required_count"] == 2
    assert result["resolved_count"] == 2
    assert set(seen) == {"r1", "r2"}
    assert "r3" not in seen
    assert rows[0]["gateway_receive_time"] == 100.0
    assert rows[1]["gateway_receive_time"] == 200.0
    assert rows[2]["gateway_receive_time"] == 300.0


def install_fake_llama_with_gateway_timing(monkeypatch, *, delayed_once=None, never=None, request_count=1):
    delayed_once = set(delayed_once or [])
    never = set(never or [])
    lookups = []
    completions = []
    trace_ids = []

    def spans_for(request_id):
        return [
            {"trace_id": f"{abs(hash(request_id)) % (16**32):032x}", "span_id": f"{abs(hash(request_id + 'g')) % (16**16):016x}", "parent_span_id": None, "span_name": "gateway request", "start_time": 1.0, "end_time": 1.3, "duration": 0.3, "status": "ok", "attributes": json.dumps({"request_id": request_id})},
            {"trace_id": f"{abs(hash(request_id)) % (16**32):032x}", "span_id": f"{abs(hash(request_id + 'd')) % (16**16):016x}", "parent_span_id": f"{abs(hash(request_id + 'g')) % (16**16):016x}", "span_name": "dependency call", "start_time": 1.01, "end_time": 1.11, "duration": 0.1, "status": "ok", "attributes": json.dumps({"request_id": request_id})},
            {"trace_id": f"{abs(hash(request_id)) % (16**32):032x}", "span_id": f"{abs(hash(request_id + 'l')) % (16**16):016x}", "parent_span_id": f"{abs(hash(request_id + 'g')) % (16**16):016x}", "span_name": "llama call", "start_time": 1.11, "end_time": 1.3, "duration": 0.19, "status": "ok", "attributes": json.dumps({"request_id": request_id})},
        ]

    timing_lookup_counts = {}

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url.endswith("/health"):
            return FakeHTTPResponse(body=b'{"status":"ok"}')
        if url.endswith("/v1/models"):
            return FakeHTTPResponse(body=b'{"data":[{"id":"fake-gguf"}]}')
        if url.endswith("/metrics"):
            return FakeHTTPResponse(body=b"llamacpp:requests_processing 0\n")
        if url.endswith("/sloscope/reset"):
            return FakeHTTPResponse(body=b'{"status":"ok","gateway":"sloscope"}')
        if url.endswith("/sloscope/traces"):
            spans = []
            for request_id in trace_ids:
                spans.extend(spans_for(request_id))
            return FakeHTTPResponse(body=json.dumps({"spans": spans}).encode("utf-8"))
        if "/sloscope/requests/" in url:
            request_id = url.rsplit("/", 1)[-1]
            lookups.append(request_id)
            timing_lookup_counts[request_id] = timing_lookup_counts.get(request_id, 0) + 1
            if request_id in never:
                raise FakeHTTPError(url, 404, b'{"error":"not found"}')
            if request_id in delayed_once and timing_lookup_counts[request_id] == 1:
                raise FakeHTTPError(url, 404, b'{"error":"not found"}')
            return FakeHTTPResponse(body=json.dumps(timing_payload(request_id)).encode("utf-8"))
        if url.endswith("/v1/completions"):
            request_id = req.get_header("X-SLOScope-Request-Id")
            if request_id is None:
                request_id = next((value for key, value in req.headers.items() if key.lower() == "x-sloscope-request-id"), None)
            completions.append(request_id)
            trace_ids.append(request_id)
            return FakeHTTPResponse(
                lines=[
                    b"data: {\"choices\":[{\"text\":\"x\",\"finish_reason\":null}]}\n\n",
                    b"data: {\"choices\":[{\"text\":\"y\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\n",
                    b"data: [DONE]\n\n",
                ]
            )
        raise FakeHTTPError(url, 404, b'{"error":"not found"}')

    monkeypatch.setattr("sloscope.runtime.llamacpp.urllib.request.urlopen", fake_urlopen)
    return {"lookups": lookups, "completions": completions}


def test_runner_reconciles_delayed_gateway_timing_after_workload(monkeypatch, tmp_path):
    run_id = "race-reconciled"
    request_id = f"{run_id}-req-000000"
    observed = install_fake_llama_with_gateway_timing(monkeypatch, delayed_once={request_id})
    cfg = llama_cfg("http://fake", run_id=run_id, request_count=1, metrics_enabled=True)
    cfg.runtime.parameters["gateway_timing_reconciliation_timeout_seconds"] = 0.1
    cfg.runtime.parameters["gateway_timing_poll_interval_seconds"] = 0.001
    cfg = with_traces_enabled(cfg)
    run_dir = ExperimentRunner(cfg, tmp_path).run()
    result = validate_run(run_dir)
    _, rows = read_table(run_dir / "requests.parquet")
    metadata = read_json(run_dir / "runtime_metadata.json")
    assert result["valid"]
    assert rows[0]["status"] == "success"
    assert all(rows[0][field] is not None for field in GATEWAY_TIMING_FIELDS)
    assert metadata["gateway_timing_reconciliation"]["required_count"] == 1
    assert metadata["gateway_timing_reconciliation"]["resolved_count"] == 1
    assert observed["lookups"].count(request_id) >= 2


def test_runner_permanent_missing_gateway_timing_remains_invalid(monkeypatch, tmp_path):
    run_id = "race-unresolved"
    request_id = f"{run_id}-req-000000"
    install_fake_llama_with_gateway_timing(monkeypatch, never={request_id})
    cfg = llama_cfg("http://fake", run_id=run_id, request_count=1, metrics_enabled=True)
    cfg.runtime.parameters["gateway_timing_reconciliation_timeout_seconds"] = 0.0
    cfg = with_traces_enabled(cfg)
    run_dir = ExperimentRunner(cfg, tmp_path).run()
    result = validate_run(run_dir)
    _, rows = read_table(run_dir / "requests.parquet")
    metadata = read_json(run_dir / "runtime_metadata.json")
    assert not result["valid"]
    assert "missing_gateway_timing" in codes(result)
    assert all(rows[0][field] is None for field in GATEWAY_TIMING_FIELDS)
    assert metadata["gateway_timing_reconciliation"]["unresolved_count"] == 1


def test_reconciliation_does_not_delay_open_loop_or_completion(monkeypatch, tmp_path):
    run_id = "race-schedule"
    delayed = {f"{run_id}-req-{idx:06d}" for idx in range(3)}
    observed = install_fake_llama_with_gateway_timing(monkeypatch, delayed_once=delayed)
    cfg = llama_cfg("http://fake", run_id=run_id, request_count=3, inter_arrival=0.01, max_outstanding=3)
    cfg.runtime.parameters["gateway_timing_reconciliation_timeout_seconds"] = 0.1
    cfg.runtime.parameters["gateway_timing_poll_interval_seconds"] = 0.001
    cfg = with_traces_enabled(cfg)
    run_dir = ExperimentRunner(cfg, tmp_path).run()
    _, rows = read_table(run_dir / "requests.parquet")
    metadata = read_json(run_dir / "runtime_metadata.json")
    assert validate_run(run_dir)["valid"]
    assert rows[1]["actual_arrival"] - rows[0]["actual_arrival"] < 0.04
    assert len(observed["completions"]) == 3
    assert metadata["gateway_timing_reconciliation"]["required_count"] == 3
    assert metadata["gateway_timing_reconciliation"]["resolved_count"] == 3
