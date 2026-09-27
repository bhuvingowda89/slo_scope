import json
import time
import urllib.error

import pytest

from sloscope.cli import main as cli_main
from sloscope.artifacts.validation import validate_run
from sloscope.artifacts.writer import read_json, read_table, write_json
from sloscope.artifacts.writer import ARTIFACT_SCHEMA_VERSION
from sloscope.config import (
    ExperimentConfig,
    MechanismConfig,
    RuntimeConfig,
    SUPPORTED_SCHEMA_VERSION,
    TelemetryConfig,
    WorkloadConfig,
    write_config,
)
from sloscope.ground_truth import GroundTruthLedger
from sloscope.injectors.mock import MockInjector
from sloscope.injectors.cpu_contention import CPUContentionInjector, worker_count_for_intensity
from sloscope.injectors.factory import create_injector
from sloscope.lifecycle import ExperimentState, FakeClock, IllegalTransitionError, Lifecycle, SystemClock
from sloscope.runtime.llamacpp import LlamaCppRuntimeAdapter, content_from_chunk, parse_prometheus_metrics, parse_sse_payload
from sloscope.runner import ExperimentRunner
from sloscope.telemetry.schemas import RequestRecord
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


def test_config_deterministic_hashing():
    a = cfg()
    b = ExperimentConfig.from_dict(json.loads(a.canonical_json(include_hash=False)))
    assert a.compute_hash() == b.compute_hash()
    assert a.with_hash().config_hash == a.compute_hash()


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
    assert metrics == {"queue_delay": 1, "ttft": 3, "service_time": 5, "total_latency": 6, "scheduler_slip": 1}


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
    with pytest.raises(ValueError):
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
