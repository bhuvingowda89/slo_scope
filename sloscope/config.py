from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union
from .prompts import PROMPT_CORPUS

SUPPORTED_SCHEMA_VERSION = "sloscope.config.v1"
LEGACY_SCHEMA_VERSIONS = {"phase3a.v1"}
SUPPORTED_SCHEMA_VERSIONS = {SUPPORTED_SCHEMA_VERSION, *LEGACY_SCHEMA_VERSIONS}


def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class RuntimeConfig:
    runtime_id: str
    runtime_type: str
    model_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadConfig:
    arrival_pattern: str
    request_count: int
    concurrency: int
    prompt_profile: str
    output_profile: str
    target_output_tokens: int
    inter_arrival_seconds: float = 0.1
    burst_size: int = 5
    burst_interval_seconds: float = 1.0
    randomized_output: bool = False
    output_jitter_tokens: int = 0
    start_offset_seconds: float = 0.0


@dataclass(frozen=True)
class MechanismConfig:
    mechanism_id: str
    mechanism_type: str
    intensity: float
    target: str
    scheduled_onset: float
    scheduled_duration: float
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def scheduled_stop(self) -> float:
        return self.scheduled_onset + self.scheduled_duration


@dataclass(frozen=True)
class TelemetryConfig:
    request_telemetry: bool = True
    system_metrics: bool = True
    runtime_metrics: bool = True
    traces: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    maximum_cpu_stress: float = 0.0
    maximum_allocated_pressure_memory: int = 0
    maximum_disk_io: int = 0
    experiment_timeout: float = 60.0
    require_requests_within_mechanism_window: bool = False
    maximum_dependency_delay_ms: int = 0
    mechanism_watchdog_interval_seconds: float = 0.25


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    run_id: str
    seed: int
    runtime: RuntimeConfig
    workload: WorkloadConfig
    mechanisms: List[MechanismConfig] = field(default_factory=list)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    config_hash: Optional[str] = None

    def to_dict(self, include_hash: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_hash:
            data.pop("config_hash", None)
        return data

    def canonical_json(self, include_hash: bool = False) -> str:
        return _canonical_json(self.to_dict(include_hash=include_hash))

    def compute_hash(self) -> str:
        return hashlib.sha256(self.canonical_json(include_hash=False).encode("utf-8")).hexdigest()

    def with_hash(self) -> "ExperimentConfig":
        return ExperimentConfig.from_dict({**self.to_dict(include_hash=False), "config_hash": self.compute_hash()})

    def validate(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unknown schema_version: {self.schema_version}")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.workload.request_count < 0:
            raise ValueError("request_count must be non-negative")
        if self.workload.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.workload.inter_arrival_seconds < 0:
            raise ValueError("inter_arrival_seconds must be non-negative")
        if self.workload.burst_size < 1:
            raise ValueError("burst_size must be >= 1")
        if self.workload.burst_interval_seconds < 0:
            raise ValueError("burst_interval_seconds must be non-negative")
        if self.workload.target_output_tokens < 0:
            raise ValueError("target_output_tokens must be non-negative")
        if self.workload.output_jitter_tokens < 0:
            raise ValueError("output_jitter_tokens must be non-negative")
        if self.workload.start_offset_seconds < 0:
            raise ValueError("start_offset_seconds must be non-negative")
        if self.workload.arrival_pattern not in {"constant_open_loop", "burst", "batched_arrival"}:
            raise ValueError(f"unknown arrival_pattern: {self.workload.arrival_pattern}")
        if self.workload.prompt_profile not in PROMPT_CORPUS:
            raise ValueError(f"unknown prompt_profile: {self.workload.prompt_profile}")
        if self.workload.output_profile != "fixed":
            raise ValueError(f"unsupported output_profile: {self.workload.output_profile}")
        if self.runtime.runtime_type not in {"mock", "llamacpp"}:
            raise ValueError(f"unknown runtime_type: {self.runtime.runtime_type}")
        if self.runtime.runtime_type == "llamacpp":
            if not self.runtime.parameters.get("base_url"):
                raise ValueError("llamacpp runtime requires parameters.base_url")
            if int(self.runtime.parameters.get("max_outstanding_requests", 1)) < 1:
                raise ValueError("max_outstanding_requests must be >= 1")
            if not self.telemetry.request_telemetry:
                raise ValueError("llamacpp Phase 3B runs require request_telemetry=true")
        ids = [m.mechanism_id for m in self.mechanisms]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate mechanism_id")
        for mech in self.mechanisms:
            if mech.scheduled_onset < 0 or mech.scheduled_duration < 0:
                raise ValueError("mechanism schedules must be non-negative")
            if mech.mechanism_type == "cpu_contention":
                if mech.target != "host_cpu":
                    raise ValueError("cpu_contention target must be host_cpu")
                if not (0 < mech.intensity <= 1.0):
                    raise ValueError("cpu_contention intensity must satisfy 0 < intensity <= 1.0")
                if mech.intensity > self.safety.maximum_cpu_stress:
                    raise ValueError("cpu_contention intensity exceeds safety.maximum_cpu_stress")
                logical_cpu_count = int(mech.parameters.get("logical_cpu_count") or (os.cpu_count() or 1))
                worker_count = int(mech.parameters.get("worker_count") or max(1, round(logical_cpu_count * mech.intensity)))
                if worker_count < 1:
                    raise ValueError("cpu_contention worker_count must be >= 1")
                effective_worker_fraction = worker_count / logical_cpu_count
                if effective_worker_fraction > self.safety.maximum_cpu_stress:
                    raise ValueError("cpu_contention effective worker fraction exceeds safety.maximum_cpu_stress")
            elif mech.mechanism_type == "downstream_latency":
                if mech.target != "synthetic_dependency":
                    raise ValueError("downstream_latency target must be synthetic_dependency")
                delay_ms = int(mech.parameters.get("delay_ms", mech.intensity))
                if delay_ms < 0:
                    raise ValueError("downstream_latency delay_ms must be non-negative")
                if delay_ms > self.safety.maximum_dependency_delay_ms:
                    raise ValueError("downstream_latency delay_ms exceeds safety.maximum_dependency_delay_ms")
        if self.safety.maximum_cpu_stress < 0:
            raise ValueError("maximum_cpu_stress must be non-negative")
        if self.safety.maximum_allocated_pressure_memory < 0:
            raise ValueError("maximum_allocated_pressure_memory must be non-negative")
        if self.safety.maximum_disk_io < 0:
            raise ValueError("maximum_disk_io must be non-negative")
        if self.safety.experiment_timeout <= 0:
            raise ValueError("experiment_timeout must be positive")
        if self.safety.maximum_dependency_delay_ms < 0:
            raise ValueError("maximum_dependency_delay_ms must be non-negative")
        if self.safety.mechanism_watchdog_interval_seconds <= 0:
            raise ValueError("mechanism_watchdog_interval_seconds must be positive")
        if self.config_hash is not None and self.config_hash != self.compute_hash():
            raise ValueError("config_hash mismatch")

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ExperimentConfig":
        return ExperimentConfig(
            schema_version=str(data["schema_version"]),
            run_id=str(data["run_id"]),
            seed=int(data["seed"]),
            runtime=RuntimeConfig(**dict(data["runtime"])),
            workload=WorkloadConfig(**dict(data["workload"])),
            mechanisms=[MechanismConfig(**dict(item)) for item in data.get("mechanisms", [])],
            telemetry=TelemetryConfig(**dict(data.get("telemetry", {}))),
            safety=SafetyConfig(**dict(data.get("safety", {}))),
            config_hash=data.get("config_hash"),
        )


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = ExperimentConfig.from_dict(json.load(fh))
    cfg.validate()
    return cfg


def write_config(path: Union[str, Path], config: ExperimentConfig) -> None:
    cfg = config.with_hash()
    Path(path).write_text(cfg.canonical_json(include_hash=True) + "\n", encoding="utf-8")
