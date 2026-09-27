from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


REQUEST_COLUMNS = {
    "request_id": "string",
    "sequence": "int64",
    "scheduled_arrival": "float64",
    "actual_arrival": "float64",
    "dispatch_time": "float64",
    "first_token_time": "float64",
    "completion_time": "float64",
    "prompt_tokens": "int64",
    "output_tokens": "int64",
    "status": "string",
    "runtime_id": "string",
    "model_id": "string",
    "http_status": "int64",
    "error_type": "string",
    "error_message": "string",
    "server_prompt_tokens": "int64",
    "server_output_tokens": "int64",
    "server_prompt_seconds": "float64",
    "server_generation_seconds": "float64",
    "finish_reason": "string",
    "scheduler_slip": "float64",
    "token_count_source": "string",
}
SYSTEM_METRIC_COLUMNS = {
    "timestamp": "float64", "host_cpu": "float64", "process_cpu": "float64",
    "memory": "float64", "memory_pressure": "float64", "disk_io": "float64", "network": "float64",
    "process_rss": "float64", "memory_used": "float64", "memory_available": "float64",
    "host_cpu_percent": "float64", "host_memory_percent": "float64",
    "host_memory_used_bytes": "float64", "host_memory_available_bytes": "float64",
    "client_process_cpu_percent": "float64", "client_process_rss_bytes": "float64",
    "server_pid": "int64", "server_process_cpu_percent": "float64", "server_process_rss_bytes": "float64",
}
RUNTIME_METRIC_COLUMNS = {
    "timestamp": "float64", "runtime_id": "string", "queue_depth": "int64", "active_requests": "int64",
    "deferred_requests": "int64", "throughput": "float64", "counters": "string",
    "metric_name": "string", "metric_value": "float64", "labels": "string",
}
TRACE_COLUMNS = {
    "trace_id": "string", "span_id": "string", "parent_span_id": "string",
    "name": "string", "start_time": "float64", "end_time": "float64", "attributes": "string",
}


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    sequence: int
    scheduled_arrival: float
    actual_arrival: Optional[float]
    dispatch_time: Optional[float]
    first_token_time: Optional[float]
    completion_time: Optional[float]
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    status: str
    runtime_id: str
    model_id: str
    http_status: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    server_prompt_tokens: Optional[int] = None
    server_output_tokens: Optional[int] = None
    server_prompt_seconds: Optional[float] = None
    server_generation_seconds: Optional[float] = None
    finish_reason: Optional[str] = None
    scheduler_slip: Optional[float] = None
    token_count_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate_ordering(self) -> list[str]:
        issues: list[str] = []
        if self.status not in {"success", "failed", "admission_failed"}:
            issues.append(f"unknown status {self.status}")
        if self.actual_arrival is None:
            return issues if self.status == "failed" else ["actual_arrival missing"]
        if self.dispatch_time is None:
            return issues if self.status == "failed" else ["dispatch_time missing"]
        if self.actual_arrival > self.dispatch_time:
            issues.append("actual_arrival after dispatch_time")
        if self.first_token_time is not None and self.dispatch_time > self.first_token_time:
            issues.append("dispatch_time after first_token_time")
        if self.completion_time is not None:
            if self.first_token_time is not None and self.first_token_time > self.completion_time:
                issues.append("first_token_time after completion_time")
            if self.dispatch_time > self.completion_time:
                issues.append("dispatch_time after completion_time")
        elif self.status != "failed":
            issues.append("completion_time missing")
        if self.status == "success":
            if self.first_token_time is None:
                issues.append("successful request missing first_token_time")
            if self.completion_time is None:
                issues.append("successful request missing completion_time")
        if self.status == "failed" and not (self.error_type or self.error_message):
            issues.append("failed request missing error metadata")
        return issues

    def derived_metrics(self) -> Dict[str, Optional[float]]:
        if self.actual_arrival is None or self.dispatch_time is None or self.completion_time is None:
            return {"queue_delay": None, "ttft": None, "service_time": None, "total_latency": None}
        return {
            "queue_delay": self.dispatch_time - self.actual_arrival,
            "ttft": None if self.first_token_time is None else self.first_token_time - self.actual_arrival,
            "service_time": self.completion_time - self.dispatch_time,
            "total_latency": self.completion_time - self.actual_arrival,
            "scheduler_slip": self.actual_arrival - self.scheduled_arrival,
        }
