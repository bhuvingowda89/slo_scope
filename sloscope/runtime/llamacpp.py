from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sloscope.config import RuntimeConfig
from sloscope.lifecycle import Clock
from sloscope.prompts import prompt_for
from sloscope.runtime.base import RuntimeAdapter
from sloscope.telemetry.schemas import RequestRecord
from sloscope.workload import RequestPlan


class LlamaCppRuntimeError(RuntimeError):
    pass


GATEWAY_TIMING_FIELDS = (
    "gateway_receive_time",
    "dependency_start_time",
    "dependency_end_time",
    "llama_dispatch_time",
    "llama_first_token_time",
    "gateway_completion_time",
    "dependency_duration",
)


@dataclass(frozen=True)
class LlamaHealth:
    state: str
    diagnostics: Dict[str, Any]


def _json_request(method: str, url: str, timeout: float, payload: Optional[dict] = None) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return int(resp.status), json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"error": body}
        return int(exc.code), parsed


def parse_sse_payload(line: bytes) -> Optional[dict | str]:
    text = line.decode("utf-8", errors="replace").strip()
    if not text or text.startswith(":"):
        return None
    if text.startswith("data:"):
        text = text[5:].strip()
    if not text:
        return None
    if text == "[DONE]":
        return "[DONE]"
    return json.loads(text)


def content_from_chunk(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    return choice.get("text") or choice.get("delta", {}).get("content") or ""


def finish_reason_from_chunk(chunk: dict) -> Optional[str]:
    choices = chunk.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def usage_from_payload(payload: dict) -> dict:
    usage = payload.get("usage") or payload.get("timings") or {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("prompt_n"))
    output_tokens = usage.get("completion_tokens", usage.get("predicted_n"))
    prompt_seconds = usage.get("prompt_ms")
    generation_seconds = usage.get("predicted_ms")
    if prompt_seconds is not None:
        prompt_seconds = float(prompt_seconds) / 1000.0
    if generation_seconds is not None:
        generation_seconds = float(generation_seconds) / 1000.0
    return {
        "server_prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "server_output_tokens": int(output_tokens) if output_tokens is not None else None,
        "server_prompt_seconds": prompt_seconds,
        "server_generation_seconds": generation_seconds,
    }


def parse_prometheus_metrics(text: str, timestamp: float, runtime_id: str) -> List[dict]:
    rows: list[dict] = []
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            text = json.loads(text)
        except Exception:
            pass
    pattern = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+([-+]?(?:[0-9]*\.?[0-9]+)(?:[eE][-+]?[0-9]+)?)$")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        try:
            metric_value = float(value)
        except ValueError:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "runtime_id": runtime_id,
                "queue_depth": None,
                "active_requests": None,
                "deferred_requests": None,
                "throughput": None,
                "counters": None,
                "metric_name": name,
                "metric_value": metric_value,
                "labels": labels or "",
            }
        )
    return rows


class LlamaCppRuntimeAdapter(RuntimeAdapter):
    def __init__(self, config: RuntimeConfig, clock: Clock, seed: int) -> None:
        self.config = config
        self.clock = clock
        self.seed = seed
        self.params = config.parameters
        self.base_url = str(self.params.get("base_url", "http://127.0.0.1:8080")).rstrip("/")
        self.request_timeout = float(self.params.get("request_timeout", 60.0))
        self.health_timeout = float(self.params.get("health_timeout", 5.0))
        self.metrics_enabled = bool(self.params.get("metrics_enabled", False))
        self.allow_model_mismatch = bool(self.params.get("allow_model_mismatch", False))
        self.endpoint = str(self.params.get("request_endpoint", "/v1/completions"))
        self.max_outstanding = int(self.params.get("max_outstanding_requests", self.params.get("client_max_outstanding", 8)))
        self.telemetry_interval_seconds = float(self.params.get("telemetry_interval_seconds", 0.25))
        self.accepted = 0
        self.succeeded = 0
        self.failed = 0
        self._counter_lock = threading.Lock()
        self.last_health: Optional[LlamaHealth] = None
        self.model_metadata: Dict[str, Any] = {}
        self.runtime_metadata: Dict[str, Any] = {}
        self.gateway_telemetry_enabled = bool(self.params.get("gateway_telemetry_enabled", True))
        self.gateway_timing_reconciliation_timeout = float(self.params.get("gateway_timing_reconciliation_timeout_seconds", 1.0))
        self.gateway_timing_poll_interval = float(self.params.get("gateway_timing_poll_interval_seconds", 0.01))
        self._trace_rows: List[dict] = []

    def _advance_epsilon(self) -> None:
        advance = getattr(self.clock, "advance", None)
        if callable(advance):
            advance(0.001)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _health(self) -> LlamaHealth:
        try:
            status, payload = _json_request("GET", self._url("/ready"), self.health_timeout)
            if status == 200 and isinstance(payload, dict) and str(payload.get("status", "")).lower() == "ok":
                return LlamaHealth("healthy", {"http_status": status, "payload": payload, "endpoint": "/ready"})
            if status == 503 and isinstance(payload, dict) and {"gateway_alive", "dependency_reachable", "llama_reachable"} & set(payload):
                return LlamaHealth("unavailable", {"http_status": status, "payload": payload, "endpoint": "/ready"})
        except Exception:
            pass
        try:
            status, payload = _json_request("GET", self._url("/health"), self.health_timeout)
        except Exception as exc:
            return LlamaHealth("unavailable", {"error": str(exc)})
        if not isinstance(payload, dict):
            return LlamaHealth("malformed", {"http_status": status, "payload": payload})
        if status == 503:
            error = payload.get("error")
            if isinstance(error, dict) and error.get("code") == 503 and "loading model" in str(error.get("message", "")).lower():
                return LlamaHealth("loading", {"http_status": status, "payload": payload})
            return LlamaHealth("unavailable", {"http_status": status, "payload": payload})
        if status >= 500:
            return LlamaHealth("unavailable", {"http_status": status, "payload": payload})
        payload_status = str(payload.get("status", "")).lower()
        if status == 200 and payload_status == "ok":
            return LlamaHealth("healthy", {"http_status": status, "payload": payload})
        if payload_status == "loading":
            return LlamaHealth("loading", {"http_status": status, "payload": payload})
        return LlamaHealth("malformed", {"http_status": status, "payload": payload})

    def reset_gateway_telemetry(self) -> Optional[dict]:
        if not self.gateway_telemetry_enabled:
            return None
        try:
            status, payload = _json_request("POST", self._url("/sloscope/reset"), min(self.health_timeout, 2.0), {})
        except Exception:
            return None
        if status == 200 and isinstance(payload, dict) and payload.get("gateway") == "sloscope":
            return payload
        return None

    def prepare(self) -> None:
        gateway_reset = self.reset_gateway_telemetry()
        health = self._health()
        self.last_health = health
        if health.state != "healthy":
            raise LlamaCppRuntimeError(f"llama.cpp healthcheck not healthy: {health.state}: {health.diagnostics}")
        self.model_metadata = self.query_models()
        reported = self.model_metadata.get("selected_model_id")
        if reported and self.config.model_id and reported != self.config.model_id and not self.allow_model_mismatch:
            raise LlamaCppRuntimeError(f"model mismatch: expected {self.config.model_id}, server reported {reported}")
        if self.metrics_enabled and not self.scrape_metrics(self.clock.monotonic()):
            raise LlamaCppRuntimeError("metrics_enabled=true but /metrics is unavailable or empty")
        self.runtime_metadata = {
            "runtime_type": "llamacpp",
            "base_url": self.base_url,
            "expected_model_id": self.config.model_id,
            "model_metadata": self.model_metadata,
            "server_metadata": self.params.get("server_metadata", {}),
            "llama_cpp_revision": self.params.get("llama_cpp_revision"),
            "llama_cpp_build_number": self.params.get("llama_cpp_build_number"),
            "server_command": self.params.get("server_command"),
            "server_arguments": self.params.get("server_arguments"),
            "prompt_cache_enabled": False if "--no-cache-prompt" in (self.params.get("server_arguments") or []) else self.params.get("prompt_cache_enabled"),
            "gateway_telemetry_enabled": self.gateway_telemetry_enabled,
            "gateway_reset": gateway_reset,
            "gateway_timing_reconciliation": {
                "timeout_seconds": self.gateway_timing_reconciliation_timeout,
                "poll_interval_seconds": self.gateway_timing_poll_interval,
                "required_count": 0,
                "resolved_count": 0,
                "unresolved_count": 0,
                "duration_seconds": 0.0,
            },
        }

    def healthcheck(self) -> bool:
        self.last_health = self._health()
        return self.last_health.state == "healthy"

    def warmup(self) -> None:
        return None

    def query_models(self) -> Dict[str, Any]:
        status, payload = _json_request("GET", self._url("/v1/models"), self.health_timeout)
        if status >= 400 or not isinstance(payload, dict):
            raise LlamaCppRuntimeError(f"failed to query /v1/models: status={status}")
        models = payload.get("data") or []
        first = models[0] if models else {}
        meta = first.get("meta") if isinstance(first.get("meta"), dict) else {}
        return {
            "raw": payload,
            "selected_model_id": first.get("id") or payload.get("id"),
            "architecture": first.get("architecture"),
            "parameter_count": first.get("parameter_count") or meta.get("n_params"),
            "training_context_size": first.get("context_length") or first.get("n_ctx_train") or meta.get("n_ctx_train"),
            "embedding_dimension": first.get("embedding_length") or first.get("n_embd") or meta.get("n_embd"),
            "model_size": first.get("size") or first.get("model_size") or meta.get("size"),
            "n_ctx": meta.get("n_ctx"),
            "vocab_type": meta.get("vocab_type"),
            "n_vocab": meta.get("n_vocab"),
            "ftype": meta.get("ftype"),
        }

    def execute(self, request: RequestPlan, actual_arrival: float) -> RequestRecord:
        return asyncio.run(self.execute_async(request, actual_arrival))

    async def execute_async(self, request: RequestPlan, actual_arrival: float) -> RequestRecord:
        return await asyncio.to_thread(self._execute_streaming, request, actual_arrival)

    def _execute_streaming(self, request: RequestPlan, actual_arrival: float) -> RequestRecord:
        with self._counter_lock:
            self.accepted += 1
        prompt_id, prompt = prompt_for(request.prompt_profile, request.sequence)
        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "stream": True,
            "temperature": self.params.get("temperature", 0),
            "max_tokens": request.target_output_tokens,
            "seed": self.params.get("seed", self.seed),
            "stream_options": {"include_usage": True},
        }
        payload.update(dict(self.params.get("completion_parameters", {})))
        dispatch = self.clock.monotonic()
        self._advance_epsilon()
        first_token = None
        completion = None
        output_text: list[str] = []
        finish_reason = None
        usage: dict[str, Any] = {}
        http_status = None
        try:
            req = urllib.request.Request(
                self._url(self.endpoint),
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json", "Accept": "text/event-stream", "X-SLOScope-Request-Id": request.request_id},
            )
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                http_status = int(resp.status)
                for raw in resp:
                    parsed = parse_sse_payload(raw)
                    if parsed is None:
                        continue
                    if parsed == "[DONE]":
                        break
                    if not isinstance(parsed, dict):
                        continue
                    if "error" in parsed:
                        raise LlamaCppRuntimeError(str(parsed["error"]))
                    usage.update({k: v for k, v in usage_from_payload(parsed).items() if v is not None})
                    finish_reason = finish_reason_from_chunk(parsed) or finish_reason
                    content = content_from_chunk(parsed)
                    if not content:
                        continue
                    if first_token is None:
                        first_token = self.clock.monotonic()
                    output_text.append(content)
                    self._advance_epsilon()
            completion = self.clock.monotonic()
            if first_token is None:
                raise LlamaCppRuntimeError("stream completed without a content-bearing token")
            with self._counter_lock:
                self.succeeded += 1
            gateway_timing = self._gateway_timing(request.request_id)
            return RequestRecord(
                request.request_id,
                request.sequence,
                request.scheduled_arrival,
                actual_arrival,
                dispatch,
                first_token,
                completion,
                None,
                None,
                "success",
                self.config.runtime_id,
                self.config.model_id,
                http_status=http_status,
                server_prompt_tokens=usage.get("server_prompt_tokens"),
                server_output_tokens=usage.get("server_output_tokens"),
                server_prompt_seconds=usage.get("server_prompt_seconds"),
                server_generation_seconds=usage.get("server_generation_seconds"),
                finish_reason=finish_reason,
                scheduler_slip=actual_arrival - request.scheduled_arrival,
                token_count_source="runtime_reported" if usage.get("server_prompt_tokens") is not None or usage.get("server_output_tokens") is not None else None,
                **gateway_timing,
            )
        except TimeoutError as exc:
            return self._failed_record(request, actual_arrival, dispatch, "timeout", str(exc), http_status)
        except urllib.error.HTTPError as exc:
            return self._failed_record(request, actual_arrival, dispatch, "http_error", exc.read().decode("utf-8", errors="replace"), int(exc.code))
        except json.JSONDecodeError as exc:
            return self._failed_record(request, actual_arrival, dispatch, "malformed_stream", str(exc), http_status)
        except Exception as exc:
            return self._failed_record(request, actual_arrival, dispatch, type(exc).__name__, str(exc), http_status)

    def _failed_record(self, request: RequestPlan, actual_arrival: float, dispatch: float, error_type: str, message: str, http_status: Optional[int]) -> RequestRecord:
        with self._counter_lock:
            self.failed += 1
        return RequestRecord(
            request.request_id,
            request.sequence,
            request.scheduled_arrival,
            actual_arrival,
            dispatch,
            None,
            self.clock.monotonic(),
            None,
            None,
            "failed",
            self.config.runtime_id,
            self.config.model_id,
            http_status=http_status,
            error_type=error_type,
            error_message=message,
            scheduler_slip=actual_arrival - request.scheduled_arrival,
            **self._gateway_timing(request.request_id),
        )

    def _gateway_timing(self, request_id: str) -> Dict[str, Any]:
        if not self.gateway_telemetry_enabled:
            return {}
        try:
            status, payload = _json_request("GET", self._url(f"/sloscope/requests/{request_id}"), min(self.health_timeout, 2.0))
            if status >= 400 or not isinstance(payload, dict):
                return {}
            return {
                "gateway_receive_time": payload.get("gateway_receive_time"),
                "dependency_start_time": payload.get("dependency_start_time"),
                "dependency_end_time": payload.get("dependency_end_time"),
                "llama_dispatch_time": payload.get("llama_dispatch_time"),
                "llama_first_token_time": payload.get("llama_first_token_time"),
                "gateway_completion_time": payload.get("gateway_completion_time"),
                "dependency_duration": payload.get("dependency_duration"),
            }
        except Exception:
            return {}

    @staticmethod
    def _has_complete_gateway_timing(row: Dict[str, Any]) -> bool:
        return all(row.get(field) is not None for field in GATEWAY_TIMING_FIELDS)

    def reconcile_gateway_timings(self, request_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fill missing gateway timings after measured workload completion.

        This deliberately runs outside the request execution path. It may mutate
        only gateway-side telemetry fields in the provided request dictionaries.
        """
        started = time.monotonic()
        if not self.gateway_telemetry_enabled:
            result = {
                "required_count": 0,
                "resolved_count": 0,
                "unresolved_count": 0,
                "duration_seconds": 0.0,
                "timeout_seconds": self.gateway_timing_reconciliation_timeout,
                "poll_interval_seconds": self.gateway_timing_poll_interval,
                "polled_request_ids": [],
            }
            self.runtime_metadata["gateway_timing_reconciliation"] = result
            return result
        by_id = {
            row.get("request_id"): row
            for row in request_rows
            if row.get("status") == "success"
            and row.get("request_id")
            and not self._has_complete_gateway_timing(row)
        }
        unresolved = set(by_id)
        required_count = len(unresolved)
        polled: list[str] = []
        deadline = started + max(0.0, self.gateway_timing_reconciliation_timeout)
        while unresolved:
            for request_id in list(unresolved):
                polled.append(str(request_id))
                timing = self._gateway_timing(str(request_id))
                if all(timing.get(field) is not None for field in GATEWAY_TIMING_FIELDS):
                    for field in GATEWAY_TIMING_FIELDS:
                        by_id[request_id][field] = timing.get(field)
                    unresolved.remove(request_id)
            if not unresolved or time.monotonic() >= deadline:
                break
            time.sleep(max(0.001, self.gateway_timing_poll_interval))
        ended = time.monotonic()
        result = {
            "required_count": required_count,
            "resolved_count": required_count - len(unresolved),
            "unresolved_count": len(unresolved),
            "duration_seconds": ended - started,
            "timeout_seconds": self.gateway_timing_reconciliation_timeout,
            "poll_interval_seconds": self.gateway_timing_poll_interval,
            "polled_request_ids": polled,
            "unresolved_request_ids": sorted(str(item) for item in unresolved),
        }
        self.runtime_metadata["gateway_timing_reconciliation"] = result
        return result

    def collect_trace_rows(self) -> List[dict]:
        if not self.gateway_telemetry_enabled:
            return []
        try:
            status, payload = _json_request("GET", self._url("/sloscope/traces"), min(self.health_timeout, 2.0))
            if status >= 400 or not isinstance(payload, dict):
                return []
            rows = payload.get("spans") or []
            if not isinstance(rows, list):
                return []
            self._trace_rows = [row for row in rows if isinstance(row, dict)]
            return list(self._trace_rows)
        except Exception:
            return []

    def scrape_metrics(self, timestamp: float) -> List[dict]:
        try:
            req = urllib.request.Request(self._url("/metrics"), method="GET")
            with urllib.request.urlopen(req, timeout=self.health_timeout) as resp:
                if int(resp.status) >= 400:
                    return []
                return parse_prometheus_metrics(resp.read().decode("utf-8", errors="replace"), timestamp, self.config.runtime_id)
        except Exception:
            return []

    def collect_runtime_metrics(self, timestamp: float) -> Dict[str, Any]:
        return {
            "timestamp": timestamp,
            "runtime_id": self.config.runtime_id,
            "queue_depth": None,
            "active_requests": None,
            "deferred_requests": None,
            "throughput": None,
            "counters": f"accepted_requests_total={self.accepted};successful_requests_total={self.succeeded};failed_requests_total={self.failed}",
            "metric_name": None,
            "metric_value": None,
            "labels": None,
        }

    def collect_runtime_metric_rows(self, timestamp: float) -> List[dict]:
        rows = [self.collect_runtime_metrics(timestamp)]
        if self.metrics_enabled:
            rows.extend(self.scrape_metrics(timestamp))
        return rows

    def shutdown(self) -> None:
        return None
