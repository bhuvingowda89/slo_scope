from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Optional

from sloscope.config import MechanismConfig
from sloscope.injectors.base import DegradationInjector, VerificationEvidence
from sloscope.lifecycle import Clock


def _json_request(method: str, url: str, timeout: float, payload: Optional[dict] = None) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return int(resp.status), json.loads(body) if body else {}


class DownstreamLatencyInjector(DegradationInjector):
    def __init__(self, config: MechanismConfig, clock: Clock) -> None:
        self.config = config
        self.mechanism_id = config.mechanism_id
        self.mechanism_type = config.mechanism_type
        self.intensity = config.intensity
        self.target = config.target
        self.clock = clock
        self.dependency_base_url = str(config.parameters.get("dependency_base_url", "http://127.0.0.1:8091")).rstrip("/")
        self.delay_ms = int(config.parameters.get("delay_ms", config.intensity))
        self.timeout = float(config.parameters.get("control_timeout_seconds", 2.0))
        self.verify_request = bool(config.parameters.get("verify_request", True))
        self.verify_min_fraction = float(config.parameters.get("verify_min_fraction", 0.8))
        self.activation_time: Optional[float] = None
        self.verification_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.cleanup_time: Optional[float] = None
        self.started = False
        self.stopped = False
        self.cleaned = False
        self.last_evidence: Optional[VerificationEvidence] = None
        self._cleanup_evidence: dict = {}

    def prepare(self) -> None:
        if self.mechanism_type != "downstream_latency":
            raise ValueError("DownstreamLatencyInjector requires mechanism_type=downstream_latency")
        if self.target != "synthetic_dependency":
            raise ValueError("DownstreamLatencyInjector target must be synthetic_dependency")
        if self.delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        status = self._status()
        if status.get("status") != "ok":
            raise RuntimeError("dependency service unavailable")

    def _status(self) -> dict:
        status, payload = _json_request("GET", f"{self.dependency_base_url}/control/status", self.timeout)
        if status >= 400 or not isinstance(payload, dict):
            raise RuntimeError(f"dependency status failed: {status}")
        return payload

    def _set_delay(self, delay_ms: int) -> dict:
        status, payload = _json_request("POST", f"{self.dependency_base_url}/control/delay", self.timeout, {"delay_ms": int(delay_ms)})
        if status >= 400 or not isinstance(payload, dict):
            raise RuntimeError(f"dependency delay configuration failed: {status}")
        return payload

    def start(self) -> None:
        observed = self._set_delay(self.delay_ms)
        if int(observed.get("delay_ms", -1)) != self.delay_ms:
            raise RuntimeError("dependency did not accept configured delay")
        self.activation_time = self.clock.monotonic()
        self.started = True

    def verify(self) -> VerificationEvidence:
        observed = self._status()
        verification_start = self.clock.monotonic()
        observed_request_duration = None
        if self.verify_request:
            start = time.monotonic()
            _json_request("GET", f"{self.dependency_base_url}/dependency", self.timeout + self.delay_ms / 1000.0 + 1.0)
            observed_request_duration = time.monotonic() - start
        self.verification_time = self.clock.monotonic()
        expected_seconds = self.delay_ms / 1000.0
        request_ok = observed_request_duration is None or observed_request_duration >= expected_seconds * self.verify_min_fraction
        state_ok = int(observed.get("delay_ms", -1)) == self.delay_ms and observed.get("status") == "ok"
        verified = self.started and state_ok and request_ok
        evidence = VerificationEvidence(
            verified=verified,
            evidence_type="downstream-latency-control-state",
            evidence_value={
                "configured_delay_ms": self.delay_ms,
                "observed_control_state": observed,
                "activation_time": self.activation_time,
                "verification_time": self.verification_time,
                "verification_start_time": verification_start,
                "dependency_service_id": observed.get("service_id"),
                "dependency_port": observed.get("port"),
                "observed_dependency_request_duration": observed_request_duration,
                "verify_minimum_duration": expected_seconds * self.verify_min_fraction if self.verify_request else None,
                "verified": verified,
            },
            timestamp=self.verification_time,
        )
        self.last_evidence = evidence
        return evidence

    def stop(self) -> None:
        observed = self._set_delay(0)
        self.stop_time = self.clock.monotonic()
        self.stopped = True
        if int(observed.get("delay_ms", -1)) != 0:
            raise RuntimeError("dependency stop did not restore zero delay")

    def cleanup(self) -> None:
        observed = self._set_delay(0)
        self.cleanup_time = self.clock.monotonic()
        self.cleaned = int(observed.get("delay_ms", -1)) == 0
        self._cleanup_evidence = {
            "stop_time": self.stop_time,
            "cleanup_time": self.cleanup_time,
            "observed_control_state": observed,
            "cleaned": self.cleaned,
        }
        if not self.cleaned:
            raise RuntimeError("dependency cleanup did not restore zero delay")

    def active_failure(self) -> Optional[str]:
        try:
            status = self._status()
        except Exception as exc:
            return f"dependency service unavailable during active interval: {exc}"
        if int(status.get("delay_ms", -1)) != self.delay_ms:
            return f"dependency delay changed during active interval: expected {self.delay_ms}, got {status.get('delay_ms')}"
        return None

    def cleanup_evidence(self) -> dict:
        return dict(self._cleanup_evidence)
