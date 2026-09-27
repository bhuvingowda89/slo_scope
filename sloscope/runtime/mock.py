from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Set

from sloscope.config import RuntimeConfig
from sloscope.lifecycle import Clock
from sloscope.runtime.base import RuntimeAdapter
from sloscope.telemetry.schemas import RequestRecord
from sloscope.workload import RequestPlan


class MockRuntimeAdapter(RuntimeAdapter):
    def __init__(self, config: RuntimeConfig, clock: Clock, seed: int, fail_sequences: Iterable[int] = ()) -> None:
        self.config = config
        self.clock = clock
        self.rng = random.Random(seed)
        self.fail_sequences: Set[int] = set(fail_sequences)
        self.prepared = False
        self.shutdown_called = False
        self.accepted = 0
        self.succeeded = 0
        self.failed = 0

    def prepare(self) -> None:
        if self.config.parameters.get("raise_on_prepare"):
            raise RuntimeError("mock prepare failure")
        self.prepared = True

    def healthcheck(self) -> bool:
        return self.prepared and not self.config.parameters.get("unhealthy", False)

    def warmup(self) -> None:
        if self.config.parameters.get("raise_on_warmup"):
            raise RuntimeError("mock warmup failure")

    def execute(self, request: RequestPlan, actual_arrival: float) -> RequestRecord:
        if self.config.parameters.get("raise_on_sequence") == request.sequence:
            raise RuntimeError(f"mock runtime exception at sequence {request.sequence}")
        self.accepted += 1
        dispatch = actual_arrival + 0.001
        prompt_tokens = 8 + (request.sequence % 5)
        if request.sequence in self.fail_sequences or request.sequence in set(self.config.parameters.get("fail_sequences", [])):
            self.failed += 1
            return RequestRecord(request.request_id, request.sequence, request.scheduled_arrival, actual_arrival, dispatch, None, dispatch + 0.002, prompt_tokens, 0, "failed", self.config.runtime_id, self.config.model_id)
        first = dispatch + 0.01 + (request.sequence % 3) * 0.001
        output_tokens = request.target_output_tokens
        completion = first + output_tokens * 0.002
        self.succeeded += 1
        return RequestRecord(request.request_id, request.sequence, request.scheduled_arrival, actual_arrival, dispatch, first, completion, prompt_tokens, output_tokens, "success", self.config.runtime_id, self.config.model_id)

    def collect_runtime_metrics(self, timestamp: float) -> Dict[str, Any]:
        return {
            "timestamp": timestamp,
            "runtime_id": self.config.runtime_id,
            "queue_depth": 0,
            "active_requests": 0,
            "deferred_requests": 0,
            "throughput": None,
            "counters": f"accepted_requests_total={self.accepted};successful_requests_total={self.succeeded};failed_requests_total={self.failed}",
        }

    def shutdown(self) -> None:
        if self.config.parameters.get("raise_on_shutdown"):
            raise RuntimeError("mock shutdown failure")
        self.shutdown_called = True
