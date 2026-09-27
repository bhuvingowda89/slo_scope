from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from sloscope.telemetry.schemas import RequestRecord
from sloscope.workload import RequestPlan


class RuntimeAdapter(ABC):
    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def healthcheck(self) -> bool: ...

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def execute(self, request: RequestPlan, actual_arrival: float) -> RequestRecord: ...

    @abstractmethod
    def collect_runtime_metrics(self, timestamp: float) -> Dict[str, Any]: ...

    @abstractmethod
    def shutdown(self) -> None: ...
