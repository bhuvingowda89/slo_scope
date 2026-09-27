from __future__ import annotations

import os
from typing import Any, Dict

from sloscope.lifecycle import Clock


class SystemTelemetryCollector:
    def __init__(self, clock: Clock, server_pid: int | None = None) -> None:
        self.clock = clock
        try:
            import psutil
        except ModuleNotFoundError as exc:
            raise RuntimeError("psutil is required for Phase 3B system telemetry") from exc
        self.psutil = psutil
        self.client_process = psutil.Process(os.getpid())
        self.server_pid = server_pid
        self.server_process = psutil.Process(server_pid) if server_pid is not None else None
        self.client_process.cpu_percent(None)
        if self.server_process is not None:
            self.server_process.cpu_percent(None)
        psutil.cpu_percent(None)

    def sample(self) -> Dict[str, Any]:
        vm = self.psutil.virtual_memory()
        return {
            "timestamp": self.clock.monotonic(),
            "host_cpu": None,
            "process_cpu": None,
            "memory": None,
            "memory_pressure": None,
            "disk_io": None,
            "network": None,
            "process_rss": None,
            "memory_used": None,
            "memory_available": None,
            "host_cpu_percent": float(self.psutil.cpu_percent(None)),
            "host_memory_percent": float(vm.percent),
            "host_memory_used_bytes": float(vm.used),
            "host_memory_available_bytes": float(vm.available),
            "client_process_cpu_percent": float(self.client_process.cpu_percent(None)),
            "client_process_rss_bytes": float(self.client_process.memory_info().rss),
            "server_pid": self.server_pid,
            "server_process_cpu_percent": None if self.server_process is None else float(self.server_process.cpu_percent(None)),
            "server_process_rss_bytes": None if self.server_process is None else float(self.server_process.memory_info().rss),
        }
