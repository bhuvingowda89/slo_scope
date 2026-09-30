from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil

from sloscope.config import MechanismConfig
from sloscope.injectors.base import DegradationInjector, VerificationEvidence
from sloscope.lifecycle import Clock


def cpu_worker(stop_event, do_work: bool = True) -> None:
    value = 1
    while not stop_event.is_set():
        if do_work:
            value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
            value ^= (value << 13) & 0xFFFFFFFF
        else:
            time.sleep(0.01)


def exiting_worker(stop_event) -> None:
    return None


def worker_count_for_intensity(logical_cpu_count: int, intensity: float) -> int:
    if intensity <= 0 or intensity > 1.0:
        raise ValueError("intensity must satisfy 0 < intensity <= 1.0")
    return max(1, round(logical_cpu_count * intensity))


class CPUContentionInjector(DegradationInjector):
    def __init__(self, config: MechanismConfig, clock: Clock) -> None:
        self.config = config
        self.mechanism_id = config.mechanism_id
        self.mechanism_type = config.mechanism_type
        self.intensity = config.intensity
        self.target = config.target
        self.clock = clock
        self.logical_cpu_count = int(config.parameters.get("logical_cpu_count") or (os.cpu_count() or 1))
        self.worker_count = int(config.parameters.get("worker_count") or worker_count_for_intensity(self.logical_cpu_count, self.intensity))
        self.verification_interval_seconds = float(config.parameters.get("verification_interval_seconds", 0.30))
        self.stop_timeout_seconds = float(config.parameters.get("stop_timeout_seconds", 2.0))
        self.do_work = bool(config.parameters.get("do_work", True))
        self.exit_immediately = bool(config.parameters.get("exit_immediately", False))
        self.stop_event = None
        self.processes: List[mp.Process] = []
        self.worker_pids: List[int] = []
        self.activation_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.cleanup_time: Optional[float] = None
        self.cleaned = False
        self.last_evidence: Optional[VerificationEvidence] = None
        self.exit_status: Dict[int, Optional[int]] = {}

    def prepare(self) -> None:
        if self.mechanism_type != "cpu_contention":
            raise ValueError("CPUContentionInjector requires mechanism_type=cpu_contention")
        if self.target != "host_cpu":
            raise ValueError("CPUContentionInjector target must be host_cpu")
        if not (0 < self.intensity <= 1.0):
            raise ValueError("CPUContentionInjector intensity must satisfy 0 < intensity <= 1.0")

    def start(self) -> None:
        self.stop_event = mp.Event()
        self.processes = []
        self.worker_pids = []
        for _ in range(self.worker_count):
            target = exiting_worker if self.exit_immediately else cpu_worker
            args = (self.stop_event,) if self.exit_immediately else (self.stop_event, self.do_work)
            proc = mp.Process(target=target, args=args)
            proc.start()
            self.processes.append(proc)
            if proc.pid is not None:
                self.worker_pids.append(proc.pid)
        if len(self.worker_pids) != self.worker_count:
            raise RuntimeError("failed to start all CPU contention workers")
        self.activation_time = self.clock.monotonic()

    def _cpu_times(self) -> Dict[int, float]:
        times: Dict[int, float] = {}
        for pid in self.worker_pids:
            try:
                proc = psutil.Process(pid)
                cpu = proc.cpu_times()
                times[pid] = float(cpu.user + cpu.system)
            except psutil.Error:
                times[pid] = 0.0
        return times

    def verify(self) -> VerificationEvidence:
        before = self._cpu_times()
        time.sleep(self.verification_interval_seconds)
        after = self._cpu_times()
        verification_time = self.clock.monotonic()
        deltas = {pid: max(0.0, after.get(pid, 0.0) - before.get(pid, 0.0)) for pid in self.worker_pids}
        all_alive = all(proc.is_alive() for proc in self.processes)
        total_delta = sum(deltas.values())
        verified = (not self.exit_immediately) and all_alive and len(self.worker_pids) == self.worker_count and total_delta > 0
        evidence = VerificationEvidence(
            verified=verified,
            evidence_type="cpu-contention-worker-cputime",
            evidence_value={
                "logical_cpu_count": self.logical_cpu_count,
                "configured_intensity": self.intensity,
                "actual_worker_count": len(self.worker_pids),
                "actual_worker_fraction": len(self.worker_pids) / self.logical_cpu_count,
                "worker_pids": list(self.worker_pids),
                "verification_interval_seconds": self.verification_interval_seconds,
                "per_worker_cpu_time_delta": deltas,
                "total_worker_cpu_time_delta": total_delta,
                "activation_time": self.activation_time,
                "verification_time": verification_time,
                "workers_alive": all_alive,
                "verified": verified,
            },
            timestamp=verification_time,
        )
        self.last_evidence = evidence
        return evidence

    def stop(self) -> None:
        self.stop_time = self.clock.monotonic()
        if self.stop_event is not None:
            self.stop_event.set()
        deadline = time.time() + self.stop_timeout_seconds
        for proc in self.processes:
            remaining = max(0.0, deadline - time.time())
            proc.join(timeout=remaining)
            self.exit_status[proc.pid or -1] = proc.exitcode

    def cleanup(self) -> None:
        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=self.stop_timeout_seconds)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=self.stop_timeout_seconds)
            self.exit_status[proc.pid or -1] = proc.exitcode
        self.cleanup_time = self.clock.monotonic()
        self.cleaned = all(not proc.is_alive() for proc in self.processes)
        if not self.cleaned:
            raise RuntimeError("CPU contention worker cleanup failed")

    def active_failure(self) -> Optional[str]:
        dead = [proc.pid for proc in self.processes if not proc.is_alive()]
        if dead:
            return f"CPU contention worker exited during active interval: {dead}"
        return None

    def cleanup_evidence(self) -> dict:
        return {
            "stop_time": self.stop_time,
            "cleanup_time": self.cleanup_time,
            "worker_exit_status": dict(self.exit_status),
            "worker_pids": list(self.worker_pids),
            "cleaned": self.cleaned,
        }
