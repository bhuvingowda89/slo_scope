from __future__ import annotations

from sloscope.config import MechanismConfig
from sloscope.injectors.cpu_contention import CPUContentionInjector
from sloscope.injectors.downstream_latency import DownstreamLatencyInjector
from sloscope.injectors.mock import MockInjector
from sloscope.lifecycle import Clock


def create_injector(config: MechanismConfig, clock: Clock):
    if config.mechanism_type == "mock-degradation":
        return MockInjector(config, clock)
    if config.mechanism_type == "cpu_contention":
        return CPUContentionInjector(config, clock)
    if config.mechanism_type == "downstream_latency":
        return DownstreamLatencyInjector(config, clock)
    raise ValueError(f"unsupported degradation mechanism_type: {config.mechanism_type}")
