from __future__ import annotations

from typing import List

from sloscope.config import MechanismConfig
from sloscope.injectors.base import DegradationInjector, VerificationEvidence
from sloscope.lifecycle import Clock


class MockInjector(DegradationInjector):
    def __init__(self, config: MechanismConfig, clock: Clock) -> None:
        self.config = config
        self.mechanism_id = config.mechanism_id
        self.mechanism_type = config.mechanism_type
        self.intensity = config.intensity
        self.target = config.target
        self.clock = clock
        self.started = False
        self.cleaned = False
        self.stopped = False
        self.activation_time = None
        self.history: List[str] = []

    def prepare(self) -> None:
        self.history.append("prepare")

    def start(self) -> None:
        if self.config.parameters.get("raise_on_start"):
            raise RuntimeError(f"mock injector start failure: {self.mechanism_id}")
        self.started = True
        self.activation_time = self.clock.monotonic()
        self.history.append("start")

    def verify(self) -> VerificationEvidence:
        self.history.append("verify")
        verified = self.started and not self.config.parameters.get("verification_fails", False)
        reason = "activated" if verified else ("not-started" if not self.started else "configured-verification-failure")
        verification_time = self.clock.monotonic()
        return VerificationEvidence(
            verified=verified,
            evidence_type="mock-state",
            evidence_value={"started": self.started, "reason": reason, "history": list(self.history), "activation_time": self.activation_time, "verification_time": verification_time},
            timestamp=verification_time,
        )

    def stop(self) -> None:
        if self.config.parameters.get("raise_on_stop"):
            raise RuntimeError(f"mock injector stop failure: {self.mechanism_id}")
        self.stopped = True
        self.history.append("stop")

    def cleanup(self) -> None:
        if self.config.parameters.get("raise_on_cleanup"):
            raise RuntimeError(f"mock injector cleanup failure: {self.mechanism_id}")
        self.cleaned = True
        self.history.append("cleanup")
