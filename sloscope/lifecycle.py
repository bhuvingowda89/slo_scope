from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...
    def wall_time(self) -> str: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def wall_time(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def wall_time(self) -> str:
        return f"fake-{self.now:.6f}"

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot move monotonic clock backwards")
        self.now += seconds


class ExperimentState(str, Enum):
    PLANNED = "PLANNED"
    PREPARING = "PREPARING"
    WARMUP = "WARMUP"
    BASELINE = "BASELINE"
    INJECTING = "INJECTING"
    RECOVERY = "RECOVERY"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


LEGAL_TRANSITIONS: Dict[ExperimentState, set[ExperimentState]] = {
    ExperimentState.PLANNED: {ExperimentState.PREPARING, ExperimentState.FAILED},
    ExperimentState.PREPARING: {ExperimentState.WARMUP, ExperimentState.FAILED},
    ExperimentState.WARMUP: {ExperimentState.BASELINE, ExperimentState.FAILED},
    ExperimentState.BASELINE: {ExperimentState.INJECTING, ExperimentState.RECOVERY, ExperimentState.FAILED},
    ExperimentState.INJECTING: {ExperimentState.RECOVERY, ExperimentState.FAILED},
    ExperimentState.RECOVERY: {ExperimentState.COMPLETE, ExperimentState.FAILED},
    ExperimentState.COMPLETE: set(),
    ExperimentState.FAILED: set(),
}


@dataclass(frozen=True)
class ExperimentEvent:
    event_type: str
    state: str
    monotonic_time: float
    wall_time: str
    message: str = ""
    details: Optional[dict] = None


class IllegalTransitionError(ValueError):
    pass


class Lifecycle:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.state = ExperimentState.PLANNED
        self.events: List[ExperimentEvent] = [
            ExperimentEvent("state", self.state.value, clock.monotonic(), clock.wall_time(), "initial")
        ]

    def transition(self, target: ExperimentState, message: str = "") -> None:
        if target not in LEGAL_TRANSITIONS[self.state]:
            raise IllegalTransitionError(f"illegal transition {self.state.value} -> {target.value}")
        self.state = target
        self.events.append(ExperimentEvent("state", target.value, self.clock.monotonic(), self.clock.wall_time(), message))

    def event(self, event_type: str, message: str = "", details: Optional[dict] = None) -> None:
        self.events.append(ExperimentEvent(event_type, self.state.value, self.clock.monotonic(), self.clock.wall_time(), message, details))


def validate_lifecycle_history(events: Iterable[dict]) -> list[str]:
    issues: list[str] = []
    state: Optional[ExperimentState] = None
    last_time: Optional[float] = None
    terminal_count = 0
    for event in events:
        current_time = event.get("monotonic_time")
        if isinstance(current_time, (int, float)):
            if last_time is not None and current_time < last_time:
                issues.append("event timestamps must be monotonic")
            last_time = float(current_time)
        if event.get("event_type") != "state":
            continue
        try:
            next_state = ExperimentState(event["state"])
        except Exception:
            issues.append(f"unknown lifecycle state {event.get('state')}")
            continue
        if state is None:
            if next_state != ExperimentState.PLANNED:
                issues.append("lifecycle history must start at PLANNED")
            state = next_state
            continue
        if next_state not in LEGAL_TRANSITIONS[state]:
            issues.append(f"illegal lifecycle transition {state.value} -> {next_state.value}")
        state = next_state
        if next_state in {ExperimentState.COMPLETE, ExperimentState.FAILED}:
            terminal_count += 1
    if terminal_count != 1:
        issues.append("lifecycle history must contain exactly one terminal state")
    return issues
