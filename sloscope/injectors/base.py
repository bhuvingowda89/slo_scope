from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class VerificationEvidence:
    verified: bool
    evidence_type: str
    evidence_value: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


class DegradationInjector(ABC):
    mechanism_id: str
    mechanism_type: str
    intensity: float
    target: str

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def verify(self) -> VerificationEvidence: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...
