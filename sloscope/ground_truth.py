from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .config import MechanismConfig
from .injectors.base import VerificationEvidence


@dataclass
class MechanismTruth:
    mechanism_id: str
    mechanism_type: str
    intensity: float
    target: str
    scheduled_onset: float
    scheduled_stop: float
    actual_onset: Optional[float] = None
    verification_time: Optional[float] = None
    actual_stop: Optional[float] = None
    verification_state: str = "PENDING"
    verification_evidence: Optional[Dict[str, Any]] = None
    cleaned_up: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class GroundTruthLedger:
    def __init__(self, mechanisms: List[MechanismConfig]) -> None:
        self.records = {
            mech.mechanism_id: MechanismTruth(
                mechanism_id=mech.mechanism_id,
                mechanism_type=mech.mechanism_type,
                intensity=mech.intensity,
                target=mech.target,
                scheduled_onset=mech.scheduled_onset,
                scheduled_stop=mech.scheduled_stop,
            )
            for mech in mechanisms
        }

    def mark_verified(self, mechanism_id: str, evidence: VerificationEvidence) -> None:
        rec = self.records[mechanism_id]
        rec.verification_evidence = evidence.to_dict()
        rec.verification_time = evidence.evidence_value.get("verification_time", evidence.timestamp)
        if evidence.verified:
            rec.verification_state = "ACTIVE"
            rec.actual_onset = evidence.evidence_value.get("activation_time", evidence.timestamp)
        else:
            rec.verification_state = "VERIFICATION_FAILED"

    def mark_stopped(self, mechanism_id: str, timestamp: float) -> None:
        rec = self.records[mechanism_id]
        rec.actual_stop = timestamp
        if rec.verification_state == "ACTIVE":
            rec.verification_state = "STOPPED"

    def mark_cleaned(self, mechanism_id: str) -> None:
        self.records[mechanism_id].cleaned_up = True

    def to_dict(self) -> dict:
        return {
            "run_condition": "HEALTHY_NO_DEGRADATION" if not self.records else "DEGRADATION_INJECTED",
            "mechanisms": [record.to_dict() for record in self.records.values()],
        }
