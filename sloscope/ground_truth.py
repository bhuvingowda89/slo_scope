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
    magnitude_value: Optional[float]
    magnitude_unit: Optional[str]
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
        def magnitude(mech: MechanismConfig) -> tuple[float, str]:
            if mech.mechanism_type == "cpu_contention":
                return float(mech.intensity), "logical_cpu_worker_fraction"
            if mech.mechanism_type == "downstream_latency":
                return float(mech.parameters.get("delay_ms", mech.intensity)), "ms"
            return float(mech.intensity), "intensity"

        self.records = {
            mech.mechanism_id: MechanismTruth(
                mechanism_id=mech.mechanism_id,
                mechanism_type=mech.mechanism_type,
                intensity=mech.intensity,
                magnitude_value=magnitude(mech)[0],
                magnitude_unit=magnitude(mech)[1],
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


def active_mechanism_set(ground_truth: Dict[str, Any]) -> set[str]:
    mechanisms = ground_truth.get("mechanisms", []) if isinstance(ground_truth, dict) else []
    return {
        str(mech.get("mechanism_type"))
        for mech in mechanisms
        if isinstance(mech, dict)
        and mech.get("verification_state") == "STOPPED"
        and mech.get("verification_evidence", {}).get("verified") is True
    }
