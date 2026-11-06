from __future__ import annotations

from dataclasses import asdict, dataclass

from detector.url_filter import url_ad_score
from runtime.state_serialization import serialize_state

from .types import ProbeRecord


@dataclass(frozen=True, slots=True)
class ProbeAssessment:
    score: float
    information_value: float
    evidence: list[str]
    route: str

    def to_dict(self) -> dict:
        return asdict(self)


class ProbeAnalyzer:
    def __init__(self, review_threshold: float = 0.15, model_threshold: float = 0.30):
        self.review_threshold = review_threshold
        self.model_threshold = model_threshold

    def assess(self, record: ProbeRecord) -> ProbeAssessment:
        evidence: list[str] = []
        information = 0.0
        risk_signal = 0.0
        new_hosts = record.delta["new_resource_hosts"]
        if new_hosts:
            evidence.append("new_resource_host")
            information += 0.30
            risk_signal += 0.10
        for host in new_hosts:
            host_score, host_evidence = url_ad_score(f"https://{host}/")
            if host_score > 0:
                risk_signal = max(risk_signal, host_score * 0.50)
                evidence.extend(f"probe_{item}" for item in host_evidence)
        if record.delta["visibility_changed"] and record.after["visible"]:
            evidence.append("became_visible")
            information += 0.25
            risk_signal += 0.20
        if record.delta["viewport_changed"] and record.after["in_viewport"]:
            evidence.append("entered_viewport")
            information += 0.10
            risk_signal += 0.05
        if record.delta["children"] > 0:
            evidence.append("children_added")
            information += min(0.20, record.delta["children"] * 0.04)
            risk_signal += min(0.15, record.delta["children"] * 0.03)
        if record.delta["mutation_count"] >= 3:
            evidence.append("mutation_burst")
            information += min(0.20, record.delta["mutation_count"] * 0.02)
            risk_signal += min(0.10, record.delta["mutation_count"] * 0.01)
        if abs(record.delta["width"]) >= 20 or abs(record.delta["height"]) >= 20:
            evidence.append("layout_expansion")
            information += 0.15
            risk_signal += 0.10
        if record.after["fixed"] and record.after["visible"]:
            evidence.append("visible_fixed_element")
            information += 0.10
            risk_signal += 0.10
        information = min(1.0, information)
        score = min(1.0, record.detector_score * 0.55 + risk_signal)
        if information < self.review_threshold:
            route = "discard"
        elif information >= self.model_threshold:
            route = "model"
        else:
            route = "review"
        return ProbeAssessment(round(score, 6), round(information, 6), sorted(set(evidence)), route)

    @staticmethod
    def build_state(record: ProbeRecord, assessment: ProbeAssessment) -> str:
        payload = {
            "probe": {
                "type": record.probe_type,
                "domain": record.domain,
                "detector_score": record.detector_score,
                "original_action": record.original_action,
                "before": record.before,
                "after": record.after,
                "delta": record.delta,
                "evidence": assessment.evidence,
                "probe_score": assessment.score,
                "information_value": assessment.information_value,
            }
        }
        return serialize_state(payload)
