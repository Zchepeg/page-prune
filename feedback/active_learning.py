from __future__ import annotations

from dataclasses import dataclass

from detector.candidate_ranker import CandidateAssessment
from jev.types import Decision


@dataclass(frozen=True, slots=True)
class SamplePriority:
    score: float
    reason: str
    components: dict[str, float]


class ActiveLearningSampler:
    def __init__(self, min_priority: float = 0.45, block_score: float = 0.85, allow_score: float = 0.12):
        self.min_priority = min_priority
        self.block_score = block_score
        self.allow_score = allow_score

    def evaluate(self, assessment: CandidateAssessment, decision: Decision | None) -> SamplePriority:
        disagreement = float(bool(decision and decision.diagnostics.get("resolution") == "review_disagreement"))
        probabilities = decision.probabilities if decision else {}
        ranked = sorted((float(value) for value in probabilities.values()), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
        uncertainty = max(float(probabilities.get("uncertain", 0.0)), 1.0 - margin) if probabilities else 0.0
        boundary_distance = min(abs(assessment.score - self.block_score), abs(assessment.score - self.allow_score))
        boundary = max(0.0, 1.0 - boundary_distance / 0.25)
        score = min(
            1.0,
            max(
                disagreement,
                uncertainty * 0.75,
                boundary * 0.55,
                uncertainty * 0.55 + boundary * 0.35,
            ),
        )
        if disagreement:
            reason = "rule_model_disagreement"
        elif uncertainty >= 0.5:
            reason = "model_uncertainty"
        elif boundary >= 0.5:
            reason = "rule_boundary"
        else:
            reason = "low_information_value"
        return SamplePriority(
            round(score, 6),
            reason,
            {
                "disagreement": disagreement,
                "uncertainty": round(uncertainty, 6),
                "rule_boundary": round(boundary, 6),
            },
        )

    def accepts(self, priority: SamplePriority) -> bool:
        return priority.score >= self.min_priority
