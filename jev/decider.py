from __future__ import annotations

from .backend import DecisionBackend
from .questions import AD_TREATMENT, TypedQuestion
from .types import Decision, Verdict


class JevDecider:
    def __init__(
        self,
        backend: DecisionBackend,
        uncertainty_margin: float = 0.08,
        min_confidence: float = 0.0,
    ):
        self.backend = backend
        self.uncertainty_margin = uncertainty_margin
        self.min_confidence = min_confidence

    def runtime_status(self) -> dict[str, object]:
        status = getattr(self.backend, "runtime_status", None)
        return status() if callable(status) else {"backend": type(self.backend).__name__, "loaded": True}

    def prepare_many(self, requests, question=AD_TREATMENT):
        prepare = getattr(self.backend, 'prepare', None)
        return prepare([state for _, state, _ in requests], question) if callable(prepare) else True

    def decide(
        self,
        candidate_id: str,
        state: str,
        evidence: list[str],
        question: TypedQuestion = AD_TREATMENT,
    ) -> Decision:
        return self._decision(candidate_id, evidence, question, self.backend.decide(state, question))

    def decide_many(
        self,
        requests: list[tuple[str, str, list[str]]],
        question: TypedQuestion = AD_TREATMENT,
    ) -> list[Decision]:
        states = [state for _, state, _ in requests]
        batch_decide = getattr(self.backend, "decide_many", None)
        results = (
            batch_decide(states, question)
            if callable(batch_decide)
            else [self.backend.decide(state, question) for state in states]
        )
        if len(results) != len(requests):
            raise ValueError("backend returned a different number of decisions")
        return [
            self._decision(candidate_id, evidence, question, result)
            for (candidate_id, _, evidence), result in zip(requests, results)
        ]

    def _decision(self, candidate_id, evidence, question, result) -> Decision:
        semantic_probs = result.probabilities
        probs = {
            "block": semantic_probs.get("ad", semantic_probs.get("block", 0.0)),
            "allow": semantic_probs.get("non_ad", semantic_probs.get("allow", 0.0)),
            "uncertain": semantic_probs.get("insufficient_evidence", semantic_probs.get("uncertain", 0.0)),
        }
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        verdict = Verdict(ranked[0][0])
        semantic = sorted((probs["block"], probs["allow"]), reverse=True)
        if verdict is not Verdict.UNCERTAIN and (
            ranked[0][1] < self.min_confidence
            or semantic[0] - semantic[1] < self.uncertainty_margin
        ):
            verdict = Verdict.UNCERTAIN
        return Decision(
            candidate_id=candidate_id,
            verdict=verdict,
            probabilities=probs,
            level=result.level,
            evidence=evidence,
            model_version=result.model_version,
            question_version=question.version,
            diagnostics={
                **result.diagnostics,
                "semantic_probabilities": semantic_probs,
                "selective_min_confidence": self.min_confidence,
                "selective_margin": self.uncertainty_margin,
            },
        )
