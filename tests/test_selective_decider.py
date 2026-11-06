from jev.backend import BackendResult
from jev.decider import JevDecider
from jev.types import DecisionLevel, Verdict


class Backend:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def decide(self, state, question):
        return BackendResult(self.probabilities, DecisionLevel.L2, "test", {})


def test_binary_model_abstains_below_confidence_threshold():
    decider = JevDecider(Backend({"ad": 0.53, "non_ad": 0.47}), min_confidence=0.6)
    assert decider.decide("id", "{}", []).verdict is Verdict.UNCERTAIN


def test_binary_model_returns_semantic_verdict_above_threshold():
    decider = JevDecider(Backend({"ad": 0.8, "non_ad": 0.2}), min_confidence=0.6)
    assert decider.decide("id", "{}", []).verdict is Verdict.BLOCK
