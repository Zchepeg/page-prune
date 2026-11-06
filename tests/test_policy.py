from collector.schemas import Candidate, PageContext
from jev.types import Decision, DecisionLevel, Verdict
from policy.blocking_policy import Action, BlockingPolicy


def decision(probability: float, level: DecisionLevel) -> Decision:
    return Decision(
        candidate_id="c1",
        verdict=Verdict.BLOCK,
        probabilities={"block": probability, "allow": 1 - probability, "uncertain": 0},
        level=level,
    )


def test_l0_high_confidence_is_only_hidden_because_it_is_uncalibrated():
    item = Candidate("c1", PageContext("https://example.test"), tag="DIV")
    result = BlockingPolicy().apply(item, decision(0.95, DecisionLevel.L0))
    assert result.action is Action.HIDE


def test_l2_high_confidence_is_blocked():
    item = Candidate("c1", PageContext("https://example.test"), tag="DIV")
    result = BlockingPolicy().apply(item, decision(0.95, DecisionLevel.L2))
    assert result.action is Action.BLOCK


def test_protected_element_is_never_blocked():
    item = Candidate("c1", PageContext("https://example.test"), tag="FORM")
    result = BlockingPolicy().apply(item, decision(0.99, DecisionLevel.L2))
    assert result.action is Action.ALLOW


def test_header_is_never_blocked():
    item = Candidate("c1", PageContext("https://example.test"), tag="HEADER")
    result = BlockingPolicy().apply(item, decision(0.99, DecisionLevel.L2))
    assert result.action is Action.ALLOW
