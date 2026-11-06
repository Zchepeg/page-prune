from collector.schemas import Candidate, PageContext
from detector.candidate_ranker import CandidateAssessment, CandidateRanker
from feedback.active_learning import ActiveLearningSampler
from feedback.disagreement_store import DisagreementStore
from jev.backend import BackendResult
from jev.decider import JevDecider
from jev.protocol import DecisionProtocol, ProtocolRoute
from jev.types import Decision, DecisionLevel, Verdict
from policy.blocking_policy import BlockingPolicy
from service.orchestrator import AdBlockOrchestrator


def model_decision(verdict: Verdict, probabilities: dict[str, float]) -> Decision:
    return Decision("candidate", verdict, probabilities, DecisionLevel.L2, model_version="test")


def test_high_confidence_rule_uses_fast_path():
    plan = DecisionProtocol().plan(CandidateAssessment(1.0, ["known_ad_host"], "block"))
    assert plan.route is ProtocolRoute.RULE
    assert plan.rule_verdict is Verdict.BLOCK


def test_boundary_rule_is_challenged():
    plan = DecisionProtocol().plan(CandidateAssessment(0.85, ["ad_semantic_marker"], "block"))
    assert plan.route is ProtocolRoute.CHALLENGE


def test_unvalidated_model_cannot_downgrade_rule_block():
    plan = DecisionProtocol(challenge_rule_blocks=False).plan(
        CandidateAssessment(0.85, ["ad_semantic_marker"], "block")
    )
    assert plan.route is ProtocolRoute.RULE


def test_opposite_verdict_becomes_reviewable_uncertainty():
    protocol = DecisionProtocol()
    plan = protocol.plan(CandidateAssessment(0.85, ["ad_semantic_marker"], "block"))
    result = protocol.resolve(plan, model_decision(Verdict.ALLOW, {"block": 0.1, "allow": 0.8, "uncertain": 0.1}))
    assert result.disagreement
    assert result.decision.verdict is Verdict.UNCERTAIN
    assert sum(result.decision.probabilities.values()) == 1.0
    assert result.decision.diagnostics["resolution"] == "review_disagreement"


def test_active_learning_prioritizes_disagreement():
    decision = model_decision(Verdict.UNCERTAIN, {"block": 0.1, "allow": 0.1, "uncertain": 0.8})
    decision.diagnostics["resolution"] = "review_disagreement"
    priority = ActiveLearningSampler().evaluate(CandidateAssessment(0.85, [], "block"), decision)
    assert priority.score == 1.0
    assert priority.reason == "rule_model_disagreement"


class AllowBackend:
    def decide(self, state, question):
        return BackendResult(
            {"ad": 0.05, "non_ad": 0.9, "insufficient_evidence": 0.05},
            DecisionLevel.L2,
            "allow-test",
            {},
        )


class RejectBackend:
    def decide(self, state, question):
        raise AssertionError("model must not run for compound deterministic evidence")


def test_orchestrator_records_rule_model_disagreement(tmp_path):
    orchestrator = AdBlockOrchestrator(
        CandidateRanker(),
        JevDecider(AllowBackend()),
        BlockingPolicy(),
        disagreements=DisagreementStore(tmp_path / "disagreements.jsonl"),
    )
    candidate = Candidate(
        "boundary-ad",
        PageContext("https://news.test", top_level_domain="news.test"),
        classes=["ad-content"],
        width=300,
        height=250,
    )
    result = orchestrator.process(candidate)
    assert result["source"] == "disagreement"
    assert result["decision"]["verdict"] == "uncertain"
    assert (tmp_path / "disagreements.jsonl").read_text(encoding="utf-8")


def test_compound_overlay_evidence_bypasses_model_and_blocks():
    orchestrator = AdBlockOrchestrator(
        CandidateRanker(),
        JevDecider(RejectBackend()),
        BlockingPolicy(),
    )
    candidate = Candidate(
        "overlay-ad",
        PageContext("https://publisher.test", top_level_domain="publisher.test"),
        classes=["ad-overlay"],
        width=640,
        height=480,
        fixed=True,
    )
    result = orchestrator.process(candidate)
    assert result["source"] == "rules"
    assert result["decision"]["verdict"] == "block"
    assert result["command"]["action"] == "block"
