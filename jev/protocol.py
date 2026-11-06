from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from detector.candidate_ranker import CandidateAssessment

from .types import Decision, Verdict


PROTOCOL_VERSION = "jev-ad-decision-v1"


class ProtocolRoute(StrEnum):
    RULE = "rule"
    MODEL = "model"
    CHALLENGE = "challenge"


@dataclass(frozen=True, slots=True)
class DecisionPlan:
    route: ProtocolRoute
    rule_verdict: Verdict | None
    reason: str


@dataclass(frozen=True, slots=True)
class ProtocolResolution:
    decision: Decision
    source: str
    disagreement: bool
    disagreement_kind: str


class DecisionProtocol:
    def __init__(
        self,
        challenge_block_below: float = 0.98,
        challenge_allow_above: float = 0.02,
        version: str = PROTOCOL_VERSION,
        challenge_rule_blocks: bool = True,
    ):
        self.challenge_block_below = challenge_block_below
        self.challenge_allow_above = challenge_allow_above
        self.version = version
        self.challenge_rule_blocks = challenge_rule_blocks

    def plan(self, assessment: CandidateAssessment) -> DecisionPlan:
        if assessment.route == "jev":
            return DecisionPlan(ProtocolRoute.MODEL, None, "ambiguous_rule_score")
        if assessment.route == "block":
            route = (
                ProtocolRoute.CHALLENGE
                if self.challenge_rule_blocks and assessment.score < self.challenge_block_below
                else ProtocolRoute.RULE
            )
            reason = "challenge_rule_block" if route is ProtocolRoute.CHALLENGE else "high_confidence_rule_block"
            return DecisionPlan(route, Verdict.BLOCK, reason)
        route = ProtocolRoute.CHALLENGE if assessment.score > self.challenge_allow_above else ProtocolRoute.RULE
        reason = "challenge_rule_allow" if route is ProtocolRoute.CHALLENGE else "high_confidence_rule_allow"
        return DecisionPlan(route, Verdict.ALLOW, reason)

    def resolve(self, plan: DecisionPlan, model_decision: Decision) -> ProtocolResolution:
        diagnostics = {
            **model_decision.diagnostics,
            "protocol_version": self.version,
            "protocol_route": plan.route.value,
            "protocol_reason": plan.reason,
            "rule_verdict": plan.rule_verdict.value if plan.rule_verdict else "",
        }
        if plan.route is ProtocolRoute.MODEL or plan.rule_verdict is None:
            return ProtocolResolution(replace(model_decision, diagnostics=diagnostics), "jev", False, "")
        if model_decision.verdict is plan.rule_verdict:
            diagnostics["resolution"] = "rule_model_consensus"
            return ProtocolResolution(replace(model_decision, diagnostics=diagnostics), "consensus", False, "")
        kind = "model_uncertain" if model_decision.verdict is Verdict.UNCERTAIN else "opposite_verdict"
        diagnostics["resolution"] = "review_disagreement"
        diagnostics["model_verdict"] = model_decision.verdict.value
        decision = replace(
            model_decision,
            verdict=Verdict.UNCERTAIN,
            diagnostics=diagnostics,
        )
        return ProtocolResolution(decision, "disagreement", True, kind)
