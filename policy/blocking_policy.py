from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from collector.schemas import Candidate
from jev.types import Decision, DecisionLevel, Verdict


class Action(StrEnum):
    BLOCK = "block"
    HIDE = "hide"
    ALLOW = "allow"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class PolicyResult:
    action: Action
    reason: str


class BlockingPolicy:
    def __init__(
        self,
        block_threshold: float = 0.90,
        hide_threshold: float = 0.72,
        min_decision_level: DecisionLevel = DecisionLevel.L1,
        protected_tags: set[str] | None = None,
        fail_open: bool = True,
    ):
        self.block_threshold = block_threshold
        self.hide_threshold = hide_threshold
        self.min_decision_level = min_decision_level
        self.protected_tags = protected_tags or {"ARTICLE", "MAIN", "NAV", "HEADER", "FORM", "INPUT", "BUTTON"}
        self.fail_open = fail_open

    def apply(self, candidate: Candidate, decision: Decision) -> PolicyResult:
        if candidate.tag in self.protected_tags:
            return PolicyResult(Action.ALLOW, "protected_element")
        if decision.verdict is Verdict.ALLOW:
            return PolicyResult(Action.ALLOW, "decision_allow")
        if decision.verdict is Verdict.UNCERTAIN:
            return PolicyResult(Action.ALLOW if self.fail_open else Action.REVIEW, "decision_uncertain")
        probability = decision.probabilities.get("block", 0.0)
        if decision.level is DecisionLevel.RULE and probability >= self.block_threshold:
            return PolicyResult(Action.BLOCK, "deterministic_rule")
        if decision.level >= self.min_decision_level and probability >= self.block_threshold:
            return PolicyResult(Action.BLOCK, "calibrated_high_confidence")
        if probability >= self.hide_threshold:
            return PolicyResult(Action.HIDE, "recoverable_medium_confidence")
        return PolicyResult(Action.ALLOW if self.fail_open else Action.REVIEW, "below_policy_threshold")
