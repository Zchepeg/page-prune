from __future__ import annotations

from dataclasses import dataclass

from collector.schemas import Candidate
from .dom_heuristics import dom_ad_score
from .url_filter import assess_url


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    score: float
    evidence: list[str]
    route: str  # block, allow, or jev


class CandidateRanker:
    def __init__(self, block_score: float = 0.85, allow_score: float = 0.12):
        self.block_score = block_score
        self.allow_score = allow_score

    def assess(self, candidate: Candidate) -> CandidateAssessment:
        dom_score, evidence = dom_ad_score(candidate)
        ownership_known = candidate.resource_ownership_known or bool(candidate.direct_resource_urls)
        direct_urls = candidate.direct_resource_urls if ownership_known else candidate.resource_urls
        direct_set = set(direct_urls)
        direct_risks = [assess_url(url) for url in direct_urls]
        inherited_risks = [assess_url(url) for url in candidate.resource_urls if url not in direct_set]
        for risk in direct_risks:
            evidence.extend(risk.evidence)
        for risk in inherited_risks:
            evidence.extend(f"descendant_{item}" for item in risk.evidence)
        direct_ad = max((risk.ad_score for risk in direct_risks), default=0.0)
        direct_tracker = max((risk.tracker_score for risk in direct_risks), default=0.0)
        inherited_ad = max((risk.ad_score for risk in inherited_risks), default=0.0)
        tracker_element = candidate.tag in {"SCRIPT", "IFRAME"} or (
            candidate.width <= 2 and candidate.height <= 2
        )
        tracker_score = direct_tracker if tracker_element else direct_tracker * 0.35
        inherited_score = inherited_ad * 0.35
        score = max(
            dom_score,
            direct_ad,
            tracker_score,
            min(1.0, dom_score + direct_ad * 0.75),
            min(1.0, dom_score + inherited_score),
        )
        if not candidate.visible:
            evidence.append("not_visible")
            if score < self.block_score:
                return CandidateAssessment(score, sorted(set(evidence)), "allow")
        route = "block" if score >= self.block_score else "allow" if score <= self.allow_score else "jev"
        return CandidateAssessment(score, sorted(set(evidence)), route)
