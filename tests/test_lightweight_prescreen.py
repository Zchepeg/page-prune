import pytest

pytest.importorskip('sklearn')

from collector.schemas import Candidate, PageContext
from detector.candidate_ranker import CandidateAssessment
from jev.protocol import DecisionProtocol
from policy.blocking_policy import BlockingPolicy
from scripts.evaluate_lightweight_prescreen import eligible, select_threshold


def test_prescreen_threshold_excludes_calibration_ads_and_equal_scores():
    rows = [{'eligible': True, 'domain': 'a.test', 'label': 1, 'p_ad': .0001}, {'eligible': True, 'domain': 'b.test', 'label': 1, 'p_ad': .001}, {'eligible': True, 'domain': 'c.test', 'label': 0, 'p_ad': .002}]
    assert select_threshold(rows, min_samples=2, min_domains=2) == .001
    assert select_threshold(rows, min_samples=3, min_domains=2) is None


def test_prescreen_cannot_override_rules_challenges_or_protected_elements():
    candidate = Candidate('node', PageContext('', '', 'example.test'), tag='DIV')
    policy, protocol = BlockingPolicy(), DecisionProtocol(challenge_rule_blocks=False)
    assert not eligible(candidate, CandidateAssessment(1, [], 'block'), protocol, policy)
    assert not eligible(candidate, CandidateAssessment(.1, [], 'allow'), protocol, policy)
    assert eligible(candidate, CandidateAssessment(.5, [], 'jev'), protocol, policy)
    candidate.tag = 'NAV'
    assert not eligible(candidate, CandidateAssessment(.5, [], 'jev'), protocol, policy)
