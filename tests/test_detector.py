from collector.schemas import Candidate, PageContext
from detector.candidate_ranker import CandidateRanker


def candidate(**overrides):
    values = {
        "candidate_id": "c1",
        "page": PageContext("https://news.test", top_level_domain="news.test"),
        "tag": "DIV",
        "visible": True,
    }
    values.update(overrides)
    return Candidate(**values)


def test_known_ad_host_takes_fast_block_path():
    item = candidate(resource_urls=["https://ads.doubleclick.net/banner"])
    result = CandidateRanker().assess(item)
    assert result.route == "block"
    assert "known_ad_host" in result.evidence


def test_article_is_allowed_by_fast_path():
    item = candidate(tag="ARTICLE", classes=["article-content"])
    result = CandidateRanker().assess(item)
    assert result.route == "allow"
    assert "protected_element_type" in result.evidence


def test_ambiguous_banner_goes_to_jev():
    item = candidate(width=300, height=250)
    result = CandidateRanker().assess(item)
    assert result.route == "jev"


def test_semantic_marker_and_standard_size_take_fast_block_path():
    item = candidate(classes=["ad-content"], width=300, height=250)
    result = CandidateRanker().assess(item)
    assert result.route == "block"
    assert result.score == 0.85
    assert "content_semantic_marker" not in result.evidence


def test_semantic_marker_and_ad_url_take_fast_block_path():
    item = candidate(classes=["sponsored"], resource_urls=["https://cdn.test/ads/banner.js"])
    result = CandidateRanker().assess(item)
    assert result.route == "block"


def test_hidden_known_ad_host_remains_a_deterministic_block():
    item = candidate(visible=False, resource_urls=["https://securepubads.g.doubleclick.net/ad"])
    result = CandidateRanker().assess(item)
    assert result.route == "block"
    assert "not_visible" in result.evidence


def test_explicit_ad_aria_and_iframe_are_independent_signals():
    item = candidate(tag="IFRAME", attributes={"aria_label": "Advertisement"})
    result = CandidateRanker(block_score=0.55).assess(item)
    assert result.route == "block"
    assert "explicit_ad_attribute" in result.evidence
    assert "ad_container_element" in result.evidence


def test_near_standard_size_tolerates_layout_rounding():
    item = candidate(classes=["ad-banner"], width=298, height=248)
    result = CandidateRanker().assess(item)
    assert result.route == "block"
    assert "common_ad_dimensions" in result.evidence


def test_curated_analytics_host_takes_fast_block_path():
    item = candidate(tag="SCRIPT", resource_urls=["https://www.clarity.ms/tag/example"])
    result = CandidateRanker().assess(item)
    assert result.route == "block"
    assert "known_tracker_host" in result.evidence


def test_descendant_tracker_does_not_block_generic_container():
    url = "https://www.clarity.ms/tag/example"
    direct = "https://news.test/card.jpg"
    item = candidate(
        resource_urls=[direct, url],
        direct_resource_urls=[direct],
        resource_ownership_known=True,
    )
    result = CandidateRanker().assess(item)
    assert result.route == "allow"
    assert "descendant_known_tracker_host" in result.evidence


def test_container_with_only_descendant_tracker_is_not_treated_as_direct():
    url = "https://www.clarity.ms/tag/example"
    item = candidate(
        resource_urls=[url],
        direct_resource_urls=[],
        resource_ownership_known=True,
    )
    result = CandidateRanker().assess(item)
    assert result.route == "allow"
    assert "descendant_known_tracker_host" in result.evidence


def test_direct_tracker_on_regular_image_requires_review():
    url = "https://www.clarity.ms/tag/example"
    item = candidate(
        tag="IMG",
        width=640,
        height=360,
        resource_urls=[url],
        direct_resource_urls=[url],
        resource_ownership_known=True,
    )
    result = CandidateRanker().assess(item)
    assert result.route == "jev"


def test_direct_tracking_pixel_takes_fast_block_path():
    url = "https://www.clarity.ms/tag/example"
    item = candidate(
        tag="IMG",
        width=1,
        height=1,
        resource_urls=[url],
        direct_resource_urls=[url],
        resource_ownership_known=True,
    )
    result = CandidateRanker().assess(item)
    assert result.route == "block"


def test_semantic_marker_and_fixed_overlay_take_high_confidence_rule_path():
    item = candidate(classes=["ad-overlay"], width=640, height=480, fixed=True)
    result = CandidateRanker().assess(item)
    assert result.route == "block"
    assert result.score == 1.0
    assert "compound_ad_overlay" in result.evidence


def test_multiple_ad_identity_markers_block_hidden_container():
    item = candidate(classes=["interstitial-ad", "fullscreen-ad", "modal-ad"], visible=False)
    result = CandidateRanker().assess(item)
    assert result.route == "block"
    assert result.score == 1.0
    assert "multiple_ad_identity_markers" in result.evidence


def test_large_fixed_dialog_without_ad_marker_is_not_fast_blocked():
    item = candidate(classes=["dialog", "modal"], width=640, height=480, fixed=True)
    result = CandidateRanker().assess(item)
    assert result.route != "block"


def test_single_hidden_ad_marker_remains_fail_open():
    item = candidate(classes=["ad-content"], visible=False)
    result = CandidateRanker().assess(item)
    assert result.route == "allow"


def test_known_dfp_slot_marker_takes_fast_block_path():
    item = candidate(classes=["dfp-leaderboard-container"], width=980, height=250)
    result = CandidateRanker().assess(item)
    assert result.route == "block"
    assert "known_ad_slot_marker" in result.evidence


def test_known_ad_to_list_marker_takes_fast_block_path():
    item = candidate(classes=["ad_to_list"], width=768, height=118)
    result = CandidateRanker().assess(item)
    assert result.route == "block"
    assert "known_ad_slot_marker" in result.evidence


def test_membership_banner_is_allowed_by_fast_path():
    item = candidate(classes=["custom-widget-membership", "skinny-banner"], width=1265, height=50, fixed=True)
    result = CandidateRanker().assess(item)
    assert result.route == "allow"
    assert "content_semantic_marker" in result.evidence


def test_navigation_and_header_are_protected_element_types():
    for tag in ("NAV", "HEADER"):
        result = CandidateRanker().assess(candidate(tag=tag, width=1265, height=98))
        assert result.route == "allow"
        assert "protected_element_type" in result.evidence
