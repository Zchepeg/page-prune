from __future__ import annotations

import re

from collector.schemas import Candidate

_TOKEN_LEFT = r"(^|[-_\s])"
_TOKEN_RIGHT = r"([-_\s]|$)"
_AD_WORD = re.compile(
    _TOKEN_LEFT + r"(ad|ads|advert|advertisement|sponsor(ed)?|promoted|推广|广告)" + _TOKEN_RIGHT,
    re.I,
)
_AD_SLOT_WORD = re.compile(
    _TOKEN_LEFT + r"(dfp|adsbygoogle|ad[-_]?slot|ad[-_]?unit|ad[-_]?to[-_]?list)" + _TOKEN_RIGHT,
    re.I,
)
_SAFE_WORD = re.compile(
    _TOKEN_LEFT + r"(article|content|navigation|checkout|login|comment|member(ship)?|community|newsletter|正文|登录|支付)" + _TOKEN_RIGHT,
    re.I,
)
_COMMON_AD_SIZES = {(300, 250), (336, 280), (728, 90), (970, 250), (320, 50)}


def _is_common_ad_size(width: float, height: float) -> bool:
    """Allow small layout/zoom differences around standard creative sizes."""
    return any(
        abs(width - expected_width) <= max(8, expected_width * 0.05)
        and abs(height - expected_height) <= max(6, expected_height * 0.08)
        for expected_width, expected_height in _COMMON_AD_SIZES
    )


def dom_ad_score(candidate: Candidate) -> tuple[float, list[str]]:
    identity = " ".join([candidate.element_id, *candidate.classes])
    attributes = " ".join(
        candidate.attributes.get(key, "") for key in ("aria_label", "title", "name", "role")
    )
    text = candidate.text[:160]
    score = 0.0
    evidence: list[str] = []
    identity_marker_count = sum(
        1 for value in (candidate.element_id, *candidate.classes) if _AD_WORD.search(value)
    )
    has_identity_marker = bool(_AD_WORD.search(identity))
    has_attribute_marker = bool(_AD_WORD.search(attributes))
    has_text_marker = bool(_AD_WORD.search(text))
    has_ad_marker = has_identity_marker or has_attribute_marker or has_text_marker
    has_ad_slot_marker = bool(_AD_SLOT_WORD.search(identity))
    if has_ad_slot_marker:
        score = max(score, 0.85)
        evidence.append("known_ad_slot_marker")
    if has_identity_marker:
        score += 0.60
        evidence.append("ad_semantic_marker")
    elif has_text_marker:
        score += 0.40
        evidence.append("ad_text_marker")
    if has_attribute_marker:
        score += 0.30
        evidence.append("explicit_ad_attribute")
    if has_ad_marker and candidate.tag in {"IFRAME", "INS"}:
        score += 0.25
        evidence.append("ad_container_element")
    if not has_ad_marker and _SAFE_WORD.search(identity):
        score -= 0.45
        evidence.append("content_semantic_marker")
    if _is_common_ad_size(candidate.width, candidate.height):
        score += 0.25
        evidence.append("common_ad_dimensions")
    if candidate.fixed and candidate.width * candidate.height > 20_000:
        score += 0.15
        evidence.append("large_fixed_overlay")
    if has_identity_marker and candidate.fixed and candidate.width * candidate.height > 20_000:
        score = max(score, 1.0)
        evidence.append("compound_ad_overlay")
    if identity_marker_count >= 2:
        score = max(score, 1.0)
        evidence.append("multiple_ad_identity_markers")
    if candidate.tag in {"ARTICLE", "MAIN", "NAV", "HEADER", "FORM", "INPUT", "BUTTON"}:
        score -= 0.5
        evidence.append("protected_element_type")
    return max(0.0, min(1.0, score)), evidence
