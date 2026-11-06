from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TypedQuestion:
    name: str
    text: str
    options: tuple[str, ...]
    version: str


AD_TREATMENT = TypedQuestion(
    name="ad_classification",
    text=(
        "Classify this candidate web element. Choose ad only for paid promotional content; "
        "choose non_ad for normal content and controls; choose insufficient_evidence when the state cannot decide."
    ),
    options=("ad", "non_ad", "insufficient_evidence"),
    version="ad-classification-v1",
)

PROBE_AD_TREATMENT = TypedQuestion(
    name="probe_ad_classification",
    text=(
        "Classify whether this observed web-element transition is advertising behavior. "
        "Choose ad only when the transition provides advertising evidence; choose non_ad for normal dynamic content; "
        "choose insufficient_evidence when the transition cannot decide."
    ),
    options=("ad", "non_ad", "insufficient_evidence"),
    version="probe-ad-classification-v1",
)
