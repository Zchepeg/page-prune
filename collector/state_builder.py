from __future__ import annotations

from typing import Any

from runtime.state_serialization import serialize_state

from .schemas import Candidate


STATE_SCHEMA_VERSION = "candidate-features-v2"
OWNERSHIP_SCHEMA_VERSION = "candidate-features-v3"


def build_state_payload(candidate: Candidate, evidence: list[str], schema_version: str = STATE_SCHEMA_VERSION) -> dict[str, Any]:
    if schema_version not in {STATE_SCHEMA_VERSION, OWNERSHIP_SCHEMA_VERSION}:
        raise ValueError("unsupported candidate state schema")
    payload = {
        "schema_version": schema_version,
        "detector_evidence": sorted(set(evidence)),
        "tag": candidate.tag,
        "element_id": candidate.element_id,
        "classes": candidate.classes,
        "attributes": dict(sorted(candidate.attributes.items())),
        "resource_urls": candidate.resource_urls,
        "geometry": [round(float(candidate.width), 1), round(float(candidate.height), 1)],
        "fixed": candidate.fixed,
        "visible": candidate.visible,
        "page_domain": candidate.page.top_level_domain,
        "text": candidate.text,
    }
    if schema_version == OWNERSHIP_SCHEMA_VERSION:
        payload["direct_resource_urls"] = candidate.direct_resource_urls
        payload["resource_ownership_known"] = candidate.resource_ownership_known or bool(candidate.direct_resource_urls)
    return payload


def build_state(candidate: Candidate, evidence: list[str]) -> str:
    return serialize_state(build_state_payload(candidate, evidence))
