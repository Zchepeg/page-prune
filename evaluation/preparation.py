from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any

from collector.schemas import Candidate
from collector.state_builder import build_state
from detector.candidate_ranker import CandidateRanker
from jev.backend import QwenPointerBackend
from jev.decider import JevDecider
from jev.protocol import DecisionProtocol, ProtocolRoute
from jev.types import Verdict
from service.config import load_config


_SPACE = re.compile(r"\s+")


def _normalized(value: Any, limit: int = 1000) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()[:limit]


def candidate_fingerprint(item: dict[str, Any]) -> str:
    candidate = Candidate.from_dict(item.get("candidate", {}))
    payload = {
        "domain": str(item.get("domain", "")).lower(),
        "page": candidate.page.url.split("?", 1)[0].split("#", 1)[0],
        "tag": candidate.tag,
        "text": _normalized(candidate.text),
        "element_id": _normalized(candidate.element_id, 200),
        "classes": sorted(_normalized(value, 200) for value in candidate.classes),
        "attributes": sorted((str(key), _normalized(value, 200)) for key, value in candidate.attributes.items()),
        "resources": sorted(url.split("?", 1)[0].split("#", 1)[0] for url in candidate.resource_urls),
        "width": round(candidate.width, 1),
        "height": round(candidate.height, 1),
        "fixed": candidate.fixed,
        "visible": candidate.visible,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rule_prelabel(item: dict[str, Any], ranker: CandidateRanker) -> tuple[dict[str, Any], Any]:
    candidate = Candidate.from_dict(item["candidate"])
    assessment = ranker.assess(candidate)
    if assessment.route == "block":
        label = "ad"
        confidence = assessment.score
    elif assessment.route == "allow":
        label = "non_ad"
        confidence = 1.0 - assessment.score
    else:
        label = "insufficient_evidence"
        confidence = max(0.34, 1.0 - abs(assessment.score - 0.5))
    ambiguity = 1.0 - min(1.0, abs(assessment.score - 0.5) * 2.0)
    priority = 0.25 + 0.55 * ambiguity
    if item.get("review_context", {}).get("candidate_kind") == "suspicious":
        priority += 0.1
    if candidate.tag in {"ARTICLE", "MAIN", "NAV", "FORM", "INPUT", "BUTTON"} and assessment.score > 0.12:
        priority += 0.1
    return {
        "id": str(item["id"]),
        "suggested_label": label,
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "source": "deterministic_rule" if assessment.route != "jev" else "rule_uncertainty",
        "route": assessment.route,
        "score": round(assessment.score, 6),
        "evidence": assessment.evidence,
        "review_priority": round(max(0.0, min(1.0, priority)), 6),
    }, assessment


def _stratified(items: list[dict[str, Any]], priorities: dict[str, float]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get("domain", ""))].append(item)
    queues = {
        domain: deque(sorted(values, key=lambda value: (-priorities[str(value["id"])], str(value["id"]))))
        for domain, values in groups.items()
    }
    domains = sorted(queues, key=lambda domain: (-priorities[str(queues[domain][0]["id"])], domain))
    result = []
    while domains:
        next_domains = []
        for domain in domains:
            result.append(queues[domain].popleft())
            if queues[domain]:
                next_domains.append(domain)
        domains = sorted(next_domains, key=lambda domain: (-priorities[str(queues[domain][0]["id"])], domain))
    return result


def prepare_records(pool: list[dict[str, Any]], screenshot_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(screenshot_root).resolve()
    ranker = CandidateRanker()
    seen_ids = set()
    seen_fingerprints = {}
    curated = []
    prelabels = []
    rejected = Counter()
    screenshots = Counter()
    routes = Counter()
    duplicate_groups = Counter()
    for source in pool:
        item = deepcopy(source)
        identifier = str(item.get("id", ""))
        candidate_value = item.get("candidate")
        if not identifier or not isinstance(candidate_value, dict) or not str(item.get("domain", "")):
            rejected["invalid_record"] += 1
            continue
        if identifier in seen_ids:
            rejected["duplicate_id"] += 1
            continue
        seen_ids.add(identifier)
        fingerprint = candidate_fingerprint(item)
        if fingerprint in seen_fingerprints:
            rejected["duplicate_candidate"] += 1
            duplicate_groups[seen_fingerprints[fingerprint]] += 1
            continue
        seen_fingerprints[fingerprint] = identifier
        screenshot_value = str(item.get("review_context", {}).get("screenshot", ""))
        screenshot = Path(screenshot_value) if screenshot_value else None
        screenshot_valid = False
        if screenshot:
            try:
                resolved = screenshot.resolve(strict=True)
                resolved.relative_to(root)
                screenshot_valid = resolved.is_file()
            except (OSError, ValueError):
                screenshot_valid = False
        screenshots["available" if screenshot_valid else "missing"] += 1
        item.setdefault("review_context", {})["screenshot_available"] = screenshot_valid
        item["review_context"]["fingerprint"] = fingerprint
        prelabel, assessment = _rule_prelabel(item, ranker)
        if not screenshot_valid:
            prelabel["review_priority"] = round(max(0.0, prelabel["review_priority"] - 0.35), 6)
        routes[assessment.route] += 1
        curated.append(item)
        prelabels.append(prelabel)
    priorities = {value["id"]: float(value["review_priority"]) for value in prelabels}
    curated = _stratified(curated, priorities)
    prelabels.sort(key=lambda value: (-float(value["review_priority"]), str(value["id"])))
    report = {
        "input_examples": len(pool),
        "curated_examples": len(curated),
        "removed_examples": len(pool) - len(curated),
        "domains": len({str(item.get("domain", "")) for item in curated}),
        "rejected": dict(sorted(rejected.items())),
        "screenshots": dict(sorted(screenshots.items())),
        "routes": dict(sorted(routes.items())),
        "duplicate_groups": len(duplicate_groups),
        "prelabel_sources": dict(sorted(Counter(value["source"] for value in prelabels).items())),
        "suggested_labels": dict(sorted(Counter(value["suggested_label"] for value in prelabels).items())),
    }
    return curated, prelabels, report


def apply_qwen_prelabels(
    pool: list[dict[str, Any]],
    prelabels: list[dict[str, Any]],
    config_path: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    settings = config["jev"]
    if settings.get("backend") != "qwen_pointer":
        raise ValueError("Qwen prelabeling requires jev.backend=qwen_pointer")
    backend = QwenPointerBackend(
        settings["checkpoint"],
        settings.get("device", "cuda"),
        settings.get("quantization", ""),
        int(settings.get("gpu_memory_budget_mb", 0)),
        int(settings.get("inference_max_length", 0)),
    )
    decider = JevDecider(
        backend,
        float(settings.get("uncertainty_margin", 0.08)),
        float(settings.get("min_confidence", 0.0)),
    )
    ranker = CandidateRanker()
    protocol_settings = config.get("decision_protocol", {})
    protocol = DecisionProtocol(
        float(protocol_settings.get("challenge_block_below", 0.98)),
        float(protocol_settings.get("challenge_allow_above", 0.02)),
        str(protocol_settings.get("version", "jev-ad-decision-v1")),
    )
    prelabel_map = {str(value["id"]): value for value in prelabels}
    processed = Counter()
    for item in pool:
        identifier = str(item["id"])
        candidate = Candidate.from_dict(item["candidate"])
        assessment = ranker.assess(candidate)
        plan = protocol.plan(assessment)
        if plan.route is ProtocolRoute.RULE:
            continue
        model_decision = decider.decide(identifier, build_state(candidate, assessment.evidence), assessment.evidence)
        resolution = protocol.resolve(plan, model_decision)
        labels = {
            Verdict.BLOCK: "ad",
            Verdict.ALLOW: "non_ad",
            Verdict.UNCERTAIN: "insufficient_evidence",
        }
        value = prelabel_map[identifier]
        value["suggested_label"] = labels[resolution.decision.verdict]
        value["confidence"] = round(max(resolution.decision.probabilities.values()), 6)
        value["source"] = "qwen_protocol"
        value["route"] = plan.route.value
        value["probabilities"] = {key: round(number, 6) for key, number in resolution.decision.probabilities.items()}
        value["model_version"] = resolution.decision.model_version
        value["disagreement"] = resolution.disagreement
        value["review_priority"] = round(
            min(1.0, max(float(value["review_priority"]), 1.0 - float(value["confidence"]) + (0.2 if resolution.disagreement else 0.0))),
            6,
        )
        processed["disagreement" if resolution.disagreement else plan.route.value] += 1
    priorities = {identifier: float(value["review_priority"]) for identifier, value in prelabel_map.items()}
    reordered = _stratified(pool, priorities)
    pool[:] = reordered
    prelabels.sort(key=lambda value: (-float(value["review_priority"]), str(value["id"])))
    return {
        "qwen_processed": sum(processed.values()),
        "qwen_routes": dict(sorted(processed.items())),
        "prelabel_sources": dict(sorted(Counter(value["source"] for value in prelabels).items())),
        "suggested_labels": dict(sorted(Counter(value["suggested_label"] for value in prelabels).items())),
    }


def write_jsonl(path: str | Path, values: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )
