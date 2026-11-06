from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

from collector.schemas import Candidate
from collector.state_builder import build_state_payload
from detector.candidate_ranker import CandidateRanker


TAGS = ("DIV", "IFRAME", "IMG", "INS", "SCRIPT", "A", "ARTICLE", "ASIDE", "BUTTON", "HEADER", "FOOTER", "NAV")
EVIDENCE = (
    "known_ad_host",
    "known_tracker_host",
    "ad_semantic_marker",
    "ad_text_marker",
    "explicit_ad_attribute",
    "ad_container_element",
    "ad_like_url",
    "common_ad_dimensions",
    "large_fixed_overlay",
    "compound_ad_overlay",
    "multiple_ad_identity_markers",
    "content_semantic_marker",
    "protected_element_type",
    "not_visible",
)
_AD = re.compile(r"(^|[-_\s])(ad|ads|advert|advertisement|sponsor(ed)?|promoted)([-_\s]|$)|广告|推广", re.I)
_PROMO = re.compile(r"promo|partner|recommend|commercial|affiliate", re.I)
_SAFE = re.compile(r"article|content|navigation|checkout|login|comment|header|footer|menu", re.I)


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    weight: torch.Tensor
    bias: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReferenceModel:
    members: tuple[EnsembleMember, ...]
    ad_threshold: float
    non_ad_threshold: float
    reference_sha256: str
    diagnostics: dict[str, Any]


def _host(value: str) -> str:
    try:
        return (urlsplit(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def feature_vector(candidate: Candidate, ranker: CandidateRanker | None = None) -> list[float]:
    ranker = ranker or CandidateRanker()
    assessment = ranker.assess(candidate)
    evidence = set(assessment.evidence)
    identity = " ".join([candidate.element_id, *candidate.classes])
    attributes = " ".join(candidate.attributes.values())
    urls = " ".join(candidate.resource_urls)
    page_host = candidate.page.top_level_domain.lower()
    if page_host.startswith("www."):
        page_host = page_host[4:]
    resource_hosts = [_host(value) for value in candidate.resource_urls]
    third_party = sum(
        bool(host and page_host and host != page_host and not host.endswith("." + page_host))
        for host in resource_hosts
    )
    width = max(0.0, candidate.width)
    height = max(0.0, candidate.height)
    values = [
        assessment.score,
        math.log1p(width) / 10.0,
        math.log1p(height) / 10.0,
        math.log1p(width * height) / 20.0,
        math.log1p(len(candidate.text)) / 10.0,
        min(len(candidate.classes), 20) / 20.0,
        min(len(candidate.resource_urls), 8) / 8.0,
        min(third_party, 8) / 8.0,
        float(candidate.fixed),
        float(candidate.visible),
        float(bool(_AD.search(identity))),
        float(bool(_AD.search(candidate.text[:240]))),
        float(bool(_AD.search(attributes))),
        float(bool(_AD.search(urls))),
        float(bool(_PROMO.search(identity + " " + candidate.text[:240]))),
        float(bool(_SAFE.search(identity))),
        float(not candidate.element_id),
        float(assessment.route == "block"),
        float(assessment.route == "allow"),
        float(assessment.route == "jev"),
    ]
    values.extend(float(candidate.tag == tag) for tag in TAGS)
    values.append(float(candidate.tag not in TAGS))
    values.extend(float(name in evidence) for name in EVIDENCE)
    return values


def _fit_member(features: torch.Tensor, labels: torch.Tensor, indices: list[int], seed: int) -> EnsembleMember:
    torch.manual_seed(seed)
    subset = features[indices]
    targets = labels[indices]
    mean = subset.mean(dim=0)
    scale = subset.std(dim=0).clamp_min(0.05)
    normalized = (subset - mean) / scale
    layer = torch.nn.Linear(normalized.shape[1], 1)
    positives = targets.sum().clamp_min(1.0)
    negatives = (1.0 - targets).sum().clamp_min(1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=negatives / positives)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.03, weight_decay=0.02)
    for _ in range(700):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(layer(normalized).squeeze(1), targets)
        loss.backward()
        optimizer.step()
    return EnsembleMember(
        layer.weight.detach().cpu().squeeze(0),
        layer.bias.detach().cpu().squeeze(0),
        mean.detach().cpu(),
        scale.detach().cpu(),
    )


def _predict_member(member: EnsembleMember, features: torch.Tensor) -> torch.Tensor:
    normalized = (features - member.mean) / member.scale
    return torch.sigmoid(normalized @ member.weight + member.bias)


def _thresholds(probabilities: list[float], labels: list[int], precision_target: float) -> tuple[float, float]:
    ad_threshold = 1.01
    for threshold in sorted(set(probabilities)):
        selected = [index for index, value in enumerate(probabilities) if value >= threshold]
        if len(selected) >= 5 and sum(labels[index] for index in selected) / len(selected) >= precision_target:
            ad_threshold = threshold
            break
    non_ad_threshold = -0.01
    for threshold in sorted(set(probabilities), reverse=True):
        selected = [index for index, value in enumerate(probabilities) if value <= threshold]
        if len(selected) >= 5 and sum(1 - labels[index] for index in selected) / len(selected) >= precision_target:
            non_ad_threshold = threshold
            break
    return max(0.5, ad_threshold), min(0.5, non_ad_threshold)


def train_reference_model(
    records: list[dict[str, Any]],
    folds: int = 5,
    seed: int = 42,
    precision_target: float = 0.95,
) -> ReferenceModel:
    usable = [record for record in records if record.get("label") in {"ad", "non_ad", 0, 1}]
    if len(usable) < 100:
        raise ValueError("at least 100 binary reference labels are required")
    ranker = CandidateRanker()
    features = torch.tensor(
        [feature_vector(Candidate.from_dict(record["candidate"]), ranker) for record in usable],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [1.0 if record["label"] in {"ad", 0} else 0.0 for record in usable],
        dtype=torch.float32,
    )
    domains = [str(record.get("domain", record["candidate"].get("page", {}).get("top_level_domain", ""))) for record in usable]
    fold_ids = [int(hashlib.sha256(f"{seed}:{domain}".encode()).hexdigest()[:8], 16) % folds for domain in domains]
    members = []
    out_of_fold = [0.5] * len(usable)
    for fold in range(folds):
        train_indices = [index for index, value in enumerate(fold_ids) if value != fold]
        validation_indices = [index for index, value in enumerate(fold_ids) if value == fold]
        if not train_indices or not validation_indices:
            raise ValueError("reference domain folds are empty")
        member = _fit_member(features, labels, train_indices, seed + fold)
        members.append(member)
        values = _predict_member(member, features[validation_indices]).tolist()
        for index, value in zip(validation_indices, values):
            out_of_fold[index] = float(value)
    binary_labels = [int(value) for value in labels.tolist()]
    ad_threshold, non_ad_threshold = _thresholds(out_of_fold, binary_labels, precision_target)
    predictions = [
        1 if probability >= ad_threshold else 0 if probability <= non_ad_threshold else -1
        for probability in out_of_fold
    ]
    accepted = [index for index, value in enumerate(predictions) if value >= 0]
    correct = sum(predictions[index] == binary_labels[index] for index in accepted)
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in usable
    ).encode("utf-8")
    return ReferenceModel(
        tuple(members),
        ad_threshold,
        non_ad_threshold,
        hashlib.sha256(encoded).hexdigest(),
        {
            "reference_examples": len(usable),
            "reference_domains": len(set(domains)),
            "reference_labels": dict(Counter("ad" if value else "non_ad" for value in binary_labels)),
            "folds": folds,
            "precision_target": precision_target,
            "oof_accepted": len(accepted),
            "oof_coverage": len(accepted) / len(usable),
            "oof_accuracy": correct / len(accepted) if accepted else 0.0,
        },
    )


def label_candidates(
    model: ReferenceModel,
    pool: list[dict[str, Any]],
    max_std: float = 0.12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ranker = CandidateRanker()
    accepted = []
    review = []
    counts = Counter()
    for item in pool:
        candidate = Candidate.from_dict(item["candidate"])
        vector = torch.tensor([feature_vector(candidate, ranker)], dtype=torch.float32)
        probabilities = [float(_predict_member(member, vector)[0]) for member in model.members]
        mean = sum(probabilities) / len(probabilities)
        variance = sum((value - mean) ** 2 for value in probabilities) / len(probabilities)
        deviation = math.sqrt(variance)
        if mean >= model.ad_threshold:
            suggested = "ad"
            label = 0
        elif mean <= model.non_ad_threshold:
            suggested = "non_ad"
            label = 1
        else:
            suggested = "review"
            label = None
        if deviation > max_std:
            label = None
        assessment = ranker.assess(candidate)
        provenance = {
            "label_source": "reference_guided_ensemble" if label is not None else "reference_guided_review",
            "reference_sha256": model.reference_sha256,
            "ad_probability": round(mean, 6),
            "ensemble_std": round(deviation, 6),
            "ad_threshold": round(model.ad_threshold, 6),
            "non_ad_threshold": round(model.non_ad_threshold, 6),
            "rule_score": round(assessment.score, 6),
            "rule_evidence": assessment.evidence,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if label is not None:
            accepted.append(
                {
                    "id": f"reference-guided-{item['id']}",
                    "source": str(item["domain"]),
                    "state": build_state_payload(candidate, assessment.evidence),
                    "question": {
                        "instruction": "Classify this candidate web element.",
                        "options": ["ad", "non_ad"],
                        "label": label,
                    },
                    "provenance": provenance,
                }
            )
            counts[suggested] += 1
        else:
            review.append(
                {
                    "id": str(item["id"]),
                    "domain": str(item["domain"]),
                    "candidate": item["candidate"],
                    "suggested_label": suggested,
                    "provenance": provenance,
                }
            )
            counts["review"] += 1
    report = {
        **model.diagnostics,
        "reference_sha256": model.reference_sha256,
        "ad_threshold": model.ad_threshold,
        "non_ad_threshold": model.non_ad_threshold,
        "input_examples": len(pool),
        "input_domains": len({str(item["domain"]) for item in pool}),
        "accepted_examples": len(accepted),
        "accepted_domains": len({str(item["source"]) for item in accepted}),
        "labels": dict(counts),
        "max_ensemble_std": max_std,
    }
    return accepted, review, report


def augment_reference_ads(
    reference: list[dict[str, Any]],
    domains: list[str],
    ad_hosts: list[str],
    count: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    positives = [Candidate.from_dict(item["candidate"]) for item in reference if item.get("label") in {"ad", 0}]
    if not positives or not domains or not ad_hosts or count <= 0:
        return []
    rng = random.Random(seed)
    markers = ("ad-slot", "advertisement", "sponsored-placement", "promoted-unit", "commercial-card", "partner-ad")
    texts = ("Advertisement", "Sponsored", "Promoted", "广告", "推广", "")
    records = []
    ranker = CandidateRanker()
    distribution = Counter((item.tag, round(item.width / 20), round(item.height / 20)) for item in positives)
    distribution_hash = hashlib.sha256(
        json.dumps(
            sorted((tag, width, height, frequency) for (tag, width, height), frequency in distribution.items()),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    for index in range(count):
        template = rng.choice(positives)
        domain = domains[index % len(domains)]
        host = ad_hosts[index % len(ad_hosts)]
        marker = markers[index % len(markers)]
        digest = hashlib.sha256(f"{seed}:{index}:{domain}:{host}:{marker}".encode()).hexdigest()
        width = max(1.0, template.width * rng.uniform(0.92, 1.08)) if template.width else rng.choice([300, 320, 336, 728])
        height = max(1.0, template.height * rng.uniform(0.92, 1.08)) if template.height else rng.choice([50, 90, 250, 280])
        tag = template.tag if template.tag not in {"ARTICLE", "MAIN", "NAV", "FORM", "INPUT", "BUTTON"} else "DIV"
        use_third_party = index % 4 != 0
        candidate = Candidate.from_dict(
            {
                "candidate_id": f"reference-augmented-{digest[:20]}",
                "page": {"url": f"https://{domain}/", "top_level_domain": domain},
                "tag": tag,
                "text": texts[index % len(texts)],
                "element_id": f"{marker}-{digest[:8]}",
                "classes": [marker, rng.choice(("slot", "widget", "placement", "creative", "module"))],
                "attributes": {"aria_label": texts[index % len(texts)]} if index % 3 == 0 else {},
                "resource_urls": [
                    f"https://{host}/ads/{digest[8:20]}/creative.js"
                    if use_third_party
                    else f"https://{domain}/ads/{digest[8:20]}/banner.js"
                ],
                "width": round(width, 1),
                "height": round(height, 1),
                "fixed": bool(template.fixed and index % 3 == 0),
                "visible": True,
            }
        )
        assessment = ranker.assess(candidate)
        records.append(
            {
                "id": candidate.candidate_id,
                "source": domain,
                "state": build_state_payload(candidate, assessment.evidence),
                "question": {
                    "instruction": "Classify this candidate web element.",
                    "options": ["ad", "non_ad"],
                    "label": 0,
                },
                "provenance": {
                    "label_source": "reference_guided_counterfactual",
                    "reference_distribution_sha256": distribution_hash,
                    "construction_seed": seed,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )
    return records


def load_reference(pool_path: str | Path, annotation_path: str | Path) -> list[dict[str, Any]]:
    pool = {
        str(item["id"]): item
        for item in (
            json.loads(line)
            for line in Path(pool_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    annotations: dict[str, dict[str, Any]] = {}
    for line in Path(annotation_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        annotation = json.loads(line)
        annotations[str(annotation.get("id", ""))] = annotation
    records = []
    for identifier, item in pool.items():
        annotation = annotations.get(identifier, {})
        if item and annotation.get("label") in {"ad", "non_ad"}:
            records.append({**item, "label": annotation["label"], "annotation_mode": annotation.get("annotation_mode", "")})
    return records


def write_dataset(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
