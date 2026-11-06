from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from collector.schemas import Candidate
from collector.state_builder import build_state_payload
from detector.candidate_ranker import CandidateAssessment
from detector.candidate_ranker import CandidateRanker
from jev.types import Decision

from .active_learning import ActiveLearningSampler

_STRONG_EVIDENCE = {"known_ad_host", "known_tracker_host"}
_SUPPORTING_EVIDENCE = {
    "ad_like_url",
    "common_ad_dimensions",
    "large_fixed_overlay",
    "explicit_ad_attribute",
    "ad_container_element",
}
_REVIEW_EVIDENCE = {"ad_semantic_marker", "ad_text_marker", "ad_like_url", "explicit_ad_attribute"}


@dataclass(frozen=True, slots=True)
class CaptureResult:
    saved: bool
    reason: str
    example_id: str = ""
    queued: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "saved": self.saved,
            "queued": self.queued,
            "reason": self.reason,
            "example_id": self.example_id,
        }


def _safe_url(value: str) -> str:
    """Keep routing evidence while dropping query strings, fragments and credentials."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{parsed.hostname}{port}", parsed.path[:512], "", ""))


class TrainingCaptureStore:
    """Append-only pseudo-label store for strong, deterministic ad examples.

    Model predictions are intentionally ineligible: otherwise a deployed model would
    reinforce its own mistakes. Every record carries its rule evidence and provenance.
    """

    def __init__(
        self,
        path: str | Path,
        review_path: str | Path | None = None,
        reviewed_path: str | Path | None = None,
        enabled: bool = True,
        min_score: float = 0.85,
        review_min_score: float = 0.35,
        per_domain_limit: int = 500,
        review_per_domain_limit: int = 200,
        control_review_per_domain_limit: int = 12,
        max_text_length: int = 240,
        excluded_domains: list[str] | set[str] | None = None,
        sampler: ActiveLearningSampler | None = None,
    ):
        self.path = Path(path)
        self.review_path = Path(review_path) if review_path else self.path.with_name("pending_review.jsonl")
        self.reviewed_path = Path(reviewed_path) if reviewed_path else self.path.with_name("reviewed.jsonl")
        self.enabled = enabled
        self.min_score = min_score
        self.review_min_score = review_min_score
        self.per_domain_limit = per_domain_limit
        self.review_per_domain_limit = review_per_domain_limit
        self.control_review_per_domain_limit = max(0, control_review_per_domain_limit)
        self.max_text_length = max_text_length
        self.excluded_domains = {
            str(domain).lower().strip().lstrip(".") for domain in (excluded_domains or []) if str(domain).strip()
        }
        self.sampler = sampler or ActiveLearningSampler()
        self._lock = threading.Lock()
        self._fingerprints: set[str] = set()
        self._review_fingerprints: set[str] = set()
        self._reviewed_fingerprints: set[str] = set()
        self._domains: Counter[str] = Counter()
        self._review_domains: Counter[str] = Counter()
        self._reviewed_domains: Counter[str] = Counter()
        self._control_review_domains: Counter[str] = Counter()
        self._load_index()

    def _load_index(self) -> None:
        for path, fingerprints, domains in (
            (self.path, self._fingerprints, self._domains),
            (self.review_path, self._review_fingerprints, self._review_domains),
            (self.reviewed_path, self._reviewed_fingerprints, self._reviewed_domains),
        ):
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        fingerprints.add(str(record["id"]))
                        domain = str(record.get("source", ""))
                        domains[domain] += 1
                        if path == self.review_path and record.get("provenance", {}).get("selection_reason") == "random_control":
                            self._control_review_domains[domain] += 1
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue

    def _state(self, candidate: Candidate, evidence: list[str]) -> dict[str, Any]:
        bounded = replace(
            candidate,
            text=candidate.text[: self.max_text_length],
            element_id=candidate.element_id[:128],
            classes=candidate.classes[:20],
            attributes={key: value[:256] for key, value in candidate.attributes.items()},
            resource_urls=[url for value in candidate.resource_urls if (url := _safe_url(value))][:5],
        )
        return build_state_payload(bounded, evidence)

    @staticmethod
    def _has_strong_evidence(evidence: set[str]) -> bool:
        return bool(evidence & _STRONG_EVIDENCE) or (
            "ad_semantic_marker" in evidence and bool(evidence & _SUPPORTING_EVIDENCE)
        ) or (
            {"explicit_ad_attribute", "ad_container_element"} <= evidence
        )

    def capture(
        self,
        candidate: Candidate,
        assessment: CandidateAssessment,
        decision: Decision | None = None,
    ) -> CaptureResult:
        if not self.enabled:
            return CaptureResult(False, "disabled")
        evidence = set(assessment.evidence)
        domain = candidate.page.top_level_domain.lower().strip()
        if not domain:
            return CaptureResult(False, "missing_domain")
        if any(domain == excluded or domain.endswith("." + excluded) for excluded in self.excluded_domains):
            return CaptureResult(False, "excluded_domain")
        deferred = bool(decision and decision.diagnostics.get("model_budget_deferred"))
        review_evidence = assessment.score >= self.review_min_score and bool(evidence & _REVIEW_EVIDENCE)
        if deferred and not review_evidence:
            return CaptureResult(False, "model_budget_deferred")
        state = self._state(candidate, assessment.evidence)
        fingerprint_source = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        example_id = "auto-ad-" + hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:20]
        disagreed = bool(decision and decision.diagnostics.get("resolution") == "review_disagreement")
        strong = (
            assessment.route == "block"
            and assessment.score >= self.min_score
            and self._has_strong_evidence(evidence)
            and not disagreed
        )
        if not strong:
            return self._queue_for_review(candidate, assessment, decision, evidence, domain, state, example_id)
        record = {
            "id": example_id,
            "source": domain,
            "state": state,
            "question": {
                "instruction": "Classify this candidate web element.",
                "options": ["ad", "non_ad", "insufficient_evidence"],
                "label": 0,
            },
            "provenance": {
                "label_source": "deterministic_rule",
                "score": round(assessment.score, 6),
                "evidence": sorted(evidence),
                "captured_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            if example_id in self._fingerprints:
                return CaptureResult(False, "duplicate", example_id)
            if self._domains[domain] >= self.per_domain_limit:
                return CaptureResult(False, "domain_limit", example_id)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._fingerprints.add(example_id)
            self._domains[domain] += 1
        return CaptureResult(True, "saved", example_id)

    def _queue_for_review(
        self,
        candidate: Candidate,
        assessment: CandidateAssessment,
        decision: Decision | None,
        evidence: set[str],
        domain: str,
        state: dict[str, Any],
        example_id: str,
    ) -> CaptureResult:
        priority = self.sampler.evaluate(assessment, decision)
        control = candidate.collection_stratum == "control"
        legacy_review = assessment.score >= self.review_min_score and bool(evidence & _REVIEW_EVIDENCE)
        if not control and not legacy_review and not self.sampler.accepts(priority):
            return CaptureResult(False, "below_review_threshold")
        review_id = example_id.replace("auto-ad-", "review-")
        record = {
            "id": review_id,
            "source": domain,
            "state": state,
            "question": {
                "instruction": "Classify this candidate web element.",
                "options": ["ad", "non_ad", "insufficient_evidence"],
                "label": None,
            },
            "provenance": {
                "label_source": "unreviewed_candidate",
                "score": round(assessment.score, 6),
                "route": assessment.route,
                "evidence": sorted(evidence),
                "selection_reason": "random_control" if control else priority.reason,
                "priority": priority.score,
                "priority_components": priority.components,
                "model_verdict": decision.diagnostics.get("model_verdict", decision.verdict.value) if decision else "",
                "protocol_version": decision.diagnostics.get("protocol_version", "") if decision else "",
                "captured_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            if review_id in self._review_fingerprints:
                return CaptureResult(False, "review_duplicate", review_id)
            if self._review_domains[domain] >= self.review_per_domain_limit:
                return CaptureResult(False, "review_domain_limit", review_id)
            if control and self._control_review_domains[domain] >= self.control_review_per_domain_limit:
                return CaptureResult(False, "control_domain_limit", review_id)
            self.review_path.parent.mkdir(parents=True, exist_ok=True)
            with self.review_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._review_fingerprints.add(review_id)
            self._review_domains[domain] += 1
            if control:
                self._control_review_domains[domain] += 1
        return CaptureResult(False, "queued_for_review", review_id, queued=True)

    def record_manual(self, candidate: Candidate, label: int = 0) -> CaptureResult:
        """Persist a user-confirmed label separately from rule pseudo-labels."""
        if not self.enabled:
            return CaptureResult(False, "disabled")
        if label not in {0, 1, 2}:
            return CaptureResult(False, "invalid_label")
        domain = candidate.page.top_level_domain.lower().strip()
        if not domain:
            return CaptureResult(False, "missing_domain")
        assessment = CandidateRanker().assess(candidate)
        state = self._state(candidate, assessment.evidence)
        fingerprint_source = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        example_id = "manual-" + hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:20]
        record = {
            "id": example_id,
            "source": domain,
            "state": state,
            "question": {
                "instruction": "Classify this candidate web element.",
                "options": ["ad", "non_ad", "insufficient_evidence"],
                "label": label,
            },
            "provenance": {
                "label_source": "human_manual",
                "captured_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            if example_id in self._reviewed_fingerprints:
                return CaptureResult(False, "manual_duplicate", example_id)
            self.reviewed_path.parent.mkdir(parents=True, exist_ok=True)
            with self.reviewed_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            self._reviewed_fingerprints.add(example_id)
            self._reviewed_domains[domain] += 1
        return CaptureResult(True, "manual_saved", example_id)

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        if not self.review_path.exists():
            return []
        items = []
        with self._lock:
            with self.review_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    identifier = str(record.get("id", ""))
                    if not identifier or identifier in self._reviewed_fingerprints:
                        continue
                    items.append(record)
                    if len(items) >= limit:
                        break
        return items

    def resolve_review(self, review_id: str, label: int) -> dict[str, Any]:
        if label not in {0, 1, 2}:
            return {"saved": False, "reason": "invalid_label", "review_id": review_id}
        if not review_id:
            return {"saved": False, "reason": "missing_review_id", "review_id": review_id}
        with self._lock:
            if review_id in self._reviewed_fingerprints:
                return {"saved": False, "reason": "already_reviewed", "review_id": review_id}
            pending = None
            if self.review_path.exists():
                with self.review_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if str(record.get("id", "")) == review_id:
                            pending = record
                            break
            if pending is None:
                return {"saved": False, "reason": "not_found", "review_id": review_id}
            reviewed = deepcopy(pending)
            reviewed["question"]["label"] = label
            original_source = str(reviewed.get("provenance", {}).get("label_source", ""))
            reviewed["provenance"] = {
                **reviewed.get("provenance", {}),
                "label_source": "human_reviewed",
                "original_label_source": original_source,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.reviewed_path.parent.mkdir(parents=True, exist_ok=True)
            with self.reviewed_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(reviewed, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
            domain = str(reviewed.get("source", ""))
            self._reviewed_fingerprints.add(review_id)
            self._reviewed_domains[domain] += 1
        return {"saved": True, "reason": "review_saved", "review_id": review_id, "label": label}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "path": str(self.path),
                "examples": len(self._fingerprints),
                "domains": len(self._domains),
                "per_domain": dict(self._domains.most_common(20)),
                "review_path": str(self.review_path),
                "review_examples": len(self._review_fingerprints - self._reviewed_fingerprints),
                "review_total_examples": len(self._review_fingerprints),
                "review_domains": len(self._review_domains),
                "review_per_domain": dict(self._review_domains.most_common(20)),
                "control_review_per_domain": dict(self._control_review_domains.most_common(20)),
                "reviewed_path": str(self.reviewed_path),
                "reviewed_examples": len(self._reviewed_fingerprints),
                "reviewed_domains": len(self._reviewed_domains),
            }
