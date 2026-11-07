from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev.types import Decision

from .analyzer import ProbeAssessment
from .types import ProbeRecord


class ProbeReviewStore:
    def __init__(self, pending_path: str | Path, reviewed_path: str | Path):
        self.pending_path = Path(pending_path)
        self.reviewed_path = Path(reviewed_path)
        self._lock = threading.Lock()
        self._inflight: set[str] = set()

    def queue(self, record: ProbeRecord, assessment: ProbeAssessment, decision: Decision | None) -> dict[str, Any]:
        result, _, _ = self.queue_with_decision(record, assessment, lambda: (decision, "not_requested"))
        return result

    def queue_with_decision(
        self, record: ProbeRecord, assessment: ProbeAssessment,
        decide: Callable[[], tuple[Decision | None, str]],
    ) -> tuple[dict[str, Any], Decision | None, str]:
        identity = json.dumps(
            [record.domain, record.candidate_fingerprint, record.delta],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        review_id = "probe-review-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        with self._lock:
            if (review_id in self._inflight or review_id in self._ids(self.pending_path)
                    or review_id in self._ids(self.reviewed_path)):
                return {"queued": False, "reason": "duplicate", "review_id": review_id}, None, "duplicate_review"
            self._inflight.add(review_id)
        try:
            decision, model_status = decide()
            self._write_pending(review_id, record, assessment, decision)
            return {"queued": True, "reason": "queued", "review_id": review_id}, decision, model_status
        finally:
            with self._lock:
                self._inflight.remove(review_id)

    def _write_pending(self, review_id: str, record: ProbeRecord, assessment: ProbeAssessment, decision: Decision | None) -> None:
        value = {
            "id": review_id,
            "source": record.domain,
            "state": json.loads(record_to_state(record, assessment)),
            "question": {
                "instruction": "Classify whether this observed element transition is advertising behavior.",
                "options": ["ad", "non_ad", "insufficient_evidence"],
                "label": None,
            },
            "provenance": {
                "label_source": "probe_pending_review",
                "probe_id": record.probe_id,
                "probe_type": record.probe_type,
                "probe_score": assessment.score,
                "information_value": assessment.information_value,
                "evidence": assessment.evidence,
                "model_verdict": decision.verdict.value if decision else "",
                "model_probabilities": decision.probabilities if decision else {},
                "model_version": decision.model_version if decision else "",
                "captured_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        with self._lock:
            self.pending_path.parent.mkdir(parents=True, exist_ok=True)
            with self.pending_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        with self._lock:
            reviewed = self._ids(self.reviewed_path)
            values = [value for value in self._records(self.pending_path) if str(value.get("id", "")) not in reviewed]
        values.sort(
            key=lambda value: (
                float(value.get("provenance", {}).get("information_value", 0)),
                float(value.get("provenance", {}).get("probe_score", 0)),
            ),
            reverse=True,
        )
        return values[:limit]

    def resolve(self, review_id: str, label: int) -> dict[str, Any]:
        if label not in {0, 1, 2}:
            raise ValueError("label must be 0, 1, or 2")
        with self._lock:
            reviewed = self._ids(self.reviewed_path)
            if review_id in reviewed:
                return {"saved": False, "reason": "already_reviewed", "review_id": review_id}
            source = next((value for value in self._records(self.pending_path) if value.get("id") == review_id), None)
            if source is None:
                return {"saved": False, "reason": "not_found", "review_id": review_id}
            value = json.loads(json.dumps(source, ensure_ascii=False))
            value["question"]["label"] = label
            value["provenance"]["label_source"] = "human_probe_review"
            value["provenance"]["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            self.reviewed_path.parent.mkdir(parents=True, exist_ok=True)
            with self.reviewed_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        return {"saved": True, "reason": "reviewed", "review_id": review_id}

    def stats(self) -> dict[str, int]:
        with self._lock:
            pending = self._ids(self.pending_path)
            reviewed = self._ids(self.reviewed_path)
        return {"pending": len(pending - reviewed), "reviewed": len(reviewed)}

    @staticmethod
    def _records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        values = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return values

    @classmethod
    def _ids(cls, path: Path) -> set[str]:
        return {str(value.get("id", "")) for value in cls._records(path) if value.get("id")}


def record_to_state(record: ProbeRecord, assessment: ProbeAssessment) -> str:
    from .analyzer import ProbeAnalyzer

    return ProbeAnalyzer.build_state(record, assessment)
