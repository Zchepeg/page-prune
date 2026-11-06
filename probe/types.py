from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any

from .sampler import sampling_rate


PROBE_TYPES = {"delayed_recheck", "dom_change", "visibility", "resource_association"}


@dataclass(frozen=True, slots=True)
class ProbeRecord:
    probe_id: str
    protocol_version: str
    probe_type: str
    domain: str
    candidate_fingerprint: str
    seed_digest: str
    detector_score: float
    detector_route: str
    original_action: str
    sampling_rate: float
    scheduled_delay_ms: int
    started_at: str
    completed_at: str
    before: dict[str, Any]
    after: dict[str, Any]
    delta: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProbeRecord":
        allowed = {
            "probe_id", "protocol_version", "probe_type", "domain", "candidate_fingerprint",
            "seed_digest", "detector_score", "detector_route", "original_action", "sampling_rate",
            "scheduled_delay_ms", "started_at", "completed_at", "before", "after", "delta",
        }
        if set(value) - allowed:
            raise ValueError("unexpected probe fields")
        record = cls(
            probe_id=str(value.get("probe_id", "")),
            protocol_version=str(value.get("protocol_version", "")),
            probe_type=str(value.get("probe_type", "")),
            domain=str(value.get("domain", "")).lower().strip(),
            candidate_fingerprint=str(value.get("candidate_fingerprint", "")),
            seed_digest=str(value.get("seed_digest", "")),
            detector_score=float(value.get("detector_score", -1)),
            detector_route=str(value.get("detector_route", "")),
            original_action=str(value.get("original_action", "")),
            sampling_rate=float(value.get("sampling_rate", 0)),
            scheduled_delay_ms=int(value.get("scheduled_delay_ms", 0)),
            started_at=str(value.get("started_at", "")),
            completed_at=str(value.get("completed_at", "")),
            before=_snapshot(value.get("before")),
            after=_snapshot(value.get("after")),
            delta=_delta(value.get("delta")),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.probe_id or len(self.probe_id) > 128:
            raise ValueError("invalid probe_id")
        if self.protocol_version != "ssa-probe-v1":
            raise ValueError("unsupported protocol_version")
        if self.probe_type not in PROBE_TYPES:
            raise ValueError("invalid probe_type")
        if not self.domain or len(self.domain) > 253:
            raise ValueError("invalid domain")
        if not re.fullmatch(r"[0-9a-f]{64}", self.candidate_fingerprint) or not re.fullmatch(r"[0-9a-f]{16}", self.seed_digest):
            raise ValueError("invalid fingerprint")
        if not 0.45 <= self.detector_score < 0.90:
            raise ValueError("detector_score outside audit band")
        if self.detector_route not in {"block", "allow", "jev"}:
            raise ValueError("invalid detector_route")
        if self.original_action not in {"allow", "review"}:
            raise ValueError("invalid original_action")
        if abs(self.sampling_rate - sampling_rate(self.detector_score)) > 1e-12:
            raise ValueError("invalid sampling_rate")
        if not 100 <= self.scheduled_delay_ms <= 5000:
            raise ValueError("invalid scheduled_delay_ms")
        if not self.started_at or not self.completed_at:
            raise ValueError("timestamps are required")
        try:
            started = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            completed = datetime.fromisoformat(self.completed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid timestamps") from exc
        if started.tzinfo is None or completed.tzinfo is None or not 0 <= (completed - started).total_seconds() <= 10:
            raise ValueError("invalid probe duration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "protocol_version": self.protocol_version,
            "probe_type": self.probe_type,
            "domain": self.domain,
            "candidate_fingerprint": self.candidate_fingerprint,
            "seed_digest": self.seed_digest,
            "detector_score": self.detector_score,
            "detector_route": self.detector_route,
            "original_action": self.original_action,
            "sampling_rate": self.sampling_rate,
            "scheduled_delay_ms": self.scheduled_delay_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
        }


def _snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("snapshot must be an object")
    allowed = {"width", "height", "visible", "in_viewport", "fixed", "child_count", "resource_hosts", "connected"}
    if set(value) - allowed:
        raise ValueError("unexpected snapshot fields")
    hosts = value.get("resource_hosts", [])
    if not isinstance(hosts, list) or len(hosts) > 8:
        raise ValueError("invalid resource_hosts")
    return {
        "width": round(float(value.get("width", 0)), 1),
        "height": round(float(value.get("height", 0)), 1),
        "visible": bool(value.get("visible", False)),
        "in_viewport": bool(value.get("in_viewport", False)),
        "fixed": bool(value.get("fixed", False)),
        "child_count": max(0, min(int(value.get("child_count", 0)), 1000)),
        "resource_hosts": [str(host).lower()[:253] for host in hosts if str(host).strip()],
        "connected": bool(value.get("connected", False)),
    }


def _delta(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("delta must be an object")
    allowed = {"width", "height", "visibility_changed", "viewport_changed", "children", "new_resource_hosts", "mutation_count", "disconnected"}
    if set(value) - allowed:
        raise ValueError("unexpected delta fields")
    return {
        "width": round(float(value.get("width", 0)), 1),
        "height": round(float(value.get("height", 0)), 1),
        "visibility_changed": bool(value.get("visibility_changed", False)),
        "viewport_changed": bool(value.get("viewport_changed", False)),
        "children": max(-1000, min(int(value.get("children", 0)), 1000)),
        "new_resource_hosts": [str(host).lower()[:253] for host in value.get("new_resource_hosts", [])[:8]],
        "mutation_count": max(0, min(int(value.get("mutation_count", 0)), 10000)),
        "disconnected": bool(value.get("disconnected", False)),
    }
