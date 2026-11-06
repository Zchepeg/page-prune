from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from collector.schemas import Candidate
from detector.candidate_ranker import CandidateAssessment
from jev.protocol import ProtocolResolution


class DisagreementStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def append(
        self,
        candidate: Candidate,
        assessment: CandidateAssessment,
        resolution: ProtocolResolution,
    ) -> None:
        if self.path is None or not resolution.disagreement:
            return
        event = {
            "candidate_id": candidate.candidate_id,
            "domain": candidate.page.top_level_domain,
            "rule_route": assessment.route,
            "rule_score": round(assessment.score, 6),
            "rule_evidence": assessment.evidence,
            "resolved_verdict": resolution.decision.verdict.value,
            "model_verdict": resolution.decision.diagnostics.get("model_verdict", ""),
            "kind": resolution.disagreement_kind,
            "protocol_version": resolution.decision.diagnostics.get("protocol_version", ""),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
