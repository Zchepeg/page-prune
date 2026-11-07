from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path

from .types import ProbeRecord


class ProbeStore:
    def __init__(self, path: str | Path | None = None, enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = enabled
        self._lock = threading.Lock()
        self._count = 0
        self._types: Counter[str] = Counter()
        self._routes: Counter[str] = Counter()
        self._model_outcomes: Counter[str] = Counter()
        self._review_outcomes: Counter[str] = Counter()

    def append(self, value: dict) -> dict:
        if not self.enabled:
            return {"accepted": False, "reason": "disabled", "probe_id": ""}
        record = ProbeRecord.from_dict(value)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                self._count += 1
                self._types[record.probe_type] += 1
        return {"accepted": True, "probe_id": record.probe_id}

    def record_outcome(self, route: str, model_status: str, review_status: str) -> None:
        with self._lock:
            self._routes[route] += 1
            self._model_outcomes[model_status] += 1
            self._review_outcomes[review_status] += 1

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "records": self._count,
                "types": dict(self._types),
                "outcomes": {
                    "routes": dict(self._routes),
                    "model": dict(self._model_outcomes),
                    "review": dict(self._review_outcomes),
                },
            }
