from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class EventStore:
    """Append-only JSONL audit log; no raw page text is stored."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        self.append_many([event])

    def append_many(self, events: list[dict[str, Any]]) -> None:
        if self.path is None or not events:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            for event in events
        )
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(content)
