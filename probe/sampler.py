from __future__ import annotations

import hashlib
import threading
from collections import Counter


def sampling_rate(score: float) -> float:
    if score < 0.45 or score >= 0.90:
        return 0.0
    if score < 0.60:
        return 0.001
    if score < 0.75:
        return 0.005
    return 0.01


def deterministic_selection(material: str, score: float) -> tuple[bool, float]:
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    value = int(digest[:13], 16) / 16**13
    rate = sampling_rate(score)
    return value < rate, value


class ProbeBudget:
    def __init__(self, page_limit: int = 3, domain_limit: int = 20):
        self.page_limit = page_limit
        self.domain_limit = domain_limit
        self._pages: Counter[str] = Counter()
        self._domains: Counter[tuple[str, int]] = Counter()
        self._lock = threading.Lock()

    def consume(self, page_id: str, domain: str, hour_window: int) -> bool:
        with self._lock:
            key = (domain, hour_window)
            if self._pages[page_id] >= self.page_limit or self._domains[key] >= self.domain_limit:
                return False
            self._pages[page_id] += 1
            self._domains[key] += 1
            return True
