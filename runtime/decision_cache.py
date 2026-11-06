from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from collector.schemas import Candidate
from jev.types import Decision


@dataclass(slots=True)
class _Entry:
    expires_at: float
    decision: Decision


class DecisionCache:
    def __init__(self, max_size: int = 2048, ttl_seconds: int = 900, version: str = "candidate-features-v2"):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.version = version
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = self._misses = self._expired = self._evictions = 0

    def key(self, candidate: Candidate) -> str:
        stable = {
            "version": self.version,
            "domain": candidate.page.top_level_domain,
            "tag": candidate.tag,
            "id": candidate.element_id,
            "text": candidate.text[:160],
            "classes": candidate.classes,
            "attributes": dict(sorted(candidate.attributes.items())),
            "urls": candidate.resource_urls,
            "direct_urls": candidate.direct_resource_urls,
            "resource_ownership_known": candidate.resource_ownership_known,
            "width": round(candidate.width),
            "height": round(candidate.height),
            "fixed": candidate.fixed,
            "visible": candidate.visible,
        }
        encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, candidate: Candidate) -> Decision | None:
        return self.get_by_key(self.key(candidate))

    def get_by_key(self, key: str) -> Decision | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at <= time.monotonic():
                self._misses += 1
                self._expired += 1
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.decision

    def put(self, candidate: Candidate, decision: Decision) -> None:
        self.put_by_key(self.key(candidate), decision)

    def put_by_key(self, key: str, decision: Decision) -> None:
        with self._lock:
            self._entries[key] = _Entry(time.monotonic() + self.ttl_seconds, decision)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
                self._evictions += 1

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {'hits': self._hits, 'misses': self._misses, 'expired_misses': self._expired, 'evictions': self._evictions, 'resident_entries': len(self._entries), 'max_size': self.max_size, 'ttl_seconds': self.ttl_seconds, 'lookup_hit_fraction': self._hits/total if total else 0}
