from __future__ import annotations

from urllib.parse import urlparse

import json
import re
from dataclasses import dataclass
from pathlib import Path

_AD_HOSTS = {
    "doubleclick.net",
    "googlesyndication.com",
    "googleadservices.com",
    "adnxs.com",
    "taboola.com",
    "outbrain.com",
    "criteo.com",
    "criteo.net",
    "amazon-adsystem.com",
    "rubiconproject.com",
    "pubmatic.com",
    "openx.net",
    "casalemedia.com",
    "smartadserver.com",
}
_AD_PATH = re.compile(r"/(ads?|advert|sponsor|promoted|banner)([/_.?=-]|$)", re.I)


def _load_tracker_hosts() -> set[str]:
    project_root = Path(__file__).resolve().parent.parent
    runtime_hosts = project_root / "data" / "runtime" / "tracker_hosts.json"
    try:
        payload = json.loads(runtime_hosts.read_text(encoding="utf-8"))
        return {str(host).lower().rstrip(".") for host in payload["hosts"]}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return set()


_TRACKER_HOSTS = _load_tracker_hosts()


@dataclass(frozen=True, slots=True)
class UrlRisk:
    ad_score: float
    tracker_score: float
    evidence: tuple[str, ...]


def _matches_host(hostname: str, domains: set[str]) -> bool:
    labels = hostname.lower().rstrip(".").split(".")
    return any(".".join(labels[index:]) in domains for index in range(len(labels)))


def _is_ad_host(hostname: str) -> bool:
    return _matches_host(hostname, _AD_HOSTS)


def _is_tracker_host(hostname: str) -> bool:
    return _matches_host(hostname, _TRACKER_HOSTS)


def assess_url(url: str) -> UrlRisk:
    try:
        parsed = urlparse(url)
    except ValueError:
        return UrlRisk(0.0, 0.0, ())
    evidence: list[str] = []
    ad_score = 0.0
    tracker_score = 0.0
    hostname = parsed.hostname or ""
    if _is_ad_host(hostname):
        ad_score = 0.95
        evidence.append("known_ad_host")
    elif _is_tracker_host(hostname):
        tracker_score = 0.95
        evidence.append("known_tracker_host")
    if _AD_PATH.search(parsed.path + "?" + parsed.query):
        ad_score = min(1.0, ad_score + 0.40)
        evidence.append("ad_like_url")
    return UrlRisk(ad_score, tracker_score, tuple(evidence))


def url_ad_score(url: str) -> tuple[float, list[str]]:
    risk = assess_url(url)
    return max(risk.ad_score, risk.tracker_score), list(risk.evidence)
