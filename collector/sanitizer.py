from __future__ import annotations

import re
from dataclasses import replace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .schemas import Candidate

_SECRET_KEYS = re.compile(r"pass(word)?|token|cookie|authorization|secret|session", re.I)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_LONG_NUMBER = re.compile(r"\b\d{9,}\b")


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    keys = list(dict.fromkeys(key for key, _ in parse_qsl(parsed.query, keep_blank_values=True)))[:12]
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path, urlencode([(key, "") for key in keys]), ""))


def sanitize_candidate(candidate: Candidate, max_text: int = 500, max_html: int = 2000) -> Candidate:
    attributes = {
        key.strip().lower().replace("-", "_"): value[:256]
        for key, value in candidate.attributes.items()
        if not _SECRET_KEYS.search(key)
    }
    text = _LONG_NUMBER.sub("[NUMBER]", _EMAIL.sub("[EMAIL]", candidate.text))[:max_text]
    html = _LONG_NUMBER.sub("[NUMBER]", _EMAIL.sub("[EMAIL]", candidate.html))[:max_html]
    direct_resource_urls = list(dict.fromkeys(filter(None, (_safe_url(value) for value in candidate.direct_resource_urls))))[:8]
    resource_urls = list(
        dict.fromkeys(
            [
                *direct_resource_urls,
                *filter(None, (_safe_url(value) for value in candidate.resource_urls)),
            ]
        )
    )[:8]
    return replace(
        candidate,
        text=text,
        html=html,
        attributes=attributes,
        resource_urls=resource_urls,
        direct_resource_urls=direct_resource_urls,
    )
