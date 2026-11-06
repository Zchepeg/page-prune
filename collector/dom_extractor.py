"""Compatibility module.

DOM extraction lives in the browser content script so raw pages and form values do not
need to leave the browser. This module normalizes the resulting payload.
"""

from .schemas import Candidate


def normalize_dom_candidate(payload: dict) -> Candidate:
    return Candidate.from_dict(payload)
