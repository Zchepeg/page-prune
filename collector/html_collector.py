"""Deprecated raw-HTML collector.

Kept as a small compatibility shim. New code should send bounded candidate snippets
through ``collector.schemas.Candidate`` rather than collecting an entire document.
"""

from .dom_extractor import normalize_dom_candidate

__all__ = ["normalize_dom_candidate"]
