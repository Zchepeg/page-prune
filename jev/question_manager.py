"""Versioned typed-question registry."""

from .questions import AD_TREATMENT, TypedQuestion

QUESTIONS = {AD_TREATMENT.name: AD_TREATMENT}

__all__ = ["AD_TREATMENT", "QUESTIONS", "TypedQuestion"]
