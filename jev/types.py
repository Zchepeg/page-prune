from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class Verdict(StrEnum):
    BLOCK = "block"
    ALLOW = "allow"
    UNCERTAIN = "uncertain"


class DecisionLevel(IntEnum):
    RULE = 0
    L0 = 1
    L1 = 2
    L2 = 3

    @classmethod
    def parse(cls, value: str) -> "DecisionLevel":
        return cls[value.upper()]


@dataclass(slots=True)
class Decision:
    candidate_id: str
    verdict: Verdict
    probabilities: dict[str, float]
    level: DecisionLevel
    evidence: list[str] = field(default_factory=list)
    model_version: str = ""
    question_version: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["verdict"] = self.verdict.value
        value["level"] = self.level.name
        return value
