from __future__ import annotations

from dataclasses import asdict, dataclass

from policy.blocking_policy import Action, PolicyResult


@dataclass(frozen=True, slots=True)
class ExecutionCommand:
    candidate_id: str
    action: Action
    reason: str
    recoverable: bool

    @classmethod
    def from_policy(cls, candidate_id: str, result: PolicyResult) -> "ExecutionCommand":
        return cls(
            candidate_id=candidate_id,
            action=result.action,
            reason=result.reason,
            recoverable=result.action in {Action.HIDE, Action.BLOCK},
        )

    def to_dict(self) -> dict:
        value = asdict(self)
        value["action"] = self.action.value
        return value
