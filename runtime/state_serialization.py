from __future__ import annotations

import json
from typing import Any


def serialize_state(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def validate_state_schema(state: str, expected: str | None) -> None:
    if expected and json.loads(state).get("schema_version") != expected:
        raise ValueError("candidate state schema does not match checkpoint")
