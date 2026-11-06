from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("Config must be JSON-compatible YAML unless PyYAML is installed") from exc
        value = yaml.safe_load(text)
        if not isinstance(value, dict):
            raise ValueError("Configuration root must be an object")
        return value
