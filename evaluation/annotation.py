from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


LABELS = {"ad", "non_ad", "insufficient_evidence"}


def clean_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path[:500], "", ""))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    values = []
    source = Path(path)
    if not source.exists():
        return values
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"record must be an object at {source}:{line_number}")
            values.append(value)
    return values


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def latest_annotations(path: str | Path) -> dict[str, dict[str, Any]]:
    result = {}
    for value in load_jsonl(path):
        identifier = str(value.get("id", ""))
        label = str(value.get("label", ""))
        if identifier and label in LABELS:
            result[identifier] = value
    return result


def pool_status(pool: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    valid_ids = {str(item.get("id", "")) for item in pool}
    labels = Counter(
        str(value["label"])
        for identifier, value in annotations.items()
        if identifier in valid_ids and value.get("label") in LABELS
    )
    reviewed_ids = {identifier for identifier in annotations if identifier in valid_ids}
    reviewed_domains = {
        str(item.get("domain", ""))
        for item in pool
        if str(item.get("id", "")) in reviewed_ids
    }
    return {
        "total": len(pool),
        "reviewed": len(reviewed_ids),
        "remaining": len(pool) - len(reviewed_ids),
        "domains": len({str(item.get("domain", "")) for item in pool}),
        "reviewed_domains": len(reviewed_domains),
        "labels": {label: labels.get(label, 0) for label in sorted(LABELS)},
    }


def freeze_pool(
    pool_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path,
    minimum_examples: int = 300,
    minimum_domains: int = 30,
    minimum_per_label: int = 50,
    required_labels: set[str] | None = None,
    ground_truth_task: str = "three_class",
    allowed_annotation_modes: set[str] | None = None,
) -> dict[str, Any]:
    pool = load_jsonl(pool_path)
    all_annotations = latest_annotations(annotations_path)
    annotations = {
        identifier: value
        for identifier, value in all_annotations.items()
        if allowed_annotation_modes is None or str(value.get("annotation_mode", "")) in allowed_annotation_modes
    }
    if ground_truth_task == 'binary_ad_detection':
        annotations = {identifier: value for identifier, value in annotations.items() if value.get('label') in {'ad', 'non_ad'}}
    status = pool_status(pool, annotations)
    reasons = []
    if status["reviewed"] < minimum_examples:
        reasons.append(f"need_{minimum_examples}_reviewed_examples")
    if status["reviewed_domains"] < minimum_domains:
        reasons.append(f"need_{minimum_domains}_reviewed_domains")
    required = required_labels or LABELS
    for label in sorted(required):
        count = status["labels"].get(label, 0)
        if count < minimum_per_label:
            reasons.append(f"need_{minimum_per_label}_{label}_examples")
    if reasons:
        raise ValueError(", ".join(reasons))
    selected = []
    for item in pool:
        identifier = str(item.get("id", ""))
        annotation = annotations.get(identifier)
        if not annotation:
            continue
        candidate = deepcopy(item["candidate"])
        candidate["html"] = ""
        selected.append(
            {
                "id": identifier,
                "domain": item["domain"],
                "label": annotation["label"],
                "candidate": candidate,
            }
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        for item in selected
    )
    encoded = content.encode("utf-8")
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    metadata = {
        "annotation_method": "independent_human",
        "ground_truth_task": ground_truth_task,
        "required_labels": sorted(required),
        "annotation_modes": sorted(allowed_annotation_modes or {"unspecified"}),
        "excluded_annotations": len(all_annotations) - len(annotations),
        "labels_frozen_before_evaluation": True,
        "used_for_supervision": False,
        "provenance": "Local browser capture followed by explicit human review",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "examples": len(selected),
        "domains": status["reviewed_domains"],
        "labels": status["labels"],
        "sha256": digest,
    }
    Path(metadata_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"output": str(output.resolve()), "metadata": str(Path(metadata_path).resolve()), **metadata}
