from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _state_fingerprint(record: dict[str, Any]) -> str:
    state = dict(record["state"])
    state.pop("detector_evidence", None)
    encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _domain_key(value: object) -> str:
    domain = str(value).strip().lower().rstrip(".")
    return domain[4:] if domain.startswith("www.") else domain


def _provenance_priority(record: dict[str, Any]) -> int:
    source = str(record.get("provenance", {}).get("label_source", "")).lower()
    if any(value in source for value in ("human", "manual", "reviewed")):
        return 100
    if "consensus" in source:
        return 80
    if source == "deterministic_rule":
        return 60
    if source in {"external_rule_bootstrap", "curated_tracker_taxonomy"}:
        return 50
    if source == "reference_guided_ensemble":
        return 40
    if "counterfactual" in source or "synthetic" in source:
        return 20
    return 0


def _is_human_record(record: dict[str, Any]) -> bool:
    source = str(record.get("provenance", {}).get("label_source", "")).lower()
    return any(value in source for value in ("human", "manual", "reviewed"))


@dataclass(frozen=True, slots=True)
class DatasetReadiness:
    ready: bool
    reason: str
    examples: int
    domains: int
    labels: dict[int, int]
    calibration_examples: int
    calibration_labels: dict[int, int]
    human_examples: int
    human_domains: int
    human_labels: dict[int, int]
    human_calibration_labels: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "examples": self.examples,
            "domains": self.domains,
            "labels": {str(key): value for key, value in self.labels.items()},
            "calibration_examples": self.calibration_examples,
            "calibration_labels": {str(key): value for key, value in self.calibration_labels.items()},
            "human_examples": self.human_examples,
            "human_domains": self.human_domains,
            "human_labels": {str(key): value for key, value in self.human_labels.items()},
            "human_calibration_labels": {str(key): value for key, value in self.human_calibration_labels.items()},
        }


def read_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    label = int(record["question"]["label"])
                    if label not in {0, 1, 2}:
                        raise ValueError("label must be 0, 1, or 2")
                    record_id = str(record["id"])
                    if not str(record.get("source", "")).strip():
                        raise ValueError("source domain is required")
                    existing = records_by_id.get(record_id)
                    if existing is not None:
                        if _state_fingerprint(existing) != _state_fingerprint(record):
                            raise ValueError("duplicate id has different state")
                        if int(existing["question"]["label"]) != label:
                            raise ValueError("duplicate id has conflicting label")
                        if _provenance_priority(record) > _provenance_priority(existing):
                            records_by_id[record_id] = record
                    else:
                        records_by_id[record_id] = record
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid training record at {path}:{line_number}: {exc}") from exc
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records_by_id.values():
        groups.setdefault(_state_fingerprint(record), []).append(record)
    records: list[dict[str, Any]] = []
    for fingerprint, group in groups.items():
        best_priority = max(_provenance_priority(record) for record in group)
        preferred = [record for record in group if _provenance_priority(record) == best_priority]
        labels = {int(record["question"]["label"]) for record in preferred}
        if len(labels) != 1:
            identifiers = ", ".join(sorted(str(record["id"]) for record in preferred)[:5])
            raise ValueError(f"conflicting labels for state {fingerprint[:12]}: {identifiers}")
        records.append(preferred[0])
    return records


def dataset_readiness(
    training_records: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    min_examples: int,
    min_per_label: int,
    min_domains: int,
    min_calibration_per_label: int,
    required_labels: tuple[int, ...] = (0, 1, 2),
    min_human_examples: int = 0,
    min_human_domains: int = 0,
    min_human_per_label: int = 0,
    min_human_calibration_per_label: int = 0,
) -> DatasetReadiness:
    labels = Counter(int(record["question"]["label"]) for record in training_records)
    calibration_labels = Counter(int(record["question"]["label"]) for record in calibration_records)
    domains = {_domain_key(record["source"]) for record in training_records}
    calibration_domains = {_domain_key(record["source"]) for record in calibration_records}
    calibration_ids = {str(record["id"]) for record in calibration_records}
    human_records = [record for record in training_records if _is_human_record(record)]
    human_calibration = [record for record in calibration_records if _is_human_record(record)]
    human_labels = Counter(int(record["question"]["label"]) for record in human_records)
    human_calibration_labels = Counter(int(record["question"]["label"]) for record in human_calibration)
    human_domains = {_domain_key(record["source"]) for record in human_records}
    overlap = calibration_ids & {str(record["id"]) for record in training_records}
    state_overlap = {_state_fingerprint(record) for record in training_records} & {
        _state_fingerprint(record) for record in calibration_records
    }
    reason = "ready"
    if overlap:
        reason = "train_calibration_overlap"
    elif domains & calibration_domains:
        reason = "train_calibration_domain_overlap"
    elif state_overlap:
        reason = "train_calibration_state_overlap"
    elif len(training_records) < min_examples:
        reason = "not_enough_examples"
    elif len(domains) < min_domains:
        reason = "not_enough_domains"
    elif any(labels[index] < min_per_label for index in required_labels):
        reason = "class_imbalance"
    elif any(calibration_labels[index] < min_calibration_per_label for index in required_labels):
        reason = "calibration_not_ready"
    elif len(human_records) < min_human_examples:
        reason = "not_enough_human_examples"
    elif len(human_domains) < min_human_domains:
        reason = "not_enough_human_domains"
    elif any(human_labels[index] < min_human_per_label for index in required_labels):
        reason = "human_class_imbalance"
    elif any(human_calibration_labels[index] < min_human_calibration_per_label for index in required_labels):
        reason = "human_calibration_not_ready"
    return DatasetReadiness(
        ready=reason == "ready",
        reason=reason,
        examples=len(training_records),
        domains=len(domains),
        labels=dict(labels),
        calibration_examples=len(calibration_records),
        calibration_labels=dict(calibration_labels),
        human_examples=len(human_records),
        human_domains=len(human_domains),
        human_labels=dict(human_labels),
        human_calibration_labels=dict(human_calibration_labels),
    )


def split_training_records(
    records: list[dict],
    validation_fraction: float = 0.15,
    seed: int = 42,
    required_labels: tuple[int, ...] = (0, 1),
) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    domains_by_label: dict[int, set[str]] = {label: set() for label in required_labels}
    for record in records:
        label = int(record["question"]["label"])
        if label in domains_by_label:
            domains_by_label[label].add(_domain_key(record["source"]))
    validation_domains: set[str] = set()
    for label, domains in domains_by_label.items():
        ordered = sorted(
            domains,
            key=lambda domain: hashlib.sha256(f"{seed}:{label}:{domain}".encode()).digest(),
        )
        count = max(1, round(len(ordered) * validation_fraction)) if ordered else 0
        validation_domains.update(ordered[:count])
    training = [record for record in records if _domain_key(record["source"]) not in validation_domains]
    validation = [record for record in records if _domain_key(record["source"]) in validation_domains]
    if not training or not validation:
        raise ValueError("domain split produced an empty partition")
    return training, validation


def write_snapshot(
    directory: str | Path,
    training_records: list[dict],
    calibration_records: list[dict],
    validation_fraction: float = 0.15,
    seed: int = 42,
    required_labels: tuple[int, ...] = (0, 1),
) -> tuple[Path, Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    training_records, validation_records = split_training_records(
        training_records,
        validation_fraction,
        seed,
        required_labels,
    )
    train_path = directory / "train.jsonl"
    validation_path = directory / "validation.jsonl"
    calibration_path = directory / "calibration.jsonl"
    train_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in training_records),
        encoding="utf-8",
    )
    validation_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in validation_records),
        encoding="utf-8",
    )
    calibration_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in calibration_records),
        encoding="utf-8",
    )
    return train_path, validation_path, calibration_path
