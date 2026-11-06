import json

import pytest

from training.auto_dataset import dataset_readiness, read_records, split_training_records
from training.auto_trainer import AutoTrainer


def record(identifier: str, domain: str, label: int):
    return {
        "id": identifier,
        "source": domain,
        "state": {"tag": "DIV", "page_domain": domain},
        "question": {
            "instruction": "Classify this candidate web element.",
            "options": ["ad", "non_ad", "insufficient_evidence"],
            "label": label,
        },
    }


def test_auto_training_refuses_positive_only_data():
    training = [record(f"ad-{index}", f"site-{index}.test", 0) for index in range(12)]
    calibration = [record(f"cal-{label}", f"cal-{label}.test", label) for label in range(3)]
    result = dataset_readiness(training, calibration, 9, 3, 3, 1)
    assert not result.ready
    assert result.reason == "class_imbalance"


def test_balanced_disjoint_data_is_ready():
    training = [
        record(f"train-{label}-{index}", f"site-{label}-{index}.test", label)
        for label in range(3)
        for index in range(3)
    ]
    calibration = [record(f"cal-{label}", f"cal-{label}.test", label) for label in range(3)]
    result = dataset_readiness(training, calibration, 9, 3, 3, 1)
    assert result.ready
    assert result.reason == "ready"


def test_calibration_overlap_is_rejected():
    training = [record(f"item-{label}-{index}", f"site-{index}.test", label) for label in range(3) for index in range(3)]
    calibration = [training[0], record("cal-1", "cal.test", 1), record("cal-2", "cal.test", 2)]
    result = dataset_readiness(training, calibration, 9, 3, 3, 1)
    assert not result.ready
    assert result.reason == "train_calibration_overlap"


def test_binary_readiness_does_not_require_abstention_label():
    training = [
        record(f"train-{label}-{index}", f"site-{label}-{index}.test", label)
        for label in range(2)
        for index in range(3)
    ]
    calibration = [record(f"cal-{label}", f"cal-{label}.test", label) for label in range(2)]
    result = dataset_readiness(training, calibration, 6, 3, 3, 1, required_labels=(0, 1))
    assert result.ready


def test_calibration_domain_overlap_is_rejected():
    training = [
        record(f"train-{label}-{index}", f"site-{label}-{index}.test", label)
        for label in range(2)
        for index in range(3)
    ]
    calibration = [record("cal-0", "site-0-0.test", 0), record("cal-1", "cal.test", 1)]
    result = dataset_readiness(training, calibration, 6, 3, 3, 1, required_labels=(0, 1))
    assert result.reason == "train_calibration_domain_overlap"


def test_validation_split_is_domain_disjoint_and_reproducible():
    records = [
        record(f"train-{label}-{index}", f"site-{label}-{index}.test", label)
        for label in range(2)
        for index in range(20)
    ]
    first_train, first_validation = split_training_records(records, 0.2, 7)
    second_train, second_validation = split_training_records(records, 0.2, 7)
    assert [value["id"] for value in first_train] == [value["id"] for value in second_train]
    assert [value["id"] for value in first_validation] == [value["id"] for value in second_validation]
    assert {value["source"] for value in first_train}.isdisjoint(
        {value["source"] for value in first_validation}
    )


def test_read_records_prefers_human_label_for_same_state(tmp_path):
    weak = record("weak", "example.test", 0)
    weak["provenance"] = {"label_source": "deterministic_rule"}
    human = record("human", "example.test", 1)
    human["provenance"] = {"label_source": "human_reviewed"}
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(value) for value in (weak, human)), encoding="utf-8")
    values = read_records([path])
    assert len(values) == 1
    assert values[0]["id"] == "human"


def test_read_records_rejects_equal_priority_conflicts(tmp_path):
    first = record("first", "example.test", 0)
    second = record("second", "example.test", 1)
    first["provenance"] = {"label_source": "human_reviewed"}
    second["provenance"] = {"label_source": "manual"}
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(value) for value in (first, second)), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting labels for state"):
        read_records([path])


def test_calibration_state_overlap_is_rejected_with_different_ids():
    training = [record(f"train-{label}-{index}", f"site-{label}-{index}.test", label) for label in range(2) for index in range(3)]
    calibration = [record("other-id", "cal.test", 0), record("cal-1", "cal-1.test", 1)]
    calibration[0]["state"] = dict(training[0]["state"])
    result = dataset_readiness(training, calibration, 6, 3, 3, 1, required_labels=(0, 1))
    assert result.reason == "train_calibration_state_overlap"


def test_synthetic_only_data_fails_human_quality_gate():
    training = [
        record(f"train-{label}-{index}", f"site-{label}-{index}.test", label)
        for label in range(2)
        for index in range(3)
    ]
    calibration = [record(f"cal-{label}", f"cal-{label}.test", label) for label in range(2)]

    result = dataset_readiness(
        training,
        calibration,
        6,
        3,
        3,
        1,
        required_labels=(0, 1),
        min_human_examples=2,
    )

    assert result.reason == "not_enough_human_examples"


def test_disabled_auto_trainer_does_not_expose_stale_state(tmp_path):
    trainer = AutoTrainer(tmp_path, {"enabled": False})
    assert trainer.status() == {"enabled": False, "status": "disabled", "running": False}
