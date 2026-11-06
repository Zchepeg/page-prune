from __future__ import annotations

from collections.abc import Sequence


def classification_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, float]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty and equally sized")
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(labels),
    }


def expected_calibration_error(
    labels: Sequence[bool], probabilities: Sequence[float], bins: int = 10
) -> float:
    if len(labels) != len(probabilities) or not labels or bins < 1:
        raise ValueError("inputs must be non-empty, equally sized, and bins must be positive")
    if any(probability < 0 or probability > 1 for probability in probabilities):
        raise ValueError("probabilities must be between zero and one")
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            i
            for i, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1)
        ]
        if not members:
            continue
        confidence = sum(probabilities[i] for i in members) / len(members)
        accuracy = sum(bool(labels[i]) for i in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error
