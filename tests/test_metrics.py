from evaluation.metrics import classification_metrics, expected_calibration_error


def test_classification_metrics_surface_false_positive_rate():
    result = classification_metrics([True, True, False, False], [True, False, True, False])
    assert result == {"precision": 0.5, "recall": 0.5, "false_positive_rate": 0.5, "accuracy": 0.5}


def test_ece_is_zero_for_perfect_confidence():
    assert abs(expected_calibration_error([True, False], [1.0, 0.0])) < 1e-12
