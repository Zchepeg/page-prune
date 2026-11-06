from pathlib import Path

from service.config import load_config
from evaluation.evaluate_checkpoint import parse_args as parse_evaluation_args
from training.calibrate import parse_args as parse_calibration_args


def test_default_runtime_is_quantized_decision_only():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "config.yaml")
    assert config["jev"]["backend"] == "heuristic"
    assert config["jev"]["quantization"] == "nf4"
    assert config["jev"]["gpu_memory_budget_mb"] == 4096
    assert config["jev"]["inference_max_length"] == 384
    assert config["jev"]["micro_batch_size"] == 2
    assert config["auto_training"]["enabled"] is False
    assert config["training_capture"]["enabled"] is True
    assert config["runtime"]["max_model_candidates_per_batch"] == 12
    assert config['runtime']['defer_model_startup'] is False
    assert config["training_capture"]["control_review_per_domain_limit"] == 12


def test_calibration_and_checkpoint_evaluation_accept_runtime_quantization():
    calibration = parse_calibration_args(
        [
            "--checkpoint",
            "checkpoint",
            "--data",
            "calibration.jsonl",
            "--quantization",
            "nf4",
            "--gpu-memory-budget-mb",
            "4096",
            "--max-length",
            "384",
        ]
    )
    evaluation = parse_evaluation_args(
        [
            "--checkpoint",
            "checkpoint",
            "--data",
            "evaluation.jsonl",
            "--quantization",
            "nf4",
            "--gpu-memory-budget-mb",
            "4096",
            "--max-length",
            "384",
        ]
    )
    assert calibration.quantization == evaluation.quantization == "nf4"
    assert calibration.gpu_memory_budget_mb == evaluation.gpu_memory_budget_mb == 4096
    assert calibration.max_length == evaluation.max_length == 384
