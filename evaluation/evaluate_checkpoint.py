from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from modeling.decision_model import load_decision_runtime
from modeling.encoding import encode_example
from training.dataset import load_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen decision checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--gpu-memory-budget-mb", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    model, tokenizer, metadata = load_decision_runtime(
        checkpoint,
        args.device,
        args.quantization,
        args.gpu_memory_budget_mb,
    )
    max_length = min(int(metadata["max_length"]), args.max_length) if args.max_length > 0 else int(metadata["max_length"])

    label_mode = str(metadata.get("label_mode", "three_class"))
    label_count = len(metadata.get("labels", ["ad", "non_ad", "insufficient_evidence"]))
    confusion = [[0 for _ in range(label_count)] for _ in range(label_count)]
    confidences: list[float] = []
    correct: list[bool] = []
    log_loss = 0.0
    examples = load_jsonl(args.data, label_mode, metadata.get("instruction"), metadata.get("state_schema"))
    with torch.inference_mode():
        for example in examples:
            if example.label is None:
                raise ValueError("evaluation examples require labels")
            encoded = encode_example(tokenizer, example, max_length)
            probabilities = torch.softmax(model.forward_encoded(encoded).float(), dim=-1).cpu()
            prediction = int(probabilities.argmax())
            confidence = float(probabilities[prediction])
            confusion[example.label][prediction] += 1
            confidences.append(confidence)
            correct.append(prediction == example.label)
            log_loss -= math.log(max(float(probabilities[example.label]), 1e-12))

    per_label = {}
    for label in range(label_count):
        tp = confusion[label][label]
        actual = sum(confusion[label])
        predicted = sum(row[label] for row in confusion)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        per_label[str(label)] = {"precision": precision, "recall": recall, "support": actual}

    ece = 0.0
    for bin_index in range(10):
        lower, upper = bin_index / 10, (bin_index + 1) / 10
        members = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bin_index == 9 and confidence == 1.0)
        ]
        if members:
            bin_confidence = sum(confidences[index] for index in members) / len(members)
            bin_accuracy = sum(correct[index] for index in members) / len(members)
            ece += len(members) / len(examples) * abs(bin_accuracy - bin_confidence)

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "data": str(Path(args.data).resolve()),
        "examples": len(examples),
        "labels": dict(Counter(example.label for example in examples)),
        "accuracy": sum(correct) / len(correct),
        "mean_log_loss": log_loss / len(examples),
        "ece_10_bin": ece,
        "confusion_actual_rows_predicted_columns": confusion,
        "per_label": per_label,
        "temperature": metadata.get("temperature", 1.0),
        "quantization": args.quantization,
        "max_length": max_length,
    }
    if args.device.startswith("cuda"):
        report["cuda_allocated_mb"] = round(torch.cuda.memory_allocated(args.device) / 1024**2, 1)
        report["cuda_reserved_mb"] = round(torch.cuda.memory_reserved(args.device) / 1024**2, 1)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
