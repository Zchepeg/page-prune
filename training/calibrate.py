from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from modeling.decision_model import load_decision_runtime
from modeling.encoding import encode_example
from .dataset import load_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit one temperature on a held-out domain calibration set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
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
    model.head.temperature.fill_(1.0)
    max_length = min(int(metadata["max_length"]), args.max_length) if args.max_length > 0 else int(metadata["max_length"])

    logits, labels = [], []
    with torch.inference_mode():
        for example in load_jsonl(args.data, str(metadata.get("label_mode", "three_class")), metadata.get("instruction"), metadata.get("state_schema")):
            if example.label is None:
                raise ValueError("calibration examples require labels")
            logits.append(model.forward_encoded(encode_example(tokenizer, example, max_length)).cpu())
            labels.append(example.label)
    logits_tensor = torch.stack(logits).float()
    labels_tensor = torch.tensor(labels)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_tensor / log_temperature.exp(), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp().detach().clamp(0.05, 20.0))
    before_nll = float(F.cross_entropy(logits_tensor, labels_tensor))
    after_nll = float(F.cross_entropy(logits_tensor / temperature, labels_tensor))
    if not math.isfinite(temperature) or not math.isfinite(after_nll) or after_nll > before_nll + 1e-6:
        raise RuntimeError("calibration failed to produce a finite non-degrading fit; checkpoint unchanged")
    model.head.temperature.fill_(temperature)
    metadata["temperature"] = temperature
    metadata["calibration_examples"] = len(labels)
    metadata["calibration_quantization"] = args.quantization
    metadata["calibration_max_length"] = max_length
    metadata["calibration_data_sha256"] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    metadata["calibration_labels"] = dict(Counter(labels))
    metadata["calibration_nll_before"] = before_nll
    metadata["calibration_nll_after"] = after_nll
    torch.save({"state_dict": model.head.state_dict(), "metadata": metadata}, checkpoint / "pointer_head.pt")
    print(f"temperature={temperature:.6f} n={len(labels)}")


if __name__ == "__main__":
    main()
