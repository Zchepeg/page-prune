from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jev.backend import QwenPointerBackend
from jev.questions import AD_TREATMENT
from training.dataset import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--gpu-memory-budget-mb", type=int, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--disable-allocator-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.data, "binary_selective")[: args.samples]
    states = [example.state for example in examples]
    if args.disable_allocator_warmup:
        import transformers.modeling_utils

        transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    started = time.perf_counter()
    backend = QwenPointerBackend(
        args.checkpoint,
        "cuda",
        "nf4",
        args.gpu_memory_budget_mb,
        args.max_length,
        args.micro_batch_size,
    )
    load_ms = (time.perf_counter() - started) * 1000
    timings = []
    probabilities = []
    for start in range(0, len(states), args.micro_batch_size):
        batch = states[start : start + args.micro_batch_size]
        batch_started = time.perf_counter()
        results = backend.decide_many(batch, AD_TREATMENT)
        timings.append((time.perf_counter() - batch_started) * 1000)
        probabilities.extend(result.probabilities for result in results)
    digest_source = json.dumps(probabilities, sort_keys=True, separators=(",", ":"))
    report = {
        "budget_mb": args.gpu_memory_budget_mb,
        "samples": len(states),
        "allocator_warmup_disabled": args.disable_allocator_warmup,
        "load_ms": round(load_ms, 3),
        "batch_mean_ms": round(statistics.mean(timings), 3),
        "batch_p95_ms": round(sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))], 3),
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
        "prediction_sha256": hashlib.sha256(digest_source.encode()).hexdigest(),
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
