from __future__ import annotations

import argparse
import hashlib
from collections import Counter
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer

from modeling.decision_model import QwenDecisionModel
from modeling.encoding import encode_example
from .augment import augment_example
from .dataset import load_jsonl
from jev.questions import AD_TREATMENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the local Qwen ad-decision model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--validation", default="")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--revision", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--pointer-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--label-mode", choices=("binary_selective", "three_class"), default="binary_selective")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evidence-dropout", type=float, default=0.5)
    parser.add_argument("--feature-dropout", type=float, default=0.1)
    parser.add_argument("--domain-balance-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.lr, args.head_lr, args.lora_rank, args.pointer_dim, args.accum, args.patience) <= 0:
        parser.error("training values must be positive")
    if not 0 <= args.evidence_dropout <= 1 or not 0 <= args.feature_dropout <= 1:
        parser.error("dropout values must be between 0 and 1")
    if not 0 <= args.domain_balance_power <= 1:
        parser.error("domain balance power must be between 0 and 1")
    if not 0 <= args.label_smoothing < 1:
        parser.error("label smoothing must be between 0 and 1")
    if Path(args.out).exists():
        parser.error("refusing to overwrite an existing output directory")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3-4B training profile")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("this training profile requires BF16 support")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    examples = load_jsonl(args.data, args.label_mode, AD_TREATMENT.text)
    schemas = {json.loads(example.state)["schema_version"] for example in examples}
    if len(schemas) != 1:
        raise ValueError("training data must use a single state schema")
    state_schema = schemas.pop()
    validation = load_jsonl(args.validation, args.label_mode, AD_TREATMENT.text, state_schema) if args.validation else []
    domain_counts = Counter(example.source for example in examples)
    domain_weights = {
        domain: count ** (-args.domain_balance_power)
        for domain, count in domain_counts.items()
    }
    weight_scale = len(examples) / sum(domain_weights[example.source] for example in examples)
    model_source = args.model
    if args.local_files_only and not Path(model_source).exists():
        from huggingface_hub import snapshot_download

        model_source = snapshot_download(
            repo_id=args.model,
            revision=args.revision or None,
            local_files_only=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=args.revision or None,
        local_files_only=args.local_files_only,
    )
    model = QwenDecisionModel.create_for_training(
        model_source,
        revision=args.revision or None,
        lora_rank=args.lora_rank,
        pointer_dim=args.pointer_dim,
        dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to("cuda")
    model.train()

    adapter_params = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    # Each epoch flushes its final partial accumulation, so rounding must occur
    # per epoch rather than once across all epochs.
    total_updates = math.ceil(len(examples) / args.accum) * args.epochs
    started = time.time()
    update = 0
    best_epoch = 0
    best_validation_loss = float("inf")
    best_adapter = None
    best_head = None
    stale_epochs = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        running_loss = 0.0
        for step, index in enumerate(order, start=1):
            example = augment_example(
                examples[index],
                rng,
                args.evidence_dropout,
                args.feature_dropout,
            )
            if example.label is None:
                raise ValueError(f"training example {example.example_id or index} has no label")
            encoded = encode_example(tokenizer, example, args.max_length)
            logits = model.forward_encoded(encoded)
            target = torch.tensor([example.label], device=logits.device)
            group_start = ((step - 1) // args.accum) * args.accum
            group_size = min(args.accum, len(order) - group_start)
            sample_weight = domain_weights[example.source] * weight_scale
            loss = F.cross_entropy(
                logits[None],
                target,
                label_smoothing=args.label_smoothing,
            ) * sample_weight / group_size
            loss.backward()
            running_loss += loss.item() * group_size
            if step % args.accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "update": update,
                            "updates": total_updates,
                            "loss": round(running_loss / step, 6),
                            "gpu_mb": round(torch.cuda.max_memory_allocated() / 1024**2),
                        }
                    ),
                    flush=True,
                )
        if validation:
            model.eval()
            validation_loss = 0.0
            with torch.inference_mode():
                for example in validation:
                    encoded = encode_example(tokenizer, example, args.max_length)
                    logits = model.forward_encoded(encoded)
                    target = torch.tensor([example.label], device=logits.device)
                    validation_loss += float(F.cross_entropy(logits[None], target))
            validation_loss /= len(validation)
            print(json.dumps({"epoch": epoch + 1, "validation_loss": round(validation_loss, 6)}), flush=True)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                best_adapter = {
                    key: value.detach().cpu().clone()
                    for key, value in get_peft_model_state_dict(model.backbone).items()
                }
                best_head = {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            model.train()
            if stale_epochs >= args.patience:
                break

    if best_adapter is not None and best_head is not None:
        set_peft_model_state_dict(model.backbone, best_adapter)
        model.head.load_state_dict(best_head)

    metadata = {
        "seed": args.seed,
        "training_data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "validation_data_sha256": hashlib.sha256(Path(args.validation).read_bytes()).hexdigest() if args.validation else None,
        "state_schema": state_schema,
        "instruction": AD_TREATMENT.text,
        "instruction_version": "checkpoint-bound-v1",
        "architecture": "qwen-pointer-selective-v2" if args.label_mode == "binary_selective" else "qwen-pointer-decision-v1",
        "base_model": args.model,
        "requested_model": args.model,
        "base_revision": args.revision,
        "lora_rank": args.lora_rank,
        "pointer_dim": args.pointer_dim,
        "max_length": args.max_length,
        "labels": ["ad", "non_ad"] if args.label_mode == "binary_selective" else ["ad", "non_ad", "insufficient_evidence"],
        "label_mode": args.label_mode,
        "training_examples": len(examples),
        "validation_examples": len(validation),
        "best_epoch": best_epoch or args.epochs,
        "best_validation_loss": None if not validation else best_validation_loss,
        "epochs": args.epochs,
        "elapsed_seconds": round(time.time() - started, 3),
        "evidence_dropout": args.evidence_dropout,
        "feature_dropout": args.feature_dropout,
        "domain_balance_power": args.domain_balance_power,
        "label_smoothing": args.label_smoothing,
    }
    model.save_checkpoint(args.out, metadata)
    tokenizer.save_pretrained(Path(args.out) / "tokenizer")
    print(json.dumps({"saved": args.out, **metadata}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
