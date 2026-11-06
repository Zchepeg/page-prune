from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.reference_labeler import (
    augment_reference_ads,
    label_candidates,
    load_reference,
    train_reference_model,
    write_dataset,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reference-guided binary training dataset")
    parser.add_argument("--reference-pool", type=Path, default=ROOT / "runs/evaluation-annotation/pool_curated.jsonl")
    parser.add_argument("--annotations", type=Path, default=ROOT / "runs/evaluation-annotation/annotations.jsonl")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--precision-target", type=float, default=0.95)
    parser.add_argument("--max-ensemble-std", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance-counterfactuals", action="store_true")
    parser.add_argument("--vendors", type=Path, default=ROOT / "data/runtime/tracker_vendors.json")
    parser.add_argument("--exclude-domain-data", type=Path, action="append", default=[])
    args = parser.parse_args()
    reference = load_reference(args.reference_pool, args.annotations)
    pool = load_jsonl(args.pool)
    excluded_domains = set()
    for path in args.exclude_domain_data:
        for item in load_jsonl(path):
            domain = str(item.get("domain", item.get("source", ""))).lower().strip()
            if domain:
                excluded_domains.add(domain)
    pool = [
        item
        for item in pool
        if not any(
            str(item["domain"]).lower() == domain
            or str(item["domain"]).lower().endswith("." + domain)
            or domain.endswith("." + str(item["domain"]).lower())
            for domain in excluded_domains
        )
    ]
    reference_domains = {str(item["domain"]).lower() for item in reference}
    pool_domains = {str(item["domain"]).lower() for item in pool}
    overlap = {
        domain
        for domain in pool_domains
        if any(
            domain == reference_domain
            or domain.endswith("." + reference_domain)
            or reference_domain.endswith("." + domain)
            for reference_domain in reference_domains
        )
    }
    if overlap:
        raise ValueError(f"new pool overlaps reference domains: {', '.join(sorted(overlap))}")
    model = train_reference_model(reference, seed=args.seed, precision_target=args.precision_target)
    accepted, review, report = label_candidates(model, pool, args.max_ensemble_std)
    if args.balance_counterfactuals:
        vendors = json.loads(args.vendors.read_text(encoding="utf-8"))
        hosts = sorted(
            {
                str(host)
                for vendor_hosts in vendors.get("advertising_network", {}).values()
                for host in vendor_hosts
            }
        )
        labels = {0: 0, 1: 0}
        for record in accepted:
            labels[int(record["question"]["label"])] += 1
        augmented = augment_reference_ads(
            reference,
            sorted(pool_domains),
            hosts,
            max(0, labels[1] - labels[0]),
            args.seed,
        )
        accepted.extend(augmented)
        report["counterfactual_ads"] = len(augmented)
        report["accepted_examples"] = len(accepted)
        report["accepted_domains"] = len({str(item["source"]) for item in accepted})
        report["training_labels"] = {
            "ad": sum(int(item["question"]["label"]) == 0 for item in accepted),
            "non_ad": sum(int(item["question"]["label"]) == 1 for item in accepted),
        }
    write_dataset(args.out, accepted)
    write_dataset(args.review_out, review)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
