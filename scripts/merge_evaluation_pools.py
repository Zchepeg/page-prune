from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.annotation import LABELS, load_jsonl
from evaluation.benchmark import training_domains
from evaluation.preparation import candidate_fingerprint, write_jsonl
from scripts.build_evaluation_pool import overlaps
from service.config import load_config


def merge_pools(inputs: list[Path], output: Path, excluded_domains: set[str]) -> dict[str, object]:
    screenshots = output.parent / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    accepted = []
    seen_ids = set()
    seen_fingerprints = set()
    rejected = Counter()
    for source in inputs:
        root = (source.resolve().parent / "screenshots").resolve()
        for item in load_jsonl(source):
            identifier = str(item.get("id", ""))
            domain = str(item.get("domain", "")).lower().strip()
            if not identifier or not domain or not isinstance(item.get("candidate"), dict):
                rejected["invalid_record"] += 1
                continue
            if overlaps(domain, excluded_domains):
                rejected["excluded_domain"] += 1
                continue
            fingerprint = candidate_fingerprint(item)
            if identifier in seen_ids or fingerprint in seen_fingerprints:
                rejected["duplicate_candidate"] += 1
                continue
            seen_ids.add(identifier)
            seen_fingerprints.add(fingerprint)
            screenshot_value = str(item.get("review_context", {}).get("screenshot", ""))
            target = screenshots / f"{identifier}.png"
            if screenshot_value:
                try:
                    screenshot = Path(screenshot_value).resolve(strict=True)
                    screenshot.relative_to(root)
                    shutil.copy2(screenshot, target)
                    item.setdefault("review_context", {})["screenshot"] = str(target.resolve())
                except (OSError, ValueError):
                    item.setdefault("review_context", {})["screenshot"] = ""
                    rejected["missing_screenshot"] += 1
            accepted.append(item)
    accepted.sort(key=lambda item: (str(item["domain"]), str(item["id"])))
    write_jsonl(output, accepted)
    return {
        "input_examples": sum(len(load_jsonl(path)) for path in inputs),
        "examples": len(accepted),
        "domains": len({str(item["domain"]) for item in accepted}),
        "rejected": dict(sorted(rejected.items())),
        "output": str(output.resolve()),
    }


def merge_annotations(inputs: list[Path], pool_path: Path, output: Path) -> dict[str, object]:
    pool = load_jsonl(pool_path)
    valid_ids = {str(item.get("id", "")) for item in pool}
    latest = {}
    rejected = Counter()
    for source in inputs:
        for item in load_jsonl(source):
            identifier = str(item.get("id", ""))
            label = str(item.get("label", ""))
            if identifier not in valid_ids:
                rejected["unknown_candidate"] += 1
                continue
            if label not in LABELS:
                rejected["invalid_label"] += 1
                continue
            latest[identifier] = item
    ordered = [latest[str(item["id"])] for item in pool if str(item["id"]) in latest]
    write_jsonl(output, ordered)
    return {
        "annotations": len(ordered),
        "labels": dict(sorted(Counter(str(item["label"]) for item in ordered).items())),
        "annotation_modes": dict(sorted(Counter(str(item.get("annotation_mode", "")) for item in ordered).items())),
        "rejected_annotations": dict(sorted(rejected.items())),
        "annotations_output": str(output.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    parser.add_argument("--exclude-data", type=Path, action="append", default=[])
    parser.add_argument("--min-examples", type=int, default=300)
    parser.add_argument("--min-domains", type=int, default=30)
    parser.add_argument("--annotations", type=Path, nargs="+")
    parser.add_argument("--annotations-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sources = [ROOT / value for value in config.get("auto_training", {}).get("training_sources", [])]
    excluded = training_domains(sources)
    for path in args.exclude_data:
        for item in load_jsonl(path):
            domain = str(item.get("domain", item.get("source", ""))).lower().strip()
            if domain:
                excluded.add(domain)
    report = merge_pools(args.inputs, args.output, excluded)
    if bool(args.annotations) != bool(args.annotations_output):
        raise ValueError("--annotations and --annotations-output must be provided together")
    if args.annotations:
        report.update(merge_annotations(args.annotations, args.output, args.annotations_output))
    report_path = args.output.parent / "merge_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if int(report["examples"]) >= args.min_examples and int(report["domains"]) >= args.min_domains else 1


if __name__ == "__main__":
    raise SystemExit(main())
