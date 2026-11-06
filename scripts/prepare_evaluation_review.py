from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.annotation import load_jsonl
from evaluation.preparation import apply_qwen_prelabels, prepare_records, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=ROOT / "runs/evaluation-annotation/pool.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/evaluation-annotation/pool_curated.jsonl")
    parser.add_argument("--prelabels", type=Path, default=ROOT / "runs/evaluation-annotation/prelabels.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "runs/evaluation-annotation/preparation_report.json")
    parser.add_argument("--qwen-config", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = load_jsonl(args.pool)
    curated, prelabels, report = prepare_records(pool, args.pool.parent / "screenshots")
    if args.qwen_config:
        report.update(apply_qwen_prelabels(curated, prelabels, args.qwen_config))
    report["prepared_at"] = datetime.now(timezone.utc).isoformat()
    report["pool"] = str(args.output.resolve())
    report["prelabels"] = str(args.prelabels.resolve())
    write_jsonl(args.output, curated)
    write_jsonl(args.prelabels, prelabels)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
