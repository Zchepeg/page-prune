from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.paired_statistics import paired_statistics
from evaluation.benchmark import load_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite statistics')
    source = json.loads(args.report.read_text(encoding='utf-8'))
    digest = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    if digest != source['benchmark_sha256']:
        raise ValueError('benchmark hash mismatch')
    truth = {row.case_id: row for row in load_benchmark(args.benchmark)}
    for mode in source['modes'].values():
        rows = mode['rows']
        if len(truth) != len(rows) or {row['id'] for row in rows} != set(truth):
            raise ValueError('benchmark prediction IDs mismatch')
        if any(row['label'] != truth[row['id']].label or row['domain'] != truth[row['id']].domain for row in rows):
            raise ValueError('benchmark truth mismatch')
    result = {'benchmark_sha256': digest, 'prediction_report_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(), 'scope': 'retrospective uncertainty analysis of historical frozen predictions, not new model evaluation', 'comparisons': {name: paired_statistics(source['modes']['rules_only']['rows'], source['modes'][name]['rows'], args.repeats) for name in ('rule_model_cascade', 'jev_ad_protocol')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
