from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def paired_recall(baseline, candidate, domains, repeats=2000):
    left = {row['id']: row for row in baseline}
    right = {row['id']: row for row in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate) or set(left) != set(right):
        raise ValueError('prediction IDs must match exactly and be unique')
    grouped = defaultdict(list)
    for identifier, row in left.items():
        if row['label'] != right[identifier]['label']:
            raise ValueError('label mismatch')
        grouped[domains[identifier]].append(identifier)
    keys = sorted(grouped)
    rng = random.Random(42)
    report = {}
    for name, pred in (('argmax', lambda r: r['argmax_ad']), ('selective', lambda r: r['verdict'] == 'block'), ('policy', lambda r: r['action'] in ('block', 'hide'))):
        def delta(ids):
            ads = [identifier for identifier in ids if left[identifier]['label'] == 0]
            return sum(int(pred(right[i]))-int(pred(left[i])) for i in ads) / len(ads) if ads else None
        point = delta(list(left))
        values = []
        for _ in range(repeats):
            ids = [identifier for key in rng.choices(keys, k=len(keys)) for identifier in grouped[key]]
            value = delta(ids)
            if value is not None:
                values.append(value)
        values.sort()
        report[name] = {'delta_recall': point, 'domain_bootstrap_95_percentile': [values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]] if values else None, 'valid_resamples': len(values)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    left = json.loads(args.baseline.read_text(encoding='utf-8'))
    right = json.loads(args.candidate.read_text(encoding='utf-8'))
    import hashlib

    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if left['data_sha256'] != digest or right['data_sha256'] != digest:
        raise ValueError('data hashes differ')
    domains = {row['id']: row['source'] for row in map(json.loads, args.data.read_text(encoding='utf-8').splitlines())}
    report = {'usage': 'development_only', 'method': 'paired domain percentile bootstrap, 2000 repeats, seed42; descriptive intervals on a small reused development set', 'variants': {variant: paired_recall(left['variants'][variant]['rows'], right['variants'][variant]['rows'], domains) for variant in left['variants']}}
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
