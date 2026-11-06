from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.schemas import Candidate
from evaluation.benchmark import BenchmarkCase, domain_leakage, training_domains
from service.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pools', nargs='+', required=True, type=Path)
    parser.add_argument('--out', default='runs/calibration-supplement-v1', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('refusing to overwrite an annotation directory')
    config = load_config(ROOT / 'config/local.yaml')
    excluded = training_domains([ROOT / p for p in config['auto_training']['training_sources']] + list((ROOT / 'data/training/snapshots').glob('*/*.jsonl')) + list((ROOT / 'data/training').glob('*.jsonl')))
    for path in (ROOT / 'data/evaluation').glob('independent_test*.jsonl'):
        excluded.update(json.loads(line)['domain'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    records, identifiers, images = [], set(), []
    for pool in args.pools:
        capture_hash = hashlib.sha256(pool.read_bytes()).hexdigest()
        for line in pool.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record['id'] in identifiers:
                raise ValueError('duplicate candidate ID')
            candidate = Candidate.from_dict(record['candidate'])
            if candidate.page.top_level_domain != record['domain']:
                raise ValueError('candidate domain mismatch')
            if not domain_leakage([BenchmarkCase(record['id'], record['domain'], 1, candidate)], excluded)['domain_disjoint']:
                raise ValueError('domain overlaps supervision or benchmark')
            if 'label' in record:
                raise ValueError('merge expects unlabelled capture records')
            identifiers.add(record['id'])
            raw_image = record.get('review_context', {}).get('screenshot', '')
            if raw_image:
                source = Path(raw_image).resolve()
                source.relative_to((pool.parent / 'screenshots').resolve())
                if source.is_file():
                    target = args.out / 'screenshots' / f"{record['id']}.png"
                    images.append((source, target))
                    record['review_context']['screenshot'] = str(target.resolve())
                else:
                    record['review_context']['screenshot'] = ''
            record['provenance'] = {'purpose': 'calibration_supplement', 'label_source': 'unreviewed_capture', 'capture_sha256': capture_hash}
            records.append(record)
    counts = Counter(record['domain'] for record in records)
    if max(counts.values(), default=0) > 8:
        raise ValueError('per-domain cap exceeded')
    (args.out / 'screenshots').mkdir(parents=True)
    for source, target in images:
        shutil.copy2(source, target)
    output = args.out / 'pool.jsonl'
    output.write_text(''.join(json.dumps(record, ensure_ascii=False)+'\n' for record in records), encoding='utf-8')
    report = {'examples': len(records), 'domains': len(counts), 'screenshots': len(images), 'pool_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'domain_disjoint': True, 'labels': 'none', 'ready_for_calibration': False}
    (args.out / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
