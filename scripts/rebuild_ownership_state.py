from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collector.schemas import Candidate
from collector.state_builder import OWNERSHIP_SCHEMA_VERSION, build_state_payload


def rebuild(snapshot, raw_path, output):
    if output.exists():
        raise ValueError('refusing to overwrite snapshot')
    raw = {}
    for line in raw_path.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        if record['id'] in raw:
            raise ValueError('duplicate raw ID')
        raw[record['id']] = record
    report = {'state_schema': OWNERSHIP_SCHEMA_VERSION, 'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(), 'splits': {}, 'usage': 'experimental_reconstruction_not_new_labels'}
    pending = {}
    for source in sorted(snapshot.glob('*.jsonl')):
        records, counts = [], Counter()
        for line in source.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            candidate = Candidate.from_dict(raw[record['id']]['candidate'])
            if candidate.page.top_level_domain != record['source']:
                raise ValueError('domain mismatch')
            record['state'] = build_state_payload(candidate, record['state'].get('detector_evidence', []), OWNERSHIP_SCHEMA_VERSION)
            counts['known' if record['state']['resource_ownership_known'] else 'unknown'] += 1
            records.append(record)
        pending[source.name] = records
        report['splits'][source.stem] = {'examples': len(records), 'domains': len({r['source'] for r in records}), 'labels': dict(Counter(r['question']['label'] for r in records)), 'ownership': dict(counts), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    if not pending:
        raise ValueError('empty snapshot')
    output.mkdir(parents=True)
    for name, records in pending.items():
        (output/name).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records), encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rebuild(args.snapshot, args.raw, args.out), indent=2))
