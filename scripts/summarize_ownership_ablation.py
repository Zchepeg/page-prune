from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_domain_bootstrap import paired_recall


def main():
    directory = ROOT/'runs/ownership-paired-experiment-v1'
    reports = {}
    for version in ('v2', 'v3'):
        data = ROOT/f'data/training/snapshots/calibration-supplement-{version}/validation.jsonl'
        digest = hashlib.sha256(data.read_bytes()).hexdigest()
        for mode, suffix in (('calibrated', ''), ('temperature1', '-temp1')):
            path = directory/f'{version}{suffix}-validation.json'
            report = json.loads(path.read_text(encoding='utf-8'))
            if report['data_sha256'] != digest:
                raise ValueError('validation data changed')
            reports[version, mode] = report
    rows = [json.loads(line) for line in (ROOT/'data/training/snapshots/calibration-supplement-v2/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    domains = {row['id']: row['source'] for row in rows}
    output = {'usage': 'reused_development_set_only', 'examples': len(rows), 'domains': len(set(domains.values())), 'single_training_seed': True, 'modes': {}}
    for mode in ('temperature1', 'calibrated'):
        left, right = reports['v2', mode], reports['v3', mode]
        output['modes'][mode] = {'temperatures': {'v2': left['temperature'], 'v3': right['temperature']}, 'variants': {}}
        for variant in ('with_evidence', 'without_evidence'):
            a, b = left['variants'][variant], right['variants'][variant]
            if {row['id'] for row in a['rows']} != set(domains):
                raise ValueError('prediction coverage mismatch')
            output['modes'][mode]['variants'][variant] = {'v2': a['summary'], 'v3': b['summary'], 'paired_domain_delta_v3_minus_v2': paired_recall(a['rows'], b['rows'], domains)}
    (directory/'summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
