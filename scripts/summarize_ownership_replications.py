from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_domain_bootstrap import paired_recall


def main():
    directory = ROOT/'runs/ownership-replications-v1'
    previous = ROOT/'runs/ownership-paired-experiment-v1'
    paths = {'v2-seed42': previous/'v2-validation.json', 'v3-seed42': previous/'v3-validation.json', **{name: directory/f'{name}-validation.json' for name in ('v2-seed7', 'v3-seed7', 'v3-constant-seed42')}}
    data = [json.loads(line) for line in (ROOT/'data/training/snapshots/calibration-supplement-v3/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    labels = {row['id']: row['question']['label'] for row in data}
    domains = {row['id']: row['source'] for row in data}
    report = {'usage': 'reused_development_only', 'seeds': [42, 7], 'training_seed_count': 2, 'examples': len(data), 'models': {}, 'comparisons': {}}
    loaded = {}
    for mode in ('calibrated', 'temperature1'):
        loaded[mode] = {}
        for name, original in paths.items():
            path = original if mode == 'calibrated' else original.with_name(original.name.replace('-validation.json', '-temp1-validation.json'))
            raw = json.loads(path.read_text(encoding='utf-8'))
            for variant in raw['variants'].values():
                if {row['id']: row['label'] for row in variant['rows']} != labels:
                    raise ValueError('prediction IDs or labels differ')
            loaded[mode][name] = raw
            report['models'].setdefault(name, {})[mode] = {'temperature': raw['temperature'], 'variants': {variant: value['summary'] for variant, value in raw['variants'].items()}}
        comparisons = [('v3_minus_v2_seed42', 'v2-seed42', 'v3-seed42'), ('v3_minus_v2_seed7', 'v2-seed7', 'v3-seed7'), ('constant_minus_real_seed42', 'v3-seed42', 'v3-constant-seed42')]
        report['comparisons'][mode] = {name: {variant: paired_recall(loaded[mode][left]['variants'][variant]['rows'], loaded[mode][right]['variants'][variant]['rows'], domains) for variant in ('with_evidence', 'without_evidence')} for name, left, right in comparisons}
    (directory/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, modes in report['models'].items():
        for mode, values in modes.items():
            print(json.dumps({'name': name, 'mode': mode, 'temperature': values['temperature'], 'results': {k: {metric: v[metric] for metric in ('argmax', 'policy', 'nll', 'truncated')} for k, v in values['variants'].items()}}))


if __name__ == '__main__':
    main()
