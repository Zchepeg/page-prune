from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def constant_copy(source, output):
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    for record in records:
        record['state']['direct_resource_urls'] = []
        record['state']['resource_ownership_known'] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records), encoding='utf-8')


def main():
    logs = ROOT/'runs/ownership-replications-v1'
    controlled = ROOT/'data/training/snapshots/ownership-constant-control-v1'
    plan = [('v2-seed7', 7, 'v2'), ('v3-seed7', 7, 'v3'), ('v3-constant-seed42', 42, 'constant')]
    if logs.exists() or controlled.exists() or any((ROOT/f'runs/ownership-{name}').exists() for name, _, _ in plan):
        raise ValueError('refusing to overwrite experiment artifacts')
    for split in ('train', 'validation'):
        constant_copy(ROOT/f'data/training/snapshots/ownership-paired-v1/v3/{split}.jsonl', controlled/'training'/f'{split}.jsonl')
    for split in ('calibration', 'validation'):
        constant_copy(ROOT/f'data/training/snapshots/calibration-supplement-v3/{split}.jsonl', controlled/'heldout'/f'{split}.jsonl')
    logs.mkdir(parents=True)
    configurations = []
    for name, seed, version in plan:
        training = controlled/'training' if version == 'constant' else ROOT/f'data/training/snapshots/ownership-paired-v1/{version}'
        heldout = controlled/'heldout' if version == 'constant' else ROOT/f'data/training/snapshots/calibration-supplement-{version}'
        configurations.append({'name': name, 'seed': seed, 'training': str(training), 'heldout': str(heldout), 'input_hashes': {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in [('training', training/'train.jsonl'), ('early_validation', training/'validation.jsonl'), ('calibration', heldout/'calibration.jsonl'), ('validation', heldout/'validation.jsonl')]}})
    (logs/'plan.json').write_text(json.dumps({'usage': 'reused_development_experiments_not_independent_claims', 'configurations': configurations, 'epochs': 2, 'evidence_dropout': .5, 'max_length': 384, 'constant_control': 'unknown ownership in all train/calibration/evaluation inputs'}, indent=2), encoding='utf-8')
    for config in configurations:
        name = config['name']
        training, heldout = Path(config['training']), Path(config['heldout'])
        checkpoint = ROOT/f'runs/ownership-{name}'
        stages = [
            ('train', ['-m', 'training.train', '--data', str(training/'train.jsonl'), '--validation', str(training/'validation.jsonl'), '--model', 'Qwen/Qwen3-4B-Base', '--label-mode', 'binary_selective', '--epochs', '2', '--patience', '2', '--accum', '8', '--max-length', '384', '--evidence-dropout', '.5', '--feature-dropout', '.1', '--domain-balance-power', '.5', '--label-smoothing', '.05', '--seed', str(config['seed']), '--local-files-only', '--out', str(checkpoint)]),
            ('calibrate', ['-m', 'training.calibrate', '--checkpoint', str(checkpoint), '--data', str(heldout/'calibration.jsonl'), '--quantization', 'nf4', '--gpu-memory-budget-mb', '4096', '--max-length', '384']),
            ('evaluate', ['scripts/compare_human_baseline.py', '--checkpoint', str(checkpoint), '--data', str(heldout/'validation.jsonl'), '--out', str(logs/f'{name}-validation.json')]),
            ('temperature1', ['scripts/compare_human_baseline.py', '--checkpoint', str(checkpoint), '--data', str(heldout/'validation.jsonl'), '--temperature', '1', '--out', str(logs/f'{name}-temp1-validation.json')]),
        ]
        for stage, command in stages:
            print(json.dumps({'name': name, 'stage': stage, 'status': 'started'}), flush=True)
            with (logs/f'{name}-{stage}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} {stage} failed; inspect log')
            print(json.dumps({'name': name, 'stage': stage, 'status': 'completed'}), flush=True)


if __name__ == '__main__':
    main()
