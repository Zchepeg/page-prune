from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for version in ('v2', 'v3'):
        if (ROOT/f'runs/ownership-paired-{version}-v1').exists():
            raise ValueError('refusing to overwrite a candidate')
    logs = ROOT/'runs/ownership-paired-experiment-v1'
    if logs.exists():
        raise ValueError('refusing to overwrite experiment logs')
    logs.mkdir(parents=True)
    for version in ('v2', 'v3'):
        snapshot = f'data/training/snapshots/ownership-paired-v1/{version}'
        heldout = 'data/training/snapshots/calibration-supplement-'+('v2' if version == 'v2' else 'v3')
        checkpoint = f'runs/ownership-paired-{version}-v1'
        stages = [
            ('train', ['-m', 'training.train', '--data', snapshot+'/train.jsonl', '--validation', snapshot+'/validation.jsonl', '--model', 'Qwen/Qwen3-4B-Base', '--label-mode', 'binary_selective', '--epochs', '2', '--patience', '2', '--accum', '8', '--max-length', '384', '--evidence-dropout', '.5', '--feature-dropout', '.1', '--domain-balance-power', '.5', '--label-smoothing', '.05', '--local-files-only', '--out', checkpoint]),
            ('calibrate', ['-m', 'training.calibrate', '--checkpoint', checkpoint, '--data', heldout+'/calibration.jsonl', '--quantization', 'nf4', '--gpu-memory-budget-mb', '4096', '--max-length', '384']),
            ('evaluate', ['scripts/compare_human_baseline.py', '--checkpoint', checkpoint, '--data', heldout+'/validation.jsonl', '--out', f'runs/ownership-paired-experiment-v1/{version}-validation.json']),
            ('temperature1', ['scripts/compare_human_baseline.py', '--checkpoint', checkpoint, '--data', heldout+'/validation.jsonl', '--temperature', '1', '--out', f'runs/ownership-paired-experiment-v1/{version}-temp1-validation.json']),
        ]
        for stage, command in stages:
            print(json.dumps({'version': version, 'stage': stage, 'status': 'started'}), flush=True)
            with (logs/f'{version}-{stage}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{version} {stage} failed; inspect experiment log')
            print(json.dumps({'version': version, 'stage': stage, 'status': 'completed'}), flush=True)


if __name__ == '__main__':
    main()
