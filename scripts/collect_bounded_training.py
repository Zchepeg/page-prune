from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'runs/ownership-training-bounded-v1'
    if output.exists():
        raise ValueError('refusing to overwrite capture')
    output.mkdir(parents=True)
    (output/'screenshots').mkdir()
    sites = 'publicbooks.org budgetbytes.com dezeen.com archdaily.com design-milk.com itsnicethat.com creativeboom.com hyperallergic.com hackaday.com circuitdigest.com electronics-lab.com sparkfun.com adafruit.com instructables.com'.split()
    exclusions = list((ROOT/'data/training/snapshots').glob('*/*.jsonl')) + list((ROOT/'data/training').glob('*.jsonl')) + list((ROOT/'data/evaluation').glob('independent_test*.jsonl')) + list((ROOT/'runs').glob('calibration*/pool.jsonl'))
    started = time.monotonic()
    rows, seen, pages = [], set(), []
    for index, site in enumerate(sites):
        remaining = 180 - (time.monotonic()-started)
        if remaining < 10:
            break
        part = output/'parts'/str(index)/'pool.jsonl'
        command = [sys.executable, str(ROOT/'scripts/build_evaluation_pool.py'), 'https://'+site+'/', '--config', str(ROOT/'config/local.yaml'), '--output', str(part), '--target', '8', '--per-domain', '8', '--timeout-ms', '8000', '--seed', str(20261031+index)]
        for path in exclusions:
            command.extend(['--exclude-data', str(path)])
        status = {'site': site}
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=min(30, remaining), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            status['exit_code'] = result.returncode
            status['stderr_tail'] = result.stderr.decode('utf-8', errors='replace')[-800:]
        except subprocess.TimeoutExpired:
            status['timeout_seconds'] = min(30, remaining)
        if part.exists():
            for line in part.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if row['id'] in seen:
                    continue
                seen.add(row['id'])
                image = Path(row['review_context'].get('screenshot', ''))
                if image.is_file():
                    image.resolve().relative_to((part.parent/'screenshots').resolve())
                    target = output/'screenshots'/f"{row['id']}.png"
                    shutil.copy2(image, target)
                    row['review_context']['screenshot'] = str(target.resolve())
                row['provenance'] = {'purpose': 'ownership_training_capture', 'label_source': 'unreviewed_capture'}
                rows.append(row)
        status['captured_total'] = len(rows)
        pages.append(status)
        (output/'pool.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')
        report = {'examples': len(rows), 'domains': len({row['domain'] for row in rows}), 'budget_seconds': 180, 'pages': pages, 'labels': 'none'}
        (output/'collection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(status), flush=True)
        if len(rows) >= 96:
            break


if __name__ == '__main__':
    main()
