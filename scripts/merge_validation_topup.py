from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def main():
    original = Path('runs/calibration-supplement-v1')
    topup = Path('runs/calibration-validation-topup-v1')
    output = Path('runs/calibration-supplement-v2')
    if output.exists():
        raise ValueError('refusing to overwrite annotation directory')
    rows = [json.loads(line) for line in (original/'pool.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    domains = sorted({row['domain'] for row in rows}, key=lambda d: hashlib.sha256(('calibration-supplement-v1:'+d).encode()).hexdigest())
    validation = set(domains[::3])
    identifiers = {row['id'] for row in rows}
    additions = []
    for line in (topup/'pool.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['domain'] not in validation:
            raise ValueError('topup outside fixed validation domains')
        if row['id'] in identifiers:
            continue
        identifiers.add(row['id'])
        row['provenance'] = {'purpose': 'fixed_validation_topup', 'label_source': 'unreviewed_capture'}
        additions.append(row)
    if not additions:
        raise ValueError('no unique additions')
    (output/'screenshots').mkdir(parents=True)
    for row in rows + additions:
        raw = row.get('review_context', {}).get('screenshot', '')
        if not raw:
            continue
        image = Path(raw).resolve()
        if image.parent not in {(original/'screenshots').resolve(), (topup/'screenshots').resolve()}:
            raise ValueError('screenshot outside capture directory')
        if image.is_file():
            destination = output/'screenshots'/f"{row['id']}.png"
            shutil.copy2(image, destination)
            row['review_context']['screenshot'] = str(destination.resolve())
        else:
            row['review_context']['screenshot'] = ''
    (output/'pool.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows+additions), encoding='utf-8')
    shutil.copy2(original/'annotations.jsonl', output/'annotations.jsonl')
    report = {'original_examples': len(rows), 'unique_additions': len(additions), 'total': len(rows)+len(additions), 'domains': len(domains), 'validation_domains': sorted(validation), 'calibration_domains': sorted(set(domains)-validation), 'addition_domains': dict(Counter(row['domain'] for row in additions)), 'split_unchanged': True, 'original_annotations_sha256': hashlib.sha256((original/'annotations.jsonl').read_bytes()).hexdigest(), 'sampling_note': 'fixed validation-domain topup after class shortfall; development only, not a blind benchmark'}
    (output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
