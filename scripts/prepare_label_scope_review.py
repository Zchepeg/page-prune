from __future__ import annotations

import json
import shutil
from pathlib import Path


def main():
    source = Path('runs/calibration-supplement-v2')
    output = Path('runs/label-scope-review-v1')
    if output.exists():
        raise ValueError('refusing to overwrite review directory')
    identifier = 'eval-f60697bc279d78aff2b0'
    record = next(row for row in map(json.loads, (source/'pool.jsonl').read_text(encoding='utf-8').splitlines()) if row['id'] == identifier)
    screenshot = Path(record['review_context']['screenshot']).resolve()
    screenshot.relative_to((source/'screenshots').resolve())
    (output/'screenshots').mkdir(parents=True)
    target = output/'screenshots'/f'{identifier}.png'
    shutil.copy2(screenshot, target)
    record['review_context']['screenshot'] = str(target.resolve())
    (output/'pool.jsonl').write_text(json.dumps(record, ensure_ascii=False)+'\n', encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps({'purpose': 'label_scope_recheck', 'examples': 1, 'original_labels_unchanged': True, 'automatic_label_correction': False}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
