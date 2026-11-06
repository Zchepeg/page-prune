from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.freeze_fresh_blind import validate_study
from scripts.build_evaluation_pool import write_records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite merged study')
    studies = [(source.resolve(), *validate_study(source.resolve())) for source in args.sources]
    original = studies[0][1]
    for _, plan, _ in studies[1:]:
        for key in ('models', 'policy', 'detector', 'protocol', 'jev_thresholds', 'code_sha256', 'exclusion_domains'):
            if original[key] != plan[key]:
                raise ValueError('incompatible preregistration: '+key)
    sources = []
    seen_ids, seen_domains = set(), set()
    rows = []
    for directory, _, pool in studies:
        if (directory/'annotations.jsonl').exists():
            raise ValueError('merge must precede annotation')
        accepted_domains = {row['domain'] for row in pool} - seen_domains
        for row in pool:
            if row['domain'] in accepted_domains and row['id'] not in seen_ids:
                rows.append((directory, row))
                seen_ids.add(row['id'])
        seen_domains.update(accepted_domains)
        sources.append({'directory':str(directory.relative_to(ROOT)), 'plan_sha256':digest(directory/'plan.json'), 'capture_manifest_sha256':digest(directory/'capture-manifest.json')})
    output = args.output.resolve()
    screenshots = output/'screenshots'
    screenshots.mkdir(parents=True)
    merged = []
    for directory, row in rows:
        image = Path(row['review_context'].get('screenshot', ''))
        if image.is_file():
            image.resolve().relative_to((directory/'screenshots').resolve())
            destination = screenshots/f"{row['id']}.png"
            shutil.copy2(image, destination)
            row['review_context']['screenshot'] = str(destination)
        merged.append(row)
    plan = dict(original)
    plan.update({'assembled_at':datetime.now(timezone.utc).isoformat(), 'status':'assembled_from_preregistered_unlabelled_batches', 'capture_sources':sources, 'merge_policy':'all unique domains, first source takes precedence for duplicate domains; no labels or predictions used'})
    plan['runtime_assets'] = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((ROOT/'data/runtime').glob('*.json'))}
    plan['runtime_assets_registration'] = 'frozen at unlabelled pool assembly, before human labels or any model inference; not asserted to have been registered before capture'
    (output/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    write_records(output/'pool.jsonl', merged)
    manifest = {'plan_sha256':digest(output/'plan.json'), 'pool_sha256':digest(output/'pool.jsonl'), 'screenshots':{p.name:digest(p) for p in sorted(screenshots.glob('*.png'))}, 'captured':len(merged), 'domains':len(seen_domains), 'annotations':0, 'inference_executed':False}
    (output/'capture-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'merged':len(merged), 'domains':len(seen_domains), 'screenshots':len(manifest['screenshots']), 'inference_executed':False}))


if __name__ == '__main__':
    main()
