from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.annotation import freeze_pool, latest_annotations, load_jsonl
from scripts.build_evaluation_pool import overlaps


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_study(directory):
    plan_path = directory/'plan.json'
    manifest = json.loads((directory/'capture-manifest.json').read_text(encoding='utf-8'))
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if manifest['plan_sha256'] != sha256(plan_path) or manifest['pool_sha256'] != sha256(directory/'pool.jsonl'):
        raise ValueError('capture or plan hash changed')
    for name, expected in manifest['screenshots'].items():
        image = directory/'screenshots'/name
        image.resolve().relative_to((directory/'screenshots').resolve())
        if sha256(image) != expected:
            raise ValueError('screenshot changed')
    for model in plan['models'].values():
        for relative, expected in model['files'].items():
            if sha256(ROOT/relative) != expected:
                raise ValueError('frozen model changed')
    for relative, expected in plan['code_sha256'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError('frozen decision code changed')
    for relative, expected in plan.get('runtime_assets', {}).items():
        if sha256(ROOT/relative) != expected:
            raise ValueError('frozen runtime rule asset changed')
    for source in plan.get('capture_sources', []):
        parent = ROOT/source['directory']
        if sha256(parent/'plan.json') != source['plan_sha256'] or sha256(parent/'capture-manifest.json') != source['capture_manifest_sha256']:
            raise ValueError('source preregistration changed')
    pool = load_jsonl(directory/'pool.jsonl')
    identifiers = {row['id'] for row in pool}
    if not pool or len(identifiers) != len(pool):
        raise ValueError('empty pool or duplicate IDs')
    known = set(plan['exclusion_domains'])
    if any(overlaps(row['domain'], known) for row in pool):
        raise ValueError('domain overlap with prior data')
    return plan, pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan, pool = validate_study(directory)
    annotations = latest_annotations(directory/'annotations.jsonl')
    missing = sum(row['id'] not in annotations or annotations[row['id']].get('annotation_mode') != 'human_independent' for row in pool)
    if args.check_only:
        print(json.dumps({'assets_verified': True, 'captured':len(pool), 'domains':len({row['domain'] for row in pool}), 'missing_independent_labels':missing, 'inference_started_by_this_command':False}))
        return
    if missing:
        raise ValueError(f'{missing} candidates still require independent human annotation')
    output = directory/'benchmark.jsonl'
    metadata = directory/'benchmark.metadata.json'
    if output.exists() or metadata.exists():
        raise ValueError('refusing to overwrite frozen benchmark')
    result = freeze_pool(directory/'pool.jsonl', directory/'annotations.jsonl', output, metadata, minimum_examples=plan['minimum_binary_examples'], minimum_domains=plan['minimum_domains'], minimum_per_label=plan['minimum_per_binary_label'], required_labels={'ad', 'non_ad'}, ground_truth_task='binary_ad_detection', allowed_annotation_modes={'human_independent'})
    metadata_value = json.loads(metadata.read_text(encoding='utf-8'))
    metadata_value.update({'study_plan_sha256':sha256(directory/'plan.json'), 'capture_manifest_sha256':sha256(directory/'capture-manifest.json'), 'annotations_sha256':sha256(directory/'annotations.jsonl'), 'complete_pool_independently_reviewed':True})
    metadata.write_text(json.dumps(metadata_value, indent=2), encoding='utf-8')
    print(json.dumps({'frozen':True, 'examples':result['examples'], 'domains':result['domains'], 'inference_started_by_this_command':False}))


if __name__ == '__main__':
    main()
