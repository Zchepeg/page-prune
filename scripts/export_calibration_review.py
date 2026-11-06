from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.schemas import Candidate
from collector.state_builder import build_state_payload
from detector.candidate_ranker import CandidateRanker
from evaluation.annotation import latest_annotations, load_jsonl, pool_status
from evaluation.benchmark import BenchmarkCase, domain_leakage, training_domains
from jev.questions import AD_TREATMENT
from service.config import load_config


def prepare(pool, annotations, excluded):
    ids = [row['id'] for row in pool]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate pool IDs')
    domains = sorted({row['domain'] for row in pool}, key=lambda d: hashlib.sha256(('calibration-supplement-v1:'+d).encode()).hexdigest())
    validation_domains = set(domains[::3])
    partitions = {'calibration': [], 'validation': []}
    reasons = []
    if any(identifier not in annotations for identifier in ids):
        reasons.append('incomplete_human_review')
    ranker = CandidateRanker()
    for row in pool:
        candidate = Candidate.from_dict(row['candidate'])
        if candidate.page.top_level_domain != row['domain']:
            raise ValueError('candidate domain mismatch')
        if not domain_leakage([BenchmarkCase(row['id'], row['domain'], 1, candidate)], excluded)['domain_disjoint']:
            raise ValueError('domain overlaps existing supervision or benchmark')
        annotation = annotations.get(row['id'])
        if annotation is None:
            continue
        if annotation.get('annotation_mode') != 'human_independent':
            reasons.append('non_independent_annotation')
            continue
        if annotation['label'] == 'insufficient_evidence':
            continue
        if annotation['label'] not in ('ad', 'non_ad'):
            raise ValueError('invalid label')
        split = 'validation' if row['domain'] in validation_domains else 'calibration'
        partitions[split].append({'id': row['id'], 'source': row['domain'], 'state': build_state_payload(candidate, ranker.assess(candidate).evidence), 'question': {'instruction': AD_TREATMENT.text, 'options': ['ad', 'non_ad'], 'label': int(annotation['label'] == 'non_ad')}, 'provenance': {'label_source': 'human_independent', 'purpose': 'calibration_development', 'split': split, 'annotated_at': annotation.get('annotated_at', '')}})
    stats = {}
    for split, records in partitions.items():
        labels = Counter(record['question']['label'] for record in records)
        stats[split] = {'examples': len(records), 'domains': len({record['source'] for record in records}), 'labels': dict(labels)}
        if min(labels.get(0, 0), labels.get(1, 0)) < 15:
            reasons.append(split+'_requires_15_per_class')
        if stats[split]['domains'] < (10 if split == 'calibration' else 5):
            reasons.append(split+'_not_enough_domains')
    fit_domains = {record['source'] for record in partitions['calibration']}
    if any(not domain_leakage([BenchmarkCase(record['id'], record['source'], 1, Candidate.from_dict({'page': {'top_level_domain': record['source']}}))], fit_domains)['domain_disjoint'] for record in partitions['validation']):
        raise ValueError('parent/subdomain overlap between splits')
    return partitions, {'ready': not reasons, 'reasons': sorted(set(reasons)), 'status': pool_status(pool, annotations), 'splits': stats, 'usage': 'calibration_development_only', 'used_for_supervision': True, 'split_rule': 'domain SHA256 order, every third domain held out; fixed before labels'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'runs/calibration-supplement-v1')
    parser.add_argument('--out', type=Path, default=ROOT/'data/training/snapshots/calibration-supplement-v1')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    config = load_config(ROOT/'config/local.yaml')
    excluded = training_domains([ROOT/p for p in config['auto_training']['training_sources']] + list((ROOT/'data/training/snapshots').glob('*/*.jsonl')) + list((ROOT/'data/training').glob('*.jsonl')))
    for path in (ROOT/'data/evaluation').glob('independent_test*.jsonl'):
        excluded.update(row['domain'] for row in load_jsonl(path))
    pool = load_jsonl(args.run/'pool.jsonl')
    if not pool:
        raise ValueError('empty review pool')
    partitions, report = prepare(pool, latest_annotations(args.run/'annotations.jsonl'), excluded)
    report['pool_sha256'] = hashlib.sha256((args.run/'pool.jsonl').read_bytes()).hexdigest()
    if args.export:
        if not report['ready']:
            raise ValueError('export blocked: '+', '.join(report['reasons']))
        if args.out.exists():
            raise ValueError('refusing to overwrite existing export')
        args.out.mkdir(parents=True)
        for split, records in partitions.items():
            (args.out/(split+'.jsonl')).write_text(''.join(json.dumps(record, ensure_ascii=False)+'\n' for record in records), encoding='utf-8')
        (args.out/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
