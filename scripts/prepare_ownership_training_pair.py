from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.schemas import Candidate
from collector.state_builder import STATE_SCHEMA_VERSION, OWNERSHIP_SCHEMA_VERSION, build_state_payload
from detector.candidate_ranker import CandidateRanker
from evaluation.annotation import load_jsonl, latest_annotations
from evaluation.benchmark import BenchmarkCase, domain_leakage, training_domains
from jev.questions import AD_TREATMENT


def main():
    output = ROOT/'data/training/snapshots/ownership-paired-v1'
    if output.exists():
        raise ValueError('refusing to overwrite paired dataset')
    old = ROOT/'data/training/snapshots/human-development-v3'
    capture = ROOT/'runs/ownership-training-bounded-v1'
    pool = load_jsonl(capture/'pool.jsonl')
    annotations = latest_annotations(capture/'annotations.jsonl')
    if any(row['id'] not in annotations for row in pool):
        raise ValueError('human review incomplete')
    excluded = training_domains([old/'validation.jsonl', old/'calibration.jsonl'])
    for path in (ROOT/'data/evaluation').glob('independent_test_v2.jsonl'):
        excluded.update(row['domain'] for row in load_jsonl(path))
    for path in (ROOT/'data/training/snapshots/calibration-supplement-v3').glob('*.jsonl'):
        excluded.update(row['source'] for row in load_jsonl(path))
    training = load_jsonl(old/'train.jsonl')
    identifiers = {row['id'] for row in training}
    ranker = CandidateRanker()
    additions = []
    for row in pool:
        annotation = annotations[row['id']]
        if annotation.get('annotation_mode') != 'human_independent':
            raise ValueError('independent human labels required')
        if annotation['label'] == 'insufficient_evidence':
            continue
        candidate = Candidate.from_dict(row['candidate'])
        if not domain_leakage([BenchmarkCase(row['id'], row['domain'], 1, candidate)], excluded)['domain_disjoint']:
            raise ValueError('new training domain overlaps held-out data')
        if not candidate.resource_ownership_known or row['id'] in identifiers:
            raise ValueError('known ownership and unique IDs required')
        identifiers.add(row['id'])
        additions.append({'id': row['id'], 'source': row['domain'], 'state': build_state_payload(candidate, ranker.assess(candidate).evidence, OWNERSHIP_SCHEMA_VERSION), 'question': {'instruction': AD_TREATMENT.text, 'options': ['ad', 'non_ad'], 'label': int(annotation['label'] == 'non_ad')}, 'provenance': {'label_source': 'human_independent', 'origin': 'ownership_training_capture', 'annotated_at': annotation.get('annotated_at', '')}})
    training += additions
    report = {'usage': 'paired_development_training_preparation', 'new_human_examples': len(additions), 'new_human_labels': dict(Counter(row['question']['label'] for row in additions)), 'total_training': len(training), 'train_domains': len({row['source'] for row in training}), 'training_labels': dict(Counter(row['question']['label'] for row in training)), 'annotation_sha256': hashlib.sha256((capture/'annotations.jsonl').read_bytes()).hexdigest(), 'variants': {}, 'trained': False}
    for name, schema in (('v2', STATE_SCHEMA_VERSION), ('v3', OWNERSHIP_SCHEMA_VERSION)):
        directory = output/name
        directory.mkdir(parents=True)
        report['variants'][name] = {}
        for split, records in (('train', training), ('validation', load_jsonl(old/'validation.jsonl')), ('calibration', load_jsonl(old/'calibration.jsonl'))):
            converted = []
            for row in records:
                record = json.loads(json.dumps(row))
                record['state']['schema_version'] = schema
                if name == 'v2':
                    record['state'].pop('direct_resource_urls', None)
                    record['state'].pop('resource_ownership_known', None)
                converted.append(record)
            path = directory/(split+'.jsonl')
            path.write_text(''.join(json.dumps(record, ensure_ascii=False)+'\n' for record in converted), encoding='utf-8')
            report['variants'][name][split] = {'examples': len(converted), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
