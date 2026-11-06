from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.state_builder import build_state_payload
from detector.candidate_ranker import CandidateRanker
from evaluation.benchmark import load_benchmark, domain_leakage
from jev.questions import AD_TREATMENT


def prepare(source: Path, excluded: Path, output: Path):
    metadata = json.loads(source.with_suffix('.metadata.json').read_text(encoding='utf-8'))
    if metadata.get('used_for_supervision') is not True or metadata.get('annotation_method') != 'independent_human':
        raise ValueError('source must be independent human labels already designated for supervision')
    if output.exists():
        raise ValueError('refusing to overwrite an existing dataset')
    cases = load_benchmark(source)
    excluded_domains = {case.domain for case in load_benchmark(excluded)}
    cases = [case for case in cases if domain_leakage([case], excluded_domains)['domain_disjoint'] and case.label in (0, 1)]
    domains = sorted({case.domain for case in cases})
    if len(domains) < 10:
        raise ValueError('not enough domains')
    selected = None
    for seed in range(1000):
        shuffled = domains.copy()
        random.Random(seed).shuffle(shuffled)
        n = max(2, round(len(domains) * .2))
        groups = {'validation': set(shuffled[:n]), 'calibration': set(shuffled[n:2*n]), 'train': set(shuffled[2*n:])}
        partitions = {name: [case for case in cases if case.domain in group] for name, group in groups.items()}
        if any(not domain_leakage(partitions[left], groups[right])['domain_disjoint'] for left, right in (('train', 'validation'), ('train', 'calibration'), ('validation', 'calibration'))):
            continue
        if all(min(Counter(case.label for case in items).get(label, 0) for label in (0, 1)) >= 15 for items in partitions.values()):
            selected = seed, partitions
            break
    if selected is None:
        raise ValueError('cannot obtain domain-disjoint splits with at least 15 examples per class')
    seed, partitions = selected
    ranker = CandidateRanker()
    manifest = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'excluded_sha256': hashlib.sha256(excluded.read_bytes()).hexdigest(), 'seed': seed, 'usage': 'development_only', 'splits': {}}
    output.mkdir(parents=True)
    for name, items in partitions.items():
        records = [{'id': case.case_id, 'source': case.domain, 'state': build_state_payload(case.candidate, ranker.assess(case.candidate).evidence), 'question': {'instruction': AD_TREATMENT.text, 'options': ['ad', 'non_ad'], 'label': case.label}, 'provenance': {'label_source': 'human_independent', 'origin': 'retired_supervision_benchmark', 'split': name}} for case in items]
        path = output / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records), encoding='utf-8')
        manifest['splits'][name] = {'examples': len(items), 'domains': len({case.domain for case in items}), 'labels': dict(Counter(case.label for case in items)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/evaluation/independent_test.jsonl', type=Path)
    parser.add_argument('--exclude', default='data/evaluation/independent_test_v2.jsonl', type=Path)
    parser.add_argument('--out', default='data/training/snapshots/human-development-v1', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.exclude, args.out), indent=2))
