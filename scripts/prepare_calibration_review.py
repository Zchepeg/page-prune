from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.schemas import Candidate
from collector.state_builder import build_state
from detector.candidate_ranker import CandidateRanker
from evaluation.benchmark import domain_leakage, BenchmarkCase, training_domains
from service.config import load_config


def main():
    output = ROOT / 'runs/calibration-review-v1'
    if output.exists():
        raise ValueError('refusing to overwrite review queue')
    config = load_config(ROOT / 'config/local.yaml')
    excluded = training_domains([ROOT / p for p in config['auto_training']['training_sources']])
    for path in (ROOT / 'data/evaluation').glob('independent_test*.jsonl'):
        excluded.update(json.loads(line)['domain'] for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    for path in (ROOT / 'data/training/snapshots').glob('*/*.jsonl'):
        excluded.update(json.loads(line).get('source', '') for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    excluded.discard('')
    sources = [ROOT / 'runs/reference-guided-v1/pool.jsonl', ROOT / 'runs/gpt56-machine-annotation-20261022/fresh_raw.jsonl']
    buckets = defaultdict(list)
    seen, counts = set(), Counter()
    ranker = CandidateRanker()
    for path in sources:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            candidate = Candidate.from_dict(record['candidate'])
            domain = candidate.page.top_level_domain.lower().strip()
            if not domain or not domain_leakage([BenchmarkCase('', domain, 1, candidate)], excluded)['domain_disjoint']:
                counts['excluded_domain'] += 1
                continue
            fingerprint = hashlib.sha256(build_state(candidate, []).encode()).hexdigest()
            if fingerprint in seen:
                counts['duplicates'] += 1
                continue
            seen.add(fingerprint)
            score = ranker.assess(candidate).score
            stratum = 'low_rule_score' if score < .12 else 'high_rule_score' if score >= .85 else 'rule_boundary'
            buckets[domain, stratum].append({'id': record['id'], 'domain': domain, 'candidate': candidate.to_dict(), 'captured_at': record.get('captured_at', ''), 'review_context': {'candidate_kind': stratum}, 'provenance': {'purpose': 'calibration_supplement_review', 'label_source': 'unreviewed_candidate', 'source_file': str(path.relative_to(ROOT)), 'fingerprint': fingerprint}})
    selected = []
    for domain in sorted({key[0] for key in buckets}):
        for stratum in ('rule_boundary', 'high_rule_score', 'low_rule_score'):
            selected.extend(buckets[domain, stratum][:4])
    output.mkdir()
    (output / 'pool.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in selected), encoding='utf-8')
    report = {'examples': len(selected), 'domains': len({r['domain'] for r in selected}), 'strata': dict(Counter(r['review_context']['candidate_kind'] for r in selected)), 'filtered': dict(counts), 'label_status': 'unlabelled', 'ready_for_calibration': False, 'reason': 'human_labels_and_more_domains_required', 'per_domain_stratum_limit': 4, 'excluded_domain_count': len(excluded)}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
