from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageStat
from collector.schemas import Candidate
from detector.candidate_ranker import CandidateRanker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', default='runs/calibration-fresh-v1/pool.jsonl', type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.pool.read_text(encoding='utf-8').splitlines() if line.strip()]
    kinds, strata, evidence, ownership = Counter(), Counter(), Counter(), Counter()
    issues = []
    ranker = CandidateRanker()
    for record in records:
        raw_candidate = record['candidate']
        known = raw_candidate.get('resource_ownership_known', 'direct_resource_urls' in raw_candidate)
        ownership['known_direct_resources' if known and raw_candidate.get('direct_resource_urls') else 'known_no_direct_resources' if known else 'unknown'] += 1
        kinds[record.get('review_context', {}).get('candidate_kind', 'missing')] += 1
        score = ranker.assess(Candidate.from_dict(record['candidate'])).score
        strata['low' if score < .12 else 'high' if score >= .85 else 'boundary'] += 1
        path = Path(record.get('review_context', {}).get('screenshot', ''))
        status = 'missing'
        if path.is_file():
            try:
                with Image.open(path) as image:
                    stat = ImageStat.Stat(image.convert('RGB').resize((64, 64)))
                    status = 'near_uniform' if max(stat.stddev) < 2 else 'has_visual_variation'
            except OSError:
                status = 'unreadable'
        evidence[status] += 1
        if status != 'has_visual_variation':
            issues.append({'id': record['id'], 'status': status})
    report = {'examples': len(records), 'domains': len({r['domain'] for r in records}), 'candidate_kinds': dict(kinds), 'rule_score_strata': dict(strata), 'resource_ownership': dict(ownership), 'screenshot_quality': dict(evidence), 'issues': issues, 'labels': 'none', 'ready_for_training_or_calibration': False, 'note': 'Visual variation is not proof of sufficient annotation evidence; blank screenshots do not imply non-ad.'}
    (args.pool.parent / 'evidence-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'issues'}, indent=2))


if __name__ == '__main__':
    main()
