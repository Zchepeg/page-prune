from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev.backend import QwenPointerBackend
from jev.decider import JevDecider
from jev.questions import AD_TREATMENT
from jev.types import DecisionLevel
from policy.blocking_policy import BlockingPolicy
from service.config import load_config
from training.dataset import _candidate_from_state, load_jsonl
from runtime.state_serialization import serialize_state
from evaluation.metrics import expected_calibration_error


def alter_ownership(state, mode, donor=None):
    state = json.loads(json.dumps(state))
    if mode == 'unknown':
        state['direct_resource_urls'] = []
        state['resource_ownership_known'] = False
    elif mode == 'known_empty':
        state['direct_resource_urls'] = []
        state['resource_ownership_known'] = True
    elif mode == 'shuffle':
        state['direct_resource_urls'] = list(donor.get('direct_resource_urls', []))
        state['resource_ownership_known'] = bool(donor.get('resource_ownership_known', False))
    elif mode == 'omitted':
        state.pop('direct_resource_urls', None)
        state.pop('resource_ownership_known', None)
    elif mode != 'original':
        raise ValueError('unsupported ownership intervention')
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default='config/local.yaml')
    parser.add_argument('--out', required=True)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--ownership-ablation', action='store_true')
    parser.add_argument('--ownership-modes', nargs='+', choices=('original', 'unknown', 'known_empty', 'shuffle', 'omitted'))
    args = parser.parse_args()
    config = load_config(Path(args.config))
    pc, jc = config['policy'], config['jev']
    backend = QwenPointerBackend(args.checkpoint, 'cuda', 'nf4', 4096, 384, 2)
    if args.temperature is not None:
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        backend.model.head.temperature.fill_(args.temperature)
        backend.metadata['temperature'] = args.temperature
    decider = JevDecider(backend, jc['uncertainty_margin'], jc['min_confidence'])
    policy = BlockingPolicy(pc['block_threshold'], pc['hide_threshold'], DecisionLevel.parse(pc['min_decision_level']), set(pc['protected_tags']), pc['fail_open'])
    examples = load_jsonl(args.data, 'binary_selective', state_schema=backend.metadata.get('state_schema'))
    donors = {}
    if args.ownership_modes and not args.ownership_ablation:
        raise ValueError('ownership modes require --ownership-ablation')
    if args.ownership_ablation:
        if backend.metadata.get('state_schema') != 'candidate-features-v3':
            raise ValueError('ownership ablation requires an explicit v3 checkpoint')
        for domain in sorted({example.source for example in examples}):
            group = sorted([example for example in examples if example.source == domain], key=lambda example: example.example_id)
            shuffled = group.copy()
            random.Random(42).shuffle(shuffled)
            for example, donor in zip(group, shuffled):
                donors[example.example_id] = json.loads(donor.state)
    report = {'checkpoint': args.checkpoint, 'data_sha256': hashlib.sha256(Path(args.data).read_bytes()).hexdigest(), 'usage': 'development_only', 'temperature': backend.metadata.get('temperature', 1), 'instruction': backend.metadata.get('instruction', AD_TREATMENT.text), 'variants': {}}
    modes = (args.ownership_modes or ('original', 'unknown', 'known_empty', 'shuffle')) if args.ownership_ablation else ('original',)
    variants = [(mode, evidence) for mode in modes for evidence in ('with_evidence', 'without_evidence')]
    for mode, evidence_mode in variants:
        variant = mode+'.'+evidence_mode if args.ownership_ablation else evidence_mode
        rows = []
        changed = 0
        for example in examples:
            state = json.loads(example.state)
            candidate = _candidate_from_state(state, example.source)
            altered = alter_ownership(state, mode, donors.get(example.example_id))
            changed += altered != state
            state = altered
            if evidence_mode == 'without_evidence':
                state['detector_evidence'] = []
            evidence = state.get('detector_evidence', [])
            decision = decider.decide(example.example_id, serialize_state(state), evidence)
            action = policy.apply(candidate, decision)
            p = decision.probabilities['block']
            rows.append({'id': example.example_id, 'label': example.label, 'p_ad': p, 'argmax_ad': p > decision.probabilities['allow'], 'verdict': decision.verdict.value, 'action': action.action.value, 'reason': action.reason, 'truncated': decision.diagnostics.get('truncated', False)})
        ads = sum(row['label'] == 0 for row in rows)
        non_ads = len(rows) - ads
        stats = {'examples': len(rows), 'ads': ads, 'non_ads': non_ads, 'verdicts': dict(Counter(row['verdict'] for row in rows)), 'actions': dict(Counter(row['action'] for row in rows)), 'truncated': sum(row['truncated'] for row in rows)}
        stats['ownership_changed'] = changed
        stats['nll'] = -sum(math.log(max(1e-12, row['p_ad'] if row['label'] == 0 else 1-row['p_ad'])) for row in rows) / len(rows)
        stats['brier'] = sum((row['p_ad'] - (row['label'] == 0))**2 for row in rows) / len(rows)
        stats['ad_ece'] = expected_calibration_error([row['label'] == 0 for row in rows], [row['p_ad'] for row in rows])
        for name, pred in (('argmax', lambda r: r['argmax_ad']), ('selective', lambda r: r['verdict'] == 'block'), ('policy', lambda r: r['action'] in ('hide', 'block'))):
            tp = sum(row['label'] == 0 and pred(row) for row in rows)
            fp = sum(row['label'] == 1 and pred(row) for row in rows)
            stats[name] = {'true_ads': tp, 'false_ads': fp, 'recall': tp / ads if ads else 0, 'fpr': fp / non_ads if non_ads else 0}
        report['variants'][variant] = {'summary': stats, 'rows': rows}
        print(json.dumps({variant: stats}), flush=True)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
