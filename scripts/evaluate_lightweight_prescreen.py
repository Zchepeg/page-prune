from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.schemas import Candidate
from detector.candidate_ranker import CandidateRanker
from jev.lightweight_backend import TfidfBackend
from jev.protocol import DecisionProtocol, ProtocolRoute
from jev.questions import AD_TREATMENT
from jev.types import Decision, DecisionLevel, Verdict
from policy.blocking_policy import BlockingPolicy
from runtime.state_serialization import serialize_state
from service.config import load_config
from training.dataset import load_jsonl

GRID = (0, .0001, .0005, .001, .002, .005, .01, .02, .05)


def eligible(candidate, assessment, protocol, policy):
    return candidate.visible and candidate.tag not in policy.protected_tags and protocol.plan(assessment).route is ProtocolRoute.MODEL


def select_threshold(rows, min_samples=10, min_domains=3):
    feasible = []
    for threshold in GRID:
        accepted = [row for row in rows if row['eligible'] and row['p_ad'] <= threshold]
        if len(accepted) >= min_samples and len({row['domain'] for row in accepted}) >= min_domains and not any(row['label'] == 0 for row in accepted):
            feasible.append(threshold)
    return max(feasible) if feasible else None


def summarize(rows):
    return {'examples': len(rows), 'ads': sum(row['label'] == 0 for row in rows), 'non_ads': sum(row['label'] == 1 for row in rows), 'classified_ads': sum(row['label'] == 0 and row['verdict'] == 'block' for row in rows), 'false_ad_classifications': sum(row['label'] == 1 and row['verdict'] == 'block' for row in rows), 'action_ads': sum(row['label'] == 0 and row['action'] in ('block', 'hide') for row in rows), 'action_false_positives': sum(row['label'] == 1 and row['action'] in ('block', 'hide') for row in rows), 'qwen_calls': sum(row['qwen_called'] for row in rows), 'lightweight_calls': sum(row['lightweight_called'] for row in rows), 'prescreen_allowed': sum(row['prescreen_allowed'] for row in rows), 'prescreen_allowed_ads': sum(row['prescreen_allowed'] and row['label'] == 0 for row in rows)}


def main():
    output = ROOT/'runs/lightweight-prescreen-v1'
    if output.exists():
        raise ValueError('refusing to overwrite prescreen experiment')
    config = load_config(ROOT/'config/local.yaml')
    detector, pc, protocol_config = config['detector'], config['policy'], config['decision_protocol']
    ranker = CandidateRanker(detector['block_score'], detector['allow_score'])
    policy = BlockingPolicy(pc['block_threshold'], pc['hide_threshold'], DecisionLevel.parse(pc['min_decision_level']), set(pc['protected_tags']), pc['fail_open'])
    protocol = DecisionProtocol(protocol_config['challenge_block_below'], protocol_config['challenge_allow_above'], challenge_rule_blocks=protocol_config['challenge_rule_blocks'])
    backend = TfidfBackend(ROOT/'runs/tfidf-baseline-v1')
    raw = {row['id']: row for row in map(json.loads, (ROOT/'runs/calibration-supplement-v2/pool.jsonl').read_text(encoding='utf-8').splitlines())}
    calibration_path = ROOT/'data/training/snapshots/calibration-supplement-v2/calibration.jsonl'
    validation_path = ROOT/'data/training/snapshots/calibration-supplement-v2/validation.jsonl'
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != backend.metadata['calibration_data_sha256']:
        raise ValueError('calibration data differs from baseline')
    output.mkdir()
    plan = {'grid': GRID, 'minimum_calibration_accepts': 10, 'minimum_calibration_domains': 3, 'maximum_observed_calibration_ad_misses': 0, 'eligibility': 'visible non-protected gray model route only; rules and challenge routes bypass gate', 'usage': 'reused_development_only', 'temperature': backend.temperature}
    (output/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    calibration = []
    for example in load_jsonl(calibration_path, 'binary_selective'):
        candidate = Candidate.from_dict(raw[example.example_id]['candidate'])
        assessment = ranker.assess(candidate)
        calibration.append({'id': example.example_id, 'domain': example.source, 'label': example.label, 'eligible': eligible(candidate, assessment, protocol, policy), 'p_ad': backend.decide(example.state, AD_TREATMENT).probabilities['ad']})
    threshold = select_threshold(calibration)
    accepted = [row for row in calibration if threshold is not None and row['eligible'] and row['p_ad'] <= threshold]
    report = {'plan': plan, 'threshold': threshold, 'enabled': threshold is not None, 'calibration': {'examples': len(calibration), 'eligible': sum(row['eligible'] for row in calibration), 'eligible_ads': sum(row['eligible'] and row['label'] == 0 for row in calibration), 'accepted': len(accepted), 'accepted_domains': len({row['domain'] for row in accepted}), 'accepted_ads': sum(row['label'] == 0 for row in accepted)}, 'models': {}}
    report['calibration']['grid'] = [{'threshold': t, 'accepted': sum(row['eligible'] and row['p_ad'] <= t for row in calibration), 'accepted_domains': len({row['domain'] for row in calibration if row['eligible'] and row['p_ad'] <= t}), 'accepted_ads': sum(row['eligible'] and row['p_ad'] <= t and row['label'] == 0 for row in calibration)} for t in GRID]
    digest = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    validation = load_jsonl(validation_path, 'binary_selective')
    for name, path in [('qwen_v2_seed42', ROOT/'runs/ownership-paired-experiment-v1/v2-validation.json'), ('qwen_v2_seed7', ROOT/'runs/ownership-replications-v1/v2-seed7-validation.json')]:
        cached = json.loads(path.read_text(encoding='utf-8'))
        if cached['data_sha256'] != digest:
            raise ValueError('cached Qwen predictions use a different validation set')
        predictions = {row['id']: row for row in cached['variants']['with_evidence']['rows']}
        if set(predictions) != {example.example_id for example in validation}:
            raise ValueError('cached prediction coverage mismatch')
        variants = {}
        for use_gate in (False, True):
            rows = []
            for example in validation:
                candidate = Candidate.from_dict(raw[example.example_id]['candidate'])
                assessment = ranker.assess(candidate)
                route = protocol.plan(assessment)
                light_called = use_gate and threshold is not None and eligible(candidate, assessment, protocol, policy)
                p_light = backend.decide(example.state, AD_TREATMENT).probabilities['ad'] if light_called else None
                allowed = light_called and p_light <= threshold
                qwen_called = route.route is not ProtocolRoute.RULE and not allowed
                if allowed:
                    decision = Decision(example.example_id, Verdict.ALLOW, {'block': p_light, 'allow': 1-p_light, 'uncertain': 0}, DecisionLevel.L2, [], 'tfidf-prescreen-v1', 'ad-classification-v1')
                elif route.route is ProtocolRoute.RULE:
                    decision = Decision(example.example_id, route.rule_verdict, {'block': assessment.score, 'allow': 1-assessment.score, 'uncertain': 0}, DecisionLevel.RULE, assessment.evidence, 'rules', 'ad-classification-v1')
                else:
                    cached_row = predictions[example.example_id]
                    if cached_row['label'] != example.label:
                        raise ValueError('cached label mismatch')
                    p = cached_row['p_ad']
                    decision = Decision(example.example_id, Verdict(cached_row['verdict']), {'block': p, 'allow': 1-p, 'uncertain': 0}, DecisionLevel.L2, assessment.evidence, 'cached-qwen', 'ad-classification-v1')
                    decision = protocol.resolve(route, decision).decision
                action = policy.apply(candidate, decision)
                rows.append({'id': example.example_id, 'label': example.label, 'verdict': decision.verdict.value, 'action': action.action.value, 'qwen_called': qwen_called, 'lightweight_called': light_called, 'prescreen_allowed': allowed})
            variants['prescreen' if use_gate else 'baseline'] = {'summary': summarize(rows), 'rows': rows}
        base, gated = variants['baseline']['summary'], variants['prescreen']['summary']
        variants['comparison'] = {'qwen_calls_saved': base['qwen_calls']-gated['qwen_calls'], 'fraction_of_baseline_calls_saved': (base['qwen_calls']-gated['qwen_calls'])/base['qwen_calls'] if base['qwen_calls'] else 0, 'change_in_action_ads': gated['action_ads']-base['action_ads'], 'change_in_action_false_positives': gated['action_false_positives']-base['action_false_positives']}
        report['models'][name] = variants
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'threshold': threshold, 'enabled': report['enabled'], 'calibration': report['calibration'], 'models': {name: {'baseline': values['baseline']['summary'], 'prescreen': values['prescreen']['summary'], 'comparison': values['comparison']} for name, values in report['models'].items()}}, indent=2))


if __name__ == '__main__':
    main()
