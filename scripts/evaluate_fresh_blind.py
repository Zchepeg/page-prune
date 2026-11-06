from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.state_builder import build_state_payload
from detector.candidate_ranker import CandidateRanker
from evaluation.benchmark import AblationEvaluator, AblationMode, claim_eligibility, domain_leakage, load_benchmark
from jev.backend import HeuristicBackend
from jev.decider import JevDecider
from jev.protocol import DecisionProtocol
from jev.types import DecisionLevel
from policy.blocking_policy import BlockingPolicy
from runtime.state_serialization import serialize_state
from scripts.freeze_fresh_blind import validate_study


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeEvaluator(AblationEvaluator):
    def __init__(self, *args, schema='candidate-features-v2', **kwargs):
        super().__init__(*args, **kwargs)
        self.schema = schema
        self.last_model_ms = None

    def model_decision(self, candidate, evidence):
        state = serialize_state(build_state_payload(candidate, evidence, self.schema))
        started = time.perf_counter()
        decision = self.decider.decide(candidate.candidate_id, state, evidence)
        self.last_model_ms = (time.perf_counter()-started)*1000
        return decision

    def _decide(self, candidate, assessment, mode):
        if mode is AblationMode.MODEL_ONLY:
            return self.model_decision(candidate, []), 'model'
        if mode is AblationMode.CASCADE and assessment.route == 'jev':
            return self.model_decision(candidate, assessment.evidence), 'model'
        return self._rule_decision(candidate, assessment, allow_uncertain=mode is AblationMode.RULES_ONLY), 'rules'

    def run_mode(self, cases, mode):
        rows = []
        for case in cases:
            self.last_model_ms = None
            started = time.perf_counter()
            assessment = self.ranker.assess(case.candidate)
            decision, source = self._decide(case.candidate, assessment, mode)
            action = self.policy.apply(case.candidate, decision)
            rows.append({'id':case.case_id, 'domain':case.domain, 'label':case.label, 'prediction':self._prediction(decision), 'block_probability':float(decision.probabilities['block']), 'verdict':decision.verdict.value, 'action':action.action.value, 'source':source, 'model_called':source == 'model', 'model_call_ms':self.last_model_ms, 'truncated':bool(decision.diagnostics.get('truncated', False)), 'latency_ms':(time.perf_counter()-started)*1000, 'ssa_flagged':False, 'ssa_route':'not_executed', 'protocol_route':'ordinary_cascade' if mode is AblationMode.CASCADE else mode.value})
        report = self._summarize(rows)
        timings = [row['model_call_ms'] for row in rows if row['model_called']]
        report['model_call_timing_ms'] = {'samples':len(timings), 'mean':sum(timings)/len(timings) if timings else None, 'p95':self._percentile(timings, .95) if timings else None}
        report['truncated'] = sum(row['truncated'] for row in rows)
        report['page_action_ad_recall'] = sum(row['label'] == 0 and row['action'] in ('hide','block') for row in rows)/sum(row['label'] == 0 for row in rows)
        return report


def frozen_inputs(directory):
    plan, _ = validate_study(directory)
    metadata = json.loads((directory/'benchmark.metadata.json').read_text(encoding='utf-8'))
    if metadata['sha256'] != digest(directory/'benchmark.jsonl') or metadata['study_plan_sha256'] != digest(directory/'plan.json') or metadata['annotations_sha256'] != digest(directory/'annotations.jsonl') or metadata['capture_manifest_sha256'] != digest(directory/'capture-manifest.json'):
        raise ValueError('frozen benchmark inputs changed')
    if not metadata.get('complete_pool_independently_reviewed') or metadata['annotation_modes'] != ['human_independent']:
        raise ValueError('independent human review required')
    cases = load_benchmark(directory/'benchmark.jsonl')
    eligibility = claim_eligibility(cases, domain_leakage(cases, set(plan['exclusion_domains'])), metadata)
    if not eligibility['eligible']:
        raise ValueError('benchmark claim gates failed: '+str(eligibility['reasons']))
    return plan, metadata, cases, eligibility


def make_evaluator(plan, backend, schema):
    pc, dc, jc = plan['policy'], plan['detector'], plan['jev_thresholds']
    return NativeEvaluator(CandidateRanker(dc['block_score'], dc['allow_score']), JevDecider(backend, jc['uncertainty_margin'], jc['min_confidence']), BlockingPolicy(pc['block_threshold'], pc['hide_threshold'], DecisionLevel.parse(pc['min_decision_level']), set(pc['protected_tags']), pc['fail_open']), DecisionProtocol(), schema=schema)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=ROOT/'runs/fresh-blind-review-v1')
    parser.add_argument('--model', required=True, choices=('rules', 'tfidf', 'qwen-v2-seed42', 'qwen-v2-seed7', 'qwen-v3-seed42', 'qwen-v3-seed7'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan, metadata, cases, eligibility = frozen_inputs(directory)
    output = directory/'evaluation-v1'/f'{args.model}.json'
    if args.check_only:
        print(json.dumps({'eligible':eligibility['eligible'], 'examples':len(cases), 'model':args.model, 'inference_executed':False}))
        return
    if output.exists():
        raise ValueError('refusing to overwrite model evaluation')
    output.parent.mkdir(exist_ok=True)
    registration = output.with_suffix('.execution.json')
    if registration.exists():
        raise ValueError('execution already registered; inspect prior attempt before retry')
    registration.write_text(json.dumps({'started_at':datetime.now(timezone.utc).isoformat(), 'model':args.model, 'benchmark_sha256':metadata['sha256'], 'plan_sha256':digest(directory/'plan.json'), 'runner_sha256':digest(Path(__file__)), 'scope':'implementation recorded after human label freeze, before this model inference; no model selection or threshold fitting', 'primary':'rules_only' if args.model == 'rules' else 'rule_model_cascade', 'secondary':'model_only_without_evidence', 'timing':'serial single candidate, warmed decision core; excludes HTTP, browser, cache, batching and load; production GPU service may remain resident'},indent=2),encoding='utf-8')
    started = time.perf_counter()
    if args.model == 'rules':
        backend = HeuristicBackend()
        schema = 'candidate-features-v2'
    else:
        checkpoint = ROOT/plan['models'][args.model]['checkpoint']
        if args.model == 'tfidf':
            from jev.lightweight_backend import TfidfBackend
            backend = TfidfBackend(checkpoint)
        else:
            free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip().splitlines()[0])
            if free < 4608:
                raise RuntimeError('insufficient free GPU memory for isolated evaluation; production service untouched')
            from jev.backend import QwenPointerBackend
            backend = QwenPointerBackend(str(checkpoint), 'cuda', 'nf4', 4096, 384, 2)
        schema = backend.metadata['state_schema']
        expected = 'candidate-features-v3' if 'v3-' in args.model else 'candidate-features-v2'
        if schema != expected:
            raise ValueError('checkpoint native schema mismatch')
    load_ms = (time.perf_counter()-started)*1000
    evaluator = make_evaluator(plan, backend, schema)
    warmup_started = time.perf_counter()
    if args.model != 'rules':
        for evidence in ([], evaluator.ranker.assess(cases[0].candidate).evidence):
            evaluator.model_decision(cases[0].candidate, evidence)
    warmup_ms = (time.perf_counter()-warmup_started)*1000
    modes = [AblationMode.RULES_ONLY] if args.model == 'rules' else [AblationMode.CASCADE, AblationMode.MODEL_ONLY]
    results = {}
    for mode in modes:
        results[mode.value] = evaluator.run_mode(cases, mode)
        summary = results[mode.value]
        print(json.dumps({'model':args.model, 'mode':mode.value, 'recall':summary['binary_ad_metrics']['recall'], 'action_ad_recall':summary['page_action_ad_recall'], 'action_fpr':summary['page_action_false_positive_rate'], 'model_calls':summary['routing']['model_calls']}),flush=True)
    frozen_inputs(directory)
    report = {'model':args.model, 'generated_at':datetime.now(timezone.utc).isoformat(), 'benchmark_sha256':metadata['sha256'], 'plan_sha256':digest(directory/'plan.json'), 'runner_sha256':digest(Path(__file__)), 'state_schema':schema, 'load_ms':load_ms, 'warmup_ms':warmup_ms, 'model_metadata':{k:v for k,v in getattr(backend,'metadata',{}).items() if k in ('temperature','state_schema','architecture','instruction','seed')}, 'runtime':backend.runtime_status() if hasattr(backend,'runtime_status') else {}, 'claim_eligibility':eligibility, 'modes':results}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'saved':args.model, 'load_ms':round(load_ms,3), 'warmup_ms':round(warmup_ms,3)}),flush=True)


if __name__ == '__main__':
    main()
