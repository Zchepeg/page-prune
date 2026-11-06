from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from jev.decider import JevDecider
from jev.lightweight_backend import TfidfBackend
from jev.types import DecisionLevel
from policy.blocking_policy import BlockingPolicy
from service.config import load_config
from training.dataset import load_jsonl, _candidate_from_state
from runtime.state_serialization import serialize_state
from evaluation.metrics import expected_calibration_error
from scripts.crossvalidate_temperature import loss


def main():
    output = ROOT/'runs/tfidf-baseline-v1'
    if output.exists():
        raise ValueError('refusing to overwrite baseline')
    training_path = ROOT/'data/training/snapshots/ownership-paired-v1/v2/train.jsonl'
    calibration_path = ROOT/'data/training/snapshots/calibration-supplement-v2/calibration.jsonl'
    validation_path = ROOT/'data/training/snapshots/calibration-supplement-v2/validation.jsonl'
    training = load_jsonl(training_path, 'binary_selective')
    calibration = load_jsonl(calibration_path, 'binary_selective')
    validation = load_jsonl(validation_path, 'binary_selective')
    counts = Counter(example.source for example in training)
    states, targets, weights = [], [], []
    for example in training:
        for keep in (True, False):
            state = json.loads(example.state)
            if not keep:
                state['detector_evidence'] = []
            states.append(serialize_state(state))
            targets.append(int(example.label == 0))
            weights.append(counts[example.source]**(-.5))
    weights = np.asarray(weights)
    weights *= len(training)/weights.sum()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=20000, sublinear_tf=True)
    classifier = LogisticRegression(C=1, solver='liblinear', max_iter=1000, random_state=42)
    started = time.perf_counter()
    matrix = vectorizer.fit_transform(states)
    classifier.fit(matrix, targets, sample_weight=weights)
    training_ms = (time.perf_counter()-started)*1000
    logits = classifier.decision_function(vectorizer.transform([example.state for example in calibration]))
    labels = [int(example.label == 0) for example in calibration]
    grid = [math.exp(math.log(.05)+i*math.log(400)/400) for i in range(401)]
    temperature = min(grid, key=lambda t: sum(loss(float(z), y, t) for z, y in zip(logits, labels)))
    metadata = {'temperature': temperature, 'state_schema': 'candidate-features-v2', 'architecture': 'tfidf-logistic-baseline-v1', 'seed': 42, 'training_ms': training_ms, 'training_examples': len(training), 'augmented_examples': len(states), 'vocabulary_size': len(vectorizer.vocabulary_), 'training_data_sha256': hashlib.sha256(training_path.read_bytes()).hexdigest(), 'calibration_data_sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(), 'calibration_nll_before': sum(loss(float(z), y, 1) for z, y in zip(logits, labels))/len(labels), 'calibration_nll_after': sum(loss(float(z), y, temperature) for z, y in zip(logits, labels))/len(labels)}
    output.mkdir()
    joblib.dump({'vectorizer': vectorizer, 'classifier': classifier, 'metadata': metadata}, output/'model.joblib', compress=3)
    backend = TfidfBackend(output)
    config = load_config(ROOT/'config/local.yaml')
    pc, jc = config['policy'], config['jev']
    decider = JevDecider(backend, jc['uncertainty_margin'], jc['min_confidence'])
    policy = BlockingPolicy(pc['block_threshold'], pc['hide_threshold'], DecisionLevel.parse(pc['min_decision_level']), set(pc['protected_tags']), pc['fail_open'])
    report = {'usage': 'reused_development_only', 'metadata': metadata, 'artifact_bytes': (output/'model.joblib').stat().st_size, 'data_sha256': hashlib.sha256(validation_path.read_bytes()).hexdigest(), 'modes': {}}
    for mode, value in (('temperature1', 1), ('calibrated', temperature)):
        backend.temperature = value
        report['modes'][mode] = {}
        for variant in ('with_evidence', 'without_evidence'):
            rows = []
            for example in validation:
                state = json.loads(example.state)
                candidate = _candidate_from_state(state, example.source)
                if variant == 'without_evidence':
                    state['detector_evidence'] = []
                decision = decider.decide(example.example_id, serialize_state(state), state.get('detector_evidence', []))
                action = policy.apply(candidate, decision)
                p = decision.probabilities['block']
                rows.append({'id': example.example_id, 'label': example.label, 'p_ad': p, 'argmax_ad': p > .5, 'verdict': decision.verdict.value, 'action': action.action.value})
            summary = {'examples': len(rows), 'ads': sum(r['label'] == 0 for r in rows), 'non_ads': sum(r['label'] == 1 for r in rows), 'nll': -sum(math.log(max(1e-12, r['p_ad'] if r['label'] == 0 else 1-r['p_ad'])) for r in rows)/len(rows), 'brier': sum((r['p_ad']-(r['label'] == 0))**2 for r in rows)/len(rows), 'ad_ece': expected_calibration_error([r['label'] == 0 for r in rows], [r['p_ad'] for r in rows])}
            for name, predicate in (('argmax', lambda r: r['argmax_ad']), ('selective', lambda r: r['verdict'] == 'block'), ('policy', lambda r: r['action'] in ('block', 'hide'))):
                summary[name] = {'true_ads': sum(r['label'] == 0 and predicate(r) for r in rows), 'false_ads': sum(r['label'] == 1 and predicate(r) for r in rows)}
            report['modes'][mode][variant] = {'summary': summary, 'rows': rows}
            print(json.dumps({mode+'.'+variant: summary}), flush=True)
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'training_ms': training_ms, 'temperature': temperature, 'artifact_bytes': report['artifact_bytes']}))


if __name__ == '__main__':
    main()
