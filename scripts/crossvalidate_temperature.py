from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


def loss(logit, label, temperature):
    z = logit / temperature
    return max(z, 0) + math.log1p(math.exp(-abs(z))) - label * z


def metrics(rows, temperatures):
    probabilities = [1 / (1 + math.exp(-max(-700, min(700, row['logit'] / temperature)))) for row, temperature in zip(rows, temperatures)]
    return {
        'nll': sum(loss(row['logit'], row['target'], temperature) for row, temperature in zip(rows, temperatures)) / len(rows),
        'brier': sum((p-row['target'])**2 for row, p in zip(rows, probabilities)) / len(rows),
        'ads_above_hide_threshold': sum(row['target'] == 1 and p >= .72 for row, p in zip(rows, probabilities)),
        'non_ads_above_hide_threshold': sum(row['target'] == 0 and p >= .72 for row, p in zip(rows, probabilities)),
    }


def crossvalidate(rows, seed, folds=5):
    domains = sorted({row['domain'] for row in rows})
    if len(domains) < folds:
        raise ValueError('not enough independent domains')
    random.Random(seed).shuffle(domains)
    assignments = {domain: index % folds for index, domain in enumerate(domains)}
    temperatures = [1.0] * len(rows)
    reports = []
    grid = [math.exp(math.log(.05) + i * math.log(400) / 400) for i in range(401)]
    for fold in range(folds):
        train = [row for row in rows if assignments[row['domain']] != fold]
        indices = [i for i, row in enumerate(rows) if assignments[row['domain']] == fold]
        test = [rows[i] for i in indices]
        if len({row['target'] for row in train}) != 2:
            raise ValueError('fit partition lacks a class')
        temperature = min(grid, key=lambda t: sum(loss(row['logit'], row['target'], t) for row in train))
        for index in indices:
            temperatures[index] = temperature
        reports.append({'fold': fold, 'fit_examples': len(train), 'heldout_examples': len(test), 'heldout_domains': len({row['domain'] for row in test}), 'heldout_ads': sum(row['target'] for row in test), 'temperature': temperature, 'baseline': metrics(test, [1]*len(test)), 'calibrated': metrics(test, [temperature]*len(test))})
    return {'seed': seed, 'folds': reports, 'baseline': metrics(rows, [1]*len(rows)), 'out_of_fold': metrics(rows, temperatures)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', default='runs/human-calibration-temp1.json')
    parser.add_argument('--data', default='data/training/snapshots/human-development-v1/calibration.jsonl')
    parser.add_argument('--out', default='runs/temperature-crossvalidation.json')
    args = parser.parse_args()
    source = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    if source['data_sha256'] != hashlib.sha256(Path(args.data).read_bytes()).hexdigest():
        raise ValueError('prediction data hash mismatch')
    if source['temperature'] != 1:
        raise ValueError('raw temperature-1 predictions required')
    records = {row['id']: row for row in map(json.loads, Path(args.data).read_text(encoding='utf-8').splitlines())}
    rows = []
    identifiers = set()
    for row in source['variants']['with_evidence']['rows']:
        if row['id'] in identifiers:
            raise ValueError('duplicate prediction id')
        identifiers.add(row['id'])
        record = records[row['id']]
        if row['label'] != record['question']['label']:
            raise ValueError('label mismatch')
        p = max(1e-12, min(1-1e-12, row['p_ad']))
        rows.append({'domain': record['source'], 'target': int(row['label'] == 0), 'logit': math.log(p/(1-p))})
    if identifiers != set(records):
        raise ValueError('predictions must cover the complete source dataset')
    report = {'usage': 'calibration_stability_diagnostic_only', 'examples': len(rows), 'domains': len({row['domain'] for row in rows}), 'labels': dict(Counter(row['target'] for row in rows)), 'method': 'five domain folds, fixed temperature grid 0.05 to 20, seeds 0 to 9', 'runs': [crossvalidate(rows, seed) for seed in range(10)]}
    report['summary'] = {'mean_baseline_nll': report['runs'][0]['baseline']['nll'], 'mean_oof_nll': sum(run['out_of_fold']['nll'] for run in report['runs']) / 10, 'nll_improved_runs': sum(run['out_of_fold']['nll'] < run['baseline']['nll'] for run in report['runs']), 'temperature_range': [min(fold['temperature'] for run in report['runs'] for fold in run['folds']), max(fold['temperature'] for run in report['runs'] for fold in run['folds'])]}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
