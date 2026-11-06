from __future__ import annotations

import math
import random
from collections import defaultdict


def exact_mcnemar(improved, regressed):
    total = improved + regressed
    if total == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(total, k) for k in range(min(improved, regressed) + 1)) / 2**total)


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def _metrics(rows, action=False):
    tp = fp = fn = 0
    for row in rows:
        truth = row['label'] == 0
        positive = row['action'] in ('block', 'hide') if action else row['prediction'] == 0
        tp += truth and positive
        fp += not truth and positive
        fn += truth and not positive
    return {'tp': tp, 'fp': fp, 'fn': fn, 'recall': tp/(tp+fn) if tp+fn else None, 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def _interval(values):
    values = sorted(values)
    return [values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]] if values else None


def paired_statistics(baseline, candidate, repeats=10000, seed=42):
    if repeats < 100:
        raise ValueError('at least 100 bootstrap resamples required')
    left = {row['id']: row for row in baseline}
    right = {row['id']: row for row in candidate}
    if not left or len(left) != len(baseline) or len(right) != len(candidate) or set(left) != set(right):
        raise ValueError('nonempty unique prediction IDs must match exactly')
    grouped = defaultdict(list)
    for identifier, row in left.items():
        other = right[identifier]
        if row['label'] not in (0, 1) or row['label'] != other['label'] or not row['domain'] or row['domain'] != other['domain']:
            raise ValueError('binary labels and nonempty domains must match')
        if row['prediction'] not in (0, 1, 2) or other['prediction'] not in (0, 1, 2):
            raise ValueError('invalid prediction')
        if row['action'] not in ('block', 'hide', 'allow', 'review') or other['action'] not in ('block', 'hide', 'allow', 'review'):
            raise ValueError('invalid action')
        grouped[row['domain']].append(identifier)
    ids = sorted(left)
    domains = sorted(grouped)
    report = {'examples': len(ids), 'domains': len(domains), 'repeats': repeats, 'seed': seed, 'method': 'paired domain cluster percentile bootstrap; domains sampled with replacement, all candidates retained; micro metrics', 'limitations': 'conditional on frozen predictions; no training-seed uncertainty; candidate-level McNemar assumes independent pairs and does not adjust domain clustering', 'metrics': {}}
    rng = random.Random(seed)
    samples = {'classification': {'recall': [], 'f1': []}, 'page_action': {'recall': [], 'f1': []}}
    domain_counts = {}
    for domain in domains:
        domain_counts[domain] = {name: (_metrics([left[i] for i in grouped[domain]], action), _metrics([right[i] for i in grouped[domain]], action)) for name, action in (('classification', False), ('page_action', True))}
    for _ in range(repeats):
        selected = rng.choices(domains, k=len(domains))
        for name in samples:
            totals = [{key: sum(domain_counts[d][name][side][key] for d in selected) for key in ('tp', 'fp', 'fn')} for side in (0, 1)]
            for metric in ('recall', 'f1'):
                values = []
                for value in totals:
                    denominator = value['tp']+value['fn'] if metric == 'recall' else 2*value['tp']+value['fp']+value['fn']
                    values.append((value['tp'] if metric == 'recall' else 2*value['tp'])/denominator if denominator else (None if metric == 'recall' else 0.0))
                if all(value is not None for value in values):
                    samples[name][metric].append(values[1]-values[0])
    for name, action in (('classification', False), ('page_action', True)):
        a, b = _metrics(baseline, action), _metrics(candidate, action)
        def positive(row):
            return row['action'] in ('block', 'hide') if action else row['prediction'] == 0
        improved = sum(left[i]['label'] == 0 and not positive(left[i]) and positive(right[i]) for i in ids)
        regressed = sum(left[i]['label'] == 0 and positive(left[i]) and not positive(right[i]) for i in ids)
        report['metrics'][name] = {'baseline': a, 'candidate': b, 'baseline_recall_wilson_95': wilson(a['tp'], a['tp']+a['fn']), 'candidate_recall_wilson_95': wilson(b['tp'], b['tp']+b['fn']), 'delta_recall': b['recall']-a['recall'] if a['recall'] is not None else None, 'delta_f1': b['f1']-a['f1'], 'delta_recall_domain_bootstrap_95': _interval(samples[name]['recall']), 'delta_f1_domain_bootstrap_95': _interval(samples[name]['f1']), 'recall_valid_resamples': len(samples[name]['recall']), 'ad_recall_mcnemar': {'improved': improved, 'regressed': regressed, 'two_sided_exact_p': exact_mcnemar(improved, regressed)}}
    improved = sum(left[i]['prediction'] != left[i]['label'] and right[i]['prediction'] == right[i]['label'] for i in ids)
    regressed = sum(left[i]['prediction'] == left[i]['label'] and right[i]['prediction'] != right[i]['label'] for i in ids)
    report['all_candidate_classification_mcnemar'] = {'improved': improved, 'regressed': regressed, 'two_sided_exact_p': exact_mcnemar(improved, regressed), 'scope': 'exact class correctness including abstentions as incorrect; not an ad-recall or safety test'}
    return report
