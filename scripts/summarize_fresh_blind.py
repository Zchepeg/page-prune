from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.paired_statistics import paired_statistics, wilson
from scripts.evaluate_fresh_blind import frozen_inputs, digest

MODEL_ORDER = ('rules', 'tfidf', 'qwen-v2-seed42', 'qwen-v2-seed7', 'qwen-v3-seed42', 'qwen-v3-seed7')


def action_fpr_interval(left, right):
    right_by_id = {row['id']:row for row in right}
    grouped = defaultdict(lambda:[0,0])
    for row in left:
        value = grouped[row['domain']]
        if row['label'] == 1:
            value[0] += 1
            value[1] += int(right_by_id[row['id']]['action'] in ('hide','block'))-int(row['action'] in ('hide','block'))
    keys = sorted(grouped)
    rng = random.Random(42)
    values = []
    for _ in range(10000):
        selected = rng.choices(keys, k=len(keys))
        denominator = sum(grouped[key][0] for key in selected)
        if denominator:
            values.append(sum(grouped[key][1] for key in selected)/denominator)
    values.sort()
    last = len(values) - 1
    return {
        'delta_action_fpr_domain_bootstrap_95': [values[int(.025 * last)], values[int(.975 * last)]],
        'valid_resamples': len(values),
        'limitation': 'The bootstrap reflects only false positives observed in this sample.',
    }


def summarize(directory):
    plan, metadata, cases, _ = frozen_inputs(directory)
    folder = directory/'evaluation-v1'
    output = folder/'summary.json'
    if output.exists():
        raise ValueError('refusing to overwrite frozen evaluation summary')
    loaded = {}
    for name in MODEL_ORDER:
        value = json.loads((folder/f'{name}.json').read_text(encoding='utf-8'))
        if value['benchmark_sha256'] != metadata['sha256'] or value['plan_sha256'] != digest(directory/'plan.json'):
            raise ValueError('evaluation provenance mismatch')
        if value['runner_sha256'] != digest(ROOT/'scripts/evaluate_fresh_blind.py'):
            raise ValueError('evaluation implementation changed')
        truth = {case.case_id:(case.domain,case.label) for case in cases}
        for mode in value['modes'].values():
            rows = mode['rows']
            if len(rows) != len(truth) or {r['id']:(r['domain'],r['label']) for r in rows} != truth:
                raise ValueError('prediction rows do not match frozen benchmark')
        loaded[name] = value
    baseline = loaded['rules']['modes']['rules_only']['rows']
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'benchmark_sha256': metadata['sha256'],
        'plan_sha256': digest(directory / 'plan.json'),
        'data': {
            'examples': len(cases),
            'domains': len({c.domain for c in cases}),
            'ads': sum(c.label == 0 for c in cases),
            'non_ads': sum(c.label == 1 for c in cases),
            'annotation_method': 'independent human blind',
            'used_for_supervision': False,
            'sampling': (
                'mixed targeted/control/neutral/ambiguous public-site candidates, '
                'access failures retained locally; not natural-web prevalence'
            ),
        },
        'status': 'frozen candidate evaluation; no automatic model promotion',
        'primary': 'ordinary rule-model cascade versus same rules, page-action recall and FPR',
        'timing_scope': (
            'warmed serial candidate decision core only; excludes HTTP, browser, cache, '
            'batching and model load; GPU service remained resident'
        ),
        'statistical_scope': (
            '10000 paired domain percentile bootstrap draws, seed42; per-comparison intervals, '
            'five descriptive comparisons; McNemar candidate-independence assumption, '
            'no multiplicity-adjusted confirmatory claims; training-seed uncertainty not included'
        ),
        'models': {},
        'comparisons': {},
    }
    for name, value in loaded.items():
        primary = value['modes']['rules_only' if name == 'rules' else 'rule_model_cascade']
        rows = primary['rows']
        tp = sum(r['label'] == 0 and r['action'] in ('hide','block') for r in rows)
        fp = sum(r['label'] == 1 and r['action'] in ('hide','block') for r in rows)
        report['models'][name] = {key:value[key] for key in ('state_schema','load_ms','warmup_ms','model_metadata','runtime')}
        report['models'][name].update({
            'modes': {
                mode: {key: content for key, content in data.items() if key != 'rows'}
                for mode, data in value['modes'].items()
            },
            'primary_action_counts': {
                'true_ads': tp,
                'false_non_ads': fp,
                'ad_total': report['data']['ads'],
                'non_ad_total': report['data']['non_ads'],
            },
            'action_fpr_wilson_95_candidate_independence': wilson(fp, report['data']['non_ads']),
        })
        if name != 'rules':
            stats = paired_statistics(baseline, rows)
            stats.update(action_fpr_interval(baseline, rows))
            report['comparisons'][name] = stats
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = [
        '# 新域名人工盲测',
        '',
        '2026-11-01。352 条人工标签来自 36 个域名，其中广告 69 条、非广告 283 条。推理前固定标签、候选和阈值。',
        '',
        '## 纯规则与普通级联',
        '',
        '| 候选 | 分类广告 TP / FP | 策略处理广告 | 动作误杀 | 覆盖率 | ECE | 模型调用 | 决策核心 P95 ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, value in loaded.items():
        primary = value['modes']['rules_only' if name == 'rules' else 'rule_model_cascade']
        rows = primary['rows']
        tp = sum(r['label'] == 0 and r['prediction'] == 0 for r in rows)
        fp = sum(r['label'] == 1 and r['prediction'] == 0 for r in rows)
        counts = report['models'][name]['primary_action_counts']
        lines.append(
            f"| {name} | {tp} / {fp} | {counts['true_ads']}/69 | {counts['false_non_ads']}/283 | "
            f"{primary['coverage']:.4f} | {primary['ece_10_bin']:.4f} | "
            f"{primary['routing']['model_calls']}/352 | {primary['latency_ms']['p95']:.3f} |"
        )
    lines += [
        '', '## 配对域名聚类区间', '',
        '候选和纯规则共用 10,000 次域名重采样，随机种子为 42。区间按每项比较分别计算，未做多重比较调整。',
        '',
        '| 候选 | 动作召回差值 | 95%区间 | 动作FPR差值95%区间 | 广告动作配对改善/退步 | 精确p |',
        '| --- | ---: | --- | --- | ---: | ---: |',
    ]
    for name, stats in report['comparisons'].items():
        action = stats['metrics']['page_action']
        interval = action['delta_recall_domain_bootstrap_95']
        fpr = stats['delta_action_fpr_domain_bootstrap_95']
        test = action['ad_recall_mcnemar']
        lines.append(
            f"| {name} | {action['delta_recall']:.4f} | [{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"[{fpr[0]:.4f}, {fpr[1]:.4f}] | {test['improved']}/{test['regressed']} | "
            f"{test['two_sided_exact_p']:.6g} |"
        )
    lines += ['', '## 无规则证据的纯模型诊断', '', '| 候选 | 分类识别广告 | 策略处理广告 | 动作误杀 | 截断数 |', '| --- | ---: | ---: | ---: | ---: |']
    for name, value in loaded.items():
        if name == 'rules':
            continue
        diagnostic = value['modes']['model_only']
        rows = diagnostic['rows']
        classified_ads = sum(row['label'] == 0 and row['prediction'] == 0 for row in rows)
        acted_ads = sum(row['label'] == 0 and row['action'] in ('hide', 'block') for row in rows)
        acted_non_ads = sum(row['label'] == 1 and row['action'] in ('hide', 'block') for row in rows)
        lines.append(
            f"| {name} | {classified_ads}/69 | {acted_ads}/69 | {acted_non_ads}/283 | "
            f"{diagnostic['truncated']} |"
        )
    lines += [
        '', '## 测量口径', '',
        '- 纯模型诊断绕过规则分流。分类结果和页面动作分别计数。',
        '- 延迟只计预热后的单候选决策。HTTP、页面渲染和模型加载分别处理，评测使用串行推理，关闭缓存。',
        '- 精确 McNemar 检验按候选配对，域名重采样区间反映站点抽样差异。按候选独立计算，0/283 误杀的 Wilson 上界约为 1.34%。',
        '- 结果适用于这批公开站点候选。浏览器静态网络拦截另行测量。',
        '- 评估器在标签冻结后、模型推理前完成并记录哈希。报告保留全部种子。若用于下一轮选模型，应另建盲测集。',
        '- 逐样本预测和截图保存在本机 `runs/`，公开 JSON 只提供汇总。',
        '',
    ]
    (folder/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=ROOT/'runs/fresh-blind-review-v1')
    args = parser.parse_args()
    summarize(args.directory.resolve())


if __name__ == '__main__':
    main()
