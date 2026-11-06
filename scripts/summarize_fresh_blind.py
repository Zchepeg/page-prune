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
    return {'delta_action_fpr_domain_bootstrap_95':[values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]], 'valid_resamples':len(values), 'limitation':'percentile resampling cannot create unobserved false positives; zero endpoints do not prove safety'}


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
        report['models'][name].update({'modes':{mode:{key:content for key,content in data.items() if key != 'rows'} for mode,data in value['modes'].items()}, 'primary_action_counts':{'true_ads':tp,'false_non_ads':fp,'ad_total':report['data']['ads'],'non_ad_total':report['data']['non_ads']}, 'action_fpr_wilson_95_candidate_independence':wilson(fp,report['data']['non_ads'])})
        if name != 'rules':
            stats = paired_statistics(baseline, rows)
            stats.update(action_fpr_interval(baseline, rows))
            report['comparisons'][name] = stats
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = ['# 新域名冻结人工盲测结果', '', '日期：2026-11-01。352条、36域名；广告69条、非广告283条。全部标签在推理前冻结，五候选与阈值未按结果调整，未自动部署。', '', '## 主比较：规则与普通级联', '', '| 候选 | 分类广告 TP / FP | 策略处理广告 | 动作误杀 | 覆盖率 | ECE | 模型调用 | 决策核心 P95 ms |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, value in loaded.items():
        primary = value['modes']['rules_only' if name == 'rules' else 'rule_model_cascade']
        rows = primary['rows']
        tp = sum(r['label'] == 0 and r['prediction'] == 0 for r in rows)
        fp = sum(r['label'] == 1 and r['prediction'] == 0 for r in rows)
        counts = report['models'][name]['primary_action_counts']
        lines.append(f"| {name} | {tp} / {fp} | {counts['true_ads']}/69 | {counts['false_non_ads']}/283 | {primary['coverage']:.4f} | {primary['ece_10_bin']:.4f} | {primary['routing']['model_calls']}/352 | {primary['latency_ms']['p95']:.3f} |")
    lines += ['', '## 配对域名聚类区间', '', '10000次、种子42，所有候选与规则使用同一域名重采样。以下为每项比较的95%百分位区间；五项比较为描述性结果，没有多重比较调整，不能据此挑选最佳种子并声称确认性显著。', '', '| 候选 | 动作召回差值 | 95%区间 | 动作FPR差值95%区间 | 广告动作配对改善/退步 | 精确p |', '| --- | ---: | --- | --- | ---: | ---: |']
    for name, stats in report['comparisons'].items():
        action = stats['metrics']['page_action']
        interval = action['delta_recall_domain_bootstrap_95']
        fpr = stats['delta_action_fpr_domain_bootstrap_95']
        test = action['ad_recall_mcnemar']
        lines.append(f"| {name} | {action['delta_recall']:.4f} | [{interval[0]:.4f}, {interval[1]:.4f}] | [{fpr[0]:.4f}, {fpr[1]:.4f}] | {test['improved']}/{test['regressed']} | {test['two_sided_exact_p']:.6g} |")
    lines += ['', '## 无规则证据的纯模型诊断', '', '| 候选 | 分类识别广告 | 策略处理广告 | 动作误杀 | 截断数 |', '| --- | ---: | ---: | ---: | ---: |']
    for name, value in loaded.items():
        if name == 'rules':
            continue
        diagnostic = value['modes']['model_only']
        rows = diagnostic['rows']
        lines.append(f"| {name} | {sum(r['label']==0 and r['prediction']==0 for r in rows)}/69 | {sum(r['label']==0 and r['action'] in ('hide','block') for r in rows)}/69 | {sum(r['label']==1 and r['action'] in ('hide','block') for r in rows)}/283 | {diagnostic['truncated']} |")
    lines += ['', '## 解释边界', '', '- 分类广告识别与最终页面动作分开；纯模型诊断不经过规则快速路径。', '- 耗时为预热后的串行单候选决策核心，不是浏览器整页或HTTP请求时间；没有缓存或批处理，另保存模型调用耗时与加载、预热耗时。', '- 候选级精确McNemar没有处理域名相关性，只作补充；域名区间不包含训练随机性。', '- 零误杀观察不代表风险为零；0/283的候选独立Wilson上界约1.34%，也不是域名聚类安全保证。', '- 样本来自混合候选采样与可访问的公开站点，不能外推全网拦截率，也不包含浏览器静态网络拦截的独立效果。', '- 评估器实现记录在标签冻结后、各模型推理前，原生schema和规则分流经过测试；这是执行实现时序披露，不伪称实现本身在采集前预注册。', '- 所有种子均报告，不自动选优或部署；此盲测若用于下一轮模型选择，将不再是后续版本的盲测。', '- 聚合JSON不含逐样本、域名、截图或本机路径；原始预测留在runs目录。', '']
    (folder/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=ROOT/'runs/fresh-blind-review-v1')
    args = parser.parse_args()
    summarize(args.directory.resolve())


if __name__ == '__main__':
    main()
