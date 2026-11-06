# Independent evaluation and ablation

## Human review preparation

Prepare, validate, deduplicate and stratify the captured evaluation pool before annotation:

```powershell
python scripts/prepare_evaluation_review.py --qwen-config config/local.yaml
python scripts/review_evaluation_pool.py serve
```

Machine suggestions are stored separately from human annotations and remain hidden until explicitly requested in the review interface. Direct label buttons record an independent human decision; accepting a revealed suggestion records assisted review provenance. Neither machine suggestion type is treated as an independent human label.

For a new claim-eligible benchmark, start the review service with `--blind`. Blind mode does not send prelabels to the browser and removes the suggestion controls:

```powershell
python scripts/review_evaluation_pool.py serve --blind
```

独立评测集使用 `data/schema/benchmark_schema.json` 定义的 JSONL 格式。每行包含稳定样本 ID、页面域名、人工标签和完整候选对象。标签为 `ad`、`non_ad` 或 `insufficient_evidence`。

测试域名必须与所有训练来源按域名隔离。评测命令默认发现任意重叠就终止，只有诊断旧数据时才应使用 `--allow-domain-overlap`。

评测文件可以配套同名 `.metadata.json`。例如 `independent_test.jsonl` 对应
`independent_test.metadata.json`。只有同时满足以下条件的结果才会标记为可用于性能声明：

- 独立人工或独立专家人工标注，并在运行评测前冻结标签；
- 至少 300 个样本、30 个域名，每个标签至少 50 个样本；
- 与全部训练来源保持域名隔离；
- 元数据记录数据来源。

不满足条件时仍会生成诊断报告，但 `claim_eligibility.eligible` 为 `false`，Markdown 报告也会列出原因。

## 独立浏览器基准集

安装开发依赖后采集 500 条本地候选：

```powershell
python scripts/build_evaluation_pool.py --target 500
```

An external CSV catalog with a `Domain` column can supply fresh sites while the same training-domain and redirect checks remain active:

```powershell
python scripts/build_evaluation_pool.py --target 500 --catalog-csv <catalog.csv>
```

采集器会在访问前以及跳转后排除训练域名。临时截图、页面证据和标注历史保存在 `runs/evaluation-annotation/`，不会被 Git 收录。

启动本地复核界面：

```powershell
python scripts/review_evaluation_pool.py serve
```

打开终端打印的回环地址，使用 `1`、`2`、`3` 将候选标记为广告、非广告或证据不足。至少完成 300 条复核、覆盖 30 个域名且每类达到 50 条后才能冻结。

也可以从命令行查看进度和冻结：

```powershell
python scripts/review_evaluation_pool.py status
python scripts/review_evaluation_pool.py freeze --ground-truth-task binary_ad_detection --independent-only
```

二分类冻结要求人工标签中同时存在 `ad` 与 `non_ad`。`--independent-only` 会排除查看或接受过机器建议的标注，只保留直接人工判断。冻结过程会移除临时 HTML，并写入被 Git 忽略的 `data/evaluation/independent_test.jsonl` 和 `data/evaluation/independent_test.metadata.json`。冻结后再运行五模式消融：

冻结时元数据会写入 `used_for_supervision: false`。一旦该测试集被用于训练、弱标注、规则修改、提示设计或阈值选择，必须将其改为 `true`；评测器会拒绝把这类结果标记为可用于性能声明。

```powershell
python -m evaluation.run_ablation --data data/evaluation/independent_test.jsonl --config config/local.yaml --out runs/evaluation-independent
```

逐样本结果只保存在 `runs/`。需要生成可公开结果时，增加 `--public-summary`，输出会自动移除样本 ID、域名、逐条预测、本机路径和原始元数据，只保留聚合指标：

```powershell
python -m evaluation.run_ablation `
  --data data/evaluation/independent_test.jsonl `
  --config config/local.yaml `
  --out artifacts/evaluation-independent `
  --public-summary
```

## Five modes

| 模式 | 规则 | 模型 | 协议挑战与合议 |
| --- | --- | --- | --- |
| `rules_only` | 是 | 否 | 否 |
| `model_only` | 否 | 是 | 否 |
| `rule_model_cascade` | 是 | 仅灰区 | 否 |
| `jev_ad_protocol` | 是 | 灰区与边界挑战 | 是 |
| `jev_ad_protocol_with_ssa` | 是 | 灰区、边界挑战和高信息探针 | 是 |

`model_only` 的状态中不包含检测器证据，避免模型基线间接使用规则输出。四组实验共享同一策略配置和同一批候选。

## Metrics

- 广告类别 precision、recall、F1、false-positive rate 和 accuracy；
- 三分类 accuracy 和非不确定覆盖率；
- 10 桶 ECE；
- 对人工标注非广告执行 `block` 或 `hide` 的页面动作误杀率；
- 平均延迟和 P95 延迟；
- SSA 复核率、广告发现召回率和复核命中率；
- 三分类混淆矩阵、决策来源和逐样本结果。

页面动作误杀率与分类误杀率分开统计，因为模型判断和策略实际执行是两个不同风险边界。

## Command

```powershell
python -m evaluation.run_ablation `
  --data data/evaluation/independent_test.jsonl `
  --config config/config.yaml `
  --out artifacts/evaluation
```

在没有加载 Qwen 权重的开发环境中，可以用启发式后端验证评测链路：

```powershell
python -m evaluation.run_ablation `
  --data <benchmark.jsonl> `
  --backend heuristic `
  --out artifacts/evaluation-smoke
```

输出包括 `ablation.json`、`ablation.csv` 和 `ablation.md`。默认 JSON 包含逐样本诊断，只应写入被 Git 忽略的本地目录；公开目录必须使用 `--public-summary`。只有人工独立标注、标签冻结在评测前、与训练监督完全隔离并使用正式 Qwen 检查点得到的结果，才能作为项目效果声明。若一个旧基准后来参与了弱标签器、阈值、提示、规则或训练数据的构建，它只能继续描述此前未接触该基准的旧检查点，不能用于评估后续模型。
