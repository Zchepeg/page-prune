# Reference-guided training data

`reference_guided_v1` 使用本地保存的 500 条人工二分类标注作为判定标准，为新采集且域名隔离的候选生成弱标签。原始候选、人工标注、训练 JSONL 和复核队列均被 Git 忽略。

## Construction

1. 读取标注日志中每个候选的最后一次标注，得到 132 条广告与 368 条非广告，共 55 个域名。
2. 从 DOM 语义、元素尺寸与位置、资源 URL、候选标签和规则证据中提取 `candidate-features-v2` 结构特征。
3. 按域名进行五折训练与折外预测，使用折外结果确定广告和非广告阈值。
4. 对 500 条新候选运行五模型集成，仅接收达到阈值且集成标准差不超过 0.10 的结果。
5. 删除与参考域名或校准域名重叠的候选，将不确定结果送入人工复核队列。
6. 根据人工广告样本的结构分布与广告供应商类别生成反事实广告样本，使训练输出类别平衡。反事实记录使用独立 provenance，不伪装为网页人工标注。

## Result

- 采集池：500 条；排除校准域名后输入 466 条、24 个域名。
- 高置信网页候选：294 条，其中广告 32 条、非广告 262 条。
- 结构反事实广告：230 条。
- 最终训练集：524 条，广告与非广告各 262 条。
- 待人工复核：172 条，不进入训练。
- 与 55 个参考域名、90 个校准域名的交集均为零。
- 候选 ID 和归一化状态均无重复。

折外阈值目标为 90% 精度，折外接收覆盖率为 78.4%，接收样本准确率为 92.35%。这些数字只描述参考集上的弱标注器选择行为，不是新数据的人工验证准确率，也不是广告屏蔽效果。

## Rebuild

准备好新的候选池后运行：

```powershell
python -m pip install -r requirements-model.txt

python scripts/build_reference_guided_dataset.py `
  --pool runs/reference-guided-v1/pool.jsonl `
  --out data/training/reference_guided_v1.jsonl `
  --review-out data/training/reference_guided_v1_review.jsonl `
  --report runs/reference-guided-v1/labeling_report.json `
  --precision-target 0.90 `
  --max-ensemble-std 0.10 `
  --balance-counterfactuals `
  --exclude-domain-data data/training/calibration_v2.jsonl
```

输出已加入 `auto_training.training_sources`，但自动训练保持关闭。只有用户明确运行训练命令时才会使用该数据。

## Human review

启动专用复核页面：

```powershell
python scripts/review_reference_guided.py serve
```

标注会以追加日志保存，关闭页面或重启服务后可以继续。按 `1` 标记广告、`2` 标记非广告、`3` 标记证据不足，按 `0` 才显示弱标注模型信息。页面中的“导出已复核样本”会把广告与非广告写入
`data/training/reference_guided_v1_human.jsonl`；证据不足样本不会进入训练。该输出已列入训练源，但自动训练仍保持关闭。

## Evaluation boundary

这 500 条人工标注已经成为开发监督，原有评测结果只能保留为旧检查点的历史基线。任何使用本数据训练或调参的模型都必须在重新采集、域名隔离、标签预先冻结的盲测集上评估。
