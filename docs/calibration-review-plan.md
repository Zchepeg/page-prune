# 校准数据补充盘点

已检查本地候选池，排除配置训练来源、全部训练快照及两版人工基准的域名和父子域名关系，共602个已用域名。早期盘点的7个域名在纳入全部快照排除后剩6个，不能据此认为已有足够新数据。

生成runs/calibration-review-v1/pool.jsonl：30条无标签候选，6域名。每域名每规则分层最多4条；高规则分数6条、低规则分数24条、边界候选0条。规则分层只用于覆盖统计，不代表真实广告类别或标签。

当前不足：域名少、缺边界候选、无新人工标签，且记录为历史采集证据，不能假定对应当前网页。此队列不是新校准集，不会进入训练或更改检查点。页面截图或上下文不足时应标记证据不足，不能仅凭类名和规则分数确定标签。

复核可使用已有盲标界面并指定独立输出，避免覆盖测试集：

```powershell
python scripts/review_evaluation_pool.py serve --blind --pool runs/calibration-review-v1/pool.jsonl --annotations runs/calibration-review-v1/annotations.jsonl --output runs/calibration-review-v1/reviewed.jsonl --metadata runs/calibration-review-v1/reviewed.metadata.json --port 8878
```

该工具默认冻结门槛针对独立评测，30条队列不应强行冻结为独立基准。后续需要专门按校准监督用途导出，保留人工标注来源；本轮只准备队列，不导出标签。

下一批采集应预先保留新域名，覆盖内容站、论坛、工具站和交易页面，并同时采集广告疑似区域、边界区域与随机正常对照。至少补到30个新域名后再审视标签与结构覆盖；条数和域名数量不是有效性的保证。保留未查看的新域名用于最终盲测，其余用于校准开发。禁止依据模型输出直接生成所谓人工标签。

复现入口scripts/prepare_calibration_review.py。队列和manifest保存在runs下，不进入公开仓库；脚本拒绝覆盖已有复核队列。

## 新网页采集结果

本轮使用隔离、未登录Edge配置访问43个公开站点，排除配置训练源、所有训练快照、历史人工基准及上一批复核域名。成功采集183条、24个新域名，每域名最多8条；15站HTTP错误、1站访问异常、3站域名重叠被跳过，未达到300条和30域名目标。数据位于runs/calibration-fresh-v1/pool.jsonl。

候选分层：疑似35、随机对照59、普通结构52、模糊尺寸37。重新计算的规则分层：低分134、高分14、边界35；这些都不是人工广告标签。截图初检154条有视觉变化、25条近空白、4条缺图。近空白不等于非广告，有视觉变化也不保证证据充分。部分成功页面可能是同意页或付费墙，复核时需明确页面上下文。

父子域名隔离、ID唯一与每域名限额检查通过。原始采集与质量审计保存在本地runs目录；没有训练或重新校准模型。后续需补充至少6个新域名，并完成直接人工标注，保留证据不足选项；当前数据不能宣称校准效果或广告类别覆盖达标。

新采集池盲标入口：

```powershell
python scripts/review_evaluation_pool.py serve --blind --pool runs/calibration-fresh-v1/pool.jsonl --annotations runs/calibration-fresh-v1/annotations.jsonl --output runs/calibration-fresh-v1/reviewed.jsonl --metadata runs/calibration-fresh-v1/reviewed.metadata.json --port 8878
```

截图初检入口：scripts/audit_calibration_capture.py。盲标后仍需校准用途的受控导出及域名切分，不能沿用独立评测冻结元数据作为监督数据声明。

## 补充完成与统一复核包

第二批新增48条、6域名，来自公开调查新闻、国际科技、学术传播和科普内容站。两批合并为runs/calibration-supplement-v1/pool.jsonl，共231条、30个新域名；226张对应截图已复制到统一screenshots目录。

规则分层为低分180、高分14、边界37；候选类型为疑似41、随机对照77、普通结构67、模糊尺寸46。截图初检201条有视觉变化、25条近空白、5条缺图。域名、ID、每域名8条限制核验通过，仍无人工标签，尚不能用于校准。

统一复核命令使用上述盲标命令，将calibration-fresh-v1替换为calibration-supplement-v1。复核时使用1广告、2非广告、3证据不足；不要点击冻结来宣称独立测试，后续通过专门的校准用途导出处理。匿名汇总与原始证据均保存在runs下，公开仓库不收录。

复现合并：scripts/merge_calibration_capture.py --pools runs/calibration-fresh-v1/pool.jsonl runs/calibration-fresh-v2/pool.jsonl。脚本拒绝覆盖已有标注目录，并保留来源池哈希。

## 校准用途导出

专用入口scripts/export_calibration_review.py默认只报告就绪状态。增加--export后，只有全部样本完成人工复核、无机器建议辅助来源、排除证据不足后每切分每类至少15条、校准至少10域名且验证至少5域名时才导出。

按固定字符串与域名SHA256排序，每第三个域名进入验证，其余进入校准，划分规则不依赖标签或模型成绩。父子域名重叠、已用监督或测试域名、重复ID均拒绝；导出元数据明确为calibration_development_only、used_for_supervision=true。数据写入新的快照目录，拒绝覆盖既有导出。

```powershell
python scripts/export_calibration_review.py
python scripts/export_calibration_review.py --export
```

当前231条均未复核，导出检查正确报告未就绪，没有创建训练或校准数据文件。三项测试验证了标签翻转不改变切分、辅助/缺失标注阻止导出，以及父域名泄漏被拒绝。

## 人工复核完成后的固定验证域名补采

231条现已全部人工复核：40广告、191非广告。固定校准切分155条（广告28、非广告127），固定验证切分76条（广告12、非广告64）；验证广告未达每类15条门槛，未导出或重新校准。

在原10个验证域名上用新随机种子、每域名最多12条进行整批补采，得到110条；按已有ID去重后新增77条，来自其中9个域名。未按模型广告预测或人工结果筛选补采记录。合并包runs/calibration-supplement-v2共308条，保留原231条人工标注及其哈希，待复核77条；所有校准记录、20个校准域名和10个验证域名保持不变。原每域名8条限额仅针对首轮，本轮补采增加了验证域名内记录数，后续指标需关注域名聚类与重复页面带来的相关性。

此补采因验证类别不足而启动，因此必须继续标为开发验证，不能视为未接触的独立盲测。原始标注与v1目录保留，未覆盖。新页面端口8879；专用导出检查使用`python scripts/export_calibration_review.py --run runs/calibration-supplement-v2`，完成后仍需达到原门槛。
