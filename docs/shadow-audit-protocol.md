# SSA Shadow Audit Protocol v1

SSA 用极低比例审计最终仍被放行、但确定性检测分数处于中高区间的候选。协议版本为 `ssa-probe-v1`。它只观察用户已经加载的页面状态，不创建广告请求、不点击元素、不模拟曝光或转化，也不执行页面提供的未知代码。

## Eligibility and sampling

候选必须同时满足：

- 最终动作为 `allow` 或 `review`；
- 检测分数大于等于 0.45 且小于 0.90；
- 页面、域名和小时预算仍有余额；
- SHA-256 确定性采样命中。

| 分数区间 | 采样率 |
| --- | ---: |
| 0.45–0.60 | 0.1% |
| 0.60–0.75 | 0.5% |
| 0.75–0.90 | 1.0% |

采样材料由域名、脱敏候选指纹、小时窗口和协议版本组成。同一小时窗口内同一候选的选择稳定，下一窗口可以重新探索。

每个浏览器标签页每小时最多接受 3 个探针，每个域名每小时最多接受 20 个探针。内容脚本还维护当前文档最多 3 个探针的本地限制。

## Probe families

- `delayed_recheck`：延迟后重新获取稳定状态；
- `dom_change`：统计有限的子节点和属性变化；
- `visibility`：观察可见性与视口状态变化；
- `resource_association`：观察已有元素新增的资源主机。

探针类型和等待时间由同一确定性摘要选择。等待时间限制在 200–3000 毫秒内，MutationObserver 在完成后立即销毁。

## Privacy boundary

探针允许记录尺寸、可见性、视口状态、固定定位、子节点数量、最多八个资源主机、连接状态和变化计数。探针禁止记录：

- 页面文本或完整 HTML；
- 元素 ID 或类名；
- 完整页面 URL 和资源路径；
- URL 查询参数、凭据或片段；
- Cookie、Token、表单值和用户输入。

服务端使用严格字段白名单，出现额外字段时拒绝整个记录。有效记录仅写入 `data/events/probes.jsonl`，不属于训练来源。

## Analysis and review

服务端将探针变化转换为 `probe_score`、`information_value` 和可审计证据。信息价值低于 0.15 的记录只保留审计统计。达到 0.15 的新观测进入待复核。达到 0.30 时，服务尝试用 `probe-ad-classification-v1` 进行模型判断。重复观测、模型加载或故障期间会跳过推理，符合条件的新观测仍可进入人工复核。

Qwen 的探针判断是实验性辅助信号，不能改变当前页面动作，也不能直接成为训练标签。待复核记录保存在 `data/training/pending_probe_review.jsonl`，人工结果单独保存在 `data/training/reviewed_probes.jsonl`。本地服务运行时可访问 `http://127.0.0.1:8765/review`，在同一页面复核 SSA 探针与普通浏览器候选。

本地 `GET /v1/status` 的 `probes.outcomes` 按处理路线、模型状态和复核结果分别计数。例如 `model.decided` 是实际得到模型判断的观测数，`model.duplicate_review` 是跳过重复推理的次数，`review.queued` 是进入人工复核的次数。这些计数自服务启动累计，不含域名、探针 ID 或页面内容。它们用于确认探针有没有触发及产生复核记录，无法直接衡量拦截收益。

## Failure behavior

探针失败、超时、本地服务不可用、元素断开或预算耗尽时，页面保持原决策，不重试外部请求，也不升级权限。扩展与本地服务使用 `jev-local-session-v1` 配对，探针请求和响应通过临时 P-256 ECDH 会话派生的 AES-256-GCM 密钥保护；详细边界见 `docs/local-security.md`。
