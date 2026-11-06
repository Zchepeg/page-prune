# 参考材料与数据来源

本项目不包含下列上游项目的完整源码副本或网页快照。表中说明它们对架构、训练数据和已编译规则的影响。

| 类型 | 名称 | 上游地址 | 用途 | 上游许可 | 仓库中的结果 |
|---|---|---|---|---|---|
| 架构参考 | AnyJev | https://github.com/nokia-applied-research/AnyJev | JEV 类型化决策与提前退出思路 | Apache-2.0 | 仅保留引用，不分发上游源码 |
| 架构参考 | Kev | https://github.com/jaredpalmer/kev | Qwen 决策模型训练与评估思路 | Apache-2.0 | 仅保留引用，不分发上游源码 |
| 架构参考 | Laya | https://github.com/NandhaKishorM/laya | JEV 推理与服务实现参考 | Apache-2.0 | 仅保留引用，不分发上游源码 |
| 规则来源 | anti-AD | https://github.com/privacy-protection-tools/anti-AD | 大规模广告域名规则 | MIT | `browser/extension/rules/anti_ad_rules.json` |
| 训练来源 | EasyList | https://github.com/easylist/easylist | 广告域名提取与 bootstrap 样本构建 | GPL-3.0-or-later AND CC-BY-SA-3.0 | 生成训练记录，不包含原始列表 |
| 训练来源 | AdGuard Base Filter | https://github.com/AdguardTeam/AdguardFilters | 广告域名提取与 bootstrap 样本构建 | GPL-3.0-only | 生成训练记录，不包含原始列表 |
| 训练来源 | Tranco | https://tranco-list.eu/ | 非广告页面域名输入 | 快照未声明许可 | 生成训练记录，不包含原始排名文件 |
| 评测采集来源 | Majestic Million | https://majestic.com/reports/majestic-million | 独立评测候选网站抽样 | CC BY 3.0 | 仅本地使用排名目录，不提交原始 CSV、网页或截图 |
| 规则来源 | Turtlecute Host List | https://github.com/Turtlecute33/adblocktest | 广告、分析、社交像素和遥测域名 | CC BY-NC-SA 4.0 | `data/runtime/tracker_hosts.json` 与 `tracker_rules.json` 的一部分 |
| 测试参考 | AdsBlocks Test | https://adsblocks.com/tools/ad-blocker-test | 浏览器规则覆盖验证 | 页面未声明许可 | 仅保留独立编写的规则，不分发网页快照 |
| 测试参考 | TryBlock Advanced | https://tryblock.org/adblock-test/advanced | 中高级探针验证 | 页面未声明许可 | 仅保留独立编写的规则，不分发网页快照 |
| 测试参考 | MobileProxy Test | https://mobileproxy.space/ch/adblock-test.html | 广告和跟踪探针验证 | 页面未声明许可 | 仅保留提取后的域名规则，不分发网页快照 |
| 测试参考 | Super AdBlock Test | https://superadblocktest.com/ | 广告探针验证 | 页面未声明许可 | 仅保留提取后的域名规则，不分发网页快照 |
| 测试参考 | AdBlock Tester | https://adblock-tester.com/ | 网络与元素隐藏验证 | 页面未声明许可 | 仅保留独立编写的规则，不分发网页快照 |
| 测试参考 | Ads Blocker Test | https://ads-blocker.com/zh-Hans/%E5%B9%BF%E5%91%8A%E6%8B%A6%E6%88%AA%E6%B5%8B%E8%AF%95/ | 网络和控制请求验证 | 页面未声明许可 | 仅保留独立编写的规则，不分发网页快照 |
| 测试参考 | Can You Block It | https://canyoublockit.com/extreme-test/ | 极端广告组件验证 | 页面未声明许可 | 仅保留独立编写的规则，不分发网页快照 |
| 测试参考 | iBlockAds | https://iblockads.net/test | 广告和隐私端点验证 | 页面未声明许可 | 仅保留提取后的域名规则，不分发网页快照 |
| 测试参考 | Maurizio Fonte AdBlock Test | https://www.mauriziofonte.it/tools/adblock-test | 网络请求和元素诱饵验证 | 页面未声明许可 | 仅保留提取后的域名及独立 CSS 规则 |
| 测试参考 | AdBlockBench | https://adblockbench.com/ | 多类型广告组件验证 | 页面未声明许可 | 仅保留提取后的域名及独立 CSS 规则 |
| 测试参考 | PaileActivist Toolz | https://paileactivist.github.io/toolz/adblock.html | 广告和跟踪端点验证 | 页面未声明许可 | 仅保留提取后的域名规则，不分发网页快照 |

`data/training/bootstrap_manifest.json` 和 `data/training/tracker_manifest.json` 保存构建时所用快照的 SHA-256、生成数量和切分统计。重新生成数据时需要由维护者自行取得对应来源，并通过构建脚本参数显式传入。
