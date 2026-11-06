# 第三方声明

根目录 Apache-2.0 仅覆盖项目自有代码与文档。下列文件继续受对应上游条款约束，详见
`browser/extension/rules/LICENSE.md` 与 `REFERENCES.md`。

## anti-AD

- 来源：https://github.com/privacy-protection-tools/anti-AD
- 作者版权：Copyright (c) 2017-2019 gently
- 许可：MIT
- 派生文件：`browser/extension/rules/anti_ad_rules.json`
- 完整许可文本：`browser/extension/THIRD_PARTY_NOTICES.txt`

## EasyList

- 来源：https://github.com/easylist/easylist
- 许可：GPL-3.0-or-later AND CC-BY-SA-3.0
- 用途：构建 bootstrap 训练数据时提取广告服务域名
- 原始列表不随项目分发

## AdGuard Base Filter

- 来源：https://github.com/AdguardTeam/AdguardFilters
- 许可：GPL-3.0-only
- 用途：构建 bootstrap 训练数据时提取广告服务域名
- 原始列表不随项目分发

## Turtlecute Host List

- 来源：https://adblock.turtlecute.org/d3host.adblock
- 上游：[Turtlecute33/adblocktest](https://github.com/Turtlecute33/adblocktest)，源自 [d3ward/toolz](https://github.com/d3ward/toolz)
- 许可：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。以下两个派生文件按此许可分发。
- 用途：生成本地快速判断域名和扩展静态规则
- 相关文件：`data/runtime/tracker_hosts.json`、`browser/extension/rules/tracker_rules.json`

## Tranco

- 来源：https://tranco-list.eu/
- 用途：构建 bootstrap 数据中的非广告页面域名
- 本项目取得的快照未声明许可，原始排名文件不随项目分发

其余仅用于验证的公开网页列在 `REFERENCES.md`。这些网页的 HTML 和 JavaScript 快照不随项目分发。
