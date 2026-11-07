# GitHub 发布清单

本文件描述公开源码仓库与版本附件的发布边界。

## 必须确认

- 根目录 `LICENSE` 已采用 Apache-2.0，`NOTICE` 的版权主体为 `Zchepeg`。
- `data/runtime/tracker_hosts.json` 与 `browser/extension/rules/tracker_rules.json` 含有 CC BY-NC-SA 来源，不受根目录 Apache-2.0 覆盖，不应用于不符合其非商业和相同方式共享条件的分发。
- 决定模型权重的发布位置。`runs/` 不进入 Git，应使用 Hugging Face 或 GitHub Release，并公布 SHA-256。
- `artifacts/evaluation-independent/` 只包含旧检查点的历史聚合指标；对应的 500 条人工标注现已用于训练集构建，不能再作为新检查点的独立评测。
- `artifacts/evaluation-independent-v2/` 只包含当前检查点在 490 条、63 域名独立盲标集上的去标识聚合结果；原始候选、域名和逐条预测不得提交。

## 不得提交

- `browser/extension.pem`
- `browser/extension.crx`
- `runs/`、`checkpoints/`、`models/`
- 自动采集、人工标记、待复核数据和事件日志
- 完整训练、校准和独立评测记录
- IDE、测试缓存、服务日志和 Chrome `_metadata`

这些路径已加入 `.gitignore`。首次提交前仍应检查暂存区，避免使用强制添加。

## 发布前验证

当前本地干净环境验证结果见 `docs/release-validation.md`：Windows Python 3.11 CPU 环境 180 项测试通过。远端 CI 和浏览器验收仍须分别确认。

```powershell
python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pip check
python -m pytest
node --check browser/extension/background.js
node --check browser/extension/content.js
node --check browser/extension/newtab.js
node --check browser/extension/popup.js
node --check browser/extension/screenshot-background.js
node --check browser/extension/screenshot-content.js
node --check browser/extension/screenshot-review.js
node --check browser/extension/options.js
node --check browser/extension/probe.js
node --check browser/extension/privacy_guard.js
```

上述 CPU 安装步骤用于独立验证环境，已有 CUDA 环境不应替换为 CPU PyTorch。CI 在 Windows、Ubuntu 的 Python 3.11 和 3.13 上执行相同检查，并先验证只安装基础依赖的公开服务启动。CI 配置已提供，远端运行结果需在上传后确认。

在 Chromium 中加载 `browser/extension/`，验证：

1. 三个静态规则集均成功加载。
2. 新标签页显示热点榜、搜索框和本地收藏。验证热点日期和离线状态，以及网格导航、添加网址、删除文件夹和刷新后的持久化。
3. 工具栏弹窗默认进入人工标记。勾选截图复核后，验证遮盖预览、保存与丢弃，并确认未授予全站权限。
4. 本地服务关闭时扩展保持 fail-open。
5. 本地服务开启时模型判断、自动采集和人工标记正常。

## 首次提交

仓库初始化后执行：

```powershell
git init
git add .
git status
```

确认暂存区中没有私钥、模型权重、浏览历史或用户采集数据后再创建首次提交。

## 版本附件

扩展发布包应从受控文件清单生成，不包含 `_metadata`、`.pem`、本机配置或数据目录。模型权重
与扩展包分开发布，并为每个附件提供 SHA-256。未经独立、跨域人工评测，不得把合成集或
校准集指标表述为真实准确率。

当前独立 v2 评测显示，JEV 级联相对纯规则仅将 F1 从 0.5294 提升到 0.5333，额外找回 1 条广告且没有增加页面动作误杀；纯模型广告召回率为 0。发布说明必须保留这些限制，不得宣称模型可以替代规则或已经带来显著性能提升。v2 一旦用于训练、规则、提示、阈值或模型选择，后续版本必须重新建立独立盲测集。
