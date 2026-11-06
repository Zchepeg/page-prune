# 发布前干净环境验证

验证日期：2026-11-01。范围为当前工作区源码，不代表已发布版本或远端 CI 结果。

## 已验证

在项目忽略目录 `.venv/release-check` 中创建独立 Conda 环境；使用 Git 跟踪文件和未忽略新文件的当前内容生成源码副本，在该副本中运行。副本不含 `.git`、本机配置、模型权重、采集记录或私有评测集。原有 GPU 环境与运行中的本地服务未修改。

| 检查 | 结果 |
| --- | --- |
| 仅安装基础依赖后启动公开默认配置 | 通过，未导入 PyTorch |
| 随机本地端口 `/health`、`/v1/status` | 通过，配对存储使用临时目录 |
| 完整 CPU 测试依赖安装 | 通过 |
| `python -m pip check` | 无依赖冲突 |
| 离线完整测试 | 180 passed，9.25 秒，无跳过 |
| 扩展 7 个脚本 `node --check` | 全部通过 |

测试环境为 Windows、Python 3.11.18，PyTorch 2.8.0+cpu，CUDA 不可用。其余关键版本：Transformers 4.57.6、PEFT 0.21.2、Accelerate 1.15.0、scikit-learn 1.9.1、pytest 8.4.2、cryptography 46.0.7。测试设置 Hugging Face 离线环境变量，不下载模型权重。

## 复现入口

在独立环境中先安装 `requirements.txt`，执行 `python scripts/smoke_public_runtime.py`；然后安装 CPU PyTorch 和 `requirements-test.txt`，执行 `pip check` 与完整 pytest。具体命令见根目录 `PUBLISHING.md`。已有 CUDA 环境无需替换 PyTorch。

基础运行依赖、开发工具、完整 CPU 测试和 GPU 模型依赖分别位于 `requirements.txt`、`requirements-dev.txt`、`requirements-test.txt` 和 `requirements-model.txt`。

## 仍需确认

- GitHub Actions 已配置 Windows、Ubuntu 与 Python 3.11、3.13 的四组合矩阵，尚未在远端运行；本次只验证 Windows Python 3.11 的干净 CPU 环境。
- 语法检查与服务测试不能替代浏览器扩展安装、配对、人工标记和页面兼容性验收。
- 发布前仍须检查最终暂存区及 Git 历史中的敏感文件，确定附件与权重发布范围。
- 本次验证证明安装与测试可复现，不构成模型效果、浏览器性能或商业许可的新增证据。
