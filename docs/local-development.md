# 本地开发指南

以下命令均在仓库根目录运行。扩展安装和服务配对见 [README](../README.md)。

## 测试环境

可用 Conda 创建基础环境：

```powershell
conda env create -f environment.yml
conda activate jev_ad_block
```

完整测试可使用独立 CPU 环境：

```powershell
python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pip check
python -m pytest
```

已有 CUDA 环境时，直接安装 `requirements-test.txt`，保留现有 CUDA 版 PyTorch。

```powershell
python scripts/smoke_public_runtime.py
python scripts/browser_compatibility.py
python scripts/browser_scenarios.py
python scripts/paired_browser_e2e.py
```

浏览器脚本使用临时 Edge 配置。兼容性和页面场景报告写到 `runs/`，场景截图写到 `runs/visual/`。配对测试会另启隔离服务。[发布验收记录](release-validation.md)

## 切换到 Qwen

安装模型依赖。首次准备本机配置时，从公开配置复制一份：

```powershell
python -m pip install -r requirements-model.txt
Copy-Item config/config.yaml config/local.yaml
```

将检查点放到 `models/jev-qwen-pointer/`，或在 `config/local.yaml` 中指定 `jev.checkpoint`。把 `jev.backend` 改成 `qwen_pointer`，然后启动：

```powershell
python -m service.api --config config/local.yaml
```

当前模型配置使用 NF4、384 token 输入上限、微批量 2 和 4096 MiB CUDA 分配器预算。每批候选调用上限由 `runtime.max_model_candidates_per_batch` 控制。诊断信息中的已分配与保留显存是进程内数据。[性能记录](runtime-performance.md)

## 接口和数据

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
Invoke-RestMethod http://127.0.0.1:8765/v1/status
Invoke-RestMethod http://127.0.0.1:8765/v1/training/stats
```

`POST /v1/decisions` 接收候选批次。配对后的消息格式见[本地通信文档](local-security.md)，候选字段见 [data/schema](../data/schema/)。结果中的 `decision` 是规则或模型判断，`command` 是最终页面动作。

高置信规则样本可写入 `data/training/auto_ads.jsonl`。普通对照样本和不确定候选进入 `pending_review.jsonl`。网页上点选并保存的人工标签写入 `reviewed.jsonl`，标记的元素随即隐藏。采集设置位于 `training_capture`，自动训练开关 `auto_training.enabled` 默认为关闭。

## 训练与校准

按域名分开训练、验证、校准和测试数据。正式训练前检查类别覆盖和人工标签来源。项目建议训练集至少有 300 条记录，覆盖 20 个域名，每类至少 60 条。独立校准集建议每类至少 15 条。[数据来源与构建](reference-guided-data.md)

```powershell
python -m training.train `
  --data data/training/train.jsonl `
  --validation data/training/validation.jsonl `
  --model Qwen/Qwen3-4B-Base `
  --label-mode binary_selective `
  --epochs 4 --patience 2 --accum 8 --max-length 384 `
  --evidence-dropout 0.5 --feature-dropout 0.1 `
  --domain-balance-power 0.5 --label-smoothing 0.05 `
  --out runs/qwen3-4b-ad-v2

python -m training.calibrate `
  --checkpoint runs/qwen3-4b-ad-v2 `
  --data data/training/calibration.jsonl `
  --quantization nf4 --gpu-memory-budget-mb 4096 --max-length 384
```

用与部署一致的量化配置做独立评测，检查误杀、校准和延迟。通过后手动修改 `jev.checkpoint` 并重启服务。[评测入口与指标](evaluation.md)

## 重建规则

浏览器使用编译后的域名与页面规则。anti-AD 域名规则针对第三方请求，主文档导航排除在网络阻断之外。重建时需准备来源快照：

```powershell
python scripts/build_tracker_dataset.py `
  --host-list <d3host-file> `
  --anti-ad-domains <anti-ad-domains-file> `
  --benchmark-source <optional-saved-probe-file>
```

来源与分发许可见 [REFERENCES](../REFERENCES.md) 和 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。
