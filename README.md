# CLARA: Claim-Level Argument Reliability Adjudication

面向闭卷（不检索外部证据）、纯文本虚假新闻检测的 LLM+SLM 协同框架，以及一套受控的多种子评测流程。

冻结的 LLM 对每条新闻只调用一次：把新闻拆成若干原子 claim，并为每个 claim 同时写出"最强支持论证"和"最强质疑论证"，给出怀疑度评分。可训练的小模型（裁决器）在 claim 级学习一个可靠性门控，决定在哪些 claim 上采信 LLM，再用 noisy-OR 形式的多示例学习（MIL）聚合成新闻级判定。训练只需要新闻级标签。

> 实验结果、图表与论文见 `results/` 和 `paper/`（实验完成后生成）。

## 目录结构

```
clara/
  llm/tasks.py        所有离线 LLM 任务的 prompt、JSON schema 与解析器
  text.py             分句（prompt、claim→句对齐、学生模型共用同一套切分）
  encoder.py          冻结前缀编码缓存（只训练编码器最后一层）
  models/clara.py     CLARA 裁决器及全部消融开关
  models/student.py   CLARA-D：无需 LLM 的学生模型（claim→句对齐蒸馏）
  models/baselines.py BERT、ARG、FactGuard 及其蒸馏版本
  train.py            统一的训练/评测循环（沿用 ARG 协议）
  experiment.py       实验会话：一个数据集共享一份编码缓存
  metrics.py          检测、校准、选择性预测、显著性检验、定位指标
  selective.py        拒答（选择性预测）
  corrupt.py          理由噪声注入
configs/sweeps.py     实验配置（主实验、消融、蒸馏等）
scripts/
  llm_generate.py     vLLM 批量离线生成（可断点续跑）
  build_dataset.py    合并原始数据与 LLM 缓存
  run.py              运行一组实验（已有结果自动跳过）
  lora_baseline.py    LoRA 微调 7B 直接分类基线
  make_perturb.py     风格攻击集、事实篡改定位集的构建
  aggregate.py        汇总 mean ± std
  llm_only.py         免训练的 LLM 基线
tests/test_smoke.py   合成数据上的端到端冒烟测试
```

## 环境

两个 Python 环境（训练与生成的依赖版本互相冲突，故分开）：

| 用途 | 主要依赖 |
|---|---|
| 生成（`scripts/llm_generate.py`） | vllm 0.26、torch 2.11 |
| 训练与分析（其余全部） | torch 2.4、transformers 4.46、peft 0.13、scikit-learn、matplotlib |

本地模型路径通过环境变量覆盖：`CLARA_BERT_ZH`、`CLARA_BERT_EN`（默认见 `clara/experiment.py`）。

系统 CUDA 工具链较旧时，vLLM 的 FlashInfer 采样器无法 JIT 编译，需设置 `VLLM_USE_FLASHINFER_SAMPLER=0`（`scripts/gen_main.sh` 已设置）。

## 数据

- `data/ARG-CN`（Weibo21）、`data/ARG-EN`（GossipCop）：ARG 论文发布的按时间划分的数据，需向原作者申请，**本仓库不包含也不得再分发**。
- LLM 生成的缓存与合并后的数据集文件由上述数据派生，同样不入库（见 `.gitignore`）。

## 复现流程

```bash
# 1. 离线生成（单卡 3090，Qwen2.5-7B-Instruct，约 2 小时）
bash scripts/gen_main.sh /path/to/Qwen2.5-7B-Instruct qwen25

# 2.（可选）恢复 Weibo21 领域标签，用于跨域实验
python -I scripts/map_weibo21_domains.py /path/to/weibo21_pkl_dir data/ARG-CN outputs/datasets/ARG-CN.domains.json

# 3. 合并数据集
bash scripts/build_all.sh qwen25

# 4. 训练与评测
python scripts/run.py --dataset outputs/datasets/ARG-CN.qwen25.json --sweep main
python scripts/aggregate.py ARG-CN.qwen25

# 5. 测试
python -m pytest tests -q
```

## 评测协议

所有可训练方法共用同一协议（沿用 ARG 官方实现）：编码器只微调最后一层；batch size 64；最多 50 轮；按验证集 macro-F1 早停（patience 5）；判定阈值固定为 0.5；测试集只在选定的 checkpoint 上评一次。在此基础上：

- 每个配置跑多个随机种子，报告均值 ± 标准差；
- 所有方法在同一组学习率上搜索，按验证集 macro-F1 选择；
- 需要 LLM 输出的基线（ARG、FactGuard）与 CLARA 使用**同一个** LLM 重新生成输入。

冻结层只前向一次并缓存（`clara/encoder.py`）。与每步重算相比，唯一差别是冻结层在训练时没有 dropout 噪声；`scripts/run.py --live` 可关闭缓存以核对这一点。
