# CLARA: Claim-Level Argument Reliability Adjudication

面向闭卷（不检索外部证据）、纯文本虚假新闻检测的 LLM+SLM 协同框架，以及一套受控的多种子评测流程。

冻结的 LLM 把新闻拆成若干原子 claim，并为每个 claim 同时写出"最强支持论证"和"最强质疑论证"，给出怀疑度评分。可训练的小模型（裁决器）逐个实例决定是否采用 LLM 的分析：

- **相关性门控**判断一个分析单元是否真的属于这条新闻；不属于的单元变成不携带证据的"空实例"。全部为空时，预测精确退回到只看新闻文本的分支。
- **判定门控**调节 LLM 证据的权重。
- 实例经 noisy-OR 多示例学习（MIL）聚合为新闻级判定，训练只需要新闻级标签。

论文与全部结果见 `paper/` 与 `results/`。结论并不都是正面的，README 末尾有简要说明。

## 目录结构

```
clara/
  llm/tasks.py        所有离线 LLM 任务的 prompt、JSON schema 与解析器
  text.py             分句（prompt、claim→句对齐、学生模型共用同一套切分）
  encoder.py          冻结前缀编码缓存（只训练编码器最后一层）
  models/clara.py     CLARA 裁决器及全部消融开关
  models/student.py   CLARA-D：无需 LLM 的学生模型（claim→句对齐蒸馏）
  models/baselines.py BERT、BERT+ZS、ARG、FactGuard 及蒸馏版本
  train.py            统一的训练/评测循环（沿用 ARG 协议）
  experiment.py       实验会话：一个数据集共享一份编码缓存
  corrupt.py          LLM 分析的替换噪声（整条 / 单元级）
  metrics.py          检测、校准、选择性预测、定位指标
  analysis.py         论文中各项分析（显著性、门控、定位、拒答）
  selective.py        拒答（选择性预测）
  plotting.py         统一的作图样式
configs/sweeps.py     实验配置（主实验、消融、对照、蒸馏、敏感性）
scripts/
  llm_generate.py     vLLM 批量离线生成（可断点续跑）
  gen_main.sh         两个主数据集的全部 LLM 输出
  gen_perturb.sh      风格改写集、单句事实篡改集、遮挡集
  build_dataset.py    合并原始数据与 LLM 缓存
  run.py              运行一组实验（已有结果自动跳过）
  lora_baseline.py    LoRA 微调 7B 直接分类基线
  pipeline_train.sh   论文全部训练实验（分阶段，可续跑）
  pipeline_extra.sh   补充实验（风格归一化输入、更换分析 LLM）
  analyze.py          从已完成的运行计算论文用到的全部数字 → results/analysis.json
  make_assets.py      由 analysis.json 生成表格、图与正文中的数字
  aggregate.py        快速查看 mean ± std
  llm_only.py         免训练的 LLM 基线
  dev/val_check.py    只使用验证集的开发诊断
tests/test_smoke.py   合成数据上的端到端测试
paper/                论文 LaTeX 源文件
results/              每次运行的指标（runs/）与汇总（analysis.json）
```

## 环境

两个 Python 环境（训练与生成的依赖版本互相冲突，故分开）：

| 用途 | 主要依赖 |
|---|---|
| 生成（`scripts/llm_generate.py`） | vllm 0.26、torch 2.11 |
| 训练与分析（其余全部） | torch 2.4、transformers 4.46、peft 0.13、scikit-learn、scipy、matplotlib |

本地模型路径通过环境变量覆盖：`CLARA_BERT_ZH`、`CLARA_BERT_EN`（默认见 `clara/experiment.py`）。

系统 CUDA 工具链较旧时，vLLM 的 FlashInfer 采样器无法 JIT 编译，需设置 `VLLM_USE_FLASHINFER_SAMPLER=0`（生成脚本已设置）。

## 数据

- `data/ARG-CN`（Weibo21）、`data/ARG-EN`（GossipCop）：ARG 论文发布的按时间划分的数据，需向原作者申请，**本仓库不包含也不得再分发**。
- LLM 生成的缓存、合并后的数据集文件、逐样本预测由上述数据派生，同样不入库（见 `.gitignore`）。入库的 `results/` 只含汇总指标与配置。

## 复现流程

```bash
# 1. 离线生成（单卡 3090，Qwen2.5-7B-Instruct，约 2 小时）
bash scripts/gen_main.sh /path/to/Qwen2.5-7B-Instruct qwen25

# 2.（可选）恢复 Weibo21 领域标签，用于留一领域实验
python -I scripts/map_weibo21_domains.py /path/to/weibo21_pkl_dir data/ARG-CN outputs/datasets/ARG-CN.domains.json

# 3. 合并数据集
bash scripts/build_all.sh qwen25

# 4. 受控测试集：改写与篡改由另一个模型（Qwen3-8B）完成，再由分析模型重新分析
bash scripts/gen_perturb.sh /path/to/Qwen3-8B /path/to/Qwen2.5-7B-Instruct qwen25

# 5. 全部训练实验（约 20 GPU 小时，可随时中断后重新执行）
bash scripts/pipeline_train.sh

# 6. 汇总、制表作图、编译论文
python scripts/analyze.py && python scripts/make_assets.py && make -C paper

# 测试
python -m pytest tests -q
```

## 评测协议

所有基于 SLM 的方法共用同一协议（沿用 ARG 官方实现）：编码器只微调最后一层；batch size 64；最多 50 轮；按验证集 macro-F1 早停（patience 5）；判定阈值固定为 0.5；测试集只在选定的 checkpoint 上评一次。在此基础上：

- 主实验每个配置 5 个随机种子，报告均值 ± 标准差，并做配对 bootstrap 检验；
- 所有方法在同一组学习率 {2e-5, 5e-5} 上运行，按验证集 macro-F1 选择；
- 需要 LLM 输出的基线（ARG、FactGuard）与 CLARA 使用**同一个** LLM 重新生成输入；
- CLARA 的损失权重等超参数为预先设定的固定值，未做搜索；开发阶段的结构选择只参考了训练集抽样与验证集（`scripts/dev/val_check.py`）。

冻结层只前向一次并缓存（`clara/encoder.py`），可训练的最后一层用混合精度计算。`scripts/run.py --live` 关闭缓存与混合精度、每步重算冻结层，用于核对两种方式的一致性。

## 需要事先知道的结论

- 分布内精度上，CLARA 与最强的 LLM+SLM 基线持平，并未超过"BERT + LLM 零样本分数后融合"这一最简单的做法；LoRA 直接微调 LLM 更强。
- CLARA 的相关性门控能可靠地识别不属于该新闻的分析单元，使模型在分析被替换时平缓退回文本分支。
- 风格改写与单句事实篡改两项受控测试显示：包括 LLM 自身判断在内的所有闭卷检测器都主要依赖文风，对局部事实改动几乎不敏感。
- 判定门控并没有学到"LLM 是否判断正确"，它跟随的是新闻标签。

细节与数字见论文。
