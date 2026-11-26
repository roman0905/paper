# 基于大语言模型的虚假新闻检测：闭卷裁决与证据驱动验证

本仓库是一篇硕士学位论文的全部代码、实验结果与论文源文件，包含先后两部分工作：

1. **闭卷检测（`clara/`）**：只看新闻文本、不检索外部证据。提出声明级论证裁决方法 CLARA，并设计了一套受控的多种子评测流程。结论以负面发现为主：闭卷检测器主要依赖文风（见文末）。
2. **开卷验证（`openbook/`）**：在 RAWFC 与 LIAR-RAW 上，利用数据集自带的原始报道作为证据验证声明。包括对证据来源的泄漏审计、两种证据协议，以及证据自筛选验证方法 SIEVE（LoRA 微调的 LLM 在同一次前向计算中给证据句打分并判断真伪，训练时混入错配证据，推理时先读、再选、后判）。

学位论文（江西师范大学硕士学位论文模板）在 `thesis/`；闭卷部分另有一份文章体例的稿件在 `paper/`。

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
  build_dataset.py    合并原始数据与 LLM 缓存（build_all.sh 一次合并全部）
  make_perturb.py     由改写/篡改模型的输出构建受控测试集
  map_weibo21_domains.py  从 Weibo21 原始发布恢复领域标签（留一领域实验用）
  run.py              运行一组实验（已有结果自动跳过）
  lora_baseline.py    LoRA 微调 7B 直接分类基线
  selected_lr.py      输出各方法按验证集选中的配置名（供流水线脚本使用）
  pipeline_train.sh   论文全部训练实验（分阶段，可续跑）
  pipeline_extra.sh   补充实验（只读 LLM 复述的分类器、更换分析 LLM）
  pipeline_llm2.sh    用第二个 LLM 重新生成全部输出并重训依赖 LLM 的方法
  pipeline_checks.sh  核对实验（交叉评测、全精度复核、每步重算冻结层）
  collect_official_arg.py  汇总 ARG 官方代码的运行结果 → results/official_arg.json
  analyze.py          从已完成的运行计算论文用到的全部数字 → results/analysis.json
  make_assets.py      由 analysis.json 生成表格、图与正文中的数字
  aggregate.py        快速查看 mean ± std
  llm_only.py         免训练的 LLM 基线
  dev/                只使用训练集抽样与验证集的开发诊断
openbook/             开卷部分
  data.py             RAWFC / LIAR-RAW 的规范化、报道的泄漏标记、两种证据协议
  retrieval.py        候选句池与 BM25 第一阶段排序
  prompts.py          验证器的提示词
  verifier.py         LLM 验证器（选项字母读出、LoRA、证据打分头）与训练循环
  views.py            训练/测试查询的构造：证据重采样、错配证据、各种证据受控条件
  pipeline.py         单遍推理、读—选—判推理与证据打分头的诊断指标
  metrics.py          多分类指标与配对 bootstrap
  report.py           读取结果、方法间比较
scripts/ob_*.py|sh    开卷部分的数据准备、训练评测、流水线、汇总与制表
tests/                合成数据上的端到端测试（test_smoke.py 闭卷，test_openbook.py 开卷）
paper/                闭卷部分的文章稿（LaTeX）
thesis/               学位论文（LaTeX，见 thesis/README.md）
results/              每次运行的指标与汇总：results/analysis.json（闭卷），results/ob/（开卷）
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

## 闭卷部分的复现流程

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

# 6. 补充实验：只读 LLM 复述的分类器；用第二个 LLM（Qwen3-8B）重新生成并重训
MODEL2_PATH=/path/to/Qwen3-8B MODEL2_TAG=qwen3 bash scripts/pipeline_extra.sh

# 7. 核对实验：交叉评测、主实验配置的全精度复核、每步重算冻结层
bash scripts/pipeline_checks.sh cross fp32 fp32_main live_clara

# 8.（可选）与 ARG 官方代码对照：在其仓库中用多个种子运行 run_zh.sh，再汇总测试指标
python scripts/collect_official_arg.py /path/to/official_arg_logs

# 9. 汇总、制表作图、编译论文
python scripts/analyze.py && python scripts/make_assets.py && make -C paper

# 测试
python -m pytest tests -q
```

## 开卷部分的复现流程

```bash
# 1. 规范化数据并计算 BM25 排序（约 1.5 分钟；数据放在 data/RAWFC、data/LIAR-RAW）
python -I scripts/ob_prepare.py RAWFC data/RAWFC
python -I scripts/ob_prepare.py LIAR-RAW data/LIAR-RAW

# 2. 全部实验（分阶段、可续跑；单卡 3090 约 4 天）
bash scripts/ob_pipeline.sh rawfc_main liar_main ablation all_protocol

# 3. 汇总、制表作图、编译学位论文
python scripts/ob_analyze.py && python scripts/ob_assets.py && make -C thesis
```

RAWFC 与 LIAR-RAW 由其作者公开发布（Yang et al., COLING 2022），本仓库不包含数据。数据集自带的事实核查结论文本与银标证据标签只用作训练信号，不进入任何测试输入。

## 评测协议（闭卷部分）

所有基于 SLM 的方法共用同一协议（沿用 ARG 官方实现）：编码器只微调最后一层；batch size 64；最多 50 轮；按验证集 macro-F1 早停（patience 5）；判定阈值固定为 0.5；测试集只在选定的 checkpoint 上评一次。在此基础上：

- 主实验每个配置 5 个随机种子，报告均值 ± 标准差；
- 显著性同时报告对测试样本的配对 bootstrap 与对随机种子的 Welch 检验，**两者都小于 0.05 才称显著**，未做多重比较校正；
- 所有方法在同一组学习率 {2e-5, 5e-5} 上运行，按验证集 macro-F1 选择；
- 需要 LLM 输出的基线（ARG、FactGuard）与 CLARA 使用**同一个** LLM 重新生成输入；
- CLARA 的损失权重等超参数为预先设定的固定值，未做搜索；开发阶段的结构选择只参考了训练集抽样与验证集（`scripts/dev/val_check.py`）。

冻结层只前向一次并缓存（`clara/encoder.py`），可训练的最后一层用混合精度计算（`scripts/run.py --amp`，主实验的协议）。不带 `--amp` 为全精度；`--live` 关闭缓存、每步重算冻结层，与官方实现的计算一致。论文附录 C 用后两种方式复核了主实验：定性结论不变，但 ARG 与 FactGuard 的数值对精度敏感，1 个点左右的差异会随之出现或消失。

## 闭卷部分：需要事先知道的结论

结果并不都支持最初的设想（macro-F1，Weibo21 / GossipCop，5 个种子）：

| 方法 | Weibo21 | GossipCop |
|---|---|---|
| BERT | 76.46 ± 0.35 | 76.66 ± 1.34 |
| LoRA 微调 Qwen2.5-7B（3 个种子） | 79.74 ± 1.36 | 77.04 ± 0.56 |
| BERT + LLM 零样本评分 | 78.00 ± 0.43 | 77.73 ± 0.47 |
| ARG | 77.41 ± 0.97 | 77.17 ± 1.04 |
| FactGuard | 76.38 ± 0.80 | 77.24 ± 1.01 |
| CLARA | 77.41 ± 0.49 | 78.15 ± 1.06 |

- **分布内持平。** CLARA 与 ARG、"BERT + LLM 零样本评分"之间的差异都不显著；LoRA 直接微调 LLM 在 Weibo21 上更强，在 GossipCop 上没有优势。在留一领域实验中，最简单的评分融合显著优于 CLARA。
- **论证文本、声明分解与判定门控对精度没有可测的贡献**（消融）。在 Weibo21 上 CLARA 相对 BERT 的提升几乎全部来自 LLM 的数值评分。
- **相关性门控有效但作用范围窄。** 它能识别来自其他新闻的分析单元（AUC 约 0.99），去掉它模型在分析被替换时会崩溃；但它识别不了同话题而在某处与文本事实不符的分析（AUC 约 0.5），并且是软门控，退回并不精确。ARG 本身对整条替换已相当稳健。
- **判定门控没有学到"LLM 是否判断正确"**，它跟随的是新闻标签；ARG 的理由可用性预测器同样如此。
- **所有闭卷检测器都主要依赖文风。** 风格改写后 Weibo21 上各小模型方法降到 50–53，LLM 自身零样本判断的 AUC 从 83.2 降到 58.7；各方法对单句事实改动几乎没有反应，声明级定位只略高于随机。
- 蒸馏得到的学生模型都没有显著超过 BERT。

细节与数字见论文（`paper/main.pdf`）；论文中的每个数字都由 `scripts/make_assets.py` 从 `results/analysis.json` 生成。
