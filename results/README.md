# results/

只包含汇总指标与配置，不包含任何数据集文本或逐样本预测（这些在 `outputs/` 下，不入库）。

| 路径 | 内容 |
|---|---|
| `runs/<数据集>.<LLM>/<实验名>/s<种子>.json` | 一次运行：配置、各评测集指标、每轮验证曲线、训练后的标量参数、耗时 |
| `runs/<数据集>.<LLM>-holdout-<领域>/` | 留一领域实验 |
| `runs/ARG-CN.qwen25-live/`、`-fp32/` | 协议核对：每步重算冻结层 / 缓存但全精度 |
| `runs/<数据集>.qwen25-cross/` | 篡改文本配原文分析的交叉评测 |
| `runs/<数据集>.qwen3/` | 更换分析用 LLM（Qwen3-8B） |
| `llm_only/*.json` | 不训练的 LLM 基线 |
| `analysis.json` | 论文用到的全部数字，由 `scripts/analyze.py` 从上面的文件与本地预测计算得到 |

实验名的含义见 `configs/sweeps.py`：`name@lr<学习率>` 是主实验的网格点（按验证集 macro-F1 选择），`clara-<变体>` 是消融与敏感性实验，`*_d` 是蒸馏学生。

每个运行文件里的 `metrics` 包含：`val`、`test`，以及（视实验而定）`noiseXX`（整条替换 XX% 的样本）、`unitXX`（每个分析单元以 XX% 的概率被替换）、`style`（风格改写集）、`tamper`（单句事实篡改集）、`occl`（遮挡变体）、`cross`（交叉评测）。

`analysis.json` 中 `cases` 一项含两条测试样例的文本，用于论文附录的样例展示。
