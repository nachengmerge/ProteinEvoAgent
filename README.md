# ProteinEvoAgent

面向GB1蛋白质定向进化的科学智能体。此版本按用户提供的《实验报告.docx》整理，只保留对应的方法实现、必要数据、主要结果和交互演示，不包含历史迁移包、全阶段归档或模型权重。

仓库：https://github.com/nachengmerge/ProteinEvoAgent （公开，可直接访问）。

## 1. 报告与代码对应

| 报告章节 | 实现 | 对应结果 |
|---|---|---|
| 第2章 数据集 | `dataset.py`、`config.py`、`data/splits/` | 149,361条实测；发现/验证/测试112,040/14,928/22,393；初始512条 |
| 第3章 Ridge/P2 | `models.py`、`residual_model.py`、`features.py` | `results/predictor/metrics_*.csv`；512与560标签，固定测试集22,393条 |
| 第4章 LLM Agent | `active_science/planner.py`、`llama_baseline.py`、`delivery/workflow.py` | `results/agent/` 原始调用、工具评分和反馈 |
| 第5章 知识增强 | `knowledge.py`、`contextual_science.py`、`evidence_graph.py` | 主实验增加氨基酸性质/条件规则；图谱为单独补充的已知证据索引，不将其增益归入主表 |
| 第6章 三轮实验 | `scripts/run_p34_llama.py`、`candidates.py`、`oracle.py` | `results/evolution/` 八组×六方法×三轮的完整推荐及统计 |
| 第7章 失败案例 | `results/evolution/failure_cases.csv` 和完整推荐 | 45005组FWAA/KWAA；不同轮次模型，不作同模型配对结论 |
| 补充自审 | `grounded_designer_v2.py` | `results/supplement/self_review_summary.csv`；独立面板，不混入主实验 |

文件名中的P34是原实验的标识；保留核心代码的原字节以减少整理引入的语义变化。未写入报告的大规模扩大对照、ESM/RL扩展、失败阶段工程目录不在当前提交范围。详细归因与范围见[报告对应说明](docs/report_alignment.md)。

## 2. 环境与安装

Python ≥3.10。本机实测Windows/Python 3.13.9；CPU版本使用NumPy、pandas、SciPy、scikit-learn、joblib、psutil及openpyxl。P2需要PyTorch；真实Llama需要适合本机平台的llama-cpp-python后端。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m proteinevo.delivery doctor
```

Linux/macOS激活方式为`source .venv/bin/activate`。CPU示例不要求Torch或模型权重；本机原正式环境为PyTorch 2.11.0+cu128/CUDA和llama-cpp-python 0.3.36/Vulkan，两个后端分别核验。不能将提供安装命令当作已经验证全新机器或GPU跨平台复现。

## 3. 小规模复现

```powershell
python scripts/reproduce_small.py --output runs/small_cpu
python -m unittest discover -s tests -v
```

一条命令完成512条输入的校验/全长序列整理、Ridge内部80/20评价、候选生成、三轮推荐和反馈重训。CPU版采用随机、模型直推、工具规则，三轮128候选/8反馈，共72条。这里重新隐藏的是已经提供的观测，属于工程流程复现，不替代报告的真实正式对照。输入中合法的零fitness保留。

需要真实LLM四方法时，先安装本机推理后端并核验权重：

```powershell
python scripts/download_local_llm.py
python scripts/download_local_llm.py --verify-only
python -m proteinevo.delivery doctor --verify-model
python scripts/reproduce_small.py --include-llm --output runs/small_llama
```

四方法为随机、模型直推、普通LLM和知识LLM；三轮共96条已知记录反馈，保存原始prompt/输出、评分、推荐、已记录亲本和反馈。模型缺失或推理失败保留错误，不以规则文本冒充LLM。无动作导致零调用时不能宣称已验证LLM。

所有输出目录必须是新目录。`runs/`不会提交到Git。小流程不支持断点恢复；正式runner只支持其对应注册状态的恢复，不覆盖complete/failed结果。

## 4. 分步运行与正式配方

```powershell
# 观测内训练/评价，然后用512条拟合工作模型。
python -m proteinevo.delivery train --output runs/train_cpu
# 推荐只输出预测，不揭示候选真实fitness。
python -m proteinevo.delivery recommend --model runs/train_cpu/model.joblib --method full_greedy --output runs/recommend
# P2训练需先安装神经依赖。
python -m pip install -e ".[neural]"
python -m proteinevo.delivery train --predictor blend_resmlp_pairwise --device cuda --output runs/train_p2
# 读取已完成实验的汇总，不重新推理。
python -m proteinevo.delivery results
# 完整主实验配方：8组、6方法、三轮800候选/16反馈；需要P2和真实Llama环境。
python scripts/run_p34_llama.py run --config configs/llama_baseline_v1.json --output runs/full_main
```

完整配方是新运行，不是恢复历史成绩。小规模Ridge示例不能称为重现报告P2测试分数。原始正式源码哈希与本提交源文件对应见`SOURCE_PROVENANCE.json`；本次精简改变了交付入口和目录，未声称旧全阶段manifest可以在此无条件严格恢复。

## 5. 交互演示页

在线演示：[GitHub Pages — ProteinEvoAgent 实验交互演示](https://nachengmerge.github.io/ProteinEvoAgent/)（公开，无需登录或启动本地服务）。

双击`demo/index.html`即可查看，无需安装前端依赖；也可从仓库根目录启动：

```powershell
python -m http.server 8000 --directory demo
```

浏览器打开 http://localhost:8000 。支持8组/6方法/3轮筛选、Top-k、全长序列与替换高亮、预测/实测曲线、原始LLM输入输出、双背景检验及失败案例。CSV可导出。页面只浏览归档数据，不实时训练或推理。证据源为2,304条主实验推荐、144个状态和94次真实调用。

`python scripts/build_demo.py`可从提交的结果表与压缩决策状态重建`demo/data.js`，不重新训练。前端逻辑可用`node tests/test_demo.cjs`核验（Node.js ≥18）。

## 6. 主要结果

| 方法 | 三轮推荐均值 ± SD | 终局最佳增益 | fitness≥5事件 |
|---|---:|---:|---:|
| 全局随机 | 0.0960 ± 0.0677 | 0.0000 | 0 |
| 单P2直推 | 3.3360 ± 0.3445 | 1.7855 | 44 |
| 集成直推 | 3.5326 ± 0.3819 | 2.3773 | 55 |
| 工具规则 | 3.4108 ± 0.4248 | 1.8726 | 50 |
| 普通LLM | 3.4564 ± 0.3731 | 1.8249 | 48 |
| 知识LLM | 3.4879 ± 0.4026 | 1.8726 | 51 |

先汇总每组48条，再对八组取均值和SD。最佳增益相对于初始最佳，≥5计事件，非唯一变体或生物重复。P2在512标签测试时Spearman0.4353、Pearson0.5662、MSE0.1017、Top-20召回0.1375；完整分组表同时保留。集成直推优于两类LLM；知识增量仅3组正、3组负、2组相同，不宣称稳定独立优势。全局随机与精英邻域分布不同；反馈后各策略轨迹分化，不能称三轮完全同池。

## 7. 数据与使用说明

数据来自[Wu et al., eLife 2016, 5:e16965](https://doi.org/10.7554/eLife.16965)补充文件1，仅实测记录，不使用插补fitness。四位点39/40/41/54，WT=VDGV，56 aa全长由`config.py`生成。`data/splits/`是必要的已知/未知划分，只有评价器在决策后揭示所选ID；`gb1_oracle.csv.gz`不得直接交给Agent训练。历史测试集曾被使用，不称全新盲测。

完整原始Excel按需下载，不占源码仓库：

```powershell
python scripts/download_gb1.py
python -m proteinevo.delivery prepare --source data/raw/elife-16965-supp1-v4.xlsx --output runs/prepared_data
```

Llama基础模型为Meta Llama3.1 8B Instruct，GGUF来自bartowski Q4_K_M，权重不提交。模型和第三方来源、许可及ChatGPT/Codex辅助工程/核验/写作方式见[来源声明](docs/THIRD_PARTY_NOTICES.md)。原始LLM理由是可见输出，不是已验证的内部思维；数值/引用/合法性程序保证不归为LLM推理收益。
